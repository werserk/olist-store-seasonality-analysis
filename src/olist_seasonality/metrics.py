from __future__ import annotations

import json
import math
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import FIGURES, MONTH_NAMES, NOTES, PROCESSED, RESULTS, TABLES, WEEKDAY_NAMES
from .utils import mode_or_nan, safe_div, save_fig, table_to_markdown

def order_level(enriched: pd.DataFrame) -> pd.DataFrame:
    delivered = enriched[enriched["is_delivered"]].copy()
    order_cols = [
        "order_id",
        "order_purchase_timestamp",
        "year",
        "month",
        "year_month",
        "week",
        "weekday",
        "payment_value",
        "payment_installments",
        "payment_type",
        "customer_state",
        "delivery_time_days",
        "estimated_delivery_delta_days",
        "review_score",
    ]
    order_agg = (
        delivered.groupby(order_cols, dropna=False)
        .agg(
            revenue=("price", "sum"),
            freight_sum=("freight_value", "sum"),
            items_sold=("order_item_id", "count"),
            categories_count=("product_category_name_english", "nunique"),
        )
        .reset_index()
    )
    order_agg["gmv"] = order_agg["revenue"] + order_agg["freight_sum"]
    return order_agg


def aggregate_metrics(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delivered = enriched[enriched["is_delivered"]].copy()
    orders = order_level(enriched)
    p90 = orders["payment_value"].quantile(0.9)
    p75 = orders["payment_value"].quantile(0.75)
    orders["large_purchase_p90"] = orders["payment_value"].ge(p90)
    orders["large_purchase_p75"] = orders["payment_value"].ge(p75)
    orders.to_parquet(PROCESSED / "orders_enriched_order_level.parquet", index=False)

    monthly_items = (
        delivered.groupby("year_month")
        .agg(
            items_sold=("order_item_id", "count"),
            products_sold=("product_id", "nunique"),
            revenue=("price", "sum"),
            freight_sum=("freight_value", "sum"),
            avg_item_price=("price", "mean"),
            avg_freight_value=("freight_value", "mean"),
            avg_review_score=("review_score", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
        )
        .reset_index()
    )
    monthly_orders = (
        orders.groupby("year_month")
        .agg(
            orders_count=("order_id", "nunique"),
            payment_value_sum=("payment_value", "sum"),
            avg_order_value=("payment_value", "mean"),
            large_purchase_share=("large_purchase_p90", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
        )
        .reset_index()
    )
    monthly = monthly_orders.merge(monthly_items, on="year_month", how="left")
    monthly["gmv"] = monthly["revenue"] + monthly["freight_sum"]
    monthly["year"] = pd.PeriodIndex(monthly["year_month"], freq="M").year
    monthly["month"] = pd.PeriodIndex(monthly["year_month"], freq="M").month
    monthly.to_parquet(PROCESSED / "monthly_metrics.parquet", index=False)
    monthly.to_csv(TABLES / "monthly_metrics.csv", index=False)

    weekly = (
        orders.groupby("week")
        .agg(
            orders_count=("order_id", "nunique"),
            revenue=("revenue", "sum"),
            avg_order_value=("payment_value", "mean"),
            large_purchase_share=("large_purchase_p90", "mean"),
        )
        .reset_index()
    )
    weekly.to_parquet(PROCESSED / "weekly_metrics.parquet", index=False)

    # Figures 2-4.
    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.lineplot(data=monthly, x="year_month", y="orders_count", marker="o", ax=ax)
    ax.set_title("Delivered orders by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=65)
    save_fig(FIGURES / "02_orders_by_month.png")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.lineplot(data=monthly, x="year_month", y="revenue", marker="o", color="#2a9d8f", ax=ax)
    ax.set_title("Revenue by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue, BRL")
    ax.tick_params(axis="x", rotation=65)
    save_fig(FIGURES / "03_revenue_by_month.png")

    heat = monthly.pivot(index="year", columns="month", values="orders_count").reindex(columns=range(1, 13))
    plt.figure(figsize=(11, 4.6))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Orders"})
    plt.title("Delivered orders heatmap: year × month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    save_fig(FIGURES / "04_year_month_heatmap.png")
    return monthly, weekly
