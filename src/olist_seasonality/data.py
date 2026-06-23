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
from .utils import mode_or_nan, read_csv, save_fig, table_to_markdown

def load_tables() -> dict[str, pd.DataFrame]:
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = read_csv("olist_orders_dataset.csv", parse_dates=date_cols)
    items = read_csv("olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    products = read_csv("olist_products_dataset.csv")
    translations = read_csv("product_category_name_translation.csv")
    payments = read_csv("olist_order_payments_dataset.csv")
    reviews = read_csv(
        "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )
    customers = read_csv("olist_customers_dataset.csv")
    sellers = read_csv("olist_sellers_dataset.csv")
    geolocation = read_csv("olist_geolocation_dataset.csv")
    return {
        "orders": orders,
        "items": items,
        "products": products,
        "translations": translations,
        "payments": payments,
        "reviews": reviews,
        "customers": customers,
        "sellers": sellers,
        "geolocation": geolocation,
    }


def audit_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": len(df.columns),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum()),
                "columns_list": ", ".join(df.columns),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(PROCESSED / "data_audit.csv", index=False)

    lines = ["# Data audit", "", "## Tables", "", table_to_markdown(audit.drop(columns=["columns_list"])), ""]
    for name, df in tables.items():
        lines.append(f"## `{name}` columns")
        lines.append("")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(15).rename("missing").reset_index().rename(columns={"index": "column"})
        lines.append("Columns: " + ", ".join(df.columns))
        lines.append("")
        if not miss.empty:
            lines.append("Top missing columns:")
            lines.append("")
            lines.append(table_to_markdown(miss))
            lines.append("")
    (PROCESSED / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def build_enriched(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"].copy()
    items = tables["items"].copy()
    products = tables["products"].copy()
    translations = tables["translations"].copy()
    customers = tables["customers"].copy()
    sellers = tables["sellers"].copy()

    payments_agg = (
        tables["payments"]
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", mode_or_nan),
            payment_methods_count=("payment_type", "nunique"),
            payment_rows=("payment_type", "size"),
        )
        .reset_index()
    )
    reviews_agg = (
        tables["reviews"]
        .groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "nunique"),
            review_creation_date=("review_creation_date", "min"),
        )
        .reset_index()
    )
    products = products.merge(translations, on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("unknown")

    enriched = (
        items.merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(reviews_agg, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left", suffixes=("", "_seller"))
    )

    ts = enriched["order_purchase_timestamp"]
    enriched["year"] = ts.dt.year
    enriched["quarter"] = ts.dt.quarter
    enriched["month"] = ts.dt.month
    enriched["month_name"] = enriched["month"].map(MONTH_NAMES)
    enriched["year_month"] = ts.dt.to_period("M").astype(str)
    enriched["week"] = ts.dt.to_period("W").astype(str)
    enriched["order_date"] = ts.dt.date
    enriched["weekday"] = ts.dt.weekday
    enriched["weekday_name"] = enriched["weekday"].map(WEEKDAY_NAMES)
    enriched["product_volume_cm3"] = (
        enriched["product_length_cm"] * enriched["product_height_cm"] * enriched["product_width_cm"]
    )
    enriched["delivery_time_days"] = (
        enriched["order_delivered_customer_date"] - enriched["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    enriched["estimated_delivery_delta_days"] = (
        enriched["order_estimated_delivery_date"] - enriched["order_delivered_customer_date"]
    ).dt.total_seconds() / 86400
    enriched["freight_share"] = enriched["freight_value"] / (enriched["price"] + enriched["freight_value"])
    enriched["is_delivered"] = enriched["order_status"].eq("delivered")
    enriched["is_low_review"] = enriched["review_score"].le(2)
    enriched["is_high_review"] = enriched["review_score"].ge(4)
    enriched.to_parquet(PROCESSED / "orders_items_enriched.parquet", index=False)
    enriched.to_csv(PROCESSED / "orders_items_enriched_sample.csv", index=False)
    return enriched


def coverage_analysis(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"].copy()
    rows = []
    for label, df in [("all_orders", orders), ("delivered", orders[orders["order_status"].eq("delivered")])]:
        s = df["order_purchase_timestamp"].dropna()
        for year, g in s.groupby(s.dt.year):
            first, last = g.min().date(), g.max().date()
            total_days = 366 if calendar.isleap(int(year)) else 365
            rows.append(
                {
                    "scope": label,
                    "year": int(year),
                    "orders": int(len(g)),
                    "first_order": first.isoformat(),
                    "last_order": last.isoformat(),
                    "span_days": int((last - first).days + 1),
                    "active_days": int(g.dt.date.nunique()),
                    "calendar_days": total_days,
                    "missing_before": int((first - pd.Timestamp(year=int(year), month=1, day=1).date()).days),
                    "missing_after": int((pd.Timestamp(year=int(year), month=12, day=31).date() - last).days),
                }
            )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(PROCESSED / "coverage_by_year.csv", index=False)

    delivered = orders[orders["order_status"].eq("delivered")].copy()
    delivered["year"] = delivered["order_purchase_timestamp"].dt.year
    delivered["month"] = delivered["order_purchase_timestamp"].dt.month
    ym = delivered.groupby(["year", "month"])["order_id"].nunique().unstack(fill_value=0).reindex(columns=range(1, 13))
    plt.figure(figsize=(11, 4.6))
    sns.heatmap(ym, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Delivered orders"})
    plt.title("Data coverage: delivered orders by year/month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    save_fig(FIGURES / "01_data_coverage_by_year_month.png")
    return coverage
