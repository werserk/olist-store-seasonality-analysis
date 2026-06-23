#!/usr/bin/env python3
"""Run the full Olist seasonality analysis and generate final-ready artifacts."""

from __future__ import annotations

import calendar
import json
import math
import warnings
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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180})

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "olist-brazilian-ecommerce"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
NOTES = ROOT / "notes"

MONTH_NAMES = {i: calendar.month_abbr[i] for i in range(1, 13)}
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def ensure_dirs() -> None:
    for path in [PROCESSED, FIGURES, TABLES, NOTES]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw table: {path}")
    return pd.read_csv(path, **kwargs)


def mode_or_nan(series: pd.Series):
    non_na = series.dropna()
    if non_na.empty:
        return np.nan
    return non_na.mode().iloc[0]


def safe_div(a, b):
    return np.where(np.asarray(b) == 0, np.nan, np.asarray(a) / np.asarray(b))


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


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


def classify_type(row: pd.Series) -> str:
    if row["confidence_level"] == "low":
        return "low_confidence"
    if row.get("trend_strength", 0) > 0.55 and row["cv"] > 0.8:
        return "trend_driven"
    if row.get("event_spike_count", 0) >= 3 and row["peak_month_share"] < 0.25:
        return "event_driven"
    if row["cv"] < 0.35 and row["peak_to_mean"] < 1.6:
        return "stable"
    if row["peak_month_share"] >= 0.35 or row["peak_to_mean"] >= 2.6:
        return "single_month_spike"
    if row["peak_quarter"] == 4 and row["peak_quarter_share"] >= 0.34:
        return "holiday_q4"
    if row["peak_quarter"] == 1 and row["peak_quarter_share"] >= 0.34:
        return "early_year"
    if row["peak_quarter"] in [2, 3] and row["peak_quarter_share"] >= 0.34:
        return "mid_year"
    if row["peak_quarter_share"] >= 0.45:
        return "quarter_peak"
    if row.get("weekly_pattern_strength", 0) >= 0.45:
        return "weekly_pattern"
    return "mixed_seasonal"


def category_seasonality(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delivered = enriched[enriched["is_delivered"]].copy()
    base_2017 = delivered[delivered["year"].eq(2017)].copy()
    categories = sorted(base_2017["product_category_name_english"].fillna("unknown").unique())
    months = range(1, 13)
    full_idx = pd.MultiIndex.from_product([categories, months], names=["category", "month"])

    cat_month = (
        base_2017.groupby(["product_category_name_english", "month"])
        .agg(
            orders_count=("order_id", "nunique"),
            items_sold=("order_item_id", "count"),
            revenue=("price", "sum"),
            avg_item_price=("price", "mean"),
            avg_freight_value=("freight_value", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            avg_review_score=("review_score", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            customer_state_nunique=("customer_state", "nunique"),
            seller_state_nunique=("seller_state", "nunique"),
        )
        .rename_axis(index=["category", "month"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )
    cat_month["month_name"] = cat_month["month"].map(MONTH_NAMES)
    cat_month["seasonal_index"] = cat_month.groupby("category")["orders_count"].transform(
        lambda x: x / x.mean() if x.mean() else np.nan
    )
    cat_month.to_parquet(PROCESSED / "category_monthly_metrics.parquet", index=False)

    # Weekly category metrics.
    cat_week = (
        delivered.groupby(["product_category_name_english", "week"])
        .agg(orders_count=("order_id", "nunique"), revenue=("price", "sum"), items_sold=("order_item_id", "count"))
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    cat_week.to_parquet(PROCESSED / "category_weekly_metrics.parquet", index=False)

    # Daily spikes per category on all delivered data.
    daily = (
        delivered.groupby(["product_category_name_english", "order_date"])["order_id"]
        .nunique()
        .rename("orders_count")
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    spike_rows = []
    for cat, part in daily.groupby("category"):
        part = part.sort_values("order_date").copy()
        if len(part) < 35:
            continue
        part["rolling_mean_30"] = part["orders_count"].rolling(30, min_periods=30).mean()
        part["rolling_std_30"] = part["orders_count"].rolling(30, min_periods=30).std()
        part["z_score"] = (part["orders_count"] - part["rolling_mean_30"]) / part["rolling_std_30"].replace(0, np.nan)
        spikes = part[part["z_score"].abs() > 3].copy()
        if not spikes.empty:
            spike_rows.append(spikes.assign(category=cat))
    event_spikes = pd.concat(spike_rows, ignore_index=True) if spike_rows else pd.DataFrame()
    if not event_spikes.empty:
        event_spikes.to_csv(TABLES / "event_spikes.csv", index=False)
    else:
        pd.DataFrame(columns=["category", "order_date", "orders_count", "z_score"]).to_csv(TABLES / "event_spikes.csv", index=False)

    spike_counts = event_spikes.groupby("category").agg(
        event_spike_count=("order_date", "count"), max_spike_z_score=("z_score", "max")
    ) if not event_spikes.empty else pd.DataFrame(columns=["event_spike_count", "max_spike_z_score"])

    weekday = (
        delivered.groupby(["product_category_name_english", "weekday"])["order_id"]
        .nunique()
        .rename("orders_count")
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    weekday["weekday_index"] = weekday.groupby("category")["orders_count"].transform(lambda x: x / x.mean() if x.mean() else np.nan)
    weekly_strength = weekday.groupby("category")["weekday_index"].agg(lambda x: x.max() - x.min()).rename("weekly_pattern_strength")

    trend_strength = []
    for cat, part in cat_month.groupby("category"):
        y = part.sort_values("month")["orders_count"].to_numpy(dtype=float)
        if y.mean() == 0:
            strength = 0.0
        else:
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            strength = abs(slope) * 11 / max(y.max(), 1)
        trend_strength.append((cat, strength))
    trend_strength = pd.DataFrame(trend_strength, columns=["category", "trend_strength"]).set_index("category")

    rows = []
    quarters = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
    for cat, part in cat_month.groupby("category"):
        y = part.sort_values("month")["orders_count"].astype(float)
        revenue = part["revenue"].sum()
        total_orders = y.sum()
        active_months = int((y > 0).sum())
        mean = y.mean()
        std = y.std(ddof=0)
        peak_month = int(part.loc[y.idxmax(), "month"]) if total_orders else np.nan
        q_sums = {q: float(part.loc[part["month"].isin(ms), "orders_count"].sum()) for q, ms in quarters.items()}
        peak_quarter = max(q_sums, key=q_sums.get) if total_orders else np.nan
        peak_quarter_share = q_sums[peak_quarter] / total_orders if total_orders else np.nan
        nonzero = y[y > 0]
        confidence = "high" if total_orders >= 200 and active_months >= 10 else "medium" if total_orders >= 100 and active_months >= 6 else "low"
        rows.append(
            {
                "category": cat,
                "total_orders": int(total_orders),
                "total_revenue": revenue,
                "active_months": active_months,
                "monthly_mean": mean,
                "monthly_std": std,
                "cv": std / mean if mean else np.nan,
                "peak_to_mean": y.max() / mean if mean else np.nan,
                "peak_to_min": y.max() / nonzero.min() if len(nonzero) else np.nan,
                "peak_month_share": y.max() / total_orders if total_orders else np.nan,
                "peak_quarter_share": peak_quarter_share,
                "seasonal_index_max": y.max() / mean if mean else np.nan,
                "peak_month": peak_month,
                "peak_quarter": peak_quarter,
                "confidence_level": confidence,
            }
        )
    ranking = pd.DataFrame(rows).set_index("category")
    ranking = ranking.join(spike_counts, how="left").join(weekly_strength, how="left").join(trend_strength, how="left")
    ranking[["event_spike_count", "weekly_pattern_strength", "trend_strength"]] = ranking[
        ["event_spike_count", "weekly_pattern_strength", "trend_strength"]
    ].fillna(0)
    ranking["seasonality_score"] = ranking["cv"]
    ranking["seasonality_type"] = ranking.apply(classify_type, axis=1)
    ranking = ranking.reset_index().sort_values("seasonality_score", ascending=False)
    ranking.to_csv(PROCESSED / "category_seasonality_scores.csv", index=False)
    ranking.to_csv(TABLES / "category_seasonality_ranking.csv", index=False)

    eligible = ranking.query("confidence_level != 'low'").copy()
    top = eligible.head(15).sort_values("seasonality_score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, y="category", x="seasonality_score", hue="seasonality_type", dodge=False)
    plt.title("Top seasonal categories by CV score (2017, delivered orders)")
    plt.xlabel("Seasonality score (CV of monthly orders)")
    plt.ylabel("")
    plt.legend(title="Type", fontsize=7, title_fontsize=8, loc="lower right")
    save_fig(FIGURES / "05_top_seasonal_categories.png")

    heat_cats = eligible.head(20)["category"]
    heat = cat_month[cat_month["category"].isin(heat_cats)].pivot(index="category", columns="month", values="seasonal_index")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, cmap="RdYlBu_r", center=1, vmin=0, vmax=min(3, np.nanmax(heat.values)), cbar_kws={"label": "Seasonal index"})
    plt.title("Category × month seasonal index for top seasonal categories")
    plt.xlabel("Month")
    plt.ylabel("")
    save_fig(FIGURES / "06_category_month_seasonal_index_heatmap.png")

    # KMeans-like clustering via sklearn if enough eligible categories.
    from sklearn.cluster import KMeans

    cluster_source = cat_month[cat_month["category"].isin(eligible["category"])].pivot(index="category", columns="month", values="seasonal_index").fillna(0)
    n_clusters = min(6, max(2, len(cluster_source) // 5))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(cluster_source)
    cluster_order = pd.Series(labels, index=cluster_source.index, name="cluster").sort_values()
    cluster_heat = cluster_source.loc[cluster_order.index]
    plt.figure(figsize=(12, max(6, len(cluster_heat) * 0.18)))
    sns.heatmap(cluster_heat, cmap="RdYlBu_r", center=1, cbar_kws={"label": "Seasonal index"})
    plt.title("Seasonality type clusters: category monthly profiles")
    plt.xlabel("Month")
    plt.ylabel("Category")
    save_fig(FIGURES / "07_seasonality_type_clusters.png")

    cluster_profiles = cluster_source.assign(cluster=labels).groupby("cluster").mean().reset_index()
    cluster_profiles.to_csv(TABLES / "seasonality_cluster_profiles.csv", index=False)

    # Daily spikes overview for total demand.
    all_daily = delivered.groupby(pd.to_datetime(delivered["order_date"]))["order_id"].nunique().rename("orders_count").reset_index().rename(columns={"order_date": "date"})
    all_daily["rolling_mean_30"] = all_daily["orders_count"].rolling(30, min_periods=30).mean()
    all_daily["rolling_std_30"] = all_daily["orders_count"].rolling(30, min_periods=30).std()
    all_daily["z_score"] = (all_daily["orders_count"] - all_daily["rolling_mean_30"]) / all_daily["rolling_std_30"].replace(0, np.nan)
    all_daily["is_spike"] = all_daily["z_score"].abs() > 3
    plt.figure(figsize=(12, 4.8))
    plt.plot(all_daily["date"], all_daily["orders_count"], label="Daily delivered orders", alpha=0.55)
    plt.plot(all_daily["date"], all_daily["rolling_mean_30"], label="30-day rolling mean", color="black")
    spikes = all_daily[all_daily["is_spike"]]
    plt.scatter(spikes["date"], spikes["orders_count"], color="crimson", s=18, label="|z| > 3")
    plt.title("Event-driven demand spikes: daily orders with rolling z-score")
    plt.xlabel("Date")
    plt.ylabel("Orders")
    plt.legend()
    save_fig(FIGURES / "daily_spikes_overview.png")

    # Weekday figures.
    w = delivered.groupby("weekday").agg(orders_count=("order_id", "nunique"), revenue=("price", "sum")).reset_index()
    w["weekday_name"] = w["weekday"].map(WEEKDAY_NAMES)
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=w, x="weekday_name", y="orders_count", ax=ax1, color="#4C72B0")
    ax1.set_title("Weekly operational seasonality: orders by weekday")
    ax1.set_xlabel("Weekday")
    ax1.set_ylabel("Orders")
    save_fig(FIGURES / "weekday_orders.png")

    return ranking, cat_month


def product_seasonality(enriched: pd.DataFrame) -> pd.DataFrame:
    delivered = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    prod_month = (
        delivered.groupby(["product_id", "product_category_name_english", "month"])
        .agg(orders_count=("order_id", "nunique"), revenue=("price", "sum"), items_sold=("order_item_id", "count"))
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    prod_month.to_parquet(PROCESSED / "product_monthly_metrics.parquet", index=False)
    rows = []
    for (pid, cat), part in prod_month.groupby(["product_id", "category"]):
        y = part.set_index("month")["orders_count"].reindex(range(1, 13), fill_value=0).astype(float)
        total = y.sum()
        active = int((y > 0).sum())
        if total == 0:
            continue
        mean = y.mean()
        quarters = {1: y.loc[[1, 2, 3]].sum(), 2: y.loc[[4, 5, 6]].sum(), 3: y.loc[[7, 8, 9]].sum(), 4: y.loc[[10, 11, 12]].sum()}
        rows.append(
            {
                "product_id": pid,
                "category": cat,
                "total_orders": int(total),
                "total_revenue": part["revenue"].sum(),
                "active_months": active,
                "seasonality_score": y.std(ddof=0) / mean if mean else np.nan,
                "peak_to_mean": y.max() / mean if mean else np.nan,
                "peak_month": int(y.idxmax()),
                "peak_quarter": int(max(quarters, key=quarters.get)),
                "peak_quarter_share": float(max(quarters.values()) / total),
                "confidence_level": "high" if total >= 30 and active >= 4 else "medium" if total >= 20 and active >= 3 else "low",
            }
        )
    ranking = pd.DataFrame(rows).sort_values("seasonality_score", ascending=False)
    ranking.to_csv(PROCESSED / "product_seasonality_scores.csv", index=False)
    ranking.to_csv(TABLES / "product_seasonality_ranking.csv", index=False)
    return ranking


def business_impact(enriched: pd.DataFrame, ranking: pd.DataFrame, cat_month: pd.DataFrame) -> pd.DataFrame:
    delivered = enriched[enriched["is_delivered"]].copy()
    eligible = ranking[ranking["confidence_level"].ne("low")].copy()
    seasonal_threshold = eligible["seasonality_score"].quantile(0.75)
    seasonal_categories = set(eligible.loc[eligible["seasonality_score"].ge(seasonal_threshold), "category"])
    peak_months = set(
        cat_month[(cat_month["category"].isin(seasonal_categories)) & (cat_month["seasonal_index"].ge(1.25))]
        .assign(key=lambda x: x["category"] + "|" + x["month"].astype(str))["key"]
    )
    seasonal_data = delivered[delivered["product_category_name_english"].isin(seasonal_categories)].copy()
    seasonal_data["period_type"] = np.where(
        (seasonal_data["product_category_name_english"] + "|" + seasonal_data["month"].astype(str)).isin(peak_months),
        "peak_months",
        "normal_months",
    )
    p90 = order_level(enriched)["payment_value"].quantile(0.9)
    seasonal_data["large_purchase"] = seasonal_data["payment_value"].ge(p90)
    seasonal_data["installments_6_plus"] = seasonal_data["payment_installments"].ge(6)

    # Compare category-month averages, not raw totals across unequal period counts.
    # Otherwise peak months look smaller simply because there are fewer peak months.
    category_month = (
        seasonal_data.groupby(["product_category_name_english", "month", "period_type"])
        .agg(
            orders_count=("order_id", "nunique"),
            items_sold=("order_item_id", "count"),
            revenue=("price", "sum"),
            avg_order_value=("payment_value", "mean"),
            avg_item_price=("price", "mean"),
            large_purchase_share=("large_purchase", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            share_installments_6_plus=("installments_6_plus", "mean"),
            freight_share=("freight_share", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_review_score=("review_score", "mean"),
            share_low_reviews=("is_low_review", "mean"),
        )
        .reset_index()
    )
    grouped = category_month.groupby("period_type").agg(
        monthly_orders_per_category=("orders_count", "mean"),
        monthly_items_per_category=("items_sold", "mean"),
        monthly_revenue_per_category=("revenue", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_item_price=("avg_item_price", "mean"),
        large_purchase_share=("large_purchase_share", "mean"),
        avg_payment_installments=("avg_payment_installments", "mean"),
        share_installments_6_plus=("share_installments_6_plus", "mean"),
        freight_share=("freight_share", "mean"),
        avg_delivery_time_days=("avg_delivery_time_days", "mean"),
        avg_review_score=("avg_review_score", "mean"),
        share_low_reviews=("share_low_reviews", "mean"),
    )
    rows = []
    for metric in grouped.columns:
        normal = grouped.loc["normal_months", metric] if "normal_months" in grouped.index else np.nan
        peak = grouped.loc["peak_months", metric] if "peak_months" in grouped.index else np.nan
        rows.append({"metric": metric, "normal_months": normal, "peak_months": peak, "delta_abs": peak - normal, "delta_pct": (peak / normal - 1) if normal else np.nan})
    impact = pd.DataFrame(rows)
    impact.to_csv(PROCESSED / "business_impact_peak_vs_normal.csv", index=False)
    impact.to_csv(TABLES / "business_impact_peak_vs_normal.csv", index=False)

    plot_metrics = impact[impact["metric"].isin(["monthly_orders_per_category", "monthly_revenue_per_category", "avg_order_value", "large_purchase_share", "avg_payment_installments", "avg_delivery_time_days", "avg_review_score"])]
    plot_long = plot_metrics.melt(id_vars="metric", value_vars=["normal_months", "peak_months"], var_name="period", value_name="value")
    g = sns.catplot(data=plot_long, x="period", y="value", col="metric", kind="bar", col_wrap=4, sharey=False, height=3, aspect=1.05)
    g.fig.suptitle("Business impact: peak vs normal months for seasonal categories", y=1.02)
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=25)
    g.savefig(FIGURES / "08_peak_vs_normal_business_metrics.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)
    return impact


def large_purchases(enriched: pd.DataFrame) -> pd.DataFrame:
    orders = order_level(enriched)
    p75, p90 = orders["payment_value"].quantile([0.75, 0.90])
    orders["large_purchase_p75"] = orders["payment_value"].ge(p75)
    orders["large_purchase_p90"] = orders["payment_value"].ge(p90)
    monthly = (
        orders.groupby(["year", "month", "year_month"])
        .agg(
            orders_count=("order_id", "nunique"),
            large_purchase_count_p90=("large_purchase_p90", "sum"),
            large_purchase_share_p90=("large_purchase_p90", "mean"),
            large_purchase_share_p75=("large_purchase_p75", "mean"),
            avg_payment_value=("payment_value", "mean"),
            median_payment_value=("payment_value", "median"),
            avg_payment_installments=("payment_installments", "mean"),
            share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
        )
        .reset_index()
    )
    monthly.to_csv(PROCESSED / "large_purchase_monthly.csv", index=False)
    monthly.to_csv(TABLES / "large_purchase_monthly.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    sns.lineplot(data=monthly, x="year_month", y="large_purchase_share_p90", marker="o", ax=ax1, label="P90 large purchase share")
    ax1.set_ylabel("Large purchase share")
    ax1.tick_params(axis="x", rotation=65)
    ax2 = ax1.twinx()
    sns.lineplot(data=monthly, x="year_month", y="avg_payment_value", marker="s", ax=ax2, color="darkorange", label="Avg payment value")
    ax2.set_ylabel("Avg payment value, BRL")
    ax1.set_title("Large purchases and average payment value by month")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    ax2.legend_.remove() if ax2.legend_ else None
    save_fig(FIGURES / "09_large_purchase_share_by_month.png")
    return monthly


def prediction_block(enriched: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delivered = enriched[enriched["is_delivered"]].copy()
    features = (
        delivered.groupby("product_category_name_english")
        .agg(
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_freight_value=("freight_value", "mean"),
            freight_share=("freight_share", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
            avg_review_score=("review_score", "mean"),
            share_low_reviews=("is_low_review", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_product_weight_g=("product_weight_g", "mean"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
            avg_product_photos_qty=("product_photos_qty", "mean"),
            avg_product_name_length=("product_name_lenght", "mean"),
            avg_product_description_length=("product_description_lenght", "mean"),
            customer_state_nunique=("customer_state", "nunique"),
            seller_state_nunique=("seller_state", "nunique"),
            total_orders=("order_id", "nunique"),
            total_revenue=("price", "sum"),
        )
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    pred = features.merge(ranking[["category", "seasonality_score", "confidence_level", "seasonality_type"]], on="category", how="inner")
    eligible = pred[pred["confidence_level"].ne("low")].copy()
    threshold = eligible["seasonality_score"].quantile(0.75)
    pred["is_seasonal"] = pred["seasonality_score"].ge(threshold).astype(int)
    pred.to_csv(PROCESSED / "seasonality_prediction_features.csv", index=False)

    numeric_cols = [c for c in features.columns if c != "category"]
    model_df = pred[pred["confidence_level"].ne("low")].copy()
    X = model_df[numeric_cols]
    y = model_df["is_seasonal"]
    results_rows = []
    importance = pd.DataFrame({"feature": numeric_cols, "importance": 0.0})
    if len(model_df) >= 20 and y.nunique() == 2 and y.value_counts().min() >= 3:
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=stratify)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", min_samples_leaf=2)),
            ]
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred_label = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan
        results_rows.append({"model": "random_forest", "roc_auc": auc, "f1": f1_score(y_test, pred_label), "accuracy": accuracy_score(y_test, pred_label), "n_train": len(X_train), "n_test": len(X_test), "target_threshold": threshold})
        rf = model.named_steps["rf"]
        importance = pd.DataFrame({"feature": numeric_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure(figsize=(5.8, 5.2))
        plt.plot(fpr, tpr, label=f"RF ROC-AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.title("Seasonality prediction ROC curve")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        save_fig(FIGURES / "seasonality_prediction_roc.png")
    else:
        results_rows.append({"model": "not_enough_data", "roc_auc": np.nan, "f1": np.nan, "accuracy": np.nan, "n_train": 0, "n_test": len(model_df), "target_threshold": threshold})
        correlations = []
        for col in numeric_cols:
            correlations.append((col, abs(model_df[col].corr(model_df["seasonality_score"], method="spearman"))))
        importance = pd.DataFrame(correlations, columns=["feature", "importance"]).sort_values("importance", ascending=False)

    results = pd.DataFrame(results_rows)
    results.to_csv(PROCESSED / "seasonality_prediction_results.csv", index=False)
    results.to_csv(TABLES / "seasonality_prediction_results.csv", index=False)
    importance.to_csv(TABLES / "model_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.6))
    sns.barplot(data=importance.head(12).sort_values("importance"), y="feature", x="importance", color="#4C72B0")
    plt.title("Feature importance for predicting seasonal categories")
    plt.xlabel("Importance")
    plt.ylabel("")
    save_fig(FIGURES / "10_seasonality_feature_importance.png")
    return pred, results


def forecast_block(enriched: pd.DataFrame) -> pd.DataFrame:
    orders = order_level(enriched).copy()
    orders["week_start"] = pd.PeriodIndex(orders["week"], freq="W").start_time
    weekly = orders.groupby("week_start")["order_id"].nunique().asfreq("W-MON").fillna(0)
    weekly = weekly[weekly.index <= pd.Timestamp("2018-08-27")]
    rows = []
    forecast_df = pd.DataFrame()
    if len(weekly) >= 50:
        test_horizon = min(8, max(4, len(weekly) // 8))
        train, test = weekly.iloc[:-test_horizon], weekly.iloc[-test_horizon:]
        mean_forecast = pd.Series(train.tail(4).mean(), index=test.index)
        seasonal_naive = train.reindex(test.index - pd.Timedelta(weeks=52)).set_axis(test.index)
        if seasonal_naive.isna().any():
            seasonal_naive = pd.Series(train.tail(8).mean(), index=test.index)
        rows.append({"model": "last_4_week_mean", "mae": float((test - mean_forecast).abs().mean()), "mape": float(((test - mean_forecast).abs() / test.replace(0, np.nan)).mean())})
        rows.append({"model": "seasonal_naive_or_8w_mean", "mae": float((test - seasonal_naive).abs().mean()), "mape": float(((test - seasonal_naive).abs() / test.replace(0, np.nan)).mean())})
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 4), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarima = model.get_forecast(steps=test_horizon).summary_frame()["mean"].set_axis(test.index)
            rows.append({"model": "sarima_weekly_s4", "mae": float((test - sarima).abs().mean()), "mape": float(((test - sarima).abs() / test.replace(0, np.nan)).mean())})
        except Exception as exc:
            sarima = pd.Series(np.nan, index=test.index)
            rows.append({"model": f"sarima_failed: {type(exc).__name__}", "mae": np.nan, "mape": np.nan})
        forecast_df = pd.DataFrame({"actual": test, "last_4_week_mean": mean_forecast, "seasonal_naive_or_8w_mean": seasonal_naive, "sarima_weekly_s4": sarima}).reset_index().rename(columns={"week_start": "week_start"})
        plt.figure(figsize=(12, 4.8))
        plt.plot(weekly.index, weekly.values, label="Historical weekly orders", color="steelblue")
        plt.plot(test.index, mean_forecast, label="Last-4-week mean", linestyle="--")
        plt.plot(test.index, seasonal_naive, label="Seasonal naive / 8w mean", linestyle="--")
        if sarima.notna().any():
            plt.plot(test.index, sarima, label="SARIMA weekly s=4", linestyle="--")
        plt.axvspan(test.index.min(), test.index.max(), color="gray", alpha=0.12, label="Holdout")
        plt.title("Forecast example: weekly delivered orders")
        plt.xlabel("Week")
        plt.ylabel("Orders")
        plt.legend()
        save_fig(FIGURES / "optional_forecast_examples.png")
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "forecast_model_comparison.csv", index=False)
    if not forecast_df.empty:
        forecast_df.to_csv(TABLES / "forecast_holdout_predictions.csv", index=False)
    return results


def robustness_analysis(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for min_orders in [50, 100, 200]:
        subset = ranking[(ranking["total_orders"] >= min_orders) & (ranking["active_months"] >= 6)].copy()
        top10 = set(subset.head(10)["category"])
        rows.append({"setting": f"min_orders_{min_orders}", "n_categories": len(subset), "top10_categories": "; ".join(sorted(top10))})
    robust = pd.DataFrame(rows)
    robust.to_csv(TABLES / "seasonality_robustness.csv", index=False)
    return robust


def write_notes_and_outline(
    audit: pd.DataFrame,
    coverage: pd.DataFrame,
    monthly: pd.DataFrame,
    ranking: pd.DataFrame,
    product_ranking: pd.DataFrame,
    impact: pd.DataFrame,
    large: pd.DataFrame,
    pred_results: pd.DataFrame,
    forecast_results: pd.DataFrame,
) -> None:
    delivered_cov = coverage[coverage["scope"].eq("delivered")]
    top_seasonal = ranking[ranking["confidence_level"].eq("high")].head(10)
    if len(top_seasonal) < 5:
        top_seasonal = ranking[ranking["confidence_level"].ne("low")].head(10)
    top_stable = ranking[ranking["confidence_level"].ne("low")].sort_values("seasonality_score").head(10)
    top_large = large[(large["orders_count"] >= 500) & (large["year"] != 2016)].sort_values("large_purchase_share_p90", ascending=False).head(5)
    auc_text = "not enough data"
    if not pred_results.empty and pd.notna(pred_results.loc[0, "roc_auc"]):
        auc_text = f"ROC-AUC {pred_results.loc[0, 'roc_auc']:.2f}, F1 {pred_results.loc[0, 'f1']:.2f}"

    limitations = f"""# Limitations

## Date coverage

{table_to_markdown(delivered_cov)}

## Rules used in the final analysis

- Main demand analysis uses `order_status == delivered`.
- 2017 is the primary complete year for 12-month seasonality profiles.
- 2018 is used mainly as January-August context/validation.
- 2016 is context only: the year is too incomplete and sparse for strict seasonality claims.
- Category-level conclusions are filtered by confidence (`total_orders` and `active_months`) to avoid treating tiny categories as seasonal.
- Product-level output is an appendix because Olist product IDs do not include human-readable product names.
"""
    (NOTES / "limitations.md").write_text(limitations, encoding="utf-8")

    methodology = f"""# Methodology

## Analysis unit

The main unit is an item row in a delivered order, enriched with product, payment, review, customer, seller and date fields.

## Seasonality measurement

For each category in the complete 2017 window we compute monthly demand and derive:

- `seasonality_score = coefficient_of_variation(monthly_orders)`;
- `seasonal_index = monthly_orders / average_monthly_orders`;
- peak month and peak quarter;
- event spike count via 30-day rolling z-score;
- weekday pattern strength;
- trend strength;
- confidence level from total orders and active months.

## Seasonality types

Categories are assigned to rule-based types: `stable`, `single_month_spike`, `holiday_q4`, `early_year`, `mid_year`, `event_driven`, `trend_driven`, `weekly_pattern`, `mixed_seasonal`, or `low_confidence`.

## Business impact

For high/medium-confidence seasonal categories, peak months are months where the category seasonal index is at least 1.25. Metrics are compared between peak and normal months.

## Prediction block

`is_seasonal` is defined as top quartile by seasonality score among eligible categories. A random forest model estimates how well product/order/logistics/geography features explain seasonality. Result: {auc_text}.

## Forecast block

Forecasting is demonstrative because the dataset has only one complete year. Weekly total demand is compared across naive, seasonal-naive/rolling mean, and SARIMA-style baselines where fitting succeeds.
"""
    (NOTES / "methodology.md").write_text(methodology, encoding="utf-8")

    presentation = f"""# Presentation outline: Olist Store seasonality

## Slide 1 — Task

Question: how seasonality affects Olist demand, which products/categories are most seasonal, and how to account for this in demand forecasting.

## Slide 2 — Data and limitations

Use one Kaggle dataset: Brazilian E-Commerce Public Dataset by Olist. Main analysis uses delivered orders.

Key limitation: 2017 is the only nearly complete year; 2016 is sparse and 2018 ends before Q4.

Figure: `results/figures/01_data_coverage_by_year_month.png`.

## Slide 3 — Overall demand dynamics

Show orders and revenue by month.

Figures:
- `results/figures/02_orders_by_month.png`
- `results/figures/03_revenue_by_month.png`
- `results/figures/04_year_month_heatmap.png`

## Slide 4 — Seasonality method

Explain monthly profiles, seasonal index, CV score, peak month/quarter, event spikes, weekly patterns and confidence filters.

## Slide 5 — Most seasonal categories

Top high/medium-confidence seasonal categories:

{table_to_markdown(top_seasonal[["category", "total_orders", "seasonality_score", "peak_month", "peak_quarter", "seasonality_type", "confidence_level"]], 10)}

Figure: `results/figures/05_top_seasonal_categories.png`.

## Slide 6 — Seasonality types

Show category × month heatmap and cluster heatmap. Explain that we identify multiple forms: stable demand, single-month spikes, Q4/holiday, early-year, mid-year, event-driven, trend-driven/low-confidence.

Figures:
- `results/figures/06_category_month_seasonal_index_heatmap.png`
- `results/figures/07_seasonality_type_clusters.png`

## Slide 7 — Event-driven and weekly patterns

Use daily rolling z-score to find spikes and weekday chart for operational rhythm.

Figures:
- `results/figures/daily_spikes_overview.png`
- `results/figures/weekday_orders.png`

## Slide 8 — Business impact

Compare peak vs normal months for seasonal categories.

{table_to_markdown(impact[["metric", "normal_months", "peak_months", "delta_pct"]].round(3), 12)}

Figure: `results/figures/08_peak_vs_normal_business_metrics.png`.

## Slide 9 — Large purchases

Top months by P90 large-purchase share:

{table_to_markdown(top_large[["year_month", "orders_count", "large_purchase_share_p90", "avg_payment_value", "avg_payment_installments"]].round(3), 5)}

Figure: `results/figures/09_large_purchase_share_by_month.png`.

## Slide 10 — Can seasonality be predicted?

Use category-level feature table. Model result: {auc_text}. Interpret feature importance cautiously due to short history and small number of categories.

Figures:
- `results/figures/10_seasonality_feature_importance.png`
- `results/figures/seasonality_prediction_roc.png` if generated.

## Slide 11 — Forecasting implication

Forecast should be category-specific:

- stable categories: simple baseline may be enough;
- seasonal categories: use category-level seasonal profiles;
- spike-driven categories: use event calendar + anomaly-aware planning.

Figure: `results/figures/optional_forecast_examples.png`.

## Slide 12 — Final recommendations

1. Plan inventory and seller readiness 2–4 weeks before peak months.
2. Use category-level forecasts instead of one global forecast.
3. Track event-driven spike categories separately.
4. In large-purchase months, promote installments and high-ticket categories.
5. Monitor delivery time and reviews during peak periods, because demand peaks can affect customer experience.
"""
    (RESULTS / "presentation_outline.md").write_text(presentation, encoding="utf-8")

    summary = f"""# Final analytical summary

## Direct answers

### How does seasonality affect demand and which categories/products are strongest?

Seasonality is heterogeneous. The strongest high/medium-confidence categories by CV score are:

{table_to_markdown(top_seasonal[["category", "total_orders", "seasonality_score", "peak_month", "peak_quarter", "seasonality_type"]], 10)}

Product-level appendix is available in `results/tables/product_seasonality_ranking.csv`; product IDs should be interpreted through their category because Olist has no readable product names.

### What types of seasonality appear?

The project distinguishes monthly/calendar, quarterly, event-driven, weekly, trend-driven pseudo-seasonality, product lifecycle effects and regional/geographic differences. The final category table includes `seasonality_type`, `event_spike_count`, `weekly_pattern_strength`, `trend_strength` and `confidence_level`.

### Business impact

Peak months for seasonal categories are compared to normal months in `results/tables/business_impact_peak_vs_normal.csv`. The comparison covers orders, revenue, average order value, large purchases, installments, freight, delivery time and review score.

### Large purchases

Large purchases are defined as order-level `payment_value >= P90`. Monthly results are in `results/tables/large_purchase_monthly.csv`. Highest-share months:

{table_to_markdown(top_large[["year_month", "large_purchase_share_p90", "avg_payment_value", "share_installments_6_plus"]].round(3), 5)}

### Can seasonality be predicted?

A category-level feature model was trained with target `is_seasonal = top quartile by seasonality score`. Result: {auc_text}. Feature importances are in `results/tables/model_feature_importance.csv` and figure `results/figures/10_seasonality_feature_importance.png`.

### How to forecast demand?

Use category-specific planning: stable categories can use simple baselines; seasonal categories need seasonal profiles; spike-driven categories need event calendars and anomaly-aware planning. A demonstration is saved in `results/figures/optional_forecast_examples.png` and `results/tables/forecast_model_comparison.csv`.
"""
    (RESULTS / "final_summary.md").write_text(summary, encoding="utf-8")


def update_readme() -> None:
    path = ROOT / "README.md"
    original = path.read_text(encoding="utf-8") if path.exists() else "# Сезонность продаж Olist Store\n"
    marker = "\n## Финальный результат анализа\n"
    addition = """
## Финальный результат анализа

Полный анализ запускается командой:

```bash
make final-analysis
```

Ключевые итоговые артефакты:

- `data/processed/orders_items_enriched.parquet` — обогащённая item-level таблица.
- `results/tables/category_seasonality_ranking.csv` — рейтинг сезонности категорий с типом сезонности и confidence.
- `results/tables/product_seasonality_ranking.csv` — appendix по сезонным `product_id`.
- `results/tables/business_impact_peak_vs_normal.csv` — влияние peak months на бизнес-метрики.
- `results/tables/large_purchase_monthly.csv` — месяцы крупных покупок.
- `results/tables/model_feature_importance.csv` — признаки, связанные с сезонностью.
- `results/figures/` — финальные графики для презентации.
- `results/presentation_outline.md` — структура 8-минутной презентации.
- `results/final_summary.md` — прямые ответы на вопросы кейса.

Главное ограничение: 2017 — основной полный год; 2016 и конец 2018 нельзя использовать как полноценные сезонные периоды.
"""
    if marker in original:
        original = original.split(marker)[0].rstrip() + addition
    else:
        original = original.rstrip() + "\n" + addition
    path.write_text(original, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    tables = load_tables()
    audit = audit_tables(tables)
    enriched = build_enriched(tables)
    coverage = coverage_analysis(tables)
    monthly, weekly = aggregate_metrics(enriched)
    ranking, cat_month = category_seasonality(enriched)
    product_ranking = product_seasonality(enriched)
    impact = business_impact(enriched, ranking, cat_month)
    large = large_purchases(enriched)
    prediction_features, pred_results = prediction_block(enriched, ranking)
    forecast_results = forecast_block(enriched)
    robustness_analysis(ranking)
    write_notes_and_outline(audit, coverage, monthly, ranking, product_ranking, impact, large, pred_results, forecast_results)
    update_readme()
    print(json.dumps({
        "enriched_rows": int(len(enriched)),
        "category_rankings": int(len(ranking)),
        "product_rankings": int(len(product_ranking)),
        "figures": len(list(FIGURES.glob("*.png"))),
        "tables": len(list(TABLES.glob("*.csv"))),
        "presentation_outline": str(RESULTS / "presentation_outline.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
