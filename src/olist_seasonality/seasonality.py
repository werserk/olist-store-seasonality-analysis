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
from .metrics import order_level
from .utils import mode_or_nan, safe_div, save_fig, table_to_markdown

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
    save_fig(FIGURES / "08_daily_spikes_overview.png")

    # Weekday figures.
    w = delivered.groupby("weekday").agg(orders_count=("order_id", "nunique"), revenue=("price", "sum")).reset_index()
    w["weekday_name"] = w["weekday"].map(WEEKDAY_NAMES)
    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    sns.barplot(data=w, x="weekday_name", y="orders_count", ax=ax1, color="#4C72B0")
    ax1.set_title("Weekly operational seasonality: orders by weekday")
    ax1.set_xlabel("Weekday")
    ax1.set_ylabel("Orders")
    save_fig(FIGURES / "09_weekday_orders.png")

    return ranking, cat_month


def _top_n_by_period(
    data: pd.DataFrame,
    period_cols: list[str],
    entity_cols: list[str],
    n: int,
) -> pd.DataFrame:
    grouped = (
        data.groupby(period_cols + entity_cols, dropna=False)
        .agg(
            orders_count=("order_id", "nunique"),
            items_sold=("order_item_id", "count"),
            revenue=("price", "sum"),
        )
        .reset_index()
        .sort_values(
            period_cols + ["orders_count", "revenue", "items_sold"],
            ascending=[True] * len(period_cols) + [False, False, False],
        )
    )
    grouped["rank"] = grouped.groupby(period_cols).cumcount() + 1
    return grouped[grouped["rank"].le(n)].copy()


def granularity_artifacts(enriched: pd.DataFrame, ranking: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create line-chart and top-N artifacts across requested granularities."""
    delivered_2017 = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    delivered_2017["category"] = delivered_2017["product_category_name_english"].fillna("unknown")

    line_categories = (
        ranking[ranking["confidence_level"].ne("low")]
        .sort_values("seasonality_score", ascending=False)
        .head(8)["category"]
        .tolist()
    )
    line_data = delivered_2017[delivered_2017["category"].isin(line_categories)].copy()

    category_quarterly = (
        line_data.groupby(["category", "quarter"])
        .agg(orders_count=("order_id", "nunique"), items_sold=("order_item_id", "count"), revenue=("price", "sum"))
        .reset_index()
    )
    category_quarterly.to_csv(TABLES / "category_quarterly_line_metrics.csv", index=False)

    category_monthly = (
        line_data.groupby(["category", "month"])
        .agg(orders_count=("order_id", "nunique"), items_sold=("order_item_id", "count"), revenue=("price", "sum"))
        .reset_index()
    )
    category_monthly["month_name"] = category_monthly["month"].map(MONTH_NAMES)
    category_monthly.to_csv(TABLES / "category_monthly_line_metrics.csv", index=False)

    category_weekly = (
        line_data.groupby(["category", "week"])
        .agg(orders_count=("order_id", "nunique"), items_sold=("order_item_id", "count"), revenue=("price", "sum"))
        .reset_index()
        .sort_values(["week", "category"])
    )
    category_weekly["week_start"] = pd.PeriodIndex(category_weekly["week"], freq="W").start_time
    category_weekly.to_csv(TABLES / "category_weekly_line_metrics.csv", index=False)

    category_weekday = (
        line_data.groupby(["category", "weekday"])
        .agg(orders_count=("order_id", "nunique"), items_sold=("order_item_id", "count"), revenue=("price", "sum"))
        .reset_index()
    )
    category_weekday["weekday_name"] = category_weekday["weekday"].map(WEEKDAY_NAMES)
    category_weekday.to_csv(TABLES / "category_weekday_line_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.lineplot(data=category_quarterly, x="quarter", y="orders_count", hue="category", marker="o", ax=ax)
    ax.set_title("Seasonal categories by quarter: delivered orders, 2017")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Orders")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(title="Category", fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "15_category_quarterly_line_orders.png")

    fig, ax = plt.subplots(figsize=(12, 5.4))
    sns.lineplot(data=category_monthly, x="month", y="orders_count", hue="category", marker="o", ax=ax)
    ax.set_title("Seasonal categories by month: delivered orders, 2017")
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    ax.set_xticks(list(range(1, 13)))
    ax.legend(title="Category", fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "16_category_monthly_line_orders.png")

    fig, ax = plt.subplots(figsize=(13, 5.4))
    sns.lineplot(data=category_weekly, x="week_start", y="orders_count", hue="category", ax=ax)
    ax.set_title("Seasonal categories by week: delivered orders, 2017")
    ax.set_xlabel("Week")
    ax.set_ylabel("Orders")
    ax.legend(title="Category", fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "17_category_weekly_line_orders.png")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.lineplot(data=category_weekday, x="weekday", y="orders_count", hue="category", marker="o", ax=ax)
    ax.set_title("Seasonal categories by weekday: delivered orders, 2017")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Orders")
    ax.set_xticks(list(WEEKDAY_NAMES.keys()))
    ax.set_xticklabels([WEEKDAY_NAMES[i] for i in WEEKDAY_NAMES])
    ax.legend(title="Category", fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "18_category_weekday_line_orders.png")

    top_categories_by_month = _top_n_by_period(delivered_2017, ["month"], ["category"], 3)
    top_categories_by_month["month_name"] = top_categories_by_month["month"].map(MONTH_NAMES)
    top_categories_by_month.to_csv(TABLES / "top3_categories_by_month.csv", index=False)

    top_products_by_month = _top_n_by_period(delivered_2017, ["month"], ["product_id", "category"], 5)
    top_products_by_month["month_name"] = top_products_by_month["month"].map(MONTH_NAMES)
    top_products_by_month.to_csv(TABLES / "top5_products_by_month.csv", index=False)

    top_categories_by_quarter = _top_n_by_period(delivered_2017, ["quarter"], ["category"], 3)
    top_categories_by_quarter.to_csv(TABLES / "top3_categories_by_quarter.csv", index=False)

    top_products_by_quarter = _top_n_by_period(delivered_2017, ["quarter"], ["product_id", "category"], 5)
    top_products_by_quarter.to_csv(TABLES / "top5_products_by_quarter.csv", index=False)

    return {
        "category_quarterly": category_quarterly,
        "category_monthly": category_monthly,
        "category_weekly": category_weekly,
        "category_weekday": category_weekday,
        "top_categories_by_month": top_categories_by_month,
        "top_products_by_month": top_products_by_month,
        "top_categories_by_quarter": top_categories_by_quarter,
        "top_products_by_quarter": top_products_by_quarter,
    }


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
    g.savefig(FIGURES / "10_peak_vs_normal_business_metrics.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)
    return impact


def large_purchases(enriched: pd.DataFrame) -> pd.DataFrame:
    """Analyze item-level expensive goods and their revenue distribution."""
    items = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    thresholds = {
        "top5": 0.95,
        "top10": 0.90,
        "top15": 0.85,
        "top20": 0.80,
    }
    threshold_rows = []
    for label, quantile in thresholds.items():
        value = float(items["price"].quantile(quantile))
        items[f"large_item_{label}"] = items["price"].ge(value)
        items[f"large_revenue_{label}"] = np.where(items[f"large_item_{label}"], items["price"], 0.0)
        threshold_rows.append(
            {
                "threshold": label,
                "price_quantile": quantile,
                "min_item_price": value,
                "items_count": int(items[f"large_item_{label}"].sum()),
                "items_share": float(items[f"large_item_{label}"].mean()),
                "revenue": float(items[f"large_revenue_{label}"].sum()),
                "revenue_share": float(items[f"large_revenue_{label}"].sum() / items["price"].sum()),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(TABLES / "large_purchase_price_thresholds.csv", index=False)

    def aggregate(period_cols: list[str], name: str) -> pd.DataFrame:
        grouped = (
            items.groupby(period_cols, dropna=False)
            .agg(
                orders_count=("order_id", "nunique"),
                items_count=("order_item_id", "count"),
                total_revenue=("price", "sum"),
                avg_item_price=("price", "mean"),
                median_item_price=("price", "median"),
            )
            .reset_index()
        )
        for label in thresholds:
            stats = (
                items.groupby(period_cols, dropna=False)
                .agg(
                    **{
                        f"large_items_count_{label}": (f"large_item_{label}", "sum"),
                        f"large_revenue_{label}": (f"large_revenue_{label}", "sum"),
                    }
                )
                .reset_index()
            )
            grouped = grouped.merge(stats, on=period_cols, how="left")
            grouped[f"large_items_share_{label}"] = grouped[f"large_items_count_{label}"] / grouped["items_count"]
            grouped[f"large_revenue_share_of_period_{label}"] = grouped[f"large_revenue_{label}"] / grouped["total_revenue"]
            total_large_revenue = grouped[f"large_revenue_{label}"].sum()
            grouped[f"large_revenue_distribution_{label}"] = grouped[f"large_revenue_{label}"] / total_large_revenue if total_large_revenue else np.nan
        grouped.to_csv(PROCESSED / f"large_purchase_revenue_by_{name}.csv", index=False)
        grouped.to_csv(TABLES / f"large_purchase_revenue_by_{name}.csv", index=False)
        return grouped

    monthly = aggregate(["month", "year_month"], "month")
    quarterly = aggregate(["quarter"], "quarter")
    weekly = aggregate(["week"], "week")
    weekday = aggregate(["weekday", "weekday_name"], "weekday")

    # Backward-compatible summary name used by the report.
    monthly.to_csv(TABLES / "large_purchase_monthly.csv", index=False)

    figure_numbers = {
        "top10": {"month": 11, "quarter": 19, "week": 20, "weekday": 21},
        "top5": {"month": 22, "quarter": 23, "week": 24, "weekday": 25},
        "top15": {"month": 26, "quarter": 27, "week": 28, "weekday": 29},
        "top20": {"month": 30, "quarter": 31, "week": 32, "weekday": 33},
    }
    threshold_titles = {"top5": "top-5%", "top10": "top-10%", "top15": "top-15%", "top20": "top-20%"}

    def save_monthly_revenue_plot(label: str) -> None:
        title = threshold_titles[label]
        fig, ax1 = plt.subplots(figsize=(12, 4.8))
        sns.lineplot(
            data=monthly,
            x="month",
            y=f"large_revenue_distribution_{label}",
            marker="o",
            ax=ax1,
            label=f"{title} revenue distribution",
        )
        ax1.set_ylabel(f"Share of {title} item revenue")
        ax1.set_xlabel("Month")
        ax1.set_xticks(list(range(1, 13)))
        ax2 = ax1.twinx()
        sns.lineplot(
            data=monthly,
            x="month",
            y=f"large_revenue_share_of_period_{label}",
            marker="s",
            ax=ax2,
            color="darkorange",
            label=f"{title} share of month revenue",
        )
        ax2.set_ylabel("Share of month revenue")
        ax1.set_title(f"Revenue from {title} expensive goods by month, 2017")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        ax2.legend_.remove() if ax2.legend_ else None
        save_fig(FIGURES / f"{figure_numbers[label]['month']:02d}_large_purchase_revenue_{label}_by_month.png")

    def save_distribution_plot(df: pd.DataFrame, label: str, x_col: str, grain: str) -> None:
        title = threshold_titles[label]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        sns.lineplot(data=df, x=x_col, y=f"large_revenue_distribution_{label}", marker="o", ax=ax)
        ax.set_title(f"Distribution of {title} expensive-goods revenue by {grain}, 2017")
        ax.set_xlabel(grain.title())
        ax.set_ylabel(f"Share of {title} item revenue")
        if x_col == "weekday":
            ax.set_xticks(list(WEEKDAY_NAMES.keys()))
            ax.set_xticklabels([WEEKDAY_NAMES[i] for i in WEEKDAY_NAMES])
        save_fig(FIGURES / f"{figure_numbers[label][grain]:02d}_large_purchase_revenue_{label}_by_{grain}.png")

    weekly_plot = weekly.assign(week_start=pd.PeriodIndex(weekly["week"], freq="W").start_time)
    weekday_plot = weekday.sort_values("weekday")
    for label in thresholds:
        save_monthly_revenue_plot(label)
        save_distribution_plot(quarterly, label, "quarter", "quarter")
        save_distribution_plot(weekly_plot, label, "week_start", "week")
        save_distribution_plot(weekday_plot, label, "weekday", "weekday")

    return monthly


def robustness_analysis(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for min_orders in [50, 100, 200]:
        subset = ranking[(ranking["total_orders"] >= min_orders) & (ranking["active_months"] >= 6)].copy()
        top10 = set(subset.head(10)["category"])
        rows.append({"setting": f"min_orders_{min_orders}", "n_categories": len(subset), "top10_categories": "; ".join(sorted(top10))})
    robust = pd.DataFrame(rows)
    robust.to_csv(TABLES / "seasonality_robustness.csv", index=False)
    return robust
