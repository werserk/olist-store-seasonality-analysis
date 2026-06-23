from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import FIGURES, MONTH_NAMES, RESULTS, TABLES, WEEKDAY_NAMES
from .utils import save_fig


def _seasonal_category_sets(ranking: pd.DataFrame, cat_month: pd.DataFrame) -> tuple[pd.DataFrame, set[str], set[str]]:
    eligible = ranking[ranking["confidence_level"].ne("low")].copy()
    threshold = eligible["seasonality_score"].quantile(0.75)
    seasonal = eligible[eligible["seasonality_score"].ge(threshold)].copy()
    seasonal_categories = set(seasonal["category"])
    peak_keys = set(
        cat_month[(cat_month["category"].isin(seasonal_categories)) & (cat_month["seasonal_index"].ge(1.25))]
        .assign(key=lambda x: x["category"] + "|" + x["month"].astype(str))["key"]
    )
    return seasonal, seasonal_categories, peak_keys


def seasonality_completion_figures(
    ranking: pd.DataFrame,
    cat_month: pd.DataFrame,
    product_ranking: pd.DataFrame,
) -> None:
    """Create direct-answer seasonality figures missing from the final graph plan."""
    eligible = ranking[ranking["confidence_level"].ne("low")].copy()

    # G01: ranked chart with peak-month labels.
    top = eligible.sort_values("seasonality_score", ascending=False).head(15).sort_values("seasonality_score")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.barplot(data=top, y="category", x="seasonality_score", hue="seasonality_type", dodge=False, ax=ax)
    for patch, peak_month in zip(ax.patches, top["peak_month"]):
        width = getattr(patch, "get_width")()
        y = getattr(patch, "get_y")() + getattr(patch, "get_height")() / 2
        ax.text(width + 0.015, y, f"M{int(peak_month)}", va="center", fontsize=8)
    ax.set_title("Most seasonal categories with peak month labels (2017)")
    ax.set_xlabel("Seasonality score: CV of monthly delivered orders")
    ax.set_ylabel("")
    ax.legend(title="Type", fontsize=7, title_fontsize=8, loc="lower right")
    save_fig(FIGURES / "35_top_seasonal_categories_with_peak_month.png")

    # G02: monthly profiles for top 6 categories.
    top6 = eligible.sort_values("seasonality_score", ascending=False).head(6)["category"].tolist()
    profiles = cat_month[cat_month["category"].isin(top6)].copy()
    g = sns.relplot(
        data=profiles,
        x="month",
        y="orders_count",
        col="category",
        col_wrap=3,
        kind="line",
        marker="o",
        height=3.0,
        aspect=1.25,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Month", "Delivered orders")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.set_xticks(list(range(1, 13)))
    g.fig.suptitle("Monthly demand profiles for top seasonal categories", y=1.03)
    g.savefig(FIGURES / "36_top6_category_monthly_profiles.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)

    # G03: composition of seasonality types by confidence.
    type_counts = (
        ranking.groupby(["seasonality_type", "confidence_level"])
        .size()
        .rename("categories_count")
        .reset_index()
    )
    type_counts.to_csv(TABLES / "seasonality_type_composition.csv", index=False)
    order = type_counts.groupby("seasonality_type")["categories_count"].sum().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=type_counts, x="seasonality_type", y="categories_count", hue="confidence_level", order=order, ax=ax)
    ax.set_title("Seasonality type composition by confidence level")
    ax.set_xlabel("Seasonality type")
    ax.set_ylabel("Categories")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Confidence")
    save_fig(FIGURES / "37_seasonality_type_composition.png")

    # G04: product appendix.
    prod = product_ranking[product_ranking["confidence_level"].ne("low")].copy()
    prod = prod.sort_values("seasonality_score", ascending=False).head(20).copy()
    prod["product_short"] = prod["product_id"].str[:8]
    fig, ax = plt.subplots(figsize=(11, 7.0))
    sns.barplot(data=prod.sort_values("seasonality_score"), y="product_short", x="seasonality_score", hue="category", dodge=False, ax=ax)
    ax.set_title("Appendix: top seasonal product_id values (short IDs)")
    ax.set_xlabel("Seasonality score: CV of monthly orders")
    ax.set_ylabel("product_id prefix")
    ax.legend(title="Category", fontsize=6, title_fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "38_top20_seasonal_products_appendix.png")


def business_completion_figures(enriched: pd.DataFrame, ranking: pd.DataFrame, cat_month: pd.DataFrame) -> None:
    """Create paired business-impact figures by category."""
    delivered = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    seasonal, seasonal_categories, peak_keys = _seasonal_category_sets(ranking, cat_month)
    p90 = delivered["price"].quantile(0.9)
    delivered["category"] = delivered["product_category_name_english"].fillna("unknown")
    seasonal_data = delivered[delivered["category"].isin(seasonal_categories)].copy()
    seasonal_data["period_type"] = np.where(
        (seasonal_data["category"] + "|" + seasonal_data["month"].astype(str)).isin(peak_keys),
        "peak_months",
        "normal_months",
    )
    seasonal_data["large_item_top10"] = seasonal_data["price"].ge(p90)
    seasonal_data["installments_6_plus"] = seasonal_data["payment_installments"].ge(6)

    category_month = (
        seasonal_data.groupby(["category", "month", "period_type"])
        .agg(
            orders_count=("order_id", "nunique"),
            items_sold=("order_item_id", "count"),
            revenue=("price", "sum"),
            avg_item_price=("price", "mean"),
            large_purchase_share=("large_item_top10", "mean"),
            avg_payment_installments=("payment_installments", "mean"),
            avg_delivery_time_days=("delivery_time_days", "mean"),
            avg_review_score=("review_score", "mean"),
        )
        .reset_index()
    )
    category_month["items_per_order"] = category_month["items_sold"] / category_month["orders_count"]

    metrics = [
        "orders_count",
        "revenue",
        "avg_item_price",
        "large_purchase_share",
        "avg_payment_installments",
        "avg_delivery_time_days",
        "avg_review_score",
    ]
    rows = []
    for category, part in category_month.groupby("category"):
        normal = part[part["period_type"].eq("normal_months")][metrics].mean()
        peak = part[part["period_type"].eq("peak_months")][metrics].mean()
        if peak.isna().all() or normal.isna().all():
            continue
        for metric in metrics:
            n = normal[metric]
            p = peak[metric]
            rows.append(
                {
                    "category": category,
                    "metric": metric,
                    "normal_months": n,
                    "peak_months": p,
                    "delta_abs": p - n,
                    "delta_pct": (p / n - 1) if pd.notna(n) and n else np.nan,
                }
            )
    uplift = pd.DataFrame(rows)
    uplift.to_csv(TABLES / "business_impact_by_category_uplift.csv", index=False)

    plot_metrics = ["orders_count", "revenue", "avg_item_price", "large_purchase_share", "avg_payment_installments"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    plot = uplift[uplift["metric"].isin(plot_metrics)].copy()
    plot["delta_pct_display"] = plot["delta_pct"] * 100
    sns.boxplot(data=plot, x="metric", y="delta_pct_display", ax=ax, color="#9ecae1")
    sns.stripplot(data=plot, x="metric", y="delta_pct_display", ax=ax, color="#2b4c7e", alpha=0.65, size=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Peak-month uplift by category for seasonal categories")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Peak vs normal change, %")
    ax.tick_params(axis="x", rotation=25)
    save_fig(FIGURES / "39_business_impact_by_category_uplift.png")

    decomposition_metrics = ["orders_count", "items_per_order", "avg_item_price", "revenue"]
    decomp = category_month.groupby("period_type")[decomposition_metrics].mean().reset_index()
    decomp.to_csv(TABLES / "business_revenue_decomposition_peak_vs_normal.csv", index=False)
    decomp_long = decomp.melt(id_vars="period_type", var_name="metric", value_name="value")
    g = sns.catplot(data=decomp_long, x="period_type", y="value", col="metric", kind="bar", col_wrap=4, sharey=False, height=3, aspect=1.0)
    g.fig.suptitle("Revenue decomposition: peak vs normal months", y=1.04)
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=25)
    g.savefig(FIGURES / "40_revenue_decomposition_peak_vs_normal.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)


def early_prediction_completion_figures(enriched: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Evaluate whether H1 signals can predict H2 seasonality."""
    delivered = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    delivered["category"] = delivered["product_category_name_english"].fillna("unknown")
    h1 = delivered[delivered["month"].between(1, 6)].copy()
    h2 = delivered[delivered["month"].between(7, 12)].copy()

    h1_features = (
        h1.groupby("category")
        .agg(
            h1_orders=("order_id", "nunique"),
            h1_revenue=("price", "sum"),
            h1_avg_item_price=("price", "mean"),
            h1_avg_payment_installments=("payment_installments", "mean"),
            h1_share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
            h1_freight_share=("freight_share", "mean"),
            h1_avg_review_score=("review_score", "mean"),
            h1_customer_state_nunique=("customer_state", "nunique"),
            h1_seller_state_nunique=("seller_state", "nunique"),
        )
        .reset_index()
    )
    jan = h1[h1["month"].eq(1)].groupby("category")["order_id"].nunique()
    jun = h1[h1["month"].eq(6)].groupby("category")["order_id"].nunique()
    growth = ((jun + 1) / (jan + 1) - 1).rename("h1_growth_jan_to_jun")
    weekday = h1.groupby(["category", "weekday"])["order_id"].nunique().reset_index()
    weekday["weekday_index"] = weekday.groupby("category")["order_id"].transform(lambda x: x / x.mean() if x.mean() else np.nan)
    weekday_strength = weekday.groupby("category")["weekday_index"].agg(lambda x: x.max() - x.min()).rename("h1_weekday_concentration")
    h1_features = h1_features.merge(growth, on="category", how="left").merge(weekday_strength, on="category", how="left")

    h2_month = h2.groupby(["category", "month"])["order_id"].nunique().rename("orders_count").reset_index()
    rows = []
    for category, part in h2_month.groupby("category"):
        y = part.set_index("month")["orders_count"].reindex(range(7, 13), fill_value=0).astype(float)
        mean = y.mean()
        rows.append({"category": category, "h2_seasonality_score": y.std(ddof=0) / mean if mean else np.nan, "h2_orders": int(y.sum())})
    h2_target = pd.DataFrame(rows)
    pred = h1_features.merge(h2_target, on="category", how="inner").merge(
        ranking[["category", "confidence_level", "seasonality_type", "seasonality_score"]], on="category", how="left"
    )
    eligible = pred[(pred["confidence_level"].ne("low")) & (pred["h1_orders"].ge(20)) & (pred["h2_orders"].ge(20))].copy()
    threshold = eligible["h2_seasonality_score"].quantile(0.75)
    pred["is_h2_seasonal"] = pred["h2_seasonality_score"].ge(threshold).astype(int)
    eligible["is_h2_seasonal"] = eligible["h2_seasonality_score"].ge(threshold).astype(int)
    pred.to_csv(TABLES / "early_seasonality_prediction_features.csv", index=False)

    numeric_cols = [c for c in h1_features.columns if c != "category"]
    results_rows = []
    importance = pd.DataFrame({"feature": numeric_cols, "importance": 0.0})
    roc_created = False
    if len(eligible) >= 18 and eligible["is_h2_seasonal"].nunique() == 2 and eligible["is_h2_seasonal"].value_counts().min() >= 3:
        X = eligible[numeric_cols]
        y = eligible["is_h2_seasonal"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2)),
            ]
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred_label = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan
        results_rows.append({"model": "h1_to_h2_random_forest", "roc_auc": auc, "f1": f1_score(y_test, pred_label), "accuracy": accuracy_score(y_test, pred_label), "n_train": len(X_train), "n_test": len(X_test), "target_threshold": threshold})
        importance = pd.DataFrame({"feature": numeric_cols, "importance": model.named_steps["rf"].feature_importances_}).sort_values("importance", ascending=False)
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.plot(fpr, tpr, label=f"H1→H2 RF ROC-AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title("Early seasonality prediction: H1 signals → H2 seasonality")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        save_fig(FIGURES / "41_early_prediction_roc.png")
        roc_created = True
    else:
        results_rows.append({"model": "not_enough_data", "roc_auc": np.nan, "f1": np.nan, "accuracy": np.nan, "n_train": 0, "n_test": len(eligible), "target_threshold": threshold})
        correlations = []
        for col in numeric_cols:
            correlations.append((col, abs(eligible[col].corr(eligible["h2_seasonality_score"], method="spearman"))))
        importance = pd.DataFrame(correlations, columns=["feature", "importance"]).sort_values("importance", ascending=False)

    results = pd.DataFrame(results_rows)
    results.to_csv(TABLES / "early_seasonality_prediction_results.csv", index=False)
    importance.to_csv(TABLES / "early_seasonality_feature_importance.csv", index=False)

    if not roc_created:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        sns.histplot(data=eligible, x="h2_seasonality_score", hue="is_h2_seasonal", bins=12, ax=ax)
        ax.set_title("Early prediction target distribution: H2 seasonality")
        save_fig(FIGURES / "41_early_prediction_roc.png")

    fig, ax = plt.subplots(figsize=(9, 5.6))
    sns.barplot(data=importance.head(12).sort_values("importance"), y="feature", x="importance", color="#4C72B0", ax=ax)
    ax.set_title("Early prediction feature importance: H1 signals")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    save_fig(FIGURES / "42_early_prediction_feature_importance.png")

    score_cols = ["h1_growth_jan_to_jun", "h1_weekday_concentration", "h1_avg_item_price", "h1_avg_payment_installments", "h1_customer_state_nunique"]
    score_source = pred.copy()
    for col in score_cols:
        s = score_source[col].replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() > 1 and s.max() != s.min():
            score_source[f"{col}_rank"] = s.rank(pct=True)
        else:
            score_source[f"{col}_rank"] = 0.0
    rank_cols = [f"{c}_rank" for c in score_cols]
    score_source["early_warning_score"] = score_source[rank_cols].mean(axis=1)
    score_source.to_csv(TABLES / "early_warning_scorecard.csv", index=False)
    plot = score_source[score_source["confidence_level"].ne("low")].copy()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.scatterplot(data=plot, x="early_warning_score", y="seasonality_score", hue="seasonality_type", size="h1_orders", sizes=(30, 220), ax=ax)
    ax.set_title("Explainable early warning score vs final seasonality")
    ax.set_xlabel("Early warning score from Jan–Jun signals")
    ax.set_ylabel("Final 2017 seasonality score")
    ax.legend(fontsize=7, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(FIGURES / "43_early_warning_score_vs_final_seasonality.png")


def large_purchase_completion_figures(enriched: pd.DataFrame, ranking: pd.DataFrame, cat_month: pd.DataFrame) -> None:
    """Create missing large-purchase explanatory figures."""
    items = enriched[(enriched["is_delivered"]) & (enriched["year"].eq(2017))].copy()
    items["category"] = items["product_category_name_english"].fillna("unknown")
    p90 = items["price"].quantile(0.90)
    p80 = items["price"].quantile(0.80)
    items["large_item_top10"] = items["price"].ge(p90)
    items["large_item_top20"] = items["price"].ge(p80)
    items["order_date"] = pd.to_datetime(items["order_date"])

    # G10: Black Friday / Q4 event window.
    window = items[(items["order_date"] >= "2017-10-15") & (items["order_date"] <= "2017-12-15")].copy()
    daily = (
        window.groupby("order_date")
        .agg(
            all_revenue=("price", "sum"),
            top10_revenue=("price", lambda s: s[window.loc[s.index, "large_item_top10"]].sum()),
            top20_revenue=("price", lambda s: s[window.loc[s.index, "large_item_top20"]].sum()),
        )
        .reset_index()
    )
    daily.to_csv(TABLES / "large_purchase_black_friday_window.csv", index=False)
    daily_long = daily.melt(id_vars="order_date", var_name="segment", value_name="revenue")
    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.lineplot(data=daily_long, x="order_date", y="revenue", hue="segment", ax=ax)
    ax.axvline(pd.Timestamp("2017-11-24"), color="crimson", linestyle="--", linewidth=1.4, label="Black Friday 2017")
    ax.set_title("Daily revenue around Black Friday/Q4: all vs expensive goods")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily item revenue")
    ax.legend(title="Segment")
    save_fig(FIGURES / "44_large_purchase_black_friday_window.png")

    # G11: expensive-goods category mix in peak vs normal months.
    _, seasonal_categories, peak_keys = _seasonal_category_sets(ranking, cat_month)
    seasonal_items = items[items["category"].isin(seasonal_categories)].copy()
    seasonal_items["period_type"] = np.where(
        (seasonal_items["category"] + "|" + seasonal_items["month"].astype(str)).isin(peak_keys),
        "peak_months",
        "normal_months",
    )
    mix = (
        seasonal_items[seasonal_items["large_item_top10"]]
        .groupby(["period_type", "category"])
        .agg(top10_expensive_revenue=("price", "sum"), top10_expensive_items=("order_item_id", "count"))
        .reset_index()
    )
    mix["period_revenue_share"] = mix["top10_expensive_revenue"] / mix.groupby("period_type")["top10_expensive_revenue"].transform("sum")
    mix.to_csv(TABLES / "large_purchase_category_mix_peak_vs_normal.csv", index=False)
    top_categories = mix.groupby("category")["top10_expensive_revenue"].sum().sort_values(ascending=False).head(10).index
    plot = mix[mix["category"].isin(top_categories)].copy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.barplot(data=plot, y="category", x="period_revenue_share", hue="period_type", ax=ax)
    ax.set_title("Top-10 expensive-goods category mix: peak vs normal months")
    ax.set_xlabel("Share of top-10 expensive revenue within period")
    ax.set_ylabel("")
    ax.legend(title="Period")
    save_fig(FIGURES / "45_large_purchase_category_mix_peak_vs_normal.png")

    # G12: installments for all vs top-10 expensive purchases by month.
    inst = (
        items.groupby("month")
        .agg(
            all_avg_installments=("payment_installments", "mean"),
            all_share_installments_6_plus=("payment_installments", lambda x: x.ge(6).mean()),
            top10_avg_installments=("payment_installments", lambda x: x[items.loc[x.index, "large_item_top10"]].mean()),
            top10_share_installments_6_plus=("payment_installments", lambda x: x[items.loc[x.index, "large_item_top10"]].ge(6).mean()),
        )
        .reset_index()
    )
    inst.to_csv(TABLES / "large_purchase_installments_by_month.csv", index=False)
    inst_long = inst.melt(id_vars="month", var_name="metric", value_name="value")
    inst_long["measure"] = np.where(inst_long["metric"].str.contains("avg"), "Average installments", "Share with 6+ installments")
    inst_long["segment"] = np.where(inst_long["metric"].str.startswith("top10"), "Top-10% expensive goods", "All purchases")
    g = sns.relplot(data=inst_long, x="month", y="value", hue="segment", col="measure", kind="line", marker="o", facet_kws={"sharey": False}, height=4.0, aspect=1.35)
    g.set_axis_labels("Month", "Value")
    for ax in g.axes.flatten():
        ax.set_xticks(list(range(1, 13)))
    g.fig.suptitle("Installment behavior: all purchases vs top-10 expensive goods", y=1.04)
    g.savefig(FIGURES / "46_large_purchase_installments_by_month.png", bbox_inches="tight", dpi=180)
    plt.close(g.fig)


def refresh_final_figure_set() -> None:
    """Rebuild cleaned presentation-first figure bundle."""
    output = RESULTS / "final_figure_set"
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("*.png"):
        path.unlink()
    figures = [
        ("01_data_coverage_main_year.png", "01_data_coverage_by_year_month.png", "Data coverage: why 2017 is the main analysis year"),
        ("02_overall_orders_by_month.png", "02_orders_by_month.png", "Overall demand: delivered orders by month"),
        ("03_overall_revenue_by_month.png", "03_revenue_by_month.png", "Overall sales: revenue by month"),
        ("04_top_seasonal_categories_with_peak_month.png", "35_top_seasonal_categories_with_peak_month.png", "Most seasonal categories with peak month labels"),
        ("05_top6_category_monthly_profiles.png", "36_top6_category_monthly_profiles.png", "Monthly profiles for top seasonal categories"),
        ("06_category_monthly_seasonal_index_heatmap.png", "06_category_month_seasonal_index_heatmap.png", "Category × month seasonal index heatmap"),
        ("07_seasonality_type_composition.png", "37_seasonality_type_composition.png", "Composition of seasonality types"),
        ("08_daily_event_spikes.png", "08_daily_spikes_overview.png", "Daily event-driven spikes via rolling z-score"),
        ("09_peak_vs_normal_business_metrics.png", "10_peak_vs_normal_business_metrics.png", "Business metrics: peak vs normal months"),
        ("10_business_impact_by_category_uplift.png", "39_business_impact_by_category_uplift.png", "Paired peak uplift by category"),
        ("11_revenue_decomposition_peak_vs_normal.png", "40_revenue_decomposition_peak_vs_normal.png", "Revenue decomposition for peak vs normal months"),
        ("12_large_purchase_top10_by_month.png", "11_large_purchase_revenue_top10_by_month.png", "Top-10 expensive goods revenue by month"),
        ("13_large_purchase_black_friday_window.png", "44_large_purchase_black_friday_window.png", "Daily expensive-goods revenue around Black Friday"),
        ("14_large_purchase_category_mix_peak_vs_normal.png", "45_large_purchase_category_mix_peak_vs_normal.png", "Category mix of expensive purchases"),
        ("15_weekday_revenue_distribution_all_top20_top10.png", "34_weekday_revenue_distribution_all_top20_top10.png", "Weekday revenue distribution: all vs top-20/top-10"),
        ("16_early_prediction_roc.png", "41_early_prediction_roc.png", "Early H1→H2 seasonality prediction"),
        ("17_early_prediction_feature_importance.png", "42_early_prediction_feature_importance.png", "Early prediction feature importance"),
        ("18_early_warning_score_vs_final_seasonality.png", "43_early_warning_score_vs_final_seasonality.png", "Explainable early warning score"),
        ("19_forecast_examples.png", "14_optional_forecast_examples.png", "Forecasting examples and implication"),
    ]
    lines = [
        "# Final cleaned figure set",
        "",
        "Presentation-first subset of the generated figures. Appendix and sensitivity plots remain in `results/figures/`.",
        "",
        "## Figures",
        "",
    ]
    for dest_name, source_name, description in figures:
        source = FIGURES / source_name
        if not source.exists():
            raise FileNotFoundError(f"Missing source figure: {source}")
        shutil.copy2(source, output / dest_name)
        lines.append(f"- `{dest_name}` — {description}. Source: `results/figures/{source_name}`.")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
