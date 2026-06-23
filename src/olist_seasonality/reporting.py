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

from .config import FIGURES, MONTH_NAMES, NOTES, PROCESSED, RESULTS, ROOT, TABLES, WEEKDAY_NAMES
from .utils import mode_or_nan, safe_div, save_fig, table_to_markdown

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

Figure: `results/figures/10_peak_vs_normal_business_metrics.png`.

## Slide 9 — Large purchases

Top months by P90 large-purchase share:

{table_to_markdown(top_large[["year_month", "orders_count", "large_purchase_share_p90", "avg_payment_value", "avg_payment_installments"]].round(3), 5)}

Figure: `results/figures/11_large_purchase_share_by_month.png`.

## Slide 10 — Can seasonality be predicted?

Use category-level feature table. Model result: {auc_text}. Interpret feature importance cautiously due to short history and small number of categories.

Figures:
- `results/figures/12_seasonality_prediction_roc.png` if generated.
- `results/figures/13_seasonality_feature_importance.png`

## Slide 11 — Forecasting implication

Forecast should be category-specific:

- stable categories: simple baseline may be enough;
- seasonal categories: use category-level seasonal profiles;
- spike-driven categories: use event calendar + anomaly-aware planning.

Figure: `results/figures/14_optional_forecast_examples.png`.

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

A category-level feature model was trained with target `is_seasonal = top quartile by seasonality score`. Result: {auc_text}. Feature importances are in `results/tables/model_feature_importance.csv` and figure `results/figures/13_seasonality_feature_importance.png`.

### How to forecast demand?

Use category-specific planning: stable categories can use simple baselines; seasonal categories need seasonal profiles; spike-driven categories need event calendars and anomaly-aware planning. A demonstration is saved in `results/figures/14_optional_forecast_examples.png` and `results/tables/forecast_model_comparison.csv`.
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

Архитектура pipeline:

- `scripts/run_final_analysis.py` — тонкая точка входа.
- `src/olist_seasonality/data.py` — загрузка, audit, enrich, coverage.
- `src/olist_seasonality/metrics.py` — order-level и агрегированные метрики.
- `src/olist_seasonality/seasonality.py` — сезонность категорий/товаров, business impact, крупные покупки.
- `src/olist_seasonality/modeling.py` — prediction и forecast blocks.
- `src/olist_seasonality/reporting.py` — итоговые markdown-артефакты и README.
- `notebooks/final_analysis.ipynb` — notebook-витрина для запуска pipeline и просмотра графиков/таблиц.

Ключевые итоговые артефакты:

- `data/processed/orders_items_enriched.parquet` — обогащённая item-level таблица.
- `results/tables/category_seasonality_ranking.csv` — рейтинг сезонности категорий с типом сезонности и confidence.
- `results/tables/product_seasonality_ranking.csv` — appendix по сезонным `product_id`.
- `results/tables/business_impact_peak_vs_normal.csv` — влияние peak months на бизнес-метрики.
- `results/tables/large_purchase_monthly.csv` — месяцы крупных покупок.
- `results/tables/model_feature_importance.csv` — признаки, связанные с сезонностью.
- `results/figures/` — пронумерованные финальные графики для презентации (`01_...png`, `02_...png`, ...).
- `results/presentation_outline.md` — структура 8-минутной презентации.
- `results/final_summary.md` — прямые ответы на вопросы кейса.

Главное ограничение: 2017 — основной полный год; 2016 и конец 2018 нельзя использовать как полноценные сезонные периоды.
"""
    if marker in original:
        original = original.split(marker)[0].rstrip() + addition
    else:
        original = original.rstrip() + "\n" + addition
    path.write_text(original, encoding="utf-8")
