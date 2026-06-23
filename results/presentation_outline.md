# Presentation outline: Olist Store seasonality

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

| category               |   total_orders |   seasonality_score |   peak_month |   peak_quarter | seasonality_type   | confidence_level   |
|:-----------------------|---------------:|--------------------:|-------------:|---------------:|:-------------------|:-------------------|
| stationery             |            897 |            0.780273 |           12 |              4 | single_month_spike | high               |
| electronics            |            810 |            0.778976 |           12 |              4 | event_driven       | high               |
| home_appliances        |            232 |            0.737056 |            6 |              3 | event_driven       | high               |
| watches_gifts          |           2073 |            0.703136 |           11 |              4 | event_driven       | high               |
| consoles_games         |            640 |            0.646867 |           11 |              4 | event_driven       | high               |
| garden_tools           |           1934 |            0.623369 |           11 |              4 | event_driven       | high               |
| toys                   |           2404 |            0.614123 |           11 |              4 | event_driven       | high               |
| books_general_interest |            214 |            0.594689 |           11 |              4 | event_driven       | high               |
| home_confort           |            246 |            0.580433 |           11 |              3 | event_driven       | high               |
| musical_instruments    |            241 |            0.572599 |           11 |              4 | event_driven       | high               |

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

| metric                       |   normal_months |   peak_months |   delta_pct |
|:-----------------------------|----------------:|--------------:|------------:|
| monthly_orders_per_category  |         120.282 |        88.382 |      -0.265 |
| monthly_items_per_category   |         131.388 |        95.765 |      -0.271 |
| monthly_revenue_per_category |       18657.5   |     17388.3   |      -0.068 |
| avg_order_value              |         308.395 |       320.1   |       0.038 |
| avg_item_price               |         226.199 |       250.475 |       0.107 |
| large_purchase_share         |           0.21  |         0.244 |       0.162 |
| avg_payment_installments     |           3.078 |         3.305 |       0.074 |
| share_installments_6_plus    |           0.186 |         0.214 |       0.148 |
| freight_share                |           0.222 |         0.201 |      -0.095 |
| avg_delivery_time_days       |          12.447 |        13.15  |       0.056 |
| avg_review_score             |           4.13  |         4.064 |      -0.016 |
| share_low_reviews            |           0.136 |         0.14  |       0.033 |

Figure: `results/figures/10_peak_vs_normal_business_metrics.png`.

## Slide 9 — Large purchases

Top months by P90 large-purchase share:

| year_month   |   orders_count |   large_purchase_share_p90 |   avg_payment_value |   avg_payment_installments |
|:-------------|---------------:|---------------------------:|--------------------:|---------------------------:|
| 2017-01      |            750 |                      0.14  |             170.061 |                      2.975 |
| 2017-04      |           2303 |                      0.115 |             169.758 |                      3.173 |
| 2017-10      |           4478 |                      0.113 |             167.74  |                      2.947 |
| 2017-09      |           4150 |                      0.11  |             168.957 |                      3.076 |
| 2017-03      |           2546 |                      0.108 |             162.753 |                      2.973 |

Figure: `results/figures/11_large_purchase_share_by_month.png`.

## Slide 10 — Can seasonality be predicted?

Use category-level feature table. Model result: ROC-AUC 0.33, F1 0.00. Interpret feature importance cautiously due to short history and small number of categories.

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
