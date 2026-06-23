# Final analytical summary

## Direct answers

### How does seasonality affect demand and which categories/products are strongest?

Seasonality is heterogeneous. The strongest high/medium-confidence categories by CV score are:

| category               |   total_orders |   seasonality_score |   peak_month |   peak_quarter | seasonality_type   |
|:-----------------------|---------------:|--------------------:|-------------:|---------------:|:-------------------|
| stationery             |            897 |            0.780273 |           12 |              4 | single_month_spike |
| electronics            |            810 |            0.778976 |           12 |              4 | event_driven       |
| home_appliances        |            232 |            0.737056 |            6 |              3 | event_driven       |
| watches_gifts          |           2073 |            0.703136 |           11 |              4 | event_driven       |
| consoles_games         |            640 |            0.646867 |           11 |              4 | event_driven       |
| garden_tools           |           1934 |            0.623369 |           11 |              4 | event_driven       |
| toys                   |           2404 |            0.614123 |           11 |              4 | event_driven       |
| books_general_interest |            214 |            0.594689 |           11 |              4 | event_driven       |
| home_confort           |            246 |            0.580433 |           11 |              3 | event_driven       |
| musical_instruments    |            241 |            0.572599 |           11 |              4 | event_driven       |

Product-level appendix is available in `results/tables/product_seasonality_ranking.csv`; product IDs should be interpreted through their category because Olist has no readable product names.

### What types of seasonality appear?

The project distinguishes monthly/calendar, quarterly, event-driven, weekly, trend-driven pseudo-seasonality, product lifecycle effects and regional/geographic differences. The final category table includes `seasonality_type`, `event_spike_count`, `weekly_pattern_strength`, `trend_strength` and `confidence_level`.

### Business impact

Peak months for seasonal categories are compared to normal months in `results/tables/business_impact_peak_vs_normal.csv`. The comparison covers orders, revenue, average order value, large purchases, installments, freight, delivery time and review score.

### Large purchases

Large purchases are defined as order-level `payment_value >= P90`. Monthly results are in `results/tables/large_purchase_monthly.csv`. Highest-share months:

| year_month   |   large_purchase_share_p90 |   avg_payment_value |   share_installments_6_plus |
|:-------------|---------------------------:|--------------------:|----------------------------:|
| 2017-01      |                      0.14  |             170.061 |                       0.183 |
| 2017-04      |                      0.115 |             169.758 |                       0.193 |
| 2017-10      |                      0.113 |             167.74  |                       0.163 |
| 2017-09      |                      0.11  |             168.957 |                       0.178 |
| 2017-03      |                      0.108 |             162.753 |                       0.176 |

### Can seasonality be predicted?

A category-level feature model was trained with target `is_seasonal = top quartile by seasonality score`. Result: ROC-AUC 0.33, F1 0.00. Feature importances are in `results/tables/model_feature_importance.csv` and figure `results/figures/10_seasonality_feature_importance.png`.

### How to forecast demand?

Use category-specific planning: stable categories can use simple baselines; seasonal categories need seasonal profiles; spike-driven categories need event calendars and anomaly-aware planning. A demonstration is saved in `results/figures/optional_forecast_examples.png` and `results/tables/forecast_model_comparison.csv`.
