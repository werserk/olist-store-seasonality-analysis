# Methodology

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

## Large purchases

Large purchases are item-level expensive goods: a product item belongs to the selected top-price bucket by `price`. The main definition uses top-10% (`price >= P90`); sensitivity tables also cover top-5%, top-15% and top-20%.

## Prediction block

`is_seasonal` is defined as top quartile by seasonality score among eligible categories. A random forest model estimates how well product/order/logistics/geography features explain seasonality. Result: ROC-AUC 0.33, F1 0.00.

## Forecast block

Forecasting is demonstrative because the dataset has only one complete year. Weekly total demand is compared across naive, seasonal-naive/rolling mean, and SARIMA-style baselines where fitting succeeds.
