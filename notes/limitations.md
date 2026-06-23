# Limitations

## Date coverage

| scope     |   year |   orders | first_order   | last_order   |   span_days |   active_days |   calendar_days |   missing_before |   missing_after |
|:----------|-------:|---------:|:--------------|:-------------|------------:|--------------:|----------------:|-----------------:|----------------:|
| delivered |   2016 |      267 | 2016-09-15    | 2016-12-23   |         100 |            10 |             366 |              258 |               8 |
| delivered |   2017 |    43428 | 2017-01-05    | 2017-12-31   |         361 |           361 |             365 |                4 |               0 |
| delivered |   2018 |    52783 | 2018-01-01    | 2018-08-29   |         241 |           241 |             365 |                0 |             124 |

## Rules used in the final analysis

- Main demand analysis uses `order_status == delivered`.
- 2017 is the primary complete year for 12-month seasonality profiles.
- 2018 is used mainly as January-August context/validation.
- 2016 is context only: the year is too incomplete and sparse for strict seasonality claims.
- Category-level conclusions are filtered by confidence (`total_orders` and `active_months`) to avoid treating tiny categories as seasonal.
- Product-level output is an appendix because Olist product IDs do not include human-readable product names.
