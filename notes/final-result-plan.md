# Финальный образ результата и план достижения

## 0. Короткий вердикт

Идеальный результат проекта — не общий EDA по Olist, а цельное аналитическое исследование:

> Мы измерили сезонность спроса в Olist, нашли категории и товары с наиболее выраженной сезонностью, классифицировали разные типы сезонности, оценили влияние сезонности на бизнес-метрики и проверили, можно ли заранее предсказывать сезонный спрос по признакам товара, заказа, оплаты, логистики и географии.

Финальная работа должна отвечать на исходные вопросы задачи:

1. **Как сезонность влияет на спрос и какие товары подвержены ей сильнее всего?**
2. **Как сезонность влияет на продажи и бизнес-метрики в целом?**
3. **Можно ли заранее предсказать, что на определённый товар будет сезонный спрос? По каким признакам?**
4. **В какие периоды люди больше склонны к крупным покупкам?**
5. **Какая сезонность бывает и как её учитывать при прогнозе спроса?**

Главный принцип: сезонность нужно рассматривать не как один тип “пик в месяце X”, а как несколько разных паттернов: месячная, квартальная, event-driven, недельная, региональная, трендовая псевдосезонность и жизненный цикл товара.

---

## 1. Входные данные

### 1.1 Основной датасет

Используется один датасет:

- **Brazilian E-Commerce Public Dataset by Olist**
- Kaggle: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data>
- Период: 2016–2018
- Содержание: заказы, товары, категории, продавцы, покупатели, платежи, отзывы, география.

### 1.2 Основные CSV-таблицы

Нужны таблицы:

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_products_dataset.csv
product_category_name_translation.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_customers_dataset.csv
olist_sellers_dataset.csv
olist_geolocation_dataset.csv  # опционально для более глубокой географии
```

### 1.3 Главные ключи

```text
order_id      # связь orders / order_items / payments / reviews
product_id    # связь order_items / products
customer_id   # связь orders / customers
seller_id     # связь order_items / sellers
```

---

## 2. Важные ограничения данных

### 2.1 Покрытие по годам

По `order_purchase_timestamp`:

#### Все заказы

| Год | Заказов | Первый заказ | Последний заказ | Покрытие дней | Активных дней |
|---|---:|---|---|---:|---:|
| 2016 | 329 | 2016-09-04 | 2016-12-23 | 111 из 366 | 15 |
| 2017 | 45 101 | 2017-01-05 | 2017-12-31 | 361 из 365 | 361 |
| 2018 | 54 011 | 2018-01-01 | 2018-10-17 | 290 из 365 | 258 |

#### Только delivered-заказы

| Год | Заказов | Первый заказ | Последний заказ | Покрытие дней | Активных дней |
|---|---:|---|---|---:|---:|
| 2016 | 267 | 2016-09-15 | 2016-12-23 | 100 из 366 | 10 |
| 2017 | 43 428 | 2017-01-05 | 2017-12-31 | 361 из 365 | 361 |
| 2018 | 52 783 | 2018-01-01 | 2018-08-29 | 241 из 365 | 241 |

### 2.2 Следствия для анализа

1. **2016 нельзя использовать как полноценный год сезонности.**
   - Есть только кусок сентября–декабря.
   - Очень мало delivered-заказов.
   - Использовать максимум как контекст.

2. **2017 — основной полный год.**
   - Лучший год для 12-месячных сезонных профилей.

3. **2018 неполный.**
   - Можно использовать январь–август как проверку паттернов 2017.
   - Нельзя интерпретировать отсутствие продаж в сентябре–декабре 2018 как падение спроса.

4. **Для сезонности нужно использовать maturity/coverage logic из HW2.**
   - Не заполнять ненаблюдаемые будущие периоды нулями.
   - Явно отмечать неполные годы/месяцы.

---

## 3. Итоговые артефакты проекта

### 3.1 Processed data

В `data/processed/` должны появиться:

```text
orders_items_enriched.parquet
monthly_metrics.parquet
weekly_metrics.parquet
category_monthly_metrics.parquet
category_weekly_metrics.parquet
product_monthly_metrics.parquet
category_seasonality_scores.csv
product_seasonality_scores.csv
business_impact_peak_vs_normal.csv
large_purchase_monthly.csv
seasonality_prediction_features.csv
seasonality_prediction_results.csv
```

### 3.2 Results

В `results/`:

```text
results/figures/
  01_data_coverage_by_year_month.png
  02_orders_by_month.png
  03_revenue_by_month.png
  04_year_month_heatmap.png
  05_top_seasonal_categories.png
  06_category_month_seasonal_index_heatmap.png
  07_seasonality_type_clusters.png
  08_peak_vs_normal_business_metrics.png
  09_large_purchase_share_by_month.png
  10_seasonality_feature_importance.png
  optional_forecast_examples.png

results/tables/
  category_seasonality_ranking.csv
  product_seasonality_ranking.csv
  business_impact_peak_vs_normal.csv
  large_purchase_monthly.csv
  model_feature_importance.csv

results/presentation_outline.md
```

### 3.3 Notes

В `notes/`:

```text
00-brief.md
methodology.md
limitations.md
final-result-plan.md
```

---

## 4. Главная аналитическая таблица

### 4.1 Единица анализа

Основная единица анализа: **строка товара в заказе**.

То есть один заказ с несколькими товарами даёт несколько строк.

### 4.2 Целевая таблица

Файл:

```text
data/processed/orders_items_enriched.parquet
```

Ключевые поля:

```text
order_id
order_item_id
customer_id
product_id
seller_id
order_status
order_purchase_timestamp
order_approved_at
order_delivered_customer_date
order_estimated_delivery_date
year
quarter
month
year_month
week
weekday
price
freight_value
payment_value
payment_type
payment_installments
review_score
review_creation_date
product_category_name
product_category_name_english
product_name_lenght
product_description_lenght
product_photos_qty
product_weight_g
product_length_cm
product_height_cm
product_width_cm
product_volume_cm3
customer_city
customer_state
seller_city
seller_state
delivery_time_days
estimated_delivery_delta_days
freight_share
```

### 4.3 Базовый фильтр

Для основного анализа спроса:

```text
order_status == "delivered"
```

Отдельно можно держать анализ cancelled/unavailable как операционный контекст, но не смешивать с основным спросом.

---

## 5. Метрики спроса и бизнес-метрики

### 5.1 Спрос

Основные метрики спроса:

```text
orders_count       = число уникальных order_id
items_sold         = число строк order_items / сумма quantity, если выводится quantity
products_sold      = число product_id в заказах
category_orders    = число заказов в категории
```

### 5.2 Деньги

```text
revenue = sum(price)
gmv = sum(price + freight_value)
payment_value_sum = sum(payment_value)
avg_order_value = mean(payment_value на уровне заказа)
avg_item_price = mean(price)
```

### 5.3 Логистика

```text
freight_sum
avg_freight_value
freight_share = freight_value / (price + freight_value)
delivery_time_days = delivered_customer_date - purchase_timestamp
estimated_delivery_delta_days = estimated_delivery_date - delivered_customer_date
```

### 5.4 Клиентский опыт

```text
avg_review_score
review_score_distribution
share_low_reviews = share(review_score <= 2)
share_high_reviews = share(review_score >= 4)
```

### 5.5 Крупные покупки

Крупная покупка:

```text
large_purchase = payment_value >= 90th percentile
```

Проверить устойчивость на 75-м и 90-м перцентилях.

Метрики:

```text
large_purchase_count
large_purchase_share
avg_large_purchase_value
avg_payment_installments
share_installments_6_plus
```

---

## 6. Какие типы сезонности учитывать

Нельзя ограничиваться одним типом сезонности. Нужно рассмотреть несколько разных паттернов.

### 6.1 Месячная / календарная сезонность

Повторяемость по месяцам года.

Примеры:

- декабрьский пик;
- январский спад;
- мартовский рост;
- июнь-июльский пик.

Метрики:

```text
monthly_orders
monthly_revenue
monthly_index = monthly_orders / avg_monthly_orders_for_category
```

Главный график:

```text
category × month heatmap
```

### 6.2 Квартальная сезонность

Пик не в одном месяце, а в группе месяцев.

Примеры:

- Q4: октябрь–декабрь;
- Q1: январь–март;
- Q2/Q3 летний спрос.

Метрики:

```text
quarter_orders
quarter_revenue
peak_quarter_share = sales_in_peak_quarter / annual_sales
```

Зачем нужна: категория может не иметь одного экстремального месяца, но иметь устойчивый сезон в квартале.

### 6.3 Event-driven сезонность / всплески

Резкие всплески вокруг событий:

- Black Friday;
- Christmas;
- Mother’s Day;
- Valentine’s Day;
- Brazilian holidays;
- school season;
- payday / 13th salary в Бразилии.

Метрики:

```text
rolling_mean_30
rolling_std_30
z_score = (daily_sales - rolling_mean_30) / rolling_std_30
is_spike = abs(z_score) > 3
spike_count
max_spike_z_score
```

График:

```text
daily sales + rolling mean + anomaly points
```

### 6.4 Недельная сезонность

Различия по дням недели:

- будни vs выходные;
- Monday effect;
- payday-adjacent effects.

Метрики:

```text
weekday_orders
weekday_revenue
weekday_index = weekday_orders / avg_daily_orders
```

Графики:

```text
weekday bar chart
weekday × category heatmap
```

Это вторичный, но полезный слой для операционного прогноза.

### 6.5 Трендовая псевдосезонность

Категория может выглядеть сезонной, потому что она росла или падала, а не потому что спрос повторяется календарно.

Пример:

```text
Jan: 10
Feb: 20
Mar: 40
Apr: 80
```

Это тренд, не сезонность.

Как отличать:

- trend/seasonal/residual decomposition;
- detrended monthly profile;
- сравнение внутри 2017;
- отдельная метрика trend_strength.

### 6.6 Жизненный цикл товара

Product-level пик может означать не сезонность, а то, что товар появился/исчез.

Пример:

- товар появился в июле;
- продавался 2 месяца;
- исчез.

Фильтры:

```text
active_months >= 6
total_orders >= 20 или 30 для product_id
total_orders >= 100 или 200 для category
```

### 6.7 Региональная сезонность

В Бразилии регионы отличаются по климату, доходам, праздникам, логистике.

Анализ:

```text
category × customer_state × month
region × month
```

Графики:

```text
state × month heatmap
category × region seasonal index
```

Не обязательно делать главным блоком, но это сильный дополнительный анализ.

---

## 7. Классификация сезонности

### 7.1 Итоговая типология

Для каждой категории/товара присваивать тип:

| Тип | Смысл |
|---|---|
| `stable` | Низкий разброс, нет сильных пиков |
| `single_month_spike` | Один месяц сильно выше среднего |
| `quarter_peak` | 2–3 соседних месяца дают основную долю продаж |
| `holiday_q4` | Пик в октябре–декабре |
| `early_year` | Пик в январе–марте |
| `mid_year` | Пик в апреле–августе |
| `weekly_pattern` | Сильные различия по дням недели |
| `event_driven` | Есть резкие дневные/недельные всплески |
| `trend_driven` | Рост/падение сильнее сезонной компоненты |
| `low_confidence` | Мало продаж или мало активных месяцев |

### 7.2 Поля итоговой таблицы

```text
category
seasonality_strength
seasonality_type
peak_month
peak_quarter
weekly_pattern_strength
event_spike_count
trend_strength
total_orders
active_months
confidence_level
business_impact
```

---

## 8. Метрики силы сезонности

### 8.1 На уровне категории

Для каждой категории:

```text
total_orders
active_months
monthly_mean
monthly_std
coefficient_of_variation = monthly_std / monthly_mean
peak_to_mean = max(monthly_orders) / monthly_mean
peak_to_min = max(monthly_orders) / min_nonzero(monthly_orders)
peak_month_share = max(monthly_orders) / total_orders
peak_quarter_share = max(quarter_orders) / total_orders
seasonal_index_max = max(monthly_orders / monthly_mean)
```

### 8.2 Seasonality score

Базовый вариант:

```text
seasonality_score = coefficient_of_variation
```

Расширенный вариант:

```text
seasonality_score = 0.4 * normalized_cv
                    + 0.3 * normalized_peak_to_mean
                    + 0.3 * normalized_peak_quarter_share
```

Для презентации лучше объяснять через CV и peak share, чтобы не перегружать.

### 8.3 Confidence level

Высокий score при малом числе продаж может быть шумом. Поэтому нужен `confidence_level`.

Признаки confidence:

```text
total_orders
active_months
years_observed
months_observed
share_observed_months
```

Пример классификации:

```text
high    = total_orders >= 200 and active_months >= 10
medium  = total_orders >= 100 and active_months >= 6
low     = otherwise
```

`low_confidence` нельзя подавать как сильный вывод.

---

## 9. Анализ 1: data audit и maturity

### 9.1 Цель

Понять, какие периоды реально наблюдаются, и не исказить сезонность неполными годами.

### 9.2 Сделать

1. Прочитать все CSV.
2. Проверить размеры таблиц.
3. Проверить ключи и дубликаты.
4. Проверить missing values.
5. Посчитать coverage по датам.
6. Отдельно посчитать coverage для delivered-заказов.

### 9.3 Графики

1. `orders by day`
2. `orders by month`
3. `coverage by year/month heatmap`
4. `delivered vs all orders by month`

### 9.4 Вывод

В финальном тексте явно написать:

- основной год для 12-месячной сезонности — 2017;
- 2018 используется для проверки январь–август;
- 2016 исключается из строгих сезонных выводов.

---

## 10. Анализ 2: общая динамика спроса

### 10.1 Цель

Показать общий временной профиль Olist.

### 10.2 Метрики по месяцам

```text
orders_count
items_sold
revenue
gmv
avg_order_value
avg_item_price
large_purchase_share
avg_payment_installments
```

### 10.3 Графики

1. **Orders by month**
2. **Revenue by month**
3. **Average order value by month**
4. **Year × month heatmap**
5. **Orders and revenue indexed to monthly average**

### 10.4 Ожидаемый смысл

Этот блок отвечает на вопрос:

> Есть ли сезонные пики на уровне всего магазина?

Но общий магазин может сглаживать категорийные пики, поэтому далее нужен category-level анализ.

---

## 11. Анализ 3: сезонность категорий

### 11.1 Цель

Найти категории с самой сильной и самой слабой сезонностью.

### 11.2 Фильтры

Рекомендуемый минимум:

```text
category_total_orders >= 100
active_months >= 6
```

Для robustness:

```text
category_total_orders >= 50
category_total_orders >= 200
```

### 11.3 Расчёты

По `product_category_name_english`:

```text
monthly_orders
monthly_revenue
seasonal_index
cv
peak_to_mean
peak_quarter_share
peak_month
peak_quarter
confidence_level
```

### 11.4 Графики

1. **Top-15 seasonal categories by score**
2. **Top-15 stable categories**
3. **Category × month seasonal index heatmap**
4. **Monthly profiles for top seasonal categories**
5. **Peak month distribution**
6. **Seasonality score × total orders scatter**

### 11.5 Главный результат

Файл:

```text
results/tables/category_seasonality_ranking.csv
```

Колонки:

```text
category
total_orders
total_revenue
active_months
seasonality_score
cv
peak_to_mean
peak_quarter_share
peak_month
peak_quarter
seasonality_type
confidence_level
```

---

## 12. Анализ 4: сезонность товаров

### 12.1 Проблема

В Olist нет нормальных названий товаров, только `product_id` и категория. Поэтому product-level анализ должен быть appendix/детализацией, а основной рассказ лучше строить по категориям.

### 12.2 Фильтры

```text
product_total_orders >= 20 или 30
active_months >= 3 или 4
```

### 12.3 Метрики

Те же, что для категорий:

```text
seasonality_score
peak_month
peak_quarter
seasonality_type
confidence_level
```

### 12.4 Графики

1. **Top seasonal products inside top seasonal categories**
2. **Product monthly profiles for examples**
3. **Product seasonality score × total orders**

### 12.5 Как презентовать

Не говорить “товар X” как человекочитаемое название. Формулировать так:

> На уровне конкретных `product_id` наиболее сезонные товары в основном находятся в категориях X, Y, Z.

---

## 13. Анализ 5: типы сезонности

### 13.1 Цель

Ответить на вопрос:

> Какая сезонность бывает?

### 13.2 Подход

Для каждой категории построить 12-месячный профиль:

```text
[Jan_index, Feb_index, ..., Dec_index]
```

Дальше:

1. нормализовать профиль;
2. кластеризовать категории;
3. вручную интерпретировать кластеры;
4. присвоить `seasonality_type`.

### 13.3 Методы

- KMeans;
- hierarchical clustering;
- cosine/euclidean distance по 12-месячным профилям;
- ручные правила для явных типов.

### 13.4 Графики

1. **Cluster heatmap: category profiles**
2. **Average profile by cluster**
3. **Examples per seasonality type**
4. **Distribution of seasonality types**

### 13.5 Итоговая типология

В финале должно быть 4–6 понятных типов, например:

```text
stable demand
single-month spike
Q4 / holiday peak
early-year peak
mid-year peak
event-driven / anomaly-driven
trend-driven / low-confidence
```

---

## 14. Анализ 6: event-driven всплески

### 14.1 Цель

Понять, есть ли резкие всплески спроса, которые не видны в месячной агрегации.

### 14.2 Методика из HW2

Использовать rolling z-score:

```text
rolling_mean_30 = daily_orders.rolling(30).mean()
rolling_std_30 = daily_orders.rolling(30).std()
z_score = (daily_orders - rolling_mean_30) / rolling_std_30
is_anomaly = abs(z_score) > 3
```

### 14.3 Уровни анализа

1. общий daily demand;
2. top seasonal categories;
3. top revenue categories.

### 14.4 Графики

1. **Daily orders + rolling mean + spikes**
2. **Spike calendar**
3. **Categories with most spikes**
4. **Revenue impact of spike days**

### 14.5 Вывод

Этот блок нужен, чтобы отделить:

- нормальную календарную сезонность;
- разовые события;
- промо/праздничные всплески.

---

## 15. Анализ 7: недельная сезонность

### 15.1 Цель

Проверить операционные паттерны спроса.

### 15.2 Метрики

```text
weekday_orders
weekday_revenue
weekday_aov
weekday_index
```

### 15.3 Графики

1. **Orders by weekday**
2. **Revenue by weekday**
3. **Weekday × category heatmap**

### 15.4 Как использовать

В основной презентации этот блок можно оставить коротким:

> Помимо годовой/месячной сезонности, у спроса есть недельный операционный ритм, который важен для краткосрочного прогноза и логистики.

---

## 16. Анализ 8: бизнес-эффект сезонности

### 16.1 Цель

Показать, как сезонность влияет не только на спрос, но и на бизнес.

### 16.2 Сравнение peak vs normal

Для каждой сезонной категории:

```text
peak_months = месяцы, где seasonal_index >= 1.25 или top months
normal_months = остальные наблюдаемые месяцы
```

Сравнить:

```text
orders_count
revenue
avg_order_value
avg_item_price
large_purchase_share
avg_payment_installments
freight_share
avg_delivery_time_days
avg_review_score
share_low_reviews
```

### 16.3 Таблица

```text
metric
normal_months
peak_months
delta_abs
delta_pct
```

### 16.4 Графики

1. **Peak vs normal business metrics bar chart**
2. **Revenue seasonal/non-seasonal categories by month**
3. **Delivery time peak vs normal**
4. **Review score peak vs normal**
5. **Freight share peak vs normal**

### 16.5 Главный смысл

Ответить:

> Сезонность создаёт не только рост заказов, но и меняет структуру выручки, средний чек, нагрузку на логистику и клиентский опыт.

---

## 17. Анализ 9: крупные покупки

### 17.1 Цель

Ответить:

> В какие периоды люди больше склонны к крупным покупкам?

### 17.2 Определение крупной покупки

Основное:

```text
large_purchase = payment_value >= P90(payment_value)
```

Robustness:

```text
P75
P90
```

### 17.3 Метрики по месяцам

```text
large_purchase_count
large_purchase_share
avg_payment_value
median_payment_value
avg_payment_installments
share_installments_6_plus
top_categories_large_purchases
```

### 17.4 Графики

1. **Large purchase share by month**
2. **Average payment value by month**
3. **Payment installments by month**
4. **Top categories among large purchases**
5. **Large purchase share: seasonal vs non-seasonal categories**

### 17.5 Вывод

Нужен прямой ответ:

> В месяцы X/Y/Z выше доля крупных покупок; это связано с категориями A/B/C и/или с рассрочками.

---

## 18. Анализ 10: можно ли предсказать сезонность заранее

### 18.1 Цель

Ответить:

> Можно ли заранее предсказать, что на товар/категорию будет сезонный спрос? По каким признакам?

### 18.2 Target

На уровне категории:

```text
is_seasonal = seasonality_score in top 25%
```

Альтернативы для robustness:

```text
top 20%
top 30%
cv >= fixed threshold
```

На уровне товара — опционально, если достаточно наблюдений.

### 18.3 Признаки

```text
avg_price
median_price
avg_freight_value
freight_share
avg_payment_installments
share_installments_6_plus
avg_review_score
share_low_reviews
avg_delivery_time_days
avg_product_weight_g
avg_product_volume_cm3
avg_product_photos_qty
avg_product_name_length
avg_product_description_length
customer_state_nunique
seller_state_nunique
region_diversity
total_orders
total_revenue
```

### 18.4 Модели

Минимум:

```text
logistic regression
random forest classifier
```

Можно добавить:

```text
gradient boosting
```

### 18.5 Метрики

```text
ROC-AUC
F1
precision/recall
confusion matrix
```

### 18.6 Графики

1. **Feature importance**
2. **ROC curve**
3. **Confusion matrix**
4. **Boxplots: seasonal vs non-seasonal**
   - price;
   - freight;
   - installments;
   - weight/volume;
   - review score;
   - delivery time.

### 18.7 Как интерпретировать

Если ROC-AUC заметно выше 0.5:

> Да, сезонность частично предсказуема по признакам товара/цены/логистики/географии.

Если ROC-AUC около 0.5:

> По доступным признакам сезонность заранее плохо предсказывается; надёжнее использовать исторический спрос по категории.

Оба результата приемлемы, если честно объяснить.

---

## 19. Анализ 11: прогноз спроса

### 19.1 Цель

Показать, как сезонность можно учитывать при forecasting.

### 19.2 Ограничение

Данных мало для сильного годового прогноза:

- полный год фактически один — 2017;
- 2018 неполный;
- product-level ряды часто разрежены.

Поэтому forecast должен быть демонстрационным, не главным результатом.

### 19.3 Уровень прогноза

Лучше прогнозировать:

```text
overall weekly orders
category weekly orders для top seasonal categories
```

Не стоит массово прогнозировать каждый `product_id`.

### 19.4 Модели

1. **Naive baseline**
   - прогноз = среднее последних N периодов.

2. **Seasonal naive**
   - прогноз = значение аналогичного сезонного периода.

3. **SARIMA**
   - методика из HW2;
   - сезонный период `s=7` для дневных данных;
   - для месячных данных `s=12`, но точек мало.

### 19.5 Графики

1. **Historical + forecast**
2. **Forecast with confidence interval**
3. **Model comparison: MAE/MAPE**

### 19.6 Главный вывод

> Для сезонных категорий forecast должен быть category-specific. Общий forecast сглаживает пики и может недооценивать спрос в peak periods.

---

## 20. Использование методик из HW1–HW3

### 20.1 HW1: retention / LTV / experiments

Что применить:

1. **Когортная логика**
   - когорты покупателей по первому заказу;
   - retention покупателей;
   - revenue per customer.

2. **Cumulative LTV**
   - сравнить покупателей, пришедших в сезонные месяцы, с обычными.

3. **Before/after / DiD-логика**
   - seasonal vs non-seasonal categories;
   - peak vs normal months.

Пример вопроса:

> Сезонные месяцы дают долгосрочных клиентов или разовые покупки?

### 20.2 HW2: cohort maturity / time series / SARIMA

Самый важный источник методик.

Что применить:

1. **Maturity logic**
   - не сравнивать незрелые периоды;
   - не заполнять ненаблюдаемые месяцы нулями.

2. **Heatmaps**
   - category × month;
   - year × month.

3. **Rolling z-score anomalies**
   - event-driven spikes.

4. **Seasonal decomposition**
   - trend / seasonal / residual.

5. **ADF / stationarity**
   - опционально для forecast.

6. **SARIMA**
   - демонстрационный forecast.

### 20.3 HW3: confidence / ROC / sensitivity

Что применить:

1. **Confidence scoring**
   - не принимать малые категории за сезонные.

2. **ROC/feature scoring**
   - проверить предсказуемость `is_seasonal`.

3. **Sensitivity analysis**
   - устойчивость результатов к:
     - min_orders threshold;
     - seasonality_score definition;
     - выбору периода: 2017 vs 2017 + Jan–Aug 2018;
     - P75/P90 для large purchases.

---

## 21. Минимальный набор графиков для презентации

Для 8 минут нужно не больше 10–12 графиков.

### Must-have графики

1. **Data coverage by year/month**
   - доказать ограничения данных.

2. **Orders by month**
   - общий спрос.

3. **Revenue by month**
   - бизнес-эффект.

4. **Year × month heatmap**
   - общий сезонный рисунок.

5. **Top seasonal categories**
   - прямой ответ “кто сезонный”.

6. **Category × month seasonal index heatmap**
   - кто и когда пикует.

7. **Seasonality type clusters**
   - какие сезонности бывают.

8. **Daily/weekly spikes with rolling mean**
   - event-driven сезонность.

9. **Peak vs normal business metrics**
   - влияние на бизнес.

10. **Large purchase share / AOV by month**
   - периоды крупных покупок.

11. **Feature importance for seasonality prediction**
   - можно ли предсказать.

12. **Optional forecast example**
   - если останется место.

---

## 22. Структура финальной презентации на 8 минут

### Слайд 1. Задача

- Olist, 2016–2018.
- Главный вопрос: сезонность спроса.
- Что считаем спросом: orders/items/revenue.

### Слайд 2. Данные и ограничения

- схема таблиц;
- только delivered orders для основного анализа;
- 2017 полный, 2016 и 2018 неполные.

### Слайд 3. Общая динамика спроса

- orders by month;
- revenue by month;
- первые признаки сезонности.

### Слайд 4. Методика измерения сезонности

- monthly profile;
- seasonal index;
- seasonality score;
- confidence threshold;
- фильтр малых категорий.

### Слайд 5. Самые сезонные категории

- top seasonal categories;
- top stable categories для контраста.

### Слайд 6. Типы сезонности

- cluster heatmap;
- типы: stable, single-month spike, Q4/holiday, early-year, mid-year, event-driven, trend/low-confidence.

### Слайд 7. Влияние на бизнес

- peak vs normal months;
- revenue, AOV, delivery, review, installments.

### Слайд 8. Крупные покупки

- large purchase share by month;
- AOV/payment installments;
- категории крупных покупок.

### Слайд 9. Можно ли предсказать сезонность

- признаки;
- feature importance;
- ROC-AUC / качество;
- честное ограничение.

### Слайд 10. Как учитывать в forecast и бизнес-решениях

- category-level forecast;
- inventory planning;
- marketing calendar;
- logistics capacity;
- recommendations.

---

## 23. Финальные бизнес-рекомендации

Финал должен содержать не только аналитику, но и действия.

### 23.1 Inventory planning

Для категорий с понятным peak month:

```text
увеличивать stock / seller readiness за 2–4 недели до peak period
```

### 23.2 Marketing calendar

Для сезонных категорий:

```text
кампании запускать до начала peak period, а не в момент пика
```

### 23.3 Category-specific forecast

Не использовать один общий forecast для всех товаров:

```text
stable categories → simple baseline может работать
seasonal categories → category-level seasonal model
spike-driven categories → event calendar + anomaly-aware planning
```

### 23.4 Logistics capacity

Если peak months ухудшают delivery/reviews:

```text
заранее планировать логистическую нагрузку
следить за freight_share и delivery_time
```

### 23.5 Large purchase strategy

В месяцы высокого AOV/large_purchase_share:

```text
продвигать рассрочки
поднимать видимость дорогих категорий
планировать revenue target отдельно от order target
```

---

## 24. Пошаговый план выполнения

### Шаг 1. Подготовка данных

1. Скачать Olist dataset.
2. Распаковать в `data/raw/olist-brazilian-ecommerce/`.
3. Проверить список таблиц.
4. Сделать audit строк, колонок, ключей, missing values.
5. Сохранить audit summary.

Выход:

```text
data/processed/data_audit.md
notes/limitations.md
```

### Шаг 2. Сбор enriched table

1. Merge `orders` + `order_items`.
2. Добавить `products`.
3. Добавить English category names.
4. Добавить `payments`.
5. Добавить `reviews`.
6. Добавить `customers` и `sellers`.
7. Посчитать derived fields:
   - year/month/week/weekday;
   - product_volume;
   - delivery_time;
   - freight_share.

Выход:

```text
data/processed/orders_items_enriched.parquet
```

### Шаг 3. Coverage и maturity

1. Посчитать coverage по всем заказам.
2. Посчитать coverage по delivered.
3. Построить coverage heatmap.
4. Зафиксировать правила периодов:
   - 2017 = основной год;
   - 2018 Jan–Aug = validation/context;
   - 2016 = context only.

Выход:

```text
results/figures/01_data_coverage_by_year_month.png
notes/limitations.md
```

### Шаг 4. Общие monthly/weekly метрики

1. Посчитать monthly metrics.
2. Посчитать weekly metrics.
3. Построить orders/revenue/AOV graphs.

Выход:

```text
data/processed/monthly_metrics.parquet
results/figures/02_orders_by_month.png
results/figures/03_revenue_by_month.png
results/figures/04_year_month_heatmap.png
```

### Шаг 5. Category seasonality

1. Посчитать monthly category metrics.
2. Отфильтровать малые категории.
3. Посчитать seasonality scores.
4. Назначить confidence levels.
5. Построить top seasonal/stable categories.

Выход:

```text
data/processed/category_monthly_metrics.parquet
results/tables/category_seasonality_ranking.csv
results/figures/05_top_seasonal_categories.png
results/figures/06_category_month_seasonal_index_heatmap.png
```

### Шаг 6. Product seasonality

1. Посчитать product-level monthly metrics.
2. Отфильтровать товары с малым числом продаж.
3. Посчитать scores.
4. Связать с категориями.

Выход:

```text
results/tables/product_seasonality_ranking.csv
```

### Шаг 7. Seasonality type clustering

1. Построить 12-месячные профили категорий.
2. Кластеризовать.
3. Интерпретировать кластеры.
4. Назначить `seasonality_type`.

Выход:

```text
results/figures/07_seasonality_type_clusters.png
results/tables/category_seasonality_ranking.csv  # обновлённый с type
```

### Шаг 8. Event-driven spikes

1. Построить daily/weekly demand.
2. Посчитать rolling z-score.
3. Найти spike days/weeks.
4. Посчитать, какие категории дают spikes.

Выход:

```text
results/figures/daily_spikes_overview.png
results/tables/event_spikes.csv
```

### Шаг 9. Business impact

1. Определить peak vs normal months.
2. Посчитать business metrics.
3. Сравнить seasonal vs non-seasonal categories.

Выход:

```text
results/tables/business_impact_peak_vs_normal.csv
results/figures/08_peak_vs_normal_business_metrics.png
```

### Шаг 10. Large purchases

1. Определить P75/P90 thresholds.
2. Посчитать monthly large purchase share.
3. Найти месяцы/категории крупных покупок.

Выход:

```text
results/tables/large_purchase_monthly.csv
results/figures/09_large_purchase_share_by_month.png
```

### Шаг 11. Predicting seasonality

1. Собрать feature table.
2. Определить `is_seasonal`.
3. Обучить logistic regression / random forest.
4. Посчитать ROC-AUC/F1.
5. Построить feature importance.
6. Сделать sensitivity analysis.

Выход:

```text
data/processed/seasonality_prediction_features.csv
results/tables/model_feature_importance.csv
results/figures/10_seasonality_feature_importance.png
```

### Шаг 12. Forecast block

1. Построить seasonal naive baseline.
2. Построить SARIMA для общего спроса или топ-категорий.
3. Сравнить ошибки.
4. Подготовить optional figure.

Выход:

```text
results/figures/optional_forecast_examples.png
```

### Шаг 13. Финальная упаковка

1. Собрать лучшие графики.
2. Написать `results/presentation_outline.md`.
3. Обновить README коротким summary.
4. Проверить, что выводы соответствуют исходным вопросам.

---

## 25. Definition of Done

Проект можно считать готовым, если есть:

1. **Чистая enriched table** с заказами, товарами, платежами, категориями, отзывами и географией.
2. **Документированные ограничения данных** по 2016/2017/2018.
3. **Метрика силы сезонности** и ranking категорий.
4. **Классификация типов сезонности**, а не только один общий score.
5. **Product-level appendix** для топ сезонных товаров.
6. **Business impact table** peak vs normal.
7. **Large purchase analysis** по месяцам.
8. **Prediction block** с feature importance или честным выводом, что признаки слабо предсказывают сезонность.
9. **10–12 финальных графиков** для презентации.
10. **Presentation outline** с прямыми ответами на вопросы.

---

## 26. Главный ожидаемый итоговый вывод

Финальный вывод должен звучать примерно так:

> Сезонность в Olist неоднородна: часть категорий имеет стабильный спрос, часть — широкие квартальные пики, часть — резкие event-driven всплески. Наиболее сезонные категории отличаются не только объёмом продаж, но и бизнес-профилем: в пиковые месяцы меняются выручка, средний чек, доля крупных покупок, рассрочки, логистика и оценки клиентов. Для прогноза спроса лучше использовать category-level seasonal profiles и учитывать тип сезонности: стабильные категории можно прогнозировать простыми baseline-моделями, сезонные — отдельными сезонными моделями, spike-driven — через календарь событий и anomaly-aware planning.

Это и есть идеальный образ результата: не набор графиков, а связанный аналитический ответ на исходную бизнес-задачу.
