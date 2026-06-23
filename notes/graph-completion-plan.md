# План графиков для ответа на целевые вопросы

Цель документа — зафиксировать, какие визуализации нужны, чтобы работа была завершена по исходным исследовательским вопросам, какие из них уже построены, а какие ещё стоит добавить.

## Execution status

План выполнен: недостающие графики G01–G12 добавлены в pipeline, итоговый cleaned set пересобирается автоматически в `results/final_figure_set/` после `make final-analysis`.

Новые графики:

- G01: `results/figures/35_top_seasonal_categories_with_peak_month.png`
- G02: `results/figures/36_top6_category_monthly_profiles.png`
- G03: `results/figures/37_seasonality_type_composition.png`
- G04: `results/figures/38_top20_seasonal_products_appendix.png`
- G05: `results/figures/39_business_impact_by_category_uplift.png`
- G06: `results/figures/40_revenue_decomposition_peak_vs_normal.png`
- G07: `results/figures/41_early_prediction_roc.png`
- G08: `results/figures/42_early_prediction_feature_importance.png`
- G09: `results/figures/43_early_warning_score_vs_final_seasonality.png`
- G10: `results/figures/44_large_purchase_black_friday_window.png`
- G11: `results/figures/45_large_purchase_category_mix_peak_vs_normal.png`
- G12: `results/figures/46_large_purchase_installments_by_month.png`

## Целевые вопросы

1. **Основной вопрос:** как сезонность влияет на спрос и какие товары/категории подвержены ей сильнее всего?
2. **Бизнес-метрики:** как сезонность влияет на продажи и бизнес-метрики в целом?
3. **Предсказуемость:** можно ли заранее предсказать, что на определённый товар/категорию будет сезонный спрос, и по каким признакам?
4. **Крупные покупки:** в какие периоды люди больше склонны к крупным покупкам?

## Критерий готовности

Работа считается визуально закрытой, когда для каждого вопроса есть:

- один главный график для презентации;
- одна supporting visualization для проверки механики;
- таблица с численными значениями;
- явная связь `вопрос → метрика → график → вывод`.

## 1. Сезонность спроса и самые сезонные категории/товары

### 1.1. Data coverage / почему 2017 основной год

**Вопрос, который закрывает:** можно ли вообще анализировать сезонность по годам?

- Статус: **готово**.
- График: `results/figures/01_data_coverage_by_year_month.png`.
- Таблица/источник: coverage analysis из pipeline.
- Вывод: 2017 — основной полный год; 2016 и 2018 неполные.

### 1.2. Общий спрос по месяцам

**Вопрос, который закрывает:** есть ли общий сезонный профиль спроса на уровне всего Olist?

- Статус: **готово**.
- Графики:
  - `results/figures/02_orders_by_month.png`
  - `results/figures/03_revenue_by_month.png`
  - `results/figures/04_year_month_heatmap.png`
- Таблица: `results/tables/monthly_metrics.csv`.
- Вывод: общий рынок растёт и имеет Q4/Nov эффекты, но общий тренд нельзя путать с сезонностью отдельных категорий.

### 1.3. Рейтинг сезонности категорий

**Вопрос, который закрывает:** какие категории сильнее всего подвержены сезонности?

- Статус: **готово, но нужно улучшить читаемость**.
- Текущий график: `results/figures/05_top_seasonal_categories.png`.
- Таблица: `results/tables/category_seasonality_ranking.csv`.
- Метрики:
  - `seasonality_score = CV(monthly_orders)`;
  - `peak_month`;
  - `peak_quarter`;
  - `seasonality_type`;
  - `confidence_level`.
- Нужно добавить:
  - **G01 — итоговый ranked chart с peak month labels**: top seasonal categories, цвет = `seasonality_type`, подпись = `peak_month`.
- Приоритет: **must-have**.

### 1.4. Профили топ-сезонных категорий

**Вопрос, который закрывает:** как именно выглядит сезонность у главных категорий?

- Статус: **частично готово**.
- Текущие графики:
  - `results/figures/06_category_month_seasonal_index_heatmap.png`
  - `results/figures/16_category_monthly_line_orders.png`
  - `results/figures/15_category_quarterly_line_orders.png`
  - `results/figures/17_category_weekly_line_orders.png`
- Нужно добавить:
  - **G02 — small multiples для top-6 seasonal categories**: `stationery`, `electronics`, `watches_gifts`, `toys`, `garden_tools`, `home_appliances`; x = month, y = orders and/or seasonal index.
- Приоритет: **must-have**.

### 1.5. Типы сезонности

**Вопрос, который закрывает:** сезонность бывает одного типа или нескольких?

- Статус: **готово**.
- Графики:
  - `results/figures/07_seasonality_type_clusters.png`
  - `results/figures/08_daily_spikes_overview.png`
  - `results/figures/09_weekday_orders.png`
  - `results/figures/18_category_weekday_line_orders.png`
- Таблицы:
  - `results/tables/event_spikes.csv`
  - `results/tables/seasonality_cluster_profiles.csv`
  - `results/tables/category_weekday_line_metrics.csv`
- Метод:
  - monthly CV;
  - peak month/quarter share;
  - daily rolling z-score spikes;
  - weekday index spread;
  - trend strength.
- Нужно добавить:
  - **G03 — type composition chart**: bar chart с количеством категорий по `seasonality_type`, отдельно high/medium/low confidence.
- Приоритет: **should-have**.

### 1.6. Product-level сезонность

**Вопрос, который закрывает:** какие конкретные товары/product_id наиболее сезонны?

- Статус: **таблица есть, графика нет**.
- Таблица: `results/tables/product_seasonality_ranking.csv`.
- Ограничение: Olist `product_id` не человекочитаемый, поэтому product-level должен быть appendix, а не главный вывод.
- Нужно добавить:
  - **G04 — top-20 seasonal product_id appendix**: y = product_id, x = seasonality_score, цвет = category, фильтр `confidence_level != low`.
- Приоритет: **should-have**.

## 2. Влияние сезонности на продажи и бизнес-метрики

### 2.1. Peak vs normal months

**Вопрос, который закрывает:** что меняется в peak periods у сезонных категорий?

- Статус: **готово, но агрегат нужно усилить paired-анализом**.
- Текущий график: `results/figures/10_peak_vs_normal_business_metrics.png`.
- Таблица: `results/tables/business_impact_peak_vs_normal.csv`.
- Метрики:
  - orders;
  - items;
  - revenue;
  - AOV;
  - avg item price;
  - large purchase share;
  - installments;
  - freight share;
  - delivery time;
  - review score;
  - low reviews.

### 2.2. Paired uplift by category

**Вопрос, который закрывает:** сезонность реально повышает бизнес-метрики внутри категории или эффект смазан агрегацией?

- Статус: **нужно добавить**.
- Новый график:
  - **G05 — paired uplift distribution**: для каждой seasonal category сравнить `peak_month_value` с `normal_month_average`; показать boxplot/bar распределения uplift по метрикам.
- Метрики для первого графика:
  - `orders_count`;
  - `revenue`;
  - `avg_item_price`;
  - `large_purchase_share`;
  - `avg_payment_installments`;
  - `avg_delivery_time_days`;
  - `avg_review_score`.
- Таблица:
  - `business_impact_by_category_uplift.csv`.
- Приоритет: **must-have**.

### 2.3. Revenue decomposition

**Вопрос, который закрывает:** сезонность влияет на revenue через объём, цену или структуру заказа?

- Статус: **нужно добавить**.
- Новый график:
  - **G06 — revenue decomposition peak vs normal**: orders, items/order, avg item price, revenue per category-month.
- Приоритет: **should-have**.

## 3. Можно ли заранее предсказать сезонный спрос?

### 3.1. Текущая category-level модель

**Вопрос, который закрывает:** можно ли классифицировать сезонность по текущим category features?

- Статус: **готово, но результат слабый**.
- Графики:
  - `results/figures/12_seasonality_prediction_roc.png`
  - `results/figures/13_seasonality_feature_importance.png`
- Таблицы:
  - `results/tables/seasonality_prediction_results.csv`
  - `results/tables/model_feature_importance.csv`
- Текущий вывод: ROC-AUC около `0.33`, F1 около `0.00`; убедительного early prediction нет.

### 3.2. Early-signal prediction

**Вопрос, который закрывает:** можно ли заранее, по первой половине года, предсказать сезонность второй половины?

- Статус: **нужно добавить**.
- Новый анализ:
  - признаки только из Jan–Jun;
  - цель — seasonal/high-spike behavior в Jul–Dec или top quartile final seasonality.
- Признаки:
  - H1 orders;
  - H1 revenue;
  - avg item price;
  - growth Jan→Jun;
  - weekday concentration;
  - installments;
  - freight share;
  - review score;
  - customer/seller state diversity.
- Новые графики:
  - **G07 — early prediction ROC/PR or score distribution**;
  - **G08 — early feature importance / coefficient chart**.
- Новые таблицы:
  - `early_seasonality_prediction_results.csv`;
  - `early_seasonality_feature_importance.csv`.
- Приоритет: **must-have**.

### 3.3. Explainable early warning scorecard

**Вопрос, который закрывает:** если ML слабый, какие признаки всё равно полезны как warning signals?

- Статус: **нужно добавить**.
- Новый график:
  - **G09 — early warning score vs final seasonality_score**: x = early warning score, y = final seasonality_score, цвет = seasonality_type.
- Приоритет: **should-have**.

## 4. Крупные покупки

### 4.1. Пороги крупной покупки

**Вопрос, который закрывает:** что именно считается крупной покупкой?

- Статус: **готово**.
- Таблица: `results/tables/large_purchase_price_thresholds.csv`.
- Определение:
  - основной порог: top-10% item-level `price`;
  - sensitivity: top-5%, top-15%, top-20%.

### 4.2. Крупные покупки по месяцам/кварталам/неделям/weekday

**Вопрос, который закрывает:** в какие периоды люди больше склонны к крупным покупкам?

- Статус: **готово**.
- Графики:
  - `results/figures/11_large_purchase_revenue_top10_by_month.png`
  - `results/figures/19_large_purchase_revenue_top10_by_quarter.png`
  - `results/figures/20_large_purchase_revenue_top10_by_week.png`
  - `results/figures/21_large_purchase_revenue_top10_by_weekday.png`
  - `results/figures/22_large_purchase_revenue_top5_by_month.png`
  - `results/figures/26_large_purchase_revenue_top15_by_month.png`
  - `results/figures/30_large_purchase_revenue_top20_by_month.png`
  - `results/figures/34_weekday_revenue_distribution_all_top20_top10.png`
- Таблицы:
  - `large_purchase_revenue_by_month.csv`;
  - `large_purchase_revenue_by_quarter.csv`;
  - `large_purchase_revenue_by_week.csv`;
  - `large_purchase_revenue_by_weekday.csv`.

### 4.3. Event window for expensive purchases

**Вопрос, который закрывает:** связаны ли крупные покупки с Black Friday / Q4 event windows?

- Статус: **нужно добавить**.
- Новый график:
  - **G10 — daily expensive-goods revenue around Black Friday/Q4**: x = date, y = revenue, линии = all revenue / top-10 / top-20, vertical line = Black Friday 2017-11-24.
- Приоритет: **must-have**.

### 4.4. Category mix of large purchases

**Вопрос, который закрывает:** какие категории создают крупные покупки?

- Статус: **нужно добавить**.
- Новый график:
  - **G11 — top categories by expensive-goods revenue**: сравнить peak months vs normal months.
- Новая таблица:
  - `large_purchase_category_mix_peak_vs_normal.csv`.
- Приоритет: **must-have**.

### 4.5. Installments and large purchases

**Вопрос, который закрывает:** сопровождаются ли крупные покупки рассрочкой?

- Статус: **нужно добавить**.
- Новый график:
  - **G12 — installments for all vs top-10 expensive purchases by month**: avg installments and share installments >= 6.
- Приоритет: **should-have**.

## Итоговый cleaned figure set

Для презентации не нужно показывать все 34 графика. Основной очищенный комплект должен лежать в `results/final_figure_set/` и содержать только ключевые графики, переименованные по логике презентации.

Текущий комплект должен включать:

1. data coverage;
2. overall orders by month;
3. overall revenue by month;
4. top seasonal categories;
5. category monthly heatmap;
6. seasonality clusters/types;
7. daily event spikes;
8. weekday operational pattern;
9. peak vs normal business metrics;
10. top-10 large purchases by month;
11. top-10 large purchases by quarter;
12. all/top-20/top-10 revenue distribution by weekday;
13. prediction ROC;
14. feature importance;
15. forecast examples.

После добавления недостающих графиков комплект нужно обновить:

- заменить текущий top seasonal chart на G01;
- добавить G02 category small multiples;
- заменить/дополнить business impact графиком G05;
- добавить G07/G08 по early prediction;
- добавить G10/G11 по крупным покупкам.

## Очерёдность реализации

1. G01 + G02 — закрыть основной вопрос по категориям.
2. G05 + G06 — закрыть бизнес-метрики.
3. G10 + G11 — закрыть крупные покупки.
4. G07 + G08 + G09 — закрыть предсказуемость.
5. G03 + G04 + G12 — appendix / усиление презентации.

## Acceptance checklist

- [x] Каждый целевой вопрос имеет минимум один главный график.
- [x] Каждый главный график имеет таблицу-источник в `results/tables/`.
- [x] Итоговая папка `results/final_figure_set/` содержит только презентационно важные графики.
- [x] Все новые графики имеют стабильную нумерацию и читаемые имена.
- [x] `results/final_summary.md` и `results/presentation_outline.md` ссылаются на актуальные графики.
- [x] `make final-analysis` проходит после всех изменений.
