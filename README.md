# Сезонность продаж Olist Store

Учебный аналитический проект по кейсу №4 из файла «Проекты 2026».

## Кейс

**Датасет:** Brazilian E-Commerce Public Dataset by Olist

**Источник:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data

В датасете доступны продажи бразильской компании Olist Store за период 2016–2018 гг. Нужно проанализировать, какие товары обладают ярко выраженной сезонностью, какая сезонность бывает и как её можно учитывать при прогнозе спроса.

## Основной вопрос

Как сезонность влияет на спрос и какие товары подвержены ей сильнее всего?

## Дополнительные вопросы

- Как сезонность влияет на продажи и бизнес-метрики в целом?
- Можно ли заранее предсказать, что на определённый товар будет сезонный спрос? По каким признакам?
- В какие периоды люди больше склонны к крупным покупкам?

## Ожидаемый результат

Проект сдаётся в виде презентации на 8 минут. Структура презентации:

1. Постановка задачи — что делали и какую задачу решали.
2. Обзор данных — какие данные доступны и нужны ли дополнительные источники.
3. Что сделали — ключевые инсайты, действия и подходы к решению.
4. Финальные аналитические выводы — прямой ответ на основной вопрос.

Фокус презентации — пункты 3–4: работа, инсайты и выводы.

## Структура репозитория

```text
data/
  datasets.json  # реестр источников данных проекта
  raw/            # исходные данные, не редактировать вручную
  processed/      # очищенные и промежуточные датасеты
notebooks/        # разведочный анализ
scripts/          # утилиты проекта
src/              # переиспользуемый код для загрузки, очистки и анализа
results/          # графики, таблицы, артефакты для презентации
notes/            # краткие заметки по задаче и решениям
```

## Базовый setup

Проект использует `uv` для Python-окружения и `make` для коротких CLI-команд.

1. Подготовить окружение:

   ```bash
   make sync
   ```

2. Настроить Kaggle API credentials: положить `kaggle.json` в `~/.kaggle/` или задать переменные `KAGGLE_USERNAME` и `KAGGLE_KEY`.

3. Посмотреть реестр датасетов:

   ```bash
   make list-datasets
   ```

4. Скачать архив датасета:

   ```bash
   make download-data
   ```

5. Распаковать архив:

   ```bash
   make extract-data
   ```

Команда `download-data` сохраняет архив в `data/raw/archives/olist-brazilian-ecommerce/`, а `extract-data` распаковывает его в `data/raw/olist-brazilian-ecommerce/`.

## Ближайший следующий шаг

Скачать и распаковать датасет, затем сделать первичный audit таблиц: строки, колонки, ключи, даты заказов, категории товаров, цены, freight и статусы заказов.
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
- `results/tables/large_purchase_price_thresholds.csv` — пороги top-5/10/15/20% по цене товара.
- `results/tables/large_purchase_revenue_by_month.csv` — распределение выручки крупных покупок по месяцам.
- `results/tables/large_purchase_revenue_by_quarter.csv`, `large_purchase_revenue_by_week.csv`, `large_purchase_revenue_by_weekday.csv` — другие гранулярности.
- `results/tables/model_feature_importance.csv` — признаки, связанные с сезонностью.
- `results/tables/top3_categories_by_month.csv` и `top5_products_by_month.csv` — топы по каждому месяцу.
- `results/tables/top3_categories_by_quarter.csv` и `top5_products_by_quarter.csv` — топы по каждому кварталу.
- `results/figures/` — пронумерованные финальные графики для презентации (`01_...png`, `02_...png`, ...).
- `results/presentation_outline.md` — структура 8-минутной презентации.
- `results/final_summary.md` — прямые ответы на вопросы кейса.

Главное ограничение: 2017 — основной полный год; 2016 и конец 2018 нельзя использовать как полноценные сезонные периоды.
