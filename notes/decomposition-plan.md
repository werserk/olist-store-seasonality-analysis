# План декомпозиции аналитического pipeline

## Проблема

`scripts/run_final_analysis.py` вырос до 1147 строк и смешивал роли:

- загрузка и аудит данных;
- сбор enriched датасета;
- агрегации;
- scoring сезонности;
- business-impact анализ;
- ML/forecast блок;
- построение графиков;
- генерация markdown-отчётов;
- обновление README.

Такой файл трудно читать, защищать и менять точечно.

## Целевая архитектура

```text
scripts/run_final_analysis.py          # тонкая CLI-точка входа
src/olist_seasonality/
  __init__.py
  config.py                            # пути, константы месяцев/дней
  utils.py                             # ensure_dirs, read_csv, save_fig, helpers
  data.py                              # load/audit/enrich/coverage
  metrics.py                           # order-level и aggregate metrics
  seasonality.py                       # category/product seasonality, impact, large purchases
  modeling.py                          # prediction и forecast blocks
  reporting.py                         # final_summary, outline, README update
  pipeline.py                          # orchestration порядка шагов
notebooks/final_analysis.ipynb         # воспроизводимый обзор/перегенерация результатов
```

## Правила для графиков

- Все презентационные картинки сохраняются в `results/figures/`.
- Имена файлов начинаются с номера `NN_`, чтобы порядок был устойчивым для презентации.
- Таблицы сохраняются в `results/tables/`.
- Notebook не заменяет pipeline, а служит удобной витриной: запуск pipeline, просмотр таблиц и вставка графиков.

## Этапы реализации

1. Вынести конфигурацию и shared helpers.
2. Разнести функции по доменным модулям.
3. Оставить `scripts/run_final_analysis.py` тонким wrapper вокруг `olist_seasonality.pipeline.main`.
4. Пронумеровать все сохраняемые графики.
5. Добавить notebook-витрину для повторного запуска и просмотра outputs.
6. Прогнать `py_compile`, `make final-analysis`, проверить количество графиков/таблиц.
7. Закоммитить и запушить изменения.
