.PHONY: sync list-datasets download-data extract-data final-analysis

DATASET ?= olist-brazilian-ecommerce

sync:
	uv sync

list-datasets:
	uv run python scripts/download_data.py list

download-data:
	uv run python scripts/download_data.py download $(DATASET)

extract-data:
	uv run python scripts/download_data.py extract $(DATASET)

final-analysis:
	uv run python scripts/run_final_analysis.py
