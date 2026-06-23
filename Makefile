.PHONY: sync list-datasets download-data extract-data

DATASET ?= olist-brazilian-ecommerce

sync:
	uv sync

list-datasets:
	uv run python scripts/download_data.py list

download-data:
	uv run python scripts/download_data.py download $(DATASET)

extract-data:
	uv run python scripts/download_data.py extract $(DATASET)
