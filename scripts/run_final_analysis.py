#!/usr/bin/env python3
"""Run the full Olist seasonality analysis pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist_seasonality.pipeline import main


if __name__ == "__main__":
    main()
