from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, NOTES, PROCESSED, RAW, TABLES

def ensure_dirs() -> None:
    for path in [PROCESSED, FIGURES, TABLES, NOTES]:
        path.mkdir(parents=True, exist_ok=True)


def reset_presentation_outputs() -> None:
    """Remove generated presentation artifacts before regenerating them."""
    ensure_dirs()
    for pattern, directory in [("*.png", FIGURES), ("*.csv", TABLES)]:
        for path in directory.glob(pattern):
            path.unlink()


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw table: {path}")
    return pd.read_csv(path, **kwargs)


def mode_or_nan(series: pd.Series):
    non_na = series.dropna()
    if non_na.empty:
        return np.nan
    return non_na.mode().iloc[0]


def safe_div(a, b):
    return np.where(np.asarray(b) == 0, np.nan, np.asarray(a) / np.asarray(b))


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
