from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from .config import FIGURES, NOTES, PROCESSED, RAW, TABLES

LIGHT_THEME_RCPARAMS = {
    "text.color": "#1e293b",
    "axes.labelcolor": "#1e293b",
    "axes.titlecolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.edgecolor": "#94a3b8",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

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


def configure_plot_style() -> None:
    """Light-theme defaults for figures shown on white Slidev slides."""
    plt.rcParams.update(LIGHT_THEME_RCPARAMS)


def style_heatmap_annotations(
    ax,
    data: pd.DataFrame | np.ndarray,
    *,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    threshold: float = 0.55,
) -> None:
    """Pick annotation colors for light slide backgrounds: dark by default, white on dark cells."""
    values = np.asarray(data, dtype=float)
    vmin = float(np.nanmin(values)) if vmin is None else vmin
    vmax = float(np.nanmax(values)) if vmax is None else vmax
    cmap_obj = cm.get_cmap(cmap)
    norm: Normalize | TwoSlopeNorm
    if center is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    nrows, ncols = values.shape
    for text in ax.texts:
        text.set_color("#1e293b")
        x, y = text.get_position()
        col = int(round(x - 0.5))
        row = int(round(y - 0.5))
        if row < 0 or col < 0 or row >= nrows or col >= ncols:
            continue
        val = values[row, col]
        if not np.isfinite(val):
            continue
        r, g, b, _ = cmap_obj(norm(val))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum < threshold:
            text.set_color("white")


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
