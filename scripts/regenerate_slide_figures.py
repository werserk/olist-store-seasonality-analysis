#!/usr/bin/env python3
"""Regenerate slide figures with readable text on light backgrounds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from olist_seasonality.completion import refresh_final_figure_set
from olist_seasonality.config import FIGURES, TABLES
from olist_seasonality.utils import configure_plot_style, save_fig, style_heatmap_annotations

PRES_ASSETS = ROOT / "presentation" / "assets" / "figures"


def coverage_figure() -> None:
    monthly = pd.read_csv(TABLES / "monthly_metrics.csv")
    ym = (
        monthly.groupby(["year", "month"])["orders_count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )
    plt.figure(figsize=(11, 4.6))
    ax = sns.heatmap(ym, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Delivered orders"})
    style_heatmap_annotations(ax, ym, cmap="Blues")
    plt.title("Data coverage: delivered orders by year/month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    save_fig(FIGURES / "01_data_coverage_by_year_month.png")


def top_seasonal_figure() -> None:
    ranking = pd.read_csv(TABLES / "category_seasonality_ranking.csv")
    eligible = ranking[ranking["confidence_level"].ne("low")].copy()
    top = eligible.sort_values("seasonality_score", ascending=False).head(15).sort_values("seasonality_score")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    sns.barplot(data=top, y="category", x="seasonality_score", hue="seasonality_type", dodge=False, ax=ax)
    for patch, peak_month in zip(ax.patches, top["peak_month"]):
        width = getattr(patch, "get_width")()
        y = getattr(patch, "get_y")() + getattr(patch, "get_height")() / 2
        ax.text(width + 0.015, y, f"M{int(peak_month)}", va="center", fontsize=8, color="#1e293b")
    ax.set_title("Most seasonal categories with peak month labels (2017)")
    ax.set_xlabel("Seasonality score: CV of monthly delivered orders")
    ax.set_ylabel("")
    ax.legend(title="Type", fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(FIGURES / "35_top_seasonal_categories_with_peak_month.png")


def main() -> None:
    configure_plot_style()
    sns.set_theme(style="whitegrid")
    coverage_figure()
    top_seasonal_figure()
    refresh_final_figure_set()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_slidev_figures.py")], check=True)
    print("OK regenerated slide figures ->", PRES_ASSETS)


if __name__ == "__main__":
    main()
