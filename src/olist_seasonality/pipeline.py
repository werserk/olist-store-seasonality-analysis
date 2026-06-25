from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from .completion import (
    business_completion_figures,
    early_prediction_completion_figures,
    large_purchase_completion_figures,
    refresh_final_figure_set,
    seasonality_completion_figures,
)
from .data import audit_tables, build_enriched, coverage_analysis, load_tables
from .config import FIGURES, RESULTS, TABLES
from .metrics import aggregate_metrics
from .modeling import forecast_block, prediction_block
from .reporting import update_readme, write_notes_and_outline
from .seasonality import (
    business_impact,
    category_seasonality,
    granularity_artifacts,
    large_purchases,
    product_seasonality,
    robustness_analysis,
)
from .utils import reset_presentation_outputs, configure_plot_style

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
configure_plot_style()
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180})


def main() -> None:
    reset_presentation_outputs()
    tables = load_tables()
    audit = audit_tables(tables)
    enriched = build_enriched(tables)
    coverage = coverage_analysis(tables)
    monthly, weekly = aggregate_metrics(enriched)
    ranking, cat_month = category_seasonality(enriched)
    granularity_artifacts(enriched, ranking)
    product_ranking = product_seasonality(enriched)
    seasonality_completion_figures(ranking, cat_month, product_ranking)
    impact = business_impact(enriched, ranking, cat_month)
    business_completion_figures(enriched, ranking, cat_month)
    large = large_purchases(enriched)
    large_purchase_completion_figures(enriched, ranking, cat_month)
    prediction_features, pred_results = prediction_block(enriched, ranking)
    early_prediction_completion_figures(enriched, ranking)
    forecast_results = forecast_block(enriched)
    robustness_analysis(ranking)
    write_notes_and_outline(
        audit,
        coverage,
        monthly,
        ranking,
        product_ranking,
        impact,
        large,
        pred_results,
        forecast_results,
    )
    update_readme()
    refresh_final_figure_set()
    import json

    print(json.dumps({
        "enriched_rows": int(len(enriched)),
        "category_rankings": int(len(ranking)),
        "product_rankings": int(len(product_ranking)),
        "figures": len(list(FIGURES.glob("*.png"))),
        "tables": len(list(TABLES.glob("*.csv"))),
        "presentation_outline": str(RESULTS / "presentation_outline.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
