"""Evaluation metrics and reporting."""

from .domain_eval import (
    aggregate_domain_results,
    compute_domain_gap,
    evaluate_per_domain,
    held_out_domain_evaluation,
)
from .metrics import (
    bootstrap_metric,
    compute_act_dcf,
    compute_auc,
    compute_cllr,
    compute_eer,
    compute_min_dcf,
    compute_tdcf,
    compute_threshold_metrics,
)
from .reports import (
    generate_overall_metrics,
    generate_per_domain_report,
    generate_scorefile,
    save_domain_tables,
    save_metrics_report,
    save_predictions,
)

__all__ = [
    # Metrics
    "compute_eer",
    "compute_min_dcf",
    "compute_cllr",
    "compute_act_dcf",
    "compute_auc",
    "compute_threshold_metrics",
    "compute_tdcf",
    "bootstrap_metric",
    # Domain evaluation
    "evaluate_per_domain",
    "compute_domain_gap",
    "held_out_domain_evaluation",
    "aggregate_domain_results",
    # Reports
    "generate_overall_metrics",
    "generate_per_domain_report",
    "save_predictions",
    "save_metrics_report",
    "save_domain_tables",
    "generate_scorefile",
]
