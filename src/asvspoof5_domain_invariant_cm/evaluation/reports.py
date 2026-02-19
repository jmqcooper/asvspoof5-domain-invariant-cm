"""Reporting utilities for evaluation results."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .domain_eval import evaluate_per_domain
from .metrics import (
    bootstrap_metric,
    compute_auc,
    compute_eer,
    compute_min_dcf,
    compute_tdcf,
    compute_threshold_metrics,
)


def generate_overall_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    asv_scores: Optional[np.ndarray] = None,
    asv_labels: Optional[np.ndarray] = None,
) -> dict:
    """Generate overall metrics with optional bootstrap CIs.

    Args:
        scores: Detection scores (higher = more bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        bootstrap: Whether to compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.
        asv_scores: ASV verification scores (for t-DCF computation).
        asv_labels: ASV labels (for t-DCF computation).

    Returns:
        Dictionary of metrics including:
        - EER and threshold
        - minDCF
        - AUC
        - F1, Precision, Recall at EER threshold
        - t-DCF (if ASV scores provided)
        - Sample counts
    """
    eer, eer_threshold = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)
    auc = compute_auc(scores, labels)

    # Compute threshold-based metrics at EER threshold
    threshold_metrics = compute_threshold_metrics(scores, labels, eer_threshold)

    # Compute t-DCF (will return note if ASV scores unavailable)
    tdcf_result = compute_tdcf(
        scores, labels,
        asv_scores=asv_scores,
        asv_labels=asv_labels,
    )

    metrics = {
        # Primary metrics
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "min_dcf": float(min_dcf),
        "auc": float(auc),

        # Threshold-based metrics at EER operating point
        "f1_macro": threshold_metrics["f1_macro"],
        "precision_macro": threshold_metrics["precision_macro"],
        "recall_macro": threshold_metrics["recall_macro"],
        "f1_bonafide": threshold_metrics["f1_bonafide"],
        "f1_spoof": threshold_metrics["f1_spoof"],
        "precision_bonafide": threshold_metrics["precision_bonafide"],
        "precision_spoof": threshold_metrics["precision_spoof"],
        "recall_bonafide": threshold_metrics["recall_bonafide"],
        "recall_spoof": threshold_metrics["recall_spoof"],

        # t-DCF (ASVspoof-specific)
        "tdcf_min": tdcf_result["min_tdcf"],
        "tdcf_threshold": tdcf_result["tdcf_threshold"],
        "tdcf_asv_available": tdcf_result["asv_available"],
        "tdcf_note": tdcf_result["note"],

        # Sample counts
        "n_samples": len(scores),
        "n_bonafide": int((labels == 0).sum()),
        "n_spoof": int((labels == 1).sum()),
    }

    if bootstrap:
        # EER bootstrap
        eer_mean, eer_lower, eer_upper = bootstrap_metric(
            scores, labels, compute_eer_only, n_bootstrap, confidence, seed
        )
        metrics["eer_ci_lower"] = float(eer_lower)
        metrics["eer_ci_upper"] = float(eer_upper)

        # minDCF bootstrap
        dcf_mean, dcf_lower, dcf_upper = bootstrap_metric(
            scores, labels, compute_min_dcf, n_bootstrap, confidence, seed
        )
        metrics["min_dcf_ci_lower"] = float(dcf_lower)
        metrics["min_dcf_ci_upper"] = float(dcf_upper)

        # AUC bootstrap
        auc_mean, auc_lower, auc_upper = bootstrap_metric(
            scores, labels, compute_auc, n_bootstrap, confidence, seed
        )
        metrics["auc_ci_lower"] = float(auc_lower)
        metrics["auc_ci_upper"] = float(auc_upper)

    return metrics


def compute_eer_only(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute EER only (for bootstrap)."""
    eer, _ = compute_eer(scores, labels)
    return eer


def generate_per_domain_report(
    df: pd.DataFrame,
    domain_col: str,
    score_col: str = "score",
    label_col: str = "y_task",
) -> pd.DataFrame:
    """Generate per-domain metrics report.

    Args:
        df: DataFrame with predictions.
        domain_col: Name of domain column ('codec' or 'codec_q').
        score_col: Name of score column.
        label_col: Name of label column.

    Returns:
        DataFrame with per-domain metrics.
    """
    return evaluate_per_domain(df, score_col, label_col, domain_col)


def save_predictions(
    predictions: list[dict],
    output_path: Path,
) -> None:
    """Save predictions to TSV file.

    Args:
        predictions: List of prediction dictionaries.
        output_path: Output path.
    """
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, sep="\t", index=False)


def save_metrics_report(
    metrics: dict,
    output_path: Path,
) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Metrics dictionary.
        output_path: Output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_domain_tables(
    df: pd.DataFrame,
    output_dir: Path,
    score_col: str = "score",
    label_col: str = "y_task",
) -> dict[str, Path]:
    """Save per-domain breakdown tables.

    Args:
        df: DataFrame with predictions.
        output_dir: Output directory.
        score_col: Name of score column.
        label_col: Name of label column.

    Returns:
        Dictionary mapping domain type to output path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for domain_col in ["codec", "codec_q"]:
        if domain_col in df.columns:
            domain_df = generate_per_domain_report(
                df, domain_col, score_col, label_col
            )
            output_path = output_dir / f"metrics_by_{domain_col}.csv"
            domain_df.to_csv(output_path, index=False)
            paths[domain_col] = output_path

    # Optional: by attack label
    if "attack_label" in df.columns:
        attack_df = generate_per_domain_report(
            df, "attack_label", score_col, label_col
        )
        output_path = output_dir / "metrics_by_attack_label.csv"
        attack_df.to_csv(output_path, index=False)
        paths["attack_label"] = output_path

    return paths


def generate_scorefile(
    df: pd.DataFrame,
    output_path: Path,
    score_col: str = "score",
    id_col: str = "flac_file",
) -> None:
    """Generate score file compatible with official ASVspoof evaluation.

    Format: utterance_id score

    Args:
        df: DataFrame with predictions.
        output_path: Output path.
        score_col: Name of score column.
        id_col: Name of utterance ID column.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row[id_col]} {row[score_col]:.6f}\n")
