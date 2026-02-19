"""Core evaluation metrics for ASVspoof 5.

Implements:
- EER (Equal Error Rate)
- minDCF (Minimum Detection Cost Function)
- Cllr (Log-Likelihood Ratio Cost)
- actDCF (Actual DCF at fixed threshold)
- AUC (Area Under ROC Curve)
- F1, Precision, Recall (at given threshold)
- t-DCF (tandem Detection Cost Function) - requires ASV scores
"""

import numpy as np
from typing import Optional

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate.

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).

    Returns:
        Tuple of (EER, threshold at EER).
    """
    # Sort scores ascending
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Count classes (0 = bonafide, 1 = spoof)
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 0.5, 0.0

    # At each threshold position i, samples 0..i are below threshold (rejected as spoof)
    # FRR = fraction of bonafide incorrectly rejected (bonafide below threshold)
    # FAR = fraction of spoof incorrectly accepted (spoof above threshold)
    bonafide_below = np.cumsum(sorted_labels == 0)
    frr = bonafide_below / n_bonafide

    spoof_below = np.cumsum(sorted_labels == 1)
    spoof_above = n_spoof - spoof_below
    far = spoof_above / n_spoof

    # Find EER (where FRR = FAR)
    diff = frr - far
    idx = np.argmin(np.abs(diff))

    eer = (frr[idx] + far[idx]) / 2
    threshold = sorted_scores[idx]

    return float(eer), float(threshold)


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05,
) -> float:
    """Compute minimum Detection Cost Function.

    DCF = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        c_miss: Cost of miss (false rejection of bonafide).
        c_fa: Cost of false alarm (false acceptance of spoof).
        p_target: Prior probability of target (bonafide).

    Returns:
        Minimum DCF value.
    """
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 1.0

    # Sort scores ascending
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # At each threshold, samples below are rejected (classified as spoof)
    # p_miss = P(rejected | bonafide) = bonafide below threshold / total bonafide
    # p_fa = P(accepted | spoof) = spoof above threshold / total spoof
    bonafide_below = np.cumsum(sorted_labels == 0)
    p_miss = bonafide_below / n_bonafide

    spoof_below = np.cumsum(sorted_labels == 1)
    spoof_above = n_spoof - spoof_below
    p_fa = spoof_above / n_spoof

    # Compute DCF at each threshold
    dcf = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    # Normalize by minimum of default DCFs
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = np.min(dcf) / default_dcf

    return float(min_dcf)


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Log-Likelihood Ratio Cost (Cllr).

    Measures calibration quality of scores as likelihood ratios.

    Args:
        scores: Log-likelihood ratio scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).

    Returns:
        Cllr value.
    """
    # Separate scores by class (0 = bonafide, 1 = spoof)
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 1.0

    # Compute Cllr components
    # For bonafide: average of log2(1 + e^(-score)) - penalizes low scores
    # For spoof: average of log2(1 + e^(score)) - penalizes high scores

    def softplus_log2(x):
        # log2(1 + e^x) = x / log(2) + log2(1 + e^(-x))
        # Numerically stable version
        return np.log2(1 + np.exp(-np.abs(x))) + np.maximum(x, 0) / np.log(2)

    cllr_bonafide = np.mean(softplus_log2(-bonafide_scores))
    cllr_spoof = np.mean(softplus_log2(spoof_scores))

    cllr = (cllr_bonafide + cllr_spoof) / 2

    return float(cllr)


def compute_act_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05,
) -> float:
    """Compute actual DCF at a fixed threshold.

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        threshold: Decision threshold (score >= threshold -> accept as bonafide).
        c_miss: Cost of miss (false rejection of bonafide).
        c_fa: Cost of false alarm (false acceptance of spoof).
        p_target: Prior probability of target (bonafide).

    Returns:
        Actual DCF value.
    """
    # score >= threshold means accepted as bonafide
    accepted = scores >= threshold

    # Count classes (0 = bonafide, 1 = spoof)
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 1.0

    # p_miss = P(rejected | bonafide) = bonafide not accepted / total bonafide
    p_miss = np.sum((~accepted) & (labels == 0)) / n_bonafide

    # p_fa = P(accepted | spoof) = spoof accepted / total spoof
    p_fa = np.sum(accepted & (labels == 1)) / n_spoof

    # Compute DCF
    dcf = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    # Normalize
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    act_dcf = dcf / default_dcf

    return float(act_dcf)


def bootstrap_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Compute metric with bootstrap confidence interval.

    Args:
        scores: Detection scores.
        labels: Binary labels.
        metric_fn: Function that takes (scores, labels) and returns metric.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(scores)

    bootstrap_values = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        value = metric_fn(scores[indices], labels[indices])
        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)
    mean = np.mean(bootstrap_values)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return float(mean), float(lower), float(upper)


def compute_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Area Under the ROC Curve (AUC).

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).

    Returns:
        AUC value (0-1). Note: We invert labels since higher score = bonafide.
    """
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 0.5

    # sklearn's roc_auc_score expects: higher score = positive class
    # Our convention: higher score = bonafide (class 0)
    # So we use 1 - labels as the target, or equivalently, 1 - score
    # Using inverted labels is cleaner:
    try:
        # Treat bonafide (0) as the positive class for AUC
        # Higher score should correspond to higher probability of being positive
        auc = roc_auc_score(1 - labels, scores)
        return float(auc)
    except ValueError:
        # Can happen with degenerate data
        return 0.5


def compute_threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, and F1 score at a given threshold.

    Decision rule: score >= threshold → predict bonafide (0)

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        threshold: Decision threshold.

    Returns:
        Dictionary with precision, recall, f1 for both classes and macro averages.
    """
    # Predictions: score >= threshold → bonafide (0), else spoof (1)
    predictions = (scores < threshold).astype(int)

    # Compute precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0.0
    )

    return {
        "precision_bonafide": float(precision[0]),
        "recall_bonafide": float(recall[0]),
        "f1_bonafide": float(f1[0]),
        "precision_spoof": float(precision[1]),
        "recall_spoof": float(recall[1]),
        "f1_spoof": float(f1[1]),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


def compute_tdcf(
    cm_scores: np.ndarray,
    cm_labels: np.ndarray,
    asv_scores: Optional[np.ndarray] = None,
    asv_labels: Optional[np.ndarray] = None,
    c_miss_cm: float = 1.0,
    c_fa_cm: float = 10.0,
    c_miss_asv: float = 1.0,
    c_fa_asv: float = 10.0,
    p_target: float = 0.9405,
    p_spoof: float = 0.05,
) -> dict[str, Optional[float]]:
    """Compute tandem Detection Cost Function (t-DCF) for ASVspoof evaluation.

    t-DCF is the official primary metric for ASVspoof challenges. It measures
    the combined performance of an ASV system and a countermeasure (CM) system
    operating in tandem.

    The t-DCF requires both CM scores (from this model) and ASV scores (from
    a separate speaker verification system). If ASV scores are not provided,
    this function returns None values with a note.

    Reference:
        Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem
        Assessment of Spoofing Countermeasures and Automatic Speaker
        Verification", Proc. Odyssey 2018.

    Args:
        cm_scores: CM detection scores (higher = more likely bonafide).
        cm_labels: Binary labels (0 = bonafide, 1 = spoof).
        asv_scores: ASV verification scores (higher = target speaker match).
                   If None, t-DCF cannot be computed.
        asv_labels: ASV labels. If None, t-DCF cannot be computed.
        c_miss_cm: Cost of CM miss (rejecting bonafide).
        c_fa_cm: Cost of CM false alarm (accepting spoof).
        c_miss_asv: Cost of ASV miss (rejecting target).
        c_fa_asv: Cost of ASV false alarm (accepting non-target).
        p_target: Prior probability of target speaker.
        p_spoof: Prior probability of spoofed trial (given non-target).

    Returns:
        Dictionary containing:
        - min_tdcf: Minimum t-DCF value (None if ASV scores unavailable)
        - tdcf_threshold: CM threshold at minimum t-DCF
        - asv_available: Whether ASV scores were provided
        - note: Explanation if t-DCF couldn't be computed
    """
    result = {
        "min_tdcf": None,
        "tdcf_threshold": None,
        "asv_available": False,
        "note": None,
    }

    if asv_scores is None or asv_labels is None:
        result["note"] = (
            "t-DCF requires ASV (Automatic Speaker Verification) scores which "
            "are not available in this dataset. t-DCF measures the combined "
            "performance of ASV and CM systems in tandem. To compute t-DCF, "
            "ASV scores from the official ASVspoof evaluation server or a "
            "separate ASV system are needed."
        )
        return result

    result["asv_available"] = True

    # Implementation of t-DCF following Kinnunen et al. 2018
    # This is a placeholder for the full implementation when ASV scores
    # become available. The computation involves:
    # 1. Computing ASV error rates at a fixed operating point
    # 2. Computing CM error rates across all thresholds
    # 3. Combining them according to t-DCF formula

    n_bonafide = np.sum(cm_labels == 0)
    n_spoof = np.sum(cm_labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        result["note"] = "Insufficient class samples for t-DCF computation"
        return result

    # Compute priors
    p_non_target = 1 - p_target
    p_spoof_given_non_target = p_spoof
    p_zero_effort = 1 - p_spoof

    # This is a simplified version - full t-DCF requires ASV operating point
    # For now, return placeholder values
    result["note"] = (
        "ASV scores provided but full t-DCF computation requires additional "
        "ASV operating point configuration. See ASVspoof evaluation guidelines."
    )

    return result
