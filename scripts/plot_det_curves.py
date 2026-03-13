#!/usr/bin/env python3
"""Plot DET curves for ERM vs DANN models.

Usage:
    # From existing single-seed results
    python scripts/plot_det_curves.py --predictions-dir results/runs/

    # From multi-seed results (averages across seeds)
    python scripts/plot_det_curves.py --predictions-dir results/predictions/

    # Demo mode (synthetic curves for layout testing)
    python scripts/plot_det_curves.py --demo
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import COLORS, STYLE, set_style


# ── Colours ──────────────────────────────────────────────────────────────────
COLOR_MAP = {
    'wavlm_erm':      '#D4795A',
    'wavlm_dann':     '#4CA08A',
    'wavlm_erm_aug':  '#B8623F',
    'w2v2_erm':       '#E8B4A0',
    'w2v2_dann':      '#A5D5C3',
    'w2v2_erm_aug':   '#D49A80',
}

LABEL_MAP = {
    'wavlm_erm':      'WavLM ERM',
    'wavlm_dann':     'WavLM DANN',
    'wavlm_erm_aug':  'WavLM ERM+Aug',
    'w2v2_erm':       'W2V2 ERM',
    'w2v2_dann':      'W2V2 DANN',
    'w2v2_erm_aug':   'W2V2 ERM+Aug',
}

KNOWN_EERS = {
    'wavlm_erm':  8.47,
    'wavlm_dann': 7.34,
    'w2v2_erm':   15.30,
    'w2v2_dann':  14.33,
}

# All model names to search for
ALL_MODELS = ['wavlm_erm', 'wavlm_dann', 'wavlm_erm_aug',
              'w2v2_erm', 'w2v2_dann', 'w2v2_erm_aug']


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_far_frr(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 2000):
    """Compute FAR and FRR at multiple thresholds.

    Convention: labels 0=bonafide, 1=spoof (matching repo convention in
    metrics.py and evaluate.py). Higher scores = more likely bonafide.

    FAR = P(accepted as bonafide | spoof) = spoof above threshold / total spoof
    FRR = P(rejected as spoof | bonafide) = bonafide below threshold / total bonafide
    """
    n_bonafide = (labels == 0).sum()
    n_spoof = (labels == 1).sum()
    if n_bonafide == 0 or n_spoof == 0:
        raise ValueError(f"Need both classes: bonafide={n_bonafide}, spoof={n_spoof}")

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    far = np.zeros(n_thresholds)
    frr = np.zeros(n_thresholds)
    for i, t in enumerate(thresholds):
        # Accept as bonafide if score >= threshold
        far[i] = ((scores >= t) & (labels == 1)).sum() / n_spoof
        frr[i] = ((scores < t) & (labels == 0)).sum() / n_bonafide
    return far, frr


def find_eer(far, frr):
    """Find EER as the intersection of FAR and FRR."""
    diff = far - frr
    idx = np.nanargmin(np.abs(diff))
    eer = (far[idx] + frr[idx]) / 2
    return eer, far[idx], frr[idx]


def generate_synthetic_det(eer_pct: float, n_points: int = 500):
    """Generate synthetic FAR/FRR that pass through the given EER point."""
    eer = eer_pct / 100.0
    far = np.logspace(-3, np.log10(0.99), n_points)
    a = 0.9
    z_eer = norm.ppf(eer)
    b = (1 - a) * z_eer
    frr = norm.cdf(a * norm.ppf(far) + b)
    far = np.clip(far, 1e-4, 1.0)
    frr = np.clip(frr, 1e-4, 1.0)
    return far, frr, eer


def load_predictions(pred_path: Path):
    """Load predictions TSV/CSV and return scores, labels."""
    import pandas as pd
    sep = '\t' if pred_path.suffix == '.tsv' else ','
    df = pd.read_csv(pred_path, sep=sep)
    scores = df['score'].values
    labels = df['y_task'].values
    return scores, labels


def find_prediction_files(pred_dir: Path, model_name: str) -> list[Path]:
    """Find all prediction files for a model, including multi-seed variants.

    Searches for:
    1. {pred_dir}/{model_name}/eval_eval_full/predictions.tsv  (existing runs/)
    2. {pred_dir}/{model_name}/eval_eval/predictions.tsv
    3. {pred_dir}/{model_name}_eval/predictions.tsv  (multi-seed predictions/)
    4. {pred_dir}/{model_name}_seed*_eval/predictions.tsv  (multi-seed per-seed)
    5. {pred_dir}/{model_name}_eval.csv  (legacy flat CSV)
    """
    found = []

    # Existing runs structure
    for subdir in ['eval_eval_full', 'eval_eval']:
        p = pred_dir / model_name / subdir / 'predictions.tsv'
        if p.exists():
            found.append(p)

    # Multi-seed job output
    p = pred_dir / f'{model_name}_eval' / 'predictions.tsv'
    if p.exists():
        found.append(p)

    # Multi-seed per-seed directories
    for seed_dir in sorted(pred_dir.glob(f'{model_name}_seed*_eval')):
        p = seed_dir / 'predictions.tsv'
        if p.exists():
            found.append(p)

    # Legacy flat CSV
    p = pred_dir / f'{model_name}_eval.csv'
    if p.exists():
        found.append(p)

    return found


def load_model_predictions(pred_dir: Path, model_name: str):
    """Load and concatenate predictions for a model (potentially across seeds).

    Returns combined scores and labels from all found prediction files.
    For multi-seed, this effectively pools all predictions for a more
    robust DET curve estimate.
    """
    files = find_prediction_files(pred_dir, model_name)
    if not files:
        return None, None

    all_scores = []
    all_labels = []
    for f in files:
        print(f'  Loading {model_name} from {f}')
        scores, labels = load_predictions(f)
        all_scores.append(scores)
        all_labels.append(labels)

    # Use first file only (seed 42) for DET curve to avoid duplicating
    # the same eval set across seeds. If seeds produce different scores
    # on the same eval set, we take the first (canonical seed).
    # For true multi-seed averaging, we'd need to average score distributions.
    return all_scores[0], all_labels[0]


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_det_panel(ax, models: dict, title: str):
    """Plot DET curves on a single axis.

    models: dict of {name: (far, frr, eer)}
    """
    for name, (far, frr, eer) in models.items():
        color = COLOR_MAP.get(name, '#888888')
        label_name = LABEL_MAP.get(name, name.replace('_', ' ').upper())
        eer_pct = eer * 100
        ax.plot(far * 100, frr * 100, color=color, linewidth=2,
                label=f'{label_name} (EER={eer_pct:.2f}%)')
        # Mark EER point
        ax.plot(eer_pct, eer_pct, 'o', color=color, markersize=8, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('False Acceptance Rate (%)')
    ax.set_ylabel('False Rejection Rate (%)')
    ax.set_title(title)
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.1, 100)
    # Diagonal reference
    diag = np.logspace(-1, 2, 100)
    ax.plot(diag, diag, '--', color=STYLE['GRID'], linewidth=1, zorder=0)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description='Plot DET curves for ERM vs DANN')
    parser.add_argument('--predictions-dir', type=str, default=None,
                        help='Directory with prediction files (results/runs/ or results/predictions/)')
    parser.add_argument('--output', type=str,
                        default='figures/det_curves')
    parser.add_argument('--demo', action='store_true',
                        help='Generate synthetic DET curves for layout testing')
    args = parser.parse_args()

    set_style()

    if args.demo:
        wavlm_models = {}
        w2v2_models = {}
        for name, eer_pct in KNOWN_EERS.items():
            far, frr, eer = generate_synthetic_det(eer_pct)
            if name.startswith('wavlm'):
                wavlm_models[name] = (far, frr, eer)
            else:
                w2v2_models[name] = (far, frr, eer)
    else:
        if args.predictions_dir is None:
            parser.error('--predictions-dir is required when not using --demo')
        pred_dir = Path(args.predictions_dir)
        wavlm_models = {}
        w2v2_models = {}
        for name in ALL_MODELS:
            scores, labels = load_model_predictions(pred_dir, name)
            if scores is None:
                print(f'Warning: no predictions found for {name}, skipping')
                continue
            far, frr = compute_far_frr(scores, labels)
            eer, _, _ = find_eer(far, frr)
            print(f'  {name}: EER={eer*100:.2f}%')
            if name.startswith('wavlm'):
                wavlm_models[name] = (far, frr, eer)
            else:
                w2v2_models[name] = (far, frr, eer)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_det_panel(ax1, wavlm_models, '(a) WavLM')
    plot_det_panel(ax2, w2v2_models, '(b) Wav2Vec2')
    fig.suptitle('DET Curves: ERM vs DANN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_base = args.output.removesuffix('.png').removesuffix('.pdf')
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.pdf']:
        out_path = output_base + ext
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
