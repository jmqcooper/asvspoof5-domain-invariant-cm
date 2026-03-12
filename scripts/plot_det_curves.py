#!/usr/bin/env python3
"""Plot DET curves for ERM vs DANN models.

Usage:
    python scripts/plot_det_curves.py --predictions-dir results/predictions/
    python scripts/plot_det_curves.py --demo  # synthetic curves for layout testing
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, '/root/.openclaw/workspace/master-thesis-uva/figures/')
from thesis_style import COLORS, STYLE, set_style


# ── Colours ──────────────────────────────────────────────────────────────────
COLOR_MAP = {
    'wavlm_erm':  '#D4795A',
    'wavlm_dann': '#4CA08A',
    'w2v2_erm':   '#E8B4A0',
    'w2v2_dann':  '#A5D5C3',
}

KNOWN_EERS = {
    'wavlm_erm':  8.47,
    'wavlm_dann': 7.34,
    'w2v2_erm':   15.30,
    'w2v2_dann':  14.33,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_far_frr(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 1000):
    """Compute FAR and FRR at multiple thresholds. labels: 1=bonafide, 0=spoof."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    far = np.zeros(n_thresholds)
    frr = np.zeros(n_thresholds)
    n_bonafide = (labels == 1).sum()
    n_spoof = (labels == 0).sum()
    for i, t in enumerate(thresholds):
        # Predict bonafide if score >= threshold
        far[i] = ((scores >= t) & (labels == 0)).sum() / max(n_spoof, 1)
        frr[i] = ((scores < t) & (labels == 1)).sum() / max(n_bonafide, 1)
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
    # Use probit (normal deviate) model: FRR = Phi(Phi^{-1}(FAR) + shift)
    # At EER: FAR = FRR = eer, so shift ≈ 0 at that point
    # We parameterise via a spread parameter
    far = np.logspace(-3, np.log10(0.99), n_points)
    # Calibrate: at EER, probit(eer) + shift = probit(eer) → shift=0
    # We need asymmetry. Use: FRR = Phi(a * Phi^{-1}(FAR) + b)
    # At EER: eer = Phi(a * Phi^{-1}(eer) + b)
    # Phi^{-1}(eer) = a * Phi^{-1}(eer) + b → b = (1-a)*Phi^{-1}(eer)
    a = 0.9  # slight asymmetry
    z_eer = norm.ppf(eer)
    b = (1 - a) * z_eer
    frr = norm.cdf(a * norm.ppf(far) + b)
    # Clip
    far = np.clip(far, 1e-4, 1.0)
    frr = np.clip(frr, 1e-4, 1.0)
    return far, frr, eer


def load_predictions(csv_path: Path):
    """Load predictions CSV and return scores, labels."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    scores = df['score'].values
    labels = df['y_task'].values
    return scores, labels


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_det_panel(ax, models: dict, title: str):
    """Plot DET curves on a single axis.

    models: dict of {name: (far, frr, eer)}
    """
    for name, (far, frr, eer) in models.items():
        color = COLOR_MAP[name]
        label_name = name.replace('_', ' ').upper()
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
                        help='Directory with prediction CSVs')
    parser.add_argument('--output', type=str,
                        default='/root/.openclaw/workspace/master-thesis-uva/figures/det_curves')
    parser.add_argument('--demo', action='store_true',
                        help='Generate synthetic DET curves for layout testing')
    args = parser.parse_args()

    set_style()

    if args.demo:
        # Generate synthetic curves
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
        for name in ['wavlm_erm', 'wavlm_dann', 'w2v2_erm', 'w2v2_dann']:
            csv_path = pred_dir / f'{name}_eval.csv'
            if not csv_path.exists():
                print(f'Warning: {csv_path} not found, skipping {name}')
                continue
            scores, labels = load_predictions(csv_path)
            far, frr = compute_far_frr(scores, labels)
            eer, _, _ = find_eer(far, frr)
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
    for ext in ['.png', '.pdf']:
        out_path = output_base + ext
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
