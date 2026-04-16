#!/usr/bin/env python3
"""Per-codec EER bias relative to uncoded (NONE), multi-seed averaged.

Generates a 2-panel figure (WavLM + W2V2) showing ΔEER = EER[codec] - EER[NONE]
for ERM, DANN, and ERM+Aug methods. Error bars are std across seeds.

Codecs covered by synthetic augmentation (C05 MP3, C06 AAC) are bolded.

Usage:
    python scripts/plot_per_codec_bias.py \
        --predictions-dir results/predictions \
        --output figures/per_codec_bias.png

Expects directory structure:
    results/predictions/{model}_{seed}_eval/tables/metrics_by_codec.csv
    or
    results/predictions/{model}_seed{N}_eval/tables/metrics_by_codec.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from thesis_style import set_style, PALETTE
except ImportError:
    PALETTE = {}
    def set_style():
        pass


# ── Configuration ────────────────────────────────────────────────────────────
CODEC_ORDER = ["C01", "C02", "C03", "C04", "C05", "C06",
               "C07", "C08", "C09", "C10", "C11"]

# Codecs covered by our synthetic augmentation (MP3, AAC)
AUGMENTED_CODECS = {"C05", "C06"}

METHODS = ["erm", "dann", "erm_aug"]
METHOD_LABELS = {
    "erm":     "ERM",
    "dann":    "DANN",
    "erm_aug": "ERM+Aug",
}

BACKBONES = ["wavlm", "w2v2"]
BACKBONE_LABELS = {
    "wavlm": "WavLM",
    "w2v2":  "Wav2Vec 2.0",
}

# Default colors (terracotta/teal/amber family from thesis)
DEFAULT_COLORS = {
    "erm":     "#D4795A",   # terracotta
    "dann":    "#4CA08A",   # teal
    "erm_aug": "#C49A3C",   # gold
}


def find_seed_dirs(predictions_dir: Path, backbone: str, method: str) -> List[Path]:
    """Find all seed directories for a given backbone/method combination.

    Looks for patterns like:
      {backbone}_{method}_eval/
      {backbone}_{method}_seed{N}_eval/
      {backbone}_{method}_seed{N}_v2_eval/
    """
    prefix = f"{backbone}_{method}"
    dirs = []
    for p in predictions_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        # Match {backbone}_{method}_*_eval
        if name.startswith(prefix + "_") and name.endswith("_eval"):
            # Exclude wrong matches like wavlm_erm_* matching wavlm_erm_aug
            remainder = name[len(prefix)+1:-5]  # strip prefix_ and _eval
            # Must be empty, or start with "seed" or match "seedN_v2"
            if remainder == "" or remainder.startswith("seed"):
                dirs.append(p)
    return sorted(dirs)


def load_per_codec_eer(eval_dir: Path) -> Dict[str, float]:
    """Load per-codec EER from metrics_by_codec.csv."""
    csv_path = eval_dir / "tables" / "metrics_by_codec.csv"
    if not csv_path.exists():
        csv_path = eval_dir / "metrics_by_codec.csv"
    if not csv_path.exists():
        return {}

    codec_eer = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            codec = row.get("domain") or row.get("codec")
            eer = float(row["eer"])
            codec_eer[codec] = eer * 100  # convert to percentage
    return codec_eer


def compute_bias_to_none(predictions_dir: Path, backbone: str, method: str
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean ± std bias across seeds for a backbone/method."""
    seed_dirs = find_seed_dirs(predictions_dir, backbone, method)
    if not seed_dirs:
        print(f"  ⚠ No seeds found for {backbone}_{method}")
        return None, None

    per_seed_bias = []
    for seed_dir in seed_dirs:
        codec_eer = load_per_codec_eer(seed_dir)
        if "NONE" not in codec_eer:
            print(f"  ⚠ No NONE in {seed_dir.name}")
            continue
        none_eer = codec_eer["NONE"]
        bias = np.array([codec_eer.get(c, np.nan) - none_eer for c in CODEC_ORDER])
        per_seed_bias.append(bias)

    if not per_seed_bias:
        return None, None

    arr = np.array(per_seed_bias)
    print(f"  {backbone}_{method}: {len(per_seed_bias)} seeds")
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def plot_panel(ax, predictions_dir: Path, backbone: str, colors: Dict[str, str]):
    """Plot one panel (one backbone)."""
    x = np.arange(len(CODEC_ORDER))
    width = 0.27

    for i, method in enumerate(METHODS):
        mean, std = compute_bias_to_none(predictions_dir, backbone, method)
        if mean is None:
            continue
        offset = (i - 1) * width
        ax.bar(x + offset, mean, width,
               color=colors[method], edgecolor='white', linewidth=0.5,
               yerr=std, capsize=2.5,
               error_kw={'linewidth': 0.9, 'ecolor': '#444'},
               label=METHOD_LABELS[method])

    ax.set_xticks(x)
    # Bold codecs covered by synthetic augmentation
    labels = []
    for c in CODEC_ORDER:
        if c in AUGMENTED_CODECS:
            labels.append(r"$\mathbf{" + c + r"}$")
        else:
            labels.append(c)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("ΔEER vs NONE (%)", fontsize=10)
    ax.set_title(f"{BACKBONE_LABELS[backbone]}", fontsize=12, fontweight='bold')
    ax.axhline(0, color='#333', linewidth=0.8, zorder=0)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper left', frameon=False, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description='Per-codec bias-to-NONE plot')
    parser.add_argument('--predictions-dir', type=Path,
                        default=Path('results/predictions'),
                        help='Directory with {model}_eval/ subdirs')
    parser.add_argument('--output', type=Path,
                        default=Path('figures/per_codec_bias.png'))
    parser.add_argument('--use-thesis-style', action='store_true',
                        help='Use thesis_style.py PALETTE (if available)')
    args = parser.parse_args()

    set_style()

    # Choose colors
    if args.use_thesis_style and PALETTE:
        colors = {
            "erm":     PALETTE.get("wavlm_erm",     DEFAULT_COLORS["erm"]),
            "dann":    PALETTE.get("wavlm_dann",    DEFAULT_COLORS["dann"]),
            "erm_aug": PALETTE.get("wavlm_erm_aug", DEFAULT_COLORS["erm_aug"]),
        }
    else:
        colors = DEFAULT_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, backbone in zip(axes, BACKBONES):
        print(f"\n[{backbone}]")
        plot_panel(ax, args.predictions_dir, backbone, colors)

    # Shared y-axis: compute range across both panels
    # sharey=True already handles this, but let's ensure clean limits
    all_ylim = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in all_ylim)
    ymax = max(y[1] for y in all_ylim)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Per-Codec EER Bias Relative to Uncoded (NONE) — multi-seed",
                 fontsize=13, fontweight='bold', y=1.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    pdf_path = args.output.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved: {args.output}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
