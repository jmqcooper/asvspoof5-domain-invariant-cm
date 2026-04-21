#!/usr/bin/env python3
"""Learned layer-pooling weights — multi-seed, both backbones.

Reads softmax-normalized pool weights from results/pool_weights/*.json
(produced by scripts/extract_pool_weights.py) and plots mean ± std across
three seeds for ERM and DANN, for WavLM and W2V2 side-by-side.

Previous version hard-coded values from a single seed-42 checkpoint that
has since been overwritten. The current script loads from disk so the
figure always reflects the actual trained checkpoints.

Usage:
    python scripts/plot_pooling_weights.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import PALETTE, set_style


ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT / "results" / "pool_weights"
OUT_DIR = ROOT / "figures" / "rq4"

ERM_SEEDS = ["seed42", "seed123", "seed456"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456"]
BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}
N_LAYERS = 12


def load(name: str) -> np.ndarray:
    path = POOL_DIR / f"{name}.json"
    return np.array(json.loads(path.read_text())["softmax_weights"]) * 100


def stack(backbone: str, method: str) -> np.ndarray:
    """Return (n_seeds, 12) array of pool weights in percent."""
    seeds = ERM_SEEDS if method == "erm" else DANN_SEEDS
    return np.vstack([load(f"{backbone}_{method}_{s}") for s in seeds])


def plot_weights_panel(ax, backbone: str) -> None:
    erm = stack(backbone, "erm")
    dann = stack(backbone, "dann")
    x = np.arange(N_LAYERS)
    width = 0.38

    ax.bar(x - width / 2, erm.mean(0), width,
           yerr=erm.std(0, ddof=1), capsize=3,
           color=PALETTE[f"{backbone}_erm"], edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 0.8, "ecolor": "#374151"},
           label="ERM (3 seeds)")
    ax.bar(x + width / 2, dann.mean(0), width,
           yerr=dann.std(0, ddof=1), capsize=3,
           color=PALETTE[f"{backbone}_dann"], edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 0.8, "ecolor": "#374151"},
           label="DANN (3 seeds)")

    ax.axhline(100 / N_LAYERS, color="#6B7280", linestyle="--",
               linewidth=0.9, alpha=0.7, label=f"Uniform ({100/N_LAYERS:.1f}%)")

    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Pooling weight (%)", fontsize=11)
    ax.set_title(BACKBONE_LABEL[backbone], fontsize=13, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
    ymax = max(erm.mean(0).max(), dann.mean(0).max()) * 1.25
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9, loc="upper right", ncols=1)


def plot_delta_panel(ax, backbone: str) -> None:
    erm = stack(backbone, "erm")
    dann = stack(backbone, "dann")
    # Per-seed pairing (same seed index in erm/dann stacks)
    per_seed = dann - erm
    mean = per_seed.mean(0)
    std = per_seed.std(0, ddof=1)
    x = np.arange(N_LAYERS)

    colors = [PALETTE[f"{backbone}_dann"] if d < 0 else PALETTE[f"{backbone}_erm"]
              for d in mean]
    ax.bar(x, mean, yerr=std, capsize=3,
           color=colors, edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 0.8, "ecolor": "#374151"})
    ax.axhline(0, color="#374151", linewidth=0.6)

    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Δ weight: DANN − ERM (pp)", fontsize=11)
    ax.set_title(f"{BACKBONE_LABEL[backbone]} — weight shift",
                 fontsize=12, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
    lim = max(np.max(np.abs(mean + std)), 0.3) * 1.25
    ax.set_ylim(-lim, lim)


def main() -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_weights_panel(axes[0, 0], "wavlm")
    plot_weights_panel(axes[0, 1], "w2v2")
    plot_delta_panel(axes[1, 0], "wavlm")
    plot_delta_panel(axes[1, 1], "w2v2")

    fig.suptitle(
        "Learned layer-pooling weights (mean ± std across 3 seeds)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"pooling_weights_comparison.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
