#!/usr/bin/env python3
"""Learned layer-pooling weights — multi-seed, both backbones.

2x2 panel layout:
    (a) WavLM   pool weights (ERM + DANN, mean +/- std across 3 seeds)
    (b) Wav2Vec 2.0 pool weights
    (c) WavLM   DANN - ERM weight shift
    (d) Wav2Vec 2.0 DANN - ERM weight shift

Bottom-row panels share a common y-axis so WavLM's near-zero redistribution
is immediately visible as a contrast against W2V2's substantial shifts.

Uses the thesis terracotta/teal palette.

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
from thesis_style import set_style


ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT / "results" / "pool_weights"
OUT_DIR = ROOT / "figures" / "rq4"

COLORS = {
    "wavlm_erm":  "#D4795A",  # terracotta
    "w2v2_erm":   "#E9B69C",  # terracotta (light)
    "wavlm_dann": "#4CA08A",  # teal
    "w2v2_dann":  "#8EC6B4",  # teal (light)
}

ERM_SEEDS = ["seed42", "seed123", "seed456", "seed789", "seed2024"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456", "seed789", "seed2024"]
BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}
N_LAYERS = 12
PANEL_TAGS = {("wavlm", "weights"): "(a)", ("w2v2", "weights"): "(b)",
              ("wavlm", "delta"):   "(c)", ("w2v2", "delta"):   "(d)"}


def load(name: str) -> np.ndarray:
    return np.array(json.loads((POOL_DIR / f"{name}.json").read_text())["softmax_weights"]) * 100


def stack(backbone: str, method: str) -> np.ndarray:
    seeds = ERM_SEEDS if method == "erm" else DANN_SEEDS
    return np.vstack([load(f"{backbone}_{method}_{s}") for s in seeds])


def _panel_label(ax, tag: str) -> None:
    ax.text(
        0.01, 0.985, tag,
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="top", ha="left",
    )


def plot_weights_panel(ax, backbone: str) -> None:
    erm = stack(backbone, "erm")
    dann = stack(backbone, "dann")
    x = np.arange(N_LAYERS)
    width = 0.38

    ax.bar(
        x - width / 2, erm.mean(0), width,
        yerr=erm.std(0, ddof=1), capsize=3,
        color=COLORS[f"{backbone}_erm"], edgecolor="#1F2937", linewidth=0.5,
        error_kw={"linewidth": 0.7, "ecolor": "#374151"},
        label="ERM (3 seeds)",
    )
    ax.bar(
        x + width / 2, dann.mean(0), width,
        yerr=dann.std(0, ddof=1), capsize=3,
        color=COLORS[f"{backbone}_dann"], edgecolor="#1F2937", linewidth=0.5,
        error_kw={"linewidth": 0.7, "ecolor": "#374151"},
        label="DANN (3 seeds)",
    )

    ax.axhline(
        100 / N_LAYERS, color="#6B7280", linestyle="--",
        linewidth=0.9, alpha=0.7, label=f"Uniform ({100/N_LAYERS:.1f}%)",
    )

    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Pooling weight (%)", fontsize=11)
    ax.set_title(BACKBONE_LABEL[backbone], fontsize=12, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
    ax.set_ylim(0, max(erm.mean(0).max(), dann.mean(0).max()) * 1.28)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    _panel_label(ax, PANEL_TAGS[(backbone, "weights")])


def plot_delta_panel(ax, backbone: str, ylim: tuple[float, float]) -> None:
    erm = stack(backbone, "erm")
    dann = stack(backbone, "dann")
    per_seed = dann - erm
    mean = per_seed.mean(0)
    std = per_seed.std(0, ddof=1)
    x = np.arange(N_LAYERS)

    # Color by sign: negative = DANN down-weighted (teal), positive = ERM-like (terracotta)
    colors = [COLORS[f"{backbone}_dann"] if d < 0 else COLORS[f"{backbone}_erm"] for d in mean]
    ax.bar(
        x, mean, yerr=std, capsize=3,
        color=colors, edgecolor="#1F2937", linewidth=0.5,
        error_kw={"linewidth": 0.7, "ecolor": "#374151"},
    )
    ax.axhline(0, color="#1F2937", linewidth=0.8)

    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Δ weight: DANN − ERM (pp)", fontsize=11)
    ax.set_title(f"{BACKBONE_LABEL[backbone]} — weight shift",
                 fontsize=12, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
    ax.set_ylim(*ylim)
    _panel_label(ax, PANEL_TAGS[(backbone, "delta")])


def main() -> None:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    plot_weights_panel(axes[0, 0], "wavlm")
    plot_weights_panel(axes[0, 1], "w2v2")

    # Shared Δ-axis: use the larger (W2V2) range on both so WavLM's flat
    # bars make the backbone contrast immediately obvious.
    w2v2_per_seed = stack("w2v2", "dann") - stack("w2v2", "erm")
    mag = float(np.max(np.abs(w2v2_per_seed.mean(0)) + w2v2_per_seed.std(0, ddof=1))) * 1.25
    mag = max(mag, 2.5)
    shared_ylim = (-mag, mag)

    plot_delta_panel(axes[1, 0], "wavlm", shared_ylim)
    plot_delta_panel(axes[1, 1], "w2v2", shared_ylim)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"pooling_weights_comparison.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
