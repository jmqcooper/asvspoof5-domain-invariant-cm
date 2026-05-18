#!/usr/bin/env python3
"""Codec probe profile + DANN-vs-ERM pool-weight shift, per backbone.

Two-panel figure (WavLM | W2V2). On each panel:
  - line plot (left y-axis): per-layer codec probe accuracy (frozen-backbone
    property, identical across seeds).
  - signed bars (right y-axis): mean DANN-ERM pool-weight delta (pp) with
    error bars = std across 3 seeds.

Annotates Pearson r (mean +/- std across 3 seeds) per backbone. Uses the
terracotta/teal/gold palette that matches the rest of the thesis figures.

Usage:
    python scripts/plot_codec_probe_vs_pool_shift.py
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
PROBE_DIR = ROOT / "results" / "domain_probes"
OUT_DIR = ROOT / "figures" / "rq4"

# Thesis palette (terracotta / teal / lighter variants for W2V2)
COLORS = {
    "wavlm_erm":  "#D4795A",  # terracotta (bold)
    "w2v2_erm":   "#E9B69C",  # terracotta (light)
    "wavlm_dann": "#4CA08A",  # teal (bold)
    "w2v2_dann":  "#8EC6B4",  # teal (light)
}

BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}
ERM_SEEDS = ["seed42", "seed123", "seed456", "seed789", "seed2024"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456", "seed789", "seed2024"]


def load_pool(backbone: str, method: str, seed: str) -> np.ndarray:
    return np.array(json.loads((POOL_DIR / f"{backbone}_{method}_{seed}.json").read_text())["softmax_weights"]) * 100


def load_codec_profile(backbone: str) -> np.ndarray:
    per_layer = json.loads((PROBE_DIR / f"{backbone}_erm_seed42" / "probe_results.json").read_text())["model"]["codec"]["per_layer"]
    return np.array([per_layer[str(i)]["accuracy"] * 100 for i in range(12)])


def per_seed_deltas(backbone: str) -> np.ndarray:
    return np.vstack([
        load_pool(backbone, "dann", sd) - load_pool(backbone, "erm", se)
        for se, sd in zip(ERM_SEEDS, DANN_SEEDS)
    ])


def plot_panel(ax, backbone: str) -> None:
    codec = load_codec_profile(backbone)
    deltas = per_seed_deltas(backbone)
    mean_delta = deltas.mean(axis=0)
    std_delta = deltas.std(axis=0, ddof=1)
    rs = np.array([float(np.corrcoef(codec, deltas[i])[0, 1]) for i in range(deltas.shape[0])])
    r_mean, r_std = float(np.mean(rs)), float(np.std(rs, ddof=1))

    layers = np.arange(12)
    erm_color = COLORS[f"{backbone}_erm"]
    dann_color = COLORS[f"{backbone}_dann"]

    # Bars: DANN-ERM pool-weight delta on right y-axis
    ax_r = ax.twinx()
    bar_colors = [dann_color if d < 0 else erm_color for d in mean_delta]
    ax_r.bar(
        layers, mean_delta, yerr=std_delta,
        color=bar_colors, edgecolor="#374151", linewidth=0.4,
        alpha=0.85, capsize=3,
        error_kw={"linewidth": 0.7, "ecolor": "#374151"},
        zorder=2,
    )
    ax_r.axhline(0, color="#374151", linewidth=0.6, zorder=1)
    ax_r.set_ylabel("DANN − ERM pool weight (pp)", fontsize=11)
    ax_r.spines["top"].set_visible(False)
    lim = max(np.max(np.abs(mean_delta + std_delta)), 0.5) * 1.18
    ax_r.set_ylim(-lim, lim)

    # Line: codec probe accuracy on left y-axis
    ax.plot(
        layers, codec, "o-",
        color="#374151", linewidth=2.0, markersize=6,
        markerfacecolor=erm_color, markeredgecolor="#374151", markeredgewidth=1.0,
        zorder=5,
    )
    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("Codec probe accuracy (%)", fontsize=11)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax.set_title(BACKBONE_LABEL[backbone], fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(max(0, codec.min() - 12), min(100, codec.max() + 10))

    # r annotation — upper-right of the line axis, well clear of bars and markers
    ax.text(
        0.985, 0.965,
        f"$r = {r_mean:+.3f} \\pm {r_std:.3f}$  (n=3 seeds)",
        transform=ax.transAxes, fontsize=10,
        va="top", ha="right",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white", edgecolor="#374151",
            alpha=0.92, linewidth=0.8,
        ),
    )


def main() -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, bb in zip(axes, ("wavlm", "w2v2")):
        plot_panel(ax, bb)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"codec_probe_vs_pool_shift.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
