#!/usr/bin/env python3
"""Codec probe profile + DANN-vs-ERM pool-weight shift, per backbone.

Two-panel figure (WavLM | W2V2). On each panel:
  - line plot (left y-axis): per-layer codec probe accuracy (frozen-backbone
    property, identical across seeds — we use any seed-42 run).
  - signed bars (right y-axis): mean DANN-ERM pool-weight delta (pp) with
    error bars = std across 3 seeds.

Annotates Pearson r (mean ± std across 3 seeds) per backbone.

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
from thesis_style import PALETTE, set_style


ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT / "results" / "pool_weights"
PROBE_DIR = ROOT / "results" / "domain_probes"
OUT_DIR = ROOT / "figures" / "rq4"

BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}
ERM_SEEDS = ["seed42", "seed123", "seed456"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456"]


def load_pool(backbone: str, method: str, seed: str) -> np.ndarray:
    path = POOL_DIR / f"{backbone}_{method}_{seed}.json"
    return np.array(json.loads(path.read_text())["softmax_weights"]) * 100


def load_codec_profile(backbone: str) -> np.ndarray:
    # Backbone is frozen -> identical across seeds; pick any seed's ERM probe.
    path = PROBE_DIR / f"{backbone}_erm_seed42" / "probe_results.json"
    per_layer = json.loads(path.read_text())["model"]["codec"]["per_layer"]
    return np.array([per_layer[str(i)]["accuracy"] * 100 for i in range(12)])


def per_seed_deltas(backbone: str) -> np.ndarray:
    """Return (n_seeds, 12) array of DANN-ERM pool-weight deltas in pp."""
    deltas = []
    for se, sd in zip(ERM_SEEDS, DANN_SEEDS):
        deltas.append(load_pool(backbone, "dann", sd) - load_pool(backbone, "erm", se))
    return np.array(deltas)


def per_seed_pearson_r(deltas: np.ndarray, codec: np.ndarray) -> np.ndarray:
    return np.array([float(np.corrcoef(codec, deltas[i])[0, 1]) for i in range(deltas.shape[0])])


def plot_panel(ax, backbone: str) -> None:
    codec = load_codec_profile(backbone)
    deltas = per_seed_deltas(backbone)
    mean_delta = deltas.mean(axis=0)
    std_delta = deltas.std(axis=0, ddof=1)
    rs = per_seed_pearson_r(deltas, codec)
    r_mean, r_std = float(np.mean(rs)), float(np.std(rs, ddof=1))

    layers = np.arange(12)

    # Bars: Δ weight on right y-axis
    ax_r = ax.twinx()
    bar_colors = [PALETTE["wavlm_dann"] if d < 0 else PALETTE["wavlm_erm"]
                  if backbone == "wavlm"
                  else PALETTE["w2v2_dann"] if d < 0 else PALETTE["w2v2_erm"]
                  for d in mean_delta]
    ax_r.bar(layers, mean_delta, yerr=std_delta,
             color=bar_colors, edgecolor="black", linewidth=0.4,
             alpha=0.85, capsize=3, error_kw={"linewidth": 0.7, "ecolor": "#374151"},
             zorder=2)
    ax_r.axhline(0, color="#374151", linewidth=0.6, zorder=1)
    ax_r.set_ylabel("DANN − ERM pool weight (pp)", fontsize=11)
    ax_r.spines["top"].set_visible(False)
    # Symmetric y-range so zero is visually centered
    lim = max(np.max(np.abs(mean_delta + std_delta)), 0.5) * 1.15
    ax_r.set_ylim(-lim, lim)

    # Line: codec probe accuracy on left y-axis
    line_color = PALETTE[f"{backbone}_erm"]
    ax.plot(layers, codec, "o-",
            color=line_color, linewidth=2.2, markersize=7,
            markerfacecolor="white", markeredgewidth=1.8,
            label="Codec probe accuracy", zorder=5)
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Codec probe accuracy (%)", fontsize=11, color=line_color)
    ax.tick_params(axis="y", labelcolor=line_color)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax.set_title(BACKBONE_LABEL[backbone], fontsize=13, fontweight="bold", pad=8)
    ax.set_ylim(max(0, codec.min() - 10), min(100, codec.max() + 8))

    # r annotation
    ax.text(0.03, 0.05,
            f"r = {r_mean:+.3f} ± {r_std:.3f}\n(n=3 seeds)",
            transform=ax.transAxes, fontsize=10.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor=PALETTE[f"{backbone}_dann"],
                      alpha=0.95, linewidth=1.1))


def main() -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, bb in zip(axes, ("wavlm", "w2v2")):
        plot_panel(ax, bb)

    fig.suptitle(
        "Per-layer codec information vs. DANN pool-weight redistribution",
        fontsize=14, fontweight="bold", y=1.02,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"codec_probe_vs_pool_shift.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
