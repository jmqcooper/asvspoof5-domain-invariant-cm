#!/usr/bin/env python3
"""Pool-weight trajectory across training epochs (DANN seed 42 v2).

For each backbone, shows how DANN's per-layer pooling weights evolve from
initialization through the saved intermediate epoch checkpoints (epoch_0,
epoch_5, epoch_10, ...). Overlaid on top: the matching ERM final pool
weights (averaged across 3 seeds for context). Colorbar / line-color
encodes training epoch + lambda_grl at that epoch.

This figure is the diagnostic: it shows whether DANN's pool-weight
redistribution mechanism would have emerged later in training, even when
"best" save policy froze the checkpoint at low lambda.

Usage:
    python scripts/plot_pool_weight_trajectory.py
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
TRAJ_DIR = ROOT / "results" / "pool_weights_trajectory"
POOL_DIR = ROOT / "results" / "pool_weights"
OUT_DIR = ROOT / "figures" / "rq4"

# Same palette as the rest of the RQ4 plots
COLORS = {
    "wavlm_erm":  "#D4795A",  # terracotta (bold)
    "w2v2_erm":   "#E9B69C",  # terracotta (light)
    "wavlm_dann": "#4CA08A",  # teal (bold)
    "w2v2_dann":  "#8EC6B4",  # teal (light)
}
BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}
N_LAYERS = 12
ERM_SEEDS = ["seed42", "seed123", "seed456", "seed789", "seed2024"]


def load_trajectory(name: str) -> dict:
    return json.loads((TRAJ_DIR / f"{name}.json").read_text())["entries"]


def load_pool(name: str) -> np.ndarray:
    return np.array(json.loads((POOL_DIR / f"{name}.json").read_text())["softmax_weights"]) * 100


def erm_mean(backbone: str) -> np.ndarray:
    return np.mean([load_pool(f"{backbone}_erm_{s}") for s in ERM_SEEDS], axis=0)


def epoch_sort_key(entry: tuple[str, dict]) -> tuple[int, int]:
    """Sort: numeric epochs ascending, then "best", then "last"."""
    label, data = entry
    if label == "best":
        return (1, 0)
    if label == "last":
        return (2, 0)
    ep = data.get("epoch")
    return (0, ep if ep is not None else -1)


def plot_panel(ax, backbone: str) -> None:
    traj = load_trajectory(f"{backbone}_dann_seed42_v2")
    erm_ref = erm_mean(backbone)

    items = sorted(
        ((label, data) for label, data in traj.items() if "softmax_weights" in data),
        key=epoch_sort_key,
    )

    # ERM reference (mean of 3 seeds)
    layers = np.arange(N_LAYERS)
    ax.plot(
        layers, erm_ref, marker="o", linewidth=2.0, markersize=5,
        color=COLORS[f"{backbone}_erm"],
        label=f"ERM (mean of 3 seeds)",
        zorder=4,
    )

    # DANN trajectory: light → dark teal as training progresses
    numeric_items = [(l, d) for l, d in items if l.startswith("epoch_")]
    n = len(numeric_items)
    cmap = plt.get_cmap("Greens")
    for i, (label, data) in enumerate(numeric_items):
        ep = data.get("epoch")
        lam = data.get("lambda_grl")
        w = np.array(data["softmax_weights"]) * 100
        ax.plot(
            layers, w, marker="s", linewidth=1.6, markersize=4,
            color=cmap(0.35 + 0.55 * i / max(n - 1, 1)),
            label=(f"DANN epoch {ep}"
                   + (f" (λ={lam:.2f})" if lam is not None else "")),
            alpha=0.95,
        )

    # "best" and "last" with distinct markers
    for label in ("best", "last"):
        if label not in traj or "softmax_weights" not in traj[label]:
            continue
        data = traj[label]
        ep = data.get("epoch")
        lam = data.get("lambda_grl")
        w = np.array(data["softmax_weights"]) * 100
        marker = "*" if label == "best" else "P"
        ax.plot(
            layers, w, marker=marker, linewidth=1.4, markersize=11,
            color=COLORS[f"{backbone}_dann"],
            label=(f"DANN {label} (epoch {ep}"
                   + (f", λ={lam:.2f}" if lam is not None else "")
                   + ")"),
            markeredgecolor="#1F2937", markeredgewidth=0.8,
            zorder=6,
        )

    ax.axhline(100 / N_LAYERS, color="#6B7280", linestyle="--",
               linewidth=0.9, alpha=0.6,
               label=f"Uniform ({100/N_LAYERS:.1f}%)")

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=9)
    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Pooling weight (%)", fontsize=11)
    ax.set_title(BACKBONE_LABEL[backbone], fontsize=12, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, ncol=1)


def main() -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    plot_panel(axes[0], "wavlm")
    plot_panel(axes[1], "w2v2")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"pool_weight_trajectory.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
