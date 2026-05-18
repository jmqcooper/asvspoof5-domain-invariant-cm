#!/usr/bin/env python3
"""Component transplant results — grouped bars per intervention mode.

Two-panel figure:
  Panel A — Delta EER (pp) vs ERM baseline.
  Panel B — max per-layer codec probe accuracy (%).

Each panel groups by intervention mode (pool_weight_transplant,
layer_patch_mixed, layer_patch_repr, layer_patch_hidden) with paired bars
for WavLM (bold terracotta/teal variant) and Wav2Vec 2.0 (lighter).
Error bars = std across 3 seeds.

Uses the thesis terracotta/teal palette. No title — the LaTeX caption
provides the heading.

Usage:
    python scripts/plot_transplant_results.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import set_style


ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "results" / "component_transplant"
OUT_DIR = ROOT / "figures" / "rq4"

# Transplant is its own comparison dimension (not ERM-vs-DANN), so use the
# gold/amber family the thesis already reserves for "transplant / augment"
# operations (see plot_per_codec_bias ERM+Aug). WavLM = bold gold,
# W2V2 = lighter gold.
COLORS = {
    "wavlm": "#C49A3C",  # gold (bold)
    "w2v2":  "#E6C878",  # gold (light)
}

MODES = [
    "pool_weight_transplant",
    "layer_patch_mixed",
    "layer_patch_repr",
    "layer_patch_hidden",
]
MODE_LABELS = {
    "pool_weight_transplant": "Pool\nweights",
    "layer_patch_mixed":      "Mixed\nrepr.",
    "layer_patch_repr":       "Projection",
    "layer_patch_hidden":     "Hidden\nstates",
}
BACKBONES = ["wavlm", "w2v2"]
SEED_TAGS = ["seed42", "seed123", "seed456", "seed789", "seed2024"]
BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}


def load_all_results() -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for bb in BACKBONES:
        for seed in SEED_TAGS:
            path = IN_DIR / f"{bb}_{seed}_results_summary.csv"
            if not path.exists():
                print(f"WARN: missing {path}", file=sys.stderr)
                continue
            with path.open() as fh:
                for row in csv.DictReader(fh):
                    out[(bb, row["mode"])].append(row)
    return out


def aggregate(rows: list[dict], field: str, scale: float = 1.0) -> tuple[float, float]:
    vals = [float(r[field]) * scale for r in rows]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1 if len(vals) > 1 else 0))


def plot_panel(ax, data, field: str, scale: float, ylabel: str,
               zero_line: bool, panel_label: str) -> None:
    x = np.arange(len(MODES))
    width = 0.36

    for i, bb in enumerate(BACKBONES):
        means, stds = [], []
        for m in MODES:
            mean, std = aggregate(data.get((bb, m), []), field, scale)
            means.append(mean)
            stds.append(std)
        offset = (-width / 2) if i == 0 else (+width / 2)
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            color=COLORS[bb],
            edgecolor="#1F2937",
            linewidth=0.8,
            error_kw={"linewidth": 0.8, "ecolor": "#374151"},
            label=BACKBONE_LABEL[bb],
        )

    if zero_line:
        ax.axhline(0, color="#1F2937", linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.text(
        0.01, 0.985, panel_label,
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="top", ha="left",
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.92)


def main() -> None:
    set_style()
    data = load_all_results()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    plot_panel(
        axes[0], data,
        field="delta_eer_vs_base", scale=100.0,
        ylabel="Δ EER vs. ERM baseline (pp)",
        zero_line=True,
        panel_label="(a)",
    )
    plot_panel(
        axes[1], data,
        field="max_probe_acc", scale=100.0,
        ylabel="Max per-layer codec probe accuracy (%)",
        zero_line=False,
        panel_label="(b)",
    )

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"component_transplant.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
