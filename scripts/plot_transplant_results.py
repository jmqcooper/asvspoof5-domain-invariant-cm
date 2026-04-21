#!/usr/bin/env python3
"""Component transplant results — ΔEER and Δprobe per intervention mode.

Two-panel figure:
  Panel A — ΔEER (pp) vs ERM baseline, grouped bars.
  Panel B — max probe codec-leakage accuracy (%), grouped bars.

For each intervention mode (pool_weight_transplant, layer_patch_repr,
layer_patch_mixed, layer_patch_hidden) we show WavLM and W2V2 bars side-by-
side with error bars = std across 3 seeds. W2V2 bars rendered in the full
DANN colour; WavLM bars desaturated to emphasise the backbone contrast.

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
from thesis_style import PALETTE, set_style


ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "results" / "component_transplant"
OUT_DIR = ROOT / "figures" / "rq4"

MODES = [
    "pool_weight_transplant",
    "layer_patch_mixed",
    "layer_patch_repr",
    "layer_patch_hidden",
]
MODE_LABELS = {
    "pool_weight_transplant": "Pool weights",
    "layer_patch_mixed":      "Mixed repr.",
    "layer_patch_repr":       "Projection",
    "layer_patch_hidden":     "Hidden states",
}
BACKBONES = ["wavlm", "w2v2"]
SEED_TAGS = ["seed42", "seed123", "seed456"]
BACKBONE_LABEL = {"wavlm": "WavLM", "w2v2": "Wav2Vec 2.0"}


def load_all_results() -> dict[tuple[str, str], list[dict]]:
    """Returns {(backbone, mode): [row_for_seed42, row_for_seed123, ...]}"""
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


def _bar_kwargs(backbone: str) -> dict:
    """Follow thesis palette: WavLM bold, W2V2 lighter. Story reads through
    bar heights (W2V2 pool-weight bar is dominant because the effect is ~1.6 pp
    vs ~0.05 pp for WavLM)."""
    return dict(
        color=PALETTE[f"{backbone}_dann"],
        edgecolor="black",
        linewidth=0.8,
    )


def plot_panel(ax, data, field: str, scale: float, ylabel: str,
               zero_line: bool, title: str) -> None:
    x = np.arange(len(MODES))
    width = 0.36

    for i, bb in enumerate(BACKBONES):
        means, stds = [], []
        for m in MODES:
            mean, std = aggregate(data.get((bb, m), []), field, scale)
            means.append(mean)
            stds.append(std)
        kwargs = _bar_kwargs(bb)
        offset = (-width / 2) if i == 0 else (+width / 2)
        ax.bar(x + offset, means, width,
               yerr=stds, capsize=3,
               label=BACKBONE_LABEL[bb],
               error_kw={"linewidth": 0.8, "ecolor": "#374151"},
               **kwargs)

    if zero_line:
        ax.axhline(0, color="#374151", linewidth=0.7, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.legend(loc="best", fontsize=10)


def main() -> None:
    set_style()
    data = load_all_results()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel A: Δ EER (pp)
    plot_panel(
        axes[0], data,
        field="delta_eer_vs_base", scale=100.0,
        ylabel="Δ EER vs. ERM baseline (pp)",
        zero_line=True,
        title="Causal effect on EER",
    )
    # Panel B: max probe accuracy (absolute %)
    plot_panel(
        axes[1], data,
        field="max_probe_acc", scale=100.0,
        ylabel="Max per-layer codec probe accuracy (%)",
        zero_line=False,
        title="Post-intervention codec leakage",
    )

    fig.suptitle(
        "Component transplant: W2V2 pool-weight transplant is the dominant causal effect",
        fontsize=14, fontweight="bold", y=1.02,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"component_transplant.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
