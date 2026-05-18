#!/usr/bin/env python3
"""Multi-seed out-of-distribution gap figure.

Paired dev / eval EER bars per (backbone, method), with mean +/- std error
bars across 5 training seeds and per-seed scatter dots overlaid.
Terracotta (ERM) / teal (DANN) palette, dev = lighter shade, eval = bold
shade. Matches the RQ4 mechanism plots.

Replaces the single-seed figure in scripts/plot_ood_gap.py.

Usage:
    python scripts/plot_ood_gap_multi_seed.py
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
DEV_JSON = ROOT / "results" / "dev_eer_multiseed.json"
OUT_DIR = ROOT / "figures"

# Pale shade = Dev (in-domain), bold shade = Eval (out-of-domain).
# Hue = method (ERM terracotta, DANN teal); matches RQ4 mechanism plots.
COLORS = {
    "erm_dev":   "#ECC7B5",
    "erm_eval":  "#D4795A",
    "dann_dev":  "#B2DACC",
    "dann_eval": "#4CA08A",
}

# (clean-name, backbone, method, seed-label) per column on the x-axis
ERM_SEEDS  = ["seed42",    "seed123", "seed456", "seed789", "seed2024"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456", "seed789", "seed2024"]
GROUPS = [
    ("WavLM\nERM",   "wavlm", "erm",  ERM_SEEDS),
    ("WavLM\nDANN",  "wavlm", "dann", DANN_SEEDS),
    ("W2V2\nERM",    "w2v2",  "erm",  ERM_SEEDS),
    ("W2V2\nDANN",   "w2v2",  "dann", DANN_SEEDS),
]

# W2V2 ERM seed-42 dev EER is not in the cluster metrics_train.json
# (older run format). Fall back to the canonical Table 5.1 value.
DEV_SEED42_FALLBACK = {
    "w2v2_erm_seed42": 0.0424,
}


def eval_metrics_path(backbone: str, method: str, seed: str) -> Path:
    """Canonical seed-42 ERM evals live under results/runs/; all other
    multi-seed outputs live under results/predictions/."""
    run = f"{backbone}_{method}_{seed}"
    p = ROOT / "results" / "predictions" / f"{run}_eval" / "metrics.json"
    if p.exists():
        return p
    # Canonical seed-42 non-v2 fallback
    if seed == "seed42":
        p = ROOT / "results" / "runs" / f"{backbone}_{method}" / "eval_eval_full" / "metrics.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"No eval metrics.json for {run}")


def load_dev_eers(backbone: str, method: str, seeds: list[str]) -> np.ndarray:
    data = json.loads(DEV_JSON.read_text())
    out = []
    for seed in seeds:
        key = f"{backbone}_{method}_{seed}"
        entry = data.get(key, {})
        eer = entry.get("best_eer")
        if eer is None:
            eer = DEV_SEED42_FALLBACK.get(key)
            if eer is None:
                raise RuntimeError(f"Missing dev EER for {key} and no fallback")
        out.append(float(eer) * 100)
    return np.array(out)


def load_eval_eers(backbone: str, method: str, seeds: list[str]) -> np.ndarray:
    out = []
    for seed in seeds:
        d = json.loads(eval_metrics_path(backbone, method, seed).read_text())
        out.append(float(d["eer"]) * 100)
    return np.array(out)


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    n = len(GROUPS)
    x = np.arange(n)
    width = 0.38
    rng = np.random.default_rng(0)

    eval_means_all, gaps_means, gaps_stds = [], [], []

    for i, (label, backbone, method, seeds) in enumerate(GROUPS):
        dev = load_dev_eers(backbone, method, seeds)
        evl = load_eval_eers(backbone, method, seeds)
        gap_per_seed = evl - dev
        gaps_means.append(gap_per_seed.mean())
        gaps_stds.append(gap_per_seed.std(ddof=1))
        eval_means_all.append(evl.mean())

        dev_color = COLORS[f"{method}_dev"]
        eval_color = COLORS[f"{method}_eval"]

        # Bars
        ax.bar(
            x[i] - width / 2, dev.mean(), width, yerr=dev.std(ddof=1), capsize=4,
            color=dev_color, edgecolor="#1F2937", linewidth=0.7,
            error_kw={"linewidth": 0.8, "ecolor": "#374151"},
            label="Dev (in-domain)" if i == 0 else None,
        )
        ax.bar(
            x[i] + width / 2, evl.mean(), width, yerr=evl.std(ddof=1), capsize=4,
            color=eval_color, edgecolor="#1F2937", linewidth=0.7,
            error_kw={"linewidth": 0.8, "ecolor": "#374151"},
            label="Eval (out-of-domain)" if i == 0 else None,
        )

        # Per-seed scatter dots
        jitter = rng.uniform(-0.06, 0.06, size=dev.size)
        ax.scatter(x[i] - width / 2 + jitter, dev, s=18, color="white",
                   edgecolor=dev_color, linewidth=1.3, zorder=6)
        ax.scatter(x[i] + width / 2 + jitter, evl, s=18, color="white",
                   edgecolor=eval_color, linewidth=1.3, zorder=6)

    # Gap annotations per group (above the eval bar)
    for i, (label, backbone, method, seeds) in enumerate(GROUPS):
        eval_color = COLORS[f"{method}_eval"]
        y = eval_means_all[i] + 1.2
        ax.text(
            x[i], y,
            f"+{gaps_means[i]:.1f} ± {gaps_stds[i]:.1f} pp",
            ha="center", va="bottom", fontsize=9.5, color=eval_color, fontweight="bold",
        )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=10.5)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=11)
    ax.set_ylim(0, max(eval_means_all) * 1.25)

    # Dashed separator between backbones
    ax.axvline(1.5, color="#6B7280", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=10)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"ood_gap.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
