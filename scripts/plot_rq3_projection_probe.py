#!/usr/bin/env python3
"""Projection-output (Lrepr) codec probe — multi-seed, both backbones.

Replaces the single-seed Figure 5.4. Reads the multi-seed probe results
produced by scripts/jobs/probe_multi_seed.job and plots mean +/- std
(across 3 training seeds) of the Lrepr codec probe accuracy for ERM and
DANN on both WavLM and Wav2Vec 2.0. Also annotates the per-seed deltas
so the sign-instability is visible.

Terracotta (ERM) / teal (DANN) palette, WavLM bold / W2V2 lighter,
matching the other RQ4 mechanism plots.

Usage:
    python scripts/plot_rq3_projection_probe.py
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
PROBE_DIR = ROOT / "results" / "domain_probes"
OUT_DIR = ROOT / "figures"

COLORS = {
    "wavlm_erm":  "#D4795A",  # terracotta (bold)
    "w2v2_erm":   "#E9B69C",  # terracotta (light)
    "wavlm_dann": "#4CA08A",  # teal (bold)
    "w2v2_dann":  "#8EC6B4",  # teal (light)
}

ERM_SEEDS = ["seed42", "seed123", "seed456"]
DANN_SEEDS = ["seed42_v2", "seed123", "seed456"]
CHANCE = 1.0 / 12  # 12 codec classes


def lrepr_accuracy(run: str) -> float:
    path = PROBE_DIR / run / "probe_results.json"
    d = json.loads(path.read_text())
    return float(d["model"]["codec"]["per_layer"]["repr"]["accuracy"])


def collect(backbone: str):
    erm = np.array([lrepr_accuracy(f"{backbone}_erm_{s}") for s in ERM_SEEDS])
    dann = np.array([lrepr_accuracy(f"{backbone}_dann_{s}") for s in DANN_SEEDS])
    return erm, dann


def plot_panel(ax, backbone: str, label: str) -> None:
    erm, dann = collect(backbone)
    erm_mean, erm_std = erm.mean() * 100, erm.std(ddof=1) * 100
    dann_mean, dann_std = dann.mean() * 100, dann.std(ddof=1) * 100
    per_seed_delta = (dann - erm) * 100
    delta_mean = per_seed_delta.mean()
    delta_std = per_seed_delta.std(ddof=1)

    x = [0, 1]
    means = [erm_mean, dann_mean]
    stds = [erm_std, dann_std]
    colors = [COLORS[f"{backbone}_erm"], COLORS[f"{backbone}_dann"]]

    bars = ax.bar(
        x, means, width=0.55, yerr=stds, capsize=5,
        color=colors, edgecolor="#1F2937", linewidth=0.7,
        error_kw={"linewidth": 0.9, "ecolor": "#374151"},
    )

    # Per-seed points overlaid on each bar
    rng = np.random.default_rng(0)
    jitter_erm = rng.uniform(-0.08, 0.08, size=len(erm))
    jitter_dann = rng.uniform(-0.08, 0.08, size=len(dann))
    ax.scatter(
        np.zeros(len(erm)) + jitter_erm, erm * 100,
        s=22, color="white", edgecolor=COLORS[f"{backbone}_erm"],
        linewidth=1.4, zorder=5,
    )
    ax.scatter(
        np.ones(len(dann)) + jitter_dann, dann * 100,
        s=22, color="white", edgecolor=COLORS[f"{backbone}_dann"],
        linewidth=1.4, zorder=5,
    )

    # Annotate bar values
    for xi, m, s in zip(x, means, stds):
        ax.text(
            xi, m + s + 1.0, f"{m:.1f} ± {s:.1f}",
            ha="center", va="bottom", fontsize=10, color="#1F2937",
        )

    # Δ annotation
    sign = "+" if delta_mean >= 0 else "−"
    ax.text(
        0.5, max(means) + max(stds) + 6.5,
        f"Δ = {sign}{abs(delta_mean):.2f} ± {delta_std:.2f} pp\n(per-seed: "
        + ", ".join(f"{d:+.2f}" for d in per_seed_delta) + ")",
        transform=ax.transData, ha="center", va="bottom",
        fontsize=9.5, color="#1F2937",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#374151", alpha=0.92, linewidth=0.7),
    )

    ax.axhline(
        CHANCE * 100, color="#6B7280", linestyle="--", linewidth=0.9, alpha=0.7,
    )
    ax.text(
        1.5 - 0.02, CHANCE * 100 + 0.6, f"Chance (1/12)",
        fontsize=8.5, color="#6B7280", ha="right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["ERM", "DANN"], fontsize=11)
    ax.set_ylabel("Codec probe accuracy (%)", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, max(60.0, max(means) + max(stds) + 16))


def main() -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    plot_panel(axes[0], "wavlm", "WavLM")
    plot_panel(axes[1], "w2v2", "Wav2Vec 2.0")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"rq3_projection_probe.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
