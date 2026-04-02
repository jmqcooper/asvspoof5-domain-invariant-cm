#!/usr/bin/env python3
"""Generate pooling weights comparison figure: ERM vs DANN."""

import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import PALETTE, set_style

# Final learned pooling weights (softmax-normalized) from training logs
# ERM: best checkpoint (epoch 12)
erm_weights = [0.12677821516990662, 0.11494231224060059, 0.11170808225870132,
               0.10658012330532074, 0.09402822703123093, 0.08467204123735428,
               0.07432181388139725, 0.06754684448242188, 0.06348063051700592,
               0.057763781398534775, 0.05290167033672333, 0.04527627304196358]

# DANN: best checkpoint (epoch 6)
dann_weights = [0.07566410303115845, 0.09675665199756622, 0.08147844672203064,
                0.1085059642791748, 0.11087941378355026, 0.09928752481937408,
                0.08613286167383194, 0.08080056309700012, 0.08029918372631073,
                0.071660615503788, 0.057381756603717804, 0.05115285515785217]

layers = list(range(12))
erm_w = np.array(erm_weights)
dann_w = np.array(dann_weights)

set_style()
fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

# --- Left: side-by-side bar chart ---
ax = axes[0]
x = np.arange(len(layers))
width = 0.35

bars_erm = ax.bar(x - width/2, erm_w * 100, width, label="ERM", color=PALETTE["wavlm_erm"], edgecolor="black", linewidth=0.5)
bars_dann = ax.bar(x + width/2, dann_w * 100, width, label="DANN", color=PALETTE["wavlm_dann"], edgecolor="black", linewidth=0.5)

ax.set_xlabel("Transformer Layer", fontsize=13)
ax.set_ylabel("Pooling Weight (%)", fontsize=13)
ax.set_title("Learned Layer Pooling Weights", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"L{i}" for i in layers], fontsize=10)
ax.axhline(y=100/12, color="gray", linestyle="--", alpha=0.5, label="Uniform (8.3%)")
ax.legend(fontsize=11)
ax.set_ylim(0, 15)
ax.grid(axis="y", alpha=0.3)

# --- Right: delta (DANN - ERM) ---
ax2 = axes[1]
delta = (dann_w - erm_w) * 100
colors = [PALETTE["wavlm_dann"] if d > 0 else PALETTE["wavlm_erm"] for d in delta]
ax2.barh(x, delta, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_yticks(x)
ax2.set_yticklabels([f"L{i}" for i in layers], fontsize=10)
ax2.set_xlabel("Δ Weight (DANN − ERM) [pp]", fontsize=12)
ax2.set_title("Weight Shift", fontsize=14, fontweight="bold")
ax2.axvline(x=0, color="black", linewidth=0.8)
ax2.grid(axis="x", alpha=0.3)
ax2.invert_yaxis()

# Annotate key insight
ax2.annotate(
    "DANN shifts weight\nfrom early → middle layers",
    xy=(2.5, 4), fontsize=9, fontstyle="italic",
    ha="center", color="#555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
)

plt.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "figures" / "rq4"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "pooling_weights_comparison.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
