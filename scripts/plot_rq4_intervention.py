#!/usr/bin/env python3
"""Generate RQ4 visualizations: CKA analysis and activation patching results.

This script creates publication-ready figures for RQ4:
"Where does DANN achieve domain invariance?"

Figures generated:
1. CKA heatmap: Layer-by-layer similarity between ERM and DANN
2. Intervention ablation: Effect of patching on EER and probe accuracy
3. Layer contribution divergence: Bar chart showing per-layer CKA

Key findings visualized:
- Layer 11 shows dramatic divergence (CKA=0.098) despite frozen backbone
- Projection layer patching reduces domain leakage while preserving EER
- DANN's effect is concentrated in pooling weights + projection head

Usage:
    python scripts/plot_rq4_intervention.py \\
        --cka-results rq4_cka_results.csv \\
        --intervention-results rq4_results_summary.csv \\
        --output-dir figures/rq4

    # From Snellius results
    python scripts/plot_rq4_intervention.py \\
        --cka-results /projects/prjs1904/runs/wavlm_dann/rq4_cka_results.csv \\
        --intervention-results /projects/prjs1904/runs/wavlm_dann/rq4_results_summary.csv \\
        --output-dir figures/rq4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Publication-quality settings
FIGSIZE_SINGLE = (7.2, 4.8)
FIGSIZE_WIDE = (12, 4.8)
DPI = 300

# Style contract matching plot_rq3_combined.py
STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Thesis palette aligned with RQ3 styling
COLORS = {
    "erm": "#E57373",
    "dann": "#64B5F6",
    "wavlm": "#4C72B0",
    "w2v2": "#DD8452",
    "chance": "#9E9E9E",
    "neutral": "#90A4AE",
    "highlight": "#DD8452",
    "info_blue": "#6FA8DC",
    "divergent": "#D32F2F",
}

INTERVENTION_ORDER = [
    "layer_patch_mixed",
    "pool_weight_transplant",
    "layer_patch_repr",
    "layer_patch_hidden",
]

INTERVENTION_LABELS = {
    "layer_patch_hidden": "Baseline\n(no patch)",
    "layer_patch_repr": "Patch\nProjection",
    "layer_patch_mixed": "Patch\nMixed",
    "pool_weight_transplant": "Transplant\nPool Weights",
}


def sort_interventions(df: pd.DataFrame) -> pd.DataFrame:
    """Return intervention rows in a stable, presentation-friendly order."""
    ordered_modes = [mode for mode in INTERVENTION_ORDER if mode in set(df["mode"].values)]
    sorted_df = (
        df.assign(mode=pd.Categorical(df["mode"], categories=ordered_modes, ordered=True))
        .sort_values("mode")
        .copy()
    )
    sorted_df["mode"] = sorted_df["mode"].astype(str)
    return sorted_df


def infer_probe_chance_from_results(df: pd.DataFrame) -> tuple[float, str]:
    """Infer probe chance accuracy from RQ4 results metadata columns."""
    if "probe_chance_acc" in df.columns:
        valid_values = pd.to_numeric(df["probe_chance_acc"], errors="coerce").dropna()
        if not valid_values.empty:
            chance_acc = float(valid_values.mode().iloc[0])
            if chance_acc > 0:
                classes = int(round(1.0 / chance_acc))
                return chance_acc * 100.0, f"Chance ({classes} codecs)"
    if "probe_target_unique" in df.columns:
        valid_values = pd.to_numeric(df["probe_target_unique"], errors="coerce").dropna()
        if not valid_values.empty:
            classes = int(valid_values.mode().iloc[0])
            if classes > 0:
                return 100.0 / classes, f"Chance ({classes} codecs)"

    logger.warning("Could not infer codec cardinality from intervention CSV; defaulting chance to 33.3%%")
    return 33.3, "Chance (3 codecs)"


def plot_cka_layer_bar(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart showing per-layer CKA between ERM and DANN.
    
    Highlights layer 11's dramatic divergence.
    """
    plt.rcParams.update(STYLE_CONFIG)

    # Filter to pool_weight_transplant layer_contrib (the meaningful comparison)
    layer_data = df[
        (df['mode'] == 'pool_weight_transplant') & 
        (df['representation_mode'] == 'layer_contrib')
    ].copy()
    
    if layer_data.empty:
        logger.warning("No pool_weight_transplant layer_contrib data found")
        return
    
    # Sort by layer key numerically
    layer_data['layer_num'] = layer_data['layer_key'].astype(int)
    layer_data = layer_data.sort_values('layer_num')
    
    layers = layer_data['layer_num'].values
    cka_values = layer_data['cka'].values
    
    # Color bars based on divergence
    colors = [COLORS["divergent"] if cka < 0.5 else COLORS["wavlm"] for cka in cka_values]
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    bars = ax.bar(layers, cka_values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    
    # Add horizontal reference lines
    ax.axhline(y=1.0, color=COLORS["chance"], linestyle="--", linewidth=1.5, label="Identical")
    ax.axhline(y=0.5, color="#D9A441", linestyle=":", linewidth=2.0, label="Moderate similarity")
    
    # Highlight layer 11 without overlaying text on bars.
    layer_11_idx = np.where(layers == 11)[0]
    if len(layer_11_idx) > 0:
        idx = layer_11_idx[0]
        ax.text(
            11,
            cka_values[idx] + 0.04,
            f"L11={cka_values[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["divergent"],
        )
    
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("CKA Similarity (ERM vs DANN)")
    ax.set_title("Layer Contribution Similarity: ERM vs DANN", fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.08)
    ax.margins(x=0.02)
    legend_handles = [
        Line2D([0], [0], color=COLORS["chance"], linestyle="--", linewidth=1.5, label="Identical"),
        Line2D([0], [0], color="#D9A441", linestyle=":", linewidth=2.0, label="Moderate similarity"),
        Patch(facecolor=COLORS["divergent"], edgecolor="black", label="Layer 11 divergence"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CKA layer bar chart: {output_path}")


def plot_intervention_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing intervention effects on EER and probe accuracy."""

    plt.rcParams.update(STYLE_CONFIG)
    df_sorted = sort_interventions(df)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    modes = df_sorted["mode"].values
    labels = [INTERVENTION_LABELS.get(mode, mode) for mode in modes]
    eer_values = df_sorted["eer"].values * 100
    probe_values = df_sorted["max_probe_acc"].values * 100
    chance_pct, chance_label = infer_probe_chance_from_results(df_sorted)

    x = np.arange(len(modes))
    width = 0.62

    mode_colors = []
    for mode in modes:
        if mode == "layer_patch_hidden":
            mode_colors.append(COLORS["neutral"])
        elif mode == "layer_patch_repr":
            mode_colors.append(COLORS["highlight"])
        elif mode == "pool_weight_transplant":
            mode_colors.append(COLORS["wavlm"])
        else:
            mode_colors.append(COLORS["info_blue"])

    # Left panel: EER
    eer_bars = axes[0].bar(
        x,
        eer_values,
        width,
        color=mode_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    axes[0].set_ylabel("EER (%)")
    axes[0].set_title("Spoofing Detection Performance", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, max(eer_values) + 0.8)

    for bar, value in zip(eer_bars, eer_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Right panel: Probe accuracy (domain leakage)
    probe_colors = [
        COLORS["highlight"] if mode == "layer_patch_repr" else COLORS["wavlm"]
        for mode in modes
    ]
    probe_bars = axes[1].bar(
        x,
        probe_values,
        width,
        color=probe_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    axes[1].set_ylabel("Codec Probe Accuracy (%)")
    axes[1].set_title("Domain Information Leakage", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(
        y=chance_pct,
        color=COLORS["chance"],
        linestyle="--",
        linewidth=1.5,
        label=chance_label,
    )
    axes[1].legend(loc="upper right", framealpha=0.9)

    for bar, value in zip(probe_bars, probe_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Remove over-assertive callouts; keep figure descriptive and uncluttered.

    plt.suptitle("RQ4: Activation Patching Ablation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved intervention comparison: {output_path}")


def plot_cka_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing CKA across different intervention modes."""
    plt.rcParams.update(STYLE_CONFIG)

    # Get unique modes and their CKA summaries
    modes = df['mode'].unique()
    
    # For each mode, get the mean/summary CKA
    summary_data = []
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        if 'layer_key' in mode_df.columns:
            # Has layer-wise data
            for _, row in mode_df.iterrows():
                summary_data.append({
                    'mode': mode,
                    'layer': str(row['layer_key']),
                    'cka': row['cka']
                })
        else:
            summary_data.append({
                'mode': mode,
                'layer': 'all',
                'cka': mode_df['cka'].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Pivot for heatmap (only for pool_weight_transplant which has per-layer data)
    layer_data = summary_df[summary_df['mode'] == 'pool_weight_transplant'].copy()
    
    if layer_data.empty:
        logger.warning("No layer-wise data for heatmap")
        return
    
    layer_data['layer_num'] = pd.to_numeric(layer_data['layer'], errors='coerce')
    layer_data = layer_data.dropna(subset=['layer_num'])
    layer_data = layer_data.sort_values('layer_num')
    
    # Dynamically determine number of layers from data
    n_layers = len(layer_data)
    layer_nums = layer_data['layer_num'].astype(int).values
    
    # Create simple heatmap (1 row x n_layers)
    fig, ax = plt.subplots(figsize=(max(9.5, n_layers * 0.9), 2.8))
    
    cka_matrix = layer_data['cka'].values.reshape(1, -1)
    
    im = ax.imshow(cka_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    # Labels - use actual layer numbers from data
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in layer_nums], fontsize=11)
    ax.set_yticks([0])
    ax.set_yticklabels(["ERM↔DANN"], fontsize=11)
    ax.set_xlabel("Transformer Layer")
    ax.set_title("CKA Similarity: Layer Contributions (ERM vs DANN)", fontweight="bold", pad=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.34, shrink=0.82)
    cbar.set_label("CKA Similarity")
    
    # Annotate low CKA values
    for i, cka in enumerate(layer_data['cka'].values):
        color = "white" if cka <= 0.35 else "black"
        ax.text(i, 0, f"{cka:.2f}", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CKA heatmap: {output_path}")


def plot_delta_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot: Δ EER vs Δ Probe accuracy for each intervention."""
    plt.rcParams.update(STYLE_CONFIG)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # Filter out baseline (delta = 0)
    plot_df = sort_interventions(df[df["mode"] != "layer_patch_hidden"].copy())

    delta_eer = plot_df["delta_eer_vs_base"].values * 100
    delta_probe = plot_df["delta_probe_vs_base"].values * 100
    modes = plot_df["mode"].values

    mode_labels = {
        "layer_patch_repr": "Projection patch",
        "layer_patch_mixed": "Mixed patch",
        "pool_weight_transplant": "Pool-weight transplant",
    }
    mode_colors = {
        "layer_patch_repr": COLORS["highlight"],
        "layer_patch_mixed": COLORS["info_blue"],
        "pool_weight_transplant": COLORS["wavlm"],
    }
    legend_handles = []
    for de, dp, mode in zip(delta_eer, delta_probe, modes):
        ax.scatter(
            de,
            dp,
            s=520,
            c=mode_colors.get(mode, COLORS["neutral"]),
            edgecolor="black",
            linewidth=2.0,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=mode_colors.get(mode, COLORS["neutral"]),
                markeredgecolor="black",
                markersize=9,
                label=mode_labels.get(mode, mode),
            )
        )

    # Add quadrant lines
    ax.axhline(y=0, color=COLORS["chance"], linestyle="-", alpha=0.5, linewidth=1.2)
    ax.axvline(x=0, color=COLORS["chance"], linestyle="-", alpha=0.5, linewidth=1.2)

    # Quadrant labels
    ax.text(
        0.02,
        0.90,
        "Better EER\nMore leakage",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=COLORS["chance"],
        style="italic",
    )
    ax.text(
        0.98,
        0.90,
        "Worse EER\nMore leakage",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["chance"],
        style="italic",
    )
    ax.text(
        0.02,
        0.02,
        "Better EER\nLess leakage",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color=COLORS["highlight"],
        style="italic",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.02,
        "Worse EER\nLess leakage",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=COLORS["chance"],
        style="italic",
    )

    x_padding = 0.02
    y_padding = 0.60
    ax.set_xlim(delta_eer.min() - x_padding, delta_eer.max() + x_padding)
    ax.set_ylim(delta_probe.min() - y_padding, delta_probe.max() + y_padding)
    ax.set_xlabel("Δ EER (%) vs Baseline")
    ax.set_ylabel("Δ Probe Accuracy (%) vs Baseline")
    ax.set_title("Intervention Trade-offs: Detection vs Domain Invariance", fontweight="bold", pad=12)
    ax.margins(x=0.08, y=0.08)
    # Deduplicate legend entries while preserving order.
    dedup_handles = []
    seen_labels = set()
    for handle in legend_handles:
        if handle.get_label() not in seen_labels:
            dedup_handles.append(handle)
            seen_labels.add(handle.get_label())
    ax.legend(handles=dedup_handles, loc="upper center", ncol=3, framealpha=0.9)

    plt.tight_layout(pad=1.25)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved delta scatter: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--cka-results',
        type=Path,
        default=Path('results/rq4_cka_results.csv'),
        help='Path to rq4_cka_results.csv (default: results/rq4_cka_results.csv)',
    )
    parser.add_argument(
        '--intervention-results',
        type=Path,
        default=Path('results/rq4_results_summary.csv'),
        help='Path to rq4_results_summary.csv (default: results/rq4_results_summary.csv)',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('figures/rq4'), help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading CKA results from {args.cka_results}")
    cka_df = pd.read_csv(args.cka_results)
    
    logger.info(f"Loading intervention results from {args.intervention_results}")
    intervention_df = pd.read_csv(args.intervention_results)
    
    # Generate figures
    plot_cka_layer_bar(cka_df, args.output_dir / 'cka_layer_divergence.png')
    plot_cka_heatmap(cka_df, args.output_dir / 'cka_heatmap.png')
    plot_intervention_comparison(intervention_df, args.output_dir / 'intervention_comparison.png')
    plot_delta_scatter(intervention_df, args.output_dir / 'intervention_tradeoff.png')
    
    logger.info(f"All RQ4 figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
