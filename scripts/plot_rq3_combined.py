#!/usr/bin/env python3
"""Generate combined RQ3 figure for thesis: backbone vs projection layer domain invariance.

This script creates a publication-ready figure addressing RQ3:
"Does DANN reduce domain information in learned representations?"

The figure has two panels:
1. LEFT: Backbone layer-wise codec probe accuracy (layers 0-11)
   - Shows that ERM and DANN produce identical backbone features (frozen backbone)
   - Displays both WavLM and W2V2 when available
   
2. RIGHT: Projection layer codec probe accuracy (ERM vs DANN)
   - Shows DANN reduces codec information in the projection output
   - Quantifies the relative reduction (~10.7% for WavLM)

Key insight: The backbone is frozen, so ERM and DANN produce identical
representations at layers 0-11. The difference emerges at the trainable
projection head, where DANN's gradient reversal reduces domain information.

Usage:
    python scripts/plot_rq3_combined.py \\
        --backbone-probes probe_results/comparison_results.json \\
        --projection-probes results/rq3_projection.json \\
        --output figures/rq3_combined.png

    # With both WavLM and W2V2 projection probes
    python scripts/plot_rq3_combined.py \\
        --backbone-probes probe_results/comparison_results.json \\
        --projection-probes results/rq3_projection.json \\
        --projection-probes-w2v2 results/rq3_projection_w2v2.json \\
        --output figures/rq3_combined.png

    # With custom backbone selection
    python scripts/plot_rq3_combined.py \\
        --backbone-probes probe_results/comparison_results.json \\
        --projection-probes results/rq3_projection.json \\
        --backbone wavlm \\
        --output figures/rq3_wavlm_combined.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Number of codec classes (for chance level)
N_CODEC_CLASSES = 12
CHANCE_LEVEL = 1.0 / N_CODEC_CLASSES  # ~0.083

# Color scheme matching existing repo aesthetics
COLORS = {
    "wavlm": "#4C72B0",      # Steel blue
    "w2v2": "#DD8452",       # Coral/orange
    "erm": "#E57373",        # Light red/coral (from existing plot)
    "dann": "#64B5F6",       # Light blue (from existing plot)
    "chance": "#9E9E9E",     # Gray
}

# Plot style settings
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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_backbone_probes(path: Path) -> Dict[str, Any]:
    """Load backbone layer-wise probe results.
    
    Expected format (comparison_results.json):
    {
        "wavlm": {
            "description": "...",
            "per_layer": {
                "0": {"accuracy": 0.936, "accuracy_std": 0.006, ...},
                ...
            }
        },
        "w2v2": {...}
    }
    """
    logger.info(f"Loading backbone probes from: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    # Handle different possible formats
    if "wavlm" in data or "w2v2" in data:
        return data
    elif "per_layer" in data:
        # Single backbone format
        return {"wavlm": data}
    else:
        raise ValueError(f"Unrecognized backbone probe format in {path}")


def load_projection_probes(path: Path) -> Dict[str, Any]:
    """Load projection layer probe results.
    
    Expected format (rq3_projection.json):
    {
        "results": {
            "erm": {"codec": {"accuracy": 0.434, "accuracy_std": 0.009, ...}},
            "dann": {"codec": {"accuracy": 0.388, "accuracy_std": 0.009, ...}}
        },
        "comparison": {
            "codec": {"reduction": 0.046, "relative_reduction": 0.107, ...}
        }
    }
    """
    logger.info(f"Loading projection probes from: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    return data


# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------
def extract_backbone_data(
    data: Dict[str, Any],
    backbone: str = "wavlm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract layer-wise accuracy and std from backbone probe data.
    
    Returns:
        layers: Array of layer indices (0-11)
        accuracies: Array of probe accuracies
        stds: Array of standard deviations
    """
    if backbone not in data:
        available = list(data.keys())
        raise ValueError(f"Backbone '{backbone}' not found. Available: {available}")
    
    backbone_data = data[backbone]
    per_layer = backbone_data.get("per_layer", backbone_data)
    
    layers = []
    accuracies = []
    stds = []
    
    for layer_idx in range(12):
        layer_key = str(layer_idx)
        if layer_key in per_layer:
            layer_data = per_layer[layer_key]
            layers.append(layer_idx)
            accuracies.append(layer_data["accuracy"])
            stds.append(layer_data.get("accuracy_std", 0.0))
    
    return np.array(layers), np.array(accuracies), np.array(stds)


def extract_projection_data(
    data: Dict[str, Any],
    probe_target: str = "codec",
) -> Dict[str, Dict[str, float]]:
    """Extract ERM vs DANN projection layer probe data.
    
    Returns:
        Dictionary with 'erm' and 'dann' keys, each containing
        'accuracy', 'accuracy_std', and comparison metrics.
    """
    results = data.get("results", data)
    comparison = data.get("comparison", {})
    
    extracted = {}
    
    for model_type in ["erm", "dann"]:
        if model_type in results:
            model_data = results[model_type]
            if probe_target in model_data:
                probe_data = model_data[probe_target]
                extracted[model_type] = {
                    "accuracy": probe_data.get("accuracy", 0.0),
                    "accuracy_std": probe_data.get("accuracy_std", 0.0),
                }
            else:
                # Fallback: maybe accuracy is at model level
                extracted[model_type] = {
                    "accuracy": model_data.get("accuracy", 0.0),
                    "accuracy_std": model_data.get("accuracy_std", 0.0),
                }
    
    # Add comparison metrics
    if probe_target in comparison:
        extracted["comparison"] = comparison[probe_target]
    elif "reduction" in comparison:
        extracted["comparison"] = comparison
    
    return extracted


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
def plot_backbone_panel(
    ax: plt.Axes,
    backbone_data: Dict[str, Any],
    backbones: list[str],
    show_legend: bool = True,
) -> None:
    """Plot backbone layer-wise codec probe accuracy (left panel)."""
    
    for backbone in backbones:
        if backbone not in backbone_data:
            continue
            
        layers, accuracies, stds = extract_backbone_data(backbone_data, backbone)
        color = COLORS.get(backbone, COLORS["wavlm"])
        
        # Line plot with error band
        ax.plot(
            layers, accuracies,
            marker='o', markersize=6,
            linewidth=2, color=color,
            label=backbone.upper() if backbone == "w2v2" else "WavLM",
        )
        
        # Error band (shaded region)
        ax.fill_between(
            layers,
            accuracies - stds,
            accuracies + stds,
            alpha=0.2, color=color,
        )
    
    # Chance level line
    ax.axhline(
        y=CHANCE_LEVEL, color=COLORS["chance"],
        linestyle='--', linewidth=1.5,
        label=f'Chance (1/{N_CODEC_CLASSES})',
    )
    
    # Formatting
    ax.set_xlabel('Backbone Layer')
    ax.set_ylabel('Codec Probe Accuracy')
    ax.set_title('(a) Backbone Layer Probes\n(Frozen — identical for ERM & DANN)')
    
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'{i}' for i in range(12)])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 1.0)
    
    if show_legend:
        ax.legend(loc='lower left', framealpha=0.9)


def plot_projection_panel(
    ax: plt.Axes,
    projection_data: Dict[str, Dict[str, float]],
    backbone_name: str = "WavLM",
    projection_data_w2v2: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Plot projection layer ERM vs DANN comparison (right panel).
    
    When projection_data_w2v2 is provided, shows 4 bars grouped by backbone:
    [WavLM ERM, WavLM DANN] [W2V2 ERM, W2V2 DANN]
    """
    
    if projection_data_w2v2 is not None:
        # Four-bar layout: grouped by backbone
        _plot_projection_panel_dual(ax, projection_data, projection_data_w2v2)
    else:
        # Original two-bar layout
        _plot_projection_panel_single(ax, projection_data, backbone_name)


def _plot_projection_panel_single(
    ax: plt.Axes,
    projection_data: Dict[str, Dict[str, float]],
    backbone_name: str = "WavLM",
) -> None:
    """Plot single-backbone projection panel (2 bars: ERM vs DANN)."""
    
    models = ['erm', 'dann']
    x = np.arange(len(models))
    width = 0.6
    
    accuracies = []
    stds = []
    colors_list = []
    
    for model in models:
        if model in projection_data:
            accuracies.append(projection_data[model]["accuracy"])
            stds.append(projection_data[model]["accuracy_std"])
            colors_list.append(COLORS[model])
        else:
            accuracies.append(0)
            stds.append(0)
            colors_list.append(COLORS["chance"])
    
    # Bar plot
    bars = ax.bar(
        x, accuracies,
        width, yerr=stds,
        color=colors_list, alpha=0.85,
        capsize=5, error_kw={'linewidth': 1.5},
        edgecolor='black', linewidth=0.5,
    )
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.annotate(
            f'{acc:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
        )
    
    # Chance level line
    ax.axhline(
        y=CHANCE_LEVEL, color=COLORS["chance"],
        linestyle='--', linewidth=1.5,
        label=f'Chance ({CHANCE_LEVEL:.3f})',
    )
    
    # Add reduction annotation
    if "comparison" in projection_data:
        comparison = projection_data["comparison"]
        reduction = comparison.get("reduction", 0)
        rel_reduction = comparison.get("relative_reduction", 0)
        
        # Position annotation between bars, below
        ax.annotate(
            f'Reduction: {reduction:.3f} ({rel_reduction*100:.1f}% relative)',
            xy=(0.5, 0.05),
            xycoords='axes fraction',
            ha='center', va='bottom',
            fontsize=10, style='italic',
            color='#333333',
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Codec Probe Accuracy')
    ax.set_title(f'(b) Projection Layer Probes ({backbone_name})\n(Trainable — DANN reduces codec info)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['ERM', 'DANN'], fontsize=12)
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='upper right', framealpha=0.9)


def _plot_projection_panel_dual(
    ax: plt.Axes,
    projection_data_wavlm: Dict[str, Dict[str, float]],
    projection_data_w2v2: Dict[str, Dict[str, float]],
) -> None:
    """Plot dual-backbone projection panel (4 bars: WavLM ERM/DANN, W2V2 ERM/DANN)."""
    
    # Bar positions: grouped by backbone with gap between groups
    # Group 1 (WavLM): positions 0, 1
    # Group 2 (W2V2): positions 2.5, 3.5
    width = 0.7
    x_wavlm = np.array([0, 1])
    x_w2v2 = np.array([2.5, 3.5])
    
    # Extract data for WavLM
    wavlm_accs = []
    wavlm_stds = []
    for model in ['erm', 'dann']:
        if model in projection_data_wavlm:
            wavlm_accs.append(projection_data_wavlm[model]["accuracy"])
            wavlm_stds.append(projection_data_wavlm[model]["accuracy_std"])
        else:
            wavlm_accs.append(0)
            wavlm_stds.append(0)
    
    # Extract data for W2V2
    w2v2_accs = []
    w2v2_stds = []
    for model in ['erm', 'dann']:
        if model in projection_data_w2v2:
            w2v2_accs.append(projection_data_w2v2[model]["accuracy"])
            w2v2_stds.append(projection_data_w2v2[model]["accuracy_std"])
        else:
            w2v2_accs.append(0)
            w2v2_stds.append(0)
    
    # Color scheme: use backbone colors with ERM darker, DANN lighter
    wavlm_colors = [COLORS["wavlm"], COLORS["wavlm"]]
    w2v2_colors = [COLORS["w2v2"], COLORS["w2v2"]]
    
    # Create hatching to distinguish ERM vs DANN
    hatches = ['', '///']  # ERM solid, DANN hatched
    
    # Plot WavLM bars
    for i, (x_pos, acc, std, color, hatch) in enumerate(
        zip(x_wavlm, wavlm_accs, wavlm_stds, wavlm_colors, hatches)
    ):
        bar = ax.bar(
            x_pos, acc, width,
            yerr=std, color=color,
            alpha=0.85 if i == 0 else 0.65,
            hatch=hatch,
            capsize=4, error_kw={'linewidth': 1.5},
            edgecolor='black', linewidth=0.5,
        )
        # Value label
        ax.annotate(
            f'{acc:.3f}',
            xy=(x_pos, acc + std + 0.02),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )
    
    # Plot W2V2 bars
    for i, (x_pos, acc, std, color, hatch) in enumerate(
        zip(x_w2v2, w2v2_accs, w2v2_stds, w2v2_colors, hatches)
    ):
        bar = ax.bar(
            x_pos, acc, width,
            yerr=std, color=color,
            alpha=0.85 if i == 0 else 0.65,
            hatch=hatch,
            capsize=4, error_kw={'linewidth': 1.5},
            edgecolor='black', linewidth=0.5,
        )
        # Value label
        ax.annotate(
            f'{acc:.3f}',
            xy=(x_pos, acc + std + 0.02),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )
    
    # Chance level line
    ax.axhline(
        y=CHANCE_LEVEL, color=COLORS["chance"],
        linestyle='--', linewidth=1.5,
        label=f'Chance ({CHANCE_LEVEL:.3f})',
    )
    
    # Add reduction annotations below x-axis under each backbone group.
    # Use axis transform so x follows data coordinates while y is axis-relative.
    reduction_label_y = -0.17

    # WavLM reduction
    if "comparison" in projection_data_wavlm:
        comp = projection_data_wavlm["comparison"]
        rel_red = comp.get("relative_reduction", 0) * 100
        ax.annotate(
            f'↓{rel_red:.1f}%',
            xy=(0.5, reduction_label_y),
            ha='center', va='top',
            fontsize=10, fontweight='bold',
            color='#333333',
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            annotation_clip=False,
        )
    
    # W2V2 reduction
    if "comparison" in projection_data_w2v2:
        comp = projection_data_w2v2["comparison"]
        rel_red = comp.get("relative_reduction", 0) * 100
        ax.annotate(
            f'↓{rel_red:.1f}%',
            xy=(3.0, reduction_label_y),
            ha='center', va='top',
            fontsize=10, fontweight='bold',
            color='#333333',
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            annotation_clip=False,
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Codec Probe Accuracy')
    ax.set_title('(b) Projection Layer Probes (WavLM & W2V2)\n(Trainable — DANN reduces codec info)')
    
    # X-axis labels
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(['ERM', 'DANN', 'ERM', 'DANN'], fontsize=10)
    
    # Add backbone labels below
    ax.text(0.5, -0.10, 'WavLM', ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold',
            color=COLORS["wavlm"])
    ax.text(3.0, -0.10, 'W2V2', ha='center', va='top',
            transform=ax.get_xaxis_transform(), fontsize=11, fontweight='bold',
            color=COLORS["w2v2"])
    
    ax.set_xlim(-0.6, 4.1)
    ax.set_ylim(0, 1.0)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["wavlm"], edgecolor='black', label='WavLM'),
        Patch(facecolor=COLORS["w2v2"], edgecolor='black', label='W2V2'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='DANN'),
        plt.Line2D([0], [0], color=COLORS["chance"], linestyle='--', label=f'Chance ({CHANCE_LEVEL:.3f})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)


def create_combined_figure(
    backbone_data: Dict[str, Any],
    projection_data: Dict[str, Dict[str, float]],
    backbones: list[str],
    backbone_name: str,
    figsize: tuple[float, float] = (12, 5),
    projection_data_w2v2: Optional[Dict[str, Dict[str, float]]] = None,
) -> plt.Figure:
    """Create the combined RQ3 figure with two panels.
    
    Args:
        backbone_data: Layer-wise probe data for backbone panel
        projection_data: WavLM projection probe data (ERM vs DANN)
        backbones: List of backbones to show in left panel
        backbone_name: Display name for backbone(s)
        figsize: Figure dimensions
        projection_data_w2v2: Optional W2V2 projection probe data for 4-bar display
    """
    
    # Apply style
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Backbone layer-wise probes
    plot_backbone_panel(ax1, backbone_data, backbones)
    
    # Right panel: Projection layer comparison
    plot_projection_panel(ax2, projection_data, backbone_name, projection_data_w2v2)
    
    # Overall title
    fig.suptitle(
        'RQ3: Does DANN Reduce Domain Information in Learned Representations?',
        fontsize=14, fontweight='bold', y=1.02,
    )
    
    # Reserve extra bottom margin for below-axis annotations in the dual projection panel.
    plt.tight_layout(rect=(0, 0.14, 1, 1))
    
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate combined RQ3 figure for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input files
    p.add_argument(
        "--backbone-probes",
        type=Path,
        required=True,
        help="Path to backbone probe results JSON (e.g., probe_results/comparison_results.json)",
    )
    p.add_argument(
        "--projection-probes",
        type=Path,
        required=True,
        help="Path to WavLM projection probe results JSON (e.g., results/rq3_projection.json)",
    )
    p.add_argument(
        "--projection-probes-w2v2",
        type=Path,
        default=None,
        help="Path to W2V2 projection probe results JSON (optional, e.g., results/rq3_projection_w2v2.json)",
    )
    
    # Backbone selection
    p.add_argument(
        "--backbone",
        type=str,
        choices=["wavlm", "w2v2", "both"],
        default="wavlm",
        help="Which backbone to show in plots (default: wavlm)",
    )
    
    # Output options
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/rq3_combined.png"),
        help="Output file path (PNG and PDF will be saved)",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 5],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 12 5)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats (default: 300)",
    )
    
    # Verbosity
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input files exist
    if not args.backbone_probes.exists():
        logger.error(f"Backbone probes file not found: {args.backbone_probes}")
        return 1
    
    if not args.projection_probes.exists():
        logger.error(f"Projection probes file not found: {args.projection_probes}")
        return 1
    
    # Load data
    try:
        backbone_data = load_backbone_probes(args.backbone_probes)
        projection_data_raw = load_projection_probes(args.projection_probes)
        projection_data = extract_projection_data(projection_data_raw)
        
        # Load W2V2 projection data if provided
        projection_data_w2v2 = None
        if args.projection_probes_w2v2:
            if not args.projection_probes_w2v2.exists():
                logger.error(f"W2V2 projection probes file not found: {args.projection_probes_w2v2}")
                return 1
            projection_data_w2v2_raw = load_projection_probes(args.projection_probes_w2v2)
            projection_data_w2v2 = extract_projection_data(projection_data_w2v2_raw)
            logger.info(f"Loaded W2V2 projection probes from: {args.projection_probes_w2v2}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Determine backbones to plot
    if args.backbone == "both":
        backbones = [b for b in ["wavlm", "w2v2"] if b in backbone_data]
        backbone_name = "WavLM & W2V2"
    else:
        backbones = [args.backbone]
        backbone_name = "WavLM" if args.backbone == "wavlm" else "Wav2Vec2"
    
    # Override backbone_name if showing both projection probes
    if projection_data_w2v2 is not None:
        backbone_name = "WavLM & W2V2"
    
    logger.info(f"Plotting backbones: {backbones}")
    if projection_data_w2v2:
        logger.info("Projection panel: showing both WavLM and W2V2 (4 bars)")
    
    # Create figure
    fig = create_combined_figure(
        backbone_data=backbone_data,
        projection_data=projection_data,
        backbones=backbones,
        backbone_name=backbone_name,
        figsize=tuple(args.figsize),
        projection_data_w2v2=projection_data_w2v2,
    )
    
    # Save outputs
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    png_path = args.output.with_suffix('.png')
    fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved PNG: {png_path}")
    
    # Save PDF
    pdf_path = args.output.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved PDF: {pdf_path}")
    
    plt.close(fig)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("RQ3 Combined Figure Summary")
    logger.info("=" * 60)
    
    for backbone in backbones:
        layers, accs, _ = extract_backbone_data(backbone_data, backbone)
        logger.info(f"\n{backbone.upper()} Backbone:")
        logger.info(f"  Layer range: {layers[0]}-{layers[-1]}")
        logger.info(f"  Accuracy range: {accs.min():.3f} - {accs.max():.3f}")
        logger.info(f"  Peak layer: {layers[np.argmax(accs)]} ({accs.max():.3f})")
    
    logger.info(f"\nProjection Layer (WavLM):")
    if "erm" in projection_data:
        logger.info(f"  ERM accuracy:  {projection_data['erm']['accuracy']:.3f}")
    if "dann" in projection_data:
        logger.info(f"  DANN accuracy: {projection_data['dann']['accuracy']:.3f}")
    if "comparison" in projection_data:
        comp = projection_data["comparison"]
        logger.info(f"  Reduction:     {comp.get('reduction', 0):.3f} "
                   f"({comp.get('relative_reduction', 0)*100:.1f}% relative)")
    
    if projection_data_w2v2:
        logger.info(f"\nProjection Layer (W2V2):")
        if "erm" in projection_data_w2v2:
            logger.info(f"  ERM accuracy:  {projection_data_w2v2['erm']['accuracy']:.3f}")
        if "dann" in projection_data_w2v2:
            logger.info(f"  DANN accuracy: {projection_data_w2v2['dann']['accuracy']:.3f}")
        if "comparison" in projection_data_w2v2:
            comp = projection_data_w2v2["comparison"]
            logger.info(f"  Reduction:     {comp.get('reduction', 0):.3f} "
                       f"({comp.get('relative_reduction', 0)*100:.1f}% relative)")
    
    logger.info(f"\nOutputs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
