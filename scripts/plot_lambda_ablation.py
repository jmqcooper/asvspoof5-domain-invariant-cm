#!/usr/bin/env python3
"""Generate F5: Lambda schedule ablation visualization for thesis.

This script visualizes the comparison between different λ (lambda) 
scheduling strategies for DANN training:
- v1: Exponential ramp-up
- v2: Linear ramp-up

Usage:
    python scripts/plot_lambda_ablation.py \\
        --input results/lambda_ablation.json \\
        --output figures/lambda_ablation.png \\
        --verbose

    # With demo data for testing
    python scripts/plot_lambda_ablation.py --demo --output figures/lambda_ablation.png
    
    # Show training dynamics overlay
    python scripts/plot_lambda_ablation.py --demo --show-dynamics --output figures/lambda_ablation.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Style Configuration (matching plot_rq3_combined.py)
# ---------------------------------------------------------------------------
COLORS = {
    "v1_exponential": "#E57373",  # Light red
    "v2_linear": "#64B5F6",       # Light blue
    "v3_cosine": "#81C784",       # Light green (optional)
    "loss": "#9E9E9E",            # Gray
    "eer": "#FF9800",             # Orange
}

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
# Lambda Schedule Functions
# ---------------------------------------------------------------------------
def exponential_schedule(epoch: float, max_epochs: int, gamma: float = 10.0) -> float:
    """Exponential lambda schedule (DANN paper style).
    
    λ(p) = 2 / (1 + exp(-γp)) - 1
    where p = epoch / max_epochs
    """
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def linear_schedule(epoch: float, max_epochs: int, warmup_epochs: int = 0) -> float:
    """Linear lambda schedule.
    
    λ = 0 for epoch < warmup_epochs
    λ = (epoch - warmup) / (max_epochs - warmup) otherwise
    """
    if epoch < warmup_epochs:
        return 0.0
    return (epoch - warmup_epochs) / (max_epochs - warmup_epochs)


def cosine_schedule(epoch: float, max_epochs: int) -> float:
    """Cosine annealing lambda schedule.
    
    λ = 0.5 * (1 - cos(π * epoch / max_epochs))
    """
    return 0.5 * (1.0 - np.cos(np.pi * epoch / max_epochs))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ablation_data(path: Path) -> Dict[str, Any]:
    """Load lambda ablation results.
    
    Expected format:
    {
        "schedules": {
            "exponential": {
                "final_eer": 0.065,
                "training_loss": [...],
                "validation_eer": [...]
            },
            "linear": {...}
        },
        "config": {
            "max_epochs": 10,
            "gamma": 10.0
        }
    }
    """
    logger.info(f"Loading ablation data from: {path}")
    
    with open(path) as f:
        return json.load(f)


def generate_demo_data(max_epochs: int = 10) -> Dict[str, Any]:
    """Generate demo data for testing."""
    logger.info("Generating demo data")
    
    np.random.seed(42)
    epochs = np.arange(max_epochs + 1)
    
    # Simulated training curves
    # Exponential schedule: converges faster but may be less stable
    exp_eer = 0.12 * np.exp(-0.2 * epochs) + 0.065 + np.random.normal(0, 0.003, len(epochs))
    exp_loss = 0.5 * np.exp(-0.15 * epochs) + 0.1 + np.random.normal(0, 0.01, len(epochs))
    
    # Linear schedule: slower but more stable convergence
    lin_eer = 0.10 * np.exp(-0.18 * epochs) + 0.062 + np.random.normal(0, 0.002, len(epochs))
    lin_loss = 0.45 * np.exp(-0.12 * epochs) + 0.12 + np.random.normal(0, 0.008, len(epochs))
    
    return {
        "schedules": {
            "exponential": {
                "final_eer": float(exp_eer[-1]),
                "final_loss": float(exp_loss[-1]),
                "validation_eer": exp_eer.tolist(),
                "training_loss": exp_loss.tolist(),
            },
            "linear": {
                "final_eer": float(lin_eer[-1]),
                "final_loss": float(lin_loss[-1]),
                "validation_eer": lin_eer.tolist(),
                "training_loss": lin_loss.tolist(),
            },
        },
        "config": {
            "max_epochs": max_epochs,
            "gamma": 10.0,
            "warmup_epochs": 1,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_lambda_schedules(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (10, 5),
    show_dynamics: bool = False,
) -> plt.Figure:
    """Plot lambda schedules with optional training dynamics.
    
    Args:
        data: Ablation data with schedules and config
        figsize: Figure dimensions
        show_dynamics: If True, overlay training curves
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    config = data.get("config", {})
    max_epochs = config.get("max_epochs", 10)
    gamma = config.get("gamma", 10.0)
    warmup_epochs = config.get("warmup_epochs", 0)
    
    # Create epoch array for smooth curves
    epochs_smooth = np.linspace(0, max_epochs, 200)
    epochs_discrete = np.arange(max_epochs + 1)
    
    # Compute lambda curves
    lambda_exp = [exponential_schedule(e, max_epochs, gamma) for e in epochs_smooth]
    lambda_lin = [linear_schedule(e, max_epochs, warmup_epochs) for e in epochs_smooth]
    lambda_cos = [cosine_schedule(e, max_epochs) for e in epochs_smooth]
    
    if show_dynamics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.2, figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
    
    # Left panel: Lambda schedules
    ax1.plot(
        epochs_smooth, lambda_exp,
        linewidth=2.5, color=COLORS["v1_exponential"],
        label=f'Exponential (γ={gamma})',
    )
    ax1.plot(
        epochs_smooth, lambda_lin,
        linewidth=2.5, color=COLORS["v2_linear"],
        label='Linear',
    )
    ax1.plot(
        epochs_smooth, lambda_cos,
        linewidth=2.5, color=COLORS["v3_cosine"],
        linestyle='--',
        label='Cosine (reference)',
        alpha=0.7,
    )
    
    # Formatting for left panel
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('λ (Domain Loss Weight)')
    ax1.set_title('(a) Lambda Scheduling Strategies')
    ax1.set_xlim(0, max_epochs)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Training dynamics (if requested)
    if show_dynamics and ax2 is not None:
        schedules = data.get("schedules", {})
        
        for sched_name, props in [
            ("exponential", {"color": COLORS["v1_exponential"], "label": "Exponential"}),
            ("linear", {"color": COLORS["v2_linear"], "label": "Linear"}),
        ]:
            sched_data = schedules.get(sched_name, {})
            val_eer = sched_data.get("validation_eer", [])
            
            if val_eer:
                epochs_data = np.arange(len(val_eer))
                ax2.plot(
                    epochs_data, np.array(val_eer) * 100,  # Convert to percentage
                    marker='o', markersize=4,
                    linewidth=2, color=props["color"],
                    label=f'{props["label"]} (EER={val_eer[-1]*100:.2f}%)',
                )
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation EER (%)')
        ax2.set_title('(b) Training Dynamics')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_lambda_comparison_bars(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot bar chart comparing final results of different schedules."""
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    schedules = data.get("schedules", {})
    
    schedule_names = ["exponential", "linear"]
    labels = ["Exponential", "Linear"]
    colors = [COLORS["v1_exponential"], COLORS["v2_linear"]]
    
    eers = []
    for name in schedule_names:
        sched_data = schedules.get(name, {})
        eer = sched_data.get("final_eer", 0)
        eers.append(eer * 100)  # Convert to percentage
    
    x = np.arange(len(schedule_names))
    bars = ax.bar(x, eers, width=0.5, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels
    for bar, eer in zip(bars, eers):
        height = bar.get_height()
        ax.annotate(
            f'{eer:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
        )
    
    ax.set_xlabel('')
    ax.set_ylabel('Final EER (%)')
    ax.set_title('Lambda Schedule Comparison: Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(eers) * 1.2)
    
    plt.tight_layout()
    
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate lambda schedule ablation visualization for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    p.add_argument(
        "--input",
        type=Path,
        default=Path("results/lambda_ablation.json"),
        help="Path to lambda ablation results JSON",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/lambda_ablation.png"),
        help="Output file path (PNG and PDF will be saved)",
    )
    p.add_argument(
        "--style",
        choices=["schedules", "comparison", "both"],
        default="schedules",
        help="Visualization style",
    )
    p.add_argument(
        "--show-dynamics",
        action="store_true",
        help="Show training dynamics alongside schedules",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data instead of loading from file",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum epochs for demo data (default: 10)",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 5],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 10 5)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats (default: 300)",
    )
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
    
    # Load or generate data
    if args.demo:
        data = generate_demo_data(args.max_epochs)
    elif args.input.exists():
        data = load_ablation_data(args.input)
    else:
        logger.warning(f"Input file not found: {args.input}")
        logger.info("Generating demo data instead")
        data = generate_demo_data(args.max_epochs)
    
    # Output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.style in ["schedules", "both"]:
        fig = plot_lambda_schedules(
            data,
            figsize=tuple(args.figsize),
            show_dynamics=args.show_dynamics,
        )
        
        # Save PNG
        png_path = args.output.with_suffix('.png')
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG: {png_path}")
        
        # Save PDF
        pdf_path = args.output.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
    
    if args.style in ["comparison", "both"]:
        fig = plot_lambda_comparison_bars(data, figsize=tuple(args.figsize))
        
        suffix = "_comparison" if args.style == "both" else ""
        
        # Save PNG
        png_path = args.output.with_stem(args.output.stem + suffix).with_suffix('.png')
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG: {png_path}")
        
        # Save PDF
        pdf_path = args.output.with_stem(args.output.stem + suffix).with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Lambda Ablation Figure Summary")
    logger.info("=" * 60)
    
    config = data.get("config", {})
    logger.info(f"Max epochs: {config.get('max_epochs', 10)}")
    logger.info(f"Gamma (exponential): {config.get('gamma', 10.0)}")
    
    schedules = data.get("schedules", {})
    for name, sched_data in schedules.items():
        final_eer = sched_data.get("final_eer", 0)
        logger.info(f"{name.capitalize()} schedule: Final EER = {final_eer*100:.2f}%")
    
    logger.info(f"\nOutput: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
