#!/usr/bin/env python3
"""Generate F2: OOD Gap visualization for thesis.

This script visualizes the in-domain (dev) vs out-of-domain (eval) 
generalization gap, showing how DANN reduces the gap compared to ERM.

Usage:
    python scripts/plot_ood_gap.py \\
        --input results/main_results.json \\
        --output figures/ood_gap.png \\
        --verbose

    # With demo data for testing
    python scripts/plot_ood_gap.py --demo --output figures/ood_gap.png
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_RUN_DIRS = {
    "wavlm_erm": "wavlm_erm",
    "wavlm_dann": "wavlm_dann",
    "w2v2_erm": "w2v2_erm",
    "w2v2_dann": "w2v2_dann",
    "lfcc_gmm": "lfcc_gmm_32",
    "trillsson_logistic": "trillsson_logistic",
    "trillsson_mlp": "trillsson_mlp",
}


# ---------------------------------------------------------------------------
# Style Configuration (matching plot_rq3_combined.py)
# ---------------------------------------------------------------------------
COLORS = {
    "wavlm": "#4C72B0",      # Steel blue
    "w2v2": "#DD8452",       # Coral/orange
    "erm": "#E57373",        # Light red/coral
    "dann": "#64B5F6",       # Light blue
    "dev": "#90CAF9",        # Light blue (in-domain)
    "eval": "#EF9A9A",       # Light red (out-of-domain)
    "gap_arrow": "#333333",  # Dark gray
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
# Data Loading
# ---------------------------------------------------------------------------
def load_main_results(path: Path) -> Dict[str, Any]:
    """Load main results data.
    
    Expected format:
    {
        "wavlm_erm": {"dev_eer": 0.05, "eval_eer": 0.08, ...},
        "wavlm_dann": {"dev_eer": 0.04, "eval_eer": 0.06, ...},
        "w2v2_erm": {...},
        "w2v2_dann": {...}
    }
    """
    logger.info(f"Loading main results from: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    return data


def safe_get(data: Optional[Dict[str, Any]], *keys: str, default=None):
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def get_best_dev_eer(logs_path: Path) -> Optional[float]:
    """Extract best validation EER from logs.jsonl."""
    if not logs_path.exists():
        return None

    latest_best_event_eer: Optional[float] = None
    latest_message_best_eer: Optional[float] = None
    latest_epoch_val_eer: Optional[float] = None

    with logs_path.open("r", encoding="utf-8") as logs_file:
        for raw_line in logs_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("event_type") == "epoch_complete":
                event_data = entry.get("data", {})
                if event_data.get("is_best"):
                    candidate_eer = safe_get(event_data, "val", "eer", default=None)
                    try:
                        if candidate_eer is not None:
                            latest_best_event_eer = float(candidate_eer)
                    except (TypeError, ValueError):
                        pass

            message = str(entry.get("message", ""))
            new_best_match = re.search(r"New best eer:\s*([0-9]*\.?[0-9]+)", message)
            if new_best_match:
                try:
                    latest_message_best_eer = float(new_best_match.group(1))
                except ValueError:
                    pass

            epoch_val_match = re.search(r"Epoch\s+\d+\s+val:.*eer=([0-9]*\.?[0-9]+)", message)
            if epoch_val_match:
                try:
                    latest_epoch_val_eer = float(epoch_val_match.group(1))
                except ValueError:
                    pass

    if latest_best_event_eer is not None:
        return latest_best_event_eer
    if latest_message_best_eer is not None:
        return latest_message_best_eer
    return latest_epoch_val_eer


def extract_dev_eer_from_metrics_payload(payload: Dict[str, Any]) -> Optional[float]:
    for key in ["val_eer", "dev_eer", "eer"]:
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    final_val_eer = safe_get(payload, "final_val", "eer", default=None)
    if final_val_eer is not None:
        try:
            return float(final_val_eer)
        except (TypeError, ValueError):
            pass

    best_eer = payload.get("best_eer")
    if best_eer is not None:
        try:
            return float(best_eer)
        except (TypeError, ValueError):
            pass

    return None


def load_main_results_from_runs(results_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Load eval metrics and dev EER from runs directory structure."""
    results: Dict[str, Dict[str, Optional[float]]] = {}

    for model_key, run_dir_name in MODEL_RUN_DIRS.items():
        model_dir = results_dir / run_dir_name
        eval_dir_name = resolve_eval_results_dir(model_dir)
        if eval_dir_name is None:
            continue
        eval_metrics_path = model_dir / eval_dir_name / "metrics.json"
        if not eval_metrics_path.exists():
            continue

        eval_payload = load_main_results(eval_metrics_path)
        eval_eer = eval_payload.get("eer")
        eval_mindcf = eval_payload.get("min_dcf")
        model_result: Dict[str, Optional[float]] = {
            "eval_eer": float(eval_eer) if isinstance(eval_eer, (int, float)) else None,
            "eval_mindcf": float(eval_mindcf) if isinstance(eval_mindcf, (int, float)) else None,
            "dev_eer": None,
        }

        dev_eer = get_best_dev_eer(model_dir / "logs.jsonl")
        if dev_eer is None:
            for fallback_path in [
                model_dir / "eval_dev" / "metrics.json",
                model_dir / "metrics.json",
                model_dir / "metrics_train.json",
            ]:
                if not fallback_path.exists():
                    continue
                fallback_payload = load_main_results(fallback_path)
                dev_eer = extract_dev_eer_from_metrics_payload(fallback_payload)
                if dev_eer is not None:
                    break

        model_result["dev_eer"] = dev_eer
        results[model_key] = model_result

    return results


def resolve_eval_results_dir(model_dir: Path) -> Optional[str]:
    """Pick best available eval directory (prefer full eval)."""
    candidate_dirs = ["eval_eval_full", "eval_eval", "eval_eval_epoch5"]
    for candidate_dir in candidate_dirs:
        if (model_dir / candidate_dir / "metrics.json").exists():
            return candidate_dir
    return None


def generate_demo_data() -> Dict[str, Any]:
    """Generate demo data for testing."""
    logger.info("Generating demo data")
    
    return {
        "wavlm_erm": {
            "dev_eer": 0.035,
            "eval_eer": 0.082,
            "eval_mindcf": 0.245,
        },
        "wavlm_dann": {
            "dev_eer": 0.038,
            "eval_eer": 0.065,
            "eval_mindcf": 0.198,
        },
        "w2v2_erm": {
            "dev_eer": 0.045,
            "eval_eer": 0.095,
            "eval_mindcf": 0.285,
        },
        "w2v2_dann": {
            "dev_eer": 0.048,
            "eval_eer": 0.078,
            "eval_mindcf": 0.235,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ood_gap_bars(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (12, 6.5),
) -> plt.Figure:
    """Create paired bar chart showing dev vs eval EER with gap annotations.
    
    Layout: Groups by backbone, then ERM vs DANN within each backbone.
    Each group has 2 bars (dev and eval) with connecting arrows showing the gap.
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define groups
    groups = [
        ("WavLM ERM", "wavlm_erm"),
        ("WavLM DANN", "wavlm_dann"),
        ("W2V2 ERM", "w2v2_erm"),
        ("W2V2 DANN", "w2v2_dann"),
        ("LFCC-GMM", "lfcc_gmm"),
        ("TRILLsson Logistic", "trillsson_logistic"),
        ("TRILLsson MLP", "trillsson_mlp"),
    ]
    groups = [group for group in groups if group[1] in data]
    
    n_groups = len(groups)
    x = np.arange(n_groups)
    width = 0.35
    
    # Extract data
    dev_eers = []
    eval_eers = []
    gaps = []
    
    for _, key in groups:
        model_data = data.get(key, {})
        dev_raw = model_data.get("dev_eer", 0)
        eval_raw = model_data.get("eval_eer", 0)
        dev_eer = (dev_raw if isinstance(dev_raw, (int, float)) else 0.0) * 100
        eval_eer = (eval_raw if isinstance(eval_raw, (int, float)) else 0.0) * 100
        dev_eers.append(dev_eer)
        eval_eers.append(eval_eer)
        gaps.append(eval_eer - dev_eer)
    
    # Plot bars
    bars_dev = ax.bar(
        x - width/2, dev_eers, width,
        label='Dev (In-Domain)',
        color=COLORS["dev"],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
    )
    
    bars_eval = ax.bar(
        x + width/2, eval_eers, width,
        label='Eval (OOD)',
        color=COLORS["eval"],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
    )
    
    # Add gap arrows and annotations
    for i, (dev_bar, eval_bar, gap) in enumerate(zip(bars_dev, bars_eval, gaps)):
        dev_height = dev_bar.get_height()
        eval_height = eval_bar.get_height()
        
        # Arrow from dev to eval
        arrow_x = x[i]
        ax.annotate(
            '',
            xy=(arrow_x + width/2, eval_height),
            xytext=(arrow_x - width/2, dev_height),
            arrowprops=dict(
                arrowstyle='->',
                color=COLORS["gap_arrow"],
                lw=1.5,
                connectionstyle='arc3,rad=0.2',
            ),
        )
        
        # Gap label
        gap_y = max(dev_height, eval_height) + 0.5
        ax.annotate(
            f'+{gap:.1f}%',
            xy=(arrow_x, gap_y),
            ha='center', va='bottom',
            fontsize=10,
            fontweight='bold',
            color=COLORS["gap_arrow"],
        )
    
    # Calculate gap reductions
    max_y = max(eval_eers) + 3 if eval_eers else 1

    key_to_gap = {key: gap for (_, key), gap in zip(groups, gaps)}
    if "wavlm_erm" in key_to_gap and "wavlm_dann" in key_to_gap and key_to_gap["wavlm_erm"] > 0:
        wavlm_reduction = ((key_to_gap["wavlm_erm"] - key_to_gap["wavlm_dann"]) / key_to_gap["wavlm_erm"] * 100)
        ax.annotate(
            f'Gap ↓{wavlm_reduction:.1f}%',
            xy=(0.5, max_y + 1),
            ha='center', va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8),
        )

    if "w2v2_erm" in key_to_gap and "w2v2_dann" in key_to_gap and key_to_gap["w2v2_erm"] > 0:
        w2v2_reduction = ((key_to_gap["w2v2_erm"] - key_to_gap["w2v2_dann"]) / key_to_gap["w2v2_erm"] * 100)
        x_anchor = 2.5 if len(groups) >= 4 else max(0.5, len(groups) - 1)
        ax.annotate(
            f'Gap ↓{w2v2_reduction:.1f}%',
            xy=(x_anchor, max_y + 1),
            ha='center', va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8),
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('EER (%)')
    ax.set_title('OOD Generalization Gap: Dev (In-Domain) vs Eval (Out-of-Domain)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], rotation=20, ha='right')
    
    # Y-axis limits
    ax.set_ylim(0, max(eval_eers) + 6)
    
    if len(groups) >= 4:
        ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    return fig


def plot_ood_gap_slope(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (12, 6.5),
) -> plt.Figure:
    """Create slope chart showing dev→eval transition with gap annotations.
    
    Alternative visualization showing the "slope" from dev to eval for each model.
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define models and their properties
    models = {
        "wavlm_erm": {"label": "WavLM ERM", "color": "#E57373", "marker": "o", "linestyle": "-"},
        "wavlm_dann": {"label": "WavLM DANN", "color": "#64B5F6", "marker": "s", "linestyle": "-"},
        "w2v2_erm": {"label": "W2V2 ERM", "color": "#FFB74D", "marker": "^", "linestyle": "--"},
        "w2v2_dann": {"label": "W2V2 DANN", "color": "#81C784", "marker": "D", "linestyle": "--"},
        "lfcc_gmm": {"label": "LFCC-GMM", "color": "#A1887F", "marker": "P", "linestyle": "-."},
        "trillsson_logistic": {"label": "TRILLsson Logistic", "color": "#4DB6AC", "marker": "X", "linestyle": "-."},
        "trillsson_mlp": {"label": "TRILLsson MLP", "color": "#9575CD", "marker": "*", "linestyle": "-."},
    }
    
    x_positions = [0, 1]  # Dev (0) and Eval (1)
    
    annotation_offsets = {
        "wavlm_erm": 0.00,
        "wavlm_dann": -0.30,
        "w2v2_erm": 0.35,
        "w2v2_dann": -0.45,
        "lfcc_gmm": 0.20,
        "trillsson_logistic": -0.20,
        "trillsson_mlp": 0.10,
    }

    for model_key, props in models.items():
        if model_key not in data:
            continue
        model_data = data.get(model_key, {})
        dev_raw = model_data.get("dev_eer", 0)
        eval_raw = model_data.get("eval_eer", 0)
        dev_eer = (dev_raw if isinstance(dev_raw, (int, float)) else 0.0) * 100
        eval_eer = (eval_raw if isinstance(eval_raw, (int, float)) else 0.0) * 100
        
        ax.plot(
            x_positions, [dev_eer, eval_eer],
            marker=props["marker"],
            markersize=9,
            linewidth=2.2,
            color=props["color"],
            linestyle=props["linestyle"],
            label=props["label"],
        )
        
        # Add gap annotation at the eval point
        gap = eval_eer - dev_eer
        ax.annotate(
            f'+{gap:.1f}%',
            xy=(1.05, eval_eer + annotation_offsets.get(model_key, 0.0)),
            ha='left', va='center',
            fontsize=9,
            color=props["color"],
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('EER (%)')
    ax.set_title('OOD Generalization: Dev → Eval Transition')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Dev\n(In-Domain)', 'Eval\n(Out-of-Domain)'], fontsize=11)
    ax.set_xlim(-0.25, 1.45)
    
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate OOD gap visualization for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Path to runs directory for auto-loading metrics",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to main results JSON (override runs loading)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/ood_gap.png"),
        help="Output file path (PNG and PDF will be saved)",
    )
    p.add_argument(
        "--style",
        choices=["bars", "slope", "both"],
        default="bars",
        help="Visualization style: bars (grouped), slope (line), or both",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data instead of loading from file",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6.5],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 12 6.5)",
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
        data = generate_demo_data()
    elif args.input is not None:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            logger.info("Use --demo or omit --input to load from --results-dir")
            return 1
        data = load_main_results(args.input)
    else:
        data = load_main_results_from_runs(args.results_dir)
        if not data:
            logger.error(f"No run metrics found in: {args.results_dir}")
            return 1
    
    # Output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.style in ["bars", "both"]:
        fig = plot_ood_gap_bars(data, figsize=tuple(args.figsize))
        
        # Save PNG
        png_path = args.output.with_suffix('.png')
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG: {png_path}")
        
        # Save PDF
        pdf_path = args.output.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
    
    if args.style in ["slope", "both"]:
        fig = plot_ood_gap_slope(data, figsize=tuple(args.figsize))
        
        suffix = "_slope" if args.style == "both" else ""
        
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
    logger.info("OOD Gap Figure Summary")
    logger.info("=" * 60)
    
    for model_key in [
        "wavlm_erm",
        "wavlm_dann",
        "w2v2_erm",
        "w2v2_dann",
        "lfcc_gmm",
        "trillsson_logistic",
        "trillsson_mlp",
    ]:
        if model_key not in data:
            continue
        model_data = data.get(model_key, {})
        dev_raw = model_data.get("dev_eer", 0)
        eval_raw = model_data.get("eval_eer", 0)
        dev_eer = float(dev_raw) if isinstance(dev_raw, (int, float)) else 0.0
        eval_eer = float(eval_raw) if isinstance(eval_raw, (int, float)) else 0.0
        gap = (eval_eer - dev_eer) * 100
        logger.info(f"{model_key}: Dev={dev_eer*100:.2f}%, Eval={eval_eer*100:.2f}%, Gap={gap:.2f}%")
    
    # Calculate reductions
    wavlm_erm_gap = (data.get("wavlm_erm", {}).get("eval_eer", 0) - 
                    data.get("wavlm_erm", {}).get("dev_eer", 0))
    wavlm_dann_gap = (data.get("wavlm_dann", {}).get("eval_eer", 0) - 
                     data.get("wavlm_dann", {}).get("dev_eer", 0))
    
    if wavlm_erm_gap > 0:
        wavlm_reduction = (wavlm_erm_gap - wavlm_dann_gap) / wavlm_erm_gap * 100
        logger.info(f"\nWavLM Gap Reduction: {wavlm_reduction:.1f}%")
    
    logger.info(f"\nOutput: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
