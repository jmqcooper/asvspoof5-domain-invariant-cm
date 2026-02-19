#!/usr/bin/env python3
"""Generate F1: Per-codec EER bar chart for thesis.

This script creates a grouped bar chart showing per-codec EER
for all 4 models (WavLM ERM, WavLM DANN, W2V2 ERM, W2V2 DANN).

Usage:
    python scripts/plot_per_codec.py \\
        --input results/per_codec_eer.json \\
        --output figures/per_codec_eer.png \\
        --verbose

    # With demo data for testing
    python scripts/plot_per_codec.py --demo --output figures/per_codec_eer.png
"""

from __future__ import annotations

import argparse
import csv
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
    "wavlm_erm": "#E57373",   # Light red/coral
    "wavlm_dann": "#64B5F6",  # Light blue
    "w2v2_erm": "#FFB74D",    # Light orange
    "w2v2_dann": "#81C784",   # Light green
    "lfcc_gmm": "#A1887F",
    "trillsson_logistic": "#4DB6AC",
    "trillsson_mlp": "#9575CD",
}

MODEL_RUN_DIRS = {
    "wavlm_erm": "wavlm_erm",
    "wavlm_dann": "wavlm_dann",
    "w2v2_erm": "w2v2_erm",
    "w2v2_dann": "w2v2_dann",
    "lfcc_gmm": "lfcc_gmm_32",
    "trillsson_logistic": "trillsson_logistic",
    "trillsson_mlp": "trillsson_mlp",
}

MODEL_LABELS = {
    "wavlm_erm": "WavLM ERM",
    "wavlm_dann": "WavLM DANN",
    "w2v2_erm": "W2V2 ERM",
    "w2v2_dann": "W2V2 DANN",
    "lfcc_gmm": "LFCC-GMM",
    "trillsson_logistic": "TRILLsson Logistic",
    "trillsson_mlp": "TRILLsson MLP",
}

CODEC_NAMES = {
    "C01": "AMR-WB",
    "C02": "EVS",
    "C03": "G.722",
    "C04": "G.726",
    "C05": "GSM-FR",
    "C06": "iLBC",
    "C07": "MP3",
    "C08": "Opus",
    "C09": "Speex",
    "C10": "Vorbis",
    "C11": "mu-law",
    "NONE": "Uncoded",
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
def load_per_codec_data(path: Path) -> Dict[str, Any]:
    """Load per-codec EER results.
    
    Expected format:
    {
        "C01": {"wavlm_erm": 0.05, "wavlm_dann": 0.04, "w2v2_erm": 0.06, "w2v2_dann": 0.05},
        "C02": {...},
        ...
    }
    
    Or alternative format:
    {
        "wavlm_erm": {"C01": 0.05, "C02": 0.06, ...},
        "wavlm_dann": {...},
        ...
    }
    """
    logger.info(f"Loading per-codec data from: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    # Handle alternative format (model -> codec -> eer)
    if "wavlm_erm" in data and isinstance(data["wavlm_erm"], dict):
        # Transpose to codec -> model -> eer
        codecs = set()
        for model_data in data.values():
            if isinstance(model_data, dict):
                codecs.update(model_data.keys())
        
        transposed = {}
        for codec in codecs:
            transposed[codec] = {}
            for model, model_data in data.items():
                if isinstance(model_data, dict) and codec in model_data:
                    transposed[codec][model] = model_data[codec]
        return transposed
    
    return data


def load_per_codec_from_runs(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load per-codec EER from eval tables under results/runs."""
    data: Dict[str, Dict[str, float]] = {}
    for model_key, run_dir_name in MODEL_RUN_DIRS.items():
        model_dir = results_dir / run_dir_name
        eval_dir_name = resolve_eval_results_dir(model_dir)
        if eval_dir_name is None:
            continue
        csv_path = model_dir / eval_dir_name / "tables" / "metrics_by_codec.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                codec = row.get("domain")
                eer_raw = row.get("eer")
                if not codec or eer_raw is None:
                    continue
                try:
                    data.setdefault(codec, {})[model_key] = float(eer_raw)
                except (TypeError, ValueError):
                    continue
    return data


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
    
    codecs = ["C01", "C02", "C03", "C04", "C05", "C06",
              "C07", "C08", "C09", "C10", "C11", "NONE"]
    
    # Simulated EER values (reasonable ranges)
    np.random.seed(42)
    data = {}
    
    for codec in codecs:
        base_eer = np.random.uniform(0.03, 0.12)
        data[codec] = {
            "wavlm_erm": base_eer + np.random.uniform(0, 0.02),
            "wavlm_dann": base_eer - np.random.uniform(0, 0.02),
            "w2v2_erm": base_eer + np.random.uniform(0.01, 0.03),
            "w2v2_dann": base_eer + np.random.uniform(-0.01, 0.02),
        }
    
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_per_codec_eer(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (15, 7),
    show_values: bool = False,
) -> plt.Figure:
    """Create grouped bar chart for per-codec EER.
    
    Args:
        data: Per-codec EER data (codec -> model -> eer)
        figsize: Figure dimensions
        show_values: Whether to show EER values above bars
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    # Get codecs in order
    codecs = ["C01", "C02", "C03", "C04", "C05", "C06",
              "C07", "C08", "C09", "C10", "C11", "NONE"]
    codecs = [c for c in codecs if c in data]
    
    # Models in order (all available in data)
    model_order = list(MODEL_RUN_DIRS.keys())
    available_models = {
        model_key
        for codec_payload in data.values()
        for model_key in codec_payload.keys()
    }
    models = [model_key for model_key in model_order if model_key in available_models]
    
    # Prepare data arrays
    n_codecs = len(codecs)
    n_models = len(models)
    
    x = np.arange(n_codecs)
    # Keep grouped bars within each codec bucket to avoid overlap.
    width = min(0.20, 0.80 / max(1, n_models))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars for each model
    for i, model in enumerate(models):
        eers = []
        for codec in codecs:
            eer = data.get(codec, {}).get(model, 0)
            # Convert to percentage if needed
            if isinstance(eer, float) and eer < 1:
                eer = eer * 100
            eers.append(eer)
        
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, eers, width,
            label=MODEL_LABELS.get(model, model),
            color=COLORS.get(model, "#9E9E9E"),
            edgecolor='black',
            linewidth=0.5,
            alpha=0.85,
        )
        
        # Optionally show values above bars
        if show_values:
            for bar, eer in zip(bars, eers):
                height = bar.get_height()
                ax.annotate(
                    f'{eer:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7,
                    rotation=90,
                )
    
    # Formatting
    ax.set_xlabel('Codec')
    ax.set_ylabel('EER (%)')
    ax.set_title('Per-Codec EER Comparison on Eval Set')
    
    ax.set_xticks(x)
    codec_labels = [f"{codec}\n({CODEC_NAMES.get(codec, codec)})" for codec in codecs]
    ax.set_xticklabels(codec_labels, rotation=28, ha='right')
    
    # Set y-axis limits
    all_eers = []
    for codec in codecs:
        for model in models:
            eer = data.get(codec, {}).get(model, 0)
            if isinstance(eer, float) and eer < 1:
                eer = eer * 100
            all_eers.append(eer)
    
    if all_eers:
        max_eer = max(all_eers)
        ax.set_ylim(0, max_eer * 1.15)
    
    # Legend
    legend_cols = 2 if n_models <= 6 else 3
    ax.legend(loc='upper left', framealpha=0.92, ncol=legend_cols)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.9, alpha=0.32)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-codec EER bar chart for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Path to runs directory for auto-loading per-codec CSV metrics",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to per-codec EER JSON (override runs loading)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/per_codec_eer.png"),
        help="Output file path (PNG and PDF will be saved)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data instead of loading from file",
    )
    p.add_argument(
        "--show-values",
        action="store_true",
        help="Show EER values above each bar",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[15, 7],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 15 7)",
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
        data = load_per_codec_data(args.input)
    else:
        data = load_per_codec_from_runs(args.results_dir)
        if not data:
            logger.error(f"No per-codec CSV metrics found in: {args.results_dir}")
            return 1
    
    # Create figure
    fig = plot_per_codec_eer(
        data,
        figsize=tuple(args.figsize),
        show_values=args.show_values,
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
    logger.info("Per-Codec EER Figure Summary")
    logger.info("=" * 60)
    logger.info(f"Codecs: {len(data)}")
    logger.info(f"Models: {len(COLORS)}")
    logger.info(f"Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
