#!/usr/bin/env python3
"""Generate LaTeX and Markdown tables for thesis.

This script generates all thesis tables:
- T1: Main results table (Dev/Eval EER, minDCF)
- T2: Per-codec EER comparison
- T3: OOD gap analysis
- T4: Projection probe results
- T5: Dataset statistics
- T6: Synthetic augmentation coverage
- T7: Hyperparameters

Usage:
    # Generate all tables with default paths
    python scripts/generate_thesis_tables.py

    # Generate with custom paths
    python scripts/generate_thesis_tables.py \\
        --main-results results/main_results.json \\
        --per-codec results/per_codec_eer.json \\
        --projection-wavlm results/rq3_projection.json \\
        --projection-w2v2 results/rq3_projection_w2v2.json \\
        --output-dir figures/tables \\
        --verbose

    # Generate only specific tables
    python scripts/generate_thesis_tables.py --tables T1 T4 T5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "w2v2_dann_v2": "w2v2_dann_v2",
    "lfcc_gmm": "lfcc_gmm_32",
    "trillsson_logistic": "trillsson_logistic",
    "trillsson_mlp": "trillsson_mlp",
}

T1_EXTRA_MODEL_RUN_DIRS = {
    "wavlm_erm_aug": "wavlm_erm_aug",
    "w2v2_erm_aug": "w2v2_erm_aug",
}

MODEL_LABELS = {
    "wavlm_erm": "WavLM ERM",
    "wavlm_erm_aug": "WavLM ERM + Aug",
    "wavlm_dann": "WavLM DANN",
    "w2v2_erm": "W2V2 ERM",
    "w2v2_erm_aug": "W2V2 ERM + Aug",
    "w2v2_dann": "W2V2 DANN",
    "w2v2_dann_v2": "W2V2 DANN v2",
    "lfcc_gmm": "LFCC-GMM",
    "trillsson_logistic": "TRILLsson Logistic",
    "trillsson_mlp": "TRILLsson MLP",
}


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(data: Optional[Dict], *keys, default=None):
    """Safely get nested dictionary value."""
    if data is None:
        return default
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


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

    main_results_run_dirs = {**MODEL_RUN_DIRS, **T1_EXTRA_MODEL_RUN_DIRS}
    for model_key, run_dir_name in main_results_run_dirs.items():
        model_dir = results_dir / run_dir_name
        eval_dir_name = resolve_eval_results_dir(model_dir)
        if eval_dir_name is None:
            continue
        eval_metrics_path = model_dir / eval_dir_name / "metrics.json"
        if not eval_metrics_path.exists():
            continue

        eval_payload = load_json(eval_metrics_path)
        if eval_payload is None:
            continue

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
                fallback_payload = load_json(fallback_path)
                if fallback_payload is None:
                    continue
                dev_eer = extract_dev_eer_from_metrics_payload(fallback_payload)
                if dev_eer is not None:
                    break

        model_result["dev_eer"] = dev_eer
        results[model_key] = model_result

    return results


def resolve_eval_results_dir(model_dir: Path) -> Optional[str]:
    """Pick the best available eval results directory for a run.

    Preference order:
    1) eval_eval_full (full eval)
    2) eval_eval (default eval)
    3) eval_eval_epoch5 (legacy wavlm_dann patched checkpoint eval)
    """
    candidate_dirs = ["eval_eval_full", "eval_eval", "eval_eval_epoch5"]
    for candidate_dir in candidate_dirs:
        if (model_dir / candidate_dir / "metrics.json").exists():
            return candidate_dir
    return None


def load_per_codec_from_runs(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load per-codec EER from eval tables under results/runs."""
    per_codec: Dict[str, Dict[str, float]] = {}
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
                    per_codec.setdefault(codec, {})[model_key] = float(eer_raw)
                except (TypeError, ValueError):
                    continue
    return per_codec


# ---------------------------------------------------------------------------
# Table Generation: LaTeX and Markdown
# ---------------------------------------------------------------------------
def to_latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None,
    note: str = "",
) -> str:
    """Convert headers and rows to LaTeX table format."""
    if column_format is None:
        column_format = "l" + "c" * (len(headers) - 1)
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(r"\vspace{0.4em}")
        lines.append(rf"{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Convert headers and rows to Markdown table format."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_table(
    output_dir: Path,
    name: str,
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    latex_note: str = "",
) -> None:
    """Save table in both LaTeX and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LaTeX
    tex_path = output_dir / f"{name}.tex"
    tex_content = to_latex_table(headers, rows, caption, label, note=latex_note)
    tex_path.write_text(tex_content)
    logger.info(f"Saved LaTeX: {tex_path}")
    
    # Markdown
    md_path = output_dir / f"{name}.md"
    md_content = f"# {caption}\n\n" + to_markdown_table(headers, rows)
    md_path.write_text(md_content)
    logger.info(f"Saved Markdown: {md_path}")


# ---------------------------------------------------------------------------
# T1: Main Results Table
# ---------------------------------------------------------------------------
def generate_t1_main_results(
    main_results: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T1: Main results table."""
    logger.info("Generating T1: Main Results Table")
    
    headers = ["Model", "Backbone", "Dev EER (%)", "Eval EER (%)", "Eval minDCF"]
    
    # Default/placeholder data if no results file
    if main_results is None:
        rows = [
            ["ERM", "WavLM", "—", "—", "—"],
            ["ERM + Aug", "WavLM", "—", "—", "—"],
            ["DANN", "WavLM", "—", "—", "—"],
            ["ERM", "W2V2", "—", "—", "—"],
            ["ERM + Aug", "W2V2", "—", "—", "—"],
            ["DANN", "W2V2", "—", "—", "—"],
            ["LFCC-GMM", "LFCC", "—", "—", "—"],
            ["TRILLsson Logistic", "TRILLsson", "—", "—", "—"],
            ["TRILLsson MLP", "TRILLsson", "—", "—", "—"],
        ]
        logger.warning("Using placeholder data for T1 (no main_results.json)")
    else:
        rows = []
        model_order = [
            "wavlm_erm",
            "wavlm_erm_aug",
            "wavlm_dann",
            "w2v2_erm",
            "w2v2_erm_aug",
            "w2v2_dann",
            "lfcc_gmm",
            "trillsson_logistic",
            "trillsson_mlp",
        ]
        for model_key in model_order:
            if model_key not in main_results:
                continue
            model_data = main_results.get(model_key, {})
            if model_key == "lfcc_gmm":
                backbone = "LFCC"
                method = "LFCC-GMM"
            elif model_key == "trillsson_logistic":
                backbone = "TRILLsson"
                method = "TRILLsson Logistic"
            elif model_key == "trillsson_mlp":
                backbone = "TRILLsson"
                method = "TRILLsson MLP"
            else:
                backbone = "WavLM" if "wavlm" in model_key else "W2V2"
                if "dann" in model_key:
                    method = "DANN"
                elif "erm_aug" in model_key:
                    method = "ERM + Aug"
                else:
                    method = "ERM"
            
            dev_eer = model_data.get("dev_eer", "—")
            eval_eer = model_data.get("eval_eer", "—")
            eval_mindcf = model_data.get("eval_mindcf", "—")
            
            if isinstance(dev_eer, float):
                dev_eer = f"{dev_eer * 100:.2f}"
            if isinstance(eval_eer, float):
                eval_eer = f"{eval_eer * 100:.2f}"
            if isinstance(eval_mindcf, float):
                eval_mindcf = f"{eval_mindcf:.4f}"
            
            rows.append([method, backbone, str(dev_eer), str(eval_eer), str(eval_mindcf)])
    
    save_table(
        output_dir, "T1_main_results", headers, rows,
        caption="Main Results: EER and minDCF for ERM vs DANN",
        label="tab:main_results",
        latex_note=(
            r"W2V2 DANN uses weighted pooling across all 12 layers "
            r"with an exponential $\lambda$ schedule 0.01$\rightarrow$1.0 and no warmup."
        ),
    )
    return True


# ---------------------------------------------------------------------------
# T2: Per-Codec EER Comparison
# ---------------------------------------------------------------------------
def generate_t2_per_codec(
    per_codec: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T2: Per-codec EER comparison table."""
    logger.info("Generating T2: Per-Codec EER Comparison")
    
    model_order = [
        "wavlm_erm",
        "wavlm_dann",
        "w2v2_erm",
        "w2v2_dann",
        "lfcc_gmm",
        "trillsson_logistic",
        "trillsson_mlp",
    ]
    if per_codec:
        available_models = {
            model_key
            for codec_payload in per_codec.values()
            for model_key in codec_payload.keys()
        }
        model_order = [key for key in model_order if key in available_models]
    headers = ["Codec"] + [MODEL_LABELS.get(key, key) for key in model_order]
    
    # Default codec list
    codecs = ["C01", "C02", "C03", "C04", "C05", "C06", 
              "C07", "C08", "C09", "C10", "C11", "NONE"]
    
    if per_codec is None:
        rows = [[codec] + ["—"] * len(model_order) for codec in codecs]
        logger.warning("Using placeholder data for T2 (no per_codec_eer.json)")
    else:
        rows = []
        for codec in codecs:
            codec_data = per_codec.get(codec, {})
            row = [codec]
            for model in model_order:
                eer = codec_data.get(model, "—")
                if isinstance(eer, float):
                    eer = f"{eer * 100:.2f}"
                row.append(str(eer))
            rows.append(row)
    
    save_table(
        output_dir, "T2_per_codec", headers, rows,
        caption="Per-Codec EER (\\%) on Eval Set",
        label="tab:per_codec",
    )
    return True


# ---------------------------------------------------------------------------
# T3: OOD Gap Analysis
# ---------------------------------------------------------------------------
def generate_t3_ood_gap(
    main_results: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T3: OOD gap analysis table."""
    logger.info("Generating T3: OOD Gap Analysis")
    
    headers = ["Model", "Backbone", "Dev EER (%)", "Eval EER (%)", "Gap", "Gap Reduction"]
    
    if main_results is None:
        rows = [
            ["ERM", "WavLM", "—", "—", "—", "—"],
            ["DANN", "WavLM", "—", "—", "—", "—"],
            ["ERM", "W2V2", "—", "—", "—", "—"],
            ["DANN", "W2V2", "—", "—", "—", "—"],
        ]
        logger.warning("Using placeholder data for T3 (no main_results.json)")
    else:
        rows = []
        # Calculate baseline gaps for each backbone
        for backbone in ["wavlm", "w2v2"]:
            backbone_name = "WavLM" if backbone == "wavlm" else "W2V2"
            method_pairs = [("ERM", f"{backbone}_erm"), ("DANN", f"{backbone}_dann")]

            baseline_data = main_results.get(f"{backbone}_erm", {})
            baseline_dev = baseline_data.get("dev_eer", 0)
            baseline_eval = baseline_data.get("eval_eer", 0)
            baseline_gap = (
                baseline_eval - baseline_dev
                if isinstance(baseline_dev, float) and isinstance(baseline_eval, float)
                else None
            )

            for method_name, model_key in method_pairs:
                model_data = main_results.get(model_key, {})
                if not model_data:
                    continue
            
                dev = model_data.get("dev_eer", 0)
                eval_value = model_data.get("eval_eer", 0)
                gap = (eval_value - dev) if isinstance(dev, float) and isinstance(eval_value, float) else None

                dev_str = f"{dev * 100:.2f}" if isinstance(dev, float) else "—"
                eval_str = f"{eval_value * 100:.2f}" if isinstance(eval_value, float) else "—"
                gap_str = f"{gap * 100:.2f}" if gap is not None else "—"

                if method_name == "ERM":
                    reduction_str = "(baseline)"
                elif baseline_gap is not None and gap is not None and baseline_gap > 0:
                    reduction = ((baseline_gap - gap) / baseline_gap) * 100
                    reduction_str = f"{reduction:.1f}\\%"
                else:
                    reduction_str = "—"

                rows.append([method_name, backbone_name, dev_str, eval_str, gap_str, reduction_str])
    
    save_table(
        output_dir, "T3_ood_gap", headers, rows,
        caption="OOD Gap Analysis: Dev vs Eval Generalization",
        label="tab:ood_gap",
    )
    return True


# ---------------------------------------------------------------------------
# T4: Projection Probe Results
# ---------------------------------------------------------------------------
def generate_t4_projection_probes(
    projection_wavlm: Optional[Dict[str, Any]],
    projection_w2v2: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T4: Projection probe results table."""
    logger.info("Generating T4: Projection Probe Results")
    
    headers = ["Backbone", "ERM Probe Acc", "DANN Probe Acc", "Reduction", "Rel. Reduction"]
    rows = []
    
    for backbone_name, proj_data in [("WavLM", projection_wavlm), ("W2V2", projection_w2v2)]:
        if proj_data is None:
            rows.append([backbone_name, "—", "—", "—", "—"])
            continue
        
        erm_acc = safe_get(proj_data, "results", "erm", "codec", "accuracy", default=None)
        dann_acc = safe_get(proj_data, "results", "dann", "codec", "accuracy", default=None)
        reduction = safe_get(proj_data, "comparison", "codec", "reduction", default=None)
        rel_reduction = safe_get(proj_data, "comparison", "codec", "relative_reduction", default=None)
        
        erm_str = f"{erm_acc:.3f}" if erm_acc is not None else "—"
        dann_str = f"{dann_acc:.3f}" if dann_acc is not None else "—"
        red_str = f"{reduction:.3f}" if reduction is not None else "—"
        rel_str = f"{rel_reduction * 100:.1f}\\%" if rel_reduction is not None else "—"
        
        rows.append([backbone_name, erm_str, dann_str, red_str, rel_str])
    
    save_table(
        output_dir, "T4_projection_probes", headers, rows,
        caption="Projection Layer Codec Probe Accuracy (RQ3)",
        label="tab:projection_probes",
    )
    return True


# ---------------------------------------------------------------------------
# T5: Dataset Statistics
# ---------------------------------------------------------------------------
def generate_t5_dataset_stats(output_dir: Path) -> bool:
    """Generate T5: Dataset statistics table."""
    logger.info("Generating T5: Dataset Statistics")
    
    headers = ["Split", "Bonafide", "Spoof", "Total", "Codecs"]
    
    # Counts are aligned to project-observed protocol/eval outputs.
    # Duration hours require audio-level metadata aggregation and are left as TBD.
    rows = [
        ["Train", "18,797", "163,560", "182,357", "—"],
        ["Dev", "31,334", "109,616", "140,950", "—"],
        ["Eval", "138,688", "542,086", "680,774", "12"],
    ]
    
    save_table(
        output_dir, "T5_dataset_stats", headers, rows,
        caption="ASVspoof 5 Track 1 Dataset Statistics",
        label="tab:dataset_stats",
    )
    return True


# ---------------------------------------------------------------------------
# T6: Synthetic Augmentation Coverage
# ---------------------------------------------------------------------------
def generate_t6_augmentation(output_dir: Path) -> bool:
    """Generate T6: Synthetic augmentation coverage table."""
    logger.info("Generating T6: Synthetic Augmentation Coverage")
    
    headers = ["Codec", "Quality Levels", "Bitrates", "Configured", "Observed in runs"]

    # Matches config request (MP3/AAC/OPUS) and observed ffmpeg support in logs.
    rows = [
        ["MP3 (libmp3lame)", "5 (1-5)", "64k, 96k, 128k, 192k, 256k", "Yes", "Yes"],
        ["AAC (aac)", "5 (1-5)", "32k, 64k, 96k, 128k, 192k", "Yes", "Yes"],
        ["Opus (libopus)", "5 (1-5)", "12k, 24k, 48k, 64k, 96k", "Yes", "No (ffmpeg unsupported)"],
    ]
    
    save_table(
        output_dir, "T6_augmentation", headers, rows,
        caption="Synthetic Codec Augmentation for DANN Training",
        label="tab:augmentation",
    )
    return True


# ---------------------------------------------------------------------------
# T7: Hyperparameters
# ---------------------------------------------------------------------------
def generate_t7_hyperparameters(output_dir: Path) -> bool:
    """Generate T7: Hyperparameters table."""
    logger.info("Generating T7: Hyperparameters")
    
    headers = [
        "Parameter",
        "WavLM ERM",
        "WavLM DANN",
        "W2V2 ERM",
        "W2V2 DANN",
        "LFCC-GMM",
        "TRILLsson Logistic",
        "TRILLsson MLP",
    ]

    # Values taken from configs/*.yaml and results/runs/* metrics where applicable.
    rows = [
        ["Model type", "SSL + linear head", "SSL + DANN", "SSL + linear head", "SSL + DANN", "GMM baseline", "Linear baseline", "MLP baseline"],
        ["Backbone/features", "WavLM Base+", "WavLM Base+", "Wav2Vec2 Base", "Wav2Vec2 Base", "LFCC (120-dim)", "TRILLsson (1024-dim)", "TRILLsson (1024-dim)"],
        ["Layer selection", "weighted (k=6)", "weighted (k=6)", "weighted (k=6)", "weighted (k=6)", "—", "—", "—"],
        ["Backbone frozen", "true", "true", "true", "true", "—", "—", "—"],
        ["Projection output dim", "256", "256", "256", "256", "—", "—", "—"],
        ["Batch size", "256", "256", "256", "256", "—", "—", "—"],
        ["Learning rate", "1e-4", "1e-4", "5e-5", "5e-5", "—", "—", "—"],
        ["Optimizer", "AdamW", "AdamW", "AdamW", "AdamW", "—", "scikit-learn", "scikit-learn"],
        ["Weight decay", "0.01", "0.01", "0.01", "0.01", "—", "—", "—"],
        ["Max epochs", "50", "50", "50", "50", "—", "—", "—"],
        ["Patience", "10", "10", "10", "10", "—", "—", "—"],
        ["Gradient clip", "1.0", "0.5", "0.5", "0.5", "—", "—", "—"],
        [r"$\\lambda$ schedule", "N/A (ERM)", "linear 0.1→0.75 (warmup 3)", "N/A (ERM)", "exponential 0.01→1.0", "N/A", "N/A", "N/A"],
        ["Augmentation codecs", "N/A", "MP3, AAC, OPUS", "N/A", "MP3, AAC, OPUS", "N/A", "N/A", "N/A"],
        ["Augmentation qualities", "N/A", "1-5", "N/A", "1-5", "N/A", "N/A", "N/A"],
        ["Random seed", "42", "42", "42", "42", "42", "42", "42"],
    ]
    
    save_table(
        output_dir, "T7_hyperparameters", headers, rows,
        caption="Key Hyperparameters by Model Family",
        label="tab:hyperparameters",
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LaTeX and Markdown tables for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input files
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Path to runs directory for auto-loading metrics",
    )
    p.add_argument(
        "--main-results",
        type=Path,
        default=None,
        help="Path to main results JSON (override runs loading)",
    )
    p.add_argument(
        "--per-codec",
        type=Path,
        default=None,
        help="Path to per-codec EER JSON (override runs loading)",
    )
    p.add_argument(
        "--projection-wavlm",
        type=Path,
        default=Path("results/rq3_projection.json"),
        help="Path to WavLM projection probes JSON",
    )
    p.add_argument(
        "--projection-w2v2",
        type=Path,
        default=Path("results/rq3_projection_w2v2.json"),
        help="Path to W2V2 projection probes JSON",
    )
    
    # Output
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/tables"),
        help="Output directory for tables",
    )
    
    # Table selection
    p.add_argument(
        "--tables",
        nargs="+",
        choices=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "all"],
        default=["all"],
        help="Which tables to generate (default: all)",
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
    
    # Load available data
    if args.main_results is not None:
        main_results = load_json(args.main_results)
    else:
        main_results = load_main_results_from_runs(args.results_dir)
        if not main_results:
            main_results = None
            logger.warning(f"No run metrics found in: {args.results_dir}")
    if args.per_codec is not None:
        per_codec = load_json(args.per_codec)
    else:
        per_codec = load_per_codec_from_runs(args.results_dir)
        if not per_codec:
            per_codec = None
            logger.warning(f"No per-codec CSV metrics found in: {args.results_dir}")
    projection_wavlm = load_json(args.projection_wavlm)
    projection_w2v2 = load_json(args.projection_w2v2)
    
    # Determine which tables to generate
    tables = args.tables
    if "all" in tables:
        tables = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    
    # Generate tables
    output_dir = args.output_dir
    success_count = 0
    
    if "T1" in tables:
        if generate_t1_main_results(main_results, output_dir):
            success_count += 1
    
    if "T2" in tables:
        if generate_t2_per_codec(per_codec, output_dir):
            success_count += 1
    
    if "T3" in tables:
        if generate_t3_ood_gap(main_results, output_dir):
            success_count += 1
    
    if "T4" in tables:
        if generate_t4_projection_probes(projection_wavlm, projection_w2v2, output_dir):
            success_count += 1
    
    if "T5" in tables:
        if generate_t5_dataset_stats(output_dir):
            success_count += 1
    
    if "T6" in tables:
        if generate_t6_augmentation(output_dir):
            success_count += 1
    
    if "T7" in tables:
        if generate_t7_hyperparameters(output_dir):
            success_count += 1
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Generated {success_count}/{len(tables)} tables")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
