#!/usr/bin/env python3
"""Evaluate trained sklearn baselines on eval split.

Supports TRILLsson classifiers and LFCC-GMM models.

Usage:
    # Evaluate TRILLsson logistic on eval
    python scripts/eval_sklearn_baseline.py \
        --model-path runs/trillsson_logistic/model.joblib \
        --features-dir data/features/trillsson \
        --split eval \
        --model-type trillsson

    # Evaluate LFCC-GMM on eval
    python scripts/eval_sklearn_baseline.py \
        --model-path runs/lfcc_gmm_32/model.joblib \
        --features-dir data/features/lfcc \
        --split eval \
        --model-type lfcc_gmm
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sklearn baseline on eval split")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (joblib file)",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory with extracted features",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "eval"],
        default="eval",
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["trillsson", "lfcc_gmm"],
        required=True,
        help="Type of model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: model parent dir / eval_{split})",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Compute per-domain breakdown",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-dann",
        help="Wandb project name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for wandb (default: inferred from model path)",
    )
    return parser.parse_args()


def load_features(features_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata for a split."""
    embeddings_path = features_dir / f"{split}.npy"
    metadata_path = features_dir / f"{split}_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Features not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    return embeddings, metadata


def predict_trillsson(model_data: dict, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with TRILLsson classifier."""
    clf = model_data["classifier"]
    scaler = model_data["scaler"]

    # Standardize
    embeddings_scaled = scaler.transform(embeddings)

    # Predict
    preds = clf.predict(embeddings_scaled)
    probs = clf.predict_proba(embeddings_scaled)

    # Score convention: higher = more bonafide (class 0)
    scores = probs[:, 0]
    return preds, scores


def predict_lfcc_gmm(model_data: dict, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with LFCC-GMM."""
    gmm_bonafide = model_data["gmm_bonafide"]
    gmm_spoof = model_data["gmm_spoof"]
    scaler = model_data["scaler"]

    # Standardize
    embeddings_scaled = scaler.transform(embeddings)

    # Compute log-likelihoods
    log_prob_bonafide = gmm_bonafide.score_samples(embeddings_scaled)
    log_prob_spoof = gmm_spoof.score_samples(embeddings_scaled)

    # Score = log P(x | bonafide) - log P(x | spoof)
    # Higher score = more likely bonafide
    scores = log_prob_bonafide - log_prob_spoof

    # Predictions: positive score = bonafide (class 0)
    preds = (scores < 0).astype(int)  # spoof if score < 0

    return preds, scores


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute EER and minDCF."""
    from asvspoof5_domain_invariant_cm.evaluation import compute_eer, compute_min_dcf

    eer, eer_threshold = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)

    n_bonafide = (labels == 0).sum()
    n_spoof = (labels == 1).sum()

    return {
        "eer": float(eer),
        "min_dcf": float(min_dcf),
        "eer_threshold": float(eer_threshold),
        "n_samples": len(labels),
        "n_bonafide": int(n_bonafide),
        "n_spoof": int(n_spoof),
    }


def compute_per_domain_metrics(df: pd.DataFrame, domain_col: str) -> dict:
    """Compute metrics per domain."""
    from asvspoof5_domain_invariant_cm.evaluation import compute_eer, compute_min_dcf

    domain_metrics = {}
    for domain_name in df[domain_col].unique():
        mask = df[domain_col] == domain_name
        if mask.sum() > 10:
            domain_scores = df.loc[mask, "score"].values
            domain_labels = df.loc[mask, "y_task"].values
            if len(np.unique(domain_labels)) == 2:
                eer, _ = compute_eer(domain_scores, domain_labels)
                min_dcf = compute_min_dcf(domain_scores, domain_labels)
                domain_metrics[domain_name] = {
                    "eer": float(eer),
                    "min_dcf": float(min_dcf),
                    "n_samples": int(mask.sum()),
                }
    return domain_metrics


def main():
    args = parse_args()

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    model_data = joblib.load(args.model_path)

    # Load features
    logger.info(f"Loading features from: {args.features_dir}")
    embeddings, metadata = load_features(args.features_dir, args.split)
    labels = metadata["y_task"].values

    logger.info(f"Loaded {len(embeddings)} samples, embedding dim: {embeddings.shape[1]}")

    # Run inference
    logger.info("Running inference...")
    if args.model_type == "trillsson":
        preds, scores = predict_trillsson(model_data, embeddings)
    elif args.model_type == "lfcc_gmm":
        preds, scores = predict_lfcc_gmm(model_data, embeddings)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Compute metrics
    metrics = compute_metrics(scores, labels)

    logger.info("=" * 60)
    logger.info(f"Overall metrics ({args.split}):")
    logger.info(f"  EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    logger.info(f"  minDCF: {metrics['min_dcf']:.4f}")
    logger.info(f"  Samples: {metrics['n_samples']} (bonafide: {metrics['n_bonafide']}, spoof: {metrics['n_spoof']})")
    logger.info("=" * 60)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.model_path.parent / f"eval_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    result_df = metadata.copy()
    result_df["score"] = scores
    result_df["prediction"] = preds
    result_df.to_csv(output_dir / f"predictions_{args.split}.tsv", sep="\t", index=False)

    # Per-domain breakdown
    codec_metrics = None
    codec_q_metrics = None

    if args.per_domain:
        for domain_col in ["codec", "codec_q"]:
            if domain_col in result_df.columns:
                domain_metrics = compute_per_domain_metrics(result_df, domain_col)
                if domain_metrics:
                    domain_df = pd.DataFrame([
                        {"domain": k, "eer": v["eer"], "min_dcf": v["min_dcf"], "n_samples": v["n_samples"]}
                        for k, v in domain_metrics.items()
                    ]).sort_values("eer", ascending=False)

                    tables_dir = output_dir / "tables"
                    tables_dir.mkdir(exist_ok=True)
                    domain_df.to_csv(tables_dir / f"metrics_by_{domain_col}.csv", index=False)

                    logger.info(f"\n{domain_col.upper()} breakdown:")
                    logger.info(domain_df.to_string(index=False))

                    if domain_col == "codec":
                        codec_metrics = domain_metrics
                    else:
                        codec_q_metrics = domain_metrics

    # Save metrics
    run_name = args.run_name or args.model_path.parent.name
    full_metrics = {
        "model_type": args.model_type,
        "model_path": str(args.model_path),
        "split": args.split,
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    logger.info(f"Results saved to: {output_dir}")

    # Wandb logging
    if args.wandb and WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"eval_{run_name}_{args.split}",
                config={
                    "model_type": args.model_type,
                    "model_path": str(args.model_path),
                    "split": args.split,
                },
                job_type="evaluation",
                tags=["baseline", args.model_type, f"eval_{args.split}"],
            )

            wandb.log({
                f"eval/{args.split}/eer": metrics["eer"],
                f"eval/{args.split}/min_dcf": metrics["min_dcf"],
                f"eval/{args.split}/n_samples": metrics["n_samples"],
            })

            if codec_metrics:
                codec_table = wandb.Table(
                    columns=["codec", "eer", "min_dcf", "n_samples"],
                    data=[[k, v["eer"], v["min_dcf"], v["n_samples"]] for k, v in codec_metrics.items()]
                )
                wandb.log({f"eval/{args.split}/per_codec": codec_table})

            if codec_q_metrics:
                codec_q_table = wandb.Table(
                    columns=["codec_q", "eer", "min_dcf", "n_samples"],
                    data=[[k, v["eer"], v["min_dcf"], v["n_samples"]] for k, v in codec_q_metrics.items()]
                )
                wandb.log({f"eval/{args.split}/per_codec_q": codec_q_table})

            wandb.run.summary[f"{args.split}_eer"] = metrics["eer"]
            wandb.run.summary[f"{args.split}_min_dcf"] = metrics["min_dcf"]
            wandb.finish()
            logger.info("Logged to Wandb")

        except Exception as e:
            logger.warning(f"Wandb logging failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
