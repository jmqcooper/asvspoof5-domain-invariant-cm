#!/usr/bin/env python3
"""Evaluation entrypoint for trained models with comprehensive logging.

Usage:
    # Evaluate on dev set
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split dev

    # Evaluate with per-domain breakdown
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split dev --per-domain

    # Evaluate on eval set with wandb logging
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split eval --wandb
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset,
    AudioCollator,
    load_vocab,
    normalize_domain_value,
)
from asvspoof5_domain_invariant_cm.evaluation import (
    generate_overall_metrics,
    save_domain_tables,
    save_metrics_report,
    save_predictions,
    generate_scorefile,
)
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_experiment_context,
    get_manifest_path,
    load_config,
    setup_logging,
)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "eval"],
        default="dev",
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: checkpoint parent dir)",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Compute per-domain (CODEC, CODEC_Q) metrics",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--scorefile",
        action="store_true",
        help="Generate official-format score file",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Wandb",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-dann",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (team or username)",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Evaluate on a random subset for quick signal (uses --max-samples)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples for evaluation (default: all samples). Implies --quick-eval if set.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed for quick-eval subset sampling",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint (uses run_dir vocabs for head sizes).
    
    Auto-detects discriminator input dimension from checkpoint weights for
    backwards compatibility with both old (post-projection, 256-dim) and 
    new (pre-projection, 1536-dim) checkpoints.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state_dict", {})

    # Get vocab sizes from checkpoint or load from run dir
    run_dir = checkpoint_path.parent.parent
    codec_vocab = load_vocab(run_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(run_dir / "codec_q_vocab.json")

    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)

    # Build model
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,  # Always freeze for inference
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    if pooling_method == "stats":
        proj_input_dim = backbone.hidden_size * 2
    else:
        proj_input_dim = backbone.hidden_size

    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)

    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        # Auto-detect discriminator input dimension from checkpoint weights
        # This provides backwards compatibility with old checkpoints trained
        # with post-projection discriminator input (256-dim) vs new architecture
        # that uses pre-projection input (1536-dim from stats pooling)
        disc_weight_key = "domain_discriminator.shared.0.weight"
        if disc_weight_key in state_dict:
            disc_input_dim = state_dict[disc_weight_key].shape[1]
            logger.info(f"Auto-detected discriminator input dim from checkpoint: {disc_input_dim}")
        else:
            # Fallback to config or default to pre-projection dim
            disc_input_dim = disc_cfg.get("input_dim", proj_input_dim)
            logger.info(f"Using config discriminator input dim: {disc_input_dim}")

        domain_discriminator = MultiHeadDomainDiscriminator(
            input_dim=disc_input_dim,
            num_codecs=num_codecs,
            num_codec_qs=num_codec_qs,
            hidden_dim=disc_cfg.get("hidden_dim", 512),
            dropout=disc_cfg.get("dropout", 0.1),
        )

        model = DANNModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
            domain_discriminator=domain_discriminator,
            lambda_=0.0,  # No GRL effect during inference
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Prefer manifest vocabs for evaluation reporting if available.
    manifest_codec_vocab_path = run_dir / "manifest_codec_vocab.json"
    manifest_codec_q_vocab_path = run_dir / "manifest_codec_q_vocab.json"
    if manifest_codec_vocab_path.exists() and manifest_codec_q_vocab_path.exists():
        eval_codec_vocab = load_vocab(manifest_codec_vocab_path)
        eval_codec_q_vocab = load_vocab(manifest_codec_q_vocab_path)
    else:
        eval_codec_vocab = codec_vocab
        eval_codec_q_vocab = codec_q_vocab

    return model, config, codec_vocab, codec_q_vocab, eval_codec_vocab, eval_codec_q_vocab


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Run inference and collect predictions.

    Args:
        model: Trained model.
        dataloader: Evaluation dataloader.
        device: Device.

    Returns:
        List of prediction dictionaries.
    """
    predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        # Score convention: higher = more bonafide (class 0)
        probs = torch.softmax(outputs["task_logits"], dim=-1)
        scores = probs[:, 0]  # P(bonafide)
        preds = outputs["task_logits"].argmax(dim=-1)

        # Collect predictions
        metadata = batch["metadata"]
        batch_size = waveform.shape[0]

        for i in range(batch_size):
            pred_dict = {
                "flac_file": metadata["flac_file"][i],
                "score": scores[i].cpu().item(),
                "prediction": preds[i].cpu().item(),
                "y_task": batch["y_task"][i].item(),
                "y_codec": batch["y_codec"][i].item(),
                "y_codec_q": batch["y_codec_q"][i].item(),
            }

            # Add optional metadata
            for key in ["speaker_id", "codec_seed", "codec", "codec_q", "attack_label", "attack_tag"]:
                if key in metadata:
                    pred_dict[key] = metadata[key][i]

            predictions.append(pred_dict)

    return predictions


def plot_score_distribution(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Score Distribution",
) -> None:
    """Plot score distribution by class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    bonafide_scores = df[df["y_task"] == 0]["score"]
    spoof_scores = df[df["y_task"] == 1]["score"]

    ax.hist(bonafide_scores, bins=50, alpha=0.7, label="Bonafide", density=True)
    ax.hist(spoof_scores, bins=50, alpha=0.7, label="Spoof", density=True)
    ax.set_xlabel("Score (P(bonafide))")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend()

    # Box plot
    ax = axes[1]
    data = [bonafide_scores, spoof_scores]
    ax.boxplot(data, labels=["Bonafide", "Spoof"])
    ax.set_ylabel("Score")
    ax.set_title("Score Box Plot")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_domain_eer(
    domain_metrics: dict,
    output_path: Path,
    domain_name: str = "CODEC",
) -> None:
    """Plot EER by domain."""
    if not domain_metrics:
        return

    domains = list(domain_metrics.keys())
    eers = [domain_metrics[d].get("eer", 0) * 100 for d in domains]  # Convert to %
    samples = [domain_metrics[d].get("n_samples", 0) for d in domains]

    # Sort by EER
    sorted_idx = np.argsort(eers)[::-1]
    domains = [domains[i] for i in sorted_idx]
    eers = [eers[i] for i in sorted_idx]
    samples = [samples[i] for i in sorted_idx]

    # Limit to top 20 for readability
    if len(domains) > 20:
        domains = domains[:20]
        eers = eers[:20]
        samples = samples[:20]

    fig, ax = plt.subplots(figsize=(10, max(6, len(domains) * 0.3)))

    colors = plt.cm.RdYlGn_r(np.array(eers) / max(eers) if max(eers) > 0 else np.zeros_like(eers))

    bars = ax.barh(domains, eers, color=colors)

    # Add sample counts as text
    for i, (bar, n) in enumerate(zip(bars, samples)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'n={n}', va='center', fontsize=8)

    ax.set_xlabel("EER (%)")
    ax.set_title(f"EER by {domain_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def log_to_wandb(
    args,
    metrics: dict,
    df: pd.DataFrame,
    output_dir: Path,
    config: dict,
    codec_metrics: Optional[dict] = None,
    codec_q_metrics: Optional[dict] = None,
) -> None:
    """Log comprehensive results to wandb."""
    if not WANDB_AVAILABLE:
        logger.warning("Wandb not available, skipping logging")
        return

    run_name = args.checkpoint.parent.parent.name

    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval_{run_name}_{args.split}",
            config=config,
            job_type="evaluation",
        )

        # Log overall metrics
        wandb_metrics = {
            # Primary metrics
            f"eval/{args.split}/eer": metrics["eer"],
            f"eval/{args.split}/min_dcf": metrics["min_dcf"],
            f"eval/{args.split}/auc": metrics["auc"],
            f"eval/{args.split}/eer_threshold": metrics["eer_threshold"],

            # Threshold-based metrics at EER operating point
            f"eval/{args.split}/f1_macro": metrics["f1_macro"],
            f"eval/{args.split}/precision_macro": metrics["precision_macro"],
            f"eval/{args.split}/recall_macro": metrics["recall_macro"],
            f"eval/{args.split}/f1_bonafide": metrics["f1_bonafide"],
            f"eval/{args.split}/f1_spoof": metrics["f1_spoof"],

            # Sample counts
            f"eval/{args.split}/n_samples": metrics["n_samples"],
            f"eval/{args.split}/n_bonafide": metrics["n_bonafide"],
            f"eval/{args.split}/n_spoof": metrics["n_spoof"],
        }

        # Add t-DCF if available
        if metrics.get("tdcf_min") is not None:
            wandb_metrics[f"eval/{args.split}/tdcf_min"] = metrics["tdcf_min"]

        wandb.log(wandb_metrics)

        # Log score distribution as histogram
        bonafide_scores = df[df["y_task"] == 0]["score"].tolist()
        spoof_scores = df[df["y_task"] == 1]["score"].tolist()

        wandb.log({
            f"eval/{args.split}/bonafide_scores": wandb.Histogram(bonafide_scores),
            f"eval/{args.split}/spoof_scores": wandb.Histogram(spoof_scores),
        })

        # Log score distribution plot
        score_plot_path = output_dir / "score_distribution.png"
        if score_plot_path.exists():
            wandb.log({
                f"eval/{args.split}/score_distribution": wandb.Image(str(score_plot_path))
            })

        # Log per-domain metrics as tables with all metrics
        if codec_metrics:
            codec_table = wandb.Table(
                columns=["codec", "eer", "auc", "f1_macro", "n_samples", "n_bonafide", "n_spoof"],
                data=[
                    [k, v["eer"], v.get("auc", 0), v.get("f1_macro", 0),
                     v["n_samples"], v.get("n_bonafide", 0), v.get("n_spoof", 0)]
                    for k, v in codec_metrics.items()
                ]
            )
            wandb.log({f"eval/{args.split}/per_codec": codec_table})

            # Log per-codec scalar metrics for easy comparison
            for codec_name, codec_vals in codec_metrics.items():
                safe_name = codec_name.replace("/", "_").replace(" ", "_")
                wandb.log({
                    f"eval/{args.split}/codec/{safe_name}/eer": codec_vals["eer"],
                    f"eval/{args.split}/codec/{safe_name}/auc": codec_vals.get("auc", 0),
                    f"eval/{args.split}/codec/{safe_name}/f1_macro": codec_vals.get("f1_macro", 0),
                })

            # Log codec EER plot
            codec_plot_path = output_dir / "eer_by_codec.png"
            if codec_plot_path.exists():
                wandb.log({
                    f"eval/{args.split}/eer_by_codec_plot": wandb.Image(str(codec_plot_path))
                })

        if codec_q_metrics:
            codec_q_table = wandb.Table(
                columns=["codec_q", "eer", "auc", "f1_macro", "n_samples", "n_bonafide", "n_spoof"],
                data=[
                    [k, v["eer"], v.get("auc", 0), v.get("f1_macro", 0),
                     v["n_samples"], v.get("n_bonafide", 0), v.get("n_spoof", 0)]
                    for k, v in codec_q_metrics.items()
                ]
            )
            wandb.log({f"eval/{args.split}/per_codec_q": codec_q_table})

            # Log per-codec_q scalar metrics for easy comparison
            for cq_name, cq_vals in codec_q_metrics.items():
                safe_name = str(cq_name).replace("/", "_").replace(" ", "_")
                wandb.log({
                    f"eval/{args.split}/codec_q/{safe_name}/eer": cq_vals["eer"],
                    f"eval/{args.split}/codec_q/{safe_name}/auc": cq_vals.get("auc", 0),
                    f"eval/{args.split}/codec_q/{safe_name}/f1_macro": cq_vals.get("f1_macro", 0),
                })

            # Log codec_q EER plot
            codec_q_plot_path = output_dir / "eer_by_codec_q.png"
            if codec_q_plot_path.exists():
                wandb.log({
                    f"eval/{args.split}/eer_by_codec_q_plot": wandb.Image(str(codec_q_plot_path))
                })

        # Log predictions sample as table (first 100)
        sample_df = df.head(100)[["flac_file", "score", "prediction", "y_task"]]
        if "codec" in df.columns:
            sample_df = df.head(100)[["flac_file", "score", "prediction", "y_task", "codec"]]
        pred_table = wandb.Table(dataframe=sample_df)
        wandb.log({f"eval/{args.split}/predictions_sample": pred_table})

        # Set summary with all key metrics
        wandb.run.summary[f"{args.split}_eer"] = metrics["eer"]
        wandb.run.summary[f"{args.split}_min_dcf"] = metrics["min_dcf"]
        wandb.run.summary[f"{args.split}_auc"] = metrics["auc"]
        wandb.run.summary[f"{args.split}_f1_macro"] = metrics["f1_macro"]
        if metrics.get("tdcf_min") is not None:
            wandb.run.summary[f"{args.split}_tdcf_min"] = metrics["tdcf_min"]

        wandb.finish()
        logger.info("Logged metrics to Wandb")

    except Exception as e:
        logger.warning(f"Wandb logging failed: {e}")


def main():
    args = parse_args()

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, config, _model_codec_vocab, _model_codec_q_vocab, eval_codec_vocab, eval_codec_q_vocab = load_model_from_checkpoint(
        args.checkpoint, device
    )

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.checkpoint.parent.parent / f"eval_{args.split}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with JSON output
    setup_logging(output_dir, json_output=True)
    logger.info(f"Output directory: {output_dir}")

    # Log experiment context
    context = get_experiment_context(config)

    # Load dataset
    audio_cfg = config.get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)

    dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path(args.split),
        codec_vocab=eval_codec_vocab,
        codec_q_vocab=eval_codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    # Optional quick-eval subset
    total_samples = len(dataset)
    max_samples = args.max_samples if args.max_samples is not None else (50000 if args.quick_eval else total_samples)
    if args.quick_eval or (args.max_samples is not None and args.max_samples < total_samples):
        subset_size = min(max_samples, total_samples)
        rng = np.random.default_rng(args.subset_seed)
        subset_indices = rng.choice(total_samples, size=subset_size, replace=False)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        logger.info(
            f"Quick-eval enabled: using {subset_size} / {total_samples} samples "
            f"(seed={args.subset_seed})"
        )
    else:
        logger.info(f"Evaluation samples: {total_samples}")

    # Create dataloader
    fixed_length = int(max_duration * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Run inference
    predictions = run_inference(model, dataloader, device)

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Normalize domain labels for consistent reporting
    if "codec" in df.columns:
        df["codec"] = df["codec"].apply(lambda v: normalize_domain_value(v, is_codec_q=False))
    if "codec_q" in df.columns:
        df["codec_q"] = df["codec_q"].apply(lambda v: normalize_domain_value(v, is_codec_q=True))

    # Save predictions
    pred_path = output_dir / "predictions.tsv"
    save_predictions(predictions, pred_path)
    logger.info(f"Saved predictions: {pred_path}")

    # Generate score file
    if args.scorefile:
        score_path = output_dir / f"scores_{args.split}.txt"
        generate_scorefile(df, score_path)
        logger.info(f"Saved score file: {score_path}")

    # Compute overall metrics
    scores = df["score"].values
    labels = df["y_task"].values

    metrics = generate_overall_metrics(
        scores, labels,
        bootstrap=args.bootstrap,
        seed=config.get("seed", 42),
    )

    logger.info("=" * 60)
    logger.info(f"Overall metrics ({args.split}):")
    logger.info(f"  EER: {metrics['eer']:.4f} (threshold: {metrics['eer_threshold']:.4f})")
    logger.info(f"  minDCF: {metrics['min_dcf']:.4f}")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info("-" * 40)
    logger.info(f"  Metrics at EER threshold:")
    logger.info(f"    F1 (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"    Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"    Recall (macro): {metrics['recall_macro']:.4f}")
    logger.info(f"    F1 (bonafide/spoof): {metrics['f1_bonafide']:.4f} / {metrics['f1_spoof']:.4f}")
    logger.info("-" * 40)
    if metrics.get('tdcf_min') is not None:
        logger.info(f"  t-DCF (min): {metrics['tdcf_min']:.4f}")
    else:
        logger.info(f"  t-DCF: Not computed (ASV scores required)")
    logger.info("-" * 40)
    logger.info(f"  Samples: {metrics['n_samples']} (bonafide: {metrics['n_bonafide']}, spoof: {metrics['n_spoof']})")
    logger.info("=" * 60)

    # Save overall metrics
    save_metrics_report(metrics, output_dir / "metrics.json")

    # Plot score distribution
    plot_score_distribution(
        df,
        output_dir / "score_distribution.png",
        title=f"Score Distribution ({args.split})",
    )

    # Per-domain breakdown
    codec_metrics = None
    codec_q_metrics = None

    if args.per_domain:
        tables_dir = output_dir / "tables"
        table_paths = save_domain_tables(df, tables_dir)

        for domain, path in table_paths.items():
            logger.info(f"Saved {domain} table: {path}")

            # Log summary
            domain_df = pd.read_csv(path)
            logger.info(f"\n{domain.upper()} breakdown:")
            logger.info(domain_df.to_string(index=False))

        # Compute per-domain metrics for wandb (use normalized string labels)
        from asvspoof5_domain_invariant_cm.evaluation.metrics import (
            compute_eer,
            compute_auc,
            compute_threshold_metrics,
        )

        # Per-CODEC metrics
        codec_metrics = {}
        for codec_name in df["codec"].unique():
            mask = df["codec"] == codec_name
            if mask.sum() > 10:
                codec_scores = df.loc[mask, "score"].values
                codec_labels = df.loc[mask, "y_task"].values
                if len(np.unique(codec_labels)) == 2:
                    codec_eer, codec_eer_thresh = compute_eer(codec_scores, codec_labels)
                    codec_auc = compute_auc(codec_scores, codec_labels)
                    codec_thresh_metrics = compute_threshold_metrics(
                        codec_scores, codec_labels, codec_eer_thresh
                    )
                    codec_metrics[codec_name] = {
                        "eer": float(codec_eer),
                        "auc": float(codec_auc),
                        "f1_macro": codec_thresh_metrics["f1_macro"],
                        "n_samples": int(mask.sum()),
                        "n_bonafide": int((codec_labels == 0).sum()),
                        "n_spoof": int((codec_labels == 1).sum()),
                    }

        # Plot EER by CODEC
        if codec_metrics:
            plot_per_domain_eer(
                codec_metrics,
                output_dir / "eer_by_codec.png",
                "CODEC",
            )

        # Per-CODEC_Q metrics
        codec_q_metrics = {}
        for codec_q_name in df["codec_q"].unique():
            mask = df["codec_q"] == codec_q_name
            if mask.sum() > 10:
                cq_scores = df.loc[mask, "score"].values
                cq_labels = df.loc[mask, "y_task"].values
                if len(np.unique(cq_labels)) == 2:
                    cq_eer, cq_eer_thresh = compute_eer(cq_scores, cq_labels)
                    cq_auc = compute_auc(cq_scores, cq_labels)
                    cq_thresh_metrics = compute_threshold_metrics(
                        cq_scores, cq_labels, cq_eer_thresh
                    )
                    codec_q_metrics[codec_q_name] = {
                        "eer": float(cq_eer),
                        "auc": float(cq_auc),
                        "f1_macro": cq_thresh_metrics["f1_macro"],
                        "n_samples": int(mask.sum()),
                        "n_bonafide": int((cq_labels == 0).sum()),
                        "n_spoof": int((cq_labels == 1).sum()),
                    }

        # Plot EER by CODEC_Q
        if codec_q_metrics:
            plot_per_domain_eer(
                codec_q_metrics,
                output_dir / "eer_by_codec_q.png",
                "CODEC_Q",
            )

    logger.info(f"\nResults saved to: {output_dir}")

    # Build evaluation complete wide event
    eval_event = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if codec_metrics:
        eval_event["n_codecs_evaluated"] = len(codec_metrics)
    if codec_q_metrics:
        eval_event["n_codec_qs_evaluated"] = len(codec_q_metrics)

    # Save evaluation wide event
    with open(output_dir / "evaluation_event.json", "w") as f:
        json.dump(eval_event, f, indent=2, default=str)

    # Wandb logging
    if args.wandb:
        log_to_wandb(
            args,
            metrics,
            df,
            output_dir,
            config,
            codec_metrics=codec_metrics,
            codec_q_metrics=codec_q_metrics,
        )

    return 0


if __name__ == "__main__":
    exit(main())
