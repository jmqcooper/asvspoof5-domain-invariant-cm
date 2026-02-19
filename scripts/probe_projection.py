#!/usr/bin/env python3
"""Probe projection layer outputs to measure domain invariance.

This script is the key analysis for RQ3: "Does DANN reduce domain information
in learned representations?"

Unlike backbone probing (which gives identical results for ERM and DANN since
the backbone is frozen), projection layer probing reveals the effect of DANN's
adversarial training on the learned representations.

**Why this matters:**
- Backbone layers (0-11) are frozen → ERM and DANN produce identical features
- The projection head is trainable → DANN's gradient reversal affects it
- Lower codec probe accuracy on projection output = more domain-invariant

**Expected results:**
- ERM: Projection output retains codec information (higher probe accuracy)
- DANN: Projection output is domain-invariant (lower probe accuracy)
- The difference quantifies DANN's effectiveness for RQ3

Usage:
    # Single model analysis
    python scripts/probe_projection.py \
        --checkpoint runs/dann_wavlm/checkpoints/best.pt \
        --output results/dann_projection_probe.json

    # Compare ERM vs DANN (recommended for RQ3)
    python scripts/probe_projection.py \
        --erm-checkpoint runs/erm_wavlm/checkpoints/best.pt \
        --dann-checkpoint runs/dann_wavlm/checkpoints/best.pt \
        --output results/rq3_projection_comparison.json

    # With wandb logging
    python scripts/probe_projection.py \
        --erm-checkpoint ... --dann-checkpoint ... \
        --wandb --wandb-project asvspoof5-dann
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import project modules
from asvspoof5_domain_invariant_cm.data import ASVspoof5Dataset, AudioCollator, load_vocab
from asvspoof5_domain_invariant_cm.data.codec_augment import create_augmentor
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)
from asvspoof5_domain_invariant_cm.utils import get_device, get_manifest_path

# Optional wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe projection layer output for domain invariance (RQ3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to single model checkpoint",
    )
    p.add_argument(
        "--erm-checkpoint",
        type=Path,
        default=None,
        help="Path to ERM model checkpoint (for comparison)",
    )
    p.add_argument(
        "--dann-checkpoint",
        type=Path,
        default=None,
        help="Path to DANN model checkpoint (for comparison)",
    )

    # Data options
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (uses default manifest paths if not set)",
    )
    p.add_argument(
        "--split",
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split to use (default: dev)",
    )
    p.add_argument(
        "--domain-source",
        choices=["protocol", "synthetic", "auto"],
        default="auto",
        help="Domain label source: protocol=manifest labels, synthetic=augmentation (default: auto)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to use (default: 5000)",
    )

    # Probe options
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    p.add_argument(
        "--probe-target",
        choices=["codec", "codec_q", "both"],
        default="codec",
        help="Domain to probe for (default: codec)",
    )

    # Output options
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot",
    )

    # Runtime options
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction (default: 32)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Wandb options
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Wandb",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-dann",
        help="Wandb project name",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity",
    )

    return p.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict, dict, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Target device.

    Returns:
        Tuple of (model, config, codec_vocab, codec_q_vocab).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Load vocabs from run directory
    run_dir = checkpoint_path.parent.parent
    codec_vocab = load_vocab(run_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(run_dir / "codec_q_vocab.json")

    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)

    # Extract config sections
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    # Build backbone
    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    # Build pooling
    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    if pooling_method == "stats":
        proj_input_dim = backbone.hidden_size * 2
    else:
        proj_input_dim = backbone.hidden_size

    # Build projection head
    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)

    # Build task head
    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    # Build model
    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        disc_input_dim = disc_cfg.get("input_dim", proj_input_dim)
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
            lambda_=0.0,
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, codec_vocab, codec_q_vocab


@torch.no_grad()
def extract_projection_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int,
    domain_source: str = "protocol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract projection layer outputs and domain labels.

    Args:
        model: Model to extract from.
        dataloader: DataLoader.
        device: Device.
        max_samples: Maximum samples to extract.
        domain_source: Label source ('protocol' or 'synthetic').

    Returns:
        Tuple of (projection_outputs, codec_labels, codec_q_labels).
    """
    all_repr = []
    all_codec = []
    all_codec_q = []
    n_samples = 0

    for batch in tqdm(dataloader, desc="Extracting projection outputs"):
        if n_samples >= max_samples:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        # Get projection output (repr)
        all_repr.append(outputs["repr"].cpu().numpy())

        # Get domain labels
        if domain_source == "synthetic":
            all_codec.append(batch["y_codec_aug"].numpy())
            all_codec_q.append(batch["y_codec_q_aug"].numpy())
        else:
            all_codec.append(batch["y_codec"].numpy())
            all_codec_q.append(batch["y_codec_q"].numpy())

        n_samples += waveform.shape[0]

    # Concatenate and truncate
    repr_array = np.concatenate(all_repr, axis=0)[:max_samples]
    codec_array = np.concatenate(all_codec)[:max_samples]
    codec_q_array = np.concatenate(all_codec_q)[:max_samples]

    return repr_array, codec_array, codec_q_array


def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Train a logistic regression probe to predict domain from embeddings.

    Args:
        embeddings: Feature embeddings of shape (N, D).
        labels: Domain labels of shape (N,).
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Dictionary with probe results.
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)

    if n_classes < 2:
        return {
            "status": "skipped",
            "skip_reason": "only_one_class",
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "cv_scores": [],
            "n_samples": len(embeddings),
            "n_classes": int(n_classes),
            "class_distribution": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
        }

    # Ensure we have enough samples per class for CV
    min_class_count = int(label_counts.min())
    cv_folds_used = min(cv_folds, min_class_count)

    if cv_folds_used < 2:
        return {
            "status": "skipped",
            "skip_reason": "insufficient_samples_per_class",
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "cv_scores": [],
            "n_samples": len(embeddings),
            "n_classes": int(n_classes),
            "class_distribution": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
        }

    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds_used, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(embeddings_scaled, labels):
        clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        clf.fit(embeddings_scaled[train_idx], labels[train_idx])
        score = clf.score(embeddings_scaled[test_idx], labels[test_idx])
        scores.append(score)

    return {
        "status": "ok",
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores],
        "n_samples": len(embeddings),
        "n_classes": int(n_classes),
        "cv_folds_used": cv_folds_used,
        "class_distribution": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
    }


def plot_comparison(
    erm_result: dict,
    dann_result: dict,
    output_path: Path,
    domain: str,
) -> None:
    """Create a bar chart comparing ERM vs DANN projection probe accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ["ERM", "DANN"]
    accuracies = [erm_result["accuracy"], dann_result["accuracy"]]
    stds = [erm_result["accuracy_std"], dann_result["accuracy_std"]]

    colors = ["#E74C3C", "#3498DB"]  # Red for ERM, Blue for DANN
    bars = ax.bar(models, accuracies, yerr=stds, capsize=5, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add chance level line
    n_classes = max(erm_result.get("n_classes", 2), dann_result.get("n_classes", 2))
    chance = 1.0 / n_classes
    ax.axhline(y=chance, color="gray", linestyle="--", alpha=0.7, label=f"Chance ({chance:.3f})")

    ax.set_ylabel(f"{domain.upper()} Probe Accuracy", fontsize=12)
    ax.set_title(
        f"Projection Layer Domain Invariance (RQ3)\n"
        f"Lower DANN accuracy = more domain-invariant",
        fontsize=13,
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add reduction annotation
    reduction = erm_result["accuracy"] - dann_result["accuracy"]
    relative_reduction = reduction / max(erm_result["accuracy"], 1e-6) * 100
    ax.text(
        0.5,
        0.02,
        f"Reduction: {reduction:.3f} ({relative_reduction:.1f}% relative)",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")


def main() -> int:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Determine mode
    if args.erm_checkpoint and args.dann_checkpoint:
        mode = "comparison"
        checkpoints = {"erm": args.erm_checkpoint, "dann": args.dann_checkpoint}
    elif args.checkpoint:
        mode = "single"
        checkpoints = {"model": args.checkpoint}
    else:
        logger.error(
            "Provide --checkpoint for single model or both "
            "--erm-checkpoint and --dann-checkpoint for comparison"
        )
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        first_ckpt = list(checkpoints.values())[0]
        output_dir = first_ckpt.parent.parent / "projection_probe"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"projection_probe_{args.split}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine domains to probe
    if args.probe_target == "both":
        domains = ["codec", "codec_q"]
    else:
        domains = [args.probe_target]

    logger.info("=" * 60)
    logger.info("Projection Layer Domain Probe (RQ3 Analysis)")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Domains: {domains}")

    # Load first model to get config
    first_ckpt = list(checkpoints.values())[0]
    _, config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(first_ckpt, device)

    # Resolve domain source
    if args.domain_source == "auto":
        domain_source = "protocol" if args.split == "eval" else "synthetic"
    else:
        domain_source = args.domain_source

    logger.info(f"Domain source: {domain_source}")

    if domain_source == "protocol" and args.split != "eval":
        logger.warning(
            "Protocol labels on train/dev are often single-class. "
            "Consider --domain-source synthetic."
        )

    # Create dataset
    audio_cfg = config.get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)
    augmentation_cfg = config.get("augmentation", {})

    if domain_source == "synthetic":
        aug_config = dict(augmentation_cfg)
        aug_config["sample_rate"] = sample_rate
        augmentor = create_augmentor(aug_config)
        if augmentor is None:
            logger.error(
                "Synthetic probing requested but augmentation is disabled. "
                "Use --domain-source protocol or enable augmentation."
            )
            return 1
        dataset = ASVspoof5Dataset(
            manifest_path=get_manifest_path(args.split),
            codec_vocab=codec_vocab,
            codec_q_vocab=codec_q_vocab,
            max_duration_sec=max_duration,
            sample_rate=sample_rate,
            mode="train",
            augmentor=augmentor,
            use_synthetic_labels=True,
        )
    else:
        dataset = ASVspoof5Dataset(
            manifest_path=get_manifest_path(args.split),
            codec_vocab=codec_vocab,
            codec_q_vocab=codec_q_vocab,
            max_duration_sec=max_duration,
            sample_rate=sample_rate,
            mode="eval",
        )

    # Subsample if needed
    if args.n_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    logger.info(f"Dataset size: {len(dataset)}")

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

    # Process each model
    all_results = {}

    for model_name, ckpt_path in checkpoints.items():
        logger.info(f"\n--- Processing {model_name.upper()}: {ckpt_path} ---")

        model, _, _, _ = load_model_from_checkpoint(ckpt_path, device)

        repr_outputs, codec_labels, codec_q_labels = extract_projection_outputs(
            model, dataloader, device, args.n_samples, domain_source
        )

        logger.info(f"  Extracted {len(repr_outputs)} projection outputs (dim={repr_outputs.shape[1]})")

        # Log label distribution
        for name, labels in [("codec", codec_labels), ("codec_q", codec_q_labels)]:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(unique, counts)}
            logger.info(f"  {name} distribution: {dist}")

        model_results = {}

        for domain in domains:
            labels = codec_labels if domain == "codec" else codec_q_labels

            result = train_probe(repr_outputs, labels, cv_folds=args.cv_folds, seed=args.seed)

            model_results[domain] = result

            if result["status"] == "ok":
                logger.info(
                    f"  {domain.upper()} probe: {result['accuracy']:.4f} ± {result['accuracy_std']:.4f}"
                )
            else:
                logger.warning(f"  {domain.upper()} probe skipped: {result['skip_reason']}")

        all_results[model_name] = model_results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute comparison metrics if in comparison mode
    comparison = {}
    if mode == "comparison":
        for domain in domains:
            erm_result = all_results["erm"][domain]
            dann_result = all_results["dann"][domain]

            if erm_result["status"] == "ok" and dann_result["status"] == "ok":
                reduction = erm_result["accuracy"] - dann_result["accuracy"]
                relative_reduction = reduction / max(erm_result["accuracy"], 1e-6)

                comparison[domain] = {
                    "erm_accuracy": erm_result["accuracy"],
                    "dann_accuracy": dann_result["accuracy"],
                    "reduction": float(reduction),
                    "relative_reduction": float(relative_reduction),
                    "dann_is_more_invariant": dann_result["accuracy"] < erm_result["accuracy"],
                }

                logger.info(f"\n{domain.upper()} Comparison:")
                logger.info(f"  ERM accuracy:  {erm_result['accuracy']:.4f}")
                logger.info(f"  DANN accuracy: {dann_result['accuracy']:.4f}")
                logger.info(f"  Reduction:     {reduction:.4f} ({relative_reduction * 100:.1f}%)")
                logger.info(f"  DANN more invariant: {comparison[domain]['dann_is_more_invariant']}")

        # Generate plot if requested
        if args.plot:
            for domain in domains:
                if domain in comparison:
                    plot_path = output_path.with_suffix("").with_name(
                        f"{output_path.stem}_{domain}_comparison.png"
                    )
                    plot_comparison(
                        all_results["erm"][domain],
                        all_results["dann"][domain],
                        plot_path,
                        domain,
                    )

    # Build output
    output = {
        "analysis": "projection_layer_probe",
        "description": "Probe projection layer outputs for domain invariance (RQ3)",
        "split": args.split,
        "domain_source": domain_source,
        "n_samples": len(dataset),
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        "results": all_results,
    }

    if comparison:
        output["comparison"] = comparison
        output["rq3_summary"] = {
            "question": "Does DANN reduce domain information in learned representations?",
            "answer": all(c.get("dann_is_more_invariant", False) for c in comparison.values()),
            "evidence": comparison,
        }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved: {output_path}")

    # Wandb logging
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Wandb not available")
        else:
            try:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=f"projection_probe_{args.split}",
                    job_type="analysis",
                )

                for model_name, model_results in all_results.items():
                    for domain, result in model_results.items():
                        if result["status"] == "ok":
                            wandb.log({
                                f"projection_probe/{model_name}/{domain}": result["accuracy"]
                            })

                if comparison:
                    for domain, comp in comparison.items():
                        wandb.log({
                            f"projection_probe/comparison/{domain}/reduction": comp["reduction"],
                            f"projection_probe/comparison/{domain}/relative_reduction": comp["relative_reduction"],
                        })

                # Log plots
                for png in output_path.parent.glob("*.png"):
                    wandb.log({f"plots/{png.stem}": wandb.Image(str(png))})

                wandb.finish()
                logger.info("Logged to Wandb")

            except Exception as e:
                logger.warning(f"Wandb logging failed: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
