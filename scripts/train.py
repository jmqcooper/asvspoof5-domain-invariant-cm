#!/usr/bin/env python3
"""Training entrypoint for ERM and DANN models.

Usage:
    # ERM with WavLM
    python scripts/train.py --config configs/wavlm_erm.yaml

    # DANN with WavLM
    python scripts/train.py --config configs/wavlm_dann.yaml

    # With separate config files
    python scripts/train.py \
        --train-config configs/train/erm.yaml \
        --model-config configs/model/wavlm_base.yaml \
        --data-config configs/data/asvspoof5_track1.yaml

    # Override run name
    python scripts/train.py --config configs/wavlm_dann.yaml --name my_experiment

    # With wandb logging
    python scripts/train.py --config configs/wavlm_dann.yaml --wandb
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch

from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset,
    AudioCollator,
    SYNTHETIC_CODEC_VOCAB,
    SYNTHETIC_QUALITY_VOCAB,
    create_augmentor,
    load_vocab,
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
from asvspoof5_domain_invariant_cm.training import (
    LambdaScheduler,
    Trainer,
    build_loss,
    build_lr_scheduler,
    build_optimizer,
)
from asvspoof5_domain_invariant_cm.utils import (
    get_aug_cache_dir,
    get_device,
    get_experiment_context,
    get_git_info,
    get_manifest_path,
    get_manifests_dir,
    get_run_dir,
    load_config,
    merge_configs,
    set_seed,
    setup_logging,
)

# Setup basic logging first (will be reconfigured per-run)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Seed random for reproducible augmentation in DataLoader workers.
    
    Seeds Python random, NumPy, and torch for full reproducibility.
    Must be at module level for multiprocessing pickling.
    """
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
    # Also seed torch for any torch.randint calls (e.g., random crop in audio.py)
    torch.manual_seed(worker_seed + worker_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to combined config (contains all settings)",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Path to training config (erm.yaml or dann.yaml)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to model config (wavlm_base.yaml, w2v2_base.yaml, etc.)",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/asvspoof5_track1.yaml"),
        help="Path to data config",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for model optimization (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Wandb logging",
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
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated wandb tags",
    )
    return parser.parse_args()


def load_configs(args) -> dict:
    """Load and merge configuration files."""
    if args.config is not None:
        # Single combined config
        config = load_config(args.config)
    else:
        # Separate config files
        configs = []

        if args.data_config and args.data_config.exists():
            configs.append(load_config(args.data_config))

        if args.model_config and args.model_config.exists():
            configs.append(load_config(args.model_config))

        if args.train_config and args.train_config.exists():
            configs.append(load_config(args.train_config))

        if not configs:
            raise ValueError(
                "No config files provided. Use --config or --train-config + --model-config"
            )

        config = merge_configs(*configs)

    # Override with CLI args
    if args.seed is not None:
        config["seed"] = args.seed

    return config


def build_model(config: dict, num_codecs: int, num_codec_qs: int) -> torch.nn.Module:
    """Build model from config."""
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    # Backbone
    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=backbone_cfg.get("freeze", True),
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    # Pooling
    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    # Calculate projection input dim
    if pooling_method == "stats":
        proj_input_dim = backbone.hidden_size * 2
    else:
        proj_input_dim = backbone.hidden_size

    # Projection head
    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)

    # Task head
    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    # Build model based on method
    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        # Discriminator taps pre-projection pooled features (e.g. 1536-dim stats pooling)
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
            lambda_=dann_cfg.get("lambda_", 0.1),
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    return model


def get_dataset_distribution(dataset, codec_vocab: dict, codec_q_vocab: dict) -> dict:
    """Get class and domain distribution from dataset."""
    import pandas as pd

    # Try to get manifest for stats
    if hasattr(dataset, "manifest"):
        df = dataset.manifest
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "manifest"):
        df = dataset.dataset.manifest
    else:
        return {}

    distribution = {}

    # Class distribution
    if "y_task" in df.columns:
        class_counts = df["y_task"].value_counts().to_dict()
        distribution["class"] = {
            "bonafide": class_counts.get(0, 0),
            "spoof": class_counts.get(1, 0),
        }
    elif "key" in df.columns:
        class_counts = df["key"].value_counts().to_dict()
        distribution["class"] = class_counts

    # CODEC distribution (top 10)
    if "codec" in df.columns:
        codec_counts = df["codec"].value_counts().head(10).to_dict()
        distribution["codec_top10"] = codec_counts

    # CODEC_Q distribution (top 10)
    if "codec_q" in df.columns:
        codec_q_counts = df["codec_q"].value_counts().head(10).to_dict()
        distribution["codec_q_top10"] = codec_q_counts

    return distribution


def main():
    args = parse_args()

    # Load config
    config = load_configs(args)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=config.get("deterministic", True))

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Performance optimizations for CUDA
    if device.type == "cuda":
        # Enable TF32 for matmul (8x faster on A100, slight precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto-tune convolution algorithms for fixed input sizes
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimizations: TF32=True, cudnn.benchmark=True")

    # Setup run directory with git commit hash for traceability
    # Format: {name}_{commit_hash} e.g., wavlm_dann_cd52b83
    # Falls back to timestamp if git unavailable to prevent overwrites
    git_info = get_git_info()
    if git_info.get("commit"):
        run_suffix = git_info["commit"][:7]
    else:
        run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.name:
        base_name = args.name
    else:
        method = config.get("training", {}).get("method", "erm")
        backbone_name = config.get("backbone", {}).get("name", "wavlm")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{backbone_name}_{method}_{timestamp}"
    
    run_name = f"{base_name}_{run_suffix}"
    run_dir = get_run_dir(run_name)

    # Reconfigure logging with JSON output to run directory
    setup_logging(run_dir, json_output=True)
    logger.info(f"Run directory: {run_dir}")

    # Log experiment context
    context = get_experiment_context(config)
    logger.info(f"Git commit: {context['git'].get('commit', 'N/A')[:8]}")
    logger.info(f"Hardware: {context['hardware'].get('gpu_name', 'CPU')}")

    # Load vocabularies
    manifests_dir = get_manifests_dir()
    codec_vocab = load_vocab(manifests_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(manifests_dir / "codec_q_vocab.json")

    logger.info(f"CODEC classes (manifest): {len(codec_vocab)}")
    logger.info(f"CODEC_Q classes (manifest): {len(codec_q_vocab)}")

    # Note: vocab files are saved to run_dir AFTER augmentor is created,
    # so we know whether to use synthetic or manifest vocabs

    # Data config
    data_cfg = config.get("dataset", config.get("data", {}))
    audio_cfg = config.get("audio", {})
    dataloader_cfg = config.get("dataloader", {})
    training_cfg = config.get("training", {})

    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)
    batch_size = dataloader_cfg.get("batch_size", 32)
    num_workers = dataloader_cfg.get("num_workers", 4)

    # Create codec augmentor for DANN
    method = training_cfg.get("method", "erm")
    aug_cfg = config.get("augmentation", {})
    augmentor = None

    # Codec augmentation can be used as a control even for ERM:
    #   A) ERM (no aug)
    #   B) ERM (codec aug)
    #   C) DANN (codec aug, adversarial λ>0)
    if aug_cfg.get("enabled", False):
        # Resolve cache_dir: config > env var > None (on-the-fly)
        if not aug_cfg.get("cache_dir"):
            env_cache_dir = get_aug_cache_dir()
            if env_cache_dir:
                logger.info(f"Using AUGMENTATION_CACHE_DIR from environment: {env_cache_dir}")
                aug_cfg["cache_dir"] = str(env_cache_dir)
        
        # Inject sample_rate into augmentation config
        aug_cfg_with_sr = {**aug_cfg, "sample_rate": sample_rate}
        augmentor = create_augmentor(aug_cfg_with_sr)

        if augmentor is not None:
            logger.info(
                f"Codec augmentor initialized: supported_codecs={augmentor.supported_codecs}, "
                f"codec_prob={aug_cfg.get('codec_prob', 0.5)}"
            )
            # Critical: DANN requires domain diversity
            if method == "dann" and len(augmentor.supported_codecs) < 2:
                raise RuntimeError(
                    f"DANN requires >=2 supported codecs, got {len(augmentor.supported_codecs)}. "
                    f"Supported: {augmentor.supported_codecs}. "
                    "Check ffmpeg encoder support: ffmpeg -encoders | grep -E 'mp3|aac|opus'"
                )
        else:
            logger.warning(
                "Augmentation enabled in config but augmentor is None - "
                "codec augmentation will be disabled for this run."
            )

    # Create datasets
    train_dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path("train"),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="train",
        augmentor=augmentor,
        # Expose y_codec_aug/y_codec_q_aug when augmentation is enabled so we can:
        # - assert domain diversity early for DANN
        # - log augmentation rate for ERM+aug control runs
        use_synthetic_labels=augmentor is not None and aug_cfg.get("use_synthetic_labels", True),
    )

    val_dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path("dev"),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Get dataset distribution for logging
    train_distribution = get_dataset_distribution(train_dataset, codec_vocab, codec_q_vocab)
    if train_distribution:
        logger.info(f"Train class distribution: {train_distribution.get('class', {})}")

    # Create dataloaders
    fixed_length = int(max_duration * sample_rate)

    train_collator = AudioCollator(fixed_length=fixed_length, mode="train")
    val_collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    # DataLoader performance settings
    prefetch_factor = dataloader_cfg.get("prefetch_factor", 2)
    persistent_workers = dataloader_cfg.get("persistent_workers", False) and num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collator,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )

    # Build model vocab sizes and save vocabs.
    #
    # IMPORTANT: for DANN+augmentation, the discriminator head sizes MUST match the
    # synthetic vocab used to generate y_codec_aug/y_codec_q_aug.
    #
    # For ERM(+augmentation), evaluation should remain interpretable wrt real ASVspoof5
    # codec labels, so we keep the manifest vocabs as the primary run vocabs.
    import json
    if method == "dann" and augmentor is not None:
        num_codecs = len(augmentor.codec_vocab)  # 1 + len(supported_codecs)
        num_codec_qs = len(SYNTHETIC_QUALITY_VOCAB)
        logger.info(
            "DANN with augmentation: domain discriminator sizes "
            f"num_codecs={num_codecs}, num_codec_qs={num_codec_qs}"
        )
        # Save synthetic vocabs so evaluate.py can rebuild model with correct head sizes
        with open(run_dir / "codec_vocab.json", "w") as f:
            json.dump(augmentor.codec_vocab, f, indent=2)
        with open(run_dir / "codec_q_vocab.json", "w") as f:
            json.dump(SYNTHETIC_QUALITY_VOCAB, f, indent=2)
        # Also save manifest vocabs for reporting on eval (12/9 classes)
        shutil.copy(manifests_dir / "codec_vocab.json", run_dir / "manifest_codec_vocab.json")
        shutil.copy(manifests_dir / "codec_q_vocab.json", run_dir / "manifest_codec_q_vocab.json")
        logger.info("Saved synthetic vocabs to run dir (DANN with augmentation)")
    else:
        num_codecs = len(codec_vocab)
        num_codec_qs = len(codec_q_vocab)
        # Save manifest vocabs for ERM or DANN without augmentation
        shutil.copy(manifests_dir / "codec_vocab.json", run_dir / "codec_vocab.json")
        shutil.copy(manifests_dir / "codec_q_vocab.json", run_dir / "codec_q_vocab.json")
        # If augmentation is enabled as a control, also save the synthetic vocab used
        # to generate y_codec_aug/y_codec_q_aug for transparency/debugging.
        if augmentor is not None:
            with open(run_dir / "synthetic_codec_vocab.json", "w") as f:
                json.dump(augmentor.codec_vocab, f, indent=2)
            with open(run_dir / "synthetic_codec_q_vocab.json", "w") as f:
                json.dump(SYNTHETIC_QUALITY_VOCAB, f, indent=2)

    model = build_model(config, num_codecs, num_codec_qs)
    model = model.to(device)

    # Optional: torch.compile for PyTorch 2.0+ speedup
    if args.compile:
        logger.info("Compiling model with torch.compile (this may take a few minutes)...")
        model = torch.compile(model)
        logger.info("Model compiled successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler configs (training_cfg and method already defined above)
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})

    # Build optimizer
    optimizer = build_optimizer(
        model,
        name=optimizer_cfg.get("name", "adamw"),
        lr=optimizer_cfg.get("lr", 1e-4),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
    )

    # Build scheduler
    num_training_steps = len(train_loader) * training_cfg.get("max_epochs", 50)
    scheduler = build_lr_scheduler(
        optimizer,
        name=scheduler_cfg.get("name", "cosine"),
        num_warmup_steps=scheduler_cfg.get("warmup_steps", 500),
        num_training_steps=num_training_steps,
        min_lr_ratio=scheduler_cfg.get("min_lr", 1e-6) / optimizer_cfg.get("lr", 1e-4),
    )

    # Build loss
    loss_cfg = config.get("loss", {})
    task_loss_cfg = loss_cfg.get("task", {})
    dann_cfg = config.get("dann", {})

    loss_fn = build_loss(
        method=method,
        task_label_smoothing=task_loss_cfg.get("label_smoothing", 0.0),
        lambda_domain=dann_cfg.get("lambda_", 0.1) if method == "dann" else 0.0,
    )

    # Build lambda scheduler (for DANN)
    lambda_scheduler = None
    if method == "dann":
        lambda_sched_cfg = dann_cfg.get("lambda_schedule", {})
        if lambda_sched_cfg.get("enabled", False):
            lambda_scheduler = LambdaScheduler(
                schedule_type=lambda_sched_cfg.get("type", "linear"),
                start_value=lambda_sched_cfg.get("start", 0.0),
                end_value=lambda_sched_cfg.get("end", 1.0),
                warmup_epochs=lambda_sched_cfg.get("warmup_epochs", 5),
                total_epochs=training_cfg.get("max_epochs", 50),
            )

    # Logging config
    logging_cfg = config.get("logging", {})
    wandb_cfg = config.get("wandb", {})

    # Wandb is enabled by default (can be disabled via config or --no-wandb)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    use_wandb = args.wandb or wandb_cfg.get("enabled", True)
    if use_wandb and not wandb_api_key:
        logger.warning(
            "Wandb enabled but WANDB_API_KEY not set. "
            "Set it in .env or export before running. Disabling wandb."
        )
        use_wandb = False

    # Parse wandb tags
    wandb_tags = None
    if args.wandb_tags:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(",")]
    elif wandb_cfg.get("tags"):
        wandb_tags = wandb_cfg.get("tags")
    else:
        # Generate tags from config
        wandb_tags = [
            config.get("backbone", {}).get("name", "unknown"),
            method,
            "track1",
        ]

    # Create trainer with enhanced logging
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        config=config,
        method=method,
        max_epochs=training_cfg.get("max_epochs", 50),
        patience=training_cfg.get("patience", 10),
        min_delta=training_cfg.get("min_delta", 0.001),
        train_loss_threshold=training_cfg.get("train_loss_threshold", 0.01),
        plateau_patience=training_cfg.get("plateau_patience", 3),
        gradient_clip=training_cfg.get("gradient_clip", 1.0),
        use_amp=args.amp,
        log_interval=logging_cfg.get("log_every_n_steps", 50),
        val_interval=logging_cfg.get("val_every_n_epochs", 1),
        save_every_n_epochs=training_cfg.get("save_every_n_epochs", 5),
        monitor_metric=training_cfg.get("monitor_metric", "eer"),
        monitor_mode=training_cfg.get("monitor_mode", "min"),
        lambda_scheduler=lambda_scheduler,
        # Enhanced logging options
        use_wandb=use_wandb,
        wandb_project=args.wandb_project or wandb_cfg.get("project", "asvspoof5-dann"),
        wandb_entity=args.wandb_entity or wandb_cfg.get("entity"),
        wandb_run_name=run_name,
        wandb_tags=wandb_tags,
        batch_sample_rate=logging_cfg.get("log_batch_samples", 0.02),
        track_gradients=logging_cfg.get("track_gradients", True),
        log_domain_breakdown_every=logging_cfg.get("log_domain_breakdown_every", 5),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        nan_grad_abort_count=training_cfg.get("nan_grad_abort_count"),
    )

    # Train
    logger.info("=" * 60)
    logger.info(f"Training {method.upper()} model")
    logger.info(f"Backbone: {config.get('backbone', {}).get('name', 'unknown')}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 60)

    final_metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete!")
    monitor_metric = training_cfg.get('monitor_metric', 'eer')
    best_metric_key = f'best_{monitor_metric}'
    best_value = final_metrics.get(best_metric_key, 'N/A')
    logger.info(f"Best {monitor_metric}: {best_value}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
