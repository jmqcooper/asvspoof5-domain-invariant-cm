"""Training and validation loops with comprehensive logging."""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.io import save_checkpoint, save_config, save_metrics
from ..utils.logging import (
    ExperimentLogger,
    compute_grad_norm,
    get_non_finite_grad_parameter_names,
    get_gpu_memory_usage,
)

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    method: str = "erm",
    log_interval: int = 50,
    batch_sample_rate: float = 0.0,
    track_gradients: bool = True,
    exp_logger: Optional[ExperimentLogger] = None,
    global_step_start: int = 0,
    log_step_interval: int = 100,
    nan_grad_abort_count: Optional[int] = None,
) -> tuple[dict, int]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device.
        scheduler: Optional LR scheduler.
        scaler: Optional gradient scaler for AMP.
        gradient_clip: Gradient clipping value.
        method: Training method ('erm' or 'dann').
        log_interval: Logging interval for progress bar.
        batch_sample_rate: Rate of batches to log detailed info (0.0-1.0).
        track_gradients: Whether to track gradient norms.
        exp_logger: Optional ExperimentLogger for step-level wandb logging.
        global_step_start: Starting global step for this epoch.
        log_step_interval: Interval for step-level wandb logging.

    Returns:
        Tuple of (metrics dict, final global step).
    """
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_codec_loss = 0.0
    total_codec_q_loss = 0.0
    total_task_acc = 0.0
    total_codec_acc = 0.0
    total_codec_q_acc = 0.0
    num_batches = 0

    # Gradient tracking
    grad_norms = []
    grad_clips_count = 0
    nan_grad_count = 0

    # Augmentation rate tracking (for DANN and ERM+aug controls)
    total_aug_samples = 0
    total_non_none_samples = 0
    saw_aug_labels = False

    # Batch-level logging samples
    batch_samples = []

    # Global step tracking
    global_step = global_step_start

    # Fail-fast domain diversity tracking for DANN (across first N batches)
    dann_domain_check_batches = 200
    seen_codec_ids_first_n_batches: set[int] = set()

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch_start_time = time.time()

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"].to(device)

        # Pre-forward sanity checks to catch the most common NaN sources early.
        # (e.g., 0-length waveforms -> empty attention -> backbone NaNs)
        if lengths.numel() > 0:
            min_len = int(lengths.min().item())
            if min_len <= 0:
                metadata = batch.get("metadata", {})
                bad_idx = (lengths <= 0).nonzero(as_tuple=False).view(-1).tolist()
                bad_samples = []
                for i in bad_idx[:5]:
                    bad_samples.append(
                        {
                            "flac_file": (metadata.get("flac_file") or [None] * len(bad_idx))[i],
                            "codec_seed": (metadata.get("codec_seed") or [None] * len(bad_idx))[i],
                            "speaker_id": (metadata.get("speaker_id") or [None] * len(bad_idx))[i],
                            "length": int(lengths[i].item()),
                        }
                    )
                logger.warning(
                    f"Skipping batch {batch_idx} (global_step={global_step}): "
                    f"non-positive lengths detected (min_len={min_len}). "
                    f"bad_samples={bad_samples}"
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

        if not torch.isfinite(waveform).all().item():
            metadata = batch.get("metadata", {})
            bad_samples = []
            for i in range(min(waveform.shape[0], 5)):
                bad_samples.append(
                    {
                        "flac_file": (metadata.get("flac_file") or [None])[i],
                        "codec_seed": (metadata.get("codec_seed") or [None])[i],
                        "speaker_id": (metadata.get("speaker_id") or [None])[i],
                    }
                )
            logger.warning(
                f"Skipping batch {batch_idx} (global_step={global_step}): "
                "non-finite waveform values detected. "
                f"waveform_stats={{'min': {float(waveform.min().item()):.6g}, "
                f"'max': {float(waveform.max().item()):.6g}, "
                f"'mean': {float(waveform.mean().item()):.6g}}} "
                f"samples={bad_samples}"
            )
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue

        # Attention mask must have at least one True per sample.
        if attention_mask.numel() > 0:
            per_sample_valid = attention_mask.any(dim=1)
            if (~per_sample_valid).any().item():
                metadata = batch.get("metadata", {})
                bad_idx = (~per_sample_valid).nonzero(as_tuple=False).view(-1).tolist()
                bad_samples = []
                for i in bad_idx[:5]:
                    bad_samples.append(
                        {
                            "flac_file": (metadata.get("flac_file") or [None])[i],
                            "codec_seed": (metadata.get("codec_seed") or [None])[i],
                            "speaker_id": (metadata.get("speaker_id") or [None])[i],
                            "attention_true": int(attention_mask[i].sum().item()),
                            "length": int(lengths[i].item()) if lengths.numel() else None,
                        }
                    )
                logger.warning(
                    f"Skipping batch {batch_idx} (global_step={global_step}): "
                    "empty attention masks detected. "
                    f"bad_samples={bad_samples}"
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

        # Use augmented domain labels when available (for DANN with synthetic augmentation)
        if "y_codec_aug" in batch and batch["y_codec_aug"] is not None:
            y_codec = batch["y_codec_aug"].to(device)
            y_codec_q = batch["y_codec_q_aug"].to(device)
            saw_aug_labels = True
        else:
            y_codec = batch["y_codec"].to(device)
            y_codec_q = batch["y_codec_q"].to(device)

        optimizer.zero_grad()

        # Forward pass
        use_amp = scaler is not None
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(waveform, attention_mask, lengths)

            if method == "dann":
                # Fail-fast: DANN requires domain diversity in early training.
                # Requirement: >1 unique y_codec_aug class across the first N batches.
                if batch_idx < dann_domain_check_batches:
                    for v in y_codec.detach().cpu().unique().tolist():
                        seen_codec_ids_first_n_batches.add(int(v))
                    if batch_idx == dann_domain_check_batches - 1 and len(seen_codec_ids_first_n_batches) < 2:
                        raise RuntimeError(
                            "DANN requires domain diversity but the first "
                            f"{dann_domain_check_batches} batches only contained "
                            f"{len(seen_codec_ids_first_n_batches)} unique codec id(s): "
                            f"{sorted(seen_codec_ids_first_n_batches)}. "
                            "Check: augmentor wired? ffmpeg available? supported_codecs>=2? codec_prob>0?"
                        )

                losses = loss_fn(
                    outputs["task_logits"],
                    outputs["codec_logits"],
                    outputs["codec_q_logits"],
                    y_task,
                    y_codec,
                    y_codec_q,
                )
            else:
                losses = loss_fn(outputs["task_logits"], y_task)

        loss = losses["total_loss"]

        # Fast fail / diagnostics: skip batches with non-finite loss.
        # This happens in unstable regimes (too-large LR, AMP overflow, bad batch),
        # and letting it backpropagate will poison the optimizer state.
        if not torch.isfinite(loss).item():
            metadata = batch.get("metadata", {})
            sample_ids = []
            for i in range(min(waveform.shape[0], 5)):
                sample_ids.append(
                    {
                        "flac_file": (metadata.get("flac_file") or [None] * min(waveform.shape[0], 5))[i],
                        "codec_seed": (metadata.get("codec_seed") or [None] * min(waveform.shape[0], 5))[i],
                        "speaker_id": (metadata.get("speaker_id") or [None] * min(waveform.shape[0], 5))[i],
                    }
                )
            repr_stats = None
            if isinstance(outputs, dict) and "repr" in outputs:
                repr_tensor = outputs["repr"].detach().float()
                repr_stats = {
                    "min": float(repr_tensor.min().item()),
                    "max": float(repr_tensor.max().item()),
                    "mean": float(repr_tensor.mean().item()),
                }
            non_finite_loss_event = {
                "batch_idx": batch_idx,
                "global_step": global_step,
                "method": method,
                "sample_ids": sample_ids,
                "waveform_stats": {
                    "min": float(waveform.min().item()),
                    "max": float(waveform.max().item()),
                    "mean": float(waveform.mean().item()),
                },
                "repr_stats": repr_stats,
                "losses": {
                    k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                    for k, v in losses.items()
                },
            }
            logger.warning(
                f"Non-finite loss at batch {batch_idx} (global_step={global_step}). "
                "Skipping optimizer/scheduler step for this batch."
            )
            if exp_logger is not None:
                exp_logger.log_wide_event("non_finite_loss_batch", non_finite_loss_event)
            # Clear grads just in case and continue.
            optimizer.zero_grad(set_to_none=True)
            # Still advance global_step so `train/global_step` stays monotonic.
            global_step += 1
            continue

        # Backward pass
        optimizer_step_taken = False
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Track gradients before clipping
            grad_norm: Optional[float] = None
            if track_gradients:
                grad_norm_value = compute_grad_norm(model)
                grad_norms.append(grad_norm_value)
                # grad_norm can be NaN if any gradients are non-finite
                grad_norm = float(grad_norm_value) if np.isfinite(grad_norm_value) else None

            # Clip gradients
            orig_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            orig_norm_value = float(orig_norm)
            grads_are_finite = np.isfinite(orig_norm_value)

            if not grads_are_finite:
                nan_grad_count += 1
                non_finite_grad_params = get_non_finite_grad_parameter_names(model, max_names=10)
                metadata = batch.get("metadata", {})
                sample_ids = []
                for i in range(min(waveform.shape[0], 5)):
                    sample_ids.append(
                        {
                            "flac_file": (metadata.get("flac_file") or [None])[i],
                            "codec_seed": (metadata.get("codec_seed") or [None])[i],
                            "speaker_id": (metadata.get("speaker_id") or [None])[i],
                        }
                    )
                repr_stats = None
                if isinstance(outputs, dict) and "repr" in outputs:
                    repr_tensor = outputs["repr"].detach().float()
                    repr_stats = {
                        "min": float(repr_tensor.min().item()),
                        "max": float(repr_tensor.max().item()),
                        "mean": float(repr_tensor.mean().item()),
                    }
                logger.warning(
                    "Non-finite gradients detected at batch "
                    f"{batch_idx} (global_step={global_step}, grad_norm={orig_norm_value}). "
                    f"params={non_finite_grad_params} "
                    f"sample_ids={sample_ids} "
                    f"waveform_stats={{'min': {float(waveform.min().item()):.6g}, "
                    f"'max': {float(waveform.max().item()):.6g}, "
                    f"'mean': {float(waveform.mean().item()):.6g}}} "
                    f"repr_stats={repr_stats}"
                )
                if exp_logger is not None:
                    exp_logger.log_wide_event(
                        "non_finite_grad_batch",
                        {
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "method": method,
                            "grad_norm": grad_norm,
                            "clipped_grad_norm": orig_norm_value,
                            "params": non_finite_grad_params,
                            "sample_ids": sample_ids,
                            "waveform_stats": {
                                "min": float(waveform.min().item()),
                                "max": float(waveform.max().item()),
                                "mean": float(waveform.mean().item()),
                            },
                            "repr_stats": repr_stats,
                            "losses": {
                                k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                                for k, v in losses.items()
                            },
                        },
                    )
                # Skip optimizer step to avoid poisoning the run.
                # GradScaler will automatically backoff when update() is called without step().
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if nan_grad_abort_count is not None and nan_grad_count >= nan_grad_abort_count:
                    raise RuntimeError(
                        f"Aborting: {nan_grad_count} non-finite-gradient batches "
                        f"in epoch (threshold={nan_grad_abort_count}). "
                        "Reduce LR, increase warmup, tighten grad clip, or disable AMP."
                    )
            else:
                if orig_norm_value > gradient_clip:
                    grad_clips_count += 1

                # Let GradScaler decide whether to skip the optimizer step (inf/nan grads).
                # If it skips, we also skip the scheduler step to keep LR aligned with updates.
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scaler_skipped_step = scaler.get_scale() < prev_scale
                optimizer_step_taken = not scaler_skipped_step
        else:
            loss.backward()

            # Track gradients before clipping
            grad_norm: Optional[float] = None
            if track_gradients:
                grad_norm_value = compute_grad_norm(model)
                grad_norms.append(grad_norm_value)
                grad_norm = float(grad_norm_value) if np.isfinite(grad_norm_value) else None

            orig_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            orig_norm_value = float(orig_norm)
            grads_are_finite = np.isfinite(orig_norm_value)

            if not grads_are_finite:
                nan_grad_count += 1
                non_finite_grad_params = get_non_finite_grad_parameter_names(model, max_names=10)
                metadata = batch.get("metadata", {})
                sample_ids = []
                for i in range(min(waveform.shape[0], 5)):
                    sample_ids.append(
                        {
                            "flac_file": (metadata.get("flac_file") or [None])[i],
                            "codec_seed": (metadata.get("codec_seed") or [None])[i],
                            "speaker_id": (metadata.get("speaker_id") or [None])[i],
                        }
                    )
                repr_stats = None
                if isinstance(outputs, dict) and "repr" in outputs:
                    repr_tensor = outputs["repr"].detach().float()
                    repr_stats = {
                        "min": float(repr_tensor.min().item()),
                        "max": float(repr_tensor.max().item()),
                        "mean": float(repr_tensor.mean().item()),
                    }
                logger.warning(
                    "Non-finite gradients detected at batch "
                    f"{batch_idx} (global_step={global_step}, grad_norm={orig_norm_value}). "
                    f"params={non_finite_grad_params} "
                    f"sample_ids={sample_ids} "
                    f"waveform_stats={{'min': {float(waveform.min().item()):.6g}, "
                    f"'max': {float(waveform.max().item()):.6g}, "
                    f"'mean': {float(waveform.mean().item()):.6g}}} "
                    f"repr_stats={repr_stats}"
                )
                if exp_logger is not None:
                    exp_logger.log_wide_event(
                        "non_finite_grad_batch",
                        {
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "method": method,
                            "grad_norm": grad_norm,
                            "clipped_grad_norm": orig_norm_value,
                            "params": non_finite_grad_params,
                            "sample_ids": sample_ids,
                            "waveform_stats": {
                                "min": float(waveform.min().item()),
                                "max": float(waveform.max().item()),
                                "mean": float(waveform.mean().item()),
                            },
                            "repr_stats": repr_stats,
                            "losses": {
                                k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                                for k, v in losses.items()
                            },
                        },
                    )
                optimizer.zero_grad(set_to_none=True)
                if nan_grad_abort_count is not None and nan_grad_count >= nan_grad_abort_count:
                    raise RuntimeError(
                        f"Aborting: {nan_grad_count} non-finite-gradient batches "
                        f"in epoch (threshold={nan_grad_abort_count}). "
                        "Reduce LR, increase warmup, tighten grad clip, or disable AMP."
                    )
            else:
                if orig_norm_value > gradient_clip:
                    grad_clips_count += 1
                optimizer.step()
                optimizer_step_taken = True

        if scheduler is not None:
            if optimizer_step_taken:
                scheduler.step()

        # Increment global step
        global_step += 1

        # Compute accuracies
        task_acc = compute_accuracy(outputs["task_logits"], y_task)

        # Accumulate metrics
        total_loss += losses["total_loss"].item()
        total_task_loss += losses["task_loss"].item()
        total_task_acc += task_acc
        num_batches += 1

        if method == "dann":
            total_codec_loss += losses["codec_loss"].item()
            total_codec_q_loss += losses["codec_q_loss"].item()
            codec_acc = compute_accuracy(outputs["codec_logits"], y_codec)
            codec_q_acc = compute_accuracy(outputs["codec_q_logits"], y_codec_q)
            total_codec_acc += codec_acc
            total_codec_q_acc += codec_q_acc

        # Track augmentation rate whenever synthetic labels are present.
        if saw_aug_labels:
            total_aug_samples += y_codec.numel()
            total_non_none_samples += (y_codec != 0).sum().item()

            # Log augmentation rate periodically and fail if too low (DANN only)
            if batch_idx > 0 and batch_idx % 100 == 0:
                aug_rate = total_non_none_samples / max(total_aug_samples, 1)
                logger.info(
                    f"Step {batch_idx}: cumulative augmentation rate = {aug_rate:.1%} "
                    f"({total_non_none_samples}/{total_aug_samples} samples coded)"
                )

                # Fail if augmentation rate is near zero after sufficient steps
                if method == "dann" and batch_idx >= 500 and aug_rate < 0.05:
                    raise RuntimeError(
                        f"Augmentation rate {aug_rate:.1%} < 5% after {batch_idx} steps. "
                        f"DANN requires domain diversity. Check codec_prob and ffmpeg codec support."
                    )

        # Sample batches for detailed logging
        if batch_sample_rate > 0 and random.random() < batch_sample_rate:
            batch_duration = time.time() - batch_start_time
            batch_event = {
                "batch_idx": batch_idx,
                "loss": losses["total_loss"].item(),
                "task_loss": losses["task_loss"].item(),
                "task_acc": task_acc,
                "grad_norm": grad_norms[-1] if grad_norms else None,
                "batch_duration_sec": batch_duration,
                "batch_size": waveform.shape[0],
            }
            if method == "dann":
                batch_event["codec_loss"] = losses["codec_loss"].item()
                batch_event["codec_q_loss"] = losses["codec_q_loss"].item()
                batch_event["codec_acc"] = codec_acc
                batch_event["codec_q_acc"] = codec_q_acc
            batch_samples.append(batch_event)

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "task_acc": f"{total_task_acc / num_batches:.4f}",
            })

        # Step-level wandb logging
        if exp_logger is not None and global_step % log_step_interval == 0:
            step_metrics = {
                "train/step_loss": losses["total_loss"].item(),
                "train/step_task_loss": losses["task_loss"].item(),
                "train/step_task_acc": task_acc,
            }
            if method == "dann":
                step_metrics["train/step_codec_loss"] = losses["codec_loss"].item()
                step_metrics["train/step_codec_q_loss"] = losses["codec_q_loss"].item()
                step_metrics["train/step_codec_acc"] = codec_acc
                step_metrics["train/step_codec_q_acc"] = codec_q_acc
            if track_gradients and grad_norms:
                step_metrics["train/step_grad_norm"] = grad_norms[-1]
            exp_logger.log_step_metrics(step_metrics, step=global_step)

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "task_acc": total_task_acc / num_batches,
    }

    if method == "dann":
        metrics["codec_loss"] = total_codec_loss / num_batches
        metrics["codec_q_loss"] = total_codec_q_loss / num_batches
        metrics["codec_acc"] = total_codec_acc / num_batches
        metrics["codec_q_acc"] = total_codec_q_acc / num_batches
    # Augmentation rate metric (for DANN and ERM+aug controls)
    if saw_aug_labels and total_aug_samples > 0:
        metrics["aug_rate"] = total_non_none_samples / total_aug_samples

    # Gradient statistics
    if track_gradients and grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(grad_norms))
        metrics["grad_norm_max"] = float(np.max(grad_norms))
        metrics["grad_clips"] = grad_clips_count
        metrics["nan_grads"] = nan_grad_count

    # Include batch samples for wide event logging
    if batch_samples:
        metrics["_batch_samples"] = batch_samples

    return metrics, global_step


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    method: str = "erm",
    compute_domain_breakdown: bool = False,
    codec_vocab: Optional[dict] = None,
    codec_q_vocab: Optional[dict] = None,
) -> dict:
    """Validate for one epoch.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: Device.
        method: Training method ('erm' or 'dann').
        compute_domain_breakdown: Whether to compute per-domain metrics.
        codec_vocab: CODEC vocabulary (for domain breakdown).
        codec_q_vocab: CODEC_Q vocabulary (for domain breakdown).

    Returns:
        Dictionary of average metrics for the epoch.
    """
    model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_codec_loss = 0.0
    total_codec_q_loss = 0.0
    total_task_acc = 0.0
    total_codec_acc = 0.0
    total_codec_q_acc = 0.0
    num_batches = 0

    all_scores = []
    all_labels = []
    all_codec_labels = []
    all_codec_q_labels = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"].to(device)
        y_codec = batch["y_codec"].to(device)
        y_codec_q = batch["y_codec_q"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        if method == "dann":
            # During validation, skip domain loss computation.
            # Validation uses manifest labels which may not match the synthetic
            # vocab used during training (e.g., manifest has 12 codecs, model expects 6).
            # Domain invariance is measured separately via domain probe accuracy.
            losses = loss_fn(
                outputs["task_logits"],
                None,  # Skip domain loss - codec_logits
                None,  # Skip domain loss - codec_q_logits
                y_task,
                None,  # Skip domain loss - codec_labels
                None,  # Skip domain loss - codec_q_labels
            )
        else:
            losses = loss_fn(outputs["task_logits"], y_task)

        # Compute accuracies
        task_acc = compute_accuracy(outputs["task_logits"], y_task)

        # Accumulate metrics
        total_loss += losses["total_loss"].item()
        total_task_loss += losses["task_loss"].item()
        total_task_acc += task_acc
        num_batches += 1

        # Note: For DANN validation, we skip domain accuracy computation because
        # validation labels come from manifest vocab (may differ from training vocab).
        # Domain invariance should be evaluated via domain probe accuracy instead.

        # Collect scores for EER computation
        # Score convention: higher = more bonafide (class 0)
        # Use logit difference or softmax probability
        scores = torch.softmax(outputs["task_logits"], dim=-1)[:, 0]  # P(bonafide)
        all_scores.append(scores.cpu())
        all_labels.append(y_task.cpu())

        # Collect domain labels for breakdown
        if compute_domain_breakdown:
            all_codec_labels.append(y_codec.cpu())
            all_codec_q_labels.append(y_codec_q.cpu())

    # Compute EER
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from ..evaluation.metrics import compute_eer, compute_min_dcf

    eer, _ = compute_eer(all_scores, all_labels)
    min_dcf = compute_min_dcf(all_scores, all_labels)

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "task_acc": total_task_acc / num_batches,
        "eer": eer,
        "min_dcf": min_dcf,
    }

    # Note: For DANN validation, domain metrics are not computed because
    # validation labels come from manifest vocab (may differ from training vocab).
    # Domain invariance is measured separately via domain probe accuracy.

    # Per-domain breakdown
    if compute_domain_breakdown and codec_vocab and codec_q_vocab:
        all_codec_labels = torch.cat(all_codec_labels).numpy()
        all_codec_q_labels = torch.cat(all_codec_q_labels).numpy()

        # Compute EER per CODEC
        codec_id_to_name = {v: k for k, v in codec_vocab.items()}
        metrics["per_codec"] = {}
        for codec_id in np.unique(all_codec_labels):
            mask = all_codec_labels == codec_id
            if mask.sum() > 10:  # Need enough samples
                codec_scores = all_scores[mask]
                codec_labels = all_labels[mask]
                if len(np.unique(codec_labels)) == 2:  # Need both classes
                    codec_eer, _ = compute_eer(codec_scores, codec_labels)
                    codec_name = codec_id_to_name.get(codec_id, str(codec_id))
                    metrics["per_codec"][codec_name] = {
                        "eer": float(codec_eer),
                        "n_samples": int(mask.sum()),
                    }

        # Compute EER per CODEC_Q
        codec_q_id_to_name = {v: k for k, v in codec_q_vocab.items()}
        metrics["per_codec_q"] = {}
        for codec_q_id in np.unique(all_codec_q_labels):
            mask = all_codec_q_labels == codec_q_id
            if mask.sum() > 10:
                cq_scores = all_scores[mask]
                cq_labels = all_labels[mask]
                if len(np.unique(cq_labels)) == 2:
                    cq_eer, _ = compute_eer(cq_scores, cq_labels)
                    cq_name = codec_q_id_to_name.get(codec_q_id, str(codec_q_id))
                    metrics["per_codec_q"][cq_name] = {
                        "eer": float(cq_eer),
                        "n_samples": int(mask.sum()),
                    }

    return metrics


def get_layer_weights(model: nn.Module) -> Optional[list[float]]:
    """Extract learned layer mixing weights from model.

    Args:
        model: Model with potential layer_pooling.

    Returns:
        List of layer weights or None if not available.
    """
    try:
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer_pooling"):
            weights = model.backbone.layer_pooling.weights
            normalized = torch.softmax(weights, dim=0)
            return normalized.detach().cpu().tolist()
    except Exception:
        pass
    return None


class Trainer:
    """Training loop manager with checkpointing, early stopping, and comprehensive logging.

    Args:
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler.
        device: Device.
        run_dir: Directory for saving outputs.
        config: Full resolved config.
        method: Training method ('erm' or 'dann').
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum improvement to count as progress (e.g., 0.001 for EER).
        train_loss_threshold: Stop if train loss is below this for plateau_patience epochs.
        plateau_patience: Epochs of low train loss before stopping (prevents overfitting).
        gradient_clip: Gradient clipping value.
        use_amp: Whether to use automatic mixed precision.
        log_interval: Logging interval (steps).
        val_interval: Validation interval (epochs).
        save_every_n_epochs: Checkpoint save interval.
        monitor_metric: Metric to monitor for best model.
        monitor_mode: 'min' or 'max'.
        lambda_scheduler: Optional DANN lambda scheduler.
        use_wandb: Whether to use wandb logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity (team or username).
        wandb_run_name: Wandb run name.
        wandb_tags: Optional list of wandb tags.
        batch_sample_rate: Rate of batches to log in detail (0.0-1.0).
        track_gradients: Whether to track gradient norms.
        log_domain_breakdown_every: Epochs between per-domain metric computation.
        codec_vocab: CODEC vocabulary for domain breakdown.
        codec_q_vocab: CODEC_Q vocabulary for domain breakdown.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        run_dir: Path,
        config: dict,
        method: str = "erm",
        max_epochs: int = 50,
        patience: int = 10,
        min_delta: float = 0.001,
        train_loss_threshold: float = 0.01,
        plateau_patience: int = 3,
        gradient_clip: float = 1.0,
        use_amp: bool = False,
        log_interval: int = 50,
        val_interval: int = 1,
        save_every_n_epochs: int = 5,
        monitor_metric: str = "eer",
        monitor_mode: str = "min",
        lambda_scheduler=None,
        use_wandb: bool = False,
        wandb_project: str = "asvspoof5-dann",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        batch_sample_rate: float = 0.02,
        track_gradients: bool = True,
        log_domain_breakdown_every: int = 5,
        codec_vocab: Optional[dict] = None,
        codec_q_vocab: Optional[dict] = None,
        nan_grad_abort_count: Optional[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = Path(run_dir)
        self.config = config
        self.method = method
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.train_loss_threshold = train_loss_threshold
        self.plateau_patience = plateau_patience
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.lambda_scheduler = lambda_scheduler
        self.batch_sample_rate = batch_sample_rate
        self.track_gradients = track_gradients
        self.log_domain_breakdown_every = log_domain_breakdown_every
        self.codec_vocab = codec_vocab
        self.codec_q_vocab = codec_q_vocab
        self.nan_grad_abort_count = nan_grad_abort_count

        self.scaler = GradScaler("cuda") if use_amp else None

        # Setup directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize ExperimentLogger
        self.exp_logger = ExperimentLogger(
            run_dir=self.run_dir,
            run_name=wandb_run_name or self.run_dir.name,
            config=config,
            use_wandb=use_wandb and WANDB_AVAILABLE,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
        )

        # Log experiment start
        self._log_experiment_start()

        # State
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
        self.epochs_at_train_plateau = 0  # Track consecutive epochs with very low train loss
        self.current_epoch = 0
        self.global_step = 0

        # Training log
        self.train_log = []

        # Save config
        save_config(config, self.run_dir / "config_resolved.yaml")

    def _log_experiment_start(self) -> None:
        """Log comprehensive experiment start event."""
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.exp_logger.log_model_summary(self.model, watch_gradients=False)

        # Dataset stats
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        self.exp_logger.log_dataset_stats(
            train_size=train_size,
            val_size=val_size,
        )

        # Log experiment start wide event
        backbone_name = self.config.get("backbone", {}).get("name", "unknown")
        start_event = {
            "method": self.method,
            "backbone": backbone_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.config.get("dataloader", {}).get("batch_size", 32),
            "learning_rate": self.config.get("training", {}).get("optimizer", {}).get("lr"),
            "model": {
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            "dataset": {
                "train_size": train_size,
                "val_size": val_size,
            },
        }

        if self.method == "dann":
            start_event["dann"] = {
                "lambda_init": self.config.get("dann", {}).get("lambda_", 0.1),
                "lambda_schedule": self.config.get("dann", {}).get("lambda_schedule", {}),
            }

        self.exp_logger.log_wide_event("experiment_start", start_event)

        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Run directory: {self.run_dir}")
        logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
        logger.info(f"Dataset: train={train_size}, val={val_size}")

    def train(self) -> dict:
        """Run full training loop.

        Returns:
            Dictionary of best metrics.
        """
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            # Update DANN lambda if scheduled
            current_lambda = None
            if self.lambda_scheduler is not None and self.method == "dann":
                current_lambda = self.lambda_scheduler.get_lambda(epoch)
                self.model.set_lambda(current_lambda)
                # Sync loss function lambda with GRL for mathematical consistency (DANN paper)
                if hasattr(self.loss_fn, "set_lambda"):
                    self.loss_fn.set_lambda(current_lambda)
                lambda_grl = float(self.model.get_lambda()) if hasattr(self.model, "get_lambda") else float(current_lambda)
                if hasattr(self.loss_fn, "lambda_domain"):
                    lambda_domain_loss = float(self.loss_fn.lambda_domain)
                elif hasattr(self.loss_fn, "lambda_"):
                    lambda_domain_loss = float(self.loss_fn.lambda_)
                else:
                    lambda_domain_loss = float(current_lambda) if current_lambda is not None else None
                logger.info(
                    f"Epoch {epoch}: lambda_grl={lambda_grl:.6f}, "
                    f"lambda_domain_loss={float(lambda_domain_loss):.6f}"
                )
            elif self.method == "dann":
                # Even with a constant λ (no scheduler), log the current values.
                lambda_grl = float(self.model.get_lambda()) if hasattr(self.model, "get_lambda") else None
                if hasattr(self.loss_fn, "lambda_domain"):
                    lambda_domain_loss = float(self.loss_fn.lambda_domain)
                elif hasattr(self.loss_fn, "lambda_"):
                    lambda_domain_loss = float(self.loss_fn.lambda_)
                else:
                    lambda_domain_loss = float(lambda_grl) if lambda_grl is not None else 0.0
                if lambda_grl is not None:
                    logger.info(
                        f"Epoch {epoch}: lambda_grl={lambda_grl:.6f}, "
                        f"lambda_domain_loss={lambda_domain_loss:.6f} (no scheduler)"
                    )

            # Fail-fast correctness assertions for DANN.
            # Requirement: λ_grl must be > 0 once the warmup phase is over.
            #
            # Timing semantics (0-indexed epochs):
            #   - LambdaScheduler.get_lambda(epoch) returns 0.0 when epoch < warmup_epochs
            #   - So with warmup_epochs=3, epochs 0,1,2 return 0, epoch 3 returns >0
            #   - check_epoch = warmup_epochs means we check at the first non-zero epoch
            #   - The `max(1, ...)` ensures we check at least by epoch 1 when there's no
            #     scheduler or warmup_epochs=0 (constant lambda should be >0 immediately)
            warmup_epochs = getattr(self.lambda_scheduler, "warmup_epochs", 0) if self.lambda_scheduler else 0
            check_epoch = max(1, warmup_epochs)  # First epoch where lambda must be >0
            if self.method == "dann" and epoch >= check_epoch and hasattr(self.model, "get_lambda"):
                lambda_grl_epoch = float(self.model.get_lambda())
                if lambda_grl_epoch <= 0.0:
                    raise RuntimeError(
                        f"DANN requires lambda_grl > 0 at epoch {check_epoch} (first post-warmup epoch), "
                        f"but got lambda_grl={lambda_grl_epoch}. "
                        "Fix: check lambda_schedule config or ensure training isn't cut off early."
                    )

            # Train
            train_metrics, self.global_step = train_epoch(
                self.model,
                self.train_loader,
                self.loss_fn,
                self.optimizer,
                self.device,
                scheduler=self.scheduler,
                scaler=self.scaler,
                gradient_clip=self.gradient_clip,
                method=self.method,
                log_interval=self.log_interval,
                batch_sample_rate=self.batch_sample_rate,
                track_gradients=self.track_gradients,
                exp_logger=self.exp_logger,
                global_step_start=self.global_step,
                log_step_interval=self.log_interval,
                nan_grad_abort_count=self.nan_grad_abort_count,
            )

            # Extract batch samples for separate logging
            batch_samples = train_metrics.pop("_batch_samples", [])

            # Log training metrics
            logger.info(
                f"Epoch {epoch} train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"task_acc={train_metrics['task_acc']:.4f}"
            )

            # Validate
            val_metrics = None
            if epoch % self.val_interval == 0:
                # Compute domain breakdown periodically
                compute_breakdown = (
                    epoch % self.log_domain_breakdown_every == 0
                    and self.codec_vocab is not None
                    and self.codec_q_vocab is not None
                )

                val_metrics = validate_epoch(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.device,
                    method=self.method,
                    compute_domain_breakdown=compute_breakdown,
                    codec_vocab=self.codec_vocab,
                    codec_q_vocab=self.codec_q_vocab,
                )

                logger.info(
                    f"Epoch {epoch} val: "
                    f"loss={val_metrics['loss']:.4f}, "
                    f"eer={val_metrics['eer']:.4f}, "
                    f"min_dcf={val_metrics['min_dcf']:.4f}"
                )

                # Check for improvement (must improve by at least min_delta)
                current_metric = val_metrics[self.monitor_metric]
                if self.monitor_mode == "min":
                    is_better = current_metric < (self.best_metric - self.min_delta)
                else:
                    is_better = current_metric > (self.best_metric + self.min_delta)

                if is_better:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0

                    # Save best model
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_metrics,
                        self.run_dir / "checkpoints" / "best.pt",
                        config=self.config,
                    )
                    logger.info(f"New best {self.monitor_metric}: {self.best_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1

            # Track train loss plateau (model has memorized training data)
            if train_metrics["loss"] < self.train_loss_threshold:
                self.epochs_at_train_plateau += 1
            else:
                self.epochs_at_train_plateau = 0

            # Build epoch-complete wide event
            epoch_duration = time.time() - epoch_start_time
            epoch_event = {
                "epoch": epoch,
                "duration_sec": round(epoch_duration, 2),
                "train": {
                    k: v for k, v in train_metrics.items()
                    if isinstance(v, (int, float))
                },
                "is_best": val_metrics is not None and self.epochs_without_improvement == 0,
            }

            if val_metrics:
                epoch_event["val"] = {
                    k: v for k, v in val_metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("per_")
                }
                # Include domain breakdown if computed
                if "per_codec" in val_metrics:
                    epoch_event["per_codec"] = val_metrics["per_codec"]
                if "per_codec_q" in val_metrics:
                    epoch_event["per_codec_q"] = val_metrics["per_codec_q"]

            # Learning rate
            if self.scheduler is not None:
                epoch_event["learning_rate"] = self.scheduler.get_last_lr()[0]

            # DANN lambda (log both GRL and domain-loss weight separately)
            if self.method == "dann":
                if hasattr(self.model, "get_lambda"):
                    epoch_event["lambda_grl"] = float(self.model.get_lambda())
                if hasattr(self.loss_fn, "lambda_domain"):
                    epoch_event["lambda_domain_loss"] = float(self.loss_fn.lambda_domain)
                elif hasattr(self.loss_fn, "lambda_"):
                    epoch_event["lambda_domain_loss"] = float(self.loss_fn.lambda_)
                if current_lambda is not None:
                    epoch_event["lambda_domain"] = float(current_lambda)

            # Layer weights
            layer_weights = get_layer_weights(self.model)
            if layer_weights:
                epoch_event["layer_weights"] = layer_weights

            # GPU memory
            gpu_mem = get_gpu_memory_usage()
            if gpu_mem is not None:
                epoch_event["gpu_memory_gb"] = gpu_mem

            # Log wide event
            self.exp_logger.log_wide_event("epoch_complete", epoch_event)

            # Record to log
            log_entry = {
                "epoch": epoch,
                "train": train_metrics,
            }
            if val_metrics:
                log_entry["val"] = {
                    k: v for k, v in val_metrics.items()
                    if not k.startswith("per_")
                }
            if current_lambda is not None:
                log_entry["lambda"] = current_lambda
            self.train_log.append(log_entry)

            # Periodic checkpoint
            if epoch % self.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics,
                    self.run_dir / "checkpoints" / f"epoch_{epoch}.pt",
                )

            # Early stopping checks
            should_stop = False
            stop_reason = ""

            # 1. Val metric hasn't improved for patience epochs
            if self.epochs_without_improvement >= self.patience:
                should_stop = True
                stop_reason = f"val {self.monitor_metric} hasn't improved for {self.patience} epochs"

            # 2. Train loss plateau: model has memorized data, further training is overfitting
            if self.epochs_at_train_plateau >= self.plateau_patience:
                should_stop = True
                stop_reason = (
                    f"train loss below {self.train_loss_threshold} for "
                    f"{self.plateau_patience} consecutive epochs (model memorized training data)"
                )

            if should_stop:
                logger.info(f"Early stopping at epoch {epoch}: {stop_reason}")
                break

        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            train_metrics,
            self.run_dir / "checkpoints" / "last.pt",
        )

        # Save training log
        with open(self.run_dir / "train_log.jsonl", "w") as f:
            for entry in self.train_log:
                f.write(json.dumps(entry, default=str) + "\n")

        # Build final metrics
        final_metrics = {
            "best_epoch": self.current_epoch - self.epochs_without_improvement,
            f"best_{self.monitor_metric}": self.best_metric,
            "final_epoch": self.current_epoch,
        }
        if self.train_log:
            final_metrics["final_val"] = self.train_log[-1].get("val", {})

        save_metrics(final_metrics, self.run_dir / "metrics_train.json")

        # Log training complete wide event
        self.exp_logger.log_wide_event("training_complete", final_metrics)

        # Set wandb summary
        self.exp_logger.set_summary({
            "best_epoch": final_metrics["best_epoch"],
            f"best_{self.monitor_metric}": self.best_metric,
            "total_epochs": self.current_epoch + 1,
        })

        # Finish logging
        self.exp_logger.finish()

        return final_metrics
