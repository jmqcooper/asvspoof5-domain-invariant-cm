#!/usr/bin/env python3
"""Compare layer-wise codec encoding between WavLM and Wav2Vec2 backbones.

This script investigates why DANN performs differently on WavLM vs wav2vec2
by analyzing how each backbone encodes codec information across transformer layers.

Key questions this script answers:
1. Which layers encode codec information most strongly in each backbone?
2. Are there systematic differences in how WavLM vs wav2vec2 encode codecs?
3. Does WavLM's denoising pretraining make it more/less codec-sensitive?
4. Do the learned layer weights correlate with codec encoding strength?

Background:
- WavLM achieved 3.59% EER at epoch 9 with DANN (beats ERM baseline)
- wav2vec2 at 4.87% EER at epoch 11 with DANN (WORSE than ERM at 4.24%)

Architectural differences that may explain this:
- WavLM: gated relative position bias (better local temporal modeling)
- WavLM: denoising pretraining objective (trained on overlapped/noisy speech)
- wav2vec2: convolutional position embeddings (less adaptive)

Usage:
    # Compare both backbones on ASVspoof5 data
    python scripts/compare_backbone_codec_probes.py --num-samples 3000

    # Quick test with synthetic audio
    python scripts/compare_backbone_codec_probes.py --synthetic --num-samples 500

    # Use pre-augmented cache
    python scripts/compare_backbone_codec_probes.py --cache-dir /path/to/augmented_cache

    # Compare with specific checkpoints to analyze learned layer weights
    python scripts/compare_backbone_codec_probes.py --wavlm-checkpoint runs/wavlm_dann/best.pt \
        --w2v2-checkpoint runs/w2v2_dann/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKBONES = {
    "wavlm": {
        "pretrained": "microsoft/wavlm-base-plus",
        "num_layers": 12,
        "hidden_size": 768,
        "description": "WavLM Base+ (gated rel pos bias + denoising pretraining)",
    },
    "w2v2": {
        "pretrained": "facebook/wav2vec2-base",
        "num_layers": 12,
        "hidden_size": 768,
        "description": "Wav2Vec 2.0 Base (conv pos embeddings + contrastive pretraining)",
    },
}

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 4.0
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLE_RATE)

# Codec configs
CODEC_CONFIGS = {
    "MP3": ("libmp3lame", "mp3", ".mp3", [64, 128, 256]),
    "AAC": ("aac", "adts", ".aac", [32, 96, 192]),
    "OPUS": ("libopus", "opus", ".opus", [12, 48, 96]),
}


@dataclass
class ProbeResult:
    """Results from a single layer probe."""
    layer: int
    backbone: str
    accuracy: float
    accuracy_std: float
    n_samples: int
    n_classes: int


@dataclass
class BackboneComparison:
    """Full comparison between backbones."""
    wavlm_results: dict[int, ProbeResult]
    w2v2_results: dict[int, ProbeResult]
    wavlm_layer_weights: Optional[np.ndarray]
    w2v2_layer_weights: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare layer-wise codec encoding between WavLM and Wav2Vec2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--num-samples", type=int, default=3000,
        help="Number of audio samples to use (default: 3000)",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic audio instead of ASVspoof5 data",
    )
    p.add_argument(
        "--split", choices=["train", "dev"], default="train",
        help="ASVspoof5 split to load (default: train)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for feature extraction (default: 8)",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory with pre-augmented audio cache",
    )
    p.add_argument(
        "--wavlm-checkpoint", type=Path, default=None,
        help="Optional WavLM DANN checkpoint to extract learned layer weights",
    )
    p.add_argument(
        "--w2v2-checkpoint", type=Path, default=None,
        help="Optional Wav2Vec2 DANN checkpoint to extract learned layer weights",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: results/backbone_comparison)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# FFmpeg helpers (reused from run_domain_probe.py)
# ---------------------------------------------------------------------------
def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def check_codec_support() -> dict[str, bool]:
    supported = {}
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10,
        )
        for name, (encoder, *_) in CODEC_CONFIGS.items():
            supported[name] = encoder in result.stdout
    except Exception:
        for name in CODEC_CONFIGS:
            supported[name] = False
    return supported


def apply_codec(
    waveform: np.ndarray,
    codec_name: str,
    bitrate_kbps: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray | None:
    import soundfile as sf

    encoder, fmt, ext, _ = CODEC_CONFIGS[codec_name]
    tmp_in = tmp_enc = tmp_out = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_in = f.name
        sf.write(tmp_in, waveform, sample_rate)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            tmp_enc = f.name
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in,
                "-ar", str(sample_rate), "-ac", "1",
                "-c:a", encoder, "-b:a", f"{bitrate_kbps}k",
                "-f", fmt, tmp_enc,
            ],
            capture_output=True, check=True, timeout=30,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_out = f.name
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_enc,
                "-ar", str(sample_rate), "-ac", "1", tmp_out,
            ],
            capture_output=True, check=True, timeout=30,
        )

        data, _ = sf.read(tmp_out, dtype="float32")
        return data

    except Exception:
        return None

    finally:
        for p in (tmp_in, tmp_enc, tmp_out):
            if p and os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_real_audio(split: str, num_samples: int, seed: int) -> list[np.ndarray]:
    import soundfile as sf

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from asvspoof5_domain_invariant_cm.utils.paths import get_manifest_path
        manifest_path = get_manifest_path(split)
    except Exception:
        project_root = Path(__file__).resolve().parent.parent
        manifest_path = project_root / "data" / "manifests" / f"{split}.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Use --synthetic for testing."
        )

    import pandas as pd
    df = pd.read_parquet(manifest_path)

    rng = np.random.RandomState(seed)
    n = min(num_samples, len(df))
    indices = rng.choice(len(df), size=n, replace=False)

    waveforms = []
    for idx in tqdm(indices, desc="Loading audio"):
        audio_path = df.iloc[idx]["audio_path"]
        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if len(data) > MAX_SAMPLES:
                start = rng.randint(0, len(data) - MAX_SAMPLES)
                data = data[start:start + MAX_SAMPLES]
            elif len(data) < MAX_SAMPLES:
                data = np.pad(data, (0, MAX_SAMPLES - len(data)))
            waveforms.append(data)
        except Exception:
            continue

    logger.info(f"Loaded {len(waveforms)} audio samples from {split} split")
    return waveforms


def generate_synthetic_audio(num_samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    waveforms = []
    t = np.linspace(0, MAX_DURATION_SEC, MAX_SAMPLES, dtype=np.float32)

    for _ in range(num_samples):
        f0 = rng.uniform(80, 300)
        signal = np.zeros(MAX_SAMPLES, dtype=np.float32)

        for h in range(1, rng.randint(4, 12)):
            amp = rng.uniform(0.05, 0.3) / h
            signal += np.float32(amp) * np.sin(
                np.float32(2 * np.pi * f0 * h) * t + np.float32(rng.uniform(0, 2 * np.pi))
            )

        for _ in range(rng.randint(2, 5)):
            formant_f = rng.uniform(300, 4000)
            amp = rng.uniform(0.02, 0.15)
            signal += np.float32(amp) * np.sin(np.float32(2 * np.pi * formant_f) * t)

        noise_level = rng.uniform(0.001, 0.02)
        signal += noise_level * rng.randn(MAX_SAMPLES).astype(np.float32)

        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak * rng.uniform(0.3, 0.9)

        waveforms.append(signal)

    logger.info(f"Generated {num_samples} synthetic audio samples")
    return waveforms


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------
def create_augmented_dataset(
    waveforms: list[np.ndarray],
    supported_codecs: dict[str, bool],
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Create pairs of original + codec-augmented audio.
    
    Returns (all_waveforms, binary_labels, multiclass_labels)
    """
    codecs_available = [c for c, ok in supported_codecs.items() if ok]
    if not codecs_available:
        raise RuntimeError("No codecs available!")

    multiclass_map = {"NONE": 0, "MP3": 1, "AAC": 2, "OPUS": 3}
    rng = np.random.RandomState(seed)

    all_waveforms = []
    binary_labels = []
    multiclass_labels = []

    for wav in tqdm(waveforms, desc="Creating augmented pairs"):
        # Original
        all_waveforms.append(wav)
        binary_labels.append(0)
        multiclass_labels.append(0)

        # Augmented version
        codec = rng.choice(codecs_available)
        _, _, _, bitrates = CODEC_CONFIGS[codec]
        bitrate = rng.choice(bitrates)

        augmented = apply_codec(wav, codec, bitrate)
        if augmented is not None:
            if len(augmented) > MAX_SAMPLES:
                augmented = augmented[:MAX_SAMPLES]
            elif len(augmented) < MAX_SAMPLES:
                augmented = np.pad(augmented, (0, MAX_SAMPLES - len(augmented)))

            all_waveforms.append(augmented)
            binary_labels.append(1)
            multiclass_labels.append(multiclass_map[codec])

    logger.info(f"Created {len(all_waveforms)} samples (original + augmented)")
    return all_waveforms, np.array(binary_labels), np.array(multiclass_labels)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_backbone(name: str) -> tuple[torch.nn.Module, dict]:
    cfg = BACKBONES[name]
    logger.info(f"Loading {name}: {cfg['pretrained']}")

    if name == "wavlm":
        from transformers import WavLMModel
        model = WavLMModel.from_pretrained(cfg["pretrained"], output_hidden_states=True)
    else:
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(cfg["pretrained"], output_hidden_states=True)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, cfg


@torch.no_grad()
def extract_all_layer_features(
    model: torch.nn.Module,
    waveforms: list[np.ndarray],
    original_lengths: list[int],
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled features from every transformer layer.
    
    Args:
        model: HuggingFace WavLM or Wav2Vec2 model
        waveforms: List of padded waveform arrays
        original_lengths: Original sample lengths before padding (for attention mask)
        batch_size: Batch size for inference
    
    Returns:
        Dictionary mapping layer index to pooled features
    """
    num_layers = model.config.num_hidden_layers
    layer_features: dict[int, list] = {i: [] for i in range(num_layers)}
    
    # Get model device
    device = next(model.parameters()).device

    n_batches = (len(waveforms) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(waveforms))
        batch_np = waveforms[start:end]
        batch_lengths = original_lengths[start:end]

        # Move tensor to model device
        batch_tensor = torch.tensor(np.stack(batch_np), dtype=torch.float32).to(device)
        
        # Create sample-level attention mask based on original lengths
        max_len = batch_tensor.shape[1]
        attention_mask = torch.zeros(len(batch_np), max_len, dtype=torch.long, device=device)
        for i, length in enumerate(batch_lengths):
            attention_mask[i, :length] = 1
        
        outputs = model(batch_tensor, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        frame_length = hidden_states[1].shape[1]
        frame_attention_mask = model._get_feature_vector_attention_mask(
            frame_length,
            attention_mask,
        )
        frame_mask = frame_attention_mask.unsqueeze(-1).to(hidden_states[1].dtype)

        for layer_idx in range(num_layers):
            hs = hidden_states[layer_idx + 1]  # Skip CNN output (layer 0 is CNN)
            # Masked mean pooling at encoder-frame resolution.
            masked_hs = hs * frame_mask
            valid_frame_count = frame_mask.sum(dim=1).clamp(min=1.0)
            pooled = (masked_hs.sum(dim=1) / valid_frame_count).cpu().numpy()
            layer_features[layer_idx].append(pooled)

    return {k: np.concatenate(v, axis=0) for k, v in layer_features.items()}


def extract_learned_layer_weights(checkpoint_path: Path) -> Optional[np.ndarray]:
    """Extract learned layer weights from a DANN checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Look for layer pooling weights
        for key in state_dict:
            if "layer_pooling.weights" in key:
                weights = state_dict[key].numpy()
                # Apply softmax
                weights = np.exp(weights) / np.exp(weights).sum()
                return weights

        logger.warning(f"No layer weights found in {checkpoint_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def train_probe(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Train a logistic regression probe."""
    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2 or counts.min() < 4:
        return {"accuracy": float("nan"), "accuracy_std": float("nan"), "status": "skipped"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    scores = []

    for train_idx, test_idx in sss.split(X_scaled, y):
        clf = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
        clf.fit(X_scaled[train_idx], y[train_idx])
        scores.append(clf.score(X_scaled[test_idx], y[test_idx]))

    return {
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "scores": [float(s) for s in scores],
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_layer_comparison(
    wavlm_results: dict,
    w2v2_results: dict,
    output_path: Path,
    wavlm_weights: Optional[np.ndarray] = None,
    w2v2_weights: Optional[np.ndarray] = None,
):
    """Create comprehensive comparison plot."""
    num_layers = len(wavlm_results)
    layers = list(range(num_layers))

    wavlm_acc = [wavlm_results[l]["accuracy"] for l in layers]
    w2v2_acc = [w2v2_results[l]["accuracy"] for l in layers]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Layer-wise codec probe accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(num_layers)
    width = 0.35

    bars1 = ax1.bar(x - width/2, wavlm_acc, width, label='WavLM', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, w2v2_acc, width, label='wav2vec2', color='coral', alpha=0.8)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_xlabel('Transformer Layer')
    ax1.set_ylabel('Codec Classification Accuracy')
    ax1.set_title('Layer-wise Codec Encoding Strength')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{i}' for i in layers])
    ax1.legend()
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Difference plot
    ax2 = axes[0, 1]
    diff = np.array(wavlm_acc) - np.array(w2v2_acc)
    colors = ['steelblue' if d > 0 else 'coral' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Transformer Layer')
    ax2.set_ylabel('WavLM - wav2vec2 Accuracy Difference')
    ax2.set_title('Codec Encoding Difference (positive = WavLM encodes more)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i}' for i in layers])
    ax2.grid(axis='y', alpha=0.3)

    # 3. Learned layer weights (if available)
    ax3 = axes[1, 0]
    if wavlm_weights is not None and w2v2_weights is not None:
        ax3.bar(x - width/2, wavlm_weights, width, label='WavLM DANN', color='steelblue', alpha=0.8)
        ax3.bar(x + width/2, w2v2_weights, width, label='wav2vec2 DANN', color='coral', alpha=0.8)
        ax3.set_xlabel('Transformer Layer')
        ax3.set_ylabel('Layer Weight (softmax)')
        ax3.set_title('Learned Layer Mixing Weights (from DANN training)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'L{i}' for i in layers])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No checkpoint weights available\n(use --wavlm-checkpoint and --w2v2-checkpoint)',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Learned Layer Mixing Weights')

    # 4. Correlation analysis
    ax4 = axes[1, 1]
    ax4.scatter(wavlm_acc, w2v2_acc, s=100, alpha=0.7, c=layers, cmap='viridis')
    for i, (wa, w2a) in enumerate(zip(wavlm_acc, w2v2_acc)):
        ax4.annotate(f'L{i}', (wa, w2a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Add diagonal line
    min_val = min(min(wavlm_acc), min(w2v2_acc))
    max_val = max(max(wavlm_acc), max(w2v2_acc))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal encoding')
    
    ax4.set_xlabel('WavLM Codec Accuracy')
    ax4.set_ylabel('wav2vec2 Codec Accuracy')
    ax4.set_title('Layer-wise Encoding Correlation')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")


def plot_heatmap_comparison(
    wavlm_results: dict,
    w2v2_results: dict,
    output_path: Path,
):
    """Create heatmap comparison of codec encoding."""
    num_layers = len(wavlm_results)
    layers = list(range(num_layers))

    wavlm_acc = [wavlm_results[l]["accuracy"] for l in layers]
    w2v2_acc = [w2v2_results[l]["accuracy"] for l in layers]

    data = np.array([wavlm_acc, w2v2_acc])

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=0.4, vmax=1.0)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['WavLM', 'wav2vec2'])
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f'L{i}' for i in layers])
    ax.set_xlabel('Transformer Layer')
    ax.set_title('Codec Encoding Heatmap (accuracy)')

    # Add values
    for i in range(2):
        for j in range(num_layers):
            color = 'white' if data[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved heatmap: {output_path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_results(
    wavlm_results: dict,
    w2v2_results: dict,
) -> dict:
    """Compute summary statistics and insights."""
    num_layers = len(wavlm_results)
    layers = list(range(num_layers))

    wavlm_acc = np.array([wavlm_results[l]["accuracy"] for l in layers])
    w2v2_acc = np.array([w2v2_results[l]["accuracy"] for l in layers])

    # Find peaks
    wavlm_peak_layer = int(np.argmax(wavlm_acc))
    w2v2_peak_layer = int(np.argmax(w2v2_acc))

    # Compute statistics
    analysis = {
        "wavlm": {
            "mean_accuracy": float(np.mean(wavlm_acc)),
            "std_accuracy": float(np.std(wavlm_acc)),
            "peak_layer": wavlm_peak_layer,
            "peak_accuracy": float(wavlm_acc[wavlm_peak_layer]),
            "early_layers_mean": float(np.mean(wavlm_acc[:4])),  # L0-L3
            "middle_layers_mean": float(np.mean(wavlm_acc[4:8])),  # L4-L7
            "late_layers_mean": float(np.mean(wavlm_acc[8:])),  # L8-L11
        },
        "w2v2": {
            "mean_accuracy": float(np.mean(w2v2_acc)),
            "std_accuracy": float(np.std(w2v2_acc)),
            "peak_layer": w2v2_peak_layer,
            "peak_accuracy": float(w2v2_acc[w2v2_peak_layer]),
            "early_layers_mean": float(np.mean(w2v2_acc[:4])),
            "middle_layers_mean": float(np.mean(w2v2_acc[4:8])),
            "late_layers_mean": float(np.mean(w2v2_acc[8:])),
        },
        "comparison": {
            "mean_difference": float(np.mean(wavlm_acc) - np.mean(w2v2_acc)),
            "correlation": float(np.corrcoef(wavlm_acc, w2v2_acc)[0, 1]),
            "layers_where_wavlm_higher": [int(l) for l in layers if wavlm_acc[l] > w2v2_acc[l]],
            "layers_where_w2v2_higher": [int(l) for l in layers if w2v2_acc[l] > wavlm_acc[l]],
        }
    }

    return analysis


def generate_hypothesis_report(analysis: dict) -> str:
    """Generate a hypothesis report based on the analysis."""
    report = []
    report.append("=" * 70)
    report.append("BACKBONE COMPARISON: CODEC ENCODING ANALYSIS")
    report.append("=" * 70)
    report.append("")

    # Summary stats
    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"WavLM  - Mean accuracy: {analysis['wavlm']['mean_accuracy']:.3f} ± {analysis['wavlm']['std_accuracy']:.3f}")
    report.append(f"         Peak: Layer {analysis['wavlm']['peak_layer']} ({analysis['wavlm']['peak_accuracy']:.3f})")
    report.append(f"wav2vec2 - Mean accuracy: {analysis['w2v2']['mean_accuracy']:.3f} ± {analysis['w2v2']['std_accuracy']:.3f}")
    report.append(f"         Peak: Layer {analysis['w2v2']['peak_layer']} ({analysis['w2v2']['peak_accuracy']:.3f})")
    report.append("")

    # Layer distribution
    report.append("LAYER-WISE DISTRIBUTION")
    report.append("-" * 40)
    report.append(f"WavLM  - Early (L0-3): {analysis['wavlm']['early_layers_mean']:.3f}, "
                 f"Mid (L4-7): {analysis['wavlm']['middle_layers_mean']:.3f}, "
                 f"Late (L8-11): {analysis['wavlm']['late_layers_mean']:.3f}")
    report.append(f"wav2vec2 - Early (L0-3): {analysis['w2v2']['early_layers_mean']:.3f}, "
                 f"Mid (L4-7): {analysis['w2v2']['middle_layers_mean']:.3f}, "
                 f"Late (L8-11): {analysis['w2v2']['late_layers_mean']:.3f}")
    report.append("")

    # Hypotheses
    report.append("HYPOTHESES FOR DANN PERFORMANCE DIFFERENCE")
    report.append("-" * 40)

    # Hypothesis 1: Overall encoding strength
    if analysis['wavlm']['mean_accuracy'] < analysis['w2v2']['mean_accuracy']:
        report.append("1. WavLM encodes LESS codec information overall")
        report.append("   → Denoising pretraining may have learned some codec invariance")
        report.append("   → Less signal for DANN to adversarially remove = easier task")
    else:
        report.append("1. WavLM encodes MORE codec information overall")
        report.append("   → But DANN still works better - suggests layer weighting matters more")

    # Hypothesis 2: Layer concentration
    wavlm_range = analysis['wavlm']['peak_accuracy'] - min(
        analysis['wavlm']['early_layers_mean'],
        analysis['wavlm']['middle_layers_mean'],
        analysis['wavlm']['late_layers_mean']
    )
    w2v2_range = analysis['w2v2']['peak_accuracy'] - min(
        analysis['w2v2']['early_layers_mean'],
        analysis['w2v2']['middle_layers_mean'],
        analysis['w2v2']['late_layers_mean']
    )

    report.append("")
    if wavlm_range < w2v2_range:
        report.append("2. WavLM has MORE UNIFORM codec encoding across layers")
        report.append("   → Easier for layer weighting to find good combination")
        report.append("   → wav2vec2's peaked distribution may concentrate codec info in specific layers")
    else:
        report.append("2. wav2vec2 has MORE UNIFORM codec encoding across layers")
        report.append("   → May make it harder to find layers that avoid codec information")

    # Hypothesis 3: Gated relative position bias
    report.append("")
    report.append("3. ARCHITECTURAL FACTOR: Gated Relative Position Bias")
    report.append("   → WavLM's gated mechanism may adaptively weight local context")
    report.append("   → Could help separate content from acoustic artifacts (codecs)")
    report.append("   → wav2vec2's conv pos embeddings are less adaptive")

    # Recommendations
    report.append("")
    report.append("RECOMMENDATIONS FOR IMPROVING wav2vec2 DANN")
    report.append("-" * 40)

    # layers_where_wavlm_higher = layers where WavLM has higher probe accuracy
    # Higher probe accuracy = more codec info encoded (easier to classify codec)
    # So these are layers where wav2vec2 encodes LESS codec info than WavLM
    w2v2_low_codec_layers = analysis['comparison']['layers_where_wavlm_higher']
    if w2v2_low_codec_layers:
        report.append(f"1. Focus on layers {w2v2_low_codec_layers} where wav2vec2 encodes less codec info")
        report.append("   → Consider 'specific' layer selection with these layers")

    report.append("2. Try unfreezing top layers of wav2vec2 backbone")
    report.append("   → May allow learning codec-invariant representations")

    report.append("3. Consider stronger gradient reversal (higher lambda)")
    report.append("   → wav2vec2's codec encoding may need stronger adversarial pressure")

    report.append("4. Experiment with different layer selection strategies")
    report.append("   → 'first_k' or 'specific' instead of 'weighted' for all layers")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output directory
    if args.output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        args.output_dir = project_root / "results" / "backbone_comparison"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("WavLM vs Wav2Vec2 Codec Probe Comparison")
    logger.info("=" * 60)
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Data source: {'synthetic' if args.synthetic else f'ASVspoof5 {args.split}'}")
    logger.info(f"Output: {args.output_dir}")

    # Check ffmpeg
    if not check_ffmpeg():
        logger.error("ffmpeg not found. Install it first.")
        return 1

    supported_codecs = check_codec_support()
    available = [c for c, ok in supported_codecs.items() if ok]
    logger.info(f"Available codecs: {available}")

    if not available:
        logger.error("No codecs available!")
        return 1

    # Step 1: Load audio
    logger.info("\n--- Step 1: Loading audio ---")
    if args.synthetic:
        waveforms = generate_synthetic_audio(args.num_samples, args.seed)
    else:
        waveforms = load_real_audio(args.split, args.num_samples, args.seed)

    # Step 2: Create augmented dataset
    logger.info("\n--- Step 2: Creating codec-augmented pairs ---")
    all_waveforms, binary_labels, multiclass_labels = create_augmented_dataset(
        waveforms, supported_codecs, seed=args.seed
    )

    # Step 3: Extract features from both backbones
    logger.info("\n--- Step 3: Extracting features ---")
    
    # All waveforms are padded to MAX_SAMPLES, so lengths are uniform
    # (load_real_audio and generate_synthetic_audio both pad to MAX_SAMPLES)
    original_lengths = [MAX_SAMPLES] * len(all_waveforms)

    # WavLM
    logger.info("Processing WavLM...")
    wavlm_model, wavlm_cfg = load_backbone("wavlm")
    wavlm_features = extract_all_layer_features(wavlm_model, all_waveforms, original_lengths, args.batch_size)
    del wavlm_model

    # Wav2Vec2
    logger.info("Processing wav2vec2...")
    w2v2_model, w2v2_cfg = load_backbone("w2v2")
    w2v2_features = extract_all_layer_features(w2v2_model, all_waveforms, original_lengths, args.batch_size)
    del w2v2_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 4: Train probes
    logger.info("\n--- Step 4: Training probes ---")
    wavlm_results = {}
    w2v2_results = {}

    for layer in range(12):
        logger.info(f"  Layer {layer}...")
        wavlm_results[layer] = train_probe(wavlm_features[layer], binary_labels, args.seed)
        w2v2_results[layer] = train_probe(w2v2_features[layer], binary_labels, args.seed)

    # Step 5: Load learned layer weights (if checkpoints provided)
    wavlm_weights = None
    w2v2_weights = None
    if args.wavlm_checkpoint:
        wavlm_weights = extract_learned_layer_weights(args.wavlm_checkpoint)
    if args.w2v2_checkpoint:
        w2v2_weights = extract_learned_layer_weights(args.w2v2_checkpoint)

    # Step 6: Analysis
    logger.info("\n--- Step 5: Analysis ---")
    analysis = analyze_results(wavlm_results, w2v2_results)
    report = generate_hypothesis_report(analysis)
    print(report)

    # Step 7: Save results
    logger.info("\n--- Step 6: Saving results ---")

    # Plots
    plot_layer_comparison(
        wavlm_results, w2v2_results,
        args.output_dir / "backbone_comparison.png",
        wavlm_weights, w2v2_weights
    )
    plot_heatmap_comparison(
        wavlm_results, w2v2_results,
        args.output_dir / "codec_encoding_heatmap.png"
    )

    # JSON results
    results = {
        "wavlm": {
            "description": BACKBONES["wavlm"]["description"],
            "per_layer": {str(k): v for k, v in wavlm_results.items()},
        },
        "w2v2": {
            "description": BACKBONES["w2v2"]["description"],
            "per_layer": {str(k): v for k, v in w2v2_results.items()},
        },
        "analysis": analysis,
        "config": {
            "num_samples": len(waveforms),
            "data_source": "synthetic" if args.synthetic else f"asvspoof5_{args.split}",
            "codecs_available": available,
            "seed": args.seed,
        }
    }

    with open(args.output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save report
    with open(args.output_dir / "hypothesis_report.txt", "w") as f:
        f.write(report)

    logger.info(f"\nAll results saved to: {args.output_dir}")
    logger.info("  - backbone_comparison.png")
    logger.info("  - codec_encoding_heatmap.png")
    logger.info("  - comparison_results.json")
    logger.info("  - hypothesis_report.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
