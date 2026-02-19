#!/usr/bin/env python3
"""Visualize codec augmentation effects on audio spectrograms.

This script generates a publication-ready figure showing the effect of
codec augmentation on audio samples. Used as a sanity check to prove
augmentation is actually being applied during training.

The figure shows spectrograms of:
1. Original (NONE) - uncompressed audio
2. MP3 compressed - lossy compression artifacts
3. AAC compressed - different compression characteristics

Usage:
    # Visualize random samples from training set
    python scripts/visualize_augmentation.py \
        --n-samples 3 \
        --output figures/augmentation_examples.png

    # Visualize specific file
    python scripts/visualize_augmentation.py \
        --audio-file /path/to/audio.flac \
        --output figures/augmentation_single.png

    # With specific quality level
    python scripts/visualize_augmentation.py \
        --n-samples 3 \
        --quality 3 \
        --output figures/augmentation_q3.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asvspoof5_domain_invariant_cm.data.codec_augment import CodecAugmentor, CodecAugmentConfig
from asvspoof5_domain_invariant_cm.data.audio import load_waveform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style Configuration (matching rq3_combined.py)
# ---------------------------------------------------------------------------
COLORS = {
    "none": "#4C72B0",       # Steel blue (original)
    "mp3": "#DD8452",        # Coral/orange
    "aac": "#55A868",        # Green
    "opus": "#C44E52",       # Red
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
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Spectrogram parameters
SPEC_PARAMS = {
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "fmin": 20,
    "fmax": 8000,
}


# ---------------------------------------------------------------------------
# Spectrogram Computation
# ---------------------------------------------------------------------------
def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Compute mel spectrogram from waveform.
    
    Returns log-mel spectrogram in dB scale.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for spectrogram computation")
    
    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=SPEC_PARAMS["n_fft"],
        hop_length=SPEC_PARAMS["hop_length"],
        n_mels=SPEC_PARAMS["n_mels"],
        fmin=SPEC_PARAMS["fmin"],
        fmax=SPEC_PARAMS["fmax"],
    )
    
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def compute_difference_spectrogram(
    spec_original: np.ndarray,
    spec_augmented: np.ndarray,
) -> np.ndarray:
    """Compute absolute difference between spectrograms."""
    return np.abs(spec_original - spec_augmented)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_spectrogram(
    ax: plt.Axes,
    spectrogram: np.ndarray,
    title: str,
    sample_rate: int = 16000,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
) -> None:
    """Plot a spectrogram on the given axes."""
    hop_length = SPEC_PARAMS["hop_length"]
    
    # Compute time axis
    n_frames = spectrogram.shape[1]
    duration = n_frames * hop_length / sample_rate
    
    # Plot
    img = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, duration, 0, SPEC_PARAMS["fmax"] / 1000],
    )
    
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    
    if colorbar:
        plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)


def create_augmentation_figure(
    audio_path: Path,
    augmentor: CodecAugmentor,
    quality: int = 3,
    sample_rate: int = 16000,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Create figure showing original vs augmented spectrograms.
    
    Layout:
    Row 1: Original | MP3 | AAC
    Row 2: Diff (Orig-MP3) | Diff (Orig-AAC) | Waveform comparison
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    # Load original audio
    waveform_orig, sr = load_waveform(str(audio_path), target_sr=sample_rate)
    if isinstance(waveform_orig, np.ndarray) is False:
        waveform_orig = waveform_orig.numpy()
    waveform_orig = waveform_orig.squeeze()
    
    # Apply augmentations
    augmented = {}
    for codec in ["MP3", "AAC"]:
        if codec in augmentor.supported_codecs:
            aug_waveform, _, _ = augmentor.apply_codec(
                waveform_orig, sample_rate, codec, quality
            )
            augmented[codec] = aug_waveform
        else:
            logger.warning(f"Codec {codec} not supported, skipping")
    
    if not augmented:
        raise RuntimeError("No codecs available for augmentation")
    
    # Compute spectrograms
    spec_orig = compute_mel_spectrogram(waveform_orig, sample_rate)
    specs_aug = {
        codec: compute_mel_spectrogram(wav, sample_rate)
        for codec, wav in augmented.items()
    }
    
    # Create figure
    n_codecs = len(augmented)
    fig, axes = plt.subplots(2, n_codecs + 1, figsize=figsize)
    
    # Determine global color scale for spectrograms
    all_specs = [spec_orig] + list(specs_aug.values())
    vmin = min(s.min() for s in all_specs)
    vmax = max(s.max() for s in all_specs)
    
    # Row 1: Spectrograms
    plot_spectrogram(
        axes[0, 0], spec_orig, "Original (NONE)",
        sample_rate, vmin=vmin, vmax=vmax,
    )
    
    for idx, (codec, spec) in enumerate(specs_aug.items()):
        color_key = codec.lower()
        plot_spectrogram(
            axes[0, idx + 1], spec, f"{codec} (Q{quality})",
            sample_rate, vmin=vmin, vmax=vmax,
        )
    
    # Row 2: Difference spectrograms and waveform
    for idx, (codec, spec) in enumerate(specs_aug.items()):
        diff = compute_difference_spectrogram(spec_orig, spec)
        plot_spectrogram(
            axes[1, idx], diff, f"Difference: Original - {codec}",
            sample_rate, cmap="hot", vmin=0, vmax=20,
        )
    
    # Waveform comparison in last column
    ax_wave = axes[1, -1]
    time = np.arange(len(waveform_orig)) / sample_rate
    
    # Plot subset for clarity (first 0.5s)
    max_samples = int(0.5 * sample_rate)
    time_sub = time[:max_samples]
    
    ax_wave.plot(time_sub, waveform_orig[:max_samples], 
                 color=COLORS["none"], alpha=0.7, linewidth=0.5, label="Original")
    
    for codec, wav in augmented.items():
        ax_wave.plot(time_sub, wav[:max_samples],
                     color=COLORS[codec.lower()], alpha=0.7, linewidth=0.5, label=codec)
    
    ax_wave.set_title("Waveform (first 0.5s)", fontweight="bold")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_xlim(0, 0.5)
    
    # Empty last cell in row 1 if only one codec
    if n_codecs < 2:
        axes[0, -1].axis("off")
    
    # Overall title
    filename = audio_path.name
    fig.suptitle(
        f"Codec Augmentation Visualization: {filename}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    
    plt.tight_layout()
    return fig


def create_multi_sample_figure(
    audio_paths: list[Path],
    augmentor: CodecAugmentor,
    quality: int = 3,
    sample_rate: int = 16000,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Create figure showing multiple samples with original and augmented.
    
    Layout: One row per sample, columns for NONE | MP3 | AAC
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    n_samples = len(audio_paths)
    codecs = [c for c in ["MP3", "AAC"] if c in augmentor.supported_codecs]
    n_cols = 1 + len(codecs)  # Original + codecs
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, audio_path in enumerate(audio_paths):
        # Load original
        waveform_orig, sr = load_waveform(str(audio_path), target_sr=sample_rate)
        if hasattr(waveform_orig, 'numpy'):
            waveform_orig = waveform_orig.numpy()
        waveform_orig = waveform_orig.squeeze()
        
        spec_orig = compute_mel_spectrogram(waveform_orig, sample_rate)
        
        # Plot original
        title = "Original" if row == 0 else ""
        ylabel = audio_path.stem[:15] + "..." if len(audio_path.stem) > 15 else audio_path.stem
        plot_spectrogram(axes[row, 0], spec_orig, title, sample_rate, colorbar=False)
        axes[row, 0].set_ylabel(ylabel, fontsize=9)
        
        # Plot augmented versions
        for col, codec in enumerate(codecs, start=1):
            aug_waveform, _, _ = augmentor.apply_codec(
                waveform_orig, sample_rate, codec, quality
            )
            spec_aug = compute_mel_spectrogram(aug_waveform, sample_rate)
            
            title = f"{codec} (Q{quality})" if row == 0 else ""
            plot_spectrogram(axes[row, col], spec_aug, title, sample_rate, colorbar=False)
            axes[row, col].set_ylabel("")
    
    # Overall title
    fig.suptitle(
        f"Codec Augmentation Examples (Quality={quality})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize codec augmentation effects on spectrograms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    p.add_argument(
        "--audio-file",
        type=Path,
        help="Single audio file to visualize",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of random samples from training set (default: 3)",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Codec quality level (default: 3)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/augmentation_examples.png"),
        help="Output figure path",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate (default: 16000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create augmentor
    config = CodecAugmentConfig(
        codec_prob=1.0,  # Always apply for visualization
        codecs=["MP3", "AAC", "OPUS"],
        qualities=[1, 2, 3, 4, 5],
        sample_rate=args.sample_rate,
    )
    
    try:
        augmentor = CodecAugmentor(config)
    except Exception as e:
        logger.error(f"Failed to create augmentor: {e}")
        logger.error("Make sure ffmpeg is installed with codec support")
        return 1
    
    logger.info(f"Augmentor initialized: supported_codecs={augmentor.supported_codecs}")
    
    if not augmentor.supported_codecs:
        logger.error("No codecs supported! Check ffmpeg installation.")
        return 1
    
    # Get audio files
    if args.audio_file:
        audio_paths = [args.audio_file]
    else:
        # Get random samples from training manifest
        manifest_path = Path(os.environ.get("ASVSPOOF5_ROOT", "")) / "manifests" / "train.parquet"
        
        if not manifest_path.exists():
            logger.error(f"Training manifest not found: {manifest_path}")
            logger.error("Set ASVSPOOF5_ROOT or use --audio-file")
            return 1
        
        import pandas as pd
        df = pd.read_parquet(manifest_path)
        
        # Random sample
        indices = np.random.choice(len(df), min(args.n_samples, len(df)), replace=False)
        audio_paths = [Path(df.iloc[i]["audio_path"]) for i in indices]
        
        logger.info(f"Selected {len(audio_paths)} random samples")
    
    # Verify files exist
    for path in audio_paths:
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return 1
    
    # Create figure
    if len(audio_paths) == 1:
        fig = create_augmentation_figure(
            audio_paths[0], augmentor, args.quality, args.sample_rate,
        )
    else:
        fig = create_multi_sample_figure(
            audio_paths, augmentor, args.quality, args.sample_rate,
        )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure: {args.output}")
    
    # Also save PDF
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info(f"Saved PDF: {pdf_path}")
    
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
