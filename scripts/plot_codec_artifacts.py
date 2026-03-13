#!/usr/bin/env python3
"""Visualize codec artifacts: waveform, spectrogram, and residual per codec.

Creates a 12-row × 3-column grid showing how each ASVspoof 5 codec condition
affects the same speaker's bonafide audio.

Usage:
    python scripts/plot_codec_artifacts.py \
        --data-root /projects/prjs1904/data/asvspoof5 \
        --predictions results/runs/wavlm_erm/eval_eval_full/predictions.tsv \
        --speaker E_0002 \
        --output figures/codec_artifacts

Columns: Waveform | Spectrogram | Residual (vs NONE reference)
Rows: NONE, C01, C02, ..., C11
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import STYLE, set_style

# ASVspoof 5 codec labels (human-readable)
CODEC_LABELS = {
    'NONE': 'No codec (original)',
    'C01': 'C01: mp3 (8kHz)',
    'C02': 'C02: aac (8kHz)',
    'C03': 'C03: ogg (8kHz)',
    'C04': 'C04: Encodec',
    'C05': 'C05: mp3 (16kHz)',
    'C06': 'C06: aac (16kHz)',
    'C07': 'C07: mp3+Encodec',
    'C08': 'C08: G.722',
    'C09': 'C09: GSM-FR',
    'C10': 'C10: G.711 μ-law',
    'C11': 'C11: Device effects',
}

CODEC_ORDER = ['NONE', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
               'C08', 'C09', 'C10', 'C11']


def load_audio(flac_path: Path, target_sr: int = 16000):
    """Load a FLAC file and return waveform + sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(flac_path))
    except ImportError:
        import librosa
        audio, sr = librosa.load(str(flac_path), sr=target_sr)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono
    return audio, sr


def compute_spectrogram(audio, sr, n_fft=1024, hop_length=256):
    """Compute log-magnitude spectrogram."""
    # Use numpy STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    window = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        fft = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft)
    # Log magnitude
    spec_db = 20 * np.log10(np.maximum(spec, 1e-10))
    return spec_db


def align_and_trim(audio, ref_audio, max_samples=None):
    """Trim both signals to the same length."""
    min_len = min(len(audio), len(ref_audio))
    if max_samples is not None:
        min_len = min(min_len, max_samples)
    return audio[:min_len], ref_audio[:min_len]


def main():
    parser = argparse.ArgumentParser(description='Visualize codec artifacts')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to ASVspoof5 dataset root')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions.tsv with codec labels')
    parser.add_argument('--speaker', type=str, default='E_0002',
                        help='Speaker ID to visualize')
    parser.add_argument('--output', type=str, default='figures/codec_artifacts',
                        help='Output path (without extension)')
    parser.add_argument('--max-seconds', type=float, default=3.0,
                        help='Max duration to display (seconds)')
    args = parser.parse_args()

    set_style()

    data_root = Path(args.data_root)
    # Eval FLAC files are in flac_E_eval/
    eval_dir = data_root / 'flac_E_eval'
    if not eval_dir.exists():
        # Try alternative structure
        for candidate in ['flac_E', 'eval']:
            if (data_root / candidate).exists():
                eval_dir = data_root / candidate
                break

    # Load predictions to find utterances
    import pandas as pd
    df = pd.read_csv(args.predictions, sep='\t')
    bon = df[(df['y_task'] == 0) & (df['speaker_id'] == args.speaker)]

    if len(bon) == 0:
        print(f'Error: no bonafide utterances for speaker {args.speaker}')
        sys.exit(1)

    # Pick one utterance per codec
    utterances = {}
    for codec in CODEC_ORDER:
        subset = bon[bon['codec'] == codec]
        if len(subset) == 0:
            print(f'Warning: no {codec} utterance for speaker {args.speaker}')
            continue
        utterances[codec] = subset.iloc[0]['flac_file']

    print(f'Speaker {args.speaker}: found {len(utterances)}/{len(CODEC_ORDER)} codecs')

    # Load reference (NONE) audio
    if 'NONE' not in utterances:
        print('Error: no NONE (uncoded) reference utterance found')
        sys.exit(1)

    ref_path = eval_dir / f'{utterances["NONE"]}.flac'
    if not ref_path.exists():
        print(f'Error: reference file not found: {ref_path}')
        sys.exit(1)

    ref_audio, sr = load_audio(ref_path)
    max_samples = int(args.max_seconds * sr)
    ref_audio = ref_audio[:max_samples]
    print(f'Reference: {ref_path.name} ({len(ref_audio)/sr:.2f}s, {sr}Hz)')

    # Create figure: 12 rows × 3 columns
    n_codecs = len(utterances)
    fig = plt.figure(figsize=(16, 2.2 * n_codecs))
    gs = gridspec.GridSpec(n_codecs, 3, figure=fig,
                           width_ratios=[1, 1.2, 1],
                           hspace=0.35, wspace=0.25)

    # Column headers
    col_titles = ['Waveform', 'Spectrogram', 'Residual vs Original']

    for row_idx, codec in enumerate(CODEC_ORDER):
        if codec not in utterances:
            continue

        flac_path = eval_dir / f'{utterances[codec]}.flac'
        if not flac_path.exists():
            print(f'Warning: file not found: {flac_path}')
            continue

        audio, _ = load_audio(flac_path)
        audio = audio[:max_samples]
        t = np.arange(len(audio)) / sr

        label = CODEC_LABELS.get(codec, codec)
        print(f'  {codec}: {flac_path.name} ({len(audio)/sr:.2f}s)')

        # ── Column 1: Waveform ──
        ax_wave = fig.add_subplot(gs[row_idx, 0])
        ax_wave.plot(t, audio, color='#4CA08A', linewidth=0.3, alpha=0.8)
        ax_wave.set_ylabel(label, fontsize=8, fontweight='bold', rotation=0,
                          labelpad=120, ha='left', va='center')
        ax_wave.set_ylim(-1, 1)
        ax_wave.set_xlim(0, args.max_seconds)
        if row_idx == 0:
            ax_wave.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=10)
        if row_idx < n_codecs - 1:
            ax_wave.set_xticklabels([])
        else:
            ax_wave.set_xlabel('Time (s)', fontsize=9)
        ax_wave.tick_params(labelsize=7)

        # ── Column 2: Spectrogram ──
        ax_spec = fig.add_subplot(gs[row_idx, 1])
        spec = compute_spectrogram(audio, sr)
        freq_bins = np.linspace(0, sr / 2, spec.shape[0])
        time_bins = np.linspace(0, len(audio) / sr, spec.shape[1])
        vmin, vmax = -80, 0
        ax_spec.pcolormesh(time_bins, freq_bins / 1000, spec,
                          cmap='magma', vmin=vmin, vmax=vmax,
                          shading='gouraud', rasterized=True)
        ax_spec.set_ylim(0, 8)
        if row_idx == 0:
            ax_spec.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=10)
        if row_idx < n_codecs - 1:
            ax_spec.set_xticklabels([])
        else:
            ax_spec.set_xlabel('Time (s)', fontsize=9)
        ax_spec.set_ylabel('kHz', fontsize=8)
        ax_spec.tick_params(labelsize=7)

        # ── Column 3: Residual ──
        ax_res = fig.add_subplot(gs[row_idx, 2])
        if codec == 'NONE':
            # No residual for reference — show flat line
            ax_res.axhline(y=0, color='#9CA3AF', linewidth=0.5)
            ax_res.text(0.5, 0.5, '(reference)', transform=ax_res.transAxes,
                       ha='center', va='center', fontsize=9, color='#9CA3AF')
        else:
            # Compute residual against NONE reference
            # Note: different utterances, so residual shows structural differences
            # rather than per-sample codec artifacts. Use spectrogram difference instead.
            spec_ref = compute_spectrogram(ref_audio, sr)
            # Align spectrograms to shorter length
            min_frames = min(spec.shape[1], spec_ref.shape[1])
            residual = spec[:, :min_frames] - spec_ref[:, :min_frames]
            time_res = np.linspace(0, min(len(audio), len(ref_audio)) / sr, min_frames)
            ax_res.pcolormesh(time_res, freq_bins / 1000, residual,
                            cmap='RdBu_r', vmin=-30, vmax=30,
                            shading='gouraud', rasterized=True)
            ax_res.set_ylim(0, 8)

        if row_idx == 0:
            ax_res.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=10)
        if row_idx < n_codecs - 1:
            ax_res.set_xticklabels([])
        else:
            ax_res.set_xlabel('Time (s)', fontsize=9)
        if codec != 'NONE':
            ax_res.set_ylabel('kHz', fontsize=8)
        ax_res.tick_params(labelsize=7)

    fig.suptitle(f'Codec Artifacts: Speaker {args.speaker} (bonafide)',
                fontsize=14, fontweight='bold', y=1.0)

    # Save
    output_base = args.output.removesuffix('.png').removesuffix('.pdf')
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.pdf']:
        out_path = output_base + ext
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
