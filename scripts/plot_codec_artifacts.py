#!/usr/bin/env python3
"""Visualize codec artifacts: waveform, spectrogram, and spectral envelope.

Creates a 12-row × 3-column grid showing how each ASVspoof 5 codec condition
affects the same speaker's bonafide audio. Rows are grouped by codec category.

Usage:
    python scripts/plot_codec_artifacts.py \
        --data-root /projects/prjs1904/data/asvspoof5 \
        --predictions results/runs/wavlm_erm/eval_eval_full/predictions.tsv \
        --speaker E_0002 \
        --output figures/codec_artifacts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import COLORS, STYLE, set_style


# ── Codec metadata ───────────────────────────────────────────────────────────
CODEC_LABELS = {
    'NONE': 'No Codec (Original)',
    'C01':  'C01 · Opus WB',
    'C02':  'C02 · AMR-WB',
    'C03':  'C03 · Speex WB',
    'C04':  'C04 · Encodec',
    'C05':  'C05 · MP3 WB',
    'C06':  'C06 · AAC WB',
    'C07':  'C07 · MP3 + Encodec',
    'C08':  'C08 · Opus NB',
    'C09':  'C09 · AMR-NB',
    'C10':  'C10 · Speex NB',
    'C11':  'C11 · Device Effects',
}

CODEC_ORDER = ['NONE', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
               'C08', 'C09', 'C10', 'C11']

# Group boundaries (draw separator BEFORE these indices)
# Groups: Original | Wideband (C01-C07) | Narrowband (C08-C10) | Device (C11)
GROUP_STARTS = {1: 'Wideband Codecs', 8: 'Narrowband Codecs', 11: 'Device'}

CODEC_COLORS = {
    'NONE': '#333333',
    'C01':  '#D4795A', 'C02': '#E8946E', 'C03': '#F0A882',
    'C04':  '#7C3AED',
    'C05':  '#4CA08A', 'C06': '#6BC4AE',
    'C07':  '#9B59B6',
    'C08':  '#3B82F6', 'C09': '#60A5FA', 'C10': '#93C5FD',
    'C11':  '#F59E0B',
}


# ── Audio helpers ────────────────────────────────────────────────────────────
def load_audio(flac_path: Path, target_sr: int = 16000):
    try:
        import soundfile as sf
        audio, sr = sf.read(str(flac_path))
    except ImportError:
        import librosa
        audio, sr = librosa.load(str(flac_path), sr=target_sr)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return audio, sr


def compute_spectrogram(audio, sr, n_fft=1024, hop_length=256):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    return 20 * np.log10(np.maximum(spec, 1e-10))


def compute_spectral_envelope(spec_db):
    return np.mean(spec_db, axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize codec artifacts')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--speaker', type=str, default='E_0002')
    parser.add_argument('--output', type=str, default='figures/codec_artifacts')
    parser.add_argument('--label', type=str, default='bonafide', choices=['bonafide', 'spoof'],
                        help='Filter by bonafide (y_task=0) or spoof (y_task=1)')
    parser.add_argument('--max-seconds', type=float, default=3.0)
    args = parser.parse_args()

    set_style()

    data_root = Path(args.data_root)
    eval_dir = data_root / 'flac_E_eval'
    if not eval_dir.exists():
        for candidate in ['flac_E', 'eval']:
            if (data_root / candidate).exists():
                eval_dir = data_root / candidate
                break

    import pandas as pd
    df = pd.read_csv(args.predictions, sep='\t')
    y_val = 0 if args.label == 'bonafide' else 1
    bon = df[(df['y_task'] == y_val) & (df['speaker_id'] == args.speaker)]

    utterances = {}
    for codec in CODEC_ORDER:
        subset = bon[bon['codec'] == codec]
        if len(subset) == 0:
            print(f'Warning: no {codec} utterance for speaker {args.speaker}')
            continue
        utterances[codec] = subset.iloc[0]['flac_file']

    print(f'Speaker {args.speaker}: {len(utterances)}/{len(CODEC_ORDER)} codecs')
    if 'NONE' not in utterances:
        print('Error: no NONE reference'); sys.exit(1)

    ref_path = eval_dir / f'{utterances["NONE"]}.flac'
    ref_audio, sr = load_audio(ref_path)
    max_samples = int(args.max_seconds * sr)
    ref_audio = ref_audio[:max_samples]
    ref_spec = compute_spectrogram(ref_audio, sr)
    ref_envelope = compute_spectral_envelope(ref_spec)
    freq_bins = np.linspace(0, sr / 2, ref_spec.shape[0])
    print(f'Reference: {ref_path.name} ({len(ref_audio)/sr:.2f}s, {sr}Hz)')

    # ── Figure layout: 12 rows × 3 cols + colorbar row ──────────────────────
    n_codecs = len(utterances)
    row_h = 1.8
    fig_h = row_h * n_codecs + 1.2
    fig = plt.figure(figsize=(15, fig_h))

    gs = gridspec.GridSpec(
        n_codecs + 1, 3, figure=fig,
        width_ratios=[1, 1.4, 0.9],
        height_ratios=[1] * n_codecs + [0.06],
        hspace=0.40, wspace=0.28,
    )

    vmin_spec, vmax_spec = -80, 0

    for row_idx, codec in enumerate(CODEC_ORDER):
        if codec not in utterances:
            continue

        flac_path = eval_dir / f'{utterances[codec]}.flac'
        if not flac_path.exists():
            print(f'Warning: {flac_path} not found'); continue

        audio, _ = load_audio(flac_path)
        audio = audio[:max_samples]
        t = np.arange(len(audio)) / sr
        spec = compute_spectrogram(audio, sr)
        envelope = compute_spectral_envelope(spec)
        label = CODEC_LABELS.get(codec, codec)
        color = CODEC_COLORS.get(codec, '#888888')
        print(f'  {codec}: {flac_path.name} ({len(audio)/sr:.2f}s)')

        # ── Col 0: Waveform ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.plot(t, audio, color=color, linewidth=0.25, alpha=0.85)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, args.max_seconds)
        ax.set_ylabel(label, fontsize=9, fontweight='bold',
                      rotation=0, labelpad=120, ha='left', va='center')
        ax.yaxis.set_label_coords(-0.58, 0.5)
        if row_idx == 0:
            ax.set_title('Waveform', fontsize=12, fontweight='bold', pad=8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_yticks([-0.5, 0, 0.5])

        # ── Col 1: Spectrogram ───────────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 1])
        time_bins = np.linspace(0, len(audio) / sr, spec.shape[1])
        im_spec = ax.pcolormesh(
            time_bins, freq_bins / 1000, spec,
            cmap='magma', vmin=vmin_spec, vmax=vmax_spec,
            shading='gouraud', rasterized=True,
        )
        ax.set_ylim(0, 8)
        if row_idx == 0:
            ax.set_title('Spectrogram', fontsize=12, fontweight='bold', pad=8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('kHz', fontsize=9)
        ax.tick_params(labelsize=8)

        # ── Col 2: Spectral envelope ─────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.plot(ref_envelope, freq_bins / 1000, color='#BBBBBB',
                linewidth=1.0, alpha=0.6, linestyle='--')
        ax.plot(envelope, freq_bins / 1000, color=color, linewidth=1.8, alpha=0.9)
        ax.set_ylim(0, 8)
        ax.set_xlim(0, -80)  # Flipped: high power (0) on left, low (-80) on right
        if row_idx == 0:
            ax.set_title('Spectral Envelope', fontsize=12, fontweight='bold', pad=8)
            # Legend
            ax.plot([], [], color='#BBBBBB', linewidth=1.0, linestyle='--',
                    label='Original (ref)')
            ax.plot([], [], color=color, linewidth=1.8, label='Codec')
            ax.legend(loc='lower left', fontsize=7, framealpha=0.8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Power (dB)', fontsize=10)
        ax.set_ylabel('kHz', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.axhline(y=4, color=STYLE['GRID'], linewidth=0.6, linestyle=':', alpha=0.6)

    # ── Group separators ─────────────────────────────────────────────────────
    # Draw horizontal lines between codec groups across all columns
    for group_row, group_label in GROUP_STARTS.items():
        # Convert row index to figure coordinates
        # Use the top of the row's axes as the separator position
        if group_row < n_codecs:
            ax_ref = fig.axes[group_row * 3]  # first column of the group row
            bbox = ax_ref.get_position()
            y_pos = bbox.y1 + 0.008
            fig.add_artist(plt.Line2D(
                [0.08, 0.95], [y_pos, y_pos],
                transform=fig.transFigure,
                color=STYLE['GRID'], linewidth=1.2, linestyle='-', alpha=0.8,
            ))

    # ── Colorbar ─────────────────────────────────────────────────────────────
    ax_cb = fig.add_subplot(gs[n_codecs, 1])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin_spec, vmax_spec), cmap='magma'),
        cax=ax_cb, orientation='horizontal',
    )
    cb.set_label('Power (dB)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Hide unused bottom cells
    for col in [0, 2]:
        ax_empty = fig.add_subplot(gs[n_codecs, col])
        ax_empty.axis('off')

    fig.suptitle(
        f'Codec Artifacts — ASVspoof 5 Eval, Speaker {args.speaker} ({args.label.title()})',
        fontsize=14, fontweight='bold', y=1.005,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_base = args.output.removesuffix('.png').removesuffix('.pdf')
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.pdf']:
        out = output_base + ext
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
