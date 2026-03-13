#!/usr/bin/env python3
"""Visualize codec artifacts: bonafide vs spoof paired per codec.

Creates a 24-row × 3-column grid (12 codecs × 2 labels) showing how each
ASVspoof 5 codec condition affects bonafide and spoof audio from the same speaker.

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
CODEC_SHORT = {
    'NONE': 'No Codec',
    'C01':  'C01 · Opus WB',
    'C02':  'C02 · AMR-WB',
    'C03':  'C03 · Speex WB',
    'C04':  'C04 · Encodec',
    'C05':  'C05 · MP3 WB',
    'C06':  'C06 · AAC WB',
    'C07':  'C07 · MP3+Encodec',
    'C08':  'C08 · Opus NB',
    'C09':  'C09 · AMR-NB',
    'C10':  'C10 · Speex NB',
    'C11':  'C11 · Device',
}

CODEC_ORDER = ['NONE', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
               'C08', 'C09', 'C10', 'C11']

# Group separators: draw line BEFORE this codec index
GROUP_BOUNDARIES = [1, 8, 11]  # before C01, C08, C11

CODEC_COLORS = {
    'NONE': '#333333',
    'C01':  '#D4795A', 'C02': '#E8946E', 'C03': '#F0A882',
    'C04':  '#7C3AED',
    'C05':  '#4CA08A', 'C06': '#6BC4AE',
    'C07':  '#9B59B6',
    'C08':  '#3B82F6', 'C09': '#60A5FA', 'C10': '#93C5FD',
    'C11':  '#F59E0B',
}

BONAFIDE_ALPHA = 0.90
SPOOF_ALPHA = 0.70


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
    spk = df[df['speaker_id'] == args.speaker]

    # Pick one bonafide + one spoof utterance per codec
    rows = []  # list of (codec, label_str, y_task, flac_file)
    for codec in CODEC_ORDER:
        for y_task, label_suffix in [(0, 'Bonafide'), (1, 'Spoof')]:
            subset = spk[(spk['y_task'] == y_task) & (spk['codec'] == codec)]
            if len(subset) == 0:
                print(f'Warning: no {label_suffix.lower()} for {codec}')
                continue
            rows.append((codec, label_suffix, y_task, subset.iloc[0]['flac_file']))

    print(f'Speaker {args.speaker}: {len(rows)} rows ({len(rows)//2} codecs × 2)')

    # Load NONE bonafide as reference for spectral envelope
    ref_entry = [r for r in rows if r[0] == 'NONE' and r[1] == 'Bonafide']
    if not ref_entry:
        print('Error: no NONE bonafide reference'); sys.exit(1)
    ref_path = eval_dir / f'{ref_entry[0][3]}.flac'
    ref_audio, sr = load_audio(ref_path)
    max_samples = int(args.max_seconds * sr)
    ref_audio = ref_audio[:max_samples]
    ref_spec = compute_spectrogram(ref_audio, sr)
    ref_envelope = compute_spectral_envelope(ref_spec)
    freq_bins = np.linspace(0, sr / 2, ref_spec.shape[0])
    print(f'Reference: {ref_path.name} ({len(ref_audio)/sr:.2f}s, {sr}Hz)')

    # ── Figure layout ────────────────────────────────────────────────────────
    n_rows = len(rows)
    row_h = 0.72
    fig_h = row_h * n_rows + 1.2
    fig = plt.figure(figsize=(10, fig_h))

    gs = gridspec.GridSpec(
        n_rows + 1, 3, figure=fig,
        width_ratios=[1, 1.4, 0.9],
        height_ratios=[1] * n_rows + [0.05],
        hspace=0.30, wspace=0.25,
        top=0.97, bottom=0.03,
    )

    vmin_spec, vmax_spec = -80, 0

    # Track which codec row we're in for group separators
    prev_codec = None
    codec_row_start = {}  # codec → first grid row index

    for row_idx, (codec, label_suffix, y_task, flac_id) in enumerate(rows):
        flac_path = eval_dir / f'{flac_id}.flac'
        if not flac_path.exists():
            print(f'Warning: {flac_path} not found'); continue

        audio, _ = load_audio(flac_path)
        audio = audio[:max_samples]
        t = np.arange(len(audio)) / sr
        spec = compute_spectrogram(audio, sr)
        envelope = compute_spectral_envelope(spec)

        color = CODEC_COLORS.get(codec, '#888888')
        alpha = BONAFIDE_ALPHA if y_task == 0 else SPOOF_ALPHA
        is_spoof = y_task == 1

        # Row label: "C01 · Opus WB\nBonafide" or just "Spoof" for second row
        if label_suffix == 'Bonafide':
            row_label = f'{CODEC_SHORT[codec]}\n  ● Bonafide'
            codec_row_start[codec] = row_idx
        else:
            row_label = f'  ○ Spoof'

        print(f'  {codec} {label_suffix}: {flac_id} ({len(audio)/sr:.2f}s)')

        # ── Col 0: Waveform ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 0])
        lw = 0.25 if not is_spoof else 0.20
        ax.plot(t, audio, color=color, linewidth=lw, alpha=alpha,
                linestyle='-' if not is_spoof else '-')
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, args.max_seconds)
        ax.set_ylabel(row_label, fontsize=6,
                      fontweight='bold' if not is_spoof else 'normal',
                      rotation=0, labelpad=95, ha='left', va='center',
                      fontstyle='normal' if not is_spoof else 'italic')
        ax.yaxis.set_label_coords(-0.50, 0.5)
        if row_idx == 0:
            ax.set_title('Waveform', fontsize=8, fontweight='bold', pad=4)
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (s)', fontsize=6)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_yticklabels([])
        # Light background tint for spoof rows
        if is_spoof:
            ax.set_facecolor('#FDF2F2')

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
            ax.set_title('Spectrogram', fontsize=8, fontweight='bold', pad=4)
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('')
        ax.tick_params(labelsize=5, length=2, pad=1)

        # ── Col 2: Spectral envelope ─────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.plot(ref_envelope, freq_bins / 1000, color='#CCCCCC',
                linewidth=0.8, alpha=0.5, linestyle='--')
        ls = '-' if not is_spoof else '--'
        ax.plot(envelope, freq_bins / 1000, color=color,
                linewidth=1.6, alpha=alpha, linestyle=ls)
        ax.set_ylim(0, 8)
        ax.set_xlim(0, -80)
        if row_idx == 0:
            ax.set_title('Spectral Envelope', fontsize=8, fontweight='bold', pad=4)
            ax.plot([], [], color='#CCCCCC', linewidth=0.8, linestyle='--', label='Ref (uncoded)')
            ax.plot([], [], color='#555555', linewidth=1.6, linestyle='-', label='Bonafide')
            ax.plot([], [], color='#555555', linewidth=1.6, linestyle='--', label='Spoof')
            ax.legend(loc='lower left', fontsize=5, framealpha=0.8, handlelength=1.5)
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Power (dB)', fontsize=6)
        ax.set_ylabel('')
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.axhline(y=4, color=STYLE['GRID'], linewidth=0.5, linestyle=':', alpha=0.5)
        if is_spoof:
            ax.set_facecolor('#FDF2F2')

    # ── Group separators ─────────────────────────────────────────────────────
    fig.canvas.draw()
    for boundary_codec_idx in GROUP_BOUNDARIES:
        codec = CODEC_ORDER[boundary_codec_idx]
        if codec in codec_row_start:
            grid_row = codec_row_start[codec]
            ax_ref = fig.axes[grid_row * 3]
            bbox = ax_ref.get_position()
            y_pos = bbox.y1 + 0.005
            fig.add_artist(plt.Line2D(
                [0.06, 0.96], [y_pos, y_pos],
                transform=fig.transFigure,
                color='#999999', linewidth=1.0, linestyle='-', alpha=0.6,
            ))

    # ── Colorbar ─────────────────────────────────────────────────────────────
    ax_cb = fig.add_subplot(gs[n_rows, 1])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin_spec, vmax_spec), cmap='magma'),
        cax=ax_cb, orientation='horizontal',
    )
    cb.set_label('Power (dB)', fontsize=6)
    cb.ax.tick_params(labelsize=5)

    for col in [0, 2]:
        ax_empty = fig.add_subplot(gs[n_rows, col])
        ax_empty.axis('off')

    fig.suptitle(
        f'Codec Artifacts — ASVspoof 5 Eval, Speaker {args.speaker}',
        fontsize=9, fontweight='bold', y=0.99,
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
