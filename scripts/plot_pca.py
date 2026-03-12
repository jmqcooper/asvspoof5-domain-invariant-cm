#!/usr/bin/env python3
"""Plot PCA of projection layer representations: ERM vs DANN.

Usage:
    python scripts/plot_pca.py --erm-repr erm.npy --dann-repr dann.npy --labels labels.npy
    python scripts/plot_pca.py --demo  # synthetic data for layout testing
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, '/root/.openclaw/workspace/master-thesis-uva/figures/')
from thesis_style import COLORS, STYLE, set_style

COLOR_CODED = '#D4795A'    # Terracotta
COLOR_UNCODED = '#4CA08A'  # Teal
MAX_POINTS = 2000
ALPHA = 0.3


def subsample(data, labels_binary, max_n=MAX_POINTS, rng=None):
    """Random subsample to max_n points."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(data) <= max_n:
        return data, labels_binary
    idx = rng.choice(len(data), max_n, replace=False)
    return data[idx], labels_binary[idx]


def make_binary_labels(codec_labels):
    """Convert codec labels to binary: 0=uncoded (NONE), 1=coded (anything else).

    Handles both string labels and integer-encoded labels where 0 = NONE.
    """
    if codec_labels.dtype.kind in ('U', 'S', 'O'):  # string types
        return (codec_labels != 'NONE').astype(int)
    else:
        # Assume 0 = NONE/uncoded
        return (codec_labels != 0).astype(int)


def generate_demo_data(n=4000, dim=256):
    """Generate synthetic clustered data for layout testing."""
    rng = np.random.default_rng(42)

    # ERM: overlapping clusters (less separated)
    erm_coded = rng.normal(loc=[2, 1] + [0] * (dim - 2), scale=1.5, size=(n // 2, dim))
    erm_uncoded = rng.normal(loc=[-1, -0.5] + [0] * (dim - 2), scale=1.5, size=(n // 2, dim))
    erm_data = np.vstack([erm_coded, erm_uncoded])

    # DANN: more overlapping (domain-invariant)
    dann_coded = rng.normal(loc=[0.5, 0.3] + [0] * (dim - 2), scale=1.5, size=(n // 2, dim))
    dann_uncoded = rng.normal(loc=[-0.3, -0.2] + [0] * (dim - 2), scale=1.5, size=(n // 2, dim))
    dann_data = np.vstack([dann_coded, dann_uncoded])

    labels = np.array([1] * (n // 2) + [0] * (n // 2))  # 1=coded, 0=uncoded
    return erm_data, dann_data, labels


def plot_pca_panel(ax, data_2d, labels_binary, subtitle):
    """Scatter plot of PCA-projected data with binary colouring."""
    coded_mask = labels_binary == 1
    uncoded_mask = labels_binary == 0

    ax.scatter(data_2d[uncoded_mask, 0], data_2d[uncoded_mask, 1],
               c=COLOR_UNCODED, alpha=ALPHA, s=10, label='Uncoded', rasterized=True)
    ax.scatter(data_2d[coded_mask, 0], data_2d[coded_mask, 1],
               c=COLOR_CODED, alpha=ALPHA, s=10, label='Coded', rasterized=True)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(subtitle)
    ax.legend(loc='upper right', markerscale=3)


def main():
    parser = argparse.ArgumentParser(description='PCA of projection layer representations')
    parser.add_argument('--erm-repr', type=str, help='Path to ERM representations .npy')
    parser.add_argument('--dann-repr', type=str, help='Path to DANN representations .npy')
    parser.add_argument('--labels', type=str, help='Path to codec labels .npy')
    parser.add_argument('--output', type=str,
                        default='/root/.openclaw/workspace/master-thesis-uva/figures/pca_representations')
    parser.add_argument('--demo', action='store_true', help='Generate synthetic data for layout testing')
    args = parser.parse_args()

    set_style()
    rng = np.random.default_rng(42)

    if args.demo:
        erm_data, dann_data, labels_binary = generate_demo_data()
    else:
        if not all([args.erm_repr, args.dann_repr, args.labels]):
            parser.error('--erm-repr, --dann-repr, and --labels are required when not using --demo')
        erm_data = np.load(args.erm_repr)
        dann_data = np.load(args.dann_repr)
        codec_labels = np.load(args.labels, allow_pickle=True)
        labels_binary = make_binary_labels(codec_labels)

    # Subsample
    erm_sub, labels_erm = subsample(erm_data, labels_binary, rng=rng)
    dann_sub, labels_dann = subsample(dann_data, labels_binary, rng=rng)

    # Fit PCA on combined data for consistent axes
    pca = PCA(n_components=2, random_state=42)
    combined = np.vstack([erm_sub, dann_sub])
    pca.fit(combined)

    erm_2d = pca.transform(erm_sub)
    dann_2d = pca.transform(dann_sub)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_pca_panel(ax1, erm_2d, labels_erm, '(a) ERM')
    plot_pca_panel(ax2, dann_2d, labels_dann, '(b) DANN')
    fig.suptitle('Projection Layer Representations: ERM vs DANN',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_base = args.output.removesuffix('.png').removesuffix('.pdf')
    for ext in ['.png', '.pdf']:
        out_path = output_base + ext
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
