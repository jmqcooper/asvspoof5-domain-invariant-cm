"""
Thesis figure style — Anthropic-inspired academic palette.
Usage: from thesis_style import STYLE, set_style, COLORS
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    'WAVLM_ERM':  '#D4795A',  # Terracotta
    'WAVLM_DANN': '#4CA08A',  # Muted teal
    'W2V2_ERM':   '#E8B4A0',  # Light terracotta
    'W2V2_DANN':  '#A5D5C3',  # Light teal
    'BASELINE':   '#9CA3AF',  # Gray
    'ACCENT':     '#3B4D6B',  # Dark navy
    'HIGHLIGHT':  '#E74C3C',  # Red for outlier highlight
}

STYLE = {
    'BG':      '#FAFAFA',
    'PLOT_BG': '#F5F5F5',
    'GRID':    '#E0E0E0',
    'AXIS':    '#444444',
    'TEXT':    '#333333',
    'TICK':    '#555555',
}

# CKA sequential: light blue → dark blue
CKA_CMAP_COLORS = ['#F0F4FF', '#C7D7FE', '#93B4FD', '#5B8DEF', '#2563EB', '#1D4ED8', '#1E3A8A']


def set_style():
    """Apply thesis matplotlib style globally."""
    mpl.rcParams.update({
        'figure.facecolor': STYLE['BG'],
        'axes.facecolor': STYLE['PLOT_BG'],
        'axes.edgecolor': STYLE['AXIS'],
        'axes.labelcolor': STYLE['AXIS'],
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': STYLE['GRID'],
        'grid.linewidth': 0.5,
        'grid.alpha': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': STYLE['TICK'],
        'ytick.color': STYLE['TICK'],
        'text.color': STYLE['TEXT'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': STYLE['BG'],
    })


# NOTE: No hardcoded DATA dict. Plotting scripts should load results
# dynamically from prediction files or result CSVs. See e.g.
# plot_det_curves.py which loads from predictions.tsv files.
