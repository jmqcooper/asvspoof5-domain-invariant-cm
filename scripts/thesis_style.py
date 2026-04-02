"""
Thesis figure style — consistent academic palette.

Color logic (per Angela's feedback):
  - Same COLOR = same METHOD across backbones
  - Backbone distinguished by SATURATION (WavLM = bold, W2V2 = lighter)
  - Every shade appears in every legend

Methods:
  ERM     → Blue family
  DANN    → Red/coral family  
  ERM+Aug → Amber/gold family

Backbones:
  WavLM   → Bold/saturated variant
  W2V2    → Lighter/desaturated variant

Usage:
    from thesis_style import set_style, PALETTE, MODEL_LABELS, get_color
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Core palette ─────────────────────────────────────────────────────────────

# Method × Backbone color matrix
PALETTE = {
    # Method colors: WavLM (bold) / W2V2 (light)
    "wavlm_erm":      "#2563EB",  # Blue 600
    "w2v2_erm":        "#93C5FD",  # Blue 300
    "wavlm_dann":      "#DC2626",  # Red 600
    "w2v2_dann":       "#FCA5A5",  # Red 300
    "wavlm_erm_aug":   "#D97706",  # Amber 600
    "w2v2_erm_aug":    "#FCD34D",  # Amber 300

    # Baselines (gray family)
    "lfcc_gmm":            "#9CA3AF",  # Gray 400
    "trillsson_logistic":  "#6B7280",  # Gray 500
    "trillsson_mlp":       "#4B5563",  # Gray 600

    # Semantic colors
    "chance":     "#9CA3AF",  # Gray 400 dashed
    "highlight":  "#7C3AED",  # Violet 600
    "divergent":  "#DC2626",  # Red 600
    "neutral":    "#E5E7EB",  # Gray 200
}

# Hatching for B&W distinguishability (optional, for print)
HATCHES = {
    "wavlm_erm":     None,
    "w2v2_erm":      "//",
    "wavlm_dann":    None,
    "w2v2_dann":     "//",
    "wavlm_erm_aug": None,
    "w2v2_erm_aug":  "//",
}

MODEL_LABELS = {
    "wavlm_erm":      "WavLM ERM",
    "wavlm_dann":     "WavLM DANN",
    "wavlm_erm_aug":  "WavLM ERM+Aug",
    "w2v2_erm":       "W2V2 ERM",
    "w2v2_dann":      "W2V2 DANN",
    "w2v2_erm_aug":   "W2V2 ERM+Aug",
    "lfcc_gmm":             "LFCC-GMM",
    "trillsson_logistic":   "TRILLsson Logistic",
    "trillsson_mlp":        "TRILLsson MLP",
}

# Canonical model ordering for legends
MODEL_ORDER = [
    "wavlm_erm", "wavlm_dann", "wavlm_erm_aug",
    "w2v2_erm",  "w2v2_dann",  "w2v2_erm_aug",
    "lfcc_gmm", "trillsson_logistic", "trillsson_mlp",
]

# Method legend (for grouped plots)
METHOD_COLORS = {
    "ERM":     ("#2563EB", "#93C5FD"),  # bold, light
    "DANN":    ("#DC2626", "#FCA5A5"),
    "ERM+Aug": ("#D97706", "#FCD34D"),
}

# ── Style tokens ─────────────────────────────────────────────────────────────

STYLE = {
    "BG":      "#FFFFFF",
    "PLOT_BG": "#FAFAFA",
    "GRID":    "#E5E7EB",
    "AXIS":    "#374151",
    "TEXT":    "#1F2937",
    "TICK":    "#4B5563",
}

# CKA sequential colormap: white → red
CKA_CMAP_COLORS = [
    "#FEF2F2", "#FECACA", "#FCA5A5", "#F87171",
    "#EF4444", "#DC2626", "#B91C1C", "#7F1D1D",
]

# DET curve line styles per model
LINE_STYLES = {
    "wavlm_erm":     {"ls": "-",  "lw": 2.0},
    "wavlm_dann":    {"ls": "-",  "lw": 2.0},
    "wavlm_erm_aug": {"ls": "-",  "lw": 2.0},
    "w2v2_erm":      {"ls": "--", "lw": 1.8},
    "w2v2_dann":     {"ls": "--", "lw": 1.8},
    "w2v2_erm_aug":  {"ls": "--", "lw": 1.8},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_color(model_key: str) -> str:
    """Get color for a model key, with fallback."""
    return PALETTE.get(model_key, "#9CA3AF")


def get_label(model_key: str) -> str:
    """Get display label for a model key."""
    return MODEL_LABELS.get(model_key, model_key.replace("_", " ").upper())


def set_style():
    """Apply thesis matplotlib style globally."""
    mpl.rcParams.update({
        "figure.facecolor": STYLE["BG"],
        "axes.facecolor":   STYLE["PLOT_BG"],
        "axes.edgecolor":   STYLE["AXIS"],
        "axes.labelcolor":  STYLE["AXIS"],
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "axes.axisbelow":   True,
        "grid.color":       STYLE["GRID"],
        "grid.linewidth":   0.5,
        "grid.alpha":       0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.color":  STYLE["TICK"],
        "ytick.color":  STYLE["TICK"],
        "text.color":   STYLE["TEXT"],
        "font.family":  "serif",
        "font.serif":   ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":     11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "legend.frameon":   True,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  STYLE["GRID"],
        "figure.dpi":    150,
        "savefig.dpi":   300,
        "savefig.bbox":  "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": STYLE["BG"],
    })


# Backward compat aliases
COLORS = PALETTE
