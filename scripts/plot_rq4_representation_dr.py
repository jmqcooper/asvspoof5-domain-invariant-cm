#!/usr/bin/env python3
"""Create PCA/UMAP/t-SNE visualization for RQ4 projection representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("results/rq4_repr_cache.npz"),
        help="Path to representation cache NPZ from rq4_activation_patching.py",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/rq4_results_summary.csv"),
        help="Optional RQ4 summary CSV for subtitle metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/rq4/representation_dr.png"),
        help="Output PNG path (PDF also saved)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Maximum points per model for DR plotting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and reducers",
    )
    return parser.parse_args()


def maybe_subsample(features: np.ndarray, labels: np.ndarray, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, labels
    indices = rng.choice(features.shape[0], size=max_points, replace=False)
    return features[indices], labels[indices]


def run_reducers(
    erm_features: np.ndarray,
    dann_features: np.ndarray,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    merged_features = np.concatenate([erm_features, dann_features], axis=0)
    n_erm = erm_features.shape[0]
    n_total = merged_features.shape[0]
    if n_total < 10:
        raise ValueError("Not enough samples for dimensionality reduction.")

    scaled_features = StandardScaler().fit_transform(merged_features)
    pca_components = min(50, scaled_features.shape[1], n_total - 1)
    pca_model = PCA(n_components=pca_components, random_state=seed)
    pca_features = pca_model.fit_transform(scaled_features)
    pca_2d = pca_features[:, :2]

    perplexity = min(30, max(5, (n_total - 1) // 3))
    tsne_model = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        perplexity=perplexity,
    )
    tsne_2d = tsne_model.fit_transform(pca_features)

    umap_model = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=seed,
        metric="euclidean",
    )
    umap_2d = umap_model.fit_transform(pca_features)

    return {
        "PCA": (pca_2d[:n_erm], pca_2d[n_erm:]),
        "UMAP": (umap_2d[:n_erm], umap_2d[n_erm:]),
        "t-SNE": (tsne_2d[:n_erm], tsne_2d[n_erm:]),
    }


def main() -> int:
    args = parse_args()
    plt.rcParams.update(STYLE_CONFIG)

    if not args.cache.exists():
        raise FileNotFoundError(f"Representation cache not found: {args.cache}")
    cache_payload = np.load(args.cache, allow_pickle=True)
    erm_features = np.asarray(cache_payload["erm_repr"])
    dann_features = np.asarray(cache_payload["dann_repr"])
    erm_labels = np.asarray(cache_payload["erm_labels"])
    dann_labels = np.asarray(cache_payload["dann_labels"])
    probe_target = str(cache_payload["probe_target"][0]) if "probe_target" in cache_payload else "CODEC"
    probe_split = str(cache_payload["probe_split"][0]) if "probe_split" in cache_payload else "unknown"

    rng = np.random.default_rng(args.seed)
    erm_features, erm_labels = maybe_subsample(erm_features, erm_labels, args.max_points, rng)
    dann_features, dann_labels = maybe_subsample(dann_features, dann_labels, args.max_points, rng)

    embeddings = run_reducers(erm_features, dann_features, seed=args.seed)
    all_labels = np.concatenate([erm_labels, dann_labels], axis=0)
    unique_labels = sorted(np.unique(all_labels).tolist())
    label_to_color = {label: index for index, label in enumerate(unique_labels)}
    color_map = plt.get_cmap("tab20", max(2, len(unique_labels)))

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 13.6))
    method_order = ["PCA", "UMAP", "t-SNE"]
    model_titles = ["ERM", "DANN"]

    for row_idx, method_name in enumerate(method_order):
        erm_embed, dann_embed = embeddings[method_name]
        model_embeds = [erm_embed, dann_embed]
        model_labels = [erm_labels, dann_labels]
        for col_idx in range(2):
            axis = axes[row_idx, col_idx]
            points = model_embeds[col_idx]
            labels = model_labels[col_idx]
            colors = [label_to_color[int(label)] for label in labels]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                c=colors,
                cmap=color_map,
                s=7,
                alpha=0.45,
                linewidths=0.0,
            )
            axis.set_title(f"{method_name}: {model_titles[col_idx]}", fontweight="bold")
            axis.set_xlabel("Component 1")
            axis.set_ylabel("Component 2")

    handles = []
    for label in unique_labels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map(label_to_color[label]),
                markeredgecolor="none",
                markersize=6,
                label=f"{probe_target}={label}",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)),
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.03),
    )

    subtitle = f"Split={probe_split} | Samples: ERM={erm_features.shape[0]} DANN={dann_features.shape[0]}"
    if args.results is not None and args.results.exists():
        results_df = pd.read_csv(args.results)
        if "eval_split" in results_df.columns and not results_df.empty:
            subtitle = (
                f"Eval split={results_df['eval_split'].iloc[0]} | "
                f"Probe split={results_df.get('probe_split', pd.Series([probe_split])).iloc[0]} | "
                f"Samples: ERM={erm_features.shape[0]} DANN={dann_features.shape[0]}"
            )

    fig.suptitle("RQ4 Representation Structure: PCA vs UMAP vs t-SNE", fontsize=15, fontweight="bold", y=0.988)
    fig.text(0.5, 0.962, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved DR figure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
