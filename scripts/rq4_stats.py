#!/usr/bin/env python3
"""Compute uncertainty/significance summaries for RQ4 intervention results.

Inputs:
- RQ4 summary CSV produced by notebooks/rq4_activation_patching.py
- Optional stats cache NPZ with per-sample score/label arrays

Outputs:
- CSV summary with bootstrap confidence intervals and p-estimates
- Markdown summary for direct thesis/paper insertion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asvspoof5_domain_invariant_cm.evaluation import compute_eer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/rq4_results_summary.csv"),
        help="Path to RQ4 summary CSV",
    )
    parser.add_argument(
        "--stats-cache",
        type=Path,
        default=Path("results/rq4_stats_cache.npz"),
        help="Path to NPZ score cache emitted by rq4_activation_patching.py",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rq4_stats_summary.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def bootstrap_eer_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_samples = scores.shape[0]
    eers = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n_samples, n_samples)
        sampled_scores = scores[sample_idx]
        sampled_labels = labels[sample_idx]
        eer_value, _ = compute_eer(sampled_scores, sampled_labels)
        eers[idx] = eer_value
    return eers


def bootstrap_probe_accuracy_distribution(
    observed_accuracy: float,
    n_samples: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draws = rng.binomial(n_samples, observed_accuracy, size=n_bootstrap)
    return draws.astype(float) / float(n_samples)


def compute_delta_p_estimate(delta_distribution: np.ndarray) -> float:
    positive_mass = float(np.mean(delta_distribution >= 0.0))
    negative_mass = float(np.mean(delta_distribution <= 0.0))
    return 2.0 * min(positive_mass, negative_mass)


def load_mode_arrays(cache_payload: np.lib.npyio.NpzFile, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    score_key = f"{mode}__scores"
    label_key = f"{mode}__labels"
    if score_key not in cache_payload.files or label_key not in cache_payload.files:
        return None
    return np.asarray(cache_payload[score_key]), np.asarray(cache_payload[label_key])


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.results.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")
    results_df = pd.read_csv(args.results)
    if results_df.empty:
        raise ValueError(f"No rows found in {args.results}")

    cache_payload = None
    if args.stats_cache.exists():
        cache_payload = np.load(args.stats_cache, allow_pickle=True)

    output_rows: list[dict[str, float | str]] = []

    baseline_eer_bootstrap = None
    baseline_scores = None
    baseline_labels = None
    if cache_payload is not None and "baseline_scores" in cache_payload.files and "baseline_labels" in cache_payload.files:
        baseline_scores = np.asarray(cache_payload["baseline_scores"])
        baseline_labels = np.asarray(cache_payload["baseline_labels"])
        baseline_eer_bootstrap = bootstrap_eer_distribution(
            baseline_scores,
            baseline_labels,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

    for _, row in results_df.iterrows():
        mode = str(row["mode"])
        mode_eer = float(row["eer"])
        mode_probe = float(row["max_probe_acc"])
        probe_n = int(row.get("probe_num_samples", 0))
        if probe_n <= 0:
            probe_n = 5000

        # Probe uncertainty from parametric bootstrap around observed accuracy.
        mode_probe_bootstrap = bootstrap_probe_accuracy_distribution(
            observed_accuracy=mode_probe,
            n_samples=probe_n,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        probe_ci_low, probe_ci_high = bootstrap_ci(mode_probe_bootstrap)

        mode_eer_ci_low = np.nan
        mode_eer_ci_high = np.nan
        delta_eer_ci_low = np.nan
        delta_eer_ci_high = np.nan
        delta_eer_p = np.nan

        if cache_payload is not None and baseline_eer_bootstrap is not None:
            mode_arrays = load_mode_arrays(cache_payload, mode)
            if mode_arrays is not None:
                mode_scores, mode_labels = mode_arrays
                mode_eer_bootstrap = bootstrap_eer_distribution(
                    mode_scores,
                    mode_labels,
                    n_bootstrap=args.n_bootstrap,
                    rng=rng,
                )
                mode_eer_ci_low, mode_eer_ci_high = bootstrap_ci(mode_eer_bootstrap)

                # Paired delta if the evaluated subset aligns.
                if baseline_scores is not None and mode_scores.shape[0] == baseline_scores.shape[0]:
                    n_samples = mode_scores.shape[0]
                    delta_bootstrap = np.empty(args.n_bootstrap, dtype=float)
                    for bootstrap_idx in range(args.n_bootstrap):
                        sample_idx = rng.integers(0, n_samples, n_samples)
                        baseline_eer, _ = compute_eer(
                            baseline_scores[sample_idx],
                            baseline_labels[sample_idx],
                        )
                        mode_eer_sampled, _ = compute_eer(
                            mode_scores[sample_idx],
                            mode_labels[sample_idx],
                        )
                        delta_bootstrap[bootstrap_idx] = mode_eer_sampled - baseline_eer
                    delta_eer_ci_low, delta_eer_ci_high = bootstrap_ci(delta_bootstrap)
                    delta_eer_p = compute_delta_p_estimate(delta_bootstrap)

        output_rows.append(
            {
                "mode": mode,
                "eer": mode_eer,
                "eer_ci_low": mode_eer_ci_low,
                "eer_ci_high": mode_eer_ci_high,
                "max_probe_acc": mode_probe,
                "probe_ci_low": probe_ci_low,
                "probe_ci_high": probe_ci_high,
                "delta_eer_vs_base": float(row["delta_eer_vs_base"]),
                "delta_eer_ci_low": delta_eer_ci_low,
                "delta_eer_ci_high": delta_eer_ci_high,
                "delta_eer_p_bootstrap": delta_eer_p,
                "delta_probe_vs_base": float(row["delta_probe_vs_base"]),
                "probe_num_samples": probe_n,
                "probe_target": str(row.get("probe_target", "CODEC")),
                "probe_target_unique": int(row.get("probe_target_unique", 0)),
                "eval_split": str(row.get("eval_split", "unknown")),
                "probe_split": str(row.get("probe_split", "unknown")),
            }
        )

    summary_df = pd.DataFrame(output_rows).sort_values("delta_probe_vs_base", ascending=True)

    output_csv = args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    output_md = output_csv.with_suffix(".md")
    md_lines = [
        "# RQ4 Statistical Summary",
        "",
        f"- Results source: `{args.results}`",
        f"- Score cache: `{args.stats_cache}`",
        f"- Bootstrap samples: `{args.n_bootstrap}`",
        "",
        "| Mode | EER | EER 95% CI | Probe Acc | Probe 95% CI | ΔEER vs base | ΔEER 95% CI | Bootstrap p | ΔProbe vs base |",
        "| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        eer_ci = (
            f"[{row['eer_ci_low']:.4f}, {row['eer_ci_high']:.4f}]"
            if np.isfinite(row["eer_ci_low"])
            else "n/a"
        )
        delta_ci = (
            f"[{row['delta_eer_ci_low']:.4f}, {row['delta_eer_ci_high']:.4f}]"
            if np.isfinite(row["delta_eer_ci_low"])
            else "n/a"
        )
        p_text = f"{row['delta_eer_p_bootstrap']:.4f}" if np.isfinite(row["delta_eer_p_bootstrap"]) else "n/a"
        md_lines.append(
            f"| {row['mode']} | {row['eer']:.4f} | {eer_ci} | "
            f"{row['max_probe_acc']:.4f} | [{row['probe_ci_low']:.4f}, {row['probe_ci_high']:.4f}] | "
            f"{row['delta_eer_vs_base']:+.4f} | {delta_ci} | {p_text} | {row['delta_probe_vs_base']:+.4f} |"
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved stats CSV: {output_csv}")
    print(f"Saved stats Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
