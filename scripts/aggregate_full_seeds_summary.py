#!/usr/bin/env python3
"""Build a comprehensive per-seed summary across all 5 seeds and 6 models.

Outputs two CSVs:
    results/headline_seeds_summary.csv     # one row per (model, seed) +
                                           # aggregate (model, "mean") and
                                           # (model, "std") rows.
                                           # Columns: eer, min_dcf, auc,
                                           # eer_ci_lower, eer_ci_upper,
                                           # min_dcf_ci_lower, min_dcf_ci_upper,
                                           # auc_ci_lower, auc_ci_upper,
                                           # f1_macro, n_samples.
    results/per_codec_seeds_summary.csv    # already produced by
                                           # aggregate_per_codec_seeds.py
                                           # — left untouched.

The headline CSV is what populates T1 (main results) in the thesis with
mean ± std over all 5 seeds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
OUT_CSV = ROOT / "results" / "headline_seeds_summary.csv"

SEEDS = [42, 123, 456, 789, 2024]
V2_MODELS = {"wavlm_dann", "wavlm_erm_aug", "w2v2_dann", "w2v2_erm_aug"}
MODELS = ["wavlm_erm", "wavlm_erm_aug", "wavlm_dann",
          "w2v2_erm", "w2v2_erm_aug", "w2v2_dann"]

METRIC_COLS = [
    "eer", "eer_ci_lower", "eer_ci_upper",
    "min_dcf", "min_dcf_ci_lower", "min_dcf_ci_upper",
    "auc", "auc_ci_lower", "auc_ci_upper",
    "f1_macro", "n_samples",
]


def metrics_path(model: str, seed: int) -> Path:
    suffix = f"seed{seed}_v2_eval" if (seed == 42 and model in V2_MODELS) else f"seed{seed}_eval"
    return PRED / f"{model}_{suffix}" / "metrics.json"


def main() -> None:
    rows: list[dict] = []
    for model in MODELS:
        per_seed: dict[str, list[float]] = {c: [] for c in METRIC_COLS}
        for seed in SEEDS:
            p = metrics_path(model, seed)
            if not p.exists():
                raise FileNotFoundError(p)
            d = json.loads(p.read_text())
            row = {"model": model, "seed": str(seed)}
            for col in METRIC_COLS:
                v = d.get(col)
                row[col] = f"{v:.6f}" if isinstance(v, float) else (str(v) if v is not None else "")
                if isinstance(v, (int, float)) and col != "n_samples":
                    per_seed[col].append(float(v))
            rows.append(row)

        # aggregate rows
        for agg_label, fn in (("mean", mean), ("std", stdev)):
            row = {"model": model, "seed": agg_label}
            for col in METRIC_COLS:
                vs = per_seed[col]
                if vs and col != "n_samples":
                    row[col] = f"{fn(vs):.6f}"
                else:
                    row[col] = ""
            rows.append(row)

    fieldnames = ["model", "seed"] + METRIC_COLS
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
