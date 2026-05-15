#!/usr/bin/env python3
"""Recompute per-codec EER + minDCF for seed-42 plain ERM runs using the
*current* (post-22a87ea) metric definitions.

Background: predictions/{wavlm,w2v2}_erm_seed42_eval/tables/metrics_by_codec.csv
are symlinks to runs/<model>/eval_eval_full/tables/metrics_by_codec.csv,
which were written on Feb 15 with the buggy minDCF defaults
(c_miss=1, c_fa=1, p_target=0.05). Commit 22a87ea (Mar 13) fixed these
to the ASVspoof 5 Track 1 values (c_miss=1, c_fa=10, p_target=0.95).
EER is unaffected; only minDCF needs recomputing.

This script reads each model's predictions.tsv and writes a fresh
metrics_by_codec.csv next to it, broken down by codec, using the
current compute_eer / compute_min_dcf implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from asvspoof5_domain_invariant_cm.evaluation.metrics import (
    compute_eer,
    compute_min_dcf,
)

ROOT = Path(__file__).resolve().parents[1]

# (predictions.tsv source, output metrics_by_codec.csv)
# Outputs land in predictions/ so we replace the stale Apr-16 symlinks with
# real files. The Feb 15 runs/<model>/eval_eval_full/tables/*.csv files are
# left untouched as historical artifacts.
TARGETS = [
    (
        ROOT / "results/runs/wavlm_erm/eval_eval_full/predictions.tsv",
        ROOT / "results/predictions/wavlm_erm_seed42_eval/tables/metrics_by_codec.csv",
    ),
    (
        ROOT / "results/runs/w2v2_erm/eval_eval_full/predictions.tsv",
        ROOT / "results/predictions/w2v2_erm_seed42_eval/tables/metrics_by_codec.csv",
    ),
]


def labels_from_y_task(y_task) -> "pd.Series":
    # Verified against seed 123 CSV: y_task already encodes 1=spoof, 0=bonafide,
    # which is exactly what compute_eer / compute_min_dcf expect.
    return y_task.astype(int)


def main() -> None:
    for tsv_path, out_path in TARGETS:
        if not tsv_path.exists():
            raise FileNotFoundError(tsv_path)
        df = pd.read_csv(tsv_path, sep="\t")
        if "codec" not in df.columns:
            raise ValueError(f"{tsv_path} missing 'codec' column")

        labels = labels_from_y_task(df["y_task"])  # 0=bonafide, 1=spoof
        scores = df["score"].to_numpy()

        rows = []
        for codec, idx in df.groupby("codec").groups.items():
            sub_scores = scores[idx]
            sub_labels = labels.iloc[idx].to_numpy()
            n_bonafide = int((sub_labels == 0).sum())
            n_spoof = int((sub_labels == 1).sum())
            eer, eer_thr = compute_eer(sub_scores, sub_labels)
            mdcf = compute_min_dcf(sub_scores, sub_labels)
            rows.append({
                "domain": codec,
                "n_samples": len(idx),
                "n_bonafide": n_bonafide,
                "n_spoof": n_spoof,
                "eer": eer,
                "eer_threshold": eer_thr,
                "min_dcf": mdcf,
            })

        rows.sort(key=lambda r: (r["domain"] != "NONE", r["domain"]))  # NONE last
        # Match original column order
        cols = ["domain", "n_samples", "n_bonafide", "n_spoof", "eer", "eer_threshold", "min_dcf"]

        # If a stale symlink exists at the destination, remove it before
        # writing — otherwise the open() would follow the link and clobber
        # the historical Feb-15 file in runs/.
        if out_path.is_symlink():
            out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
