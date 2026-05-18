#!/usr/bin/env python3
"""Recompute per-codec + overall EER + minDCF for seed-42 plain ERM runs
using the *current* (post-22a87ea) metric definitions.

Background: predictions/{wavlm,w2v2}_erm_seed42_eval/tables/metrics_by_codec.csv
are symlinks to runs/<model>/eval_eval_full/tables/metrics_by_codec.csv,
which were written on Feb 15 with the buggy minDCF defaults
(c_miss=1, c_fa=1, p_target=0.05). Commit 22a87ea (Mar 13) fixed these
to the ASVspoof 5 Track 1 values (c_miss=1, c_fa=10, p_target=0.95).
EER is unaffected; only minDCF needs recomputing.

This script reads each model's predictions.tsv and writes both:
  - results/predictions/{model}_seed42_eval/tables/metrics_by_codec.csv
  - results/predictions/{model}_seed42_eval/metrics.json
with overall + per-codec EER/minDCF and bootstrap CIs, using the current
compute_eer / compute_min_dcf implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import json

import numpy as np

from asvspoof5_domain_invariant_cm.evaluation.metrics import (
    bootstrap_metric,
    compute_eer,
    compute_min_dcf,
)
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

# (predictions.tsv source, output predictions/ subdir)
# Outputs land in predictions/ so we replace the stale Apr-16 symlinks with
# real files. The Feb 15 runs/<model>/eval_eval_full/ files are left
# untouched as historical artifacts.
TARGETS = [
    (
        ROOT / "results/runs/wavlm_erm/eval_eval_full/predictions.tsv",
        ROOT / "results/predictions/wavlm_erm_seed42_eval",
    ),
    (
        ROOT / "results/runs/w2v2_erm/eval_eval_full/predictions.tsv",
        ROOT / "results/predictions/w2v2_erm_seed42_eval",
    ),
]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def labels_from_y_task(y_task) -> "pd.Series":
    # Verified against seed 123 CSV: y_task already encodes 1=spoof, 0=bonafide,
    # which is exactly what compute_eer / compute_min_dcf expect.
    return y_task.astype(int)


def main() -> None:
    for tsv_path, pred_dir in TARGETS:
        if not tsv_path.exists():
            raise FileNotFoundError(tsv_path)
        df = pd.read_csv(tsv_path, sep="\t")
        if "codec" not in df.columns:
            raise ValueError(f"{tsv_path} missing 'codec' column")

        labels_all = labels_from_y_task(df["y_task"]).to_numpy()  # 0=bona, 1=spoof
        scores_all = df["score"].to_numpy()

        # ---- per-codec ----
        rows = []
        for codec, idx in df.groupby("codec").groups.items():
            sub_scores = scores_all[idx]
            sub_labels = labels_all[idx]
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
        cols = ["domain", "n_samples", "n_bonafide", "n_spoof", "eer", "eer_threshold", "min_dcf"]

        per_codec_csv = pred_dir / "tables" / "metrics_by_codec.csv"
        if per_codec_csv.is_symlink():
            per_codec_csv.unlink()
        per_codec_csv.parent.mkdir(parents=True, exist_ok=True)
        with per_codec_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote: {per_codec_csv}")

        # ---- overall metrics.json (with bootstrap CIs) ----
        eer, eer_thr = compute_eer(scores_all, labels_all)
        mdcf = compute_min_dcf(scores_all, labels_all)
        # AUC convention in this codebase: higher score = bonafide, so we
        # invert labels for sklearn (0=bonafide, 1=spoof → flip for auc).
        auc = float(roc_auc_score(1 - labels_all, scores_all))

        _, eer_lo, eer_hi = bootstrap_metric(
            scores_all, labels_all,
            lambda s, l: compute_eer(s, l)[0],
            n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        _, dcf_lo, dcf_hi = bootstrap_metric(
            scores_all, labels_all, compute_min_dcf,
            n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        _, auc_lo, auc_hi = bootstrap_metric(
            scores_all, labels_all,
            lambda s, l: float(roc_auc_score(1 - l, s)),
            n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )

        # Read the existing (Feb 15) metrics.json for the rest of the fields
        # so the file structure matches the seeds-123/456 outputs.
        legacy_json = (
            ROOT / "results/runs"
            / tsv_path.parents[1].name  # wavlm_erm / w2v2_erm
            / tsv_path.parent.name      # eval_eval_full
            / "metrics.json"
        )
        legacy = json.loads(legacy_json.read_text()) if legacy_json.exists() else {}

        out = {
            "eer": eer,
            "eer_threshold": eer_thr,
            "min_dcf": mdcf,
            "auc": auc,
            # Carry forward F1/precision/recall — those are score-independent
            # of the cost params and remain valid.
            "f1_macro": legacy.get("f1_macro"),
            "precision_macro": legacy.get("precision_macro"),
            "recall_macro": legacy.get("recall_macro"),
            "f1_bonafide": legacy.get("f1_bonafide"),
            "f1_spoof": legacy.get("f1_spoof"),
            "precision_bonafide": legacy.get("precision_bonafide"),
            "precision_spoof": legacy.get("precision_spoof"),
            "recall_bonafide": legacy.get("recall_bonafide"),
            "recall_spoof": legacy.get("recall_spoof"),
            "tdcf_min": None,
            "tdcf_threshold": None,
            "tdcf_asv_available": False,
            "tdcf_note": legacy.get("tdcf_note", ""),
            "n_samples": int(len(df)),
            "n_bonafide": int((labels_all == 0).sum()),
            "n_spoof": int((labels_all == 1).sum()),
            "eer_ci_lower": eer_lo,
            "eer_ci_upper": eer_hi,
            "min_dcf_ci_lower": dcf_lo,
            "min_dcf_ci_upper": dcf_hi,
            "auc_ci_lower": auc_lo,
            "auc_ci_upper": auc_hi,
        }
        out_metrics = pred_dir / "metrics.json"
        out_metrics.write_text(json.dumps(out, indent=2))
        print(f"Wrote: {out_metrics}")


if __name__ == "__main__":
    main()
