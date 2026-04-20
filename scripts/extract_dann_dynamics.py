#!/usr/bin/env python3
"""Extract DANN training dynamics (best_epoch + lambda at save) per run.

Parses metrics_train.json for best_epoch, then parses logs.jsonl to find the
lambda_grl value logged at the "Epoch {best_epoch}:" line.

Usage:
    python scripts/extract_dann_dynamics.py \
        --runs-dir /gpfs/work5/0/prjs1904/runs \
        --output results/dann_dynamics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_RUN_NAMES = [
    "wavlm_dann",  # pre-v2 original
    "wavlm_dann_seed42_v2_1e2d5c7",
    "wavlm_dann_seed123_5223811",
    "wavlm_dann_seed456_5223811",
    "w2v2_dann",  # original
    "w2v2_dann_v2",
    "w2v2_dann_seed42_v2_1e2d5c7",
    "w2v2_dann_seed123_5223811",
    "w2v2_dann_seed456_5223811",
]


LAMBDA_RE = re.compile(r"Epoch\s+(\d+):\s*lambda_grl=([\d.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=DEFAULT_RUN_NAMES,
        help="Run names to inspect (default: all known DANN runs)",
    )
    return parser.parse_args()


def find_lambda_at_epoch(logs_path: Path, target_epoch: int) -> float | None:
    """Scan logs.jsonl for the lambda value at a given epoch.

    Returns the most recent match (logs often contain multiple training
    attempts; the latest run's value is what the saved best.pt reflects).
    """
    if not logs_path.exists():
        return None
    last_value: float | None = None
    with logs_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message", "")
            match = LAMBDA_RE.search(msg)
            if match:
                epoch = int(match.group(1))
                if epoch == target_epoch:
                    last_value = float(match.group(2))
    return last_value


def row_for_run(runs_dir: Path, run_name: str) -> dict:
    run_dir = runs_dir / run_name
    metrics_path = run_dir / "metrics_train.json"
    logs_path = run_dir / "logs.jsonl"

    if not metrics_path.exists():
        return {"run": run_name, "status": "missing_metrics"}

    metrics = json.loads(metrics_path.read_text())
    best_epoch = metrics.get("best_epoch")
    best_eer = metrics.get("best_eer")
    final_epoch = metrics.get("final_epoch")

    lambda_at_save = find_lambda_at_epoch(logs_path, best_epoch) if best_epoch is not None else None

    return {
        "run": run_name,
        "status": "ok",
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_eer": best_eer,
        "lambda_at_save": lambda_at_save,
    }


def main() -> None:
    args = parse_args()
    rows = [row_for_run(args.runs_dir, r) for r in args.runs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "status", "best_epoch", "final_epoch", "best_eer", "lambda_at_save"]
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {args.output}\n")
    print(f'{"run":<40}{"status":<18}{"best_ep":<8}{"final":<7}{"best_eer":<10}{"λ@save":<10}')
    for r in rows:
        print(
            f"{r['run']:<40}{r['status']:<18}"
            f"{str(r.get('best_epoch', '')):<8}"
            f"{str(r.get('final_epoch', '')):<7}"
            f"{(f'{r[\"best_eer\"]:.4f}' if r.get('best_eer') is not None else ''):<10}"
            f"{(f'{r[\"lambda_at_save\"]:.3f}' if r.get('lambda_at_save') is not None else ''):<10}"
        )


if __name__ == "__main__":
    main()
