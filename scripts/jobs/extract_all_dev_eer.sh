#!/usr/bin/env bash
# One-shot: dump best_eer (dev EER) for all 20 multi-seed runs into
# a single JSON. Run on cluster login node.
#
# Covers 5 seeds (42, 123, 456, 789, 2024) for ERM and DANN on both
# backbones. Run dirs for seeds 789/2024 are resolved by glob to avoid
# hardcoding the training-time commit suffix.

set -u

cd "${SLURM_SUBMIT_DIR:-.}"

RUNS_BASE="${RUNS_DIR:-/gpfs/work5/0/prjs1904/runs}"
OUT="results/dev_eer_multiseed.json"
mkdir -p "$(dirname "$OUT")"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked >/dev/null 2>&1 || true

uv run python - <<PY
import json
from pathlib import Path

RUNS_BASE = Path("$RUNS_BASE")

# (clean_name, run_dir_or_glob).
# Exact names for runs with known suffixes; glob patterns (resolved to
# newest match) for seeds 789/2024 trained on commit fca361a.
mapping = {
    "wavlm_erm_seed42":      "wavlm_erm",
    "wavlm_erm_seed123":     "wavlm_erm_seed123_5223811",
    "wavlm_erm_seed456":     "wavlm_erm_seed456_5223811",
    "wavlm_erm_seed789":     "wavlm_erm_seed789_*",
    "wavlm_erm_seed2024":    "wavlm_erm_seed2024_*",
    "wavlm_dann_seed42_v2":  "wavlm_dann_seed42_v2_1e2d5c7",
    "wavlm_dann_seed123":    "wavlm_dann_seed123_5223811",
    "wavlm_dann_seed456":    "wavlm_dann_seed456_5223811",
    "wavlm_dann_seed789":    "wavlm_dann_seed789_*",
    "wavlm_dann_seed2024":   "wavlm_dann_seed2024_*",
    "w2v2_erm_seed42":       "w2v2_erm",
    "w2v2_erm_seed123":      "w2v2_erm_seed123_5223811",
    "w2v2_erm_seed456":      "w2v2_erm_seed456_5223811",
    "w2v2_erm_seed789":      "w2v2_erm_seed789_*",
    "w2v2_erm_seed2024":     "w2v2_erm_seed2024_*",
    "w2v2_dann_seed42_v2":   "w2v2_dann_seed42_v2_1e2d5c7",
    "w2v2_dann_seed123":     "w2v2_dann_seed123_5223811",
    "w2v2_dann_seed456":     "w2v2_dann_seed456_5223811",
    "w2v2_dann_seed789":     "w2v2_dann_seed789_*",
    "w2v2_dann_seed2024":    "w2v2_dann_seed2024_*",
}


def resolve(pattern: str) -> Path | None:
    """Resolve a literal name or glob to the newest matching run dir."""
    if "*" not in pattern:
        candidate = RUNS_BASE / pattern
        return candidate if candidate.exists() else None
    matches = sorted(RUNS_BASE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


out = {}
for clean, pattern in mapping.items():
    run_dir = resolve(pattern)
    if run_dir is None:
        out[clean] = {"error": f"no match for {pattern}", "run_dir": pattern}
        continue
    p = run_dir / "metrics_train.json"
    try:
        d = json.load(open(p))
        out[clean] = {
            "best_epoch": d.get("best_epoch"),
            "best_eer": d.get("best_eer"),
            "run_dir": run_dir.name,
        }
    except Exception as e:
        out[clean] = {"error": str(e), "run_dir": run_dir.name}

output_path = Path("$OUT")
output_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {output_path}")
for k, v in out.items():
    eer = v.get("best_eer")
    print(f"  {k:<30} {eer if eer is not None else v.get('error','?')}")
PY
