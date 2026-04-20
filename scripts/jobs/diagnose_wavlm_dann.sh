#!/usr/bin/env bash
# One-shot diagnostic: compare pre-v2 vs v2 WavLM DANN, check multi-seed
# probe job status. Run on cluster login node after git pull.
#
# Usage (from repo root):
#     bash scripts/jobs/diagnose_wavlm_dann.sh

set -u

cd "${SLURM_SUBMIT_DIR:-.}"
if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root" >&2
  exit 1
fi

RUNS="${RUNS_DIR:-/gpfs/work5/0/prjs1904/runs}"
OUT="results/pool_weights"
mkdir -p "$OUT"

# ---------- 1. Slurm job status ----------
echo "=========================================="
echo "1. SLURM QUEUE (your jobs)"
echo "=========================================="
squeue -u "$USER" 2>/dev/null || echo "(squeue unavailable)"
echo ""
echo "=========================================="
echo "2. Recent ProbeMultiSeed job logs"
echo "=========================================="
shopt -s nullglob
probe_logs=( scripts/jobs/out/ProbeMultiSeed_*.out )
shopt -u nullglob
if (( ${#probe_logs[@]} == 0 )); then
  echo "  (no ProbeMultiSeed_*.out logs found -- job may not have been submitted)"
else
  for f in "${probe_logs[@]}"; do
    echo ""
    echo "--- $f ---"
    echo "(last 20 lines)"
    tail -20 "$f"
  done
fi
echo ""
echo "=========================================="
echo "3. Probe outputs written so far"
echo "=========================================="
if [[ -d results/domain_probes ]]; then
  for d in results/domain_probes/*/; do
    name=$(basename "$d")
    done_marker="no"
    [[ -f "$d/probe_results.json" ]] && done_marker="YES"
    echo "  $name  probe_results.json=$done_marker"
  done
else
  echo "  (results/domain_probes/ does not exist yet)"
fi
echo ""

# ---------- 4. Metrics for v2 DANN seeds ----------
echo "=========================================="
echo "4. Best-epoch metrics for v2 DANN seeds"
echo "=========================================="
for c in wavlm_dann_seed42_v2_1e2d5c7 wavlm_dann_seed123_5223811 wavlm_dann_seed456_5223811; do
  p="$RUNS/$c/metrics_train.json"
  echo ""
  echo "--- $c ---"
  if [[ -f "$p" ]]; then
    cat "$p"
  else
    echo "  (missing: $p)"
    # Look for alternatives
    ls "$RUNS/$c/" 2>/dev/null | head -20
  fi
done
echo ""

# ---------- 5. Extract pre-v2 WavLM DANN pool weights ----------
echo "=========================================="
echo "5. Pre-v2 wavlm_dann pool weights"
echo "=========================================="
PREV2_CKPT="$RUNS/wavlm_dann/checkpoints/best.pt"
PREV2_OUT="$OUT/wavlm_dann_prev2.json"
if [[ -f "$PREV2_CKPT" ]]; then
  if [[ -f "$PREV2_OUT" ]]; then
    echo "(Already extracted: $PREV2_OUT)"
  else
    # uv should already be on PATH from previous session; if not, install
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    uv run python scripts/extract_pool_weights.py \
      --checkpoint "$PREV2_CKPT" \
      --output "$PREV2_OUT"
  fi
  echo ""
  echo "--- contents of $PREV2_OUT ---"
  cat "$PREV2_OUT"
else
  echo "  (missing: $PREV2_CKPT)"
  echo "  Files under $RUNS/wavlm_dann/:"
  ls -la "$RUNS/wavlm_dann/" 2>/dev/null | head -20
  echo "  Files under $RUNS/wavlm_dann/checkpoints/:"
  ls -la "$RUNS/wavlm_dann/checkpoints/" 2>/dev/null | head -20
fi
echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
