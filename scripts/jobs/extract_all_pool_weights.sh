#!/usr/bin/env bash
# Extract softmax-normalized layer-pooling weights from all 12 multi-seed
# checkpoints (WavLM/W2V2 x ERM/DANN x 3 seeds). No GPU needed; runs on a
# login node in a few minutes.
#
# Usage (from repo root):
#     bash scripts/jobs/extract_all_pool_weights.sh
#
# Outputs: results/pool_weights/<clean_name>.json, one file per checkpoint.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found in $(pwd). Run from repo root." >&2
  exit 1
fi

# Install uv if needed; sync env
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked

RUNS_BASE="${RUNS_DIR:-/gpfs/work5/0/prjs1904/runs}"
OUT_DIR="results/pool_weights"
mkdir -p "$OUT_DIR"

# (cluster_run_dir_or_glob, clean_output_name) pairs.
# Clean names mirror the convention used in results/predictions/ so downstream
# scripts can join pool weights with probe results by name.
# For runs trained with new commit suffixes (seeds 789/2024 trained on
# fca361a) we use a glob pattern; resolve_ckpt picks the newest match.
CHECKPOINTS=(
  "wavlm_erm:wavlm_erm_seed42"
  "wavlm_erm_seed123_5223811:wavlm_erm_seed123"
  "wavlm_erm_seed456_5223811:wavlm_erm_seed456"
  "wavlm_erm_seed789_*:wavlm_erm_seed789"
  "wavlm_erm_seed2024_*:wavlm_erm_seed2024"
  "wavlm_dann_seed42_v2_1e2d5c7:wavlm_dann_seed42_v2"
  "wavlm_dann_seed123_5223811:wavlm_dann_seed123"
  "wavlm_dann_seed456_5223811:wavlm_dann_seed456"
  "wavlm_dann_seed789_*:wavlm_dann_seed789"
  "wavlm_dann_seed2024_*:wavlm_dann_seed2024"
  "w2v2_erm:w2v2_erm_seed42"
  "w2v2_erm_seed123_5223811:w2v2_erm_seed123"
  "w2v2_erm_seed456_5223811:w2v2_erm_seed456"
  "w2v2_erm_seed789_*:w2v2_erm_seed789"
  "w2v2_erm_seed2024_*:w2v2_erm_seed2024"
  "w2v2_dann_seed42_v2_1e2d5c7:w2v2_dann_seed42_v2"
  "w2v2_dann_seed123_5223811:w2v2_dann_seed123"
  "w2v2_dann_seed456_5223811:w2v2_dann_seed456"
  "w2v2_dann_seed789_*:w2v2_dann_seed789"
  "w2v2_dann_seed2024_*:w2v2_dann_seed2024"
)

resolve_ckpt() {
  # Resolve "${RUNS_BASE}/${pattern}/checkpoints/best.pt", taking the
  # most-recently-modified match if the pattern is a glob.
  # Variables must be unquoted at the glob site so bash performs
  # pathname expansion; otherwise "*" is treated literally.
  local pattern="$1"
  local -a candidates
  shopt -s nullglob
  # shellcheck disable=SC2206
  candidates=( ${RUNS_BASE}/${pattern}/checkpoints/best.pt )
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    echo ""
  else
    # Sort by mtime (newest first), keep first
    ls -1t "${candidates[@]}" 2>/dev/null | head -1
  fi
}

echo "=== Extracting pool weights from ${#CHECKPOINTS[@]} checkpoints ==="
echo "Runs base:  $RUNS_BASE"
echo "Output dir: $OUT_DIR"
echo ""

for entry in "${CHECKPOINTS[@]}"; do
  src_name="${entry%%:*}"
  out_name="${entry##*:}"
  ckpt=$(resolve_ckpt "$src_name")
  out_path="${OUT_DIR}/${out_name}.json"

  if [[ -f "$out_path" ]]; then
    echo "SKIP  $out_name (already exists)"
    continue
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "WARN  no checkpoint match for: ${RUNS_BASE}/${src_name}/checkpoints/best.pt" >&2
    continue
  fi

  uv run python scripts/extract_pool_weights.py \
    --checkpoint "$ckpt" \
    --output "$out_path"
done

echo ""
echo "=== Done. Files in $OUT_DIR/ ==="
ls -la "$OUT_DIR"/
