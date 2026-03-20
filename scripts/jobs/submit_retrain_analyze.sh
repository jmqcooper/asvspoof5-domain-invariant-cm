#!/bin/bash
# =============================================================================
# Submit 4 parallel retrain jobs + dependent analysis job
#
# Usage: ./scripts/jobs/submit_retrain_analyze.sh [--dry-run]
# =============================================================================
set -e

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

cd "$(dirname "$0")/../.."
mkdir -p scripts/jobs/out

CONFIGS=(wavlm_dann wavlm_erm_aug w2v2_dann w2v2_erm_aug)
JOB_IDS=()

echo "=== Submitting 4 retrain jobs (parallel) ==="
for CONFIG in "${CONFIGS[@]}"; do
  if $DRY_RUN; then
    echo "  [DRY RUN] Would submit: retrain_seed42_single.job ${CONFIG}"
    JOB_IDS+=("DRYRUN")
  else
    output=$(sbatch scripts/jobs/retrain_seed42_single.job "${CONFIG}" 2>&1)
    if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      job_id="${BASH_REMATCH[1]}"
      JOB_IDS+=("${job_id}")
      echo "  Submitted: ${CONFIG} (Job ID: ${job_id})"
    else
      echo "  FAILED: ${CONFIG}: ${output}"
      exit 1
    fi
  fi
done

echo ""
echo "=== Submitting analysis job (depends on all 4 retrains) ==="
if $DRY_RUN; then
  echo "  [DRY RUN] Would submit: analyze_seed42.job (after retrains)"
else
  DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
  output=$(sbatch --dependency=afterok:${DEPS} scripts/jobs/analyze_seed42.job 2>&1)
  if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    analysis_id="${BASH_REMATCH[1]}"
    echo "  Submitted: analyze_seed42.job (Job ID: ${analysis_id}, depends on: ${DEPS})"
  else
    echo "  FAILED: ${output}"
    exit 1
  fi
fi

echo ""
echo "=== Summary ==="
echo "Retrain jobs: ${JOB_IDS[*]}"
echo "Analysis job: ${analysis_id:-DRY_RUN}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Original seed 42 results are UNTOUCHED (v2 runs use separate dirs)"
