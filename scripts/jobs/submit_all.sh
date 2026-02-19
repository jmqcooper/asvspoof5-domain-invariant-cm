#!/bin/bash
# =============================================================================
# Master Launcher Script for ASVspoof 5 Domain-Invariant CM Pipeline
# Submits all jobs with proper dependency management to SLURM queue
#
# Usage:
#   ./scripts/jobs/submit_all.sh [--dry-run] [--skip-staging] [--skip-baselines] [--run-held-out]
#
# Options:
#   --dry-run        Show what would be submitted without actually submitting
#   --skip-staging   Skip data staging/setup phases (use when data already staged)
#   --skip-baselines Skip LFCC/MFCC baseline jobs
#   --run-held-out   Enable held-out codec experiment (disabled by default because
#                    ASVspoof5 train/dev have no codec diversity)
# =============================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
DRY_RUN=false
SKIP_BASELINES=false
# NOTE: Held-out codec experiment is disabled by default because ASVspoof5
# train/dev sets have no codec diversity (all samples are NONE/uncoded).
# Codec diversity only exists in the eval set.
# See docs/evaluation.md for details.
SKIP_HELD_OUT=true
SKIP_STAGING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-baselines)
            SKIP_BASELINES=true
            shift
            ;;
        --run-held-out)
            # Enable held-out experiment (disabled by default)
            SKIP_HELD_OUT=false
            shift
            ;;
        --skip-staging)
            SKIP_STAGING=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--dry-run] [--skip-staging] [--skip-baselines] [--run-held-out]"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Ensure job output directory exists
mkdir -p ./scripts/jobs/out

# Require ASVSPOOF5_ROOT before submitting any jobs
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
if [ -z "${ASVSPOOF5_ROOT:-}" ]; then
    echo -e "${RED}ERROR: ASVSPOOF5_ROOT is not set.${NC}"
    echo -e "${RED}       Set it in .env or export it before running submit_all.sh.${NC}"
    echo ""
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No jobs will be submitted${NC}"
    echo ""
fi

if [ "$SKIP_STAGING" = true ]; then
    echo -e "${YELLOW}SKIP STAGING MODE - Assuming data is already staged${NC}"
    echo ""
fi

if [ "$DRY_RUN" = false ]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo -e "${RED}ERROR: sbatch is not available in this shell.${NC}"
        echo -e "${RED}       Load your Slurm environment/module and try again.${NC}"
        echo ""
        exit 1
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ASVspoof 5 Domain-Invariant CM       ${NC}"
echo -e "${BLUE}  Pipeline Job Submission              ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Job scripts
STAGE_SCRIPT="scripts/jobs/stage_dataset.job"
SETUP_SCRIPT="scripts/jobs/setup_environment.job"
TRAIN_SCRIPTS=(
    "scripts/jobs/train_wavlm_erm.job"
    "scripts/jobs/train_wavlm_erm_aug.job"
    "scripts/jobs/train_wavlm_dann.job"
    "scripts/jobs/train_wavlm_dann_exp.job"  # Exponential λ schedule (better for WavLM)
    "scripts/jobs/train_w2v2_erm.job"
    "scripts/jobs/train_w2v2_erm_aug.job"
    "scripts/jobs/train_w2v2_dann.job"       # v2 config with first_k=6
)
EVAL_SCRIPT="scripts/jobs/evaluate_models.job"
PROBE_SCRIPT="scripts/jobs/probe_domain.job"  # ERM vs DANN domain probe comparison
ANALYSIS_SCRIPT="scripts/jobs/run_analysis.job"
BASELINES_SCRIPT="scripts/jobs/run_baselines.job"
HELD_OUT_SCRIPT="scripts/jobs/run_held_out.job"

# Track job IDs
declare -a ALL_JOB_IDS
declare -a TRAIN_JOB_IDS

# Function to extract job info
get_job_info() {
    local script=$1
    if [ -f "$script" ]; then
        local job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        local partition=$(grep -m1 "#SBATCH --partition=" "$script" | cut -d'=' -f2)
        local time=$(grep -m1 "#SBATCH --time=" "$script" | cut -d'=' -f2)
        echo "$job_name ($partition, $time)"
    else
        echo "${RED}NOT FOUND${NC}"
    fi
}

# Show job summary
echo "Jobs to submit:"
echo "---------------"
echo ""
if [ "$SKIP_STAGING" = true ]; then
    echo -e "  ${YELLOW}Phase 1: Dataset staging - SKIPPED${NC}"
    echo -e "  ${YELLOW}Phase 1b: Setup - SKIPPED${NC}"
else
    echo -e "  ${GREEN}Phase 1: Dataset staging${NC}"
    echo "    - $(get_job_info $STAGE_SCRIPT)"
    echo ""
    echo -e "  ${GREEN}Phase 1b: Setup${NC}"
    echo "    - $(get_job_info $SETUP_SCRIPT)"
fi
echo ""
echo -e "  ${GREEN}Phase 2: Training (parallel)${NC}"
for script in "${TRAIN_SCRIPTS[@]}"; do
    echo "    - $(get_job_info $script)"
done
echo ""
if [ "$SKIP_BASELINES" = false ]; then
    echo -e "  ${GREEN}Phase 2b: Baselines (parallel with training)${NC}"
    echo "    - $(get_job_info $BASELINES_SCRIPT)"
    echo ""
fi
echo -e "  ${GREEN}Phase 3: Evaluation (after training)${NC}"
echo "    - $(get_job_info $EVAL_SCRIPT)"
echo ""
echo -e "  ${GREEN}Phase 4: Analysis (after ERM+DANN training)${NC}"
echo "    - $(get_job_info $ANALYSIS_SCRIPT)"
echo ""
echo -e "  ${GREEN}Phase 5: Domain Probe Comparison (after training)${NC}"
echo "    - $(get_job_info $PROBE_SCRIPT)"
echo ""
if [ "$SKIP_HELD_OUT" = false ]; then
    echo -e "  ${GREEN}Phase 6: Held-Out Codec Experiment${NC}"
    echo "    - $(get_job_info $HELD_OUT_SCRIPT)"
    echo ""
fi

# Confirmation
if [ "$DRY_RUN" = false ]; then
    read -p "Submit all jobs? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

echo "Submitting jobs..."
echo "==================="
echo ""

# Phase 1: Dataset staging
STAGE_JOB_ID=""
SETUP_JOB_ID=""

if [ "$SKIP_STAGING" = true ]; then
    echo -e "${YELLOW}Phase 1: Dataset staging - SKIPPED${NC}"
    echo -e "${YELLOW}Phase 1b: Setup - SKIPPED${NC}"
    echo ""
else
    echo -e "${BLUE}Phase 1: Dataset staging${NC}"
    if [ -f "$STAGE_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$STAGE_SCRIPT" | cut -d'=' -f2)

        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC}"
            STAGE_JOB_ID="DRYRUN_STAGE"
        else
            set +e
            output=$(sbatch --chdir="$PROJECT_ROOT" "$STAGE_SCRIPT" 2>&1)
            sbatch_exit_code=$?
            set -e
            if [ $sbatch_exit_code -ne 0 ]; then
                echo -e "  ${RED}Failed to submit: $output${NC}"
                exit 1
            elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                STAGE_JOB_ID="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$STAGE_JOB_ID")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $STAGE_JOB_ID)"
            else
                echo -e "  ${RED}Failed to submit: $output${NC}"
                exit 1
            fi
        fi
    else
        echo -e "  ${RED}Stage script not found: $STAGE_SCRIPT${NC}"
        exit 1
    fi
    echo ""

    # Phase 1b: Setup (depends on staging)
    echo -e "${BLUE}Phase 1b: Setup${NC}"
    if [ -f "$SETUP_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$SETUP_SCRIPT" | cut -d'=' -f2)

        if [ "$DRY_RUN" = true ]; then
            echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after staging)"
            SETUP_JOB_ID="DRYRUN_SETUP"
        else
            set +e
            output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$STAGE_JOB_ID "$SETUP_SCRIPT" 2>&1)
            sbatch_exit_code=$?
            set -e
            if [ $sbatch_exit_code -ne 0 ]; then
                echo -e "  ${RED}Failed to submit: $output${NC}"
                exit 1
            elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                SETUP_JOB_ID="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$SETUP_JOB_ID")
                echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $SETUP_JOB_ID, depends on: $STAGE_JOB_ID)"
            else
                echo -e "  ${RED}Failed to submit: $output${NC}"
                exit 1
            fi
        fi
    else
        echo -e "  ${RED}Setup script not found: $SETUP_SCRIPT${NC}"
        exit 1
    fi
    echo ""
fi

# Phase 2: Training (depends on setup, or no dependency if --skip-staging)
echo -e "${BLUE}Phase 2: Training${NC}"
for script in "${TRAIN_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$script" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            if [ "$SKIP_STAGING" = true ]; then
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (no dependency)"
            else
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
            fi
        else
            set +e
            if [ "$SKIP_STAGING" = true ]; then
                output=$(sbatch --chdir="$PROJECT_ROOT" "$script" 2>&1)
            else
                output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$SETUP_JOB_ID "$script" 2>&1)
            fi
            sbatch_exit_code=$?
            set -e
            if [ $sbatch_exit_code -ne 0 ]; then
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                TRAIN_JOB_IDS+=("$job_id")
                if [ "$SKIP_STAGING" = true ]; then
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id)"
                else
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
                fi
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
done
echo ""

# Phase 2b: Baselines (depends on setup, parallel with training)
if [ "$SKIP_BASELINES" = false ]; then
    echo -e "${BLUE}Phase 2b: Baselines${NC}"
    if [ -f "$BASELINES_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$BASELINES_SCRIPT" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            if [ "$SKIP_STAGING" = true ]; then
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (no dependency)"
            else
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
            fi
        else
            set +e
            if [ "$SKIP_STAGING" = true ]; then
                output=$(sbatch --chdir="$PROJECT_ROOT" "$BASELINES_SCRIPT" 2>&1)
            else
                output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$SETUP_JOB_ID "$BASELINES_SCRIPT" 2>&1)
            fi
            sbatch_exit_code=$?
            set -e
            if [ $sbatch_exit_code -ne 0 ]; then
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                if [ "$SKIP_STAGING" = true ]; then
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id)"
                else
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
                fi
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
    echo ""
fi

# Phase 3: Evaluation (depends on all training jobs)
echo -e "${BLUE}Phase 3: Evaluation${NC}"
if [ -f "$EVAL_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$EVAL_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after training)"
    else
        # Build dependency string for all training jobs
        TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
        set +e
        output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$TRAIN_DEPS "$EVAL_SCRIPT" 2>&1)
        sbatch_exit_code=$?
        set -e
        if [ $sbatch_exit_code -ne 0 ]; then
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$job_id")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $TRAIN_DEPS)"
        else
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        fi
    fi
fi
echo ""

# Phase 4: Analysis (depends on training - specifically ERM and DANN pairs)
echo -e "${BLUE}Phase 4: Analysis${NC}"
if [ -f "$ANALYSIS_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$ANALYSIS_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after training)"
    else
        TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
        set +e
        output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$TRAIN_DEPS "$ANALYSIS_SCRIPT" 2>&1)
        sbatch_exit_code=$?
        set -e
        if [ $sbatch_exit_code -ne 0 ]; then
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$job_id")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $TRAIN_DEPS)"
        else
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        fi
    fi
fi
echo ""

# Phase 5: Domain Probe Comparison (depends on training)
echo -e "${BLUE}Phase 5: Domain Probe Comparison${NC}"
if [ -f "$PROBE_SCRIPT" ]; then
    job_name=$(grep -m1 "#SBATCH --job-name=" "$PROBE_SCRIPT" | cut -d'=' -f2)
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after training)"
    else
        TRAIN_DEPS=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
        set +e
        output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$TRAIN_DEPS "$PROBE_SCRIPT" 2>&1)
        sbatch_exit_code=$?
        set -e
        if [ $sbatch_exit_code -ne 0 ]; then
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
            job_id="${BASH_REMATCH[1]}"
            ALL_JOB_IDS+=("$job_id")
            echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $TRAIN_DEPS)"
        else
            echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
        fi
    fi
else
    echo -e "  ${YELLOW}Probe script not found: $PROBE_SCRIPT${NC}"
fi
echo ""

# Phase 6: Held-Out Codec Experiment (depends on setup only - does its own training)
if [ "$SKIP_HELD_OUT" = false ]; then
    echo -e "${BLUE}Phase 6: Held-Out Codec Experiment${NC}"
    if [ -f "$HELD_OUT_SCRIPT" ]; then
        job_name=$(grep -m1 "#SBATCH --job-name=" "$HELD_OUT_SCRIPT" | cut -d'=' -f2)
        
        if [ "$DRY_RUN" = true ]; then
            if [ "$SKIP_STAGING" = true ]; then
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (no dependency)"
            else
                echo -e "  [DRY RUN] Would submit: ${GREEN}$job_name${NC} (after setup)"
            fi
        else
            set +e
            if [ "$SKIP_STAGING" = true ]; then
                output=$(sbatch --chdir="$PROJECT_ROOT" "$HELD_OUT_SCRIPT" 2>&1)
            else
                output=$(sbatch --chdir="$PROJECT_ROOT" --dependency=afterok:$SETUP_JOB_ID "$HELD_OUT_SCRIPT" 2>&1)
            fi
            sbatch_exit_code=$?
            set -e
            if [ $sbatch_exit_code -ne 0 ]; then
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            elif [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                ALL_JOB_IDS+=("$job_id")
                if [ "$SKIP_STAGING" = true ]; then
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id)"
                else
                    echo -e "  Submitted: ${GREEN}$job_name${NC} (Job ID: $job_id, depends on: $SETUP_JOB_ID)"
                fi
            else
                echo -e "  ${YELLOW}Failed to submit $job_name: $output${NC}"
            fi
        fi
    fi
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Summary                               ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "Would submit ${GREEN}${#ALL_JOB_IDS[@]}${NC} jobs"
else
    echo -e "Submitted ${GREEN}${#ALL_JOB_IDS[@]}${NC} jobs"
    echo ""
    echo "Job IDs: ${ALL_JOB_IDS[*]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel all with:"
    echo "  scancel ${ALL_JOB_IDS[*]}"
fi
echo ""
