# ASVspoof 5 Domain-Invariant CM

Domain-adversarial training (DANN) for codec-robust speech deepfake detection.

**Thesis:** *Can domain-adversarial training improve generalization of speech deepfake detectors to unseen transmission codecs?*

## Results

| Model | Backbone | Eval EER | OOD Gap | minDCF |
|-------|----------|----------|---------|--------|
| **DANN** | **WavLM** | **7.34%** | **+54%** | **0.585** |
| ERM | WavLM | 8.47% | +160% | 0.639 |
| DANN | W2V2 | 14.33% | +222% | 1.000 |
| ERM | W2V2 | 15.30% | +261% | 1.000 |

DANN reduces the OOD gap by 66% (160% → 54%) compared to standard ERM training.

## Setup

We use [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/Jmqcooper1/asvspoof5-domain-invariant-cm.git
cd asvspoof5-domain-invariant-cm
uv venv && source .venv/activate
uv pip install -e ".[dev]"

# Verify
uv run pytest tests/ -v --tb=short
```

## Dataset

ASVspoof 5 is hosted on [Zenodo](https://zenodo.org/records/14498691) (~100GB full).

### Fast Download (aria2)

The download script uses `aria2c` for parallel downloading (16 connections by default):

```bash
# Set data directory
export ASVSPOOF5_ROOT=/path/to/data

# Download (protocols + minimal subset for dev)
bash scripts/download_asvspoof5.sh --ipv4

# Download full dataset (train + dev + eval)
bash scripts/download_asvspoof5.sh --full --ipv4 --parallel 16

# Unpack
bash scripts/unpack_asvspoof5.sh

# Create manifests
uv run python scripts/make_manifest.py --validate
```

**Tips:**
- Use `--ipv4` if IPv6 is slow
- Install `aria2c` for 10-20x faster downloads vs wget/curl
- `--parallel N` controls connections per file (max 16 for aria2)

## Running on Snellius (SLURM)

### Prerequisites

1. Stage the dataset tarballs to `$ASVSPOOF5_ROOT` (download separately or transfer)
2. Create `.env` with your paths:

```bash
cp .env.example .env
```

Example `.env` for Snellius:
```bash
# Dataset location
ASVSPOOF5_ROOT=/projects/prjs1904/data/asvspoof5

# Pre-computed augmentations cache (speeds up DANN training ~5x)
AUGMENTATION_CACHE_DIR=/scratch-shared/jcooper/asvspoof5_augmented_cache

# Checkpoints and results (use persistent storage, not scratch)
RUNS_DIR=/projects/prjs1904/runs

# HuggingFace model cache
HF_HOME=/scratch-shared/jcooper/.cache/huggingface

# Wandb (optional)
WANDB_API_KEY=your_key
WANDB_PROJECT=asvspoof5-dann
```

### Submit All Jobs

```bash
# Dry run (see what would be submitted)
./scripts/jobs/submit_all.sh --dry-run

# Submit full pipeline
./scripts/jobs/submit_all.sh

# Skip data staging if already done
./scripts/jobs/submit_all.sh --skip-staging
```

### Job Pipeline

```mermaid
flowchart TD
    subgraph Setup
        A[stage_dataset.job<br><i>Unpack tarballs, create manifests</i>]
        B[setup_environment.job<br><i>Download HuggingFace models</i>]
    end

    subgraph Training [Training - parallel]
        C1[train_wavlm_erm.job]
        C2[train_wavlm_erm_aug.job]
        C3[train_wavlm_dann.job]
        C4[train_wavlm_dann_exp.job]
        C5[train_w2v2_erm.job]
        C6[train_w2v2_erm_aug.job]
        C7[train_w2v2_dann.job]
    end

    subgraph Analysis [Analysis - after training]
        D[evaluate_models.job<br><i>Eval all checkpoints</i>]
        E[probe_domain.job<br><i>ERM vs DANN domain probes</i>]
        F[run_analysis.job<br><i>CKA, activation patching</i>]
    end

    A --> B
    B --> C1 & C2 & C3 & C4 & C5 & C6 & C7
    C1 & C2 & C3 & C4 & C5 & C6 & C7 --> D & E & F
```

### Monitor Jobs

```bash
squeue -u $USER              # Check status
scancel <job_id>             # Cancel job
cat scripts/jobs/out/*.out   # View logs
```

## Local Training

```bash
export ASVSPOOF5_ROOT=/path/to/data

# Train WavLM ERM baseline
uv run python scripts/train.py --config configs/wavlm_erm.yaml --name wavlm_erm

# Train WavLM DANN
uv run python scripts/train.py --config configs/wavlm_dann.yaml --name wavlm_dann

# Evaluate
uv run python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt --per-domain
```

## Analysis Scripts

```bash
# Domain probes (ERM vs DANN comparison)
uv run python scripts/probe_domain.py \
    --erm-checkpoint runs/wavlm_erm/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann/checkpoints/best.pt

# CKA representation similarity
uv run python scripts/run_cka.py \
    --erm-checkpoint runs/wavlm_erm/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann/checkpoints/best.pt

# Activation patching
uv run python scripts/run_patching.py \
    --source runs/wavlm_dann/checkpoints/best.pt \
    --target runs/wavlm_erm/checkpoints/best.pt
```

## Project Structure

```
configs/           YAML configs for models and training
scripts/           Training, evaluation, analysis scripts
scripts/jobs/      SLURM job files for cluster
src/               Main package code
tests/             Unit tests
runs/              Experiment outputs (created at runtime)
```

## Key Configs

| Config | Description |
|--------|-------------|
| `wavlm_erm.yaml` | WavLM + ERM baseline |
| `wavlm_dann.yaml` | WavLM + DANN (linear λ) |
| `wavlm_dann_exponential.yaml` | WavLM + DANN (exponential λ, best) |
| `w2v2_erm.yaml` | Wav2Vec2 + ERM |
| `w2v2_dann.yaml` | Wav2Vec2 + DANN |

## Citation

```bibtex
@mastersthesis{cooper2026dann,
  author = {Cooper, Mike},
  title = {Domain-Adversarial Training for Codec-Robust Speech Deepfake Detection},
  school = {University of Amsterdam},
  year = {2026}
}
```

## License

MIT
