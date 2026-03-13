# Multi-Seed Experiments

Run each model configuration with 3 seeds to measure variance and establish statistical significance.

**Seed 42 is already complete** for all configs. This job runs seeds 123 and 456 only.

## Configurations

| Config | Seed 42 | Seeds 123 + 456 |
|--------|---------|-----------------|
| `wavlm_erm` | ✅ Done | 2 new runs |
| `wavlm_dann` | ✅ Done | 2 new runs |
| `wavlm_erm_aug` | ✅ Done | 2 new runs |
| `w2v2_erm` | ✅ Done | 2 new runs |
| `w2v2_dann` | ✅ Done | 2 new runs |
| `w2v2_erm_aug` | ✅ Done | 2 new runs |

## Prerequisites

### Pre-compute augmentation cache (speeds up training ~5x)

```bash
JOB_ID=$(sbatch --parsable scripts/jobs/precompute_augmentations.job)
sbatch --dependency=afterok:$JOB_ID scripts/jobs/merge_augmentation_manifests.job
```

Wait for both jobs to complete before submitting training.

## Running Experiments

Submit all 6 configs (each launches 2 array jobs = 12 total new runs):

```bash
for cfg in wavlm_erm wavlm_dann wavlm_erm_aug w2v2_erm w2v2_dann w2v2_erm_aug; do
  sbatch scripts/jobs/train_multi_seed.job $cfg
done
```

Monitor progress:

```bash
squeue -u $USER
```

Each array job trains with one seed, then evaluates on both `dev` and `eval` splits with per-domain breakdown, bootstrap CIs, and score files.

## Output Structure

```
results/predictions/
├── wavlm_erm_seed123_dev/
│   ├── predictions.tsv
│   └── metrics.json
├── wavlm_erm_seed123_eval/
│   ├── predictions.tsv
│   └── metrics.json
├── wavlm_erm_seed456_dev/
│   └── ...
├── wavlm_erm_seed456_eval/
│   └── ...
└── ...
```

Predictions TSV has columns: `flac_file, score, prediction, y_task, y_codec, y_codec_q, ...`

**Note:** Seed 42 results are in `$RUNS_DIR/{config}/eval_eval_full/` — the DET curve script searches both `results/predictions/` and `results/runs/` automatically.

## Generating Figures (after all training completes)

### DET Curves

```bash
# From existing single-seed results
python scripts/plot_det_curves.py --predictions-dir results/runs/

# Or from multi-seed predictions directory
python scripts/plot_det_curves.py --predictions-dir results/predictions/

# Demo mode (synthetic curves for layout testing)
python scripts/plot_det_curves.py --demo
```

Output: `figures/det_curves.{png,pdf}`

### PCA of Representations

```bash
# With real data (after extracting representations via RQ4 notebook)
python scripts/plot_pca.py \
    --erm-repr results/representations/erm_proj.npy \
    --dann-repr results/representations/dann_proj.npy \
    --labels results/representations/codec_labels.npy

# Demo mode
python scripts/plot_pca.py --demo
```

Output: `figures/pca_representations.{png,pdf}`

## Metrics

Both EER and minDCF are computed with corrected ASVspoof 5 Track 1 parameters:
- `C_miss=1, C_fa=10, π_spf=0.05` (β ≈ 1.90)

Results are logged to W&B for systematic comparison.
