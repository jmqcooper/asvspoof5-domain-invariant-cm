# Multi-Seed Experiments

Run each model configuration with 5 seeds (42, 123, 456, 789, 1337) to measure variance and generate DET curves.

## Configurations

| Config | Description |
|--------|-------------|
| `wavlm_erm` | WavLM backbone, standard ERM training |
| `wavlm_dann` | WavLM backbone, DANN domain adaptation |
| `w2v2_erm` | Wav2Vec2 backbone, standard ERM training |
| `w2v2_dann` | Wav2Vec2 backbone, DANN domain adaptation |

## Running Experiments

Submit all 4 configs (each launches 5 array jobs = 20 total runs):

```bash
sbatch scripts/jobs/train_multi_seed.job wavlm_erm
sbatch scripts/jobs/train_multi_seed.job wavlm_dann
sbatch scripts/jobs/train_multi_seed.job w2v2_erm
sbatch scripts/jobs/train_multi_seed.job w2v2_dann
```

Monitor progress:

```bash
squeue -u $USER
```

Each array job trains with one seed, then evaluates on both `dev` and `eval` splits. Predictions are saved to `results/predictions/`.

## Output Structure

```
results/predictions/
├── wavlm_erm_seed42_eval.csv
├── wavlm_erm_seed42_dev.csv
├── wavlm_erm_seed123_eval.csv
├── ...
├── w2v2_dann_seed1337_eval.csv
└── w2v2_dann_seed1337_dev.csv
```

Each CSV has columns: `flac_file, score, prediction, y_task, codec`.

## Generating Figures

### DET Curves

```bash
# With real predictions (after experiments complete)
python scripts/plot_det_curves.py --predictions-dir results/predictions/

# Demo mode (synthetic curves for layout testing)
python scripts/plot_det_curves.py --demo
```

Output: `master-thesis-uva/figures/det_curves.{png,pdf}`

### PCA of Representations

First extract representations (see `scripts/extract_representations.py`), then:

```bash
# With real data
python scripts/plot_pca.py \
    --erm-repr results/representations/erm_proj.npy \
    --dann-repr results/representations/dann_proj.npy \
    --labels results/representations/codec_labels.npy

# Demo mode
python scripts/plot_pca.py --demo
```

Output: `master-thesis-uva/figures/pca_representations.{png,pdf}`

## Aggregating Results

To compute mean ± std EER across seeds for a config:

```bash
# Example: collect EER from all wavlm_erm seeds
for f in results/predictions/wavlm_erm_seed*_eval.csv; do
    echo "$f"
done
```

Use the evaluation metrics logged to W&B for systematic comparison, or parse the prediction CSVs with the DET curve script.
