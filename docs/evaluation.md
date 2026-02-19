# Evaluation

## Metrics

| Metric | Description | Primary |
|--------|-------------|---------|
| **minDCF** | Minimum Detection Cost Function (p_target=0.05) | ✓ |
| **EER** | Equal Error Rate | Secondary |
| Cllr | Log-likelihood ratio cost | Optional |

## Score Convention

**Higher score = more likely bonafide**

| Label | Value |
|-------|-------|
| Bonafide | 0 |
| Spoof | 1 |

## Per-Domain Evaluation

Per-domain breakdown is **only meaningful on eval set** (train/dev have no codec diversity).

| Split | Codec Diversity | Per-Domain Analysis |
|-------|-----------------|---------------------|
| Train | None | N/A |
| Dev | None | N/A |
| Eval | C01-C11 + uncoded | ✓ |

## Domain Invariance Metrics

| Metric | Description |
|--------|-------------|
| **OOD Gap** | Eval EER − Dev EER (lower = better generalization) |
| **Probe Accuracy** | Linear classifier predicting codec from embeddings (lower = more invariant) |
| **CKA** | Representation similarity between ERM and DANN |

## Statistical Reliability

- Bootstrap CIs (n=1000) for eval metrics
- Multiple seeds recommended for training variance
