# Methodology

## Training Paradigms

| Method | Description |
|--------|-------------|
| **ERM** | Standard supervised training (bonafide/spoof classification only) |
| **DANN** | Domain-adversarial training with gradient reversal |

## Architecture

```
Audio → SSL Backbone (frozen) → Layer Mixing → Projection Head → Task Classifier
                                                      ↓
                                              GRL → Domain Discriminator
```

### Components

| Component | Details |
|-----------|---------|
| **SSL Backbone** | WavLM Base+ or Wav2Vec2 Base (12 layers, 768-dim, frozen) |
| **Layer Mixing** | Learnable weighted sum of all layer outputs |
| **Projection Head** | MLP: 768 → 512 → 256 |
| **Task Classifier** | Linear: 256 → 2 (bonafide/spoof) |
| **Domain Discriminator** | Multi-head: CODEC (6 classes) + CODEC_Q (6 classes) |

## DANN Training

### Domain Labels

| Context | Source | Classes |
|---------|--------|---------|
| Training | Synthetic augmentation | NONE, MP3, AAC, OPUS, SPEEX, AMR |
| Evaluation | Protocol metadata | C01-C11 + uncoded |

Train/dev have no native codec diversity — synthetic augmentation creates domain labels.

### Loss Function

```
L_total = L_task + λ × (L_codec + L_codec_q)
```

- **L_task**: Cross-entropy for bonafide/spoof
- **L_codec / L_codec_q**: Cross-entropy for domain prediction (after GRL)
- **λ**: Adversarial weight (default: exponential schedule 0→1)

### Gradient Reversal Layer (GRL)

- Forward: identity
- Backward: negate gradients by λ

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (AdamW) |
| Batch size | 32 |
| Max epochs | 10 |
| λ schedule | Exponential (0→1) |
| Projection dim | 256 |
| Dropout | 0.1 |

## Baselines

1. **ERM**: Same architecture, no domain discriminator
2. **TRILLsson**: Frozen embeddings + logistic regression
3. **LFCC-GMM**: Classical features + GMM classifier
