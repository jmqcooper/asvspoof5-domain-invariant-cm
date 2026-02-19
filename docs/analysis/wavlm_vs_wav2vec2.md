# WavLM vs Wav2Vec2: Architectural Differences and DANN Performance

## Executive Summary

WavLM DANN achieves **3.59% EER** (epoch 9), outperforming ERM baseline.
Wav2Vec2 DANN achieves **4.87% EER** (epoch 11), **worse** than its ERM baseline (4.24%).

This document analyzes the architectural differences that may explain why DANN works
effectively with WavLM but underperforms with wav2vec2.

---

## 1. Architectural Differences

### 1.1 Core Architecture Comparison

| Feature | WavLM Base+ | Wav2Vec2 Base |
|---------|-------------|---------------|
| Transformer Layers | 12 | 12 |
| Hidden Dimension | 768 | 768 |
| CNN Feature Extractor | 7-layer, same config | 7-layer |
| Position Encoding | **Gated Relative Position Bias** | Convolutional Position Embeddings |
| Pretraining Objective | **Masked Speech Denoising** + Prediction | Contrastive Learning |
| Training Data | 94k hours (denoised/overlapped) | 60k hours (clean speech) |

### 1.2 Key Architectural Differences

#### Gated Relative Position Bias (WavLM only)

WavLM replaces wav2vec2's convolutional position embeddings with a **gated relative 
position bias** mechanism in the self-attention layers:

```
attention_scores = Q @ K^T / sqrt(d) + gated_bias(rel_pos)
```

where `gated_bias` applies a learned gating mechanism:

```
gated_bias = gate * bias
gate = sigmoid(W_g @ hidden_state)
```

**Impact on codec encoding:**
- Adaptive weighting of local temporal context
- May help the model distinguish content from acoustic artifacts
- Could enable more flexible separation of speaker/content from channel effects

#### Denoising Pretraining (WavLM only)

WavLM is trained with a **denoising objective** where:
1. Clean speech is mixed with noise/overlapped utterances
2. Model must predict clean targets from corrupted input
3. Forces learning of robust representations

**Impact on codec encoding:**
- Codec compression artifacts are similar to certain types of corruption
- Model may have learned to be partially invariant to such distortions
- Representations may already have some codec robustness "baked in"

### 1.3 Pretraining Data Differences

| Aspect | WavLM | Wav2Vec2 |
|--------|-------|----------|
| Hours | 94,000 | 60,000 |
| Sources | LibriLight, VoxPopuli, GigaSpeech | LibriLight |
| Data Quality | Mixed (clean + noisy) | Primarily clean |
| Speaker Diversity | Higher | Lower |

---

## 2. Hypotheses for DANN Performance Difference

### Hypothesis 1: Denoising Pretraining Creates Natural Codec Invariance

WavLM's denoising objective may have already learned to be partially invariant
to compression artifacts:
- Codec compression introduces similar distortions to the noise used in pretraining
- The model learns to recover clean representations from degraded input
- Result: Less codec information leaks into the final representations

**Prediction:** WavLM layers should show lower codec probe accuracy than wav2vec2

### Hypothesis 2: Gated Relative Position Bias Enables Content-Channel Separation

The gating mechanism in WavLM's position bias allows the model to:
- Dynamically adjust attention based on input content
- Potentially attend differently to codec-affected regions
- Better separate phonetic content from channel characteristics

**Prediction:** WavLM should show more uniform codec encoding across layers

### Hypothesis 3: Layer-wise Codec Distribution Differences

Different pretraining objectives may create different layer-wise distributions
of codec information:
- Wav2vec2: Codec info may be concentrated in specific layers
- WavLM: Codec info may be more uniformly distributed

**Impact:** If codec info is concentrated in specific layers for wav2vec2,
DANN's gradient reversal may not effectively suppress all codec information.

### Hypothesis 4: Gradient Flow Differences

The gated mechanism in WavLM may create different gradient dynamics:
- Smoother gradient flow through gated layers
- More stable adversarial training
- Better coordination between task and domain losses

---

## 3. Recommendations for Improving Wav2Vec2 DANN

### 3.1 Layer Selection Strategy

Instead of using `weighted` layer selection for all 12 layers, try:

```yaml
backbone:
  layer_selection:
    method: first_k
    k: 6
```

This focuses on early layers which typically encode less codec information.

### 3.2 Backbone Unfreezing

Consider unfreezing the top layers of wav2vec2:

```yaml
backbone:
  freeze: false
  freeze_layers: [0, 1, 2, 3, 4, 5, 6, 7]  # Freeze lower 8 layers
```

This allows DANN to learn codec-invariant representations directly.

### 3.3 Stronger Adversarial Pressure

Increase gradient reversal strength for wav2vec2:

```yaml
dann:
  lambda_schedule:
    start: 0.05  # Higher initial lambda
    end: 2.0     # Higher final lambda
```

### 3.4 Domain Discriminator Architecture

Try a deeper discriminator to better capture wav2vec2's codec patterns:

```yaml
dann:
  discriminator:
    hidden_dim: 1024
    num_layers: 3
```

### 3.5 Different Pooling Strategy

Consider attention pooling instead of stats pooling:

```yaml
pooling:
  method: attention
  heads: 4
```

---

## 4. Analysis Script

Use the provided script to empirically validate these hypotheses:

```bash
# Compare codec encoding between backbones
python scripts/compare_backbone_codec_probes.py --num-samples 3000

# With DANN checkpoints to analyze learned layer weights
python scripts/compare_backbone_codec_probes.py \
    --wavlm-checkpoint runs/wavlm_dann/best.pt \
    --w2v2-checkpoint runs/w2v2_dann/best.pt
```

This will generate:
- Layer-wise codec probe accuracy comparison
- Heatmap of codec encoding strength
- Analysis report with specific recommendations

---

## 5. References

1. Chen et al. (2022). "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack 
   Speech Processing." IEEE JSTSP. https://arxiv.org/abs/2110.13900

2. Baevski et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of 
   Speech Representations." NeurIPS. https://arxiv.org/abs/2006.11477

3. Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR.
   https://arxiv.org/abs/1505.07818
