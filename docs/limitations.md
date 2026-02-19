# Known Limitations

This document describes known limitations and constraints of the codebase.

## Domain Mismatch

### Synthetic vs Real Codec Taxonomy

**Limitation:** Synthetic codec augmentation does not perfectly match ASVspoof5 eval codec taxonomy.

| Synthetic Codec | ASVspoof5 Codec | Match Quality |
|-----------------|-----------------|---------------|
| MP3 | C05 (mp3_wb) | Good |
| AAC | C06 (m4a_wb) | Good |
| OPUS | C01 (opus_wb), C08 (opus_nb) | Good |
| SPEEX | C03 (speex_wb), C10 (speex_nb) | Good (if encoder available) |
| AMR | C02 (amr_wb), C09 (amr_nb) | Good (if encoder available) |

### Uncovered Eval Codecs

| Eval Codec | Why Not Covered | Potential Mitigation |
|------------|-----------------|----------------------|
| **C04 (Encodec)** | Neural codec; fundamentally different artifacts | Would require Encodec library integration |
| **C07 (MP3+Encodec cascade)** | Compound degradation pattern | Would need cascaded pipeline |
| **C11 (Device/channel)** | Acoustic effects, not codec | Requires room impulse responses, device models |

**Expected Impact:** DANN may show limited or no improvement for C04, C07, C11 compared to ERM.

---

## CODEC_Q Semantic Mismatch

### Training vs Evaluation Quality Levels

**Limitation:** Quality levels have different meanings in training vs evaluation.

| Context | Quality Values | Semantics |
|---------|----------------|-----------|
| Training (synthetic) | 1-5 | Arbitrary bitrate tiers per codec |
| Eval (C01-C10) | 1-5 | Codec-specific bitrate tiers |
| Eval (C11) | 6-8 | Device variants (Bluetooth, cable, MST) |
| Uncoded | 0 / "-" | No codec applied |

**Implication:**
- CODEC_Q adversarial head learns to be invariant to synthetic tiers 1-5
- This does NOT directly transfer to eval CODEC_Q semantics
- Eval CODEC_Q breakdown is purely analytical, not training-aligned

### C11 Device Variants

Eval CODEC_Q values 6, 7, 8 correspond to:
- 6: MST (microphone simulation tool)
- 7: Bluetooth transmission
- 8: Cable transmission

These are **never seen during training** and have no synthetic equivalent.

---

## FFmpeg Encoder Availability

### Encoder Requirements

| Codec | Encoder | Status |
|-------|---------|--------|
| MP3 | libmp3lame | Usually available |
| AAC | aac (built-in) | Usually available |
| OPUS | libopus | Usually available |
| SPEEX | libspeex | Often missing |
| AMR | libopencore_amrnb | Often missing |

**Limitation:** SPEEX and AMR encoders are often not included in default ffmpeg builds.

### Fallback Behavior

If an encoder is missing:
1. Codec is excluded from `supported_codecs`
2. Training proceeds with remaining codecs
3. Warning logged: "Some requested codecs not supported by ffmpeg"

**Minimum requirement:** At least 2 supported codecs for meaningful DANN training.

### Installation Issues

Some systems require manual compilation of ffmpeg with additional encoders:

```bash
# Ubuntu: Install additional codec packages
sudo apt install libspeex-dev libopencore-amrnb-dev

# Rebuild ffmpeg or use a pre-built binary with more codecs
```

---

## Computational Considerations

### Training Time

| Model | GPU | Train Time (per epoch) | Full Training |
|-------|-----|------------------------|---------------|
| WavLM Base+ (frozen) | A100 40GB | ~1.5 hours | ~15-30 hours |
| Wav2Vec2 Base (frozen) | A100 40GB | ~1.2 hours | ~12-25 hours |
| WavLM Base+ (frozen) | Consumer GPU (RTX 3090) | ~3-4 hours | ~30-50 hours |

### Memory Requirements

| Model | Batch Size | GPU Memory |
|-------|------------|------------|
| WavLM Base+ (frozen) | 32 | ~16 GB |
| WavLM Base+ (frozen) | 16 | ~10 GB |
| WavLM Base+ (frozen) | 8 | ~6 GB |

**Recommendation:** Use batch size 32 on A100, reduce to 16 or 8 on consumer GPUs.

### Disk Usage

| Component | Size |
|-----------|------|
| Full dataset (train+dev+eval) | ~100 GB |
| Augmentation cache (full) | ~50-100 GB |
| Model checkpoints | ~400 MB per checkpoint |
| Probe/CKA embeddings | ~1-5 GB |

---

## Train/Dev Domain Homogeneity

### The Core Problem

**Train and dev sets have NO codec diversity.**

| Split | CODEC unique values | CODEC_Q unique values |
|-------|---------------------|----------------------|
| Train | 1 (`"-"`) | 1 (`"-"`) |
| Dev | 1 (`"-"`) | 1 (`"-"`) |
| Eval | 12 (`"-"` + C01-C11) | 9 (0-8) |

### Implications

1. **Per-domain validation metrics are trivial:** Dev set breakdown shows 100% NONE domain
2. **Domain probe accuracy on dev is meaningless:** Single class = undefined probe accuracy
3. **DANN validation loss is uninformative:** Domain discriminator always correct on dev

### Workarounds

- Use synthetic domain labels during validation (not implemented)
- Rely on overall EER/minDCF for validation
- Per-domain analysis only on eval set

---

## Label and Score Convention Consistency

### Fixed Convention

| Label | Value | Score Direction |
|-------|-------|-----------------|
| Bonafide | 0 | Higher score = more bonafide |
| Spoof | 1 | Lower score = more bonafide |

**This is consistent across:**
- Dataset (`KEY_TO_LABEL`)
- Model output (softmax[:, 0] = P(bonafide))
- Metrics (EER, minDCF)
- Evaluation scripts

### Historical Note

Previous versions had inconsistent label conventions in some documentation and tests. These have been fixed .

---

## Augmentation Rate Dependencies

### Minimum Domain Diversity

DANN training will fail-fast if domain diversity is insufficient:

```python
# In training loop (first 10 batches)
if unique_codecs < 2:
    raise RuntimeError("DANN requires domain diversity...")
```

### Augmentation Rate Monitoring

If cumulative augmentation rate drops below 5% after 500 steps, training fails:

```python
if batch_idx >= 500 and aug_rate < 0.05:
    raise RuntimeError("Augmentation rate < 5%...")
```

### Common Causes of Low Augmentation Rate

1. `codec_prob` set too low
2. ffmpeg not available
3. All requested codecs unsupported
4. Augmentation errors (check logs for ffmpeg failures)

---

## Analysis Scope Limitations

### Held-Out Codec Experiments

**Limitation:** "Held-out codec" analysis uses **synthetic** domains, not protocol domains.

- Cannot hold out C01-C11 from train (they don't exist in train)
- Can only hold out synthetic codec families (MP3, AAC, OPUS, etc.)
- This tests generalization to unseen synthetic codecs, not unseen protocol codecs

### Layer Probe Interpretation

**Limitation:** Probe accuracy is influenced by class imbalance.

- With `codec_prob=0.5`, ~50% samples are NONE
- High probe accuracy for NONE is expected
- Focus on relative ERM vs DANN comparison, not absolute numbers

### CKA Analysis

**Limitation:** CKA measures representation similarity, not quality.

- High CKA between ERM and DANN = similar representations
- Does NOT indicate which is better for the task
- Use alongside probe accuracy and task metrics

---

## Future Work (Out of Scope)

The following are acknowledged limitations not addressed in this codebase:

1. **Neural codec simulation (Encodec):** Would improve C04 coverage
2. **Acoustic channel simulation:** Would improve C11 coverage
3. **Speaker-aware domain adaptation:** Speakers may have domain-specific characteristics
4. **Attack-aware domain adaptation:** Some attacks may interact with codecs
5. **Multi-task learning with attack classification:** Could improve robustness
