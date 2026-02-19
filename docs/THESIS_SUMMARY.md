# Thesis Summary: Domain-Adversarial Training for Codec-Robust Speech Deepfake Detection

**Author:** Mike Cooper  
**Institution:** University of Amsterdam  
**Date:** February 2026  

## Research Question

> **Does domain invariance learned on synthetic codec augmentation (MP3/AAC) transfer to real benchmark codec conditions (C01-C11) in speech deepfake detection?**

We frame this as a *transfer problem*: DANN is trained to be invariant to synthetic codecs created via ffmpeg, then evaluated on ASVspoof5's real codec conditions which include neural codecs (Encodec) and device effects not seen during training.

---

## Key Results (Full Eval, 680k samples)

| Model | Backbone | Dev EER | Eval EER | OOD Gap | minDCF |
|-------|----------|---------|----------|---------|--------|
| ERM | WavLM | 3.26% | 8.47% | +160% | 0.639 |
| ~~ERM+Aug~~ | ~~WavLM~~ | ~~3.26%~~ | ~~8.47%~~ | ~~+160%~~ | ~~0.639~~ |
| **DANN** | **WavLM** | **4.76%** | **7.34%** | **+54%** | **0.585** |
| ERM | W2V2 | 4.24% | 15.30% | +261% | 1.000* |
| ~~ERM+Aug~~ | ~~W2V2~~ | ~~3.26%~~ | ~~15.79%~~ | ~~+383%~~ | ~~1.000~~ |
| DANN | W2V2 | 4.45% | 14.33% | +222% | 1.000* |

*\*W2V2 minDCF saturates at 1.0 due to over-confident score distribution (EER threshold ≈ 0.9999). EER is the primary metric for this backbone.*

~~Strikethrough~~ = **Pending re-run** (ERM+Aug had zero augmentation due to missing ffmpeg encoders — see Issue #100)

**Baselines:**
| Model | Eval EER |
|-------|----------|
| TRILLsson Logistic | 23.75% |
| TRILLsson MLP | 25.65% |
| LFCC-GMM | 43.33% |

---

## Research Questions Answered

### RQ1: Does DANN reduce the OOD gap?
**✅ Yes.** WavLM OOD gap reduced from 160% to 54% (66% relative reduction).

*Note: The OOD gap reflects codec shift combined with other benchmark differences (dev/eval have different shortcut removal post-processing). We cannot isolate codec-only effects without additional controls.*

### RQ2: Does DANN improve per-codec performance?
**✅ Yes.** Improvement observed across all codecs in the evaluation set, including codecs not covered by synthetic augmentation (C04/Encodec, C07/cascade, C11/device).

### RQ3: Where does domain invariance emerge?
**✅ Projection layer shows partial invariance.** Probing analysis shows codec probe accuracy drops 43.4% → 38.8% in the projection layer (10.6% relative reduction). Backbone layers are identical (frozen).

*Important framing:* Probe accuracy remains above chance (~16.7% for 6 classes), indicating *partial* domain invariance — DANN reduces but does not eliminate domain information. The probe predicts *synthetic* domain labels (MP3/AAC/NONE), not eval codecs (C01-C11); transfer to real codecs is indirect.

### RQ4: What components cause domain invariance?
**✅ Learned layer weights + projection head.** CKA analysis compares `hidden_state × layer_weight` between ERM and DANN. Since the backbone is frozen, raw hidden states are identical; the divergence comes from **different learned layer mixing weights**.

Layer 11 divergence (CKA=0.098) indicates DANN learns to down-weight contributions from the final transformer layer, which contains the most codec-specific information. This is a consequence of the frozen backbone design — DANN can only adjust the learnable components (layer weights, projection head).

---

## Novelty Claim

> **"We provide a systematic evaluation of domain-adversarial training (DANN) for codec-robust speech deepfake detection on ASVspoof5, including transfer from synthetic codec augmentation to real benchmark conditions and representation-level analysis via probing, CKA, and activation patching."**

### Related Work
- **"Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment"** (Sept 2025) — Uses adversarial training for TTS/VC domain shift, not codec robustness.
- Most prior work uses **data augmentation** (codec simulation) without adversarial objectives.
- **Face anti-spoofing** has used DANN for domain adaptation, but not speech deepfake detection.

### Why This is a Contribution
1. DANN applied specifically to codec mismatch problem with frozen SSL backbone
2. Evidence that invariance can emerge in a lightweight projection head without fine-tuning billion-parameter models
3. Multi-method analysis (probing + CKA + patching) beyond "it works"

---

## Synthetic Augmentation Design

**Codecs used:** MP3, AAC (OPUS failed silently during training)
**Quality tiers:** 1-5 (arbitrary bitrate levels per codec)

| Synthetic Domain | Maps to Eval Codecs |
|------------------|---------------------|
| NONE (uncoded) | Uncoded ("-") |
| MP3 | C05 (mp3_wb) |
| AAC | C06 (m4a_wb) |

**Not covered by synthetic augmentation:**
- C01/C08 (Opus) — encoder failed
- C02/C09 (AMR) — not included
- C03/C10 (Speex) — not included
- **C04 (Encodec)** — neural codec, fundamentally different
- **C07 (MP3+Encodec cascade)** — compound degradation
- **C11 (Device/channel)** — acoustic effects, not codec

*Note: DANN still improved on uncovered codecs (C04, C07, C11), suggesting some transfer of general robustness, not just codec-specific invariance.*

---

## Limitations

1. **Only 2 synthetic codecs** (MP3, AAC) — OPUS failed silently
2. **ERM+Aug ablation invalid** — pending re-run (Issue #100)
3. **Single training seed** — bootstrap CIs on eval only
4. **Backbone choice matters more than DANN** — WavLM >> W2V2
5. **W2V2 minDCF saturates** — over-confident scores
6. **Partial invariance only** — probe accuracy above chance
7. **OOD gap conflates multiple shifts** — codec + shortcuts + attack distribution

---

## Pre-Publication Checklist

- [x] Full eval (680k samples) completed
- [x] Bootstrap CIs computed
- [x] RQ4 ablation + visualization done
- [ ] **ERM+Aug re-run with working ffmpeg** (Issue #100)
- [ ] Paper intro/related work draft
- [ ] Multi-seed runs (optional, strengthens paper)

---

*Last updated: 2026-02-15 (framing fixes, ERM+Aug bug noted)*
