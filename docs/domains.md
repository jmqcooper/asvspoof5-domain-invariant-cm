# Domain Labels & Augmentation

## The Problem

Train/dev sets have **no codec diversity** — all samples are uncoded.

| Split | CODEC | CODEC_Q | Samples |
|-------|-------|---------|---------|
| Train | 100% "-" | 100% "-" | 182,357 |
| Dev | 100% "-" | 100% "-" | 140,950 |
| Eval | C01-C11 + "-" | 0-8 | 680,774 |

Without augmentation, DANN degenerates to ERM (domain discriminator learns nothing).

## Synthetic Augmentation

We create domain diversity via ffmpeg codec compression:

| Synthetic | ffmpeg Encoder | Maps to Eval |
|-----------|----------------|--------------|
| NONE | — | Uncoded ("-") |
| MP3 | libmp3lame | C05 |
| AAC | aac | C06 |
| OPUS | libopus | C01, C08 |
| SPEEX | libspeex | C03, C10 |
| AMR | libopencore_amrnb | C02, C09 |

Quality levels 1-5 correspond to bitrate tiers per codec.

## Coverage Gaps

| Eval Codec | Covered | Notes |
|------------|---------|-------|
| C01-C03, C05-C06, C08-C10 | ✓ | Good match |
| **C04 (Encodec)** | ✗ | Neural codec, different artifacts |
| **C07 (MP3+Encodec)** | ⚠️ | Only MP3 portion |
| **C11 (Device)** | ✗ | Acoustic effects, not codec |

DANN improvement expected for covered codecs; limited/no improvement for C04, C07, C11.

## Eval Codec Reference

| ID | Codec | Bandwidth |
|----|-------|-----------|
| C01 | opus_wb | Wideband |
| C02 | amr_wb | Wideband |
| C03 | speex_wb | Wideband |
| C04 | encodec_wb | Wideband |
| C05 | mp3_wb | Wideband |
| C06 | m4a_wb | Wideband |
| C07 | mp3_encodec_wb | Cascade |
| C08 | opus_nb | Narrowband |
| C09 | amr_nb | Narrowband |
| C10 | speex_nb | Narrowband |
| C11 | Device | BT/cable/MST |

## Domain Normalization

| Value | Normalized to |
|-------|---------------|
| "-" | NONE |
| "0" | NONE |

## CODEC_Q Semantic Mismatch

Training quality levels (1-5) are arbitrary bitrate tiers. Eval levels 6-8 are C11 device variants (never seen in training).
