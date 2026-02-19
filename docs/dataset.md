# ASVspoof 5 Dataset (Track 1)

## Download

Source: [Zenodo](https://zenodo.org/records/14498691) (~100GB full)

```bash
# Fast download with aria2
bash scripts/download_asvspoof5.sh --full --ipv4 --parallel 16

# Unpack
bash scripts/unpack_asvspoof5.sh

# Create manifests
uv run python scripts/make_manifest.py --validate
```

## Directory Structure

```
$ASVSPOOF5_ROOT/
├── ASVspoof5_protocols/
│   ├── ASVspoof5.train.tsv
│   ├── ASVspoof5.dev.track_1.tsv
│   └── ASVspoof5.eval.track_1.tsv
├── flac_T/      # Training (~182k files)
├── flac_D/      # Dev (~141k files)
└── flac_E_eval/ # Eval (~681k files)
```

## Protocol Format

Files are **whitespace-separated** (not tab), despite `.tsv` extension.

| Column | Description |
|--------|-------------|
| FLAC_FILE_NAME | Audio filename |
| CODEC | Codec type (C01-C11 or "-") |
| CODEC_Q | Codec quality (0-8 or "-") |
| CODEC_SEED | Links coded variants to originals |
| KEY | bonafide / spoof |

## Domain Distribution

| Split | Samples | CODEC | CODEC_Q |
|-------|---------|-------|---------|
| Train | 182,357 | 100% uncoded | 100% uncoded |
| Dev | 140,950 | 100% uncoded | 100% uncoded |
| Eval | 680,774 | 25% uncoded, 75% coded | 0-8 |

**Critical:** Train/dev have no codec diversity. DANN requires synthetic augmentation.

## Audio Format

- FLAC, 16 kHz, mono
