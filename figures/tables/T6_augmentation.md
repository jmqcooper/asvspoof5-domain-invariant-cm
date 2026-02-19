# Synthetic Codec Augmentation for DANN Training

| Codec | Quality Levels | Bitrates | Configured | Observed in runs |
| --- | --- | --- | --- | --- |
| MP3 (libmp3lame) | 5 (1-5) | 64k, 96k, 128k, 192k, 256k | Yes | Yes |
| AAC (aac) | 5 (1-5) | 32k, 64k, 96k, 128k, 192k | Yes | Yes |
| Opus (libopus) | 5 (1-5) | 12k, 24k, 48k, 64k, 96k | Yes | No (ffmpeg unsupported) |