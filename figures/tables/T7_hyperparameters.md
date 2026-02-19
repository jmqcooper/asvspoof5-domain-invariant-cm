# Key Hyperparameters by Model Family

| Parameter | WavLM ERM | WavLM DANN | W2V2 ERM | W2V2 DANN | LFCC-GMM | TRILLsson Logistic | TRILLsson MLP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model type | SSL + linear head | SSL + DANN | SSL + linear head | SSL + DANN | GMM baseline | Linear baseline | MLP baseline |
| Backbone/features | WavLM Base+ | WavLM Base+ | Wav2Vec2 Base | Wav2Vec2 Base | LFCC (120-dim) | TRILLsson (1024-dim) | TRILLsson (1024-dim) |
| Layer selection | weighted (k=6) | weighted (k=6) | weighted (k=6) | weighted (k=6) | — | — | — |
| Backbone frozen | true | true | true | true | — | — | — |
| Projection output dim | 256 | 256 | 256 | 256 | — | — | — |
| Batch size | 256 | 256 | 256 | 256 | — | — | — |
| Learning rate | 1e-4 | 1e-4 | 5e-5 | 5e-5 | — | — | — |
| Optimizer | AdamW | AdamW | AdamW | AdamW | — | scikit-learn | scikit-learn |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 | — | — | — |
| Max epochs | 50 | 50 | 50 | 50 | — | — | — |
| Patience | 10 | 10 | 10 | 10 | — | — | — |
| Gradient clip | 1.0 | 0.5 | 0.5 | 0.5 | — | — | — |
| $\\lambda$ schedule | N/A (ERM) | linear 0.1→0.75 (warmup 3) | N/A (ERM) | exponential 0.01→1.0 | N/A | N/A | N/A |
| Augmentation codecs | N/A | MP3, AAC, OPUS | N/A | MP3, AAC, OPUS | N/A | N/A | N/A |
| Augmentation qualities | N/A | 1-5 | N/A | 1-5 | N/A | N/A | N/A |
| Random seed | 42 | 42 | 42 | 42 | 42 | 42 | 42 |