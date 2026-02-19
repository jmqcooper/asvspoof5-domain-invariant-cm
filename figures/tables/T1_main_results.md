# Main Results: EER and minDCF for ERM vs DANN

| Model | Backbone | Dev EER (%) | Eval EER (%) | Eval minDCF |
| --- | --- | --- | --- | --- |
| ERM | WavLM | 3.26 | 8.47 | 0.6388 |
| ERM + Aug | WavLM | 3.26 | 7.98 | 0.6052 |
| DANN | WavLM | 4.76 | 7.34 | 0.5853 |
| ERM | W2V2 | 4.24 | 15.30 | 1.0000 |
| ERM + Aug | W2V2 | 4.34 | 18.02 | 0.9992 |
| DANN | W2V2 | 4.45 | 14.33 | 1.0000 |
| LFCC-GMM | LFCC | 17.59 | 43.33 | 0.9995 |
| TRILLsson Logistic | TRILLsson | 19.35 | 23.75 | 1.0000 |
| TRILLsson MLP | TRILLsson | 20.32 | 25.65 | 1.0000 |