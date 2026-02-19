# OOD Gap Analysis: Dev vs Eval Generalization

| Model | Backbone | Dev EER (%) | Eval EER (%) | Gap | Gap Reduction |
| --- | --- | --- | --- | --- | --- |
| ERM | WavLM | 3.26 | 8.47 | 5.21 | (baseline) |
| DANN | WavLM | 4.76 | 7.34 | 2.58 | 50.4\% |
| ERM | W2V2 | 4.24 | 15.30 | 11.06 | (baseline) |
| DANN | W2V2 | 4.45 | 14.33 | 9.88 | 10.7\% |