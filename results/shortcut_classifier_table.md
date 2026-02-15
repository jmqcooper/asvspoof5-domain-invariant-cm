# Shortcut Classifier Results

## Method
Logistic regression trained using only 5 shortcut features from ASVspoof5 §4.2:
- Peak amplitude
- Total energy
- Total duration
- Leading non-speech duration
- Trailing non-speech duration

## Results

| Split | Accuracy | AUC   | EER   | n      |
|-------|----------|-------|-------|--------|
| Train | 82.6%    | 0.903 | 17.7% | 10,000 |
| Dev   | 24.5%    | 0.451 | 53.6% | 10,000 |
| Eval  | 20.4%    | 0.483 | 52.7% | 10,000 |

## Feature Coefficients

| Feature       | Coefficient |
|---------------|-------------|
| peak_amp      | -0.663      |
| energy        | -0.499      |
| duration      | -1.104      |
| lead_sil      | -0.300      |
| trail_sil     | -0.630      |

## Interpretation

These five shortcut features are strongly predictive in train (82.6% acc, 0.903 AUC) but are non-predictive on dev/eval (~50% EER, ~0.45 AUC), indicating they are train-specific artifacts.

**Conclusion:** The OOD performance gap is NOT primarily explained by these five shortcut artifacts (ASVspoof5 §4.2).
