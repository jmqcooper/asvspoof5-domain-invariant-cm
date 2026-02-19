# RQ4 Findings: Activation Patching & CKA Analysis

## Key Findings (2026-02-14)

### 1. Layer 11 Divergence
Even with frozen backbone, layer 11's **contribution** to the final representation diverges dramatically between ERM and DANN:
- **CKA = 0.098** for layer 11 (vs >0.87 for layers 0-10)
- Pooling weight delta: L2=0.690, max_abs=0.356

**Interpretation:** Layer 11 (final transformer layer) contains the most codec-specific information. DANN learns to either suppress it via pooling weights or transform it away in the projection layer.

### 2. Domain Invariance in Projection Head
`layer_patch_repr` intervention shows:
- Probe accuracy drops: 76.9% → 38.8% (**-38.1 pp**)
- EER only slightly affected: 3.24% → 2.94% (-0.30 pp)

**Interpretation:** Domain invariance happens specifically in the projection head, not the backbone. The 38 percentage point drop in codec probe accuracy confirms domain information is actively removed.

### 3. Intervention Results Summary

| Mode | EER | Δ EER (pp) | Probe Acc | Δ Probe (pp) |
|------|-----|------------|-----------|--------------|
| layer_patch_hidden (baseline) | 3.24% | — | 76.9% | — |
| layer_patch_repr | 2.94% | -0.30 | 38.8% | **-38.1** |
| layer_patch_mixed | 2.58% | -0.66 | 70.1% | -6.8 |
| pool_weight_transplant | 2.58% | -0.66 | 76.8% | -0.1 |

*pp = percentage points*

### 4. CKA Layer-by-Layer (ERM vs DANN)

```
Layer  0: 0.970  ████████████████████░
Layer  1: 0.989  ████████████████████░
Layer  2: 0.975  ████████████████████░
Layer  3: 0.986  ████████████████████░
Layer  4: 0.978  ████████████████████░
Layer  5: 0.970  ████████████████████░
Layer  6: 0.930  ███████████████████░░
Layer  7: 0.872  █████████████████░░░░
Layer  8: 0.890  ██████████████████░░░
Layer  9: 0.884  ██████████████████░░░
Layer 10: 0.844  █████████████████░░░░
Layer 11: 0.098  ██░░░░░░░░░░░░░░░░░░░  ← DANN diverges here
```

### 5. Activation Patching Interpretation

We performed activation patching by:
1. Taking ERM model as base
2. Patching in DANN's learned components (pooling weights, projection layer)
3. Measuring effect on both EER and codec probe accuracy

This reveals **which components** cause DANN's domain invariance:
- Projection layer patching (`layer_patch_repr`) → reduces domain leakage
- Pooling weight transplant → improves EER without affecting leakage

## Thesis Implications

1. **DANN works at the projection layer** — backbone remains unchanged
2. **Layer 11 is the information bottleneck** — most codec info flows through final transformer layer
3. **Pooling weights + projection head = the intervention targets** — future work could focus on these components

## Visualization TODO

- [ ] CKA heatmap (layers × model pairs)
- [ ] Pooling weight comparison bar chart (ERM vs DANN)
- [ ] Intervention effect scatter plot (Δ EER vs Δ probe)
- [ ] Layer contribution flow diagram
