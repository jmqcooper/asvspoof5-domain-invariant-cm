# RQ4 Statistical Summary

- Results source: `rq4_results_summary.csv`
- Score cache: `rq4_stats_cache.npz`
- Bootstrap samples: `1000`

| Mode | EER | EER 95% CI | Probe Acc | Probe 95% CI | ΔEER vs base | ΔEER 95% CI | Bootstrap p | ΔProbe vs base |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |
| layer_patch_repr | 0.0747 | [0.0704, 0.0787] | 0.3880 | [0.3742, 0.4014] | -0.0101 | [-0.0131, -0.0080] | 0.0000 | -0.0476 |
| layer_patch_mixed | 0.0757 | [0.0715, 0.0805] | 0.7014 | [0.6892, 0.7140] | -0.0091 | [-0.0119, -0.0066] | 0.0000 | +0.0000 |
| pool_weight_transplant | 0.0757 | [0.0711, 0.0801] | 0.7682 | [0.7572, 0.7796] | -0.0091 | [-0.0118, -0.0062] | 0.0000 | +0.0000 |
| layer_patch_hidden | 0.0848 | [0.0803, 0.0891] | 0.7686 | [0.7572, 0.7810] | +0.0000 | [0.0000, 0.0000] | 2.0000 | +0.0000 |
