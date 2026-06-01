# Draft Notes

Canonical source: `paper/main.tex`.

Current scope: **gsplat CUDA controlled probes of DefaultStrategy-style opacity
reset only**.

## Claim Boundary

The draft demonstrates that a full opacity reset can push effective per-pixel
alpha below gsplat's `1/255` contribution cutoff, removing image-driven opacity
gradients for affected contributions. Controlled CUDA probes show large
post-reset drops under full reset and smaller drops when only a subset is
reset.

The paper does not claim:

- cross-backend validation,
- real-scene CUDA validation,
- cross-backend validation,
- MCMC failure under the same mechanism,
- age-gating superiority over random subset selection at the same reset
  fraction.

## Paper Code

Run on CUDA with `gsplat` installed:

```bash
python experiments/reproduce_gsplat_opacity_reset.py
```

The script writes:

```text
support_collapse_paper_results.json
```

Checked-in result snapshots:

- `results/colab/environment_and_full_record.json`
- `results/colab/run1_n100k_synthetic_t4.json`
- `results/colab/run2_n500k_synthetic_t4.json`
- `results/colab/run3_n100k_mature_t4.json`
- `results/colab/run4_ablation_max_age_t4.json`
- `results/colab/threshold_sweep_t4.json`

## Retained Tables

### Table 1: gsplat CUDA controlled probes

| Experiment | N | Train steps | Full reset | Youngest-20% reset | Gap |
|---|---:|---:|---:|---:|---:|
| Exp 1 | 100K | 300 | -6.5 dB | -0.7 dB | +5.8 dB |
| Exp 2 | 500K | 200 | -2.7 dB | -0.0 dB | +2.7 dB |
| Exp 3 | 100K | 1000 | -7.6 dB | -1.8 dB | +5.8 dB |

### Table 2: reset-fraction ablation

| Method | Reset % | Drop | Rec@100 |
|---|---:|---:|---:|
| No reset | 0 | 0.00 | 12.49 |
| Age-50 | 5 | -0.45 | 12.46 |
| Age-100 | 10 | -0.88 | 12.44 |
| Age-200 | 20 | -1.76 | 12.37 |
| Age-500 | 50 | -4.10 | 12.00 |
| Random-20% | 20 | -1.78 | 12.36 |
| Full reset | 100 | -7.59 | 11.64 |

### Table 3: threshold isolation

Current source: `results/colab/threshold_sweep_t4.json`.

| Reset opacity | vs. `1/255` | Recovery |
|---:|---|---:|
| 0.0035 | below | 0.00 dB |
| 0.00395 | near boundary | +0.10 dB |
| 0.0050 | above | +0.13 dB |

The artifact is an archived T4 sweep packaged in JSON. A fully rerunnable
threshold-sweep path should be added before archival submission.
