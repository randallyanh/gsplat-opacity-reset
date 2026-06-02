# Paper Evidence Checklist

Current paper scope: **gsplat CUDA controlled probes of
DefaultStrategy-style opacity reset only**. Cross-backend rows,
natural-scene proxy rows, MCMC claims, and product-trainer evidence are
intentionally excluded.

Every retained claim should map to one table, one artifact, and one command.

## Runtime Evidence Policy

Colab and Kaggle are separate runtimes in this project.

| Runtime | Role | Artifact policy |
|---|---|---|
| Colab | Archived paper evidence | Checked in under `results/colab/`; do not overwrite during remote reruns |
| Kaggle | Reproducibility runtime | Download under ignored `results/kaggle/` run directories with commit-pinned metadata |

Runtime outputs should be consistent, not identical. The required standard is:
same protocol and metric semantics, same qualitative conclusions, same
paper-table trends, PSNR/drop/gap metrics within 0.25 dB, exact reset
percentages, semantic equality for the threshold-sweep JSON, pinned Kaggle
environment fields, and clean commit-pinned Kaggle run metadata.

Runtime command:

```bash
python experiments/compare_runtime_artifacts.py --gate runtime
```

The script writes `results/runtime_consistency_report.json` and exports
normalized Kaggle values for paper-table review. This gate can pass while still
warning that multiple Kaggle source revisions were used.

Publication command:

```bash
python experiments/compare_runtime_artifacts.py --gate publication
```

The publication gate writes `results/publication_gate_report.json` and fails
unless all final Kaggle table runs share one clean source revision.

For the planned 2026-06-02 Colab rerun, use
`docs/evidence/publication-checklist.md`. Raw Colab reruns should be staged
under ignored `results/colab-reruns/` and promoted into `results/colab/` only
after acceptance.

## Mechanism Evidence

| Claim | Artifact | Paper location |
|---|---|---|
| gsplat CUDA uses an `alpha < 1/255` contribution cutoff | `results/colab/environment_and_full_record.json` (`alpha_cutoff_evidence`) | §1, §2 |
| Below-cutoff contributions supply no image-driven opacity gradient | `results/colab/threshold_sweep_t4.json`; notes in `docs/evidence/threshold-boundary-notes.md` | §2.2 |
| Reset opacity sweep crosses the boundary at `1/255` | `results/colab/threshold_sweep_t4.json` | Table 3 |
| MCMC is outside the demonstrated reset-path claim | Strategy-code inspection; no result artifact in this project | §5 strategy scope |

Command:

```bash
python experiments/reproduce_gsplat_opacity_reset.py
```

Run on CUDA with `gsplat` installed. Checked-in result snapshots are under
`results/colab/`.

## Table 1: gsplat CUDA Support Collapse

| Row | Artifact |
|---|---|
| Exp 1: 100K / 300 steps | `results/colab/environment_and_full_record.json` |
| Exp 2: 500K / 200 steps | `results/colab/environment_and_full_record.json` |
| Exp 3: 100K / 1000 mature steps | `results/colab/environment_and_full_record.json` |

Boundary: these rows test a `DefaultStrategy`-style full opacity reset. They do
not validate or falsify gsplat `MCMCStrategy`.

## Table 2: Reset-Fraction Ablation

| Row family | Artifact |
|---|---|
| No reset / age gates / full reset | `results/colab/run4_ablation_max_age_t4.json` |
| Random 20% baseline | `results/colab/run4_ablation_max_age_t4.json` |

Interpretation retained in the paper: reset fraction dominates in this
synthetic probe; age-gating is adopted as a densification-lifecycle selector,
not because it beats random selection at equal reset fraction in this setup.

## Table 3: Threshold Isolation

| Row family | Artifact |
|---|---|
| Reset opacity sweep around `1/255` | `results/colab/threshold_sweep_t4.json` |

Status: JSON-backed as an archived T4 sweep. The next robustness step is to
make the sweep fully runnable from `experiments/reproduce_gsplat_opacity_reset.py`
rather than only emitted as an archived artifact.
