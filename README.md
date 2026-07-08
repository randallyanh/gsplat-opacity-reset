# gsplat Opacity Reset Deadlock

Standalone paper and reproduction package for the gsplat opacity-reset support
collapse note.

## Upstream

- Mechanism report: [nerfstudio-project/gsplat#1011](https://github.com/nerfstudio-project/gsplat/issues/1011)
- Proposed fix (age-gated reset, `DefaultStrategy.reset_max_age`): [PR #1012](https://github.com/nerfstudio-project/gsplat/pull/1012)

## Scope

This project is intentionally limited to the gsplat CUDA, reset-based training
path. The current paper studies a `DefaultStrategy`-style global opacity reset
and its interaction with the rasterizer's `alpha < 1/255` contribution cutoff.

It does not include backend-runtime code, product trainer code, or MCMC
experiments.

## Layout

- `paper/main.tex`: paper source.
- `experiments/reproduce_gsplat_opacity_reset.py`: CUDA reproduction script for
  the paper-facing controlled probes.
- `experiments/compare_runtime_artifacts.py`: Colab-vs-Kaggle artifact
  consistency check and normalized paper-table export.
- `results/colab/`: checked-in T4 result snapshots used by the paper.
- `results/runtime_consistency_report.json`: latest checked consistency report
  comparing archived Colab evidence with downloaded Kaggle reruns.
- `results/publication_gate_report.json`: latest strict final-table readiness
  report for Kaggle reruns.
- `docs/evidence/checklist.md`: claim-to-artifact evidence map.
- `docs/evidence/publication-checklist.md`: Colab rerun and final release
  readiness checklist.
- `docs/evidence/threshold-boundary-notes.md`: notes behind the archived
  threshold-boundary artifact.
- `docs/notes/draft-notes.md`: scope and table notes.

## Reproduce

Run on a CUDA machine with PyTorch and gsplat installed:

```bash
pip install -r requirements.txt
python experiments/reproduce_gsplat_opacity_reset.py
```

The script writes generated artifacts under ignored `results/local-runs/` by
default. Use `--result-dir`, `--output`, and `--log-jsonl` to route run outputs.
Use `--paper-archive-dir` when you need JSON exports matching the checked-in
`results/colab/` paper layout.

For a small CUDA sanity check before a long run:

```bash
python experiments/reproduce_gsplat_opacity_reset.py --only smoke --log-jsonl results/local-runs/smoke.log.jsonl
```

For incremental runs, use `--only exp1`, `--only exp2`, `--only exp3`, or
`--only ablation`. Use `--result-dir` and `--output` to route generated JSONs
outside the default locations. Use `--log-jsonl` for structured progress logs.

To emit only the archived Table 3 threshold-sweep artifact:

```bash
python experiments/reproduce_gsplat_opacity_reset.py --only threshold
```

The script refuses to write inside the checked-in `results/colab/` archive
unless `--allow-colab-archive-write` is passed. Stage reruns elsewhere first and
only use that flag during an intentional promotion.

## Kaggle Training

Use the project-local Python 3.12 environment and Kaggle CLI. The Kaggle
workflow lives under `.codex/skills/kaggle-training/` and packages GPU runs as:

1. a commit-pinned offline wheelhouse under `dist/kaggle/`;
2. a private Kaggle Dataset, `randallyan/gsplat-opacity-reset-source`, for the
   source snapshot;
3. a private Kaggle Dataset, `randallyan/gsplat-opacity-reset-wheelhouse`, for
   the dependency wheels;
4. a private Kaggle script kernel with `dataset_sources` pointing at both
   Datasets.

Remote runs write JSON artifacts plus `results/kaggle/run_kaggle.log.jsonl`,
per-experiment JSONL logs, `training_status.json`, and
`training_run_metadata.json`. Use Kaggle `--run-mode gpucheck` before smoke
when diagnosing accelerator assignment. GPU kernel metadata uses
`docker_image_pinning_type: latest` and `machine_shape: NvidiaTeslaT4`; the
machine shape is required for Kaggle to expose NVIDIA devices. Do not run
smoke/full training until `gpucheck` sees `nvidia-smi` or `/dev/nvidia*` in
Kaggle logs.

After downloading Kaggle outputs, compare the latest Kaggle reruns against the
checked-in Colab archive:

```bash
python experiments/compare_runtime_artifacts.py --gate runtime
python experiments/compare_runtime_artifacts.py --gate publication
```

The runtime gate writes `results/runtime_consistency_report.json`. The
publication gate writes `results/publication_gate_report.json` and additionally
requires all final Kaggle table runs to come from one clean source revision. Both
gates treat Colab and Kaggle as separate runtimes: artifacts must preserve the
same protocol, metric semantics, conclusions, and paper-table trends, but they
are not expected to be byte-identical or exact floating-point matches.

For a final Colab rerun, follow `docs/evidence/publication-checklist.md`. Stage
raw rerun outputs under ignored `results/colab-reruns/`, review them, and only
then promote accepted JSON artifacts into `results/colab/`.

## Build Paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Evidence Status

Tables 1, 2, and 3 are backed by checked-in JSON artifacts under
`results/colab/`. These are the archived paper evidence snapshots and should not
be overwritten by remote runtime downloads.

Kaggle is maintained as a separate reproducibility runtime. Downloaded Kaggle
outputs stay under ignored `results/kaggle/` run directories and are compared
against the archive with `experiments/compare_runtime_artifacts.py`. The runtime
gate uses a 0.25 dB tolerance for PSNR, drop, and gap metrics, exact reset
percentages, semantic JSON equality for the threshold sweep, pinned environment
fields, and clean commit-pinned Kaggle run metadata. The publication gate uses
the same checks and fails unless all final Kaggle table runs share one source
commit.

`threshold_sweep_t4.json` is an archived T4 sweep packaged from the project
evidence notes; rerunning that sweep directly from code is the next robustness
step for an archival submission.
