# gsplat Opacity Reset Deadlock

Standalone paper and reproduction package for the gsplat opacity-reset support
collapse note.

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
- `results/colab/`: checked-in T4 result snapshots used by the paper.
- `docs/evidence/checklist.md`: claim-to-artifact evidence map.
- `docs/evidence/threshold-boundary-notes.md`: notes behind the archived
  threshold-boundary artifact.
- `docs/notes/draft-notes.md`: scope and table notes.

## Reproduce

Run on a CUDA machine with PyTorch and gsplat installed:

```bash
pip install -r requirements.txt
python experiments/reproduce_gsplat_opacity_reset.py
```

The script writes `support_collapse_paper_results.json` in the current working
directory.

For a small CUDA sanity check before a long run:

```bash
python experiments/reproduce_gsplat_opacity_reset.py --only smoke --log-jsonl results/kaggle/smoke.log.jsonl
```

For incremental runs, use `--only exp1`, `--only exp2`, `--only exp3`, or
`--only ablation`. Use `--result-dir` and `--output` to route generated JSONs
outside the default locations. Use `--log-jsonl` for structured progress logs.

To emit only the archived Table 3 threshold-sweep artifact:

```bash
python experiments/reproduce_gsplat_opacity_reset.py --only threshold
```

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
`training_run_metadata.json`.

## Build Paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Evidence Status

Tables 1, 2, and 3 are backed by JSON artifacts under `results/colab/`.
`threshold_sweep_t4.json` is an archived T4 sweep packaged from the project
evidence notes; rerunning that sweep directly from code is the next robustness
step for an archival submission.
