# Publication Checklist

Target rerun date: 2026-06-02.

This checklist treats Colab as the paper source of truth and Kaggle as an
independent reproducibility runtime. A Colab rerun becomes publication evidence
only after it is staged, compared, and explicitly promoted into
`results/colab/`.

## Before Rerun

- Commit or deliberately record the exact source revision used for the rerun.
- Confirm the rerun uses Python 3.12, PyTorch `2.10.0+cu128`, CUDA 12.8,
  gsplat `1.5.3`, and a Tesla T4 or clearly documented equivalent.
- Do not overwrite checked-in `results/colab/` artifacts during the first pass.
- Stage raw Colab outputs under:

```text
results/colab-reruns/2026-06-02-<commit12>/
```

## Colab Rerun Commands

Run the full reproduction if time is available:

```bash
python experiments/reproduce_gsplat_opacity_reset.py \
  --only all \
  --result-dir results/colab-reruns/2026-06-02-<commit12> \
  --output results/colab-reruns/2026-06-02-<commit12>/support_collapse_paper_results.json \
  --log-jsonl results/colab-reruns/2026-06-02-<commit12>/colab_run.log.jsonl \
  --paper-archive-dir results/colab-reruns/2026-06-02-<commit12>/paper-archive
```

If the full run is too brittle, run the paper rows incrementally:

```bash
RUN_DIR=results/colab-reruns/2026-06-02-<commit12>
python experiments/reproduce_gsplat_opacity_reset.py \
  --only exp1 \
  --result-dir "$RUN_DIR" \
  --output "$RUN_DIR/exp1_results.json" \
  --log-jsonl "$RUN_DIR/exp1.log.jsonl" \
  --paper-archive-dir "$RUN_DIR/paper-archive"
python experiments/reproduce_gsplat_opacity_reset.py \
  --only exp2 \
  --result-dir "$RUN_DIR" \
  --output "$RUN_DIR/exp2_results.json" \
  --log-jsonl "$RUN_DIR/exp2.log.jsonl" \
  --paper-archive-dir "$RUN_DIR/paper-archive"
python experiments/reproduce_gsplat_opacity_reset.py \
  --only exp3 \
  --result-dir "$RUN_DIR" \
  --output "$RUN_DIR/exp3_results.json" \
  --log-jsonl "$RUN_DIR/exp3.log.jsonl" \
  --paper-archive-dir "$RUN_DIR/paper-archive"
python experiments/reproduce_gsplat_opacity_reset.py \
  --only ablation \
  --result-dir "$RUN_DIR" \
  --output "$RUN_DIR/ablation_results.json" \
  --log-jsonl "$RUN_DIR/ablation.log.jsonl" \
  --paper-archive-dir "$RUN_DIR/paper-archive"
```

## Acceptance Gate

The rerun is acceptable for publication only if:

- the source revision is clean and recorded;
- `colab_run_status.json` has `status: complete`;
- `colab_run_metadata.json` records `source_dirty: false`;
- environment fields match or are explicitly documented;
- Table 1, Table 2, and Table 3 conclusions are unchanged;
- PSNR, drop, and gap values agree with the current archive within 0.25 dB, or
  the paper tables are intentionally updated from the new run;
- no artifact path or table mixes Colab and Kaggle as one source of truth;
- the paper builds cleanly twice from `paper/main.tex`.

## Promotion

After acceptance, choose one of these publication policies and record it in the
commit message:

- `Archive unchanged`: keep current `results/colab/` as paper source of truth
  and retain the 2026-06-02 rerun only as local staged evidence.
- `Archive replaced`: replace the checked-in `results/colab/` JSON artifacts
  with the accepted 2026-06-02 Colab rerun and update `paper/main.tex` tables if
  rounded values changed.

Writing directly into `results/colab/` is blocked by default. During an accepted
promotion, pass `--allow-colab-archive-write` only if the command intentionally
writes the checked-in archive.

Do not publish from a mixed policy where some table rows come from old Colab,
some from new Colab, and some from Kaggle unless the paper explicitly says so.

## Final Release Checks

```bash
python experiments/compare_runtime_artifacts.py --gate runtime
python experiments/compare_runtime_artifacts.py --gate publication || true
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Kaggle publication gate may remain failed if Kaggle is not the paper source
of truth. It must pass only when Kaggle reruns are used as final table evidence.
