# Double-Blind Submission Package

This document defines the internal release process for the double-blind paper
package. It is not itself a MICCAI submission artifact.

Canonical package builder:

```bash
.venv/bin/python experiments/build_submission_package.py
```

The builder is manifest-driven and uses
`results/paper_artifact_manifest.json` as the source of truth.

## Venue Boundary

For MICCAI 2026 main-conference review, the primary submission artifact is the
paper PDF. The official guidelines require the provided MICCAI 2026 template,
double-blind anonymization, and a limit of 8 pages of content plus up to 2 pages
of references.

MICCAI 2026 supplementary material is restricted to multimedia content and must
not contain extra proofs, analyses, additional results, or identifying markers.
Therefore, the ZIP produced by this project is an internal audit package, or a
package to provide only when explicitly requested and allowed by the review
process. Do not upload it as CMT supplementary material by default.

Official guideline:

```text
https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html
```

## Package Contents

The double-blind package must contain exactly the files listed by the artifact
manifest:

```text
paper/main.pdf
paper/llncs.cls
paper/splncs04.bst
results/paper_artifact_manifest.json
results/colab/environment_and_full_record.json
results/colab/run1_n100k_synthetic_t4.json
results/colab/run2_n500k_synthetic_t4.json
results/colab/run3_n100k_mature_t4.json
results/colab/run4_ablation_max_age_t4.json
results/colab/threshold_sweep_t4.json
```

The package must not include broad repository files or generated runtime
reports. In particular, exclude:

```text
README.md
results/runtime_consistency_report.json
results/publication_gate_report.json
results/kaggle/
results/colab-reruns/
dist/kaggle/
kaggle/
.kaggle/
.venv/
```

These files and directories are either not part of the paper evidence package or
may contain non-anonymous account, dataset, path, or provenance strings.

## Runtime Policy

Colab and Kaggle are separate runtimes in this project.

| Runtime | Role | Submission-package policy |
|---|---|---|
| Colab T4 | Paper source of truth | Checked-in JSON evidence under `results/colab/` |
| Kaggle | Independent reproducibility runtime | Do not include in the double-blind package |
| Local machine | Development and verification | Do not include in the double-blind package |

The current paper tables are sourced from the checked-in Colab T4 archive.
Kaggle results may pass the runtime consistency gate while still failing the
publication gate if the Kaggle runs were produced from multiple source commits.
That failure is not a blocker unless Kaggle is promoted to final paper-table
evidence.

## Build

Use the project virtual environment. The expected Python runtime is 3.12.

```bash
.venv/bin/python --version
.venv/bin/python experiments/build_submission_package.py
```

Default output:

```text
dist/submission/submission.zip
dist/submission/submission.zip.sha256
```

The builder performs two hard gates before writing the ZIP:

- Integrity gate: every package member with a manifest entry must match its
  recorded SHA-256.
- Anonymity gate: every package member is scanned for configured identity
  strings; PDFs are scanned through rendered text using `pdftotext`.

The ZIP is written through a temporary file and renamed into place only after a
successful build. A failure during validation or ZIP creation must not corrupt an
existing package.

To scan additional identity strings, repeat `--extra-needle`:

```bash
.venv/bin/python experiments/build_submission_package.py \
  --extra-needle "lab name" \
  --extra-needle "author surname"
```

Do not use `--allow-identity` for a double-blind build. That option exists only
for deliberately non-blind release testing.

## Verification

Check the output ZIP contents:

```bash
unzip -l dist/submission/submission.zip
```

Check the ZIP digest:

```bash
shasum -a 256 dist/submission/submission.zip
(cd dist/submission && shasum -a 256 -c submission.zip.sha256)
```

Run a negative sentinel test. This must exit with code `2` and must not rewrite
the existing ZIP:

```bash
before=$(shasum -a 256 dist/submission/submission.zip | awk '{print $1}')
set +e
.venv/bin/python experiments/build_submission_package.py --extra-needle gsplat
rc=$?
set -e
after=$(shasum -a 256 dist/submission/submission.zip | awk '{print $1}')
test "$rc" = 2
test "$before" = "$after"
```

The sentinel word `gsplat` is not an identity string. It is intentionally used
as a high-signal test needle because it should appear in the PDF, evidence JSONs,
and manifest. The expected result is a refusal to package.

Run an independent unpack-and-scan check for known identity strings:

```bash
IDENTITY_RE='AUTHOR_REAL_NAME|AUTHOR_PUBLIC_HANDLE|AUTHOR_EMAIL|REAL_REPOSITORY_URL'
tmp=$(mktemp -d)
unzip -q dist/submission/submission.zip -d "$tmp"
rg -n -i "$IDENTITY_RE" "$tmp" || true
pdftotext "$tmp/paper/main.pdf" - | \
  rg -n -i "$IDENTITY_RE" || true
rm -rf "$tmp"
```

Both scans should produce no identity matches.

## Pre-Submission Audit

Before final upload, verify:

- `paper/main.pdf` is built from the official MICCAI/LNCS template.
- The rendered PDF has no real author, affiliation, account, repository, email,
  ORCID, funding, or acknowledgment strings.
- `pdfinfo paper/main.pdf` shows no identifying Author/Title/Keywords metadata.
- `results/paper_artifact_manifest.json` records `source_dirty: false`.
- The manifest SHA-256 entries match all packaged evidence files.
- The runtime gate passes:

```bash
python experiments/compare_runtime_artifacts.py --gate runtime
```

The publication gate may fail only under the documented policy that Kaggle is an
independent reproducibility runtime and not the final paper-table source:

```bash
python experiments/compare_runtime_artifacts.py --gate publication || true
```

## Commit Policy

Commit the package builder and this document. Do not commit generated files under
`dist/submission/`.

If the paper PDF, evidence JSONs, template files, or manifest change, regenerate
the manifest from a clean commit, rebuild the package, and rerun the verification
commands above.
