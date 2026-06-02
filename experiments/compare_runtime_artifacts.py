#!/usr/bin/env python3
# Copyright 2026 gsplat-opacity-reset contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare archived Colab evidence against Kaggle rerun artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_OUTPUT = Path("results/runtime_consistency_report.json")
DEFAULT_PUBLICATION_OUTPUT = Path("results/publication_gate_report.json")
DEFAULT_DB_TOLERANCE = 0.25
KAGGLE_VERSION_RE = re.compile(r"-v(?P<version>\d+)$")
RUN_MODES = ("exp1", "exp2", "exp3", "ablation")
EXPECTED_KAGGLE_ENV = {
    "cuda_version": "12.8",
    "gpu_name": "Tesla T4",
    "gsplat": "1.5.3",
    "pytorch": "2.10.0+cu128",
}
EXPECTED_PYTHON_PREFIX = "3.12."


class RuntimeArtifactError(Exception):
    """Raised when expected runtime evidence is missing or malformed."""


@dataclass(frozen=True)
class MetricComparison:
    group: str
    metric: str
    colab: float
    kaggle: float
    delta: float
    tolerance: float
    status: str


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeArtifactError(f"Missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeArtifactError(f"Invalid JSON artifact: {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def latest_kaggle_result(kaggle_dir: Path, run_mode: str) -> Path:
    candidates = sorted(
        kaggle_dir.glob(f"{run_mode}-*/results/kaggle/{run_mode}_results.json")
    )
    if not candidates:
        raise RuntimeArtifactError(
            f"No Kaggle result found for {run_mode} under {kaggle_dir}"
        )

    def sort_key(path: Path) -> tuple[int, float, str]:
        run_dir = path.parents[2]
        match = KAGGLE_VERSION_RE.search(run_dir.name)
        version = int(match.group("version")) if match else -1
        return version, path.stat().st_mtime, str(path)

    return max(candidates, key=sort_key)


def load_latest_kaggle_runs(kaggle_dir: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for run_mode in RUN_MODES:
        result_path = latest_kaggle_result(kaggle_dir, run_mode)
        payload = load_json(result_path)
        metadata_path = result_path.parent / "training_run_metadata.json"
        metadata = load_json(metadata_path)
        runs[run_mode] = {
            "result_path": str(result_path),
            "metadata_path": str(metadata_path),
            "run_dir": str(result_path.parents[2]),
            "payload": payload,
            "metadata": metadata,
        }
    return runs


def colab_table1(colab_dir: Path) -> dict[str, dict[str, float]]:
    exp1 = load_json(colab_dir / "run1_n100k_synthetic_t4.json")
    exp2 = load_json(colab_dir / "run2_n500k_synthetic_t4.json")
    exp3 = load_json(colab_dir / "run3_n100k_mature_t4.json")
    return {
        "exp1": {
            "pre": exp1["psnr_before_reset"],
            "full": exp1["full_reset"]["psnr_after_reset"],
            "full_rec": exp1["full_reset"]["psnr_recovery_100"],
            "sel": exp1["selective_reset_20pct"]["psnr_after_reset"],
            "sel_rec": exp1["selective_reset_20pct"]["psnr_recovery_100"],
            "gap": exp1["selective_vs_full_gap_at_reset"],
        },
        "exp2": {
            "pre": exp2["psnr_before"],
            "full": exp2["full_reset_psnr_after"],
            "full_rec": exp2["full_reset_recovery_50"],
            "sel": exp2["selective_20pct_psnr_after"],
            "sel_rec": exp2["selective_20pct_recovery_50"],
            "gap": exp2["gap_at_reset"],
        },
        "exp3": {
            "pre": exp3["psnr_before"],
            "full": exp3["full_reset"]["psnr_after"],
            "full_rec": exp3["full_reset"]["recovery_100"],
            "sel": exp3["selective_20pct"]["psnr_after"],
            "sel_rec": exp3["selective_20pct"]["recovery_100"],
            "gap": exp3["gap_at_reset"],
            "permanent_full": exp3["full_reset"]["permanent_loss"],
            "permanent_sel": exp3["selective_20pct"]["permanent_loss"],
        },
    }


def kaggle_table1(kaggle_runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for run_mode in ("exp1", "exp2", "exp3"):
        payload = kaggle_runs[run_mode]["payload"]
        try:
            table[run_mode] = payload["table1"][run_mode]
        except KeyError as exc:
            raise RuntimeArtifactError(
                f"Kaggle {run_mode} result is missing table1.{run_mode}"
            ) from exc
    return table


def colab_ablation(colab_dir: Path) -> dict[str, Any]:
    return load_json(colab_dir / "run4_ablation_max_age_t4.json")["results"]


def kaggle_ablation(kaggle_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = kaggle_runs["ablation"]["payload"]
    try:
        return payload["table2_ablation"]["results"]
    except KeyError as exc:
        raise RuntimeArtifactError(
            "Kaggle ablation result is missing table2_ablation.results"
        ) from exc


def compare_metric(
    group: str,
    metric: str,
    colab_value: float,
    kaggle_value: float,
    tolerance: float,
) -> MetricComparison:
    delta = kaggle_value - colab_value
    status = "pass" if abs(delta) <= tolerance else "fail"
    return MetricComparison(
        group=group,
        metric=metric,
        colab=colab_value,
        kaggle=kaggle_value,
        delta=delta,
        tolerance=tolerance,
        status=status,
    )


def compare_tables(
    old_table1: dict[str, dict[str, float]],
    new_table1: dict[str, dict[str, float]],
    old_ablation: dict[str, Any],
    new_ablation: dict[str, Any],
    *,
    db_tolerance: float,
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for experiment, old_row in old_table1.items():
        new_row = new_table1[experiment]
        for metric, colab_value in old_row.items():
            if metric not in new_row:
                continue
            comparisons.append(
                compare_metric(
                    f"table1.{experiment}",
                    metric,
                    float(colab_value),
                    float(new_row[metric]),
                    db_tolerance,
                )
            )

    for policy, old_row in old_ablation.items():
        if policy not in new_ablation:
            raise RuntimeArtifactError(
                f"Kaggle ablation result is missing policy {policy}"
            )
        new_row = new_ablation[policy]
        comparisons.append(
            compare_metric(
                f"table2.{policy}",
                "reset_pct",
                float(old_row["reset_pct"]),
                float(new_row["reset_pct"]),
                1e-9,
            )
        )
        for metric in ("psnr_after", "psnr_rec", "drop"):
            comparisons.append(
                compare_metric(
                    f"table2.{policy}",
                    metric,
                    float(old_row[metric]),
                    float(new_row[metric]),
                    db_tolerance,
                )
            )
    return comparisons


def compare_threshold_sweeps(
    colab_dir: Path,
    kaggle_runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    colab_threshold = load_json(colab_dir / "threshold_sweep_t4.json")
    results = []
    for run_mode, run in kaggle_runs.items():
        threshold = run["payload"].get("table3_threshold_sweep")
        if threshold is None:
            results.append(
                {
                    "run_mode": run_mode,
                    "colab_path": str(colab_dir / "threshold_sweep_t4.json"),
                    "kaggle_result_path": run["result_path"],
                    "semantic_equal": False,
                    "status": "fail",
                    "reason": "missing table3_threshold_sweep",
                }
            )
            continue
        results.append(
            {
                "run_mode": run_mode,
                "colab_path": str(colab_dir / "threshold_sweep_t4.json"),
                "kaggle_result_path": run["result_path"],
                "semantic_equal": threshold == colab_threshold,
                "status": "pass" if threshold == colab_threshold else "fail",
            }
        )
    return results


def paper_table1_from_kaggle(table1: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for experiment in ("exp1", "exp2", "exp3"):
        row = table1[experiment]
        full_drop = row["full"] - row["pre"]
        selective_drop = row["sel"] - row["pre"]
        rows.append(
            {
                "experiment": experiment,
                "pre_reset_psnr": row["pre"],
                "full_reset_drop_db": full_drop,
                "selective_reset_drop_db": selective_drop,
                "gap_db": row["gap"],
                "paper_round_1dp": {
                    "full_reset_drop_db": round(full_drop, 1),
                    "selective_reset_drop_db": round(selective_drop, 1),
                    "gap_db": round(row["gap"], 1),
                },
            }
        )
    return rows


def paper_table2_from_kaggle(ablation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for policy in (
        "no_reset",
        "age_50",
        "age_100",
        "age_200",
        "age_500",
        "random_20pct",
        "full_reset",
    ):
        row = ablation[policy]
        rows.append(
            {
                "policy": policy,
                "reset_pct": row["reset_pct"],
                "drop_db": row["drop"],
                "psnr_rec": row["psnr_rec"],
                "paper_round_2dp": {
                    "drop_db": round(row["drop"], 2),
                    "psnr_rec": round(row["psnr_rec"], 2),
                },
            }
        )
    return rows


def check_record(name: str, status: bool, message: str, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if status else "fail",
        "message": message,
        **fields,
    }


def selected_checks(kaggle_runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for run_mode, run in kaggle_runs.items():
        selected = run["payload"].get("selected")
        checks.append(
            check_record(
                f"{run_mode}.selected",
                selected == run_mode,
                "Kaggle result selected mode matches the run directory mode.",
                run_mode=run_mode,
                selected=selected,
            )
        )
    return checks


def environment_checks(kaggle_runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for run_mode, run in kaggle_runs.items():
        environment = run["payload"].get("environment", {})
        python_version = str(environment.get("python", ""))
        checks.append(
            check_record(
                f"{run_mode}.python",
                python_version.startswith(EXPECTED_PYTHON_PREFIX),
                "Kaggle Python version matches the project Python 3.12 policy.",
                run_mode=run_mode,
                actual=python_version,
                expected_prefix=EXPECTED_PYTHON_PREFIX,
            )
        )
        for field, expected in EXPECTED_KAGGLE_ENV.items():
            actual = environment.get(field)
            checks.append(
                check_record(
                    f"{run_mode}.{field}",
                    actual == expected,
                    "Kaggle runtime field matches the pinned evidence environment.",
                    run_mode=run_mode,
                    field=field,
                    actual=actual,
                    expected=expected,
                )
            )
    return checks


def provenance_checks(kaggle_runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for run_mode, run in kaggle_runs.items():
        metadata = run["metadata"]
        source_revision = metadata.get("source_revision")
        training_version = metadata.get("training_version", "")
        source_dirty = metadata.get("source_dirty")
        commit12 = source_revision[:12] if isinstance(source_revision, str) else ""
        expected_training_version = f"run-{run_mode}-{commit12}" if commit12 else None
        checks.append(
            {
                "run_mode": run_mode,
                "source_revision": source_revision,
                "source_dirty": source_dirty,
                "training_version": training_version,
                "metadata_run_mode": metadata.get("run_mode"),
                "expected_training_version": expected_training_version,
                "status": (
                    "pass"
                    if (
                        source_dirty is False
                        and isinstance(source_revision, str)
                        and len(source_revision) == 40
                        and metadata.get("run_mode") == run_mode
                        and training_version == expected_training_version
                    )
                    else "fail"
                ),
            }
        )
    return checks


def publication_checks(source_revisions: list[str]) -> list[dict[str, Any]]:
    return [
        check_record(
            "single_kaggle_source_revision",
            len(source_revisions) == 1,
            "Publication-table Kaggle runs must all come from one source commit.",
            source_revisions=source_revisions,
        )
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    colab_dir = Path(args.colab_dir)
    kaggle_dir = Path(args.kaggle_dir)
    kaggle_runs = load_latest_kaggle_runs(kaggle_dir)
    old_table1 = colab_table1(colab_dir)
    new_table1 = kaggle_table1(kaggle_runs)
    old_ablation = colab_ablation(colab_dir)
    new_ablation = kaggle_ablation(kaggle_runs)
    metric_comparisons = compare_tables(
        old_table1,
        new_table1,
        old_ablation,
        new_ablation,
        db_tolerance=args.db_tolerance,
    )
    threshold_comparisons = compare_threshold_sweeps(colab_dir, kaggle_runs)
    provenance = provenance_checks(kaggle_runs)
    selected = selected_checks(kaggle_runs)
    environment = environment_checks(kaggle_runs)

    failed_metrics = [item for item in metric_comparisons if item.status != "pass"]
    failed_thresholds = [
        item for item in threshold_comparisons if item["status"] != "pass"
    ]
    failed_provenance = [item for item in provenance if item["status"] != "pass"]
    failed_selected = [item for item in selected if item["status"] != "pass"]
    failed_environment = [item for item in environment if item["status"] != "pass"]
    source_revisions = sorted(
        {
            run["metadata"].get("source_revision")
            for run in kaggle_runs.values()
            if run["metadata"].get("source_revision")
        }
    )
    publication = publication_checks(source_revisions)
    failed_publication = [item for item in publication if item["status"] != "pass"]
    blocking_publication = failed_publication if args.gate == "publication" else []
    warnings = []
    if args.gate == "runtime" and len(source_revisions) > 1:
        warnings.append(
            "Kaggle runs use multiple source revisions; review run metadata "
            "before using them as a single final table source."
        )

    overall_status = (
        "pass"
        if not (
            failed_metrics
            or failed_thresholds
            or failed_provenance
            or failed_selected
            or failed_environment
            or blocking_publication
        )
        else "fail"
    )
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gate": args.gate,
        "runtime_policy": {
            "colab": "archived checked-in paper evidence",
            "kaggle": "remote reproducibility runtime evidence",
            "expected_consistency": [
                "same experiment protocol and metric semantics",
                "same qualitative conclusions and paper-table trends",
                "numeric agreement within the configured tolerance",
                "not byte-identical and not exact floating-point equality across runtimes",
            ],
            "publication_gate": [
                "all final Kaggle table runs must share one source revision",
                "Kaggle run metadata must be clean and training-version pinned",
                "Kaggle runtime fields must match the pinned evidence environment",
                "runtime consistency checks must pass first",
            ],
        },
        "tolerances": {
            "db_metrics": args.db_tolerance,
            "reset_pct": 1e-9,
            "threshold_sweep": "semantic JSON equality",
        },
        "inputs": {
            "colab_dir": str(colab_dir),
            "kaggle_dir": str(kaggle_dir),
            "kaggle_runs": {
                run_mode: {
                    "run_dir": run["run_dir"],
                    "result_path": run["result_path"],
                    "metadata_path": run["metadata_path"],
                    "environment": run["payload"].get("environment", {}),
                    "metadata": run["metadata"],
                }
                for run_mode, run in kaggle_runs.items()
            },
        },
        "summary": {
            "gate": args.gate,
            "overall_status": overall_status,
            "metric_comparisons": len(metric_comparisons),
            "metric_failures": len(failed_metrics),
            "threshold_comparisons": len(threshold_comparisons),
            "threshold_failures": len(failed_thresholds),
            "provenance_checks": len(provenance),
            "provenance_failures": len(failed_provenance),
            "selected_checks": len(selected),
            "selected_failures": len(failed_selected),
            "environment_checks": len(environment),
            "environment_failures": len(failed_environment),
            "publication_checks": len(publication),
            "publication_failures": len(failed_publication),
            "publication_blocking_failures": len(blocking_publication),
            "warnings": warnings,
        },
        "metric_comparisons": [asdict(item) for item in metric_comparisons],
        "threshold_comparisons": threshold_comparisons,
        "selected_checks": selected,
        "environment_checks": environment,
        "provenance_checks": provenance,
        "publication_checks": publication,
        "normalized_kaggle_outputs": {
            "paper_table1": paper_table1_from_kaggle(new_table1),
            "paper_table2": paper_table2_from_kaggle(new_ablation),
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Colab archive artifacts with latest Kaggle rerun outputs."
    )
    parser.add_argument("--colab-dir", default="results/colab")
    parser.add_argument("--kaggle-dir", default="results/kaggle")
    parser.add_argument(
        "--gate",
        choices=("runtime", "publication"),
        default="runtime",
        help="runtime checks reproducibility; publication also enforces final-table provenance.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report path. Defaults depend on --gate.",
    )
    parser.add_argument(
        "--db-tolerance",
        type=float,
        default=DEFAULT_DB_TOLERANCE,
        help="Allowed absolute delta for PSNR/drop/gap metrics.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary without writing the JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.output is None:
        args.output = str(
            DEFAULT_PUBLICATION_OUTPUT
            if args.gate == "publication"
            else DEFAULT_RUNTIME_OUTPUT
        )
    try:
        report = build_report(args)
    except RuntimeArtifactError as exc:
        print(f"runtime artifact comparison failed: {exc}", file=sys.stderr)
        return 2

    if not args.no_write:
        write_json(Path(args.output), report)

    summary = report["summary"]
    print(f"Gate: {summary['gate']}")
    print(f"Overall: {summary['overall_status'].upper()}")
    print(
        "Metric comparisons: "
        f"{summary['metric_comparisons'] - summary['metric_failures']} passed, "
        f"{summary['metric_failures']} failed"
    )
    print(
        "Threshold comparisons: "
        f"{summary['threshold_comparisons'] - summary['threshold_failures']} passed, "
        f"{summary['threshold_failures']} failed"
    )
    print(
        "Provenance checks: "
        f"{summary['provenance_checks'] - summary['provenance_failures']} passed, "
        f"{summary['provenance_failures']} failed"
    )
    print(
        "Environment checks: "
        f"{summary['environment_checks'] - summary['environment_failures']} passed, "
        f"{summary['environment_failures']} failed"
    )
    print(
        "Publication checks: "
        f"{summary['publication_checks'] - summary['publication_failures']} passed, "
        f"{summary['publication_failures']} failed, "
        f"{summary['publication_blocking_failures']} blocking"
    )
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    if not args.no_write:
        print(f"Wrote {args.output}")

    return 0 if summary["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
