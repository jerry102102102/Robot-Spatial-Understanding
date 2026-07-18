#!/usr/bin/env python3
"""Grade a public suite of blind URDF edit tasks with isolated private keys."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from robot_spatial import SpatialError
from spatial_edit_evaluation import EditEvaluationError, grade_edit, json_dump
from spatial_invariants import InvariantError


SUITE_SCHEMA = "robot-spatial-edit-suite.v1"
SUITE_KEY_SCHEMA = "robot-spatial-edit-suite-key.v1"
SUITE_REPORT_SCHEMA = "robot-spatial-edit-suite-report.v1"


class EditSuiteError(ValueError):
    """An invalid public suite, private suite key, or submission layout."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EditSuiteError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(data, dict):
        raise EditSuiteError(f"{label} must contain a JSON object")
    return data


def _string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise EditSuiteError(f"{label}.{field} must be a non-empty string")
    return value


def _resolve(base: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else base / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise EditSuiteError(f"{label} does not exist: {resolved}")
    return resolved


def grade_suite(
    manifest_path: Path,
    key_manifest_path: Path,
    submissions_root: Path,
    minimum_pass_rate: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(minimum_pass_rate) or minimum_pass_rate < 0.0 or minimum_pass_rate > 1.0:
        raise EditSuiteError("minimum_pass_rate must be between 0 and 1")
    manifest = _read_json(manifest_path, "public edit-suite manifest")
    if manifest.get("schema_version") != SUITE_SCHEMA:
        raise EditSuiteError(f"public manifest must use schema_version {SUITE_SCHEMA}")
    suite_id = _string(manifest, "suite_id", "manifest")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise EditSuiteError("manifest.tasks must be a non-empty array")

    private_manifest = _read_json(key_manifest_path, "private edit-suite key manifest")
    if private_manifest.get("schema_version") != SUITE_KEY_SCHEMA:
        raise EditSuiteError(f"private key manifest must use schema_version {SUITE_KEY_SCHEMA}")
    if private_manifest.get("suite_id") != suite_id:
        raise EditSuiteError("private key manifest suite_id does not match public manifest")
    keys = private_manifest.get("keys")
    if not isinstance(keys, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in keys.items()):
        raise EditSuiteError("private key manifest keys must map task IDs to key paths")

    task_ids: list[str] = []
    normalized_tasks: list[dict[str, str | None]] = []
    for index, task in enumerate(tasks):
        label = f"manifest.tasks[{index}]"
        if not isinstance(task, dict):
            raise EditSuiteError(f"{label} must be an object")
        task_id = _string(task, "task_id", label)
        if task_id in task_ids:
            raise EditSuiteError(f"duplicate task_id {task_id!r}")
        task_ids.append(task_id)
        candidate_change_set = task.get("candidate_change_set")
        if candidate_change_set is not None and (not isinstance(candidate_change_set, str) or not candidate_change_set):
            raise EditSuiteError(f"{label}.candidate_change_set must be a non-empty path when provided")
        normalized_tasks.append({
            "task_id": task_id,
            "category": _string(task, "category", label),
            "task": _string(task, "task", label),
            "candidate_urdf": _string(task, "candidate_urdf", label),
            "candidate_invariants": _string(task, "candidate_invariants", label),
            "candidate_change_set": candidate_change_set,
        })
    if set(keys) != set(task_ids):
        raise EditSuiteError("private key IDs must exactly match public task IDs")

    reports: list[dict[str, Any]] = []
    for task in normalized_tasks:
        task_id = str(task["task_id"])
        task_path = _resolve(manifest_path.parent, str(task["task"]), f"task {task_id}")
        key_path = _resolve(key_manifest_path.parent, keys[task_id], f"private key {task_id}")
        candidate_urdf = _resolve(submissions_root, str(task["candidate_urdf"]), f"candidate URDF {task_id}")
        candidate_invariants = _resolve(
            submissions_root,
            str(task["candidate_invariants"]),
            f"candidate invariant contract {task_id}",
        )
        candidate_change_set = (
            _resolve(submissions_root, str(task["candidate_change_set"]), f"candidate graph change set {task_id}")
            if task["candidate_change_set"] is not None
            else None
        )
        report = grade_edit(task_path, key_path, candidate_urdf, candidate_invariants, candidate_change_set)
        reports.append({
            "task_id": task_id,
            "category": task["category"],
            "status": report["status"],
            "passed_checks": report["passed_count"],
            "failed_checks": report["failed_checks"],
            "report": report,
        })

    passed_count = sum(report["status"] == "passed" for report in reports)
    task_count = len(reports)
    pass_rate = passed_count / task_count
    category_summary: dict[str, dict[str, Any]] = {}
    for report in reports:
        summary = category_summary.setdefault(report["category"], {"task_count": 0, "passed_count": 0})
        summary["task_count"] += 1
        summary["passed_count"] += report["status"] == "passed"
    for summary in category_summary.values():
        summary["pass_rate"] = summary["passed_count"] / summary["task_count"]
    passed = pass_rate + 1e-15 >= minimum_pass_rate
    return {
        "schema_version": SUITE_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "meaning": "suite success requires the configured fraction of independently keyed edit tasks to pass every task-level gate",
        "suite_id": suite_id,
        "task_count": task_count,
        "passed_count": passed_count,
        "failed_count": task_count - passed_count,
        "pass_rate": pass_rate,
        "minimum_pass_rate": minimum_pass_rate,
        "category_summary": dict(sorted(category_summary.items())),
        "tasks": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="public robot-spatial-edit-suite.v1 manifest")
    parser.add_argument("key_manifest", type=Path, help="evaluator-private robot-spatial-edit-suite-key.v1 manifest")
    parser.add_argument("--submissions-root", type=Path, required=True)
    parser.add_argument("--minimum-pass-rate", type=float, default=1.0)
    parser.add_argument("--report", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    report = grade_suite(args.manifest, args.key_manifest, args.submissions_root, args.minimum_pass_rate)
    serialized = json_dump(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, EditSuiteError, EditEvaluationError, SpatialError, InvariantError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
