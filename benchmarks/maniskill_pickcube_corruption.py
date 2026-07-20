#!/usr/bin/env python3
"""Apply deterministic corruption controls to one real normalized PickCube run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from robot_spatial_understanding.corruption import corrupt_run
from robot_spatial_understanding.errors import IntegrityError
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.simulation import SimulationRun
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


CONTROLS = (
    ("dropped-pose", "dropped-frame", "pose", "cube_follows_tcp", "unknown"),
    ("stale-pose", "stale-tail", "pose", "cube_at_goal", "unknown"),
    ("out-of-order-pose", "out-of-order", "pose", "cube_at_goal", "conflicting"),
    ("conflicting-pose", "conflicting-duplicate", "pose", "cube_at_goal", "conflicting"),
    ("wrong-pose-identity", "wrong-id", "pose", "cube_at_goal", "unknown"),
    ("missing-contact", "missing-channel", "contact", "grasped", "unknown"),
    ("missing-collision", "missing-channel", "collision", "collision_free", "unknown"),
)


def run(source: Path, task_path: Path, output: Path) -> dict[str, Any]:
    ensure_new_directory(output)
    source_run = SimulationRun.load(source)
    task = TaskSpec.load(task_path)
    baseline = AssuranceReport.evaluate(source_run, task)
    cases: list[dict[str, Any]] = []
    for case_id, kind, channel, predicate_id, expected in CONTROLS:
        case_root = output / "cases" / case_id
        corrupted_path = corrupt_run(source_run.root, case_root / "run", kind=kind, channel=channel)
        corrupted = SimulationRun.load(corrupted_path)
        report = AssuranceReport.evaluate(corrupted, task)
        report.write(case_root / "result")
        statuses = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        actual = statuses[predicate_id]
        cases.append(
            {
                "case_id": case_id,
                "corruption": kind,
                "channel": channel,
                "predicate_id": predicate_id,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
                "run_manifest_sha256": corrupted.digest,
                "report_sha256": report.digest,
            }
        )

    terminal_root = output / "cases" / "terminal-status-tamper"
    terminal_path = corrupt_run(source_run.root, terminal_root / "run", kind="terminal-status-tamper")
    terminal_report = AssuranceReport.evaluate(SimulationRun.load(terminal_path), task)
    terminal_report.write(terminal_root / "result")
    baseline_verdict = baseline.data["verdict"]["simulation_bounded_physical_success"]
    terminal_verdict = terminal_report.data["verdict"]["simulation_bounded_physical_success"]
    cases.append(
        {
            "case_id": "terminal-status-tamper",
            "corruption": "terminal-status-tamper",
            "expected": baseline_verdict,
            "actual": terminal_verdict,
            "passed": terminal_verdict == baseline_verdict,
            "report_sha256": terminal_report.digest,
        }
    )

    digest_path = corrupt_run(
        source_run.root,
        output / "cases" / "digest-tamper" / "run",
        kind="digest-tamper",
        channel="pose",
    )
    rejected = False
    try:
        SimulationRun.load(digest_path)
    except IntegrityError:
        rejected = True
    cases.append(
        {
            "case_id": "digest-tamper",
            "corruption": "digest-tamper",
            "expected": "rejected_before_evaluation",
            "actual": "rejected_before_evaluation" if rejected else "accepted",
            "passed": rejected,
        }
    )

    result: dict[str, Any] = {
        "schema_version": "robot-spatial-maniskill-corruption-matrix.v1",
        "source_run_manifest_sha256": source_run.digest,
        "task_spec_sha256": task.digest,
        "baseline_report_sha256": baseline.digest,
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case["passed"]),
        "cases": cases,
        "limitations": [
            "Corruptions are deterministic evidence-integrity controls over one real normalized simulator run; they are not additional physical episodes."
        ],
    }
    result["matrix_sha256"] = sha256_json(result)
    write_json(output / "corruption-matrix.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.task, args.out)
    print(result["passed_count"], "/", result["case_count"])
    return 0 if result["passed_count"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
