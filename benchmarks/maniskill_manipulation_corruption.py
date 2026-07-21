#!/usr/bin/env python3
"""Cross-profile evidence corruption and abstention gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from robot_spatial_understanding.corruption import corrupt_run
from robot_spatial_understanding.errors import IntegrityError
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.simulation import SimulationRun
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


@dataclass(frozen=True)
class SourceCase:
    profile_id: str
    example_dir: str
    case_leaf: str


SOURCES = (
    SourceCase("peg-panda", "peginsertion-live", "seed-004-solver"),
    SourceCase("pick-xarm", "pickcube-xarm-live", "seed-000-solver"),
    SourceCase("push-panda", "pushcube-live", "seed-000-solver"),
    SourceCase("stack-panda", "stackcube-live", "seed-000-solver"),
)


def run(benchmark_root: Path, output: Path) -> dict[str, Any]:
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    cases: list[dict[str, Any]] = []
    source_digests: dict[str, str] = {}
    task_digests: dict[str, str] = {}
    for source in SOURCES:
        source_run = SimulationRun.load(
            benchmark_root / "candidates" / source.profile_id / source.case_leaf / "run"
        )
        task = TaskSpec.load(repo_root / "examples" / source.example_dir / "task.yaml")
        baseline = AssuranceReport.evaluate(source_run, task)
        if baseline.data["verdict"]["simulation_bounded_physical_success"] != "supported":
            raise RuntimeError(f"corruption source {source.profile_id!r} is not a supported baseline")
        source_digests[source.profile_id] = source_run.digest
        task_digests[source.profile_id] = task.digest
        for case_name, kind, expected in (
            ("stale-pose", "stale-tail", "unknown"),
            ("out-of-order-pose", "out-of-order", "conflicting"),
            ("missing-pose", "missing-channel", "unknown"),
        ):
            case_id = f"{source.profile_id}/{case_name}"
            case_root = output / "cases" / source.profile_id / case_name
            path = corrupt_run(source_run.root, case_root / "run", kind=kind, channel="pose")
            report = AssuranceReport.evaluate(SimulationRun.load(path), task)
            report.write(case_root / "result")
            actual = report.data["verdict"]["goal_status"]
            cases.append(
                {
                    "case_id": case_id,
                    "corruption": kind,
                    "channel": "pose",
                    "expected": expected,
                    "actual": actual,
                    "passed": actual == expected,
                    "run_manifest_sha256": SimulationRun.load(path).digest,
                    "report_sha256": report.digest,
                }
            )

        terminal_root = output / "cases" / source.profile_id / "terminal-status-tamper"
        terminal_path = corrupt_run(
            source_run.root, terminal_root / "run", kind="terminal-status-tamper"
        )
        terminal_report = AssuranceReport.evaluate(SimulationRun.load(terminal_path), task)
        terminal_report.write(terminal_root / "result")
        actual_terminal = terminal_report.data["verdict"]["simulation_bounded_physical_success"]
        cases.append(
            {
                "case_id": f"{source.profile_id}/terminal-status-tamper",
                "corruption": "terminal-status-tamper",
                "expected": "supported",
                "actual": actual_terminal,
                "passed": actual_terminal == "supported",
                "report_sha256": terminal_report.digest,
            }
        )

        digest_path = corrupt_run(
            source_run.root,
            output / "cases" / source.profile_id / "digest-tamper" / "run",
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
                "case_id": f"{source.profile_id}/digest-tamper",
                "corruption": "digest-tamper",
                "expected": "rejected_before_evaluation",
                "actual": "rejected_before_evaluation" if rejected else "accepted",
                "passed": rejected,
            }
        )

        if source.profile_id == "stack-panda":
            contact_root = output / "cases" / source.profile_id / "missing-contact"
            contact_path = corrupt_run(
                source_run.root, contact_root / "run", kind="missing-channel", channel="contact"
            )
            contact_report = AssuranceReport.evaluate(SimulationRun.load(contact_path), task)
            contact_report.write(contact_root / "result")
            actual_contact = contact_report.data["verdict"]["goal_status"]
            cases.append(
                {
                    "case_id": f"{source.profile_id}/missing-contact",
                    "corruption": "missing-channel",
                    "channel": "contact",
                    "expected": "unknown",
                    "actual": actual_contact,
                    "passed": actual_contact == "unknown",
                    "report_sha256": contact_report.digest,
                }
            )

    result: dict[str, Any] = {
        "schema_version": "robot-spatial-maniskill-manipulation-corruption-matrix.v1",
        "source_run_manifest_sha256": source_digests,
        "task_spec_sha256": task_digests,
        "case_count": len(cases),
        "passed_count": sum(case["passed"] for case in cases),
        "cases": cases,
        "limitations": [
            "Corruptions are evidence-integrity controls over four real normalized simulator runs, not additional physical episodes."
        ],
    }
    result["matrix_sha256"] = sha256_json(result)
    write_json(output / "corruption-matrix.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.benchmark_root.resolve(), args.out.resolve())
    print(json.dumps({"passed": result["passed_count"], "total": result["case_count"]}, indent=2))
    return 0 if result["passed_count"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
