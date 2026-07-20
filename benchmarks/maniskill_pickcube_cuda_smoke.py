#!/usr/bin/env python3
"""Capture one 16-environment CUDA PickCube batch and verify honest channel abstention."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from maniskill_pickcube_evidence import environment_record
from robot_spatial_understanding.adapters import ManiSkillAdapter
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.out.resolve()
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    entity_map = args.entity_map or repo_root / "examples" / "pickcube-live" / "pickcube-entities.yaml"
    task_path = args.task or repo_root / "examples" / "pickcube-live" / "task.yaml"
    task = TaskSpec.load(task_path)
    captured = ManiSkillAdapter().capture_episode(
        output / "runs",
        env_id=args.env_id,
        seed=args.seed,
        trajectory=args.trajectory,
        entity_map=entity_map,
        sim_backend="physx_cuda",
        num_envs=args.num_envs,
        fixed_horizon=args.fixed_horizon,
        render_backend=args.render_backend,
    )
    if not isinstance(captured, list) or len(captured) != args.num_envs:
        raise RuntimeError(f"expected {args.num_envs} independently normalized CUDA runs")

    cases: list[dict[str, Any]] = []
    for index, candidate in enumerate(captured):
        report = AssuranceReport.evaluate(candidate, task)
        report.write(output / "results" / f"env-{index:03d}")
        statuses = {item["predicate_id"]: item["status"] for item in report.data["predicates"]}
        collision_channel = candidate.manifest["channels"]["collision"]
        passed = collision_channel["status"] == "unavailable" and statuses["collision_free"] == "unknown"
        cases.append(
            {
                "sub_environment_index": index,
                "seed": candidate.manifest["seed"],
                "run_manifest_sha256": candidate.digest,
                "report_sha256": report.digest,
                "collision_channel": collision_channel,
                "collision_free": statuses["collision_free"],
                "passed": passed,
            }
        )

    record: dict[str, Any] = {
        "schema_version": "robot-spatial-maniskill-cuda-parity-smoke.v1",
        "environment": environment_record(),
        "env_id": args.env_id,
        "sim_backend": "physx_cuda",
        "planner_backend": "physx_cpu",
        "fixed_horizon": args.fixed_horizon,
        "num_envs": args.num_envs,
        "trajectory_sha256": captured[0].manifest["intervention"]["trajectory_sha256"],
        "trajectory_action_count": captured[0].manifest["intervention"]["action_count"],
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case["passed"]),
        "cases": cases,
        "claim": "parallel capture and channel-availability smoke only; not a GPU-parallel motion-planner benchmark",
        "limitations": [
            "The same seed-2 CPU-planner action trajectory is broadcast to all CUDA sub-environments.",
            "GPU collision enumeration is unavailable; no CPU replay data is substituted.",
        ],
    }
    record["record_sha256"] = sha256_json(record)
    write_json(output / "cuda-parity-smoke.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fixed-horizon", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--entity-map", type=Path)
    parser.add_argument("--task", type=Path)
    args = parser.parse_args()
    record = run(args)
    print(record["passed_count"], "/", record["case_count"])
    return 0 if record["passed_count"] == record["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
