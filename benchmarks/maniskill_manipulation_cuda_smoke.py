#!/usr/bin/env python3
"""Replay sealed CPU-planner actions through four PhysX CUDA task/robot profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from maniskill_manipulation_evidence import PROFILES, environment_record, official_replay
from robot_spatial_understanding.adapters import ManiSkillAdapter
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


PROFILE_SEEDS = {
    "peg-panda": 4,
    "pick-xarm": 0,
    "push-panda": 0,
    "stack-panda": 0,
}


def run(cpu_benchmark_root: Path, output: Path, render_backend: str) -> dict[str, Any]:
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    adapter = ManiSkillAdapter()
    predictions: list[dict[str, Any]] = []
    loaded: dict[str, tuple[dict[str, Any], TaskSpec, str]] = {}
    for profile_id, seed in PROFILE_SEEDS.items():
        profile = PROFILES[profile_id]
        example_root = repo_root / "examples" / profile.example_dir
        entity_map = example_root / profile.entity_map_name
        mapping, mapping_digest = ManiSkillAdapter._load_entity_map(entity_map, profile.env_id)
        task = TaskSpec.load(example_root / "task.yaml")
        loaded[profile_id] = (mapping, task, mapping_digest)
        trajectory = cpu_benchmark_root / "action-inputs" / profile_id / f"seed-{seed:03d}-solver.h5"
        case_root = output / "candidates" / profile_id
        print(f"[cuda phase1] capture {profile_id} seed={seed}", flush=True)
        run_artifact = adapter.capture_episode(
            case_root / "run",
            env_id=profile.env_id,
            seed=seed,
            trajectory=trajectory,
            entity_map=entity_map,
            sim_backend="physx_cuda",
            num_envs=1,
            fixed_horizon=profile.fixed_horizon,
            render_backend=render_backend,
        )
        if isinstance(run_artifact, list):
            raise RuntimeError("single-environment CUDA smoke returned multiple runs")
        report = AssuranceReport.evaluate(run_artifact, task)
        report.write(case_root / "result")
        predictions.append(
            {
                "profile_id": profile_id,
                "profile": profile,
                "seed": seed,
                "trajectory": trajectory,
                "trajectory_sha256": run_artifact.manifest["intervention"]["trajectory_sha256"],
                "initial_state_sha256": run_artifact.manifest["world"]["initial_state_sha256"],
                "entity_map_sha256": mapping_digest,
                "task_spec_sha256": task.digest,
                "robot_model_sha256": run_artifact.manifest["robot"]["model_sha256"],
                "task_source_sha256": run_artifact.manifest["world"]["task_source_sha256"],
                "run_manifest_sha256": run_artifact.digest,
                "report_sha256": report.digest,
                "verdict": report.data["verdict"]["simulation_bounded_physical_success"],
                "predicates": {item["predicate_id"]: item["status"] for item in report.data["predicates"]},
                "collision_channel": run_artifact.manifest["channels"]["collision"],
            }
        )
    write_json(
        output / "prediction-phase-complete.json",
        {
            "status": "complete",
            "profile_ids": [item["profile_id"] for item in predictions],
            "report_sha256": [item["report_sha256"] for item in predictions],
            "official_evaluator_read": False,
        },
    )

    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        profile = prediction["profile"]
        mapping, _task, _mapping_digest = loaded[prediction["profile_id"]]
        print(f"[cuda phase2] official replay {prediction['profile_id']}", flush=True)
        reference = official_replay(
            profile,
            prediction["seed"],
            prediction["trajectory"],
            mapping,
            sim_backend="physx_cuda",
            render_backend=render_backend,
        )
        digests_match = (
            reference["trajectory_sha256"] == prediction["trajectory_sha256"]
            and reference["initial_state_sha256"] == prediction["initial_state_sha256"]
        )
        predicate_agreement = all(
            prediction["predicates"][predicate_id] == actual
            for predicate_id, actual in reference["predicates"].items()
        )
        verdict_agreement = prediction["verdict"] == reference["verdict"]
        collision_unavailable = prediction["collision_channel"].get("status") == "unavailable"
        cases.append(
            {
                "profile_id": prediction["profile_id"],
                "env_id": profile.env_id,
                "robot_uid": profile.robot_uid,
                "seed": prediction["seed"],
                "fixed_horizon": profile.fixed_horizon,
                "trajectory_sha256": prediction["trajectory_sha256"],
                "initial_state_sha256": prediction["initial_state_sha256"],
                "entity_map_sha256": prediction["entity_map_sha256"],
                "task_spec_sha256": prediction["task_spec_sha256"],
                "robot_model_sha256": prediction["robot_model_sha256"],
                "task_source_sha256": prediction["task_source_sha256"],
                "run_manifest_sha256": prediction["run_manifest_sha256"],
                "report_sha256": prediction["report_sha256"],
                "predicted_verdict": prediction["verdict"],
                "official_verdict": reference["verdict"],
                "digests_match": digests_match,
                "predicate_agreement": predicate_agreement,
                "verdict_agreement": verdict_agreement,
                "collision_channel": prediction["collision_channel"],
                "passed": digests_match and predicate_agreement and verdict_agreement and collision_unavailable,
            }
        )

    record: dict[str, Any] = {
        "schema_version": "robot-spatial-maniskill-manipulation-cuda-smoke.v1",
        "environment": environment_record(),
        "sim_backend": "physx_cuda",
        "planner_backend": "physx_cpu",
        "render_backend": render_backend,
        "case_count": len(cases),
        "passed_count": sum(case["passed"] for case in cases),
        "oracle_isolation": {
            "actions_reused_from_sealed_cpu_planner_inputs": True,
            "all_predictions_written_before_official_replay": True,
            "capture_discarded_step_returns": True,
        },
        "cases": cases,
        "limitations": [
            "This gate validates CUDA replay/capture/evaluator parity, not CUDA motion planning throughput.",
            "GPU scene-wide collision enumeration remains unavailable and is reported as unavailable rather than synthesized from CPU data.",
        ],
    }
    record["record_sha256"] = sha256_json(record)
    write_json(output / "cuda-smoke.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-benchmark-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-backend", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    record = run(args.cpu_benchmark_root.resolve(), args.out.resolve(), args.render_backend)
    print(json.dumps({"passed": record["passed_count"], "total": record["case_count"]}, indent=2))
    return 0 if record["passed_count"] == record["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
