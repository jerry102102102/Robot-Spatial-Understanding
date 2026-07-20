#!/usr/bin/env python3
"""Reproduce a live MuJoCo smoke benchmark with oracle isolation."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
from typing import Any

from robot_spatial_understanding.adapters import GymnasiumRoboticsAdapter
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


def official_replay(env_id: str, seed: int, max_steps: int, controller_gain: float) -> bool:
    """Run the official environment independently and reveal only its terminal label."""

    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    environment = gym.make(env_id, max_episode_steps=max_steps)
    try:
        observation, _reset_info = environment.reset(seed=seed)
        final_info: dict[str, Any] = {}
        for _step in range(max_steps):
            action = GymnasiumRoboticsAdapter.goal_action(observation, controller_gain)
            observation, _reward, terminated, truncated, final_info = environment.step(action)
            if bool(terminated) or bool(truncated):
                break
        return bool(final_info.get("is_success", False))
    finally:
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-id", default="FetchReach-v3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 7])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--controller-gain", type=float, default=10.0)
    args = parser.parse_args()

    output = args.out.resolve()
    ensure_new_directory(output)
    task_path = Path(__file__).resolve().parents[1] / "examples" / "fetch-reach" / "task.yaml"
    task = TaskSpec.load(task_path)
    adapter = GymnasiumRoboticsAdapter()

    # Phase 1: capture and predict every case. Capture code discards all official labels.
    predictions: list[dict[str, Any]] = []
    for seed in args.seeds:
        case_root = output / "cases" / f"seed-{seed}"
        run = adapter.capture_goal_episode(
            case_root / "run",
            env_id=args.env_id,
            seed=seed,
            max_steps=args.max_steps,
            controller_gain=args.controller_gain,
        )
        report = AssuranceReport.evaluate(run, task)
        report.write(case_root / "result")
        prediction = str(report.data["verdict"]["simulation_bounded_physical_success"])
        predictions.append(
            {
                "seed": seed,
                "run_manifest_sha256": run.digest,
                "report_sha256": report.digest,
                "prediction": prediction,
            }
        )

    # Phase 2: only after all reports exist, run the official oracle path.
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        official_success = official_replay(
            args.env_id,
            prediction["seed"],
            args.max_steps,
            args.controller_gain,
        )
        official_label = "supported" if official_success else "refuted"
        cases.append(
            {
                **prediction,
                "official_reference": official_label,
                "agreement": prediction["prediction"] == official_label,
            }
        )

    record: dict[str, Any] = {
        "schema_version": "robot-spatial-live-smoke-record.v1",
        "environment": args.env_id,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "controller_gain": args.controller_gain,
        "versions": {
            "gymnasium": metadata.version("gymnasium"),
            "gymnasium_robotics": metadata.version("gymnasium-robotics"),
            "mujoco": metadata.version("mujoco"),
        },
        "oracle_isolation": {
            "capture_discarded_reward_and_info": True,
            "all_predictions_completed_before_official_replay": True,
            "official_reference_stored_outside_candidate_runs": True,
        },
        "cases": cases,
        "agreement_count": sum(1 for case in cases if case["agreement"]),
        "case_count": len(cases),
        "limitations": [
            "This is a two-seed live simulator smoke check, not a statistical benchmark or hardware-safety claim.",
            "The controller is only a deterministic episode generator and is outside the evaluator claim.",
        ],
    }
    record["record_sha256"] = sha256_json(record)
    write_json(output / "smoke-record.json", record)
    print(output / "smoke-record.json")
    return 0 if record["agreement_count"] == record["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
