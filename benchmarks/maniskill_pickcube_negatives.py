#!/usr/bin/env python3
"""Generate controlled live PickCube negative episodes from one official solver trajectory."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_pickcube_evidence import official_replay, write_actions
from robot_spatial_understanding.adapters import ManiSkillAdapter
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


def fixed(actions: np.ndarray, horizon: int) -> np.ndarray:
    rows = [actions[min(index, len(actions) - 1)] for index in range(horizon)]
    return np.asarray(rows, dtype=np.float32)


def variants(solver: np.ndarray, hold: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    baseline = fixed(solver, horizon)
    first_closed = next((index for index, action in enumerate(baseline) if action[-1] < 0.0), horizon - 1)
    contact_hold_index = min(first_closed + 5, horizon - 1)
    short_release_index = min(first_closed + 2, horizon - 1)

    close_without_following = np.repeat(hold[None, :], horizon, axis=0)
    close_without_following[:, -1] = -1.0

    contact_only = baseline.copy()
    contact_only[contact_hold_index + 1 :] = contact_only[contact_hold_index]

    short_contact = baseline.copy()
    release_action = short_contact[short_release_index].copy()
    release_action[-1] = 1.0
    short_contact[short_release_index + 1 :] = release_action

    push_only = baseline.copy()
    push_only[:, -1] = 1.0

    lift_and_drop = baseline.copy()
    lift_and_drop[max(0, horizon - 6) :, -1] = 1.0

    return {
        "close-without-following": close_without_following,
        "contact-only": contact_only,
        "short-contact": short_contact,
        "push-only": push_only,
        "lift-and-drop": lift_and_drop,
        "no-op": np.repeat(hold[None, :], horizon, axis=0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.out.resolve()
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    entity_map = args.entity_map or repo_root / "examples" / "pickcube-live" / "pickcube-entities.yaml"
    task_path = args.task or repo_root / "examples" / "pickcube-live" / "task.yaml"
    task = TaskSpec.load(task_path)
    mapping, _ = ManiSkillAdapter._load_entity_map(entity_map, args.env_id)
    solver, _ = ManiSkillAdapter._load_actions(args.trajectory, 0)
    hold_source, _ = ManiSkillAdapter._load_actions(args.no_op_trajectory, 0)
    hold = hold_source[0]
    action_variants = variants(solver, hold, args.fixed_horizon)
    action_variants["goal-at-start"] = np.repeat(hold[None, :], args.fixed_horizon, axis=0)
    adapter = ManiSkillAdapter()
    predictions: list[dict[str, Any]] = []

    # Phase 1: every controlled episode and report is immutable before official evaluation.
    for case_id, actions in action_variants.items():
        trajectory = write_actions(output / "action-inputs" / f"{case_id}.h5", actions)
        initialization = "goal_at_cube" if case_id == "goal-at-start" else None
        case_root = output / "candidates" / case_id
        run_artifact = adapter.capture_episode(
            case_root / "run",
            env_id=args.env_id,
            seed=args.seed,
            trajectory=trajectory,
            entity_map=entity_map,
            sim_backend=args.sim_backend,
            num_envs=1,
            fixed_horizon=args.fixed_horizon,
            render_backend=args.render_backend,
            initialization=initialization,
        )
        if isinstance(run_artifact, list):
            raise RuntimeError("negative generator expected one sub-environment")
        report = AssuranceReport.evaluate(run_artifact, task)
        report.write(case_root / "result")
        predictions.append(
            {
                "case_id": case_id,
                "trajectory": trajectory,
                "initialization": initialization,
                "run": run_artifact,
                "report": report,
                "statuses": {item["predicate_id"]: item["status"] for item in report.data["predicates"]},
            }
        )
    write_json(
        output / "prediction-phase-complete.json",
        {
            "case_ids": [item["case_id"] for item in predictions],
            "report_sha256": [item["report"].digest for item in predictions],
            "official_evaluator_read": False,
        },
    )

    # Phase 2: reveal official labels in fresh environments.
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        reference = official_replay(
            args.env_id,
            args.seed,
            prediction["trajectory"],
            mapping,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            fixed_horizon=args.fixed_horizon,
            initialization=prediction["initialization"],
        )
        statuses = prediction["statuses"]
        grasp_refuted = statuses["grasped"] == "refuted"
        expected_grasp_refuted = prediction["case_id"] in {
            "close-without-following",
            "short-contact",
            "push-only",
            "lift-and-drop",
            "no-op",
            "goal-at-start",
        }
        case_passed = (
            prediction["report"].data["verdict"]["simulation_bounded_physical_success"] == reference["verdict"]
            and (not expected_grasp_refuted or grasp_refuted)
        )
        if prediction["case_id"] == "contact-only":
            case_passed = case_passed and statuses["cube_lifted"] == "refuted" and grasp_refuted
        if prediction["case_id"] == "short-contact":
            case_passed = case_passed and (
                statuses["left_finger_contact"] == "refuted" or statuses["right_finger_contact"] == "refuted"
            )
        if prediction["case_id"] == "goal-at-start":
            case_passed = case_passed and reference["verdict"] == "supported" and grasp_refuted
        cases.append(
            {
                "case_id": prediction["case_id"],
                "run_manifest_sha256": prediction["run"].digest,
                "report_sha256": prediction["report"].digest,
                "predicted_verdict": prediction["report"].data["verdict"]["simulation_bounded_physical_success"],
                "official_verdict": reference["verdict"],
                "predicate_statuses": statuses,
                "passed": case_passed,
            }
        )

    # Re-evaluate the real goal-at-start run under a policy where table contact is not allowed.
    goal_case = next(item for item in predictions if item["case_id"] == "goal-at-start")
    collision_task_data = copy.deepcopy(task.data)
    collision_predicate = next(item for item in collision_task_data["predicates"] if item["predicate_id"] == "collision_free")
    collision_predicate["parameters"] = {}
    collision_task_path = output / "nonallowed-collision-task.json"
    write_json(collision_task_path, collision_task_data)
    collision_report = AssuranceReport.evaluate(goal_case["run"], TaskSpec.load(collision_task_path))
    collision_report.write(output / "candidates" / "goal-at-start-nonallowed-collision" / "result")
    collision_statuses = {item["predicate_id"]: item["status"] for item in collision_report.data["predicates"]}
    cases.append(
        {
            "case_id": "goal-at-start-nonallowed-collision",
            "source_run_manifest_sha256": goal_case["run"].digest,
            "report_sha256": collision_report.digest,
            "predicted_verdict": collision_report.data["verdict"]["simulation_bounded_physical_success"],
            "official_verdict": "supported",
            "predicate_statuses": collision_statuses,
            "passed": collision_statuses["cube_at_goal"] == "supported" and collision_statuses["collision_free"] == "refuted",
        }
    )

    result: dict[str, Any] = {
        "schema_version": "robot-spatial-maniskill-negative-matrix.v1",
        "seed": args.seed,
        "fixed_horizon": args.fixed_horizon,
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case["passed"]),
        "oracle_isolation": {"all_predictions_written_before_official_replay": True},
        "cases": cases,
        "limitations": [
            "These are simulator-controlled action and initialization negatives, not real-hardware trials or causal proofs."
        ],
    }
    result["matrix_sha256"] = sha256_json(result)
    write_json(output / "negative-matrix.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--no-op-trajectory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--fixed-horizon", type=int, default=100)
    parser.add_argument("--sim-backend", choices=["physx_cpu"], default="physx_cpu")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--entity-map", type=Path)
    parser.add_argument("--task", type=Path)
    args = parser.parse_args()
    result = run(args)
    print(result["passed_count"], "/", result["case_count"])
    return 0 if result["passed_count"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
