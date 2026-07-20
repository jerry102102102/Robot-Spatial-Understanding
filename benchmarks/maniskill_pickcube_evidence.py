#!/usr/bin/env python3
"""Run an oracle-isolated live ManiSkill PickCube evidence benchmark.

Phase 1 generates action-only trajectories, captures raw simulator state, and writes every Robot
Spatial report. Phase 2 starts fresh same-seed environments, replays the immutable actions, and
only then reads ManiSkill's official evaluator. The prediction phase never reads recorded reward,
success, info, observations, environment states, or evaluator output.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers environments
from mani_skill.examples.motionplanning.panda.solutions import solvePickCube

from robot_spatial_understanding.adapters import ManiSkillAdapter
from robot_spatial_understanding.benchmark import _classification_metrics
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


SCHEMA = "robot-spatial-maniskill-pickcube-benchmark.v1"


class ActionRecorder(gym.Wrapper):
    """Record controller actions while leaving the official solution otherwise unchanged."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.actions: list[np.ndarray] = []
        self.hold_action: np.ndarray | None = None

    def reset(self, *args: Any, **kwargs: Any):
        result = self.env.reset(*args, **kwargs)
        qpos = self.unwrapped.agent.robot.get_qpos()[0, :7].detach().cpu().numpy()
        self.hold_action = np.hstack([qpos, 1.0]).astype(np.float32)
        return result

    def step(self, action: Any):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return self.env.step(action)


def make_env(env_id: str, sim_backend: str, render_backend: str) -> gym.Env:
    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend=render_backend,
        sim_backend=sim_backend,
    )


def write_actions(path: Path, actions: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with h5py.File(path, "w", libver="earliest") as archive:
        group = archive.create_group("traj_0", track_order=False)
        group.create_dataset("actions", data=np.asarray(actions, dtype=np.float32), dtype=np.float32)
    return path


def generate_actions(
    env_id: str,
    seed: int,
    output: Path,
    *,
    sim_backend: str,
    render_backend: str,
    fixed_horizon: int,
) -> tuple[Path, Path, dict[str, Any]]:
    print(f"[generate] seed={seed} create environment", flush=True)
    environment = ActionRecorder(make_env(env_id, sim_backend, render_backend))
    generation_error: str | None = None
    try:
        try:
            # The return value contains outcome-bearing info and is deliberately ignored.
            print(f"[generate] seed={seed} run official motion planner", flush=True)
            solvePickCube(environment, seed=seed, debug=False, vis=False)
            print(f"[generate] seed={seed} planner returned with {len(environment.actions)} actions", flush=True)
        except Exception as error:  # preserve a controller-generation failure as evidence
            generation_error = f"{type(error).__name__}: {error}"
        if environment.hold_action is None:
            environment.reset(seed=seed)
        if not environment.actions:
            environment.actions.append(environment.hold_action.copy())
        solver_actions = np.asarray(environment.actions, dtype=np.float32)
        hold_actions = np.repeat(environment.hold_action[None, :], fixed_horizon, axis=0)
    finally:
        environment.close()
    solver_path = write_actions(output / f"seed-{seed:03d}-solver.h5", solver_actions)
    no_op_path = write_actions(output / f"seed-{seed:03d}-no-op.h5", hold_actions)
    return solver_path, no_op_path, {
        "seed": seed,
        "solver_action_count": int(solver_actions.shape[0]),
        "controller_generation_error": generation_error,
        "outcome_read_during_generation": False,
    }


def initial_state_sha256(raw: Any, mapping: dict[str, Any], seed: int) -> str:
    joint_names = [joint.name for joint in raw.agent.robot.get_active_joints()]
    qpos = ManiSkillAdapter._numpy(raw.agent.robot.get_qpos())[0]
    qvel = ManiSkillAdapter._numpy(raw.agent.robot.get_qvel())[0]
    source_index = {name: index for index, name in enumerate(joint_names)}
    joint_state = {
        "time_s": 0.0,
        "positions": {
            output: float(qpos[source_index[source]])
            for output, source in mapping["joints"].items()
        },
        "velocities": {
            output: float(qvel[source_index[source]])
            for output, source in mapping["joints"].items()
        },
    }
    poses = {
        role: ManiSkillAdapter._pose(actor.pose.raw_pose, 0)
        for role, actor in {
            "tcp": raw.agent.tcp,
            "left_finger": raw.agent.finger1_link,
            "right_finger": raw.agent.finger2_link,
            "cube": raw.cube,
            "goal": raw.goal_site,
        }.items()
    }
    return sha256_json({"seed": seed, "joint_state": joint_state, "poses": poses})


def official_replay(
    env_id: str,
    seed: int,
    trajectory: Path,
    mapping: dict[str, Any],
    *,
    sim_backend: str,
    render_backend: str,
    fixed_horizon: int,
    initialization: str | None = None,
) -> dict[str, Any]:
    actions, trajectory_digest = ManiSkillAdapter._load_actions(trajectory, 0)
    environment = make_env(env_id, sim_backend, render_backend)
    try:
        environment.reset(seed=seed)
        raw = environment.unwrapped
        if initialization == "goal_at_cube":
            raw.goal_site.set_pose(raw.cube.pose)
        initial_digest = initial_state_sha256(raw, mapping, seed)
        for step in range(fixed_horizon):
            environment.step(actions[min(step, len(actions) - 1)])
        # This is the first and only point in the benchmark path that reads official evaluator output.
        official = raw.evaluate()
        placed = bool(official["is_obj_placed"].item())
        static = bool(official["is_robot_static"].item())
        success = bool(official["success"].item())
        return {
            "trajectory_sha256": trajectory_digest,
            "initial_state_sha256": initial_digest,
            "predicates": {
                "cube_at_goal": "supported" if placed else "refuted",
                "robot_static": "supported" if static else "refuted",
            },
            "verdict": "supported" if success else "refuted",
        }
    finally:
        environment.close()


def environment_record() -> dict[str, Any]:
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": driver,
        "mani_skill": metadata.version("mani-skill"),
        "sapien": metadata.version("sapien"),
        "gymnasium": metadata.version("gymnasium"),
        "numpy": np.__version__,
        "adapter": ManiSkillAdapter.version,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_supported < 0 or args.minimum_refuted < 0:
        raise ValueError("minimum reference counts must be non-negative")
    output = args.out.resolve()
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    entity_map_path = args.entity_map or repo_root / "examples" / "pickcube-live" / "pickcube-entities.yaml"
    task_path = args.task or repo_root / "examples" / "pickcube-live" / "task.yaml"
    mapping, mapping_digest = ManiSkillAdapter._load_entity_map(entity_map_path, args.env_id)
    task = TaskSpec.load(task_path)
    adapter = ManiSkillAdapter()
    predictions: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []

    # Phase 1: generate action-only inputs, capture raw state, and write every prediction.
    for seed in args.seeds:
        print(f"[phase1] seed={seed}", flush=True)
        solver_path, no_op_path, generation_record = generate_actions(
            args.env_id,
            seed,
            output / "action-inputs",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            fixed_horizon=args.fixed_horizon,
        )
        generation.append(generation_record)
        interventions = [("solver", solver_path)]
        if args.include_no_op:
            interventions.append(("no-op", no_op_path))
        for intervention, trajectory in interventions:
            case_id = f"seed-{seed:03d}-{intervention}"
            print(f"[phase1] capture {case_id}", flush=True)
            case_root = output / "candidates" / case_id
            run_artifact = adapter.capture_episode(
                case_root / "run",
                env_id=args.env_id,
                seed=seed,
                trajectory=trajectory,
                entity_map=entity_map_path,
                sim_backend=args.sim_backend,
                num_envs=1,
                fixed_horizon=args.fixed_horizon,
                render_backend=args.render_backend,
            )
            if isinstance(run_artifact, list):
                raise RuntimeError("single-environment benchmark capture returned multiple runs")
            report = AssuranceReport.evaluate(run_artifact, task)
            report.write(case_root / "result")
            predictions.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "intervention": intervention,
                    "trajectory": trajectory,
                    "trajectory_sha256": run_artifact.manifest["intervention"]["trajectory_sha256"],
                    "run_manifest_sha256": run_artifact.digest,
                    "initial_state_sha256": run_artifact.manifest["world"]["initial_state_sha256"],
                    "report_sha256": report.digest,
                    "prediction": report.data["verdict"]["simulation_bounded_physical_success"],
                    "predicate_predictions": {
                        item["predicate_id"]: item["status"] for item in report.data["predicates"]
                    },
                }
            )
    write_json(
        output / "prediction-phase-complete.json",
        {
            "status": "complete",
            "case_ids": [item["case_id"] for item in predictions],
            "report_sha256": [item["report_sha256"] for item in predictions],
            "official_evaluator_read": False,
        },
    )

    # Phase 2: new environments, same seeds/actions, official labels revealed after all reports exist.
    predicate_pairs: list[tuple[str, str]] = []
    verdict_pairs: list[tuple[str, str]] = []
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        print(f"[phase2] official replay {prediction['case_id']}", flush=True)
        reference = official_replay(
            args.env_id,
            prediction["seed"],
            prediction["trajectory"],
            mapping,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            fixed_horizon=args.fixed_horizon,
        )
        if reference["trajectory_sha256"] != prediction["trajectory_sha256"]:
            raise RuntimeError(f"trajectory digest mismatch for {prediction['case_id']}")
        if reference["initial_state_sha256"] != prediction["initial_state_sha256"]:
            raise RuntimeError(f"initial state digest mismatch for {prediction['case_id']}")
        for predicate_id, actual in reference["predicates"].items():
            predicate_pairs.append((actual, prediction["predicate_predictions"][predicate_id]))
        verdict_pairs.append((reference["verdict"], prediction["prediction"]))
        unscored = sorted(set(prediction["predicate_predictions"]) - set(reference["predicates"]))
        reference_record = {
            "schema_version": "robot-spatial-reference-result.v1",
            "case_id": prediction["case_id"],
            "run_manifest_sha256": prediction["run_manifest_sha256"],
            "task_spec_sha256": task.digest,
            "predicates": reference["predicates"],
            "unscored_predicates": unscored,
            "verdict": reference["verdict"],
        }
        reference_record["reference_sha256"] = sha256_json(reference_record)
        write_json(output / "references" / f"{prediction['case_id']}.json", reference_record)
        cases.append(
            {
                "case_id": prediction["case_id"],
                "seed": prediction["seed"],
                "intervention": prediction["intervention"],
                "run_manifest_sha256": prediction["run_manifest_sha256"],
                "report_sha256": prediction["report_sha256"],
                "reference_sha256": reference_record["reference_sha256"],
                "predicted_verdict": prediction["prediction"],
                "official_verdict": reference["verdict"],
                "agreement": prediction["prediction"] == reference["verdict"],
                "scored_predicates": sorted(reference["predicates"]),
                "unscored_predicates": unscored,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "environment": environment_record(),
        "env_id": args.env_id,
        "seeds": args.seeds,
        "fixed_horizon": args.fixed_horizon,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "entity_map_sha256": mapping_digest,
        "task_spec_sha256": task.digest,
        "case_count": len(cases),
        "oracle_isolation": {
            "action_files_contain_only_actions": True,
            "capture_discarded_step_returns": True,
            "all_predictions_written_before_official_replay": True,
            "references_outside_candidate_runs": True,
        },
        "predicate_metrics": _classification_metrics(predicate_pairs),
        "episode_metrics": _classification_metrics(verdict_pairs),
        "generation": generation,
        "cases": cases,
        "limitations": [
            "Results are bounded to the pinned ManiSkill/SAPIEN simulation, raw captured channels, seeds, actions, thresholds, and horizon.",
            "Unscored contact, grasp, following, lift, and collision diagnostics are not labeled by ManiSkill's official PickCube evaluator.",
            "This benchmark does not establish real-world causation, authorization, hardware behavior, or safety.",
        ],
    }
    supported_references = sum(case["official_verdict"] == "supported" for case in cases)
    refuted_references = sum(case["official_verdict"] == "refuted" for case in cases)
    result["acceptance"] = {
        "requires_full_episode_agreement": True,
        "minimum_supported_references": args.minimum_supported,
        "minimum_refuted_references": args.minimum_refuted,
        "supported_references": supported_references,
        "refuted_references": refuted_references,
        "passed": (
            all(case["agreement"] for case in cases)
            and supported_references >= args.minimum_supported
            and refuted_references >= args.minimum_refuted
        ),
    }
    result["benchmark_sha256"] = sha256_json(result)
    write_json(output / "benchmark.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--include-no-op", action="store_true")
    parser.add_argument("--fixed-horizon", type=int, default=100)
    parser.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--entity-map", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--minimum-supported", type=int, default=0)
    parser.add_argument("--minimum-refuted", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps({"benchmark": str((args.out / "benchmark.json").resolve()), "case_count": result["case_count"], "episode_metrics": result["episode_metrics"]}, indent=2))
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
