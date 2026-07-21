#!/usr/bin/env python3
"""Oracle-isolated cross-task and cross-robot ManiSkill evidence benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401 - registers environments
from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve as solve_peg_panda
from mani_skill.examples.motionplanning.panda.solutions.push_cube import solve as solve_push_panda
from mani_skill.examples.motionplanning.panda.solutions.stack_cube import solve as solve_stack_panda
from mani_skill.examples.motionplanning.xarm6.solutions.pick_cube import solve as solve_pick_xarm

from robot_spatial_understanding.adapters import ManiSkillAdapter
from robot_spatial_understanding.benchmark import _classification_metrics
from robot_spatial_understanding.report import AssuranceReport
from robot_spatial_understanding.task import TaskSpec
from robot_spatial_understanding.util import ensure_new_directory, sha256_json, write_json


SCHEMA = "robot-spatial-maniskill-manipulation-benchmark.v1"


@dataclass(frozen=True)
class Profile:
    profile_id: str
    env_id: str
    robot_uid: str
    example_dir: str
    entity_map_name: str
    solver: Callable[..., Any]
    fixed_horizon: int
    official_predicates: dict[str, str]


PROFILES = {
    profile.profile_id: profile
    for profile in (
        Profile(
            "push-panda",
            "PushCube-v1",
            "panda",
            "pushcube-live",
            "pushcube-entities.yaml",
            solve_push_panda,
            100,
            {"object_placed": "success"},
        ),
        Profile(
            "stack-panda",
            "StackCube-v1",
            "panda_wristcam",
            "stackcube-live",
            "stackcube-entities.yaml",
            solve_stack_panda,
            150,
            {
                "cube_a_grasped": "is_cubeA_grasped",
                "cube_a_on_cube_b": "is_cubeA_on_cubeB",
                "cube_a_static": "is_cubeA_static",
            },
        ),
        Profile(
            "peg-panda",
            "PegInsertionSide-v1",
            "panda_wristcam",
            "peginsertion-live",
            "peginsertion-entities.yaml",
            solve_peg_panda,
            150,
            {"peg_head_inserted": "success"},
        ),
        Profile(
            "pick-xarm",
            "PickCube-v1",
            "xarm6_robotiq",
            "pickcube-xarm-live",
            "pickcube-xarm-entities.yaml",
            solve_pick_xarm,
            100,
            {"cube_at_goal": "is_obj_placed", "robot_static": "is_robot_static"},
        ),
    )
}


class ActionRecorder(gym.Wrapper):
    """Record controller actions without retaining observations or evaluator outputs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.actions: list[np.ndarray] = []
        self.hold_action: np.ndarray | None = None

    def reset(self, *args: Any, **kwargs: Any):
        result = self.env.reset(*args, **kwargs)
        action_dimension = int(self.action_space.shape[-1])
        qpos = ManiSkillAdapter._numpy(self.unwrapped.agent.robot.get_qpos())[0]
        self.hold_action = np.hstack([qpos[: action_dimension - 1], 1.0]).astype(np.float32)
        return result

    def step(self, action: Any):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        return self.env.step(action)


def make_env(profile: Profile, sim_backend: str, render_backend: str) -> gym.Env:
    return gym.make(
        profile.env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend=render_backend,
        sim_backend=sim_backend,
        robot_uids=profile.robot_uid,
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
    profile: Profile,
    seed: int,
    output: Path,
    *,
    sim_backend: str,
    render_backend: str,
) -> tuple[Path, Path, dict[str, Any]]:
    environment = ActionRecorder(make_env(profile, sim_backend, render_backend))
    generation_error: str | None = None
    try:
        try:
            profile.solver(environment, seed=seed, debug=False, vis=False)
        except Exception as error:
            generation_error = f"{type(error).__name__}: {error}"
        if environment.hold_action is None:
            environment.reset(seed=seed)
        if not environment.actions:
            environment.actions.append(environment.hold_action.copy())
        solver_actions = np.asarray(environment.actions, dtype=np.float32)
        no_op_actions = np.repeat(environment.hold_action[None, :], profile.fixed_horizon, axis=0)
    finally:
        environment.close()
    solver_path = write_actions(output / f"seed-{seed:03d}-solver.h5", solver_actions)
    no_op_path = write_actions(output / f"seed-{seed:03d}-no-op.h5", no_op_actions)
    return solver_path, no_op_path, {
        "profile_id": profile.profile_id,
        "seed": seed,
        "solver_action_count": int(solver_actions.shape[0]),
        "controller_generation_error": generation_error,
        "controller_outcomes_retained": False,
    }


def initial_state_sha256(raw: Any, mapping: dict[str, Any], seed: int) -> str:
    active_joints = [joint.name for joint in raw.agent.robot.get_active_joints()]
    joint_map = {str(output): str(source) for output, source in mapping["joints"].items()}
    source_joint_index = {name: index for index, name in enumerate(active_joints)}
    joint_state, poses, measurements = ManiSkillAdapter._state_snapshot(
        raw, mapping, 0, joint_map, source_joint_index
    )
    return sha256_json(
        {
            "seed": seed,
            "joint_state": {"time_s": 0.0, **joint_state},
            "poses": poses,
            "measurements": measurements,
        }
    )


def official_replay(
    profile: Profile,
    seed: int,
    trajectory: Path,
    mapping: dict[str, Any],
    *,
    sim_backend: str,
    render_backend: str,
) -> dict[str, Any]:
    actions, trajectory_digest = ManiSkillAdapter._load_actions(trajectory, 0)
    environment = make_env(profile, sim_backend, render_backend)
    try:
        environment.reset(seed=seed)
        raw = environment.unwrapped
        initial_digest = initial_state_sha256(raw, mapping, seed)
        for step in range(profile.fixed_horizon):
            environment.step(actions[min(step, len(actions) - 1)])
        # Oracle boundary: official evaluator output is first read here, after all reports are sealed.
        official = raw.evaluate()
        predicate_labels = {
            predicate_id: "supported" if bool(official[official_key].item()) else "refuted"
            for predicate_id, official_key in profile.official_predicates.items()
        }
        success = bool(official["success"].item())
        return {
            "trajectory_sha256": trajectory_digest,
            "initial_state_sha256": initial_digest,
            "predicates": predicate_labels,
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
    output = args.out.resolve()
    ensure_new_directory(output)
    repo_root = Path(__file__).resolve().parents[1]
    profiles = [PROFILES[profile_id] for profile_id in args.profiles]
    loaded: dict[str, tuple[Path, dict[str, Any], str, Path, TaskSpec]] = {}
    for profile in profiles:
        example = repo_root / "examples" / profile.example_dir
        entity_map_path = example / profile.entity_map_name
        task_path = example / "task.yaml"
        mapping, mapping_digest = ManiSkillAdapter._load_entity_map(entity_map_path, profile.env_id)
        task = TaskSpec.load(task_path)
        loaded[profile.profile_id] = (entity_map_path, mapping, mapping_digest, task_path, task)

    adapter = ManiSkillAdapter()
    predictions: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []

    # Phase 1: every task/robot prediction is written before any official evaluator is read.
    for profile in profiles:
        entity_map_path, _mapping, _mapping_digest, _task_path, task = loaded[profile.profile_id]
        for seed in args.seeds:
            print(f"[phase1] profile={profile.profile_id} seed={seed} generate", flush=True)
            solver_path, no_op_path, generation_record = generate_actions(
                profile,
                seed,
                output / "action-inputs" / profile.profile_id,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
            )
            generation.append(generation_record)
            for intervention, trajectory in (("solver", solver_path), ("no-op", no_op_path)):
                case_id = f"{profile.profile_id}/seed-{seed:03d}-{intervention}"
                print(f"[phase1] capture {case_id}", flush=True)
                case_root = output / "candidates" / profile.profile_id / f"seed-{seed:03d}-{intervention}"
                run_artifact = adapter.capture_episode(
                    case_root / "run",
                    env_id=profile.env_id,
                    seed=seed,
                    trajectory=trajectory,
                    entity_map=entity_map_path,
                    sim_backend=args.sim_backend,
                    num_envs=1,
                    fixed_horizon=profile.fixed_horizon,
                    render_backend=args.render_backend,
                )
                if isinstance(run_artifact, list):
                    raise RuntimeError("single-environment benchmark capture returned multiple runs")
                report = AssuranceReport.evaluate(run_artifact, task)
                report.write(case_root / "result")
                predictions.append(
                    {
                        "case_id": case_id,
                        "profile": profile,
                        "seed": seed,
                        "intervention": intervention,
                        "trajectory": trajectory,
                        "trajectory_sha256": run_artifact.manifest["intervention"]["trajectory_sha256"],
                        "run_manifest_sha256": run_artifact.digest,
                        "initial_state_sha256": run_artifact.manifest["world"]["initial_state_sha256"],
                        "robot_model_sha256": run_artifact.manifest["robot"]["model_sha256"],
                        "task_source_sha256": run_artifact.manifest["world"]["task_source_sha256"],
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

    predicate_pairs: list[tuple[str, str]] = []
    verdict_pairs: list[tuple[str, str]] = []
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        profile = prediction["profile"]
        _entity_map_path, mapping, _mapping_digest, _task_path, task = loaded[profile.profile_id]
        print(f"[phase2] official replay {prediction['case_id']}", flush=True)
        reference = official_replay(
            profile,
            prediction["seed"],
            prediction["trajectory"],
            mapping,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
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
        reference_path = output / "references" / profile.profile_id / f"seed-{prediction['seed']:03d}-{prediction['intervention']}.json"
        write_json(reference_path, reference_record)
        cases.append(
            {
                "case_id": prediction["case_id"],
                "profile_id": profile.profile_id,
                "env_id": profile.env_id,
                "robot_uid": profile.robot_uid,
                "seed": prediction["seed"],
                "intervention": prediction["intervention"],
                "run_manifest_sha256": prediction["run_manifest_sha256"],
                "robot_model_sha256": prediction["robot_model_sha256"],
                "task_source_sha256": prediction["task_source_sha256"],
                "report_sha256": prediction["report_sha256"],
                "reference_sha256": reference_record["reference_sha256"],
                "predicted_verdict": prediction["prediction"],
                "official_verdict": reference["verdict"],
                "agreement": prediction["prediction"] == reference["verdict"],
                "predicate_agreement": all(
                    prediction["predicate_predictions"][predicate_id] == actual
                    for predicate_id, actual in reference["predicates"].items()
                ),
                "scored_predicates": sorted(reference["predicates"]),
                "unscored_predicates": unscored,
            }
        )

    per_profile: dict[str, Any] = {}
    acceptance_passed = True
    for profile in profiles:
        _entity_map_path, _mapping, mapping_digest, _task_path, task = loaded[profile.profile_id]
        profile_cases = [case for case in cases if case["profile_id"] == profile.profile_id]
        supported = sum(case["official_verdict"] == "supported" for case in profile_cases)
        refuted = sum(case["official_verdict"] == "refuted" for case in profile_cases)
        passed = (
            all(case["agreement"] and case["predicate_agreement"] for case in profile_cases)
            and supported >= args.minimum_supported_per_profile
            and refuted >= args.minimum_refuted_per_profile
        )
        acceptance_passed = acceptance_passed and passed
        per_profile[profile.profile_id] = {
            "env_id": profile.env_id,
            "robot_uid": profile.robot_uid,
            "fixed_horizon": profile.fixed_horizon,
            "entity_map_sha256": mapping_digest,
            "task_spec_sha256": task.digest,
            "robot_model_sha256": sorted({case["robot_model_sha256"] for case in profile_cases}),
            "task_source_sha256": sorted({case["task_source_sha256"] for case in profile_cases}),
            "case_count": len(profile_cases),
            "supported_references": supported,
            "refuted_references": refuted,
            "agreement_count": sum(case["agreement"] for case in profile_cases),
            "predicate_agreement_count": sum(case["predicate_agreement"] for case in profile_cases),
            "passed": passed,
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "environment": environment_record(),
        "profiles": per_profile,
        "seeds": args.seeds,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "case_count": len(cases),
        "oracle_isolation": {
            "action_files_contain_only_actions": True,
            "capture_discarded_step_returns": True,
            "all_cross_profile_predictions_written_before_official_replay": True,
            "references_outside_candidate_runs": True,
        },
        "predicate_metrics": _classification_metrics(predicate_pairs),
        "episode_metrics": _classification_metrics(verdict_pairs),
        "generation": generation,
        "cases": cases,
        "acceptance": {
            "requires_full_episode_agreement": True,
            "requires_full_scored_predicate_agreement": True,
            "minimum_supported_per_profile": args.minimum_supported_per_profile,
            "minimum_refuted_per_profile": args.minimum_refuted_per_profile,
            "passed": acceptance_passed,
        },
        "limitations": [
            "Results are bounded to pinned ManiSkill/SAPIEN simulation, declared raw channels, seeds, actions, geometry, thresholds, and horizons.",
            "Cross-task and cross-robot simulation agreement does not establish hardware transfer, causation, authorization, or safety.",
        ],
    }
    result["benchmark_sha256"] = sha256_json(result)
    write_json(output / "benchmark.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=sorted(PROFILES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu")
    parser.add_argument("--render-backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--minimum-supported-per-profile", type=int, default=1)
    parser.add_argument("--minimum-refuted-per-profile", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "benchmark": str((args.out / "benchmark.json").resolve()),
                "case_count": result["case_count"],
                "episode_metrics": result["episode_metrics"],
                "acceptance": result["acceptance"],
            },
            indent=2,
        )
    )
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
