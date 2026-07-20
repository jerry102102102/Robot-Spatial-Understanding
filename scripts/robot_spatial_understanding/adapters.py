"""Simulator adapters that normalize raw exports without importing task outcomes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path
import tempfile
from typing import Any

from .errors import AdapterError
from .simulation import SimulationRun
from .util import load_json, require_mapping, sha256_file, write_json


class SimulatorAdapter(ABC):
    """Adapter boundary: map simulator state fields, never task success logic."""

    name = "abstract"
    version = "0.2.0"

    @abstractmethod
    def import_source(self, source: str | Path, out: str | Path) -> SimulationRun:
        """Normalize one immutable source into `simulation-run.v1`."""


class GenericJsonAdapter(SimulatorAdapter):
    name = "generic-json"

    def import_source(self, source: str | Path, out: str | Path) -> SimulationRun:
        return SimulationRun.import_generic_trace(source, out, adapter_name=self.name, adapter_version=self.version)


class DeclaredSimulatorExportAdapter(GenericJsonAdapter):
    """Normalize a generic trace while requiring an explicit simulator family."""

    accepted_simulators: tuple[str, ...] = ()

    def import_source(self, source: str | Path, out: str | Path) -> SimulationRun:
        trace = require_mapping(load_json(Path(source)), "adapter source")
        simulator = require_mapping(trace.get("simulator"), "adapter source.simulator")
        name = str(simulator.get("name", "")).lower()
        if not any(token in name for token in self.accepted_simulators):
            raise AdapterError(
                f"adapter {self.name!r} requires simulator.name containing one of {self.accepted_simulators}; got {name!r}"
            )
        return SimulationRun.import_generic_trace(source, out, adapter_name=self.name, adapter_version=self.version)


class ManiSkillAdapter(DeclaredSimulatorExportAdapter):
    """Offline SAPIEN/ManiSkill state export adapter; reward/success fields are rejected."""

    name = "maniskill"
    accepted_simulators = ("maniskill", "sapien")


class MuJoCoAdapter(DeclaredSimulatorExportAdapter):
    """Offline MuJoCo/robosuite/Meta-World state export adapter."""

    name = "mujoco"
    accepted_simulators = ("mujoco", "robosuite", "meta-world", "metaworld")


class GymnasiumRoboticsAdapter(MuJoCoAdapter):
    """Capture raw goal-state trajectories from Gymnasium Robotics/MuJoCo.

    The capture path deliberately ignores reward and ``info``. Official task
    labels must be produced by an independent replay after prediction.
    """

    name = "gymnasium-robotics"
    accepted_simulators = (*MuJoCoAdapter.accepted_simulators, "gymnasium robotics")

    @staticmethod
    def goal_action(observation: Any, gain: float = 10.0) -> Any:
        """A deterministic demo controller; it is not part of evaluation."""

        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - core dependency
            raise AdapterError("Gymnasium Robotics capture requires NumPy") from error
        achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
        desired = np.asarray(observation["desired_goal"], dtype=np.float64)
        if achieved.shape != (3,) or desired.shape != (3,):
            raise AdapterError("live goal capture currently requires three-dimensional achieved_goal and desired_goal")
        action = np.zeros(4, dtype=np.float64)
        action[:3] = np.clip((desired - achieved) * float(gain), -1.0, 1.0)
        return action

    def capture_goal_episode(
        self,
        out: str | Path,
        *,
        env_id: str = "FetchReach-v3",
        seed: int = 2,
        max_steps: int = 50,
        controller_gain: float = 10.0,
    ) -> SimulationRun:
        """Run a fixed-horizon GoalEnv episode and normalize raw poses only."""

        if max_steps <= 0:
            raise AdapterError("max_steps must be positive")
        try:
            import gymnasium as gym
            import gymnasium_robotics
            import mujoco
            import numpy as np
        except ImportError as error:
            raise AdapterError(
                "live Gymnasium Robotics capture requires the 'mujoco' extra: "
                "pip install 'robot-spatial-understanding[mujoco]'"
            ) from error

        gym.register_envs(gymnasium_robotics)
        environment = gym.make(env_id, max_episode_steps=max_steps)
        try:
            observation, _ignored_reset_info = environment.reset(seed=int(seed))
            if not isinstance(observation, dict) or not {"achieved_goal", "desired_goal"}.issubset(observation):
                raise AdapterError(f"environment {env_id!r} is not a supported three-dimensional GoalEnv")
            timestep_s = float(environment.unwrapped.dt)
            if timestep_s <= 0.0:
                raise AdapterError(f"environment {env_id!r} reported an invalid timestep")

            with tempfile.TemporaryDirectory(prefix="robot-spatial-gymnasium-") as temporary:
                temporary_root = Path(temporary)
                model_path = temporary_root / "model.xml"
                mujoco.mj_saveLastXML(str(model_path), environment.unwrapped.model)
                model_digest = sha256_file(model_path)

                pose_samples: list[dict[str, Any]] = []
                events: list[dict[str, Any]] = [
                    {"time_s": 0.0, "type": "rollout_status", "status": "accepted", "controller": "goal_proportional"}
                ]

                def append_pose(time_s: float, current: Any) -> None:
                    achieved = np.asarray(current["achieved_goal"], dtype=np.float64)
                    desired = np.asarray(current["desired_goal"], dtype=np.float64)
                    if achieved.shape != (3,) or desired.shape != (3,):
                        raise AdapterError("goal observation dimensionality changed during capture")
                    pose_samples.append(
                        {
                            "time_s": float(time_s),
                            "entities": {
                                "end_effector": {
                                    "position_m": achieved.tolist(),
                                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                                },
                                "goal": {
                                    "position_m": desired.tolist(),
                                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                                },
                            },
                        }
                    )

                append_pose(0.0, observation)
                steps = 0
                for step in range(max_steps):
                    action = self.goal_action(observation, controller_gain)
                    # Outcome-bearing returns are intentionally ignored in the capture path.
                    observation, _ignored_reward, terminated, truncated, _ignored_info = environment.step(action)
                    steps = step + 1
                    time_s = steps * timestep_s
                    append_pose(time_s, observation)
                    events.append(
                        {
                            "time_s": time_s,
                            "type": "command",
                            "controller": "goal_proportional",
                            "action": [float(value) for value in action],
                        }
                    )
                    if bool(terminated):
                        raise AdapterError(
                            "live capture refuses environments with outcome-dependent early termination; "
                            "use an immutable raw export or a fixed-horizon environment"
                        )
                    if bool(truncated):
                        break
                end_s = steps * timestep_s
                events.append(
                    {
                        "time_s": end_s,
                        "type": "rollout_status",
                        "status": "completed",
                        "steps": steps,
                    }
                )
                package_versions = {
                    "gymnasium": metadata.version("gymnasium"),
                    "gymnasium_robotics": metadata.version("gymnasium-robotics"),
                    "mujoco": metadata.version("mujoco"),
                }
                trace = {
                    "schema_version": "robot-spatial-generic-trace.v1",
                    "run_id": f"gymnasium-robotics/{env_id}/seed-{int(seed)}",
                    "simulator": {"name": "Gymnasium Robotics/MuJoCo", "version": package_versions["mujoco"]},
                    "seed": int(seed),
                    "timestep_s": timestep_s,
                    "clock": {"clock_id": "simulation/episode", "domain": "simulated_monotonic"},
                    "interval": {"start_s": 0.0, "end_s": end_s},
                    "task_id": env_id,
                    "intervention": {
                        "type": "action",
                        "controller": "goal_proportional",
                        "controller_gain": float(controller_gain),
                        "max_steps": int(max_steps),
                    },
                    "robot": {
                        "robot_id": "fetch",
                        "root_frame": "world",
                        "model_sha256": model_digest,
                    },
                    "world": {"world_id": env_id, "world_sha256": model_digest},
                    "conventions": {
                        "length_unit": "m",
                        "angle_unit": "rad",
                        "quaternion_order": "xyzw",
                        "pose_direction": "world_from_entity",
                    },
                    "channel_policies": {"pose": {"max_gap_s": timestep_s * 1.5}},
                    "samples": {"pose": pose_samples},
                    "events": events,
                    "assets": [
                        {
                            "kind": "compiled_mjcf",
                            "sha256": model_digest,
                            "redistributed": False,
                            "package_versions": package_versions,
                        }
                    ],
                }
                trace_path = temporary_root / "trace.json"
                write_json(trace_path, trace)
                return SimulationRun.import_generic_trace(
                    trace_path,
                    out,
                    adapter_name=self.name,
                    adapter_version=self.version,
                )
        finally:
            environment.close()


class GazeboRos2Adapter(DeclaredSimulatorExportAdapter):
    """Offline Gazebo/ROS 2 export adapter; live topic capture remains an optional integration."""

    name = "gazebo-ros2"
    accepted_simulators = ("gazebo", "gz sim", "ignition")


class DeformableJsonAdapter(DeclaredSimulatorExportAdapter):
    """Offline deformable keypoint/mesh-state adapter for OmniGibson-like exports."""

    name = "deformable-json"
    accepted_simulators = ("omnigibson", "behavior", "isaac", "sapien")


BUILTIN_ADAPTERS: dict[str, type[SimulatorAdapter]] = {
    adapter.name: adapter
    for adapter in (
        GenericJsonAdapter,
        ManiSkillAdapter,
        MuJoCoAdapter,
        GymnasiumRoboticsAdapter,
        GazeboRos2Adapter,
        DeformableJsonAdapter,
    )
}


def available_adapters() -> list[str]:
    names = set(BUILTIN_ADAPTERS)
    try:
        points = metadata.entry_points()
        selected = points.select(group="robot_spatial.adapters") if hasattr(points, "select") else points.get("robot_spatial.adapters", [])
        names.update(point.name for point in selected)
    except Exception:
        pass
    return sorted(names)


def adapter_for(name: str) -> SimulatorAdapter:
    if name in BUILTIN_ADAPTERS:
        return BUILTIN_ADAPTERS[name]()
    try:
        points = metadata.entry_points()
        selected = points.select(group="robot_spatial.adapters", name=name) if hasattr(points, "select") else [
            point for point in points.get("robot_spatial.adapters", []) if point.name == name
        ]
        for point in selected:
            adapter_type: Any = point.load()
            adapter = adapter_type()
            if not isinstance(adapter, SimulatorAdapter):
                raise AdapterError(f"adapter entry point {name!r} does not implement SimulatorAdapter")
            return adapter
    except AdapterError:
        raise
    except Exception as error:
        raise AdapterError(f"failed to load adapter {name!r}: {error}") from error
    raise AdapterError(f"unknown adapter {name!r}; available adapters: {available_adapters()}")
