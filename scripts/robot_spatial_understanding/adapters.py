"""Simulator adapters that normalize raw exports without importing task outcomes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata
import inspect
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .errors import AdapterError
from .simulation import SimulationRun
from .util import (
    ensure_new_directory,
    load_json,
    load_structured,
    require_list,
    require_mapping,
    require_string,
    sha256_file,
    sha256_json,
    write_json,
)


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
    version = "0.3.0"
    accepted_simulators = ("maniskill", "sapien")

    entity_map_schema = "robot-spatial-maniskill-entity-map.v1"
    entity_map_schema_v2 = "robot-spatial-maniskill-entity-map.v2"

    @staticmethod
    def _numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return value

    @staticmethod
    def _pose(raw_pose: Any, index: int) -> dict[str, list[float]]:
        values = ManiSkillAdapter._numpy(raw_pose)[index]
        return {
            "position_m": [float(value) for value in values[:3]],
            # SAPIEN stores quaternion components as wxyz; simulation-run.v1 uses xyzw.
            "quaternion_xyzw": [float(values[4]), float(values[5]), float(values[6]), float(values[3])],
        }

    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        """Resolve a declared public attribute path without evaluating code."""

        value = root
        for segment in path.split("."):
            if not segment or segment.startswith("_") or not segment.isidentifier():
                raise AdapterError(f"invalid ManiSkill source path {path!r}")
            if not hasattr(value, segment):
                raise AdapterError(f"ManiSkill source path {path!r} has no attribute {segment!r}")
            value = getattr(value, segment)
        return value

    @classmethod
    def _entity_specs(cls, mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Normalize legacy PickCube roles and declarative v2 entity specs."""

        if mapping["schema_version"] == cls.entity_map_schema:
            legacy_paths = {
                "tcp": "agent.tcp",
                "left_finger": "agent.finger1_link",
                "right_finger": "agent.finger2_link",
                "cube": "cube",
                "goal": "goal_site",
            }
            return {
                role: {
                    "source_path": legacy_paths[role],
                    "expected_name": source,
                    "capture_velocity": False,
                }
                for role, source in mapping["entities"].items()
            }
        return {str(role): dict(spec) for role, spec in mapping["entities"].items()}

    @classmethod
    def _contact_pairs(cls, mapping: dict[str, Any]) -> list[tuple[str, str]]:
        if mapping["schema_version"] == cls.entity_map_schema:
            return [("left_finger", "cube"), ("right_finger", "cube")]
        return [tuple(str(role) for role in pair) for pair in mapping.get("contact_pairs", [])]

    @classmethod
    def _resolve_entities(cls, raw: Any, mapping: dict[str, Any]) -> dict[str, Any]:
        objects: dict[str, Any] = {}
        for role, spec in cls._entity_specs(mapping).items():
            obj = cls._resolve_path(raw, str(spec["source_path"]))
            expected_name = spec.get("expected_name")
            if expected_name is not None:
                actual_name = getattr(obj, "name", None)
                if actual_name != expected_name:
                    raise AdapterError(
                        f"entity map role {role!r} expected simulator object {expected_name!r}, got {actual_name!r}"
                    )
            objects[role] = obj
        return objects

    @classmethod
    def _entity_pose(cls, obj: Any, spec: dict[str, Any], index: int) -> dict[str, Any]:
        pose = getattr(obj, "pose", obj)
        raw_pose = getattr(pose, "raw_pose", None)
        if raw_pose is None:
            raise AdapterError(f"declared ManiSkill entity at {spec['source_path']!r} does not expose a pose")
        record: dict[str, Any] = cls._pose(raw_pose, index)
        if bool(spec.get("capture_velocity", False)):
            for source_name, output_name in (
                ("linear_velocity", "linear_velocity_mps"),
                ("angular_velocity", "angular_velocity_radps"),
            ):
                if not hasattr(obj, source_name):
                    raise AdapterError(
                        f"entity {spec['source_path']!r} requested {output_name} but exposes no {source_name}"
                    )
                values = cls._numpy(getattr(obj, source_name))[index]
                record[output_name] = [float(value) for value in values]
        return record

    @classmethod
    def _measurements(cls, raw: Any, mapping: dict[str, Any], index: int) -> dict[str, float]:
        measurements: dict[str, float] = {}
        for measurement_id, raw_spec in mapping.get("measurements", {}).items():
            spec = require_mapping(raw_spec, f"entity_map.measurements.{measurement_id}")
            values = cls._numpy(cls._resolve_path(raw, str(spec["source_path"])))
            value = values if getattr(values, "ndim", 0) == 0 else values[index]
            component = spec.get("component")
            if component is not None:
                value = value[int(component)]
            try:
                scalar = float(value)
            except (TypeError, ValueError) as error:
                raise AdapterError(f"measurement {measurement_id!r} did not resolve to one scalar") from error
            if not math.isfinite(scalar):
                raise AdapterError(f"measurement {measurement_id!r} is not finite")
            measurements[str(measurement_id)] = scalar
        return measurements

    @classmethod
    def _state_snapshot(
        cls,
        raw: Any,
        mapping: dict[str, Any],
        index: int,
        joint_map: dict[str, str],
        source_joint_index: dict[str, int],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        qpos = cls._numpy(raw.agent.robot.get_qpos())
        qvel = cls._numpy(raw.agent.robot.get_qvel())
        joint_state = {
            "positions": {
                output_id: float(qpos[index, source_joint_index[source_id]])
                for output_id, source_id in joint_map.items()
            },
            "velocities": {
                output_id: float(qvel[index, source_joint_index[source_id]])
                for output_id, source_id in joint_map.items()
            },
        }
        objects = cls._resolve_entities(raw, mapping)
        specs = cls._entity_specs(mapping)
        poses = {role: cls._entity_pose(obj, specs[role], index) for role, obj in objects.items()}
        return joint_state, poses, cls._measurements(raw, mapping, index)

    @staticmethod
    def _canonical_body_name(name: str, aliases: dict[str, str]) -> str:
        source = re.sub(r"^scene-\d+_", "", name)
        return aliases.get(source, source)

    @classmethod
    def _load_entity_map(cls, path: str | Path, env_id: str) -> tuple[dict[str, Any], str]:
        data = require_mapping(load_structured(Path(path)), "ManiSkill entity map")
        schema_version = data.get("schema_version")
        if schema_version not in {cls.entity_map_schema, cls.entity_map_schema_v2}:
            raise AdapterError(
                f"ManiSkill entity map schema must be {cls.entity_map_schema!r} or {cls.entity_map_schema_v2!r}"
            )
        if require_string(data.get("env_id"), "entity_map.env_id") != env_id:
            raise AdapterError(f"entity map env_id does not match requested environment {env_id!r}")
        require_string(data.get("control_mode"), "entity_map.control_mode")
        joints = require_mapping(data.get("joints"), "entity_map.joints")
        entities = require_mapping(data.get("entities"), "entity_map.entities")
        for output_id, source_id in joints.items():
            require_string(str(output_id), "entity map output ID")
            require_string(source_id, f"entity map source for {output_id}")
        if not joints:
            raise AdapterError("entity_map.joints must not be empty")
        if schema_version == cls.entity_map_schema:
            required_entities = {"tcp", "left_finger", "right_finger", "cube", "goal"}
            if set(entities) != required_entities:
                raise AdapterError(f"entity_map.entities must contain exactly {sorted(required_entities)}")
            for output_id, source_id in entities.items():
                require_string(str(output_id), "entity map output ID")
                require_string(source_id, f"entity map source for {output_id}")
        else:
            require_string(data.get("robot_uid"), "entity_map.robot_uid")
            if not entities:
                raise AdapterError("entity_map.entities must not be empty")
            for role, raw_spec in entities.items():
                require_string(str(role), "entity map role")
                spec = require_mapping(raw_spec, f"entity_map.entities.{role}")
                require_string(spec.get("source_path"), f"entity_map.entities.{role}.source_path")
                if "expected_name" in spec:
                    require_string(spec["expected_name"], f"entity_map.entities.{role}.expected_name")
                if "capture_velocity" in spec and not isinstance(spec["capture_velocity"], bool):
                    raise AdapterError(f"entity_map.entities.{role}.capture_velocity must be boolean")
            for index, raw_pair in enumerate(require_list(data.get("contact_pairs", []), "entity_map.contact_pairs")):
                pair = require_list(raw_pair, f"entity_map.contact_pairs[{index}]")
                if len(pair) != 2:
                    raise AdapterError(f"entity_map.contact_pairs[{index}] must name two declared entity roles")
                for role_index, role in enumerate(pair):
                    require_string(role, f"entity_map.contact_pairs[{index}][{role_index}]")
                    if role not in entities:
                        raise AdapterError(f"entity_map.contact_pairs[{index}] names undeclared role {role!r}")
            measurements = require_mapping(data.get("measurements", {}), "entity_map.measurements")
            for measurement_id, raw_spec in measurements.items():
                require_string(str(measurement_id), "measurement ID")
                spec = require_mapping(raw_spec, f"entity_map.measurements.{measurement_id}")
                require_string(spec.get("source_path"), f"entity_map.measurements.{measurement_id}.source_path")
                if "component" in spec and (
                    not isinstance(spec["component"], int) or isinstance(spec["component"], bool) or spec["component"] < 0
                ):
                    raise AdapterError(f"entity_map.measurements.{measurement_id}.component must be non-negative integer")
        aliases = require_mapping(data.get("collision_aliases", {}), "entity_map.collision_aliases")
        for source_id, output_id in aliases.items():
            require_string(str(source_id), "collision alias source")
            require_string(output_id, f"collision alias {source_id}")
        return data, sha256_json(data)

    @staticmethod
    def _load_actions(path: str | Path, trajectory_index: int) -> tuple[Any, str]:
        try:
            import h5py
            import numpy as np
        except ImportError as error:
            raise AdapterError("live ManiSkill capture requires h5py and NumPy from the 'maniskill' extra") from error
        trajectory_path = Path(path)
        if not trajectory_path.is_file():
            raise AdapterError(f"ManiSkill trajectory does not exist: {trajectory_path}")
        if trajectory_index < 0:
            raise AdapterError("trajectory_index must be non-negative")
        dataset_path = f"traj_{trajectory_index}/actions"
        try:
            with h5py.File(trajectory_path, "r") as archive:
                # Intentionally open only the action dataset. Reward, success, info, observations,
                # and environment-state datasets are neither read nor copied into prediction input.
                actions = np.asarray(archive[dataset_path], dtype=np.float32)
        except (OSError, KeyError, ValueError) as error:
            raise AdapterError(f"cannot read action dataset {dataset_path!r} from {trajectory_path}: {error}") from error
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise AdapterError("ManiSkill action trajectory must have shape [steps, action_dimension]")
        if not np.all(np.isfinite(actions)):
            raise AdapterError("ManiSkill action trajectory contains non-finite values")
        return actions, sha256_file(trajectory_path)

    def capture_episode(
        self,
        out: str | Path,
        *,
        env_id: str,
        seed: int,
        trajectory: str | Path,
        entity_map: str | Path,
        sim_backend: str = "physx_cpu",
        num_envs: int = 1,
        fixed_horizon: int = 100,
        trajectory_index: int = 0,
        render_backend: str = "gpu",
        initialization: str | None = None,
    ) -> SimulationRun | list[SimulationRun]:
        """Replay action-only input and capture raw ManiSkill state without reading outcome labels."""

        if sim_backend not in {"physx_cpu", "physx_cuda"}:
            raise AdapterError("ManiSkill sim_backend must be physx_cpu or physx_cuda")
        if num_envs <= 0 or fixed_horizon <= 0:
            raise AdapterError("num_envs and fixed_horizon must be positive")
        if initialization not in {None, "goal_at_cube"}:
            raise AdapterError("ManiSkill initialization must be omitted or goal_at_cube")
        output = Path(out)
        if output.exists():
            raise AdapterError(f"output path already exists: {output}")
        mapping, mapping_digest = self._load_entity_map(entity_map, env_id)
        actions, trajectory_digest = self._load_actions(trajectory, trajectory_index)

        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401 - registers environments
            import numpy as np
            import sapien
            import torch
        except ImportError as error:
            raise AdapterError(
                "live ManiSkill capture requires the pinned simulator environment: pip install 'mani-skill==3.0.1'"
            ) from error

        control_mode = str(mapping["control_mode"])
        try:
            environment_options: dict[str, Any] = {
                "num_envs": num_envs,
                "obs_mode": "state",
                "control_mode": control_mode,
                "render_mode": None,
                "render_backend": render_backend,
                "sim_backend": sim_backend,
            }
            if mapping["schema_version"] == self.entity_map_schema_v2:
                environment_options["robot_uids"] = str(mapping["robot_uid"])
            environment = gym.make(env_id, **environment_options)
        except Exception as error:
            raise AdapterError(f"failed to create ManiSkill environment {env_id!r}: {error}") from error

        traces: list[dict[str, Any]] = []
        try:
            reset_seeds: int | list[int] = int(seed) if num_envs == 1 else [int(seed) + index for index in range(num_envs)]
            environment.reset(seed=reset_seeds)
            raw = environment.unwrapped
            if initialization == "goal_at_cube":
                objects = self._resolve_entities(raw, mapping)
                if "goal" not in objects or "cube" not in objects or not hasattr(objects["goal"], "set_pose"):
                    raise AdapterError("goal_at_cube initialization requires declared cube and mutable goal entities")
                objects["goal"].set_pose(getattr(objects["cube"], "pose"))
            active_joints = [joint.name for joint in raw.agent.robot.get_active_joints()]
            joint_map = {str(output_id): str(source_id) for output_id, source_id in mapping["joints"].items()}
            if set(joint_map.values()) != set(active_joints):
                raise AdapterError(
                    "entity map must cover every active ManiSkill joint exactly; "
                    f"expected {sorted(active_joints)}, got {sorted(joint_map.values())}"
                )
            source_joint_index = {name: index for index, name in enumerate(active_joints)}
            entity_specs = self._entity_specs(mapping)
            simulator_objects = self._resolve_entities(raw, mapping)
            contact_pairs = self._contact_pairs(mapping)
            aliases = {
                str(obj.name): role
                for role, obj in simulator_objects.items()
                if getattr(obj, "name", None) is not None
            }
            aliases.update({str(source): str(target) for source, target in mapping.get("collision_aliases", {}).items()})

            action_dimension = int(environment.action_space.shape[-1])
            if int(actions.shape[1]) != action_dimension:
                raise AdapterError(
                    f"trajectory action dimension {actions.shape[1]} does not match environment dimension {action_dimension}"
                )
            timestep_s = float(raw.control_timestep)
            if not math.isfinite(timestep_s) or timestep_s <= 0.0:
                raise AdapterError("ManiSkill environment reported an invalid control timestep")
            urdf_path = Path(raw.agent.urdf_path)
            model_digest = sha256_file(urdf_path)
            task_source_path = Path(inspect.getfile(type(raw)))
            task_source_digest = sha256_file(task_source_path)
            versions = {
                "mani_skill": metadata.version("mani-skill"),
                "sapien": metadata.version("sapien"),
                "torch": torch.__version__,
                "gymnasium": metadata.version("gymnasium"),
                "numpy": np.__version__,
            }
            config_digest = sha256_json(
                {
                    "env_id": env_id,
                    "control_mode": control_mode,
                    "sim_backend": sim_backend,
                    "render_backend": render_backend,
                    "num_envs": num_envs,
                    "fixed_horizon": fixed_horizon,
                    "entity_map_sha256": mapping_digest,
                    "initialization": initialization,
                    "robot_uid": str(mapping.get("robot_uid", raw.robot_uids)),
                }
            )
            collision_available = sim_backend == "physx_cpu" and num_envs == 1
            for index in range(num_envs):
                run_seed = int(seed) + index
                measurements = self._measurements(raw, mapping, index)
                traces.append(
                    {
                        "schema_version": "robot-spatial-generic-trace.v1",
                        "run_id": f"maniskill/{env_id}/seed-{run_seed}/{sim_backend}",
                        "simulator": {
                            "name": "ManiSkill/SAPIEN",
                            "version": versions["mani_skill"],
                            "runtime_versions": versions,
                        },
                        "seed": run_seed,
                        "timestep_s": timestep_s,
                        "clock": {"clock_id": f"simulation/{env_id}/seed-{run_seed}", "domain": "simulated_monotonic"},
                        "interval": {"start_s": 0.0, "end_s": fixed_horizon * timestep_s},
                        "task_id": env_id,
                        "intervention": {
                            "type": "action",
                            "source": "action_only_hdf5",
                            "trajectory_sha256": trajectory_digest,
                            "trajectory_index": trajectory_index,
                            "action_count": int(actions.shape[0]),
                            "fixed_horizon": fixed_horizon,
                            "padding": "repeat_final_action",
                            "initialization": initialization,
                        },
                        "robot": {
                            "robot_id": str(mapping.get("robot_uid", raw.robot_uids)),
                            "root_frame": "world",
                            "model_sha256": model_digest,
                            "active_joint_ids": sorted(joint_map),
                        },
                        "world": {
                            "world_id": f"{env_id}/seed-{run_seed}",
                            "task_source_sha256": task_source_digest,
                            "config_sha256": config_digest,
                            "entity_map_sha256": mapping_digest,
                            **(
                                {"measurements": measurements}
                                if mapping["schema_version"] == self.entity_map_schema_v2
                                else {}
                            ),
                        },
                        "conventions": {
                            "length_unit": "m",
                            "angle_unit": "rad",
                            "quaternion_order": "xyzw",
                            "pose_direction": "world_from_entity",
                        },
                        "channel_policies": {
                            "joint_state": {"max_gap_s": timestep_s * 1.5},
                            "pose": {"max_gap_s": timestep_s * 1.5},
                            "contact": {"max_gap_s": timestep_s * 1.5},
                            **({"collision": {"max_gap_s": timestep_s * 1.5}} if collision_available else {}),
                        },
                        "samples": {
                            "joint_state": [],
                            "pose": [],
                            "contact": [],
                            **({"collision": []} if collision_available else {}),
                        },
                        "events": [
                            {"time_s": 0.0, "type": "reset", "seed": run_seed},
                            {"time_s": 0.0, "type": "rollout_status", "status": "accepted"},
                        ],
                        "assets": [
                            {"kind": "robot_urdf", "name": urdf_path.name, "sha256": model_digest, "redistributed": False},
                            {"kind": "task_source", "name": task_source_path.name, "sha256": task_source_digest, "redistributed": False},
                            {"kind": "entity_map", "name": Path(entity_map).name, "sha256": mapping_digest, "redistributed": True},
                            {"kind": "action_trajectory", "name": Path(trajectory).name, "sha256": trajectory_digest, "redistributed": False},
                        ],
                        "capture": {
                            "sim_backend": sim_backend,
                            "render_backend": render_backend,
                            "num_envs": num_envs,
                            "sub_environment_index": index,
                            "versions": versions,
                            "raw_state_source": "env.unwrapped",
                            "reward_or_success_read": False,
                            "entity_mapping": "declarative_v2" if mapping["schema_version"] == self.entity_map_schema_v2 else "legacy_pickcube_v1",
                            "contact_pairs": [list(pair) for pair in contact_pairs],
                            "collision_enumeration": "complete_cpu_scene_contacts" if collision_available else "unavailable_on_gpu_backend",
                        },
                    }
                )

            previous_contacts: list[dict[str, bool] | None] = [None for _ in range(num_envs)]

            def capture_sample(time_s: float) -> None:
                qpos = np.asarray(self._numpy(raw.agent.robot.get_qpos()), dtype=np.float64)
                qvel = np.asarray(self._numpy(raw.agent.robot.get_qvel()), dtype=np.float64)
                # Re-resolve every declared pose source. Actor handles are stable, but computed
                # properties such as PegInsertionSide.peg_head_pose return one-time Pose values.
                current_objects = self._resolve_entities(raw, mapping)
                pair_vectors = {
                    pair: np.asarray(
                        self._numpy(
                            raw.scene.get_pairwise_contact_forces(
                                simulator_objects[pair[0]], simulator_objects[pair[1]]
                            )
                        ),
                        dtype=np.float64,
                    )
                    for pair in contact_pairs
                }
                collision_pairs: list[tuple[str, str]] = []
                if collision_available:
                    for contact in raw.scene.get_contacts():
                        body_a = self._canonical_body_name(contact.bodies[0].entity.name, aliases)
                        body_b = self._canonical_body_name(contact.bodies[1].entity.name, aliases)
                        collision_pairs.append(tuple(sorted((body_a, body_b))))
                    collision_pairs = sorted(set(collision_pairs))
                for env_index, trace in enumerate(traces):
                    positions = {
                        output_id: float(qpos[env_index, source_joint_index[source_id]])
                        for output_id, source_id in joint_map.items()
                    }
                    velocities = {
                        output_id: float(qvel[env_index, source_joint_index[source_id]])
                        for output_id, source_id in joint_map.items()
                    }
                    trace["samples"]["joint_state"].append(
                        {"time_s": time_s, "positions": positions, "velocities": velocities}
                    )
                    trace["samples"]["pose"].append(
                        {
                            "time_s": time_s,
                            "entities": {
                                role: self._entity_pose(actor, entity_specs[role], env_index)
                                for role, actor in current_objects.items()
                            },
                        }
                    )
                    current_contacts: dict[str, bool] = {}
                    for pair in contact_pairs:
                        label = "|".join(pair)
                        vector = pair_vectors[pair][env_index]
                        force = float(np.linalg.norm(vector))
                        active = force > 1e-12
                        current_contacts[label] = active
                        trace["samples"]["contact"].append(
                            {
                                "time_s": time_s,
                                "body_a": pair[0],
                                "body_b": pair[1],
                                "active": active,
                                "force_n": [float(value) for value in vector],
                                "normal_force_n": force,
                            }
                        )
                        previous = previous_contacts[env_index]
                        if previous is not None and previous[label] != active:
                            trace["events"].append(
                                {
                                    "time_s": time_s,
                                    "type": "contact_begin" if active else "contact_end",
                                    "body_a": pair[0],
                                    "body_b": pair[1],
                                }
                            )
                    previous_contacts[env_index] = current_contacts
                    if collision_available:
                        trace["samples"]["collision"].append(
                            {
                                "time_s": time_s,
                                "body_a": "collision_snapshot",
                                "body_b": "no_unlisted_pair",
                                "active": False,
                            }
                        )
                        for body_a, body_b in collision_pairs:
                            trace["samples"]["collision"].append(
                                {"time_s": time_s, "body_a": body_a, "body_b": body_b, "active": True}
                            )

            capture_sample(0.0)
            for step in range(fixed_horizon):
                source_index = min(step, int(actions.shape[0]) - 1)
                source_action = actions[source_index]
                applied_action = source_action if num_envs == 1 else np.broadcast_to(source_action, (num_envs, action_dimension)).copy()
                # The returned observation, reward, termination flags, and info are intentionally discarded.
                environment.step(applied_action)
                time_s = (step + 1) * timestep_s
                capture_sample(time_s)
                for env_index, trace in enumerate(traces):
                    trace["events"].extend(
                        [
                            {
                                "time_s": time_s,
                                "type": "command",
                                "step": step,
                                "trajectory_action_index": source_index,
                                "action": [float(value) for value in source_action],
                            },
                            {"time_s": time_s, "type": "step", "step": step + 1},
                        ]
                    )
            for trace in traces:
                initial_pose = trace["samples"]["pose"][0]["entities"]
                initial_state = {
                    "seed": trace["seed"],
                    "joint_state": trace["samples"]["joint_state"][0],
                    "poses": initial_pose,
                }
                if mapping["schema_version"] == self.entity_map_schema_v2:
                    initial_state["measurements"] = trace["world"]["measurements"]
                trace["world"]["initial_state_sha256"] = sha256_json(initial_state)
                trace["world"]["world_sha256"] = sha256_json(trace["world"])
                trace["events"].append(
                    {"time_s": fixed_horizon * timestep_s, "type": "rollout_status", "status": "completed", "steps": fixed_horizon}
                )
        finally:
            environment.close()

        with tempfile.TemporaryDirectory(prefix="robot-spatial-maniskill-") as temporary:
            temporary_root = Path(temporary)
            if num_envs == 1:
                trace_path = temporary_root / "trace.json"
                write_json(trace_path, traces[0])
                return SimulationRun.import_generic_trace(
                    trace_path,
                    output,
                    adapter_name=self.name,
                    adapter_version=self.version,
                )
            ensure_new_directory(output)
            runs: list[SimulationRun] = []
            try:
                for index, trace in enumerate(traces):
                    trace_path = temporary_root / f"trace-{index:03d}.json"
                    write_json(trace_path, trace)
                    runs.append(
                        SimulationRun.import_generic_trace(
                            trace_path,
                            output / f"env-{index:03d}",
                            adapter_name=self.name,
                            adapter_version=self.version,
                        )
                    )
                return runs
            except Exception:
                shutil.rmtree(output, ignore_errors=True)
                raise


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
