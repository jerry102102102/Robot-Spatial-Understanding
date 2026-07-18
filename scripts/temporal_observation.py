#!/usr/bin/env python3
"""Timestamped observations resolved against one URDF and one declared world scene.

This layer is intentionally separate from both inputs: URDF is the mechanism model,
the world scene is a static declaration, and this log records what a source reported
at particular times.  Version 1 selects the latest sample at or before query time
(zero-order hold); it never interpolates or consumes future samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from world_scene import Matrix, WorldScene, _matmul, _pose, pose_record


LOG_SCHEMA = "robot-spatial-observation-log.v1"
LOG_SCHEMA_V2 = "robot-spatial-observation-log.v2"
LOG_SCHEMAS = {LOG_SCHEMA, LOG_SCHEMA_V2}
ROS_NORMALIZATION_SCHEMA = "robot-spatial-ros-normalization-provenance.v1"
QUERY_SCHEMA = "robot-spatial-observation-query.v1"
RESOLVED_SCHEMA = "robot-spatial-resolved-observation.v1"
SOURCE_TYPES = {"measured", "synthetic", "imported", "unknown"}
FALLBACKS = {"require_observed", "allow_static_declaration"}


class ObservationError(ValueError):
    """An invalid, ambiguous, stale, or incorrectly bound observation input."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservationError(f"{label} must be an object")
    return value


def _known(value: dict[str, Any], fields: set[str], label: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ObservationError(f"{label} has unsupported fields {unknown}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationError(f"{label} must be a non-empty string")
    if "/" in value:
        raise ObservationError(f"{label} must not contain '/'")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationError(f"{label} must be a non-empty string")
    return value


def _timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObservationError(f"{label} must be a non-negative integer nanosecond timestamp")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ObservationError(f"{label} must be a finite number")
    return result


def _source(value: Any, label: str) -> dict[str, Any]:
    source = _object(value, label)
    _known(source, {"type", "reference", "sensor_id", "topic"}, label)
    source_type = source.get("type")
    if source_type not in SOURCE_TYPES:
        raise ObservationError(f"{label}.type must be one of {sorted(SOURCE_TYPES)}")
    result: dict[str, Any] = {"type": source_type}
    for field in ("reference", "sensor_id", "topic"):
        item = source.get(field)
        if item is not None:
            result[field] = _string(item, f"{label}.{field}")
        else:
            result[field] = None
    result["meaning"] = "provenance asserted by the log producer; source identity and calibration are not independently verified"
    return result


def _covariance(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    covariance = _object(value, label)
    _known(covariance, {"order", "matrix_6x6_rowmajor"}, label)
    if covariance.get("order") != "xyz_m_then_rotation_vector_rad":
        raise ObservationError(
            f"{label}.order must be 'xyz_m_then_rotation_vector_rad'"
        )
    matrix = covariance.get("matrix_6x6_rowmajor")
    if not isinstance(matrix, list) or len(matrix) != 36:
        raise ObservationError(f"{label}.matrix_6x6_rowmajor must contain 36 finite numbers")
    values = [_finite(item, f"{label}.matrix_6x6_rowmajor[{index}]") for index, item in enumerate(matrix)]
    for index in range(6):
        if values[index * 6 + index] < 0.0:
            raise ObservationError(f"{label} diagonal variances must be non-negative")
        for other in range(6):
            if abs(values[index * 6 + other] - values[other * 6 + index]) > 1e-9:
                raise ObservationError(f"{label} must be symmetric within 1e-9")
    return {
        "order": covariance["order"],
        "matrix_6x6_rowmajor": values,
        "meaning": "reported covariance is preserved as probabilistic metadata, not treated as a hard geometric bound",
    }


def _sample_header(sample: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "sample_id": _identifier(sample.get("sample_id"), f"{label}.sample_id"),
        "timestamp_ns": _timestamp(sample.get("timestamp_ns"), f"{label}.timestamp_ns"),
        "source": _source(sample.get("source"), f"{label}.source"),
    }


def _ros_normalization(value: Any, label: str) -> dict[str, Any]:
    normalization = _object(value, label)
    _known(
        normalization,
        {
            "schema_version",
            "adapter_id",
            "config_sha256",
            "capture_id",
            "capture_sha256",
            "method",
            "clock_policy",
            "authority_policy",
            "tf_policy",
        },
        label,
    )
    if normalization.get("schema_version") != ROS_NORMALIZATION_SCHEMA:
        raise ObservationError(f"{label}.schema_version must be {ROS_NORMALIZATION_SCHEMA}")
    clock = _object(normalization.get("clock_policy"), f"{label}.clock_policy")
    _known(clock, {"timestamp_source", "synchronization_verified"}, f"{label}.clock_policy")
    if not isinstance(clock.get("synchronization_verified"), bool):
        raise ObservationError(f"{label}.clock_policy.synchronization_verified must be boolean")
    authority = _object(normalization.get("authority_policy"), f"{label}.authority_policy")
    _known(authority, {"joint", "tf", "publisher_identity_visibility"}, f"{label}.authority_policy")
    tf_policy = _object(normalization.get("tf_policy"), f"{label}.tf_policy")
    _known(
        tf_policy,
        {"maximum_dynamic_edge_age_ns", "interpolation", "extrapolation", "future_samples_consumed"},
        f"{label}.tf_policy",
    )
    maximum_age = _timestamp(
        tf_policy.get("maximum_dynamic_edge_age_ns"),
        f"{label}.tf_policy.maximum_dynamic_edge_age_ns",
    )
    flags: dict[str, bool] = {}
    for field in ("interpolation", "extrapolation", "future_samples_consumed"):
        item = tf_policy.get(field)
        if not isinstance(item, bool):
            raise ObservationError(f"{label}.tf_policy.{field} must be boolean")
        flags[field] = item
    return {
        "schema_version": ROS_NORMALIZATION_SCHEMA,
        "adapter_id": _identifier(normalization.get("adapter_id"), f"{label}.adapter_id"),
        "config_sha256": _string(normalization.get("config_sha256"), f"{label}.config_sha256"),
        "capture_id": _identifier(normalization.get("capture_id"), f"{label}.capture_id"),
        "capture_sha256": _string(normalization.get("capture_sha256"), f"{label}.capture_sha256"),
        "method": _string(normalization.get("method"), f"{label}.method"),
        "clock_policy": {
            "timestamp_source": _string(
                clock.get("timestamp_source"), f"{label}.clock_policy.timestamp_source"
            ),
            "synchronization_verified": clock["synchronization_verified"],
        },
        "authority_policy": {
            "joint": _string(authority.get("joint"), f"{label}.authority_policy.joint"),
            "tf": _string(authority.get("tf"), f"{label}.authority_policy.tf"),
            "publisher_identity_visibility": _string(
                authority.get("publisher_identity_visibility"),
                f"{label}.authority_policy.publisher_identity_visibility",
            ),
        },
        "tf_policy": {"maximum_dynamic_edge_age_ns": maximum_age, **flags},
        "meaning": "digest-bound ROS transport normalization provenance; it does not independently verify clock synchronization, publisher truth, calibration, or physical completeness",
    }


def _joint_sample(value: Any, label: str) -> dict[str, Any]:
    sample = _object(value, label)
    _known(sample, {"sample_id", "timestamp_ns", "positions", "position_standard_deviation", "source"}, label)
    positions = _object(sample.get("positions"), f"{label}.positions")
    if not positions:
        raise ObservationError(f"{label}.positions must not be empty")
    converted = {
        _identifier(name, f"{label}.positions key"): _finite(item, f"{label}.positions.{name}")
        for name, item in positions.items()
    }
    deviation_value = sample.get("position_standard_deviation", {})
    deviation = _object(deviation_value, f"{label}.position_standard_deviation")
    if sorted(set(deviation) - set(converted)):
        raise ObservationError(f"{label}.position_standard_deviation may only name observed positions")
    converted_deviation = {
        name: _finite(item, f"{label}.position_standard_deviation.{name}")
        for name, item in deviation.items()
    }
    if any(item < 0.0 for item in converted_deviation.values()):
        raise ObservationError(f"{label}.position_standard_deviation values must be non-negative")
    return {
        **_sample_header(sample, label),
        "positions": converted,
        "position_standard_deviation": converted_deviation,
    }


def _pose_sample(value: Any, label: str) -> dict[str, Any]:
    sample = _object(value, label)
    _known(sample, {"sample_id", "timestamp_ns", "parent_scene_frame", "pose", "covariance", "source"}, label)
    try:
        canonical_pose, transform = _pose(sample.get("pose"), f"{label}.pose")
    except ValueError as error:
        raise ObservationError(str(error)) from error
    return {
        **_sample_header(sample, label),
        "parent_scene_frame": _identifier(sample.get("parent_scene_frame"), f"{label}.parent_scene_frame"),
        "pose_in_parent": canonical_pose,
        "parent_from_entity": transform,
        "covariance": _covariance(sample.get("covariance"), f"{label}.covariance"),
    }


def _stream(value: Any, label: str, parser: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ObservationError(f"{label} must be an array")
    records = [parser(item, f"{label}[{index}]") for index, item in enumerate(value)]
    sample_ids = [record["sample_id"] for record in records]
    timestamps = [record["timestamp_ns"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ObservationError(f"{label} sample_id values must be unique")
    if len(timestamps) != len(set(timestamps)):
        raise ObservationError(f"{label} timestamps must be unique to avoid ambiguous selection")
    return sorted(records, key=lambda record: record["timestamp_ns"])


def _read_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ObservationError(f"cannot read {label} {resolved}: {error}") from error
    return raw, _object(data, label)


class TemporalObservationLog:
    """One digest-bound log whose streams can be resolved at explicit query times."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        raw, data = _read_json(self.path, "observation log")
        schema_version = data.get("schema_version")
        allowed_fields = {"schema_version", "observation_log_id", "clock", "binding", "streams", "source"}
        if schema_version == LOG_SCHEMA_V2:
            allowed_fields.add("normalization")
        _known(data, allowed_fields, "observation log")
        if schema_version not in LOG_SCHEMAS:
            raise ObservationError(f"observation log must use one of {sorted(LOG_SCHEMAS)}")
        self.schema_version = schema_version
        self.normalization = (
            _ros_normalization(data.get("normalization"), "observation log.normalization")
            if schema_version == LOG_SCHEMA_V2
            else None
        )
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.log_id = _identifier(data.get("observation_log_id"), "observation_log_id")
        clock = _object(data.get("clock"), "clock")
        _known(clock, {"domain", "unit", "epoch"}, "clock")
        if clock.get("unit") != "nanoseconds":
            raise ObservationError("clock.unit must be 'nanoseconds'")
        self.clock = {
            "domain": _string(clock.get("domain"), "clock.domain"),
            "unit": "nanoseconds",
            "epoch": None if clock.get("epoch") is None else _string(clock["epoch"], "clock.epoch"),
        }
        binding = _object(data.get("binding"), "binding")
        _known(binding, {"robot_name", "root_link", "source_urdf_semantic_sha256", "scene_id", "scene_sha256"}, "binding")
        self.binding = {
            "robot_name": _identifier(binding.get("robot_name"), "binding.robot_name"),
            "root_link": _identifier(binding.get("root_link"), "binding.root_link"),
            "source_urdf_semantic_sha256": _string(binding.get("source_urdf_semantic_sha256"), "binding.source_urdf_semantic_sha256"),
            "scene_id": _identifier(binding.get("scene_id"), "binding.scene_id"),
            "scene_sha256": _string(binding.get("scene_sha256"), "binding.scene_sha256"),
        }
        self.source = _source(data.get("source"), "source")
        streams = _object(data.get("streams"), "streams")
        _known(streams, {"joint_states", "robot_root_poses", "object_poses"}, "streams")
        self.joint_states = _stream(streams.get("joint_states", []), "streams.joint_states", _joint_sample)
        self.robot_root_poses = _stream(streams.get("robot_root_poses", []), "streams.robot_root_poses", _pose_sample)
        raw_objects = _object(streams.get("object_poses", {}), "streams.object_poses")
        self.object_poses = {
            _identifier(object_id, "streams.object_poses key"): _stream(
                values,
                f"streams.object_poses.{object_id}",
                _pose_sample,
            )
            for object_id, values in raw_objects.items()
        }

    def _validate_binding(self, model: Any, scene: WorldScene) -> None:
        expected = {
            "robot_name": model.name,
            "root_link": model.root_link,
            "source_urdf_semantic_sha256": model.semantic_sha256,
            "scene_id": scene.scene_id,
            "scene_sha256": scene.sha256,
        }
        mismatches = {
            field: {"log": self.binding[field], "actual": value}
            for field, value in expected.items()
            if self.binding[field] != value
        }
        if mismatches:
            raise ObservationError(f"observation binding mismatch: {mismatches}")
        unknown_objects = sorted(set(self.object_poses) - set(scene.objects))
        if unknown_objects:
            raise ObservationError(f"object pose streams reference undeclared scene objects {unknown_objects}")
        known_frames = set(scene.world_from_scene_frame)
        for label, records in [("robot_root_poses", self.robot_root_poses), *[(f"object_poses.{name}", values) for name, values in self.object_poses.items()]]:
            for record in records:
                if record["parent_scene_frame"] not in known_frames:
                    raise ObservationError(
                        f"{label} sample {record['sample_id']!r} references unknown scene frame {record['parent_scene_frame']!r}"
                    )
        movable = {name for name, joint in model.joints.items() if joint.type != "fixed"}
        drivers = {name for name, joint in model.joints.items() if joint.type != "fixed" and joint.mimic is None}
        for sample in self.joint_states:
            names = set(sample["positions"])
            unknown = sorted(names - movable)
            missing = sorted(drivers - names)
            if unknown:
                raise ObservationError(f"joint sample {sample['sample_id']!r} contains fixed or unknown joints {unknown}")
            if missing:
                raise ObservationError(f"joint sample {sample['sample_id']!r} is missing independent drivers {missing}")
            resolved = model.resolve_pose({name: sample["positions"][name] for name in drivers})
            inconsistent = {
                name: {"observed": sample["positions"][name], "mimic_expected": resolved[name]}
                for name in names - drivers
                if abs(sample["positions"][name] - resolved[name]) > 1e-9
            }
            if inconsistent:
                raise ObservationError(f"joint sample {sample['sample_id']!r} has inconsistent mimic positions {inconsistent}")

    @staticmethod
    def _select(records: list[dict[str, Any]], query_time_ns: int, maximum_age_ns: int) -> dict[str, Any]:
        eligible = [record for record in records if record["timestamp_ns"] <= query_time_ns]
        future_count = len(records) - len(eligible)
        if not eligible:
            return {
                "status": "missing",
                "reason": "no sample exists at or before query time",
                "selected_sample": None,
                "age_ns": None,
                "maximum_age_ns": maximum_age_ns,
                "future_samples_ignored_count": future_count,
            }
        selected = eligible[-1]
        age = query_time_ns - selected["timestamp_ns"]
        public_sample = {key: value for key, value in selected.items() if key != "parent_from_entity"}
        return {
            "status": "current" if age <= maximum_age_ns else "stale",
            "reason": None if age <= maximum_age_ns else "latest past sample exceeds maximum age",
            "selected_sample": public_sample,
            "age_ns": age,
            "maximum_age_ns": maximum_age_ns,
            "future_samples_ignored_count": future_count,
            "_selected_internal": selected,
        }

    def resolve(self, model: Any, scene: WorldScene, query: dict[str, Any]) -> dict[str, Any]:
        self._validate_binding(model, scene)
        query_time = query["time_ns"]
        maximum_age = query["maximum_age_ns"]
        fallbacks = query["fallbacks"]
        required_objects = query["required_object_ids"]
        unknown_required = sorted(set(required_objects) - set(scene.objects))
        if unknown_required:
            raise ObservationError(f"query.required_object_ids references undeclared objects {unknown_required}")

        joint_selection = self._select(self.joint_states, query_time, maximum_age["joint_states"])
        root_selection = self._select(self.robot_root_poses, query_time, maximum_age["robot_root_pose"])
        object_selections = {
            object_id: self._select(self.object_poses.get(object_id, []), query_time, maximum_age["object_pose"])
            for object_id in scene.objects
        }
        current_joint = joint_selection["status"] == "current"
        selected_joint = joint_selection.get("_selected_internal")
        driver_names = sorted(
            name for name, joint in model.joints.items()
            if joint.type != "fixed" and joint.mimic is None
        )
        driver_pose = None if selected_joint is None else {name: selected_joint["positions"][name] for name in driver_names}
        resolved_pose = None if driver_pose is None else model.resolve_pose(driver_pose)

        root_matrix: Matrix | None = None
        root_effective_source: dict[str, Any]
        if root_selection["status"] == "current":
            selected = root_selection["_selected_internal"]
            root_matrix = _matmul(scene.world_from_scene_frame[selected["parent_scene_frame"]], selected["parent_from_entity"])
            root_effective_source = {"layer": "observation", "sample_id": selected["sample_id"]}
        elif fallbacks["robot_root"] == "allow_static_declaration":
            root_matrix = scene.world_from_robot_root
            root_effective_source = {"layer": "static_scene_declaration", "reason": root_selection["status"]}
        else:
            root_effective_source = {"layer": "unavailable", "reason": root_selection["status"]}

        object_matrices: dict[str, Matrix] = {}
        object_effective_sources: dict[str, Any] = {}
        for object_id, selection in object_selections.items():
            if selection["status"] == "current":
                selected = selection["_selected_internal"]
                object_matrices[object_id] = _matmul(
                    scene.world_from_scene_frame[selected["parent_scene_frame"]],
                    selected["parent_from_entity"],
                )
                object_effective_sources[object_id] = {"layer": "observation", "sample_id": selected["sample_id"]}
            elif object_id not in required_objects:
                object_matrices[object_id] = scene.objects[object_id]["world_from_object"]
                object_effective_sources[object_id] = {"layer": "static_scene_declaration", "reason": "object_not_required_by_query"}
            elif fallbacks["objects"] == "allow_static_declaration":
                object_matrices[object_id] = scene.objects[object_id]["world_from_object"]
                object_effective_sources[object_id] = {"layer": "static_scene_declaration", "reason": selection["status"]}
            else:
                object_effective_sources[object_id] = {"layer": "unavailable", "reason": selection["status"]}

        required_current = (
            current_joint
            and root_selection["status"] == "current"
            and all(object_selections[name]["status"] == "current" for name in required_objects)
        )
        nominal_computable = (
            current_joint
            and root_matrix is not None
            and all(name in object_matrices for name in required_objects)
        )
        fallback_entities = [
            "robot_root" if root_effective_source["layer"] == "static_scene_declaration" else None,
            *[
                f"scene_object/{name}"
                for name, source in object_effective_sources.items()
                if source["layer"] == "static_scene_declaration"
            ],
        ]
        fallback_entities = [name for name in fallback_entities if name is not None]

        def public(selection: dict[str, Any]) -> dict[str, Any]:
            return {key: value for key, value in selection.items() if not key.startswith("_")}

        report = {
            "schema_version": RESOLVED_SCHEMA,
            "status": "current" if required_current else ("nominal_with_declaration_fallback" if nominal_computable else "not_current_or_incomplete"),
            "observation_log": {
                "schema_version": self.schema_version,
                "id": self.log_id,
                "path": str(self.path),
                "sha256": self.sha256,
                "clock": self.clock,
                "source": self.source,
                "normalization": self.normalization,
            },
            "binding": self.binding,
            "query": query,
            "selection_method": {
                "name": "latest_sample_at_or_before_query_time",
                "temporal_reconstruction": "zero_order_hold",
                "future_samples_consumed": False,
                "interpolation": False,
            },
            "selections": {
                "joint_states": public(joint_selection),
                "robot_root_pose": public(root_selection),
                "object_poses": {name: public(selection) for name, selection in object_selections.items()},
            },
            "effective_state": {
                "joint_positions": None if resolved_pose is None else {name: resolved_pose[name] for name in sorted(resolved_pose)},
                "independent_driver_positions": driver_pose,
                "world_from_robot_root": None if root_matrix is None else pose_record(root_matrix),
                "world_from_objects": {name: pose_record(matrix) for name, matrix in sorted(object_matrices.items())},
                "sources": {
                    "joint_positions": {"layer": "observation" if current_joint else "unavailable", "sample_id": None if selected_joint is None else selected_joint["sample_id"]},
                    "robot_root": root_effective_source,
                    "objects": object_effective_sources,
                },
            },
            "readiness": {
                "all_required_observations_current": required_current,
                "nominal_world_state_computable": nominal_computable,
                "declaration_fallback_used": bool(fallback_entities),
                "declaration_fallback_entities": sorted(fallback_entities),
                "required_object_ids": required_objects,
                "physical_world_completeness": "not_established",
                "physical_calibration_and_source_truth": "not_established",
            },
            "epistemic_layers": {
                "model": "URDF mechanism declarations and deterministic kinematic consequences",
                "static_scene": "declared mounting, gravity, frames, and object hypotheses",
                "observation": "timestamped source reports selected under an explicit age policy",
            },
            "epistemic_scope": "current means only that every required selected sample satisfies this query's timestamp and age policy; it does not establish sensor truth, calibration, covariance-bounded geometry, omitted-object absence, or physical safety",
        }
        return {
            "report": report,
            "joint_pose": driver_pose if current_joint else None,
            "world_from_robot_root": root_matrix,
            "world_from_objects": object_matrices,
            "nominal_computable": nominal_computable,
            "all_required_current": required_current,
        }


def read_observation_query(path: Path, scene: WorldScene) -> tuple[dict[str, Any], str]:
    raw, data = _read_json(path, "observation query")
    _known(data, {"schema_version", "query_id", "time_ns", "maximum_age_ns", "fallbacks", "required_object_ids"}, "observation query")
    if data.get("schema_version") != QUERY_SCHEMA:
        raise ObservationError(f"observation query must use schema_version {QUERY_SCHEMA}")
    maximum_age = _object(data.get("maximum_age_ns"), "maximum_age_ns")
    _known(maximum_age, {"joint_states", "robot_root_pose", "object_pose"}, "maximum_age_ns")
    fallbacks = _object(data.get("fallbacks"), "fallbacks")
    _known(fallbacks, {"robot_root", "objects"}, "fallbacks")
    for field in ("robot_root", "objects"):
        if fallbacks.get(field) not in FALLBACKS:
            raise ObservationError(f"fallbacks.{field} must be one of {sorted(FALLBACKS)}")
    required = data.get("required_object_ids", sorted(scene.objects))
    if not isinstance(required, list):
        raise ObservationError("required_object_ids must be an array")
    identifiers = [_identifier(name, f"required_object_ids[{index}]") for index, name in enumerate(required)]
    if len(identifiers) != len(set(identifiers)):
        raise ObservationError("required_object_ids must not contain duplicates")
    query = {
        "schema_version": QUERY_SCHEMA,
        "query_id": _identifier(data.get("query_id"), "query_id"),
        "time_ns": _timestamp(data.get("time_ns"), "time_ns"),
        "maximum_age_ns": {
            field: _timestamp(maximum_age.get(field), f"maximum_age_ns.{field}")
            for field in ("joint_states", "robot_root_pose", "object_pose")
        },
        "fallbacks": {field: fallbacks[field] for field in ("robot_root", "objects")},
        "required_object_ids": sorted(identifiers),
    }
    return query, hashlib.sha256(raw).hexdigest()


def resolve_observation(
    log_path: Path,
    query_path: Path,
    model: Any,
    scene: WorldScene,
) -> dict[str, Any]:
    log = TemporalObservationLog(log_path)
    query, query_sha256 = read_observation_query(query_path, scene)
    resolved = log.resolve(model, scene, query)
    resolved["report"]["query_source"] = {
        "path": str(query_path.expanduser().resolve()),
        "sha256": query_sha256,
    }
    resolved["query_sha256"] = query_sha256
    resolved["log"] = log
    return resolved
