#!/usr/bin/env python3
"""Capture ROS 2 JointState/TF messages and normalize them into observation-log.v1.

The deterministic ``normalize`` path has no ROS dependency.  The optional
``capture`` path imports rclpy only when invoked, so a capture can be preserved,
replayed, audited, and normalized on a machine without ROS installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from robot_spatial import RobotModel
from world_scene import Matrix, SceneError, WorldScene, _inverse_rigid, _matmul, _pose, pose_record


CONFIG_SCHEMA = "robot-spatial-ros-adapter-config.v1"
CAPTURE_SCHEMA = "robot-spatial-ros-capture.v1"
REPORT_SCHEMA = "robot-spatial-ros-normalization-report.v1"
OBSERVATION_SCHEMA = "robot-spatial-observation-log.v2"
ROS_NORMALIZATION_PROVENANCE_SCHEMA = "robot-spatial-ros-normalization-provenance.v1"
CAPTURE_TRANSPORTS = {"live_ros2", "rosbag_replay", "imported_json", "synthetic_fixture"}
TIMESTAMP_POLICIES = {"message_header", "message_header_or_receipt"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
JOINT_DUPLICATE_TOLERANCE = 1e-12


class RosAdapterError(ValueError):
    """Invalid, ambiguous, or incorrectly bound ROS capture input."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RosAdapterError(f"{label} must be an object")
    return value


def _known(value: dict[str, Any], fields: set[str], label: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise RosAdapterError(f"{label} has unsupported fields {unknown}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RosAdapterError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label)
    if not SAFE_ID.fullmatch(result):
        raise RosAdapterError(f"{label} may contain only letters, digits, '.', ':', '_', and '-'")
    return result


def _frame(value: Any, label: str) -> str:
    result = _string(value, label)
    if result.startswith("/"):
        raise RosAdapterError(f"{label} must not start with '/'; ROS 2 frame IDs are unqualified names")
    if result.endswith("/") or "//" in result or any(part in {"", ".", ".."} for part in result.split("/")):
        raise RosAdapterError(f"{label} is not a canonical ROS frame ID")
    return result


def _topic(value: Any, label: str) -> str:
    result = _string(value, label)
    if not result.startswith("/") or "//" in result or result.endswith("/"):
        raise RosAdapterError(f"{label} must be a canonical absolute ROS topic")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RosAdapterError(f"{label} must be boolean")
    return value


def _timestamp(value: Any, label: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if optional else ""
        raise RosAdapterError(f"{label} must be a non-negative integer nanosecond timestamp{suffix}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    result = _timestamp(value, label)
    assert result is not None
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RosAdapterError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RosAdapterError(f"{label} must be a finite number")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RosAdapterError(f"{label} must be an array of {length} finite numbers")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _read_json(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RosAdapterError(f"cannot read {label} {resolved}: {error}") from error
    return resolved, raw, _object(data, label)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_new_json(path: Path, value: Any) -> str:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise RosAdapterError(f"output path already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(raw)
        temporary.replace(resolved)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def _string_array(value: Any, label: str, parser: Any = _string) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RosAdapterError(f"{label} must be a non-empty array")
    result = [parser(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise RosAdapterError(f"{label} values must be unique")
    return result


def read_config(path: Path) -> dict[str, Any]:
    resolved, raw, data = _read_json(path, "ROS adapter config")
    _known(data, {"schema_version", "adapter_id", "clock", "binding", "topics", "frames", "joint_mapping", "policies"}, "ROS adapter config")
    if data.get("schema_version") != CONFIG_SCHEMA:
        raise RosAdapterError(f"ROS adapter config must use schema_version {CONFIG_SCHEMA}")

    clock = _object(data.get("clock"), "clock")
    _known(clock, {"domain", "unit", "epoch"}, "clock")
    if clock.get("unit") != "nanoseconds":
        raise RosAdapterError("clock.unit must be 'nanoseconds'")

    binding = _object(data.get("binding"), "binding")
    _known(binding, {"robot_name", "root_link", "source_urdf_semantic_sha256", "scene_id", "scene_sha256"}, "binding")

    topics = _object(data.get("topics"), "topics")
    _known(topics, {"joint_states", "tf_dynamic", "tf_static"}, "topics")
    parsed_topics = {
        name: _string_array(topics.get(name), f"topics.{name}", _topic)
        for name in ("joint_states", "tf_dynamic", "tf_static")
    }
    all_topics = [item for values in parsed_topics.values() for item in values]
    if len(all_topics) != len(set(all_topics)):
        raise RosAdapterError("joint_states, tf_dynamic, and tf_static topic sets must be disjoint")

    frames = _object(data.get("frames"), "frames")
    _known(frames, {"ros_reference_frame", "scene_parent_frame", "robot_root_frame", "objects"}, "frames")
    objects = _object(frames.get("objects"), "frames.objects")
    parsed_objects = {
        _identifier(object_id, "frames.objects key"): _frame(frame_id, f"frames.objects.{object_id}")
        for object_id, frame_id in objects.items()
    }
    if len(set(parsed_objects.values())) != len(parsed_objects):
        raise RosAdapterError("frames.objects values must be unique")

    mapping_value = _object(data.get("joint_mapping"), "joint_mapping")
    joint_mapping = {
        _identifier(model_joint, "joint_mapping key"): _string(ros_joint, f"joint_mapping.{model_joint}")
        for model_joint, ros_joint in mapping_value.items()
    }
    if len(set(joint_mapping.values())) != len(joint_mapping):
        raise RosAdapterError("joint_mapping ROS names must be unique")

    policies = _object(data.get("policies"), "policies")
    _known(policies, {"timestamp_source", "joint_snapshot", "tf_snapshot"}, "policies")
    timestamp_source = policies.get("timestamp_source")
    if timestamp_source not in TIMESTAMP_POLICIES:
        raise RosAdapterError(f"policies.timestamp_source must be one of {sorted(TIMESTAMP_POLICIES)}")
    joint_policy = _object(policies.get("joint_snapshot"), "policies.joint_snapshot")
    _known(joint_policy, {"maximum_component_age_ns", "reject_multiple_publishers_per_joint"}, "policies.joint_snapshot")
    tf_policy = _object(policies.get("tf_snapshot"), "policies.tf_snapshot")
    _known(tf_policy, {"maximum_dynamic_edge_age_ns", "reject_multiple_publishers_per_child", "reject_parent_switches", "matrix_component_tolerance"}, "policies.tf_snapshot")
    tolerance = _finite(tf_policy.get("matrix_component_tolerance"), "policies.tf_snapshot.matrix_component_tolerance")
    if tolerance < 0.0:
        raise RosAdapterError("policies.tf_snapshot.matrix_component_tolerance must be non-negative")
    reject_joint_authority = _boolean(
        joint_policy.get("reject_multiple_publishers_per_joint"),
        "policies.joint_snapshot.reject_multiple_publishers_per_joint",
    )
    reject_tf_authority = _boolean(
        tf_policy.get("reject_multiple_publishers_per_child"),
        "policies.tf_snapshot.reject_multiple_publishers_per_child",
    )
    reject_parent_switches = _boolean(
        tf_policy.get("reject_parent_switches"),
        "policies.tf_snapshot.reject_parent_switches",
    )
    if not (reject_joint_authority and reject_tf_authority and reject_parent_switches):
        raise RosAdapterError(
            "adapter v1 requires all authority-conflict and TF parent-switch rejection policies to be true"
        )

    return {
        "path": resolved,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": CONFIG_SCHEMA,
        "adapter_id": _identifier(data.get("adapter_id"), "adapter_id"),
        "clock": {
            "domain": _string(clock.get("domain"), "clock.domain"),
            "unit": "nanoseconds",
            "epoch": _optional_string(clock.get("epoch"), "clock.epoch"),
        },
        "binding": {
            "robot_name": _identifier(binding.get("robot_name"), "binding.robot_name"),
            "root_link": _identifier(binding.get("root_link"), "binding.root_link"),
            "source_urdf_semantic_sha256": _string(binding.get("source_urdf_semantic_sha256"), "binding.source_urdf_semantic_sha256"),
            "scene_id": _identifier(binding.get("scene_id"), "binding.scene_id"),
            "scene_sha256": _string(binding.get("scene_sha256"), "binding.scene_sha256"),
        },
        "topics": parsed_topics,
        "frames": {
            "ros_reference_frame": _frame(frames.get("ros_reference_frame"), "frames.ros_reference_frame"),
            "scene_parent_frame": _identifier(frames.get("scene_parent_frame"), "frames.scene_parent_frame"),
            "robot_root_frame": _frame(frames.get("robot_root_frame"), "frames.robot_root_frame"),
            "objects": parsed_objects,
        },
        "joint_mapping": joint_mapping,
        "policies": {
            "timestamp_source": timestamp_source,
            "joint_snapshot": {
                "maximum_component_age_ns": _nonnegative_int(joint_policy.get("maximum_component_age_ns"), "policies.joint_snapshot.maximum_component_age_ns"),
                "reject_multiple_publishers_per_joint": reject_joint_authority,
            },
            "tf_snapshot": {
                "maximum_dynamic_edge_age_ns": _nonnegative_int(tf_policy.get("maximum_dynamic_edge_age_ns"), "policies.tf_snapshot.maximum_dynamic_edge_age_ns"),
                "reject_multiple_publishers_per_child": reject_tf_authority,
                "reject_parent_switches": reject_parent_switches,
                "matrix_component_tolerance": tolerance,
            },
        },
    }


def _validate_binding(config: dict[str, Any], model: RobotModel, scene: WorldScene) -> None:
    actual = {
        "robot_name": model.name,
        "root_link": model.root_link,
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "scene_id": scene.scene_id,
        "scene_sha256": scene.sha256,
    }
    mismatches = {
        field: {"config": config["binding"][field], "actual": value}
        for field, value in actual.items()
        if config["binding"][field] != value
    }
    if mismatches:
        raise RosAdapterError(f"ROS adapter binding mismatch: {mismatches}")
    if config["frames"]["scene_parent_frame"] not in scene.world_from_scene_frame:
        raise RosAdapterError(
            f"frames.scene_parent_frame {config['frames']['scene_parent_frame']!r} is not declared by the scene"
        )
    unknown_objects = sorted(set(config["frames"]["objects"]) - set(scene.objects))
    if unknown_objects:
        raise RosAdapterError(f"frames.objects references undeclared scene objects {unknown_objects}")
    drivers = {
        name for name, joint in model.joints.items()
        if joint.type != "fixed" and joint.mimic is None
    }
    if set(config["joint_mapping"]) != drivers:
        raise RosAdapterError(
            f"joint_mapping keys must exactly cover independent model drivers; missing={sorted(drivers - set(config['joint_mapping']))}, extra={sorted(set(config['joint_mapping']) - drivers)}"
        )


def _capture_timestamp(message: int | None, receipt: int, policy: str, label: str) -> tuple[int, str]:
    if message not in {None, 0}:
        assert message is not None
        return message, "message_header"
    if policy == "message_header_or_receipt":
        return receipt, "receipt_fallback_for_zero_or_missing_header"
    raise RosAdapterError(f"{label} has a zero or missing header timestamp under message_header policy")


def read_capture(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    resolved, raw, data = _read_json(path, "ROS capture")
    _known(data, {"schema_version", "capture_id", "adapter_config_sha256", "clock", "capture", "source", "records"}, "ROS capture")
    if data.get("schema_version") != CAPTURE_SCHEMA:
        raise RosAdapterError(f"ROS capture must use schema_version {CAPTURE_SCHEMA}")
    if data.get("adapter_config_sha256") != config["sha256"]:
        raise RosAdapterError("ROS capture adapter_config_sha256 does not match the supplied config bytes")

    clock = _object(data.get("clock"), "capture.clock")
    _known(clock, {"domain", "unit", "epoch"}, "capture.clock")
    capture_clock = {
        "domain": _string(clock.get("domain"), "capture.clock.domain"),
        "unit": clock.get("unit"),
        "epoch": _optional_string(clock.get("epoch"), "capture.clock.epoch"),
    }
    if capture_clock["unit"] != "nanoseconds":
        raise RosAdapterError("capture.clock.unit must be 'nanoseconds'")
    if capture_clock != config["clock"]:
        raise RosAdapterError(f"capture/config clock mismatch: capture={capture_clock}, config={config['clock']}")

    interval = _object(data.get("capture"), "capture.capture")
    _known(interval, {"started_timestamp_ns", "ended_timestamp_ns", "node_use_sim_time"}, "capture.capture")
    started = _nonnegative_int(interval.get("started_timestamp_ns"), "capture.started_timestamp_ns")
    ended = _nonnegative_int(interval.get("ended_timestamp_ns"), "capture.ended_timestamp_ns")
    if ended < started:
        raise RosAdapterError("capture.ended_timestamp_ns must be at or after capture.started_timestamp_ns")
    use_sim_time = interval.get("node_use_sim_time")
    if use_sim_time is not None and not isinstance(use_sim_time, bool):
        raise RosAdapterError("capture.node_use_sim_time must be boolean or null")

    source = _object(data.get("source"), "capture.source")
    _known(source, {"transport", "reference", "ros_distro", "authority_visibility"}, "capture.source")
    transport = source.get("transport")
    if transport not in CAPTURE_TRANSPORTS:
        raise RosAdapterError(f"capture.source.transport must be one of {sorted(CAPTURE_TRANSPORTS)}")

    records_value = data.get("records")
    if not isinstance(records_value, list):
        raise RosAdapterError("capture.records must be an array")
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    transform_ids: set[str] = set()
    allowed_topics = {item for values in config["topics"].values() for item in values}
    for index, value in enumerate(records_value):
        label = f"capture.records[{index}]"
        record = _object(value, label)
        common = {"record_id", "kind", "topic", "publisher_id", "receipt_timestamp_ns"}
        kind = record.get("kind")
        record_id = _identifier(record.get("record_id"), f"{label}.record_id")
        if record_id in record_ids:
            raise RosAdapterError(f"duplicate capture record_id {record_id!r}")
        record_ids.add(record_id)
        topic = _topic(record.get("topic"), f"{label}.topic")
        if topic not in allowed_topics:
            raise RosAdapterError(f"{label}.topic {topic!r} is not declared by the adapter config")
        publisher_id = _optional_string(record.get("publisher_id"), f"{label}.publisher_id")
        receipt = _nonnegative_int(record.get("receipt_timestamp_ns"), f"{label}.receipt_timestamp_ns")
        if receipt < started or receipt > ended:
            raise RosAdapterError(
                f"{label}.receipt_timestamp_ns must fall inside the declared capture interval"
            )
        base = {
            "record_id": record_id,
            "kind": kind,
            "topic": topic,
            "publisher_id": publisher_id,
            "receipt_timestamp_ns": receipt,
            "authority": publisher_id or f"topic:{topic}",
            "authority_directly_visible": publisher_id is not None,
        }
        if kind == "joint_state":
            _known(record, common | {"message_timestamp_ns", "names", "positions"}, label)
            if topic not in config["topics"]["joint_states"]:
                raise RosAdapterError(f"joint_state record {record_id!r} uses a non-joint topic")
            names = _string_array(record.get("names"), f"{label}.names", _string)
            positions = _finite_vector(record.get("positions"), len(names), f"{label}.positions")
            message_timestamp = _timestamp(record.get("message_timestamp_ns"), f"{label}.message_timestamp_ns", optional=True)
            timestamp, origin = _capture_timestamp(message_timestamp, receipt, config["policies"]["timestamp_source"], label)
            if origin == "message_header" and timestamp > receipt:
                raise RosAdapterError(f"{label} header timestamp is later than receipt time in the declared clock domain")
            records.append({
                **base,
                "timestamp_ns": timestamp,
                "timestamp_origin": origin,
                "names": names,
                "positions": positions,
            })
        elif kind == "tf":
            _known(record, common | {"static", "transforms"}, label)
            is_static = _boolean(record.get("static"), f"{label}.static")
            expected_topics = config["topics"]["tf_static" if is_static else "tf_dynamic"]
            if topic not in expected_topics:
                raise RosAdapterError(f"tf record {record_id!r} static/topic classification conflicts with the config")
            transforms_value = record.get("transforms")
            if not isinstance(transforms_value, list) or not transforms_value:
                raise RosAdapterError(f"{label}.transforms must be a non-empty array")
            transforms: list[dict[str, Any]] = []
            for transform_index, transform_value in enumerate(transforms_value):
                transform_label = f"{label}.transforms[{transform_index}]"
                transform = _object(transform_value, transform_label)
                _known(transform, {"transform_id", "message_timestamp_ns", "parent_frame", "child_frame", "pose"}, transform_label)
                transform_id = _identifier(transform.get("transform_id"), f"{transform_label}.transform_id")
                if transform_id in transform_ids:
                    raise RosAdapterError(f"duplicate capture transform_id {transform_id!r}")
                transform_ids.add(transform_id)
                parent = _frame(transform.get("parent_frame"), f"{transform_label}.parent_frame")
                child = _frame(transform.get("child_frame"), f"{transform_label}.child_frame")
                if parent == child:
                    raise RosAdapterError(f"{transform_label} parent and child must differ")
                message_timestamp = _timestamp(transform.get("message_timestamp_ns"), f"{transform_label}.message_timestamp_ns", optional=True)
                if is_static:
                    timestamp, origin = receipt, "static_receipt_time"
                else:
                    timestamp, origin = _capture_timestamp(message_timestamp, receipt, config["policies"]["timestamp_source"], transform_label)
                    if origin == "message_header" and timestamp > receipt:
                        raise RosAdapterError(
                            f"{transform_label} header timestamp is later than receipt time in the declared clock domain"
                        )
                try:
                    canonical_pose, matrix = _pose(transform.get("pose"), f"{transform_label}.pose")
                except SceneError as error:
                    raise RosAdapterError(str(error)) from error
                transforms.append({
                    "transform_id": transform_id,
                    "timestamp_ns": timestamp,
                    "timestamp_origin": origin,
                    "parent": parent,
                    "child": child,
                    "pose": canonical_pose,
                    "matrix": matrix,
                })
            records.append({**base, "static": is_static, "transforms": transforms})
        else:
            raise RosAdapterError(f"{label}.kind must be 'joint_state' or 'tf'")

    return {
        "path": resolved,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "capture_id": _identifier(data.get("capture_id"), "capture.capture_id"),
        "clock": capture_clock,
        "capture": {"started_timestamp_ns": started, "ended_timestamp_ns": ended, "node_use_sim_time": use_sim_time},
        "source": {
            "transport": transport,
            "reference": _optional_string(source.get("reference"), "capture.source.reference"),
            "ros_distro": _optional_string(source.get("ros_distro"), "capture.source.ros_distro"),
            "authority_visibility": _string(source.get("authority_visibility"), "capture.source.authority_visibility"),
        },
        "records": records,
    }


def _matrix_close(left: Matrix, right: Matrix, tolerance: float) -> bool:
    return all(abs(left[row][column] - right[row][column]) <= tolerance for row in range(4) for column in range(4))


def _sample_source(capture: dict[str, Any], config: dict[str, Any], topic: str, authorities: set[str], suffix: str) -> dict[str, Any]:
    directly_visible = all(not authority.startswith("topic:") for authority in authorities)
    sensor_id = None
    if len(authorities) == 1 and directly_visible:
        sensor_id = next(iter(authorities))
    elif len(authorities) > 1:
        sensor_id = "multiple_ros_authorities"
    return {
        "type": "imported",
        "reference": f"ros_capture_sha256:{capture['sha256']}#config_sha256:{config['sha256']}#{suffix}",
        "sensor_id": sensor_id,
        "topic": topic,
    }


def _normalize_joints(model: RobotModel, config: dict[str, Any], capture: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reverse_mapping = {ros_name: model_name for model_name, ros_name in config["joint_mapping"].items()}
    required = set(config["joint_mapping"])
    maximum_age = config["policies"]["joint_snapshot"]["maximum_component_age_ns"]
    reject_authority = config["policies"]["joint_snapshot"]["reject_multiple_publishers_per_joint"]
    records = sorted(
        (record for record in capture["records"] if record["kind"] == "joint_state"),
        key=lambda item: (item["timestamp_ns"], item["receipt_timestamp_ns"], item["record_id"]),
    )
    state: dict[str, dict[str, Any]] = {}
    authorities: dict[str, set[str]] = {name: set() for name in required}
    unknown_names: set[str] = set()
    receipt_fallback_records: list[str] = []
    incomplete_records: list[dict[str, Any]] = []
    stale_records: list[dict[str, Any]] = []
    candidates: dict[int, dict[str, Any]] = {}
    coalesced = 0
    exact_duplicate_components = 0
    components_by_joint_time: dict[tuple[str, int], float] = {}
    for record in records:
        if record["timestamp_origin"] != "message_header":
            receipt_fallback_records.append(record["record_id"])
        for ros_name, position in zip(record["names"], record["positions"]):
            model_name = reverse_mapping.get(ros_name)
            if model_name is None:
                unknown_names.add(ros_name)
                continue
            authorities[model_name].add(record["authority"])
            if reject_authority and len(authorities[model_name]) > 1:
                raise RosAdapterError(
                    f"multiple ROS authorities observed for joint {model_name!r}: {sorted(authorities[model_name])}"
                )
            component_key = (model_name, record["timestamp_ns"])
            previous_position = components_by_joint_time.get(component_key)
            if previous_position is not None:
                if abs(previous_position - position) > JOINT_DUPLICATE_TOLERANCE:
                    raise RosAdapterError(
                        f"conflicting positions for joint {model_name!r} at timestamp {record['timestamp_ns']}"
                    )
                exact_duplicate_components += 1
            else:
                components_by_joint_time[component_key] = position
            state[model_name] = {
                "position": position,
                "timestamp_ns": record["timestamp_ns"],
                "record_id": record["record_id"],
                "authority": record["authority"],
                "topic": record["topic"],
            }
        missing = sorted(required - set(state))
        if missing:
            incomplete_records.append({"record_id": record["record_id"], "timestamp_ns": record["timestamp_ns"], "missing_model_joints": missing})
            continue
        ages = {name: record["timestamp_ns"] - state[name]["timestamp_ns"] for name in required}
        stale = sorted(name for name, age in ages.items() if age < 0 or age > maximum_age)
        if stale:
            stale_records.append({
                "record_id": record["record_id"],
                "timestamp_ns": record["timestamp_ns"],
                "stale_model_joints": stale,
                "component_ages_ns": ages,
            })
            continue
        timestamp = record["timestamp_ns"]
        component_records = {name: state[name]["record_id"] for name in sorted(required)}
        snapshot_authorities = {state[name]["authority"] for name in required}
        snapshot_topics = sorted({state[name]["topic"] for name in required})
        topic = snapshot_topics[0] if len(snapshot_topics) == 1 else ",".join(snapshot_topics)
        positions = {name: state[name]["position"] for name in sorted(required)}
        try:
            model.resolve_pose(positions)
        except ValueError as error:
            raise RosAdapterError(
                f"joint snapshot at timestamp {timestamp} violates the bound URDF model: {error}"
            ) from error
        sample = {
            "sample_id": f"ros_joint_{len(candidates) + 1:06d}",
            "timestamp_ns": timestamp,
            "positions": positions,
            "position_standard_deviation": {},
            "source": _sample_source(capture, config, topic, snapshot_authorities, f"joint_snapshot:{timestamp}"),
            "_provenance": {
                "component_record_ids": component_records,
                "component_ages_ns": ages,
                "authorities": sorted(snapshot_authorities),
            },
        }
        if timestamp in candidates:
            coalesced += 1
        candidates[timestamp] = sample
    samples = []
    provenance = []
    for index, timestamp in enumerate(sorted(candidates), start=1):
        sample = candidates[timestamp]
        sample["sample_id"] = f"ros_joint_{index:06d}"
        provenance.append({"sample_id": sample["sample_id"], "timestamp_ns": timestamp, **sample.pop("_provenance")})
        samples.append(sample)
    return samples, {
        "input_record_count": len(records),
        "output_snapshot_count": len(samples),
        "required_model_joints": sorted(required),
        "joint_mapping": config["joint_mapping"],
        "maximum_component_age_ns": maximum_age,
        "authority_policy": "reject_multiple_publishers_per_joint" if reject_authority else "latest_message_with_conflicts_reported",
        "authorities_by_model_joint": {name: sorted(values) for name, values in sorted(authorities.items())},
        "authority_direct_visibility_complete": all(not authority.startswith("topic:") for values in authorities.values() for authority in values),
        "ignored_unmapped_ros_joint_names": sorted(unknown_names),
        "receipt_timestamp_fallback_record_ids": receipt_fallback_records,
        "incomplete_record_events": incomplete_records,
        "stale_component_events": stale_records,
        "coalesced_same_timestamp_snapshot_count": coalesced,
        "exact_duplicate_joint_component_count": exact_duplicate_components,
        "snapshots": provenance,
    }


def _normalize_tf(config: dict[str, Any], capture: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    policy = config["policies"]["tf_snapshot"]
    tolerance = policy["matrix_component_tolerance"]
    static_by_child: dict[str, dict[str, Any]] = {}
    dynamic_by_child: dict[str, dict[int, dict[str, Any]]] = {}
    parents_by_child: dict[str, set[str]] = {}
    authorities_by_child: dict[str, set[str]] = {}
    duplicate_exact = 0
    input_transform_count = 0

    tf_records = [record for record in capture["records"] if record["kind"] == "tf"]
    for record in tf_records:
        for transform in record["transforms"]:
            input_transform_count += 1
            edge = {
                **transform,
                "record_id": record["record_id"],
                "topic": record["topic"],
                "authority": record["authority"],
                "authority_directly_visible": record["authority_directly_visible"],
                "receipt_timestamp_ns": record["receipt_timestamp_ns"],
                "static": record["static"],
            }
            child = transform["child"]
            parents_by_child.setdefault(child, set()).add(transform["parent"])
            authorities_by_child.setdefault(child, set()).add(record["authority"])
            if policy["reject_parent_switches"] and len(parents_by_child[child]) > 1:
                raise RosAdapterError(f"TF child {child!r} switches parents: {sorted(parents_by_child[child])}")
            if policy["reject_multiple_publishers_per_child"] and len(authorities_by_child[child]) > 1:
                raise RosAdapterError(f"multiple ROS authorities observed for TF child {child!r}: {sorted(authorities_by_child[child])}")
            if record["static"]:
                if child in dynamic_by_child:
                    raise RosAdapterError(f"TF child {child!r} is published as both static and dynamic")
                previous = static_by_child.get(child)
                if previous is not None:
                    if previous["parent"] != edge["parent"] or not _matrix_close(previous["matrix"], edge["matrix"], tolerance):
                        raise RosAdapterError(f"conflicting duplicate static TF for child {child!r}")
                    duplicate_exact += 1
                else:
                    static_by_child[child] = edge
            else:
                if child in static_by_child:
                    raise RosAdapterError(f"TF child {child!r} is published as both static and dynamic")
                by_time = dynamic_by_child.setdefault(child, {})
                previous = by_time.get(edge["timestamp_ns"])
                if previous is not None:
                    if previous["parent"] != edge["parent"] or not _matrix_close(previous["matrix"], edge["matrix"], tolerance):
                        raise RosAdapterError(
                            f"ambiguous dynamic TF for child {child!r} at timestamp {edge['timestamp_ns']}"
                        )
                    duplicate_exact += 1
                else:
                    by_time[edge["timestamp_ns"]] = edge

    parent_map = {child: next(iter(parents)) for child, parents in parents_by_child.items() if len(parents) == 1}
    for start in parent_map:
        seen: set[str] = set()
        current = start
        while current in parent_map:
            if current in seen:
                raise RosAdapterError(f"TF topology contains a cycle through frame {current!r}")
            seen.add(current)
            current = parent_map[current]

    topology_neighbors: dict[str, list[tuple[str, str]]] = {}
    for child, parent in parent_map.items():
        topology_neighbors.setdefault(parent, []).append((child, child))
        topology_neighbors.setdefault(child, []).append((parent, child))

    reference = config["frames"]["ros_reference_frame"]
    target_map = {"robot_root": config["frames"]["robot_root_frame"]}
    target_map.update({f"scene_object/{name}": frame for name, frame in config["frames"]["objects"].items()})
    max_age = policy["maximum_dynamic_edge_age_ns"]

    def structural_path(target: str) -> list[str] | None:
        if target == reference:
            return []
        queue = deque([(reference, [])])
        visited = {reference}
        while queue:
            frame_id, edges = queue.popleft()
            for neighbor, child_edge in topology_neighbors.get(frame_id, []):
                if neighbor in visited:
                    continue
                path = [*edges, child_edge]
                if neighbor == target:
                    return path
                visited.add(neighbor)
                queue.append((neighbor, path))
        return None

    def selected_edge(child: str, timestamp: int) -> dict[str, Any] | None:
        if child in static_by_child:
            return static_by_child[child]
        by_time = dynamic_by_child.get(child, {})
        eligible = [time_value for time_value in by_time if time_value <= timestamp]
        return None if not eligible else by_time[max(eligible)]

    def compose(target: str, path_children: list[str], timestamp: int) -> tuple[Matrix | None, list[dict[str, Any]], str | None]:
        adjacency: dict[str, list[tuple[str, Matrix, dict[str, Any]]]] = {}
        selected: dict[str, dict[str, Any]] = {}
        for child in path_children:
            edge = selected_edge(child, timestamp)
            if edge is None:
                return None, [], f"missing_past_tf_edge:{child}"
            if not edge["static"]:
                age = timestamp - edge["timestamp_ns"]
                if age < 0 or age > max_age:
                    return None, [], f"stale_tf_edge:{child}:age_ns={age}"
            selected[child] = edge
            adjacency.setdefault(edge["parent"], []).append((child, edge["matrix"], edge))
            adjacency.setdefault(child, []).append((edge["parent"], _inverse_rigid(edge["matrix"]), edge))
        if target == reference:
            identity: Matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            return identity, [], None
        queue = deque([(reference, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [])])
        visited = {reference}
        while queue:
            frame_id, reference_from_frame, provenance = queue.popleft()
            for neighbor, frame_from_neighbor, edge in adjacency.get(frame_id, []):
                if neighbor in visited:
                    continue
                next_transform = _matmul(reference_from_frame, frame_from_neighbor)
                edge_record = {
                    "parent_frame": edge["parent"],
                    "child_frame": edge["child"],
                    "static": edge["static"],
                    "selected_timestamp_ns": None if edge["static"] else edge["timestamp_ns"],
                    "age_ns": None if edge["static"] else timestamp - edge["timestamp_ns"],
                    "record_id": edge["record_id"],
                    "transform_id": edge["transform_id"],
                    "topic": edge["topic"],
                    "authority": edge["authority"],
                }
                next_provenance = [*provenance, edge_record]
                if neighbor == target:
                    return next_transform, next_provenance, None
                visited.add(neighbor)
                queue.append((neighbor, next_transform, next_provenance))
        return None, [], "internal_path_composition_failure"

    root_samples: list[dict[str, Any]] = []
    object_samples: dict[str, list[dict[str, Any]]] = {name: [] for name in config["frames"]["objects"]}
    target_reports: dict[str, Any] = {}
    for target_key, target_frame in target_map.items():
        path_children = structural_path(target_frame)
        if path_children is None:
            target_reports[target_key] = {
                "ros_target_frame": target_frame,
                "status": "unreachable",
                "reason": f"no TF topology path from {reference!r}",
                "samples": [],
                "skipped_events": [],
            }
            continue
        dynamic_times = sorted({timestamp for child in path_children for timestamp in dynamic_by_child.get(child, {})})
        if dynamic_times:
            event_times = dynamic_times
        else:
            static_receipts = [static_by_child[child]["receipt_timestamp_ns"] for child in path_children if child in static_by_child]
            event_times = [max(static_receipts)] if static_receipts else [capture["capture"]["started_timestamp_ns"]]
        sample_records: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        output_by_time: dict[int, dict[str, Any]] = {}
        for timestamp in event_times:
            matrix, provenance, reason = compose(target_frame, path_children, timestamp)
            if matrix is None:
                skipped.append({"event_timestamp_ns": timestamp, "reason": reason})
                continue
            authorities = {edge["authority"] for edge in provenance}
            topics = sorted({edge["topic"] for edge in provenance})
            topic = topics[0] if len(topics) == 1 else ",".join(topics) if topics else config["topics"]["tf_static"][0]
            pose = pose_record(matrix)
            suffix = f"tf_snapshot:{target_key}:{timestamp}"
            sample = {
                "sample_id": "pending",
                "timestamp_ns": timestamp,
                "parent_scene_frame": config["frames"]["scene_parent_frame"],
                "pose": {"xyz_m": pose["translation_xyz_m"], "quaternion_xyzw": pose["quaternion_xyzw"]},
                "covariance": None,
                "source": _sample_source(capture, config, topic, authorities, suffix),
            }
            output_by_time[timestamp] = sample
            sample_records.append({
                "timestamp_ns": timestamp,
                "pose": pose,
                "path_edges": provenance,
                "reconstruction": "latest_edge_at_or_before_event_zero_order_hold",
            })
        output_samples = [output_by_time[timestamp] for timestamp in sorted(output_by_time)]
        safe_target = re.sub(r"[^A-Za-z0-9_.:-]", "_", target_key)
        for index, sample in enumerate(output_samples, start=1):
            sample["sample_id"] = f"ros_tf_{safe_target}_{index:06d}"
        if target_key == "robot_root":
            root_samples.extend(output_samples)
        else:
            object_id = target_key.split("/", 1)[1]
            object_samples[object_id].extend(output_samples)
        target_reports[target_key] = {
            "ros_target_frame": target_frame,
            "status": "normalized" if output_samples else "no_valid_snapshot",
            "topology_path_child_edges": path_children,
            "output_snapshot_count": len(output_samples),
            "samples": sample_records,
            "skipped_events": skipped,
        }

    return root_samples, object_samples, {
        "input_tf_record_count": len(tf_records),
        "input_transform_count": input_transform_count,
        "static_child_count": len(static_by_child),
        "dynamic_child_count": len(dynamic_by_child),
        "exact_duplicate_transform_count": duplicate_exact,
        "ros_reference_frame": reference,
        "scene_parent_frame": config["frames"]["scene_parent_frame"],
        "maximum_dynamic_edge_age_ns": max_age,
        "authority_policy": "reject_multiple_publishers_per_child" if policy["reject_multiple_publishers_per_child"] else "latest_message_with_conflicts_reported",
        "parent_switch_policy": "reject" if policy["reject_parent_switches"] else "allowed_but_not_recommended",
        "authorities_by_child": {name: sorted(values) for name, values in sorted(authorities_by_child.items())},
        "authority_direct_visibility_complete": all(not authority.startswith("topic:") for values in authorities_by_child.values() for authority in values),
        "targets": target_reports,
    }


def normalize(model: RobotModel, scene: WorldScene, config: dict[str, Any], capture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_binding(config, model, scene)
    joint_samples, joint_report = _normalize_joints(model, config, capture)
    root_samples, object_samples, tf_report = _normalize_tf(config, capture)
    source = {
        "type": "imported",
        "reference": f"ros_capture_sha256:{capture['sha256']}#config_sha256:{config['sha256']}",
        "sensor_id": None,
        "topic": None,
    }
    log = {
        "schema_version": OBSERVATION_SCHEMA,
        "observation_log_id": f"ros_{config['adapter_id']}_{capture['capture_id']}",
        "clock": capture["clock"],
        "binding": config["binding"],
        "source": source,
        "normalization": {
            "schema_version": ROS_NORMALIZATION_PROVENANCE_SCHEMA,
            "adapter_id": config["adapter_id"],
            "config_sha256": config["sha256"],
            "capture_id": capture["capture_id"],
            "capture_sha256": capture["sha256"],
            "method": "partial_joint_component_assembly_and_latest_past_tf_path_zero_order_hold",
            "clock_policy": {
                "timestamp_source": config["policies"]["timestamp_source"],
                "synchronization_verified": False,
            },
            "authority_policy": {
                "joint": "reject_multiple_publishers_per_joint",
                "tf": "reject_multiple_publishers_per_child_and_parent_switches",
                "publisher_identity_visibility": capture["source"]["authority_visibility"],
            },
            "tf_policy": {
                "maximum_dynamic_edge_age_ns": config["policies"]["tf_snapshot"]["maximum_dynamic_edge_age_ns"],
                "interpolation": False,
                "extrapolation": False,
                "future_samples_consumed": False,
            },
        },
        "streams": {
            "joint_states": joint_samples,
            "robot_root_poses": root_samples,
            "object_poses": object_samples,
        },
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "normalized",
        "adapter": {"id": config["adapter_id"], "config_path": str(config["path"]), "config_sha256": config["sha256"]},
        "capture": {
            "id": capture["capture_id"],
            "path": str(capture["path"]),
            "sha256": capture["sha256"],
            "source": capture["source"],
            "clock": capture["clock"],
            **capture["capture"],
        },
        "binding": config["binding"],
        "timestamp_policy": {
            "source": config["policies"]["timestamp_source"],
            "tf_reconstruction": "latest_edge_at_or_before_event_zero_order_hold",
            "interpolation": False,
            "extrapolation": False,
            "future_samples_consumed": False,
        },
        "joint_normalization": joint_report,
        "tf_normalization": tf_report,
        "output_counts": {
            "joint_state_samples": len(joint_samples),
            "robot_root_pose_samples": len(root_samples),
            "object_pose_samples": {name: len(values) for name, values in object_samples.items()},
        },
        "epistemic_boundary": {
            "transport_capture": "records ROS messages and visible publisher metadata; it does not establish sensor truth or calibration",
            "clock": "clock domain is asserted by config/capture equality; synchronization between publishers is not independently measured",
            "tf": "poses are discrete latest-past compositions; no tf2 interpolation, extrapolation, latency compensation, or covariance propagation is performed",
            "authority": "publisher GID conflicts are detectable only when publisher metadata is present; rosbag replay may hide original publisher identity",
            "physical_world": "normalized reports remain observations, not proof of completeness, collision safety, or actual robot state",
        },
    }
    return log, report


def _mapping_argument(value: str, label: str) -> tuple[str, str]:
    if "=" not in value:
        raise RosAdapterError(f"{label} must use LEFT=RIGHT")
    left, right = value.split("=", 1)
    return left, right


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    model = RobotModel(args.urdf)
    scene = WorldScene(args.scene, expected_robot_name=model.name, expected_root_link=model.root_link)
    drivers = sorted(name for name, joint in model.joints.items() if joint.type != "fixed" and joint.mimic is None)
    joint_mapping = {name: name for name in drivers}
    mapped_model_names: set[str] = set()
    for value in args.joint_map:
        model_name, ros_name = _mapping_argument(value, "--joint-map")
        if model_name not in joint_mapping:
            raise RosAdapterError(f"--joint-map references non-driver model joint {model_name!r}")
        if model_name in mapped_model_names:
            raise RosAdapterError(f"--joint-map repeats model joint {model_name!r}")
        mapped_model_names.add(model_name)
        joint_mapping[model_name] = _string(ros_name, "--joint-map ROS name")
    objects: dict[str, str] = {}
    for value in args.object_frame:
        object_id, frame_id = _mapping_argument(value, "--object-frame")
        if object_id not in scene.objects:
            raise RosAdapterError(f"--object-frame references undeclared scene object {object_id!r}")
        if object_id in objects:
            raise RosAdapterError(f"--object-frame repeats scene object {object_id!r}")
        objects[object_id] = _frame(frame_id, "--object-frame ROS frame")
    if args.scene_parent_frame not in scene.world_from_scene_frame:
        raise RosAdapterError(f"--scene-parent-frame {args.scene_parent_frame!r} is not declared by the scene")
    joint_topics = args.joint_topic or ["/joint_states"]
    dynamic_topics = args.tf_topic or ["/tf"]
    static_topics = args.tf_static_topic or ["/tf_static"]
    for label, values in (
        ("--joint-topic", joint_topics),
        ("--tf-topic", dynamic_topics),
        ("--tf-static-topic", static_topics),
    ):
        for value in values:
            _topic(value, label)
        if len(values) != len(set(values)):
            raise RosAdapterError(f"{label} values must be unique")
    if len(set(joint_topics + dynamic_topics + static_topics)) != len(joint_topics + dynamic_topics + static_topics):
        raise RosAdapterError("joint, dynamic TF, and static TF topic sets must be disjoint")
    maximum_joint_age = _nonnegative_int(args.maximum_joint_component_age_ns, "--maximum-joint-component-age-ns")
    maximum_tf_age = _nonnegative_int(args.maximum_tf_edge_age_ns, "--maximum-tf-edge-age-ns")
    matrix_tolerance = _finite(args.matrix_component_tolerance, "--matrix-component-tolerance")
    if matrix_tolerance < 0.0:
        raise RosAdapterError("--matrix-component-tolerance must be non-negative")
    return {
        "schema_version": CONFIG_SCHEMA,
        "adapter_id": _identifier(args.adapter_id, "--adapter-id"),
        "clock": {"domain": _string(args.clock_domain, "--clock-domain"), "unit": "nanoseconds", "epoch": args.clock_epoch},
        "binding": {
            "robot_name": model.name,
            "root_link": model.root_link,
            "source_urdf_semantic_sha256": model.semantic_sha256,
            "scene_id": scene.scene_id,
            "scene_sha256": scene.sha256,
        },
        "topics": {
            "joint_states": joint_topics,
            "tf_dynamic": dynamic_topics,
            "tf_static": static_topics,
        },
        "frames": {
            "ros_reference_frame": _frame(args.ros_reference_frame, "--ros-reference-frame"),
            "scene_parent_frame": _identifier(args.scene_parent_frame, "--scene-parent-frame"),
            "robot_root_frame": _frame(args.robot_root_frame or model.root_link, "--robot-root-frame"),
            "objects": objects,
        },
        "joint_mapping": joint_mapping,
        "policies": {
            "timestamp_source": args.timestamp_source,
            "joint_snapshot": {
                "maximum_component_age_ns": maximum_joint_age,
                "reject_multiple_publishers_per_joint": True,
            },
            "tf_snapshot": {
                "maximum_dynamic_edge_age_ns": maximum_tf_age,
                "reject_multiple_publishers_per_child": True,
                "reject_parent_switches": True,
                "matrix_component_tolerance": matrix_tolerance,
            },
        },
    }


def _publisher_id(message_info: Any) -> str | None:
    if isinstance(message_info, dict):
        gid = message_info.get("publisher_gid")
    else:
        gid = getattr(message_info, "publisher_gid", None)
    if gid is None:
        return None
    implementation = None
    if isinstance(gid, dict):
        implementation = gid.get("implementation_identifier")
        gid = gid.get("data")
        if gid is None:
            return None
    try:
        raw = bytes(gid)
    except (TypeError, ValueError):
        raw = bytes(getattr(gid, "data", []))
    if not raw:
        return None
    prefix = f"{implementation}:" if isinstance(implementation, str) and implementation else ""
    return prefix + raw.hex()


def capture_live(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
    except ImportError as error:
        raise RosAdapterError(
            "live capture requires a sourced ROS 2 Python environment with rclpy, sensor_msgs, and tf2_msgs"
        ) from error

    if args.duration_sec <= 0.0:
        raise RosAdapterError("--duration-sec must be positive")
    records: list[dict[str, Any]] = []
    counter = 0

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__(args.node_name)
            nonlocal counter
            dynamic_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            static_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )

            def next_id() -> str:
                nonlocal counter
                counter += 1
                return f"rec{counter:09d}"

            def joint_callback(topic_name: str):
                def callback(message: Any, info: Any) -> None:
                    record_id = next_id()
                    records.append({
                        "record_id": record_id,
                        "kind": "joint_state",
                        "topic": topic_name,
                        "publisher_id": _publisher_id(info),
                        "receipt_timestamp_ns": self.get_clock().now().nanoseconds,
                        "message_timestamp_ns": int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec),
                        "names": list(message.name),
                        "positions": [float(value) for value in message.position],
                    })
                return callback

            def tf_callback(topic_name: str, is_static: bool):
                def callback(message: Any, info: Any) -> None:
                    record_id = next_id()
                    transforms = []
                    for transform_index, stamped in enumerate(message.transforms):
                        transforms.append({
                            "transform_id": f"{record_id}_tf{transform_index:04d}",
                            "message_timestamp_ns": int(stamped.header.stamp.sec) * 1_000_000_000 + int(stamped.header.stamp.nanosec),
                            "parent_frame": stamped.header.frame_id,
                            "child_frame": stamped.child_frame_id,
                            "pose": {
                                "xyz_m": [float(stamped.transform.translation.x), float(stamped.transform.translation.y), float(stamped.transform.translation.z)],
                                "quaternion_xyzw": [float(stamped.transform.rotation.x), float(stamped.transform.rotation.y), float(stamped.transform.rotation.z), float(stamped.transform.rotation.w)],
                            },
                        })
                    if transforms:
                        records.append({
                            "record_id": record_id,
                            "kind": "tf",
                            "topic": topic_name,
                            "publisher_id": _publisher_id(info),
                            "receipt_timestamp_ns": self.get_clock().now().nanoseconds,
                            "static": is_static,
                            "transforms": transforms,
                        })
                return callback

            self.subscriptions = []
            for topic_name in config["topics"]["joint_states"]:
                self.subscriptions.append(self.create_subscription(JointState, topic_name, joint_callback(topic_name), qos_profile_sensor_data))
            for topic_name in config["topics"]["tf_dynamic"]:
                self.subscriptions.append(self.create_subscription(TFMessage, topic_name, tf_callback(topic_name, False), dynamic_qos))
            for topic_name in config["topics"]["tf_static"]:
                self.subscriptions.append(self.create_subscription(TFMessage, topic_name, tf_callback(topic_name, True), static_qos))

    rclpy.init(args=None)
    node = CaptureNode()
    started = node.get_clock().now().nanoseconds
    deadline = time.monotonic() + args.duration_sec
    try:
        while rclpy.ok() and time.monotonic() < deadline and (args.max_records <= 0 or len(records) < args.max_records):
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
        ended = node.get_clock().now().nanoseconds
        use_sim_time = bool(node.get_parameter("use_sim_time").value)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not records:
        raise RosAdapterError("capture received no JointState or TF records; no output was written")
    return {
        "schema_version": CAPTURE_SCHEMA,
        "capture_id": _identifier(args.capture_id, "--capture-id"),
        "adapter_config_sha256": config["sha256"],
        "clock": config["clock"],
        "capture": {
            "started_timestamp_ns": started,
            "ended_timestamp_ns": ended,
            "node_use_sim_time": use_sim_time,
        },
        "source": {
            "transport": "live_ros2",
            "reference": args.source_reference,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "authority_visibility": "rclpy subscription MessageInfo.publisher_gid when the RMW implementation provides it",
        },
        "records": records,
    }


def probe() -> dict[str, Any]:
    modules = {}
    for name in ("rclpy", "sensor_msgs.msg", "tf2_msgs.msg"):
        try:
            __import__(name)
            modules[name] = "available"
        except ImportError:
            modules[name] = "missing"
    return {
        "schema_version": "robot-spatial-ros-adapter-probe.v1",
        "deterministic_normalize": "available",
        "live_capture": "available" if all(value == "available" for value in modules.values()) else "unavailable",
        "python_modules": modules,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "meaning": "normalize is dependency-free; live capture additionally requires a sourced ROS 2 Python environment",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="report deterministic and live ROS adapter availability")

    template = subparsers.add_parser("make-config", help="create a digest-bound ROS adapter config from a URDF and scene")
    template.add_argument("urdf", type=Path)
    template.add_argument("--scene", type=Path, required=True)
    template.add_argument("--adapter-id", required=True)
    template.add_argument("--clock-domain", default="ros_time")
    template.add_argument("--clock-epoch")
    template.add_argument("--ros-reference-frame", required=True)
    template.add_argument("--scene-parent-frame", required=True)
    template.add_argument("--robot-root-frame")
    template.add_argument("--object-frame", action="append", default=[], metavar="OBJECT_ID=ROS_FRAME")
    template.add_argument("--joint-map", action="append", default=[], metavar="MODEL_JOINT=ROS_JOINT")
    template.add_argument("--joint-topic", action="append")
    template.add_argument("--tf-topic", action="append")
    template.add_argument("--tf-static-topic", action="append")
    template.add_argument("--timestamp-source", choices=sorted(TIMESTAMP_POLICIES), default="message_header")
    template.add_argument("--maximum-joint-component-age-ns", type=int, default=100_000_000)
    template.add_argument("--maximum-tf-edge-age-ns", type=int, default=100_000_000)
    template.add_argument("--matrix-component-tolerance", type=float, default=1e-9)
    template.add_argument("--out", type=Path, required=True)

    capture = subparsers.add_parser("capture", help="subscribe to ROS 2 JointState/TF topics and write an immutable raw capture")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--duration-sec", type=float, required=True)
    capture.add_argument("--max-records", type=int, default=0)
    capture.add_argument("--capture-id", required=True)
    capture.add_argument("--source-reference")
    capture.add_argument("--node-name", default="robot_spatial_observation_capture")
    capture.add_argument("--out", type=Path, required=True)

    normalize_parser = subparsers.add_parser("normalize", help="convert a preserved ROS capture into observation-log.v2 with normalization provenance")
    normalize_parser.add_argument("urdf", type=Path)
    normalize_parser.add_argument("--scene", type=Path, required=True)
    normalize_parser.add_argument("--config", type=Path, required=True)
    normalize_parser.add_argument("--capture", type=Path, required=True)
    normalize_parser.add_argument("--out", type=Path, required=True, help="new observation-log.v2 path")
    normalize_parser.add_argument("--report", type=Path, required=True, help="new normalization report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            print(_json_bytes(probe()).decode("utf-8"), end="")
            return 0
        if args.command == "make-config":
            value = make_config(args)
            digest = _write_new_json(args.out, value)
            print(_json_bytes({"status": "created", "config": str(args.out.expanduser().resolve()), "sha256": digest}).decode("utf-8"), end="")
            return 0
        config = read_config(args.config)
        if args.command == "capture":
            value = capture_live(config, args)
            digest = _write_new_json(args.out, value)
            print(_json_bytes({"status": "captured", "capture": str(args.out.expanduser().resolve()), "sha256": digest, "record_count": len(value["records"])}).decode("utf-8"), end="")
            return 0
        model = RobotModel(args.urdf)
        scene = WorldScene(args.scene, expected_robot_name=model.name, expected_root_link=model.root_link)
        capture = read_capture(args.capture, config)
        log, report = normalize(model, scene, config, capture)
        if args.out.expanduser().resolve() == args.report.expanduser().resolve():
            raise RosAdapterError("--out and --report must differ")
        if args.report.expanduser().resolve().exists():
            raise RosAdapterError(f"output path already exists: {args.report.expanduser().resolve()}")
        log_sha = _write_new_json(args.out, log)
        report["output"] = {
            "observation_log_path": str(args.out.expanduser().resolve()),
            "observation_log_sha256": log_sha,
            "observation_log_id": log["observation_log_id"],
        }
        report_sha = _write_new_json(args.report, report)
        print(_json_bytes({
            "status": "normalized",
            "observation_log": str(args.out.expanduser().resolve()),
            "observation_log_sha256": log_sha,
            "report": str(args.report.expanduser().resolve()),
            "report_sha256": report_sha,
            "output_counts": report["output_counts"],
        }).decode("utf-8"), end="")
        return 0
    except (RosAdapterError, SceneError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
