#!/usr/bin/env python3
"""Capture one ROS 2 action-client exchange and compile bounded lifecycle evidence.

The ``normalize`` path is dependency-free apart from the robot-spatial modules.
The optional ``execute-capture`` path imports ROS only when explicitly invoked
and requires an exact action-instance authorization token because sending a goal
may move physical hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from spatial_action_assurance import (
    BUNDLE_SCHEMA,
    SOURCE_SCHEMA,
    ActionAssuranceError,
    build_action_assurance,
)
from spatial_functional import FunctionalError, read_functional_model


CONFIG_SCHEMA = "robot-spatial-ros-action-adapter-config.v1"
CAPTURE_SCHEMA = "robot-spatial-ros-action-capture.v1"
REPORT_SCHEMA = "robot-spatial-ros-action-normalization-report.v1"
PROBE_SCHEMA = "robot-spatial-ros-action-adapter-probe.v1"
CAPTURE_TRANSPORTS = {
    "live_ros2_action_client",
    "imported_json",
    "rosbag_replay",
    "synthetic_fixture",
}
RECORD_KINDS = {
    "send_goal_request",
    "goal_response",
    "feedback",
    "status_array",
    "get_result_request",
    "result_response",
}
STATUS_VALUES = {
    1: "accepted",
    2: "executing",
    3: "canceling",
    4: "succeeded",
    5: "canceled",
    6: "aborted",
}
TERMINAL_STATUS_VALUES = {4: "succeeded", 5: "canceled", 6: "aborted"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
ACTION_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*/action/[A-Za-z][A-Za-z0-9_]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GOAL_UUID = re.compile(r"^[0-9a-f]{32}$")


class RosActionAdapterError(ValueError):
    """Invalid, ambiguous, unsafe, or incorrectly bound ROS action input."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RosActionAdapterError(f"{label} must be an object")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RosActionAdapterError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RosActionAdapterError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if not SAFE_ID.fullmatch(result):
        raise RosActionAdapterError(f"{label} may contain only letters, digits, '.', ':', '_', and '-'")
    return result


def _typed_id(value: Any, prefix: str, label: str) -> str:
    result = _text(value, label)
    if not result.startswith(f"{prefix}/") or result.endswith("/"):
        raise RosActionAdapterError(f"{label} must use typed prefix {prefix}/")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RosActionAdapterError(f"{label} must be boolean")
    return value


def _timestamp(value: Any, label: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if optional else ""
        raise RosActionAdapterError(f"{label} must be a non-negative integer nanosecond timestamp{suffix}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    result = _timestamp(value, label)
    assert result is not None
    return result


def _positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RosActionAdapterError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise RosActionAdapterError(f"{label} must be a finite {qualifier} number")
    return result


def _sha(value: Any, label: str) -> str:
    result = _text(value, label)
    if not SHA256.fullmatch(result):
        raise RosActionAdapterError(f"{label} must be lowercase SHA-256")
    return result


def _goal_uuid(value: Any, label: str) -> str:
    result = _text(value, label)
    if not GOAL_UUID.fullmatch(result):
        raise RosActionAdapterError(f"{label} must be 32 lowercase hexadecimal characters")
    return result


def _action_name(value: Any, label: str) -> str:
    result = _text(value, label)
    if not result.startswith("/") or result.endswith("/") or "//" in result:
        raise RosActionAdapterError(f"{label} must be a canonical absolute ROS 2 name")
    if any(part in {"", ".", ".."} for part in result[1:].split("/")):
        raise RosActionAdapterError(f"{label} must be a canonical absolute ROS 2 name")
    if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is None for part in result[1:].split("/")):
        raise RosActionAdapterError(f"{label} contains an invalid ROS 2 name token")
    return result


def _action_type(value: Any, label: str) -> str:
    result = _text(value, label)
    if not ACTION_TYPE.fullmatch(result):
        raise RosActionAdapterError(f"{label} must use package/action/Type")
    return result


def _clock(value: Any, label: str) -> dict[str, str]:
    raw = _object(value, label)
    _expect_keys(raw, {"domain", "unit", "epoch"}, label)
    result = {key: _text(raw[key], f"{label}.{key}") for key in ("domain", "unit", "epoch")}
    if result["unit"] != "nanoseconds":
        raise RosActionAdapterError(f"{label}.unit must be 'nanoseconds'")
    return result


def _string_map(value: Any, label: str) -> dict[str, str]:
    raw = _object(value, label)
    if not raw or any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item
        for key, item in raw.items()
    ):
        raise RosActionAdapterError(f"{label} must map non-empty strings to non-empty strings")
    return dict(sorted(raw.items()))


def _validate_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RosActionAdapterError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RosActionAdapterError(f"{label} contains a non-string object key")
        return {key: _validate_json_value(value[key], f"{label}.{key}") for key in sorted(value)}
    raise RosActionAdapterError(f"{label} contains unsupported JSON value type {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise RosActionAdapterError(f"cannot read {path}: {error}") from error


def _read_json(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RosActionAdapterError(f"cannot read {label} {resolved}: {error}") from error
    return resolved, raw, _object(value, label)


def _write_new_json(path: Path, value: Any) -> str:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise RosActionAdapterError(f"output path already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(raw)
        temporary.replace(resolved)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(raw)


def _ensure_new_distinct(paths: list[Path]) -> list[Path]:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RosActionAdapterError("output paths must be distinct")
    existing = [str(path) for path in resolved if path.exists()]
    if existing:
        raise RosActionAdapterError(f"output path already exists: {existing[0]}")
    return resolved


def _validate_action_declaration(functional_model: dict[str, Any], action: dict[str, Any]) -> None:
    affordances = functional_model["projections"]["affordances"]
    matches = [item for item in affordances if item["affordance_id"] == action["affordance_id"]]
    if len(matches) != 1:
        raise RosActionAdapterError(
            f"functional model does not contain exactly one {action['affordance_id']!r} affordance"
        )
    affordance = matches[0]
    if action["offered_by"] not in affordance["offered_by"]:
        raise RosActionAdapterError("action_instance.offered_by is not a provider of the selected affordance")
    if action["action_verb"] != affordance["action_verb"]:
        raise RosActionAdapterError("action_instance.action_verb does not match the selected affordance")
    if action["target_object_type"] not in affordance["target_object_types"]:
        raise RosActionAdapterError("action_instance.target_object_type is not accepted by the selected affordance")
    conditions = {item["condition_id"]: item for item in functional_model["projections"]["conditions"]}
    effects = {item["effect_id"]: item for item in functional_model["projections"]["effects"]}
    expected_arguments = {"actor", "target"}
    for condition_id in affordance["precondition_refs"]:
        expected_arguments.update(conditions[condition_id]["arguments"])
    for effect_id in affordance["effect_refs"]:
        expected_arguments.update(effects[effect_id]["arguments"])
    supplied_arguments = set(action["argument_bindings"])
    if supplied_arguments != expected_arguments:
        raise RosActionAdapterError(
            "action_instance.argument_bindings mismatch; "
            f"missing={sorted(expected_arguments - supplied_arguments)}, "
            f"extra={sorted(supplied_arguments - expected_arguments)}"
        )
    if action["argument_bindings"]["actor"] != action["offered_by"]:
        raise RosActionAdapterError("action_instance actor binding must equal offered_by")
    if action["argument_bindings"]["target"] != action["target_instance_id"]:
        raise RosActionAdapterError("action_instance target binding must equal target_instance_id")


def _parse_action_instance(value: Any, label: str) -> dict[str, Any]:
    raw = _object(value, label)
    _expect_keys(
        raw,
        {
            "action_instance_id",
            "affordance_id",
            "offered_by",
            "action_verb",
            "target_object_type",
            "target_instance_id",
            "argument_bindings",
        },
        label,
    )
    offered_by = _text(raw["offered_by"], f"{label}.offered_by")
    if not offered_by.startswith(("component/", "link/", "frame/")):
        raise RosActionAdapterError(f"{label}.offered_by must be a typed component/link/frame provider")
    return {
        "action_instance_id": _typed_id(raw["action_instance_id"], "action_instance", f"{label}.action_instance_id"),
        "affordance_id": _typed_id(raw["affordance_id"], "affordance", f"{label}.affordance_id"),
        "offered_by": offered_by,
        "action_verb": _text(raw["action_verb"], f"{label}.action_verb"),
        "target_object_type": _typed_id(
            raw["target_object_type"], "object_type", f"{label}.target_object_type"
        ),
        "target_instance_id": _typed_id(
            raw["target_instance_id"], "object_instance", f"{label}.target_instance_id"
        ),
        "argument_bindings": _string_map(raw["argument_bindings"], f"{label}.argument_bindings"),
    }


def _parse_evidence_policy(value: Any, label: str) -> dict[str, Any]:
    raw = _object(value, label)
    _expect_keys(
        raw,
        {"maximum_age_ns", "require_goal_acceptance_before_status", "require_terminal_result_status_match"},
        label,
    )
    maximum_age = _object(raw["maximum_age_ns"], f"{label}.maximum_age_ns")
    expected = {
        "operator_confirmation",
        "planner_verification",
        "project_assumption",
        "runtime_observation",
    }
    _expect_keys(maximum_age, expected, f"{label}.maximum_age_ns")
    return {
        "maximum_age_ns": {
            key: _nonnegative_int(maximum_age[key], f"{label}.maximum_age_ns.{key}")
            for key in sorted(expected)
        },
        "require_goal_acceptance_before_status": _boolean(
            raw["require_goal_acceptance_before_status"],
            f"{label}.require_goal_acceptance_before_status",
        ),
        "require_terminal_result_status_match": _boolean(
            raw["require_terminal_result_status_match"],
            f"{label}.require_terminal_result_status_match",
        ),
    }


def read_config(path: Path) -> dict[str, Any]:
    resolved, raw_bytes, raw = _read_json(path, "ROS action adapter config")
    _expect_keys(
        raw,
        {"schema_version", "adapter_id", "clock", "functional_model_binding", "ros_action", "action_instance", "evidence_policy", "policies"},
        "ROS action adapter config",
    )
    if raw["schema_version"] != CONFIG_SCHEMA:
        raise RosActionAdapterError(f"ROS action adapter config must use {CONFIG_SCHEMA}")
    binding = _object(raw["functional_model_binding"], "functional_model_binding")
    _expect_keys(
        binding,
        {"functional_model_id", "functional_model_sha256", "functional_model_artifact_sha256"},
        "functional_model_binding",
    )
    parsed_binding = {
        "functional_model_id": _typed_id(
            binding["functional_model_id"], "functional_model", "functional_model_binding.functional_model_id"
        ),
        "functional_model_sha256": _sha(
            binding["functional_model_sha256"], "functional_model_binding.functional_model_sha256"
        ),
        "functional_model_artifact_sha256": _sha(
            binding["functional_model_artifact_sha256"],
            "functional_model_binding.functional_model_artifact_sha256",
        ),
    }
    ros_action = _object(raw["ros_action"], "ros_action")
    _expect_keys(ros_action, {"name", "type", "status_topic"}, "ros_action")
    parsed_action = {
        "name": _action_name(ros_action["name"], "ros_action.name"),
        "type": _action_type(ros_action["type"], "ros_action.type"),
        "status_topic": _action_name(ros_action["status_topic"], "ros_action.status_topic"),
    }
    expected_status_topic = f"{parsed_action['name']}/_action/status"
    if parsed_action["status_topic"] != expected_status_topic:
        raise RosActionAdapterError(
            f"ros_action.status_topic must equal derived action status topic {expected_status_topic!r}"
        )
    policies = _object(raw["policies"], "policies")
    _expect_keys(
        policies,
        {"reject_multiple_status_publishers", "reject_conflicting_same_time_status"},
        "policies",
    )
    result = {
        "schema_version": CONFIG_SCHEMA,
        "adapter_id": _identifier(raw["adapter_id"], "adapter_id"),
        "clock": _clock(raw["clock"], "clock"),
        "functional_model_binding": parsed_binding,
        "ros_action": parsed_action,
        "action_instance": _parse_action_instance(raw["action_instance"], "action_instance"),
        "evidence_policy": _parse_evidence_policy(raw["evidence_policy"], "evidence_policy"),
        "policies": {
            "reject_multiple_status_publishers": _boolean(
                policies["reject_multiple_status_publishers"], "policies.reject_multiple_status_publishers"
            ),
            "reject_conflicting_same_time_status": _boolean(
                policies["reject_conflicting_same_time_status"],
                "policies.reject_conflicting_same_time_status",
            ),
        },
        "path": resolved,
        "sha256": _sha256_bytes(raw_bytes),
    }
    if not all(result["policies"].values()):
        raise RosActionAdapterError("ROS action adapter v1 requires both ambiguity-rejection policies to be true")
    return result


def _parse_status(value: Any, label: str) -> dict[str, Any]:
    raw = _object(value, label)
    _expect_keys(raw, {"goal_uuid", "accepted_at_ns", "status_code"}, label)
    status_code = _nonnegative_int(raw["status_code"], f"{label}.status_code")
    if status_code > 6:
        raise RosActionAdapterError(f"{label}.status_code must be in 0..6")
    return {
        "goal_uuid": _goal_uuid(raw["goal_uuid"], f"{label}.goal_uuid"),
        "accepted_at_ns": _nonnegative_int(raw["accepted_at_ns"], f"{label}.accepted_at_ns"),
        "status_code": status_code,
    }


def _parse_record_payload(kind: str, value: Any, label: str) -> dict[str, Any]:
    raw = _object(value, label)
    if kind in {"send_goal_request", "get_result_request"}:
        _expect_keys(raw, set(), label)
        return {}
    if kind == "goal_response":
        _expect_keys(raw, {"accepted", "server_acceptance_timestamp_ns"}, label)
        return {
            "accepted": _boolean(raw["accepted"], f"{label}.accepted"),
            "server_acceptance_timestamp_ns": _timestamp(
                raw["server_acceptance_timestamp_ns"],
                f"{label}.server_acceptance_timestamp_ns",
                optional=True,
            ),
        }
    if kind == "feedback":
        _expect_keys(raw, {"goal_uuid", "feedback"}, label)
        feedback = _object(raw["feedback"], f"{label}.feedback")
        return {
            "goal_uuid": _goal_uuid(raw["goal_uuid"], f"{label}.goal_uuid"),
            "feedback": _validate_json_value(feedback, f"{label}.feedback"),
        }
    if kind == "status_array":
        _expect_keys(raw, {"statuses"}, label)
        statuses = raw["statuses"]
        if not isinstance(statuses, list):
            raise RosActionAdapterError(f"{label}.statuses must be an array")
        return {
            "statuses": [
                _parse_status(item, f"{label}.statuses[{index}]")
                for index, item in enumerate(statuses)
            ]
        }
    if kind == "result_response":
        _expect_keys(raw, {"goal_uuid", "status_code", "result"}, label)
        status_code = _nonnegative_int(raw["status_code"], f"{label}.status_code")
        if status_code > 6:
            raise RosActionAdapterError(f"{label}.status_code must be in 0..6")
        result = _object(raw["result"], f"{label}.result")
        return {
            "goal_uuid": _goal_uuid(raw["goal_uuid"], f"{label}.goal_uuid"),
            "status_code": status_code,
            "result": _validate_json_value(result, f"{label}.result"),
        }
    raise RosActionAdapterError(f"{label} has unsupported record kind {kind!r}")


def read_capture(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    resolved, raw_bytes, raw = _read_json(path, "ROS action capture")
    _expect_keys(
        raw,
        {"schema_version", "capture_id", "adapter_config_sha256", "clock", "interval", "source", "action", "client", "records"},
        "ROS action capture",
    )
    if raw["schema_version"] != CAPTURE_SCHEMA:
        raise RosActionAdapterError(f"ROS action capture must use {CAPTURE_SCHEMA}")
    config_sha = _sha(raw["adapter_config_sha256"], "adapter_config_sha256")
    if config_sha != config["sha256"]:
        raise RosActionAdapterError(
            f"capture config digest mismatch; expected {config['sha256']}, got {config_sha}"
        )
    clock = _clock(raw["clock"], "capture.clock")
    if clock != config["clock"]:
        raise RosActionAdapterError("capture clock does not exactly match adapter config clock")
    interval = _object(raw["interval"], "capture.interval")
    _expect_keys(
        interval,
        {"started_at_ns", "requested_at_ns", "decision_time_ns", "evaluation_time_ns", "ended_at_ns", "termination_reason"},
        "capture.interval",
    )
    parsed_interval = {
        "started_at_ns": _nonnegative_int(interval["started_at_ns"], "capture.interval.started_at_ns"),
        "requested_at_ns": _nonnegative_int(interval["requested_at_ns"], "capture.interval.requested_at_ns"),
        "decision_time_ns": _nonnegative_int(interval["decision_time_ns"], "capture.interval.decision_time_ns"),
        "evaluation_time_ns": _nonnegative_int(interval["evaluation_time_ns"], "capture.interval.evaluation_time_ns"),
        "ended_at_ns": _nonnegative_int(interval["ended_at_ns"], "capture.interval.ended_at_ns"),
        "termination_reason": _text(interval["termination_reason"], "capture.interval.termination_reason"),
    }
    ordered_times = [parsed_interval[key] for key in (
        "started_at_ns", "requested_at_ns", "decision_time_ns", "evaluation_time_ns", "ended_at_ns"
    )]
    if ordered_times != sorted(ordered_times):
        raise RosActionAdapterError(
            "capture interval must satisfy started <= requested <= decision <= evaluation <= ended"
        )
    source = _object(raw["source"], "capture.source")
    _expect_keys(
        source,
        {"transport", "reference", "ros_distro", "status_publisher_identity_visibility", "service_server_identity_visibility", "feedback_publisher_identity_visibility"},
        "capture.source",
    )
    transport = _text(source["transport"], "capture.source.transport")
    if transport not in CAPTURE_TRANSPORTS:
        raise RosActionAdapterError(f"capture.source.transport must be one of {sorted(CAPTURE_TRANSPORTS)}")
    parsed_source = {
        "transport": transport,
        "reference": _optional_text(source["reference"], "capture.source.reference"),
        "ros_distro": _optional_text(source["ros_distro"], "capture.source.ros_distro"),
        "status_publisher_identity_visibility": _text(
            source["status_publisher_identity_visibility"],
            "capture.source.status_publisher_identity_visibility",
        ),
        "service_server_identity_visibility": _text(
            source["service_server_identity_visibility"],
            "capture.source.service_server_identity_visibility",
        ),
        "feedback_publisher_identity_visibility": _text(
            source["feedback_publisher_identity_visibility"],
            "capture.source.feedback_publisher_identity_visibility",
        ),
    }
    action = _object(raw["action"], "capture.action")
    _expect_keys(action, {"name", "type", "goal_uuid", "goal_payload", "goal_payload_sha256"}, "capture.action")
    goal_payload = _validate_json_value(_object(action["goal_payload"], "capture.action.goal_payload"), "capture.action.goal_payload")
    parsed_action = {
        "name": _action_name(action["name"], "capture.action.name"),
        "type": _action_type(action["type"], "capture.action.type"),
        "goal_uuid": _goal_uuid(action["goal_uuid"], "capture.action.goal_uuid"),
        "goal_payload": goal_payload,
        "goal_payload_sha256": _sha(action["goal_payload_sha256"], "capture.action.goal_payload_sha256"),
    }
    actual_goal_sha = _sha256_bytes(_canonical_bytes(goal_payload))
    if parsed_action["goal_payload_sha256"] != actual_goal_sha:
        raise RosActionAdapterError(
            f"goal payload digest mismatch; expected {actual_goal_sha}, got {parsed_action['goal_payload_sha256']}"
        )
    if {key: parsed_action[key] for key in ("name", "type")} != {
        key: config["ros_action"][key] for key in ("name", "type")
    }:
        raise RosActionAdapterError("capture action name/type do not match adapter config")
    client = _object(raw["client"], "capture.client")
    _expect_keys(client, {"node_name", "use_sim_time"}, "capture.client")
    parsed_client = {
        "node_name": _text(client["node_name"], "capture.client.node_name"),
        "use_sim_time": _boolean(client["use_sim_time"], "capture.client.use_sim_time"),
    }
    raw_records = raw["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise RosActionAdapterError("capture.records must be a non-empty array")
    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    previous_time: int | None = None
    for index, item in enumerate(raw_records):
        label = f"capture.records[{index}]"
        record = _object(item, label)
        _expect_keys(
            record,
            {"record_id", "sequence", "kind", "event_timestamp_ns", "publisher_id", "payload"},
            label,
        )
        record_id = _typed_id(record["record_id"], "ros_action_record", f"{label}.record_id")
        if record_id in record_ids:
            raise RosActionAdapterError(f"duplicate capture record ID {record_id!r}")
        record_ids.add(record_id)
        sequence = _nonnegative_int(record["sequence"], f"{label}.sequence")
        if sequence != index + 1:
            raise RosActionAdapterError("capture record sequences must be contiguous and start at 1")
        kind = _text(record["kind"], f"{label}.kind")
        if kind not in RECORD_KINDS:
            raise RosActionAdapterError(f"{label}.kind must be one of {sorted(RECORD_KINDS)}")
        event_time = _nonnegative_int(record["event_timestamp_ns"], f"{label}.event_timestamp_ns")
        if not parsed_interval["started_at_ns"] <= event_time <= parsed_interval["ended_at_ns"]:
            raise RosActionAdapterError(f"{label} timestamp lies outside the capture interval")
        if previous_time is not None and event_time < previous_time:
            raise RosActionAdapterError("capture record timestamps must be nondecreasing in sequence order")
        if event_time > parsed_interval["evaluation_time_ns"]:
            raise RosActionAdapterError(f"{label} occurs after capture evaluation_time_ns")
        previous_time = event_time
        records.append({
            "record_id": record_id,
            "sequence": sequence,
            "kind": kind,
            "event_timestamp_ns": event_time,
            "publisher_id": _optional_text(record["publisher_id"], f"{label}.publisher_id"),
            "payload": _parse_record_payload(kind, record["payload"], f"{label}.payload"),
        })
    requests = [item for item in records if item["kind"] == "send_goal_request"]
    if len(requests) != 1:
        raise RosActionAdapterError("capture must contain exactly one send_goal_request record")
    if requests[0]["event_timestamp_ns"] < parsed_interval["decision_time_ns"]:
        raise RosActionAdapterError("send_goal_request must not precede capture decision_time_ns")
    request_time = requests[0]["event_timestamp_ns"]
    target_uuid = parsed_action["goal_uuid"]
    for record in records:
        if record["kind"] in {"goal_response", "get_result_request"}:
            if record["event_timestamp_ns"] < request_time:
                raise RosActionAdapterError(f"{record['record_id']} precedes send_goal_request")
        if record["kind"] in {"feedback", "result_response"}:
            if record["payload"]["goal_uuid"] != target_uuid:
                raise RosActionAdapterError(f"{record['record_id']} binds a different goal UUID")
            if record["event_timestamp_ns"] < request_time:
                raise RosActionAdapterError(f"{record['record_id']} precedes send_goal_request")
        if record["kind"] == "status_array":
            matching = [item for item in record["payload"]["statuses"] if item["goal_uuid"] == target_uuid]
            if matching and record["event_timestamp_ns"] < request_time:
                raise RosActionAdapterError(f"{record['record_id']} reports target status before send_goal_request")
    for kind in ("goal_response", "get_result_request", "result_response"):
        if len([item for item in records if item["kind"] == kind]) > 1:
            raise RosActionAdapterError(f"capture contains more than one {kind} record")
    goal_responses = [item for item in records if item["kind"] == "goal_response"]
    get_result_requests = [item for item in records if item["kind"] == "get_result_request"]
    result_responses = [item for item in records if item["kind"] == "result_response"]
    if result_responses and not get_result_requests:
        raise RosActionAdapterError("result_response requires an observed get_result_request")
    if get_result_requests and (
        not goal_responses or not goal_responses[0]["payload"]["accepted"]
    ):
        raise RosActionAdapterError("get_result_request requires an observed accepted goal_response")
    return {
        "schema_version": CAPTURE_SCHEMA,
        "capture_id": _identifier(raw["capture_id"], "capture.capture_id"),
        "adapter_config_sha256": config_sha,
        "clock": clock,
        "interval": parsed_interval,
        "source": parsed_source,
        "action": parsed_action,
        "client": parsed_client,
        "records": records,
        "path": resolved,
        "sha256": _sha256_bytes(raw_bytes),
    }


def _validate_functional_binding(
    functional_model_path: Path,
    functional_model: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected = {
        "functional_model_id": functional_model["functional_model_id"],
        "functional_model_sha256": functional_model["functional_model_sha256"],
        "functional_model_artifact_sha256": _sha256_path(functional_model_path),
    }
    if config["functional_model_binding"] != expected:
        raise RosActionAdapterError(
            f"functional model binding mismatch; expected={expected}, supplied={config['functional_model_binding']}"
        )
    _validate_action_declaration(functional_model, config["action_instance"])


def _lifecycle_record(
    capture: dict[str, Any],
    record: dict[str, Any],
    evidence_type: str,
    value: str,
    suffix: str,
) -> dict[str, Any]:
    subject, predicate = {
        "goal_response": ("lifecycle/goal_response", "goal_response"),
        "action_status": ("lifecycle/action_status", "action_status"),
        "action_result": ("lifecycle/action_result", "action_result"),
    }[evidence_type]
    safe_capture = re.sub(r"[^A-Za-z0-9_.:-]", "_", capture["capture_id"])
    return {
        "record_id": f"evidence/ros_action/{safe_capture}/{record['sequence']:06d}-{suffix}",
        "evidence_type": evidence_type,
        "subject_ref": subject,
        "predicate": predicate,
        "bindings": {"action_instance": capture["action_instance_id"]},
        "value": value,
        "observed_at_ns": record["event_timestamp_ns"],
        "valid_until_ns": None,
        "claim_scope": "ROS 2 action-server protocol report observed by the bound action client.",
        "limitations": [
            "Action-server identity is not authenticated by this capture.",
            "The report does not establish physical execution, physical success, causal effect, safety, or authorization.",
            "The client receipt timestamp does not establish the server event time or transport latency.",
        ],
    }


def normalize(
    functional_model_path: Path,
    functional_model: dict[str, Any],
    config: dict[str, Any],
    capture: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_functional_binding(functional_model_path, functional_model, config)
    capture["action_instance_id"] = config["action_instance"]["action_instance_id"]
    goal_responses = [item for item in capture["records"] if item["kind"] == "goal_response"]
    if len(goal_responses) != 1:
        raise RosActionAdapterError(
            "normalization requires exactly one observed goal_response; preserve an incomplete capture but do not invent acceptance"
        )
    target_uuid = capture["action"]["goal_uuid"]
    lifecycle_records: list[dict[str, Any]] = []
    goal_record = goal_responses[0]
    goal_value = "accepted" if goal_record["payload"]["accepted"] else "rejected"
    lifecycle_records.append(_lifecycle_record(capture, goal_record, "goal_response", goal_value, "goal-response"))

    target_statuses: list[dict[str, Any]] = []
    ignored_goal_status_count = 0
    unknown_target_status_count = 0
    exact_duplicate_status_count = 0
    status_publishers: set[str] = set()
    status_identity_missing_records: list[str] = []
    same_time_values: dict[int, set[int]] = {}
    emitted_status_keys: set[tuple[int, int]] = set()
    for record in capture["records"]:
        if record["kind"] != "status_array":
            continue
        if record["publisher_id"] is None:
            status_identity_missing_records.append(record["record_id"])
        else:
            status_publishers.add(record["publisher_id"])
        for status_index, status in enumerate(record["payload"]["statuses"]):
            if status["goal_uuid"] != target_uuid:
                ignored_goal_status_count += 1
                continue
            code = status["status_code"]
            target_statuses.append({
                "capture_record_id": record["record_id"],
                "status_index": status_index,
                "receipt_timestamp_ns": record["event_timestamp_ns"],
                "server_accepted_at_ns": status["accepted_at_ns"],
                "status_code": code,
                "status_value": STATUS_VALUES.get(code, "unknown"),
                "publisher_id": record["publisher_id"],
            })
            if code == 0:
                unknown_target_status_count += 1
                continue
            same_time_values.setdefault(record["event_timestamp_ns"], set()).add(code)
            status_key = (record["event_timestamp_ns"], code)
            if status_key in emitted_status_keys:
                exact_duplicate_status_count += 1
                continue
            emitted_status_keys.add(status_key)
            lifecycle_records.append(
                _lifecycle_record(
                    capture,
                    record,
                    "action_status",
                    STATUS_VALUES[code],
                    f"status-{status_index:04d}-{STATUS_VALUES[code]}",
                )
            )
    conflicting_times = {
        timestamp: sorted(values)
        for timestamp, values in same_time_values.items()
        if len(values) > 1
    }
    if conflicting_times and config["policies"]["reject_conflicting_same_time_status"]:
        raise RosActionAdapterError(
            f"target goal has conflicting same-receipt-time status reports: {conflicting_times}"
        )
    if len(status_publishers) > 1 and config["policies"]["reject_multiple_status_publishers"]:
        raise RosActionAdapterError(
            f"target action status is attributed to multiple visible publishers: {sorted(status_publishers)}"
        )

    result_records = [item for item in capture["records"] if item["kind"] == "result_response"]
    result_report: dict[str, Any] | None = None
    if result_records:
        result_record = result_records[0]
        code = result_record["payload"]["status_code"]
        if code not in TERMINAL_STATUS_VALUES:
            raise RosActionAdapterError(
                f"result_response status_code must be terminal (4, 5, or 6), got {code}"
            )
        result_value = TERMINAL_STATUS_VALUES[code]
        lifecycle_records.append(
            _lifecycle_record(capture, result_record, "action_result", result_value, f"result-{result_value}")
        )
        result_report = {
            "capture_record_id": result_record["record_id"],
            "receipt_timestamp_ns": result_record["event_timestamp_ns"],
            "status_code": code,
            "status_value": result_value,
            "result_payload": result_record["payload"]["result"],
            "result_payload_sha256": _sha256_bytes(_canonical_bytes(result_record["payload"]["result"])),
            "promoted_to_effect_observation": False,
        }
    lifecycle_records.sort(key=lambda item: (item["observed_at_ns"], item["record_id"]))
    feedback_records = [item for item in capture["records"] if item["kind"] == "feedback"]
    source_id = f"evidence_source/ros_action/{config['adapter_id']}/{capture['capture_id']}"
    source = {
        "schema_version": SOURCE_SCHEMA,
        "source_id": source_id,
        "clock": config["clock"],
        "producer": {
            "producer_id": f"ros_action_server_report/{config['adapter_id']}",
            "producer_type": "ros2_action_server_report_via_digest_bound_client_capture",
        },
        "records": lifecycle_records,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "normalized",
        "adapter": {
            "adapter_id": config["adapter_id"],
            "config_path": str(config["path"]),
            "config_sha256": config["sha256"],
        },
        "capture": {
            "capture_id": capture["capture_id"],
            "capture_path": str(capture["path"]),
            "capture_sha256": capture["sha256"],
            "source": capture["source"],
            "interval": capture["interval"],
            "client": capture["client"],
        },
        "binding": {
            "functional_model": config["functional_model_binding"],
            "action_instance": config["action_instance"],
            "ros_action": config["ros_action"],
            "goal_uuid": target_uuid,
            "goal_payload_sha256": capture["action"]["goal_payload_sha256"],
        },
        "goal_response": {
            "capture_record_id": goal_record["record_id"],
            "receipt_timestamp_ns": goal_record["event_timestamp_ns"],
            "value": goal_value,
            "server_acceptance_timestamp_ns": goal_record["payload"]["server_acceptance_timestamp_ns"],
            "server_and_client_clock_offset_verified": False,
        },
        "status_normalization": {
            "target_status_reports": target_statuses,
            "ignored_other_goal_status_count": ignored_goal_status_count,
            "unknown_target_status_count": unknown_target_status_count,
            "exact_duplicate_status_count": exact_duplicate_status_count,
            "visible_publisher_ids": sorted(status_publishers),
            "publisher_identity_missing_record_ids": sorted(status_identity_missing_records),
            "unique_status_authority_verified": len(status_publishers) == 1 and not status_identity_missing_records,
            "publisher_truthfulness_verified": False,
        },
        "feedback": {
            "record_count": len(feedback_records),
            "records": [
                {
                    "capture_record_id": item["record_id"],
                    "receipt_timestamp_ns": item["event_timestamp_ns"],
                    "feedback_payload": item["payload"]["feedback"],
                    "feedback_payload_sha256": _sha256_bytes(_canonical_bytes(item["payload"]["feedback"])),
                    "publisher_id": item["publisher_id"],
                }
                for item in feedback_records
            ],
            "promoted_to_condition_or_effect_evidence": False,
        },
        "result": result_report,
        "output_counts": {
            "capture_records": len(capture["records"]),
            "lifecycle_evidence_records": len(lifecycle_records),
            "feedback_records": len(feedback_records),
            "target_status_reports": len(target_statuses),
        },
        "epistemic_boundary": {
            "goal_response": "accepted/rejected is an observed action-server protocol report, not dispatch authorization or physical execution proof",
            "status_and_result": "ROS status/result values are server reports; SUCCEEDED is not independent physical success evidence",
            "feedback": "feedback is preserved for audit and is not promoted to a declared condition or effect without a separate typed evidence adapter",
            "result_payload": "result payload is preserved and hashed but is not promoted to effect evidence",
            "time": "client event/receipt timestamps and server acceptance timestamps are kept distinct; synchronization and latency are not measured",
            "authority": "visible publisher uniqueness prevents silent status arbitration but does not authenticate or validate a server",
            "physical_and_causal": "physical execution, causal attribution, calibration, safety, and producer truthfulness are not established",
        },
    }
    return source, report


def _relative_artifact(path: Path, bundle_directory: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    root = bundle_directory.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RosActionAdapterError(f"{label} must be inside the evidence bundle directory") from error
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise RosActionAdapterError(f"{label} must be a regular relative artifact path")
    return relative.as_posix()


def _supplemental_reference(path: Path, bundle_directory: Path, clock: dict[str, str]) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RosActionAdapterError(f"supplemental evidence source must be a regular non-symlink file: {resolved}")
    _, _, value = _read_json(resolved, "supplemental evidence source")
    _expect_keys(value, {"schema_version", "source_id", "clock", "producer", "records"}, "supplemental evidence source")
    if value["schema_version"] != SOURCE_SCHEMA:
        raise RosActionAdapterError(f"supplemental evidence source must use {SOURCE_SCHEMA}")
    source_id = _typed_id(value["source_id"], "evidence_source", "supplemental evidence source_id")
    if _clock(value["clock"], "supplemental evidence source clock") != clock:
        raise RosActionAdapterError("supplemental evidence source clock does not exactly match adapter clock")
    if not isinstance(value["records"], list) or not value["records"]:
        raise RosActionAdapterError("supplemental evidence source records must be a non-empty array")
    return {
        "source_id": source_id,
        "path": _relative_artifact(resolved, bundle_directory, "supplemental evidence source"),
        "sha256": _sha256_path(resolved),
    }


def build_bundle_and_report(
    functional_model_path: Path,
    functional_model: dict[str, Any],
    config: dict[str, Any],
    capture: dict[str, Any],
    lifecycle_source: dict[str, Any],
    report: dict[str, Any],
    source_output: Path,
    bundle_output: Path,
    supplemental_sources: list[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_directory = bundle_output.expanduser().resolve().parent
    source_relative = _relative_artifact(source_output, bundle_directory, "lifecycle evidence source output")
    source_sha = _sha256_bytes(_json_bytes(lifecycle_source))
    references = [{
        "source_id": lifecycle_source["source_id"],
        "path": source_relative,
        "sha256": source_sha,
    }]
    references.extend(
        _supplemental_reference(path, bundle_directory, config["clock"])
        for path in supplemental_sources
    )
    source_ids = [item["source_id"] for item in references]
    source_paths = [item["path"] for item in references]
    if len(source_ids) != len(set(source_ids)) or len(source_paths) != len(set(source_paths)):
        raise RosActionAdapterError("evidence bundle would repeat an evidence source ID or path")
    action = {
        **config["action_instance"],
        "requested_at_ns": capture["interval"]["requested_at_ns"],
        "decision_time_ns": capture["interval"]["decision_time_ns"],
        "evaluation_time_ns": capture["interval"]["evaluation_time_ns"],
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": f"action_evidence_bundle/{config['adapter_id']}/{capture['capture_id']}",
        "functional_model_binding": config["functional_model_binding"],
        "clock": config["clock"],
        "action_instance": action,
        "evidence_policy": config["evidence_policy"],
        "evidence_sources": references,
    }
    with tempfile.TemporaryDirectory(prefix="robot-spatial-action-bundle-check-") as temporary:
        check_root = Path(temporary)
        for index, reference in enumerate(references):
            destination = check_root / reference["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if index == 0:
                destination.write_bytes(_json_bytes(lifecycle_source))
            else:
                destination.write_bytes(supplemental_sources[index - 1].expanduser().resolve().read_bytes())
        check_bundle = check_root / "bundle.json"
        check_bundle.write_bytes(_json_bytes(bundle))
        try:
            build_action_assurance(functional_model_path, check_bundle)
        except (ActionAssuranceError, OSError, ValueError) as error:
            raise RosActionAdapterError(
                f"generated action bundle failed full assurance-compiler prevalidation: {error}"
            ) from error
    bundle_sha = _sha256_bytes(_json_bytes(bundle))
    report["bundle_prevalidation"] = {
        "assurance_compiler_prevalidated": True,
        "functional_model_binding_verified": True,
        "evidence_source_schema_clock_and_record_contracts_verified": True,
        "content_digests_verified": True,
        "physical_causal_and_safety_truth_verified": False,
    }
    report["outputs"] = {
        "lifecycle_evidence_source": {
            "path": str(source_output.expanduser().resolve()),
            "source_id": lifecycle_source["source_id"],
            "sha256": source_sha,
        },
        "action_evidence_bundle": {
            "path": str(bundle_output.expanduser().resolve()),
            "bundle_id": bundle["bundle_id"],
            "sha256": bundle_sha,
        },
        "supplemental_evidence_sources": references[1:],
        "functional_model_path": str(functional_model_path.expanduser().resolve()),
        "functional_model_semantic_sha256": functional_model["functional_model_sha256"],
    }
    return bundle, report


def _mapping_arguments(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RosActionAdapterError("--argument-binding must use NAME=VALUE")
        name, item = value.split("=", 1)
        if not name or not item:
            raise RosActionAdapterError("--argument-binding must use non-empty NAME=VALUE")
        if name in result:
            raise RosActionAdapterError(f"--argument-binding repeats {name!r}")
        result[name] = item
    return dict(sorted(result.items()))


def make_config(functional_model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    try:
        functional_model = read_functional_model(functional_model_path)
    except FunctionalError as error:
        raise RosActionAdapterError(f"cannot read functional model: {error}") from error
    affordances = [
        item
        for item in functional_model["projections"]["affordances"]
        if item["affordance_id"] == args.affordance_id
    ]
    if len(affordances) != 1:
        raise RosActionAdapterError(f"functional model does not contain affordance {args.affordance_id!r}")
    affordance = affordances[0]
    bindings = {"actor": args.offered_by, "target": args.target_instance_id}
    for key, value in _mapping_arguments(args.argument_binding).items():
        if key in bindings and bindings[key] != value:
            raise RosActionAdapterError(f"--argument-binding {key} conflicts with the selected actor/target")
        bindings[key] = value
    action = {
        "action_instance_id": _typed_id(args.action_instance_id, "action_instance", "--action-instance-id"),
        "affordance_id": _typed_id(args.affordance_id, "affordance", "--affordance-id"),
        "offered_by": _text(args.offered_by, "--offered-by"),
        "action_verb": affordance["action_verb"],
        "target_object_type": _typed_id(
            args.target_object_type, "object_type", "--target-object-type"
        ),
        "target_instance_id": _typed_id(
            args.target_instance_id, "object_instance", "--target-instance-id"
        ),
        "argument_bindings": dict(sorted(bindings.items())),
    }
    _validate_action_declaration(functional_model, action)
    action_name = _action_name(args.action_name, "--action-name")
    return {
        "schema_version": CONFIG_SCHEMA,
        "adapter_id": _identifier(args.adapter_id, "--adapter-id"),
        "clock": {
            "domain": _text(args.clock_domain, "--clock-domain"),
            "unit": "nanoseconds",
            "epoch": _text(args.clock_epoch, "--clock-epoch"),
        },
        "functional_model_binding": {
            "functional_model_id": functional_model["functional_model_id"],
            "functional_model_sha256": functional_model["functional_model_sha256"],
            "functional_model_artifact_sha256": _sha256_path(functional_model_path.expanduser().resolve()),
        },
        "ros_action": {
            "name": action_name,
            "type": _action_type(args.action_type, "--action-type"),
            "status_topic": f"{action_name}/_action/status",
        },
        "action_instance": action,
        "evidence_policy": {
            "maximum_age_ns": {
                "operator_confirmation": _nonnegative_int(
                    args.maximum_operator_confirmation_age_ns,
                    "--maximum-operator-confirmation-age-ns",
                ),
                "planner_verification": _nonnegative_int(
                    args.maximum_planner_verification_age_ns,
                    "--maximum-planner-verification-age-ns",
                ),
                "project_assumption": _nonnegative_int(
                    args.maximum_project_assumption_age_ns,
                    "--maximum-project-assumption-age-ns",
                ),
                "runtime_observation": _nonnegative_int(
                    args.maximum_runtime_observation_age_ns,
                    "--maximum-runtime-observation-age-ns",
                ),
            },
            "require_goal_acceptance_before_status": True,
            "require_terminal_result_status_match": True,
        },
        "policies": {
            "reject_multiple_status_publishers": True,
            "reject_conflicting_same_time_status": True,
        },
    }


def _publisher_id(message_info: Any) -> str | None:
    if isinstance(message_info, dict):
        gid = message_info.get("publisher_gid")
    else:
        gid = getattr(message_info, "publisher_gid", None)
    if gid is None:
        return None
    if isinstance(gid, dict):
        implementation = gid.get("implementation_identifier")
        data = gid.get("data")
    else:
        implementation = getattr(gid, "implementation_identifier", None)
        data = getattr(gid, "data", gid)
    if data is None:
        return None
    try:
        raw = bytes(data)
    except (TypeError, ValueError):
        return None
    if not raw:
        return None
    prefix = f"{implementation}:" if isinstance(implementation, str) and implementation else ""
    return prefix + raw.hex()


def _plain_ros(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_ros(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain_ros(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_ros(value.tolist())
    return _validate_json_value(value, "ROS message conversion")


def execute_capture(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expected_authorization = config["action_instance"]["action_instance_id"]
    if args.authorize_dispatch != expected_authorization:
        raise RosActionAdapterError(
            "--authorize-dispatch must exactly equal the configured action_instance_id; "
            "sending a ROS action goal may move physical hardware"
        )
    _positive_float(args.server_timeout_sec, "--server-timeout-sec")
    _positive_float(args.goal_response_timeout_sec, "--goal-response-timeout-sec")
    _positive_float(args.result_timeout_sec, "--result-timeout-sec")
    _positive_float(args.settle_sec, "--settle-sec", allow_zero=True)
    _, _, goal_payload_raw = _read_json(args.goal, "ROS action goal payload")
    goal_payload = _validate_json_value(goal_payload_raw, "ROS action goal payload")
    try:
        import rclpy
        from action_msgs.msg import GoalStatusArray
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_action_status_default
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.set_message import set_message_fields
        from rosidl_runtime_py.utilities import get_action
        from unique_identifier_msgs.msg import UUID
    except ImportError as error:
        raise RosActionAdapterError(
            "execute-capture requires a sourced ROS 2 Python environment with rclpy, action_msgs, "
            "rosidl_runtime_py, and unique_identifier_msgs"
        ) from error

    records: list[dict[str, Any]] = []
    action_type = get_action(config["ros_action"]["type"])
    goal_message = action_type.Goal()
    try:
        set_message_fields(goal_message, goal_payload)
    except (AttributeError, TypeError, ValueError) as error:
        raise RosActionAdapterError(f"goal JSON does not match {config['ros_action']['type']} Goal: {error}") from error
    goal_bytes = uuid.uuid4().bytes
    goal_uuid_text = goal_bytes.hex()
    goal_uuid_message = UUID(uuid=list(goal_bytes))

    rclpy.init(args=None)
    node = rclpy.create_node(_text(args.node_name, "--node-name"))
    action_client = ActionClient(node, action_type, config["ros_action"]["name"])
    started_at = node.get_clock().now().nanoseconds
    requested_at = started_at

    def add_record(kind: str, payload: dict[str, Any], publisher_id: str | None = None) -> None:
        sequence = len(records) + 1
        records.append({
            "record_id": f"ros_action_record/{args.capture_id}/{sequence:06d}",
            "sequence": sequence,
            "kind": kind,
            "event_timestamp_ns": node.get_clock().now().nanoseconds,
            "publisher_id": publisher_id,
            "payload": payload,
        })

    def status_callback(message: Any, message_info: Any) -> None:
        statuses = []
        for status in message.status_list:
            statuses.append({
                "goal_uuid": bytes(status.goal_info.goal_id.uuid).hex(),
                "accepted_at_ns": int(status.goal_info.stamp.sec) * 1_000_000_000
                + int(status.goal_info.stamp.nanosec),
                "status_code": int(status.status),
            })
        add_record("status_array", {"statuses": statuses}, _publisher_id(message_info))

    def feedback_callback(message: Any) -> None:
        add_record(
            "feedback",
            {
                "goal_uuid": bytes(message.goal_id.uuid).hex(),
                "feedback": _plain_ros(message_to_ordereddict(message.feedback)),
            },
            None,
        )

    status_subscription = node.create_subscription(
        GoalStatusArray,
        config["ros_action"]["status_topic"],
        status_callback,
        qos_profile_action_status_default,
    )
    termination_reason = "client_error"
    decision_time = requested_at
    evaluation_time = requested_at
    try:
        if not action_client.wait_for_server(timeout_sec=args.server_timeout_sec):
            raise RosActionAdapterError(
                f"action server {config['ros_action']['name']!r} was unavailable within {args.server_timeout_sec}s"
            )
        decision_time = node.get_clock().now().nanoseconds
        add_record("send_goal_request", {})
        goal_future = action_client.send_goal_async(
            goal_message,
            feedback_callback=feedback_callback,
            goal_uuid=goal_uuid_message,
        )
        rclpy.spin_until_future_complete(node, goal_future, timeout_sec=args.goal_response_timeout_sec)
        if not goal_future.done():
            termination_reason = "goal_response_timeout"
        elif goal_future.exception() is not None:
            raise RosActionAdapterError(f"send_goal_async failed: {goal_future.exception()}")
        else:
            goal_handle = goal_future.result()
            if goal_handle is None:
                raise RosActionAdapterError("send_goal_async returned no goal handle")
            server_stamp = goal_handle.stamp
            add_record("goal_response", {
                "accepted": bool(goal_handle.accepted),
                "server_acceptance_timestamp_ns": (
                    int(server_stamp.sec) * 1_000_000_000 + int(server_stamp.nanosec)
                ),
            })
            if not goal_handle.accepted:
                termination_reason = "goal_rejected"
            else:
                add_record("get_result_request", {})
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(node, result_future, timeout_sec=args.result_timeout_sec)
                if not result_future.done():
                    termination_reason = "result_timeout"
                elif result_future.exception() is not None:
                    raise RosActionAdapterError(f"get_result_async failed: {result_future.exception()}")
                else:
                    response = result_future.result()
                    if response is None:
                        raise RosActionAdapterError("get_result_async returned no response")
                    add_record("result_response", {
                        "goal_uuid": goal_uuid_text,
                        "status_code": int(response.status),
                        "result": _plain_ros(message_to_ordereddict(response.result)),
                    })
                    termination_reason = "result_received"
        settle_deadline = time.monotonic() + args.settle_sec
        while rclpy.ok() and time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, settle_deadline - time.monotonic())))
        evaluation_time = node.get_clock().now().nanoseconds
        ended_at = evaluation_time
        use_sim_time = bool(node.get_parameter("use_sim_time").value)
    finally:
        node.destroy_subscription(status_subscription)
        action_client.destroy()
        node.destroy_node()
        rclpy.shutdown()
    return {
        "schema_version": CAPTURE_SCHEMA,
        "capture_id": _identifier(args.capture_id, "--capture-id"),
        "adapter_config_sha256": config["sha256"],
        "clock": config["clock"],
        "interval": {
            "started_at_ns": started_at,
            "requested_at_ns": requested_at,
            "decision_time_ns": decision_time,
            "evaluation_time_ns": evaluation_time,
            "ended_at_ns": ended_at,
            "termination_reason": termination_reason,
        },
        "source": {
            "transport": "live_ros2_action_client",
            "reference": args.source_reference,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "status_publisher_identity_visibility": "rclpy MessageInfo.publisher_gid requested; null means unavailable from the RMW callback",
            "service_server_identity_visibility": "unavailable through the rclpy ActionClient Future API",
            "feedback_publisher_identity_visibility": "unavailable through the rclpy ActionClient feedback callback API",
        },
        "action": {
            "name": config["ros_action"]["name"],
            "type": config["ros_action"]["type"],
            "goal_uuid": goal_uuid_text,
            "goal_payload": goal_payload,
            "goal_payload_sha256": _sha256_bytes(_canonical_bytes(goal_payload)),
        },
        "client": {
            "node_name": args.node_name,
            "use_sim_time": use_sim_time,
        },
        "records": records,
    }


def probe() -> dict[str, Any]:
    modules: dict[str, str] = {}
    for name in (
        "rclpy",
        "action_msgs.msg",
        "rosidl_runtime_py.convert",
        "rosidl_runtime_py.set_message",
        "rosidl_runtime_py.utilities",
        "unique_identifier_msgs.msg",
    ):
        try:
            __import__(name)
            modules[name] = "available"
        except ImportError:
            modules[name] = "missing"
    return {
        "schema_version": PROBE_SCHEMA,
        "deterministic_normalize": "available",
        "live_execute_capture": (
            "available" if all(value == "available" for value in modules.values()) else "unavailable"
        ),
        "python_modules": modules,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "dispatch_boundary": "execute-capture requires exact --authorize-dispatch action_instance_id and may move hardware",
        "meaning": "offline normalization remains available without ROS; probe does not verify a server, QoS delivery, hardware, or safety",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="report offline and live ROS action adapter availability")

    template = subparsers.add_parser(
        "make-config",
        help="create a functional-model-bound ROS action adapter config",
    )
    template.add_argument("functional_model", type=Path)
    template.add_argument("--adapter-id", required=True)
    template.add_argument("--clock-domain", default="ros_time")
    template.add_argument("--clock-epoch", required=True)
    template.add_argument("--action-name", required=True)
    template.add_argument("--action-type", required=True)
    template.add_argument("--action-instance-id", required=True)
    template.add_argument("--affordance-id", required=True)
    template.add_argument("--offered-by", required=True)
    template.add_argument("--target-object-type", required=True)
    template.add_argument("--target-instance-id", required=True)
    template.add_argument("--argument-binding", action="append", default=[], metavar="NAME=VALUE")
    template.add_argument("--maximum-runtime-observation-age-ns", type=int, default=100_000_000)
    template.add_argument("--maximum-planner-verification-age-ns", type=int, default=100_000_000)
    template.add_argument("--maximum-operator-confirmation-age-ns", type=int, default=100_000_000)
    template.add_argument("--maximum-project-assumption-age-ns", type=int, default=1_000_000_000)
    template.add_argument("--out", type=Path, required=True)

    live = subparsers.add_parser(
        "execute-capture",
        help="DANGEROUS: send one bound ROS 2 action goal and preserve the client exchange",
    )
    live.add_argument("--config", type=Path, required=True)
    live.add_argument("--goal", type=Path, required=True, help="JSON object matching the configured action Goal")
    live.add_argument("--capture-id", required=True)
    live.add_argument(
        "--authorize-dispatch",
        required=True,
        help="must exactly equal configured action_instance_id; the goal may move hardware",
    )
    live.add_argument("--server-timeout-sec", type=float, default=5.0)
    live.add_argument("--goal-response-timeout-sec", type=float, default=5.0)
    live.add_argument("--result-timeout-sec", type=float, default=60.0)
    live.add_argument("--settle-sec", type=float, default=0.2)
    live.add_argument("--source-reference")
    live.add_argument("--node-name", default="robot_spatial_action_capture")
    live.add_argument("--out", type=Path, required=True)

    normalizer = subparsers.add_parser(
        "normalize",
        help="compile an immutable action capture into lifecycle evidence and an assurance bundle",
    )
    normalizer.add_argument("functional_model", type=Path)
    normalizer.add_argument("--config", type=Path, required=True)
    normalizer.add_argument("--capture", type=Path, required=True)
    normalizer.add_argument("--evidence-source", type=Path, required=True)
    normalizer.add_argument("--bundle", type=Path, required=True)
    normalizer.add_argument("--report", type=Path, required=True)
    normalizer.add_argument(
        "--supplemental-source",
        type=Path,
        action="append",
        default=[],
        help="existing condition/effect evidence source inside the bundle directory; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            print(_json_bytes(probe()).decode("utf-8"), end="")
            return 0
        if args.command == "make-config":
            config = make_config(args.functional_model, args)
            digest = _write_new_json(args.out, config)
            print(_json_bytes({
                "status": "created",
                "config": str(args.out.expanduser().resolve()),
                "sha256": digest,
            }).decode("utf-8"), end="")
            return 0
        config = read_config(args.config)
        if args.command == "execute-capture":
            capture = execute_capture(config, args)
            digest = _write_new_json(args.out, capture)
            print(_json_bytes({
                "status": "captured",
                "capture": str(args.out.expanduser().resolve()),
                "sha256": digest,
                "termination_reason": capture["interval"]["termination_reason"],
                "record_count": len(capture["records"]),
            }).decode("utf-8"), end="")
            return 0
        outputs = _ensure_new_distinct([args.evidence_source, args.bundle, args.report])
        functional_model_path = args.functional_model.expanduser().resolve()
        try:
            functional_model = read_functional_model(functional_model_path)
        except FunctionalError as error:
            raise RosActionAdapterError(f"cannot read functional model: {error}") from error
        capture = read_capture(args.capture, config)
        source, report = normalize(functional_model_path, functional_model, config, capture)
        bundle, report = build_bundle_and_report(
            functional_model_path,
            functional_model,
            config,
            capture,
            source,
            report,
            outputs[0],
            outputs[1],
            args.supplemental_source,
        )
        source_sha = _write_new_json(outputs[0], source)
        bundle_sha = _write_new_json(outputs[1], bundle)
        if source_sha != report["outputs"]["lifecycle_evidence_source"]["sha256"]:
            raise RosActionAdapterError("internal lifecycle evidence digest mismatch")
        if bundle_sha != report["outputs"]["action_evidence_bundle"]["sha256"]:
            raise RosActionAdapterError("internal action evidence bundle digest mismatch")
        report_sha = _write_new_json(outputs[2], report)
        print(_json_bytes({
            "status": "normalized",
            "lifecycle_evidence_source": str(outputs[0]),
            "lifecycle_evidence_source_sha256": source_sha,
            "action_evidence_bundle": str(outputs[1]),
            "action_evidence_bundle_sha256": bundle_sha,
            "report": str(outputs[2]),
            "report_sha256": report_sha,
            "output_counts": report["output_counts"],
        }).decode("utf-8"), end="")
        return 0
    except (RosActionAdapterError, FunctionalError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
