"""Versioned simulator-neutral run artifacts and completeness checks."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import IntegrityError, SchemaError
from .util import (
    ensure_new_directory,
    find_forbidden_keys,
    finite_number,
    load_json,
    quaternion_normalized,
    require_list,
    require_mapping,
    require_string,
    safe_relative_path,
    sha256_file,
    sha256_json,
    write_deterministic_npz,
    write_json,
)


RUN_SCHEMA = "robot-spatial-simulation-run.v1"
GENERIC_TRACE_SCHEMA = "robot-spatial-generic-trace.v1"
COMPLETENESS_SCHEMA = "robot-spatial-simulation-completeness.v1"
STANDARD_CHANNELS = (
    "joint_state",
    "pose",
    "odometry",
    "contact",
    "collision",
    "force_torque",
    "deformable",
)


def _as_time(value: Any, label: str) -> float:
    result = finite_number(value, label)
    if result < 0.0:
        raise SchemaError(f"{label} must be non-negative")
    return result


def _vector(value: Any, length: int, label: str) -> list[float]:
    entries = require_list(value, label)
    if len(entries) != length:
        raise SchemaError(f"{label} must contain {length} values")
    return [finite_number(entry, f"{label}[{index}]") for index, entry in enumerate(entries)]


def _times_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "time_s" not in arrays:
        raise SchemaError("stream does not contain time_s")
    times = np.asarray(arrays["time_s"], dtype=np.float64)
    if times.ndim != 1:
        raise SchemaError("stream time_s must be one-dimensional")
    return times


def _rows_conflict(arrays: dict[str, np.ndarray], left: int, right: int) -> bool:
    for name, values in arrays.items():
        if name == "time_s" or values.ndim == 0 or values.shape[0] != len(arrays["time_s"]):
            continue
        a = values[left]
        b = values[right]
        if np.issubdtype(values.dtype, np.floating):
            if not np.array_equal(a, b, equal_nan=True):
                return True
        elif not np.array_equal(a, b):
            return True
    return False


def _channel_completeness(
    name: str,
    arrays: dict[str, np.ndarray],
    interval: dict[str, float],
    max_gap_s: float,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    times = _times_from_arrays(arrays)
    sample_count = int(times.size)
    if sample_count == 0:
        issues.append({"type": "no_samples"})
    if not np.all(np.isfinite(times)):
        issues.append({"type": "invalid_time"})
    for index in range(1, sample_count):
        if float(times[index]) < float(times[index - 1]):
            issues.append(
                {
                    "type": "out_of_order",
                    "left_index": index - 1,
                    "right_index": index,
                    "left_time_s": float(times[index - 1]),
                    "right_time_s": float(times[index]),
                }
            )
        elif float(times[index]) == float(times[index - 1]):
            same_event_identity = True
            if name in {"contact", "collision"}:
                left_pair = {
                    str(arrays["body_a"][index - 1]),
                    str(arrays["body_b"][index - 1]),
                }
                right_pair = {
                    str(arrays["body_a"][index]),
                    str(arrays["body_b"][index]),
                }
                same_event_identity = left_pair == right_pair
            if same_event_identity and _rows_conflict(arrays, index - 1, index):
                issues.append(
                    {
                        "type": "conflicting_duplicate",
                        "left_index": index - 1,
                        "right_index": index,
                        "time_s": float(times[index]),
                    }
                )
    if sample_count > 1 and max_gap_s > 0.0:
        gaps = np.diff(times)
        for index in np.flatnonzero(gaps > max_gap_s):
            issues.append(
                {
                    "type": "gap",
                    "left_index": int(index),
                    "right_index": int(index + 1),
                    "start_s": float(times[index]),
                    "end_s": float(times[index + 1]),
                    "gap_s": float(gaps[index]),
                    "max_gap_s": max_gap_s,
                }
            )
    if sample_count:
        if float(times[0]) > interval["start_s"] + max_gap_s:
            issues.append(
                {
                    "type": "incomplete_start_coverage",
                    "expected_start_s": interval["start_s"],
                    "first_sample_s": float(times[0]),
                }
            )
        if float(times[-1]) < interval["end_s"] - max_gap_s:
            issues.append(
                {
                    "type": "incomplete_end_coverage",
                    "expected_end_s": interval["end_s"],
                    "last_sample_s": float(times[-1]),
                }
            )
    presence_masks = {
        "position": "position_present",
        "velocity": "velocity_present",
        "effort": "effort_present",
        "position_m": "present",
        "quaternion_xyzw": "present",
        "linear_velocity_mps": "linear_velocity_present",
        "angular_velocity_radps": "angular_velocity_present",
        "force_n": "present",
        "torque_nm": "present",
        "normal_force_n": "normal_force_present",
        "keypoints_m": "keypoint_present",
    }
    for array_name, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and array_name in presence_masks:
            mask_name = presence_masks[array_name]
            if mask_name in arrays:
                mask = np.asarray(arrays[mask_name], dtype=np.bool_)
                expanded = mask
                while expanded.ndim < values.ndim:
                    expanded = np.expand_dims(expanded, axis=-1)
                expanded = np.broadcast_to(expanded, values.shape)
                invalid_present = np.logical_and(expanded, np.isnan(values))
                if invalid_present.any():
                    issues.append(
                        {
                            "type": "missing_values",
                            "array": array_name,
                            "count": int(invalid_present.sum()),
                        }
                    )
                # A channel can omit an optional field entirely. Once one entity provides it,
                # holes for that same entity are completeness gaps rather than silent NaNs.
                if mask.ndim >= 2:
                    active_entities = np.any(mask, axis=0)
                    missing_mask = np.logical_and(~mask, np.broadcast_to(active_entities, mask.shape))
                    if missing_mask.any():
                        issues.append(
                            {
                                "type": "missing_values",
                                "array": mask_name,
                                "count": int(missing_mask.sum()),
                            }
                        )
            elif np.isnan(values).any():
                issues.append(
                    {
                        "type": "missing_values",
                        "array": array_name,
                        "count": int(np.isnan(values).sum()),
                    }
                )
        if array_name == "quaternion_xyzw" and values.size:
            norms = np.linalg.norm(values, axis=-1)
            invalid = np.logical_or(~np.isfinite(norms), np.abs(norms - 1.0) > 1e-6)
            if "present" in arrays and arrays["present"].shape == invalid.shape:
                invalid = np.logical_and(invalid, arrays["present"])
            if invalid.any():
                issues.append({"type": "invalid_quaternion", "count": int(invalid.sum())})
    invalid_types = {"invalid_time", "out_of_order", "conflicting_duplicate", "invalid_quaternion"}
    status = "invalid" if any(issue["type"] in invalid_types for issue in issues) else "incomplete" if issues else "complete"
    return {
        "channel": name,
        "status": status,
        "sample_count": sample_count,
        "observed_interval": None
        if sample_count == 0
        else {"start_s": float(times[0]), "end_s": float(times[-1])},
        "max_gap_s": max_gap_s,
        "issues": issues,
    }


def evaluate_completeness(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    interval_raw = require_mapping(manifest.get("interval"), "run.interval")
    interval = {
        "start_s": _as_time(interval_raw.get("start_s"), "run.interval.start_s"),
        "end_s": _as_time(interval_raw.get("end_s"), "run.interval.end_s"),
    }
    if interval["end_s"] < interval["start_s"]:
        raise SchemaError("run.interval.end_s must be at or after start_s")
    channels = require_mapping(manifest.get("channels"), "run.channels")
    results: dict[str, Any] = {}
    for name in STANDARD_CHANNELS:
        channel = require_mapping(channels.get(name), f"run.channels.{name}")
        if channel.get("status") != "available":
            results[name] = {
                "channel": name,
                "status": "unavailable",
                "sample_count": 0,
                "observed_interval": None,
                "max_gap_s": None,
                "issues": [{"type": "unavailable", "reason": channel.get("reason", "not supplied")}],
            }
            continue
        path = safe_relative_path(root, require_string(channel.get("path"), f"channels.{name}.path"), f"channels.{name}.path")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key]) for key in archive.files}
        default_gap = max(float(manifest.get("timestep_s", 0.0)) * 2.5, 1e-9)
        max_gap_s = finite_number(channel.get("max_gap_s", default_gap), f"channels.{name}.max_gap_s")
        results[name] = _channel_completeness(name, arrays, interval, max_gap_s)
    available = [result for result in results.values() if result["status"] != "unavailable"]
    overall = "invalid" if any(result["status"] == "invalid" for result in available) else "incomplete" if any(result["status"] == "incomplete" for result in available) else "complete_for_available_channels"
    report = {
        "schema_version": COMPLETENESS_SCHEMA,
        "run_id": manifest["run_id"],
        "run_manifest_sha256": manifest["manifest_sha256"],
        "status": overall,
        "channels": results,
        "limitations": [
            "Completeness describes recorded channels and declared sampling only; it does not prove sensor truth, simulator correctness, or physical safety."
        ],
    }
    report["completeness_sha256"] = sha256_json(report)
    return report


def _joint_arrays(samples: list[Any]) -> dict[str, np.ndarray]:
    records = [require_mapping(record, f"samples.joint_state[{index}]") for index, record in enumerate(samples)]
    joint_ids = sorted(
        {
            str(joint)
            for record in records
            for field in ("positions", "velocities", "efforts")
            for joint in require_mapping(record.get(field, {}), f"joint_state.{field}")
        }
    )
    if not joint_ids:
        raise SchemaError("joint_state samples do not contain any joint IDs")
    index_by_joint = {joint: index for index, joint in enumerate(joint_ids)}
    shape = (len(records), len(joint_ids))
    arrays: dict[str, np.ndarray] = {
        "time_s": np.empty(len(records), dtype=np.float64),
        "joint_ids": np.asarray(joint_ids, dtype="U"),
        "position": np.full(shape, np.nan, dtype=np.float64),
        "velocity": np.full(shape, np.nan, dtype=np.float64),
        "effort": np.full(shape, np.nan, dtype=np.float64),
        "position_present": np.zeros(shape, dtype=np.bool_),
        "velocity_present": np.zeros(shape, dtype=np.bool_),
        "effort_present": np.zeros(shape, dtype=np.bool_),
    }
    for row, record in enumerate(records):
        arrays["time_s"][row] = _as_time(record.get("time_s"), f"joint_state[{row}].time_s")
        for source_name, output_name in (("positions", "position"), ("velocities", "velocity"), ("efforts", "effort")):
            for joint, value in require_mapping(record.get(source_name, {}), f"joint_state[{row}].{source_name}").items():
                column = index_by_joint[str(joint)]
                arrays[output_name][row, column] = finite_number(value, f"joint_state[{row}].{source_name}.{joint}")
                arrays[f"{output_name}_present"][row, column] = True
    return arrays


def _pose_arrays(samples: list[Any], label: str = "pose") -> dict[str, np.ndarray]:
    records = [require_mapping(record, f"samples.{label}[{index}]") for index, record in enumerate(samples)]
    entity_ids = sorted(
        {
            str(entity)
            for record in records
            for entity in require_mapping(record.get("entities"), f"{label}.entities")
        }
    )
    if not entity_ids:
        raise SchemaError(f"{label} samples do not contain any entity IDs")
    index_by_entity = {entity: index for index, entity in enumerate(entity_ids)}
    shape = (len(records), len(entity_ids))
    arrays = {
        "time_s": np.empty(len(records), dtype=np.float64),
        "entity_ids": np.asarray(entity_ids, dtype="U"),
        "position_m": np.full((*shape, 3), np.nan, dtype=np.float64),
        "quaternion_xyzw": np.full((*shape, 4), np.nan, dtype=np.float64),
        "present": np.zeros(shape, dtype=np.bool_),
        "linear_velocity_mps": np.full((*shape, 3), np.nan, dtype=np.float64),
        "angular_velocity_radps": np.full((*shape, 3), np.nan, dtype=np.float64),
        "linear_velocity_present": np.zeros(shape, dtype=np.bool_),
        "angular_velocity_present": np.zeros(shape, dtype=np.bool_),
    }
    for row, record in enumerate(records):
        arrays["time_s"][row] = _as_time(record.get("time_s"), f"{label}[{row}].time_s")
        for entity, raw_pose in require_mapping(record.get("entities"), f"{label}[{row}].entities").items():
            pose = require_mapping(raw_pose, f"{label}[{row}].entities.{entity}")
            column = index_by_entity[str(entity)]
            arrays["position_m"][row, column] = _vector(
                pose.get("position_m"), 3, f"{label}[{row}].entities.{entity}.position_m"
            )
            arrays["quaternion_xyzw"][row, column] = quaternion_normalized(
                _vector(pose.get("quaternion_xyzw"), 4, f"{label}[{row}].entities.{entity}.quaternion_xyzw")
            )
            arrays["present"][row, column] = True
            for source_name, output_name in (
                ("linear_velocity_mps", "linear_velocity_mps"),
                ("angular_velocity_radps", "angular_velocity_radps"),
            ):
                if source_name in pose:
                    arrays[output_name][row, column] = _vector(
                        pose[source_name], 3, f"{label}[{row}].entities.{entity}.{source_name}"
                    )
                    mask_name = (
                        "linear_velocity_present"
                        if output_name == "linear_velocity_mps"
                        else "angular_velocity_present"
                    )
                    arrays[mask_name][row, column] = True
    return arrays


def _pair_event_arrays(samples: list[Any], label: str) -> dict[str, np.ndarray]:
    records = [require_mapping(record, f"samples.{label}[{index}]") for index, record in enumerate(samples)]
    arrays: dict[str, np.ndarray] = {
        "time_s": np.empty(len(records), dtype=np.float64),
        "body_a": np.empty(len(records), dtype="U128"),
        "body_b": np.empty(len(records), dtype="U128"),
        "active": np.empty(len(records), dtype=np.bool_),
    }
    if label == "contact":
        arrays["normal_force_n"] = np.full(len(records), np.nan, dtype=np.float64)
        arrays["normal_force_present"] = np.zeros(len(records), dtype=np.bool_)
        arrays["force_n"] = np.full((len(records), 3), np.nan, dtype=np.float64)
        arrays["present"] = np.zeros(len(records), dtype=np.bool_)
    for row, record in enumerate(records):
        arrays["time_s"][row] = _as_time(record.get("time_s"), f"{label}[{row}].time_s")
        arrays["body_a"][row] = require_string(record.get("body_a"), f"{label}[{row}].body_a")
        arrays["body_b"][row] = require_string(record.get("body_b"), f"{label}[{row}].body_b")
        if not isinstance(record.get("active"), bool):
            raise SchemaError(f"{label}[{row}].active must be boolean")
        arrays["active"][row] = record["active"]
        if label == "contact" and "normal_force_n" in record:
            arrays["normal_force_n"][row] = finite_number(record["normal_force_n"], f"contact[{row}].normal_force_n")
            arrays["normal_force_present"][row] = True
        if label == "contact" and "force_n" in record:
            arrays["force_n"][row] = _vector(record["force_n"], 3, f"contact[{row}].force_n")
            arrays["present"][row] = True
    return arrays


def _odometry_arrays(samples: list[Any]) -> dict[str, np.ndarray]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(samples):
        record = require_mapping(raw, f"samples.odometry[{index}]")
        entity = require_string(record.get("entity"), f"odometry[{index}].entity")
        normalized.append(
            {
                "time_s": record.get("time_s"),
                "entities": {
                    entity: {
                        "position_m": record.get("position_m"),
                        "quaternion_xyzw": record.get("quaternion_xyzw"),
                    }
                },
                "linear_velocity_mps": record.get("linear_velocity_mps"),
                "angular_velocity_radps": record.get("angular_velocity_radps"),
            }
        )
    arrays = _pose_arrays(normalized, "odometry")
    entity_ids = list(arrays["entity_ids"])
    index_by_entity = {str(entity): index for index, entity in enumerate(entity_ids)}
    shape = (len(normalized), len(entity_ids), 3)
    arrays["linear_velocity_mps"] = np.full(shape, np.nan, dtype=np.float64)
    arrays["angular_velocity_radps"] = np.full(shape, np.nan, dtype=np.float64)
    for row, record in enumerate(samples):
        entity = str(record["entity"])
        column = index_by_entity[entity]
        if "linear_velocity_mps" in record:
            arrays["linear_velocity_mps"][row, column] = _vector(record["linear_velocity_mps"], 3, f"odometry[{row}].linear_velocity_mps")
            arrays["linear_velocity_present"][row, column] = True
        if "angular_velocity_radps" in record:
            arrays["angular_velocity_radps"][row, column] = _vector(record["angular_velocity_radps"], 3, f"odometry[{row}].angular_velocity_radps")
            arrays["angular_velocity_present"][row, column] = True
    return arrays


def _force_torque_arrays(samples: list[Any]) -> dict[str, np.ndarray]:
    records = [require_mapping(record, f"samples.force_torque[{index}]") for index, record in enumerate(samples)]
    sensor_ids = sorted({require_string(record.get("sensor"), "force_torque.sensor") for record in records})
    index_by_sensor = {sensor: index for index, sensor in enumerate(sensor_ids)}
    shape = (len(records), len(sensor_ids), 3)
    arrays = {
        "time_s": np.empty(len(records), dtype=np.float64),
        "sensor_ids": np.asarray(sensor_ids, dtype="U"),
        "force_n": np.full(shape, np.nan, dtype=np.float64),
        "torque_nm": np.full(shape, np.nan, dtype=np.float64),
        "present": np.zeros(shape[:2], dtype=np.bool_),
    }
    for row, record in enumerate(records):
        arrays["time_s"][row] = _as_time(record.get("time_s"), f"force_torque[{row}].time_s")
        column = index_by_sensor[str(record["sensor"])]
        arrays["force_n"][row, column] = _vector(record.get("force_n"), 3, f"force_torque[{row}].force_n")
        arrays["torque_nm"][row, column] = _vector(record.get("torque_nm"), 3, f"force_torque[{row}].torque_nm")
        arrays["present"][row, column] = True
    return arrays


def _deformable_arrays(samples: list[Any]) -> dict[str, np.ndarray]:
    records = [require_mapping(record, f"samples.deformable[{index}]") for index, record in enumerate(samples)]
    entity_ids = sorted({require_string(record.get("entity"), "deformable.entity") for record in records})
    keypoint_count = max((len(require_list(record.get("keypoints_m"), "deformable.keypoints_m")) for record in records), default=0)
    if not entity_ids or keypoint_count == 0:
        raise SchemaError("deformable samples require entity and keypoints_m")
    index_by_entity = {entity: index for index, entity in enumerate(entity_ids)}
    arrays = {
        "time_s": np.empty(len(records), dtype=np.float64),
        "entity_ids": np.asarray(entity_ids, dtype="U"),
        "keypoints_m": np.full((len(records), len(entity_ids), keypoint_count, 3), np.nan, dtype=np.float64),
        "keypoint_present": np.zeros((len(records), len(entity_ids), keypoint_count), dtype=np.bool_),
    }
    for row, record in enumerate(records):
        arrays["time_s"][row] = _as_time(record.get("time_s"), f"deformable[{row}].time_s")
        column = index_by_entity[str(record["entity"])]
        for keypoint, value in enumerate(require_list(record.get("keypoints_m"), f"deformable[{row}].keypoints_m")):
            arrays["keypoints_m"][row, column, keypoint] = _vector(value, 3, f"deformable[{row}].keypoints_m[{keypoint}]")
            arrays["keypoint_present"][row, column, keypoint] = True
    return arrays


ARRAY_BUILDERS = {
    "joint_state": _joint_arrays,
    "pose": _pose_arrays,
    "odometry": _odometry_arrays,
    "contact": lambda samples: _pair_event_arrays(samples, "contact"),
    "collision": lambda samples: _pair_event_arrays(samples, "collision"),
    "force_torque": _force_torque_arrays,
    "deformable": _deformable_arrays,
}


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})


@dataclass
class SimulationRun:
    """Loaded, digest-verified `simulation-run.v1` directory."""

    root: Path
    manifest: dict[str, Any]
    completeness: dict[str, Any]
    _streams: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: str | Path, *, verify_digests: bool = True) -> "SimulationRun":
        directory = Path(root)
        manifest_path = directory / "run.json"
        completeness_path = directory / "completeness.json"
        manifest = require_mapping(load_json(manifest_path), "run")
        if manifest.get("schema_version") != RUN_SCHEMA:
            raise SchemaError(f"run schema must be {RUN_SCHEMA!r}")
        require_string(manifest.get("run_id"), "run.run_id")
        forbidden = find_forbidden_keys(manifest)
        if forbidden:
            raise IntegrityError(f"run manifest contains prohibited oracle fields: {forbidden}")
        expected_manifest_digest = _manifest_digest(manifest)
        if verify_digests and manifest.get("manifest_sha256") != expected_manifest_digest:
            raise IntegrityError("run.json manifest_sha256 does not match its canonical content")
        channels = require_mapping(manifest.get("channels"), "run.channels")
        for name in STANDARD_CHANNELS:
            channel = require_mapping(channels.get(name), f"run.channels.{name}")
            if channel.get("status") == "available":
                path = safe_relative_path(directory, require_string(channel.get("path"), f"channels.{name}.path"), f"channels.{name}.path")
                if not path.is_file():
                    raise IntegrityError(f"available channel file is missing: {path}")
                if verify_digests and sha256_file(path) != channel.get("sha256"):
                    raise IntegrityError(f"channel {name!r} digest mismatch")
            elif channel.get("status") != "unavailable":
                raise SchemaError(f"channels.{name}.status must be available or unavailable")
        events = require_mapping(manifest.get("events"), "run.events")
        events_path = safe_relative_path(directory, require_string(events.get("path"), "run.events.path"), "run.events.path")
        if verify_digests and sha256_file(events_path) != events.get("sha256"):
            raise IntegrityError("events.jsonl digest mismatch")
        completeness = require_mapping(load_json(completeness_path), "completeness")
        if completeness.get("schema_version") != COMPLETENESS_SCHEMA:
            raise SchemaError(f"completeness schema must be {COMPLETENESS_SCHEMA!r}")
        expected_completeness_digest = sha256_json(
            {key: value for key, value in completeness.items() if key != "completeness_sha256"}
        )
        if verify_digests and completeness.get("completeness_sha256") != expected_completeness_digest:
            raise IntegrityError("completeness.json digest mismatch")
        if completeness.get("run_manifest_sha256") != manifest["manifest_sha256"]:
            raise IntegrityError("completeness report is bound to a different run manifest")
        return cls(directory, manifest, completeness)

    @classmethod
    def import_generic_trace(
        cls,
        source: str | Path,
        out: str | Path,
        *,
        adapter_name: str = "generic-json",
        adapter_version: str = "0.2.0",
    ) -> "SimulationRun":
        source_path = Path(source)
        trace = require_mapping(load_json(source_path), "generic trace")
        if trace.get("schema_version") != GENERIC_TRACE_SCHEMA:
            raise SchemaError(f"generic trace schema must be {GENERIC_TRACE_SCHEMA!r}")
        forbidden = find_forbidden_keys(trace)
        if forbidden:
            raise IntegrityError(
                "raw trace contains official outcome/reward fields that would leak the oracle: " + ", ".join(forbidden)
            )
        output = Path(out)
        ensure_new_directory(output)
        try:
            samples = require_mapping(trace.get("samples", {}), "trace.samples")
            timestep_s = finite_number(trace.get("timestep_s"), "trace.timestep_s")
            if timestep_s <= 0.0:
                raise SchemaError("trace.timestep_s must be positive")
            channels: dict[str, Any] = {}
            all_times: list[float] = []
            channel_policies = require_mapping(trace.get("channel_policies", {}), "trace.channel_policies")
            for name in STANDARD_CHANNELS:
                raw_samples = samples.get(name)
                if raw_samples is None:
                    channels[name] = {"status": "unavailable", "reason": "channel not supplied by source"}
                    continue
                records = require_list(raw_samples, f"trace.samples.{name}")
                if not records:
                    channels[name] = {"status": "unavailable", "reason": "source supplied an empty channel"}
                    continue
                arrays = ARRAY_BUILDERS[name](records)
                all_times.extend(float(value) for value in arrays["time_s"] if math.isfinite(float(value)))
                relative = f"streams/{name}.npz"
                path = output / relative
                write_deterministic_npz(path, arrays)
                policy = require_mapping(channel_policies.get(name, {}), f"trace.channel_policies.{name}")
                channels[name] = {
                    "status": "available",
                    "path": relative,
                    "sha256": sha256_file(path),
                    "arrays": sorted(arrays),
                    "max_gap_s": float(
                        policy.get("max_gap_s", timestep_s * 2.5)
                    ),
                }
            events_raw = require_list(trace.get("events", []), "trace.events")
            events_path = output / "events.jsonl"
            event_lines: list[str] = []
            for index, raw_event in enumerate(events_raw):
                event = require_mapping(raw_event, f"trace.events[{index}]")
                _as_time(event.get("time_s"), f"trace.events[{index}].time_s")
                require_string(event.get("type"), f"trace.events[{index}].type")
                event_lines.append(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                all_times.append(float(event["time_s"]))
            events_path.write_text("".join(line + "\n" for line in event_lines), encoding="utf-8")
            if not all_times:
                raise SchemaError("trace contains no timestamped samples or events")
            declared_interval = require_mapping(trace.get("interval", {}), "trace.interval")
            interval = {
                "start_s": _as_time(declared_interval.get("start_s", min(all_times)), "trace.interval.start_s"),
                "end_s": _as_time(declared_interval.get("end_s", max(all_times)), "trace.interval.end_s"),
            }
            if interval["end_s"] < interval["start_s"]:
                raise SchemaError("trace interval ends before it starts")
            clock = require_mapping(trace.get("clock"), "trace.clock")
            conventions = require_mapping(trace.get("conventions"), "trace.conventions")
            expected_conventions = {
                "length_unit": "m",
                "angle_unit": "rad",
                "quaternion_order": "xyzw",
                "pose_direction": "world_from_entity",
            }
            for key, expected in expected_conventions.items():
                if conventions.get(key) != expected:
                    raise SchemaError(f"trace.conventions.{key} must be {expected!r}")
            manifest: dict[str, Any] = {
                "schema_version": RUN_SCHEMA,
                "run_id": require_string(trace.get("run_id"), "trace.run_id"),
                "simulator": require_mapping(trace.get("simulator"), "trace.simulator"),
                "adapter": {"name": adapter_name, "version": adapter_version, "oracle_fields_accepted": False},
                "seed": trace.get("seed"),
                "timestep_s": timestep_s,
                "clock": clock,
                "interval": interval,
                "task_id": require_string(trace.get("task_id"), "trace.task_id"),
                "intervention": require_mapping(
                    trace.get("intervention", {"type": "unspecified"}),
                    "trace.intervention",
                ),
                "robot": require_mapping(trace.get("robot"), "trace.robot"),
                "world": require_mapping(trace.get("world"), "trace.world"),
                "conventions": conventions,
                "channels": channels,
                "events": {
                    "path": "events.jsonl",
                    "sha256": sha256_file(events_path),
                    "count": len(event_lines),
                },
                "source": {
                    "type": GENERIC_TRACE_SCHEMA,
                    "sha256": sha256_file(source_path),
                },
                "assets": require_list(trace.get("assets", []), "trace.assets"),
                "boundaries": {
                    "simulation_only": True,
                    "official_reward_or_success_imported": False,
                    "hardware_truth_established": False,
                    "causation_established": False,
                    "safety_established": False,
                },
            }
            for label in ("name", "version"):
                require_string(manifest["simulator"].get(label), f"trace.simulator.{label}")
            require_string(manifest["clock"].get("clock_id"), "trace.clock.clock_id")
            require_string(manifest["clock"].get("domain"), "trace.clock.domain")
            manifest["manifest_sha256"] = _manifest_digest(manifest)
            write_json(output / "run.json", manifest)
            completeness = evaluate_completeness(output, manifest)
            write_json(output / "completeness.json", completeness)
            return cls.load(output)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise

    def channel_available(self, name: str) -> bool:
        return require_mapping(self.manifest["channels"].get(name), f"channels.{name}").get("status") == "available"

    def stream(self, name: str) -> dict[str, np.ndarray]:
        if name in self._streams:
            return self._streams[name]
        channel = require_mapping(self.manifest["channels"].get(name), f"channels.{name}")
        if channel.get("status") != "available":
            raise SchemaError(f"channel {name!r} is unavailable: {channel.get('reason', 'not supplied')}")
        path = safe_relative_path(self.root, channel["path"], f"channels.{name}.path")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key]) for key in archive.files}
        self._streams[name] = arrays
        return arrays

    def channel_completeness(self, name: str) -> dict[str, Any]:
        return require_mapping(self.completeness["channels"].get(name), f"completeness.channels.{name}")

    def events(self) -> list[dict[str, Any]]:
        event_path = safe_relative_path(self.root, self.manifest["events"]["path"], "events.path")
        events: list[dict[str, Any]] = []
        for index, line in enumerate(event_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"events.jsonl line {index + 1} is invalid JSON: {error}") from error
            events.append(require_mapping(event, f"events[{index}]"))
        forbidden = find_forbidden_keys(events)
        if forbidden:
            raise IntegrityError(f"events contain prohibited oracle fields: {forbidden}")
        return events

    @property
    def digest(self) -> str:
        return str(self.manifest["manifest_sha256"])
