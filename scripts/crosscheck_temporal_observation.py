#!/usr/bin/env python3
"""Independent randomized oracle for timestamp selection and effective world poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from robot_spatial import RobotModel
from temporal_observation import ObservationError, TemporalObservationLog, read_observation_query
from world_scene import SceneError, WorldScene


Matrix = list[list[float]]


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def identity() -> Matrix:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)] for row in range(4)]


def pose_matrix(record: dict[str, Any] | None) -> Matrix:
    record = record or {}
    xyz = [float(value) for value in record.get("xyz_m", [0.0, 0.0, 0.0])]
    if "quaternion_xyzw" in record:
        x, y, z, w = [float(value) for value in record["quaternion_xyzw"]]
        result = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        roll, pitch, yaw = [float(value) for value in record.get("rpy_rad", [0.0, 0.0, 0.0])]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        result = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
            [-sp, cp * sr, cp * cr, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    for axis in range(3):
        result[axis][3] = xyz[axis]
    return result


def rotation_z(angle: float) -> Matrix:
    result = identity()
    result[0][0], result[0][1] = math.cos(angle), -math.sin(angle)
    result[1][0], result[1][1] = math.sin(angle), math.cos(angle)
    return result


def translation(x: float, y: float, z: float) -> Matrix:
    result = identity()
    result[0][3], result[1][3], result[2][3] = x, y, z
    return result


def maximum_error(left: Matrix, right: Matrix) -> float:
    return max(abs(left[row][column] - right[row][column]) for row in range(4) for column in range(4))


def independent_scene(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    world = data["world_frame"]
    frames = data.get("frames", {})
    world_from: dict[str, Matrix] = {world: identity()}

    def resolve(name: str) -> Matrix:
        if name not in world_from:
            record = frames[name]
            world_from[name] = multiply(resolve(record["parent"]), pose_matrix(record.get("pose")))
        return world_from[name]

    for name in frames:
        resolve(name)
    robot = data["robot"]
    static_root = multiply(resolve(robot["parent_frame"]), pose_matrix(robot.get("pose")))
    objects = {
        name: multiply(resolve(record["parent_frame"]), pose_matrix(record.get("pose")))
        for name, record in data.get("objects", {}).items()
    }
    return {"frames": world_from, "static_root": static_root, "static_objects": objects}


def select(records: list[dict[str, Any]], query_time: int, maximum_age: int) -> dict[str, Any]:
    past = [record for record in records if record["timestamp_ns"] <= query_time]
    if not past:
        return {"status": "missing", "record": None, "age_ns": None, "future_count": len(records)}
    selected = max(past, key=lambda record: record["timestamp_ns"])
    age = query_time - selected["timestamp_ns"]
    return {
        "status": "current" if age <= maximum_age else "stale",
        "record": selected,
        "age_ns": age,
        "future_count": len(records) - len(past),
    }


def source() -> dict[str, Any]:
    return {"type": "synthetic", "reference": "independent temporal oracle", "sensor_id": None, "topic": None}


def random_pose(rng: random.Random) -> dict[str, Any]:
    return {
        "xyz_m": [rng.uniform(-3.0, 3.0) for _ in range(3)],
        "rpy_rad": [rng.uniform(-math.pi, math.pi) for _ in range(3)],
    }


def pose_sample(rng: random.Random, prefix: str, timestamp: int, parent: str) -> dict[str, Any]:
    return {
        "sample_id": f"{prefix}_{timestamp}",
        "timestamp_ns": timestamp,
        "parent_scene_frame": parent,
        "pose": random_pose(rng),
        "source": source(),
    }


def crosscheck(case_count: int, seed: int, transform_tolerance: float) -> dict[str, Any]:
    if case_count < 1:
        raise ObservationError("case count must be positive")
    rng = random.Random(seed)
    discrepancies: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    maximum_transform_error = 0.0
    comparison_count = 0
    status_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="robot-spatial-temporal-oracle-") as temp_dir:
        directory = Path(temp_dir)
        urdf_path = directory / "robot.urdf"
        urdf_path.write_text(
            '<robot name="temporal_oracle_robot">'
            '<link name="base"/><link name="arm"/><link name="tool"/>'
            '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
            '<origin xyz="0.5 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-3.141592653589793" upper="3.141592653589793" effort="1" velocity="1"/></joint>'
            '<joint name="tool_mount" type="fixed"><parent link="arm"/><child link="tool"/>'
            '<origin xyz="1 0 0" rpy="0 0 0"/></joint></robot>',
            encoding="utf-8",
        )
        model = RobotModel(urdf_path)
        scene_data = {
            "schema_version": "robot-spatial-world-scene.v1",
            "scene_id": "temporal_oracle_scene",
            "snapshot": {"id": "static_baseline", "time_semantics": "static_snapshot", "captured_at": None, "valid_until": None},
            "world_frame": "world",
            "frames": {"offset": {"parent": "world", "pose": {"xyz_m": [2.0, -1.0, 0.5], "rpy_rad": [0.0, 0.0, 0.3]}}},
            "robot": {
                "instance_id": "arm",
                "robot_name": model.name,
                "root_link": model.root_link,
                "parent_frame": "offset",
                "pose": {"xyz_m": [0.1, 0.2, 0.3], "rpy_rad": [0.1, -0.2, 0.4]},
                "source": {"type": "synthetic", "reference": "oracle", "captured_at": None},
            },
            "objects": {
                "obstacle": {
                    "parent_frame": "offset",
                    "pose": {"xyz_m": [1.0, 1.0, 1.0], "rpy_rad": [0.0, 0.0, 0.0]},
                    "source": {"type": "synthetic", "reference": "oracle", "captured_at": None},
                    "collision_geometries": [],
                }
            },
        }
        scene_path = directory / "scene.json"
        scene_path.write_text(json.dumps(scene_data), encoding="utf-8")
        scene = WorldScene(scene_path, expected_robot_name=model.name, expected_root_link=model.root_link)
        oracle_scene = independent_scene(scene_path)

        for case_index in range(case_count):
            query_time = rng.randrange(100, 10000)
            stream_timestamps = {
                name: sorted(set(rng.randrange(0, 12000) for _ in range(rng.randrange(0, 7))))
                for name in ("joint", "root", "object")
            }
            # Ensure some computable/current cases while retaining missing, stale, and future-only cases.
            if case_index % 4 == 0:
                for values in stream_timestamps.values():
                    values.append(query_time)
                    values[:] = sorted(set(values))
            ages = {name: rng.randrange(0, 3000) for name in ("joint", "root", "object")}
            root_fallback = rng.choice(["require_observed", "allow_static_declaration"])
            object_fallback = rng.choice(["require_observed", "allow_static_declaration"])
            parents = ["world", "offset"]
            joint_samples = [
                {
                    "sample_id": f"joint_{timestamp}",
                    "timestamp_ns": timestamp,
                    "positions": {"hinge": rng.uniform(-math.pi, math.pi)},
                    "source": source(),
                }
                for timestamp in stream_timestamps["joint"]
            ]
            root_samples = [pose_sample(rng, "root", timestamp, rng.choice(parents)) for timestamp in stream_timestamps["root"]]
            object_samples = [pose_sample(rng, "object", timestamp, rng.choice(parents)) for timestamp in stream_timestamps["object"]]
            log_data = {
                "schema_version": "robot-spatial-observation-log.v1",
                "observation_log_id": f"case_{case_index}",
                "clock": {"domain": "oracle", "unit": "nanoseconds", "epoch": None},
                "binding": {
                    "robot_name": model.name,
                    "root_link": model.root_link,
                    "source_urdf_semantic_sha256": model.semantic_sha256,
                    "scene_id": scene.scene_id,
                    "scene_sha256": scene.sha256,
                },
                "source": source(),
                "streams": {
                    "joint_states": joint_samples,
                    "robot_root_poses": root_samples,
                    "object_poses": {"obstacle": object_samples},
                },
            }
            query_data = {
                "schema_version": "robot-spatial-observation-query.v1",
                "query_id": f"query_{case_index}",
                "time_ns": query_time,
                "maximum_age_ns": {
                    "joint_states": ages["joint"],
                    "robot_root_pose": ages["root"],
                    "object_pose": ages["object"],
                },
                "fallbacks": {"robot_root": root_fallback, "objects": object_fallback},
                "required_object_ids": ["obstacle"],
            }
            log_path = directory / f"log_{case_index}.json"
            query_path = directory / f"query_{case_index}.json"
            log_path.write_text(json.dumps(log_data), encoding="utf-8")
            query_path.write_text(json.dumps(query_data), encoding="utf-8")

            candidate_log = TemporalObservationLog(log_path)
            candidate_query, _ = read_observation_query(query_path, scene)
            candidate = candidate_log.resolve(model, scene, candidate_query)
            independent = {
                "joint": select(joint_samples, query_time, ages["joint"]),
                "root": select(root_samples, query_time, ages["root"]),
                "object": select(object_samples, query_time, ages["object"]),
            }
            expected_current = all(record["status"] == "current" for record in independent.values())
            expected_root_available = independent["root"]["status"] == "current" or root_fallback == "allow_static_declaration"
            expected_object_available = independent["object"]["status"] == "current" or object_fallback == "allow_static_declaration"
            expected_computable = independent["joint"]["status"] == "current" and expected_root_available and expected_object_available
            checks: list[bool] = []
            report = candidate["report"]
            for key, report_key in (("joint", "joint_states"), ("root", "robot_root_pose")):
                selection = report["selections"][report_key]
                expected = independent[key]
                checks.extend([
                    selection["status"] == expected["status"],
                    selection["age_ns"] == expected["age_ns"],
                    selection["future_samples_ignored_count"] == expected["future_count"],
                    (selection["selected_sample"] or {}).get("sample_id") == (expected["record"] or {}).get("sample_id"),
                ])
            object_selection = report["selections"]["object_poses"]["obstacle"]
            checks.extend([
                object_selection["status"] == independent["object"]["status"],
                object_selection["age_ns"] == independent["object"]["age_ns"],
                object_selection["future_samples_ignored_count"] == independent["object"]["future_count"],
                (object_selection["selected_sample"] or {}).get("sample_id") == (independent["object"]["record"] or {}).get("sample_id"),
                candidate["all_required_current"] == expected_current,
                candidate["nominal_computable"] == expected_computable,
            ])
            transform_error = None
            if expected_computable:
                joint_record = independent["joint"]["record"]
                root_record = independent["root"]["record"]
                if independent["root"]["status"] == "current":
                    expected_root = multiply(
                        oracle_scene["frames"][root_record["parent_scene_frame"]],
                        pose_matrix(root_record["pose"]),
                    )
                else:
                    expected_root = oracle_scene["static_root"]
                object_record = independent["object"]["record"]
                if independent["object"]["status"] == "current":
                    expected_object = multiply(
                        oracle_scene["frames"][object_record["parent_scene_frame"]],
                        pose_matrix(object_record["pose"]),
                    )
                else:
                    expected_object = oracle_scene["static_objects"]["obstacle"]
                checks.append(maximum_error(expected_root, candidate["world_from_robot_root"]) <= transform_tolerance)
                checks.append(maximum_error(expected_object, candidate["world_from_objects"]["obstacle"]) <= transform_tolerance)
                hinge = joint_record["positions"]["hinge"]
                expected_tool = multiply(
                    expected_root,
                    multiply(translation(0.5, 0.0, 0.0), multiply(rotation_z(hinge), translation(1.0, 0.0, 0.0))),
                )
                candidate_tool = scene.typed_frames(
                    model,
                    candidate["joint_pose"],
                    world_from_robot_root=candidate["world_from_robot_root"],
                    world_from_objects=candidate["world_from_objects"],
                )["robot_frame/tool"]
                transform_error = maximum_error(expected_tool, candidate_tool)
                maximum_transform_error = max(maximum_transform_error, transform_error)
                checks.append(transform_error <= transform_tolerance)
                comparison_count += 3
            case_status = "passed" if all(checks) else "failed"
            status_counts[report["status"]] = status_counts.get(report["status"], 0) + 1
            case_record = {
                "case_index": case_index,
                "query_time_ns": query_time,
                "candidate_status": report["status"],
                "expected_all_required_current": expected_current,
                "expected_nominal_computable": expected_computable,
                "tool_transform_maximum_component_error": transform_error,
                "status": case_status,
            }
            cases.append(case_record)
            if case_status == "failed":
                discrepancies.append({
                    **case_record,
                    "candidate_selections": report["selections"],
                    "oracle_selections": independent,
                })

    return {
        "schema_version": "robot-spatial-independent-temporal-observation-crosscheck.v1",
        "status": "passed" if not discrepancies else "failed",
        "seed": seed,
        "case_count": case_count,
        "selection_stream_comparison_count": case_count * 3,
        "effective_transform_comparison_count": comparison_count,
        "status_counts": dict(sorted(status_counts.items())),
        "maximum_effective_transform_component_error": maximum_transform_error,
        "tolerance": {"transform_matrix_component": transform_tolerance},
        "engines": {
            "candidate": "TemporalObservationLog resolver plus WorldScene/RobotModel effective frames",
            "oracle": "independent latest-past selector, scene composer, and analytic one-joint FK in this script",
            "python": platform.python_version(),
        },
        "coverage": {
            "compared": [
                "latest sample at or before query time",
                "age/current/stale/missing classification",
                "future sample exclusion counts",
                "explicit static root/object fallback",
                "required-current and nominal-computable gates",
                "effective root/object transforms",
                "observation-conditioned robot tool transform",
            ],
            "not_compared": [
                "source truth or calibration",
                "clock synchronization",
                "covariance or probabilistic collision",
                "continuous motion",
                "live middleware",
            ],
        },
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "case_results_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--transform-tolerance", type=float, default=1e-10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = crosscheck(args.cases, args.seed, args.transform_tolerance)
    except (OSError, ValueError, ObservationError, SceneError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2
    serialized = json_dump(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
