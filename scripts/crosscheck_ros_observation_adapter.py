#!/usr/bin/env python3
"""Independent randomized oracle for ROS capture normalization.

This script never imports ``ros_observation_adapter``.  It generates raw partial
JointState and two-edge TF captures, computes expected snapshots with a small
independent selector/composer, invokes the public CLI, and compares artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from robot_spatial import RobotModel
from world_scene import WorldScene


HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "tests" / "fixtures"
ADAPTER = HERE / "ros_observation_adapter.py"


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_bytes(value))


def _source_record(record_id: str, timestamp: int, names: list[str], positions: list[float]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "kind": "joint_state",
        "topic": "/joint_states",
        "publisher_id": "independent_joint_oracle",
        "receipt_timestamp_ns": timestamp + 1,
        "message_timestamp_ns": timestamp,
        "names": names,
        "positions": positions,
    }


def _tf_record(record_id: str, timestamp: int, parent: str, child: str, x: float, publisher: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "kind": "tf",
        "topic": "/tf",
        "publisher_id": publisher,
        "receipt_timestamp_ns": timestamp + 1,
        "static": False,
        "transforms": [{
            "transform_id": f"{record_id}_transform",
            "message_timestamp_ns": timestamp,
            "parent_frame": parent,
            "child_frame": child,
            "pose": {"xyz_m": [x, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }],
    }


def _latest(values: list[tuple[int, float]], timestamp: int) -> tuple[int, float] | None:
    eligible = [value for value in values if value[0] <= timestamp]
    return eligible[-1] if eligible else None


def _expected_joint(records: list[dict[str, Any]], maximum_age: int) -> list[dict[str, Any]]:
    mapping = {"shoulder_encoder": "shoulder", "slide_encoder": "slide"}
    state: dict[str, tuple[int, float]] = {}
    by_time: dict[int, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (item["message_timestamp_ns"], item["receipt_timestamp_ns"], item["record_id"])):
        timestamp = record["message_timestamp_ns"]
        for name, position in zip(record["names"], record["positions"]):
            if name in mapping:
                state[mapping[name]] = (timestamp, position)
        if set(state) != {"shoulder", "slide"}:
            continue
        ages = {name: timestamp - state[name][0] for name in state}
        if any(age > maximum_age for age in ages.values()):
            continue
        by_time[timestamp] = {
            "timestamp_ns": timestamp,
            "positions": {name: state[name][1] for name in sorted(state)},
        }
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _expected_path(
    upstream: list[tuple[int, float]],
    downstream: list[tuple[int, float]],
    maximum_age: int,
) -> list[dict[str, Any]]:
    results = []
    for timestamp in sorted({time_value for time_value, _ in upstream + downstream}):
        first = _latest(upstream, timestamp)
        second = _latest(downstream, timestamp)
        if first is None or second is None:
            continue
        if timestamp - first[0] > maximum_age or timestamp - second[0] > maximum_age:
            continue
        results.append({"timestamp_ns": timestamp, "x": first[1] + second[1]})
    return results


def _expected_direct(values: list[tuple[int, float]]) -> list[dict[str, Any]]:
    return [{"timestamp_ns": timestamp, "x": value} for timestamp, value in values]


def _case(rng: random.Random, index: int, root: Path, model: RobotModel, scene: WorldScene) -> dict[str, Any]:
    joint_age = rng.randint(5, 80)
    tf_age = rng.randint(5, 80)
    config = {
        "schema_version": "robot-spatial-ros-adapter-config.v1",
        "adapter_id": f"oracle_{index:04d}",
        "clock": {"domain": "oracle_time", "unit": "nanoseconds", "epoch": "oracle_epoch"},
        "binding": {
            "robot_name": model.name,
            "root_link": model.root_link,
            "source_urdf_semantic_sha256": model.semantic_sha256,
            "scene_id": scene.scene_id,
            "scene_sha256": scene.sha256,
        },
        "topics": {"joint_states": ["/joint_states"], "tf_dynamic": ["/tf"], "tf_static": ["/tf_static"]},
        "frames": {
            "ros_reference_frame": "world",
            "scene_parent_frame": "world",
            "robot_root_frame": "base_mount",
            "objects": {"near_obstacle": "near_obstacle_tf"},
        },
        "joint_mapping": {"shoulder": "shoulder_encoder", "slide": "slide_encoder"},
        "policies": {
            "timestamp_source": "message_header",
            "joint_snapshot": {
                "maximum_component_age_ns": joint_age,
                "reject_multiple_publishers_per_joint": True,
            },
            "tf_snapshot": {
                "maximum_dynamic_edge_age_ns": tf_age,
                "reject_multiple_publishers_per_child": True,
                "reject_parent_switches": True,
                "matrix_component_tolerance": 1e-9,
            },
        },
    }
    config_path = root / "config.json"
    _write(config_path, config)

    joint_records: list[dict[str, Any]] = []
    time_value = 20
    for event_index in range(rng.randint(4, 10)):
        time_value += rng.randint(1, 45)
        choice = rng.choice(("shoulder", "slide", "both"))
        shoulder = rng.uniform(-1.0, 1.0)
        slide = rng.uniform(0.0, 1.0)
        if choice == "shoulder":
            names, positions = ["shoulder_encoder"], [shoulder]
        elif choice == "slide":
            names, positions = ["slide_encoder"], [slide]
        else:
            names, positions = ["shoulder_encoder", "slide_encoder"], [shoulder, slide]
        joint_records.append(_source_record(f"case{index:04d}_joint{event_index:03d}", time_value, names, positions))

    def series(count: int, start: int) -> list[tuple[int, float]]:
        values: list[tuple[int, float]] = []
        cursor = start
        for _ in range(count):
            cursor += rng.randint(1, 55)
            values.append((cursor, rng.uniform(-2.0, 2.0)))
        return values

    upstream = series(rng.randint(2, 6), 25)
    downstream = series(rng.randint(2, 6), 30)
    object_values = series(rng.randint(2, 5), 35)
    tf_records = [
        *[
            _tf_record(f"case{index:04d}_up{event_index:03d}", timestamp, "world", "tracking", x, "upstream_pub")
            for event_index, (timestamp, x) in enumerate(upstream)
        ],
        *[
            _tf_record(f"case{index:04d}_down{event_index:03d}", timestamp, "tracking", "base_mount", x, "downstream_pub")
            for event_index, (timestamp, x) in enumerate(downstream)
        ],
        *[
            _tf_record(f"case{index:04d}_obj{event_index:03d}", timestamp, "world", "near_obstacle_tf", x, "object_pub")
            for event_index, (timestamp, x) in enumerate(object_values)
        ],
    ]
    records = [*joint_records, *tf_records]
    maximum_receipt = max(record["receipt_timestamp_ns"] for record in records)
    capture = {
        "schema_version": "robot-spatial-ros-capture.v1",
        "capture_id": f"case_{index:04d}",
        "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "clock": config["clock"],
        "capture": {"started_timestamp_ns": 0, "ended_timestamp_ns": maximum_receipt + 1, "node_use_sim_time": True},
        "source": {
            "transport": "synthetic_fixture",
            "reference": "independent randomized oracle",
            "ros_distro": None,
            "authority_visibility": "explicit independent publisher IDs",
        },
        "records": records,
    }
    capture_path = root / "capture.json"
    _write(capture_path, capture)
    output_path = root / "observations.json"
    report_path = root / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ADAPTER),
            "normalize",
            str(model.path),
            "--scene",
            str(scene.path),
            "--config",
            str(config_path),
            "--capture",
            str(capture_path),
            "--out",
            str(output_path),
            "--report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"case": index, "kind": "cli_error", "stderr": completed.stderr.strip()}
    actual = json.loads(output_path.read_text(encoding="utf-8"))
    expected_joint = _expected_joint(joint_records, joint_age)
    expected_root = _expected_path(upstream, downstream, tf_age)
    expected_object = _expected_direct(object_values)
    actual_joint = [
        {"timestamp_ns": sample["timestamp_ns"], "positions": sample["positions"]}
        for sample in actual["streams"]["joint_states"]
    ]
    actual_root = [
        {"timestamp_ns": sample["timestamp_ns"], "x": sample["pose"]["xyz_m"][0]}
        for sample in actual["streams"]["robot_root_poses"]
    ]
    actual_object = [
        {"timestamp_ns": sample["timestamp_ns"], "x": sample["pose"]["xyz_m"][0]}
        for sample in actual["streams"]["object_poses"]["near_obstacle"]
    ]

    def numeric_equal(left: Any, right: Any) -> bool:
        if isinstance(left, float) or isinstance(right, float):
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-10)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(numeric_equal(a, b) for a, b in zip(left, right))
        if isinstance(left, dict) and isinstance(right, dict):
            return set(left) == set(right) and all(numeric_equal(left[key], right[key]) for key in left)
        return left == right

    discrepancies = []
    for kind, observed, expected in (
        ("joint", actual_joint, expected_joint),
        ("root_tf", actual_root, expected_root),
        ("object_tf", actual_object, expected_object),
    ):
        if not numeric_equal(observed, expected):
            discrepancies.append({"kind": kind, "actual": observed, "expected": expected})
    provenance = actual.get("normalization", {})
    if provenance.get("capture_sha256") != hashlib.sha256(capture_path.read_bytes()).hexdigest():
        discrepancies.append({"kind": "capture_digest_binding"})
    if provenance.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        discrepancies.append({"kind": "config_digest_binding"})
    return {
        "case": index,
        "joint_event_count": len(joint_records),
        "tf_event_count": len(tf_records),
        "joint_snapshot_count": len(actual_joint),
        "root_snapshot_count": len(actual_root),
        "object_snapshot_count": len(actual_object),
        "discrepancies": discrepancies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    model = RobotModel(FIXTURES / "two_dof.urdf")
    scene = WorldScene(FIXTURES / "world_scene.json", expected_robot_name=model.name, expected_root_link=model.root_link)
    rng = random.Random(args.seed)
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        for index in range(args.cases):
            case_root = base / f"case-{index:04d}"
            case_root.mkdir()
            results.append(_case(rng, index, case_root, model, scene))
    discrepancies = [item for result in results for item in result.get("discrepancies", [])]
    cli_errors = [result for result in results if result.get("kind") == "cli_error"]
    report = {
        "schema_version": "robot-spatial-ros-adapter-crosscheck.v1",
        "status": "passed" if not discrepancies and not cli_errors else "failed",
        "seed": args.seed,
        "case_count": args.cases,
        "joint_event_count": sum(result.get("joint_event_count", 0) for result in results),
        "tf_event_count": sum(result.get("tf_event_count", 0) for result in results),
        "joint_snapshot_comparisons": sum(result.get("joint_snapshot_count", 0) for result in results),
        "root_tf_snapshot_comparisons": sum(result.get("root_snapshot_count", 0) for result in results),
        "object_tf_snapshot_comparisons": sum(result.get("object_snapshot_count", 0) for result in results),
        "discrepancy_count": len(discrepancies),
        "cli_error_count": len(cli_errors),
        "failures": [result for result in results if result.get("discrepancies") or result.get("kind") == "cli_error"][:20],
        "method": "independent latest-component joint assembly and translation-only two-edge TF composition compared against subprocess CLI output",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
