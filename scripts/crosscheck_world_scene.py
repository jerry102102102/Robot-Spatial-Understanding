#!/usr/bin/env python3
"""Independent world-scene crosschecks for mounting, gravity loads, and box collision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from crosscheck_yourdfpy import CrosscheckError, generate_poses
from robot_spatial import RobotModel, SpatialError, scene_gravity_load_analysis
from world_scene import SceneError, WorldScene


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _pose_matrix(numpy: Any, record: dict[str, Any] | None) -> Any:
    record = record or {}
    xyz = numpy.asarray(record.get("xyz_m", [0.0, 0.0, 0.0]), dtype=float)
    if "quaternion_xyzw" in record:
        x, y, z, w = (float(value) for value in record["quaternion_xyzw"])
        rotation = numpy.asarray([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])
    else:
        roll, pitch, yaw = (float(value) for value in record.get("rpy_rad", [0.0, 0.0, 0.0]))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = numpy.asarray([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
    transform = numpy.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def independent_scene_binding(path: Path) -> dict[str, Any]:
    """Parse the scene transform tree independently from world_scene.WorldScene."""
    try:
        import numpy
    except ImportError as error:
        raise CrosscheckError("NumPy is required in the active Python environment") from error
    data = json.loads(path.read_text(encoding="utf-8"))
    world = data["world_frame"]
    frames = data.get("frames", {})
    world_from: dict[str, Any] = {world: numpy.eye(4)}
    active: set[str] = set()

    def resolve(name: str) -> Any:
        if name in world_from:
            return world_from[name]
        if name in active:
            raise CrosscheckError(f"oracle scene cycle at {name!r}")
        active.add(name)
        record = frames[name]
        world_from[name] = resolve(record["parent"]) @ _pose_matrix(numpy, record.get("pose"))
        active.remove(name)
        return world_from[name]

    for name in frames:
        resolve(name)
    robot = data["robot"]
    world_from_root = resolve(robot["parent_frame"]) @ _pose_matrix(numpy, robot.get("pose"))
    gravity_record = data.get("gravity")
    if gravity_record is None:
        gravity_world = None
        gravity_root = None
    else:
        gravity_declared = numpy.asarray(gravity_record["vector_xyz_m_s2"], dtype=float)
        gravity_world = resolve(gravity_record["expressed_in_frame"])[:3, :3] @ gravity_declared
        gravity_root = world_from_root[:3, :3].T @ gravity_world
    objects: dict[str, Any] = {}
    for object_id, record in data.get("objects", {}).items():
        world_from_object = resolve(record["parent_frame"]) @ _pose_matrix(numpy, record.get("pose"))
        objects[object_id] = {
            "world_from_object": world_from_object,
            "collision_geometries": {
                geometry["id"]: {
                    "world_from_geometry": world_from_object @ _pose_matrix(numpy, geometry.get("pose")),
                    "geometry": geometry["geometry"],
                }
                for geometry in record.get("collision_geometries", [])
            },
        }
    return {
        "data": data,
        "world_from_frames": world_from,
        "world_from_root": world_from_root,
        "gravity_world": gravity_world,
        "gravity_root": gravity_root,
        "objects": objects,
    }


def _oracle_potential_relative_to_root(
    oracle: Any,
    root_link: str,
    world_from_root: Any,
    gravity_world: Any,
) -> float:
    """Compute U=-sum(m*g dot (p_world-p_root)) via independent URDF/FK."""
    import numpy

    root_origin_world = world_from_root[:3, 3]
    energy = 0.0
    for link in oracle.robot.links:
        if link.inertial is None:
            continue
        root_from_link = oracle.get_transform(link.name, root_link)
        link_from_inertial = link.inertial.origin if link.inertial.origin is not None else numpy.eye(4)
        world_from_inertial = world_from_root @ root_from_link @ link_from_inertial
        relative_world = world_from_inertial[:3, 3] - root_origin_world
        energy -= float(link.inertial.mass) * float(gravity_world @ relative_world)
    return energy


def crosscheck_scene_gravity(
    urdf_path: Path,
    scene_path: Path,
    pose_count: int,
    seed: int,
    finite_difference_step: float,
    load_tolerance: float,
    potential_tolerance_j: float,
    transform_tolerance: float,
    source_url: str | None,
    source_revision: str | None,
) -> dict[str, Any]:
    try:
        import numpy
        import yourdfpy
        from yourdfpy import URDF
    except ImportError as error:
        raise CrosscheckError("yourdfpy and NumPy are required in the active Python environment") from error
    model = RobotModel(urdf_path)
    scene = WorldScene(scene_path, expected_robot_name=model.name, expected_root_link=model.root_link)
    independent_scene = independent_scene_binding(scene_path)
    candidate_root = numpy.asarray(scene.world_from_robot_root, dtype=float)
    root_transform_error = float(numpy.max(numpy.abs(candidate_root - independent_scene["world_from_root"])))
    candidate_gravity = numpy.asarray(scene.gravity_in_robot_root()["vector_in_robot_root_xyz_m_s2"], dtype=float)
    gravity_error = float(numpy.max(numpy.abs(candidate_gravity - independent_scene["gravity_root"])))
    oracle = URDF.load(str(urdf_path.resolve()), load_meshes=False, load_collision_meshes=False)
    if oracle.base_link != model.root_link:
        raise CrosscheckError(f"root mismatch: robot-spatial={model.root_link!r}, yourdfpy={oracle.base_link!r}")
    poses, sampling = generate_poses(model, pose_count, seed)
    discrepancies: list[dict[str, Any]] = []
    if root_transform_error > transform_tolerance:
        discrepancies.append({"quantity": "world_from_robot_root", "maximum_component_error": root_transform_error})
    if gravity_error > transform_tolerance:
        discrepancies.append({"quantity": "gravity_in_robot_root", "maximum_component_error": gravity_error})
    pose_results: list[dict[str, Any]] = []
    comparison_count = 0
    maximum_load_error = 0.0
    maximum_potential_error = 0.0
    for pose_index, pose in enumerate(poses):
        candidate = scene_gravity_load_analysis(model, scene, pose, f"pose_{pose_index}")
        if candidate["status"] != "computed":
            raise CrosscheckError(f"candidate scene gravity loads are {candidate['status']!r}")
        candidate_loads = candidate["loads"]
        oracle.update_cfg(pose)
        expected_potential = _oracle_potential_relative_to_root(
            oracle,
            model.root_link,
            independent_scene["world_from_root"],
            independent_scene["gravity_world"],
        )
        potential_error = abs(candidate_loads["modeled_potential_energy_relative_to_root_origin_j"] - expected_potential)
        maximum_potential_error = max(maximum_potential_error, potential_error)
        driver_results: dict[str, Any] = {}
        for driver in candidate_loads["independent_driver_order"]:
            plus, minus = dict(pose), dict(pose)
            plus[driver] = plus.get(driver, 0.0) + finite_difference_step
            minus[driver] = minus.get(driver, 0.0) - finite_difference_step
            oracle.update_cfg(plus)
            plus_energy = _oracle_potential_relative_to_root(
                oracle,
                model.root_link,
                independent_scene["world_from_root"],
                independent_scene["gravity_world"],
            )
            oracle.update_cfg(minus)
            minus_energy = _oracle_potential_relative_to_root(
                oracle,
                model.root_link,
                independent_scene["world_from_root"],
                independent_scene["gravity_world"],
            )
            expected_force = -(plus_energy - minus_energy) / (2.0 * finite_difference_step)
            actual_force = candidate_loads["independent_driver_loads"][driver]["generalized_gravity_force"]
            error = abs(actual_force - expected_force)
            maximum_load_error = max(maximum_load_error, error)
            comparison_count += 1
            passed = error <= load_tolerance
            driver_results[driver] = {
                "candidate_generalized_gravity_force": actual_force,
                "oracle_negative_world_potential_derivative": expected_force,
                "absolute_error": error,
                "unit": candidate_loads["independent_driver_loads"][driver]["unit"],
                "status": "passed" if passed else "failed",
            }
            if not passed:
                discrepancies.append({
                    "pose_index": pose_index,
                    "joint": driver,
                    "candidate": actual_force,
                    "oracle": expected_force,
                    "absolute_error": error,
                })
        if potential_error > potential_tolerance_j:
            discrepancies.append({
                "pose_index": pose_index,
                "quantity": "root-relative world potential energy",
                "candidate_j": candidate_loads["modeled_potential_energy_relative_to_root_origin_j"],
                "oracle_j": expected_potential,
                "absolute_error_j": potential_error,
            })
        pose_results.append({
            "pose_index": pose_index,
            "pose_sha256": hashlib.sha256(json.dumps(pose, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "potential_energy_absolute_error_j": potential_error,
            "driver_results": driver_results,
            "status": "passed" if potential_error <= potential_tolerance_j and all(record["status"] == "passed" for record in driver_results.values()) else "failed",
        })
        oracle.update_cfg(pose)
    return {
        "schema_version": "robot-spatial-cross-engine-world-scene-gravity.v1",
        "status": "passed" if not discrepancies else "failed",
        "robot": model.name,
        "root_link": model.root_link,
        "scene": {
            "scene_id": scene.scene_id,
            "snapshot_id": scene.snapshot["id"],
            "path": str(scene_path.resolve()),
            "sha256": scene.sha256,
        },
        "source": {
            "urdf_path": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "upstream_url": source_url,
            "upstream_revision": source_revision,
        },
        "engines": {
            "candidate": {"name": "robot-spatial WorldScene plus static_gravity_loads"},
            "oracle": {"name": "independent JSON frame parser plus yourdfpy world-frame potential central finite difference", "yourdfpy_version": importlib.metadata.version("yourdfpy"), "numpy_version": numpy.__version__},
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "scene_binding": {
            "maximum_world_from_root_matrix_component_error": root_transform_error,
            "maximum_root_gravity_component_error_m_s2": gravity_error,
        },
        "sampling": {"pose_count": len(poses), "seed": seed, **sampling},
        "coverage": {
            "independent_driver_comparison_count": comparison_count,
            "compared": [
                "scene frame composition and world_from_robot_root",
                "world gravity rotation into the robot root",
                "root-relative potential energy in world coordinates",
                "generalized gravity force for every independent driver as -dU_world/dq",
            ],
            "not_compared": ["physical scene calibration or currency", "payload", "contacts", "motion dynamics", "hardware or controller behavior"],
        },
        "finite_difference_step_joint_units": finite_difference_step,
        "tolerances": {
            "transform_or_gravity_component": transform_tolerance,
            "generalized_force_or_torque": load_tolerance,
            "potential_energy_j": potential_tolerance_j,
        },
        "maximum_generalized_force_or_torque_error": maximum_load_error,
        "maximum_potential_energy_error_j": maximum_potential_error,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "pose_results": pose_results,
    }


def _obb_overlap(numpy: Any, left_transform: Any, left_size: Any, right_transform: Any, right_size: Any, tolerance: float) -> bool:
    """Independent 15-axis separating-axis test for two oriented boxes."""
    left_axes = left_transform[:3, :3]
    right_axes = right_transform[:3, :3]
    left_half = numpy.asarray(left_size, dtype=float) / 2.0
    right_half = numpy.asarray(right_size, dtype=float) / 2.0
    rotation = left_axes.T @ right_axes
    absolute = numpy.abs(rotation) + 1e-12
    translation = left_axes.T @ (right_transform[:3, 3] - left_transform[:3, 3])
    for axis in range(3):
        left_radius = left_half[axis]
        right_radius = float(right_half @ absolute[axis, :])
        if abs(translation[axis]) > left_radius + right_radius + tolerance:
            return False
    for axis in range(3):
        left_radius = float(left_half @ absolute[:, axis])
        right_radius = right_half[axis]
        projected = abs(float(translation @ rotation[:, axis]))
        if projected > left_radius + right_radius + tolerance:
            return False
    for left_axis in range(3):
        for right_axis in range(3):
            left_radius = left_half[(left_axis + 1) % 3] * absolute[(left_axis + 2) % 3, right_axis] + left_half[(left_axis + 2) % 3] * absolute[(left_axis + 1) % 3, right_axis]
            right_radius = right_half[(right_axis + 1) % 3] * absolute[left_axis, (right_axis + 2) % 3] + right_half[(right_axis + 2) % 3] * absolute[left_axis, (right_axis + 1) % 3]
            projected = abs(translation[(left_axis + 2) % 3] * rotation[(left_axis + 1) % 3, right_axis] - translation[(left_axis + 1) % 3] * rotation[(left_axis + 2) % 3, right_axis])
            if projected > left_radius + right_radius + tolerance:
                return False
    return True


def crosscheck_box_collisions(case_count: int, seed: int, tolerance: float, clearance_tolerance: float) -> dict[str, Any]:
    try:
        import numpy
    except ImportError as error:
        raise CrosscheckError("NumPy is required in the active Python environment") from error
    if case_count < 1:
        raise CrosscheckError("case count must be positive")
    rng = random.Random(seed)
    robot_size = [1.0, 0.6, 0.8]
    discrepancies: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    maximum_clearance_error = 0.0
    analytic_clearance_count = 0
    with tempfile.TemporaryDirectory(prefix="robot-spatial-world-box-") as temp_dir:
        directory = Path(temp_dir)
        urdf_path = directory / "box_robot.urdf"
        urdf_path.write_text(
            '<robot name="box_oracle_robot"><link name="base"><collision><geometry><box size="1.0 0.6 0.8"/></geometry></collision></link></robot>',
            encoding="utf-8",
        )
        model = RobotModel(urdf_path)
        for index in range(case_count):
            if index == 0:
                root_pose = {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
                object_pose = {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
                object_size = [3.0, 3.0, 3.0]
                analytic_clearance = None
            elif index % 5 == 0:
                clearance = rng.uniform(0.05, 2.0)
                object_size = [rng.uniform(0.2, 1.2), 0.4, 0.5]
                center_x = robot_size[0] / 2.0 + object_size[0] / 2.0 + clearance
                root_pose = {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
                object_pose = {"xyz_m": [center_x, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
                analytic_clearance = clearance
            else:
                root_pose = {
                    "xyz_m": [rng.uniform(-0.7, 0.7) for _ in range(3)],
                    "rpy_rad": [rng.uniform(-math.pi, math.pi) for _ in range(3)],
                }
                object_pose = {
                    "xyz_m": [rng.uniform(-1.5, 1.5) for _ in range(3)],
                    "rpy_rad": [rng.uniform(-math.pi, math.pi) for _ in range(3)],
                }
                object_size = [rng.uniform(0.2, 1.4) for _ in range(3)]
                analytic_clearance = None
            scene_data = {
                "schema_version": "robot-spatial-world-scene.v1",
                "scene_id": f"box_case_{index}",
                "snapshot": {"id": f"snapshot_{index}", "time_semantics": "static_snapshot", "captured_at": None, "valid_until": None},
                "world_frame": "world",
                "frames": {},
                "robot": {
                    "instance_id": "box",
                    "robot_name": "box_oracle_robot",
                    "root_link": "base",
                    "parent_frame": "world",
                    "pose": root_pose,
                    "source": {"type": "synthetic", "reference": "crosscheck", "captured_at": None},
                },
                "objects": {
                    "obstacle": {
                        "parent_frame": "world",
                        "pose": object_pose,
                        "source": {"type": "synthetic", "reference": "crosscheck", "captured_at": None},
                        "collision_geometries": [{"id": "body", "geometry": {"type": "box", "size_xyz_m": object_size}}],
                    }
                },
            }
            scene_path = directory / f"scene_{index}.json"
            scene_path.write_text(json.dumps(scene_data), encoding="utf-8")
            independent = independent_scene_binding(scene_path)
            left_transform = independent["world_from_root"]
            right_transform = independent["objects"]["obstacle"]["collision_geometries"]["body"]["world_from_geometry"]
            oracle_collision = _obb_overlap(numpy, left_transform, robot_size, right_transform, object_size, tolerance)
            scene = WorldScene(scene_path, expected_robot_name=model.name, expected_root_link=model.root_link)
            candidate = scene.robot_environment_collisions(model, {}, contact_tolerance_m=tolerance)
            candidate_collision = candidate["status"] == "collision"
            passed = candidate["status"] in {"collision", "collision_free"} and candidate_collision == oracle_collision
            clearance_error = None
            if analytic_clearance is not None:
                analytic_clearance_count += 1
                candidate_clearance = candidate["minimum_separation"].get("distance_m")
                if candidate_clearance is None:
                    passed = False
                else:
                    clearance_error = abs(candidate_clearance - analytic_clearance)
                    maximum_clearance_error = max(maximum_clearance_error, clearance_error)
                    passed = passed and clearance_error <= clearance_tolerance
            case_record = {
                "case_index": index,
                "candidate_status": candidate["status"],
                "oracle_obb_sat_collision": oracle_collision,
                "analytic_axis_aligned_clearance_m": analytic_clearance,
                "candidate_minimum_separation_m": candidate["minimum_separation"].get("distance_m"),
                "clearance_absolute_error_m": clearance_error,
                "status": "passed" if passed else "failed",
            }
            cases.append(case_record)
            if not passed:
                discrepancies.append(case_record)
    return {
        "schema_version": "robot-spatial-cross-engine-world-scene-box-collision.v1",
        "status": "passed" if not discrepancies else "failed",
        "engines": {
            "candidate": {"name": "robot-spatial exact box triangles plus closed-solid containment"},
            "oracle": {"name": "independent NumPy 15-axis OBB separating-axis theorem and analytic axis-aligned clearance", "numpy_version": numpy.__version__},
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "sampling": {"case_count": case_count, "seed": seed, "containment_case_count": 1, "analytic_axis_aligned_clearance_case_count": analytic_clearance_count},
        "coverage": {
            "compared": ["rotated oriented-box solid collision", "closed-solid containment", "axis-aligned exact positive clearance"],
            "not_compared": ["mesh solids", "cylinders", "physical scene truth", "dynamic obstacles"],
        },
        "tolerances": {"contact_m": tolerance, "clearance_m": clearance_tolerance},
        "maximum_clearance_error_m": maximum_clearance_error,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gravity = subparsers.add_parser("gravity")
    gravity.add_argument("urdf", type=Path)
    gravity.add_argument("--scene", type=Path, required=True)
    gravity.add_argument("--poses", type=int, default=16)
    gravity.add_argument("--seed", type=int, default=20260718)
    gravity.add_argument("--finite-difference-step", type=float, default=1e-6)
    gravity.add_argument("--load-tolerance", type=float, default=2e-5)
    gravity.add_argument("--potential-tolerance-j", type=float, default=1e-9)
    gravity.add_argument("--transform-tolerance", type=float, default=1e-10)
    gravity.add_argument("--source-url")
    gravity.add_argument("--source-revision")
    gravity.add_argument("--out", type=Path)
    boxes = subparsers.add_parser("boxes")
    boxes.add_argument("--cases", type=int, default=128)
    boxes.add_argument("--seed", type=int, default=20260718)
    boxes.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    boxes.add_argument("--clearance-tolerance-m", type=float, default=1e-9)
    boxes.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "gravity":
            result = crosscheck_scene_gravity(
                args.urdf,
                args.scene,
                args.poses,
                args.seed,
                args.finite_difference_step,
                args.load_tolerance,
                args.potential_tolerance_j,
                args.transform_tolerance,
                args.source_url,
                args.source_revision,
            )
        else:
            result = crosscheck_box_collisions(
                args.cases,
                args.seed,
                args.contact_tolerance_m,
                args.clearance_tolerance_m,
            )
    except (OSError, ValueError, SpatialError, SceneError, CrosscheckError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json_dump(result), encoding="utf-8")
    print(json_dump(result), end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
