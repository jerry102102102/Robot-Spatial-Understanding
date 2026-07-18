#!/usr/bin/env python3
"""Cross-check link forward kinematics against the independent yourdfpy engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any

from robot_spatial import EPSILON, RobotModel, SpatialError, clean_number


class CrosscheckError(ValueError):
    """A cross-engine setup or comparison error."""


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _driver_range(model: RobotModel, driver: str) -> tuple[float, float, list[dict[str, Any]]]:
    joint = model.joints[driver]
    lower, upper = joint.limit["lower"], joint.limit["upper"]
    source = "urdf_limits"
    if joint.type == "continuous":
        lower, upper, source = -math.pi, math.pi, "canonical_continuous_cycle"
    elif lower is None and upper is None:
        lower, upper, source = -0.5, 0.5, "crosscheck_fallback_unbounded"
    elif lower is None:
        upper = float(upper)
        lower, source = min(0.0, upper - 1.0), "crosscheck_fallback_missing_lower"
    elif upper is None:
        lower = float(lower)
        upper, source = max(0.0, lower + 1.0), "crosscheck_fallback_missing_upper"
    else:
        lower, upper = float(lower), float(upper)
    constraints: list[dict[str, Any]] = [{"joint": driver, "source": source, "lower": lower, "upper": upper}]
    for joint_name, candidate in model.joints.items():
        candidate_driver, multiplier, offset, chain = model._mimic_affine_from_driver(joint_name)
        if candidate_driver != driver or joint_name == driver or candidate.type == "continuous":
            continue
        declared_lower, declared_upper = candidate.limit["lower"], candidate.limit["upper"]
        if declared_lower is None and declared_upper is None:
            continue
        constraint: dict[str, Any] = {
            "joint": joint_name,
            "mimic_chain": chain,
            "multiplier": multiplier,
            "offset": offset,
            "declared_lower": declared_lower,
            "declared_upper": declared_upper,
        }
        constraints.append(constraint)
        if abs(multiplier) <= EPSILON:
            if (declared_lower is not None and offset < declared_lower - EPSILON) or (declared_upper is not None and offset > declared_upper + EPSILON):
                raise CrosscheckError(f"mimic joint {joint_name!r} has an infeasible constant position")
            continue
        transformed: list[float] = []
        if declared_lower is not None:
            transformed.append((float(declared_lower) - offset) / multiplier)
        if declared_upper is not None:
            transformed.append((float(declared_upper) - offset) / multiplier)
        if len(transformed) == 2:
            candidate_lower, candidate_upper = min(transformed), max(transformed)
        elif declared_lower is not None:
            boundary = transformed[0]
            candidate_lower, candidate_upper = (boundary, math.inf) if multiplier > 0.0 else (-math.inf, boundary)
        else:
            boundary = transformed[0]
            candidate_lower, candidate_upper = (-math.inf, boundary) if multiplier > 0.0 else (boundary, math.inf)
        lower, upper = max(lower, candidate_lower), min(upper, candidate_upper)
    if lower > upper + EPSILON:
        raise CrosscheckError(f"driver {driver!r} has no feasible crosscheck range")
    return lower, upper, constraints


def generate_poses(model: RobotModel, count: int, seed: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if count < 1 or count > 10000:
        raise CrosscheckError("pose count must be between 1 and 10000")
    drivers = [name for name, joint in model.joints.items() if joint.type != "fixed" and joint.mimic is None]
    ranges: dict[str, tuple[float, float]] = {}
    constraints: dict[str, Any] = {}
    for driver in drivers:
        lower, upper, records = _driver_range(model, driver)
        ranges[driver] = (lower, upper)
        constraints[driver] = records
    poses: list[dict[str, float]] = []
    baseline = {name: min(max(0.0, bounds[0]), bounds[1]) for name, bounds in ranges.items()}
    poses.append(baseline)
    if count > 1:
        poses.append({name: (bounds[0] + bounds[1]) / 2.0 for name, bounds in ranges.items()})
    generator = random.Random(seed)
    while len(poses) < count:
        poses.append({name: generator.uniform(*bounds) if bounds[1] > bounds[0] else bounds[0] for name, bounds in ranges.items()})
    for pose in poses:
        model.resolve_pose(pose)
    return poses, {
        "independent_joint_order": drivers,
        "ranges": {name: {"minimum": clean_number(bounds[0]), "maximum": clean_number(bounds[1])} for name, bounds in ranges.items()},
        "constraints": constraints,
    }


def rotation_error_deg(ours: list[list[float]], oracle: Any) -> float:
    relative = [
        [sum(ours[index][row] * float(oracle[index, column]) for index in range(3)) for column in range(3)]
        for row in range(3)
    ]
    cosine = max(-1.0, min(1.0, (sum(relative[index][index] for index in range(3)) - 1.0) / 2.0))
    sine = 0.5 * math.sqrt(
        (relative[2][1] - relative[1][2]) ** 2
        + (relative[0][2] - relative[2][0]) ** 2
        + (relative[1][0] - relative[0][1]) ** 2
    )
    return math.degrees(math.atan2(sine, cosine))


def crosscheck(
    urdf_path: Path,
    pose_count: int,
    seed: int,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
    source_url: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    if translation_tolerance_m < 0.0 or rotation_tolerance_deg < 0.0:
        raise CrosscheckError("tolerances must be non-negative")
    try:
        import numpy
        import yourdfpy
        from yourdfpy import URDF
    except ImportError as error:
        raise CrosscheckError("yourdfpy and NumPy are required in the active Python environment") from error
    model = RobotModel(urdf_path)
    oracle = URDF.load(str(urdf_path.resolve()), load_meshes=False, load_collision_meshes=False)
    if oracle.base_link != model.root_link:
        raise CrosscheckError(f"root mismatch: robot-spatial={model.root_link!r}, yourdfpy={oracle.base_link!r}")
    poses, sampling = generate_poses(model, pose_count, seed)
    pose_results: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    global_translation_error = 0.0
    global_rotation_error = 0.0
    comparisons = 0
    for pose_index, pose in enumerate(poses):
        ours_frames, resolved_pose = model.world_frames(pose)
        oracle.update_cfg(pose)
        pose_translation_error = 0.0
        pose_rotation_error = 0.0
        worst_translation_frame = model.root_link
        worst_rotation_frame = model.root_link
        for link_name in sorted(model.links):
            ours = ours_frames[link_name]
            other = oracle.get_transform(link_name, model.root_link)
            translation_error = math.sqrt(sum((ours[index][3] - float(other[index, 3])) ** 2 for index in range(3)))
            orientation_error = rotation_error_deg(ours, other)
            comparisons += 1
            if translation_error > pose_translation_error:
                pose_translation_error, worst_translation_frame = translation_error, link_name
            if orientation_error > pose_rotation_error:
                pose_rotation_error, worst_rotation_frame = orientation_error, link_name
            if translation_error > translation_tolerance_m or orientation_error > rotation_tolerance_deg:
                discrepancies.append({
                    "pose_index": pose_index,
                    "link": link_name,
                    "translation_error_m": translation_error,
                    "rotation_error_deg": orientation_error,
                })
        global_translation_error = max(global_translation_error, pose_translation_error)
        global_rotation_error = max(global_rotation_error, pose_rotation_error)
        pose_results.append({
            "pose_index": pose_index,
            "pose_sha256": hashlib.sha256(json.dumps(pose, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "supplied_independent_joint_positions": {name: clean_number(value) for name, value in pose.items()},
            "resolved_joint_positions": {name: clean_number(value) for name, value in resolved_pose.items()},
            "maximum_translation_error_m": pose_translation_error,
            "worst_translation_frame": worst_translation_frame,
            "maximum_rotation_error_deg": pose_rotation_error,
            "worst_rotation_frame": worst_rotation_frame,
        })
    return {
        "schema_version": "robot-spatial-cross-engine-fk.v1",
        "status": "passed" if not discrepancies else "failed",
        "robot": model.name,
        "root_link": model.root_link,
        "source": {
            "urdf_path": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "upstream_url": source_url,
            "upstream_revision": source_revision,
        },
        "engines": {
            "candidate": {"name": "robot-spatial", "schema_version": "robot-spatial.v2"},
            "oracle": {
                "name": "yourdfpy",
                "version": importlib.metadata.version("yourdfpy"),
                "numpy_version": numpy.__version__,
                "trimesh_version": importlib.metadata.version("trimesh"),
            },
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "sampling": {"pose_count": len(poses), "seed": seed, **sampling},
        "coverage": {
            "link_frame_count": len(model.links),
            "joint_count": len(model.joints),
            "matrix_comparison_count": comparisons,
            "compared_frames": "every URDF link frame at every generated pose",
            "not_compared": ["joint pre-motion frames", "visual/collision/inertial frames", "mesh geometry", "collision", "dynamics"],
        },
        "tolerances": {
            "translation_m": translation_tolerance_m,
            "rotation_deg": rotation_tolerance_deg,
        },
        "maximum_translation_error_m": global_translation_error,
        "maximum_rotation_error_deg": global_rotation_error,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies[:1000],
        "pose_results": pose_results,
        "warnings_from_candidate_parser": model.warnings(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--poses", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--translation-tolerance-m", type=float, default=1e-9)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1e-6)
    parser.add_argument("--source-url")
    parser.add_argument("--source-revision")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = crosscheck(
            args.urdf,
            args.poses,
            args.seed,
            args.translation_tolerance_m,
            args.rotation_tolerance_deg,
            args.source_url,
            args.source_revision,
        )
    except (OSError, SpatialError, CrosscheckError, ValueError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json_dump(result), encoding="utf-8")
    print(json_dump(result), end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
