#!/usr/bin/env python3
"""Cross-check static gravity loads by finite-differencing yourdfpy potential energy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from crosscheck_yourdfpy import CrosscheckError, generate_poses
from robot_spatial import RobotModel, SpatialError, clean_number


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def oracle_potential_energy(oracle: Any, root_link: str, gravity_root: list[float]) -> float:
    """Compute U=-sum(m*g dot p_com) through the independent parser/FK engine."""
    import numpy

    gravity = numpy.asarray(gravity_root, dtype=float)
    energy = 0.0
    for link in oracle.robot.links:
        if link.inertial is None:
            continue
        root_from_link = oracle.get_transform(link.name, root_link)
        link_from_inertial = link.inertial.origin if link.inertial.origin is not None else numpy.eye(4)
        root_from_inertial = root_from_link @ link_from_inertial
        center = root_from_inertial[:3, 3]
        energy -= float(link.inertial.mass) * float(gravity @ center)
    return energy


def crosscheck(
    urdf_path: Path,
    pose_count: int,
    seed: int,
    gravity_root: list[float],
    finite_difference_step: float,
    load_tolerance: float,
    potential_tolerance_j: float,
    source_url: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    if len(gravity_root) != 3 or not all(math.isfinite(value) for value in gravity_root):
        raise CrosscheckError("gravity must contain exactly three finite components")
    if finite_difference_step <= 0.0 or not math.isfinite(finite_difference_step):
        raise CrosscheckError("finite-difference step must be positive and finite")
    if load_tolerance < 0.0 or potential_tolerance_j < 0.0:
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
    discrepancies: list[dict[str, Any]] = []
    pose_results: list[dict[str, Any]] = []
    maximum_load_error = 0.0
    maximum_potential_error = 0.0
    comparison_count = 0
    for pose_index, pose in enumerate(poses):
        candidate = model.static_gravity_loads(pose, gravity_root, model.root_link)
        if candidate["status"] != "computed":
            raise CrosscheckError(
                f"candidate gravity loads are {candidate['status']!r}; fix invalid/incomplete selected inertials before cross-checking"
            )
        oracle.update_cfg(pose)
        expected_potential = oracle_potential_energy(oracle, model.root_link, gravity_root)
        potential_error = abs(candidate["modeled_potential_energy_relative_to_root_origin_j"] - expected_potential)
        maximum_potential_error = max(maximum_potential_error, potential_error)
        driver_results: dict[str, Any] = {}
        for driver in candidate["independent_driver_order"]:
            plus, minus = dict(pose), dict(pose)
            plus[driver] = plus.get(driver, 0.0) + finite_difference_step
            minus[driver] = minus.get(driver, 0.0) - finite_difference_step
            oracle.update_cfg(plus)
            plus_energy = oracle_potential_energy(oracle, model.root_link, gravity_root)
            oracle.update_cfg(minus)
            minus_energy = oracle_potential_energy(oracle, model.root_link, gravity_root)
            expected_generalized_force = -(plus_energy - minus_energy) / (2.0 * finite_difference_step)
            actual_generalized_force = candidate["independent_driver_loads"][driver]["generalized_gravity_force"]
            error = abs(actual_generalized_force - expected_generalized_force)
            maximum_load_error = max(maximum_load_error, error)
            comparison_count += 1
            passed = error <= load_tolerance
            driver_results[driver] = {
                "unit": candidate["independent_driver_loads"][driver]["unit"],
                "candidate_generalized_gravity_force": actual_generalized_force,
                "oracle_negative_potential_derivative": expected_generalized_force,
                "absolute_error": error,
                "status": "passed" if passed else "failed",
            }
            if not passed:
                discrepancies.append({
                    "pose_index": pose_index,
                    "joint": driver,
                    "candidate_generalized_gravity_force": actual_generalized_force,
                    "oracle_negative_potential_derivative": expected_generalized_force,
                    "absolute_error": error,
                    "unit": candidate["independent_driver_loads"][driver]["unit"],
                })
        potential_passed = potential_error <= potential_tolerance_j
        if not potential_passed:
            discrepancies.append({
                "pose_index": pose_index,
                "quantity": "modeled_potential_energy_relative_to_root_origin_j",
                "candidate": candidate["modeled_potential_energy_relative_to_root_origin_j"],
                "oracle": expected_potential,
                "absolute_error_j": potential_error,
            })
        pose_results.append({
            "pose_index": pose_index,
            "pose_sha256": hashlib.sha256(
                json.dumps(pose, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "supplied_independent_joint_positions": {
                name: clean_number(value) for name, value in pose.items()
            },
            "potential_energy": {
                "candidate_j": candidate["modeled_potential_energy_relative_to_root_origin_j"],
                "oracle_j": expected_potential,
                "absolute_error_j": potential_error,
                "status": "passed" if potential_passed else "failed",
            },
            "driver_results": driver_results,
            "status": "passed" if potential_passed and all(record["status"] == "passed" for record in driver_results.values()) else "failed",
        })
        oracle.update_cfg(pose)
    return {
        "schema_version": "robot-spatial-cross-engine-static-gravity-loads.v1",
        "status": "passed" if not discrepancies else "failed",
        "robot": model.name,
        "root_link": model.root_link,
        "source": {
            "urdf_path": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "upstream_url": source_url,
            "upstream_revision": source_revision,
        },
        "gravity": {"vector_in_root_frame_xyz_m_s2": gravity_root},
        "engines": {
            "candidate": {"name": "robot-spatial", "schema_version": "robot-spatial-static-gravity-loads.v1"},
            "oracle": {
                "name": "yourdfpy-potential-energy-central-finite-difference",
                "yourdfpy_version": importlib.metadata.version("yourdfpy"),
                "numpy_version": numpy.__version__,
            },
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "sampling": {"pose_count": len(poses), "seed": seed, **sampling},
        "coverage": {
            "pose_count": len(poses),
            "independent_driver_comparison_count": comparison_count,
            "compared": [
                "whole-tree modeled potential energy relative to the root origin",
                "generalized gravity force for every independent driver as -dU/dq",
                "mimic effects as implemented independently by the oracle engine",
            ],
            "oracle_independence": "separate URDF parser and FK engine; numerical potential-energy derivative rather than candidate force/moment projection",
            "not_compared": [
                "physical hardware or payload",
                "actual world-to-root mounting orientation",
                "contacts, friction, damping, velocity, acceleration, motor/transmission behavior, or controller execution",
                "subtree selection or gravity expressed in a non-root frame",
            ],
        },
        "finite_difference_step_joint_units": finite_difference_step,
        "tolerances": {"generalized_force_or_torque": load_tolerance, "potential_energy_j": potential_tolerance_j},
        "maximum_generalized_force_or_torque_error": maximum_load_error,
        "maximum_potential_energy_error_j": maximum_potential_error,
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "pose_results": pose_results,
        "warnings_from_candidate_parser": model.warnings(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--poses", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, -9.80665], metavar=("GX", "GY", "GZ"))
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--load-tolerance", type=float, default=2e-5)
    parser.add_argument("--potential-tolerance-j", type=float, default=1e-9)
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
            args.gravity,
            args.finite_difference_step,
            args.load_tolerance,
            args.potential_tolerance_j,
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
