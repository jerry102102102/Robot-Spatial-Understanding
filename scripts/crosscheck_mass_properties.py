#!/usr/bin/env python3
"""Cross-check declared aggregate mass properties against independent yourdfpy/NumPy."""

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


def oracle_mass_properties(oracle: Any, selected_links: list[str], reference_frame: str) -> dict[str, Any]:
    import numpy

    by_name = {link.name: link for link in oracle.robot.links}
    contributions: list[tuple[str, float, Any, Any]] = []
    missing: list[str] = []
    for link_name in selected_links:
        link = by_name[link_name]
        if link.inertial is None:
            missing.append(link_name)
            continue
        reference_from_link = oracle.get_transform(link_name, reference_frame)
        link_from_inertial = link.inertial.origin if link.inertial.origin is not None else numpy.eye(4)
        reference_from_inertial = reference_from_link @ link_from_inertial
        rotation = reference_from_inertial[:3, :3]
        center = reference_from_inertial[:3, 3]
        inertia = rotation @ link.inertial.inertia @ rotation.T
        contributions.append((link_name, float(link.inertial.mass), center, inertia))
    if not contributions:
        return {"status": "not_provided", "missing_inertial_links": missing}
    mass = sum(record[1] for record in contributions)
    center = sum((record[1] * record[2] for record in contributions), start=numpy.zeros(3)) / mass
    aggregate = numpy.zeros((3, 3))
    for _, link_mass, link_center, link_inertia in contributions:
        displacement = link_center - center
        aggregate += link_inertia + link_mass * (
            float(displacement @ displacement) * numpy.eye(3) - numpy.outer(displacement, displacement)
        )
    return {
        "status": "computed",
        "declared_mass_kg": mass,
        "center_of_mass_in_expressed_frame_m": center,
        "inertia_about_center_of_mass_in_expressed_frame_kg_m2": aggregate,
        "principal_moments_kg_m2": numpy.linalg.eigvalsh(aggregate),
        "missing_inertial_links": missing,
        "declared_inertial_links": sorted(record[0] for record in contributions),
    }


def crosscheck(
    urdf_path: Path,
    pose_count: int,
    seed: int,
    reference_frame: str | None,
    subtree_root: str | None,
    mass_tolerance_kg: float,
    center_tolerance_m: float,
    inertia_tolerance_kg_m2: float,
    source_url: str | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    if min(mass_tolerance_kg, center_tolerance_m, inertia_tolerance_kg_m2) < 0.0:
        raise CrosscheckError("tolerances must be non-negative")
    try:
        import numpy
        import yourdfpy
        from yourdfpy import URDF
    except ImportError as error:
        raise CrosscheckError("yourdfpy and NumPy are required in the active Python environment") from error
    model = RobotModel(urdf_path)
    reference = reference_frame or model.root_link
    if reference not in model.links:
        raise CrosscheckError("independent mass cross-check currently requires --frame to name a URDF link frame")
    oracle = URDF.load(str(urdf_path.resolve()), load_meshes=False, load_collision_meshes=False)
    poses, sampling = generate_poses(model, pose_count, seed)
    discrepancies: list[dict[str, Any]] = []
    pose_results: list[dict[str, Any]] = []
    maximum_mass_error = 0.0
    maximum_center_error = 0.0
    maximum_inertia_error = 0.0
    for pose_index, pose in enumerate(poses):
        candidate = model.mass_properties(pose, reference, subtree_root)
        if candidate["status"] != "computed":
            raise CrosscheckError(f"candidate mass properties are {candidate['status']!r}; fix inertial declarations before cross-checking")
        oracle.update_cfg(pose)
        expected = oracle_mass_properties(oracle, candidate["selection"]["selected_links"], reference)
        if expected["status"] != "computed":
            raise CrosscheckError("oracle found no declared inertials in the selected links")
        mass_error = abs(candidate["declared_mass_kg"] - expected["declared_mass_kg"])
        candidate_center = numpy.asarray(candidate["center_of_mass_in_expressed_frame_m"], dtype=float)
        center_error = float(numpy.linalg.norm(candidate_center - expected["center_of_mass_in_expressed_frame_m"]))
        candidate_inertia = numpy.asarray(
            candidate["inertia_about_center_of_mass_in_expressed_frame_kg_m2"]["matrix_3x3_rowmajor"],
            dtype=float,
        )
        inertia_error = float(numpy.max(numpy.abs(candidate_inertia - expected["inertia_about_center_of_mass_in_expressed_frame_kg_m2"])))
        missing_match = candidate["coverage"]["missing_inertial_links"] == expected["missing_inertial_links"]
        maximum_mass_error = max(maximum_mass_error, mass_error)
        maximum_center_error = max(maximum_center_error, center_error)
        maximum_inertia_error = max(maximum_inertia_error, inertia_error)
        passed = (
            mass_error <= mass_tolerance_kg
            and center_error <= center_tolerance_m
            and inertia_error <= inertia_tolerance_kg_m2
            and missing_match
        )
        if not passed:
            discrepancies.append({
                "pose_index": pose_index,
                "mass_error_kg": mass_error,
                "center_error_m": center_error,
                "maximum_inertia_component_error_kg_m2": inertia_error,
                "missing_inertial_links_match": missing_match,
            })
        pose_results.append({
            "pose_index": pose_index,
            "pose_sha256": hashlib.sha256(json.dumps(pose, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "declared_mass_kg": clean_number(candidate["declared_mass_kg"]),
            "center_of_mass_in_expressed_frame_m": [clean_number(value) for value in candidate_center],
            "mass_error_kg": mass_error,
            "center_error_m": center_error,
            "maximum_inertia_component_error_kg_m2": inertia_error,
            "status": "passed" if passed else "failed",
        })
    return {
        "schema_version": "robot-spatial-cross-engine-mass-properties.v1",
        "status": "passed" if not discrepancies else "failed",
        "robot": model.name,
        "root_link": model.root_link,
        "selection": {"subtree_root_link": subtree_root or model.root_link, "expressed_in_frame": reference},
        "source": {
            "urdf_path": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "upstream_url": source_url,
            "upstream_revision": source_revision,
        },
        "engines": {
            "candidate": {"name": "robot-spatial", "schema_version": "robot-spatial-mass-properties.v1"},
            "oracle": {
                "name": "yourdfpy-plus-numpy-independent-aggregation",
                "yourdfpy_version": importlib.metadata.version("yourdfpy"),
                "numpy_version": numpy.__version__,
            },
            "python": {"version": platform.python_version(), "platform": platform.platform()},
        },
        "sampling": {"pose_count": len(poses), "seed": seed, **sampling},
        "coverage": {
            "pose_count": len(poses),
            "selected_link_count": len(model.mass_properties(poses[0], reference, subtree_root)["selection"]["selected_links"]),
            "comparisons_per_pose": ["declared mass", "center of mass", "complete aggregate inertia tensor", "missing-inertial link set"],
            "not_compared": ["physical hardware mass", "payload", "dynamic response", "world objects", "links or constraints absent from URDF"],
        },
        "tolerances": {
            "mass_kg": mass_tolerance_kg,
            "center_m": center_tolerance_m,
            "inertia_kg_m2": inertia_tolerance_kg_m2,
        },
        "maximum_mass_error_kg": maximum_mass_error,
        "maximum_center_error_m": maximum_center_error,
        "maximum_inertia_component_error_kg_m2": maximum_inertia_error,
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
    parser.add_argument("--frame", help="URDF link frame for COM/inertia components; defaults to root")
    parser.add_argument("--subtree-root", help="optional selected subtree root link")
    parser.add_argument("--mass-tolerance-kg", type=float, default=1e-10)
    parser.add_argument("--center-tolerance-m", type=float, default=1e-9)
    parser.add_argument("--inertia-tolerance-kg-m2", type=float, default=1e-9)
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
            args.frame,
            args.subtree_root,
            args.mass_tolerance_kg,
            args.center_tolerance_m,
            args.inertia_tolerance_kg_m2,
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
