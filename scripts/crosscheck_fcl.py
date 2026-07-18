#!/usr/bin/env python3
"""Cross-check portable triangle distance/collision against optional python-fcl."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import fcl  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from robot_spatial import origin_matrix
from triangle_geometry import box_surface, bvh_surface_distance, cross, norm_squared, point_inside_closed_surface, subtract, triangle_triangle_distance


def fcl_triangle(triangle: tuple[list[float], list[float], list[float]]) -> Any:
    model = fcl.BVHModel()
    model.beginModel(3, 1)
    model.addSubModel(np.asarray(triangle, dtype=float), np.asarray([[0, 1, 2]], dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model)


def fcl_transform(matrix: list[list[float]]) -> Any:
    return fcl.Transform(np.asarray([row[:3] for row in matrix[:3]], dtype=float), np.asarray([row[3] for row in matrix[:3]], dtype=float))


def random_triangle(generator: random.Random) -> tuple[list[float], list[float], list[float]]:
    while True:
        triangle = tuple([generator.uniform(-3.0, 3.0) for _ in range(3)] for _ in range(3))
        area_vector = cross(subtract(triangle[1], triangle[0]), subtract(triangle[2], triangle[0]))
        if norm_squared(area_vector) > 1e-10:
            return triangle  # type: ignore[return-value]


def run(seed: int, triangle_cases: int, box_cases: int, tolerance: float) -> dict[str, Any]:
    generator = random.Random(seed)
    triangle_discrepancies: list[dict[str, Any]] = []
    maximum_triangle_distance_error = 0.0
    for case_index in range(triangle_cases):
        left, right = random_triangle(generator), random_triangle(generator)
        portable_distance = triangle_triangle_distance(left, right)[0]
        result = fcl.DistanceResult()
        fcl_distance = float(fcl.distance(fcl_triangle(left), fcl_triangle(right), fcl.DistanceRequest(enable_nearest_points=True), result))
        reference_distance = max(0.0, fcl_distance)
        error = abs(portable_distance - reference_distance)
        maximum_triangle_distance_error = max(maximum_triangle_distance_error, error)
        if error > tolerance:
            triangle_discrepancies.append({
                "case": case_index,
                "portable_distance_m": portable_distance,
                "fcl_distance_m": fcl_distance,
                "absolute_error_m": error,
            })

    box_discrepancies: list[dict[str, Any]] = []
    maximum_separated_box_distance_error = 0.0
    containment_cases = 0
    for case_index in range(box_cases):
        left_size = [generator.uniform(0.1, 2.0) for _ in range(3)]
        right_size = [generator.uniform(0.1, 2.0) for _ in range(3)]
        left_transform = origin_matrix(
            [generator.uniform(-2.0, 2.0) for _ in range(3)],
            [generator.uniform(-math.pi, math.pi) for _ in range(3)],
        )
        right_transform = origin_matrix(
            [generator.uniform(-2.0, 2.0) for _ in range(3)],
            [generator.uniform(-math.pi, math.pi) for _ in range(3)],
        )
        left_surface = box_surface(left_size, left_transform)
        right_surface = box_surface(right_size, right_transform)
        portable = bvh_surface_distance(left_surface, right_surface)
        surface_contact = portable["distance_m"] <= tolerance
        containment = False
        if not surface_contact:
            containment = point_inside_closed_surface(left_surface[0][0], right_surface) or point_inside_closed_surface(right_surface[0][0], left_surface)
            containment_cases += int(containment)
        portable_collision = surface_contact or containment
        left_object = fcl.CollisionObject(fcl.Box(*left_size), fcl_transform(left_transform))
        right_object = fcl.CollisionObject(fcl.Box(*right_size), fcl_transform(right_transform))
        collision_result = fcl.CollisionResult()
        fcl_collision = bool(fcl.collide(left_object, right_object, fcl.CollisionRequest(), collision_result))
        distance_result = fcl.DistanceResult()
        fcl_distance = float(fcl.distance(left_object, right_object, fcl.DistanceRequest(enable_nearest_points=True), distance_result))
        distance_error = None
        if not fcl_collision:
            distance_error = abs(portable["distance_m"] - fcl_distance)
            maximum_separated_box_distance_error = max(maximum_separated_box_distance_error, distance_error)
        if portable_collision != fcl_collision or (distance_error is not None and distance_error > tolerance):
            box_discrepancies.append({
                "case": case_index,
                "portable_collision": portable_collision,
                "portable_surface_distance_m": portable["distance_m"],
                "portable_containment": containment,
                "fcl_collision": fcl_collision,
                "fcl_distance_m": fcl_distance,
                "separated_distance_error_m": distance_error,
            })

    targeted_cases = [
        ([4.0, 4.0, 4.0], origin_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), [1.0, 1.0, 1.0], origin_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])),
        ([4.0, 3.0, 2.0], origin_matrix([1.0, -1.0, 0.5], [0.4, -0.2, 0.7]), [0.5, 0.5, 0.5], origin_matrix([1.0, -1.0, 0.5], [0.4, -0.2, 0.7])),
    ]
    targeted_containment_cases = 0
    for case_index, (left_size, left_transform, right_size, right_transform) in enumerate(targeted_cases):
        left_surface = box_surface(left_size, left_transform)
        right_surface = box_surface(right_size, right_transform)
        portable = bvh_surface_distance(left_surface, right_surface)
        containment = point_inside_closed_surface(left_surface[0][0], right_surface) or point_inside_closed_surface(right_surface[0][0], left_surface)
        targeted_containment_cases += int(containment)
        portable_collision = portable["distance_m"] <= tolerance or containment
        left_object = fcl.CollisionObject(fcl.Box(*left_size), fcl_transform(left_transform))
        right_object = fcl.CollisionObject(fcl.Box(*right_size), fcl_transform(right_transform))
        collision_result = fcl.CollisionResult()
        fcl_collision = bool(fcl.collide(left_object, right_object, fcl.CollisionRequest(), collision_result))
        if portable_collision != fcl_collision or not containment or portable["distance_m"] <= tolerance:
            box_discrepancies.append({
                "case": f"targeted-containment-{case_index}",
                "portable_collision": portable_collision,
                "portable_surface_distance_m": portable["distance_m"],
                "portable_containment": containment,
                "fcl_collision": fcl_collision,
            })

    return {
        "schema_version": "robot-spatial-fcl-crosscheck.v1",
        "seed": seed,
        "tolerance_m": tolerance,
        "reference": {"engine": "Flexible Collision Library", "python_fcl_version": getattr(fcl, "__version__", "unknown")},
        "triangle_pair_cases": triangle_cases,
        "triangle_pair_discrepancy_count": len(triangle_discrepancies),
        "maximum_triangle_distance_error_m": maximum_triangle_distance_error,
        "box_pair_cases": box_cases,
        "box_pair_discrepancy_count": len(box_discrepancies),
        "random_containment_cases": containment_cases,
        "targeted_containment_cases": len(targeted_cases),
        "targeted_containment_cases_verified": targeted_containment_cases,
        "maximum_separated_box_distance_error_m": maximum_separated_box_distance_error,
        "status": "passed" if not triangle_discrepancies and not box_discrepancies else "failed",
        "discrepancies": {
            "triangle_pairs": triangle_discrepancies[:20],
            "box_pairs": box_discrepancies[:20],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--triangle-cases", type=int, default=1000)
    parser.add_argument("--box-cases", type=int, default=1000)
    parser.add_argument("--tolerance-m", type=float, default=1e-8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.triangle_cases, args.box_cases, args.tolerance_m)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
