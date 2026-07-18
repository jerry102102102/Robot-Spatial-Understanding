#!/usr/bin/env python3
"""Independent randomized oracle for semantic render-atlas projection records."""

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
from typing import Any, Iterable


Matrix = list[list[float]]
Vector = list[float]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)] for row in range(4)]


def origin(xyz: Vector, yaw: float = 0.0) -> Matrix:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        [cosine, -sine, 0.0, xyz[0]],
        [sine, cosine, 0.0, xyz[1]],
        [0.0, 0.0, 1.0, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(transform: Matrix, point: Vector) -> Vector:
    return [sum(transform[row][column] * point[column] for column in range(3)) + transform[row][3] for row in range(3)]


def box_points(size: Vector) -> list[Vector]:
    half = [value / 2.0 for value in size]
    return [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]


def project(point: Vector, u_axis: Vector, v_axis: Vector) -> tuple[float, float]:
    return (
        sum(point[index] * u_axis[index] for index in range(3)),
        sum(point[index] * v_axis[index] for index in range(3)),
    )


def dot(point: Vector, axis: Vector) -> float:
    return sum(point[index] * axis[index] for index in range(3))


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def bounds(points: Iterable[tuple[float, float]]) -> dict[str, list[float]]:
    records = list(points)
    minimum = [min(point[index] for point in records) for index in range(2)]
    maximum = [max(point[index] for point in records) for index in range(2)]
    return {
        "min": minimum,
        "max": maximum,
        "extents": [maximum[index] - minimum[index] for index in range(2)],
    }


def maximum_difference(observed: Any, expected: Any) -> float:
    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            return math.inf
        return max((maximum_difference(observed[key], expected[key]) for key in expected), default=0.0)
    if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
        if len(expected) != len(observed):
            return math.inf
        return max((maximum_difference(left, right) for left, right in zip(observed, expected)), default=0.0)
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(observed) - float(expected))
    return 0.0 if observed == expected else math.inf


def view_specs() -> dict[str, tuple[Vector, Vector, Vector]]:
    return {
        "front": ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        "side": ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        "top": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        "isometric": (
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
            [-1.0 / math.sqrt(6.0), -1.0 / math.sqrt(6.0), 2.0 / math.sqrt(6.0)],
            [1.0 / math.sqrt(3.0)] * 3,
        ),
    }


def urdf_text(parameters: dict[str, Any]) -> str:
    def xyz(values: Vector) -> str:
        return " ".join(f"{value:.17g}" for value in values)

    def link(name: str, size: Vector | None, geometry_origin: Vector | None = None, geometry_yaw: float = 0.0) -> str:
        if size is None or geometry_origin is None:
            return f'<link name="{name}"/>'
        return (
            f'<link name="{name}"><collision><origin xyz="{xyz(geometry_origin)}" rpy="0 0 {geometry_yaw:.17g}"/>'
            f'<geometry><box size="{xyz(size)}"/></geometry></collision></link>'
        )

    return "\n".join([
        '<?xml version="1.0"?>',
        '<robot name="render_oracle">',
        link("root", parameters["root_size"], parameters["root_geometry_origin"], parameters["root_geometry_yaw"]),
        link("arm", parameters["arm_size"], parameters["arm_geometry_origin"], parameters["arm_geometry_yaw"]),
        link("slider", parameters["slider_size"], parameters["slider_geometry_origin"], parameters["slider_geometry_yaw"]),
        link("tip", None),
        (
            f'<joint name="spin" type="revolute"><parent link="root"/><child link="arm"/>'
            f'<origin xyz="{xyz(parameters["spin_origin"])}" rpy="0 0 {parameters["spin_origin_yaw"]:.17g}"/>'
            '<axis xyz="0 0 1"/><limit lower="-3.141592653589793" upper="3.141592653589793" effort="10" velocity="2"/></joint>'
        ),
        (
            f'<joint name="slide" type="prismatic"><parent link="arm"/><child link="slider"/>'
            f'<origin xyz="{xyz(parameters["slide_origin"])}" rpy="0 0 {parameters["slide_origin_yaw"]:.17g}"/>'
            '<axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5" effort="10" velocity="1"/></joint>'
        ),
        (
            f'<joint name="tip_mount" type="fixed"><parent link="slider"/><child link="tip"/>'
            f'<origin xyz="{xyz(parameters["tip_origin"])}" rpy="0 0 {parameters["tip_origin_yaw"]:.17g}"/></joint>'
        ),
        '</robot>',
        '',
    ])


def expected_geometry(parameters: dict[str, Any]) -> tuple[dict[str, Matrix], dict[str, list[Vector]]]:
    root = origin([0.0, 0.0, 0.0])
    arm = matmul(matmul(origin(parameters["spin_origin"], parameters["spin_origin_yaw"]), origin([0.0, 0.0, 0.0], parameters["spin_position"])), root)
    slider = matmul(
        matmul(
            arm,
            origin(parameters["slide_origin"], parameters["slide_origin_yaw"]),
        ),
        origin([parameters["slide_position"], 0.0, 0.0]),
    )
    tip = matmul(slider, origin(parameters["tip_origin"], parameters["tip_origin_yaw"]))
    frames = {"root": root, "arm": arm, "slider": slider, "tip": tip}
    geometry: dict[str, list[Vector]] = {}
    for link_name in ("root", "arm", "slider"):
        geometry_transform = matmul(
            frames[link_name],
            origin(parameters[f"{link_name}_geometry_origin"], parameters[f"{link_name}_geometry_yaw"]),
        )
        geometry[f"collision/{link_name}/0"] = [
            transform_point(geometry_transform, point)
            for point in box_points(parameters[f"{link_name}_size"])
        ]
    return frames, geometry


def random_parameters(rng: random.Random) -> dict[str, Any]:
    vector = lambda low, high: [rng.uniform(low, high) for _ in range(3)]
    size = lambda: [rng.uniform(0.04, 0.45), rng.uniform(0.05, 0.35), rng.uniform(0.03, 0.30)]
    return {
        "spin_origin": vector(-0.35, 0.55),
        "spin_origin_yaw": rng.uniform(-math.pi, math.pi),
        "spin_position": rng.uniform(-math.pi, math.pi),
        "slide_origin": vector(-0.30, 0.50),
        "slide_origin_yaw": rng.uniform(-math.pi, math.pi),
        "slide_position": rng.uniform(-0.45, 0.45),
        "tip_origin": vector(-0.20, 0.35),
        "tip_origin_yaw": rng.uniform(-math.pi, math.pi),
        "root_size": size(),
        "arm_size": size(),
        "slider_size": size(),
        "root_geometry_origin": vector(-0.15, 0.15),
        "arm_geometry_origin": vector(-0.15, 0.15),
        "slider_geometry_origin": vector(-0.15, 0.15),
        "root_geometry_yaw": rng.uniform(-math.pi, math.pi),
        "arm_geometry_yaw": rng.uniform(-math.pi, math.pi),
        "slider_geometry_yaw": rng.uniform(-math.pi, math.pi),
    }


def run_case(script: Path, case_root: Path, parameters: dict[str, Any], tolerance: float) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    urdf_path, pose_path = case_root / "robot.urdf", case_root / "pose.json"
    overview, atlas_root = case_root / "scene.svg", case_root / "atlas"
    urdf_path.write_text(urdf_text(parameters), encoding="utf-8")
    pose_path.write_text(json.dumps({
        "pose_name": "oracle_pose",
        "joints": {"spin": parameters["spin_position"], "slide": parameters["slide_position"]},
    }), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), "render", str(urdf_path), "--pose", str(pose_path), "--out", str(overview), "--atlas-out", str(atlas_root)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ([{"check": "cli", "stderr": completed.stderr, "stdout": completed.stdout}], {}, math.inf)
    atlas = json.loads((atlas_root / "manifest.json").read_text())
    frames, geometry = expected_geometry(parameters)
    frame_origins = {name: [matrix[index][3] for index in range(3)] for name, matrix in frames.items()}
    all_points = [point for points in geometry.values() for point in points] + list(frame_origins.values())
    failures: list[dict[str, Any]] = []
    counts = {"views": 0, "frame_origins": 0, "geometry_hulls": 0, "kinematic_edges": 0, "svg_entities": 0}
    max_error = 0.0

    def compare(label: str, observed: Any, expected: Any, allowed: float = tolerance) -> None:
        nonlocal max_error
        error = maximum_difference(observed, expected)
        max_error = max(max_error, error)
        if error > allowed:
            failures.append({"check": label, "maximum_absolute_error": error, "observed": observed, "expected": expected})

    expected_edges = {
        "spin": ("root", "arm"),
        "slide": ("arm", "slider"),
        "tip_mount": ("slider", "tip"),
    }
    for view_id, (u_axis, v_axis, depth_axis) in view_specs().items():
        view = atlas["views"][view_id]
        counts["views"] += 1
        compare(f"{view_id}.projection", view["projection"]["root_xyz_to_uv_matrix_2x3"], [u_axis, v_axis])
        compare(f"{view_id}.depth_axis", view["projection"]["depth_axis_in_root_xyz"], depth_axis)
        projected_all = [project(point, u_axis, v_axis) for point in all_points]
        scene_bounds = bounds(projected_all)
        expected_scene_bounds = {"min_uv": scene_bounds["min"], "max_uv": scene_bounds["max"], "extents_uv": scene_bounds["extents"]}
        compare(f"{view_id}.scene_bounds", view["scene_projection_bounds_uv_m"], expected_scene_bounds)
        span = max(scene_bounds["extents"][0], scene_bounds["extents"][1], 1e-3)
        scale = min(640.0 / span, 430.0 / span)
        center = [(scene_bounds["min"][index] + scene_bounds["max"][index]) / 2.0 for index in range(2)]
        compare(f"{view_id}.screen_center", view["screen"]["center_uv_m"], center)
        compare(f"{view_id}.screen_scale", view["screen"]["scale_px_per_m"], scale)

        def pixel(point_uv: tuple[float, float]) -> tuple[float, float]:
            return 360.0 + (point_uv[0] - center[0]) * scale, 88.0 + 215.0 - (point_uv[1] - center[1]) * scale

        frame_by_id = {record["frame_name"]: record for record in view["link_frames"]}
        for frame_name, point in frame_origins.items():
            record = frame_by_id[frame_name]
            uv = project(point, u_axis, v_axis)
            compare(f"{view_id}.frame.{frame_name}.root", record["origin_root_xyz_m"], point)
            compare(f"{view_id}.frame.{frame_name}.uv", record["projected_uv_m"], uv)
            compare(f"{view_id}.frame.{frame_name}.pixel", record["pixel_xy"], pixel(uv), 1e-5)
            counts["frame_origins"] += 1
        geometry_by_frame = {record["frame_name"]: record for record in view["geometry"]}
        for frame_name, points in geometry.items():
            record = geometry_by_frame[frame_name]
            projected = [project(point, u_axis, v_axis) for point in points]
            hull = convex_hull(projected)
            projected_bounds = bounds(projected)
            pixel_hull = [pixel(point) for point in hull]
            pixel_bounds = bounds(pixel_hull)
            depths = [dot(point, depth_axis) for point in points]
            compare(f"{view_id}.geometry.{frame_name}.uv_hull", record["projected_hull_uv_m"], hull)
            compare(
                f"{view_id}.geometry.{frame_name}.uv_bounds",
                record["projection_bounds_uv_m"],
                {"min_uv": projected_bounds["min"], "max_uv": projected_bounds["max"], "extents_uv": projected_bounds["extents"]},
            )
            compare(f"{view_id}.geometry.{frame_name}.pixel_hull", record["pixel_hull_xy"], pixel_hull, 1e-5)
            compare(
                f"{view_id}.geometry.{frame_name}.pixel_bounds",
                record["pixel_bounds_xy"],
                {"min_xy": pixel_bounds["min"], "max_xy": pixel_bounds["max"], "extents_xy": pixel_bounds["extents"]},
                1e-5,
            )
            compare(f"{view_id}.geometry.{frame_name}.depth", record["depth_interval_m"], [min(depths), max(depths)])
            counts["geometry_hulls"] += 1
        edge_by_name = {record["joint_name"]: record for record in view["kinematic_edges"]}
        for joint_name, (parent, child) in expected_edges.items():
            record = edge_by_name[joint_name]
            start, end = frame_origins[parent], frame_origins[child]
            start_uv, end_uv = project(start, u_axis, v_axis), project(end, u_axis, v_axis)
            compare(f"{view_id}.edge.{joint_name}.start", record["start_root_xyz_m"], start)
            compare(f"{view_id}.edge.{joint_name}.end", record["end_root_xyz_m"], end)
            compare(f"{view_id}.edge.{joint_name}.uv", [record["start_uv_m"], record["end_uv_m"]], [start_uv, end_uv])
            compare(f"{view_id}.edge.{joint_name}.length3d", record["length_3d_m"], math.dist(start, end))
            compare(f"{view_id}.edge.{joint_name}.projected_length", record["projected_length_m"], math.dist(start_uv, end_uv))
            counts["kinematic_edges"] += 1
        svg_path = atlas_root / view["artifact"]["path"]
        compare(f"{view_id}.svg_sha256", hashlib.sha256(svg_path.read_bytes()).hexdigest(), view["artifact"]["sha256"], 0.0)
        svg = svg_path.read_text()
        required_entities = {f"frame/{name}" for name in geometry} | {f"frame/{name}" for name in frames} | {f"joint/{name}" for name in expected_edges}
        missing = sorted(entity for entity in required_entities if f'data-entity-id="{entity}"' not in svg)
        if missing:
            failures.append({"check": f"{view_id}.svg_entities", "missing": missing})
        counts["svg_entities"] += len(required_entities)
    return failures, counts, max_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    script = Path(__file__).with_name("robot_spatial.py")
    rng = random.Random(args.seed)
    all_failures: list[dict[str, Any]] = []
    totals = {"views": 0, "frame_origins": 0, "geometry_hulls": 0, "kinematic_edges": 0, "svg_entities": 0}
    maximum_error = 0.0
    cli_errors = 0
    with tempfile.TemporaryDirectory(prefix="robot-spatial-render-crosscheck-") as temp_directory:
        root = Path(temp_directory)
        for case_index in range(args.cases):
            case_root = root / f"case-{case_index:04d}"
            case_root.mkdir()
            failures, counts, error = run_case(script, case_root, random_parameters(rng), args.tolerance)
            if failures and failures[0].get("check") == "cli":
                cli_errors += 1
            for failure in failures[:20]:
                all_failures.append({"case": case_index, **failure})
            for key, value in counts.items():
                totals[key] += value
            maximum_error = max(maximum_error, error)
    report = {
        "schema_version": "robot-spatial-render-atlas-crosscheck.v1",
        "status": "passed" if not all_failures else "failed",
        "seed": args.seed,
        "case_count": args.cases,
        "method": "independent analytic FK for randomized revolute/prismatic/fixed chains, box-corner projection, monotone-chain hull, depth interval, and screen-fit comparison against subprocess CLI output",
        "comparison_counts": totals,
        "cli_error_count": cli_errors,
        "discrepancy_count": len(all_failures),
        "maximum_absolute_numeric_error_across_all_fields": maximum_error,
        "tolerances": {
            "world_projection_and_metric_fields": args.tolerance,
            "pixel_fields_rounded_to_six_decimals": 1e-5,
        },
        "failures": all_failures[:50],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
