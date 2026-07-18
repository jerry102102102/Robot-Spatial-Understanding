#!/usr/bin/env python3
"""Independent randomized oracle for the counterfactual motion-atlas CLI.

This script deliberately does not import robot_spatial, spatial_motion, or
spatial_render. It generates URDFs, invokes the public CLI, and recomputes the
expected mimic limits, FK, SE(3) endpoint deltas, shared projections, SVG
bindings, and artifact digests with a separate implementation.
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
from typing import Any, Iterable


EPSILON = 1e-10
Matrix = list[list[float]]
Vector = list[float]


def identity() -> Matrix:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)] for row in range(4)]


def inverse_rigid(transform: Matrix) -> Matrix:
    result = identity()
    for row in range(3):
        for column in range(3):
            result[row][column] = transform[column][row]
        result[row][3] = -sum(transform[column][row] * transform[column][3] for column in range(3))
    return result


def rpy_matrix(rpy: Vector) -> Matrix:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def origin_matrix(xyz: Vector, rpy: Vector) -> Matrix:
    result = rpy_matrix(rpy)
    for index in range(3):
        result[index][3] = xyz[index]
    return result


def normalized(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def axis_angle(axis: Vector, angle: float) -> Matrix:
    x, y, z = normalized(axis)
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation(vector: Vector) -> Matrix:
    result = identity()
    for index in range(3):
        result[index][3] = vector[index]
    return result


def origin(transform: Matrix) -> Vector:
    return [transform[index][3] for index in range(3)]


def clean(value: float, digits: int = 12) -> float:
    return 0.0 if abs(value) < 1e-12 else round(value, digits)


def vector(values: Iterable[float], digits: int = 12) -> Vector:
    return [clean(float(value), digits) for value in values]


def assert_close(observed: Any, expected: Any, label: str, tolerance: float = 2e-9) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance):
            raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"{label}: list shape mismatch")
        for index, item in enumerate(expected):
            assert_close(observed[index], item, f"{label}[{index}]", tolerance)
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError(f"{label}: object keys mismatch")
        for key, item in expected.items():
            assert_close(observed[key], item, f"{label}.{key}", tolerance)
        return
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")


def random_vector(rng: random.Random, scale: float) -> Vector:
    return [round(rng.uniform(-scale, scale), 3) for _ in range(3)]


def fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def choose_position(lower: float, upper: float, step: float, mode: int, rng: random.Random) -> float:
    if mode == 0:
        return lower
    if mode == 1:
        return upper
    if mode == 2:
        return min(upper, lower + step / 3.0)
    if mode == 3:
        return max(lower, upper - step / 3.0)
    margin = min(step * 1.5, (upper - lower) * 0.2)
    return rng.uniform(lower + margin, upper - margin)


def view_specs() -> list[tuple[str, Vector, Vector, Vector]]:
    s2, s3, s6 = 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(6.0)
    return [
        ("front", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ("side", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ("top", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ("isometric", [s2, -s2, 0.0], [-s6, -s6, 2.0 * s6], [s3, s3, s3]),
    ]


def dot(left: Vector, right: Vector) -> float:
    return sum(left[index] * right[index] for index in range(3))


def rotation_angle(relative: Matrix) -> float:
    identity_error = max(
        abs(relative[row][column] - (1.0 if row == column else 0.0))
        for row in range(3)
        for column in range(3)
    )
    if identity_error <= 1e-10:
        return 0.0
    cosine = max(-1.0, min(1.0, (relative[0][0] + relative[1][1] + relative[2][2] - 1.0) / 2.0))
    return math.acos(cosine)


def make_case(rng: random.Random, case_index: int) -> dict[str, Any]:
    multiplier = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
    offset = round(rng.uniform(-0.22, 0.22), 3)
    driver_lower, driver_upper = -1.4, 1.3
    follower_lower, follower_upper = -0.75, 0.85
    transformed = sorted(((follower_lower - offset) / multiplier, (follower_upper - offset) / multiplier))
    feasible_lower = max(driver_lower, transformed[0])
    feasible_upper = min(driver_upper, transformed[1])
    angular_step = round(rng.uniform(0.06, 0.24), 3)
    linear_step = round(rng.uniform(0.015, 0.075), 3)
    driver_position = choose_position(feasible_lower, feasible_upper, angular_step, case_index % 8, rng)
    slide_lower, slide_upper = -0.25, 0.45
    slide_position = choose_position(slide_lower, slide_upper, linear_step, (case_index + 3) % 8, rng)
    axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.5])
    specs = {
        "j_drive": {"type": "revolute", "parent": "base", "child": "arm", "xyz": random_vector(rng, 0.35), "rpy": random_vector(rng, 0.45), "axis": list(rng.choice(axes))},
        "j_follow": {"type": "revolute", "parent": "arm", "child": "finger", "xyz": random_vector(rng, 0.45), "rpy": random_vector(rng, 0.45), "axis": list(rng.choice(axes)), "mimic": ("j_drive", multiplier, offset)},
        "j_mount": {"type": "fixed", "parent": "arm", "child": "carrier", "xyz": random_vector(rng, 0.45), "rpy": random_vector(rng, 0.45), "axis": [1.0, 0.0, 0.0]},
        "j_pad": {"type": "fixed", "parent": "finger", "child": "pad", "xyz": random_vector(rng, 0.25), "rpy": random_vector(rng, 0.3), "axis": [1.0, 0.0, 0.0]},
        "j_slide": {"type": "prismatic", "parent": "carrier", "child": "tip", "xyz": random_vector(rng, 0.35), "rpy": random_vector(rng, 0.45), "axis": list(rng.choice(axes))},
        "j_tool": {"type": "fixed", "parent": "tip", "child": "tool", "xyz": random_vector(rng, 0.25), "rpy": random_vector(rng, 0.3), "axis": [1.0, 0.0, 0.0]},
    }
    limits = {
        "j_drive": (driver_lower, driver_upper),
        "j_follow": (follower_lower, follower_upper),
        "j_slide": (slide_lower, slide_upper),
    }
    return {
        "specs": specs,
        "limits": limits,
        "feasible": {"j_drive": (feasible_lower, feasible_upper), "j_slide": (slide_lower, slide_upper)},
        "pose": {"j_drive": driver_position, "j_slide": slide_position},
        "steps": {"j_drive": angular_step, "j_slide": linear_step},
        "multiplier": multiplier,
        "offset": offset,
    }


def write_case(case: dict[str, Any], urdf: Path, pose: Path) -> None:
    links = ["base", "arm", "finger", "carrier", "pad", "tip", "tool"]
    lines = ['<robot name="motion_oracle">', *(f'  <link name="{name}"/>' for name in links)]
    for name, spec in case["specs"].items():
        lines.extend([
            f'  <joint name="{name}" type="{spec["type"]}">',
            f'    <parent link="{spec["parent"]}"/><child link="{spec["child"]}"/>',
            f'    <origin xyz="{fmt(spec["xyz"])}" rpy="{fmt(spec["rpy"])}"/>',
        ])
        if spec["type"] != "fixed":
            lines.append(f'    <axis xyz="{fmt(spec["axis"])}"/>')
            lower, upper = case["limits"][name]
            lines.append(f'    <limit lower="{lower:.12g}" upper="{upper:.12g}" effort="10" velocity="2"/>')
        if "mimic" in spec:
            source, multiplier, offset = spec["mimic"]
            lines.append(f'    <mimic joint="{source}" multiplier="{multiplier:.12g}" offset="{offset:.12g}"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    urdf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pose.write_text(json.dumps({"pose_name": "oracle", "joints": case["pose"]}, sort_keys=True) + "\n", encoding="utf-8")


def resolved_positions(case: dict[str, Any], independent: dict[str, float]) -> dict[str, float]:
    driver = independent["j_drive"]
    return {
        "j_drive": driver,
        "j_follow": case["multiplier"] * driver + case["offset"],
        "j_mount": 0.0,
        "j_pad": 0.0,
        "j_slide": independent["j_slide"],
        "j_tool": 0.0,
    }


def frames_for(case: dict[str, Any], independent: dict[str, float]) -> tuple[dict[str, Matrix], dict[str, float]]:
    positions = resolved_positions(case, independent)
    specs = case["specs"]
    children: dict[str, list[str]] = {name: [] for name in ("base", "arm", "finger", "carrier", "pad", "tip", "tool")}
    for name, spec in specs.items():
        children[spec["parent"]].append(name)
    frames: dict[str, Matrix] = {"base": identity()}

    def descend(link: str) -> None:
        for name in sorted(children[link]):
            spec = specs[name]
            pre = matmul(frames[link], origin_matrix(spec["xyz"], spec["rpy"]))
            frames[f"joint/{name}"] = pre
            if spec["type"] == "revolute":
                motion = axis_angle(spec["axis"], positions[name])
            elif spec["type"] == "prismatic":
                axis = normalized(spec["axis"])
                motion = translation([component * positions[name] for component in axis])
            else:
                motion = identity()
            frames[spec["child"]] = matmul(pre, motion)
            descend(spec["child"])

    descend("base")
    return frames, positions


def expected_endpoint(case: dict[str, Any], driver: str, direction: str) -> dict[str, Any]:
    baseline = case["pose"][driver]
    lower, upper = case["feasible"][driver]
    step = case["steps"][driver]
    sign = -1.0 if direction == "minus" else 1.0
    available = baseline - lower if sign < 0.0 else upper - baseline
    if available <= EPSILON:
        return {"status": "unavailable_at_feasible_limit", "applied": 0.0, "position": baseline}
    magnitude = min(step, available)
    return {
        "status": "applied_nominal_step" if magnitude >= step - EPSILON else "clipped_to_feasible_limit",
        "applied": sign * magnitude,
        "position": baseline + sign * magnitude,
    }


def validate_endpoint(case: dict[str, Any], driver: str, direction: str, record: dict[str, Any], baseline_frames: dict[str, Matrix], counters: dict[str, int]) -> tuple[dict[str, Matrix] | None, dict[str, float] | None]:
    expected = expected_endpoint(case, driver, direction)
    assert_close(record["status"], expected["status"], f"{driver}.{direction}.status")
    assert_close(record["requested_delta"], (-1.0 if direction == "minus" else 1.0) * case["steps"][driver], f"{driver}.{direction}.requested")
    assert_close(record["applied_delta"], expected["applied"], f"{driver}.{direction}.applied")
    assert_close(record["joint_position"], expected["position"], f"{driver}.{direction}.position")
    counters[expected["status"]] = counters.get(expected["status"], 0) + 1
    if expected["status"] == "unavailable_at_feasible_limit":
        if "frame_deltas" in record:
            raise AssertionError(f"{driver}.{direction}: unavailable endpoint carries frame deltas")
        return None, None
    independent = dict(case["pose"])
    independent[driver] = expected["position"]
    endpoint_frames, positions = frames_for(case, independent)
    assert_close(record["resolved_joint_positions"], {name: clean(value) for name, value in sorted(positions.items())}, f"{driver}.{direction}.resolved")
    physical = ["j_drive", "j_follow"] if driver == "j_drive" else ["j_slide"]
    assert_close(record["physical_joint_positions"], {name: clean(positions[name]) for name in physical}, f"{driver}.{direction}.physical")
    changed: set[str] = set()
    for frame_name, baseline in baseline_frames.items():
        endpoint = endpoint_frames[frame_name]
        delta = record["frame_deltas"][frame_name]
        baseline_origin, endpoint_origin = origin(baseline), origin(endpoint)
        displacement = [endpoint_origin[index] - baseline_origin[index] for index in range(3)]
        relative = matmul(inverse_rigid(baseline), endpoint)
        distance = math.sqrt(sum(value * value for value in displacement))
        angle = rotation_angle(relative)
        assert_close(delta["baseline_origin_root_xyz_m"], baseline_origin, f"{driver}.{direction}.{frame_name}.baseline")
        assert_close(delta["endpoint_origin_root_xyz_m"], endpoint_origin, f"{driver}.{direction}.{frame_name}.endpoint")
        assert_close(delta["origin_displacement_root_xyz_m"], displacement, f"{driver}.{direction}.{frame_name}.displacement")
        assert_close(delta["origin_displacement_norm_m"], distance, f"{driver}.{direction}.{frame_name}.distance")
        assert_close(delta["baseline_frame_from_endpoint_frame"]["translation_xyz_m"], origin(relative), f"{driver}.{direction}.{frame_name}.relative_translation")
        assert_close(delta["baseline_frame_from_endpoint_frame"]["angle_rad"], angle, f"{driver}.{direction}.{frame_name}.angle", 2e-8)
        frame_changed = distance > EPSILON or angle > EPSILON
        assert_close(delta["frame_changed"], frame_changed, f"{driver}.{direction}.{frame_name}.changed")
        if frame_changed:
            changed.add(frame_name)
    structural = set((set(baseline_frames) - {"base", "joint/j_drive"}) if driver == "j_drive" else {"tip", "tool", "joint/j_tool"})
    causality = record["causality_check"]
    assert_close(causality["unexpected_changed_frames"], sorted(changed - structural), f"{driver}.{direction}.unexpected")
    assert_close(causality["structurally_affected_but_endpoint_stationary_frames"], sorted(structural - changed), f"{driver}.{direction}.stationary")
    assert_close(causality["pre_motion_frame_changed"], False, f"{driver}.{direction}.pre_motion")
    return endpoint_frames, positions


def validate_views(manifest: dict[str, Any], driver: str, driver_record: dict[str, Any], sample_frames: dict[str, dict[str, Matrix] | None], output: Path, counters: dict[str, int]) -> None:
    link_names = ["arm", "base", "carrier", "finger", "pad", "tip", "tool"]
    all_points = [origin(frames[name]) for frames in sample_frames.values() if frames is not None for name in link_names]
    motion_id = manifest["motion_id"]
    for view_id, u_axis, v_axis, depth_axis in view_specs():
        view = driver_record["views"][view_id]
        assert_close(view["projection"]["root_xyz_to_uv_matrix_2x3"], [u_axis, v_axis], f"{driver}.{view_id}.projection")
        assert_close(view["projection"]["depth_axis_in_root_xyz"], depth_axis, f"{driver}.{view_id}.depth")
        projected = [(dot(u_axis, point), dot(v_axis, point)) for point in all_points]
        minimum = [min(point[index] for point in projected) for index in range(2)]
        maximum = [max(point[index] for point in projected) for index in range(2)]
        bounds = {"min_uv": vector(minimum), "max_uv": vector(maximum), "extents_uv": vector(maximum[index] - minimum[index] for index in range(2))}
        assert_close(view["combined_projection_bounds_uv_m"], bounds, f"{driver}.{view_id}.bounds")
        span = max(maximum[0] - minimum[0], maximum[1] - minimum[1], 1e-3)
        scale = min(640.0 / span, 438.0 / span)
        center = [(minimum[index] + maximum[index]) / 2.0 for index in range(2)]
        assert_close(view["screen"]["center_uv_m"], center, f"{driver}.{view_id}.center")
        assert_close(view["screen"]["scale_px_per_m"], scale, f"{driver}.{view_id}.scale", 2e-8)
        expected_ids: set[str] = set()
        for sample_name, frames in sample_frames.items():
            sample = view["samples"][sample_name]
            if frames is None:
                assert_close(sample["status"], "unavailable_at_feasible_limit", f"{driver}.{view_id}.{sample_name}.unavailable")
                continue
            link_records = {record["frame_name"]: record for record in sample["link_frames"]}
            for name in link_names:
                point = origin(frames[name])
                uv = [dot(u_axis, point), dot(v_axis, point)]
                pixel = [720.0 / 2.0 + (uv[0] - center[0]) * scale, 86.0 + 438.0 / 2.0 - (uv[1] - center[1]) * scale]
                assert_close(link_records[name]["projected_uv_m"], uv, f"{driver}.{view_id}.{sample_name}.{name}.uv")
                assert_close(link_records[name]["pixel_xy"], pixel, f"{driver}.{view_id}.{sample_name}.{name}.pixel", 2e-5)
                expected_ids.add(f"motion_sample/{motion_id}/{driver}/{sample_name}/frame/{name}")
            for joint_name in sorted(("j_drive", "j_follow", "j_mount", "j_pad", "j_slide", "j_tool")):
                expected_ids.add(f"motion_sample/{motion_id}/{driver}/{sample_name}/joint/{joint_name}")
        for motion_vector in view["motion_vectors"]:
            direction, frame_name = motion_vector["direction"], motion_vector["frame_name"]
            endpoint = sample_frames[direction]
            assert endpoint is not None
            displacement = [origin(endpoint[frame_name])[index] - origin(sample_frames["baseline"][frame_name])[index] for index in range(3)]
            projected_displacement = [dot(u_axis, displacement), dot(v_axis, displacement)]
            assert_close(motion_vector["root_origin_displacement_xyz_m"], displacement, f"{driver}.{view_id}.{direction}.{frame_name}.root_vector")
            assert_close(motion_vector["projected_displacement_uv_m"], projected_displacement, f"{driver}.{view_id}.{direction}.{frame_name}.uv_vector")
            if motion_vector["pixel_displacement_norm"] > 1e-6 or motion_vector["orientation_change_rad"] > EPSILON:
                expected_ids.add(f"motion_vector/{motion_id}/{driver}/{direction}/frame/{frame_name}")
        assert_close(view["expected_svg_entity_ids"], sorted(expected_ids), f"{driver}.{view_id}.entity_ids")
        artifact = output / view["artifact"]["path"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert_close(view["artifact"]["sha256"], digest, f"{driver}.{view_id}.digest")
        svg = artifact.read_text(encoding="utf-8")
        if any(f'data-entity-id="{entity}"' not in svg for entity in expected_ids):
            raise AssertionError(f"{driver}.{view_id}: SVG is missing an expected typed entity")
        counters["verified_views"] += 1


def validate_case(case: dict[str, Any], manifest: dict[str, Any], output: Path, counters: dict[str, int]) -> None:
    assert_close(manifest["schema_version"], "robot-spatial-motion-atlas.v1", "schema")
    assert_close(manifest["coverage"]["independent_drivers"], ["j_drive", "j_slide"], "drivers")
    baseline_frames, baseline_positions = frames_for(case, case["pose"])
    for driver in ("j_drive", "j_slide"):
        record = manifest["drivers"][driver]
        lower, upper = case["feasible"][driver]
        assert_close(record["baseline_position"], case["pose"][driver], f"{driver}.baseline")
        assert_close(record["nominal_step"], case["steps"][driver], f"{driver}.step")
        assert_close(record["feasible_interval"]["minimum"], lower, f"{driver}.minimum")
        assert_close(record["feasible_interval"]["maximum"], upper, f"{driver}.maximum")
        physical = ["j_drive", "j_follow"] if driver == "j_drive" else ["j_slide"]
        assert_close(record["physical_joints_driven"], physical, f"{driver}.physical_joints")
        affected_links = {"arm", "carrier", "finger", "pad", "tip", "tool"} if driver == "j_drive" else {"tip", "tool"}
        assert_close(sorted(record["structural_causality"]["affected_links"]), sorted(affected_links), f"{driver}.affected_links")
        sample_frames: dict[str, dict[str, Matrix] | None] = {"baseline": baseline_frames}
        for direction in ("minus", "plus"):
            frames, _ = validate_endpoint(case, driver, direction, record["endpoints"][direction], baseline_frames, counters)
            sample_frames[direction] = frames
        validate_views(manifest, driver, record, sample_frames, output, counters)
        counters["verified_drivers"] += 1
    available = sum(1 for driver in ("j_drive", "j_slide") for direction in ("minus", "plus") if expected_endpoint(case, driver, direction)["status"] != "unavailable_at_feasible_limit")
    assert_close(manifest["coverage"]["available_signed_endpoint_count"], available, "available endpoint count")
    assert_close(manifest["epistemic_scope"]["time_parameterized_trajectory"], False, "trajectory scope")
    assert_close(manifest["epistemic_scope"]["continuous_swept_volume_or_collision"], False, "swept scope")


def run(cases: int, seed: int, cli: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    counters: dict[str, int] = {
        "verified_cases": 0,
        "verified_drivers": 0,
        "verified_views": 0,
        "applied_nominal_step": 0,
        "clipped_to_feasible_limit": 0,
        "unavailable_at_feasible_limit": 0,
    }
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="motion-atlas-oracle-") as temp_directory:
        root = Path(temp_directory)
        for case_index in range(cases):
            case_root = root / f"case-{case_index:04d}"
            case_root.mkdir()
            case = make_case(rng, case_index)
            urdf, pose, output = case_root / "robot.urdf", case_root / "pose.json", case_root / "atlas"
            write_case(case, urdf, pose)
            command = [
                sys.executable,
                str(cli),
                "motion-atlas",
                str(urdf),
                "--pose",
                str(pose),
                "--motion-angular-step-rad",
                str(case["steps"]["j_drive"]),
                "--motion-linear-step-m",
                str(case["steps"]["j_slide"]),
                "--out",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                failures.append({"case": case_index, "stage": "cli", "stderr": completed.stderr[-1000:]})
                continue
            try:
                validate_case(case, json.loads((output / "manifest.json").read_text(encoding="utf-8")), output, counters)
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                failures.append({"case": case_index, "stage": "independent_oracle", "error": str(error)})
                continue
            counters["verified_cases"] += 1
    return {
        "schema_version": "robot-spatial-motion-atlas-crosscheck.v1",
        "status": "passed" if not failures and counters["verified_cases"] == cases else "failed",
        "seed": seed,
        "requested_cases": cases,
        **counters,
        "failure_count": len(failures),
        "failures": failures,
        "independence": {
            "imports_production_modules": False,
            "invokes_public_cli_as_system_under_test": True,
            "recomputed": [
                "mimic-constrained feasible intervals and endpoint statuses",
                "branched revolute/prismatic/fixed/mimic forward kinematics",
                "all-frame endpoint SE(3) deltas and causal boundaries",
                "shared per-driver orthographic projection and screen mapping",
                "typed SVG identity coverage and artifact digests",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--cli", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    report = run(args.cases, args.seed, args.cli.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
