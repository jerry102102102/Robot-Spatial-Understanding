#!/usr/bin/env python3
"""Deterministic counterfactual motion atlas for URDF kinematic causality."""

from __future__ import annotations

import json
import math
import tempfile
from html import escape
from pathlib import Path
from typing import Any

from spatial_render import (
    RenderError,
    _bounds_2d,
    _canonical_bytes,
    _clean,
    _convex_hull,
    _dot,
    _pixel_bounds,
    _projection_support,
    _sha256_bytes,
    _sha256_path,
    _vector,
    _view_specs,
)


MOTION_ATLAS_SCHEMA = "robot-spatial-motion-atlas.v1"
MOTION_VERIFICATION_SCHEMA = "robot-spatial-motion-atlas-verification.v1"
EPSILON = 1e-10
Vector = list[float]
Matrix = list[list[float]]


class MotionError(ValueError):
    """An invalid counterfactual motion-atlas request."""


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _inverse_rigid(transform: Matrix) -> Matrix:
    result = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for row in range(3):
        for column in range(3):
            result[row][column] = transform[column][row]
        result[row][3] = -sum(transform[column][row] * transform[column][3] for column in range(3))
    return result


def _origin(transform: Matrix) -> Vector:
    return [transform[index][3] for index in range(3)]


def _distance(left: Vector, right: Vector) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _rotation_delta(relative: Matrix) -> dict[str, Any]:
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    if trace > 0.0:
        scale = math.sqrt(max(trace + 1.0, 0.0)) * 2.0
        qw = 0.25 * scale
        qx = (relative[2][1] - relative[1][2]) / scale
        qy = (relative[0][2] - relative[2][0]) / scale
        qz = (relative[1][0] - relative[0][1]) / scale
    elif relative[0][0] > relative[1][1] and relative[0][0] > relative[2][2]:
        scale = math.sqrt(max(1.0 + relative[0][0] - relative[1][1] - relative[2][2], 0.0)) * 2.0
        qw = (relative[2][1] - relative[1][2]) / scale
        qx = 0.25 * scale
        qy = (relative[0][1] + relative[1][0]) / scale
        qz = (relative[0][2] + relative[2][0]) / scale
    elif relative[1][1] > relative[2][2]:
        scale = math.sqrt(max(1.0 + relative[1][1] - relative[0][0] - relative[2][2], 0.0)) * 2.0
        qw = (relative[0][2] - relative[2][0]) / scale
        qx = (relative[0][1] + relative[1][0]) / scale
        qy = 0.25 * scale
        qz = (relative[1][2] + relative[2][1]) / scale
    else:
        scale = math.sqrt(max(1.0 + relative[2][2] - relative[0][0] - relative[1][1], 0.0)) * 2.0
        qw = (relative[1][0] - relative[0][1]) / scale
        qx = (relative[0][2] + relative[2][0]) / scale
        qy = (relative[1][2] + relative[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    quaternion = [qx / norm, qy / norm, qz / norm, qw / norm]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    angle = 2.0 * math.acos(max(-1.0, min(1.0, quaternion[3])))
    sine = math.sqrt(max(0.0, 1.0 - quaternion[3] * quaternion[3]))
    axis = [0.0, 0.0, 0.0] if sine <= EPSILON else [quaternion[index] / sine for index in range(3)]
    return {
        "quaternion_xyzw": _vector(quaternion),
        "angle_rad": _clean(angle),
        "axis_in_baseline_frame_xyz": _vector(axis),
    }


def _frame_delta(baseline: Matrix, endpoint: Matrix) -> dict[str, Any]:
    baseline_origin, endpoint_origin = _origin(baseline), _origin(endpoint)
    displacement = [endpoint_origin[index] - baseline_origin[index] for index in range(3)]
    relative = _matmul(_inverse_rigid(baseline), endpoint)
    rotation = _rotation_delta(relative)
    translation_baseline = _origin(relative)
    origin_distance = _distance(baseline_origin, endpoint_origin)
    return {
        "baseline_origin_root_xyz_m": _vector(baseline_origin),
        "endpoint_origin_root_xyz_m": _vector(endpoint_origin),
        "origin_displacement_root_xyz_m": _vector(displacement),
        "origin_displacement_norm_m": _clean(origin_distance),
        "baseline_frame_from_endpoint_frame": {
            "translation_xyz_m": _vector(translation_baseline),
            **rotation,
        },
        "origin_moved": origin_distance > EPSILON,
        "orientation_changed": rotation["angle_rad"] > EPSILON,
        "frame_changed": origin_distance > EPSILON or rotation["angle_rad"] > EPSILON,
    }


def _aabb(points: list[Vector]) -> dict[str, Vector]:
    minimum = [min(point[index] for point in points) for index in range(3)]
    maximum = [max(point[index] for point in points) for index in range(3)]
    return {
        "min_xyz_m": _vector(minimum),
        "max_xyz_m": _vector(maximum),
        "center_xyz_m": _vector((minimum[index] + maximum[index]) / 2.0 for index in range(3)),
        "extents_xyz_m": _vector(maximum[index] - minimum[index] for index in range(3)),
    }


def _geometry_delta(baseline: list[Vector], endpoint: list[Vector]) -> dict[str, Any]:
    baseline_bounds, endpoint_bounds = _aabb(baseline), _aabb(endpoint)
    union_bounds = _aabb([*baseline, *endpoint])
    center_delta = [
        endpoint_bounds["center_xyz_m"][index] - baseline_bounds["center_xyz_m"][index]
        for index in range(3)
    ]
    return {
        "baseline_aabb_root_m": baseline_bounds,
        "endpoint_aabb_root_m": endpoint_bounds,
        "aabb_center_displacement_root_xyz_m": _vector(center_delta),
        "endpoint_union_aabb_root_m": union_bounds,
        "endpoint_union_is_continuous_swept_volume": False,
    }


def _sample_entity(motion_id: str, driver: str, sample: str, kind: str, name: str) -> str:
    return f"motion_sample/{motion_id}/{driver}/{sample}/{kind}/{name}"


def _svg_for_driver_view(
    motion_id: str,
    driver: str,
    driver_record: dict[str, Any],
    view: dict[str, Any],
    baseline_pose_name: str,
) -> str:
    width, height = view["screen"]["width_px"], view["screen"]["height_px"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{escape(driver)} counterfactual motion — {escape(view["title"])}</title>',
        '<desc>Baseline, feasible signed endpoint perturbations, and frame-origin motion vectors. This is not a continuous trajectory or swept-volume proof.</desc>',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="context-stroke"/></marker></defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="18" font-weight="700" fill="#0f172a">{escape(driver)} counterfactual motion</text>',
        f'<text x="24" y="47" font-family="monospace" font-size="10" fill="#475569">baseline={escape(baseline_pose_name)} type={escape(driver_record["joint_type"])} step={driver_record["nominal_step"]} {escape(driver_record["joint_position_unit"])}</text>',
        f'<text x="24" y="64" font-family="sans-serif" font-size="10" fill="#64748b">{escape(view["axis_label"])}</text>',
        '<rect x="20" y="76" width="680" height="462" rx="8" fill="white" stroke="#cbd5e1"/>',
    ]
    sample_styles = {
        "baseline": ("#475569", "#cbd5e1", "0.20", ""),
        "minus": ("#2563eb", "#bfdbfe", "0.18", ' stroke-dasharray="5 3"'),
        "plus": ("#dc2626", "#fecaca", "0.18", ' stroke-dasharray="5 3"'),
    }
    for sample_name in ("baseline", "minus", "plus"):
        sample = view["samples"].get(sample_name)
        if not isinstance(sample, dict) or sample.get("status") == "unavailable_at_feasible_limit":
            continue
        stroke, fill, opacity, dash = sample_styles[sample_name]
        for geometry in sample["geometry"]:
            hull = geometry["pixel_hull_xy"]
            entity = escape(geometry["entity_id"], quote=True)
            if len(hull) >= 3:
                points = " ".join(f"{point[0]:.6f},{point[1]:.6f}" for point in hull)
                lines.append(
                    f'<polygon data-entity-id="{entity}" points="{points}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1"{dash}/>'
                )
        for edge in sample["kinematic_edges"]:
            start, end = edge["start_pixel_xy"], edge["end_pixel_xy"]
            lines.append(
                f'<line data-entity-id="{escape(edge["entity_id"], quote=True)}" x1="{start[0]:.6f}" y1="{start[1]:.6f}" x2="{end[0]:.6f}" y2="{end[1]:.6f}" stroke="{stroke}" stroke-width="{2 if sample_name == "baseline" else 1.2}"{dash}/>'
            )
        for frame in sample["link_frames"]:
            point = frame["pixel_xy"]
            lines.append(
                f'<circle data-entity-id="{escape(frame["entity_id"], quote=True)}" cx="{point[0]:.6f}" cy="{point[1]:.6f}" r="{2.3 if sample_name == "baseline" else 1.8}" fill="{stroke}"/>'
            )
    for vector in view["motion_vectors"]:
        start, end = vector["start_pixel_xy"], vector["end_pixel_xy"]
        color = "#2563eb" if vector["direction"] == "minus" else "#dc2626"
        if vector["pixel_displacement_norm"] <= 1e-6:
            if vector["orientation_change_rad"] > EPSILON:
                lines.append(
                    f'<circle data-entity-id="{escape(vector["entity_id"], quote=True)}" cx="{start[0]:.6f}" cy="{start[1]:.6f}" r="4" fill="none" stroke="{color}" stroke-width="1.2"/>'
                )
            continue
        lines.append(
            f'<line data-entity-id="{escape(vector["entity_id"], quote=True)}" x1="{start[0]:.6f}" y1="{start[1]:.6f}" x2="{end[0]:.6f}" y2="{end[1]:.6f}" stroke="{color}" stroke-width="1.1" marker-end="url(#arrow)"/>'
        )
    lines.extend([
        '<line x1="28" y1="558" x2="48" y2="558" stroke="#475569" stroke-width="2"/><text x="54" y="562" font-family="sans-serif" font-size="10" fill="#334155">baseline</text>',
        '<line x1="128" y1="558" x2="148" y2="558" stroke="#2563eb" stroke-width="2"/><text x="154" y="562" font-family="sans-serif" font-size="10" fill="#334155">minus endpoint</text>',
        '<line x1="278" y1="558" x2="298" y2="558" stroke="#dc2626" stroke-width="2"/><text x="304" y="562" font-family="sans-serif" font-size="10" fill="#334155">plus endpoint</text>',
        '<text x="24" y="590" font-family="sans-serif" font-size="10" fill="#64748b">Finite exact-FK endpoints; not dynamics, time, interpolation, continuous collision, or swept volume.</text>',
        '</svg>',
        '',
    ])
    return "\n".join(lines)


def write_counterfactual_motion_atlas(
    output_directory: Path,
    model: Any,
    pose_name: str,
    supplied_pose: dict[str, float],
    source_binding: dict[str, Any],
    inspect_meshes: bool = False,
    package_map_path: Path | None = None,
    inspect_mesh_kinds: list[str] | None = None,
    angular_step_rad: float = 0.1,
    linear_step_m: float = 0.01,
) -> dict[str, Any]:
    """Write per-driver four-view finite counterfactual motion artifacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    baseline_frames, baseline_pose = model.world_frames(supplied_pose)
    baseline_analysis, baseline_points, _ = model.geometry_analysis(
        baseline_pose,
        inspect_meshes=inspect_meshes,
        package_map_path=package_map_path,
        inspect_mesh_kinds=inspect_mesh_kinds,
    )
    renderable = {
        name: points for name, points in baseline_points.items()
        if points and baseline_analysis.get(name, {}).get("status") == "measured"
    }
    frame_semantics = model.frame_semantics()
    link_names = sorted(name for name, record in frame_semantics.items() if record["type"] == "link")
    drivers = sorted(
        name for name, joint in model.joints.items()
        if joint.type != "fixed" and joint.mimic is None
    )
    if not drivers:
        raise MotionError("counterfactual motion atlas requires at least one independent movable joint")
    baseline_binding = {
        "name": pose_name,
        "joint_positions": {name: _clean(value) for name, value in sorted(baseline_pose.items())},
    }
    baseline_binding["sha256"] = _sha256_bytes(_canonical_bytes(baseline_binding))
    prepared: dict[str, Any] = {}
    digest_samples: dict[str, Any] = {}
    for driver in drivers:
        driver_spec = model.motion_driver_counterfactuals(
            driver,
            baseline_pose,
            angular_step_rad,
            linear_step_m,
        )
        samples: dict[str, Any] = {
            "baseline": {
                "status": "baseline",
                "joint_positions": baseline_pose,
                "frames": baseline_frames,
                "geometry_analysis": baseline_analysis,
                "geometry_points": baseline_points,
            }
        }
        for direction in ("minus", "plus"):
            endpoint = driver_spec["endpoints"][direction]
            if "resolved_joint_positions" not in endpoint:
                samples[direction] = {"status": endpoint["status"]}
                continue
            frames, positions = model.world_frames(endpoint["resolved_joint_positions"])
            analysis, points, _ = model.geometry_analysis(
                positions,
                inspect_meshes=inspect_meshes,
                package_map_path=package_map_path,
                inspect_mesh_kinds=inspect_mesh_kinds,
            )
            samples[direction] = {
                "status": endpoint["status"],
                "joint_positions": positions,
                "frames": frames,
                "geometry_analysis": analysis,
                "geometry_points": points,
            }
        prepared[driver] = {"spec": driver_spec, "samples": samples}
        digest_samples[driver] = {
            "driver_spec": driver_spec,
            "samples": {
                sample_name: (
                    {"status": sample["status"]}
                    if "frames" not in sample
                    else {
                        "status": sample["status"],
                        "joint_positions": {name: _clean(value) for name, value in sorted(sample["joint_positions"].items())},
                        "frame_matrices": {
                            name: [[_clean(value) for value in row] for row in matrix]
                            for name, matrix in sorted(sample["frames"].items())
                        },
                        "geometry_points_in_root_m": {
                            name: [_vector(point) for point in points]
                            for name, points in sorted(sample["geometry_points"].items())
                            if points and sample["geometry_analysis"].get(name, {}).get("status") == "measured"
                        },
                    }
                ) for sample_name, sample in sorted(samples.items())
            },
        }
    motion_input = {
        "source_binding": source_binding,
        "baseline_pose_binding": baseline_binding,
        "perturbation_policy": {
            "angular_step_rad": _clean(angular_step_rad),
            "linear_step_m": _clean(linear_step_m),
            "directions": ["minus", "plus"],
            "limit_policy": "clip each signed endpoint to the exact declared driver/mimic feasible interval; unavailable at an active bound",
            "other_independent_drivers": "held_fixed_at_baseline",
        },
        "drivers": digest_samples,
    }
    motion_input_sha256 = _sha256_bytes(_canonical_bytes(motion_input))
    motion_id = f"motion-{motion_input_sha256[:20]}"
    driver_records: dict[str, Any] = {}
    total_available_endpoints = 0
    for driver, prepared_driver in prepared.items():
        spec, samples = prepared_driver["spec"], prepared_driver["samples"]
        structural = spec["structural_causality"]
        endpoint_records: dict[str, Any] = {}
        for direction in ("minus", "plus"):
            endpoint_spec = spec["endpoints"][direction]
            sample = samples[direction]
            if "frames" not in sample:
                endpoint_records[direction] = endpoint_spec
                continue
            total_available_endpoints += 1
            frame_deltas = {
                frame_name: _frame_delta(baseline_frames[frame_name], sample["frames"][frame_name])
                for frame_name in sorted(baseline_frames)
            }
            changed = sorted(name for name, delta in frame_deltas.items() if delta["frame_changed"])
            expected = set(structural["affected_frames"])
            geometry_deltas = {
                name: _geometry_delta(renderable[name], sample["geometry_points"][name])
                for name in sorted(renderable)
                if sample["geometry_analysis"].get(name, {}).get("status") == "measured"
                and sample["geometry_points"].get(name)
            }
            endpoint_records[direction] = {
                **endpoint_spec,
                "frame_deltas": frame_deltas,
                "link_frame_deltas": {name: frame_deltas[name] for name in link_names},
                "geometry_endpoint_deltas": geometry_deltas,
                "causality_check": {
                    "structurally_affected_frame_count": len(expected),
                    "numerically_changed_frame_count": len(changed),
                    "unexpected_changed_frames": sorted(set(changed) - expected),
                    "structurally_affected_but_endpoint_stationary_frames": sorted(expected - set(changed)),
                    "pre_motion_frame": structural["pre_motion_frame"],
                    "pre_motion_frame_changed": frame_deltas[structural["pre_motion_frame"]]["frame_changed"],
                },
            }
        driver_record: dict[str, Any] = {
            key: spec[key] for key in (
                "driver_joint",
                "joint_type",
                "joint_position_unit",
                "baseline_position",
                "nominal_step",
                "feasible_interval",
                "physical_joints_driven",
                "baseline_physical_joint_positions",
                "structural_causality",
            )
        }
        driver_record["endpoints"] = endpoint_records
        driver_record["views"] = {}
        all_driver_points: list[Vector] = []
        for sample in samples.values():
            if "frames" not in sample:
                continue
            all_driver_points.extend(_origin(sample["frames"][name]) for name in link_names)
            for name, points in sample["geometry_points"].items():
                if points and sample["geometry_analysis"].get(name, {}).get("status") == "measured":
                    all_driver_points.extend(points)
        if not all_driver_points:
            all_driver_points = [[0.0, 0.0, 0.0]]
        for view_spec in _view_specs():
            u_axis = view_spec["u_axis_in_root_xyz"]
            v_axis = view_spec["v_axis_in_root_xyz"]
            depth_axis = view_spec["depth_axis_in_root_xyz"]
            project = lambda point: (_dot(u_axis, point), _dot(v_axis, point))
            projected_all = [project(point) for point in all_driver_points]
            bounds = _bounds_2d(projected_all)
            min_u, min_v = bounds["min_uv"]
            max_u, max_v = bounds["max_uv"]
            span = max(max_u - min_u, max_v - min_v, 1e-3)
            width, height, plot_left, plot_top, plot_width, plot_height = 720, 610, 40.0, 86.0, 640.0, 438.0
            scale = min(plot_width / span, plot_height / span)
            center = [(min_u + max_u) / 2.0, (min_v + max_v) / 2.0]
            screen = lambda uv: (
                width / 2.0 + (uv[0] - center[0]) * scale,
                plot_top + plot_height / 2.0 - (uv[1] - center[1]) * scale,
            )
            view_samples: dict[str, Any] = {}
            expected_ids: set[str] = set()
            for sample_name in ("baseline", "minus", "plus"):
                sample = samples[sample_name]
                if "frames" not in sample:
                    view_samples[sample_name] = {"status": sample["status"]}
                    continue
                geometry_records: list[dict[str, Any]] = []
                for frame_name, points in sorted(sample["geometry_points"].items()):
                    analysis = sample["geometry_analysis"].get(frame_name, {})
                    if not points or analysis.get("status") != "measured":
                        continue
                    projected = [project(point) for point in points]
                    hull = _convex_hull(projected)
                    pixel_hull = [screen(point) for point in hull]
                    depths = [_dot(depth_axis, point) for point in points]
                    entity_id = _sample_entity(motion_id, driver, sample_name, "frame", frame_name)
                    expected_ids.add(entity_id)
                    geometry_records.append({
                        "entity_id": entity_id,
                        "frame_name": frame_name,
                        "kind": analysis["kind"],
                        "owner_link": analysis["link"],
                        "geometry_type": analysis["geometry_type"],
                        "projection_support": _projection_support(analysis),
                        "projected_hull_uv_m": [_vector(point) for point in hull],
                        "projection_bounds_uv_m": _bounds_2d(projected),
                        "pixel_hull_xy": [_vector(point, 6) for point in pixel_hull],
                        "pixel_bounds_xy": _pixel_bounds(pixel_hull),
                        "depth_interval_m": [_clean(min(depths)), _clean(max(depths))],
                    })
                link_records: list[dict[str, Any]] = []
                for link_name in link_names:
                    origin = _origin(sample["frames"][link_name])
                    uv = project(origin)
                    entity_id = _sample_entity(motion_id, driver, sample_name, "frame", link_name)
                    expected_ids.add(entity_id)
                    link_records.append({
                        "entity_id": entity_id,
                        "frame_name": link_name,
                        "origin_root_xyz_m": _vector(origin),
                        "projected_uv_m": _vector(uv),
                        "pixel_xy": _vector(screen(uv), 6),
                    })
                edge_records: list[dict[str, Any]] = []
                for joint_name, joint in sorted(model.joints.items()):
                    start = _origin(sample["frames"][joint.parent])
                    end = _origin(sample["frames"][joint.child])
                    start_uv, end_uv = project(start), project(end)
                    entity_id = _sample_entity(motion_id, driver, sample_name, "joint", joint_name)
                    expected_ids.add(entity_id)
                    edge_records.append({
                        "entity_id": entity_id,
                        "joint_name": joint_name,
                        "start_root_xyz_m": _vector(start),
                        "end_root_xyz_m": _vector(end),
                        "start_pixel_xy": _vector(screen(start_uv), 6),
                        "end_pixel_xy": _vector(screen(end_uv), 6),
                    })
                view_samples[sample_name] = {
                    "status": sample["status"],
                    "joint_positions": {name: _clean(value) for name, value in sorted(sample["joint_positions"].items())},
                    "geometry": geometry_records,
                    "link_frames": link_records,
                    "kinematic_edges": edge_records,
                }
            baseline_link_by_name = {record["frame_name"]: record for record in view_samples["baseline"]["link_frames"]}
            motion_vectors: list[dict[str, Any]] = []
            for direction in ("minus", "plus"):
                endpoint_view = view_samples[direction]
                if "link_frames" not in endpoint_view:
                    continue
                for endpoint_frame in endpoint_view["link_frames"]:
                    frame_name = endpoint_frame["frame_name"]
                    baseline_frame = baseline_link_by_name[frame_name]
                    start, end = baseline_frame["pixel_xy"], endpoint_frame["pixel_xy"]
                    delta_uv = [
                        endpoint_frame["projected_uv_m"][index] - baseline_frame["projected_uv_m"][index]
                        for index in range(2)
                    ]
                    delta_pixel = [end[index] - start[index] for index in range(2)]
                    entity_id = f"motion_vector/{motion_id}/{driver}/{direction}/frame/{frame_name}"
                    vector_record = {
                        "entity_id": entity_id,
                        "direction": direction,
                        "frame_name": frame_name,
                        "start_pixel_xy": start,
                        "end_pixel_xy": end,
                        "projected_displacement_uv_m": _vector(delta_uv),
                        "projected_displacement_norm_m": _clean(math.dist([0.0, 0.0], delta_uv)),
                        "pixel_displacement_xy": _vector(delta_pixel, 6),
                        "pixel_displacement_norm": _clean(math.dist([0.0, 0.0], delta_pixel), 6),
                        "root_origin_displacement_xyz_m": endpoint_records[direction]["link_frame_deltas"][frame_name]["origin_displacement_root_xyz_m"],
                        "root_origin_displacement_norm_m": endpoint_records[direction]["link_frame_deltas"][frame_name]["origin_displacement_norm_m"],
                        "orientation_change_rad": endpoint_records[direction]["link_frame_deltas"][frame_name]["baseline_frame_from_endpoint_frame"]["angle_rad"],
                    }
                    motion_vectors.append(vector_record)
                    if vector_record["pixel_displacement_norm"] > 1e-6 or vector_record["orientation_change_rad"] > EPSILON:
                        expected_ids.add(entity_id)
            view = {
                "view_id": view_spec["id"],
                "title": view_spec["title"],
                "axis_label": view_spec["axis_label"],
                "projection": {
                    "type": "orthographic",
                    "root_xyz_to_uv_matrix_2x3": [_vector(u_axis), _vector(v_axis)],
                    "depth_axis_in_root_xyz": _vector(depth_axis),
                },
                "screen": {
                    "width_px": width,
                    "height_px": height,
                    "plot_rect_xywh_px": [plot_left, plot_top, plot_width, plot_height],
                    "center_uv_m": _vector(center),
                    "scale_px_per_m": _clean(scale),
                    "fit_scope": "baseline_and_all_available_signed_endpoints_for_this_driver",
                    "mapping": "pixel_x=width/2+(u-center_u)*scale; pixel_y=plot_top+plot_height/2-(v-center_v)*scale",
                },
                "combined_projection_bounds_uv_m": bounds,
                "samples": view_samples,
                "motion_vectors": motion_vectors,
                "expected_svg_entity_ids": sorted(expected_ids),
            }
            svg_directory = output_directory / "drivers" / driver
            svg_directory.mkdir(parents=True, exist_ok=True)
            svg_path = svg_directory / f"{view_spec['id']}.svg"
            svg_path.write_text(_svg_for_driver_view(motion_id, driver, driver_record, view, pose_name), encoding="utf-8")
            view["artifact"] = {
                "path": f"drivers/{driver}/{view_spec['id']}.svg",
                "sha256": _sha256_path(svg_path),
                "format": "svg",
            }
            driver_record["views"][view_spec["id"]] = view
        driver_records[driver] = driver_record
    unrendered = sorted(
        name for name, record in baseline_analysis.items()
        if record.get("status") != "measured" or not baseline_points.get(name)
    )
    manifest = {
        "schema_version": MOTION_ATLAS_SCHEMA,
        "motion_id": motion_id,
        "motion_input_sha256": motion_input_sha256,
        "source_binding": source_binding,
        "baseline_pose_binding": baseline_binding,
        "perturbation_policy": motion_input["perturbation_policy"],
        "coordinate_contract": {
            "root_frame": source_binding["root_frame"],
            "world_point_order": "xyz",
            "projected_point_order": "uv",
            "pixel_point_order": "xy",
            "length_unit": "m",
            "angular_unit": "rad",
            "frame_delta_transform": "baseline_frame_from_endpoint_frame",
        },
        "coverage": {
            "independent_driver_count": len(drivers),
            "independent_drivers": drivers,
            "available_signed_endpoint_count": total_available_endpoints,
            "requested_signed_endpoint_count": 2 * len(drivers),
            "link_frame_count": len(link_names),
            "declared_geometry_count": len(baseline_analysis),
            "rendered_geometry_count": len(renderable),
            "unrendered_geometry_frames": unrendered,
            "view_count_per_driver": 4,
        },
        "drivers": driver_records,
        "epistemic_scope": {
            "purpose": "make joint-as-cause and downstream-structure-as-effect explicit through exact finite FK counterfactual endpoints",
            "same_kinematics_and_geometry_oracle_as_canonical_model": True,
            "independent_motion_or_physical_oracle": False,
            "time_parameterized_trajectory": False,
            "velocity_acceleration_effort_or_dynamics": False,
            "continuous_swept_volume_or_collision": False,
            "intermediate_motion_between_endpoints": "not_evaluated",
            "hardware_motion_or_safety": "not_established",
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_counterfactual_motion_atlas(
    manifest_path: Path,
    model: Any,
    pose_name: str,
    supplied_pose: dict[str, float],
    source_binding: dict[str, Any],
    inspect_meshes: bool = False,
    package_map_path: Path | None = None,
    inspect_mesh_kinds: list[str] | None = None,
    angular_step_rad: float = 0.1,
    linear_step_m: float = 0.01,
) -> dict[str, Any]:
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MotionError(f"cannot read counterfactual motion atlas {manifest_path}: {error}") from error
    if not isinstance(actual, dict):
        raise MotionError("counterfactual motion atlas manifest must be a JSON object")
    issues: list[dict[str, Any]] = []
    issue = lambda check, expected, observed: issues.append({"check": check, "expected": expected, "observed": observed})
    with tempfile.TemporaryDirectory(prefix="robot-spatial-motion-verify-") as temp_directory:
        expected = write_counterfactual_motion_atlas(
            Path(temp_directory),
            model,
            pose_name,
            supplied_pose,
            source_binding,
            inspect_meshes,
            package_map_path,
            inspect_mesh_kinds,
            angular_step_rad,
            linear_step_m,
        )
    for key in (
        "schema_version",
        "motion_id",
        "motion_input_sha256",
        "source_binding",
        "baseline_pose_binding",
        "perturbation_policy",
        "coordinate_contract",
        "coverage",
        "drivers",
        "epistemic_scope",
    ):
        if actual.get(key) != expected[key]:
            issue(f"manifest.{key}", expected[key], actual.get(key))
    manifest_directory = manifest_path.parent.resolve()
    verified_views = 0
    drivers = actual.get("drivers", {})
    if isinstance(drivers, dict):
        for driver, driver_record in drivers.items():
            if not isinstance(driver_record, dict) or not isinstance(driver_record.get("views"), dict):
                issue(f"drivers.{driver}.views", "object", None if not isinstance(driver_record, dict) else driver_record.get("views"))
                continue
            for view_id, view in driver_record["views"].items():
                if not isinstance(view, dict) or not isinstance(view.get("artifact"), dict):
                    issue(f"drivers.{driver}.views.{view_id}.artifact", "object", None)
                    continue
                relative = view["artifact"].get("path")
                if not isinstance(relative, str):
                    issue(f"drivers.{driver}.views.{view_id}.artifact.path", "relative string", relative)
                    continue
                artifact_path = (manifest_directory / relative).resolve()
                try:
                    artifact_path.relative_to(manifest_directory)
                except ValueError:
                    issue(f"drivers.{driver}.views.{view_id}.artifact.path", "path inside motion atlas", relative)
                    continue
                if not artifact_path.is_file():
                    issue(f"drivers.{driver}.views.{view_id}.artifact.exists", True, False)
                    continue
                verified_views += 1
                observed_sha = _sha256_path(artifact_path)
                if observed_sha != view["artifact"].get("sha256"):
                    issue(f"drivers.{driver}.views.{view_id}.artifact.sha256", view["artifact"].get("sha256"), observed_sha)
                svg = artifact_path.read_text(encoding="utf-8")
                missing = [
                    entity_id for entity_id in view.get("expected_svg_entity_ids", [])
                    if f'data-entity-id="{escape(entity_id, quote=True)}"' not in svg
                ]
                if missing:
                    issue(f"drivers.{driver}.views.{view_id}.svg_entity_ids", [], missing)
    return {
        "schema_version": MOTION_VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_path(manifest_path),
        "motion_id": actual.get("motion_id"),
        "motion_input_sha256": actual.get("motion_input_sha256"),
        "verified_driver_count": len(drivers) if isinstance(drivers, dict) else 0,
        "verified_view_count": verified_views,
        "checks": [
            "model, baseline pose, perturbation policy, and signed endpoint binding",
            "mimic-constrained feasible intervals and endpoint clipping",
            "all-frame SE(3), link-origin, geometry endpoint, and structural-causality deltas",
            "shared per-driver projection/screen mappings and motion vectors",
            "standalone SVG digests and typed entity IDs",
        ],
        "issue_count": len(issues),
        "issues": issues,
        "epistemic_scope": "verifies exact finite endpoint regeneration from the same FK/geometry implementation; not continuous motion, dynamics, physical observation, or safety",
    }
