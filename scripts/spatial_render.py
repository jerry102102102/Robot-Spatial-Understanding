#!/usr/bin/env python3
"""Deterministic semantic orthographic projections for robot spatial grounding."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from html import escape
from pathlib import Path
from typing import Any, Iterable


Vector = list[float]
EPSILON = 1e-12
ATLAS_SCHEMA = "robot-spatial-render-atlas.v1"


class RenderError(ValueError):
    """An invalid semantic render request."""


def _clean(value: float, digits: int = 12) -> float:
    return 0.0 if abs(value) < EPSILON else round(value, digits)


def _vector(values: Iterable[float], digits: int = 12) -> Vector:
    return [_clean(float(value), digits) for value in values]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dot(left: Vector, right: Vector) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _distance(left: Vector, right: Vector) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _rotate(matrix: list[list[float]], vector: Vector) -> Vector:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

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


def _bounds_2d(points: list[tuple[float, float]], digits: int = 12) -> dict[str, list[float]]:
    if not points:
        raise RenderError("cannot compute 2D bounds from no points")
    minimum = [min(point[index] for point in points) for index in range(2)]
    maximum = [max(point[index] for point in points) for index in range(2)]
    return {
        "min_uv": _vector(minimum, digits),
        "max_uv": _vector(maximum, digits),
        "extents_uv": _vector((maximum[index] - minimum[index] for index in range(2)), digits),
    }


def _pixel_bounds(points: list[tuple[float, float]]) -> dict[str, list[float]]:
    if not points:
        raise RenderError("cannot compute pixel bounds from no points")
    minimum = [min(point[index] for point in points) for index in range(2)]
    maximum = [max(point[index] for point in points) for index in range(2)]
    return {
        "min_xy": _vector(minimum, 6),
        "max_xy": _vector(maximum, 6),
        "extents_xy": _vector((maximum[index] - minimum[index] for index in range(2)), 6),
    }


def _projection_support(record: dict[str, Any]) -> dict[str, Any]:
    geometry_type = record.get("geometry_type")
    if geometry_type == "mesh":
        return {
            "points": "all_transformed_mesh_vertices",
            "convex_hull": "exact_for_loaded_vertex_set",
            "surface_visibility_or_occlusion": "not_computed",
        }
    if geometry_type == "box":
        return {
            "points": "eight_transformed_box_corners",
            "convex_hull": "exact_for_declared_box_projection",
            "surface_visibility_or_occlusion": "not_computed",
        }
    if geometry_type == "cylinder":
        return {
            "points": "two_32_sample_transformed_boundary_rings",
            "convex_hull": "deterministic_curve_approximation",
            "surface_visibility_or_occlusion": "not_computed",
        }
    if geometry_type == "sphere":
        return {
            "points": "three_24_sample_transformed_great_circles",
            "convex_hull": "deterministic_curve_approximation",
            "surface_visibility_or_occlusion": "not_computed",
        }
    return {
        "points": "unknown",
        "convex_hull": "not_established",
        "surface_visibility_or_occlusion": "not_computed",
    }


def _view_specs() -> list[dict[str, Any]]:
    inverse_sqrt_2 = 1.0 / math.sqrt(2.0)
    inverse_sqrt_3 = 1.0 / math.sqrt(3.0)
    inverse_sqrt_6 = 1.0 / math.sqrt(6.0)
    return [
        {
            "id": "front",
            "title": "Front (X-Z)",
            "u_axis_in_root_xyz": [1.0, 0.0, 0.0],
            "v_axis_in_root_xyz": [0.0, 0.0, 1.0],
            "depth_axis_in_root_xyz": [0.0, 1.0, 0.0],
            "axis_label": "+X right, +Z up; depth coordinate is +Y",
        },
        {
            "id": "side",
            "title": "Side (Y-Z)",
            "u_axis_in_root_xyz": [0.0, 1.0, 0.0],
            "v_axis_in_root_xyz": [0.0, 0.0, 1.0],
            "depth_axis_in_root_xyz": [1.0, 0.0, 0.0],
            "axis_label": "+Y right, +Z up; depth coordinate is +X",
        },
        {
            "id": "top",
            "title": "Top (X-Y)",
            "u_axis_in_root_xyz": [1.0, 0.0, 0.0],
            "v_axis_in_root_xyz": [0.0, 1.0, 0.0],
            "depth_axis_in_root_xyz": [0.0, 0.0, 1.0],
            "axis_label": "+X right, +Y up; depth coordinate is +Z",
        },
        {
            "id": "isometric",
            "title": "Isometric",
            "u_axis_in_root_xyz": [inverse_sqrt_2, -inverse_sqrt_2, 0.0],
            "v_axis_in_root_xyz": [-inverse_sqrt_6, -inverse_sqrt_6, 2.0 * inverse_sqrt_6],
            "depth_axis_in_root_xyz": [inverse_sqrt_3, inverse_sqrt_3, inverse_sqrt_3],
            "axis_label": "orthographic basis; depth coordinate follows +[1,1,1]",
        },
    ]


def _svg_for_view(
    view: dict[str, Any],
    geometry_records: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    axis_records: list[dict[str, Any]],
    source_digest: str,
    pose_name: str,
) -> str:
    width, height = view["screen"]["width_px"], view["screen"]["height_px"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title>{escape(view["title"])} semantic robot projection</title>',
        f'<desc>Digest-bound semantic hull view at pose {escape(pose_name)}. Geometry visibility and occlusion are not computed.</desc>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="24" y="30" font-family="sans-serif" font-size="20" font-weight="700" fill="#0f172a">{escape(view["title"])}</text>',
        f'<text x="24" y="50" font-family="monospace" font-size="10" fill="#475569">pose={escape(pose_name)} input={source_digest[:16]}</text>',
        f'<text x="24" y="66" font-family="sans-serif" font-size="10" fill="#64748b">{escape(view["axis_label"])}</text>',
        '<rect x="20" y="78" width="680" height="460" rx="8" fill="white" stroke="#cbd5e1"/>',
    ]
    for edge in edge_records:
        start, end = edge["start_pixel_xy"], edge["end_pixel_xy"]
        entity = escape(edge["entity_id"], quote=True)
        lines.append(
            f'<line data-entity-id="{entity}" x1="{start[0]:.6f}" y1="{start[1]:.6f}" '
            f'x2="{end[0]:.6f}" y2="{end[1]:.6f}" stroke="#334155" stroke-width="2"/>'
        )
    colors = {"visual": ("#2563eb", "#93c5fd"), "collision": ("#dc2626", "#fecaca")}
    for geometry in geometry_records:
        stroke, fill = colors.get(geometry["kind"], ("#7c3aed", "#ddd6fe"))
        dash = ' stroke-dasharray="6 4"' if geometry["kind"] == "collision" else ""
        entity = escape(geometry["entity_id"], quote=True)
        hull = geometry["pixel_hull_xy"]
        if len(hull) >= 3:
            points = " ".join(f"{point[0]:.6f},{point[1]:.6f}" for point in hull)
            lines.append(
                f'<polygon data-entity-id="{entity}" points="{points}" fill="{fill}" fill-opacity="0.34" '
                f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
            )
        elif len(hull) == 2:
            lines.append(
                f'<line data-entity-id="{entity}" x1="{hull[0][0]:.6f}" y1="{hull[0][1]:.6f}" '
                f'x2="{hull[1][0]:.6f}" y2="{hull[1][1]:.6f}" stroke="{stroke}" stroke-width="2"{dash}/>'
            )
        elif hull:
            lines.append(
                f'<circle data-entity-id="{entity}" cx="{hull[0][0]:.6f}" cy="{hull[0][1]:.6f}" r="3" fill="{stroke}"/>'
            )
    for frame in frame_records:
        point = frame["pixel_xy"]
        entity = escape(frame["entity_id"], quote=True)
        radius = 3.5 if frame["highlighted"] else 2.0
        lines.append(
            f'<circle data-entity-id="{entity}" cx="{point[0]:.6f}" cy="{point[1]:.6f}" r="{radius}" '
            f'fill="#0f172a" fill-opacity="{1.0 if frame["highlighted"] else 0.55}"/>'
        )
        if frame["highlighted"]:
            lines.append(
                f'<text x="{point[0] + 5:.6f}" y="{point[1] - 5:.6f}" font-family="sans-serif" '
                f'font-size="10" fill="#0f172a">{escape(frame["frame_name"])}</text>'
            )
    axis_colors = {"x": "#ef4444", "y": "#16a34a", "z": "#2563eb"}
    for axis in axis_records:
        start, end = axis["origin_pixel_xy"], axis["endpoint_pixel_xy"]
        entity = escape(axis["entity_id"], quote=True)
        color = axis_colors[axis["axis"]]
        lines.append(
            f'<line data-entity-id="{entity}" x1="{start[0]:.6f}" y1="{start[1]:.6f}" '
            f'x2="{end[0]:.6f}" y2="{end[1]:.6f}" stroke="{color}" stroke-width="1.5"/>'
        )
    lines.extend([
        '<rect x="25" y="558" width="14" height="8" fill="#93c5fd" stroke="#2563eb"/>',
        '<text x="45" y="567" font-family="sans-serif" font-size="11" fill="#334155">visual hull</text>',
        '<rect x="145" y="558" width="14" height="8" fill="#fecaca" stroke="#dc2626" stroke-dasharray="5 3"/>',
        '<text x="165" y="567" font-family="sans-serif" font-size="11" fill="#334155">collision hull</text>',
        '<text x="300" y="567" font-family="sans-serif" font-size="11" fill="#334155">frame axes: X red, Y green, Z blue</text>',
        '<text x="24" y="590" font-family="sans-serif" font-size="10" fill="#64748b">Semantic convex projections; not photorealistic rendering, visibility, depth ordering, or an independent geometry oracle.</text>',
        '</svg>',
    ])
    return "\n".join(lines) + "\n"


def write_semantic_render_atlas(
    output_directory: Path,
    geometry_points: dict[str, list[Vector]],
    analysis: dict[str, dict[str, Any]],
    frames: dict[str, dict[str, Any]],
    joints: dict[str, dict[str, Any]],
    highlight_frames: list[str],
    source_binding: dict[str, Any],
    pose_name: str,
    joint_positions: dict[str, float],
    combined_overview_path: Path | None = None,
) -> dict[str, Any]:
    """Write four standalone semantic SVGs and a machine-verifiable projection manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    views_directory = output_directory / "views"
    views_directory.mkdir(parents=True, exist_ok=True)
    link_frame_names = sorted(name for name, record in frames.items() if record["type"] == "link")
    link_origins = {
        name: list(frames[name]["world_from_frame"]["translation_xyz_m"])
        for name in link_frame_names
    }
    renderable_geometry = {
        name: [list(point) for point in points]
        for name, points in sorted(geometry_points.items())
        if points and analysis.get(name, {}).get("status") == "measured"
    }
    unrendered = sorted(
        name for name, record in analysis.items()
        if record.get("status") != "measured" or not geometry_points.get(name)
    )
    all_points = [point for points in renderable_geometry.values() for point in points]
    all_points.extend(link_origins.values())
    if not all_points:
        all_points = [[0.0, 0.0, 0.0]]
    pose_binding = {
        "name": pose_name,
        "joint_positions": {name: _clean(value) for name, value in sorted(joint_positions.items())},
    }
    pose_binding["sha256"] = _sha256_bytes(_canonical_bytes(pose_binding))
    render_input = {
        "source_binding": source_binding,
        "pose_binding": pose_binding,
        "geometry_points_in_root_m": renderable_geometry,
        "geometry_status": {name: analysis[name].get("status") for name in sorted(analysis)},
        "highlight_frames": sorted(set(highlight_frames)),
    }
    render_input_sha256 = _sha256_bytes(_canonical_bytes(render_input))
    atlas_views: dict[str, Any] = {}
    for spec in _view_specs():
        u_axis, v_axis, depth_axis = spec["u_axis_in_root_xyz"], spec["v_axis_in_root_xyz"], spec["depth_axis_in_root_xyz"]

        def project(point: Vector) -> tuple[float, float]:
            return _dot(u_axis, point), _dot(v_axis, point)

        projected_all = [project(point) for point in all_points]
        scene_bounds = _bounds_2d(projected_all)
        min_u, min_v = scene_bounds["min_uv"]
        max_u, max_v = scene_bounds["max_uv"]
        span = max(max_u - min_u, max_v - min_v, 1e-3)
        width, height, plot_left, plot_top, plot_width, plot_height = 720, 610, 40.0, 88.0, 640.0, 430.0
        scale = min(plot_width / span, plot_height / span)
        center = [(min_u + max_u) / 2.0, (min_v + max_v) / 2.0]

        def screen(point_uv: tuple[float, float]) -> tuple[float, float]:
            return (
                width / 2.0 + (point_uv[0] - center[0]) * scale,
                plot_top + plot_height / 2.0 - (point_uv[1] - center[1]) * scale,
            )

        geometry_records: list[dict[str, Any]] = []
        for frame_name, points in renderable_geometry.items():
            projected = [project(point) for point in points]
            hull = _convex_hull(projected)
            pixel_hull = [screen(point) for point in hull]
            depths = [_dot(depth_axis, point) for point in points]
            geometry_records.append({
                "entity_id": f"frame/{frame_name}",
                "frame_name": frame_name,
                "kind": analysis[frame_name]["kind"],
                "owner_link": analysis[frame_name]["link"],
                "geometry_type": analysis[frame_name]["geometry_type"],
                "point_count": len(points),
                "projection_support": _projection_support(analysis[frame_name]),
                "projected_hull_uv_m": [_vector(point) for point in hull],
                "projection_bounds_uv_m": _bounds_2d(projected),
                "pixel_hull_xy": [_vector(point, 6) for point in pixel_hull],
                "pixel_bounds_xy": _pixel_bounds(pixel_hull),
                "depth_interval_m": [_clean(min(depths)), _clean(max(depths))],
            })
        edge_records: list[dict[str, Any]] = []
        for joint_name, joint in sorted(joints.items()):
            start_xyz, end_xyz = link_origins[joint["parent_link"]], link_origins[joint["child_link"]]
            start_uv, end_uv = project(start_xyz), project(end_xyz)
            start_pixel, end_pixel = screen(start_uv), screen(end_uv)
            edge_records.append({
                "entity_id": f"joint/{joint_name}",
                "joint_name": joint_name,
                "joint_type": joint["type"],
                "parent_entity": f"link/{joint['parent_link']}",
                "child_entity": f"link/{joint['child_link']}",
                "start_root_xyz_m": _vector(start_xyz),
                "end_root_xyz_m": _vector(end_xyz),
                "start_uv_m": _vector(start_uv),
                "end_uv_m": _vector(end_uv),
                "start_pixel_xy": _vector(start_pixel, 6),
                "end_pixel_xy": _vector(end_pixel, 6),
                "length_3d_m": _clean(_distance(start_xyz, end_xyz)),
                "projected_length_m": _clean(math.dist(start_uv, end_uv)),
                "pixel_length": _clean(math.dist(start_pixel, end_pixel), 6),
            })
        frame_records: list[dict[str, Any]] = []
        highlights = set(highlight_frames)
        for frame_name, origin in link_origins.items():
            uv = project(origin)
            frame_records.append({
                "entity_id": f"frame/{frame_name}",
                "frame_name": frame_name,
                "origin_root_xyz_m": _vector(origin),
                "projected_uv_m": _vector(uv),
                "pixel_xy": _vector(screen(uv), 6),
                "highlighted": frame_name in highlights,
            })
        axis_length = span * 0.08
        axis_records: list[dict[str, Any]] = []
        for frame_name in sorted(highlights):
            if frame_name not in frames:
                continue
            matrix = frames[frame_name]["world_from_frame"]["matrix_4x4_rowmajor"]
            origin = [matrix[index][3] for index in range(3)]
            origin_uv = project(origin)
            for axis_name, axis in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
                direction = _rotate(matrix, axis)
                endpoint = [origin[index] + axis_length * direction[index] for index in range(3)]
                endpoint_uv = project(endpoint)
                axis_records.append({
                    "entity_id": f"frame_axis/{frame_name}/{axis_name}",
                    "frame_entity": f"frame/{frame_name}",
                    "axis": axis_name,
                    "axis_length_m": _clean(axis_length),
                    "origin_root_xyz_m": _vector(origin),
                    "endpoint_root_xyz_m": _vector(endpoint),
                    "origin_uv_m": _vector(origin_uv),
                    "endpoint_uv_m": _vector(endpoint_uv),
                    "origin_pixel_xy": _vector(screen(origin_uv), 6),
                    "endpoint_pixel_xy": _vector(screen(endpoint_uv), 6),
                })
        view = {
            "view_id": spec["id"],
            "title": spec["title"],
            "axis_label": spec["axis_label"],
            "projection": {
                "type": "orthographic",
                "root_xyz_to_uv_matrix_2x3": [_vector(u_axis), _vector(v_axis)],
                "depth_axis_in_root_xyz": _vector(depth_axis),
                "depth_meaning": "signed coordinate only; visibility and near/far clipping are not inferred",
            },
            "screen": {
                "width_px": width,
                "height_px": height,
                "plot_rect_xywh_px": [plot_left, plot_top, plot_width, plot_height],
                "center_uv_m": _vector(center),
                "scale_px_per_m": _clean(scale),
                "mapping": "pixel_x=width/2+(u-center_u)*scale; pixel_y=plot_top+plot_height/2-(v-center_v)*scale",
            },
            "scene_projection_bounds_uv_m": scene_bounds,
            "geometry": geometry_records,
            "kinematic_edges": edge_records,
            "link_frames": frame_records,
            "highlight_frame_axes": axis_records,
        }
        svg_path = views_directory / f"{spec['id']}.svg"
        svg_path.write_text(
            _svg_for_view(view, geometry_records, edge_records, frame_records, axis_records, render_input_sha256, pose_name),
            encoding="utf-8",
        )
        view["artifact"] = {
            "path": f"views/{spec['id']}.svg",
            "sha256": _sha256_path(svg_path),
            "format": "svg",
        }
        atlas_views[spec["id"]] = view
    manifest: dict[str, Any] = {
        "schema_version": ATLAS_SCHEMA,
        "render_id": f"render-{render_input_sha256[:20]}",
        "render_input_sha256": render_input_sha256,
        "source_binding": source_binding,
        "pose_binding": pose_binding,
        "coordinate_contract": {
            "root_frame": source_binding["root_frame"],
            "world_point_order": "xyz",
            "projected_point_order": "uv",
            "pixel_point_order": "xy",
            "length_unit": "m",
            "pixel_origin": "top_left",
            "pixel_y_direction": "down",
        },
        "coverage": {
            "declared_geometry_count": len(analysis),
            "rendered_geometry_count": len(renderable_geometry),
            "unrendered_geometry_frames": unrendered,
            "complete_for_declared_geometry": not unrendered,
            "link_frame_count": len(link_frame_names),
            "kinematic_edge_count": len(joints),
            "view_count": len(atlas_views),
        },
        "views": atlas_views,
        "epistemic_scope": {
            "purpose": "machine-verifiable visual grounding for the same canonical robot structure and pose",
            "derived_from_same_geometry_oracle": True,
            "independent_spatial_oracle": False,
            "photorealistic": False,
            "visibility_or_occlusion_computed": False,
            "perspective_or_lens_model": False,
            "physical_world_truth": "not_established",
        },
    }
    if combined_overview_path is not None:
        manifest["combined_overview"] = {
            "path": os.path.relpath(combined_overview_path, output_directory),
            "sha256": _sha256_path(combined_overview_path),
            "format": "svg",
            "role": "backward-compatible four-panel overview outside this atlas directory",
        }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_semantic_render_atlas(
    manifest_path: Path,
    geometry_points: dict[str, list[Vector]],
    analysis: dict[str, dict[str, Any]],
    frames: dict[str, dict[str, Any]],
    joints: dict[str, dict[str, Any]],
    highlight_frames: list[str],
    source_binding: dict[str, Any],
    pose_name: str,
    joint_positions: dict[str, float],
) -> dict[str, Any]:
    """Regenerate semantic projections and verify manifest/artifact integrity."""
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RenderError(f"cannot read semantic render atlas {manifest_path}: {error}") from error
    if not isinstance(actual, dict):
        raise RenderError("semantic render atlas manifest must be a JSON object")
    issues: list[dict[str, Any]] = []

    def issue(check: str, expected: Any, observed: Any) -> None:
        issues.append({"check": check, "expected": expected, "observed": observed})

    with tempfile.TemporaryDirectory(prefix="robot-spatial-render-verify-") as temp_directory:
        expected = write_semantic_render_atlas(
            Path(temp_directory),
            geometry_points,
            analysis,
            frames,
            joints,
            highlight_frames,
            source_binding,
            pose_name,
            joint_positions,
        )
    for key in (
        "schema_version",
        "render_id",
        "render_input_sha256",
        "source_binding",
        "pose_binding",
        "coordinate_contract",
        "coverage",
        "views",
        "epistemic_scope",
    ):
        if actual.get(key) != expected[key]:
            issue(f"manifest.{key}", expected[key], actual.get(key))
    manifest_directory = manifest_path.parent.resolve()
    actual_views = actual.get("views", {})
    if isinstance(actual_views, dict):
        for view_id, view in actual_views.items():
            if not isinstance(view, dict) or not isinstance(view.get("artifact"), dict):
                issue(f"views.{view_id}.artifact", "object", None if not isinstance(view, dict) else view.get("artifact"))
                continue
            relative_path = view["artifact"].get("path")
            if not isinstance(relative_path, str):
                issue(f"views.{view_id}.artifact.path", "relative string", relative_path)
                continue
            artifact_path = (manifest_directory / relative_path).resolve()
            try:
                artifact_path.relative_to(manifest_directory)
            except ValueError:
                issue(f"views.{view_id}.artifact.path", "path inside atlas directory", relative_path)
                continue
            if not artifact_path.is_file():
                issue(f"views.{view_id}.artifact.exists", True, False)
                continue
            observed_sha256 = _sha256_path(artifact_path)
            if observed_sha256 != view["artifact"].get("sha256"):
                issue(f"views.{view_id}.artifact.sha256", view["artifact"].get("sha256"), observed_sha256)
            svg = artifact_path.read_text(encoding="utf-8")
            entity_ids = {
                record["entity_id"]
                for collection in ("geometry", "kinematic_edges", "link_frames", "highlight_frame_axes")
                for record in view.get(collection, [])
                if isinstance(record, dict) and isinstance(record.get("entity_id"), str)
            }
            missing_ids = sorted(entity_id for entity_id in entity_ids if f'data-entity-id="{escape(entity_id, quote=True)}"' not in svg)
            if missing_ids:
                issue(f"views.{view_id}.svg_entity_ids", [], missing_ids)
    combined = actual.get("combined_overview")
    if isinstance(combined, dict):
        relative_path = combined.get("path")
        if not isinstance(relative_path, str):
            issue("combined_overview.path", "relative string", relative_path)
        else:
            combined_path = (manifest_directory / relative_path).resolve()
            try:
                combined_path.relative_to(manifest_directory.parent)
            except ValueError:
                issue("combined_overview.path", "path inside atlas parent context", relative_path)
                combined_path = Path("/__invalid_render_atlas_path__")
            if not combined_path.is_file():
                issue("combined_overview.exists", True, False)
            else:
                observed_sha256 = _sha256_path(combined_path)
                if observed_sha256 != combined.get("sha256"):
                    issue("combined_overview.sha256", combined.get("sha256"), observed_sha256)
    return {
        "schema_version": "robot-spatial-render-atlas-verification.v1",
        "status": "passed" if not issues else "failed",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_path(manifest_path),
        "render_id": actual.get("render_id"),
        "render_input_sha256": actual.get("render_input_sha256"),
        "verified_view_count": len(actual_views) if isinstance(actual_views, dict) else 0,
        "checks": [
            "model semantic digest and pose binding",
            "render-input digest",
            "projection matrices and screen mappings",
            "geometry hulls, depth intervals, frame origins, kinematic edges, and frame axes",
            "standalone SVG digests and typed entity IDs",
            "combined overview digest when present",
        ],
        "issue_count": len(issues),
        "issues": issues,
        "epistemic_scope": "proves deterministic regeneration from the same canonical geometry points; it does not make the rendering an independent geometry oracle or establish physical truth",
    }
