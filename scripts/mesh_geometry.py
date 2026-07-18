#!/usr/bin/env python3
"""Dependency-free declared-geometry and STL/OBJ measurement helpers."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse


EPSILON = 1e-12
Vector = list[float]
Matrix = list[list[float]]


class GeometryError(ValueError):
    """A mesh, URI, or geometry measurement error."""


@dataclass
class MeshData:
    vertices: list[Vector]
    faces: list[tuple[int, int, int]]
    source_format: str


def clean_number(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(value, 12)


def clean_vector(vector: Iterable[float]) -> Vector:
    return [clean_number(value) for value in vector]


def transform_point(transform: Matrix, point: Vector) -> Vector:
    return [sum(transform[i][j] * point[j] for j in range(3)) + transform[i][3] for i in range(3)]


def rotate_vector(transform: Matrix, vector: Vector) -> Vector:
    return [sum(transform[i][j] * vector[j] for j in range(3)) for i in range(3)]


def bounds(points: list[Vector], method: str) -> dict[str, Any]:
    if not points:
        raise GeometryError("cannot compute bounds from no points")
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    return bounds_from_min_max(minimum, maximum, method)


def bounds_from_min_max(minimum: Vector, maximum: Vector, method: str) -> dict[str, Any]:
    return {
        "min_xyz_m": clean_vector(minimum),
        "max_xyz_m": clean_vector(maximum),
        "extents_xyz_m": clean_vector(maximum[axis] - minimum[axis] for axis in range(3)),
        "center_xyz_m": clean_vector((maximum[axis] + minimum[axis]) / 2.0 for axis in range(3)),
        "method": method,
    }


def aabb_corners(aabb: dict[str, Any]) -> list[Vector]:
    minimum, maximum = aabb["min_xyz_m"], aabb["max_xyz_m"]
    return [[x, y, z] for x in (minimum[0], maximum[0]) for y in (minimum[1], maximum[1]) for z in (minimum[2], maximum[2])]


def read_package_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeometryError(f"cannot read package map {path}: {error}") from error
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise GeometryError("package map must be a JSON object of package names to directory paths")
    result: dict[str, Path] = {}
    for package, raw_directory in data.items():
        directory = Path(raw_directory).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        result[package] = directory.resolve()
    return result


def resolve_mesh_uri(uri: str, urdf_path: Path, package_map: dict[str, Path]) -> Path:
    if uri.startswith("package://"):
        remainder = uri[len("package://"):]
        package, separator, relative = remainder.partition("/")
        if not separator or not package or not relative:
            raise GeometryError(f"invalid package URI: {uri!r}")
        if package not in package_map:
            raise GeometryError(f"mesh {uri!r} requires package {package!r}; provide it in --package-map")
        path = package_map[package] / relative
    elif uri.startswith("file://"):
        parsed = urlparse(uri)
        if parsed.netloc not in ("", "localhost"):
            raise GeometryError(f"remote file URI is not supported: {uri!r}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri).expanduser()
        if not path.is_absolute():
            path = urdf_path.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise GeometryError(f"mesh source {uri!r} does not resolve to an existing file")
    return resolved


def _vertex_key(vertex: Iterable[float]) -> tuple[float, float, float]:
    return tuple(0.0 if abs(value) < EPSILON else float(value) for value in vertex)  # type: ignore[return-value]


def _deduplicated_mesh(triangles: list[tuple[Vector, Vector, Vector]], source_format: str) -> MeshData:
    vertices: list[Vector] = []
    faces: list[tuple[int, int, int]] = []
    indices: dict[tuple[float, float, float], int] = {}
    for triangle in triangles:
        face: list[int] = []
        for vertex in triangle:
            if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
                raise GeometryError(f"{source_format} mesh contains a non-finite vertex")
            key = _vertex_key(vertex)
            if key not in indices:
                indices[key] = len(vertices)
                vertices.append(list(key))
            face.append(indices[key])
        if len(set(face)) == 3 and _norm(_cross(_subtract(vertices[face[1]], vertices[face[0]]), _subtract(vertices[face[2]], vertices[face[0]]))) > EPSILON:
            faces.append((face[0], face[1], face[2]))
    if not vertices or not faces:
        raise GeometryError(f"{source_format} mesh contains no non-degenerate triangles")
    return MeshData(vertices, faces, source_format)


def load_stl(path: Path) -> MeshData:
    raw = path.read_bytes()
    triangles: list[tuple[Vector, Vector, Vector]] = []
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        expected_size = 84 + triangle_count * 50
        trailing = raw[expected_size:] if expected_size <= len(raw) else b"nonempty"
        if expected_size <= len(raw) and (expected_size == len(raw) or not trailing.strip(b"\x00\t\r\n ")):
            for index in range(triangle_count):
                offset = 84 + index * 50 + 12
                values = struct.unpack_from("<9f", raw, offset)
                triangles.append((list(values[0:3]), list(values[3:6]), list(values[6:9])))
            return _deduplicated_mesh(triangles, "stl_binary")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GeometryError(f"STL is neither valid binary nor UTF-8 ASCII: {path}") from error
    pending: list[Vector] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.strip().split()
        if fields and fields[0].lower() == "vertex":
            if len(fields) != 4:
                raise GeometryError(f"invalid ASCII STL vertex at {path}:{line_number}")
            try:
                vertex = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise GeometryError(f"non-numeric ASCII STL vertex at {path}:{line_number}") from error
            if not all(math.isfinite(value) for value in vertex):
                raise GeometryError(f"non-finite ASCII STL vertex at {path}:{line_number}")
            pending.append(vertex)
            if len(pending) == 3:
                triangles.append((pending[0], pending[1], pending[2]))
                pending = []
    if pending:
        raise GeometryError(f"ASCII STL has an incomplete triangle: {path}")
    return _deduplicated_mesh(triangles, "stl_ascii")


def load_obj(path: Path) -> MeshData:
    vertices: list[Vector] = []
    faces: list[tuple[int, int, int]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise GeometryError(f"OBJ is not UTF-8 text: {path}") from error
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise GeometryError(f"OBJ vertex needs at least xyz at {path}:{line_number}")
            try:
                vertex = [float(value) for value in fields[1:4]]
                weight = float(fields[4]) if len(fields) > 4 else 1.0
            except ValueError as error:
                raise GeometryError(f"non-numeric OBJ vertex at {path}:{line_number}") from error
            if abs(weight) <= EPSILON:
                raise GeometryError(f"OBJ homogeneous vertex has zero weight at {path}:{line_number}")
            vertex = [value / weight for value in vertex]
            if not all(math.isfinite(value) for value in vertex):
                raise GeometryError(f"non-finite OBJ vertex at {path}:{line_number}")
            vertices.append(vertex)
        elif fields[0] == "f":
            if len(fields) < 4:
                raise GeometryError(f"OBJ face needs at least three vertices at {path}:{line_number}")
            polygon: list[int] = []
            for token in fields[1:]:
                vertex_token = token.split("/", 1)[0]
                if not vertex_token:
                    raise GeometryError(f"OBJ face has no vertex index at {path}:{line_number}")
                try:
                    raw_index = int(vertex_token)
                except ValueError as error:
                    raise GeometryError(f"invalid OBJ face index at {path}:{line_number}") from error
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if index < 0 or index >= len(vertices):
                    raise GeometryError(f"OBJ face index is out of range at {path}:{line_number}")
                polygon.append(index)
            for index in range(1, len(polygon) - 1):
                triangle = (polygon[0], polygon[index], polygon[index + 1])
                if len(set(triangle)) == 3:
                    faces.append(triangle)
    if not vertices or not faces:
        raise GeometryError(f"OBJ mesh contains no non-degenerate faces: {path}")
    remap: dict[int, int] = {}
    unique_vertices: list[Vector] = []
    indices: dict[tuple[float, float, float], int] = {}
    for old_index, vertex in enumerate(vertices):
        key = _vertex_key(vertex)
        if key not in indices:
            indices[key] = len(unique_vertices)
            unique_vertices.append(list(key))
        remap[old_index] = indices[key]
    unique_faces: list[tuple[int, int, int]] = []
    for face in faces:
        remapped = (remap[face[0]], remap[face[1]], remap[face[2]])
        if len(set(remapped)) == 3 and _norm(_cross(_subtract(unique_vertices[remapped[1]], unique_vertices[remapped[0]]), _subtract(unique_vertices[remapped[2]], unique_vertices[remapped[0]]))) > EPSILON:
            unique_faces.append(remapped)
    if not unique_faces:
        raise GeometryError(f"OBJ mesh contains no non-degenerate faces after vertex normalization: {path}")
    return MeshData(unique_vertices, unique_faces, "obj")


def load_mesh(path: Path) -> MeshData:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return load_stl(path)
    if suffix == ".obj":
        return load_obj(path)
    raise GeometryError(f"unsupported mesh format {suffix!r} for {path}; this portable engine supports STL and OBJ")


def _cross(a: Vector, b: Vector) -> Vector:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _subtract(a: Vector, b: Vector) -> Vector:
    return [a[index] - b[index] for index in range(3)]


def _dot(a: Vector, b: Vector) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _norm(vector: Vector) -> float:
    return math.sqrt(_dot(vector, vector))


def _surface_area(vertices: list[Vector], faces: list[tuple[int, int, int]]) -> float:
    return sum(_norm(_cross(_subtract(vertices[b], vertices[a]), _subtract(vertices[c], vertices[a]))) / 2.0 for a, b, c in faces)


def _edge_topology(faces: list[tuple[int, int, int]]) -> tuple[bool, bool]:
    edges: Counter[tuple[int, int]] = Counter()
    directed: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for start, end in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((start, end)))] += 1
            directed[(start, end)] += 1
    watertight = bool(edges) and all(count == 2 for count in edges.values())
    winding_consistent = watertight and all(directed[(edge[0], edge[1])] == 1 and directed[(edge[1], edge[0])] == 1 for edge in edges)
    return watertight, winding_consistent


def _volume(vertices: list[Vector], faces: list[tuple[int, int, int]], watertight: bool, winding_consistent: bool) -> float | None:
    if not watertight or not winding_consistent:
        return None
    signed = sum(_dot(vertices[a], _cross(vertices[b], vertices[c])) / 6.0 for a, b, c in faces)
    return abs(signed)


def _principal_axes(points: list[Vector]) -> dict[str, Any]:
    count = len(points)
    center = [sum(point[axis] for point in points) / count for axis in range(3)]
    covariance = [[sum((point[row] - center[row]) * (point[column] - center[column]) for point in points) / count for column in range(3)] for row in range(3)]
    matrix = [row[:] for row in covariance]
    eigenvectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(32):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(matrix[pair[0]][pair[1]]))
        if abs(matrix[p][q]) < 1e-15:
            break
        angle = 0.5 * math.atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        for index in range(3):
            if index not in (p, q):
                old_p, old_q = matrix[index][p], matrix[index][q]
                matrix[index][p] = matrix[p][index] = cosine * old_p - sine * old_q
                matrix[index][q] = matrix[q][index] = sine * old_p + cosine * old_q
        old_pp, old_qq, old_pq = matrix[p][p], matrix[q][q], matrix[p][q]
        matrix[p][p] = cosine * cosine * old_pp - 2.0 * sine * cosine * old_pq + sine * sine * old_qq
        matrix[q][q] = sine * sine * old_pp + 2.0 * sine * cosine * old_pq + cosine * cosine * old_qq
        matrix[p][q] = matrix[q][p] = 0.0
        for row in range(3):
            old_p, old_q = eigenvectors[row][p], eigenvectors[row][q]
            eigenvectors[row][p] = cosine * old_p - sine * old_q
            eigenvectors[row][q] = sine * old_p + cosine * old_q
    pairs: list[tuple[float, Vector]] = []
    for column in range(3):
        vector = [eigenvectors[row][column] for row in range(3)]
        dominant = max(range(3), key=lambda index: abs(vector[index]))
        if vector[dominant] < 0.0:
            vector = [-value for value in vector]
        pairs.append((max(0.0, matrix[column][column]), vector))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return {
        "center_xyz_m": clean_vector(center),
        "variances_m2": clean_vector(pair[0] for pair in pairs),
        "axes_in_geometry_frame": [clean_vector(pair[1]) for pair in pairs],
        "method": "vertex_covariance",
    }


def _landmarks(points: list[Vector]) -> dict[str, Vector]:
    result: dict[str, Vector] = {}
    for axis, label in enumerate(("x", "y", "z")):
        result[f"min_{label}"] = clean_vector(min(points, key=lambda point: point[axis]))
        result[f"max_{label}"] = clean_vector(max(points, key=lambda point: point[axis]))
    return result


def _shape_class(extents: Vector) -> dict[str, Any]:
    largest, middle, smallest = sorted((abs(value) for value in extents), reverse=True)
    if largest <= EPSILON:
        label = "point_or_empty"
    elif smallest <= largest * 1e-6:
        label = "planar_or_degenerate"
    elif largest >= 3.0 * max(middle, EPSILON):
        label = "elongated"
    elif middle >= 3.0 * max(smallest, EPSILON):
        label = "plate_like"
    else:
        label = "compact"
    return {
        "heuristic_label": label,
        "sorted_extent_ratios": clean_vector([1.0, middle / largest if largest else 0.0, smallest / largest if largest else 0.0]),
        "heuristic_only": True,
    }


def _mesh_analysis(mesh: MeshData, scale: Vector, world_from_geometry: Matrix, path: Path) -> tuple[dict[str, Any], list[Vector]]:
    if len(scale) != 3 or not all(math.isfinite(value) and abs(value) > EPSILON for value in scale):
        raise GeometryError(f"mesh scale must contain three finite non-zero values: {scale}")
    vertices = [[vertex[axis] * scale[axis] for axis in range(3)] for vertex in mesh.vertices]
    local_bounds = bounds(vertices, "all_mesh_vertices_exact")
    world_vertices = [transform_point(world_from_geometry, vertex) for vertex in vertices]
    world_bounds = bounds(world_vertices, "all_transformed_mesh_vertices_exact")
    watertight, winding_consistent = _edge_topology(mesh.faces)
    volume = _volume(vertices, mesh.faces, watertight, winding_consistent)
    principal = _principal_axes(vertices)
    principal["axes_in_root_frame_at_pose"] = [clean_vector(rotate_vector(world_from_geometry, axis)) for axis in principal["axes_in_geometry_frame"]]
    return ({
        "status": "measured",
        "geometry_type": "mesh",
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "format": mesh.source_format,
            "declared_scale_xyz": clean_vector(scale),
            "source_units": "unspecified; URDF scale applied",
        },
        "topology": {
            "unique_vertex_count": len(vertices),
            "triangle_count": len(mesh.faces),
            "watertight": watertight,
            "winding_consistent": winding_consistent,
        },
        "bounds_in_geometry_frame": local_bounds,
        "bounds_in_root_frame_at_pose": world_bounds,
        "surface_area_m2": clean_number(_surface_area(vertices, mesh.faces)),
        "volume_m3": clean_number(volume) if volume is not None else None,
        "volume_trust": "exact_for_closed_consistently_oriented_triangle_surface" if volume is not None else "unavailable_without_watertight_consistent_winding",
        "vertex_mean_xyz_m": clean_vector(sum(vertex[axis] for vertex in vertices) / len(vertices) for axis in range(3)),
        "principal_axes": principal,
        "landmarks_in_geometry_frame": _landmarks(vertices),
        "geometry_axis_landmarks_in_root_frame_at_pose": {name: clean_vector(transform_point(world_from_geometry, point)) for name, point in _landmarks(vertices).items()},
        "shape": _shape_class(local_bounds["extents_xyz_m"]),
    }, world_vertices)


def _box_points(size: Vector) -> list[Vector]:
    half = [value / 2.0 for value in size]
    return [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]


def _cylinder_points(radius: float, length: float, samples: int = 32) -> list[Vector]:
    half = length / 2.0
    return [[radius * math.cos(2.0 * math.pi * index / samples), radius * math.sin(2.0 * math.pi * index / samples), z] for z in (-half, half) for index in range(samples)]


def _sphere_points(radius: float, samples: int = 24) -> list[Vector]:
    points: list[Vector] = []
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        cosine, sine = math.cos(angle), math.sin(angle)
        points.extend(([radius * cosine, radius * sine, 0.0], [radius * cosine, 0.0, radius * sine], [0.0, radius * cosine, radius * sine]))
    return points


def _primitive_analysis(geometry: dict[str, Any], world_from_geometry: Matrix) -> tuple[dict[str, Any], list[Vector]]:
    geometry_type = geometry["type"]
    if geometry_type == "box":
        size = geometry["size_xyz_m"]
        if any(value <= 0.0 for value in size):
            raise GeometryError(f"box dimensions must be positive: {size}")
        points = _box_points(size)
        volume, surface_area = size[0] * size[1] * size[2], 2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2])
        local_bounds = bounds(points, "analytic_box_exact")
        world_points = [transform_point(world_from_geometry, point) for point in points]
        world_bounds = bounds(world_points, "transformed_box_corners_exact")
    elif geometry_type == "cylinder":
        radius, length = geometry["radius_m"], geometry["length_m"]
        if radius <= 0.0 or length <= 0.0:
            raise GeometryError(f"cylinder radius and length must be positive: {radius}, {length}")
        points = _cylinder_points(radius, length)
        volume, surface_area = math.pi * radius * radius * length, 2.0 * math.pi * radius * (radius + length)
        local_bounds = bounds_from_min_max([-radius, -radius, -length / 2.0], [radius, radius, length / 2.0], "analytic_cylinder_exact")
        world_points = [transform_point(world_from_geometry, point) for point in points]
        axis = rotate_vector(world_from_geometry, [0.0, 0.0, 1.0])
        center = transform_point(world_from_geometry, [0.0, 0.0, 0.0])
        extents = [abs(axis[index]) * length / 2.0 + radius * math.sqrt(max(0.0, 1.0 - axis[index] * axis[index])) for index in range(3)]
        world_bounds = bounds_from_min_max([center[index] - extents[index] for index in range(3)], [center[index] + extents[index] for index in range(3)], "analytic_oriented_cylinder_exact")
    elif geometry_type == "sphere":
        radius = geometry["radius_m"]
        if radius <= 0.0:
            raise GeometryError(f"sphere radius must be positive: {radius}")
        points = _sphere_points(radius)
        volume, surface_area = 4.0 * math.pi * radius ** 3 / 3.0, 4.0 * math.pi * radius * radius
        local_bounds = bounds_from_min_max([-radius] * 3, [radius] * 3, "analytic_sphere_exact")
        world_points = [transform_point(world_from_geometry, point) for point in points]
        center = transform_point(world_from_geometry, [0.0, 0.0, 0.0])
        world_bounds = bounds_from_min_max([value - radius for value in center], [value + radius for value in center], "analytic_sphere_exact")
    else:
        raise GeometryError(f"unsupported primitive geometry type: {geometry_type!r}")
    principal = _principal_axes(points)
    principal["axes_in_root_frame_at_pose"] = [clean_vector(rotate_vector(world_from_geometry, axis)) for axis in principal["axes_in_geometry_frame"]]
    return ({
        "status": "measured",
        "geometry_type": geometry_type,
        "source": {"type": "urdf_primitive", "source_units": "m"},
        "topology": {"watertight": True},
        "bounds_in_geometry_frame": local_bounds,
        "bounds_in_root_frame_at_pose": world_bounds,
        "surface_area_m2": clean_number(surface_area),
        "volume_m3": clean_number(volume),
        "principal_axes": principal,
        "landmarks_in_geometry_frame": _landmarks(points),
        "geometry_axis_landmarks_in_root_frame_at_pose": {name: clean_vector(transform_point(world_from_geometry, point)) for name, point in _landmarks(points).items()},
        "shape": _shape_class(local_bounds["extents_xyz_m"]),
    }, world_points)


def analyze_declared_geometry(
    geometry: dict[str, Any],
    world_from_geometry: Matrix,
    urdf_path: Path,
    package_map: dict[str, Path],
    inspect_meshes: bool,
) -> tuple[dict[str, Any], list[Vector]]:
    if geometry["type"] != "mesh":
        return _primitive_analysis(geometry, world_from_geometry)
    if not inspect_meshes:
        return ({
            "status": "not_inspected",
            "geometry_type": "mesh",
            "uri": geometry["uri"],
            "reason": "rerun with --inspect-meshes and --package-map when package:// URIs are present",
        }, [])
    path = resolve_mesh_uri(geometry["uri"], urdf_path, package_map)
    return _mesh_analysis(load_mesh(path), geometry["scale_xyz"], world_from_geometry, path)


def broadphase_overlaps(analysis: dict[str, dict[str, Any]], adjacent_links: set[frozenset[str]]) -> dict[str, Any]:
    collisions = [(name, record) for name, record in analysis.items() if record["kind"] == "collision"]
    skipped = sorted(name for name, record in collisions if record["status"] != "measured")
    measured = [(name, record) for name, record in collisions if record["status"] == "measured"]
    pairs: list[dict[str, Any]] = []
    for index, (left_name, left) in enumerate(measured):
        for right_name, right in measured[index + 1:]:
            if left["link"] == right["link"]:
                continue
            left_bounds, right_bounds = left["bounds_in_root_frame_at_pose"], right["bounds_in_root_frame_at_pose"]
            minimum = [max(left_bounds["min_xyz_m"][axis], right_bounds["min_xyz_m"][axis]) for axis in range(3)]
            maximum = [min(left_bounds["max_xyz_m"][axis], right_bounds["max_xyz_m"][axis]) for axis in range(3)]
            if all(minimum[axis] <= maximum[axis] + EPSILON for axis in range(3)):
                extents = [max(0.0, maximum[axis] - minimum[axis]) for axis in range(3)]
                pairs.append({
                    "geometry_a": left_name,
                    "link_a": left["link"],
                    "geometry_b": right_name,
                    "link_b": right["link"],
                    "links_are_adjacent": frozenset((left["link"], right["link"])) in adjacent_links,
                    "intersection_extents_xyz_m": clean_vector(extents),
                    "intersection_volume_m3": clean_number(extents[0] * extents[1] * extents[2]),
                })
    return {
        "method": "root_frame_axis_aligned_bounding_box_overlap",
        "meaning": "conservative broad-phase candidates; overlap does not prove triangle-level collision",
        "complete_for_declared_collision_geometry": not skipped,
        "skipped_unmeasured_geometry": skipped,
        "same_link_pairs_excluded": True,
        "overlap_pairs": pairs,
    }


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

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


def render_scene_svg(
    path: Path,
    geometry_points: dict[str, list[Vector]],
    analysis: dict[str, dict[str, Any]],
    frames: dict[str, dict[str, Any]],
    joints: dict[str, dict[str, Any]],
    highlight_frames: list[str],
) -> dict[str, Any]:
    projections: list[tuple[str, Callable[[Vector], tuple[float, float]], str]] = [
        ("Front (X–Z)", lambda point: (point[0], point[2]), "+X right, +Z up"),
        ("Side (Y–Z)", lambda point: (point[1], point[2]), "+Y right, +Z up"),
        ("Top (X–Y)", lambda point: (point[0], point[1]), "+X right, +Y up"),
        ("Isometric", lambda point: ((point[0] - point[1]) / math.sqrt(2.0), math.sqrt(2.0 / 3.0) * point[2] - (point[0] + point[1]) / math.sqrt(6.0)), "orthographic isometric"),
    ]
    width, height, panel_width, panel_height = 1000, 850, 480, 360
    panel_origins = [(10, 20), (510, 20), (10, 400), (510, 400)]
    all_points = [point for points in geometry_points.values() for point in points]
    all_points.extend(frame["world_from_frame"]["translation_xyz_m"] for name, frame in frames.items() if frame["type"] == "link")
    if not all_points:
        all_points = [[0.0, 0.0, 0.0]]
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfcfe"/>']
    colors = {"visual": ("#2563eb", "#93c5fd"), "collision": ("#dc2626", "#fecaca")}
    for (title, project, axes_label), (origin_x, origin_y) in zip(projections, panel_origins):
        projected_all = [project(point) for point in all_points]
        min_x, max_x = min(point[0] for point in projected_all), max(point[0] for point in projected_all)
        min_y, max_y = min(point[1] for point in projected_all), max(point[1] for point in projected_all)
        span = max(max_x - min_x, max_y - min_y, 1e-3)
        scale = min((panel_width - 60) / span, (panel_height - 70) / span)
        center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

        def screen(point: tuple[float, float]) -> tuple[float, float]:
            return (origin_x + panel_width / 2.0 + (point[0] - center_x) * scale, origin_y + panel_height / 2.0 - (point[1] - center_y) * scale)

        elements.extend([
            f'<rect x="{origin_x}" y="{origin_y}" width="{panel_width}" height="{panel_height}" rx="8" fill="white" stroke="#cbd5e1"/>',
            f'<text x="{origin_x + 14}" y="{origin_y + 24}" font-family="sans-serif" font-size="16" font-weight="600" fill="#0f172a">{escape(title)}</text>',
            f'<text x="{origin_x + 14}" y="{origin_y + 43}" font-family="sans-serif" font-size="11" fill="#64748b">{escape(axes_label)}</text>',
        ])
        for joint_name, joint in joints.items():
            parent = frames[joint["parent_link"]]["world_from_frame"]["translation_xyz_m"]
            child = frames[joint["child_link"]]["world_from_frame"]["translation_xyz_m"]
            start, end = screen(project(parent)), screen(project(child))
            elements.append(f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="#475569" stroke-width="2"/>')
        for frame_name, points in geometry_points.items():
            if not points:
                continue
            hull = _convex_hull([project(point) for point in points])
            screen_hull = [screen(point) for point in hull]
            kind = analysis[frame_name]["kind"]
            stroke, fill = colors[kind]
            dash = ' stroke-dasharray="5 3"' if kind == "collision" else ""
            if len(screen_hull) >= 3:
                point_string = " ".join(f"{point[0]:.2f},{point[1]:.2f}" for point in screen_hull)
                elements.append(f'<polygon points="{point_string}" fill="{fill}" fill-opacity="0.35" stroke="{stroke}" stroke-width="1.5"{dash}/>' )
            elif screen_hull:
                point = screen_hull[0]
                elements.append(f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="3" fill="{stroke}"/>')
        axis_length = span * 0.08
        axis_colors = (([1.0, 0.0, 0.0], "#ef4444"), ([0.0, 1.0, 0.0], "#22c55e"), ([0.0, 0.0, 1.0], "#3b82f6"))
        for highlight_index, frame_name in enumerate(highlight_frames):
            if frame_name not in frames:
                continue
            matrix = frames[frame_name]["world_from_frame"]["matrix_4x4_rowmajor"]
            origin = [matrix[index][3] for index in range(3)]
            start = screen(project(origin))
            for axis, color in axis_colors:
                direction = rotate_vector(matrix, axis)
                endpoint = [origin[index] + axis_length * direction[index] for index in range(3)]
                end = screen(project(endpoint))
                elements.append(f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" stroke="{color}" stroke-width="1.5"/>')
            label_y = start[1] - 4.0 - 10.0 * (highlight_index % 3)
            elements.append(f'<text x="{start[0] + 4:.2f}" y="{label_y:.2f}" font-family="sans-serif" font-size="9" fill="#0f172a">{escape(frame_name)}</text>')
    elements.extend([
        '<g font-family="sans-serif" font-size="11" fill="#334155">',
        '<rect x="25" y="805" width="14" height="8" fill="#93c5fd" stroke="#2563eb"/><text x="45" y="813">visual geometry</text>',
        '<rect x="175" y="805" width="14" height="8" fill="#fecaca" stroke="#dc2626"/><text x="195" y="813">collision geometry (dashed)</text>',
        '<text x="430" y="813">frame axes: X red, Y green, Z blue</text>',
        '</g>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "format": "svg",
        "views": [title for title, _, _ in projections],
        "geometry_count": len(geometry_points),
        "highlight_frames": highlight_frames,
        "rendering": "orthographic convex projection of measured geometry points with kinematic edges and frame axes",
    }
