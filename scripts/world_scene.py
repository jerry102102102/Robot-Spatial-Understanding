#!/usr/bin/env python3
"""Validated static world snapshots bound to one URDF robot instance.

The scene layer is deliberately separate from URDF.  URDF describes the robot-local
mechanism; this module describes where that mechanism and external objects are placed
for one explicitly identified snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from mesh_geometry import GeometryError, analyze_declared_geometry, load_mesh, read_package_map
from triangle_geometry import (
    aabb_distance_squared,
    box_surface,
    build_bvh,
    bvh_surface_distance,
    closest_point_on_triangle,
    point_inside_closed_surface,
    transform_triangles,
)


SCENE_SCHEMA = "robot-spatial-world-scene.v1"
BOUND_SCENE_SCHEMA = "robot-spatial-bound-world-scene.v1"
COLLISION_SCHEMA = "robot-spatial-robot-environment-collision.v1"
EPSILON = 1e-12
SOURCE_TYPES = {"declared", "measured", "calibrated", "synthetic", "imported", "unknown"}
GEOMETRY_TYPES = {"box", "cylinder", "sphere", "mesh"}

Matrix = list[list[float]]
Vector = list[float]
Triangle = tuple[Vector, Vector, Vector]


class SceneError(ValueError):
    """An invalid, inconsistent, or ambiguous world-scene snapshot."""


def _identity() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)] for row in range(4)]


def _inverse_rigid(transform: Matrix) -> Matrix:
    result = _identity()
    for row in range(3):
        for column in range(3):
            result[row][column] = transform[column][row]
        result[row][3] = -sum(transform[column][row] * transform[column][3] for column in range(3))
    return result


def _transform_point(transform: Matrix, point: Vector) -> Vector:
    return [sum(transform[row][column] * point[column] for column in range(3)) + transform[row][3] for row in range(3)]


def _rotate_vector(transform: Matrix, vector: Vector) -> Vector:
    return [sum(transform[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _rpy_matrix(rpy: Vector) -> Matrix:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _quaternion_matrix(quaternion: Vector) -> Matrix:
    x, y, z, w = quaternion
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _clean_number(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(value, 12)


def _clean_vector(values: Iterable[float]) -> Vector:
    return [_clean_number(value) for value in values]


def _clean_matrix(matrix: Matrix) -> Matrix:
    return [_clean_vector(row) for row in matrix]


def _quaternion_xyzw(transform: Matrix) -> Vector:
    trace = transform[0][0] + transform[1][1] + transform[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (transform[2][1] - transform[1][2]) / scale,
            (transform[0][2] - transform[2][0]) / scale,
            (transform[1][0] - transform[0][1]) / scale,
            0.25 * scale,
        ]
    elif transform[0][0] > transform[1][1] and transform[0][0] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[0][0] - transform[1][1] - transform[2][2]) * 2.0
        quaternion = [
            0.25 * scale,
            (transform[0][1] + transform[1][0]) / scale,
            (transform[0][2] + transform[2][0]) / scale,
            (transform[2][1] - transform[1][2]) / scale,
        ]
    elif transform[1][1] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[1][1] - transform[0][0] - transform[2][2]) * 2.0
        quaternion = [
            (transform[0][1] + transform[1][0]) / scale,
            0.25 * scale,
            (transform[1][2] + transform[2][1]) / scale,
            (transform[0][2] - transform[2][0]) / scale,
        ]
    else:
        scale = math.sqrt(1.0 + transform[2][2] - transform[0][0] - transform[1][1]) * 2.0
        quaternion = [
            (transform[0][2] + transform[2][0]) / scale,
            (transform[1][2] + transform[2][1]) / scale,
            0.25 * scale,
            (transform[1][0] - transform[0][1]) / scale,
        ]
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    quaternion = [value / magnitude for value in quaternion]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return _clean_vector(quaternion)


def pose_record(transform: Matrix) -> dict[str, Any]:
    return {
        "matrix_4x4_rowmajor": _clean_matrix(transform),
        "translation_xyz_m": _clean_vector(transform[index][3] for index in range(3)),
        "quaternion_xyzw": _quaternion_xyzw(transform),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SceneError(f"{label} must be a finite number")
    return number


def _vector(value: Any, length: int, label: str) -> Vector:
    if not isinstance(value, list) or len(value) != length:
        raise SceneError(f"{label} must be an array of {length} finite numbers")
    return [_finite_number(component, f"{label}[{index}]") for index, component in enumerate(value)]


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SceneError(f"{label} must be a non-empty string")
    if "/" in value:
        raise SceneError(f"{label} must not contain '/'; slash is reserved for typed entity IDs")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SceneError(f"{label} must be null or a non-empty string")
    return value


def _pose(value: Any, label: str) -> tuple[dict[str, Any], Matrix]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SceneError(f"{label} must be an object")
    unknown = sorted(set(value) - {"xyz_m", "rpy_rad", "quaternion_xyzw"})
    if unknown:
        raise SceneError(f"{label} has unsupported fields {unknown}")
    xyz = _vector(value.get("xyz_m", [0.0, 0.0, 0.0]), 3, f"{label}.xyz_m")
    has_rpy = "rpy_rad" in value
    has_quaternion = "quaternion_xyzw" in value
    if has_rpy and has_quaternion:
        raise SceneError(f"{label} must use either rpy_rad or quaternion_xyzw, not both")
    if has_quaternion:
        quaternion = _vector(value["quaternion_xyzw"], 4, f"{label}.quaternion_xyzw")
        magnitude = math.sqrt(sum(component * component for component in quaternion))
        if abs(magnitude - 1.0) > 1e-6:
            raise SceneError(f"{label}.quaternion_xyzw must be unit length within 1e-6; magnitude is {magnitude}")
        quaternion = [component / magnitude for component in quaternion]
        transform = _quaternion_matrix(quaternion)
        canonical = {"xyz_m": _clean_vector(xyz), "quaternion_xyzw": _clean_vector(quaternion)}
    else:
        rpy = _vector(value.get("rpy_rad", [0.0, 0.0, 0.0]), 3, f"{label}.rpy_rad")
        transform = _rpy_matrix(rpy)
        canonical = {"xyz_m": _clean_vector(xyz), "rpy_rad": _clean_vector(rpy)}
    for index in range(3):
        transform[index][3] = xyz[index]
    return canonical, transform


def _source(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {
            "type": "unknown",
            "reference": None,
            "captured_at": None,
            "meaning": "placement or geometry is exact within this file but its agreement with the physical world is not established",
        }
    if not isinstance(value, dict):
        raise SceneError(f"{label} must be an object")
    unknown = sorted(set(value) - {"type", "reference", "captured_at"})
    if unknown:
        raise SceneError(f"{label} has unsupported fields {unknown}")
    source_type = value.get("type")
    if source_type not in SOURCE_TYPES:
        raise SceneError(f"{label}.type must be one of {sorted(SOURCE_TYPES)}")
    return {
        "type": source_type,
        "reference": _optional_string(value.get("reference"), f"{label}.reference"),
        "captured_at": _optional_string(value.get("captured_at"), f"{label}.captured_at"),
        "meaning": "provenance label supplied by the scene author; it is not independently verified by this parser",
    }


def _semantics(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {"category": None, "roles": [], "meaning": None}
    if not isinstance(value, dict):
        raise SceneError(f"{label} must be an object")
    unknown = sorted(set(value) - {"category", "roles", "meaning"})
    if unknown:
        raise SceneError(f"{label} has unsupported fields {unknown}")
    category = _optional_string(value.get("category"), f"{label}.category")
    meaning = _optional_string(value.get("meaning"), f"{label}.meaning")
    roles_value = value.get("roles", [])
    if not isinstance(roles_value, list) or not all(isinstance(role, str) and role.strip() for role in roles_value):
        raise SceneError(f"{label}.roles must be an array of non-empty strings")
    if len(set(roles_value)) != len(roles_value):
        raise SceneError(f"{label}.roles must not contain duplicates")
    return {"category": category, "roles": sorted(roles_value), "meaning": meaning}


def _geometry(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SceneError(f"{label} must be an object")
    geometry_type = value.get("type")
    if geometry_type not in GEOMETRY_TYPES:
        raise SceneError(f"{label}.type must be one of {sorted(GEOMETRY_TYPES)}")
    if geometry_type == "box":
        unknown = sorted(set(value) - {"type", "size_xyz_m"})
        size = _vector(value.get("size_xyz_m"), 3, f"{label}.size_xyz_m")
        if any(component <= 0.0 for component in size):
            raise SceneError(f"{label}.size_xyz_m components must be positive")
        result = {"type": "box", "size_xyz_m": _clean_vector(size)}
    elif geometry_type == "cylinder":
        unknown = sorted(set(value) - {"type", "radius_m", "length_m"})
        radius = _finite_number(value.get("radius_m"), f"{label}.radius_m")
        length = _finite_number(value.get("length_m"), f"{label}.length_m")
        if radius <= 0.0 or length <= 0.0:
            raise SceneError(f"{label} radius and length must be positive")
        result = {"type": "cylinder", "radius_m": radius, "length_m": length}
    elif geometry_type == "sphere":
        unknown = sorted(set(value) - {"type", "radius_m"})
        radius = _finite_number(value.get("radius_m"), f"{label}.radius_m")
        if radius <= 0.0:
            raise SceneError(f"{label}.radius_m must be positive")
        result = {"type": "sphere", "radius_m": radius}
    else:
        unknown = sorted(set(value) - {"type", "uri", "scale_xyz"})
        uri = value.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            raise SceneError(f"{label}.uri must be a non-empty string")
        scale = _vector(value.get("scale_xyz", [1.0, 1.0, 1.0]), 3, f"{label}.scale_xyz")
        if any(abs(component) <= EPSILON for component in scale):
            raise SceneError(f"{label}.scale_xyz components must be non-zero")
        result = {"type": "mesh", "uri": uri, "scale_xyz": _clean_vector(scale)}
    if unknown:
        raise SceneError(f"{label} has unsupported fields {unknown}")
    return result


def _face_components(faces: list[tuple[int, int, int]]) -> list[list[int]]:
    by_vertex: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            by_vertex.setdefault(vertex_index, []).append(face_index)
    remaining = set(range(len(faces)))
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = [seed]
        component: list[int] = []
        for face_index in queue:
            component.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbor in by_vertex[vertex_index]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
        components.append(sorted(component))
    return components


class WorldScene:
    """One validated static snapshot and its typed frame graph."""

    def __init__(self, path: Path, *, expected_robot_name: str | None = None, expected_root_link: str | None = None):
        self.path = path.expanduser().resolve()
        try:
            raw = self.path.read_bytes()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise SceneError(f"cannot read world scene {self.path}: {error}") from error
        if not isinstance(data, dict):
            raise SceneError("world scene must contain a JSON object")
        if data.get("schema_version") != SCENE_SCHEMA:
            raise SceneError(f"world scene must use schema_version {SCENE_SCHEMA}")
        unknown = sorted(set(data) - {"schema_version", "scene_id", "snapshot", "world_frame", "gravity", "frames", "robot", "objects", "source"})
        if unknown:
            raise SceneError(f"world scene has unsupported top-level fields {unknown}")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.scene_id = _identifier(data.get("scene_id"), "scene_id")
        self.world_frame = _identifier(data.get("world_frame"), "world_frame")
        self.source = _source(data.get("source"), "source")
        snapshot = data.get("snapshot")
        if not isinstance(snapshot, dict):
            raise SceneError("snapshot must be an object")
        snapshot_unknown = sorted(set(snapshot) - {"id", "time_semantics", "captured_at", "valid_until"})
        if snapshot_unknown:
            raise SceneError(f"snapshot has unsupported fields {snapshot_unknown}")
        if snapshot.get("time_semantics") != "static_snapshot":
            raise SceneError("snapshot.time_semantics must be 'static_snapshot'")
        self.snapshot = {
            "id": _identifier(snapshot.get("id"), "snapshot.id"),
            "time_semantics": "static_snapshot",
            "captured_at": _optional_string(snapshot.get("captured_at"), "snapshot.captured_at"),
            "valid_until": _optional_string(snapshot.get("valid_until"), "snapshot.valid_until"),
            "meaning": "all transforms and collision results are conditional on this static snapshot; no temporal persistence is inferred",
        }

        raw_frames = data.get("frames", {})
        if not isinstance(raw_frames, dict):
            raise SceneError("frames must be an object keyed by scene-frame name")
        self.frames: dict[str, dict[str, Any]] = {}
        local_transforms: dict[str, Matrix] = {}
        for raw_name, raw_record in raw_frames.items():
            name = _identifier(raw_name, "frames key")
            if name == self.world_frame:
                raise SceneError(f"frames must not redeclare world_frame {name!r}")
            if not isinstance(raw_record, dict):
                raise SceneError(f"frames.{name} must be an object")
            frame_unknown = sorted(set(raw_record) - {"parent", "pose", "semantics", "source"})
            if frame_unknown:
                raise SceneError(f"frames.{name} has unsupported fields {frame_unknown}")
            parent = _identifier(raw_record.get("parent"), f"frames.{name}.parent")
            canonical_pose, local = _pose(raw_record.get("pose"), f"frames.{name}.pose")
            self.frames[name] = {
                "name": name,
                "parent": parent,
                "pose_in_parent": canonical_pose,
                "semantics": _semantics(raw_record.get("semantics"), f"frames.{name}.semantics"),
                "source": _source(raw_record.get("source"), f"frames.{name}.source"),
            }
            local_transforms[name] = local
        known_scene_frames = {self.world_frame, *self.frames}
        for name, record in self.frames.items():
            if record["parent"] not in known_scene_frames:
                raise SceneError(f"frames.{name}.parent references unknown scene frame {record['parent']!r}")

        self.world_from_scene_frame: dict[str, Matrix] = {self.world_frame: _identity()}
        active: set[str] = set()

        def resolve_frame(name: str) -> Matrix:
            if name in self.world_from_scene_frame:
                return self.world_from_scene_frame[name]
            if name in active:
                raise SceneError(f"scene frame cycle detected at {name!r}")
            active.add(name)
            record = self.frames[name]
            result = _matmul(resolve_frame(record["parent"]), local_transforms[name])
            active.remove(name)
            self.world_from_scene_frame[name] = result
            return result

        for name in self.frames:
            resolve_frame(name)

        robot = data.get("robot")
        if not isinstance(robot, dict):
            raise SceneError("robot must be an object describing the mounted URDF instance")
        robot_unknown = sorted(set(robot) - {"instance_id", "robot_name", "root_link", "parent_frame", "pose", "source"})
        if robot_unknown:
            raise SceneError(f"robot has unsupported fields {robot_unknown}")
        robot_name = _identifier(robot.get("robot_name"), "robot.robot_name")
        root_link = _identifier(robot.get("root_link"), "robot.root_link")
        if expected_robot_name is not None and robot_name != expected_robot_name:
            raise SceneError(f"scene robot_name {robot_name!r} does not match URDF robot {expected_robot_name!r}")
        if expected_root_link is not None and root_link != expected_root_link:
            raise SceneError(f"scene root_link {root_link!r} does not match URDF root {expected_root_link!r}")
        parent_frame = _identifier(robot.get("parent_frame"), "robot.parent_frame")
        if parent_frame not in known_scene_frames:
            raise SceneError(f"robot.parent_frame references unknown scene frame {parent_frame!r}")
        robot_pose, parent_from_root = _pose(robot.get("pose"), "robot.pose")
        self.robot = {
            "instance_id": _identifier(robot.get("instance_id"), "robot.instance_id"),
            "robot_name": robot_name,
            "root_link": root_link,
            "parent_frame": parent_frame,
            "pose_in_parent": robot_pose,
            "source": _source(robot.get("source"), "robot.source"),
        }
        self.world_from_robot_root = _matmul(self.world_from_scene_frame[parent_frame], parent_from_root)

        raw_objects = data.get("objects", {})
        if not isinstance(raw_objects, dict):
            raise SceneError("objects must be an object keyed by object ID")
        self.objects: dict[str, dict[str, Any]] = {}
        self._scene_geometry_records: dict[str, dict[str, Any]] = {}
        for raw_object_id, raw_object in raw_objects.items():
            object_id = _identifier(raw_object_id, "objects key")
            if not isinstance(raw_object, dict):
                raise SceneError(f"objects.{object_id} must be an object")
            object_unknown = sorted(set(raw_object) - {"parent_frame", "pose", "semantics", "source", "collision_geometries"})
            if object_unknown:
                raise SceneError(f"objects.{object_id} has unsupported fields {object_unknown}")
            object_parent = _identifier(raw_object.get("parent_frame"), f"objects.{object_id}.parent_frame")
            if object_parent not in known_scene_frames:
                raise SceneError(f"objects.{object_id}.parent_frame references unknown scene frame {object_parent!r}")
            object_pose, parent_from_object = _pose(raw_object.get("pose"), f"objects.{object_id}.pose")
            world_from_object = _matmul(self.world_from_scene_frame[object_parent], parent_from_object)
            object_source = _source(raw_object.get("source"), f"objects.{object_id}.source")
            raw_geometries = raw_object.get("collision_geometries", [])
            if not isinstance(raw_geometries, list):
                raise SceneError(f"objects.{object_id}.collision_geometries must be an array")
            geometries: list[dict[str, Any]] = []
            geometry_ids: set[str] = set()
            for index, raw_geometry in enumerate(raw_geometries):
                label = f"objects.{object_id}.collision_geometries[{index}]"
                if not isinstance(raw_geometry, dict):
                    raise SceneError(f"{label} must be an object")
                geometry_unknown = sorted(set(raw_geometry) - {"id", "pose", "geometry", "semantics", "source"})
                if geometry_unknown:
                    raise SceneError(f"{label} has unsupported fields {geometry_unknown}")
                geometry_id = _identifier(raw_geometry.get("id"), f"{label}.id")
                if geometry_id in geometry_ids:
                    raise SceneError(f"objects.{object_id} has duplicate collision geometry ID {geometry_id!r}")
                geometry_ids.add(geometry_id)
                geometry_pose, object_from_geometry = _pose(raw_geometry.get("pose"), f"{label}.pose")
                entity_id = f"scene_geometry/{object_id}/{geometry_id}"
                record = {
                    "entity_id": entity_id,
                    "id": geometry_id,
                    "object_id": object_id,
                    "pose_in_object": geometry_pose,
                    "geometry": _geometry(raw_geometry.get("geometry"), f"{label}.geometry"),
                    "semantics": _semantics(raw_geometry.get("semantics"), f"{label}.semantics"),
                    "source": _source(raw_geometry.get("source"), f"{label}.source") if "source" in raw_geometry else object_source,
                }
                geometries.append(record)
                self._scene_geometry_records[entity_id] = {
                    **record,
                    "object_from_geometry": object_from_geometry,
                    "world_from_geometry": _matmul(world_from_object, object_from_geometry),
                }
            self.objects[object_id] = {
                "id": object_id,
                "parent_frame": object_parent,
                "pose_in_parent": object_pose,
                "semantics": _semantics(raw_object.get("semantics"), f"objects.{object_id}.semantics"),
                "source": object_source,
                "collision_geometries": geometries,
                "world_from_object": world_from_object,
            }

        gravity = data.get("gravity")
        if gravity is None:
            self.gravity = None
        else:
            if not isinstance(gravity, dict):
                raise SceneError("gravity must be an object")
            gravity_unknown = sorted(set(gravity) - {"vector_xyz_m_s2", "expressed_in_frame", "source"})
            if gravity_unknown:
                raise SceneError(f"gravity has unsupported fields {gravity_unknown}")
            expressed = _identifier(gravity.get("expressed_in_frame"), "gravity.expressed_in_frame")
            if expressed not in known_scene_frames:
                raise SceneError(f"gravity.expressed_in_frame references unknown scene frame {expressed!r}")
            vector = _vector(gravity.get("vector_xyz_m_s2"), 3, "gravity.vector_xyz_m_s2")
            magnitude = math.sqrt(sum(component * component for component in vector))
            if magnitude <= EPSILON:
                raise SceneError("gravity.vector_xyz_m_s2 must be non-zero")
            world_vector = _rotate_vector(self.world_from_scene_frame[expressed], vector)
            self.gravity = {
                "vector_xyz_m_s2": _clean_vector(vector),
                "expressed_in_frame": expressed,
                "vector_in_world_frame_xyz_m_s2": _clean_vector(world_vector),
                "magnitude_m_s2": _clean_number(magnitude),
                "source": _source(gravity.get("source"), "gravity.source"),
            }

    def typed_frames(
        self,
        model: Any,
        supplied_pose: dict[str, float],
        *,
        world_from_robot_root: Matrix | None = None,
        world_from_objects: dict[str, Matrix] | None = None,
    ) -> dict[str, Matrix]:
        """Return typed world poses with optional observation-conditioned placements."""
        effective_root = world_from_robot_root or self.world_from_robot_root
        object_overrides = world_from_objects or {}
        frames: dict[str, Matrix] = {
            f"scene_frame/{name}": transform
            for name, transform in self.world_from_scene_frame.items()
        }
        for object_id, record in self.objects.items():
            frames[f"scene_object/{object_id}"] = object_overrides.get(object_id, record["world_from_object"])
        for entity_id, record in self._scene_geometry_records.items():
            object_id = record["object_id"]
            frames[entity_id] = _matmul(
                object_overrides.get(object_id, self.objects[object_id]["world_from_object"]),
                record["object_from_geometry"],
            )
        robot_frames, _ = model.world_frames(supplied_pose)
        for name, root_from_frame in robot_frames.items():
            world_from_frame = _matmul(effective_root, root_from_frame)
            frames[f"robot_frame/{name}"] = world_from_frame
            if name.startswith("collision/") or name.startswith("visual/"):
                frames[f"robot_geometry/{name}"] = world_from_frame
        return frames

    def transform(
        self,
        reference: str,
        target: str,
        model: Any,
        supplied_pose: dict[str, float],
        *,
        world_from_robot_root: Matrix | None = None,
        world_from_objects: dict[str, Matrix] | None = None,
    ) -> Matrix:
        frames = self.typed_frames(
            model,
            supplied_pose,
            world_from_robot_root=world_from_robot_root,
            world_from_objects=world_from_objects,
        )
        for entity in (reference, target):
            if entity not in frames:
                raise SceneError(
                    f"unknown typed scene entity {entity!r}; use scene_frame/<name>, scene_object/<id>, "
                    "scene_geometry/<object>/<geometry>, robot_frame/<URDF-frame>, or robot_geometry/<URDF-geometry-frame>"
                )
        return _matmul(_inverse_rigid(frames[reference]), frames[target])

    def gravity_in_robot_root(self, world_from_robot_root: Matrix | None = None) -> dict[str, Any]:
        if self.gravity is None:
            return {
                "status": "not_provided",
                "reason": "world scene does not declare a gravity vector",
                "scene_sha256": self.sha256,
                "snapshot_id": self.snapshot["id"],
            }
        effective_root = world_from_robot_root or self.world_from_robot_root
        root_from_world = _inverse_rigid(effective_root)
        root_vector = _rotate_vector(root_from_world, self.gravity["vector_in_world_frame_xyz_m_s2"])
        return {
            "status": "computed",
            "declared_gravity": self.gravity,
            "vector_in_robot_root_xyz_m_s2": _clean_vector(root_vector),
            "robot_root_frame": self.robot["root_link"],
            "world_from_robot_root": pose_record(effective_root),
            "scene_sha256": self.sha256,
            "snapshot_id": self.snapshot["id"],
            "epistemic_scope": "exact coordinate conversion within the declared static scene snapshot; physical mounting and gravity agreement are only as trustworthy as their supplied provenance",
        }

    @staticmethod
    def _normalize_analysis_record(record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        if "bounds_in_root_frame_at_pose" in result:
            result["bounds_in_world_frame_at_snapshot"] = result.pop("bounds_in_root_frame_at_pose")
        principal = result.get("principal_axes")
        if isinstance(principal, dict) and "axes_in_root_frame_at_pose" in principal:
            principal = dict(principal)
            principal["axes_in_world_frame_at_snapshot"] = principal.pop("axes_in_root_frame_at_pose")
            result["principal_axes"] = principal
        if "geometry_axis_landmarks_in_root_frame_at_pose" in result:
            result["geometry_axis_landmarks_in_world_frame_at_snapshot"] = result.pop(
                "geometry_axis_landmarks_in_root_frame_at_pose"
            )
        return result

    def geometry_analysis(
        self,
        model: Any,
        supplied_pose: dict[str, float],
        package_map_path: Path | None = None,
        *,
        world_from_robot_root: Matrix | None = None,
        world_from_objects: dict[str, Matrix] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        package_map = read_package_map(package_map_path)
        analysis: dict[str, dict[str, Any]] = {}
        declared: dict[str, dict[str, Any]] = {}
        effective_root = world_from_robot_root or self.world_from_robot_root
        object_overrides = world_from_objects or {}
        robot_frames, _ = model.world_frames(supplied_pose)
        for link_name, link in model.links.items():
            for geometry_record in link["collisions"]:
                original_frame = geometry_record["frame"]
                entity_id = f"robot_geometry/{original_frame}"
                world_from_geometry = _matmul(effective_root, robot_frames[original_frame])
                geometry = geometry_record["geometry"]
                try:
                    measured, _ = analyze_declared_geometry(
                        geometry,
                        world_from_geometry,
                        model.path,
                        package_map,
                        True,
                    )
                except GeometryError as error:
                    measured = {
                        "status": "not_measured",
                        "geometry_type": geometry["type"],
                        "reason": str(error),
                    }
                analysis[entity_id] = self._normalize_analysis_record({
                    "entity_id": entity_id,
                    "owner_type": "robot_link",
                    "owner": link_name,
                    "robot_geometry_frame": original_frame,
                    "geometry": geometry,
                    **measured,
                })
                declared[entity_id] = {
                    "geometry": geometry,
                    "world_from_geometry": world_from_geometry,
                    "owner_type": "robot_link",
                    "owner": link_name,
                }
        for entity_id, geometry_record in sorted(self._scene_geometry_records.items()):
            geometry = geometry_record["geometry"]
            object_id = geometry_record["object_id"]
            world_from_geometry = _matmul(
                object_overrides.get(object_id, self.objects[object_id]["world_from_object"]),
                geometry_record["object_from_geometry"],
            )
            try:
                measured, _ = analyze_declared_geometry(
                    geometry,
                    world_from_geometry,
                    self.path,
                    package_map,
                    True,
                )
            except GeometryError as error:
                measured = {
                    "status": "not_measured",
                    "geometry_type": geometry["type"],
                    "reason": str(error),
                }
            analysis[entity_id] = self._normalize_analysis_record({
                "entity_id": entity_id,
                "owner_type": "scene_object",
                "owner": geometry_record["object_id"],
                "geometry": geometry,
                **measured,
            })
            declared[entity_id] = {
                "geometry": geometry,
                "world_from_geometry": world_from_geometry,
                "owner_type": "scene_object",
                "owner": geometry_record["object_id"],
            }
        return analysis, declared

    @staticmethod
    def _surface(
        entity_id: str,
        analysis: dict[str, Any],
        declared: dict[str, Any],
    ) -> dict[str, Any] | None:
        if analysis["status"] != "measured":
            return None
        geometry = declared["geometry"]
        geometry_type = geometry["type"]
        transform = declared["world_from_geometry"]
        if geometry_type == "sphere":
            return {
                "representation": "analytic_sphere_solid",
                "type": "sphere",
                "center": _transform_point(transform, [0.0, 0.0, 0.0]),
                "radius": geometry["radius_m"],
                "solid_complete": True,
            }
        if geometry_type == "box":
            triangles = box_surface(geometry["size_xyz_m"], transform)
            components = [triangles]
            representation = "analytic_box_boundary_triangulated_exactly"
            solid_complete = True
        elif geometry_type == "mesh":
            mesh = load_mesh(Path(analysis["source"]["path"]))
            scale_xyz = analysis["source"]["declared_scale_xyz"]
            scaled_vertices = [
                [vertex[axis] * scale_xyz[axis] for axis in range(3)]
                for vertex in mesh.vertices
            ]
            triangles = transform_triangles(scaled_vertices, mesh.faces, transform)
            components = [[triangles[index] for index in indices] for indices in _face_components(mesh.faces)]
            representation = "declared_stl_or_obj_triangles_after_scale_and_world_pose"
            topology = analysis.get("topology", {})
            solid_complete = bool(topology.get("watertight") and topology.get("winding_consistent"))
        else:
            return None
        return {
            "representation": representation,
            "type": "triangles",
            "triangles": triangles,
            "bvh": build_bvh(triangles),
            "components": components,
            "component_test_points": [list(component[0][0]) for component in components],
            "solid_complete": solid_complete,
            "triangle_count": len(triangles),
            "entity_id": entity_id,
        }

    @staticmethod
    def _point_to_triangles(point: Vector, triangles: list[Triangle]) -> tuple[float, Vector]:
        best_distance = math.inf
        best_point = list(triangles[0][0])
        for triangle in triangles:
            candidate = closest_point_on_triangle(point, triangle)
            distance = math.sqrt(sum((candidate[axis] - point[axis]) ** 2 for axis in range(3)))
            if distance < best_distance:
                best_distance = distance
                best_point = candidate
        return best_distance, best_point

    @classmethod
    def _exact_pair(
        cls,
        left_name: str,
        right_name: str,
        left: dict[str, Any],
        right: dict[str, Any],
        tolerance: float,
    ) -> dict[str, Any]:
        if left["type"] == "sphere" and right["type"] == "sphere":
            delta = [right["center"][axis] - left["center"][axis] for axis in range(3)]
            center_distance = math.sqrt(sum(component * component for component in delta))
            collision = center_distance <= left["radius"] + right["radius"] + tolerance
            separation = max(0.0, center_distance - left["radius"] - right["radius"])
            if center_distance > left["radius"] + right["radius"]:
                boundary_distance = separation
            elif center_distance + min(left["radius"], right["radius"]) < max(left["radius"], right["radius"]):
                boundary_distance = max(left["radius"], right["radius"]) - center_distance - min(left["radius"], right["radius"])
            else:
                boundary_distance = 0.0
            return {
                "status": "collision" if collision else "collision_free",
                "separation_m": _clean_number(0.0 if collision else separation),
                "surface_distance_m": _clean_number(boundary_distance),
                "containment_or_solid_overlap": collision and center_distance + min(left["radius"], right["radius"]) < max(left["radius"], right["radius"]),
                "method": "analytic_sphere_solid_distance",
                "witness_point_robot_in_world_m": None,
                "witness_point_environment_in_world_m": None,
            }
        if left["type"] == "sphere" or right["type"] == "sphere":
            sphere_is_left = left["type"] == "sphere"
            sphere = left if sphere_is_left else right
            surface = right if sphere_is_left else left
            distance, closest = cls._point_to_triangles(sphere["center"], surface["triangles"])
            center_inside = surface["solid_complete"] and any(
                point_inside_closed_surface(sphere["center"], component)
                for component in surface["components"]
            )
            collision = center_inside or distance <= sphere["radius"] + tolerance
            if not collision and not surface["solid_complete"]:
                return {
                    "status": "indeterminate",
                    "reason": "triangle surface is not a complete solid and no surface contact was detected",
                    "separation_m": None,
                    "method": "analytic_sphere_to_triangle_surface_with_incomplete_solid",
                }
            separation = max(0.0, distance - sphere["radius"])
            robot_point = None
            environment_point = None
            if sphere_is_left:
                environment_point = _clean_vector(closest)
            else:
                robot_point = _clean_vector(closest)
            return {
                "status": "collision" if collision else "collision_free",
                "separation_m": _clean_number(0.0 if collision else separation),
                "surface_distance_m": None if collision else _clean_number(separation),
                "containment_or_solid_overlap": center_inside,
                "method": "analytic_sphere_to_exact_triangle_solid_distance_and_containment",
                "witness_point_robot_in_world_m": robot_point,
                "witness_point_environment_in_world_m": environment_point,
            }
        distance = bvh_surface_distance(left["triangles"], right["triangles"], left["bvh"], right["bvh"])
        within_tolerance = distance["distance_m"] <= tolerance
        containment: list[dict[str, Any]] = []
        containment_complete = left["solid_complete"] and right["solid_complete"]
        if not within_tolerance and containment_complete:
            for component_index, point in enumerate(left["component_test_points"]):
                if any(point_inside_closed_surface(point, component) for component in right["components"]):
                    containment.append({"contained": left_name, "component": component_index, "container": right_name})
            for component_index, point in enumerate(right["component_test_points"]):
                if any(point_inside_closed_surface(point, component) for component in left["components"]):
                    containment.append({"contained": right_name, "component": component_index, "container": left_name})
        collision = within_tolerance or bool(containment)
        if not collision and not containment_complete:
            return {
                "status": "indeterminate",
                "reason": "positive triangle-surface distance cannot exclude containment because one or both surfaces are not complete solids",
                "surface_distance_m": _clean_number(distance["distance_m"]),
                "separation_m": None,
                "method": "triangle_bvh_surface_distance_with_incomplete_solid",
            }
        return {
            "status": "collision" if collision else "collision_free",
            "separation_m": _clean_number(0.0 if collision else distance["distance_m"]),
            "surface_distance_m": _clean_number(distance["distance_m"]),
            "containment_or_solid_overlap": bool(containment),
            "containment": containment,
            "method": "deterministic_triangle_bvh_distance_and_closed_solid_containment",
            "witness_point_robot_in_world_m": _clean_vector(distance["witness_point_left"]),
            "witness_point_environment_in_world_m": _clean_vector(distance["witness_point_right"]),
            "triangle_pairs_tested": distance["triangle_pairs_tested"],
            "node_pairs_visited": distance["node_pairs_visited"],
        }

    def robot_environment_collisions(
        self,
        model: Any,
        supplied_pose: dict[str, float],
        package_map_path: Path | None = None,
        contact_tolerance_m: float = 1e-9,
        *,
        world_from_robot_root: Matrix | None = None,
        world_from_objects: dict[str, Matrix] | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(contact_tolerance_m) or contact_tolerance_m < 0.0:
            raise SceneError("contact tolerance must be a finite non-negative number of meters")
        effective_root = world_from_robot_root or self.world_from_robot_root
        analysis, declared = self.geometry_analysis(
            model,
            supplied_pose,
            package_map_path,
            world_from_robot_root=effective_root,
            world_from_objects=world_from_objects,
        )
        robot_names = sorted(name for name, record in analysis.items() if record["owner_type"] == "robot_link")
        scene_names = sorted(name for name, record in analysis.items() if record["owner_type"] == "scene_object")
        surfaces = {
            name: self._surface(name, analysis[name], declared[name])
            for name in analysis
        }
        pair_results: list[dict[str, Any]] = []
        for robot_name in robot_names:
            for scene_name in scene_names:
                robot_record, scene_record = analysis[robot_name], analysis[scene_name]
                base = {
                    "robot_geometry": robot_name,
                    "robot_link": robot_record["owner"],
                    "environment_geometry": scene_name,
                    "environment_object": scene_record["owner"],
                    "contact_tolerance_m": contact_tolerance_m,
                }
                if robot_record["status"] != "measured" or scene_record["status"] != "measured":
                    reasons = [
                        f"{name}: {record.get('reason', record['status'])}"
                        for name, record in ((robot_name, robot_record), (scene_name, scene_record))
                        if record["status"] != "measured"
                    ]
                    pair_results.append({
                        **base,
                        "status": "indeterminate",
                        "reason": "unmeasured declared geometry; " + "; ".join(reasons),
                        "separation_m": None,
                        "separation_lower_bound_m": 0.0,
                    })
                    continue
                robot_bounds = robot_record["bounds_in_world_frame_at_snapshot"]
                scene_bounds = scene_record["bounds_in_world_frame_at_snapshot"]
                aabb_lower_bound = math.sqrt(aabb_distance_squared(
                    robot_bounds["min_xyz_m"],
                    robot_bounds["max_xyz_m"],
                    scene_bounds["min_xyz_m"],
                    scene_bounds["max_xyz_m"],
                ))
                base["aabb_separation_lower_bound_m"] = _clean_number(aabb_lower_bound)
                left_surface, right_surface = surfaces[robot_name], surfaces[scene_name]
                if left_surface is not None and right_surface is not None:
                    pair_results.append({
                        **base,
                        **self._exact_pair(robot_name, scene_name, left_surface, right_surface, contact_tolerance_m),
                        "representation_robot": left_surface["representation"],
                        "representation_environment": right_surface["representation"],
                        "trust": "exact for the reported analytic or triangle representations up to floating-point roundoff and the explicit contact tolerance",
                    })
                elif aabb_lower_bound > contact_tolerance_m + EPSILON:
                    pair_results.append({
                        **base,
                        "status": "collision_free",
                        "separation_m": None,
                        "separation_lower_bound_m": _clean_number(aabb_lower_bound),
                        "method": "exact_disjoint_world_aabb_rejection",
                        "reason": "solid collision is impossible because the exact containing AABBs are disjoint; exact surface clearance is not available",
                        "trust": "exact collision-free classification for the declared geometry at this snapshot; distance is a conservative lower bound",
                    })
                else:
                    unsupported = [
                        name
                        for name, surface in ((robot_name, left_surface), (scene_name, right_surface))
                        if surface is None
                    ]
                    pair_results.append({
                        **base,
                        "status": "indeterminate",
                        "separation_m": None,
                        "separation_lower_bound_m": _clean_number(aabb_lower_bound),
                        "method": "world_aabb_candidate_without_exact_supported_solid_pair",
                        "reason": f"AABBs overlap or touch and exact solid classification is unavailable for {unsupported}",
                    })

        collisions = [record for record in pair_results if record["status"] == "collision"]
        unresolved = [record for record in pair_results if record["status"] == "indeterminate"]
        if not robot_names or not scene_names:
            overall_status = "not_applicable"
        elif collisions:
            overall_status = "collision"
        elif unresolved:
            overall_status = "indeterminate"
        else:
            overall_status = "collision_free"

        if not pair_results:
            minimum = {
                "status": "not_applicable",
                "distance_m": None,
                "reason": "the robot or scene snapshot declares no collision geometry",
            }
        elif collisions:
            witness = sorted(collisions, key=lambda record: (record["robot_geometry"], record["environment_geometry"]))[0]
            minimum = {
                "status": "computed",
                "distance_m": 0.0,
                "pair": {
                    "robot_geometry": witness["robot_geometry"],
                    "environment_geometry": witness["environment_geometry"],
                },
                "meaning": "zero solid separation because at least one declared robot/environment pair collides",
            }
        else:
            exact_candidates = [record for record in pair_results if record.get("separation_m") is not None]
            exact_best = min(exact_candidates, key=lambda record: record["separation_m"]) if exact_candidates else None
            inexact = [record for record in pair_results if record.get("separation_m") is None]
            inexact_lower_bounds = [float(record.get("separation_lower_bound_m", 0.0)) for record in inexact]
            lower_candidates = [record["separation_m"] for record in exact_candidates] + inexact_lower_bounds
            global_lower_bound = min(lower_candidates) if lower_candidates else None
            exact_is_global = (
                exact_best is not None
                and all(lower_bound >= exact_best["separation_m"] - EPSILON for lower_bound in inexact_lower_bounds)
            )
            if exact_is_global:
                minimum = {
                    "status": "computed",
                    "distance_m": exact_best["separation_m"],
                    "pair": {
                        "robot_geometry": exact_best["robot_geometry"],
                        "environment_geometry": exact_best["environment_geometry"],
                    },
                    "meaning": "exact global minimum: every inexact pair has an AABB lower bound no smaller than this exact pair distance",
                }
            else:
                minimum = {
                    "status": "indeterminate",
                    "distance_m": None,
                    "global_lower_bound_m": None if global_lower_bound is None else _clean_number(global_lower_bound),
                    "best_exact_candidate_m": None if exact_best is None else exact_best["separation_m"],
                    "pairs_that_could_be_closer": [
                        {
                            "robot_geometry": record["robot_geometry"],
                            "environment_geometry": record["environment_geometry"],
                            "lower_bound_m": record.get("separation_lower_bound_m", 0.0),
                        }
                        for record in inexact
                        if exact_best is None or float(record.get("separation_lower_bound_m", 0.0)) < exact_best["separation_m"] - EPSILON
                    ],
                    "meaning": "an exact global minimum cannot be promoted from partial surface coverage",
                }
        return {
            "schema_version": COLLISION_SCHEMA,
            "status": overall_status,
            "scene": {
                "scene_id": self.scene_id,
                "snapshot": self.snapshot,
                "source_path": str(self.path),
                "sha256": self.sha256,
            },
            "robot": {
                "instance_id": self.robot["instance_id"],
                "robot_name": model.name,
                "root_link": model.root_link,
                "world_from_robot_root": pose_record(effective_root),
            },
            "contact_tolerance_m": contact_tolerance_m,
            "method": "all declared robot collision geometries against all declared scene collision geometries; exact analytic/triangle solids where supported, otherwise exact AABB rejection or fail-closed indeterminate",
            "minimum_separation": minimum,
            "coverage": {
                "declared_robot_collision_geometry_count": len(robot_names),
                "measured_robot_collision_geometry_count": sum(analysis[name]["status"] == "measured" for name in robot_names),
                "declared_environment_collision_geometry_count": len(scene_names),
                "measured_environment_collision_geometry_count": sum(analysis[name]["status"] == "measured" for name in scene_names),
                "declared_cross_pair_count": len(pair_results),
                "collision_pair_count": len(collisions),
                "indeterminate_pair_count": len(unresolved),
                "exact_or_analytic_pair_count": sum(record.get("separation_m") is not None for record in pair_results),
                "aabb_only_collision_free_pair_count": sum(record.get("method") == "exact_disjoint_world_aabb_rejection" for record in pair_results),
                "unsupported_exact_cylinder_pairs_may_be_indeterminate": True,
                "physical_world_completeness": "not_established",
            },
            "geometry_analysis": analysis,
            "pair_results": pair_results,
            "epistemic_scope": "exact only for the declared static scene snapshot, stated robot pose, successfully measured geometry, supported solid representations, and explicit tolerance; it does not prove that the snapshot is current, complete, calibrated, or physically accurate",
        }

    def canonical(
        self,
        model: Any,
        supplied_pose: dict[str, float],
        package_map_path: Path | None = None,
        contact_tolerance_m: float = 1e-9,
    ) -> dict[str, Any]:
        typed = self.typed_frames(model, supplied_pose)
        gravity = self.gravity_in_robot_root()
        collision = self.robot_environment_collisions(
            model,
            supplied_pose,
            package_map_path,
            contact_tolerance_m,
        )
        return {
            "schema_version": BOUND_SCENE_SCHEMA,
            "status": "parsed_validated_and_bound",
            "scene_id": self.scene_id,
            "source": {"path": str(self.path), "sha256": self.sha256, "provenance": self.source},
            "snapshot": self.snapshot,
            "world_frame": self.world_frame,
            "identity_grammar": {
                "scene_frame": "scene_frame/<name>",
                "scene_object": "scene_object/<id>",
                "scene_geometry": "scene_geometry/<object-id>/<geometry-id>",
                "robot_frame": "robot_frame/<exact-URDF-frame-name>",
                "robot_geometry": "robot_geometry/<exact-URDF-geometry-frame-name>",
            },
            "robot_mount": {
                **self.robot,
                "parent_entity": f"scene_frame/{self.robot['parent_frame']}",
                "root_entity": f"robot_frame/{self.robot['root_link']}",
                "world_from_robot_root": pose_record(self.world_from_robot_root),
            },
            "gravity": gravity,
            "scene_frames": {
                name: {
                    **record,
                    "entity_id": f"scene_frame/{name}",
                    "world_from_frame": pose_record(self.world_from_scene_frame[name]),
                }
                for name, record in sorted(self.frames.items())
            },
            "objects": {
                object_id: {
                    **{key: value for key, value in record.items() if key != "world_from_object"},
                    "entity_id": f"scene_object/{object_id}",
                    "world_from_object": pose_record(record["world_from_object"]),
                }
                for object_id, record in sorted(self.objects.items())
            },
            "typed_frame_poses_in_world": {
                name: pose_record(transform)
                for name, transform in sorted(typed.items())
            },
            "robot_environment_collision": collision,
            "capabilities": {
                "validated_scene_frame_tree": True,
                "explicit_static_snapshot": True,
                "explicit_robot_root_mount": True,
                "world_to_robot_transform_queries": True,
                "world_gravity_to_robot_root_conversion": self.gravity is not None,
                "robot_environment_collision": True,
                "supported_exact_solid_representations": ["box", "sphere", "watertight_consistently_wound_stl_or_obj"],
                "cylinder_collision": "exact AABB rejection; overlapping candidates are indeterminate",
            },
            "epistemic_scope": "the parser proves internal consistency and deterministic consequences of this declared snapshot; scene provenance labels are retained but not independently verified, and omitted objects are unknown rather than absent from reality",
        }


def read_world_scene(path: Path | None, model: Any | None = None) -> WorldScene | None:
    if path is None:
        return None
    return WorldScene(
        path,
        expected_robot_name=None if model is None else model.name,
        expected_root_link=None if model is None else model.root_link,
    )
