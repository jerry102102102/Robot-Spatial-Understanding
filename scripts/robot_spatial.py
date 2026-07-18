#!/usr/bin/env python3
"""Deterministic URDF kinematics and declared-geometry oracle.

No third-party packages are required. It measures STL/OBJ contents, computes
conservative collision AABBs, and can verify triangle-surface distance/contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mesh_geometry import (
    GeometryError,
    analyze_declared_geometry,
    broadphase_overlaps,
    load_mesh,
    read_package_map,
    render_scene_svg,
)
from ros_workspace import (
    WorkspaceError,
    discover_packages,
    nearest_package_root,
    normalize_expanded_urdf,
    package_references_from_urdf,
    read_package_lookups,
    sha256_path,
    tree_manifest,
    write_ament_index_shim,
    xacro_environment,
)
from srdf_semantics import SRDFError, parse_srdf, resolve_named_pose
from spatial_context import ContextError, retrieve_context, write_agent_context
from spatial_articulation import (
    ARTICULATION_SCHEMA,
    ArticulationError,
    compare_articulation_grammars,
    evaluate_articulation_grammar,
    read_articulation_grammar,
    verify_articulation_grammar,
    write_articulation_grammar,
)
from spatial_kinematic_import import (
    KinematicImportError,
    detect_source_format,
    load_imported_model,
    source_binding as articulation_source_binding,
)
from spatial_constraints import (
    GRAPH_SCHEMA as CONSTRAINT_GRAPH_SCHEMA,
    ConstraintError,
    evaluate_constraint_graph,
    read_constraint_graph,
    solve_constraint_graph,
    verify_constraint_graph,
    write_constraint_graph,
)
from spatial_configuration import (
    ATLAS_SCHEMA as CONFIGURATION_ATLAS_SCHEMA,
    ConfigurationError,
    verify_configuration_atlas,
    write_configuration_atlas,
)
from spatial_concepts import (
    CONCEPT_SCHEMA,
    ConceptError,
    query_concept_graph_files,
    verify_concept_graph,
    write_concept_graph,
    write_concept_graph_from_context,
)
from spatial_functional import (
    MODEL_SCHEMA as FUNCTIONAL_MODEL_SCHEMA,
    FunctionalError,
    query_functional_model_files,
    verify_functional_model,
    write_functional_model,
    write_functional_model_from_context,
)
from spatial_action_assurance import (
    MODEL_SCHEMA as ACTION_ASSURANCE_MODEL_SCHEMA,
    ActionAssuranceError,
    query_action_assurance_files,
    verify_action_assurance,
    write_action_assurance,
)
from spatial_evaluation import EvaluationError, generate_evaluation
from spatial_invariants import InvariantError, read_invariant_contract, verify_invariant_contract
from spatial_motion import (
    MOTION_ATLAS_SCHEMA,
    MotionError,
    verify_counterfactual_motion_atlas,
    write_counterfactual_motion_atlas,
)
from spatial_render import ATLAS_SCHEMA, RenderError, verify_semantic_render_atlas, write_semantic_render_atlas
from temporal_observation import ObservationError, resolve_observation
from triangle_geometry import (
    TriangleError,
    aabb_distance_squared,
    box_surface,
    build_bvh,
    bvh_surface_distance,
    point_inside_closed_surface,
    transform_triangles,
)
from world_scene import SceneError, WorldScene, read_world_scene


SCHEMA_VERSION = "robot-spatial.v2"
SUPPORTED_JOINTS = {"fixed", "revolute", "continuous", "prismatic"}
MESH_GEOMETRY_KINDS = frozenset({"visual", "collision"})
EPSILON = 1e-12


class SpatialError(ValueError):
    """An input or semantic error that should be shown to the caller."""


def mesh_inspection_kinds(
    inspect_meshes: bool,
    inspect_mesh_kinds: Iterable[str] | None = None,
) -> set[str]:
    """Resolve legacy all-mesh inspection and an explicit geometry-kind selection."""
    if inspect_mesh_kinds is None:
        return set(MESH_GEOMETRY_KINDS) if inspect_meshes else set()
    selected = set(inspect_mesh_kinds)
    unsupported = sorted(selected - MESH_GEOMETRY_KINDS)
    if unsupported:
        raise SpatialError(
            f"unsupported mesh inspection kind(s) {unsupported}; expected visual and/or collision"
        )
    return selected


Matrix = list[list[float]]
Vector = list[float]
Tensor3 = list[list[float]]


def identity() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def transform_point(transform: Matrix, point: Vector) -> Vector:
    return [sum(transform[i][j] * point[j] for j in range(3)) + transform[i][3] for i in range(3)]


def rotate_vector(transform: Matrix, vector: Vector) -> Vector:
    return [sum(transform[i][j] * vector[j] for j in range(3)) for i in range(3)]


def cross_product(a: Vector, b: Vector) -> Vector:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def inverse_rigid(transform: Matrix) -> Matrix:
    result = identity()
    for i in range(3):
        for j in range(3):
            result[i][j] = transform[j][i]
        result[i][3] = -sum(transform[j][i] * transform[j][3] for j in range(3))
    return result


def translation(vector: Vector) -> Matrix:
    result = identity()
    result[0][3], result[1][3], result[2][3] = vector
    return result


def rpy_matrix(rpy: Vector) -> Matrix:
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


def origin_matrix(xyz: Vector, rpy: Vector) -> Matrix:
    result = rpy_matrix(rpy)
    result[0][3], result[1][3], result[2][3] = xyz
    return result


def axis_angle(axis: Vector, angle: float) -> Matrix:
    x, y, z = normalized(axis, "joint axis")
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s, 0.0],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s, 0.0],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def normalized(vector: Vector, label: str) -> Vector:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= EPSILON:
        raise SpatialError(f"{label} must be non-zero")
    return [component / magnitude for component in vector]


def quaternion_xyzw(transform: Matrix) -> Vector:
    m = transform
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    quaternion = [qx, qy, qz, qw]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return quaternion


def clean_number(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(value, 12)


def clean_vector(vector: Iterable[float]) -> Vector:
    return [clean_number(value) for value in vector]


def clean_matrix(matrix: Matrix) -> Matrix:
    return [clean_vector(row) for row in matrix]


def tensor3_from_urdf(components: dict[str, float | None]) -> Tensor3:
    """Return the symmetric inertia tensor declared in an URDF inertial frame."""
    missing = [name for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz") if components.get(name) is None]
    if missing:
        raise SpatialError(f"inertia tensor is missing components: {missing}")
    ixx, ixy, ixz = (float(components[name]) for name in ("ixx", "ixy", "ixz"))
    iyy, iyz, izz = (float(components[name]) for name in ("iyy", "iyz", "izz"))
    return [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]


def tensor3_multiply(left: Tensor3, right: Tensor3) -> Tensor3:
    return [[sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3)] for row in range(3)]


def tensor3_transpose(matrix: Tensor3) -> Tensor3:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def rotate_tensor3(transform: Matrix, tensor: Tensor3) -> Tensor3:
    rotation = [[transform[row][column] for column in range(3)] for row in range(3)]
    return tensor3_multiply(tensor3_multiply(rotation, tensor), tensor3_transpose(rotation))


def add_tensor3(left: Tensor3, right: Tensor3) -> Tensor3:
    return [[left[row][column] + right[row][column] for column in range(3)] for row in range(3)]


def parallel_axis_tensor(mass: float, displacement: Vector) -> Tensor3:
    squared_norm = sum(value * value for value in displacement)
    return [
        [mass * ((squared_norm if row == column else 0.0) - displacement[row] * displacement[column]) for column in range(3)]
        for row in range(3)
    ]


def symmetric_eigenvalues_3x3(tensor: Tensor3) -> Vector:
    """Deterministic Jacobi eigenvalues for a real symmetric 3x3 tensor."""
    matrix = [list(row) for row in tensor]
    scale = max(1.0, *(abs(value) for row in matrix for value in row))
    tolerance = scale * 1e-15
    for _ in range(32):
        row, column = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(matrix[pair[0]][pair[1]]))
        off_diagonal = matrix[row][column]
        if abs(off_diagonal) <= tolerance:
            break
        angle = 0.5 * math.atan2(2.0 * off_diagonal, matrix[column][column] - matrix[row][row])
        cosine, sine = math.cos(angle), math.sin(angle)
        diagonal_row, diagonal_column = matrix[row][row], matrix[column][column]
        matrix[row][row] = cosine * cosine * diagonal_row - 2.0 * sine * cosine * off_diagonal + sine * sine * diagonal_column
        matrix[column][column] = sine * sine * diagonal_row + 2.0 * sine * cosine * off_diagonal + cosine * cosine * diagonal_column
        matrix[row][column] = matrix[column][row] = 0.0
        for other in range(3):
            if other in {row, column}:
                continue
            other_row, other_column = matrix[other][row], matrix[other][column]
            matrix[other][row] = matrix[row][other] = cosine * other_row - sine * other_column
            matrix[other][column] = matrix[column][other] = sine * other_row + cosine * other_column
    return sorted(clean_number(matrix[index][index]) for index in range(3))


def tensor3_record(tensor: Tensor3) -> dict[str, Any]:
    cleaned = clean_matrix(tensor)
    return {
        "ixx": cleaned[0][0],
        "ixy": cleaned[0][1],
        "ixz": cleaned[0][2],
        "iyy": cleaned[1][1],
        "iyz": cleaned[1][2],
        "izz": cleaned[2][2],
        "matrix_3x3_rowmajor": cleaned,
    }


def validate_inertial_declaration(mass: float | None, components: dict[str, float | None] | None) -> dict[str, Any]:
    missing: list[str] = []
    if mass is None:
        missing.append("mass.value")
    if components is None:
        missing.append("inertia")
    else:
        missing.extend(f"inertia.{name}" for name, value in components.items() if value is None)
    if missing:
        return {"status": "incomplete", "issues": [f"missing {name}" for name in missing], "principal_moments_kg_m2": None}
    assert mass is not None and components is not None
    tensor = tensor3_from_urdf(components)
    principal = symmetric_eigenvalues_3x3(tensor)
    scale = max(1.0, *(abs(value) for row in tensor for value in row))
    tolerance = scale * 1e-10
    issues: list[str] = []
    if mass <= 0.0:
        issues.append("mass must be positive")
    if principal[0] < -tolerance:
        issues.append(f"inertia tensor is not positive semidefinite; principal moments are {principal}")
    if principal[2] > principal[0] + principal[1] + tolerance:
        issues.append(f"principal moments violate the rigid-body triangle inequality: {principal}")
    return {
        "status": "invalid" if issues else "valid",
        "issues": issues,
        "principal_moments_kg_m2": principal,
        "validation_tolerance_kg_m2": clean_number(tolerance),
    }


def radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def pose_record(transform: Matrix) -> dict[str, Any]:
    return {
        "translation_xyz_m": clean_vector([transform[0][3], transform[1][3], transform[2][3]]),
        "quaternion_xyzw": clean_vector(quaternion_xyzw(transform)),
        "matrix_4x4_rowmajor": clean_matrix(transform),
    }


def parse_vector(raw: str | None, default: Vector, label: str) -> Vector:
    if raw is None:
        return list(default)
    try:
        values = [float(value) for value in raw.split()]
    except ValueError as error:
        raise SpatialError(f"{label} must contain numeric values: {raw!r}") from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise SpatialError(f"{label} must contain exactly three finite values: {raw!r}")
    return values


def optional_float(element: ET.Element | None, name: str) -> float | None:
    if element is None or element.get(name) is None:
        return None
    try:
        value = float(element.get(name, ""))
    except ValueError as error:
        raise SpatialError(f"{element.tag}.{name} must be numeric") from error
    if not math.isfinite(value):
        raise SpatialError(f"{element.tag}.{name} must be finite")
    return value


def element_text(element: ET.Element | None) -> str | None:
    """Return normalized element text without inventing a value for absence."""
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def named_parameters(element: ET.Element, context: str) -> dict[str, str]:
    """Parse repeated <param name="...">text</param> declarations losslessly."""
    parameters: dict[str, str] = {}
    for parameter in element.findall("param"):
        name = parameter.get("name")
        if not name:
            raise SpatialError(f"{context} contains a param without a name")
        if name in parameters:
            raise SpatialError(f"{context} contains duplicate param {name!r}")
        value = element_text(parameter)
        if value is None:
            raise SpatialError(f"{context} param {name!r} has no value")
        parameters[name] = value
    return parameters


def child_required(element: ET.Element, tag: str, context: str) -> ET.Element:
    child = element.find(tag)
    if child is None:
        raise SpatialError(f"{context} is missing <{tag}>")
    return child


def origin_from(element: ET.Element) -> tuple[Vector, Vector, Matrix]:
    origin = element.find("origin")
    xyz = parse_vector(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0], "origin xyz")
    rpy = parse_vector(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0], "origin rpy")
    return xyz, rpy, origin_matrix(xyz, rpy)


def geometry_from(container: ET.Element, warnings: list[str] | None = None, context: str | None = None) -> dict[str, Any]:
    geometry = container.find("geometry")
    if geometry is None:
        raise SpatialError(f"<{container.tag}> must contain exactly one geometry primitive")
    primitive_tags = {"box", "cylinder", "sphere", "mesh"}
    primitives = [child for child in geometry if child.tag in primitive_tags]
    if len(primitives) != 1:
        raise SpatialError(f"<{container.tag}> must contain exactly one recognized geometry primitive; found {[child.tag for child in geometry]}")
    ignored = [child.tag for child in geometry if child.tag not in primitive_tags]
    if ignored and warnings is not None:
        label = context or f"<{container.tag}>"
        warnings.append(f"{label} contains misplaced non-geometry elements inside <geometry> that were ignored: {ignored}")
    shape = primitives[0]
    if shape.tag == "box":
        size = parse_vector(shape.get("size"), [], "box size")
        if any(value <= 0.0 for value in size):
            raise SpatialError(f"box dimensions must be positive: {size}")
        return {"type": "box", "size_xyz_m": size}
    if shape.tag == "cylinder":
        radius, length = optional_float(shape, "radius"), optional_float(shape, "length")
        if radius is None or length is None:
            raise SpatialError("cylinder requires radius and length")
        if radius <= 0.0 or length <= 0.0:
            raise SpatialError("cylinder radius and length must be positive")
        return {"type": "cylinder", "radius_m": radius, "length_m": length}
    if shape.tag == "sphere":
        radius = optional_float(shape, "radius")
        if radius is None:
            raise SpatialError("sphere requires radius")
        if radius <= 0.0:
            raise SpatialError("sphere radius must be positive")
        return {"type": "sphere", "radius_m": radius}
    if shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise SpatialError("mesh requires filename")
        scale = parse_vector(shape.get("scale"), [1.0, 1.0, 1.0], "mesh scale")
        if any(abs(value) <= EPSILON for value in scale):
            raise SpatialError(f"mesh scale components must be non-zero: {scale}")
        return {
            "type": "mesh",
            "uri": filename,
            "scale_xyz": scale,
        }
    raise SpatialError(f"unsupported geometry type: {shape.tag}")


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Vector
    origin_rpy: Vector
    origin: Matrix
    axis: Vector
    limit: dict[str, float | None]
    mimic: dict[str, float | str] | None
    dynamics: dict[str, Any] | None


class RobotModel:
    def __init__(self, path: Path):
        self.path = path.resolve()
        raw = self.path.read_bytes()
        if self.path.suffix == ".xacro":
            raise SpatialError("input appears to be Xacro; expand it to a concrete .urdf first")
        try:
            self.xml_root = ET.fromstring(raw)
        except ET.ParseError as error:
            raise SpatialError(f"invalid XML: {error}") from error
        if self.xml_root.tag != "robot":
            raise SpatialError("root XML element must be <robot>")
        xacro_elements = [
            element.tag
            for element in self.xml_root.iter()
            if isinstance(element.tag, str)
            and (
                element.tag.startswith("xacro:")
                or (element.tag.startswith("{") and "xacro" in element.tag.partition("}")[0].lower())
            )
        ]
        if xacro_elements:
            raise SpatialError(
                f"input contains unexpanded Xacro elements {sorted(set(xacro_elements))}; expand it to a concrete .urdf first"
            )
        self.name = self.xml_root.get("name") or "unnamed_robot"
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.semantic_sha256 = hashlib.sha256(ET.tostring(self.xml_root, encoding="utf-8")).hexdigest()
        self._parse_warnings: list[str] = []
        self.links = self._parse_links()
        self.joints = self._parse_joints()
        self.actuation = self._parse_actuation()
        self.child_joint: dict[str, Joint] = {}
        self.children: dict[str, list[Joint]] = {name: [] for name in self.links}
        for joint in self.joints.values():
            if joint.child in self.child_joint:
                other = self.child_joint[joint.child].name
                raise SpatialError(f"link {joint.child!r} has two parent joints: {other!r}, {joint.name!r}")
            self.child_joint[joint.child] = joint
            self.children[joint.parent].append(joint)
        roots = sorted(set(self.links) - set(self.child_joint))
        if len(roots) != 1:
            raise SpatialError(f"URDF must have exactly one root link; found {roots}")
        self.root_link = roots[0]
        self._validate_tree()

    def _parse_links(self) -> dict[str, dict[str, Any]]:
        links: dict[str, dict[str, Any]] = {}
        for element in self.xml_root.findall("link"):
            name = element.get("name")
            if not name:
                raise SpatialError("link is missing name")
            if name in links:
                raise SpatialError(f"duplicate link name: {name}")
            record: dict[str, Any] = {"name": name, "visuals": [], "collisions": [], "inertial": None}
            for kind, tag in (("visuals", "visual"), ("collisions", "collision")):
                for index, child in enumerate(element.findall(tag)):
                    xyz, rpy, _ = origin_from(child)
                    record[kind].append({
                        "frame": f"{tag}/{name}/{index}",
                        "name": child.get("name"),
                        "origin_xyz_m": xyz,
                        "origin_rpy_rad": rpy,
                        "geometry": geometry_from(child, self._parse_warnings, f"{tag} {name!r} index {index}"),
                    })
            inertial = element.find("inertial")
            if inertial is not None:
                xyz, rpy, _ = origin_from(inertial)
                mass_element = inertial.find("mass")
                inertia_element = inertial.find("inertia")
                mass = optional_float(mass_element, "value")
                inertia = (
                    {key: optional_float(inertia_element, key) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                    if inertia_element is not None
                    else None
                )
                record["inertial"] = {
                    "frame": f"inertial/{name}",
                    "origin_xyz_m": xyz,
                    "origin_rpy_rad": rpy,
                    "mass_kg": mass,
                    "inertia_kg_m2": inertia,
                    "validation": validate_inertial_declaration(mass, inertia),
                }
            links[name] = record
        if not links:
            raise SpatialError("URDF contains no links")
        return links

    def _parse_joints(self) -> dict[str, Joint]:
        joints: dict[str, Joint] = {}
        for element in self.xml_root.findall("joint"):
            name, joint_type = element.get("name"), element.get("type")
            if not name or not joint_type:
                raise SpatialError("joint requires name and type")
            if name in joints:
                raise SpatialError(f"duplicate joint name: {name}")
            if joint_type not in SUPPORTED_JOINTS:
                raise SpatialError(f"joint {name!r} has unsupported type {joint_type!r}")
            parent_element = child_required(element, "parent", f"joint {name!r}")
            child_element = child_required(element, "child", f"joint {name!r}")
            parent, child = parent_element.get("link"), child_element.get("link")
            if not parent or not child or parent not in self.links or child not in self.links:
                raise SpatialError(f"joint {name!r} references missing parent/child link: {parent!r} -> {child!r}")
            xyz, rpy, origin = origin_from(element)
            axis_element = element.find("axis")
            axis = parse_vector(axis_element.get("xyz") if axis_element is not None else None, [1.0, 0.0, 0.0], f"joint {name} axis")
            if joint_type != "fixed":
                axis = normalized(axis, f"joint {name} axis")
            limit_element = element.find("limit")
            limit = {key: optional_float(limit_element, key) for key in ("lower", "upper", "effort", "velocity")}
            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                source = mimic_element.get("joint")
                if not source:
                    raise SpatialError(f"joint {name!r} mimic is missing source joint")
                mimic = {
                    "joint": source,
                    "multiplier": optional_float(mimic_element, "multiplier") if mimic_element.get("multiplier") is not None else 1.0,
                    "offset": optional_float(mimic_element, "offset") if mimic_element.get("offset") is not None else 0.0,
                }
            dynamics_element = element.find("dynamics")
            dynamics = None
            if dynamics_element is not None:
                standard = {
                    key: optional_float(dynamics_element, key)
                    for key in ("damping", "friction")
                    if dynamics_element.get(key) is not None
                }
                extensions = {
                    key: value
                    for key, value in sorted(dynamics_element.attrib.items())
                    if key not in {"damping", "friction"}
                }
                dynamics = {
                    "standard_urdf": standard,
                    "uninterpreted_extension_attributes": extensions,
                }
                if extensions:
                    self._parse_warnings.append(
                        f"joint {name!r} dynamics has nonstandard attributes preserved without interpretation: {sorted(extensions)}"
                    )
            joints[name] = Joint(name, joint_type, parent, child, xyz, rpy, origin, axis, limit, mimic, dynamics)
        for joint in joints.values():
            if joint.mimic and joint.mimic["joint"] not in joints:
                raise SpatialError(f"joint {joint.name!r} mimics unknown joint {joint.mimic['joint']!r}")
            if joint.mimic:
                source = joints[str(joint.mimic["joint"])]
                angular_types = {"revolute", "continuous"}
                compatible = (joint.type in angular_types and source.type in angular_types) or (joint.type == source.type == "prismatic")
                if not compatible:
                    raise SpatialError(f"mimic joint {joint.name!r} type {joint.type!r} is incompatible with source {source.name!r} type {source.type!r}")
        return joints

    def _parse_actuation(self) -> dict[str, Any]:
        """Parse actuation/control declarations while making no runtime claim."""

        def component_name(element: ET.Element, context: str) -> str:
            name = element.get("name") or element_text(element.find("name"))
            if not name:
                raise SpatialError(f"{context} is missing a name")
            return name

        def interface_records(component: ET.Element, tag: str, context: str) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            names: set[str] = set()
            for interface in component.findall(tag):
                name = interface.get("name")
                if not name:
                    raise SpatialError(f"{context} contains <{tag}> without name")
                if name in names:
                    raise SpatialError(f"{context} contains duplicate {tag} {name!r}")
                names.add(name)
                records.append({"name": name, "parameters": named_parameters(interface, f"{context} {tag} {name!r}")})
            return records

        transmissions: dict[str, Any] = {}
        for element in self.xml_root.findall("transmission"):
            name = element.get("name")
            if not name:
                raise SpatialError("transmission is missing name")
            if name in transmissions:
                raise SpatialError(f"duplicate transmission name: {name}")
            joint_records: list[dict[str, Any]] = []
            for joint_element in element.findall("joint"):
                joint_name = component_name(joint_element, f"transmission {name!r} joint")
                if joint_name not in self.joints:
                    raise SpatialError(f"transmission {name!r} references unknown joint {joint_name!r}")
                joint_records.append({
                    "name": joint_name,
                    "hardware_interfaces": [
                        value
                        for item in joint_element.findall("hardwareInterface")
                        if (value := element_text(item)) is not None
                    ],
                })
            actuator_records: list[dict[str, Any]] = []
            actuator_names: set[str] = set()
            for actuator_element in element.findall("actuator"):
                actuator_name = component_name(actuator_element, f"transmission {name!r} actuator")
                if actuator_name in actuator_names:
                    raise SpatialError(f"transmission {name!r} contains duplicate actuator {actuator_name!r}")
                actuator_names.add(actuator_name)
                reduction_element = actuator_element.find("mechanicalReduction")
                reduction = None
                if reduction_element is not None:
                    raw_reduction = element_text(reduction_element)
                    try:
                        reduction = float(raw_reduction or "")
                    except ValueError as error:
                        raise SpatialError(
                            f"transmission {name!r} actuator {actuator_name!r} mechanicalReduction must be numeric"
                        ) from error
                    if not math.isfinite(reduction):
                        raise SpatialError(
                            f"transmission {name!r} actuator {actuator_name!r} mechanicalReduction must be finite"
                        )
                actuator_records.append({
                    "name": actuator_name,
                    "mechanical_reduction_declared": reduction,
                    "hardware_interfaces": [
                        value
                        for item in actuator_element.findall("hardwareInterface")
                        if (value := element_text(item)) is not None
                    ],
                })
            transmissions[name] = {
                "name": name,
                "type": element_text(element.find("type")),
                "joints": joint_records,
                "actuators": actuator_records,
            }

        systems: dict[str, Any] = {}
        for element in self.xml_root.findall("ros2_control"):
            name, control_type = element.get("name"), element.get("type")
            if not name or not control_type:
                raise SpatialError("ros2_control requires name and type")
            if name in systems:
                raise SpatialError(f"duplicate ros2_control name: {name}")
            hardware = element.find("hardware")
            hardware_record = None
            if hardware is not None:
                hardware_record = {
                    "plugin": element_text(hardware.find("plugin")),
                    "parameters": named_parameters(hardware, f"ros2_control {name!r} hardware"),
                }
            components: dict[str, dict[str, Any]] = {"joints": {}, "sensors": {}, "gpios": {}}
            for tag, collection in (("joint", "joints"), ("sensor", "sensors"), ("gpio", "gpios")):
                for component in element.findall(tag):
                    component_value = component_name(component, f"ros2_control {name!r} {tag}")
                    if component_value in components[collection]:
                        raise SpatialError(f"ros2_control {name!r} contains duplicate {tag} {component_value!r}")
                    if tag == "joint" and component_value not in self.joints:
                        raise SpatialError(f"ros2_control {name!r} references unknown joint {component_value!r}")
                    components[collection][component_value] = {
                        "name": component_value,
                        "command_interfaces": interface_records(
                            component,
                            "command_interface",
                            f"ros2_control {name!r} {tag} {component_value!r}",
                        ),
                        "state_interfaces": interface_records(
                            component,
                            "state_interface",
                            f"ros2_control {name!r} {tag} {component_value!r}",
                        ),
                        "parameters": named_parameters(component, f"ros2_control {name!r} {tag} {component_value!r}"),
                    }
            systems[name] = {
                "name": name,
                "type": control_type,
                "hardware": hardware_record,
                **components,
            }

        bindings: dict[str, Any] = {}
        for joint_name in sorted(self.joints):
            legacy = sorted(
                transmission_name
                for transmission_name, transmission in transmissions.items()
                if any(record["name"] == joint_name for record in transmission["joints"])
            )
            ros2: list[dict[str, Any]] = []
            for system_name, system in sorted(systems.items()):
                component = system["joints"].get(joint_name)
                if component is None:
                    continue
                ros2.append({
                    "system": system_name,
                    "command_interfaces": [record["name"] for record in component["command_interfaces"]],
                    "state_interfaces": [record["name"] for record in component["state_interfaces"]],
                })
            bindings[joint_name] = {
                "legacy_transmissions": legacy,
                "ros2_control": ros2,
                "has_declared_command_interface": any(record["command_interfaces"] for record in ros2),
                "has_declared_state_interface": any(record["state_interfaces"] for record in ros2),
            }
        movable = sorted(name for name, joint in self.joints.items() if joint.type != "fixed")
        commandable = sorted(name for name in movable if bindings[name]["has_declared_command_interface"])
        state_observed = sorted(name for name in movable if bindings[name]["has_declared_state_interface"])
        transmitted = sorted(name for name in movable if bindings[name]["legacy_transmissions"])
        return {
            "schema_version": "robot-spatial-actuation-declarations.v1",
            "legacy_transmissions": transmissions,
            "ros2_control_systems": systems,
            "joint_bindings": bindings,
            "coverage": {
                "movable_joint_count": len(movable),
                "movable_joints": movable,
                "joints_with_declared_ros2_command_interfaces": commandable,
                "joints_with_declared_ros2_state_interfaces": state_observed,
                "joints_with_legacy_transmission_bindings": transmitted,
                "movable_joints_without_declared_command_interface": sorted(set(movable) - set(commandable)),
                "ros2_control_system_count": len(systems),
                "legacy_transmission_count": len(transmissions),
            },
            "epistemic_scope": (
                "exact transcription and reference validation of declarations embedded in this expanded URDF; "
                "not proof that plugins are installed, controllers are configured, interfaces can be claimed, "
                "hardware is connected, reductions are calibrated, or commands will execute"
            ),
        }

    def _validate_tree(self) -> None:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(link: str) -> None:
            if link in active:
                raise SpatialError(f"cycle detected at link {link!r}")
            if link in visited:
                return
            active.add(link)
            for joint in self.children[link]:
                visit(joint.child)
            active.remove(link)
            visited.add(link)

        visit(self.root_link)
        missing = sorted(set(self.links) - visited)
        if missing:
            raise SpatialError(f"links are disconnected from root {self.root_link!r}: {missing}")

    def warnings(self) -> list[str]:
        warnings: list[str] = list(self._parse_warnings)
        for link_name, link in self.links.items():
            inertial = link["inertial"]
            if inertial is not None and inertial["validation"]["status"] != "valid":
                warnings.append(
                    f"link {link_name!r} has {inertial['validation']['status']} inertial declaration: "
                    f"{inertial['validation']['issues']}"
                )
        for joint in self.joints.values():
            if joint.type == "revolute" and (joint.limit["lower"] is None or joint.limit["upper"] is None):
                warnings.append(f"revolute joint {joint.name!r} has incomplete position limits")
            if joint.type in {"revolute", "continuous", "prismatic"} and joint.limit["velocity"] is None:
                warnings.append(f"movable joint {joint.name!r} has no velocity limit")
        return warnings

    def resolve_pose(self, supplied: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(supplied) - set(self.joints))
        if unknown:
            raise SpatialError(f"pose contains unknown joints: {unknown}")
        resolved: dict[str, float] = {}
        active: set[str] = set()

        def resolve(name: str) -> float:
            if name in resolved:
                return resolved[name]
            if name in active:
                raise SpatialError(f"mimic cycle detected at joint {name!r}")
            active.add(name)
            joint = self.joints[name]
            if joint.type == "fixed":
                value = 0.0
            elif joint.mimic:
                source = str(joint.mimic["joint"])
                value = float(joint.mimic["multiplier"]) * resolve(source) + float(joint.mimic["offset"])
            else:
                value = float(supplied.get(name, 0.0))
            if not math.isfinite(value):
                raise SpatialError(f"joint {name!r} pose must be finite")
            if joint.type in {"revolute", "prismatic"}:
                lower, upper = joint.limit["lower"], joint.limit["upper"]
                if lower is not None and value < lower - EPSILON:
                    raise SpatialError(f"joint {name!r} value {value} is below lower limit {lower}")
                if upper is not None and value > upper + EPSILON:
                    raise SpatialError(f"joint {name!r} value {value} is above upper limit {upper}")
            active.remove(name)
            resolved[name] = value
            return value

        for name in self.joints:
            resolve(name)
        return resolved

    def world_frames(self, supplied_pose: dict[str, float]) -> tuple[dict[str, Matrix], dict[str, float]]:
        pose = self.resolve_pose(supplied_pose)
        frames: dict[str, Matrix] = {self.root_link: identity()}

        def descend(parent_link: str) -> None:
            world_from_parent = frames[parent_link]
            for joint in sorted(self.children[parent_link], key=lambda item: item.name):
                world_from_joint = matmul(world_from_parent, joint.origin)
                frames[f"joint/{joint.name}"] = world_from_joint
                value = pose[joint.name]
                if joint.type in {"revolute", "continuous"}:
                    motion = axis_angle(joint.axis, value)
                elif joint.type == "prismatic":
                    motion = translation([component * value for component in joint.axis])
                else:
                    motion = identity()
                frames[joint.child] = matmul(world_from_joint, motion)
                descend(joint.child)

        descend(self.root_link)
        for link_name, link in self.links.items():
            world_from_link = frames[link_name]
            for key in ("visuals", "collisions"):
                for geometry in link[key]:
                    local = origin_matrix(geometry["origin_xyz_m"], geometry["origin_rpy_rad"])
                    frames[geometry["frame"]] = matmul(world_from_link, local)
            if link["inertial"]:
                inertial = link["inertial"]
                local = origin_matrix(inertial["origin_xyz_m"], inertial["origin_rpy_rad"])
                frames[inertial["frame"]] = matmul(world_from_link, local)
        return frames, pose

    def geometry_analysis(
        self,
        supplied_pose: dict[str, float],
        inspect_meshes: bool = False,
        package_map_path: Path | None = None,
        inspect_mesh_kinds: Iterable[str] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[Vector]], dict[str, Any]]:
        frames, _ = self.world_frames(supplied_pose)
        package_map = read_package_map(package_map_path)
        selected_mesh_kinds = mesh_inspection_kinds(inspect_meshes, inspect_mesh_kinds)
        analysis: dict[str, dict[str, Any]] = {}
        render_points: dict[str, list[Vector]] = {}
        for link_name, link in self.links.items():
            for collection, kind in (("visuals", "visual"), ("collisions", "collision")):
                for declared in link[collection]:
                    frame_name = declared["frame"]
                    inspect_this_mesh = kind in selected_mesh_kinds
                    measured, points = analyze_declared_geometry(
                        declared["geometry"],
                        frames[frame_name],
                        self.path,
                        package_map,
                        inspect_this_mesh,
                    )
                    if declared["geometry"]["type"] == "mesh" and not inspect_this_mesh:
                        measured["reason"] = (
                            f"rerun with --inspect-mesh-kind {kind} and --package-map when package:// URIs are present"
                        )
                    analysis[frame_name] = {
                        "frame": frame_name,
                        "kind": kind,
                        "link": link_name,
                        "declared_origin_xyz_m": clean_vector(declared["origin_xyz_m"]),
                        "declared_origin_rpy_rad": clean_vector(declared["origin_rpy_rad"]),
                        **measured,
                    }
                    if points:
                        render_points[frame_name] = points
        adjacent_links = {frozenset((joint.parent, joint.child)) for joint in self.joints.values()}
        return analysis, render_points, broadphase_overlaps(analysis, adjacent_links)

    def triangle_surfaces(
        self,
        supplied_pose: dict[str, float],
        geometry_analysis: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Build exact posed triangle surfaces without serializing the triangle payload."""
        frames, _ = self.world_frames(supplied_pose)
        declared_by_frame = {
            declared["frame"]: declared
            for link in self.links.values()
            for collection in ("visuals", "collisions")
            for declared in link[collection]
        }
        metadata: dict[str, dict[str, Any]] = {}
        internal: dict[str, dict[str, Any]] = {}

        def face_components(faces: list[tuple[int, int, int]]) -> list[list[int]]:
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

        for frame_name, measured in sorted(geometry_analysis.items()):
            declared = declared_by_frame[frame_name]
            geometry = declared["geometry"]
            geometry_type = geometry["type"]
            base = {
                "frame": frame_name,
                "kind": measured["kind"],
                "link": measured["link"],
                "geometry_type": geometry_type,
            }
            triangles: list[tuple[Vector, Vector, Vector]]
            component_indices: list[list[int]]
            if geometry_type == "box":
                triangles = box_surface(geometry["size_xyz_m"], frames[frame_name])
                component_indices = [list(range(len(triangles)))]
                watertight = True
                winding_consistent = True
                representation = "analytic_box_boundary_triangulated_exactly"
            elif geometry_type == "mesh" and measured["status"] == "measured":
                mesh = load_mesh(Path(measured["source"]["path"]))
                scale_xyz = measured["source"]["declared_scale_xyz"]
                scaled_vertices = [
                    [vertex[axis] * scale_xyz[axis] for axis in range(3)]
                    for vertex in mesh.vertices
                ]
                triangles = transform_triangles(scaled_vertices, mesh.faces, frames[frame_name])
                component_indices = face_components(mesh.faces)
                watertight = bool(measured["topology"]["watertight"])
                winding_consistent = bool(measured["topology"]["winding_consistent"])
                representation = "declared_stl_or_obj_triangles_after_scale_and_pose"
            else:
                if geometry_type == "mesh":
                    reason = measured.get("reason", "mesh was not inspected")
                else:
                    reason = (
                        f"exact triangle surface is not implemented for analytic {geometry_type}; "
                        "no tessellation is substituted because it would be approximate"
                    )
                metadata[frame_name] = {**base, "status": "unsupported", "reason": reason}
                continue
            component_triangles = [[triangles[index] for index in indices] for indices in component_indices]
            component_test_points = [list(component[0][0]) for component in component_triangles]
            solid_complete = watertight and winding_consistent
            metadata[frame_name] = {
                **base,
                "status": "exact",
                "representation": representation,
                "triangle_count": len(triangles),
                "connected_surface_component_count": len(component_triangles),
                "watertight": watertight,
                "winding_consistent": winding_consistent,
                "solid_containment_classification_available": solid_complete,
            }
            internal[frame_name] = {
                "triangles": triangles,
                "bvh": build_bvh(triangles),
                "component_triangles": component_triangles,
                "component_test_points": component_test_points,
                "solid_complete": solid_complete,
            }
        return metadata, internal

    def triangle_surface_distance(
        self,
        geometry_a: str,
        geometry_b: str,
        supplied_pose: dict[str, float],
        package_map_path: Path | None = None,
    ) -> dict[str, Any]:
        if geometry_a == geometry_b:
            raise SpatialError("surface distance requires two different geometry frames")
        frame_semantics = self.frame_semantics()
        for frame_name in (geometry_a, geometry_b):
            if frame_name not in frame_semantics or frame_semantics[frame_name]["type"] not in MESH_GEOMETRY_KINDS:
                raise SpatialError(f"unknown geometry frame {frame_name!r}")
        requested_kinds = {frame_semantics[frame_name]["type"] for frame_name in (geometry_a, geometry_b)}
        analysis, _, _ = self.geometry_analysis(
            supplied_pose,
            package_map_path=package_map_path,
            inspect_mesh_kinds=requested_kinds,
        )
        metadata, surfaces = self.triangle_surfaces(supplied_pose, analysis)
        for frame_name in (geometry_a, geometry_b):
            if frame_name not in surfaces:
                raise SpatialError(f"geometry {frame_name!r} has no exact triangle surface: {metadata[frame_name]['reason']}")
        result = bvh_surface_distance(
            surfaces[geometry_a]["triangles"],
            surfaces[geometry_b]["triangles"],
            surfaces[geometry_a]["bvh"],
            surfaces[geometry_b]["bvh"],
        )
        return {
            "geometry_a": geometry_a,
            "geometry_b": geometry_b,
            "surface_a": metadata[geometry_a],
            "surface_b": metadata[geometry_b],
            "method": "deterministic_bvh_branch_and_bound_over_exact_triangle_pair_distance",
            "distance_m": clean_number(result["distance_m"]),
            "witness_point_a_in_root_m": clean_vector(result["witness_point_left"]),
            "witness_point_b_in_root_m": clean_vector(result["witness_point_right"]),
            "triangle_index_a": result["left_triangle_index"],
            "triangle_index_b": result["right_triangle_index"],
            "node_pairs_visited": result["node_pairs_visited"],
            "triangle_pairs_tested": result["triangle_pairs_tested"],
            "trust": "exact_for_the_reported_triangle_representations_up_to_floating_point_roundoff",
        }

    def collision_surface_analysis(
        self,
        supplied_pose: dict[str, float],
        geometry_analysis: dict[str, dict[str, Any]],
        broadphase: dict[str, Any],
        contact_tolerance_m: float = 1e-9,
        disabled_pairs: dict[frozenset[str], str] | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(contact_tolerance_m) or contact_tolerance_m < 0.0:
            raise SpatialError("contact tolerance must be a finite non-negative number of meters")
        metadata, surfaces = self.triangle_surfaces(supplied_pose, geometry_analysis)
        collision_frames = sorted(
            frame_name for frame_name, record in geometry_analysis.items() if record["kind"] == "collision"
        )
        total_pairs = 0
        measured_pairs = 0
        for index, left_name in enumerate(collision_frames):
            for right_name in collision_frames[index + 1:]:
                left, right = geometry_analysis[left_name], geometry_analysis[right_name]
                if left["link"] == right["link"]:
                    continue
                total_pairs += 1
                if left["status"] == "measured" and right["status"] == "measured":
                    measured_pairs += 1

        adjacent_links = {frozenset((joint.parent, joint.child)) for joint in self.joints.values()}
        overlap_by_pair = {
            frozenset((record["geometry_a"], record["geometry_b"])): record
            for record in broadphase["overlap_pairs"]
        }
        analysis_candidates: list[dict[str, Any]] = []
        for index, left_name in enumerate(collision_frames):
            for right_name in collision_frames[index + 1:]:
                left_record, right_record = geometry_analysis[left_name], geometry_analysis[right_name]
                if left_record["link"] == right_record["link"] or left_record["status"] != "measured" or right_record["status"] != "measured":
                    continue
                pair_key = frozenset((left_name, right_name))
                if pair_key in overlap_by_pair:
                    candidate = {**overlap_by_pair[pair_key], "aabb_separation_m": 0.0, "candidate_source": "aabb_overlap"}
                else:
                    left_bounds = left_record["bounds_in_root_frame_at_pose"]
                    right_bounds = right_record["bounds_in_root_frame_at_pose"]
                    separation = math.sqrt(aabb_distance_squared(
                        left_bounds["min_xyz_m"], left_bounds["max_xyz_m"],
                        right_bounds["min_xyz_m"], right_bounds["max_xyz_m"],
                    ))
                    if separation > contact_tolerance_m + EPSILON:
                        continue
                    candidate = {
                        "geometry_a": left_name,
                        "link_a": left_record["link"],
                        "geometry_b": right_name,
                        "link_b": right_record["link"],
                        "links_are_adjacent": frozenset((left_record["link"], right_record["link"])) in adjacent_links,
                        "intersection_extents_xyz_m": [0.0, 0.0, 0.0],
                        "intersection_volume_m3": 0.0,
                        "aabb_separation_m": clean_number(separation),
                        "candidate_source": "aabb_within_contact_tolerance",
                    }
                disabled_reason = (disabled_pairs or {}).get(frozenset((candidate["link_a"], candidate["link_b"])))
                candidate["disabled_by_srdf"] = disabled_reason is not None
                candidate["srdf_disable_reason"] = disabled_reason
                analysis_candidates.append(candidate)

        pair_results: list[dict[str, Any]] = []
        for candidate in analysis_candidates:
            left_name, right_name = candidate["geometry_a"], candidate["geometry_b"]
            missing = [name for name in (left_name, right_name) if name not in surfaces]
            if missing:
                pair_results.append({
                    **candidate,
                    "status": "indeterminate",
                    "reason": "exact triangle surface unavailable for " + ", ".join(missing),
                })
                continue
            left, right = surfaces[left_name], surfaces[right_name]
            distance = bvh_surface_distance(
                left["triangles"], right["triangles"], left["bvh"], right["bvh"]
            )
            within_tolerance = distance["distance_m"] <= contact_tolerance_m
            containment: list[dict[str, Any]] = []
            containment_complete = left["solid_complete"] and right["solid_complete"]
            if not within_tolerance and containment_complete:
                for left_component, point in enumerate(left["component_test_points"]):
                    for right_component, component_surface in enumerate(right["component_triangles"]):
                        if point_inside_closed_surface(point, component_surface):
                            containment.append({
                                "contained_geometry": left_name,
                                "contained_component": left_component,
                                "container_geometry": right_name,
                                "container_component": right_component,
                                "test_point_in_root_m": clean_vector(point),
                            })
                            break
                for right_component, point in enumerate(right["component_test_points"]):
                    for left_component, component_surface in enumerate(left["component_triangles"]):
                        if point_inside_closed_surface(point, component_surface):
                            containment.append({
                                "contained_geometry": right_name,
                                "contained_component": right_component,
                                "container_geometry": left_name,
                                "container_component": left_component,
                                "test_point_in_root_m": clean_vector(point),
                            })
                            break
            collision_detected = within_tolerance or bool(containment)
            if collision_detected:
                status = "collision"
            elif containment_complete:
                status = "collision_free"
            else:
                status = "indeterminate"
            pair_results.append({
                **candidate,
                "status": status,
                "surface_distance_m": clean_number(distance["distance_m"]),
                "within_contact_tolerance": within_tolerance,
                "contact_tolerance_m": contact_tolerance_m,
                "containment_classification_complete": containment_complete,
                "containment": containment,
                "witness_point_a_in_root_m": clean_vector(distance["witness_point_left"]),
                "witness_point_b_in_root_m": clean_vector(distance["witness_point_right"]),
                "triangle_index_a": distance["left_triangle_index"],
                "triangle_index_b": distance["right_triangle_index"],
                "node_pairs_visited": distance["node_pairs_visited"],
                "triangle_pairs_tested": distance["triangle_pairs_tested"],
            })

        unresolved = [record for record in pair_results if record["status"] == "indeterminate"]
        collisions = [record for record in pair_results if record["status"] == "collision"]
        if collisions:
            overall_status = "collision"
        elif broadphase["complete_for_declared_collision_geometry"] and not unresolved:
            overall_status = "collision_free"
        else:
            overall_status = "indeterminate"
        srdf_policy_provided = disabled_pairs is not None
        policy_candidates = [record for record in pair_results if not record["disabled_by_srdf"]]
        policy_collisions = [record for record in policy_candidates if record["status"] == "collision"]
        policy_unresolved = [record for record in policy_candidates if record["status"] == "indeterminate"]
        if not srdf_policy_provided:
            policy_status = "not_provided"
        elif policy_collisions:
            policy_status = "collision"
        elif broadphase["complete_for_declared_collision_geometry"] and not policy_unresolved:
            policy_status = "collision_free"
        else:
            policy_status = "indeterminate"
        collision_metadata = {name: metadata[name] for name in collision_frames}
        return {
            "method": "exact_aabb_rejection_then_deterministic_triangle_bvh_distance_and_closed_surface_containment",
            "trust": "exact_for_reported_triangle_representations_up_to_floating_point_roundoff; contact uses the explicit tolerance",
            "pose_dependent": True,
            "same_link_pairs_excluded": True,
            "srdf_policy_is_annotation_not_physical_geometry": True,
            "srdf_policy_provided": srdf_policy_provided,
            "contact_tolerance_m": contact_tolerance_m,
            "self_collision_status": overall_status,
            "srdf_policy_filtered_self_collision_status": policy_status,
            "declared_distinct_link_geometry_pair_count": total_pairs,
            "aabb_separated_measured_pair_count": measured_pairs - len(broadphase["overlap_pairs"]),
            "aabb_separated_beyond_contact_tolerance_measured_pair_count": measured_pairs - len(analysis_candidates),
            "aabb_unknown_pair_count": total_pairs - measured_pairs,
            "aabb_overlap_candidate_count": len(broadphase["overlap_pairs"]),
            "aabb_within_contact_tolerance_candidate_count": len(analysis_candidates) - len(broadphase["overlap_pairs"]),
            "triangle_candidate_count": len(analysis_candidates),
            "exact_candidate_count": len(pair_results) - len(unresolved),
            "indeterminate_candidate_count": len(unresolved),
            "collision_pair_count": len(collisions),
            "srdf_policy_filtered_candidate_count": len(policy_candidates) if srdf_policy_provided else None,
            "srdf_policy_filtered_indeterminate_candidate_count": len(policy_unresolved) if srdf_policy_provided else None,
            "srdf_policy_filtered_collision_pair_count": len(policy_collisions) if srdf_policy_provided else None,
            "srdf_disabled_physical_collision_pair_count": (
                sum(record["status"] == "collision" and record["disabled_by_srdf"] for record in pair_results)
                if srdf_policy_provided
                else None
            ),
            "complete_for_aabb_overlap_candidates": not unresolved,
            "all_declared_collision_geometries_have_exact_triangle_surfaces": all(
                metadata[name]["status"] == "exact" for name in collision_frames
            ),
            "geometry_surfaces": collision_metadata,
            "candidate_results": pair_results,
        }

    def frame_semantics(self) -> dict[str, dict[str, str | None]]:
        frames: dict[str, dict[str, str | None]] = {
            self.root_link: {"type": "link", "parent_frame": None, "owner": self.root_link}
        }
        for joint in self.joints.values():
            frames[f"joint/{joint.name}"] = {"type": "joint_pre_motion", "parent_frame": joint.parent, "owner": joint.name}
            frames[joint.child] = {"type": "link", "parent_frame": f"joint/{joint.name}", "owner": joint.child}
        for link_name, link in self.links.items():
            for key, frame_type in (("visuals", "visual"), ("collisions", "collision")):
                for geometry in link[key]:
                    frames[geometry["frame"]] = {"type": frame_type, "parent_frame": link_name, "owner": link_name}
            if link["inertial"]:
                frames[link["inertial"]["frame"]] = {"type": "inertial", "parent_frame": link_name, "owner": link_name}
        return frames

    def transform(self, reference: str, target: str, supplied_pose: dict[str, float]) -> Matrix:
        frames, _ = self.world_frames(supplied_pose)
        for frame in (reference, target):
            if frame not in frames:
                raise SpatialError(f"unknown frame {frame!r}; run tree or export to list frames")
        return matmul(inverse_rigid(frames[reference]), frames[target])

    def axis(self, joint_name: str, reference: str, supplied_pose: dict[str, float]) -> Vector:
        if joint_name not in self.joints:
            raise SpatialError(f"unknown joint {joint_name!r}")
        joint = self.joints[joint_name]
        if joint.type == "fixed":
            raise SpatialError(f"fixed joint {joint_name!r} has no motion axis")
        frames, _ = self.world_frames(supplied_pose)
        if reference not in frames:
            raise SpatialError(f"unknown reference frame {reference!r}")
        world_axis = rotate_vector(frames[f"joint/{joint_name}"], joint.axis)
        reference_axis = rotate_vector(inverse_rigid(frames[reference]), world_axis)
        return clean_vector(normalized(reference_axis, f"joint {joint_name} axis"))

    def mass_properties(
        self,
        supplied_pose: dict[str, float],
        expressed_in_frame: str | None = None,
        subtree_root: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate declared URDF inertials with FK and the parallel-axis theorem."""
        frames, pose = self.world_frames(supplied_pose)
        reference = expressed_in_frame or self.root_link
        if reference not in frames:
            raise SpatialError(f"unknown reference frame {reference!r}")
        selected_root = subtree_root or self.root_link
        if selected_root not in self.links:
            raise SpatialError(f"unknown subtree root link {selected_root!r}")
        selected: set[str] = set()

        def descend(link_name: str) -> None:
            selected.add(link_name)
            for child_joint in self.children[link_name]:
                descend(child_joint.child)

        descend(selected_root)
        selected_links = sorted(selected)
        missing_links = [name for name in selected_links if self.links[name]["inertial"] is None]
        invalid_links = [
            name
            for name in selected_links
            if self.links[name]["inertial"] is not None
            and self.links[name]["inertial"]["validation"]["status"] != "valid"
        ]
        reference_from_world = inverse_rigid(frames[reference])
        contributions: list[dict[str, Any]] = []
        for link_name in selected_links:
            inertial = self.links[link_name]["inertial"]
            if inertial is None or inertial["validation"]["status"] != "valid":
                continue
            mass = float(inertial["mass_kg"])
            reference_from_inertial = matmul(reference_from_world, frames[inertial["frame"]])
            center = [reference_from_inertial[index][3] for index in range(3)]
            tensor = tensor3_from_urdf(inertial["inertia_kg_m2"])
            expressed_tensor = rotate_tensor3(reference_from_inertial, tensor)
            contributions.append({
                "link": link_name,
                "inertial_frame": inertial["frame"],
                "declared_mass_kg": clean_number(mass),
                "center_of_mass_in_expressed_frame_m": clean_vector(center),
                "inertia_about_own_center_of_mass_in_expressed_frame_kg_m2": tensor3_record(expressed_tensor),
                "declared_inertial_origin_in_link": {
                    "translation_xyz_m": clean_vector(inertial["origin_xyz_m"]),
                    "rpy_rad": clean_vector(inertial["origin_rpy_rad"]),
                },
                "declared_principal_moments_kg_m2": inertial["validation"]["principal_moments_kg_m2"],
            })
        coverage = {
            "selected_link_count": len(selected_links),
            "declared_inertial_link_count": len(contributions) + len(invalid_links),
            "valid_inertial_link_count": len(contributions),
            "missing_inertial_links": missing_links,
            "invalid_or_incomplete_inertial_links": invalid_links,
            "all_selected_links_declare_valid_inertial": not missing_links and not invalid_links,
            "physical_world_completeness": "not_established",
            "absence_of_inertial_is_not_proof_of_zero_physical_mass": True,
        }
        result: dict[str, Any] = {
            "schema_version": "robot-spatial-mass-properties.v1",
            "selection": {
                "type": "whole_tree" if selected_root == self.root_link else "subtree",
                "subtree_root_link": selected_root,
                "selected_links": selected_links,
            },
            "pose": {"joint_positions": {name: clean_number(value) for name, value in pose.items()}},
            "expressed_in_frame": reference,
            "coverage": coverage,
            "per_link_declared_inertials": contributions,
            "method": "URDF-declared inertial origins and tensors transformed by forward kinematics, then combined with the parallel-axis theorem",
            "epistemic_scope": "exact for the declared inertial model, selected tree, stated pose, and supported transforms; not proof of physical mass, payload, calibration, or unmodeled components",
        }
        if invalid_links:
            result.update({
                "status": "indeterminate",
                "reason": "one or more selected inertial declarations are invalid or incomplete",
                "declared_mass_kg": None,
                "center_of_mass_in_expressed_frame_m": None,
                "inertia_about_center_of_mass_in_expressed_frame_kg_m2": None,
            })
            return result
        if not contributions:
            result.update({
                "status": "not_provided",
                "reason": "no selected link declares inertial properties",
                "declared_mass_kg": None,
                "center_of_mass_in_expressed_frame_m": None,
                "inertia_about_center_of_mass_in_expressed_frame_kg_m2": None,
            })
            return result
        total_mass = sum(record["declared_mass_kg"] for record in contributions)
        center = [
            sum(record["declared_mass_kg"] * record["center_of_mass_in_expressed_frame_m"][axis] for record in contributions) / total_mass
            for axis in range(3)
        ]
        total_inertia: Tensor3 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for record in contributions:
            own_tensor = record["inertia_about_own_center_of_mass_in_expressed_frame_kg_m2"]["matrix_3x3_rowmajor"]
            displacement = [record["center_of_mass_in_expressed_frame_m"][axis] - center[axis] for axis in range(3)]
            total_inertia = add_tensor3(
                total_inertia,
                add_tensor3(own_tensor, parallel_axis_tensor(record["declared_mass_kg"], displacement)),
            )
        result.update({
            "status": "computed",
            "declared_mass_kg": clean_number(total_mass),
            "center_of_mass_in_expressed_frame_m": clean_vector(center),
            "inertia_about_center_of_mass_in_expressed_frame_kg_m2": tensor3_record(total_inertia),
            "aggregate_principal_moments_kg_m2": symmetric_eigenvalues_3x3(total_inertia),
        })
        return result

    def static_gravity_loads(
        self,
        supplied_pose: dict[str, float],
        gravity_vector: Vector | None = None,
        gravity_frame: str | None = None,
        subtree_root: str | None = None,
    ) -> dict[str, Any]:
        """Project modeled gravity forces onto independent URDF joint coordinates."""
        frames, pose = self.world_frames(supplied_pose)
        reference = gravity_frame or self.root_link
        if reference not in frames:
            raise SpatialError(f"unknown gravity frame {reference!r}")
        gravity = list(gravity_vector if gravity_vector is not None else [0.0, 0.0, -9.80665])
        if len(gravity) != 3 or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in gravity):
            raise SpatialError("gravity vector must contain exactly three finite numeric components")
        gravity = [float(value) for value in gravity]
        gravity_root = rotate_vector(frames[reference], gravity)
        selected_root = subtree_root or self.root_link
        if selected_root not in self.links:
            raise SpatialError(f"unknown subtree root link {selected_root!r}")
        selected: set[str] = set()

        def descend(link_name: str) -> None:
            selected.add(link_name)
            for child_joint in self.children[link_name]:
                descend(child_joint.child)

        descend(selected_root)
        selected_links = sorted(selected)
        missing_links = [name for name in selected_links if self.links[name]["inertial"] is None]
        invalid_links = [
            name
            for name in selected_links
            if self.links[name]["inertial"] is not None
            and self.links[name]["inertial"]["validation"]["status"] != "valid"
        ]

        path_cache: dict[str, list[Joint]] = {}

        def root_path(link_name: str) -> list[Joint]:
            if link_name in path_cache:
                return path_cache[link_name]
            reversed_path: list[Joint] = []
            cursor = link_name
            while cursor != self.root_link:
                joint = self.child_joint[cursor]
                reversed_path.append(joint)
                cursor = joint.parent
            path_cache[link_name] = list(reversed(reversed_path))
            return path_cache[link_name]

        driver_set: set[str] = set()
        for link_name in selected_links:
            for physical_joint in root_path(link_name):
                if physical_joint.type != "fixed":
                    driver_set.add(self._independent_mimic_driver(physical_joint.name)[0])

        driver_order: list[str] = []

        def collect_drivers(parent_link: str) -> None:
            for physical_joint in sorted(self.children[parent_link], key=lambda item: item.name):
                if physical_joint.type != "fixed":
                    driver = self._independent_mimic_driver(physical_joint.name)[0]
                    if driver in driver_set and driver not in driver_order:
                        driver_order.append(driver)
                collect_drivers(physical_joint.child)

        collect_drivers(self.root_link)
        aggregates: dict[str, dict[str, Any]] = {
            driver: {"generalized_gravity_force": 0.0, "physical_contributions": []}
            for driver in driver_order
        }
        per_link: list[dict[str, Any]] = []
        modeled_potential_energy = 0.0
        for link_name in selected_links:
            inertial = self.links[link_name]["inertial"]
            if inertial is None or inertial["validation"]["status"] != "valid":
                continue
            mass = float(inertial["mass_kg"])
            center_root = [frames[inertial["frame"]][axis][3] for axis in range(3)]
            force_root = [mass * component for component in gravity_root]
            potential = -mass * sum(gravity_root[axis] * center_root[axis] for axis in range(3))
            modeled_potential_energy += potential
            joint_contributions: list[dict[str, Any]] = []
            for physical_joint in root_path(link_name):
                if physical_joint.type == "fixed":
                    continue
                joint_frame = frames[f"joint/{physical_joint.name}"]
                axis_root = normalized(
                    rotate_vector(joint_frame, physical_joint.axis),
                    f"joint {physical_joint.name} axis",
                )
                joint_origin = [joint_frame[axis][3] for axis in range(3)]
                if physical_joint.type in {"revolute", "continuous"}:
                    lever = [center_root[axis] - joint_origin[axis] for axis in range(3)]
                    moment = cross_product(lever, force_root)
                    physical_force = sum(axis_root[axis] * moment[axis] for axis in range(3))
                    unit = "N*m"
                else:
                    physical_force = sum(axis_root[axis] * force_root[axis] for axis in range(3))
                    unit = "N"
                driver, derivative, affine_offset, mimic_chain = self._mimic_affine_from_driver(physical_joint.name)
                driver_contribution = derivative * physical_force
                aggregates[driver]["generalized_gravity_force"] += driver_contribution
                contribution = {
                    "link": link_name,
                    "physical_joint": physical_joint.name,
                    "independent_driver_joint": driver,
                    "physical_joint_generalized_gravity_force": clean_number(physical_force),
                    "physical_joint_unit": unit,
                    "derivative_of_physical_joint_from_driver": clean_number(derivative),
                    "position_offset_from_driver": clean_number(affine_offset),
                    "mimic_chain": mimic_chain,
                    "contribution_to_independent_driver": clean_number(driver_contribution),
                }
                aggregates[driver]["physical_contributions"].append(contribution)
                joint_contributions.append(contribution)
            per_link.append({
                "link": link_name,
                "inertial_frame": inertial["frame"],
                "declared_mass_kg": clean_number(mass),
                "center_of_mass_in_root_frame_m": clean_vector(center_root),
                "modeled_gravity_force_in_root_frame_n": clean_vector(force_root),
                "modeled_potential_energy_relative_to_root_origin_j": clean_number(potential),
                "joint_contributions": joint_contributions,
            })

        coverage = {
            "selected_link_count": len(selected_links),
            "declared_inertial_link_count": len(per_link) + len(invalid_links),
            "valid_inertial_link_count": len(per_link),
            "missing_inertial_links": missing_links,
            "invalid_or_incomplete_inertial_links": invalid_links,
            "all_selected_links_declare_valid_inertial": not missing_links and not invalid_links,
            "physical_world_completeness": "not_established",
            "absence_of_inertial_is_not_proof_of_zero_physical_mass_or_zero_gravity_load": True,
        }
        result: dict[str, Any] = {
            "schema_version": "robot-spatial-static-gravity-loads.v1",
            "selection": {
                "type": "whole_tree" if selected_root == self.root_link else "subtree_including_loads_transmitted_to_upstream_joints",
                "subtree_root_link": selected_root,
                "selected_links": selected_links,
            },
            "pose": {"joint_positions": {name: clean_number(value) for name, value in pose.items()}},
            "gravity": {
                "vector_xyz_m_s2": clean_vector(gravity),
                "expressed_in_frame": reference,
                "vector_in_root_frame_xyz_m_s2": clean_vector(gravity_root),
                "magnitude_m_s2": clean_number(math.sqrt(sum(value * value for value in gravity))),
            },
            "independent_driver_order": driver_order,
            "coverage": coverage,
            "per_link_modeled_contributions": per_link,
            "method": (
                "for each valid URDF inertial, apply F=m*g at its FK-resolved center of mass; "
                "project force/moment onto every ancestor joint axis; combine mimic followers by dq_follower/dq_driver"
            ),
            "sign_convention": {
                "generalized_gravity_force": "force or torque exerted by the modeled gravity field along positive independent joint motion",
                "ideal_static_holding_effort": "equal and opposite generalized effort required for static equilibrium in this gravity-only model",
            },
            "epistemic_scope": (
                "exact for the selected valid URDF inertials, supported tree/mimic kinematics, stated pose, and explicit gravity vector; "
                "not a full inverse-dynamics result and not proof of actual mounting orientation, payload, contacts, friction, motor/transmission behavior, controller capability, or hardware feasibility"
            ),
        }
        if invalid_links:
            result.update({
                "status": "indeterminate",
                "reason": "one or more selected inertial declarations are invalid or incomplete",
                "modeled_potential_energy_relative_to_root_origin_j": None,
                "independent_driver_loads": None,
            })
            return result
        if not per_link:
            result.update({
                "status": "not_provided",
                "reason": "no selected link declares valid inertial properties",
                "modeled_potential_energy_relative_to_root_origin_j": None,
                "independent_driver_loads": None,
            })
            return result
        driver_loads: dict[str, Any] = {}
        for driver in driver_order:
            joint = self.joints[driver]
            generalized_force = aggregates[driver]["generalized_gravity_force"]
            unit = "N" if joint.type == "prismatic" else "N*m"
            effort_limit = joint.limit.get("effort")
            driver_loads[driver] = {
                "joint_type": joint.type,
                "unit": unit,
                "generalized_gravity_force": clean_number(generalized_force),
                "ideal_static_holding_effort": clean_number(-generalized_force),
                "declared_joint_effort_limit_magnitude": effort_limit,
                "modeled_load_within_declared_joint_effort_limit_magnitude": (
                    None if effort_limit is None else abs(generalized_force) <= abs(float(effort_limit)) + EPSILON
                ),
                "physical_contributions": aggregates[driver]["physical_contributions"],
            }
        result.update({
            "status": "computed",
            "modeled_potential_energy_relative_to_root_origin_j": clean_number(modeled_potential_energy),
            "independent_driver_loads": driver_loads,
        })
        return result

    def canonical(
        self,
        supplied_pose: dict[str, float],
        pose_name: str,
        semantics: dict[str, Any] | None = None,
        inspect_meshes: bool = False,
        package_map_path: Path | None = None,
        srdf: dict[str, Any] | None = None,
        workspace_samples: int = 256,
        include_workspace_samples: bool = False,
        surface_collisions: bool = False,
        contact_tolerance_m: float = 1e-9,
        inspect_mesh_kinds: Iterable[str] | None = None,
        world_scene: WorldScene | None = None,
    ) -> dict[str, Any]:
        frames, pose = self.world_frames(supplied_pose)
        frame_semantics = self.frame_semantics()
        selected_mesh_kinds = mesh_inspection_kinds(inspect_meshes, inspect_mesh_kinds)
        if surface_collisions:
            selected_mesh_kinds.add("collision")
        geometry_analysis, _, broadphase = self.geometry_analysis(
            supplied_pose,
            package_map_path=package_map_path,
            inspect_mesh_kinds=selected_mesh_kinds,
        )
        mesh_records = [record for record in geometry_analysis.values() if record["geometry_type"] == "mesh"]
        measured_meshes = [record for record in mesh_records if record["status"] == "measured"]
        mesh_counts_by_kind = {
            kind: {
                "requested": kind in selected_mesh_kinds,
                "declared_mesh_count": sum(record["kind"] == kind for record in mesh_records),
                "measured_mesh_count": sum(record["kind"] == kind and record["status"] == "measured" for record in mesh_records),
            }
            for kind in sorted(MESH_GEOMETRY_KINDS)
        }
        for counts in mesh_counts_by_kind.values():
            counts["complete"] = counts["declared_mesh_count"] == counts["measured_mesh_count"]
        joints: dict[str, Any] = {}
        for name, joint in self.joints.items():
            joints[name] = {
                "type": joint.type,
                "parent_link": joint.parent,
                "child_link": joint.child,
                "pre_motion_frame": f"joint/{name}",
                "origin_xyz_m": clean_vector(joint.origin_xyz),
                "origin_rpy_rad": clean_vector(joint.origin_rpy),
                "axis_in_pre_motion_frame": clean_vector(joint.axis) if joint.type != "fixed" else None,
                "axis_in_root_frame_at_pose": self.axis(name, self.root_link, supplied_pose) if joint.type != "fixed" else None,
                "position_at_pose": clean_number(pose[name]),
                "position_unit": "m" if joint.type == "prismatic" else ("rad" if joint.type != "fixed" else None),
                "limits": joint.limit,
                "mimic": joint.mimic,
                "dynamics": joint.dynamics,
                "actuation_declarations": self.actuation["joint_bindings"][name],
            }
        srdf_record = srdf or {
            "status": "not_provided",
            "groups": {},
            "named_poses": {},
            "end_effectors": {},
            "passive_joints": [],
            "virtual_joints": {},
            "disabled_collisions": [],
        }
        disabled_pairs = {
            frozenset((record["link1"], record["link2"])): record["reason"]
            for record in srdf_record["disabled_collisions"]
        }
        for pair in broadphase["overlap_pairs"]:
            disabled_reason = disabled_pairs.get(frozenset((pair["link_a"], pair["link_b"])))
            pair["disabled_by_srdf"] = disabled_reason is not None
            pair["srdf_disable_reason"] = disabled_reason
        collision_surface = (
            self.collision_surface_analysis(
                supplied_pose,
                geometry_analysis,
                broadphase,
                contact_tolerance_m,
                disabled_pairs if srdf is not None else None,
            )
            if surface_collisions
            else {
                "status": "not_requested",
                "meaning": "rerun export with --surface-collisions for triangle-level distance/contact and containment analysis",
            }
        )
        target_reasons: dict[str, list[str]] = {}

        def add_target(frame_name: str | None, reason: str) -> None:
            if frame_name is not None:
                target_reasons.setdefault(frame_name, []).append(reason)

        semantic_record = semantics or {"status": "not_provided", "frames": {}, "groups": {}, "end_effectors": {}}
        for frame_name, annotation in semantic_record["frames"].items():
            if "tcp" in annotation["roles"]:
                add_target(frame_name, "asserted_tcp_role")
        for group_name, group in semantic_record["groups"].items():
            add_target(group["tip_frame"], f"semantic_group_tip:{group_name}")
        for end_effector_name, end_effector in semantic_record["end_effectors"].items():
            add_target(end_effector["tcp_frame"], f"semantic_end_effector_tcp:{end_effector_name}")
        for group_name, group in srdf_record["groups"].items():
            for chain_record in group["chains"]:
                add_target(chain_record["tip_link"], f"srdf_group_chain_tip:{group_name}")
        for end_effector_name, end_effector in srdf_record["end_effectors"].items():
            add_target(end_effector["parent_link"], f"srdf_end_effector_parent:{end_effector_name}")
        if workspace_samples < 0 or workspace_samples > 100000:
            raise SpatialError("workspace_samples must be between 0 and 100000")
        target_analysis: dict[str, Any] = {}
        for target_frame in sorted(target_reasons):
            record: dict[str, Any] = {
                "requested_by": sorted(set(target_reasons[target_frame])),
                "geometric_jacobian": self.geometric_jacobian(target_frame, supplied_pose),
            }
            if workspace_samples:
                record["sampled_workspace"] = self.workspace_envelope(target_frame, supplied_pose, workspace_samples, include_workspace_samples)
            target_analysis[target_frame] = record
        declared_mass_properties = self.mass_properties(supplied_pose, self.root_link)
        declared_static_gravity_loads = self.static_gravity_loads(
            supplied_pose,
            [0.0, 0.0, -9.80665],
            self.root_link,
        )
        bound_world_scene = (
            world_scene.canonical(
                self,
                supplied_pose,
                package_map_path,
                contact_tolerance_m,
            )
            if world_scene is not None
            else {
                "status": "not_provided",
                "meaning": "provide --scene with a robot-spatial-world-scene.v1 static snapshot to bind the URDF root, gravity, and environment objects to an external world",
            }
        )
        scene_gravity_loads = (
            scene_gravity_load_analysis(self, world_scene, supplied_pose, pose_name)
            if world_scene is not None
            else {
                "schema_version": "robot-spatial-scene-gravity-loads.v1",
                "status": "not_provided",
                "reason": "no world scene was provided",
                "loads": None,
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "source": {"urdf": str(self.path), "sha256": self.sha256, "semantic_sha256": self.semantic_sha256},
            "units": {"length": "m", "angle": "rad", "quaternion": "xyzw"},
            "robot": {"name": self.name, "root_link": self.root_link, "is_tree": True},
            "pose": {"name": pose_name, "joint_positions": {name: clean_number(value) for name, value in pose.items()}},
            "semantics": semantic_record,
            "srdf": srdf_record,
            "links": self.links,
            "joints": joints,
            "actuation": self.actuation,
            "world_scene": bound_world_scene,
            "geometry_analysis": geometry_analysis,
            "kinematic_analysis": {"targets": target_analysis},
            "physical_analysis": {
                "declared_mass_properties": declared_mass_properties,
                "declared_static_gravity_loads_under_root_frame_convention": declared_static_gravity_loads,
                "declared_static_gravity_loads_under_scene_gravity": scene_gravity_loads,
            },
            "collision_broadphase": broadphase,
            "collision_surface": collision_surface,
            "frames": {
                name: {**frame_semantics[name], "world_from_frame": pose_record(transform)}
                for name, transform in sorted(frames.items())
            },
            "validation": {"status": "valid_with_warnings" if self.warnings() else "valid", "warnings": self.warnings()},
            "capabilities": {
                "kinematic_tree": True,
                "forward_kinematics": True,
                "analytic_geometric_jacobian": True,
                "deterministic_sampled_workspace": True,
                "grounded_evaluation_generation": True,
                "declared_geometry_placement": True,
                "declared_mass_properties": {
                    "supported": True,
                    "status": declared_mass_properties["status"],
                    "declared_inertial_link_count": declared_mass_properties["coverage"]["declared_inertial_link_count"],
                    "valid_inertial_link_count": declared_mass_properties["coverage"]["valid_inertial_link_count"],
                    "all_selected_links_declare_valid_inertial": declared_mass_properties["coverage"]["all_selected_links_declare_valid_inertial"],
                    "physical_world_completeness": "not_established",
                },
                "declared_static_gravity_loads": {
                    "supported": True,
                    "status": declared_static_gravity_loads["status"],
                    "gravity_convention": "[0, 0, -9.80665] m/s^2 expressed in the URDF root frame",
                    "physical_world_completeness": "not_established",
                },
                "actuation_declarations": {
                    "supported": True,
                    "ros2_control_system_count": self.actuation["coverage"]["ros2_control_system_count"],
                    "legacy_transmission_count": self.actuation["coverage"]["legacy_transmission_count"],
                    "runtime_or_hardware_capability_established": False,
                },
                "world_scene": {
                    "provided": world_scene is not None,
                    "validated_static_snapshot": world_scene is not None,
                    "root_mount_explicit": world_scene is not None,
                    "gravity_explicit": world_scene is not None and world_scene.gravity is not None,
                    "robot_environment_collision_status": (
                        bound_world_scene["robot_environment_collision"]["status"]
                        if world_scene is not None
                        else "not_provided"
                    ),
                    "physical_world_completeness": "not_established",
                },
                "mesh_content_inspection": {
                    "requested": bool(selected_mesh_kinds),
                    "requested_kinds": sorted(selected_mesh_kinds),
                    "declared_mesh_count": len(mesh_records),
                    "measured_mesh_count": len(measured_meshes),
                    "complete": len(mesh_records) == len(measured_meshes),
                    "complete_for_all_declared_meshes": len(mesh_records) == len(measured_meshes),
                    "complete_for_requested_kinds": all(
                        mesh_counts_by_kind[kind]["complete"] for kind in selected_mesh_kinds
                    ),
                    "by_kind": mesh_counts_by_kind,
                    "supported_formats": ["stl", "obj"],
                },
                "collision_detection": surface_collisions,
                "triangle_surface_distance": {
                    "requested": surface_collisions,
                    "supported_representations": ["stl", "obj", "box"],
                    "analytic_cylinder_and_sphere": "not_supported_without_approximate_tessellation",
                },
                "collision_aabb_broadphase": broadphase["complete_for_declared_collision_geometry"],
                "srdf_semantics": srdf is not None,
                "closed_loops": False,
            },
        }

    def tree_lines(self) -> list[str]:
        lines = [self.root_link]

        def descend(parent: str, prefix: str) -> None:
            children = sorted(self.children[parent], key=lambda item: item.name)
            for index, joint in enumerate(children):
                last = index == len(children) - 1
                connector = "└─" if last else "├─"
                axis = "" if joint.type == "fixed" else f" axis={clean_vector(joint.axis)}"
                lines.append(f"{prefix}{connector} {joint.name} [{joint.type}{axis}] → {joint.child}")
                descend(joint.child, prefix + ("   " if last else "│  "))

        descend(self.root_link, "")
        return lines

    def chain(self, start_link: str, end_link: str) -> dict[str, Any]:
        for link in (start_link, end_link):
            if link not in self.links:
                raise SpatialError(f"unknown link {link!r}")
        adjacency: dict[str, list[tuple[str, Joint, str]]] = {link: [] for link in self.links}
        for joint in self.joints.values():
            adjacency[joint.parent].append((joint.child, joint, "parent_to_child"))
            adjacency[joint.child].append((joint.parent, joint, "child_to_parent"))
        queue = [start_link]
        previous: dict[str, tuple[str, Joint, str] | None] = {start_link: None}
        for current in queue:
            if current == end_link:
                break
            for neighbor, joint, direction in sorted(adjacency[current], key=lambda item: item[1].name):
                if neighbor not in previous:
                    previous[neighbor] = (current, joint, direction)
                    queue.append(neighbor)
        if end_link not in previous:
            raise SpatialError(f"no kinematic path from {start_link!r} to {end_link!r}")
        reversed_steps: list[dict[str, Any]] = []
        current = end_link
        while current != start_link:
            parent_record = previous[current]
            assert parent_record is not None
            prior, joint, direction = parent_record
            reversed_steps.append({
                "from_link": prior,
                "to_link": current,
                "joint": joint.name,
                "joint_type": joint.type,
                "traversal": direction,
                "joint_axis_in_pre_motion_frame": clean_vector(joint.axis) if joint.type != "fixed" else None,
            })
            current = prior
        steps = list(reversed(reversed_steps))
        links = [start_link] + [step["to_link"] for step in steps]
        movable = [step["joint"] for step in steps if step["joint_type"] != "fixed"]
        return {
            "from_link": start_link,
            "to_link": end_link,
            "links": links,
            "steps": steps,
            "joint_count": len(steps),
            "movable_joint_count": len(movable),
            "movable_joints": movable,
        }

    def affected_by_joint(self, joint_name: str) -> dict[str, Any]:
        if joint_name not in self.joints:
            raise SpatialError(f"unknown joint {joint_name!r}")
        requested_joint = self.joints[joint_name]
        if requested_joint.type == "fixed":
            raise SpatialError(f"fixed joint {joint_name!r} has no position change and therefore no motion effect")
        driver, requested_derivative, mimic_chain = self._independent_mimic_driver(joint_name)
        affected_links: list[str] = []
        affected_set: set[str] = set()

        def descend(link: str) -> None:
            if link in affected_set:
                return
            affected_set.add(link)
            affected_links.append(link)
            for child_joint in sorted(self.children[link], key=lambda item: item.name):
                descend(child_joint.child)

        physical_joints = []
        for candidate_name, candidate in self.joints.items():
            if candidate.type == "fixed":
                continue
            candidate_driver, _, _ = self._independent_mimic_driver(candidate_name)
            if candidate_driver == driver:
                physical_joints.append(candidate_name)
                descend(candidate.child)
        semantics = self.frame_semantics()
        affected_frames = sorted(
            frame_name for frame_name, frame in semantics.items()
            if (
                frame["type"] in {"link", "visual", "collision", "inertial"} and frame["owner"] in affected_set
            ) or (
                frame["type"] == "joint_pre_motion" and self.joints[str(frame["owner"])].parent in affected_set
            )
        )
        downstream_joints = sorted(name for name, candidate in self.joints.items() if candidate.parent in affected_set)
        return {
            "joint": joint_name,
            "joint_type": requested_joint.type,
            "independent_driver_joint": driver,
            "requested_joint_derivative_from_driver": clean_number(requested_derivative),
            "requested_joint_mimic_chain": mimic_chain,
            "physical_joints_driven": sorted(physical_joints),
            "pre_motion_frame": f"joint/{joint_name}",
            "pre_motion_frame_is_affected_by_own_motion": False,
            "affected_links": affected_links,
            "affected_frames": affected_frames,
            "downstream_joints": downstream_joints,
            "meaning": f"frames whose poses relative to the root may change when independent driver {driver!r} changes, assuming other independent joint positions stay fixed",
        }

    def _frame_attachment_link(self, frame_name: str) -> str:
        semantics = self.frame_semantics()
        if frame_name not in semantics:
            raise SpatialError(f"unknown target frame {frame_name!r}")
        frame = semantics[frame_name]
        if frame["type"] == "link":
            return frame_name
        if frame["type"] == "joint_pre_motion":
            return self.joints[str(frame["owner"])].parent
        return str(frame["owner"])

    def _independent_mimic_driver(self, joint_name: str) -> tuple[str, float, list[str]]:
        driver, multiplier, _, chain = self._mimic_affine_from_driver(joint_name)
        return driver, multiplier, chain

    def _mimic_affine_from_driver(self, joint_name: str) -> tuple[str, float, float, list[str]]:
        current = joint_name
        multiplier = 1.0
        offset = 0.0
        chain = [current]
        visited: set[str] = set()
        while self.joints[current].mimic:
            if current in visited:
                raise SpatialError(f"mimic cycle detected at joint {current!r}")
            visited.add(current)
            mimic = self.joints[current].mimic
            assert mimic is not None
            local_multiplier = float(mimic["multiplier"])
            offset = multiplier * float(mimic["offset"]) + offset
            multiplier *= local_multiplier
            current = str(mimic["joint"])
            chain.append(current)
        return current, multiplier, offset, chain

    def _workspace_driver_range(self, driver: str) -> dict[str, Any]:
        driver_joint = self.joints[driver]
        if driver_joint.type == "continuous":
            lower, upper, source = -math.pi, math.pi, "canonical_continuous_cycle"
        else:
            lower, upper = driver_joint.limit["lower"], driver_joint.limit["upper"]
            if lower is None or upper is None:
                raise SpatialError(f"cannot sample workspace: independent joint {driver!r} has incomplete position limits")
            lower, upper, source = float(lower), float(upper), "urdf_position_limits"
        constraints: list[dict[str, Any]] = [{
            "joint": driver,
            "affine_position_from_driver": {"multiplier": 1.0, "offset": 0.0},
            "declared_lower": driver_joint.limit["lower"],
            "declared_upper": driver_joint.limit["upper"],
        }]
        for joint_name, joint in self.joints.items():
            candidate_driver, multiplier, offset, _ = self._mimic_affine_from_driver(joint_name)
            if candidate_driver != driver or joint_name == driver or joint.type == "continuous":
                continue
            declared_lower, declared_upper = joint.limit["lower"], joint.limit["upper"]
            if declared_lower is None and declared_upper is None:
                continue
            constraints.append({
                "joint": joint_name,
                "affine_position_from_driver": {"multiplier": clean_number(multiplier), "offset": clean_number(offset)},
                "declared_lower": declared_lower,
                "declared_upper": declared_upper,
            })
            if abs(multiplier) <= EPSILON:
                if (declared_lower is not None and offset < declared_lower - EPSILON) or (declared_upper is not None and offset > declared_upper + EPSILON):
                    raise SpatialError(f"mimic joint {joint_name!r} has constant position {offset} outside its declared limits")
                continue
            transformed = []
            if declared_lower is not None:
                transformed.append((float(declared_lower) - offset) / multiplier)
            if declared_upper is not None:
                transformed.append((float(declared_upper) - offset) / multiplier)
            if len(transformed) == 2:
                constraint_lower, constraint_upper = min(transformed), max(transformed)
            elif declared_lower is not None:
                boundary = transformed[0]
                constraint_lower, constraint_upper = (boundary, math.inf) if multiplier > 0.0 else (-math.inf, boundary)
            else:
                boundary = transformed[0]
                constraint_lower, constraint_upper = (-math.inf, boundary) if multiplier > 0.0 else (boundary, math.inf)
            lower, upper = max(lower, constraint_lower), min(upper, constraint_upper)
        if lower > upper + EPSILON:
            raise SpatialError(f"independent driver {driver!r} has no feasible range after mimic joint limits")
        return {
            "joint": driver,
            "joint_type": driver_joint.type,
            "minimum": clean_number(lower),
            "maximum": clean_number(upper),
            "unit": "m" if driver_joint.type == "prismatic" else "rad",
            "range_source": source if len(constraints) == 1 else f"{source}_intersected_with_mimic_limits",
            "constraints": constraints,
        }

    def motion_driver_counterfactuals(
        self,
        driver: str,
        supplied_pose: dict[str, float],
        angular_step_rad: float = 0.1,
        linear_step_m: float = 0.01,
    ) -> dict[str, Any]:
        """Return feasible signed endpoint perturbations for one independent driver."""
        if driver not in self.joints:
            raise SpatialError(f"unknown joint {driver!r}")
        driver_joint = self.joints[driver]
        if driver_joint.type == "fixed":
            raise SpatialError(f"fixed joint {driver!r} has no counterfactual motion")
        if driver_joint.mimic is not None:
            independent, _, _ = self._independent_mimic_driver(driver)
            raise SpatialError(f"joint {driver!r} is mimic-driven; perturb independent driver {independent!r}")
        for label, value in (("angular_step_rad", angular_step_rad), ("linear_step_m", linear_step_m)):
            if not math.isfinite(value) or value <= 0.0:
                raise SpatialError(f"{label} must be finite and positive")
        contract = self.independent_driver_contract(driver)
        resolved = self.resolve_pose(supplied_pose)
        baseline = resolved[driver]
        domain = contract["feasible_domain"]
        lower = -math.inf if domain["minimum"] is None else float(domain["minimum"])
        upper = math.inf if domain["maximum"] is None else float(domain["maximum"])
        constraints = domain["constraints"]
        physical_joints = contract["physical_joints_driven"]
        unit = contract["unit"]
        if baseline < lower - EPSILON or baseline > upper + EPSILON:
            raise SpatialError(
                f"baseline position {baseline} for independent driver {driver!r} is outside its mimic-constrained interval"
            )
        nominal_step = linear_step_m if driver_joint.type == "prismatic" else angular_step_rad
        endpoints: dict[str, Any] = {}
        for direction, sign, boundary in (("minus", -1.0, lower), ("plus", 1.0, upper)):
            available = baseline - boundary if sign < 0.0 else boundary - baseline
            if available <= EPSILON:
                endpoints[direction] = {
                    "status": "unavailable_at_feasible_limit",
                    "requested_delta": clean_number(sign * nominal_step),
                    "applied_delta": 0.0,
                    "joint_position": clean_number(baseline),
                    "joint_position_unit": unit,
                }
                continue
            applied_magnitude = nominal_step if not math.isfinite(available) else min(nominal_step, available)
            applied_delta = sign * applied_magnitude
            requested = dict(resolved)
            requested[driver] = baseline + applied_delta
            endpoint_pose = self.resolve_pose(requested)
            endpoints[direction] = {
                "status": (
                    "applied_nominal_step"
                    if applied_magnitude >= nominal_step - EPSILON
                    else "clipped_to_feasible_limit"
                ),
                "requested_delta": clean_number(sign * nominal_step),
                "applied_delta": clean_number(applied_delta),
                "joint_position": clean_number(endpoint_pose[driver]),
                "joint_position_unit": unit,
                "resolved_joint_positions": {
                    name: clean_number(value) for name, value in sorted(endpoint_pose.items())
                },
                "physical_joint_positions": {
                    name: clean_number(endpoint_pose[name]) for name in physical_joints
                },
            }
        return {
            "driver_joint": driver,
            "joint_type": driver_joint.type,
            "joint_position_unit": unit,
            "baseline_position": clean_number(baseline),
            "nominal_step": clean_number(nominal_step),
            "feasible_interval": {
                "minimum": domain["minimum"],
                "maximum": domain["maximum"],
                "minimum_unbounded": domain["minimum_unbounded"],
                "maximum_unbounded": domain["maximum_unbounded"],
                "constraints": constraints,
            },
            "physical_joints_driven": physical_joints,
            "baseline_physical_joint_positions": {
                name: clean_number(resolved[name]) for name in physical_joints
            },
            "structural_causality": self.affected_by_joint(driver),
            "endpoints": endpoints,
        }

    def independent_driver_contract(self, driver: str) -> dict[str, Any]:
        """Return the pose-independent feasible domain and physical joints for one driver."""
        if driver not in self.joints:
            raise SpatialError(f"unknown joint {driver!r}")
        driver_joint = self.joints[driver]
        if driver_joint.type == "fixed":
            raise SpatialError(f"fixed joint {driver!r} is not an independent motion variable")
        if driver_joint.mimic is not None:
            independent, _, _ = self._independent_mimic_driver(driver)
            raise SpatialError(f"joint {driver!r} is mimic-driven; use independent driver {independent!r}")
        lower, upper = -math.inf, math.inf
        constraints: list[dict[str, Any]] = []
        physical_joints: list[str] = []
        for joint_name, joint in sorted(self.joints.items()):
            if joint.type == "fixed":
                continue
            candidate_driver, multiplier, offset, mimic_chain = self._mimic_affine_from_driver(joint_name)
            if candidate_driver != driver:
                continue
            physical_joints.append(joint_name)
            declared_lower = None if joint.type == "continuous" else joint.limit["lower"]
            declared_upper = None if joint.type == "continuous" else joint.limit["upper"]
            constraint = {
                "joint": joint_name,
                "joint_type": joint.type,
                "affine_position_from_driver": {
                    "multiplier": clean_number(multiplier),
                    "offset": clean_number(offset),
                },
                "mimic_chain": mimic_chain,
                "declared_lower": declared_lower,
                "declared_upper": declared_upper,
            }
            constraints.append(constraint)
            if abs(multiplier) <= EPSILON:
                if (
                    (declared_lower is not None and offset < float(declared_lower) - EPSILON)
                    or (declared_upper is not None and offset > float(declared_upper) + EPSILON)
                ):
                    raise SpatialError(
                        f"mimic joint {joint_name!r} has constant position {offset} outside its declared limits"
                    )
                continue
            if declared_lower is not None:
                boundary = (float(declared_lower) - offset) / multiplier
                if multiplier > 0.0:
                    lower = max(lower, boundary)
                else:
                    upper = min(upper, boundary)
            if declared_upper is not None:
                boundary = (float(declared_upper) - offset) / multiplier
                if multiplier > 0.0:
                    upper = min(upper, boundary)
                else:
                    lower = max(lower, boundary)
        if lower > upper + EPSILON:
            raise SpatialError(f"independent driver {driver!r} has no feasible position interval")
        unit = "m" if driver_joint.type == "prismatic" else "rad"
        return {
            "driver_joint": driver,
            "joint_type": driver_joint.type,
            "unit": unit,
            "feasible_domain": {
                "minimum": None if not math.isfinite(lower) else clean_number(lower),
                "maximum": None if not math.isfinite(upper) else clean_number(upper),
                "minimum_unbounded": not math.isfinite(lower),
                "maximum_unbounded": not math.isfinite(upper),
                "constraints": constraints,
            },
            "physical_joints_driven": physical_joints,
            "structural_causality": self.affected_by_joint(driver),
            "epistemic_scope": "declared URDF position limits intersected through exact affine mimic equations; no collision, dynamics, controller, or hardware feasibility is implied",
        }

    def geometric_jacobian(self, target_frame: str, supplied_pose: dict[str, float], expressed_in_frame: str | None = None) -> dict[str, Any]:
        expressed_in = expressed_in_frame or self.root_link
        frames, resolved_pose = self.world_frames(supplied_pose)
        if target_frame not in frames:
            raise SpatialError(f"unknown target frame {target_frame!r}")
        if expressed_in not in frames:
            raise SpatialError(f"unknown expressed-in frame {expressed_in!r}")
        attachment_link = self._frame_attachment_link(target_frame)
        path = self.chain(self.root_link, attachment_link)
        target_origin = [frames[target_frame][index][3] for index in range(3)]
        aggregated: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for step in path["steps"]:
            physical_joint_name = step["joint"]
            physical_joint = self.joints[physical_joint_name]
            if physical_joint.type == "fixed":
                continue
            driver, derivative_multiplier, affine_offset, mimic_chain = self._mimic_affine_from_driver(physical_joint_name)
            if driver not in aggregated:
                order.append(driver)
                aggregated[driver] = {
                    "linear_root": [0.0, 0.0, 0.0],
                    "angular_root": [0.0, 0.0, 0.0],
                    "physical_contributions": [],
                }
            joint_frame = frames[f"joint/{physical_joint_name}"]
            axis_root = normalized(rotate_vector(joint_frame, physical_joint.axis), f"joint {physical_joint_name} axis")
            joint_origin = [joint_frame[index][3] for index in range(3)]
            if physical_joint.type in {"revolute", "continuous"}:
                delta = [target_origin[index] - joint_origin[index] for index in range(3)]
                linear = cross_product(axis_root, delta)
                angular = axis_root
            else:
                linear = axis_root
                angular = [0.0, 0.0, 0.0]
            for index in range(3):
                aggregated[driver]["linear_root"][index] += derivative_multiplier * linear[index]
                aggregated[driver]["angular_root"][index] += derivative_multiplier * angular[index]
            aggregated[driver]["physical_contributions"].append({
                "joint": physical_joint_name,
                "derivative_multiplier": clean_number(derivative_multiplier),
                "position_offset_from_driver": clean_number(affine_offset),
                "mimic_chain": mimic_chain,
            })
        root_from_expressed = frames[expressed_in]
        expressed_from_root = inverse_rigid(root_from_expressed)
        columns: list[dict[str, Any]] = []
        for driver in order:
            driver_joint = self.joints[driver]
            linear = clean_vector(rotate_vector(expressed_from_root, aggregated[driver]["linear_root"]))
            angular = clean_vector(rotate_vector(expressed_from_root, aggregated[driver]["angular_root"]))
            position_unit = "m" if driver_joint.type == "prismatic" else "rad"
            columns.append({
                "joint": driver,
                "joint_type": driver_joint.type,
                "joint_position": clean_number(resolved_pose[driver]),
                "joint_position_unit": position_unit,
                "linear_xyz_per_joint_unit": linear,
                "angular_xyz_per_joint_unit": angular,
                "physical_contributions": aggregated[driver]["physical_contributions"],
            })
        matrix = [
            [column["linear_xyz_per_joint_unit"][row] for column in columns]
            for row in range(3)
        ] + [
            [column["angular_xyz_per_joint_unit"][row] for column in columns]
            for row in range(3)
        ]
        return {
            "schema_version": "robot-geometric-jacobian.v1",
            "target_frame": target_frame,
            "target_attachment_link": attachment_link,
            "motion_relative_to_frame": self.root_link,
            "components_expressed_in_orientation_of_frame": expressed_in,
            "pose_joint_positions": {name: clean_number(value) for name, value in resolved_pose.items()},
            "joint_order": order,
            "row_order": ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"],
            "matrix_6xn": matrix,
            "columns": columns,
            "meaning": "maps independent joint rates to target-origin geometric twist relative to the root; vector components use the requested frame orientation",
        }

    def workspace_envelope(self, target_frame: str, supplied_pose: dict[str, float], sample_count: int = 256, include_samples: bool = False) -> dict[str, Any]:
        if sample_count < 1 or sample_count > 100000:
            raise SpatialError("sample_count must be between 1 and 100000")
        jacobian = self.geometric_jacobian(target_frame, supplied_pose)
        joint_order = jacobian["joint_order"]
        ranges: list[dict[str, Any]] = []
        for joint_name in joint_order:
            ranges.append(self._workspace_driver_range(joint_name))

        dimensions = len(joint_order)
        fractions: list[list[float]] = []
        seen: set[tuple[float, ...]] = set()

        def add_fraction(values: list[float]) -> None:
            key = tuple(round(value, 15) for value in values)
            if key not in seen and len(fractions) < sample_count:
                seen.add(key)
                fractions.append(values)

        if dimensions == 0:
            add_fraction([])
        else:
            add_fraction([0.5] * dimensions)
            for axis in range(dimensions):
                low, high = [0.5] * dimensions, [0.5] * dimensions
                low[axis], high[axis] = 0.0, 1.0
                add_fraction(low)
                add_fraction(high)
            corner_count = 1 << dimensions if dimensions <= 20 else sample_count + 1
            if corner_count <= max(0, sample_count - len(fractions)):
                for corner in range(corner_count):
                    add_fraction([float((corner >> axis) & 1) for axis in range(dimensions)])
            else:
                add_fraction([0.0] * dimensions)
                add_fraction([1.0] * dimensions)
            primes = first_primes(dimensions)
            index = 1
            while len(fractions) < sample_count:
                add_fraction([radical_inverse(index, base) for base in primes])
                index += 1

        baseline_pose = dict(supplied_pose)
        samples: list[dict[str, Any]] = []
        origins: list[Vector] = []
        orientation_axes: dict[str, list[Vector]] = {"x": [], "y": [], "z": []}
        for values in fractions:
            sampled_pose = dict(baseline_pose)
            sampled_joint_positions: dict[str, float] = {}
            for axis, joint_name in enumerate(joint_order):
                lower, upper = ranges[axis]["minimum"], ranges[axis]["maximum"]
                value = lower + values[axis] * (upper - lower)
                sampled_pose[joint_name] = value
                sampled_joint_positions[joint_name] = clean_number(value)
            frames, _ = self.world_frames(sampled_pose)
            transform = frames[target_frame]
            origin = clean_vector([transform[index][3] for index in range(3)])
            quaternion = clean_vector(quaternion_xyzw(transform))
            origins.append(origin)
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                orientation_axes[axis_name].append(clean_vector([transform[row][axis_index] for row in range(3)]))
            samples.append({
                "independent_joint_positions": sampled_joint_positions,
                "target_origin_in_root_xyz_m": origin,
                "target_orientation_in_root_quaternion_xyzw": quaternion,
            })

        minimum = [min(origin[axis] for origin in origins) for axis in range(3)]
        maximum = [max(origin[axis] for origin in origins) for axis in range(3)]
        extents = [maximum[axis] - minimum[axis] for axis in range(3)]
        center = [(maximum[axis] + minimum[axis]) / 2.0 for axis in range(3)]
        radii = [math.sqrt(sum(value * value for value in origin)) for origin in origins]
        orientation_ranges = {
            axis_name: {
                "component_min_xyz": clean_vector(min(vector[component] for vector in vectors) for component in range(3)),
                "component_max_xyz": clean_vector(max(vector[component] for vector in vectors) for component in range(3)),
            }
            for axis_name, vectors in orientation_axes.items()
        }
        sample_digest = hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        result = {
            "schema_version": "robot-sampled-workspace.v1",
            "target_frame": target_frame,
            "root_frame": self.root_link,
            "independent_joint_order": joint_order,
            "joint_sampling_ranges": ranges,
            "sampling": {
                "method": "center_then_axis_extrema_then_corners_when_feasible_then_halton_low_discrepancy",
                "requested_sample_count": sample_count,
                "evaluated_sample_count": len(samples),
                "deterministic": True,
                "sample_sha256": sample_digest,
            },
            "observed_target_origin_aabb_in_root": {
                "min_xyz_m": clean_vector(minimum),
                "max_xyz_m": clean_vector(maximum),
                "extents_xyz_m": clean_vector(extents),
                "center_xyz_m": clean_vector(center),
            },
            "observed_radial_distance_from_root_origin_m": {
                "minimum": clean_number(min(radii)),
                "maximum": clean_number(max(radii)),
            },
            "observed_target_axis_component_ranges_in_root": orientation_ranges,
            "approximate": True,
            "meaning": "deterministic observations at finite joint samples; the AABB bounds only sampled target origins and does not prove that every point inside is reachable or that unsampled reachable points lie inside",
        }
        if include_samples:
            result["samples"] = samples
        return result


def read_pose(path: Path | None) -> tuple[str, dict[str, float]]:
    if path is None:
        return "zero", {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpatialError(f"cannot read pose JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise SpatialError("pose JSON must be an object")
    pose_name = data.get("pose_name", path.stem)
    joints = data.get("joints", data if "pose_name" not in data else None)
    if not isinstance(pose_name, str) or not isinstance(joints, dict):
        raise SpatialError("pose JSON must contain a string pose_name and an object joints")
    try:
        converted = {str(name): float(value) for name, value in joints.items()}
    except (TypeError, ValueError) as error:
        raise SpatialError("all pose joint values must be numeric") from error
    return pose_name, converted


def resolve_pose_input(args: argparse.Namespace, model: RobotModel) -> tuple[str, dict[str, float], dict[str, Any] | None]:
    srdf = parse_srdf(getattr(args, "srdf", None), model)
    pose_path = getattr(args, "pose", None)
    pose_name = getattr(args, "pose_name", None)
    if pose_path is not None and pose_name is not None:
        raise SpatialError("use either --pose JSON or --pose-name from SRDF, not both")
    if pose_name is not None:
        if srdf is None:
            raise SpatialError("--pose-name requires --srdf")
        resolved_name, pose = resolve_named_pose(srdf, pose_name)
        return resolved_name, pose, srdf
    resolved_name, pose = read_pose(pose_path)
    return resolved_name, pose, srdf


def read_semantics(path: Path | None, model: RobotModel) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SpatialError(f"cannot read semantics JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise SpatialError("semantics JSON must be an object")
    schema_version = data.get("schema_version", "robot-semantics.v1")
    if schema_version != "robot-semantics.v1":
        raise SpatialError(f"unsupported semantics schema_version: {schema_version!r}")
    known_frames = set(model.frame_semantics())
    frames = data.get("frames", {})
    groups = data.get("groups", {})
    end_effectors = data.get("end_effectors", {})
    if not isinstance(frames, dict) or not isinstance(groups, dict) or not isinstance(end_effectors, dict):
        raise SpatialError("semantics frames, groups, and end_effectors must be objects")
    canonical_frames: dict[str, Any] = {}
    for frame_name, annotation in frames.items():
        if frame_name not in known_frames:
            raise SpatialError(f"semantic annotation references unknown frame {frame_name!r}")
        if not isinstance(annotation, dict):
            raise SpatialError(f"semantic frame {frame_name!r} must be an object")
        roles = annotation.get("roles", [])
        meaning = annotation.get("meaning")
        if not isinstance(roles, list) or not all(isinstance(role, str) and role for role in roles):
            raise SpatialError(f"semantic frame {frame_name!r} roles must be non-empty strings")
        if meaning is not None and not isinstance(meaning, str):
            raise SpatialError(f"semantic frame {frame_name!r} meaning must be a string")
        canonical_frames[frame_name] = {"roles": sorted(set(roles)), "meaning": meaning}
    canonical_groups: dict[str, Any] = {}
    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not group_name or not isinstance(group, dict):
            raise SpatialError("semantic groups require non-empty names and object values")
        joint_names = group.get("joints", [])
        if not isinstance(joint_names, list) or not all(isinstance(name, str) for name in joint_names):
            raise SpatialError(f"semantic group {group_name!r} joints must be a string list")
        unknown_joints = sorted(set(joint_names) - set(model.joints))
        if unknown_joints:
            raise SpatialError(f"semantic group {group_name!r} references unknown joints: {unknown_joints}")
        base_frame, tip_frame = group.get("base_frame"), group.get("tip_frame")
        for label, frame_name in (("base_frame", base_frame), ("tip_frame", tip_frame)):
            if frame_name is not None and frame_name not in known_frames:
                raise SpatialError(f"semantic group {group_name!r} {label} references unknown frame {frame_name!r}")
        canonical_groups[group_name] = {"joints": joint_names, "base_frame": base_frame, "tip_frame": tip_frame}
    canonical_end_effectors: dict[str, Any] = {}
    for end_effector_name, end_effector in end_effectors.items():
        if not isinstance(end_effector_name, str) or not end_effector_name or not isinstance(end_effector, dict):
            raise SpatialError("semantic end_effectors require non-empty names and object values")
        mount_frame, tcp_frame = end_effector.get("mount_frame"), end_effector.get("tcp_frame")
        if mount_frame not in known_frames or tcp_frame not in known_frames:
            raise SpatialError(f"end effector {end_effector_name!r} references unknown mount/TCP frame")
        canonical_end_effectors[end_effector_name] = {"mount_frame": mount_frame, "tcp_frame": tcp_frame}
    return {
        "status": "user_or_project_asserted",
        "schema_version": schema_version,
        "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "frames": canonical_frames,
        "groups": canonical_groups,
        "end_effectors": canonical_end_effectors,
    }


def read_model_artifact(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpatialError(f"cannot read model artifact {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise SpatialError(f"{path} is not a {SCHEMA_VERSION} model artifact")
    return data


def compare_artifacts(before_path: Path, after_path: Path, translation_tolerance: float, rotation_tolerance_deg: float) -> dict[str, Any]:
    if translation_tolerance < 0.0 or rotation_tolerance_deg < 0.0:
        raise SpatialError("compare tolerances must be non-negative")
    before, after = read_model_artifact(before_path), read_model_artifact(after_path)
    before_root = before.get("robot", {}).get("root_link")
    after_root = after.get("robot", {}).get("root_link")
    if before_root != after_root:
        raise SpatialError(f"cannot numerically compare artifacts with different root frames: {before_root!r}, {after_root!r}")
    before_frames, after_frames = before.get("frames", {}), after.get("frames", {})
    added_frames = sorted(set(after_frames) - set(before_frames))
    removed_frames = sorted(set(before_frames) - set(after_frames))
    changed_frames: list[dict[str, Any]] = []
    for name in sorted(set(before_frames) & set(after_frames)):
        before_matrix = before_frames[name]["world_from_frame"]["matrix_4x4_rowmajor"]
        after_matrix = after_frames[name]["world_from_frame"]["matrix_4x4_rowmajor"]
        before_xyz = [before_matrix[index][3] for index in range(3)]
        after_xyz = [after_matrix[index][3] for index in range(3)]
        delta = [after_xyz[index] - before_xyz[index] for index in range(3)]
        translation_change = math.sqrt(sum(value * value for value in delta))
        relative_rotation = matmul(inverse_rigid(before_matrix), after_matrix)
        cosine = max(-1.0, min(1.0, (sum(relative_rotation[index][index] for index in range(3)) - 1.0) / 2.0))
        rotation_change_deg = math.degrees(math.acos(cosine))
        semantic_changed = any(before_frames[name].get(key) != after_frames[name].get(key) for key in ("type", "parent_frame", "owner"))
        if translation_change > translation_tolerance or rotation_change_deg > rotation_tolerance_deg or semantic_changed:
            changed_frames.append({
                "frame": name,
                "translation_delta_in_root_m": clean_vector(delta),
                "translation_change_m": clean_number(translation_change),
                "rotation_change_deg": clean_number(rotation_change_deg),
                "semantic_changed": semantic_changed,
            })
    before_joints = before.get("joints", {})
    after_joints = after.get("joints", {})
    changed_joint_definitions = sorted(
        name for name in set(before_joints) & set(after_joints)
        if {key: value for key, value in before_joints[name].items() if key not in {"position_at_pose", "axis_in_root_frame_at_pose"}}
        != {key: value for key, value in after_joints[name].items() if key not in {"position_at_pose", "axis_in_root_frame_at_pose"}}
    )
    before_links, after_links = before.get("links", {}), after.get("links", {})
    changed_link_definitions = sorted(name for name in set(before_links) & set(after_links) if before_links[name] != after_links[name])
    changed_declared_inertials = [
        {
            "link": name,
            "before": before_links[name].get("inertial"),
            "after": after_links[name].get("inertial"),
        }
        for name in sorted(set(before_links) & set(after_links))
        if before_links[name].get("inertial") != after_links[name].get("inertial")
    ]
    before_geometry, after_geometry = before.get("geometry_analysis", {}), after.get("geometry_analysis", {})

    def intrinsic_geometry(record: dict[str, Any]) -> dict[str, Any]:
        principal = record.get("principal_axes")
        if isinstance(principal, dict):
            principal = {key: value for key, value in principal.items() if key != "axes_in_root_frame_at_pose"}
        return {
            key: value for key, value in {
                "frame": record.get("frame"),
                "kind": record.get("kind"),
                "link": record.get("link"),
                "declared_origin_xyz_m": record.get("declared_origin_xyz_m"),
                "declared_origin_rpy_rad": record.get("declared_origin_rpy_rad"),
                "status": record.get("status"),
                "geometry_type": record.get("geometry_type"),
                "source": record.get("source"),
                "topology": record.get("topology"),
                "bounds_in_geometry_frame": record.get("bounds_in_geometry_frame"),
                "surface_area_m2": record.get("surface_area_m2"),
                "volume_m3": record.get("volume_m3"),
                "volume_trust": record.get("volume_trust"),
                "vertex_mean_xyz_m": record.get("vertex_mean_xyz_m"),
                "principal_axes": principal,
                "landmarks_in_geometry_frame": record.get("landmarks_in_geometry_frame"),
                "shape": record.get("shape"),
                "uri": record.get("uri"),
                "reason": record.get("reason"),
            }.items() if value is not None
        }

    changed_geometry_intrinsics = sorted(
        name for name in set(before_geometry) & set(after_geometry)
        if intrinsic_geometry(before_geometry[name]) != intrinsic_geometry(after_geometry[name])
    )
    changed_geometry_world_bounds: list[dict[str, Any]] = []
    for name in sorted(set(before_geometry) & set(after_geometry)):
        before_bounds = before_geometry[name].get("bounds_in_root_frame_at_pose")
        after_bounds = after_geometry[name].get("bounds_in_root_frame_at_pose")
        if before_bounds is None or after_bounds is None:
            if before_bounds != after_bounds:
                changed_geometry_world_bounds.append({"frame": name, "before": before_bounds, "after": after_bounds})
            continue
        maximum_delta = max(
            abs(before_bounds[field][axis] - after_bounds[field][axis])
            for field in ("min_xyz_m", "max_xyz_m")
            for axis in range(3)
        )
        if maximum_delta > translation_tolerance:
            changed_geometry_world_bounds.append({
                "frame": name,
                "maximum_bound_coordinate_change_m": clean_number(maximum_delta),
                "before": before_bounds,
                "after": after_bounds,
            })
    topology = {
        "added_links": sorted(set(after_links) - set(before_links)),
        "removed_links": sorted(set(before_links) - set(after_links)),
        "added_joints": sorted(set(after_joints) - set(before_joints)),
        "removed_joints": sorted(set(before_joints) - set(after_joints)),
        "added_frames": added_frames,
        "removed_frames": removed_frames,
        "added_geometry": sorted(set(after_geometry) - set(before_geometry)),
        "removed_geometry": sorted(set(before_geometry) - set(after_geometry)),
    }
    source_urdf_changed = before.get("source", {}).get("sha256") != after.get("source", {}).get("sha256")
    pose_changed = before.get("pose") != after.get("pose")
    semantics_changed = before.get("semantics") != after.get("semantics")
    srdf_changed = before.get("srdf") != after.get("srdf")
    collision_surface_changed = before.get("collision_surface") != after.get("collision_surface")
    invariant_validation_changed = before.get("invariant_validation") != after.get("invariant_validation")
    before_mass = before.get("physical_analysis", {}).get("declared_mass_properties")
    after_mass = after.get("physical_analysis", {}).get("declared_mass_properties")
    declared_mass_properties_changed = before_mass != after_mass
    declared_mass_properties_change: dict[str, Any] = {
        "changed": declared_mass_properties_changed,
        "before": before_mass,
        "after": after_mass,
    }
    if before_mass is not None and after_mass is not None:
        before_total, after_total = before_mass.get("declared_mass_kg"), after_mass.get("declared_mass_kg")
        if before_total is not None and after_total is not None:
            declared_mass_properties_change["declared_mass_delta_kg"] = clean_number(after_total - before_total)
        before_com = before_mass.get("center_of_mass_in_expressed_frame_m")
        after_com = after_mass.get("center_of_mass_in_expressed_frame_m")
        if before_com is not None and after_com is not None:
            delta = [after_com[index] - before_com[index] for index in range(3)]
            declared_mass_properties_change["center_of_mass_delta_m"] = clean_vector(delta)
            declared_mass_properties_change["center_of_mass_change_m"] = clean_number(math.sqrt(sum(value * value for value in delta)))
        before_tensor = before_mass.get("inertia_about_center_of_mass_in_expressed_frame_kg_m2")
        after_tensor = after_mass.get("inertia_about_center_of_mass_in_expressed_frame_kg_m2")
        if before_tensor is not None and after_tensor is not None:
            before_matrix = before_tensor["matrix_3x3_rowmajor"]
            after_matrix = after_tensor["matrix_3x3_rowmajor"]
            declared_mass_properties_change["maximum_inertia_component_change_kg_m2"] = clean_number(max(
                abs(after_matrix[row][column] - before_matrix[row][column])
                for row in range(3)
                for column in range(3)
            ))
    before_gravity = before.get("physical_analysis", {}).get("declared_static_gravity_loads_under_root_frame_convention")
    after_gravity = after.get("physical_analysis", {}).get("declared_static_gravity_loads_under_root_frame_convention")
    declared_static_gravity_loads_changed = before_gravity != after_gravity
    gravity_load_change: dict[str, Any] = {
        "changed": declared_static_gravity_loads_changed,
        "before": before_gravity,
        "after": after_gravity,
        "independent_driver_deltas": {},
    }
    if before_gravity is not None and after_gravity is not None:
        before_loads = before_gravity.get("independent_driver_loads") or {}
        after_loads = after_gravity.get("independent_driver_loads") or {}
        for driver in sorted(set(before_loads) | set(after_loads)):
            before_value = before_loads.get(driver, {}).get("generalized_gravity_force")
            after_value = after_loads.get(driver, {}).get("generalized_gravity_force")
            gravity_load_change["independent_driver_deltas"][driver] = {
                "before": before_value,
                "after": after_value,
                "delta": (
                    None
                    if before_value is None or after_value is None
                    else clean_number(after_value - before_value)
                ),
                "unit": after_loads.get(driver, before_loads.get(driver, {})).get("unit"),
            }
    before_actuation = before.get("actuation")
    after_actuation = after.get("actuation")
    actuation_declarations_changed = before_actuation != after_actuation
    before_world_scene = before.get("world_scene")
    after_world_scene = after.get("world_scene")
    world_scene_changed = before_world_scene != after_world_scene
    before_observed_world = before.get("observed_world")
    after_observed_world = after.get("observed_world")
    observed_world_changed = before_observed_world != after_observed_world
    before_scene_gravity = before.get("physical_analysis", {}).get("declared_static_gravity_loads_under_scene_gravity")
    after_scene_gravity = after.get("physical_analysis", {}).get("declared_static_gravity_loads_under_scene_gravity")
    scene_gravity_loads_changed = before_scene_gravity != after_scene_gravity
    scene_gravity_load_change: dict[str, Any] = {
        "changed": scene_gravity_loads_changed,
        "before": before_scene_gravity,
        "after": after_scene_gravity,
        "independent_driver_deltas": {},
    }
    if before_scene_gravity is not None and after_scene_gravity is not None:
        before_loads = ((before_scene_gravity.get("loads") or {}).get("independent_driver_loads") or {})
        after_loads = ((after_scene_gravity.get("loads") or {}).get("independent_driver_loads") or {})
        for driver in sorted(set(before_loads) | set(after_loads)):
            before_value = before_loads.get(driver, {}).get("generalized_gravity_force")
            after_value = after_loads.get(driver, {}).get("generalized_gravity_force")
            scene_gravity_load_change["independent_driver_deltas"][driver] = {
                "before": before_value,
                "after": after_value,
                "delta": None if before_value is None or after_value is None else clean_number(after_value - before_value),
                "unit": after_loads.get(driver, before_loads.get(driver, {})).get("unit"),
            }
    changed = bool(
        source_urdf_changed
        or pose_changed
        or changed_frames
        or changed_joint_definitions
        or changed_link_definitions
        or changed_declared_inertials
        or changed_geometry_intrinsics
        or changed_geometry_world_bounds
        or any(topology.values())
        or semantics_changed
        or srdf_changed
        or collision_surface_changed
        or invariant_validation_changed
        or declared_mass_properties_changed
        or declared_static_gravity_loads_changed
        or actuation_declarations_changed
        or world_scene_changed
        or observed_world_changed
        or scene_gravity_loads_changed
    )
    return {
        "schema_version": "robot-spatial-comparison.v1",
        "changed": changed,
        "reference_frame": before_root,
        "before": {"path": str(before_path.resolve()), "pose": before.get("pose"), "source": before.get("source")},
        "after": {"path": str(after_path.resolve()), "pose": after.get("pose"), "source": after.get("source")},
        "tolerances": {"translation_m": translation_tolerance, "rotation_deg": rotation_tolerance_deg},
        "topology": topology,
        "source_urdf_changed": source_urdf_changed,
        "pose_changed": pose_changed,
        "changed_joint_definitions": changed_joint_definitions,
        "changed_link_definitions": changed_link_definitions,
        "changed_declared_inertials": changed_declared_inertials,
        "declared_mass_properties_change": declared_mass_properties_change,
        "declared_static_gravity_loads_change": gravity_load_change,
        "actuation_declarations_changed": actuation_declarations_changed,
        "actuation_declarations_change": {"before": before_actuation, "after": after_actuation},
        "world_scene_changed": world_scene_changed,
        "world_scene_change": {"before": before_world_scene, "after": after_world_scene},
        "observed_world_changed": observed_world_changed,
        "observed_world_change": {"before": before_observed_world, "after": after_observed_world},
        "scene_gravity_loads_change": scene_gravity_load_change,
        "changed_geometry_intrinsics": changed_geometry_intrinsics,
        "changed_geometry_world_bounds": changed_geometry_world_bounds,
        "changed_frames": changed_frames,
        "semantic_annotations_changed": semantics_changed,
        "srdf_semantics_changed": srdf_changed,
        "collision_surface_analysis_changed": collision_surface_changed,
        "invariant_validation_changed": invariant_validation_changed,
    }


def context_markdown(model: RobotModel, canonical: dict[str, Any]) -> str:
    pose = canonical["pose"]
    lines = [
        "# Robot spatial context",
        "",
        f"- Robot: `{model.name}`",
        f"- Root frame: `{model.root_link}`",
        f"- External world scene: `{'bound' if canonical.get('world_scene', {}).get('status') == 'parsed_validated_and_bound' else 'not_provided'}`",
        f"- Pose: `{pose['name']}`",
        "- Units: meters and radians; quaternions are xyzw",
        f"- Source SHA-256: `{model.sha256}`",
        "",
        "## Progressive disclosure for AI",
        "",
        "- Read `agent-context.json` for identity rules, epistemic labels, unresolved claims, artifact digests, and question routing.",
        "- Use `query-concepts` on `concept-graph.json` for compositional topology, articulation causality, mechanism constraints, finite-node comparison, and a minimal proof closure; `concept-language.rsl` is the compact whole-robot symbolic view.",
        (
            "- Use `query-functions` on `functional-model.json` for explicit component function, capability requirement grounding, conditions, intended effects, relational affordances, and scoped inventory boundaries; structural grounding is not physical execution."
            if isinstance(canonical.get("artifacts", {}).get("functional_model"), dict)
            else "- Function/capability/affordance knowledge was not provided. Do not infer what a part is for from its name, topology, or geometry."
        ),
        "- Retrieve one exact typed record from `entity-cards.jsonl`; `link/X` and `frame/X` are deliberately different entities.",
        "- Follow the card's `fact_ids` into `facts.jsonl`, or run `retrieve` so digests and byte offsets are checked automatically.",
        "- Use a fresh deterministic CLI query when the requested pose, frame pair, or relationship is not already exported.",
        "- Keep robot-local `frame/X` distinct from snapshot-bound `robot_frame/X`; use typed scene entities and `scene-transform` for world/object relationships.",
        "- Cite facts or tool results. The controlled-language card is an index, not independent evidence.",
        "",
        "## Kinematic tree",
        "",
        "```text",
        *model.tree_lines(),
        "```",
        "",
        "## Joint state and axis",
        "",
        "| Joint | Type | Parent → child | Position | Axis in root frame |",
        "|---|---|---|---:|---|",
    ]
    for name, joint in canonical["joints"].items():
        position = joint["position_at_pose"]
        unit = joint["position_unit"] or ""
        axis = joint["axis_in_root_frame_at_pose"]
        lines.append(f"| `{name}` | {joint['type']} | `{joint['parent_link']}` → `{joint['child_link']}` | {position} {unit} | {axis if axis is not None else '—'} |")
    mass_properties = canonical["physical_analysis"]["declared_mass_properties"]
    lines.extend([
        "",
        "## Declared mass properties",
        "",
        f"- Status: `{mass_properties['status']}`; aggregation is expressed in `{mass_properties['expressed_in_frame']}` at pose `{pose['name']}`.",
        f"- Declared mass: `{mass_properties['declared_mass_kg']}` kg",
        f"- Declared center of mass: `{mass_properties['center_of_mass_in_expressed_frame_m']}` m",
        f"- Inertia about the aggregate center of mass: `{json.dumps(mass_properties['inertia_about_center_of_mass_in_expressed_frame_kg_m2'], sort_keys=True)}`",
        f"- Coverage: `{json.dumps(mass_properties['coverage'], sort_keys=True)}`",
        "- These values combine URDF-declared inertials with forward kinematics and the parallel-axis theorem. They do not prove actual hardware mass, payload, calibration, or zero mass for links without inertial elements.",
        "",
        "| Link | Inertial status | Declared mass (kg) | Link-from-inertial origin xyz (m) |",
        "|---|---|---:|---|",
    ])
    for link_name, link in canonical["links"].items():
        inertial = link["inertial"]
        if inertial is None:
            lines.append(f"| `{link_name}` | not_provided | — | — |")
        else:
            lines.append(
                f"| `{link_name}` | {inertial['validation']['status']} | {inertial['mass_kg']} | {inertial['origin_xyz_m']} |"
            )
    gravity_loads = canonical["physical_analysis"]["declared_static_gravity_loads_under_root_frame_convention"]
    lines.extend([
        "",
        "## Declared-model static gravity loads",
        "",
        f"- Status: `{gravity_loads['status']}` at pose `{pose['name']}`.",
        f"- Export convention: gravity `{gravity_loads['gravity']['vector_xyz_m_s2']}` m/s² expressed in `{gravity_loads['gravity']['expressed_in_frame']}`.",
        f"- Coverage: `{json.dumps(gravity_loads['coverage'], sort_keys=True)}`",
        "- `generalized_gravity_force` is exerted by gravity along positive independent-joint motion; `ideal_static_holding_effort` is its opposite for gravity-only equilibrium.",
        "- This does not establish actual mounting orientation, payload, contacts, friction, dynamic motion, transmission loss, controller capability, or hardware feasibility.",
        "",
        "| Independent driver | Gravity force/torque | Ideal holding effort | Unit | Declared effort limit |",
        "|---|---:|---:|---|---:|",
    ])
    if gravity_loads.get("independent_driver_loads"):
        for joint_name in gravity_loads["independent_driver_order"]:
            load = gravity_loads["independent_driver_loads"][joint_name]
            lines.append(
                f"| `{joint_name}` | {load['generalized_gravity_force']} | {load['ideal_static_holding_effort']} | {load['unit']} | {load['declared_joint_effort_limit_magnitude']} |"
            )
    else:
        lines.append("| — | indeterminate | indeterminate | — | — |")
    actuation = canonical["actuation"]
    lines.extend([
        "",
        "## Embedded actuation and control declarations",
        "",
        f"- ros2_control systems: `{sorted(actuation['ros2_control_systems'])}`",
        f"- Legacy transmissions: `{sorted(actuation['legacy_transmissions'])}`",
        f"- Coverage: `{json.dumps(actuation['coverage'], sort_keys=True)}`",
        "- These are transcribed declarations in the expanded URDF. They do not prove plugin installation, external controller configuration, interface claiming, hardware connectivity, calibration, or command execution.",
        "",
        "| Joint | ros2_control systems and interfaces | Legacy transmissions | Dynamics declaration |",
        "|---|---|---|---|",
    ])
    for joint_name, joint in canonical["joints"].items():
        binding = joint["actuation_declarations"]
        lines.append(
            f"| `{joint_name}` | `{json.dumps(binding['ros2_control'], sort_keys=True)}` | `{binding['legacy_transmissions']}` | `{json.dumps(joint['dynamics'], sort_keys=True)}` |"
        )
    world_scene = canonical.get("world_scene", {})
    lines.extend(["", "## Bound world scene", ""])
    if world_scene.get("status") != "parsed_validated_and_bound":
        lines.extend([
            "- No `robot-spatial-world-scene.v1` snapshot was provided.",
            "- URDF root-local coordinates do not establish a physical world pose, actual gravity direction, or external obstacles.",
        ])
    else:
        scene_collision = world_scene["robot_environment_collision"]
        scene_gravity = canonical["physical_analysis"]["declared_static_gravity_loads_under_scene_gravity"]
        lines.extend([
            f"- Scene / snapshot: `{world_scene['scene_id']}` / `{world_scene['snapshot']['id']}`; time semantics `{world_scene['snapshot']['time_semantics']}`.",
            f"- Scene SHA-256: `{world_scene['source']['sha256']}`; captured at `{world_scene['snapshot']['captured_at']}`; valid until `{world_scene['snapshot']['valid_until']}`.",
            f"- Robot instance: `{world_scene['robot_mount']['instance_id']}`; `{world_scene['robot_mount']['parent_entity']}` → `{world_scene['robot_mount']['root_entity']}`.",
            f"- `scene_frame/{world_scene['world_frame']}_from_robot_frame/{model.root_link}`: `{json.dumps(world_scene['robot_mount']['world_from_robot_root'], sort_keys=True)}`",
            f"- Root placement provenance: `{json.dumps(world_scene['robot_mount']['source'], sort_keys=True)}`",
            f"- Scene gravity conversion: `{json.dumps(world_scene['gravity'], sort_keys=True)}`",
            f"- Scene-bound gravity-load status: `{scene_gravity['status']}`.",
            f"- Robot/environment collision status: `{scene_collision['status']}`; minimum separation record: `{json.dumps(scene_collision['minimum_separation'], sort_keys=True)}`",
            f"- Collision coverage: `{json.dumps(scene_collision['coverage'], sort_keys=True)}`",
            "- Every result is conditional on this exact static snapshot, robot pose, scene digest, geometry coverage, and contact tolerance. Omitted objects are unknown, not empty space.",
            "",
            "| Scene object | Category | Roles | Parent scene frame | World xyz (m) | Collision geometry IDs |",
            "|---|---|---|---|---|---|",
        ])
        for object_id, object_record in world_scene["objects"].items():
            geometry_ids = [record["entity_id"] for record in object_record["collision_geometries"]]
            lines.append(
                f"| `scene_object/{object_id}` | `{object_record['semantics']['category']}` | `{object_record['semantics']['roles']}` | `scene_frame/{object_record['parent_frame']}` | {object_record['world_from_object']['translation_xyz_m']} | `{geometry_ids}` |"
            )
        if scene_collision["pair_results"]:
            lines.extend([
                "",
                "| Robot geometry | Environment geometry | Status | Separation / lower bound (m) | Method |",
                "|---|---|---|---:|---|",
            ])
            for pair in scene_collision["pair_results"]:
                separation = pair.get("separation_m", pair.get("separation_lower_bound_m"))
                lines.append(
                    f"| `{pair['robot_geometry']}` | `{pair['environment_geometry']}` | `{pair['status']}` | {separation} | `{pair.get('method')}` |"
                )
    observed_world = canonical.get("observed_world", {})
    lines.extend(["", "## Timestamped observed world", ""])
    if not isinstance(observed_world.get("observation"), dict):
        lines.extend([
            "- No timestamped observation log and query-time policy were provided.",
            "- The URDF model and static world scene above are declarations, not claims about what was observed now.",
        ])
    else:
        observation = observed_world["observation"]
        observed_analysis = observed_world["analysis"]
        lines.extend([
            f"- Observation log: `{observation['observation_log']['id']}` / SHA-256 `{observation['observation_log']['sha256']}`.",
            f"- Query: `{observation['query']['query_id']}` at `{observation['query']['time_ns']}` ns in clock `{observation['observation_log']['clock']['domain']}`.",
            f"- Resolution status: `{observation['status']}`; all required observations current: `{observation['readiness']['all_required_observations_current']}`.",
            f"- Age limits: `{json.dumps(observation['query']['maximum_age_ns'], sort_keys=True)}` ns.",
            f"- Static-declaration fallback used: `{observation['readiness']['declaration_fallback_used']}` for `{observation['readiness']['declaration_fallback_entities']}`.",
            f"- Selected joint/root/object samples: `{json.dumps(observation['selections'], sort_keys=True)}`",
            f"- Nominal observed-world analysis: `{observed_analysis['status']}`; physical-world truth and safety: `not_established`.",
            "- Version 1 uses only the latest sample at or before query time (zero-order hold), ignores future samples, and performs no interpolation.",
            "- `current` means the sample passes this query's age policy. It does not prove sensor truth, calibration, covariance-bounded geometry, omitted-object absence, or continuous-time collision freedom.",
        ])
    articulation = canonical.get("artifacts", {}).get("articulation_grammar")
    lines.extend([
        "",
        "## Pose-independent articulation grammar",
        "",
    ])
    if not isinstance(articulation, dict):
        lines.append("- Not generated. Do not generalize exported poses or finite motion endpoints into a universal joint law.")
    else:
        lines.extend([
            f"- Grammar: `articulation_grammar/{articulation['grammar_id']}` at `{articulation['path']}`; SHA-256 `{articulation['sha256']}`.",
            f"- Coverage: `{json.dumps(articulation['coverage'], sort_keys=True)}`",
            f"- Layer contract: `{json.dumps(articulation['layer_contract'], sort_keys=True)}`",
            "- Each independent variable has a complete mimic-constrained domain; each physical joint has a typed constant-plus-motion operator; each frame has an ordered root-to-frame derivation.",
            "- Run `evaluate-articulation` at a new driver binding without reparsing URDF. Run `verify-articulation-grammar` before trusting regeneration and all-frame FK consistency.",
            "",
            "| Independent driver | Unit | Default | Feasible minimum | Feasible maximum | Physical joints driven |",
            "|---|---|---:|---:|---:|---|",
        ])
        for driver, variable in sorted(articulation["independent_variables"].items()):
            domain = variable["feasible_domain"]
            lines.append(
                f"| `{driver}` | `{variable['unit']}` | {variable['default_value']} | {domain['minimum']} | {domain['maximum']} | `{variable['physical_joints_driven']}` |"
            )
    constraint_graph = canonical.get("artifacts", {}).get("constraint_graph")
    lines.extend([
        "",
        "## Supplemental mechanism constraints",
        "",
    ])
    if not isinstance(constraint_graph, dict):
        lines.append("- Not provided. Closed loops, cross-branch attachments, and coordinate couplings are unknown—not proven absent. The articulation tree must not be promoted to the complete mechanism without other evidence.")
    else:
        evaluation = constraint_graph["evaluation"]
        local = evaluation.get("local_constraint_analysis")
        lines.extend([
            f"- Constraint graph: `constraint_graph/{constraint_graph['constraint_graph_id']}` at `{constraint_graph['path']}`; SHA-256 `{constraint_graph['sha256']}`.",
            f"- The spanning tree is a parameterization rather than the complete mechanism: `{constraint_graph['structural_graph']['tree_is_parameterization_not_complete_mechanism']}`.",
            f"- Export-pose constraint status: `{evaluation['status']}`; maximum normalized residual: `{evaluation['maximum_normalized_abs']}`.",
            f"- Coverage: `{json.dumps(constraint_graph['coverage'], sort_keys=True)}`",
        ])
        if isinstance(local, dict):
            lines.append(
                f"- At this pose only: tree variables `{local['tree_independent_variable_count']}`, residual-Jacobian rank `{local['local_constraint_rank']}`, local mobility estimate `{local['local_mobility_estimate']}`. This is numerical local evidence, not global DOF."
            )
        lines.extend([
            "- Attachments and constraints are asserted mechanism semantics; their typed residuals are deterministic consequences of the embedded articulation law, not physical observations.",
            "- Run `evaluate-constraints` at a new binding, `solve-constraints` with explicit solved variables for a local branch, and `verify-constraint-graph` before trusting regeneration.",
        ])
    configuration_atlas = canonical.get("artifacts", {}).get("configuration_atlas")
    lines.extend(["", "## Finite configuration-space witnesses", ""])
    if not isinstance(configuration_atlas, dict):
        lines.append("- Not generated. A pose-local constraint rank does not establish branch count, singularity structure, or global configuration-space topology.")
    else:
        lines.extend([
            f"- Atlas: `configuration_atlas/{configuration_atlas['configuration_atlas_id']}` at `{configuration_atlas['path']}`; SHA-256 `{configuration_atlas['sha256']}`.",
            f"- Declared-sampling status: `{configuration_atlas['status']}`; coverage: `{json.dumps(configuration_atlas['coverage'], sort_keys=True)}`",
            "- Every stored node is re-evaluable against the embedded constraint graph. Proximity components and rank-drop labels remain finite numerical witnesses, not exhaustive branch enumeration, certified singularities, or a global topology proof.",
            "- Read one configuration chart and its exact node cards; run `verify-configuration-atlas` before trusting regeneration or stored witnesses.",
        ])
    lines.extend([
        "",
        "## Instantaneous motion effects",
        "",
        "The analytic geometric Jacobian maps independent joint rates to each selected target's linear and angular velocity relative to the root. Components below are expressed in the root orientation.",
        "",
    ])
    targets = canonical["kinematic_analysis"]["targets"]
    if not targets:
        lines.append("- No TCP, semantic group tip, SRDF chain tip, or end-effector parent was declared, so no target was selected automatically.")
    for target_frame, analysis in targets.items():
        jacobian = analysis["geometric_jacobian"]
        lines.extend([
            f"### Target `{target_frame}`",
            "",
            f"Selected by: {analysis['requested_by']}",
            "",
            "| Independent joint | Linear xyz per joint unit | Angular xyz per joint unit | Physical contributions |",
            "|---|---|---|---|",
        ])
        for column in jacobian["columns"]:
            physical = [record["joint"] for record in column["physical_contributions"]]
            lines.append(f"| `{column['joint']}` | {column['linear_xyz_per_joint_unit']} | {column['angular_xyz_per_joint_unit']} | {physical} |")
        workspace = analysis.get("sampled_workspace")
        if workspace:
            observed = workspace["observed_target_origin_aabb_in_root"]
            lines.extend([
                "",
                f"Sampled workspace observation ({workspace['sampling']['evaluated_sample_count']} deterministic samples): root-frame target-origin AABB min {observed['min_xyz_m']}, max {observed['max_xyz_m']}.",
                "This is finite-sample evidence, not a proof of the complete reachable set.",
                "",
            ])
    lines.extend([
        "",
        "## Frames in root",
        "",
        f"`{model.root_link}_from_frame` means the pose of the named frame expressed in `{model.root_link}`.",
        "",
        "| Frame | Semantic type | Parent frame | xyz (m) | quaternion xyzw |",
        "|---|---|---|---|---|",
    ])
    for name, frame in canonical["frames"].items():
        pose_record_value = frame["world_from_frame"]
        lines.append(f"| `{name}` | {frame['type']} | `{frame['parent_frame']}` | {pose_record_value['translation_xyz_m']} | {pose_record_value['quaternion_xyzw']} |")
    mesh_inspection = canonical["capabilities"]["mesh_content_inspection"]
    lines.extend([
        "",
        "## Declared geometry",
        "",
        f"- Mesh inspection requested kinds: `{mesh_inspection['requested_kinds']}`",
        f"- Complete for requested kinds: `{mesh_inspection['complete_for_requested_kinds']}`; complete for every declared mesh: `{mesh_inspection['complete_for_all_declared_meshes']}`",
        f"- Per-kind mesh evidence: `{json.dumps(mesh_inspection['by_kind'], sort_keys=True)}`",
        "- An unrequested or unsupported mesh remains `not_inspected`; evidence for one kind must not be generalized to the other.",
        "",
    ])
    geometry_count = 0
    for link_name, link in canonical["links"].items():
        for key in ("visuals", "collisions"):
            for geometry in link[key]:
                geometry_count += 1
                measured = canonical["geometry_analysis"][geometry["frame"]]
                if measured["status"] == "measured":
                    bounds_text = f"root AABB min {measured['bounds_in_root_frame_at_pose']['min_xyz_m']}, max {measured['bounds_in_root_frame_at_pose']['max_xyz_m']}; shape {measured['shape']['heuristic_label']}"
                else:
                    bounds_text = f"not measured: {measured['reason']}"
                lines.append(f"- `{geometry['frame']}` on `{link_name}`: `{geometry['geometry']['type']}`, local xyz {clean_vector(geometry['origin_xyz_m'])}, local rpy {clean_vector(geometry['origin_rpy_rad'])}; {bounds_text}; declaration `{json.dumps(geometry['geometry'], sort_keys=True)}`")
    if geometry_count == 0:
        lines.append("- No visual or collision geometry declared.")
    lines.extend([
        "",
        "## SRDF semantics",
        "",
    ])
    srdf = canonical["srdf"]
    if srdf["status"] == "not_provided":
        lines.append("- No SRDF provided.")
    else:
        for group_name, group in srdf["groups"].items():
            lines.append(f"- Group `{group_name}`: joints {group['expanded_joints']}; links {group['expanded_links']}")
        for pose_key, pose_record_value in srdf["named_poses"].items():
            lines.append(f"- Named pose `{pose_key}`: {pose_record_value['joints']}")
        for end_effector_name, end_effector in srdf["end_effectors"].items():
            lines.append(f"- End effector `{end_effector_name}`: parent link `{end_effector['parent_link']}`, component group `{end_effector['component_group']}`, parent group `{end_effector['parent_group']}`")
        lines.append(f"- Disabled collision pairs: {len(srdf['disabled_collisions'])}")
    lines.extend([
        "",
        "## Collision broad phase",
        "",
        f"- Complete for declared collision geometry: `{canonical['collision_broadphase']['complete_for_declared_collision_geometry']}`",
        f"- AABB overlap candidates: `{len(canonical['collision_broadphase']['overlap_pairs'])}`",
        "- These are conservative candidates, not triangle-level collision proof.",
    ])
    collision_surface = canonical["collision_surface"]
    lines.extend(["", "## Triangle surface and solid collision", ""])
    if collision_surface.get("status") == "not_requested":
        lines.append("- Not requested. Use `--surface-collisions` to verify AABB candidates with triangle surfaces and closed-solid containment.")
    else:
        lines.extend([
            f"- Self-collision status at this pose: `{collision_surface['self_collision_status']}`",
            f"- SRDF-policy-filtered status: `{collision_surface['srdf_policy_filtered_self_collision_status']}`; this filters disabled pairs but does not rewrite their physical geometry results.",
            f"- Contact tolerance: `{collision_surface['contact_tolerance_m']}` m",
            f"- AABB overlap / tolerance-added / exact / indeterminate candidates: `{collision_surface['aabb_overlap_candidate_count']}` / `{collision_surface['aabb_within_contact_tolerance_candidate_count']}` / `{collision_surface['exact_candidate_count']}` / `{collision_surface['indeterminate_candidate_count']}`",
            f"- Confirmed collision pairs: `{collision_surface['collision_pair_count']}`",
            f"- SRDF policy enabled / disabled physical collision pairs: `{collision_surface['srdf_policy_filtered_collision_pair_count']}` / `{collision_surface['srdf_disabled_physical_collision_pair_count']}`",
            "- A positive surface distance alone does not prove separation: one closed solid may contain another. Containment is tested only when both triangle surfaces are watertight with consistent winding.",
            "- SRDF disabled-collision entries are policy annotations; they do not change the physical geometry result.",
        ])
        for pair in collision_surface["candidate_results"]:
            if pair["status"] == "indeterminate":
                lines.append(f"- `{pair['geometry_a']}` ↔ `{pair['geometry_b']}`: `indeterminate`; {pair['reason']}")
            else:
                lines.append(
                    f"- `{pair['geometry_a']}` ↔ `{pair['geometry_b']}`: `{pair['status']}`; "
                    f"surface distance `{pair['surface_distance_m']}` m; containment `{bool(pair['containment'])}`; "
                    f"SRDF-disabled `{pair['disabled_by_srdf']}`"
                )
    lines.extend(["", "## Asserted semantic roles", ""])
    semantics = canonical["semantics"]
    if semantics["status"] == "not_provided":
        lines.append("- No semantic annotations provided. Do not infer base/planning/flange/TCP roles from names.")
    else:
        for frame_name, annotation in semantics["frames"].items():
            lines.append(f"- Frame `{frame_name}` roles: {annotation['roles']}; meaning: {annotation['meaning'] or 'not specified'}")
        for group_name, group in semantics["groups"].items():
            lines.append(f"- Group `{group_name}`: joints {group['joints']}, base `{group['base_frame']}`, tip `{group['tip_frame']}`")
        for end_effector_name, end_effector in semantics["end_effectors"].items():
            lines.append(f"- End effector `{end_effector_name}`: mount `{end_effector['mount_frame']}`, TCP `{end_effector['tcp_frame']}`")
    invariant_validation = canonical.get("invariant_validation", {"status": "not_provided"})
    lines.extend(["", "## Project spatial invariants", ""])
    if invariant_validation["status"] == "not_provided":
        lines.append("- No invariant contract provided. Project design intent is not enforced as an edit acceptance gate.")
    else:
        lines.extend([
            f"- Contract status: `{invariant_validation['status']}`",
            f"- Assertions passed / total: `{invariant_validation['passed_count']}` / `{invariant_validation['assertion_count']}`",
            "- These assertions encode project intent that must remain true after an edit; they are not inferred from URDF naming.",
        ])
        for invariant in invariant_validation["results"]:
            detail = f"; error: {invariant['error']}" if "error" in invariant else ""
            lines.append(f"- `{invariant['id']}` (`{invariant['type']}`, pose `{invariant['pose']}`): `{invariant['status']}`{detail}")
    render_atlas = canonical.get("artifacts", {}).get("semantic_render_atlas")
    lines.extend(["", "## Semantic visual grounding", ""])
    if not isinstance(render_atlas, dict):
        lines.append("- Not generated. Run `export --render` to create digest-bound front, side, top, and isometric semantic projections.")
    else:
        lines.extend([
            f"- Atlas: `render_atlas/{render_atlas['render_id']}`; manifest `{render_atlas['path']}`; SHA-256 `{render_atlas['manifest_sha256']}`.",
            f"- Model/pose input digest: `{render_atlas['render_input_sha256']}`; pose binding `{render_atlas['pose_binding']['sha256']}`.",
            f"- Rendered / declared geometry: `{render_atlas['coverage']['rendered_geometry_count']}` / `{render_atlas['coverage']['declared_geometry_count']}`; complete `{render_atlas['coverage']['complete_for_declared_geometry']}`.",
            "- Each `render_view/<render-id>/<view>` card preserves the root-to-UV projection, UV-to-pixel mapping, geometry hull/depth intervals, link origins, joint edges, typed SVG entity IDs, and artifact digest.",
            "- These are semantic convex projections from the same numeric geometry—not photorealistic visibility, an independent geometry oracle, a calibrated camera, or physical-world evidence.",
        ])
    lines.extend([
        "",
        "## Trust boundary",
        "",
        "Verified: tree structure, supported joint motion, robot-local frame transforms, axes, analytic geometric Jacobians, declared inertial tensors and aggregate mass properties when their status is `computed`, gravity-only generalized loads under the explicit convention when status is `computed`, embedded actuation/control declaration transcription and references, declared geometry origins, every geometry record marked `measured`, any SRDF record marked `parsed_and_validated`, every exact triangle-surface result explicitly reported above, and—when a scene is bound—the internal consistency and deterministic consequences of that exact scene/root/snapshot declaration.",
        "Approximate: finite-sample workspace observations. Not verified: physical agreement of URDF or scene declarations with current hardware/world state, completeness or freshness of scene objects, payload, unmodeled contacts, full dynamic response, controller/plugin/runtime/hardware behavior, transmission efficiency or calibration, the complete reachable set, closed loops, unsupported exact cylinder collision candidates, indeterminate pairs, or any mesh record marked `not_inspected`/`not_measured`.",
        "",
    ])
    evaluation = canonical.get("artifacts", {}).get("evaluation")
    if evaluation:
        lines.extend([
            "## Spatial understanding evaluation",
            "",
            f"- Public blind questions: `{evaluation['question_count']}`",
            f"- Capability counts: `{evaluation['capability_counts']}`",
            "- The private answer key is intentionally absent from this model/context. Keep it outside every candidate-readable filesystem and context surface.",
            "- A score measures only the generated artifact-conditioned competencies; it is not proof of unrestricted physical-world understanding.",
            "",
        ])
    warnings = canonical["validation"]["warnings"]
    if warnings:
        lines.extend(["## Warnings", ""] + [f"- {warning}" for warning in warnings] + [""])
    return "\n".join(lines)


def fact_records(model: RobotModel, canonical: dict[str, Any]) -> list[dict[str, Any]]:
    pose_name = canonical["pose"]["name"]
    urdf_sha256 = canonical["source"]["sha256"]
    facts: list[dict[str, Any]] = []

    def add(subject: str, predicate: str, object_value: Any, *, exact: bool, source_type: str, pose_dependent: bool = False, qualifiers: dict[str, Any] | None = None, source_sha256: str | None = None) -> None:
        core = {
            "schema_version": "robot-spatial-fact.v1",
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "qualifiers": {
                "pose": pose_name if pose_dependent else None,
                "pose_dependent": pose_dependent,
                **(qualifiers or {}),
            },
            "evidence": {
                "source_type": source_type,
                "source_sha256": source_sha256 or urdf_sha256,
                "exact": exact,
            },
        }
        digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()[:20]
        facts.append({"fact_id": f"fact-{digest}", **core})

    add(f"robot/{model.name}", "has_root_link", model.root_link, exact=True, source_type="urdf_declared")
    declared_mass_properties = canonical.get("physical_analysis", {}).get("declared_mass_properties", {})
    if declared_mass_properties:
        aggregate = {
            key: declared_mass_properties.get(key)
            for key in (
                "status",
                "selection",
                "expressed_in_frame",
                "coverage",
                "declared_mass_kg",
                "center_of_mass_in_expressed_frame_m",
                "inertia_about_center_of_mass_in_expressed_frame_kg_m2",
                "aggregate_principal_moments_kg_m2",
                "epistemic_scope",
            )
            if key in declared_mass_properties
        }
        add(
            f"robot/{model.name}",
            "has_declared_mass_properties",
            aggregate,
            exact=True,
            source_type="urdf_declared_inertials_forward_kinematics_parallel_axis_theorem",
            pose_dependent=True,
            qualifiers={
                "physical_world_completeness": "not_established",
                "absence_of_inertial_is_not_proof_of_zero_physical_mass": True,
            },
        )
    static_gravity = canonical.get("physical_analysis", {}).get(
        "declared_static_gravity_loads_under_root_frame_convention",
        {},
    )
    if static_gravity:
        gravity_summary = {
            key: static_gravity.get(key)
            for key in (
                "status",
                "selection",
                "pose",
                "gravity",
                "independent_driver_order",
                "coverage",
                "modeled_potential_energy_relative_to_root_origin_j",
                "sign_convention",
                "epistemic_scope",
            )
            if key in static_gravity
        }
        add(
            f"robot/{model.name}",
            "has_declared_static_gravity_load_model",
            gravity_summary,
            exact=True,
            source_type="urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
            pose_dependent=True,
            qualifiers={
                "gravity_frame": static_gravity.get("gravity", {}).get("expressed_in_frame"),
                "gravity_vector_xyz_m_s2": static_gravity.get("gravity", {}).get("vector_xyz_m_s2"),
                "physical_world_completeness": "not_established",
                "gravity_only_static_model": True,
            },
        )
        for driver, load in (static_gravity.get("independent_driver_loads") or {}).items():
            add(
                f"joint/{driver}",
                "has_declared_model_static_gravity_load",
                {
                    key: load[key]
                    for key in (
                        "joint_type",
                        "unit",
                        "generalized_gravity_force",
                        "ideal_static_holding_effort",
                        "declared_joint_effort_limit_magnitude",
                        "modeled_load_within_declared_joint_effort_limit_magnitude",
                    )
                },
                exact=True,
                source_type="urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
                pose_dependent=True,
                qualifiers={
                    "gravity": static_gravity["gravity"],
                    "sign_convention": static_gravity["sign_convention"],
                    "inertial_coverage": static_gravity["coverage"],
                    "does_not_establish_hardware_feasibility": True,
                },
            )
    for link_name, link in canonical["links"].items():
        inertial = link["inertial"]
        if inertial is None:
            add(
                f"link/{link_name}",
                "has_inertial_declaration_status",
                "not_provided",
                exact=True,
                source_type="urdf_declaration_absence",
                qualifiers={"does_not_prove_zero_physical_mass": True},
            )
            continue
        status = inertial["validation"]["status"]
        add(
            f"link/{link_name}",
            "has_inertial_declaration_status",
            status,
            exact=True,
            source_type="urdf_inertial_validation",
            qualifiers={"issues": inertial["validation"]["issues"]},
        )
        if status == "valid":
            add(
                f"link/{link_name}",
                "declares_mass_kg",
                inertial["mass_kg"],
                exact=True,
                source_type="urdf_declared",
                qualifiers={"inertial_frame": inertial["frame"]},
            )
            add(
                f"frame/{inertial['frame']}",
                "is_declared_center_of_mass_and_inertia_frame_of_link",
                link_name,
                exact=True,
                source_type="urdf_declared",
                qualifiers={
                    "link_from_inertial_translation_xyz_m": inertial["origin_xyz_m"],
                    "link_from_inertial_rpy_rad": inertial["origin_rpy_rad"],
                },
            )
            add(
                f"frame/{inertial['frame']}",
                "has_declared_inertia_tensor_kg_m2",
                tensor3_record(tensor3_from_urdf(inertial["inertia_kg_m2"])),
                exact=True,
                source_type="urdf_declared",
                qualifiers={"tensor_is_expressed_in_this_inertial_frame": True},
            )
    for joint_name, joint in canonical["joints"].items():
        add(
            f"joint/{joint_name}",
            "connects_links",
            {"parent": joint["parent_link"], "child": joint["child_link"], "joint_type": joint["type"]},
            exact=True,
            source_type="urdf_declared",
        )
        add(
            f"joint/{joint_name}",
            "has_embedded_actuation_declaration_binding",
            joint.get("actuation_declarations"),
            exact=True,
            source_type="expanded_urdf_actuation_declaration_transcription",
            qualifiers={"runtime_or_hardware_capability_established": False},
        )
        if joint.get("dynamics") is not None:
            add(
                f"joint/{joint_name}",
                "has_joint_dynamics_declaration",
                joint["dynamics"],
                exact=True,
                source_type="urdf_joint_dynamics_declaration",
                qualifiers={
                    "nonstandard_attributes_are_uninterpreted": True,
                    "does_not_establish_dynamic_response": True,
                },
            )
        if joint["axis_in_root_frame_at_pose"] is not None:
            add(
                f"joint/{joint_name}",
                "has_axis",
                joint["axis_in_root_frame_at_pose"],
                exact=True,
                source_type="forward_kinematics",
                pose_dependent=True,
                qualifiers={"expressed_in_frame": model.root_link, "unit": "unit_vector"},
            )
            if joint["mimic"] is None:
                effect = model.affected_by_joint(joint_name)
                for link_name in effect["affected_links"]:
                    add(
                        f"joint/{joint_name}",
                        "can_change_pose_of_link",
                        link_name,
                        exact=True,
                        source_type="kinematic_tree_and_mimic_derivation",
                        qualifiers={"other_independent_joint_positions_assumed_fixed": True, "physical_joints_driven": effect["physical_joints_driven"]},
                    )
            else:
                add(
                    f"joint/{joint_name}",
                    "is_driven_by_mimic_relation",
                    joint["mimic"],
                    exact=True,
                    source_type="urdf_declared",
                )
    actuation = canonical.get("actuation", {})
    for system_name, system in actuation.get("ros2_control_systems", {}).items():
        add(
            f"ros2_control_system/{system_name}",
            "declares_control_system",
            {
                "type": system["type"],
                "hardware": system["hardware"],
                "joints": sorted(system["joints"]),
                "sensors": sorted(system["sensors"]),
                "gpios": sorted(system["gpios"]),
            },
            exact=True,
            source_type="expanded_urdf_ros2_control_declaration",
            qualifiers={"runtime_or_hardware_capability_established": False},
        )
        for joint_name, component in system["joints"].items():
            add(
                f"joint/{joint_name}",
                "is_declared_in_ros2_control_system",
                {
                    "system": system_name,
                    "command_interfaces": component["command_interfaces"],
                    "state_interfaces": component["state_interfaces"],
                    "parameters": component["parameters"],
                },
                exact=True,
                source_type="expanded_urdf_ros2_control_declaration",
                qualifiers={"runtime_or_hardware_capability_established": False},
            )
        for component_type, entity_prefix in (("sensors", "control_sensor"), ("gpios", "control_gpio")):
            for component_name, component in system[component_type].items():
                add(
                    f"{entity_prefix}/{system_name}/{component_name}",
                    "is_declared_in_ros2_control_system",
                    {"system": system_name, **component},
                    exact=True,
                    source_type="expanded_urdf_ros2_control_declaration",
                    qualifiers={"runtime_or_hardware_capability_established": False},
                )
    for transmission_name, transmission in actuation.get("legacy_transmissions", {}).items():
        add(
            f"transmission/{transmission_name}",
            "declares_legacy_transmission",
            transmission,
            exact=True,
            source_type="expanded_urdf_legacy_transmission_declaration",
            qualifiers={"runtime_or_hardware_capability_established": False},
        )
        for actuator in transmission["actuators"]:
            add(
                f"actuator/{transmission_name}/{actuator['name']}",
                "is_declared_actuator_of_transmission",
                {"transmission": transmission_name, **actuator},
                exact=True,
                source_type="expanded_urdf_legacy_transmission_declaration",
                qualifiers={
                    "mechanical_reduction_is_transcribed_not_interpreted": True,
                    "runtime_or_hardware_capability_established": False,
                },
            )
    for frame_name, frame in canonical["frames"].items():
        add(
            f"frame/{frame_name}",
            "has_semantic_type",
            frame["type"],
            exact=True,
            source_type="urdf_frame_derivation",
        )
        add(
            f"frame/{frame_name}",
            "has_pose",
            frame["world_from_frame"],
            exact=True,
            source_type="forward_kinematics",
            pose_dependent=True,
            qualifiers={"transform": f"{model.root_link}_from_{frame_name}", "length_unit": "m", "quaternion_order": "xyzw"},
        )
    for target_frame, analysis in canonical["kinematic_analysis"]["targets"].items():
        jacobian = analysis["geometric_jacobian"]
        for column in jacobian["columns"]:
            qualifiers = {
                "target_frame": target_frame,
                "motion_relative_to_frame": jacobian["motion_relative_to_frame"],
                "components_expressed_in_orientation_of_frame": jacobian["components_expressed_in_orientation_of_frame"],
                "joint_position_unit": column["joint_position_unit"],
                "physical_contributions": column["physical_contributions"],
            }
            add(
                f"joint/{column['joint']}",
                "has_instantaneous_linear_effect_on",
                {"target_frame": target_frame, "xyz_per_joint_unit": column["linear_xyz_per_joint_unit"]},
                exact=True,
                source_type="analytic_geometric_jacobian",
                pose_dependent=True,
                qualifiers=qualifiers,
            )
            add(
                f"joint/{column['joint']}",
                "has_instantaneous_angular_effect_on",
                {"target_frame": target_frame, "xyz_per_joint_unit": column["angular_xyz_per_joint_unit"]},
                exact=True,
                source_type="analytic_geometric_jacobian",
                pose_dependent=True,
                qualifiers=qualifiers,
            )
        workspace = analysis.get("sampled_workspace")
        if workspace:
            add(
                f"frame/{target_frame}",
                "has_sampled_workspace_observation",
                workspace["observed_target_origin_aabb_in_root"],
                exact=False,
                source_type="deterministic_joint_space_sampling",
                qualifiers={
                    "root_frame": workspace["root_frame"],
                    "sample_count": workspace["sampling"]["evaluated_sample_count"],
                    "sample_sha256": workspace["sampling"]["sample_sha256"],
                    "does_not_prove_complete_reachable_set": True,
                },
            )
    semantics = canonical["semantics"]
    if semantics["status"] != "not_provided":
        semantic_sha256 = semantics["source"]["sha256"]
        for frame_name, annotation in semantics["frames"].items():
            for role in annotation["roles"]:
                add(f"frame/{frame_name}", "has_asserted_role", role, exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
        for group_name, group in semantics["groups"].items():
            for joint_name in group["joints"]:
                add(f"group/{group_name}", "contains_joint", joint_name, exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
            if group["base_frame"] is not None:
                add(f"group/{group_name}", "has_base_frame", group["base_frame"], exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
            if group["tip_frame"] is not None:
                add(f"group/{group_name}", "has_tip_frame", group["tip_frame"], exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
        for end_effector_name, end_effector in semantics["end_effectors"].items():
            add(f"end_effector/{end_effector_name}", "has_mount_frame", end_effector["mount_frame"], exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
            add(f"end_effector/{end_effector_name}", "has_tcp_frame", end_effector["tcp_frame"], exact=False, source_type="user_or_project_semantic_assertion", source_sha256=semantic_sha256)
    srdf = canonical["srdf"]
    if srdf["status"] != "not_provided":
        srdf_sha256 = srdf["source"]["sha256"]
        for group_name, group in srdf["groups"].items():
            for joint_name in group["expanded_joints"]:
                add(f"srdf_group/{group_name}", "contains_joint", joint_name, exact=True, source_type="srdf_declared_and_validated", source_sha256=srdf_sha256)
            for link_name in group["expanded_links"]:
                add(f"srdf_group/{group_name}", "contains_link", link_name, exact=True, source_type="srdf_declared_and_validated", source_sha256=srdf_sha256)
        for pose_key, pose_record_value in srdf["named_poses"].items():
            add(f"srdf_group/{pose_record_value['group']}", "has_named_pose", {"name": pose_key, "joints": pose_record_value["joints"]}, exact=True, source_type="srdf_declared_and_validated", source_sha256=srdf_sha256)
        for end_effector_name, end_effector in srdf["end_effectors"].items():
            add(f"srdf_end_effector/{end_effector_name}", "has_parent_link", end_effector["parent_link"], exact=True, source_type="srdf_declared_and_validated", source_sha256=srdf_sha256)
            add(f"srdf_end_effector/{end_effector_name}", "uses_component_group", end_effector["component_group"], exact=True, source_type="srdf_declared_and_validated", source_sha256=srdf_sha256)
    for geometry_frame, geometry in canonical["geometry_analysis"].items():
        if geometry["status"] != "measured":
            add(f"frame/{geometry_frame}", "geometry_measurement_status", "not_inspected", exact=True, source_type="pipeline_status")
            continue
        mesh_sha256 = geometry.get("source", {}).get("sha256")
        source_type = "measured_mesh" if geometry["geometry_type"] == "mesh" else "analytic_urdf_primitive"
        add(
            f"frame/{geometry_frame}",
            "has_root_frame_aabb",
            geometry["bounds_in_root_frame_at_pose"],
            exact=True,
            source_type=source_type,
            pose_dependent=True,
            qualifiers={"expressed_in_frame": model.root_link, "length_unit": "m"},
            source_sha256=mesh_sha256,
        )
        add(
            f"frame/{geometry_frame}",
            "has_shape_class",
            geometry["shape"]["heuristic_label"],
            exact=False,
            source_type="extent_ratio_heuristic",
            qualifiers={"heuristic_only": True, "ratios": geometry["shape"]["sorted_extent_ratios"]},
            source_sha256=mesh_sha256,
        )
    for overlap in canonical["collision_broadphase"]["overlap_pairs"]:
        add(
            f"frame/{overlap['geometry_a']}",
            "has_aabb_overlap_candidate_with",
            overlap["geometry_b"],
            exact=True,
            source_type="aabb_broadphase",
            pose_dependent=True,
            qualifiers={"does_not_prove_triangle_collision": True, "intersection_extents_xyz_m": overlap["intersection_extents_xyz_m"]},
        )
    collision_surface = canonical["collision_surface"]
    if collision_surface.get("status") != "not_requested":
        add(
            f"robot/{model.name}",
            "has_self_collision_status",
            collision_surface["self_collision_status"],
            exact=True,
            source_type="triangle_surface_and_closed_solid_analysis",
            pose_dependent=True,
            qualifiers={
                "contact_tolerance_m": collision_surface["contact_tolerance_m"],
                "indeterminate_candidate_count": collision_surface["indeterminate_candidate_count"],
                "same_link_pairs_excluded": True,
            },
        )
        if collision_surface["srdf_policy_provided"]:
            add(
                f"robot/{model.name}",
                "has_srdf_policy_filtered_self_collision_status",
                collision_surface["srdf_policy_filtered_self_collision_status"],
                exact=True,
                source_type="triangle_surface_analysis_with_srdf_policy_filter",
                pose_dependent=True,
                qualifiers={
                    "physical_self_collision_status": collision_surface["self_collision_status"],
                    "enabled_collision_pair_count": collision_surface["srdf_policy_filtered_collision_pair_count"],
                    "disabled_physical_collision_pair_count": collision_surface["srdf_disabled_physical_collision_pair_count"],
                    "policy_is_annotation_not_physical_geometry": True,
                },
            )
        for frame_name, surface in collision_surface["geometry_surfaces"].items():
            add(
                f"frame/{frame_name}",
                "has_triangle_surface_status",
                surface["status"],
                exact=True,
                source_type="triangle_surface_construction_status",
                pose_dependent=True,
                qualifiers={key: value for key, value in surface.items() if key not in {"frame", "kind", "link", "status"}},
            )
        for pair in collision_surface["candidate_results"]:
            if "surface_distance_m" not in pair:
                continue
            qualifiers = {
                "contact_tolerance_m": pair["contact_tolerance_m"],
                "within_contact_tolerance": pair["within_contact_tolerance"],
                "containment": pair["containment"],
                "containment_classification_complete": pair["containment_classification_complete"],
                "witness_point_subject_in_root_m": pair["witness_point_a_in_root_m"],
                "witness_point_object_in_root_m": pair["witness_point_b_in_root_m"],
                "disabled_by_srdf": pair["disabled_by_srdf"],
                "srdf_disable_reason": pair["srdf_disable_reason"],
                "length_unit": "m",
            }
            add(
                f"frame/{pair['geometry_a']}",
                "has_triangle_surface_distance_to",
                {"geometry_frame": pair["geometry_b"], "distance_m": pair["surface_distance_m"]},
                exact=True,
                source_type="deterministic_triangle_bvh",
                pose_dependent=True,
                qualifiers=qualifiers,
            )
            if pair["status"] == "collision":
                predicate = "has_verified_collision_with"
            elif pair["status"] == "collision_free":
                predicate = "has_verified_no_collision_with"
            else:
                continue
            add(
                f"frame/{pair['geometry_a']}",
                predicate,
                pair["geometry_b"],
                exact=True,
                source_type="triangle_surface_and_closed_solid_analysis",
                pose_dependent=True,
                qualifiers=qualifiers,
            )
    render_atlas = canonical.get("artifacts", {}).get("semantic_render_atlas")
    if isinstance(render_atlas, dict):
        render_id = render_atlas["render_id"]
        manifest_sha256 = render_atlas["manifest_sha256"]
        render_qualifiers = {
            "render_id": render_id,
            "render_input_sha256": render_atlas["render_input_sha256"],
            "pose_binding_sha256": render_atlas["pose_binding"]["sha256"],
            "root_frame": model.root_link,
            "same_canonical_geometry_not_independent_oracle": True,
            "visibility_occlusion_perspective_and_physical_camera": "not_established",
        }
        add(
            f"render_atlas/{render_id}",
            "binds_semantic_views_to_model_and_pose",
            {
                "manifest": render_atlas["path"],
                "manifest_sha256": manifest_sha256,
                "model_semantic_sha256": model.semantic_sha256,
                "pose_binding": render_atlas["pose_binding"],
                "coverage": render_atlas["coverage"],
                "view_ids": sorted(render_atlas["views"]),
            },
            exact=True,
            source_type="deterministic_semantic_render_atlas_manifest",
            pose_dependent=True,
            qualifiers=render_qualifiers,
            source_sha256=manifest_sha256,
        )
        for view_id, view in sorted(render_atlas["views"].items()):
            view_entity = f"render_view/{render_id}/{view_id}"
            view_qualifiers = {**render_qualifiers, "view_id": view_id, "view_entity": view_entity}
            add(
                view_entity,
                "has_orthographic_projection_contract",
                {
                    "projection": view["projection"],
                    "screen": view["screen"],
                    "scene_projection_bounds_uv_m": view["scene_projection_bounds_uv_m"],
                    "artifact": view["artifact"],
                },
                exact=True,
                source_type="deterministic_root_xyz_to_uv_to_pixel_projection",
                pose_dependent=True,
                qualifiers=view_qualifiers,
                source_sha256=manifest_sha256,
            )
            for frame_projection in view["link_frames"]:
                add(
                    frame_projection["entity_id"],
                    "has_projected_origin_in_render_view",
                    {
                        "view_entity": view_entity,
                        "origin_root_xyz_m": frame_projection["origin_root_xyz_m"],
                        "projected_uv_m": frame_projection["projected_uv_m"],
                        "pixel_xy": frame_projection["pixel_xy"],
                    },
                    exact=True,
                    source_type="deterministic_frame_origin_orthographic_projection",
                    pose_dependent=True,
                    qualifiers=view_qualifiers,
                    source_sha256=manifest_sha256,
                )
            for edge in view["kinematic_edges"]:
                add(
                    edge["entity_id"],
                    "has_projected_kinematic_edge_in_render_view",
                    {
                        key: edge[key]
                        for key in (
                            "view_entity",
                            "parent_entity",
                            "child_entity",
                            "start_root_xyz_m",
                            "end_root_xyz_m",
                            "start_uv_m",
                            "end_uv_m",
                            "start_pixel_xy",
                            "end_pixel_xy",
                            "length_3d_m",
                            "projected_length_m",
                            "pixel_length",
                        )
                        if key in edge
                    } | {"view_entity": view_entity},
                    exact=True,
                    source_type="deterministic_kinematic_edge_orthographic_projection",
                    pose_dependent=True,
                    qualifiers=view_qualifiers,
                    source_sha256=manifest_sha256,
                )
            for geometry in view["geometry"]:
                hull_exact = geometry["projection_support"]["convex_hull"].startswith("exact_")
                add(
                    geometry["entity_id"],
                    "has_projected_geometry_hull_in_render_view",
                    {
                        "view_entity": view_entity,
                        "kind": geometry["kind"],
                        "geometry_type": geometry["geometry_type"],
                        "projection_bounds_uv_m": geometry["projection_bounds_uv_m"],
                        "pixel_bounds_xy": geometry["pixel_bounds_xy"],
                        "depth_interval_m": geometry["depth_interval_m"],
                        "projection_support": geometry["projection_support"],
                    },
                    exact=hull_exact,
                    source_type=(
                        "deterministic_exact_vertex_set_convex_projection"
                        if hull_exact
                        else "deterministic_sampled_curve_convex_projection"
                    ),
                    pose_dependent=True,
                    qualifiers=view_qualifiers,
                    source_sha256=manifest_sha256,
                )
    articulation = canonical.get("artifacts", {}).get("articulation_grammar")
    if isinstance(articulation, dict):
        grammar_id = articulation["grammar_id"]
        grammar_sha256 = articulation["sha256"]
        grammar_qualifiers = {
            "grammar_id": grammar_id,
            "grammar_input_sha256": articulation["grammar_input_sha256"],
            "canonical_law_id": articulation["law_identity"]["canonical_law_id"],
            "canonical_law_sha256": articulation["law_identity"]["canonical_law_sha256"],
            "pose_independent": True,
            "standalone_executable": True,
            "physical_truth": "not_established",
        }
        add(
            f"articulation_grammar/{grammar_id}",
            "binds_pose_independent_joint_laws_and_frame_compositions",
            {
                "path": articulation["path"],
                "sha256": grammar_sha256,
                "law_identity": articulation["law_identity"],
                "source_binding": articulation["source_binding"],
                "coordinate_contract": articulation["coordinate_contract"],
                "language_contract": articulation["language_contract"],
                "layer_contract": articulation["layer_contract"],
                "coverage": articulation["coverage"],
            },
            exact=True,
            source_type="deterministic_typed_articulation_grammar",
            qualifiers=grammar_qualifiers,
            source_sha256=grammar_sha256,
        )
        add(
            f"articulation_grammar/{grammar_id}",
            "separates_source_binding_from_canonical_kinematic_law_identity",
            {
                "source_binding": articulation["source_binding"],
                "law_identity": articulation["law_identity"],
                "cross_representation_policy": "compare source-binding-free law projections; use explicit digest-bound typed identifier correspondence when names differ",
            },
            exact=True,
            source_type="deterministic_source_neutral_law_projection_hash",
            qualifiers=grammar_qualifiers,
            source_sha256=grammar_sha256,
        )
        for driver, variable in sorted(articulation["independent_variables"].items()):
            variable_entity = f"articulation_variable/{grammar_id}/{driver}"
            add(
                variable_entity,
                "has_mimic_constrained_independent_domain",
                variable,
                exact=True,
                source_type="source_declared_limits_and_supported_dependency_constraint_intersection",
                qualifiers={**grammar_qualifiers, "driver_joint": driver},
                source_sha256=grammar_sha256,
            )
        for joint, operator in sorted(articulation["joint_operators"].items()):
            operator_entity = f"articulation_operator/{grammar_id}/{joint}"
            add(
                operator_entity,
                "has_typed_parameterized_joint_operator",
                {
                    "operator": operator,
                    "joint_position_rule": articulation["joint_position_rules"][joint],
                },
                exact=True,
                source_type="source_anchor_axis_type_and_dependency_law_normalization",
                qualifiers={**grammar_qualifiers, "joint": joint},
                source_sha256=grammar_sha256,
            )
        for frame, derivation in sorted(articulation["frame_derivations"].items()):
            derivation_entity = f"articulation_derivation/{grammar_id}/{frame}"
            add(
                derivation_entity,
                "has_ordered_root_from_frame_composition",
                derivation,
                exact=True,
                source_type="validated_tree_path_and_typed_joint_operator_composition",
                qualifiers={**grammar_qualifiers, "frame": frame},
                source_sha256=grammar_sha256,
            )
    constraint_graph = canonical.get("artifacts", {}).get("constraint_graph")
    if isinstance(constraint_graph, dict):
        graph_id = constraint_graph["constraint_graph_id"]
        graph_sha256 = constraint_graph["sha256"]
        evaluation = constraint_graph["evaluation"]
        graph_qualifiers = {
            "constraint_graph_id": graph_id,
            "constraint_graph_sha256": constraint_graph["constraint_graph_sha256"],
            "articulation_grammar_sha256": constraint_graph["source_binding"]["articulation_grammar_sha256"],
            "constraint_spec_sha256": constraint_graph["source_binding"]["constraint_spec_sha256"],
            "supplemental_constraints_are_asserted": True,
            "physical_truth": "not_established",
        }
        graph_entity = f"constraint_graph/{graph_id}"
        add(
            graph_entity,
            "binds_spanning_tree_to_supplemental_mechanism_constraints",
            {
                "path": constraint_graph["path"],
                "sha256": graph_sha256,
                "source_binding": constraint_graph["source_binding"],
                "structural_graph": constraint_graph["structural_graph"],
                "executable_contract": constraint_graph["executable_contract"],
                "coverage": constraint_graph["coverage"],
            },
            exact=True,
            source_type="deterministic_compilation_of_digest_bound_articulation_and_asserted_constraint_spec",
            qualifiers=graph_qualifiers,
            source_sha256=graph_sha256,
        )
        add(
            graph_entity,
            "tree_is_parameterization_not_complete_mechanism",
            constraint_graph["structural_graph"]["tree_is_parameterization_not_complete_mechanism"],
            exact=True,
            source_type="deterministic_constraint_graph_topology",
            qualifiers={
                **graph_qualifiers,
                "declared_cycle_count": constraint_graph["coverage"]["declared_cycle_count"],
                "coordinate_constraint_count": constraint_graph["coverage"]["coordinate_constraint_count"],
            },
            source_sha256=graph_sha256,
        )
        attachment_evaluations = evaluation.get("attachments", {})
        for attachment in constraint_graph["attachments"]:
            attachment_entity = f"attachment/{graph_id}/{attachment['attachment_id']}"
            add(
                attachment_entity,
                "has_asserted_rigid_mechanism_attachment",
                attachment,
                exact=False,
                source_type="digest_bound_supplemental_mechanism_assertion",
                qualifiers=graph_qualifiers,
                source_sha256=constraint_graph["source_binding"]["constraint_spec_sha256"],
            )
            add(
                attachment_entity,
                "has_pose_at_constraint_export_binding",
                attachment_evaluations.get(attachment["frame_id"]),
                exact=True,
                source_type="deterministic_articulation_and_rigid_attachment_composition",
                pose_dependent=True,
                qualifiers={**graph_qualifiers, "pose": evaluation["pose"]},
                source_sha256=graph_sha256,
            )
        evaluation_by_constraint = {
            record["constraint_id"]: record
            for record in evaluation.get("constraints", [])
        }
        for constraint in constraint_graph["constraints"]:
            constraint_entity = f"constraint/{graph_id}/{constraint['constraint_id']}"
            add(
                constraint_entity,
                "has_asserted_mechanism_constraint",
                constraint,
                exact=False,
                source_type="digest_bound_supplemental_mechanism_assertion",
                qualifiers=graph_qualifiers,
                source_sha256=constraint_graph["source_binding"]["constraint_spec_sha256"],
            )
            add(
                constraint_entity,
                "has_typed_residual_at_constraint_export_binding",
                evaluation_by_constraint[constraint["constraint_id"]],
                exact=True,
                source_type="deterministic_typed_constraint_residual_evaluation",
                pose_dependent=True,
                qualifiers={**graph_qualifiers, "pose": evaluation["pose"]},
                source_sha256=graph_sha256,
            )
        local_analysis = evaluation.get("local_constraint_analysis")
        if isinstance(local_analysis, dict):
            add(
                graph_entity,
                "has_pose_conditioned_numerical_local_mobility_estimate",
                local_analysis,
                exact=True,
                source_type="normalized_residual_finite_difference_jacobian_rank",
                pose_dependent=True,
                qualifiers={
                    **graph_qualifiers,
                    "pose": evaluation["pose"],
                    "global_mechanism_dof_proof": False,
                    "rank_may_change_at_singularities": True,
                },
                source_sha256=graph_sha256,
            )
    configuration_atlas = canonical.get("artifacts", {}).get("configuration_atlas")
    if isinstance(configuration_atlas, dict):
        atlas_id = configuration_atlas["configuration_atlas_id"]
        atlas_sha256 = configuration_atlas["sha256"]
        atlas_entity = f"configuration_atlas/{atlas_id}"
        atlas_qualifiers = {
            "configuration_atlas_id": atlas_id,
            "constraint_graph_artifact_sha256": configuration_atlas["source_binding"]["constraint_graph_artifact_sha256"],
            "configuration_atlas_spec_sha256": configuration_atlas["source_binding"]["configuration_atlas_spec_sha256"],
            "finite_declared_sampling_only": True,
            "exhaustive_branch_enumeration": False,
            "certified_global_topology": False,
            "physical_truth": "not_established",
        }
        add(
            atlas_entity,
            "binds_finite_configuration_witnesses_to_constraint_graph_and_chart_contract",
            {
                "path": configuration_atlas["path"],
                "sha256": atlas_sha256,
                "status": configuration_atlas["status"],
                "source_binding": configuration_atlas["source_binding"],
                "coverage": configuration_atlas["coverage"],
                "chart_entities": [
                    f"configuration_chart/{atlas_id}/{chart['chart_id']}"
                    for chart in configuration_atlas["charts"]
                ],
            },
            exact=True,
            source_type="deterministic_digest_bound_multi_seed_configuration_atlas",
            qualifiers=atlas_qualifiers,
            source_sha256=atlas_sha256,
        )
        for chart in configuration_atlas["charts"]:
            chart_entity = f"configuration_chart/{atlas_id}/{chart['chart_id']}"
            add(
                chart_entity,
                "has_explicit_finite_one_parameter_exploration_contract",
                {
                    key: chart[key]
                    for key in (
                        "parameter_driver",
                        "parameter_values",
                        "solve_for",
                        "driver_scales",
                        "seed_contracts",
                        "solution_merge_tolerance_normalized",
                        "continuation_edge_max_distance_normalized",
                        "minimum_solutions_per_sample",
                        "periodic_driver_metric",
                        "observed_rank_reference",
                        "coverage",
                    )
                },
                exact=False,
                source_type="digest_bound_declared_sampling_contract_and_finite_numerical_exploration",
                qualifiers=atlas_qualifiers,
                source_sha256=atlas_sha256,
            )
            nodes = [node for sample in chart["samples"] for node in sample["solutions"]]
            for node in nodes:
                node_entity = f"configuration_node/{atlas_id}/{chart['chart_id']}/{node['sample_index']:04d}/{node['node_id'].split('/')[-1]}"
                add(
                    node_entity,
                    "is_executable_constraint_satisfying_configuration_witness",
                    {
                        "chart": chart_entity,
                        "parameter_driver": node["parameter_driver"],
                        "parameter_value": node["parameter_value"],
                        "independent_driver_positions": node["independent_driver_positions"],
                        "constraint_status": node["constraint_status"],
                        "maximum_normalized_abs": node["maximum_normalized_abs"],
                        "full_constraint_jacobian": node["full_constraint_jacobian"],
                        "chart_passive_jacobian": node["chart_passive_jacobian"],
                        "singularity_witness": node["singularity_witness"],
                    },
                    exact=True,
                    source_type="local_solve_followed_by_exact_standalone_constraint_evaluation_and_numerical_jacobian_diagnostics",
                    qualifiers={
                        **atlas_qualifiers,
                        "rank_drop_is_relative_to_maximum_observed_in_declared_chart": True,
                        "certified_singularity": False,
                    },
                    source_sha256=atlas_sha256,
                )
            for component in chart["witness_components"]:
                component_entity = f"configuration_component/{atlas_id}/{chart['chart_id']}/{component['component_id'].split('/')[-1]}"
                add(
                    component_entity,
                    "groups_configuration_nodes_by_declared_proximity_witness_edges",
                    {
                        "chart": chart_entity,
                        "node_ids": component["node_ids"],
                        "edge_policy": "adjacent-sample and same-sample-singularity proximity under the declared normalized metric",
                    },
                    exact=False,
                    source_type="finite_sample_configuration_proximity_graph",
                    qualifiers={
                        **atlas_qualifiers,
                        "topological_branch_certificate": False,
                    },
                    source_sha256=atlas_sha256,
                )
    motion_atlas = canonical.get("artifacts", {}).get("counterfactual_motion_atlas")
    if isinstance(motion_atlas, dict):
        motion_id = motion_atlas["motion_id"]
        manifest_sha256 = motion_atlas["manifest_sha256"]
        atlas_qualifiers = {
            "motion_id": motion_id,
            "motion_input_sha256": motion_atlas["motion_input_sha256"],
            "baseline_pose_binding_sha256": motion_atlas["baseline_pose_binding"]["sha256"],
            "other_independent_drivers_held_fixed": True,
            "finite_endpoints_not_continuous_trajectory": True,
            "same_fk_and_geometry_not_independent_oracle": True,
        }
        add(
            f"motion_atlas/{motion_id}",
            "binds_counterfactual_joint_causes_to_finite_endpoint_effects",
            {
                "manifest": motion_atlas["path"],
                "manifest_sha256": manifest_sha256,
                "model_semantic_sha256": model.semantic_sha256,
                "baseline_pose_binding": motion_atlas["baseline_pose_binding"],
                "perturbation_policy": motion_atlas["perturbation_policy"],
                "coverage": motion_atlas["coverage"],
                "driver_entities": [
                    f"motion_driver/{motion_id}/{driver}" for driver in sorted(motion_atlas["drivers"])
                ],
            },
            exact=True,
            source_type="deterministic_counterfactual_motion_atlas_manifest",
            pose_dependent=True,
            qualifiers=atlas_qualifiers,
            source_sha256=manifest_sha256,
        )
        for driver, driver_record in sorted(motion_atlas["drivers"].items()):
            driver_entity = f"motion_driver/{motion_id}/{driver}"
            driver_qualifiers = {**atlas_qualifiers, "motion_driver_entity": driver_entity, "driver_joint": driver}
            add(
                driver_entity,
                "has_independent_driver_counterfactual_contract",
                {
                    key: driver_record[key]
                    for key in (
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
                },
                exact=True,
                source_type="declared_limits_mimic_affine_constraints_and_kinematic_tree",
                pose_dependent=True,
                qualifiers=driver_qualifiers,
                source_sha256=manifest_sha256,
            )
            for direction in ("minus", "plus"):
                endpoint = driver_record["endpoints"][direction]
                endpoint_summary = {
                    key: endpoint[key]
                    for key in (
                        "status",
                        "requested_delta",
                        "applied_delta",
                        "joint_position",
                        "joint_position_unit",
                        "physical_joint_positions",
                        "link_frame_deltas",
                        "causality_check",
                    )
                    if key in endpoint
                }
                add(
                    f"joint/{driver}",
                    "has_signed_finite_counterfactual_endpoint_effect",
                    {"direction": direction, **endpoint_summary},
                    exact=True,
                    source_type="finite_endpoint_forward_kinematics_and_frame_delta",
                    pose_dependent=True,
                    qualifiers={**driver_qualifiers, "direction": direction},
                    source_sha256=manifest_sha256,
                )
            for view_id, view in sorted(driver_record["views"].items()):
                view_entity = f"motion_view/{motion_id}/{driver}/{view_id}"
                add(
                    view_entity,
                    "has_shared_screen_counterfactual_projection_contract",
                    {
                        "motion_driver_entity": driver_entity,
                        "projection": view["projection"],
                        "screen": view["screen"],
                        "combined_projection_bounds_uv_m": view["combined_projection_bounds_uv_m"],
                        "motion_vectors": view["motion_vectors"],
                        "artifact": view["artifact"],
                    },
                    exact=True,
                    source_type="deterministic_finite_endpoint_root_xyz_to_shared_uv_pixel_projection",
                    pose_dependent=True,
                    qualifiers={
                        **driver_qualifiers,
                        "view_id": view_id,
                        "motion_view_entity": view_entity,
                        "continuous_swept_volume_or_intermediate_path": "not_established",
                    },
                    source_sha256=manifest_sha256,
                )
    world_scene = canonical.get("world_scene", {})
    if world_scene.get("status") == "parsed_validated_and_bound":
        scene_sha256 = world_scene["source"]["sha256"]
        snapshot_qualifiers = {
            "scene_id": world_scene["scene_id"],
            "snapshot_id": world_scene["snapshot"]["id"],
            "time_semantics": world_scene["snapshot"]["time_semantics"],
            "physical_world_completeness": "not_established",
            "scene_currency_and_accuracy_independently_verified": False,
        }
        instance = world_scene["robot_mount"]
        add(
            f"robot_instance/{instance['instance_id']}",
            "mounts_robot_root_in_scene",
            {
                "robot": model.name,
                "root_link": model.root_link,
                "parent_entity": instance["parent_entity"],
                "root_entity": instance["root_entity"],
                "world_from_robot_root": instance["world_from_robot_root"],
                "placement_source": instance["source"],
            },
            exact=True,
            source_type="declared_world_scene_root_mount",
            pose_dependent=False,
            qualifiers=snapshot_qualifiers,
            source_sha256=scene_sha256,
        )
        add(
            f"robot/{model.name}",
            "is_bound_to_world_scene_snapshot",
            {
                "instance_id": instance["instance_id"],
                "scene_id": world_scene["scene_id"],
                "snapshot": world_scene["snapshot"],
                "world_frame": world_scene["world_frame"],
            },
            exact=True,
            source_type="declared_world_scene_binding",
            qualifiers=snapshot_qualifiers,
            source_sha256=scene_sha256,
        )
        scene_frame_names = [world_scene["world_frame"], *sorted(world_scene.get("scene_frames", {}))]
        for frame_name in scene_frame_names:
            entity_id = f"scene_frame/{frame_name}"
            add(
                entity_id,
                "has_pose_in_scene_world",
                world_scene["typed_frame_poses_in_world"][entity_id],
                exact=True,
                source_type="validated_world_scene_frame_graph",
                qualifiers={
                    **snapshot_qualifiers,
                    "transform": f"scene_frame/{world_scene['world_frame']}_from_{entity_id}",
                    "length_unit": "m",
                    "quaternion_order": "xyzw",
                },
                source_sha256=scene_sha256,
            )
        for object_id, object_record in world_scene.get("objects", {}).items():
            object_entity = f"scene_object/{object_id}"
            add(
                object_entity,
                "has_pose_in_scene_world",
                object_record["world_from_object"],
                exact=True,
                source_type="validated_world_scene_frame_graph",
                qualifiers={**snapshot_qualifiers, "placement_source": object_record["source"]},
                source_sha256=scene_sha256,
            )
            add(
                object_entity,
                "has_declared_scene_semantics",
                object_record["semantics"],
                exact=False,
                source_type="world_scene_semantic_assertion",
                qualifiers=snapshot_qualifiers,
                source_sha256=scene_sha256,
            )
        scene_collision = world_scene["robot_environment_collision"]
        add(
            f"robot_instance/{instance['instance_id']}",
            "has_robot_environment_collision_status",
            {
                "status": scene_collision["status"],
                "minimum_separation": scene_collision["minimum_separation"],
                "coverage": scene_collision["coverage"],
            },
            exact=True,
            source_type="snapshot_bound_robot_environment_solid_analysis",
            pose_dependent=True,
            qualifiers={
                **snapshot_qualifiers,
                "contact_tolerance_m": scene_collision["contact_tolerance_m"],
                "epistemic_scope": scene_collision["epistemic_scope"],
            },
            source_sha256=scene_sha256,
        )
        for geometry_id, geometry in scene_collision["geometry_analysis"].items():
            if not (geometry_id.startswith("scene_geometry/") or geometry_id.startswith("robot_geometry/")):
                continue
            if geometry["status"] == "measured":
                add(
                    geometry_id,
                    "has_world_frame_aabb_at_snapshot",
                    geometry["bounds_in_world_frame_at_snapshot"],
                    exact=True,
                    source_type=(
                        ("measured_scene_mesh" if geometry["geometry_type"] == "mesh" else "analytic_scene_primitive")
                        if geometry_id.startswith("scene_geometry/")
                        else ("measured_robot_mesh_in_scene" if geometry["geometry_type"] == "mesh" else "analytic_robot_primitive_in_scene")
                    ),
                    qualifiers=snapshot_qualifiers,
                    source_sha256=geometry.get("source", {}).get("sha256") or scene_sha256,
                )
            else:
                add(
                    geometry_id,
                    "geometry_measurement_status",
                    geometry["status"],
                    exact=True,
                    source_type="pipeline_status",
                    qualifiers={**snapshot_qualifiers, "reason": geometry.get("reason")},
                    source_sha256=scene_sha256,
                )
        for pair in scene_collision["pair_results"]:
            pair_payload = {
                key: pair.get(key)
                for key in (
                    "robot_geometry",
                    "robot_link",
                    "environment_geometry",
                    "environment_object",
                    "status",
                    "separation_m",
                    "separation_lower_bound_m",
                    "surface_distance_m",
                    "method",
                    "reason",
                )
                if key in pair
            }
            pair_qualifiers = {
                **snapshot_qualifiers,
                "contact_tolerance_m": pair["contact_tolerance_m"],
                "representation_robot": pair.get("representation_robot"),
                "representation_environment": pair.get("representation_environment"),
            }
            add(
                pair["environment_geometry"],
                "has_robot_environment_pair_result",
                pair_payload,
                exact=True,
                source_type="snapshot_bound_robot_environment_pair_analysis",
                pose_dependent=True,
                qualifiers=pair_qualifiers,
                source_sha256=scene_sha256,
            )
            add(
                pair["robot_geometry"],
                "has_robot_environment_pair_result",
                pair_payload,
                exact=True,
                source_type="snapshot_bound_robot_environment_pair_analysis",
                pose_dependent=True,
                qualifiers=pair_qualifiers,
                source_sha256=scene_sha256,
            )
        scene_gravity = canonical.get("physical_analysis", {}).get("declared_static_gravity_loads_under_scene_gravity", {})
        if scene_gravity.get("status") != "not_provided":
            add(
                f"robot_instance/{instance['instance_id']}",
                "has_scene_gravity_load_model",
                {
                    "status": scene_gravity["status"],
                    "scene_gravity": scene_gravity["scene_gravity"],
                    "load_coverage": None if scene_gravity.get("loads") is None else scene_gravity["loads"]["coverage"],
                    "sign_convention": None if scene_gravity.get("loads") is None else scene_gravity["loads"]["sign_convention"],
                },
                exact=True,
                source_type="scene_root_gravity_conversion_and_urdf_inertial_projection",
                pose_dependent=True,
                qualifiers={**snapshot_qualifiers, "physical_truth_conditional_on_scene_provenance": True},
                source_sha256=scene_sha256,
            )
            for driver, load in ((scene_gravity.get("loads") or {}).get("independent_driver_loads") or {}).items():
                add(
                    f"joint/{driver}",
                    "has_scene_bound_static_gravity_load",
                    {
                        key: load[key]
                        for key in (
                            "joint_type",
                            "unit",
                            "generalized_gravity_force",
                            "ideal_static_holding_effort",
                            "declared_joint_effort_limit_magnitude",
                            "modeled_load_within_declared_joint_effort_limit_magnitude",
                        )
                    },
                    exact=True,
                    source_type="scene_root_gravity_conversion_and_urdf_inertial_projection",
                    pose_dependent=True,
                    qualifiers={
                        **snapshot_qualifiers,
                        "gravity_in_robot_root_xyz_m_s2": scene_gravity["scene_gravity"]["vector_in_robot_root_xyz_m_s2"],
                        "physical_truth_conditional_on_scene_provenance": True,
                    },
                    source_sha256=scene_sha256,
                )
    observed_world = canonical.get("observed_world", {})
    observation = observed_world.get("observation") if isinstance(observed_world, dict) else None
    if isinstance(observation, dict):
        log = observation["observation_log"]
        query = observation["query"]
        observation_qualifiers = {
            "observation_log_id": log["id"],
            "query_id": query["query_id"],
            "query_time_ns": query["time_ns"],
            "clock": log["clock"],
            "selection_method": observation["selection_method"],
            "physical_world_completeness": "not_established",
            "source_truth_and_calibration": "not_established",
        }
        add(
            f"observation_log/{log['id']}",
            "has_time_selection_result",
            {
                "status": observation["status"],
                "selections": observation["selections"],
                "readiness": observation["readiness"],
            },
            exact=True,
            source_type="timestamp_policy_selection_from_bound_observation_log",
            qualifiers=observation_qualifiers,
            source_sha256=log["sha256"],
        )
        normalization = log.get("normalization")
        if isinstance(normalization, dict):
            add(
                f"ros_capture/{normalization['capture_id']}",
                "normalized_into_observation_log",
                {
                    "observation_log_id": log["id"],
                    "capture_sha256": normalization["capture_sha256"],
                    "adapter_id": normalization["adapter_id"],
                    "config_sha256": normalization["config_sha256"],
                    "method": normalization["method"],
                    "clock_policy": normalization["clock_policy"],
                    "authority_policy": normalization["authority_policy"],
                    "tf_policy": normalization["tf_policy"],
                },
                exact=True,
                source_type="ros_normalization_provenance_asserted_in_digest_bound_observation_log",
                qualifiers={
                    **observation_qualifiers,
                    "clock_synchronization": "not_established",
                    "publisher_truth": "not_established",
                    "physical_completeness": "not_established",
                },
                source_sha256=log["sha256"],
            )
        effective = observation["effective_state"]
        add(
            f"robot/{model.name}",
            "has_observed_joint_state_at_query_time",
            {
                "joint_positions": effective["joint_positions"],
                "source": effective["sources"]["joint_positions"],
                "selection_status": observation["selections"]["joint_states"]["status"],
            },
            exact=True,
            source_type="latest_past_joint_sample_zero_order_hold",
            pose_dependent=True,
            qualifiers=observation_qualifiers,
            source_sha256=log["sha256"],
        )
        if effective["world_from_robot_root"] is not None:
            add(
                f"robot_instance/{canonical['world_scene']['robot_mount']['instance_id']}",
                "has_effective_root_pose_at_observation_query",
                {
                    "world_from_robot_root": effective["world_from_robot_root"],
                    "source": effective["sources"]["robot_root"],
                },
                exact=True,
                source_type="timestamped_observation_or_explicit_static_declaration_fallback",
                pose_dependent=True,
                qualifiers=observation_qualifiers,
                source_sha256=log["sha256"],
            )
        for object_id, pose_record_value in effective["world_from_objects"].items():
            add(
                f"scene_object/{object_id}",
                "has_effective_pose_at_observation_query",
                {
                    "world_from_object": pose_record_value,
                    "source": effective["sources"]["objects"][object_id],
                },
                exact=True,
                source_type="timestamped_observation_or_explicit_static_declaration_fallback",
                pose_dependent=True,
                qualifiers=observation_qualifiers,
                source_sha256=log["sha256"],
            )
        observed_analysis = observed_world.get("analysis", {})
        if observed_analysis.get("robot_environment_collision") is not None:
            nominal_collision = observed_analysis["robot_environment_collision"]
            add(
                f"robot_instance/{canonical['world_scene']['robot_mount']['instance_id']}",
                "has_nominal_observation_conditioned_collision_status",
                {
                    "status": nominal_collision["status"],
                    "minimum_separation": nominal_collision["minimum_separation"],
                    "coverage": nominal_collision["coverage"],
                    "analysis_status": observed_analysis["status"],
                },
                exact=True,
                source_type="selected_observation_poses_and_declared_geometry_analysis",
                pose_dependent=True,
                qualifiers={
                    **observation_qualifiers,
                    "physical_collision_status": "not_established",
                    "safety_conclusion": "not_established",
                },
                source_sha256=log["sha256"],
            )
    invariant_validation = canonical.get("invariant_validation", {"status": "not_provided"})
    if invariant_validation["status"] != "not_provided":
        contract_sha256 = invariant_validation["contract_source"]["sha256"]
        add(
            f"robot/{model.name}",
            "has_spatial_invariant_contract_status",
            invariant_validation["status"],
            exact=True,
            source_type="project_spatial_invariant_contract_evaluation",
            qualifiers={
                "assertion_count": invariant_validation["assertion_count"],
                "passed_count": invariant_validation["passed_count"],
                "failed_count": invariant_validation["failed_count"],
                "source_urdf_sha256": invariant_validation["source_urdf_sha256"],
            },
            source_sha256=contract_sha256,
        )
        for invariant in invariant_validation["results"]:
            add(
                f"invariant/{invariant['id']}",
                "has_validation_result",
                {
                    key: invariant.get(key)
                    for key in ("type", "status", "expected", "actual", "metrics", "error")
                    if key in invariant
                },
                exact=True,
                source_type="project_spatial_invariant_contract_evaluation",
                qualifiers={"evaluated_pose": invariant["pose"], "source_urdf_sha256": invariant["source_urdf_sha256"]},
                source_sha256=contract_sha256,
            )
    return sorted(facts, key=lambda fact: (fact["subject"], fact["predicate"], fact["fact_id"]))


def jsonl_dump(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records)


def source_contains_xacro_elements(path: Path) -> bool:
    try:
        root = ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError) as error:
        raise SpatialError(f"cannot inspect XML source {path}: {error}") from error
    return any(
        isinstance(element.tag, str)
        and (
            element.tag.startswith("xacro:")
            or (element.tag.startswith("{") and "xacro" in element.tag.partition("}")[0].lower())
        )
        for element in root.iter()
    )


def expand_xacro(
    input_path: Path,
    output_path: Path,
    xacro_bin: str,
    mappings: list[str],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    located_executable = shutil.which(xacro_bin, path=(environment or os.environ).get("PATH"))
    if located_executable is None:
        raise SpatialError(f"xacro executable {xacro_bin!r} was not found; install ROS xacro or pass --xacro-bin")
    executable = str(Path(located_executable).resolve())
    for mapping in mappings:
        name, separator, _ = mapping.partition(":=")
        if not separator or not name or any(character.isspace() for character in name):
            raise SpatialError(f"invalid Xacro mapping {mapping!r}; expected name:=value")
    try:
        source = input_path.read_bytes()
    except OSError as error:
        raise SpatialError(f"cannot read Xacro input {input_path}: {error}") from error
    if input_path.resolve() == output_path.resolve():
        raise SpatialError("Xacro input and expanded URDF output must be different files")
    command = [executable, str(input_path.resolve()), *mappings]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=30, env=environment)
    except subprocess.TimeoutExpired as error:
        raise SpatialError("xacro did not finish within 30 seconds") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SpatialError(f"xacro failed with exit code {result.returncode}: {stderr}")
    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError as error:
        raise SpatialError(f"xacro output is not valid XML: {error}") from error
    if root.tag != "robot":
        raise SpatialError("xacro output root must be <robot>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="robot-spatial-xacro-", suffix=".urdf", dir=output_path.parent, delete=False) as temporary:
            temporary.write(result.stdout)
            temporary_path = Path(temporary.name)
        RobotModel(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    expanded_model = RobotModel(output_path)
    metadata = {
        "schema_version": "xacro-expansion.v1",
        "input": {"path": str(input_path.resolve()), "sha256": hashlib.sha256(source).hexdigest()},
        "output": {"path": str(output_path.resolve()), "sha256": expanded_model.sha256},
        "mappings": mappings,
        "xacro_executable": executable,
        "xacro_executable_sha256": sha256_path(Path(executable)) if Path(executable).is_file() else None,
        "xacro_stderr": result.stderr.decode("utf-8", errors="replace").strip(),
        "validation": {"robot": expanded_model.name, "root_link": expanded_model.root_link, "links": len(expanded_model.links), "joints": len(expanded_model.joints)},
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json_dump(metadata), encoding="utf-8")
    return {"status": "expanded_and_validated", **metadata, "metadata_path": str(metadata_path.resolve())}


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def attach_query_evidence(
    result: dict[str, Any],
    model: RobotModel,
    command: str,
    parameters: dict[str, Any],
    method: str,
    epistemic_scope: str,
) -> dict[str, Any]:
    identity = {
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "command": command,
        "parameters": parameters,
        "method": method,
    }
    query_id = "query-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    result["query_evidence"] = {
        "schema_version": "robot-spatial-query-evidence.v1",
        "query_id": query_id,
        "source_urdf_sha256": model.sha256,
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "command": command,
        "parameters": parameters,
        "method": method,
        "deterministic": True,
        "epistemic_scope": epistemic_scope,
    }
    return result


def attach_scene_query_evidence(
    result: dict[str, Any],
    model: RobotModel,
    scene: WorldScene,
    command: str,
    parameters: dict[str, Any],
    method: str,
    epistemic_scope: str,
) -> dict[str, Any]:
    """Attach a deterministic identity that binds both URDF and scene snapshot."""
    identity = {
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "source_world_scene_sha256": scene.sha256,
        "snapshot_id": scene.snapshot["id"],
        "command": command,
        "parameters": parameters,
        "method": method,
    }
    query_id = "query-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    result["query_evidence"] = {
        "schema_version": "robot-spatial-scene-query-evidence.v1",
        "query_id": query_id,
        "source_urdf_sha256": model.sha256,
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "source_world_scene_path": str(scene.path),
        "source_world_scene_sha256": scene.sha256,
        "scene_id": scene.scene_id,
        "snapshot_id": scene.snapshot["id"],
        "command": command,
        "parameters": parameters,
        "method": method,
        "deterministic": True,
        "epistemic_scope": epistemic_scope,
    }
    return result


def attach_observation_query_evidence(
    result: dict[str, Any],
    model: RobotModel,
    scene: WorldScene,
    resolved: dict[str, Any],
    command: str,
    parameters: dict[str, Any],
    method: str,
    epistemic_scope: str,
) -> dict[str, Any]:
    """Bind a query to model, declared scene, observation log, and time policy."""
    observation = resolved["report"]["observation_log"]
    query = resolved["report"]["query"]
    identity = {
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "source_world_scene_sha256": scene.sha256,
        "source_observation_log_sha256": observation["sha256"],
        "source_observation_query_sha256": resolved["query_sha256"],
        "query_time_ns": query["time_ns"],
        "command": command,
        "parameters": parameters,
        "method": method,
    }
    query_id = "query-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    result["query_evidence"] = {
        "schema_version": "robot-spatial-observation-query-evidence.v1",
        "query_id": query_id,
        "source_urdf_sha256": model.sha256,
        "source_urdf_semantic_sha256": model.semantic_sha256,
        "source_world_scene_path": str(scene.path),
        "source_world_scene_sha256": scene.sha256,
        "source_observation_log_path": observation["path"],
        "source_observation_log_sha256": observation["sha256"],
        "source_observation_query_path": resolved["report"]["query_source"]["path"],
        "source_observation_query_sha256": resolved["query_sha256"],
        "scene_id": scene.scene_id,
        "observation_log_id": observation["id"],
        "observation_policy_query_id": query["query_id"],
        "query_time_ns": query["time_ns"],
        "command": command,
        "parameters": parameters,
        "method": method,
        "deterministic": True,
        "epistemic_scope": epistemic_scope,
    }
    return result


def scene_gravity_load_analysis(
    model: RobotModel,
    scene: WorldScene,
    supplied_pose: dict[str, float],
    pose_name: str,
    world_from_robot_root: Matrix | None = None,
) -> dict[str, Any]:
    """Bind declared world gravity and root mounting to the robot-local load model."""
    converted = scene.gravity_in_robot_root(world_from_robot_root)
    if converted["status"] != "computed":
        return {
            "schema_version": "robot-spatial-scene-gravity-loads.v1",
            "status": "not_provided",
            "scene_gravity": converted,
            "loads": None,
            "epistemic_scope": "no world-gravity conclusion is produced when the scene snapshot omits gravity",
        }
    loads = model.static_gravity_loads(
        supplied_pose,
        converted["vector_in_robot_root_xyz_m_s2"],
        model.root_link,
    )
    loads["pose"]["name"] = pose_name
    return {
        "schema_version": "robot-spatial-scene-gravity-loads.v1",
        "status": loads["status"],
        "scene": {
            "scene_id": scene.scene_id,
            "snapshot_id": scene.snapshot["id"],
            "source_path": str(scene.path),
            "sha256": scene.sha256,
        },
        "scene_gravity": converted,
        "loads": loads,
        "epistemic_scope": "exact for valid declared URDF inertials, stated pose, and the scene-declared root mount and gravity vector; physical truth is conditional on scene provenance and excludes payload, contact, motion, transmission, controller, and hardware behavior",
    }


def observation_world_analysis(
    model: RobotModel,
    scene: WorldScene,
    resolved: dict[str, Any],
    package_map_path: Path | None,
    contact_tolerance_m: float,
) -> dict[str, Any]:
    """Compute nominal geometry and gravity only when the temporal resolver permits it."""
    if not resolved["nominal_computable"]:
        return {
            "schema_version": "robot-spatial-observed-world-analysis.v1",
            "status": "not_computed",
            "reason": "required current joint/root/object state is unavailable under the query fallback policy",
            "robot_environment_collision": None,
            "declared_static_gravity_loads": None,
            "physical_world_truth": "not_established",
        }
    pose = resolved["joint_pose"]
    root_transform = resolved["world_from_robot_root"]
    assert pose is not None and root_transform is not None
    collision = scene.robot_environment_collisions(
        model,
        pose,
        package_map_path,
        contact_tolerance_m,
        world_from_robot_root=root_transform,
        world_from_objects=resolved["world_from_objects"],
    )
    gravity = scene_gravity_load_analysis(
        model,
        scene,
        pose,
        "observed_at_query_time",
        root_transform,
    )
    return {
        "schema_version": "robot-spatial-observed-world-analysis.v1",
        "status": "computed_from_current_observations" if resolved["all_required_current"] else "computed_nominally_with_declaration_fallback",
        "robot_environment_collision": collision,
        "declared_static_gravity_loads": gravity,
        "physical_world_truth": "not_established",
        "safety_conclusion": "not_established",
        "epistemic_scope": "nominal declared-model consequences at selected poses; covariance, source truth, calibration, omitted physical objects, and continuous motion are outside this result",
    }


def _path_argument(command: list[str], option: str, value: Path | None) -> None:
    if value is not None:
        command.extend([option, str(value.expanduser().resolve())])


def prepare_project(args: argparse.Namespace) -> dict[str, Any]:
    """Compile a source-workspace URDF/Xacro project into a provenance-bound context pack."""
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SpatialError(f"project source is not a file: {source}")
    output = args.out.expanduser().resolve()
    if output.exists():
        raise SpatialError(f"prepare output must not already exist: {output}")
    owning_root = nearest_package_root(source)
    roots = [path.expanduser().resolve() for path in args.workspace_roots]
    if not roots:
        roots.append(owning_root.parent if owning_root is not None else source.parent)
    elif owning_root is not None:
        owning_is_covered = any(
            owning_root == root or root in owning_root.parents
            for root in roots
        )
        if not owning_is_covered:
            roots.append(owning_root)
    packages = discover_packages(roots)
    owning_package = next(
        (name for name, directory in packages.items() if owning_root is not None and directory == owning_root),
        None,
    )
    package_snapshots = {
        name: tree_manifest(directory, package_name=name)
        for name, directory in packages.items()
    }
    source_scope = None if owning_package is not None else tree_manifest(source.parent)
    input_format = "xacro" if source.suffix == ".xacro" or source_contains_xacro_elements(source) else "urdf"
    if input_format == "urdf" and args.mappings:
        raise SpatialError("--arg mappings are valid only when the source is Xacro")
    scene_source = args.scene.expanduser().resolve() if args.scene is not None else None
    if scene_source is not None and not scene_source.is_file():
        raise SpatialError(f"world scene source is not a file: {scene_source}")
    observation_source = args.observations.expanduser().resolve() if args.observations is not None else None
    observation_query_source = args.observation_query.expanduser().resolve() if args.observation_query is not None else None
    constraint_source = args.constraint_spec.expanduser().resolve() if args.constraint_spec is not None else None
    configuration_atlas_source = (
        args.configuration_atlas_spec.expanduser().resolve()
        if args.configuration_atlas_spec is not None
        else None
    )
    functional_source = args.functional_spec.expanduser().resolve() if args.functional_spec is not None else None
    if (observation_source is None) != (observation_query_source is None):
        raise SpatialError("--observations and --observation-query must be provided together")
    if observation_source is not None and scene_source is None:
        raise SpatialError("--observations requires --scene")
    for label, path in (("observation log", observation_source), ("observation query", observation_query_source)):
        if path is not None and not path.is_file():
            raise SpatialError(f"{label} source is not a file: {path}")
    if constraint_source is not None and not constraint_source.is_file():
        raise SpatialError(f"constraint spec source is not a file: {constraint_source}")
    if configuration_atlas_source is not None and constraint_source is None:
        raise SpatialError("--configuration-atlas-spec requires --constraint-spec")
    if configuration_atlas_source is not None and not configuration_atlas_source.is_file():
        raise SpatialError(f"configuration atlas spec source is not a file: {configuration_atlas_source}")
    if functional_source is not None and not functional_source.is_file():
        raise SpatialError(f"function/affordance spec source is not a file: {functional_source}")

    output.mkdir(parents=True, exist_ok=False)
    try:
        package_map_path = output / "package-map.json"
        package_map_path.write_text(
            json_dump({name: str(directory) for name, directory in packages.items()}),
            encoding="utf-8",
        )
        resolved_urdf = output / "resolved.urdf"
        package_lookups: list[str] = []
        if input_format == "xacro":
            with tempfile.TemporaryDirectory(prefix="robot-spatial-ament-") as temporary_directory:
                temporary_root = Path(temporary_directory)
                shim_root = write_ament_index_shim(temporary_root / "python")
                lookup_log = temporary_root / "package-lookups.txt"
                environment = xacro_environment(packages, shim_root, lookup_log)
                resolution = expand_xacro(source, resolved_urdf, args.xacro_bin, args.mappings, environment)
                package_lookups = read_package_lookups(lookup_log)
            raw_output = resolution["output"]
            normalization = normalize_expanded_urdf(resolved_urdf, packages)
            normalized_model = RobotModel(resolved_urdf)
            resolution["raw_xacro_output"] = raw_output
            resolution["output"] = {
                "path": str(resolved_urdf),
                "sha256": normalized_model.sha256,
                "semantic_sha256": normalized_model.semantic_sha256,
            }
            resolution["normalization"] = normalization
            Path(resolution["metadata_path"]).write_text(json_dump(resolution), encoding="utf-8")
        else:
            shutil.copy2(source, resolved_urdf)
            model = RobotModel(resolved_urdf)
            resolution = {
                "status": "copied_and_validated",
                "schema_version": "urdf-resolution.v1",
                "input": {"path": str(source), "sha256": sha256_path(source)},
                "output": {"path": str(resolved_urdf), "sha256": model.sha256},
                "validation": {
                    "robot": model.name,
                    "root_link": model.root_link,
                    "links": len(model.links),
                    "joints": len(model.joints),
                },
            }

        package_references = package_references_from_urdf(resolved_urdf)
        missing_packages = sorted((set(package_lookups) | set(package_references)) - set(packages))
        if missing_packages:
            raise SpatialError(f"resolved model refers to undiscovered ROS packages: {missing_packages}")
        used_package_names = sorted(
            ({owning_package} if owning_package is not None else set())
            | set(package_lookups)
            | set(package_references)
        )
        source_manifest = {
            "schema_version": "robot-spatial-source-manifest.v1",
            "identity_policy": "package tree digests bind sorted relative paths, file kinds, symlink targets, and SHA-256 content while excluding machine-local package roots",
            "entrypoint": {"path": str(source), "sha256": sha256_path(source), "input_format": input_format},
            "workspace_roots": [str(path) for path in roots],
            "discovered_packages": [
                {
                    "name": name,
                    "root": snapshot["root"],
                    "file_count": snapshot["file_count"],
                    "tree_sha256": snapshot["tree_sha256"],
                }
                for name, snapshot in package_snapshots.items()
            ],
            "xacro_package_lookups": package_lookups,
            "resolved_urdf_package_references": package_references,
            "used_packages": [package_snapshots[name] for name in used_package_names],
            "unpackaged_source_scope": source_scope,
            "world_scene": (
                None
                if scene_source is None
                else {"path": str(scene_source), "sha256": sha256_path(scene_source)}
            ),
            "temporal_observation": (
                None
                if observation_source is None
                else {
                    "log": {"path": str(observation_source), "sha256": sha256_path(observation_source)},
                    "query": {"path": str(observation_query_source), "sha256": sha256_path(observation_query_source)},
                }
            ),
            "supplemental_constraints": (
                None
                if constraint_source is None
                else {"path": str(constraint_source), "sha256": sha256_path(constraint_source)}
            ),
            "configuration_atlas": (
                None
                if configuration_atlas_source is None
                else {"path": str(configuration_atlas_source), "sha256": sha256_path(configuration_atlas_source)}
            ),
            "functional_knowledge": (
                None
                if functional_source is None
                else {"path": str(functional_source), "sha256": sha256_path(functional_source)}
            ),
        }
        source_manifest_path = output / "source-manifest.json"
        source_manifest_path.write_text(json_dump(source_manifest), encoding="utf-8")

        context_directory = output / "context"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "export",
            str(resolved_urdf),
            "--out",
            str(context_directory),
            "--package-map",
            str(package_map_path),
            "--workspace-samples",
            str(args.workspace_samples),
            "--contact-tolerance-m",
            str(args.contact_tolerance_m),
        ]
        _path_argument(command, "--pose", args.pose)
        _path_argument(command, "--srdf", args.srdf)
        _path_argument(command, "--semantics", args.semantics)
        _path_argument(command, "--invariants", args.invariants)
        _path_argument(command, "--scene", scene_source)
        _path_argument(command, "--observations", observation_source)
        _path_argument(command, "--observation-query", observation_query_source)
        _path_argument(command, "--constraint-spec", constraint_source)
        _path_argument(command, "--configuration-atlas-spec", configuration_atlas_source)
        _path_argument(command, "--functional-spec", functional_source)
        if args.pose_name:
            command.extend(["--pose-name", args.pose_name])
        if args.inspect_meshes:
            command.append("--inspect-meshes")
        for kind in args.inspect_mesh_kinds or []:
            command.extend(["--inspect-mesh-kind", kind])
        if args.render:
            command.append("--render")
        if args.motion_atlas:
            command.extend([
                "--motion-atlas",
                "--motion-angular-step-rad",
                str(args.motion_angular_step_rad),
                "--motion-linear-step-m",
                str(args.motion_linear_step_m),
            ])
        if args.include_workspace_samples:
            command.append("--include-workspace-samples")
        if args.surface_collisions:
            command.append("--surface-collisions")
        try:
            export_result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise SpatialError("context export did not finish within 120 seconds") from error
        if export_result.returncode != 0:
            detail = export_result.stderr.strip() or export_result.stdout.strip()
            raise SpatialError(f"context export failed with exit code {export_result.returncode}: {detail}")
        try:
            export_response = json.loads(export_result.stdout)
        except json.JSONDecodeError as error:
            raise SpatialError(f"context export returned invalid JSON: {error}") from error

        source_compilation = {
            "schema_version": "robot-spatial-source-compilation.v1",
            "input_format": input_format,
            "entrypoint": {"path": str(source), "sha256": sha256_path(source)},
            "resolved_urdf": {
                "path": str(resolved_urdf),
                "sha256": sha256_path(resolved_urdf),
                "semantic_sha256": RobotModel(resolved_urdf).semantic_sha256,
            },
            "mappings": list(args.mappings),
            "package_lookups": package_lookups,
            "package_references": package_references,
            "package_map": {"path": "../package-map.json", "sha256": sha256_path(package_map_path)},
            "source_manifest": {"path": "../source-manifest.json", "sha256": sha256_path(source_manifest_path)},
            "world_scene": (
                None
                if scene_source is None
                else {"path": str(scene_source), "sha256": sha256_path(scene_source)}
            ),
            "temporal_observation": (
                None
                if observation_source is None
                else {
                    "log": {"path": str(observation_source), "sha256": sha256_path(observation_source)},
                    "query": {"path": str(observation_query_source), "sha256": sha256_path(observation_query_source)},
                }
            ),
            "supplemental_constraints": source_manifest["supplemental_constraints"],
            "configuration_atlas": source_manifest["configuration_atlas"],
            "functional_knowledge": source_manifest["functional_knowledge"],
            "resolution": resolution,
        }
        for artifact_name in ("model.json", "agent-context.json"):
            artifact_path = context_directory / artifact_name
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            data["source_compilation"] = source_compilation
            artifact_path.write_text(json_dump(data), encoding="utf-8")

        report = {
            "schema_version": "robot-spatial-project-preparation.v1",
            "status": "prepared",
            "input_format": input_format,
            "entrypoint": source_compilation["entrypoint"],
            "resolved_urdf": source_compilation["resolved_urdf"],
            "robot": resolution["validation"],
            "mappings": list(args.mappings),
            "workspace_roots": [str(path) for path in roots],
            "discovered_package_count": len(packages),
            "used_packages": used_package_names,
            "xacro_package_lookups": package_lookups,
            "world_scene": source_compilation["world_scene"],
            "temporal_observation": source_compilation["temporal_observation"],
            "artifacts": {
                "package_map": {"path": "package-map.json", "sha256": sha256_path(package_map_path)},
                "source_manifest": {"path": "source-manifest.json", "sha256": sha256_path(source_manifest_path)},
                "context_manifest": {"path": "context/agent-context.json", "sha256": sha256_path(context_directory / "agent-context.json")},
                "model": {"path": "context/model.json", "sha256": sha256_path(context_directory / "model.json")},
                "articulation_grammar": {"path": "context/articulation-grammar.json", "sha256": sha256_path(context_directory / "articulation-grammar.json")},
                "concept_graph": {"path": "context/concept-graph.json", "sha256": sha256_path(context_directory / "concept-graph.json")},
                "concept_language": {"path": "context/concept-language.rsl", "sha256": sha256_path(context_directory / "concept-language.rsl")},
                **(
                    {
                        "functional_model": {
                            "path": "context/functional-model.json",
                            "sha256": sha256_path(context_directory / "functional-model.json"),
                        }
                    }
                    if functional_source is not None
                    else {}
                ),
                **(
                    {
                        "constraint_graph": {
                            "path": "context/constraint-graph.json",
                            "sha256": sha256_path(context_directory / "constraint-graph.json"),
                        },
                        "constraint_evaluation": {
                            "path": "context/constraint-evaluation.json",
                            "sha256": sha256_path(context_directory / "constraint-evaluation.json"),
                        },
                    }
                    if constraint_source is not None
                    else {}
                ),
            },
            "next_action": (
                "read context/agent-context.json; use query-concepts for structure and query-functions for explicit "
                "function/capability/affordance knowledge when functional-model.json is present; then retrieve exact "
                "typed entities and bound facts"
            ),
            "export": export_response,
        }
        (output / "prepare.json").write_text(json_dump(report), encoding="utf-8")
        return report
    except Exception:
        shutil.rmtree(output)
        raise


def add_pose_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pose", type=Path, help="JSON file with pose_name and joints")
    parser.add_argument("--srdf", type=Path, help="optional SRDF with groups, end effectors, collision policy, and named poses")
    parser.add_argument("--pose-name", help="named pose from --srdf, preferably group/name")


def add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inspect-meshes", action="store_true", help="open and measure every declared visual and collision STL/OBJ mesh")
    parser.add_argument(
        "--inspect-mesh-kind",
        action="append",
        choices=sorted(MESH_GEOMETRY_KINDS),
        dest="inspect_mesh_kinds",
        help="measure only this geometry kind; repeat for both (the option itself enables inspection)",
    )
    parser.add_argument("--package-map", type=Path, help="JSON map from package names to package directories")


def add_motion_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--motion-angular-step-rad",
        type=float,
        default=0.1,
        help="nominal signed revolute/continuous counterfactual step; clipped at feasible limits",
    )
    parser.add_argument(
        "--motion-linear-step-m",
        type=float,
        default=0.01,
        help="nominal signed prismatic counterfactual step; clipped at feasible limits",
    )


def add_observation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--scene", type=Path, required=True, help="bound robot-spatial-world-scene.v1 declaration")
    parser.add_argument("--observations", type=Path, required=True, help="robot-spatial-observation-log.v1/v2 timestamped source reports")
    parser.add_argument("--observation-query", type=Path, required=True, help="query time, age limits, required objects, and fallback policy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    expand = subparsers.add_parser("expand-xacro", help="expand Xacro through an external xacro executable and validate the URDF")
    expand.add_argument("xacro", type=Path)
    expand.add_argument("--out", type=Path, required=True)
    expand.add_argument("--xacro-bin", default="xacro")
    expand.add_argument("--arg", action="append", default=[], dest="mappings", help="Xacro mapping in name:=value form; repeat as needed")
    prepare = subparsers.add_parser(
        "prepare",
        help="discover a ROS source workspace, resolve URDF/Xacro, and export one provenance-bound agent context",
    )
    prepare.add_argument("source", type=Path, help="concrete URDF or Xacro project entrypoint")
    prepare.add_argument("--out", type=Path, required=True, help="new output directory; existing paths are rejected")
    prepare.add_argument(
        "--workspace-root",
        action="append",
        default=[],
        type=Path,
        dest="workspace_roots",
        help="ROS source root to scan for package.xml; repeat as needed (the entry package is included if these roots do not contain it)",
    )
    prepare.add_argument("--xacro-bin", default="xacro")
    prepare.add_argument("--arg", action="append", default=[], dest="mappings", help="Xacro mapping in name:=value form; repeat as needed")
    prepare.add_argument("--semantics", type=Path, help="optional robot-semantics.v1 JSON")
    prepare.add_argument("--invariants", type=Path, help="optional robot-spatial-invariants.v1 contract")
    prepare.add_argument("--constraint-spec", type=Path, help="optional robot-spatial-constraint-spec.v1 supplemental mechanism contract")
    prepare.add_argument("--configuration-atlas-spec", type=Path, help="optional robot-spatial-configuration-atlas-spec.v1; requires --constraint-spec")
    prepare.add_argument(
        "--functional-spec",
        type=Path,
        help="optional robot-spatial-function-affordance-spec.v1 project function, capability, and affordance contract",
    )
    prepare.add_argument("--scene", type=Path, help="optional robot-spatial-world-scene.v1 static snapshot")
    prepare.add_argument("--observations", type=Path, help="optional robot-spatial-observation-log.v1/v2 bound to --scene")
    prepare.add_argument("--observation-query", type=Path, help="required query-time policy when --observations is used")
    prepare.add_argument("--render", action="store_true")
    prepare.add_argument("--motion-atlas", action="store_true", help="generate per-independent-joint finite counterfactual motion views")
    add_motion_policy_arguments(prepare)
    prepare.add_argument("--workspace-samples", type=int, default=0)
    prepare.add_argument("--include-workspace-samples", action="store_true")
    prepare.add_argument("--inspect-meshes", action="store_true")
    prepare.add_argument(
        "--inspect-mesh-kind",
        action="append",
        choices=sorted(MESH_GEOMETRY_KINDS),
        dest="inspect_mesh_kinds",
    )
    prepare.add_argument("--surface-collisions", action="store_true")
    prepare.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    add_pose_argument(prepare)
    validate = subparsers.add_parser("validate", help="validate supported URDF structure")
    validate.add_argument("urdf", type=Path)
    validate.add_argument("--semantics", type=Path, help="optional robot-semantics.v1 JSON")
    validate.add_argument("--srdf", type=Path, help="optional SRDF semantic model")
    validate.add_argument("--scene", type=Path, help="optional robot-spatial-world-scene.v1 static snapshot")
    tree = subparsers.add_parser("tree", help="print the kinematic tree")
    tree.add_argument("urdf", type=Path)
    chain = subparsers.add_parser("chain", help="show the exact joint path between two links")
    chain.add_argument("urdf", type=Path)
    chain.add_argument("--from", dest="start_link", required=True)
    chain.add_argument("--to", dest="end_link", required=True)
    affects = subparsers.add_parser("affects", help="show links and frames affected by one joint")
    affects.add_argument("urdf", type=Path)
    affects.add_argument("--joint", required=True)
    actuation = subparsers.add_parser(
        "actuation",
        help="report ros2_control and legacy transmission declarations without claiming runtime or hardware capability",
    )
    actuation.add_argument("urdf", type=Path)
    actuation_selection = actuation.add_mutually_exclusive_group()
    actuation_selection.add_argument("--joint", help="select one kinematic joint and all embedded control bindings")
    actuation_selection.add_argument("--system", help="select one ros2_control system")
    actuation_selection.add_argument("--transmission", help="select one legacy transmission")
    export = subparsers.add_parser("export", help="write canonical model, overview, indexed agent context, and provenance facts")
    export.add_argument("urdf", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--semantics", type=Path, help="optional robot-semantics.v1 JSON")
    export.add_argument("--scene", type=Path, help="optional robot-spatial-world-scene.v1 static snapshot")
    export.add_argument("--observations", type=Path, help="optional robot-spatial-observation-log.v1/v2 bound to --scene")
    export.add_argument("--observation-query", type=Path, help="required query-time policy when --observations is used")
    export.add_argument("--render", action="store_true", help="write scene.svg plus a digest-bound semantic render-atlas into --out")
    export.add_argument("--motion-atlas", action="store_true", help="write a digest-bound counterfactual motion-atlas into --out")
    add_motion_policy_arguments(export)
    export.add_argument("--workspace-samples", type=int, default=256, help="deterministic samples per declared target; use 0 to omit sampled workspaces")
    export.add_argument("--include-workspace-samples", action="store_true", help="store every workspace sample in model.json for audit or downstream learning")
    export.add_argument("--generate-evaluation", action="store_true", help="write blind spatial questions, isolated answer key, and answer template")
    export.add_argument("--evaluation-key-out", type=Path, help="private answer-key path outside the candidate-readable workspace")
    export.add_argument("--invariants", type=Path, help="robot-spatial-invariants.v1 project intent contract; failed assertions make export exit non-zero")
    export.add_argument("--constraint-spec", type=Path, help="robot-spatial-constraint-spec.v1 bound to the generated articulation grammar; violated constraints make export exit non-zero")
    export.add_argument("--configuration-atlas-spec", type=Path, help="robot-spatial-configuration-atlas-spec.v1 bound to the generated constraint graph; incomplete declared sampling makes export exit non-zero")
    export.add_argument(
        "--functional-spec",
        type=Path,
        help="robot-spatial-function-affordance-spec.v1; compiles explicit function knowledge grounded against the generated concept graph",
    )
    export.add_argument("--surface-collisions", action="store_true", help="run exact triangle-surface distance/contact and closed-surface containment analysis")
    export.add_argument("--contact-tolerance-m", type=float, default=1e-9, help="surface separation at or below this value counts as contact")
    add_geometry_arguments(export)
    add_pose_argument(export)
    transform = subparsers.add_parser("transform", help="emit A_from_B transform")
    transform.add_argument("urdf", type=Path)
    transform.add_argument("--from", dest="reference", required=True)
    transform.add_argument("--to", dest="target", required=True)
    add_pose_argument(transform)
    axis = subparsers.add_parser("axis", help="express a joint axis in a frame")
    axis.add_argument("urdf", type=Path)
    axis.add_argument("--joint", required=True)
    axis.add_argument("--frame", required=True)
    add_pose_argument(axis)
    jacobian = subparsers.add_parser("jacobian", help="compute an analytic geometric Jacobian for a target frame")
    jacobian.add_argument("urdf", type=Path)
    jacobian.add_argument("--target", required=True)
    jacobian.add_argument("--frame", help="orientation frame for vector components; defaults to root")
    add_pose_argument(jacobian)
    mass_properties = subparsers.add_parser(
        "mass-properties",
        help="aggregate declared URDF mass, center of mass, and inertia for the whole tree or one subtree",
    )
    mass_properties.add_argument("urdf", type=Path)
    mass_properties.add_argument("--subtree-root", help="link whose complete descendant subtree is selected; defaults to the URDF root")
    mass_properties.add_argument("--frame", help="frame in which COM and inertia components are expressed; defaults to the URDF root")
    add_pose_argument(mass_properties)
    gravity_loads = subparsers.add_parser(
        "gravity-loads",
        help="compute gravity generalized forces and opposite ideal static holding efforts from declared inertials",
    )
    gravity_loads.add_argument("urdf", type=Path)
    gravity_loads.add_argument("--subtree-root", help="select this complete link subtree while retaining loads transmitted to upstream joints")
    gravity_loads.add_argument("--gravity-frame", help="frame whose orientation expresses --gravity; defaults to the URDF root")
    gravity_loads.add_argument(
        "--gravity",
        nargs=3,
        type=float,
        metavar=("GX", "GY", "GZ"),
        default=[0.0, 0.0, -9.80665],
        help="gravity acceleration vector in m/s^2; defaults to 0 0 -9.80665",
    )
    add_pose_argument(gravity_loads)
    scene_summary = subparsers.add_parser(
        "scene-summary",
        help="bind a validated static world snapshot to the URDF and report frames, gravity, objects, and robot/environment collision",
    )
    scene_summary.add_argument("urdf", type=Path)
    scene_summary.add_argument("--scene", type=Path, required=True)
    scene_summary.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    scene_summary.add_argument("--package-map", type=Path, help="JSON map from package names to package directories for robot or scene meshes")
    add_pose_argument(scene_summary)
    scene_transform = subparsers.add_parser(
        "scene-transform",
        help="emit a transform between typed scene, object, geometry, or robot frames",
    )
    scene_transform.add_argument("urdf", type=Path)
    scene_transform.add_argument("--scene", type=Path, required=True)
    scene_transform.add_argument("--from", dest="reference", required=True, help="typed reference entity such as scene_frame/world")
    scene_transform.add_argument("--to", dest="target", required=True, help="typed target entity such as robot_frame/tool0")
    add_pose_argument(scene_transform)
    scene_collisions = subparsers.add_parser(
        "scene-collisions",
        help="classify every declared robot collision geometry against every declared scene collision geometry",
    )
    scene_collisions.add_argument("urdf", type=Path)
    scene_collisions.add_argument("--scene", type=Path, required=True)
    scene_collisions.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    scene_collisions.add_argument("--package-map", type=Path, help="JSON map from package names to package directories for robot or scene meshes")
    scene_collisions.add_argument("--out", type=Path, help="optional JSON report path")
    add_pose_argument(scene_collisions)
    scene_gravity = subparsers.add_parser(
        "scene-gravity-loads",
        help="convert scene-declared world gravity through the root mounting and compute declared-model static joint loads",
    )
    scene_gravity.add_argument("urdf", type=Path)
    scene_gravity.add_argument("--scene", type=Path, required=True)
    add_pose_argument(scene_gravity)
    observe_summary = subparsers.add_parser(
        "observe-summary",
        help="resolve timestamped joint/root/object reports at one query time and compute a nominal observed-world summary",
    )
    add_observation_arguments(observe_summary)
    observe_summary.add_argument("--package-map", type=Path)
    observe_summary.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    observe_transform = subparsers.add_parser(
        "observe-transform",
        help="emit a transform conditioned on selected timestamped observations",
    )
    add_observation_arguments(observe_transform)
    observe_transform.add_argument("--from", dest="reference", required=True)
    observe_transform.add_argument("--to", dest="target", required=True)
    observe_collisions = subparsers.add_parser(
        "observe-collisions",
        help="classify nominal declared geometry under selected timestamped observations without claiming physical safety",
    )
    add_observation_arguments(observe_collisions)
    observe_collisions.add_argument("--package-map", type=Path)
    observe_collisions.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    observe_collisions.add_argument("--out", type=Path)
    observe_gravity = subparsers.add_parser(
        "observe-gravity-loads",
        help="compute declared-model static gravity loads under observed joint and root pose reports",
    )
    add_observation_arguments(observe_gravity)
    workspace = subparsers.add_parser("workspace", help="sample a deterministic approximate workspace for a target frame")
    workspace.add_argument("urdf", type=Path)
    workspace.add_argument("--target", required=True)
    workspace.add_argument("--samples", type=int, default=256)
    workspace.add_argument("--include-samples", action="store_true")
    add_pose_argument(workspace)
    distance = subparsers.add_parser("distance", help="distance between frame origins")
    distance.add_argument("urdf", type=Path)
    distance.add_argument("--from", dest="reference", required=True)
    distance.add_argument("--to", dest="target", required=True)
    add_pose_argument(distance)
    bounds = subparsers.add_parser("bounds", help="measure one declared visual/collision geometry frame")
    bounds.add_argument("urdf", type=Path)
    bounds.add_argument("--geometry-frame", required=True)
    add_pose_argument(bounds)
    add_geometry_arguments(bounds)
    overlaps = subparsers.add_parser("overlaps", help="list collision AABB broad-phase overlap candidates")
    overlaps.add_argument("urdf", type=Path)
    add_pose_argument(overlaps)
    add_geometry_arguments(overlaps)
    surface_distance = subparsers.add_parser("surface-distance", help="exact triangle-surface distance between two declared geometry frames")
    surface_distance.add_argument("urdf", type=Path)
    surface_distance.add_argument("--geometry-a", required=True)
    surface_distance.add_argument("--geometry-b", required=True)
    surface_distance.add_argument("--package-map", type=Path, help="JSON map from package names to package directories")
    add_pose_argument(surface_distance)
    surface_collisions = subparsers.add_parser("surface-collisions", help="verify collision candidates with exact triangle surfaces and solid containment")
    surface_collisions.add_argument("urdf", type=Path)
    surface_collisions.add_argument("--contact-tolerance-m", type=float, default=1e-9)
    surface_collisions.add_argument("--package-map", type=Path, help="JSON map from package names to package directories")
    surface_collisions.add_argument("--out", type=Path, help="optional JSON report path")
    add_pose_argument(surface_collisions)
    check_invariants = subparsers.add_parser("check-invariants", help="verify project spatial intent after a URDF edit; exits non-zero on any failed assertion")
    check_invariants.add_argument("urdf", type=Path)
    check_invariants.add_argument("--contract", type=Path, required=True)
    check_invariants.add_argument("--srdf", type=Path, help="optional SRDF collision policy and semantics")
    check_invariants.add_argument("--package-map", type=Path, help="JSON map from package names to package directories")
    check_invariants.add_argument("--scene", type=Path, help="optional robot-spatial-world-scene.v1 static snapshot used by scene assertions")
    check_invariants.add_argument("--observations", type=Path, help="optional bound observation log used by temporal assertions")
    check_invariants.add_argument("--observation-query", type=Path, help="required query-time policy when --observations is used")
    check_invariants.add_argument("--out", type=Path, help="optional invariant report JSON path")
    render = subparsers.add_parser("render", help="render an overview plus four machine-verifiable semantic SVG views")
    render.add_argument("urdf", type=Path)
    render.add_argument("--out", type=Path, required=True, help="output SVG path")
    render.add_argument("--atlas-out", type=Path, help="semantic render-atlas directory; defaults beside --out")
    render.add_argument("--semantics", type=Path, help="optional robot-semantics.v1 JSON")
    add_pose_argument(render)
    add_geometry_arguments(render)
    verify_render = subparsers.add_parser("verify-render", help="regenerate and verify one semantic render atlas")
    verify_render.add_argument("urdf", type=Path)
    verify_render.add_argument("--atlas", type=Path, required=True, help="render-atlas/manifest.json to verify")
    verify_render.add_argument("--out", type=Path, help="optional JSON verification report")
    verify_render.add_argument("--semantics", type=Path, help="same optional robot-semantics.v1 JSON used for rendering")
    add_pose_argument(verify_render)
    add_geometry_arguments(verify_render)
    motion_atlas = subparsers.add_parser(
        "motion-atlas",
        help="generate four-view finite counterfactual motion records for every independent movable joint",
    )
    motion_atlas.add_argument("urdf", type=Path)
    motion_atlas.add_argument("--out", type=Path, required=True, help="motion-atlas output directory")
    add_pose_argument(motion_atlas)
    add_geometry_arguments(motion_atlas)
    add_motion_policy_arguments(motion_atlas)
    verify_motion = subparsers.add_parser(
        "verify-motion-atlas",
        help="regenerate and verify one counterfactual motion atlas",
    )
    verify_motion.add_argument("urdf", type=Path)
    verify_motion.add_argument("--atlas", type=Path, required=True, help="motion-atlas/manifest.json to verify")
    verify_motion.add_argument("--out", type=Path, help="optional JSON verification report")
    add_pose_argument(verify_motion)
    add_geometry_arguments(verify_motion)
    add_motion_policy_arguments(verify_motion)
    articulation = subparsers.add_parser(
        "articulation-grammar",
        help="compile URDF, SDF, or canonical MJCF into a pose-independent executable kinematic law",
    )
    articulation.add_argument("urdf", type=Path)
    articulation.add_argument("--format", choices=("auto", "urdf", "sdf", "mjcf"), default="auto")
    articulation.add_argument("--out", type=Path, required=True, help="articulation grammar JSON path")
    evaluate_articulation = subparsers.add_parser(
        "evaluate-articulation",
        help="execute an articulation grammar at a new driver binding without parsing the URDF",
    )
    evaluate_articulation.add_argument("grammar", type=Path)
    evaluate_articulation.add_argument("--pose", type=Path, help="JSON file with pose_name and joints")
    evaluate_articulation.add_argument("--target", action="append", dest="targets", help="frame to evaluate; repeat as needed; defaults to all")
    evaluate_articulation.add_argument("--out", type=Path, help="optional evaluation JSON path")
    verify_articulation = subparsers.add_parser(
        "verify-articulation-grammar",
        help="regenerate and execute a grammar against all-frame source-format FK probes",
    )
    verify_articulation.add_argument("urdf", type=Path)
    verify_articulation.add_argument("--format", choices=("auto", "urdf", "sdf", "mjcf"), default="auto")
    verify_articulation.add_argument("--grammar", type=Path, required=True)
    verify_articulation.add_argument("--out", type=Path, help="optional verification JSON path")
    verify_articulation.add_argument("--tolerance", type=float, default=1e-10)
    compare_articulation = subparsers.add_parser(
        "compare-articulation-grammars",
        help="prove common-law and unseen-pose equivalence between two grammar artifacts",
    )
    compare_articulation.add_argument("reference", type=Path)
    compare_articulation.add_argument("candidate", type=Path)
    compare_articulation.add_argument("--correspondence", type=Path, help="digest-bound typed identifier correspondence when names differ")
    compare_articulation.add_argument("--tolerance", type=float, default=1e-10)
    compare_articulation.add_argument("--out", type=Path, help="optional comparison report JSON path")
    constraint_graph = subparsers.add_parser(
        "constraint-graph",
        help="compile asserted attachments and loop/coupling constraints over an articulation grammar",
    )
    constraint_graph.add_argument("grammar", type=Path)
    constraint_graph.add_argument("spec", type=Path)
    constraint_graph.add_argument("--out", type=Path, required=True, help="standalone constraint graph JSON path")
    evaluate_constraints = subparsers.add_parser(
        "evaluate-constraints",
        help="evaluate every supplemental constraint and local mechanism mobility at one driver binding",
    )
    evaluate_constraints.add_argument("graph", type=Path)
    evaluate_constraints.add_argument("--pose", type=Path, help="JSON file with pose_name and independent driver joints")
    evaluate_constraints.add_argument("--no-local-analysis", action="store_true", help="skip the finite-difference local rank/mobility analysis")
    evaluate_constraints.add_argument("--out", type=Path, help="optional evaluation JSON path")
    solve_constraints = subparsers.add_parser(
        "solve-constraints",
        help="solve asserted constraints locally from a seed over explicitly selected independent drivers",
    )
    solve_constraints.add_argument("graph", type=Path)
    solve_constraints.add_argument("--pose", type=Path, help="seed JSON with pose_name and independent driver joints")
    solve_constraints.add_argument("--solve-for", action="append", required=True, dest="solve_for", help="independent driver to solve; repeat as needed")
    solve_constraints.add_argument("--max-iterations", type=int, default=80)
    solve_constraints.add_argument("--damping", type=float, default=1e-8)
    solve_constraints.add_argument("--out", type=Path, help="optional solution JSON path")
    verify_constraints = subparsers.add_parser(
        "verify-constraint-graph",
        help="verify exact graph regeneration and standalone constraint execution",
    )
    verify_constraints.add_argument("grammar", type=Path)
    verify_constraints.add_argument("spec", type=Path)
    verify_constraints.add_argument("--graph", type=Path, required=True)
    verify_constraints.add_argument("--pose", type=Path, help="optional verification pose")
    verify_constraints.add_argument("--out", type=Path, help="optional verification JSON path")
    configuration_atlas = subparsers.add_parser(
        "configuration-atlas",
        help="explore digest-bound finite configuration witnesses over explicit one-parameter charts",
    )
    configuration_atlas.add_argument("graph", type=Path)
    configuration_atlas.add_argument("spec", type=Path)
    configuration_atlas.add_argument("--out", type=Path, required=True, help="standalone configuration atlas JSON path")
    verify_configuration = subparsers.add_parser(
        "verify-configuration-atlas",
        help="regenerate a finite configuration atlas and execute every stored witness node",
    )
    verify_configuration.add_argument("graph", type=Path)
    verify_configuration.add_argument("spec", type=Path)
    verify_configuration.add_argument("--atlas", type=Path, required=True)
    verify_configuration.add_argument("--out", type=Path, help="optional verification JSON path")
    concept_graph = subparsers.add_parser(
        "concept-graph",
        help="compile one exported context into a proof-carrying robot spatial concept language",
    )
    concept_graph.add_argument("context_directory", type=Path)
    concept_graph.add_argument("--out", type=Path, required=True, help="concept graph JSON path")
    concept_graph.add_argument("--language-out", type=Path, required=True, help="controlled concept-language text path")
    query_concepts = subparsers.add_parser(
        "query-concepts",
        help="execute one strict structural concept query and return its minimal proof closure",
    )
    query_concepts.add_argument("graph", type=Path)
    query_concepts.add_argument("query", type=Path)
    query_concepts.add_argument("--out", type=Path, help="optional concept answer JSON path")
    query_concepts.add_argument("--compact", action="store_true", help="emit single-line JSON")
    verify_concepts = subparsers.add_parser(
        "verify-concept-graph",
        help="regenerate and verify a concept graph plus controlled-language rendering",
    )
    verify_concepts.add_argument("context_directory", type=Path)
    verify_concepts.add_argument("--concept", type=Path, required=True)
    verify_concepts.add_argument("--language", type=Path, required=True)
    verify_concepts.add_argument("--out", type=Path, help="optional verification JSON path")
    functional_model = subparsers.add_parser(
        "functional-model",
        help="compile a project function/capability/affordance spec against one bound agent context",
    )
    functional_model.add_argument("context_directory", type=Path)
    functional_model.add_argument("spec", type=Path)
    functional_model.add_argument("--out", type=Path, required=True, help="functional model JSON path")
    query_functions = subparsers.add_parser(
        "query-functions",
        help="execute one strict function, capability, or affordance query with proof closure",
    )
    query_functions.add_argument("model", type=Path)
    query_functions.add_argument("query", type=Path)
    query_functions.add_argument("--out", type=Path, help="optional functional answer JSON path")
    query_functions.add_argument("--compact", action="store_true", help="emit single-line JSON")
    verify_functions = subparsers.add_parser(
        "verify-functional-model",
        help="exactly regenerate and verify one project functional model",
    )
    verify_functions.add_argument("context_directory", type=Path)
    verify_functions.add_argument("spec", type=Path)
    verify_functions.add_argument("--model", type=Path, required=True)
    verify_functions.add_argument("--out", type=Path, help="optional verification JSON path")
    action_assurance = subparsers.add_parser(
        "action-assurance",
        help="compile one replayable action-readiness, lifecycle, and effect evidence record",
    )
    action_assurance.add_argument("functional_model", type=Path)
    action_assurance.add_argument("evidence_bundle", type=Path)
    action_assurance.add_argument("--out", type=Path, required=True, help="action assurance JSON path")
    query_action = subparsers.add_parser(
        "query-action-assurance",
        help="query one action assurance record with its selected evidence and functional support",
    )
    query_action.add_argument("model", type=Path)
    query_action.add_argument("query", type=Path)
    query_action.add_argument("--out", type=Path, help="optional action assurance answer JSON path")
    query_action.add_argument("--compact", action="store_true", help="emit single-line JSON")
    verify_action = subparsers.add_parser(
        "verify-action-assurance",
        help="exactly regenerate and verify one action assurance record and all evidence-source digests",
    )
    verify_action.add_argument("functional_model", type=Path)
    verify_action.add_argument("evidence_bundle", type=Path)
    verify_action.add_argument("--model", type=Path, required=True)
    verify_action.add_argument("--out", type=Path, help="optional verification JSON path")
    compare = subparsers.add_parser("compare", help="compare two exported model.json artifacts")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--translation-tolerance", type=float, default=1e-9)
    compare.add_argument("--rotation-tolerance-deg", type=float, default=1e-7)
    retrieve = subparsers.add_parser("retrieve", help="verify and query one progressive-disclosure agent context pack")
    retrieve.add_argument("context_directory", type=Path)
    retrieve.add_argument("--entity", help="typed entity ID such as joint/shoulder or link/tool0; a bare unique name is accepted")
    retrieve.add_argument("--predicate", help="optional exact fact predicate filter")
    retrieve.add_argument("--pose", help="optional exact exported-pose qualifier filter")
    retrieve.add_argument("--evidence", choices=("all", "exact", "nonexact"), default="all")
    retrieve.add_argument("--fact-id", help="retrieve one exact fact ID")
    retrieve.add_argument("--list-entities", action="store_true")
    retrieve.add_argument("--limit", type=int, default=100)
    retrieve.add_argument("--compact", action="store_true", help="emit single-line JSON to minimize model context tokens")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "expand-xacro":
        print(json_dump(expand_xacro(args.xacro, args.out, args.xacro_bin, args.mappings)), end="")
        return 0
    if args.command == "prepare":
        print(json_dump(prepare_project(args)), end="")
        return 0
    if args.command == "compare":
        print(json_dump(compare_artifacts(args.before, args.after, args.translation_tolerance, args.rotation_tolerance_deg)), end="")
        return 0
    if args.command == "retrieve":
        result = retrieve_context(
            args.context_directory,
            entity=args.entity,
            predicate=args.predicate,
            pose=args.pose,
            evidence=args.evidence,
            fact_id=args.fact_id,
            list_entities=args.list_entities,
            limit=args.limit,
        )
        if args.compact:
            print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        else:
            print(json_dump(result), end="")
        return 0
    if args.command == "concept-graph":
        graph = write_concept_graph_from_context(
            args.context_directory,
            args.out,
            args.language_out,
        )
        result = {
            "status": "generated",
            "schema_version": CONCEPT_SCHEMA,
            "concept_graph": str(args.out.resolve()),
            "concept_language": str(args.language_out.resolve()),
            "concept_graph_id": graph["concept_graph_id"],
            "concept_graph_sha256": graph["concept_graph_sha256"],
            "coverage": graph["coverage"],
            "epistemic_scope": graph["epistemic_scope"],
        }
        print(json_dump(result), end="")
        return 0
    if args.command == "query-concepts":
        result = query_concept_graph_files(args.graph, args.query)
        serialized = (
            json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            if args.compact
            else json_dump(result)
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json_dump(result), encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "verify-concept-graph":
        result = verify_concept_graph(
            args.context_directory,
            args.concept,
            args.language,
        )
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "passed" else 1
    if args.command == "functional-model":
        model = write_functional_model_from_context(
            args.context_directory,
            args.spec,
            args.out,
        )
        result = {
            "status": "generated",
            "schema_version": FUNCTIONAL_MODEL_SCHEMA,
            "functional_model": str(args.out.resolve()),
            "functional_model_id": model["functional_model_id"],
            "functional_model_sha256": model["functional_model_sha256"],
            "grounding_status": model["status"],
            "coverage": model["coverage"],
            "epistemic_scope": model["epistemic_scope"],
        }
        print(json_dump(result), end="")
        return 0
    if args.command == "query-functions":
        result = query_functional_model_files(args.model, args.query)
        serialized = (
            json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            if args.compact
            else json_dump(result)
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json_dump(result), encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "verify-functional-model":
        result = verify_functional_model(
            args.context_directory,
            args.spec,
            args.model,
        )
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "passed" else 1
    if args.command == "action-assurance":
        model = write_action_assurance(
            args.functional_model,
            args.evidence_bundle,
            args.out,
        )
        result = {
            "status": "generated",
            "schema_version": ACTION_ASSURANCE_MODEL_SCHEMA,
            "action_assurance": str(args.out.resolve()),
            "assurance_id": model["assurance_id"],
            "assurance_sha256": model["assurance_sha256"],
            "readiness_conclusion": model["projections"]["readiness"]["conclusion"],
            "lifecycle_status": model["projections"]["lifecycle"]["status"],
            "outcome_conclusion": model["projections"]["outcome"]["conclusion"],
            "coverage": model["coverage"],
            "epistemic_scope": model["epistemic_scope"],
        }
        print(json_dump(result), end="")
        return 0
    if args.command == "query-action-assurance":
        result = query_action_assurance_files(args.model, args.query)
        serialized = (
            json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            if args.compact
            else json_dump(result)
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json_dump(result), encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "verify-action-assurance":
        result = verify_action_assurance(
            args.functional_model,
            args.evidence_bundle,
            args.model,
        )
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "passed" else 1
    if args.command == "evaluate-articulation":
        pose_name, pose = read_pose(args.pose)
        grammar = read_articulation_grammar(args.grammar)
        result = evaluate_articulation_grammar(grammar, pose, args.targets, pose_name)
        result["query_evidence"] = {
            "method": "standalone_typed_articulation_ast_execution",
            "grammar_path": str(args.grammar.resolve()),
            "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
            "exact": True,
            "scope": "exact consequence of this grammar artifact and supplied binding; no URDF parsing occurs in this command",
        }
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "compare-articulation-grammars":
        result = compare_articulation_grammars(
            args.reference,
            args.candidate,
            args.correspondence,
            args.tolerance,
        )
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "equivalent" else 1
    if args.command == "constraint-graph":
        graph = write_constraint_graph(args.out, args.grammar, args.spec)
        result = {
            "status": "generated",
            "schema_version": CONSTRAINT_GRAPH_SCHEMA,
            "constraint_graph": str(args.out.resolve()),
            "constraint_graph_artifact_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "constraint_graph_id": graph["constraint_graph_id"],
            "constraint_graph_sha256": graph["constraint_graph_sha256"],
            "coverage": graph["coverage"],
            "query_evidence": {
                "method": "digest_bound_articulation_plus_asserted_supplemental_constraint_compilation",
                "grammar_sha256": hashlib.sha256(args.grammar.read_bytes()).hexdigest(),
                "constraint_spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                "exact": True,
                "scope": "typed graph construction and executable residual semantics; asserted constraints are not independently observed physical truth",
            },
        }
        print(json_dump(result), end="")
        return 0
    if args.command == "evaluate-constraints":
        pose_name, pose = read_pose(args.pose)
        graph = read_constraint_graph(args.graph)
        result = evaluate_constraint_graph(graph, pose, pose_name, not args.no_local_analysis)
        result["query_evidence"] = {
            "method": "standalone_spanning_tree_execution_plus_typed_constraint_residuals_and_local_rank",
            "constraint_graph_path": str(args.graph.resolve()),
            "constraint_graph_artifact_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
            "exact_residuals": True,
            "local_rank_is_numerical": not args.no_local_analysis,
            "scope": "one explicit pose; local mobility is not a global configuration-space proof",
        }
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "satisfied" else 1
    if args.command == "solve-constraints":
        pose_name, pose = read_pose(args.pose)
        graph = read_constraint_graph(args.graph)
        result = solve_constraint_graph(
            graph,
            pose,
            args.solve_for,
            pose_name,
            args.max_iterations,
            args.damping,
        )
        result["query_evidence"] = {
            "method": "damped_gauss_newton_over_explicit_independent_driver_subset",
            "constraint_graph_artifact_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
            "exact_constraint_evaluation": True,
            "solver_is_local": True,
        }
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "converged" else 1
    if args.command == "verify-constraint-graph":
        pose_name, pose = read_pose(args.pose)
        result = verify_constraint_graph(args.grammar, args.spec, args.graph, pose, pose_name)
        result["query_evidence"] = {
            "method": "exact_regeneration_standalone_execution_and_local_analysis_reproducibility",
            "exact": True,
            "scope": "artifact integrity and execution; not independent validation that asserted constraints describe the physical mechanism",
        }
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "passed" else 1
    if args.command == "configuration-atlas":
        atlas = write_configuration_atlas(args.out, args.graph, args.spec)
        result = {
            "status": "generated" if atlas["status"] == "complete_for_declared_sampling" else "generated_partial",
            "schema_version": CONFIGURATION_ATLAS_SCHEMA,
            "configuration_atlas": str(args.out.resolve()),
            "configuration_atlas_artifact_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "configuration_atlas_id": atlas["configuration_atlas_id"],
            "configuration_atlas_sha256": atlas["configuration_atlas_sha256"],
            "sampling_status": atlas["status"],
            "coverage": atlas["coverage"],
            "query_evidence": {
                "method": "digest_bound_multi_seed_local_solves_over_explicit_one_parameter_charts",
                "exact_node_constraint_evaluation": True,
                "finite_numerical_branch_and_singularity_evidence": True,
                "scope": "declared finite sampling only; not exhaustive branch coverage, certified singularity classification, or global configuration-space topology",
            },
        }
        print(json_dump(result), end="")
        return 0 if atlas["status"] == "complete_for_declared_sampling" else 1
    if args.command == "verify-configuration-atlas":
        result = verify_configuration_atlas(args.graph, args.spec, args.atlas)
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result["status"] == "passed" else 1
    if args.command in {"articulation-grammar", "verify-articulation-grammar"}:
        source_format = detect_source_format(args.urdf, args.format)
        model = RobotModel(args.urdf) if source_format == "urdf" else load_imported_model(args.urdf, source_format)
        source_binding = articulation_source_binding(model, source_format)
        if args.command == "verify-articulation-grammar":
            if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
                raise SpatialError("--tolerance must be finite and positive")
            report = verify_articulation_grammar(args.grammar, model, source_binding, args.tolerance)
            report["query_evidence"] = {
                "method": "deterministic_grammar_regeneration_standalone_ast_execution_and_all_frame_source_fk_probes",
                "source_format": source_format,
                "source_sha256": model.sha256,
                "source_semantic_sha256": model.semantic_sha256,
                "exact": True,
                "scope": "supported source-tree semantics and normalized articulation law; not an independent parser oracle or physical validation",
            }
            serialized = json_dump(report)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0 if report["status"] == "passed" else 1
        grammar = write_articulation_grammar(args.out, model, source_binding)
        result = {
            "status": "generated",
            "source_format": source_format,
            "grammar": str(args.out.resolve()),
            "grammar_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "schema_version": ARTICULATION_SCHEMA,
            "grammar_id": grammar["grammar_id"],
            "grammar_input_sha256": grammar["grammar_input_sha256"],
            "law_identity": grammar["law_identity"],
            "coverage": grammar["coverage"],
            "query_evidence": {
                "method": "source_tree_normalization_to_typed_pre_motion_post_motion_articulation_law",
                "source_format": source_format,
                "source_sha256": model.sha256,
                "source_semantic_sha256": model.semantic_sha256,
                "exact": True,
                "scope": "pose-independent law for the supported source subset; not dynamics, closed loops, hardware, or physical evidence",
            },
            "next_action": "evaluate at a new driver binding, verify against the source, or compare with another representation",
        }
        print(json_dump(result), end="")
        return 0
    model = RobotModel(args.urdf)
    if args.command == "validate":
        semantics = read_semantics(args.semantics, model)
        srdf = parse_srdf(args.srdf, model)
        world_scene = read_world_scene(args.scene, model)
        result = {
            "status": "valid_with_warnings" if model.warnings() else "valid",
            "robot": model.name,
            "root_link": model.root_link,
            "links": len(model.links),
            "joints": len(model.joints),
            "warnings": model.warnings(),
            "semantics": "valid" if semantics else "not_provided",
            "srdf": "valid" if srdf else "not_provided",
            "world_scene": "parsed_validated_and_bound" if world_scene else "not_provided",
            "checked": ["XML", "single-root tree", "supported joint types", "link references", "joint axes", "mimic references and type compatibility", "declared geometry syntax and primitive dimensions", "declared inertial completeness, positive mass, positive-semidefinite tensor, and rigid-body principal-moment triangle inequality", "embedded ros2_control and legacy transmission names, interfaces, and joint references"] + ([] if world_scene is None else ["scene schema and IDs", "scene frame tree and cycle freedom", "robot/root identity binding", "scene poses, geometry declarations, gravity frame, snapshot, and provenance syntax"]),
            "not_checked": ["mesh contents", "triangle-level collision", "full dynamic response", "controller configuration outside the expanded URDF", "plugin availability", "hardware connectivity or behavior", "physical agreement of mass/inertia with hardware or payload", "physical completeness, currency, or accuracy of the declared world snapshot"] + ([] if srdf else ["SRDF"]) + ([] if world_scene else ["world scene"]),
        }
        print(json_dump(result), end="")
        return 0
    if args.command == "tree":
        print("\n".join(model.tree_lines()))
        return 0
    if args.command == "chain":
        result = attach_query_evidence(
            model.chain(args.start_link, args.end_link),
            model,
            "chain",
            {"from_link": args.start_link, "to_link": args.end_link},
            "validated_kinematic_tree_path",
            "pose-independent graph path; it does not by itself report a pose or collision state",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "affects":
        result = attach_query_evidence(
            model.affected_by_joint(args.joint),
            model,
            "affects",
            {"joint": args.joint},
            "kinematic_tree_and_mimic_causality",
            "structural ability to change descendant poses; not a claim that motion occurs in a particular commanded trajectory",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "actuation":
        if args.joint is not None:
            if args.joint not in model.joints:
                raise SpatialError(f"unknown joint {args.joint!r}")
            selection = {"type": "joint", "name": args.joint}
            payload = {
                "joint": args.joint,
                "joint_type": model.joints[args.joint].type,
                "joint_dynamics_declaration": model.joints[args.joint].dynamics,
                "bindings": model.actuation["joint_bindings"][args.joint],
                "ros2_control_systems": {
                    record["system"]: {
                        "name": record["system"],
                        "type": model.actuation["ros2_control_systems"][record["system"]]["type"],
                        "hardware": model.actuation["ros2_control_systems"][record["system"]]["hardware"],
                        "joint": model.actuation["ros2_control_systems"][record["system"]]["joints"][args.joint],
                    }
                    for record in model.actuation["joint_bindings"][args.joint]["ros2_control"]
                },
                "legacy_transmissions": {
                    name: model.actuation["legacy_transmissions"][name]
                    for name in model.actuation["joint_bindings"][args.joint]["legacy_transmissions"]
                },
            }
        elif args.system is not None:
            if args.system not in model.actuation["ros2_control_systems"]:
                raise SpatialError(f"unknown ros2_control system {args.system!r}")
            selection = {"type": "ros2_control_system", "name": args.system}
            payload = model.actuation["ros2_control_systems"][args.system]
        elif args.transmission is not None:
            if args.transmission not in model.actuation["legacy_transmissions"]:
                raise SpatialError(f"unknown legacy transmission {args.transmission!r}")
            selection = {"type": "legacy_transmission", "name": args.transmission}
            payload = model.actuation["legacy_transmissions"][args.transmission]
        else:
            selection = {"type": "all_embedded_declarations"}
            payload = model.actuation
        result = {
            "schema_version": "robot-spatial-actuation-query.v1",
            "selection": selection,
            "declarations": payload,
            "epistemic_scope": model.actuation["epistemic_scope"],
        }
        attach_query_evidence(
            result,
            model,
            "actuation",
            selection,
            "expanded_urdf_actuation_declaration_transcription_and_reference_validation",
            "describes embedded declarations only; it does not establish runtime controller, plugin, interface-claim, or hardware capability",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "check-invariants":
        srdf = parse_srdf(args.srdf, model)
        world_scene = read_world_scene(args.scene, model)
        if (args.observations is None) != (args.observation_query is None):
            raise SpatialError("--observations and --observation-query must be provided together")
        if args.observations is not None and world_scene is None:
            raise SpatialError("--observations requires --scene")
        invariant_observation = (
            None
            if args.observations is None
            else resolve_observation(args.observations, args.observation_query, model, world_scene)
        )
        contract = read_invariant_contract(args.contract, model, world_scene, invariant_observation)
        report = verify_invariant_contract(model, contract, srdf, args.package_map, world_scene, invariant_observation)
        parameters = {
            "contract_sha256": contract.get("source", {}).get("sha256"),
            "srdf_provided": srdf is not None,
            "scene_sha256": None if world_scene is None else world_scene.sha256,
        }
        if invariant_observation is None:
            attach_query_evidence(
                report,
                model,
                "check-invariants",
                parameters,
                "project_invariant_contract_evaluation",
                "asserted project intent evaluated against this URDF; a passing contract is not a general safety proof",
            )
        else:
            assert world_scene is not None
            attach_observation_query_evidence(
                report,
                model,
                world_scene,
                invariant_observation,
                "check-invariants",
                parameters,
                "project_invariant_contract_evaluation_with_time_selected_observation_state",
                "asserted project intent evaluated against bound model/scene/observation artifacts; passing does not establish sensor truth or safety",
            )
        serialized = json_dump(report)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if report["status"] == "passed" else 1
    if args.command in {"observe-summary", "observe-transform", "observe-collisions", "observe-gravity-loads"}:
        world_scene = read_world_scene(args.scene, model)
        assert world_scene is not None
        resolved = resolve_observation(
            args.observations,
            args.observation_query,
            model,
            world_scene,
        )
        observation_report = resolved["report"]
        pose = resolved["joint_pose"]
        root_transform = resolved["world_from_robot_root"]
        object_transforms = resolved["world_from_objects"]
        if args.command == "observe-transform":
            if resolved["nominal_computable"]:
                assert pose is not None and root_transform is not None
                transform_value = world_scene.transform(
                    args.reference,
                    args.target,
                    model,
                    pose,
                    world_from_robot_root=root_transform,
                    world_from_objects=object_transforms,
                )
                result = {
                    "schema_version": "robot-spatial-observed-transform.v1",
                    "status": "computed_from_current_observations" if resolved["all_required_current"] else "computed_nominally_with_declaration_fallback",
                    "observation": observation_report,
                    "transform": f"{args.reference}_from_{args.target}",
                    "meaning": f"nominal pose of typed entity {args.target} expressed in typed entity {args.reference}",
                    **pose_record(transform_value),
                    "physical_truth": "not_established",
                }
            else:
                result = {
                    "schema_version": "robot-spatial-observed-transform.v1",
                    "status": "not_computed",
                    "reason": "required current joint/root/object state is unavailable under the query fallback policy",
                    "observation": observation_report,
                    "transform": f"{args.reference}_from_{args.target}",
                    "physical_truth": "not_established",
                }
            attach_observation_query_evidence(
                result,
                model,
                world_scene,
                resolved,
                args.command,
                {"from_entity": args.reference, "to_entity": args.target},
                "latest_past_sample_zero_order_hold_then_scene_graph_and_urdf_forward_kinematics",
                "nominal transform under selected source reports and explicit declaration fallbacks; source truth, calibration, interpolation, and physical agreement are not established",
            )
            print(json_dump(result), end="")
            return 0

        collision: dict[str, Any]
        gravity_loads: dict[str, Any]
        if resolved["nominal_computable"]:
            assert pose is not None and root_transform is not None
            collision = world_scene.robot_environment_collisions(
                model,
                pose,
                getattr(args, "package_map", None),
                getattr(args, "contact_tolerance_m", 1e-9),
                world_from_robot_root=root_transform,
                world_from_objects=object_transforms,
            )
            gravity_loads = scene_gravity_load_analysis(
                model,
                world_scene,
                pose,
                "observed_at_query_time",
                root_transform,
            )
        else:
            collision = {
                "status": "not_computed",
                "reason": "required current joint/root/object state is unavailable under the query fallback policy",
            }
            gravity_loads = {
                "schema_version": "robot-spatial-observed-gravity-loads.v1",
                "status": "not_computed",
                "reason": "required current joint/root state is unavailable under the query fallback policy",
                "loads": None,
            }

        if args.command == "observe-collisions":
            result = {
                "schema_version": "robot-spatial-observed-robot-environment-collision.v1",
                "status": (
                    f"{collision['status']}_under_current_selected_observations"
                    if resolved["all_required_current"] and collision["status"] != "not_computed"
                    else ("nominal_only_not_all_required_observations_current" if collision["status"] != "not_computed" else "not_computed")
                ),
                "observation": observation_report,
                "nominal_declared_geometry_result": collision,
                "physical_collision_status": "not_established",
                "safety_conclusion": "not_established",
                "epistemic_scope": "collision geometry is evaluated at nominal selected poses only; covariance is not a hard bound, omitted physical objects remain unknown, and no continuous motion interval is checked",
            }
            attach_observation_query_evidence(
                result,
                model,
                world_scene,
                resolved,
                args.command,
                {"contact_tolerance_m": args.contact_tolerance_m},
                "latest_past_sample_zero_order_hold_then_all_declared_robot_environment_geometry_pairs",
                result["epistemic_scope"],
            )
            serialized = json_dump(result)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0

        if args.command == "observe-gravity-loads":
            result = {
                "schema_version": "robot-spatial-observed-gravity-loads.v1",
                "status": (
                    gravity_loads["status"]
                    if resolved["all_required_current"]
                    else ("nominal_only_not_all_required_observations_current" if gravity_loads["status"] != "not_computed" else "not_computed")
                ),
                "observation": observation_report,
                "nominal_declared_model_result": gravity_loads,
                "actual_hardware_load": "not_established",
                "epistemic_scope": "gravity-only static effort from declared inertials at selected joint/root reports; excludes motion, contacts, payload, transmission loss, controller behavior, and sensor truth",
            }
            attach_observation_query_evidence(
                result,
                model,
                world_scene,
                resolved,
                args.command,
                {},
                "latest_past_joint_and_root_samples_then_scene_gravity_rotation_and_declared_model_projection",
                result["epistemic_scope"],
            )
            print(json_dump(result), end="")
            return 0

        result = {
            "schema_version": "robot-spatial-observed-world-summary.v1",
            "status": observation_report["status"],
            "observation": observation_report,
            "nominal_analysis": {
                "robot_environment_collision": collision,
                "declared_static_gravity_loads": gravity_loads,
            },
            "physical_world_truth": "not_established",
        }
        attach_observation_query_evidence(
            result,
            model,
            world_scene,
            resolved,
            args.command,
            {"contact_tolerance_m": args.contact_tolerance_m},
            "latest_past_sample_zero_order_hold_then_bound_scene_kinematics_geometry_and_gravity_analysis",
            "time-policy-conditioned nominal state; current sample age does not establish calibration, omitted-object absence, or physical safety",
        )
        print(json_dump(result), end="")
        return 0
    observation_resolved: dict[str, Any] | None = None
    if args.command == "export" and (args.observations is not None or args.observation_query is not None):
        if args.observations is None or args.observation_query is None:
            raise SpatialError("--observations and --observation-query must be provided together")
        if args.scene is None:
            raise SpatialError("--observations requires --scene")
        if args.pose is not None or args.pose_name is not None:
            raise SpatialError("observed joint state is the export pose; do not combine --observations with --pose or --pose-name")
        observation_scene = read_world_scene(args.scene, model)
        assert observation_scene is not None
        observation_resolved = resolve_observation(
            args.observations,
            args.observation_query,
            model,
            observation_scene,
        )
        if observation_resolved["joint_pose"] is None:
            raise SpatialError("export requires a current joint-state sample under the observation query age policy")
        pose_name = f"observed/{observation_resolved['report']['query']['query_id']}"
        pose = observation_resolved["joint_pose"]
        srdf = parse_srdf(args.srdf, model)
    else:
        pose_name, pose, srdf = resolve_pose_input(args, model)
    if args.command == "export":
        if args.configuration_atlas_spec is not None and args.constraint_spec is None:
            raise SpatialError("--configuration-atlas-spec requires --constraint-spec")
        semantics = read_semantics(args.semantics, model)
        world_scene = read_world_scene(args.scene, model)
        canonical = model.canonical(
            pose,
            pose_name,
            semantics,
            args.inspect_meshes,
            args.package_map,
            srdf,
            args.workspace_samples,
            args.include_workspace_samples,
            args.surface_collisions,
            args.contact_tolerance_m,
            args.inspect_mesh_kinds,
            world_scene,
        )
        if observation_resolved is not None:
            assert world_scene is not None
            canonical["observed_world"] = {
                "observation": observation_resolved["report"],
                "analysis": observation_world_analysis(
                    model,
                    world_scene,
                    observation_resolved,
                    args.package_map,
                    args.contact_tolerance_m,
                ),
            }
            canonical["capabilities"]["timestamped_observation_resolution"] = True
            canonical["capabilities"]["observation_conditioned_world_analysis"] = True
        else:
            canonical["observed_world"] = {
                "status": "not_provided",
                "meaning": "provide --observations and --observation-query with --scene to distinguish timestamped source reports from model and static-scene declarations",
            }
            canonical["capabilities"]["timestamped_observation_resolution"] = False
            canonical["capabilities"]["observation_conditioned_world_analysis"] = False
        if args.invariants is not None:
            invariant_contract = read_invariant_contract(args.invariants, model, world_scene, observation_resolved)
            invariant_report = verify_invariant_contract(model, invariant_contract, srdf, args.package_map, world_scene, observation_resolved)
        else:
            invariant_report = {
                "status": "not_provided",
                "meaning": "provide --invariants to enforce project spatial intent as an edit acceptance gate",
            }
        canonical["invariant_validation"] = invariant_report
        canonical["capabilities"]["spatial_invariant_contract"] = args.invariants is not None
        canonical["capabilities"]["semantic_render_atlas"] = {
            "generated": False,
            "meaning": "run export --render for digest-bound machine-verifiable semantic projections",
        }
        canonical["capabilities"]["counterfactual_motion_atlas"] = {
            "generated": False,
            "meaning": "run export --motion-atlas to expose independent-joint causes and finite signed endpoint effects",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        articulation_path = args.out / "articulation-grammar.json"
        articulation = write_articulation_grammar(
            articulation_path,
            model,
            articulation_source_binding(model, "urdf"),
        )
        canonical.setdefault("artifacts", {})["articulation_grammar"] = {
            "path": "articulation-grammar.json",
            "sha256": hashlib.sha256(articulation_path.read_bytes()).hexdigest(),
            "schema_version": ARTICULATION_SCHEMA,
            "grammar_id": articulation["grammar_id"],
            "grammar_input_sha256": articulation["grammar_input_sha256"],
            "law_identity": articulation["law_identity"],
            "source_binding": articulation["source_binding"],
            "coordinate_contract": articulation["coordinate_contract"],
            "language_contract": articulation["language_contract"],
            "independent_variables": articulation["independent_variables"],
            "joint_position_rules": articulation["joint_position_rules"],
            "joint_operators": articulation["joint_operators"],
            "frame_derivations": articulation["frame_derivations"],
            "evaluation_contract": articulation["evaluation_contract"],
            "layer_contract": articulation["layer_contract"],
            "coverage": articulation["coverage"],
            "epistemic_scope": articulation["epistemic_scope"],
        }
        canonical["capabilities"]["articulation_grammar"] = {
            "generated": True,
            "schema_version": ARTICULATION_SCHEMA,
            "grammar_id": articulation["grammar_id"],
            "canonical_law_id": articulation["law_identity"]["canonical_law_id"],
            "canonical_law_sha256": articulation["law_identity"]["canonical_law_sha256"],
            "pose_independent": True,
            "standalone_executable": True,
            "independent_driver_count": articulation["coverage"]["independent_driver_count"],
            "frame_derivation_count": articulation["coverage"]["frame_derivation_count"],
            "verifier": "verify-articulation-grammar",
            "independent_parser_fk_oracle": False,
        }
        constraint_evaluation: dict[str, Any] | None = None
        constraint_graph: dict[str, Any] | None = None
        configuration_atlas: dict[str, Any] | None = None
        if args.constraint_spec is not None:
            constraint_graph_path = args.out / "constraint-graph.json"
            constraint_graph = write_constraint_graph(
                constraint_graph_path,
                articulation_path,
                args.constraint_spec,
            )
            constraint_evaluation = evaluate_constraint_graph(
                constraint_graph,
                pose,
                pose_name,
                True,
            )
            constraint_evaluation_path = args.out / "constraint-evaluation.json"
            constraint_evaluation_path.write_text(json_dump(constraint_evaluation), encoding="utf-8")
            canonical.setdefault("artifacts", {})["constraint_graph"] = {
                "path": "constraint-graph.json",
                "sha256": hashlib.sha256(constraint_graph_path.read_bytes()).hexdigest(),
                "schema_version": CONSTRAINT_GRAPH_SCHEMA,
                "constraint_graph_id": constraint_graph["constraint_graph_id"],
                "constraint_graph_sha256": constraint_graph["constraint_graph_sha256"],
                "source_binding": constraint_graph["source_binding"],
                "attachments": constraint_graph["attachments"],
                "constraints": constraint_graph["constraints"],
                "structural_graph": constraint_graph["structural_graph"],
                "executable_contract": constraint_graph["executable_contract"],
                "coverage": constraint_graph["coverage"],
                "epistemic_scope": constraint_graph["epistemic_scope"],
                "evaluation": {
                    "path": "constraint-evaluation.json",
                    "sha256": hashlib.sha256(constraint_evaluation_path.read_bytes()).hexdigest(),
                    "pose": constraint_evaluation["pose"],
                    "reference_frame": constraint_evaluation["reference_frame"],
                    "status": constraint_evaluation["status"],
                    "constraint_count": constraint_evaluation["constraint_count"],
                    "residual_component_count": constraint_evaluation["residual_component_count"],
                    "maximum_normalized_abs": constraint_evaluation["maximum_normalized_abs"],
                    "attachments": constraint_evaluation["attachments"],
                    "constraints": constraint_evaluation["constraints"],
                    "local_constraint_analysis": constraint_evaluation.get("local_constraint_analysis"),
                },
            }
            canonical["capabilities"]["supplemental_constraint_graph"] = {
                "generated": True,
                "schema_version": CONSTRAINT_GRAPH_SCHEMA,
                "constraint_graph_id": constraint_graph["constraint_graph_id"],
                "spanning_tree_is_complete_mechanism": not constraint_graph["structural_graph"]["tree_is_parameterization_not_complete_mechanism"],
                "declared_cycle_count": constraint_graph["coverage"]["declared_cycle_count"],
                "constraint_count": constraint_graph["coverage"]["constraint_count"],
                "export_pose_constraint_status": constraint_evaluation["status"],
                "local_mobility_is_pose_conditioned_numerical": True,
                "solver_is_local": True,
                "physical_truth": "not_established",
            }
            if args.configuration_atlas_spec is not None:
                configuration_atlas_path = args.out / "configuration-atlas.json"
                configuration_atlas = write_configuration_atlas(
                    configuration_atlas_path,
                    constraint_graph_path,
                    args.configuration_atlas_spec,
                )
                canonical.setdefault("artifacts", {})["configuration_atlas"] = {
                    "path": "configuration-atlas.json",
                    "sha256": hashlib.sha256(configuration_atlas_path.read_bytes()).hexdigest(),
                    "schema_version": CONFIGURATION_ATLAS_SCHEMA,
                    "configuration_atlas_id": configuration_atlas["configuration_atlas_id"],
                    "configuration_atlas_sha256": configuration_atlas["configuration_atlas_sha256"],
                    "status": configuration_atlas["status"],
                    "source_binding": configuration_atlas["source_binding"],
                    "exploration_contract": configuration_atlas["exploration_contract"],
                    "coverage": configuration_atlas["coverage"],
                    "charts": configuration_atlas["charts"],
                    "epistemic_scope": configuration_atlas["epistemic_scope"],
                }
                canonical["capabilities"]["finite_configuration_atlas"] = {
                    "generated": True,
                    "schema_version": CONFIGURATION_ATLAS_SCHEMA,
                    "configuration_atlas_id": configuration_atlas["configuration_atlas_id"],
                    "status": configuration_atlas["status"],
                    "declared_sample_minima_met": configuration_atlas["coverage"]["all_declared_sample_minima_met"],
                    "configuration_node_count": configuration_atlas["coverage"]["unique_solution_node_count"],
                    "singularity_candidate_node_count": configuration_atlas["coverage"]["singularity_candidate_node_count"],
                    "verifier": "verify-configuration-atlas",
                    "global_topology_or_certified_branch_enumeration": False,
                }
            else:
                canonical["capabilities"]["finite_configuration_atlas"] = {
                    "generated": False,
                    "meaning": "provide --configuration-atlas-spec with --constraint-spec to explore explicit finite one-parameter charts",
                }
        else:
            canonical["capabilities"]["supplemental_constraint_graph"] = {
                "generated": False,
                "meaning": "provide --constraint-spec when the spanning tree omits loop closures, cross-branch attachments, or coordinate couplings",
            }
            canonical["capabilities"]["finite_configuration_atlas"] = {
                "generated": False,
                "meaning": "a supplemental constraint graph and a configuration atlas spec are required",
            }
        if args.invariants is not None:
            invariant_report_path = args.out / "invariants-report.json"
            invariant_report_path.write_text(json_dump(invariant_report), encoding="utf-8")
            canonical.setdefault("artifacts", {})["invariant_report"] = {
                "path": "invariants-report.json",
                "schema_version": invariant_report["schema_version"],
                "status": invariant_report["status"],
            }
        if args.render:
            render_mesh_kinds = mesh_inspection_kinds(args.inspect_meshes, args.inspect_mesh_kinds)
            if args.surface_collisions:
                render_mesh_kinds.add("collision")
            _, render_points, _ = model.geometry_analysis(
                pose,
                package_map_path=args.package_map,
                inspect_mesh_kinds=render_mesh_kinds,
            )
            annotated_frames = list((semantics or {}).get("frames", {}))
            highlight_frames = list(dict.fromkeys([model.root_link, *annotated_frames]))
            scene_record = render_scene_svg(args.out / "scene.svg", render_points, canonical["geometry_analysis"], canonical["frames"], canonical["joints"], highlight_frames)
            scene_record["path"] = "scene.svg"
            canonical.setdefault("artifacts", {})["scene_svg"] = scene_record
            atlas = write_semantic_render_atlas(
                args.out / "render-atlas",
                render_points,
                canonical["geometry_analysis"],
                canonical["frames"],
                canonical["joints"],
                highlight_frames,
                {
                    "robot_name": model.name,
                    "root_frame": model.root_link,
                    "urdf_sha256": model.sha256,
                    "urdf_semantic_sha256": model.semantic_sha256,
                },
                pose_name,
                pose,
                args.out / "scene.svg",
            )
            atlas_manifest_path = args.out / "render-atlas" / "manifest.json"
            canonical["artifacts"]["semantic_render_atlas"] = {
                "path": "render-atlas/manifest.json",
                "manifest_sha256": hashlib.sha256(atlas_manifest_path.read_bytes()).hexdigest(),
                "schema_version": ATLAS_SCHEMA,
                "render_id": atlas["render_id"],
                "render_input_sha256": atlas["render_input_sha256"],
                "pose_binding": atlas["pose_binding"],
                "coordinate_contract": atlas["coordinate_contract"],
                "coverage": atlas["coverage"],
                "views": atlas["views"],
                "epistemic_scope": atlas["epistemic_scope"],
            }
            canonical["capabilities"]["semantic_render_atlas"] = {
                "generated": True,
                "schema_version": ATLAS_SCHEMA,
                "render_id": atlas["render_id"],
                "view_count": atlas["coverage"]["view_count"],
                "complete_for_declared_geometry": atlas["coverage"]["complete_for_declared_geometry"],
                "view_numeric_consistency_verifier": "verify-render",
                "independent_spatial_oracle": False,
            }
        if args.motion_atlas:
            motion_atlas = write_counterfactual_motion_atlas(
                args.out / "motion-atlas",
                model,
                pose_name,
                pose,
                {
                    "robot_name": model.name,
                    "root_frame": model.root_link,
                    "urdf_sha256": model.sha256,
                    "urdf_semantic_sha256": model.semantic_sha256,
                },
                args.inspect_meshes,
                args.package_map,
                args.inspect_mesh_kinds,
                args.motion_angular_step_rad,
                args.motion_linear_step_m,
            )
            motion_manifest_path = args.out / "motion-atlas" / "manifest.json"
            canonical.setdefault("artifacts", {})["counterfactual_motion_atlas"] = {
                "path": "motion-atlas/manifest.json",
                "manifest_sha256": hashlib.sha256(motion_manifest_path.read_bytes()).hexdigest(),
                "schema_version": MOTION_ATLAS_SCHEMA,
                "motion_id": motion_atlas["motion_id"],
                "motion_input_sha256": motion_atlas["motion_input_sha256"],
                "baseline_pose_binding": motion_atlas["baseline_pose_binding"],
                "perturbation_policy": motion_atlas["perturbation_policy"],
                "coordinate_contract": motion_atlas["coordinate_contract"],
                "coverage": motion_atlas["coverage"],
                "drivers": motion_atlas["drivers"],
                "epistemic_scope": motion_atlas["epistemic_scope"],
            }
            canonical["capabilities"]["counterfactual_motion_atlas"] = {
                "generated": True,
                "schema_version": MOTION_ATLAS_SCHEMA,
                "motion_id": motion_atlas["motion_id"],
                "independent_driver_count": motion_atlas["coverage"]["independent_driver_count"],
                "available_signed_endpoint_count": motion_atlas["coverage"]["available_signed_endpoint_count"],
                "finite_endpoint_consistency_verifier": "verify-motion-atlas",
                "continuous_motion_or_dynamics": False,
                "independent_motion_or_physical_oracle": False,
            }
        concept_graph_path = args.out / "concept-graph.json"
        concept_language_path = args.out / "concept-language.rsl"
        concept_graph = write_concept_graph(
            concept_graph_path,
            concept_language_path,
            canonical,
            articulation,
            constraint_graph,
            configuration_atlas,
        )
        canonical.setdefault("artifacts", {})["concept_graph"] = {
            "path": "concept-graph.json",
            "sha256": hashlib.sha256(concept_graph_path.read_bytes()).hexdigest(),
            "schema_version": CONCEPT_SCHEMA,
            "concept_graph_id": concept_graph["concept_graph_id"],
            "concept_graph_sha256": concept_graph["concept_graph_sha256"],
            "language_path": "concept-language.rsl",
            "language_sha256": hashlib.sha256(concept_language_path.read_bytes()).hexdigest(),
            "coverage": concept_graph["coverage"],
            "epistemic_scope": concept_graph["epistemic_scope"],
        }
        canonical["capabilities"]["proof_carrying_spatial_concept_graph"] = {
            "generated": True,
            "schema_version": CONCEPT_SCHEMA,
            "concept_graph_id": concept_graph["concept_graph_id"],
            "entity_count": concept_graph["coverage"]["entity_count"],
            "clause_count": concept_graph["coverage"]["clause_count"],
            "strict_query_ast": True,
            "minimal_proof_closure": True,
            "exact_regeneration_verifier": "verify-concept-graph",
            "physical_or_global_configuration_truth": False,
        }
        functional_model: dict[str, Any] | None = None
        if args.functional_spec is not None:
            functional_model_path = args.out / "functional-model.json"
            concept_graph_artifact_sha256 = hashlib.sha256(concept_graph_path.read_bytes()).hexdigest()
            functional_model = write_functional_model(
                functional_model_path,
                canonical,
                concept_graph,
                concept_graph_artifact_sha256,
                args.functional_spec,
            )
            canonical.setdefault("artifacts", {})["functional_model"] = {
                "path": "functional-model.json",
                "sha256": hashlib.sha256(functional_model_path.read_bytes()).hexdigest(),
                "schema_version": FUNCTIONAL_MODEL_SCHEMA,
                "functional_model_id": functional_model["functional_model_id"],
                "functional_model_sha256": functional_model["functional_model_sha256"],
                "function_set_id": functional_model["function_set_id"],
                "status": functional_model["status"],
                "coverage": functional_model["coverage"],
                "epistemic_scope": functional_model["epistemic_scope"],
            }
            canonical["capabilities"]["proof_carrying_function_affordance_model"] = {
                "generated": True,
                "schema_version": FUNCTIONAL_MODEL_SCHEMA,
                "functional_model_id": functional_model["functional_model_id"],
                "function_set_id": functional_model["function_set_id"],
                "grounding_status": functional_model["status"],
                "all_declared_capabilities_structurally_grounded": functional_model["coverage"][
                    "all_declared_capabilities_structurally_grounded"
                ],
                "strict_query_ast": True,
                "recursive_structural_proof_closure": True,
                "exact_regeneration_verifier": "verify-functional-model",
                "physical_capability_or_execution_truth": False,
            }
        else:
            canonical["capabilities"]["proof_carrying_function_affordance_model"] = {
                "generated": False,
                "meaning": (
                    "provide --functional-spec to declare component function, capability requirements, conditions, "
                    "intended effects, and relational affordances; never infer these from URDF names or geometry"
                ),
            }
        facts_path = args.out / "facts.jsonl"
        facts = fact_records(model, canonical)
        facts_path.write_text(jsonl_dump(facts), encoding="utf-8")
        canonical.setdefault("artifacts", {})["facts_jsonl"] = {
            "path": "facts.jsonl",
            "schema_version": "robot-spatial-fact.v1",
            "record_count": len(facts),
        }
        agent_context = write_agent_context(args.out, canonical, facts, facts_path, functional_model)
        canonical["artifacts"]["agent_context"] = agent_context
        if args.generate_evaluation:
            evaluation_key_path = args.evaluation_key_out or args.out.parent / f"{args.out.name}-evaluation-private" / "answer-key.jsonl"
            evaluation = generate_evaluation(
                canonical,
                facts,
                args.out / "evaluation",
                evaluation_key_path,
                concept_graph,
                functional_model,
            )
            canonical["artifacts"]["evaluation"] = {
                "manifest": "evaluation/manifest.json",
                "questions": "evaluation/questions.jsonl",
                "answer_template": "evaluation/answer-template.jsonl",
                "question_count": evaluation["question_count"],
                "capability_counts": evaluation["capability_counts"],
            }
        (args.out / "model.json").write_text(json_dump(canonical), encoding="utf-8")
        (args.out / "context.md").write_text(context_markdown(model, canonical), encoding="utf-8")
        invariant_failed = invariant_report["status"] == "failed"
        constraint_failed = constraint_evaluation is not None and constraint_evaluation["status"] != "satisfied"
        configuration_atlas_incomplete = (
            configuration_atlas is not None
            and configuration_atlas["status"] != "complete_for_declared_sampling"
        )
        functional_grounding_failed = (
            functional_model is not None
            and not functional_model["coverage"]["all_declared_capabilities_structurally_grounded"]
        )
        if invariant_failed:
            export_status = "exported_with_failed_invariants"
        elif constraint_failed:
            export_status = "exported_with_violated_constraints"
        elif configuration_atlas_incomplete:
            export_status = "exported_with_incomplete_configuration_atlas"
        elif functional_grounding_failed:
            export_status = "exported_with_ungrounded_functional_requirements"
        else:
            export_status = "exported"
        result = {
            "status": export_status,
            "model": str((args.out / "model.json").resolve()),
            "context": str((args.out / "context.md").resolve()),
            "facts": str(facts_path.resolve()),
            "agent_context": str((args.out / "agent-context.json").resolve()),
            "entity_cards": str((args.out / "entity-cards.jsonl").resolve()),
            "articulation_grammar": str(articulation_path.resolve()),
            "concept_graph": str(concept_graph_path.resolve()),
            "concept_language": str(concept_language_path.resolve()),
        }
        if functional_model is not None:
            result["functional_model"] = str((args.out / "functional-model.json").resolve())
        if constraint_evaluation is not None:
            result["constraint_graph"] = str((args.out / "constraint-graph.json").resolve())
            result["constraint_evaluation"] = str((args.out / "constraint-evaluation.json").resolve())
        if configuration_atlas is not None:
            result["configuration_atlas"] = str((args.out / "configuration-atlas.json").resolve())
        if args.invariants is not None:
            result["invariant_report"] = str((args.out / "invariants-report.json").resolve())
        if args.render:
            result["scene_svg"] = str((args.out / "scene.svg").resolve())
            result["render_atlas"] = str((args.out / "render-atlas" / "manifest.json").resolve())
        if args.motion_atlas:
            result["motion_atlas"] = str((args.out / "motion-atlas" / "manifest.json").resolve())
        if args.generate_evaluation:
            result["evaluation_manifest"] = evaluation["artifacts"]["manifest"]
            result["private_evaluation_answer_key"] = evaluation["private_artifacts"]["answer_key"]
        print(json_dump(result), end="")
        return 1 if invariant_failed or constraint_failed or configuration_atlas_incomplete or functional_grounding_failed else 0
    if args.command == "transform":
        transform_value = model.transform(args.reference, args.target, pose)
        result = {
            "pose": pose_name,
            "transform": f"{args.reference}_from_{args.target}",
            "meaning": f"pose of {args.target} expressed in {args.reference}",
            **pose_record(transform_value),
        }
        attach_query_evidence(
            result,
            model,
            "transform",
            {"from_frame": args.reference, "to_frame": args.target, "pose": pose_name, "joint_positions": pose},
            "forward_kinematics",
            "exact for the supported tree model, declared transforms, and stated pose",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "axis":
        result = {
            "pose": pose_name,
            "joint": args.joint,
            "expressed_in_frame": args.frame,
            "axis_unit_vector": model.axis(args.joint, args.frame, pose),
        }
        attach_query_evidence(
            result,
            model,
            "axis",
            {"joint": args.joint, "expressed_in_frame": args.frame, "pose": pose_name, "joint_positions": pose},
            "forward_kinematics_axis_rotation",
            "signed unit axis expressed in the requested frame at the stated pose",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "jacobian":
        result = model.geometric_jacobian(args.target, pose, args.frame)
        result["pose"] = pose_name
        attach_query_evidence(
            result,
            model,
            "jacobian",
            {"target_frame": args.target, "expressed_in_frame": args.frame or model.root_link, "pose": pose_name, "joint_positions": pose},
            "analytic_geometric_jacobian",
            "instantaneous local velocity mapping at the stated pose; not global reachability or collision freedom",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "mass-properties":
        result = model.mass_properties(pose, args.frame, args.subtree_root)
        result["pose"]["name"] = pose_name
        attach_query_evidence(
            result,
            model,
            "mass-properties",
            {
                "subtree_root_link": args.subtree_root or model.root_link,
                "expressed_in_frame": args.frame or model.root_link,
                "pose": pose_name,
                "joint_positions": pose,
            },
            "urdf_declared_inertials_forward_kinematics_parallel_axis_theorem",
            "exact for the selected declared inertial model and stated pose; physical completeness, payload, calibration, and hardware agreement are not established",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "gravity-loads":
        result = model.static_gravity_loads(
            pose,
            args.gravity,
            args.gravity_frame,
            args.subtree_root,
        )
        result["pose"]["name"] = pose_name
        attach_query_evidence(
            result,
            model,
            "gravity-loads",
            {
                "subtree_root_link": args.subtree_root or model.root_link,
                "gravity_frame": args.gravity_frame or model.root_link,
                "gravity_vector_xyz_m_s2": args.gravity,
                "pose": pose_name,
                "joint_positions": pose,
            },
            "urdf_declared_inertials_forward_kinematics_gravity_projection_with_mimic_chain_rule",
            "exact for the stated declared model and gravity convention; this is gravity-only static equilibrium, not full inverse dynamics or hardware feasibility",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "scene-summary":
        world_scene = read_world_scene(args.scene, model)
        assert world_scene is not None
        result = world_scene.canonical(
            model,
            pose,
            args.package_map,
            args.contact_tolerance_m,
        )
        attach_scene_query_evidence(
            result,
            model,
            world_scene,
            "scene-summary",
            {
                "pose": pose_name,
                "joint_positions": pose,
                "contact_tolerance_m": args.contact_tolerance_m,
            },
            "validated_static_scene_frame_graph_root_binding_geometry_and_collision_analysis",
            "exact consequences of the declared static snapshot; scene currency, physical completeness, and provenance accuracy are not independently established",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "scene-transform":
        world_scene = read_world_scene(args.scene, model)
        assert world_scene is not None
        transform_value = world_scene.transform(args.reference, args.target, model, pose)
        result = {
            "schema_version": "robot-spatial-scene-transform.v1",
            "scene_id": world_scene.scene_id,
            "snapshot_id": world_scene.snapshot["id"],
            "pose": pose_name,
            "transform": f"{args.reference}_from_{args.target}",
            "meaning": f"pose of typed entity {args.target} expressed in typed entity {args.reference}",
            **pose_record(transform_value),
        }
        attach_scene_query_evidence(
            result,
            model,
            world_scene,
            "scene-transform",
            {
                "from_entity": args.reference,
                "to_entity": args.target,
                "pose": pose_name,
                "joint_positions": pose,
            },
            "validated_scene_frame_graph_and_robot_forward_kinematics",
            "exact for the declared scene/root transforms, supported URDF tree, and stated pose; physical agreement is conditional on scene provenance",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "scene-collisions":
        world_scene = read_world_scene(args.scene, model)
        assert world_scene is not None
        result = world_scene.robot_environment_collisions(
            model,
            pose,
            args.package_map,
            args.contact_tolerance_m,
        )
        result["pose"] = {"name": pose_name, "joint_positions": pose}
        attach_scene_query_evidence(
            result,
            model,
            world_scene,
            "scene-collisions",
            {
                "pose": pose_name,
                "joint_positions": pose,
                "contact_tolerance_m": args.contact_tolerance_m,
            },
            "all_declared_robot_environment_geometry_pairs_with_exact_solid_or_fail_closed_aabb_analysis",
            "exact only for the declared snapshot, stated pose, measured geometry, supported representations, and tolerance; omitted or stale physical objects remain unknown",
        )
        serialized = json_dump(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "scene-gravity-loads":
        world_scene = read_world_scene(args.scene, model)
        assert world_scene is not None
        result = scene_gravity_load_analysis(model, world_scene, pose, pose_name)
        attach_scene_query_evidence(
            result,
            model,
            world_scene,
            "scene-gravity-loads",
            {"pose": pose_name, "joint_positions": pose},
            "scene_world_gravity_rotation_through_root_mount_then_urdf_inertial_gravity_projection",
            "exact for the declared model and snapshot convention; actual mounting, gravity, payload, contact, controller, and hardware truth depend on external evidence",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "workspace":
        result = model.workspace_envelope(args.target, pose, args.samples, args.include_samples)
        result["baseline_pose"] = pose_name
        attach_query_evidence(
            result,
            model,
            "workspace",
            {"target_frame": args.target, "baseline_pose": pose_name, "samples": args.samples, "include_samples": args.include_samples},
            "deterministic_finite_joint_space_sampling",
            "observed sample envelope only; it does not prove the complete reachable set or reachability of interior points",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "distance":
        transform_value = model.transform(args.reference, args.target, pose)
        delta = [transform_value[index][3] for index in range(3)]
        result = {
            "pose": pose_name,
            "from_frame": args.reference,
            "to_frame": args.target,
            "delta_expressed_in_from_frame_m": clean_vector(delta),
            "euclidean_distance_m": clean_number(math.sqrt(sum(value * value for value in delta))),
        }
        attach_query_evidence(
            result,
            model,
            "distance",
            {"from_frame": args.reference, "to_frame": args.target, "pose": pose_name, "joint_positions": pose},
            "forward_kinematics_frame_origin_distance",
            "Euclidean distance between frame origins at the stated pose; not surface clearance",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "bounds":
        analysis, _, _ = model.geometry_analysis(
            pose,
            args.inspect_meshes,
            args.package_map,
            args.inspect_mesh_kinds,
        )
        if args.geometry_frame not in analysis:
            raise SpatialError(f"unknown geometry frame {args.geometry_frame!r}")
        result = {"pose": pose_name, **analysis[args.geometry_frame]}
        attach_query_evidence(
            result,
            model,
            "bounds",
            {"geometry_frame": args.geometry_frame, "pose": pose_name, "joint_positions": pose},
            "declared_primitive_or_measured_mesh_aabb",
            "root-frame axis-aligned bounds at the stated pose; inspect status and mesh completeness before shape claims",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "overlaps":
        canonical = model.canonical(
            pose,
            pose_name,
            inspect_meshes=args.inspect_meshes,
            package_map_path=args.package_map,
            srdf=srdf,
            inspect_mesh_kinds=args.inspect_mesh_kinds,
        )
        broadphase = canonical["collision_broadphase"]
        result = {"pose": pose_name, **broadphase}
        attach_query_evidence(
            result,
            model,
            "overlaps",
            {"pose": pose_name, "joint_positions": pose},
            "root_frame_collision_aabb_broadphase",
            "overlap pairs are conservative candidates, not verified triangle or solid collisions",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "surface-distance":
        result = model.triangle_surface_distance(args.geometry_a, args.geometry_b, pose, args.package_map)
        result = {"pose": pose_name, **result}
        attach_query_evidence(
            result,
            model,
            "surface-distance",
            {"geometry_a": args.geometry_a, "geometry_b": args.geometry_b, "pose": pose_name, "joint_positions": pose},
            "triangle_bvh_surface_distance",
            "surface distance for supported measured triangles; positive separation alone does not exclude closed-solid containment",
        )
        print(json_dump(result), end="")
        return 0
    if args.command == "surface-collisions":
        canonical = model.canonical(
            pose,
            pose_name,
            package_map_path=args.package_map,
            srdf=srdf,
            surface_collisions=True,
            contact_tolerance_m=args.contact_tolerance_m,
        )
        report = {"pose": pose_name, **canonical["collision_surface"]}
        attach_query_evidence(
            report,
            model,
            "surface-collisions",
            {"pose": pose_name, "joint_positions": pose, "contact_tolerance_m": args.contact_tolerance_m},
            "triangle_surface_distance_and_closed_solid_containment",
            "exact only for supported, measured surface representations and the stated tolerance; inspect indeterminate candidates and SRDF policy separately",
        )
        serialized = json_dump(report)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command in {"motion-atlas", "verify-motion-atlas"}:
        source_binding = {
            "robot_name": model.name,
            "root_frame": model.root_link,
            "urdf_sha256": model.sha256,
            "urdf_semantic_sha256": model.semantic_sha256,
        }
        if args.command == "verify-motion-atlas":
            report = verify_counterfactual_motion_atlas(
                args.atlas,
                model,
                pose_name,
                pose,
                source_binding,
                args.inspect_meshes,
                args.package_map,
                args.inspect_mesh_kinds,
                args.motion_angular_step_rad,
                args.motion_linear_step_m,
            )
            attach_query_evidence(
                report,
                model,
                "verify-motion-atlas",
                {
                    "atlas_manifest_sha256": report["manifest_sha256"],
                    "baseline_pose": pose_name,
                    "joint_positions": pose,
                    "angular_step_rad": args.motion_angular_step_rad,
                    "linear_step_m": args.motion_linear_step_m,
                },
                "deterministic_signed_endpoint_fk_projection_and_artifact_regeneration",
                "verifies finite counterfactual endpoints against the same FK/geometry implementation; not a trajectory, dynamics engine, physical observation, or safety proof",
            )
            serialized = json_dump(report)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0 if report["status"] == "passed" else 1
        atlas = write_counterfactual_motion_atlas(
            args.out,
            model,
            pose_name,
            pose,
            source_binding,
            args.inspect_meshes,
            args.package_map,
            args.inspect_mesh_kinds,
            args.motion_angular_step_rad,
            args.motion_linear_step_m,
        )
        result = {
            "status": "generated",
            "manifest": str((args.out / "manifest.json").resolve()),
            "schema_version": MOTION_ATLAS_SCHEMA,
            "motion_id": atlas["motion_id"],
            "motion_input_sha256": atlas["motion_input_sha256"],
            "baseline_pose": pose_name,
            "coverage": atlas["coverage"],
        }
        attach_query_evidence(
            result,
            model,
            "motion-atlas",
            {
                "baseline_pose": pose_name,
                "joint_positions": pose,
                "angular_step_rad": args.motion_angular_step_rad,
                "linear_step_m": args.motion_linear_step_m,
            },
            "finite_signed_independent_driver_counterfactual_forward_kinematics",
            "exact for generated endpoint poses in the supported URDF tree; not time interpolation, continuous swept motion, dynamics, hardware behavior, or safety",
        )
        print(json_dump(result), end="")
        return 0
    if args.command in {"render", "verify-render"}:
        semantics = read_semantics(args.semantics, model)
        canonical = model.canonical(
            pose,
            pose_name,
            semantics,
            args.inspect_meshes,
            args.package_map,
            srdf,
            inspect_mesh_kinds=args.inspect_mesh_kinds,
        )
        _, render_points, _ = model.geometry_analysis(
            pose,
            args.inspect_meshes,
            args.package_map,
            args.inspect_mesh_kinds,
        )
        annotated_frames = list((semantics or {}).get("frames", {}))
        highlight_frames = list(dict.fromkeys([model.root_link, *annotated_frames]))
        if args.command == "verify-render":
            report = verify_semantic_render_atlas(
                args.atlas,
                render_points,
                canonical["geometry_analysis"],
                canonical["frames"],
                canonical["joints"],
                highlight_frames,
                {
                    "robot_name": model.name,
                    "root_frame": model.root_link,
                    "urdf_sha256": model.sha256,
                    "urdf_semantic_sha256": model.semantic_sha256,
                },
                pose_name,
                pose,
            )
            attach_query_evidence(
                report,
                model,
                "verify-render",
                {
                    "atlas_manifest_sha256": report["manifest_sha256"],
                    "pose": pose_name,
                    "joint_positions": pose,
                },
                "deterministic_semantic_projection_regeneration_and_artifact_digest_verification",
                "verifies view/numeric consistency against the same canonical geometry implementation; it is not an independent geometry engine or physical-world observation",
            )
            serialized = json_dump(report)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0 if report["status"] == "passed" else 1
        result = render_scene_svg(args.out, render_points, canonical["geometry_analysis"], canonical["frames"], canonical["joints"], highlight_frames)
        atlas_directory = args.atlas_out or args.out.parent / f"{args.out.stem}-atlas"
        atlas = write_semantic_render_atlas(
            atlas_directory,
            render_points,
            canonical["geometry_analysis"],
            canonical["frames"],
            canonical["joints"],
            highlight_frames,
            {
                "robot_name": model.name,
                "root_frame": model.root_link,
                "urdf_sha256": model.sha256,
                "urdf_semantic_sha256": model.semantic_sha256,
            },
            pose_name,
            pose,
            args.out,
        )
        result = {
            "status": "rendered",
            "pose": pose_name,
            **result,
            "semantic_render_atlas": {
                "manifest": str((atlas_directory / "manifest.json").resolve()),
                "schema_version": ATLAS_SCHEMA,
                "render_id": atlas["render_id"],
                "render_input_sha256": atlas["render_input_sha256"],
                "coverage": atlas["coverage"],
            },
        }
        attach_query_evidence(
            result,
            model,
            "render",
            {"pose": pose_name, "joint_positions": pose},
            "deterministic_orthographic_projection",
            "visual aid derived from declared or measured geometry; the image is not an independent spatial oracle",
        )
        print(json_dump(result), end="")
        return 0
    raise SpatialError(f"unknown command: {args.command}")


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, SpatialError, WorkspaceError, GeometryError, TriangleError, SRDFError, ContextError, EvaluationError, InvariantError, MotionError, ArticulationError, ConstraintError, ConfigurationError, ConceptError, FunctionalError, ActionAssuranceError, KinematicImportError, RenderError, SceneError, ObservationError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
