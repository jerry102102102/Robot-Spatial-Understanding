#!/usr/bin/env python3
"""Strict representation-neutral kinematic-tree import for SDF and canonical MJCF subsets."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EPSILON = 1e-12
SUPPORTED_FORMATS = {"sdf", "mjcf"}
SUPPORTED_JOINTS = {"fixed", "revolute", "continuous", "prismatic"}


class KinematicImportError(ValueError):
    """A source cannot be imported without guessing or dropping kinematic semantics."""


Matrix = list[list[float]]
Vector = list[float]


def _identity() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _inverse_rigid(transform: Matrix) -> Matrix:
    result = _identity()
    for row in range(3):
        for column in range(3):
            result[row][column] = transform[column][row]
        result[row][3] = -sum(transform[column][row] * transform[column][3] for column in range(3))
    return result


def _translation(vector: Vector) -> Matrix:
    result = _identity()
    for index in range(3):
        result[index][3] = vector[index]
    return result


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


def _pose_matrix(xyz: Vector, rpy: Vector) -> Matrix:
    result = _rpy_matrix(rpy)
    for index in range(3):
        result[index][3] = xyz[index]
    return result


def _quaternion_wxyz_matrix(values: Vector, label: str) -> Matrix:
    if len(values) != 4:
        raise KinematicImportError(f"{label} must contain four wxyz components")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= EPSILON:
        raise KinematicImportError(f"{label} quaternion must be non-zero")
    w, x, y, z = [value / norm for value in values]
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_angle(axis: Vector, angle: float) -> Matrix:
    x, y, z = _normalized(axis, "joint axis")
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotate(transform: Matrix, vector: Vector) -> Vector:
    return [sum(transform[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _normalized(vector: Vector, label: str) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= EPSILON:
        raise KinematicImportError(f"{label} must be non-zero")
    return [value / norm for value in vector]


def _clean(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(float(value), 12)


def _clean_vector(values: Iterable[float]) -> Vector:
    return [_clean(value) for value in values]


def _numbers(raw: str | None, count: int, label: str, default: Vector | None = None) -> Vector:
    if raw is None or not raw.strip():
        if default is None:
            raise KinematicImportError(f"{label} is required")
        return list(default)
    try:
        values = [float(value) for value in raw.split()]
    except ValueError as error:
        raise KinematicImportError(f"{label} must contain numbers") from error
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise KinematicImportError(f"{label} must contain exactly {count} finite numbers")
    return values


def _text(element: ET.Element | None, label: str, required: bool = True) -> str | None:
    value = None if element is None else "".join(element.itertext()).strip()
    if required and not value:
        raise KinematicImportError(f"{label} is required")
    return value or None


def _source_xml(path: Path, expected_root: str) -> tuple[ET.Element, bytes, str]:
    source = path.resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise KinematicImportError(f"cannot read source {source}: {error}") from error
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise KinematicImportError(f"invalid XML: {error}") from error
    if root.tag != expected_root:
        raise KinematicImportError(f"expected <{expected_root}> root, found <{root.tag}>")
    semantic = hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()
    return root, raw, semantic


def detect_source_format(path: Path, requested: str = "auto") -> str:
    if requested not in {"auto", "urdf", "sdf", "mjcf"}:
        raise KinematicImportError(f"unsupported source format {requested!r}")
    if requested != "auto":
        return requested
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise KinematicImportError(f"cannot detect XML source format: {error}") from error
    detected = {"robot": "urdf", "sdf": "sdf", "mujoco": "mjcf"}.get(root.tag)
    if detected is None:
        raise KinematicImportError(f"cannot detect source format from root element <{root.tag}>")
    return detected


@dataclass
class ImportedJoint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Vector
    origin_rpy: Vector
    origin: Matrix
    post_motion: Matrix
    axis: Vector
    limit: dict[str, float | None]
    mimic: dict[str, float | str] | None = None
    dynamics: dict[str, Any] | None = None


class KinematicTreeModel:
    """Duck-typed articulation model whose FK includes pre × motion × post edges."""

    def __init__(
        self,
        path: Path,
        source_format: str,
        name: str,
        raw: bytes,
        semantic_sha256: str,
        links: dict[str, dict[str, Any]],
        joints: dict[str, ImportedJoint],
        import_contract: dict[str, Any],
    ) -> None:
        self.path = path.resolve()
        self.source_format = source_format
        self.name = name
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.semantic_sha256 = semantic_sha256
        self.links = links
        self.joints = joints
        self.import_contract = import_contract
        self.child_joint: dict[str, ImportedJoint] = {}
        self.children: dict[str, list[ImportedJoint]] = {name: [] for name in links}
        for joint in joints.values():
            if joint.parent not in links or joint.child not in links:
                raise KinematicImportError(
                    f"joint {joint.name!r} references missing link {joint.parent!r} -> {joint.child!r}"
                )
            if joint.child in self.child_joint:
                raise KinematicImportError(f"link {joint.child!r} has multiple parent joints")
            self.child_joint[joint.child] = joint
            self.children[joint.parent].append(joint)
        roots = sorted(set(links) - set(self.child_joint))
        if len(roots) != 1:
            raise KinematicImportError(f"kinematic tree must have exactly one root link; found {roots}")
        self.root_link = roots[0]
        self._validate_tree()

    def _validate_tree(self) -> None:
        visited: set[str] = set()
        active: set[str] = set()

        def descend(link: str) -> None:
            if link in active:
                raise KinematicImportError(f"kinematic cycle detected at link {link!r}")
            if link in visited:
                return
            active.add(link)
            for joint in self.children[link]:
                descend(joint.child)
            active.remove(link)
            visited.add(link)

        descend(self.root_link)
        missing = sorted(set(self.links) - visited)
        if missing:
            raise KinematicImportError(f"links disconnected from root {self.root_link!r}: {missing}")

    def _mimic_affine_from_driver(self, joint_name: str) -> tuple[str, float, float, list[str]]:
        current, multiplier, offset = joint_name, 1.0, 0.0
        chain, visited = [current], set()
        while self.joints[current].mimic:
            if current in visited:
                raise KinematicImportError(f"mimic cycle detected at joint {current!r}")
            visited.add(current)
            mimic = self.joints[current].mimic
            assert mimic is not None
            offset = multiplier * float(mimic["offset"]) + offset
            multiplier *= float(mimic["multiplier"])
            current = str(mimic["joint"])
            chain.append(current)
        return current, multiplier, offset, chain

    def resolve_pose(self, supplied: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(supplied) - set(self.joints))
        if unknown:
            raise KinematicImportError(f"pose contains unknown joints: {unknown}")
        resolved: dict[str, float] = {}

        def resolve(name: str) -> float:
            if name in resolved:
                return resolved[name]
            joint = self.joints[name]
            if joint.type == "fixed":
                value = 0.0
            elif joint.mimic is not None:
                value = float(joint.mimic["multiplier"]) * resolve(str(joint.mimic["joint"])) + float(joint.mimic["offset"])
            else:
                value = float(supplied.get(name, 0.0))
            if not math.isfinite(value):
                raise KinematicImportError(f"joint {name!r} position must be finite")
            if joint.type in {"revolute", "prismatic"}:
                lower, upper = joint.limit["lower"], joint.limit["upper"]
                if lower is not None and value < float(lower) - EPSILON:
                    raise KinematicImportError(f"joint {name!r} is below lower limit {lower}")
                if upper is not None and value > float(upper) + EPSILON:
                    raise KinematicImportError(f"joint {name!r} is above upper limit {upper}")
            resolved[name] = value
            return value

        for name in self.joints:
            resolve(name)
        return resolved

    def world_frames(self, supplied: dict[str, float]) -> tuple[dict[str, Matrix], dict[str, float]]:
        pose = self.resolve_pose(supplied)
        frames: dict[str, Matrix] = {self.root_link: _identity()}

        def descend(parent: str) -> None:
            for joint in sorted(self.children[parent], key=lambda item: item.name):
                root_from_pre = _matmul(frames[parent], joint.origin)
                frames[f"joint/{joint.name}"] = root_from_pre
                value = pose[joint.name]
                if joint.type in {"revolute", "continuous"}:
                    motion = _axis_angle(joint.axis, value)
                elif joint.type == "prismatic":
                    motion = _translation([component * value for component in joint.axis])
                else:
                    motion = _identity()
                frames[joint.child] = _matmul(_matmul(root_from_pre, motion), joint.post_motion)
                descend(joint.child)

        descend(self.root_link)
        for link_name, link in self.links.items():
            for collection in ("visuals", "collisions", "frames"):
                for attachment in link.get(collection, []):
                    frames[attachment["frame"]] = _matmul(frames[link_name], attachment["origin_matrix"])
            inertial = link.get("inertial")
            if inertial is not None:
                frames[inertial["frame"]] = _matmul(frames[link_name], inertial["origin_matrix"])
        return frames, pose

    def frame_semantics(self) -> dict[str, dict[str, str | None]]:
        frames: dict[str, dict[str, str | None]] = {
            self.root_link: {"type": "link", "parent_frame": None, "owner": self.root_link}
        }
        for joint in self.joints.values():
            frames[f"joint/{joint.name}"] = {
                "type": "joint_pre_motion",
                "parent_frame": joint.parent,
                "owner": joint.name,
            }
            frames[joint.child] = {
                "type": "link",
                "parent_frame": f"joint/{joint.name}",
                "owner": joint.child,
            }
        for link_name, link in self.links.items():
            for collection, semantic_type in (
                ("visuals", "visual"),
                ("collisions", "collision"),
                ("frames", "declared_frame"),
            ):
                for attachment in link.get(collection, []):
                    frames[attachment["frame"]] = {
                        "type": semantic_type,
                        "parent_frame": link_name,
                        "owner": link_name,
                    }
            inertial = link.get("inertial")
            if inertial is not None:
                frames[inertial["frame"]] = {
                    "type": "inertial",
                    "parent_frame": link_name,
                    "owner": link_name,
                }
        return frames

    def chain(self, start_link: str, end_link: str) -> dict[str, Any]:
        if start_link not in self.links or end_link not in self.links:
            raise KinematicImportError(f"unknown chain endpoint {start_link!r} -> {end_link!r}")
        if start_link != self.root_link:
            raise KinematicImportError("articulation grammar derivations require a root-to-link chain")
        reversed_joints: list[ImportedJoint] = []
        cursor = end_link
        while cursor != start_link:
            joint = self.child_joint.get(cursor)
            if joint is None:
                raise KinematicImportError(f"no chain from {start_link!r} to {end_link!r}")
            reversed_joints.append(joint)
            cursor = joint.parent
        ordered = list(reversed(reversed_joints))
        steps = [
            {
                "from_link": joint.parent,
                "to_link": joint.child,
                "joint": joint.name,
                "joint_type": joint.type,
                "traversal": "parent_to_child",
                "joint_axis_in_pre_motion_frame": None if joint.type == "fixed" else _clean_vector(joint.axis),
            }
            for joint in ordered
        ]
        return {
            "from_link": start_link,
            "to_link": end_link,
            "links": [start_link, *[joint.child for joint in ordered]],
            "steps": steps,
            "joint_count": len(steps),
            "movable_joint_count": sum(joint.type != "fixed" for joint in ordered),
            "movable_joints": [joint.name for joint in ordered if joint.type != "fixed"],
        }

    def affected_by_joint(self, joint_name: str) -> dict[str, Any]:
        joint = self.joints.get(joint_name)
        if joint is None or joint.type == "fixed":
            raise KinematicImportError(f"joint {joint_name!r} is not a movable joint")
        driver, derivative, _, mimic_chain = self._mimic_affine_from_driver(joint_name)
        affected_links: set[str] = set()

        def descend(link: str) -> None:
            affected_links.add(link)
            for child_joint in self.children[link]:
                descend(child_joint.child)

        physical: list[str] = []
        for name, candidate in self.joints.items():
            if candidate.type == "fixed":
                continue
            candidate_driver, _, _, _ = self._mimic_affine_from_driver(name)
            if candidate_driver == driver:
                physical.append(name)
                descend(candidate.child)
        semantics = self.frame_semantics()
        affected_frames = sorted(
            frame_name
            for frame_name, semantic in semantics.items()
            if (
                semantic["type"] in {"link", "visual", "collision", "inertial", "declared_frame"}
                and semantic["owner"] in affected_links
            )
            or (
                semantic["type"] == "joint_pre_motion"
                and self.joints[str(semantic["owner"])].parent in affected_links
            )
        )
        return {
            "joint": joint_name,
            "joint_type": joint.type,
            "independent_driver_joint": driver,
            "requested_joint_derivative_from_driver": _clean(derivative),
            "requested_joint_mimic_chain": mimic_chain,
            "physical_joints_driven": sorted(physical),
            "pre_motion_frame": f"joint/{joint_name}",
            "pre_motion_frame_is_affected_by_own_motion": False,
            "affected_links": sorted(affected_links),
            "affected_frames": affected_frames,
            "downstream_joints": sorted(name for name, item in self.joints.items() if item.parent in affected_links),
            "meaning": f"frames whose root-relative pose can change with independent driver {driver!r}",
        }

    def independent_driver_contract(self, driver: str) -> dict[str, Any]:
        joint = self.joints.get(driver)
        if joint is None or joint.type == "fixed" or joint.mimic is not None:
            raise KinematicImportError(f"joint {driver!r} is not an independent movable driver")
        lower, upper = -math.inf, math.inf
        constraints: list[dict[str, Any]] = []
        physical: list[str] = []
        for name, candidate in sorted(self.joints.items()):
            if candidate.type == "fixed":
                continue
            source, multiplier, offset, chain = self._mimic_affine_from_driver(name)
            if source != driver:
                continue
            physical.append(name)
            declared_lower = None if candidate.type == "continuous" else candidate.limit["lower"]
            declared_upper = None if candidate.type == "continuous" else candidate.limit["upper"]
            constraints.append({
                "joint": name,
                "joint_type": candidate.type,
                "affine_position_from_driver": {"multiplier": _clean(multiplier), "offset": _clean(offset)},
                "mimic_chain": chain,
                "declared_lower": declared_lower,
                "declared_upper": declared_upper,
            })
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
            raise KinematicImportError(f"driver {driver!r} has no feasible interval")
        return {
            "driver_joint": driver,
            "joint_type": joint.type,
            "unit": "m" if joint.type == "prismatic" else "rad",
            "feasible_domain": {
                "minimum": None if not math.isfinite(lower) else _clean(lower),
                "maximum": None if not math.isfinite(upper) else _clean(upper),
                "minimum_unbounded": not math.isfinite(lower),
                "maximum_unbounded": not math.isfinite(upper),
                "constraints": constraints,
            },
            "physical_joints_driven": physical,
            "structural_causality": self.affected_by_joint(driver),
            "epistemic_scope": (
                f"declared {self.source_format} position limits normalized into the common kinematic law; "
                "no collision, dynamics, controller, hardware, or physical feasibility is implied"
            ),
        }


def _empty_link(name: str) -> dict[str, Any]:
    return {"name": name, "visuals": [], "collisions": [], "frames": [], "inertial": None}


def _attachment(frame: str, matrix: Matrix) -> dict[str, Any]:
    return {
        "frame": frame,
        "origin_matrix": matrix,
        "origin_xyz_m": _clean_vector(matrix[index][3] for index in range(3)),
        "origin_rpy_rad": None,
    }


def _sdf_pose(element: ET.Element | None, label: str) -> Matrix:
    values = _numbers(_text(element, label, required=False), 6, label, [0.0] * 6)
    return _pose_matrix(values[:3], values[3:])


def load_sdf_model(path: Path) -> KinematicTreeModel:
    root, raw, semantic = _source_xml(path, "sdf")
    models = root.findall("model")
    if len(models) != 1 or root.find("world") is not None:
        raise KinematicImportError("SDF articulation import requires exactly one direct <model> and no <world>")
    model = models[0]
    if model.findall("model") or model.findall("include"):
        raise KinematicImportError("nested or included SDF models must be flattened before articulation import")
    name = model.get("name")
    if not name:
        raise KinematicImportError("SDF model requires a name")
    link_elements = model.findall("link")
    joint_elements = model.findall("joint")
    if not link_elements:
        raise KinematicImportError("SDF model contains no links")
    links: dict[str, dict[str, Any]] = {}
    for element in link_elements:
        link_name = element.get("name")
        if not link_name or link_name in links:
            raise KinematicImportError(f"SDF link name is missing or duplicated: {link_name!r}")
        links[link_name] = _empty_link(link_name)

    joint_meta: dict[str, dict[str, Any]] = {}
    all_names = set(links)
    for element in joint_elements:
        joint_name, joint_type = element.get("name"), element.get("type")
        if not joint_name or joint_name in all_names:
            raise KinematicImportError(f"SDF joint name is missing or conflicts: {joint_name!r}")
        if joint_type not in SUPPORTED_JOINTS:
            raise KinematicImportError(f"SDF joint {joint_name!r} has unsupported type {joint_type!r}")
        parent = _text(element.find("parent"), f"SDF joint {joint_name} parent")
        child = _text(element.find("child"), f"SDF joint {joint_name} child")
        if parent not in links or child not in links:
            raise KinematicImportError(f"SDF joint {joint_name!r} references missing links {parent!r} -> {child!r}")
        if element.find("axis/mimic") is not None:
            raise KinematicImportError("SDF axis mimic is not supported by the common importer; normalize it explicitly")
        all_names.add(joint_name)
        joint_meta[joint_name] = {
            "element": element,
            "type": joint_type,
            "parent": parent,
            "child": child,
        }

    frame_elements = model.findall("frame")
    frame_meta: dict[str, dict[str, Any]] = {}
    for element in frame_elements:
        frame_name = element.get("name")
        if not frame_name or frame_name in all_names:
            raise KinematicImportError(f"SDF frame name is missing or conflicts: {frame_name!r}")
        attached_to = element.get("attached_to") or "__model__"
        all_names.add(frame_name)
        frame_meta[frame_name] = {"element": element, "attached_to": attached_to}

    pose_records: dict[str, tuple[str, Matrix]] = {"__model__": ("", _identity())}
    for element in link_elements:
        link_name = str(element.get("name"))
        pose = element.find("pose")
        pose_records[link_name] = ((pose.get("relative_to") if pose is not None else None) or "__model__", _sdf_pose(pose, f"link {link_name} pose"))
    for joint_name, record in joint_meta.items():
        pose = record["element"].find("pose")
        pose_records[joint_name] = ((pose.get("relative_to") if pose is not None else None) or record["child"], _sdf_pose(pose, f"joint {joint_name} pose"))
    for frame_name, record in frame_meta.items():
        pose = record["element"].find("pose")
        pose_records[frame_name] = ((pose.get("relative_to") if pose is not None else None) or record["attached_to"], _sdf_pose(pose, f"frame {frame_name} pose"))

    resolved: dict[str, Matrix] = {"__model__": _identity()}
    active: set[str] = set()

    def model_from(frame: str) -> Matrix:
        if frame in resolved:
            return resolved[frame]
        if frame not in pose_records:
            raise KinematicImportError(f"SDF pose references unknown frame {frame!r}")
        if frame in active:
            raise KinematicImportError(f"SDF relative_to cycle detected at {frame!r}")
        active.add(frame)
        relative_to, local = pose_records[frame]
        value = _matmul(model_from(relative_to), local)
        active.remove(frame)
        resolved[frame] = value
        return value

    for frame in pose_records:
        model_from(frame)

    joints: dict[str, ImportedJoint] = {}
    for joint_name, record in joint_meta.items():
        element, parent, child = record["element"], record["parent"], record["child"]
        parent_from_joint = _matmul(_inverse_rigid(model_from(parent)), model_from(joint_name))
        joint_from_child = _matmul(_inverse_rigid(model_from(joint_name)), model_from(child))
        axis_element = element.find("axis")
        if record["type"] == "fixed":
            axis = [1.0, 0.0, 0.0]
        else:
            if axis_element is None:
                raise KinematicImportError(f"SDF joint {joint_name!r} requires <axis>")
            xyz_element = axis_element.find("xyz")
            raw_axis = _numbers(_text(xyz_element, f"SDF joint {joint_name} axis", required=False), 3, f"SDF joint {joint_name} axis", [0.0, 0.0, 1.0])
            expressed_in = (xyz_element.get("expressed_in") if xyz_element is not None else None) or joint_name
            joint_from_expression = _matmul(_inverse_rigid(model_from(joint_name)), model_from(expressed_in))
            axis = _normalized(_rotate(joint_from_expression, raw_axis), f"SDF joint {joint_name} axis")
            if axis_element.find("use_parent_model_frame") is not None:
                raise KinematicImportError("legacy SDF use_parent_model_frame must be normalized to xyz@expressed_in")
        limit_element = None if axis_element is None else axis_element.find("limit")

        def limit_value(tag: str) -> float | None:
            raw_value = _text(None if limit_element is None else limit_element.find(tag), f"SDF joint {joint_name} {tag}", required=False)
            if raw_value is None:
                return None
            try:
                value = float(raw_value)
            except ValueError as error:
                raise KinematicImportError(f"SDF joint {joint_name} {tag} must be numeric") from error
            if not math.isfinite(value):
                return None if tag in {"lower", "upper"} else value
            return value

        joints[joint_name] = ImportedJoint(
            joint_name,
            record["type"],
            parent,
            child,
            _clean_vector(parent_from_joint[index][3] for index in range(3)),
            [0.0, 0.0, 0.0],
            parent_from_joint,
            joint_from_child,
            axis,
            {key: limit_value(key) for key in ("lower", "upper", "effort", "velocity")},
        )

    roots = sorted(set(links) - {joint.child for joint in joints.values()})
    if len(roots) != 1:
        raise KinematicImportError(f"SDF joint graph must have one root link; found {roots}")
    root_link = roots[0]

    def attached_link(frame_name: str, seen: set[str] | None = None) -> str:
        seen = set() if seen is None else seen
        if frame_name in seen:
            raise KinematicImportError(f"SDF attached_to cycle at {frame_name!r}")
        seen.add(frame_name)
        if frame_name == "__model__":
            return root_link
        if frame_name in links:
            return frame_name
        if frame_name in joint_meta:
            return str(joint_meta[frame_name]["child"])
        if frame_name in frame_meta:
            return attached_link(str(frame_meta[frame_name]["attached_to"]), seen)
        raise KinematicImportError(f"SDF frame attaches to unknown frame {frame_name!r}")

    for frame_name, record in frame_meta.items():
        owner = attached_link(str(record["attached_to"]))
        owner_from_frame = _matmul(_inverse_rigid(model_from(owner)), model_from(frame_name))
        links[owner]["frames"].append(_attachment(f"declared/{frame_name}", owner_from_frame))

    for link_element in link_elements:
        link_name = str(link_element.get("name"))
        for collection, tag in (("visuals", "visual"), ("collisions", "collision")):
            for index, element in enumerate(link_element.findall(tag)):
                pose = element.find("pose")
                relative_to = (pose.get("relative_to") if pose is not None else None) or link_name
                model_from_attachment = _matmul(model_from(relative_to), _sdf_pose(pose, f"SDF {tag} pose"))
                local = _matmul(_inverse_rigid(model_from(link_name)), model_from_attachment)
                links[link_name][collection].append(_attachment(f"{tag}/{link_name}/{index}", local))
        inertial = link_element.find("inertial")
        if inertial is not None:
            pose = inertial.find("pose")
            relative_to = (pose.get("relative_to") if pose is not None else None) or link_name
            model_from_attachment = _matmul(model_from(relative_to), _sdf_pose(pose, "SDF inertial pose"))
            local = _matmul(_inverse_rigid(model_from(link_name)), model_from_attachment)
            links[link_name]["inertial"] = _attachment(f"inertial/{link_name}", local)

    return KinematicTreeModel(
        path,
        "sdf",
        name,
        raw,
        semantic,
        links,
        joints,
        {
            "schema_version": "robot-spatial-kinematic-import.v1",
            "source_format": "sdf",
            "supported_subset": "one flat model; fixed/revolute/continuous/prismatic tree; resolved relative_to and axis expressed_in; rigid frames",
            "model_pose_policy": "excluded from robot-local law",
            "joint_pose_default": "child_link_frame",
            "unsupported_rejected": ["world", "nested_model", "include", "axis_mimic", "legacy_use_parent_model_frame", "non_tree_or_multidof_joint"],
        },
    )


def _mjcf_frame(element: ET.Element, label: str) -> Matrix:
    position = _numbers(element.get("pos"), 3, f"{label} pos", [0.0, 0.0, 0.0])
    orientation_attributes = [name for name in ("quat", "axisangle", "xyaxes", "zaxis", "euler") if element.get(name) is not None]
    if len(orientation_attributes) > 1:
        raise KinematicImportError(f"{label} specifies multiple orientation attributes")
    if not orientation_attributes:
        rotation = _identity()
    elif orientation_attributes[0] == "quat":
        rotation = _quaternion_wxyz_matrix(_numbers(element.get("quat"), 4, f"{label} quat"), f"{label} quat")
    else:
        raise KinematicImportError(
            f"{label} uses {orientation_attributes[0]!r}; save canonical MJCF with local quaternion frames before import"
        )
    for index in range(3):
        rotation[index][3] = position[index]
    return rotation


def load_mjcf_model(path: Path) -> KinematicTreeModel:
    root, raw, semantic = _source_xml(path, "mujoco")
    if root.findall("include") or root.find("default") is not None:
        raise KinematicImportError("MJCF includes/default classes must be compiled and saved as canonical MJCF before import")
    if root.find("equality") is not None:
        raise KinematicImportError("MJCF equality constraints are outside the tree grammar and cannot be dropped")
    if root.findall(".//attach") or root.findall(".//frame") or root.findall(".//replicate"):
        raise KinematicImportError("MJCF attach/frame/replicate meta-elements must be compiled and saved before import")
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("coordinate", "local") != "local":
        raise KinematicImportError("MJCF compiler coordinate must be local")
    angle_unit = "degree" if compiler is None else compiler.get("angle", "degree")
    if angle_unit not in {"degree", "radian"}:
        raise KinematicImportError(f"unsupported MJCF compiler angle {angle_unit!r}")
    autolimits = True if compiler is None else compiler.get("autolimits", "true") == "true"
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise KinematicImportError("MJCF requires <worldbody>")
    top_bodies = worldbody.findall("body")
    if len(top_bodies) != 1:
        raise KinematicImportError("MJCF articulation import requires exactly one top-level body")
    if top_bodies[0].find("joint") is not None or top_bodies[0].find("freejoint") is not None:
        raise KinematicImportError("top-level MJCF body must be welded to world; its world pose is excluded from the robot-local law")
    links: dict[str, dict[str, Any]] = {}
    body_elements: dict[str, ET.Element] = {}
    parent_by_body: dict[str, str | None] = {}
    local_by_body: dict[str, Matrix] = {}

    def collect(body: ET.Element, parent: str | None) -> None:
        name = body.get("name")
        if not name or name in links:
            raise KinematicImportError(f"MJCF body name is missing or duplicated: {name!r}")
        if body.find("freejoint") is not None or body.findall("composite") or body.findall("flexcomp"):
            raise KinematicImportError(f"MJCF body {name!r} contains unsupported free/composite/flex semantics")
        links[name] = _empty_link(name)
        body_elements[name] = body
        parent_by_body[name] = parent
        local_by_body[name] = _mjcf_frame(body, f"MJCF body {name}")
        for child in body.findall("body"):
            collect(child, name)

    collect(top_bodies[0], None)
    joints: dict[str, ImportedJoint] = {}
    for child, parent in parent_by_body.items():
        if parent is None:
            continue
        body = body_elements[child]
        joint_elements = body.findall("joint")
        if len(joint_elements) > 1:
            raise KinematicImportError(
                f"MJCF body {child!r} has multiple joints; split compound DOFs into named serial bodies before import"
            )
        parent_from_child = local_by_body[child]
        if not joint_elements:
            joint_name, joint_type = f"fixed__{parent}__{child}", "fixed"
            axis, anchor = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            lower = upper = None
        else:
            joint_element = joint_elements[0]
            joint_name = joint_element.get("name")
            if not joint_name or joint_name in joints:
                raise KinematicImportError(f"MJCF joint name is missing or duplicated: {joint_name!r}")
            raw_type = joint_element.get("type", "hinge")
            if raw_type not in {"hinge", "slide"}:
                raise KinematicImportError(f"MJCF joint {joint_name!r} has unsupported type {raw_type!r}")
            joint_type = "prismatic" if raw_type == "slide" else "revolute"
            axis = _normalized(_numbers(joint_element.get("axis"), 3, f"MJCF joint {joint_name} axis", [0.0, 0.0, 1.0]), f"MJCF joint {joint_name} axis")
            anchor = _numbers(joint_element.get("pos"), 3, f"MJCF joint {joint_name} pos", [0.0, 0.0, 0.0])
            range_raw = joint_element.get("range")
            limited_raw = joint_element.get("limited", "auto")
            limited = limited_raw == "true" or (limited_raw == "auto" and autolimits and range_raw is not None)
            if range_raw is not None and not limited:
                raise KinematicImportError(f"MJCF joint {joint_name!r} declares range while limits are disabled")
            if limited and range_raw is None:
                raise KinematicImportError(f"MJCF joint {joint_name!r} enables limits without range")
            if limited:
                lower, upper = _numbers(range_raw, 2, f"MJCF joint {joint_name} range")
                if raw_type == "hinge" and angle_unit == "degree":
                    lower, upper = math.radians(lower), math.radians(upper)
                if lower > upper:
                    raise KinematicImportError(f"MJCF joint {joint_name!r} has descending range")
            else:
                lower = upper = None
                if raw_type == "hinge":
                    joint_type = "continuous"
        child_from_joint = _translation(anchor)
        parent_from_joint = _matmul(parent_from_child, child_from_joint)
        joint_from_child = _inverse_rigid(child_from_joint)
        joints[joint_name] = ImportedJoint(
            joint_name,
            joint_type,
            parent,
            child,
            _clean_vector(parent_from_joint[index][3] for index in range(3)),
            [0.0, 0.0, 0.0],
            parent_from_joint,
            joint_from_child,
            axis,
            {"lower": lower, "upper": upper, "effort": None, "velocity": None},
        )

    for body_name, body in body_elements.items():
        for index, geom in enumerate(body.findall("geom")):
            if geom.get("fromto") is not None:
                raise KinematicImportError("MJCF geom fromto must be compiled to an explicit local frame before import")
            links[body_name]["frames"].append(_attachment(f"declared/geom/{body_name}/{index}", _mjcf_frame(geom, f"MJCF geom {body_name}/{index}")))
        for site in body.findall("site"):
            site_name = site.get("name")
            if site_name:
                if site.get("fromto") is not None:
                    raise KinematicImportError("MJCF site fromto must be compiled to an explicit local frame before import")
                links[body_name]["frames"].append(_attachment(f"declared/site/{site_name}", _mjcf_frame(site, f"MJCF site {site_name}")))
        inertial = body.find("inertial")
        if inertial is not None:
            links[body_name]["inertial"] = _attachment(f"inertial/{body_name}", _mjcf_frame(inertial, f"MJCF inertial {body_name}"))

    return KinematicTreeModel(
        path,
        "mjcf",
        root.get("model") or "unnamed_mjcf",
        raw,
        semantic,
        links,
        joints,
        {
            "schema_version": "robot-spatial-kinematic-import.v1",
            "source_format": "mjcf",
            "supported_subset": "compiled canonical local-coordinate MJCF; one welded top body; one hinge/slide joint per descendant body; welded bodies; quaternion frames",
            "top_level_body_pose_policy": "excluded from robot-local law",
            "joint_anchor_semantics": "joint pos and axis in child body frame; normalized to pre x motion x post",
            "unsupported_rejected": ["include_or_defaults", "equality_constraint", "free_or_ball_joint", "multiple_joints_per_body", "meta_elements", "non_quaternion_orientation"],
        },
    )


def load_imported_model(path: Path, source_format: str) -> KinematicTreeModel:
    if source_format == "sdf":
        return load_sdf_model(path)
    if source_format == "mjcf":
        return load_mjcf_model(path)
    raise KinematicImportError(f"source format {source_format!r} is not handled by the representation-neutral importer")


def source_binding(model: Any, source_format: str) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "source_format": source_format,
        "source_sha256": model.sha256,
        "source_semantic_sha256": model.semantic_sha256,
        "robot_name": model.name,
        "root_frame": model.root_link,
        "import_contract": getattr(model, "import_contract", {
            "schema_version": "robot-spatial-kinematic-import.v1",
            "source_format": "urdf",
            "supported_subset": "validated fixed/revolute/continuous/prismatic URDF tree with affine mimic",
            "joint_anchor_semantics": "child link coincides with the post-motion joint frame",
        }),
    }
    if source_format == "urdf":
        binding["urdf_sha256"] = model.sha256
        binding["urdf_semantic_sha256"] = model.semantic_sha256
    return binding
