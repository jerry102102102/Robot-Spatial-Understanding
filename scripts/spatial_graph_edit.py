#!/usr/bin/env python3
"""Compile typed robot graph changes into a validated URDF and topology delta."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from robot_spatial import RobotModel, SpatialError


CHANGE_SET_SCHEMA = "robot-spatial-graph-change-set.v1"
REPORT_SCHEMA = "robot-spatial-graph-edit-report.v1"
SUPPORTED_OPERATIONS = {
    "add_leaf_link",
    "remove_leaf_link",
    "add_subtree",
    "remove_subtree",
    "reparent_subtree",
}
MOVABLE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


class GraphEditError(ValueError):
    """An invalid, unsafe, or inapplicable graph change set."""


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise GraphEditError(f"{label}.{field} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GraphEditError(f"{label} must be a finite number")
    return float(value)


def _vector3(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise GraphEditError(f"{label} must be an array of length 3")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise GraphEditError(f"{label} must be a non-empty array of non-empty strings")
    if len(set(value)) != len(value):
        raise GraphEditError(f"{label} contains duplicate names")
    return sorted(value)


def _clean(value: float) -> str:
    if abs(value) < 1e-15:
        return "0"
    return repr(float(value))


def _vector_text(values: list[float]) -> str:
    return " ".join(_clean(value) for value in values)


def _link_xml(value: Any, name: str, label: str) -> str:
    if value is None:
        return f'<link name="{name}" />'
    if not isinstance(value, str) or not value:
        raise GraphEditError(f"{label}.element_xml must be a non-empty string when provided")
    try:
        element = ET.fromstring(value)
    except ET.ParseError as error:
        raise GraphEditError(f"{label}.element_xml is invalid XML: {error}") from error
    if element.tag != "link" or element.get("name") != name:
        raise GraphEditError(f"{label}.element_xml must be one <link> with the declared name")
    if element.findall("joint"):
        raise GraphEditError(f"{label}.element_xml must not contain joint declarations")
    return ET.tostring(element, encoding="unicode", short_empty_elements=True)


def _joint_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphEditError(f"{label} must be an object")
    origin = value.get("origin")
    if not isinstance(origin, dict):
        raise GraphEditError(f"{label}.origin must be an object")
    joint_type = _string(value, "joint_type", label)
    if joint_type not in {"fixed", *MOVABLE_JOINT_TYPES}:
        raise GraphEditError(f"{label}.joint_type is not supported")
    record: dict[str, Any] = {
        "name": _string(value, "name", label),
        "joint_type": joint_type,
        "parent_link": _string(value, "parent_link", label),
        "child_link": _string(value, "child_link", label),
        "origin": {
            "xyz_m": _vector3(origin.get("xyz_m"), f"{label}.origin.xyz_m"),
            "rpy_rad": _vector3(origin.get("rpy_rad"), f"{label}.origin.rpy_rad"),
        },
    }
    if joint_type == "fixed":
        if value.get("axis_xyz") is not None or value.get("limit") is not None:
            raise GraphEditError(f"{label} fixed joints must not declare axis_xyz or limit")
        return record
    record["axis_xyz"] = _vector3(value.get("axis_xyz"), f"{label}.axis_xyz")
    limit = value.get("limit")
    if not isinstance(limit, dict):
        raise GraphEditError(f"{label}.limit must be an object")
    canonical_limit = {
        "effort": _finite(limit.get("effort"), f"{label}.limit.effort"),
        "velocity": _finite(limit.get("velocity"), f"{label}.limit.velocity"),
    }
    if canonical_limit["effort"] < 0.0 or canonical_limit["velocity"] < 0.0:
        raise GraphEditError(f"{label}.limit effort and velocity must be non-negative")
    if joint_type in {"revolute", "prismatic"}:
        canonical_limit["lower"] = _finite(limit.get("lower"), f"{label}.limit.lower")
        canonical_limit["upper"] = _finite(limit.get("upper"), f"{label}.limit.upper")
        if canonical_limit["lower"] > canonical_limit["upper"]:
            raise GraphEditError(f"{label}.limit lower must not exceed upper")
    record["limit"] = canonical_limit
    return record


def _validate_new_subtree_graph(
    root_link: str,
    expected_parent_link: str,
    links: list[dict[str, Any]],
    joints: list[dict[str, Any]],
    label: str,
) -> None:
    link_names = {link["name"] for link in links}
    if root_link not in link_names:
        raise GraphEditError(f"{label}.root_link must be one of the new links")
    if len(link_names) != len(links):
        raise GraphEditError(f"{label}.links contains duplicate names")
    joint_names = {joint["name"] for joint in joints}
    if len(joint_names) != len(joints):
        raise GraphEditError(f"{label}.joints contains duplicate names")
    if len(joints) != len(links):
        raise GraphEditError(f"{label} must contain exactly one incoming joint per new link")
    child_to_parent: dict[str, str] = {}
    attachment_joints: list[dict[str, Any]] = []
    for joint in joints:
        child = joint["child_link"]
        parent = joint["parent_link"]
        if child not in link_names:
            raise GraphEditError(f"{label} joint {joint['name']!r} child must be a new link")
        if child in child_to_parent:
            raise GraphEditError(f"{label} new link {child!r} has multiple incoming joints")
        child_to_parent[child] = parent
        if parent == expected_parent_link:
            attachment_joints.append(joint)
        elif parent not in link_names:
            raise GraphEditError(f"{label} joint {joint['name']!r} parent is outside the declared subtree")
    if set(child_to_parent) != link_names:
        raise GraphEditError(f"{label} every new link must have exactly one incoming joint")
    if len(attachment_joints) != 1 or attachment_joints[0]["child_link"] != root_link:
        raise GraphEditError(f"{label} must have exactly one attachment joint from expected_parent_link to root_link")
    for link_name in link_names:
        seen: set[str] = set()
        cursor = link_name
        while cursor in link_names:
            if cursor in seen:
                raise GraphEditError(f"{label} contains a cycle among new links")
            seen.add(cursor)
            cursor = child_to_parent[cursor]
        if cursor != expected_parent_link:
            raise GraphEditError(f"{label} new links must form one tree rooted at expected_parent_link")


def read_change_set(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise GraphEditError(f"cannot read graph change set {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != CHANGE_SET_SCHEMA:
        raise GraphEditError(f"graph change set must be a {CHANGE_SET_SCHEMA} object")
    change_set_id = _string(data, "change_set_id", "change_set")
    robot = _string(data, "robot", "change_set")
    baseline_sha256 = _string(data, "baseline_urdf_sha256", "change_set")
    operations = data.get("operations")
    if not isinstance(operations, list) or not operations:
        raise GraphEditError("change_set.operations must be a non-empty array")
    canonical_operations: list[dict[str, Any]] = []
    operation_ids: set[str] = set()
    for index, operation in enumerate(operations):
        label = f"change_set.operations[{index}]"
        if not isinstance(operation, dict):
            raise GraphEditError(f"{label} must be an object")
        operation_id = _string(operation, "operation_id", label)
        if operation_id in operation_ids:
            raise GraphEditError(f"duplicate operation_id {operation_id!r}")
        operation_ids.add(operation_id)
        operation_type = _string(operation, "type", label)
        if operation_type not in SUPPORTED_OPERATIONS:
            raise GraphEditError(f"{label}.type must be one of {sorted(SUPPORTED_OPERATIONS)}")
        if operation_type == "add_leaf_link":
            origin = operation.get("origin")
            if not isinstance(origin, dict):
                raise GraphEditError(f"{label}.origin must be an object")
            canonical = {
                "operation_id": operation_id,
                "type": operation_type,
                "parent_link": _string(operation, "parent_link", label),
                "new_link": _string(operation, "new_link", label),
                "new_joint": _string(operation, "new_joint", label),
                "joint_type": operation.get("joint_type", "fixed"),
                "origin": {
                    "xyz_m": _vector3(origin.get("xyz_m"), f"{label}.origin.xyz_m"),
                    "rpy_rad": _vector3(origin.get("rpy_rad"), f"{label}.origin.rpy_rad"),
                },
            }
            if canonical["joint_type"] != "fixed":
                raise GraphEditError(f"{label}.joint_type must currently be fixed")
        elif operation_type == "remove_leaf_link":
            canonical = {
                "operation_id": operation_id,
                "type": operation_type,
                "link": _string(operation, "link", label),
                "expected_parent_link": _string(operation, "expected_parent_link", label),
                "expected_parent_joint": _string(operation, "expected_parent_joint", label),
            }
        elif operation_type == "reparent_subtree":
            origin = operation.get("new_origin")
            if not isinstance(origin, dict):
                raise GraphEditError(f"{label}.new_origin must be an object")
            expected_subtree = operation.get("expected_subtree")
            if not isinstance(expected_subtree, dict):
                raise GraphEditError(f"{label}.expected_subtree must be an object")
            canonical = {
                "operation_id": operation_id,
                "type": operation_type,
                "joint": _string(operation, "joint", label),
                "child_link": _string(operation, "child_link", label),
                "expected_parent_link": _string(operation, "expected_parent_link", label),
                "expected_joint_type": _string(operation, "expected_joint_type", label),
                "new_parent_link": _string(operation, "new_parent_link", label),
                "new_origin": {
                    "xyz_m": _vector3(origin.get("xyz_m"), f"{label}.new_origin.xyz_m"),
                    "rpy_rad": _vector3(origin.get("rpy_rad"), f"{label}.new_origin.rpy_rad"),
                },
                "expected_subtree": {
                    "links": _string_array(expected_subtree.get("links"), f"{label}.expected_subtree.links"),
                    "joints": _string_array(expected_subtree.get("joints"), f"{label}.expected_subtree.joints"),
                },
            }
        elif operation_type == "add_subtree":
            raw_links = operation.get("links")
            raw_joints = operation.get("joints")
            if not isinstance(raw_links, list) or not raw_links:
                raise GraphEditError(f"{label}.links must be a non-empty array")
            if not isinstance(raw_joints, list) or not raw_joints:
                raise GraphEditError(f"{label}.joints must be a non-empty array")
            links: list[dict[str, Any]] = []
            for link_index, raw_link in enumerate(raw_links):
                link_label = f"{label}.links[{link_index}]"
                if not isinstance(raw_link, dict):
                    raise GraphEditError(f"{link_label} must be an object")
                name = _string(raw_link, "name", link_label)
                links.append({"name": name, "element_xml": _link_xml(raw_link.get("element_xml"), name, link_label)})
            joints = [_joint_record(raw_joint, f"{label}.joints[{joint_index}]") for joint_index, raw_joint in enumerate(raw_joints)]
            root_link = _string(operation, "root_link", label)
            expected_parent_link = _string(operation, "expected_parent_link", label)
            _validate_new_subtree_graph(root_link, expected_parent_link, links, joints, label)
            canonical = {
                "operation_id": operation_id,
                "type": operation_type,
                "root_link": root_link,
                "expected_parent_link": expected_parent_link,
                "links": links,
                "joints": joints,
            }
        else:
            expected_subtree = operation.get("expected_subtree")
            if not isinstance(expected_subtree, dict):
                raise GraphEditError(f"{label}.expected_subtree must be an object")
            canonical = {
                "operation_id": operation_id,
                "type": operation_type,
                "root_link": _string(operation, "root_link", label),
                "expected_parent_link": _string(operation, "expected_parent_link", label),
                "expected_parent_joint": _string(operation, "expected_parent_joint", label),
                "expected_subtree": {
                    "links": _string_array(expected_subtree.get("links"), f"{label}.expected_subtree.links"),
                    "joints": _string_array(expected_subtree.get("joints"), f"{label}.expected_subtree.joints"),
                },
            }
        canonical_operations.append(canonical)
    return {
        "schema_version": CHANGE_SET_SCHEMA,
        "change_set_id": change_set_id,
        "robot": robot,
        "baseline_urdf_sha256": baseline_sha256,
        "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "operations": canonical_operations,
    }


def _top_level(root: ET.Element, tag: str, name: str) -> list[ET.Element]:
    return [element for element in root.findall(tag) if element.get("name") == name]


def _insert_after_last(root: ET.Element, element: ET.Element, tag: str) -> None:
    children = list(root)
    indices = [index for index, child in enumerate(children) if child.tag == tag]
    root.insert((indices[-1] + 1) if indices else len(children), element)


def _external_references(root: ET.Element, names: set[str], excluded: set[int]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []

    def visit(element: ET.Element, path: str) -> None:
        if id(element) in excluded:
            return
        for attribute, value in element.attrib.items():
            if value in names:
                references.append({"path": path, "kind": f"attribute:{attribute}", "value": value})
        if (element.text or "").strip() in names:
            references.append({"path": path, "kind": "text", "value": (element.text or "").strip()})
        for index, child in enumerate(list(element)):
            child_name = child.get("name")
            suffix = f"[@name={child_name}]" if child_name else f"[{index}]"
            visit(child, f"{path}/{child.tag}{suffix}")

    visit(root, "/robot")
    return references


def _topology(model: RobotModel) -> dict[str, Any]:
    return {
        "root_link": model.root_link,
        "links": sorted(model.links),
        "joints": sorted(model.joints),
        "edges": [
            {
                "joint": joint.name,
                "type": joint.type,
                "parent_link": joint.parent,
                "child_link": joint.child,
            }
            for joint in sorted(model.joints.values(), key=lambda item: item.name)
        ],
    }


def _subtree(model: RobotModel, root_link: str) -> dict[str, list[str]]:
    """Return a complete rooted subtree, including its incoming attachment joint."""
    links: set[str] = set()
    joints: set[str] = set()
    stack = [root_link]
    if root_link != model.root_link:
        joints.add(model.child_joint[root_link].name)
    while stack:
        link_name = stack.pop()
        if link_name in links:
            continue
        links.add(link_name)
        for joint in model.children[link_name]:
            joints.add(joint.name)
            stack.append(joint.child)
    return {"links": sorted(links), "joints": sorted(joints)}


def _joint_element(record: dict[str, Any]) -> ET.Element:
    joint = ET.Element("joint", {"name": record["name"], "type": record["joint_type"]})
    ET.SubElement(joint, "parent", {"link": record["parent_link"]})
    ET.SubElement(joint, "child", {"link": record["child_link"]})
    ET.SubElement(joint, "origin", {
        "xyz": _vector_text(record["origin"]["xyz_m"]),
        "rpy": _vector_text(record["origin"]["rpy_rad"]),
    })
    if record["joint_type"] in MOVABLE_JOINT_TYPES:
        ET.SubElement(joint, "axis", {"xyz": _vector_text(record["axis_xyz"])})
        limit_attributes = {key: _clean(value) for key, value in record["limit"].items()}
        ET.SubElement(joint, "limit", limit_attributes)
    return joint


def apply_graph_change(source_urdf: Path, change_set_path: Path, output_urdf: Path) -> dict[str, Any]:
    source_model = RobotModel(source_urdf)
    change_set = read_change_set(change_set_path)
    if change_set["robot"] != source_model.name:
        raise GraphEditError(
            f"change set robot {change_set['robot']!r} does not match URDF robot {source_model.name!r}"
        )
    if change_set["baseline_urdf_sha256"] != source_model.sha256:
        raise GraphEditError("change set baseline_urdf_sha256 does not match the source URDF")
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        root = ET.fromstring(source_urdf.read_bytes(), parser=parser)
    except ET.ParseError as error:
        raise GraphEditError(f"cannot parse source XML while preserving comments: {error}") from error
    operation_reports: list[dict[str, Any]] = []

    for operation in change_set["operations"]:
        operation_type = operation["type"]
        if operation_type == "add_leaf_link":
            current_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            parent_link = operation["parent_link"]
            new_link = operation["new_link"]
            new_joint = operation["new_joint"]
            if parent_link not in current_model.links:
                raise GraphEditError(f"operation {operation['operation_id']!r} parent link {parent_link!r} does not exist")
            if new_link in current_model.links or new_joint in current_model.joints:
                raise GraphEditError(f"operation {operation['operation_id']!r} new link or joint name already exists")
            link_element = ET.Element("link", {"name": new_link})
            joint_element = ET.Element("joint", {"name": new_joint, "type": "fixed"})
            ET.SubElement(joint_element, "parent", {"link": parent_link})
            ET.SubElement(joint_element, "child", {"link": new_link})
            ET.SubElement(joint_element, "origin", {
                "xyz": _vector_text(operation["origin"]["xyz_m"]),
                "rpy": _vector_text(operation["origin"]["rpy_rad"]),
            })
            _insert_after_last(root, link_element, "link")
            _insert_after_last(root, joint_element, "joint")
            operation_reports.append({
                "operation_id": operation["operation_id"],
                "type": operation_type,
                "status": "applied",
                "added_link": new_link,
                "added_joint": new_joint,
                "parent_link": parent_link,
                "origin": operation["origin"],
            })
        elif operation_type == "add_subtree":
            current_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            expected_parent = operation["expected_parent_link"]
            link_names = {link["name"] for link in operation["links"]}
            joint_names = {joint["name"] for joint in operation["joints"]}
            if expected_parent not in current_model.links:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} expected parent {expected_parent!r} does not exist"
                )
            if link_names & set(current_model.links) or joint_names & set(current_model.joints):
                raise GraphEditError(f"operation {operation['operation_id']!r} new link or joint name already exists")
            for link in operation["links"]:
                link_element = ET.fromstring(link["element_xml"])
                _insert_after_last(root, link_element, "link")
            for joint in operation["joints"]:
                _insert_after_last(root, _joint_element(joint), "joint")
            validated_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            actual_subtree = _subtree(validated_model, operation["root_link"])
            expected_subtree = {"links": sorted(link_names), "joints": sorted(joint_names)}
            if actual_subtree != expected_subtree:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} compiled subtree mismatch: "
                    f"expected {expected_subtree}, actual {actual_subtree}"
                )
            operation_reports.append({
                "operation_id": operation["operation_id"],
                "type": operation_type,
                "status": "applied",
                "root_link": operation["root_link"],
                "parent_link": expected_parent,
                "added_subtree": expected_subtree,
                "edges": [
                    {
                        "joint": joint["name"],
                        "type": joint["joint_type"],
                        "parent_link": joint["parent_link"],
                        "child_link": joint["child_link"],
                    }
                    for joint in operation["joints"]
                ],
            })
        elif operation_type == "remove_leaf_link":
            current_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            link_name = operation["link"]
            if link_name == current_model.root_link:
                raise GraphEditError(f"operation {operation['operation_id']!r} cannot remove the root link")
            if link_name not in current_model.links:
                raise GraphEditError(f"operation {operation['operation_id']!r} link {link_name!r} does not exist")
            if current_model.children[link_name]:
                child_names = [joint.child for joint in current_model.children[link_name]]
                raise GraphEditError(f"operation {operation['operation_id']!r} link {link_name!r} is not a leaf: {child_names}")
            parent_joint = current_model.child_joint[link_name]
            if parent_joint.parent != operation["expected_parent_link"] or parent_joint.name != operation["expected_parent_joint"]:
                raise GraphEditError(f"operation {operation['operation_id']!r} parent precondition does not match current topology")
            link_elements = _top_level(root, "link", link_name)
            joint_elements = _top_level(root, "joint", parent_joint.name)
            if len(link_elements) != 1 or len(joint_elements) != 1:
                raise GraphEditError(f"operation {operation['operation_id']!r} source elements are ambiguous")
            references = _external_references(
                root,
                {link_name, parent_joint.name},
                {id(link_elements[0]), id(joint_elements[0])},
            )
            if references:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} would leave external references: {references}"
                )
            root.remove(joint_elements[0])
            root.remove(link_elements[0])
            operation_reports.append({
                "operation_id": operation["operation_id"],
                "type": operation_type,
                "status": "applied",
                "removed_link": link_name,
                "removed_joint": parent_joint.name,
                "former_parent_link": parent_joint.parent,
            })
        elif operation_type == "remove_subtree":
            current_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            root_link = operation["root_link"]
            if root_link == current_model.root_link:
                raise GraphEditError(f"operation {operation['operation_id']!r} cannot remove the root subtree")
            if root_link not in current_model.links:
                raise GraphEditError(f"operation {operation['operation_id']!r} root link {root_link!r} does not exist")
            parent_joint = current_model.child_joint[root_link]
            if (
                parent_joint.parent != operation["expected_parent_link"]
                or parent_joint.name != operation["expected_parent_joint"]
            ):
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} attachment precondition does not match current topology"
                )
            actual_subtree = _subtree(current_model, root_link)
            if actual_subtree != operation["expected_subtree"]:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} expected_subtree does not match current topology: "
                    f"expected {operation['expected_subtree']}, actual {actual_subtree}"
                )
            link_elements: list[ET.Element] = []
            joint_elements: list[ET.Element] = []
            for link_name in actual_subtree["links"]:
                matches = _top_level(root, "link", link_name)
                if len(matches) != 1:
                    raise GraphEditError(f"operation {operation['operation_id']!r} link {link_name!r} is ambiguous")
                link_elements.append(matches[0])
            for joint_name in actual_subtree["joints"]:
                matches = _top_level(root, "joint", joint_name)
                if len(matches) != 1:
                    raise GraphEditError(f"operation {operation['operation_id']!r} joint {joint_name!r} is ambiguous")
                joint_elements.append(matches[0])
            excluded = {id(element) for element in [*link_elements, *joint_elements]}
            references = _external_references(
                root,
                set(actual_subtree["links"]) | set(actual_subtree["joints"]),
                excluded,
            )
            if references:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} would leave external references: {references}"
                )
            for element in joint_elements:
                root.remove(element)
            for element in link_elements:
                root.remove(element)
            _model_from_tree(root, output_urdf.parent, source_urdf.name)
            operation_reports.append({
                "operation_id": operation["operation_id"],
                "type": operation_type,
                "status": "applied",
                "root_link": root_link,
                "former_parent_link": parent_joint.parent,
                "former_parent_joint": parent_joint.name,
                "removed_subtree": actual_subtree,
            })
        else:
            current_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            joint_name = operation["joint"]
            child_link = operation["child_link"]
            expected_parent = operation["expected_parent_link"]
            new_parent = operation["new_parent_link"]
            if joint_name not in current_model.joints:
                raise GraphEditError(f"operation {operation['operation_id']!r} joint {joint_name!r} does not exist")
            if child_link not in current_model.links or new_parent not in current_model.links:
                raise GraphEditError(f"operation {operation['operation_id']!r} child or new parent link does not exist")
            joint = current_model.joints[joint_name]
            if (
                joint.child != child_link
                or joint.parent != expected_parent
                or joint.type != operation["expected_joint_type"]
            ):
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} attachment joint precondition does not match current topology"
                )
            actual_subtree = _subtree(current_model, child_link)
            if actual_subtree != operation["expected_subtree"]:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} expected_subtree does not match current topology: "
                    f"expected {operation['expected_subtree']}, actual {actual_subtree}"
                )
            if new_parent == expected_parent:
                raise GraphEditError(f"operation {operation['operation_id']!r} new parent must differ from current parent")
            if new_parent in actual_subtree["links"]:
                raise GraphEditError(
                    f"operation {operation['operation_id']!r} new parent {new_parent!r} is inside the moved subtree"
                )
            joint_elements = _top_level(root, "joint", joint_name)
            if len(joint_elements) != 1:
                raise GraphEditError(f"operation {operation['operation_id']!r} attachment joint element is ambiguous")
            joint_element = joint_elements[0]
            parent_elements = joint_element.findall("parent")
            origin_elements = joint_element.findall("origin")
            if len(parent_elements) != 1 or len(origin_elements) > 1:
                raise GraphEditError(f"operation {operation['operation_id']!r} joint parent/origin declaration is ambiguous")
            parent_elements[0].set("link", new_parent)
            if origin_elements:
                origin_element = origin_elements[0]
            else:
                origin_element = ET.Element("origin")
                children = list(joint_element)
                child_indices = [index for index, element in enumerate(children) if element.tag in {"parent", "child"}]
                joint_element.insert((child_indices[-1] + 1) if child_indices else 0, origin_element)
            origin_element.set("xyz", _vector_text(operation["new_origin"]["xyz_m"]))
            origin_element.set("rpy", _vector_text(operation["new_origin"]["rpy_rad"]))
            validated_model = _model_from_tree(root, output_urdf.parent, source_urdf.name)
            validated_joint = validated_model.joints[joint_name]
            if validated_joint.parent != new_parent or validated_joint.child != child_link:
                raise GraphEditError(f"operation {operation['operation_id']!r} did not produce the requested edge")
            operation_reports.append({
                "operation_id": operation["operation_id"],
                "type": operation_type,
                "status": "applied",
                "joint": joint_name,
                "joint_type": joint.type,
                "child_link": child_link,
                "former_parent_link": expected_parent,
                "new_parent_link": new_parent,
                "new_origin": operation["new_origin"],
                "moved_subtree": actual_subtree,
            })

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    result_model = RobotModel(output_urdf)
    before, after = _topology(source_model), _topology(result_model)
    before_links, after_links = set(before["links"]), set(after["links"])
    before_joints, after_joints = set(before["joints"]), set(after["joints"])
    before_edges = {edge["joint"]: edge for edge in before["edges"]}
    after_edges = {edge["joint"]: edge for edge in after["edges"]}
    changed_edges = [
        {"before": before_edges[joint], "after": after_edges[joint]}
        for joint in sorted(before_joints & after_joints)
        if before_edges[joint] != after_edges[joint]
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "applied_and_validated",
        "meaning": "typed graph operations were compiled into a connected, supported URDF tree",
        "change_set_id": change_set["change_set_id"],
        "source": {"path": str(source_urdf.resolve()), "sha256": source_model.sha256},
        "change_set": change_set["source"],
        "output": {"path": str(output_urdf.resolve()), "sha256": result_model.sha256},
        "robot": result_model.name,
        "operation_count": len(operation_reports),
        "operations": operation_reports,
        "topology_before": before,
        "topology_after": after,
        "topology_delta": {
            "added_links": sorted(after_links - before_links),
            "removed_links": sorted(before_links - after_links),
            "added_joints": sorted(after_joints - before_joints),
            "removed_joints": sorted(before_joints - after_joints),
            "changed_edges": changed_edges,
            "root_link_before": before["root_link"],
            "root_link_after": after["root_link"],
        },
    }


def _model_from_tree(root: ET.Element, directory: Path, source_name: str) -> RobotModel:
    """Validate an intermediate tree using a task-local transient file."""
    directory.mkdir(parents=True, exist_ok=True)
    transient = directory / f".{source_name}.graph-edit-intermediate.urdf"
    ET.ElementTree(root).write(transient, encoding="utf-8", xml_declaration=True)
    try:
        return RobotModel(transient)
    finally:
        transient.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_urdf", type=Path)
    parser.add_argument("change_set", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    report = apply_graph_change(args.source_urdf, args.change_set, args.out)
    serialized = json_dump(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, GraphEditError, SpatialError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
