#!/usr/bin/env python3
"""Grade blind URDF edits against authorized source and spatial-intent changes.

The public task describes what a candidate must change. The private key defines
the exact source declarations, invariant fields, and spatial outcomes that are
authorized. A submission passes only when all three views agree:

1. the URDF source changed only at allow-listed XML attributes;
2. the project invariant contract changed only at allow-listed fields; and
3. the edited robot satisfies both the requested outcome and every invariant.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from robot_spatial import RobotModel, SpatialError, pose_record
from spatial_graph_edit import GraphEditError, apply_graph_change
from spatial_invariants import InvariantError, read_invariant_contract, verify_invariant_contract


TASK_SCHEMA = "robot-spatial-edit-task.v1"
KEY_SCHEMA = "robot-spatial-edit-key.v1"
REPORT_SCHEMA = "robot-spatial-edit-report.v1"
NUMERIC_VECTOR_ATTRIBUTES = {"axis", "rgba", "rpy", "scale", "size", "xyz"}
NUMERIC_SCALAR_ATTRIBUTES = {
    "effort", "friction", "ixx", "ixy", "ixz", "iyy", "iyz", "izz", "length", "lower",
    "mass", "multiplier", "offset", "radius", "upper", "value", "velocity",
}


class EditEvaluationError(ValueError):
    """An invalid edit task, private key, or submission."""


def json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EditEvaluationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(data, dict):
        raise EditEvaluationError(f"{label} {path} must contain a JSON object")
    return data


def _required_string(data: dict[str, Any], field: str, label: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise EditEvaluationError(f"{label}.{field} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise EditEvaluationError(f"{label} must be a finite number")
    return float(value)


def _numeric_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise EditEvaluationError(f"{label} must be an array of length {length}")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _resolve(task_path: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else task_path.parent / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise EditEvaluationError(f"{label} does not exist: {resolved}")
    return resolved


def _normalize_selector(selector: dict[str, Any], label: str) -> dict[str, Any]:
    """Normalize legacy joint selectors and generic named-entity selectors."""
    if all(field in selector for field in ("joint", "child_tag", "attribute")):
        return {
            "entity": {"tag": "joint", "name": _required_string(selector, "joint", label)},
            "path": [{"tag": _required_string(selector, "child_tag", label), "index": 0}],
            "attribute": _required_string(selector, "attribute", label),
        }
    entity = selector.get("entity")
    if not isinstance(entity, dict):
        raise EditEvaluationError(
            f"{label}.entity must identify one named top-level element; legacy joint/child_tag/attribute is also accepted"
        )
    normalized_entity = {
        "tag": _required_string(entity, "tag", f"{label}.entity"),
        "name": _required_string(entity, "name", f"{label}.entity"),
    }
    raw_path = selector.get("path")
    if not isinstance(raw_path, list) or not raw_path:
        raise EditEvaluationError(f"{label}.path must be a non-empty array")
    normalized_path: list[dict[str, Any]] = []
    for index, step in enumerate(raw_path):
        step_label = f"{label}.path[{index}]"
        if not isinstance(step, dict):
            raise EditEvaluationError(f"{step_label} must be an object")
        normalized_step: dict[str, Any] = {"tag": _required_string(step, "tag", step_label)}
        if "name" in step:
            normalized_step["name"] = _required_string(step, "name", step_label)
        step_index = step.get("index", 0)
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise EditEvaluationError(f"{step_label}.index must be a non-negative integer")
        normalized_step["index"] = step_index
        normalized_path.append(normalized_step)
    return {
        "entity": normalized_entity,
        "path": normalized_path,
        "attribute": _required_string(selector, "attribute", label),
    }


def read_task(path: Path) -> dict[str, Any]:
    data = read_json(path, "public edit task")
    if data.get("schema_version") != TASK_SCHEMA:
        raise EditEvaluationError(f"public task must use schema_version {TASK_SCHEMA}")
    task_id = _required_string(data, "task_id", "task")
    robot = _required_string(data, "robot", "task")
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        raise EditEvaluationError("task.inputs must be an object")
    urdf = _resolve(path, _required_string(inputs, "urdf", "task.inputs"), "baseline URDF")
    invariants = _resolve(path, _required_string(inputs, "invariants", "task.inputs"), "baseline invariant contract")
    package_map = None
    if inputs.get("package_map") is not None:
        package_map = _resolve(path, _required_string(inputs, "package_map", "task.inputs"), "package map")
    protected_files: list[dict[str, Any]] = []
    raw_protected_files = inputs.get("protected_files", [])
    if not isinstance(raw_protected_files, list) or not all(isinstance(item, str) and item for item in raw_protected_files):
        raise EditEvaluationError("task.inputs.protected_files must be an array of non-empty paths")
    if len(set(raw_protected_files)) != len(raw_protected_files):
        raise EditEvaluationError("task.inputs.protected_files contains duplicate paths")
    for index, raw_path in enumerate(raw_protected_files):
        protected_files.append({
            "id": raw_path,
            "path": _resolve(path, raw_path, f"protected file {index}"),
        })
    return {
        **data,
        "task_id": task_id,
        "robot": robot,
        "resolved_inputs": {
            "urdf": urdf,
            "invariants": invariants,
            "package_map": package_map,
            "protected_files": protected_files,
        },
    }


def read_key(path: Path, task: dict[str, Any]) -> dict[str, Any]:
    data = read_json(path, "private edit key")
    if data.get("schema_version") != KEY_SCHEMA:
        raise EditEvaluationError(f"private key must use schema_version {KEY_SCHEMA}")
    if data.get("task_id") != task["task_id"]:
        raise EditEvaluationError("private key task_id does not match public task")
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise EditEvaluationError("key.baseline must be an object")
    for field in ("urdf_sha256", "invariants_sha256", "robot", "root_link"):
        _required_string(baseline, field, "key.baseline")
    protected_file_sha256 = baseline.get("protected_file_sha256", {})
    if not isinstance(protected_file_sha256, dict) or not all(
        isinstance(name, str) and name and isinstance(digest, str) and digest
        for name, digest in protected_file_sha256.items()
    ):
        raise EditEvaluationError("key.baseline.protected_file_sha256 must be an object of path IDs to SHA-256 strings")
    task_protected_ids = {record["id"] for record in task["resolved_inputs"]["protected_files"]}
    if set(protected_file_sha256) != task_protected_ids:
        raise EditEvaluationError(
            "key.baseline.protected_file_sha256 keys must exactly match task.inputs.protected_files"
        )

    require_graph_change_set = data.get("require_graph_change_set", False)
    if not isinstance(require_graph_change_set, bool):
        raise EditEvaluationError("key.require_graph_change_set must be boolean")
    source_changes = data.get("authorized_urdf_changes", [])
    if not isinstance(source_changes, list):
        raise EditEvaluationError("key.authorized_urdf_changes must be an array")
    for index, change in enumerate(source_changes):
        label = f"key.authorized_urdf_changes[{index}]"
        if not isinstance(change, dict):
            raise EditEvaluationError(f"{label} must be an object")
        selector = change.get("selector")
        if not isinstance(selector, dict):
            raise EditEvaluationError(f"{label}.selector must be an object")
        _normalize_selector(selector, f"{label}.selector")
        _numeric_vector(change.get("expected_numeric_vector"), 3, f"{label}.expected_numeric_vector")
        tolerance = _finite(change.get("absolute_tolerance", 1e-12), f"{label}.absolute_tolerance")
        if tolerance < 0.0:
            raise EditEvaluationError(f"{label}.absolute_tolerance must be non-negative")

    element_additions = data.get("authorized_urdf_element_additions", [])
    element_removals = data.get("authorized_urdf_element_removals", [])
    element_replacements = data.get("authorized_urdf_element_replacements", [])
    if not all(isinstance(changes, list) for changes in (element_additions, element_removals, element_replacements)):
        raise EditEvaluationError("authorized URDF element additions/removals/replacements must be arrays")
    element_identities: set[tuple[str, str]] = set()
    for field_name, changes, requires_xml in (
        ("authorized_urdf_element_additions", element_additions, True),
        ("authorized_urdf_element_removals", element_removals, False),
        ("authorized_urdf_element_replacements", element_replacements, True),
    ):
        for index, change in enumerate(changes):
            label = f"key.{field_name}[{index}]"
            if not isinstance(change, dict):
                raise EditEvaluationError(f"{label} must be an object")
            tag = _required_string(change, "tag", label)
            name = _required_string(change, "name", label)
            identity = (tag, name)
            if identity in element_identities:
                raise EditEvaluationError(f"duplicate authorized URDF element identity {identity}")
            element_identities.add(identity)
            if requires_xml:
                if not isinstance(change.get("element_xml"), str) or not change["element_xml"]:
                    raise EditEvaluationError(f"{label}.element_xml must be a non-empty string")
                _expected_element(change, label)
    if not source_changes and not element_additions and not element_removals and not element_replacements:
        raise EditEvaluationError("private key must authorize at least one URDF attribute or top-level element change")

    invariant_changes = data.get("authorized_invariant_changes", [])
    if not isinstance(invariant_changes, list):
        raise EditEvaluationError("key.authorized_invariant_changes must be an array")
    for index, change in enumerate(invariant_changes):
        label = f"key.authorized_invariant_changes[{index}]"
        if not isinstance(change, dict):
            raise EditEvaluationError(f"{label} must be an object")
        _required_string(change, "assertion_id", label)
        field_path = change.get("field_path")
        if not isinstance(field_path, list) or not field_path or not all(isinstance(item, str) and item for item in field_path):
            raise EditEvaluationError(f"{label}.field_path must be a non-empty array of strings")
        if "expected_value" not in change:
            raise EditEvaluationError(f"{label}.expected_value is required")

    invariant_additions = data.get("authorized_invariant_additions", [])
    invariant_removals = data.get("authorized_invariant_removals", [])
    if not isinstance(invariant_additions, list) or not isinstance(invariant_removals, list):
        raise EditEvaluationError("authorized invariant additions/removals must be arrays")
    addition_ids: set[str] = set()
    for index, assertion in enumerate(invariant_additions):
        label = f"key.authorized_invariant_additions[{index}]"
        if not isinstance(assertion, dict):
            raise EditEvaluationError(f"{label} must be an assertion object")
        assertion_id = _required_string(assertion, "id", label)
        _required_string(assertion, "type", label)
        if assertion_id in addition_ids:
            raise EditEvaluationError(f"duplicate authorized invariant addition {assertion_id!r}")
        addition_ids.add(assertion_id)
    if not all(isinstance(identifier, str) and identifier for identifier in invariant_removals):
        raise EditEvaluationError("key.authorized_invariant_removals must contain non-empty assertion IDs")
    if len(set(invariant_removals)) != len(invariant_removals):
        raise EditEvaluationError("key.authorized_invariant_removals contains duplicates")
    if addition_ids & set(invariant_removals):
        raise EditEvaluationError("the same invariant ID cannot be both added and removed")
    if not invariant_changes and not invariant_additions and not invariant_removals:
        raise EditEvaluationError("private key must authorize at least one invariant field or membership change")

    outcomes = data.get("required_spatial_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise EditEvaluationError("key.required_spatial_outcomes must be a non-empty array")
    for index, outcome in enumerate(outcomes):
        label = f"key.required_spatial_outcomes[{index}]"
        if not isinstance(outcome, dict) or outcome.get("type") not in {"frame_pose", "geometry_aabb", "joint_axis", "topology"}:
            raise EditEvaluationError(f"{label}.type must be frame_pose, joint_axis, geometry_aabb, or topology")
        outcome_type = outcome["type"]
        if outcome_type == "topology":
            expected = outcome.get("expected")
            if not isinstance(expected, dict):
                raise EditEvaluationError(f"{label}.expected must be an object")
            _required_string(expected, "root_link", f"{label}.expected")
            for field in ("links", "joints"):
                values = expected.get(field)
                if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
                    raise EditEvaluationError(f"{label}.expected.{field} must be an array of non-empty strings")
                if len(set(values)) != len(values):
                    raise EditEvaluationError(f"{label}.expected.{field} contains duplicates")
            edges = expected.get("edges")
            if edges is not None:
                if not isinstance(edges, list):
                    raise EditEvaluationError(f"{label}.expected.edges must be an array when provided")
                edge_joints: set[str] = set()
                for edge_index, edge in enumerate(edges):
                    edge_label = f"{label}.expected.edges[{edge_index}]"
                    if not isinstance(edge, dict):
                        raise EditEvaluationError(f"{edge_label} must be an object")
                    for field in ("joint", "type", "parent_link", "child_link"):
                        _required_string(edge, field, edge_label)
                    if edge["joint"] in edge_joints:
                        raise EditEvaluationError(f"{label}.expected.edges contains duplicate joint {edge['joint']!r}")
                    edge_joints.add(edge["joint"])
                    if edge["parent_link"] not in expected["links"] or edge["child_link"] not in expected["links"]:
                        raise EditEvaluationError(f"{edge_label} references a link outside expected.links")
                if edge_joints != set(expected["joints"]):
                    raise EditEvaluationError(f"{label}.expected.edges must describe every expected joint exactly once")
            tolerance_fields: tuple[tuple[str, float], ...] = ()
        else:
            _required_string(outcome, "pose", label)
            pose = outcome.get("joints")
            if not isinstance(pose, dict) or not all(isinstance(name, str) and name for name in pose):
                raise EditEvaluationError(f"{label}.joints must be an object")
            for name, value in pose.items():
                _finite(value, f"{label}.joints.{name}")
        if outcome_type == "frame_pose":
            for field in ("from", "to"):
                _required_string(outcome, field, label)
            expected = outcome.get("expected")
            if not isinstance(expected, dict):
                raise EditEvaluationError(f"{label}.expected must be an object")
            _numeric_vector(expected.get("translation_xyz_m"), 3, f"{label}.expected.translation_xyz_m")
            _numeric_vector(expected.get("quaternion_xyzw"), 4, f"{label}.expected.quaternion_xyzw")
            tolerance_fields = (("translation_tolerance_m", 1e-9), ("rotation_tolerance_deg", 1e-7))
        elif outcome_type == "joint_axis":
            for field in ("joint", "frame"):
                _required_string(outcome, field, label)
            _numeric_vector(outcome.get("expected_unit_vector"), 3, f"{label}.expected_unit_vector")
            tolerance_fields = (("angular_tolerance_deg", 1e-7),)
        elif outcome_type == "geometry_aabb":
            _required_string(outcome, "geometry_frame", label)
            expected = outcome.get("expected")
            if not isinstance(expected, dict):
                raise EditEvaluationError(f"{label}.expected must be an object")
            minimum = _numeric_vector(expected.get("min_xyz_m"), 3, f"{label}.expected.min_xyz_m")
            maximum = _numeric_vector(expected.get("max_xyz_m"), 3, f"{label}.expected.max_xyz_m")
            if any(minimum[axis] > maximum[axis] for axis in range(3)):
                raise EditEvaluationError(f"{label}.expected min_xyz_m must not exceed max_xyz_m")
            tolerance_fields = (("aabb_tolerance_m", 1e-9),)
        for field, default in tolerance_fields:
            tolerance = _finite(outcome.get(field, default), f"{label}.{field}")
            if tolerance < 0.0:
                raise EditEvaluationError(f"{label}.{field} must be non-negative")
    return {
        **data,
        "require_graph_change_set": require_graph_change_set,
        "authorized_urdf_changes": source_changes,
        "authorized_urdf_element_additions": element_additions,
        "authorized_urdf_element_removals": element_removals,
        "authorized_urdf_element_replacements": element_replacements,
        "authorized_invariant_changes": invariant_changes,
        "authorized_invariant_additions": invariant_additions,
        "authorized_invariant_removals": invariant_removals,
        "required_spatial_outcomes": outcomes,
    }


def _canonical_xml(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_canonical_xml(child) for child in list(element)),
    )


def _semantic_attribute(name: str, value: str) -> Any:
    if name in NUMERIC_VECTOR_ATTRIBUTES:
        try:
            numbers = tuple(0.0 if abs(float(item)) < 1e-15 else float(item) for item in value.split())
        except ValueError:
            return value
        return numbers
    if name in NUMERIC_SCALAR_ATTRIBUTES:
        try:
            number = float(value)
        except ValueError:
            return value
        return 0.0 if abs(number) < 1e-15 else number
    return value


def _canonical_added_element(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted((name, _semantic_attribute(name, value)) for name, value in element.attrib.items())),
        (element.text or "").strip(),
        tuple(_canonical_added_element(child) for child in list(element)),
    )


def _expected_element(change: dict[str, Any], label: str) -> ET.Element:
    try:
        element = ET.fromstring(change["element_xml"])
    except (ET.ParseError, KeyError) as error:
        raise EditEvaluationError(f"{label}.element_xml must contain one valid XML element: {error}") from error
    if element.tag != change["tag"] or element.get("name") != change["name"]:
        raise EditEvaluationError(f"{label}.element_xml tag/name must match its selector")
    return element


def _top_level_elements(root: ET.Element, tag: str, name: str) -> list[ET.Element]:
    return [element for element in root.findall(tag) if element.get("name") == name]


def _selected_attribute(root: ET.Element, selector: dict[str, Any], label: str) -> tuple[ET.Element, str]:
    normalized = _normalize_selector(selector, f"{label} selector")
    entity_spec = normalized["entity"]
    entities = [
        element
        for element in root.findall(entity_spec["tag"])
        if element.get("name") == entity_spec["name"]
    ]
    if len(entities) != 1:
        raise EditEvaluationError(
            f"{label} selector matched {len(entities)} <{entity_spec['tag']}> elements named {entity_spec['name']!r}"
        )
    element = entities[0]
    for depth, step in enumerate(normalized["path"]):
        matches = [child for child in element.findall(step["tag"]) if "name" not in step or child.get("name") == step["name"]]
        if step["index"] >= len(matches):
            raise EditEvaluationError(
                f"{label} selector path step {depth} requested <{step['tag']}> index {step['index']} from {len(matches)} matches"
            )
        element = matches[step["index"]]
    attribute = normalized["attribute"]
    if element.get(attribute) is None:
        raise EditEvaluationError(f"{label} selected attribute {attribute!r} is missing")
    return element, attribute


def _parse_numeric_attribute(raw: str, label: str) -> list[float]:
    try:
        values = [float(item) for item in raw.split()]
    except ValueError as error:
        raise EditEvaluationError(f"{label} must contain numeric values") from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise EditEvaluationError(f"{label} must contain exactly three finite values")
    return values


def _vector_error(expected: list[float], actual: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(expected, actual)))


def _rotation_error_deg(expected: list[float], actual: list[float]) -> float:
    expected_norm = math.sqrt(sum(value * value for value in expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if expected_norm <= 1e-15 or actual_norm <= 1e-15:
        raise EditEvaluationError("quaternion magnitude must be non-zero")
    cosine = abs(sum(left * right for left, right in zip(expected, actual)) / (expected_norm * actual_norm))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, cosine))))


def _axis_error_deg(expected: list[float], actual: list[float]) -> float:
    expected_norm = math.sqrt(sum(value * value for value in expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if expected_norm <= 1e-15 or actual_norm <= 1e-15:
        raise EditEvaluationError("axis magnitude must be non-zero")
    cosine = sum(left * right for left, right in zip(expected, actual)) / (expected_norm * actual_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _assertions_by_id(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {assertion["id"]: assertion for assertion in contract["assertions"]}


def _nested_get(data: Any, field_path: list[str], label: str) -> Any:
    cursor = data
    for field in field_path:
        if not isinstance(cursor, dict) or field not in cursor:
            raise EditEvaluationError(f"{label} has no field path {'.'.join(field_path)}")
        cursor = cursor[field]
    return cursor


def _nested_set(data: Any, field_path: list[str], value: Any, label: str) -> None:
    cursor = data
    for field in field_path[:-1]:
        if not isinstance(cursor, dict) or field not in cursor:
            raise EditEvaluationError(f"{label} has no field path {'.'.join(field_path)}")
        cursor = cursor[field]
    if not isinstance(cursor, dict) or field_path[-1] not in cursor:
        raise EditEvaluationError(f"{label} has no field path {'.'.join(field_path)}")
    cursor[field_path[-1]] = value


def _values_match(expected: Any, actual: Any, tolerance: float) -> tuple[bool, float | None]:
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in [*expected, *actual]):
            maximum_error = max((abs(float(left) - float(right)) for left, right in zip(expected, actual)), default=0.0)
            return maximum_error <= tolerance, maximum_error
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        error = abs(float(expected) - float(actual))
        return error <= tolerance, error
    return expected == actual, None


def _contract_without_source(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in contract.items() if key != "source"}


def grade_edit(
    task_path: Path,
    key_path: Path,
    candidate_urdf: Path,
    candidate_invariants: Path,
    candidate_change_set: Path | None = None,
) -> dict[str, Any]:
    task = read_task(task_path)
    key = read_key(key_path, task)
    inputs = task["resolved_inputs"]
    checks: list[dict[str, Any]] = []

    def add(identifier: str, passed: bool, meaning: str, **evidence: Any) -> None:
        checks.append({"id": identifier, "status": "passed" if passed else "failed", "meaning": meaning, **evidence})

    baseline_urdf_sha = sha256_path(inputs["urdf"])
    baseline_invariant_sha = sha256_path(inputs["invariants"])
    protected_actual = {
        record["id"]: sha256_path(record["path"])
        for record in inputs["protected_files"]
    }
    expected_baseline = key["baseline"]
    integrity_passed = (
        baseline_urdf_sha == expected_baseline["urdf_sha256"]
        and baseline_invariant_sha == expected_baseline["invariants_sha256"]
        and protected_actual == expected_baseline.get("protected_file_sha256", {})
    )
    add(
        "baseline_integrity",
        integrity_passed,
        "the public baseline artifacts must match the evaluator-private digests",
        actual={
            "urdf_sha256": baseline_urdf_sha,
            "invariants_sha256": baseline_invariant_sha,
            "protected_file_sha256": protected_actual,
        },
        expected={
            "urdf_sha256": expected_baseline["urdf_sha256"],
            "invariants_sha256": expected_baseline["invariants_sha256"],
            "protected_file_sha256": expected_baseline.get("protected_file_sha256", {}),
        },
    )
    if not integrity_passed:
        return _report(task, candidate_urdf, candidate_invariants, candidate_change_set, checks)

    baseline_model = RobotModel(inputs["urdf"])
    if baseline_model.name != task["robot"] or baseline_model.name != expected_baseline["robot"] or baseline_model.root_link != expected_baseline["root_link"]:
        raise EditEvaluationError("baseline robot identity does not match task and private key")
    baseline_contract = read_invariant_contract(inputs["invariants"], baseline_model)

    try:
        candidate_model = RobotModel(candidate_urdf)
        candidate_valid = True
        candidate_error = None
    except (OSError, SpatialError) as error:
        candidate_model = None
        candidate_valid = False
        candidate_error = str(error)
    add(
        "candidate_urdf_valid",
        candidate_valid,
        "the candidate URDF must parse as one supported, connected kinematic tree",
        **({"error": candidate_error} if candidate_error else {}),
    )
    if candidate_model is None:
        return _report(task, candidate_urdf, candidate_invariants, candidate_change_set, checks)

    identity_passed = candidate_model.name == expected_baseline["robot"] and candidate_model.root_link == expected_baseline["root_link"]
    add(
        "robot_identity_preserved",
        identity_passed,
        "robot name and kinematic root must remain unchanged",
        expected={"robot": expected_baseline["robot"], "root_link": expected_baseline["root_link"]},
        actual={"robot": candidate_model.name, "root_link": candidate_model.root_link},
    )

    if key.get("require_graph_change_set", False) or candidate_change_set is not None:
        if candidate_change_set is None:
            add(
                "graph_change_set_reproduces_candidate",
                False,
                "a required typed graph change set must compile from the pinned baseline to the submitted URDF",
                error="candidate graph change set was not provided",
            )
        else:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    compiled_urdf = Path(temp_dir) / "compiled.urdf"
                    graph_report = apply_graph_change(inputs["urdf"], candidate_change_set, compiled_urdf)
                    compiled_model = RobotModel(compiled_urdf)
                    reproduced = _canonical_xml(compiled_model.xml_root) == _canonical_xml(candidate_model.xml_root)
                add(
                    "graph_change_set_reproduces_candidate",
                    reproduced,
                    "the typed graph change set must compile from the pinned baseline to the submitted URDF",
                    change_set_sha256=sha256_path(candidate_change_set),
                    topology_delta=graph_report["topology_delta"],
                    comparison="complete_semantic_xml_tree_after_deterministic_compilation",
                )
            except (OSError, GraphEditError, SpatialError) as error:
                add(
                    "graph_change_set_reproduces_candidate",
                    False,
                    "the typed graph change set must compile from the pinned baseline to the submitted URDF",
                    error=str(error),
                )

    baseline_root = copy.deepcopy(baseline_model.xml_root)
    candidate_root = copy.deepcopy(candidate_model.xml_root)
    source_targets: list[dict[str, Any]] = []
    target_attributes_passed = True
    for index, change in enumerate(key["authorized_urdf_changes"]):
        selector = change["selector"]
        baseline_element, baseline_attribute = _selected_attribute(baseline_root, selector, f"baseline change {index}")
        candidate_element, candidate_attribute = _selected_attribute(candidate_root, selector, f"candidate change {index}")
        expected = [float(value) for value in change["expected_numeric_vector"]]
        actual = _parse_numeric_attribute(candidate_element.get(candidate_attribute, ""), f"candidate change {index}")
        error = _vector_error(expected, actual)
        tolerance = float(change.get("absolute_tolerance", 1e-12))
        passed = error <= tolerance
        target_attributes_passed = target_attributes_passed and passed
        source_targets.append({
            "selector": selector,
            "baseline": _parse_numeric_attribute(baseline_element.get(baseline_attribute, ""), f"baseline change {index}"),
            "expected": expected,
            "actual": actual,
            "euclidean_error": error,
            "absolute_tolerance": tolerance,
            "status": "passed" if passed else "failed",
        })
        candidate_element.set(candidate_attribute, baseline_element.get(baseline_attribute, ""))
    add(
        "authorized_urdf_values",
        target_attributes_passed,
        "every authorized XML attribute must contain its requested numeric value",
        changes=source_targets,
    )
    element_results: list[dict[str, Any]] = []
    elements_passed = True
    for index, change in enumerate(key.get("authorized_urdf_element_additions", [])):
        tag, name = change["tag"], change["name"]
        baseline_matches = _top_level_elements(baseline_root, tag, name)
        candidate_matches = _top_level_elements(candidate_root, tag, name)
        expected_element = _expected_element(change, f"authorized addition {index}")
        passed = (
            not baseline_matches
            and len(candidate_matches) == 1
            and _canonical_added_element(candidate_matches[0]) == _canonical_added_element(expected_element)
        )
        elements_passed = elements_passed and passed
        element_results.append({
            "operation": "addition",
            "tag": tag,
            "name": name,
            "baseline_match_count": len(baseline_matches),
            "candidate_match_count": len(candidate_matches),
            "status": "passed" if passed else "failed",
        })
        if len(candidate_matches) == 1:
            candidate_root.remove(candidate_matches[0])
    for change in key.get("authorized_urdf_element_removals", []):
        tag, name = change["tag"], change["name"]
        baseline_matches = _top_level_elements(baseline_root, tag, name)
        candidate_matches = _top_level_elements(candidate_root, tag, name)
        passed = len(baseline_matches) == 1 and not candidate_matches
        elements_passed = elements_passed and passed
        element_results.append({
            "operation": "removal",
            "tag": tag,
            "name": name,
            "baseline_match_count": len(baseline_matches),
            "candidate_match_count": len(candidate_matches),
            "status": "passed" if passed else "failed",
        })
        if len(baseline_matches) == 1:
            baseline_root.remove(baseline_matches[0])
    for index, change in enumerate(key.get("authorized_urdf_element_replacements", [])):
        tag, name = change["tag"], change["name"]
        baseline_matches = _top_level_elements(baseline_root, tag, name)
        candidate_matches = _top_level_elements(candidate_root, tag, name)
        expected_element = _expected_element(change, f"authorized replacement {index}")
        passed = (
            len(baseline_matches) == 1
            and len(candidate_matches) == 1
            and _canonical_added_element(candidate_matches[0]) == _canonical_added_element(expected_element)
        )
        elements_passed = elements_passed and passed
        element_results.append({
            "operation": "replacement",
            "tag": tag,
            "name": name,
            "baseline_match_count": len(baseline_matches),
            "candidate_match_count": len(candidate_matches),
            "status": "passed" if passed else "failed",
        })
        if len(baseline_matches) == 1:
            baseline_root.remove(baseline_matches[0])
        if len(candidate_matches) == 1:
            candidate_root.remove(candidate_matches[0])
    if element_results:
        add(
            "authorized_urdf_elements",
            elements_passed,
            "every authorized top-level link/joint addition, removal, or replacement must match its exact semantic element contract",
            changes=element_results,
        )
    only_authorized_source = _canonical_xml(baseline_root) == _canonical_xml(candidate_root)
    add(
        "urdf_change_allowlist",
        only_authorized_source,
        "after reverting authorized attributes, the complete semantic XML tree must equal the baseline",
        comparison="whitespace_and_attribute_order_insensitive_exact_xml_tree",
    )

    geometry_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for index, outcome in enumerate(key["required_spatial_outcomes"]):
        outcome_type = outcome["type"]
        try:
            if outcome_type == "frame_pose":
                evaluated_transform = candidate_model.transform(outcome["from"], outcome["to"], outcome["joints"])
            elif outcome_type == "joint_axis":
                evaluated_axis = candidate_model.axis(outcome["joint"], outcome["frame"], outcome["joints"])
            elif outcome_type == "geometry_aabb":
                pose_name = outcome["pose"]
                if pose_name not in geometry_cache:
                    geometry_cache[pose_name] = candidate_model.geometry_analysis(
                        outcome["joints"],
                        inspect_meshes=True,
                        package_map_path=inputs["package_map"],
                    )[0]
                geometry_frame = outcome["geometry_frame"]
                if geometry_frame not in geometry_cache[pose_name]:
                    raise EditEvaluationError(f"required outcome references unknown geometry frame {geometry_frame!r}")
                geometry_record = geometry_cache[pose_name][geometry_frame]
                if geometry_record["status"] != "measured":
                    raise EditEvaluationError(
                        f"required outcome geometry {geometry_frame!r} is not measured: {geometry_record.get('reason')}"
                    )
        except (EditEvaluationError, KeyError, OSError, SpatialError, ValueError) as error:
            add(
                f"required_spatial_outcome_{index + 1}",
                False,
                "the candidate model must make the evaluator-specified spatial outcome computable and correct",
                outcome_type=outcome_type,
                error=str(error),
            )
            continue
        if outcome_type == "topology":
            actual_topology = {
                "root_link": candidate_model.root_link,
                "links": sorted(candidate_model.links),
                "joints": sorted(candidate_model.joints),
            }
            expected_topology = {
                "root_link": outcome["expected"]["root_link"],
                "links": sorted(outcome["expected"]["links"]),
                "joints": sorted(outcome["expected"]["joints"]),
            }
            if "edges" in outcome["expected"]:
                actual_topology["edges"] = [
                    {
                        "joint": joint.name,
                        "type": joint.type,
                        "parent_link": joint.parent,
                        "child_link": joint.child,
                    }
                    for joint in sorted(candidate_model.joints.values(), key=lambda item: item.name)
                ]
                expected_topology["edges"] = sorted(
                    outcome["expected"]["edges"],
                    key=lambda edge: edge["joint"],
                )
            passed = actual_topology == expected_topology
            evidence = {
                "meaning": "the edited robot must have the evaluator-specified complete link/joint topology",
                "expected": expected_topology,
                "actual": actual_topology,
                "comparison": "exact_unordered_link_and_joint_sets_with_exact_root",
            }
        elif outcome_type == "frame_pose":
            actual_pose = pose_record(evaluated_transform)
            expected = outcome["expected"]
            translation_error = _vector_error(expected["translation_xyz_m"], actual_pose["translation_xyz_m"])
            rotation_error = _rotation_error_deg(expected["quaternion_xyzw"], actual_pose["quaternion_xyzw"])
            translation_tolerance = float(outcome.get("translation_tolerance_m", 1e-9))
            rotation_tolerance = float(outcome.get("rotation_tolerance_deg", 1e-7))
            passed = translation_error <= translation_tolerance and rotation_error <= rotation_tolerance
            evidence = {
                "meaning": "the edited model must produce the evaluator-specified transform at the evaluator-specified pose",
                "pose": outcome["pose"],
                "transform": f"{outcome['from']}_from_{outcome['to']}",
                "expected": expected,
                "actual": {key: actual_pose[key] for key in ("translation_xyz_m", "quaternion_xyzw")},
                "metrics": {"translation_error_m": translation_error, "rotation_error_deg": rotation_error},
                "tolerances": {"translation_m": translation_tolerance, "rotation_deg": rotation_tolerance},
            }
        elif outcome_type == "joint_axis":
            actual_axis = evaluated_axis
            expected_axis = [float(value) for value in outcome["expected_unit_vector"]]
            angular_error = _axis_error_deg(expected_axis, actual_axis)
            angular_tolerance = float(outcome.get("angular_tolerance_deg", 1e-7))
            passed = angular_error <= angular_tolerance
            evidence = {
                "meaning": "the edited joint axis must have the evaluator-specified signed direction in the requested frame",
                "pose": outcome["pose"],
                "joint": outcome["joint"],
                "expressed_in_frame": outcome["frame"],
                "expected": {"unit_vector": expected_axis},
                "actual": {"unit_vector": actual_axis},
                "metrics": {"angular_error_deg": angular_error},
                "tolerances": {"angular_deg": angular_tolerance},
            }
        else:
            pose_name = outcome["pose"]
            geometry_frame = outcome["geometry_frame"]
            record = geometry_record
            actual_bounds = {
                "min_xyz_m": record["bounds_in_root_frame_at_pose"]["min_xyz_m"],
                "max_xyz_m": record["bounds_in_root_frame_at_pose"]["max_xyz_m"],
            }
            expected_bounds = outcome["expected"]
            maximum_error = max(
                abs(actual_bounds[bound][axis] - expected_bounds[bound][axis])
                for bound in ("min_xyz_m", "max_xyz_m")
                for axis in range(3)
            )
            aabb_tolerance = float(outcome.get("aabb_tolerance_m", 1e-9))
            passed = maximum_error <= aabb_tolerance
            evidence = {
                "meaning": "the edited geometry must have the evaluator-specified root-frame AABB at the requested pose",
                "pose": pose_name,
                "geometry_frame": geometry_frame,
                "expressed_in_frame": candidate_model.root_link,
                "expected": expected_bounds,
                "actual": actual_bounds,
                "metrics": {"maximum_coordinate_error_m": maximum_error},
                "tolerances": {"aabb_m": aabb_tolerance},
            }
        meaning = evidence.pop("meaning")
        add(f"required_spatial_outcome_{index + 1}", passed, meaning, outcome_type=outcome_type, **evidence)

    try:
        candidate_contract = read_invariant_contract(candidate_invariants, candidate_model)
        contract_valid = True
        contract_error = None
    except (OSError, InvariantError, SpatialError) as error:
        candidate_contract = None
        contract_valid = False
        contract_error = str(error)
    add(
        "candidate_invariant_contract_valid",
        contract_valid,
        "the candidate invariant artifact must be a valid contract for the edited robot",
        **({"error": contract_error} if contract_error else {}),
    )
    if candidate_contract is None:
        return _report(task, candidate_urdf, candidate_invariants, candidate_change_set, checks)

    baseline_normalized = _contract_without_source(baseline_contract)
    candidate_normalized = _contract_without_source(candidate_contract)
    baseline_assertions = _assertions_by_id(baseline_normalized)
    candidate_assertions = _assertions_by_id(candidate_normalized)
    membership_results: list[dict[str, Any]] = []
    membership_passed = True
    addition_ids: set[str] = set()
    for expected_assertion in key.get("authorized_invariant_additions", []):
        assertion_id = expected_assertion["id"]
        addition_ids.add(assertion_id)
        baseline_present = assertion_id in baseline_assertions
        candidate_assertion = candidate_assertions.get(assertion_id)
        passed = not baseline_present and candidate_assertion == expected_assertion
        membership_passed = membership_passed and passed
        membership_results.append({
            "operation": "addition",
            "assertion_id": assertion_id,
            "baseline_present": baseline_present,
            "candidate_present": candidate_assertion is not None,
            "status": "passed" if passed else "failed",
        })
    removal_ids = set(key.get("authorized_invariant_removals", []))
    for assertion_id in key.get("authorized_invariant_removals", []):
        baseline_present = assertion_id in baseline_assertions
        candidate_present = assertion_id in candidate_assertions
        passed = baseline_present and not candidate_present
        membership_passed = membership_passed and passed
        membership_results.append({
            "operation": "removal",
            "assertion_id": assertion_id,
            "baseline_present": baseline_present,
            "candidate_present": candidate_present,
            "status": "passed" if passed else "failed",
        })
    if membership_results:
        add(
            "authorized_invariant_membership",
            membership_passed,
            "added and removed invariant assertions must exactly match the evaluator-approved membership delta",
            changes=membership_results,
        )
        baseline_normalized["assertions"] = [
            assertion for assertion in baseline_normalized["assertions"] if assertion["id"] not in removal_ids
        ]
        candidate_normalized["assertions"] = [
            assertion for assertion in candidate_normalized["assertions"] if assertion["id"] not in addition_ids
        ]
        baseline_assertions = _assertions_by_id(baseline_normalized)
        candidate_assertions = _assertions_by_id(candidate_normalized)
    invariant_targets: list[dict[str, Any]] = []
    invariant_values_passed = True
    for index, change in enumerate(key["authorized_invariant_changes"]):
        assertion_id = change["assertion_id"]
        field_path = change["field_path"]
        if assertion_id not in baseline_assertions or assertion_id not in candidate_assertions:
            invariant_values_passed = False
            invariant_targets.append({
                "assertion_id": assertion_id,
                "field_path": field_path,
                "status": "failed",
                "error": "assertion is missing from baseline or candidate contract",
            })
            continue
        baseline_value = _nested_get(baseline_assertions[assertion_id], field_path, f"baseline assertion {assertion_id}")
        actual_value = _nested_get(candidate_assertions[assertion_id], field_path, f"candidate assertion {assertion_id}")
        expected_value = change["expected_value"]
        tolerance = float(change.get("absolute_tolerance", 0.0))
        passed, error = _values_match(expected_value, actual_value, tolerance)
        invariant_values_passed = invariant_values_passed and passed
        invariant_targets.append({
            "assertion_id": assertion_id,
            "field_path": field_path,
            "baseline": baseline_value,
            "expected": expected_value,
            "actual": actual_value,
            "maximum_absolute_error": error,
            "absolute_tolerance": tolerance,
            "status": "passed" if passed else "failed",
        })
        _nested_set(candidate_assertions[assertion_id], field_path, copy.deepcopy(baseline_value), f"candidate assertion {assertion_id}")
    add(
        "authorized_invariant_values",
        invariant_values_passed,
        "every authorized design-intent field must contain its approved value",
        changes=invariant_targets,
    )
    only_authorized_invariants = baseline_normalized == candidate_normalized
    add(
        "invariant_change_allowlist",
        only_authorized_invariants,
        "after reverting authorized fields, the complete canonical invariant contract must equal the baseline",
        comparison="canonical_contract_exact_equality",
    )

    invariant_report = verify_invariant_contract(candidate_model, candidate_contract, package_map_path=inputs["package_map"])
    add(
        "edited_model_satisfies_invariants",
        invariant_report["status"] == "passed",
        "all updated and protected project spatial invariants must pass on the edited model",
        invariant_summary={
            "status": invariant_report["status"],
            "assertion_count": invariant_report["assertion_count"],
            "passed_count": invariant_report["passed_count"],
            "failed_count": invariant_report["failed_count"],
            "failed_ids": [result["id"] for result in invariant_report["results"] if result["status"] != "passed"],
        },
    )
    return _report(task, candidate_urdf, candidate_invariants, candidate_change_set, checks)


def _report(
    task: dict[str, Any],
    candidate_urdf: Path,
    candidate_invariants: Path,
    candidate_change_set: Path | None,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    failed = [check["id"] for check in checks if check["status"] != "passed"]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if not failed else "failed",
        "meaning": "the edit is accepted only when source scope, design-intent scope, and spatial behavior all pass",
        "task_id": task["task_id"],
        "candidate": {
            "urdf_path": str(candidate_urdf.resolve()),
            "urdf_sha256": sha256_path(candidate_urdf) if candidate_urdf.is_file() else None,
            "invariants_path": str(candidate_invariants.resolve()),
            "invariants_sha256": sha256_path(candidate_invariants) if candidate_invariants.is_file() else None,
            "change_set_path": str(candidate_change_set.resolve()) if candidate_change_set is not None else None,
            "change_set_sha256": sha256_path(candidate_change_set) if candidate_change_set is not None and candidate_change_set.is_file() else None,
        },
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="public robot-spatial-edit-task.v1 JSON")
    parser.add_argument("key", type=Path, help="evaluator-private robot-spatial-edit-key.v1 JSON")
    parser.add_argument("--candidate-urdf", type=Path, required=True)
    parser.add_argument("--candidate-invariants", type=Path, required=True)
    parser.add_argument("--candidate-change-set", type=Path, help="optional typed robot-spatial-graph-change-set.v1")
    parser.add_argument("--report", type=Path, help="optional report path")
    return parser


def run(args: argparse.Namespace) -> int:
    report = grade_edit(
        args.task,
        args.key,
        args.candidate_urdf,
        args.candidate_invariants,
        args.candidate_change_set,
    )
    serialized = json_dump(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OSError, EditEvaluationError, SpatialError, InvariantError) as error:
        print(json_dump({"status": "error", "error": str(error)}), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
