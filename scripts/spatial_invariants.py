#!/usr/bin/env python3
"""Declarative, deterministic spatial invariant contracts for URDF projects."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "robot-spatial-invariants.v1"
REPORT_VERSION = "robot-spatial-invariant-report.v1"
SUPPORTED_ASSERTIONS = {
    "actuation_declarations",
    "affected_links",
    "chain",
    "declared_mass_properties",
    "frame_distance",
    "frame_pose",
    "frame_semantics",
    "geometry_aabb",
    "joint_axis",
    "observation_collision",
    "observation_readiness",
    "observation_transform",
    "robot_environment_collision",
    "scene_gravity_loads",
    "scene_transform",
    "self_collision_status",
    "static_gravity_loads",
}
ASSERTION_TOLERANCE_FIELDS = {
    "translation_m": "translation_tolerance_m",
    "rotation_deg": "rotation_tolerance_deg",
    "axis_deg": "axis_tolerance_deg",
    "distance_m": "distance_tolerance_m",
    "aabb_m": "aabb_tolerance_m",
    "contact_m": "contact_tolerance_m",
    "mass_kg": "mass_tolerance_kg",
    "center_of_mass_m": "center_of_mass_tolerance_m",
    "inertia_kg_m2": "inertia_tolerance_kg_m2",
    "generalized_effort": "generalized_effort_tolerance",
    "gravity_m_s2": "gravity_tolerance_m_s2",
}


class InvariantError(ValueError):
    """Invalid invariant contract or assertion."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvariantError(f"{label} must be a finite number")
    return float(value)


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise InvariantError(f"{label} must be an array of length {length}")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _matrix3(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise InvariantError(f"{label} must be a 3x3 row-major array")
    return [_vector(row, 3, f"{label}[{index}]") for index, row in enumerate(value)]


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvariantError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise InvariantError(f"{label} must be an array of non-empty strings")
    return list(value)


def read_invariant_contract(
    path: Path,
    model: Any,
    world_scene: Any | None = None,
    observation_resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise InvariantError(f"cannot read invariant contract {path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise InvariantError(f"invariant contract must be a {SCHEMA_VERSION} object")
    robot = _string(data.get("robot"), "robot")
    if robot != model.name:
        raise InvariantError(f"invariant contract robot {robot!r} does not match URDF robot {model.name!r}")
    raw_scene_binding = data.get("world_scene")
    if raw_scene_binding is not None:
        if not isinstance(raw_scene_binding, dict):
            raise InvariantError("world_scene must be an object when provided")
        unknown_binding = sorted(set(raw_scene_binding) - {"scene_id", "snapshot_id", "sha256"})
        if unknown_binding:
            raise InvariantError(f"world_scene has unsupported fields {unknown_binding}")
        if world_scene is None:
            raise InvariantError("invariant contract binds a world_scene but no --scene was provided")
        expected_scene_id = _string(raw_scene_binding.get("scene_id"), "world_scene.scene_id")
        expected_snapshot_id = _string(raw_scene_binding.get("snapshot_id"), "world_scene.snapshot_id")
        if expected_scene_id != world_scene.scene_id:
            raise InvariantError(f"world_scene.scene_id {expected_scene_id!r} does not match {world_scene.scene_id!r}")
        if expected_snapshot_id != world_scene.snapshot["id"]:
            raise InvariantError(
                f"world_scene.snapshot_id {expected_snapshot_id!r} does not match {world_scene.snapshot['id']!r}"
            )
        expected_sha = raw_scene_binding.get("sha256")
        if expected_sha is not None and expected_sha != world_scene.sha256:
            raise InvariantError("world_scene.sha256 does not match the supplied scene artifact")
        scene_binding = {
            "scene_id": expected_scene_id,
            "snapshot_id": expected_snapshot_id,
            "sha256": world_scene.sha256,
        }
    else:
        scene_binding = None
    raw_observation_binding = data.get("observation")
    if raw_observation_binding is not None:
        if not isinstance(raw_observation_binding, dict):
            raise InvariantError("observation must be an object when provided")
        unknown_binding = sorted(set(raw_observation_binding) - {"log_id", "log_sha256", "query_id", "query_sha256"})
        if unknown_binding:
            raise InvariantError(f"observation has unsupported fields {unknown_binding}")
        if observation_resolved is None:
            raise InvariantError("invariant contract binds an observation but no observation log/query was provided")
        report = observation_resolved["report"]
        expected = {
            "log_id": report["observation_log"]["id"],
            "log_sha256": report["observation_log"]["sha256"],
            "query_id": report["query"]["query_id"],
            "query_sha256": observation_resolved["query_sha256"],
        }
        for field, actual in expected.items():
            supplied = raw_observation_binding.get(field)
            if supplied is not None and _string(supplied, f"observation.{field}") != actual:
                raise InvariantError(f"observation.{field} does not match the supplied observation artifacts")
        observation_binding = expected
    else:
        observation_binding = None
    raw_tolerances = data.get("default_tolerances", {})
    if not isinstance(raw_tolerances, dict):
        raise InvariantError("default_tolerances must be an object")
    tolerances = {
        "translation_m": _finite_number(raw_tolerances.get("translation_m", 1e-6), "default_tolerances.translation_m"),
        "rotation_deg": _finite_number(raw_tolerances.get("rotation_deg", 1e-5), "default_tolerances.rotation_deg"),
        "axis_deg": _finite_number(raw_tolerances.get("axis_deg", 1e-5), "default_tolerances.axis_deg"),
        "distance_m": _finite_number(raw_tolerances.get("distance_m", 1e-6), "default_tolerances.distance_m"),
        "aabb_m": _finite_number(raw_tolerances.get("aabb_m", 1e-6), "default_tolerances.aabb_m"),
        "contact_m": _finite_number(raw_tolerances.get("contact_m", 1e-9), "default_tolerances.contact_m"),
        "mass_kg": _finite_number(raw_tolerances.get("mass_kg", 1e-9), "default_tolerances.mass_kg"),
        "center_of_mass_m": _finite_number(raw_tolerances.get("center_of_mass_m", 1e-9), "default_tolerances.center_of_mass_m"),
        "inertia_kg_m2": _finite_number(raw_tolerances.get("inertia_kg_m2", 1e-9), "default_tolerances.inertia_kg_m2"),
        "generalized_effort": _finite_number(raw_tolerances.get("generalized_effort", 1e-9), "default_tolerances.generalized_effort"),
        "gravity_m_s2": _finite_number(raw_tolerances.get("gravity_m_s2", 1e-9), "default_tolerances.gravity_m_s2"),
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise InvariantError("default tolerances must be non-negative")
    raw_poses = data.get("poses", {})
    if not isinstance(raw_poses, dict):
        raise InvariantError("poses must be an object")
    poses: dict[str, dict[str, float]] = {"zero": {}}
    for pose_name, pose_record in raw_poses.items():
        _string(pose_name, "pose name")
        if pose_name == "zero":
            raise InvariantError("pose name 'zero' is reserved for the all-default pose")
        joints = pose_record.get("joints") if isinstance(pose_record, dict) and "joints" in pose_record else pose_record
        if not isinstance(joints, dict) or not all(isinstance(name, str) and name for name in joints):
            raise InvariantError(f"pose {pose_name!r} must be an object of joint values or contain a joints object")
        poses[pose_name] = {name: _finite_number(value, f"poses.{pose_name}.{name}") for name, value in joints.items()}
        model.resolve_pose(poses[pose_name])
    raw_assertions = data.get("assertions")
    if not isinstance(raw_assertions, list) or not raw_assertions:
        raise InvariantError("assertions must be a non-empty array")
    assertions: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, assertion in enumerate(raw_assertions):
        label = f"assertions[{index}]"
        if not isinstance(assertion, dict):
            raise InvariantError(f"{label} must be an object")
        identifier = _string(assertion.get("id"), f"{label}.id")
        if identifier in identifiers:
            raise InvariantError(f"duplicate invariant id {identifier!r}")
        identifiers.add(identifier)
        assertion_type = _string(assertion.get("type"), f"{label}.type")
        if assertion_type not in SUPPORTED_ASSERTIONS:
            raise InvariantError(f"unsupported invariant type {assertion_type!r}; supported: {sorted(SUPPORTED_ASSERTIONS)}")
        canonical = dict(assertion)
        canonical["id"], canonical["type"] = identifier, assertion_type
        pose_dependent = assertion_type in {"declared_mass_properties", "frame_distance", "frame_pose", "geometry_aabb", "joint_axis", "robot_environment_collision", "scene_gravity_loads", "scene_transform", "self_collision_status", "static_gravity_loads"}
        if pose_dependent:
            pose_name = assertion.get("pose", "zero")
            if pose_name not in poses:
                raise InvariantError(f"{label}.pose references undefined pose {pose_name!r}")
            canonical["pose"] = pose_name
        if assertion_type == "frame_pose":
            canonical["from"] = _string(assertion.get("from"), f"{label}.from")
            canonical["to"] = _string(assertion.get("to"), f"{label}.to")
            expected = assertion.get("expected")
            if not isinstance(expected, dict):
                raise InvariantError(f"{label}.expected must be an object")
            canonical_expected = {
                "translation_xyz_m": _vector(expected.get("translation_xyz_m"), 3, f"{label}.expected.translation_xyz_m"),
                "quaternion_xyzw": _vector(expected.get("quaternion_xyzw"), 4, f"{label}.expected.quaternion_xyzw"),
            }
            if sum(value * value for value in canonical_expected["quaternion_xyzw"]) <= 1e-30:
                raise InvariantError(f"{label}.expected.quaternion_xyzw must have non-zero magnitude")
            canonical["expected"] = canonical_expected
        elif assertion_type == "scene_transform":
            if world_scene is None:
                raise InvariantError(f"{label} requires a supplied world scene")
            canonical["from"] = _string(assertion.get("from"), f"{label}.from")
            canonical["to"] = _string(assertion.get("to"), f"{label}.to")
            known_entities = world_scene.typed_frames(model, poses[canonical["pose"]])
            unknown_entities = sorted({canonical["from"], canonical["to"]} - set(known_entities))
            if unknown_entities:
                raise InvariantError(f"{label} references unknown typed scene entities {unknown_entities}")
            expected = assertion.get("expected")
            if not isinstance(expected, dict):
                raise InvariantError(f"{label}.expected must be an object")
            canonical_expected = {
                "translation_xyz_m": _vector(expected.get("translation_xyz_m"), 3, f"{label}.expected.translation_xyz_m"),
                "quaternion_xyzw": _vector(expected.get("quaternion_xyzw"), 4, f"{label}.expected.quaternion_xyzw"),
            }
            if sum(value * value for value in canonical_expected["quaternion_xyzw"]) <= 1e-30:
                raise InvariantError(f"{label}.expected.quaternion_xyzw must have non-zero magnitude")
            canonical["expected"] = canonical_expected
        elif assertion_type == "frame_distance":
            canonical["from"] = _string(assertion.get("from"), f"{label}.from")
            canonical["to"] = _string(assertion.get("to"), f"{label}.to")
            canonical["expected_m"] = _finite_number(assertion.get("expected_m"), f"{label}.expected_m")
            if canonical["expected_m"] < 0.0:
                raise InvariantError(f"{label}.expected_m must be non-negative")
        elif assertion_type == "declared_mass_properties":
            canonical["subtree_root"] = _string(assertion.get("subtree_root", model.root_link), f"{label}.subtree_root")
            canonical["frame"] = _string(assertion.get("frame", model.root_link), f"{label}.frame")
            if canonical["subtree_root"] not in model.links:
                raise InvariantError(f"{label}.subtree_root references unknown link {canonical['subtree_root']!r}")
            if canonical["frame"] not in model.frame_semantics():
                raise InvariantError(f"{label}.frame references unknown frame {canonical['frame']!r}")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "status",
                "declared_mass_kg",
                "center_of_mass_xyz_m",
                "inertia_about_center_of_mass_matrix_3x3_kg_m2",
                "missing_inertial_links",
                "invalid_or_incomplete_inertial_links",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "status" in expected:
                if expected["status"] not in {"computed", "indeterminate", "not_provided"}:
                    raise InvariantError(f"{label}.expected.status must be computed, indeterminate, or not_provided")
                canonical_expected["status"] = expected["status"]
            if "declared_mass_kg" in expected:
                canonical_expected["declared_mass_kg"] = _finite_number(expected["declared_mass_kg"], f"{label}.expected.declared_mass_kg")
                if canonical_expected["declared_mass_kg"] <= 0.0:
                    raise InvariantError(f"{label}.expected.declared_mass_kg must be positive")
            if "center_of_mass_xyz_m" in expected:
                canonical_expected["center_of_mass_xyz_m"] = _vector(expected["center_of_mass_xyz_m"], 3, f"{label}.expected.center_of_mass_xyz_m")
            if "inertia_about_center_of_mass_matrix_3x3_kg_m2" in expected:
                canonical_expected["inertia_about_center_of_mass_matrix_3x3_kg_m2"] = _matrix3(
                    expected["inertia_about_center_of_mass_matrix_3x3_kg_m2"],
                    f"{label}.expected.inertia_about_center_of_mass_matrix_3x3_kg_m2",
                )
            for field in ("missing_inertial_links", "invalid_or_incomplete_inertial_links"):
                if field in expected:
                    canonical_expected[field] = sorted(_string_list(expected[field], f"{label}.expected.{field}"))
            numeric_fields = {
                "declared_mass_kg",
                "center_of_mass_xyz_m",
                "inertia_about_center_of_mass_matrix_3x3_kg_m2",
            }
            if numeric_fields & canonical_expected.keys() and canonical_expected.get("status", "computed") != "computed":
                raise InvariantError(f"{label}.expected numeric mass properties require status computed")
            canonical["expected"] = canonical_expected
        elif assertion_type == "actuation_declarations":
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "ros2_control_systems",
                "legacy_transmissions",
                "joint_command_interfaces",
                "joint_state_interfaces",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            for field in ("ros2_control_systems", "legacy_transmissions"):
                if field in expected:
                    canonical_expected[field] = sorted(_string_list(expected[field], f"{label}.expected.{field}"))
            for field in ("joint_command_interfaces", "joint_state_interfaces"):
                if field not in expected:
                    continue
                if not isinstance(expected[field], dict):
                    raise InvariantError(f"{label}.expected.{field} must be a joint-to-string-list object")
                unknown_joints = sorted(set(expected[field]) - set(model.joints))
                if unknown_joints:
                    raise InvariantError(f"{label}.expected.{field} references unknown joints {unknown_joints}")
                canonical_expected[field] = {
                    joint: sorted(_string_list(interfaces, f"{label}.expected.{field}.{joint}"))
                    for joint, interfaces in sorted(expected[field].items())
                }
            canonical["expected"] = canonical_expected
        elif assertion_type == "joint_axis":
            canonical["joint"] = _string(assertion.get("joint"), f"{label}.joint")
            canonical["frame"] = _string(assertion.get("frame"), f"{label}.frame")
            canonical["expected_unit_vector"] = _vector(assertion.get("expected_unit_vector"), 3, f"{label}.expected_unit_vector")
            if sum(value * value for value in canonical["expected_unit_vector"]) <= 1e-30:
                raise InvariantError(f"{label}.expected_unit_vector must have non-zero magnitude")
        elif assertion_type == "static_gravity_loads":
            canonical["subtree_root"] = _string(assertion.get("subtree_root", model.root_link), f"{label}.subtree_root")
            canonical["gravity_frame"] = _string(assertion.get("gravity_frame", model.root_link), f"{label}.gravity_frame")
            if canonical["subtree_root"] not in model.links:
                raise InvariantError(f"{label}.subtree_root references unknown link {canonical['subtree_root']!r}")
            if canonical["gravity_frame"] not in model.frame_semantics():
                raise InvariantError(f"{label}.gravity_frame references unknown frame {canonical['gravity_frame']!r}")
            canonical["gravity_vector_xyz_m_s2"] = _vector(
                assertion.get("gravity_vector_xyz_m_s2", [0.0, 0.0, -9.80665]),
                3,
                f"{label}.gravity_vector_xyz_m_s2",
            )
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "status",
                "generalized_gravity_forces",
                "ideal_static_holding_efforts",
                "missing_inertial_links",
                "invalid_or_incomplete_inertial_links",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "status" in expected:
                if expected["status"] not in {"computed", "indeterminate", "not_provided"}:
                    raise InvariantError(f"{label}.expected.status must be computed, indeterminate, or not_provided")
                canonical_expected["status"] = expected["status"]
            for field in ("generalized_gravity_forces", "ideal_static_holding_efforts"):
                if field in expected:
                    if not isinstance(expected[field], dict) or not expected[field]:
                        raise InvariantError(f"{label}.expected.{field} must be a non-empty joint-to-number object")
                    unknown_joints = sorted(set(expected[field]) - set(model.joints))
                    if unknown_joints:
                        raise InvariantError(f"{label}.expected.{field} references unknown joints {unknown_joints}")
                    non_drivers = sorted(
                        joint
                        for joint in expected[field]
                        if model.joints[joint].type == "fixed" or model.joints[joint].mimic is not None
                    )
                    if non_drivers:
                        raise InvariantError(
                            f"{label}.expected.{field} must name independent movable driver joints, not {non_drivers}"
                        )
                    canonical_expected[field] = {
                        joint: _finite_number(value, f"{label}.expected.{field}.{joint}")
                        for joint, value in sorted(expected[field].items())
                    }
            for field in ("missing_inertial_links", "invalid_or_incomplete_inertial_links"):
                if field in expected:
                    canonical_expected[field] = sorted(_string_list(expected[field], f"{label}.expected.{field}"))
            if {"generalized_gravity_forces", "ideal_static_holding_efforts"} & canonical_expected.keys() and canonical_expected.get("status", "computed") != "computed":
                raise InvariantError(f"{label}.expected numeric gravity loads require status computed")
            canonical["expected"] = canonical_expected
        elif assertion_type == "scene_gravity_loads":
            if world_scene is None:
                raise InvariantError(f"{label} requires a supplied world scene")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "status",
                "gravity_in_robot_root_xyz_m_s2",
                "generalized_gravity_forces",
                "ideal_static_holding_efforts",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "status" in expected:
                if expected["status"] not in {"computed", "indeterminate", "not_provided"}:
                    raise InvariantError(f"{label}.expected.status must be computed, indeterminate, or not_provided")
                canonical_expected["status"] = expected["status"]
            if "gravity_in_robot_root_xyz_m_s2" in expected:
                canonical_expected["gravity_in_robot_root_xyz_m_s2"] = _vector(
                    expected["gravity_in_robot_root_xyz_m_s2"],
                    3,
                    f"{label}.expected.gravity_in_robot_root_xyz_m_s2",
                )
            for field in ("generalized_gravity_forces", "ideal_static_holding_efforts"):
                if field not in expected:
                    continue
                if not isinstance(expected[field], dict) or not expected[field]:
                    raise InvariantError(f"{label}.expected.{field} must be a non-empty joint-to-number object")
                unknown_joints = sorted(set(expected[field]) - set(model.joints))
                if unknown_joints:
                    raise InvariantError(f"{label}.expected.{field} references unknown joints {unknown_joints}")
                canonical_expected[field] = {
                    joint: _finite_number(value, f"{label}.expected.{field}.{joint}")
                    for joint, value in sorted(expected[field].items())
                }
            canonical["expected"] = canonical_expected
        elif assertion_type == "robot_environment_collision":
            if world_scene is None:
                raise InvariantError(f"{label} requires a supplied world scene")
            contact_tolerance = assertion.get("contact_tolerance_m")
            if contact_tolerance is not None:
                canonical["contact_tolerance_m"] = _finite_number(contact_tolerance, f"{label}.contact_tolerance_m")
                if canonical["contact_tolerance_m"] < 0.0:
                    raise InvariantError(f"{label}.contact_tolerance_m must be non-negative")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "status",
                "minimum_separation_status",
                "minimum_separation_m",
                "collision_pairs",
                "indeterminate_pair_count",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "status" in expected:
                if expected["status"] not in {"collision", "collision_free", "indeterminate", "not_applicable"}:
                    raise InvariantError(f"{label}.expected.status has an unsupported value")
                canonical_expected["status"] = expected["status"]
            if "minimum_separation_status" in expected:
                if expected["minimum_separation_status"] not in {"computed", "indeterminate", "not_applicable"}:
                    raise InvariantError(f"{label}.expected.minimum_separation_status has an unsupported value")
                canonical_expected["minimum_separation_status"] = expected["minimum_separation_status"]
            if "minimum_separation_m" in expected:
                value = _finite_number(expected["minimum_separation_m"], f"{label}.expected.minimum_separation_m")
                if value < 0.0:
                    raise InvariantError(f"{label}.expected.minimum_separation_m must be non-negative")
                canonical_expected["minimum_separation_m"] = value
            if "indeterminate_pair_count" in expected:
                count = expected["indeterminate_pair_count"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise InvariantError(f"{label}.expected.indeterminate_pair_count must be a non-negative integer")
                canonical_expected["indeterminate_pair_count"] = count
            if "collision_pairs" in expected:
                if not isinstance(expected["collision_pairs"], list):
                    raise InvariantError(f"{label}.expected.collision_pairs must be an array")
                pairs: list[dict[str, str]] = []
                for pair_index, pair in enumerate(expected["collision_pairs"]):
                    if not isinstance(pair, dict) or set(pair) != {"robot_geometry", "environment_geometry"}:
                        raise InvariantError(
                            f"{label}.expected.collision_pairs[{pair_index}] must contain robot_geometry and environment_geometry"
                        )
                    pairs.append({
                        "robot_geometry": _string(pair["robot_geometry"], f"{label}.expected.collision_pairs[{pair_index}].robot_geometry"),
                        "environment_geometry": _string(pair["environment_geometry"], f"{label}.expected.collision_pairs[{pair_index}].environment_geometry"),
                    })
                canonical_expected["collision_pairs"] = sorted(
                    pairs,
                    key=lambda pair: (pair["robot_geometry"], pair["environment_geometry"]),
                )
            canonical["expected"] = canonical_expected
        elif assertion_type == "observation_readiness":
            if observation_resolved is None:
                raise InvariantError(f"{label} requires supplied observation artifacts")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {
                "status",
                "all_required_observations_current",
                "nominal_world_state_computable",
                "declaration_fallback_used",
                "declaration_fallback_entities",
            }
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "status" in expected:
                allowed = {"current", "nominal_with_declaration_fallback", "not_current_or_incomplete"}
                if expected["status"] not in allowed:
                    raise InvariantError(f"{label}.expected.status must be one of {sorted(allowed)}")
                canonical_expected["status"] = expected["status"]
            for field in (
                "all_required_observations_current",
                "nominal_world_state_computable",
                "declaration_fallback_used",
            ):
                if field in expected:
                    if not isinstance(expected[field], bool):
                        raise InvariantError(f"{label}.expected.{field} must be boolean")
                    canonical_expected[field] = expected[field]
            if "declaration_fallback_entities" in expected:
                canonical_expected["declaration_fallback_entities"] = sorted(
                    _string_list(expected["declaration_fallback_entities"], f"{label}.expected.declaration_fallback_entities")
                )
            canonical["expected"] = canonical_expected
        elif assertion_type == "observation_transform":
            if observation_resolved is None or world_scene is None:
                raise InvariantError(f"{label} requires supplied observation artifacts and world scene")
            canonical["from"] = _string(assertion.get("from"), f"{label}.from")
            canonical["to"] = _string(assertion.get("to"), f"{label}.to")
            if not observation_resolved["nominal_computable"]:
                raise InvariantError(f"{label} cannot be evaluated because nominal observed world state is unavailable")
            known_entities = world_scene.typed_frames(
                model,
                observation_resolved["joint_pose"],
                world_from_robot_root=observation_resolved["world_from_robot_root"],
                world_from_objects=observation_resolved["world_from_objects"],
            )
            unknown_entities = sorted({canonical["from"], canonical["to"]} - set(known_entities))
            if unknown_entities:
                raise InvariantError(f"{label} references unknown typed observed entities {unknown_entities}")
            expected = assertion.get("expected")
            if not isinstance(expected, dict):
                raise InvariantError(f"{label}.expected must be an object")
            canonical_expected = {
                "translation_xyz_m": _vector(expected.get("translation_xyz_m"), 3, f"{label}.expected.translation_xyz_m"),
                "quaternion_xyzw": _vector(expected.get("quaternion_xyzw"), 4, f"{label}.expected.quaternion_xyzw"),
            }
            if sum(value * value for value in canonical_expected["quaternion_xyzw"]) <= 1e-30:
                raise InvariantError(f"{label}.expected.quaternion_xyzw must have non-zero magnitude")
            canonical["expected"] = canonical_expected
        elif assertion_type == "observation_collision":
            if observation_resolved is None or world_scene is None:
                raise InvariantError(f"{label} requires supplied observation artifacts and world scene")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise InvariantError(f"{label}.expected must be a non-empty object")
            supported = {"nominal_status", "analysis_status", "all_required_observations_current"}
            unknown = sorted(set(expected) - supported)
            if unknown:
                raise InvariantError(f"{label}.expected has unsupported fields {unknown}")
            canonical_expected: dict[str, Any] = {}
            if "nominal_status" in expected:
                if expected["nominal_status"] not in {"collision", "collision_free", "indeterminate", "not_applicable", "not_computed"}:
                    raise InvariantError(f"{label}.expected.nominal_status has an unsupported value")
                canonical_expected["nominal_status"] = expected["nominal_status"]
            if "analysis_status" in expected:
                if expected["analysis_status"] not in {"computed_from_current_observations", "computed_nominally_with_declaration_fallback", "not_computed"}:
                    raise InvariantError(f"{label}.expected.analysis_status has an unsupported value")
                canonical_expected["analysis_status"] = expected["analysis_status"]
            if "all_required_observations_current" in expected:
                if not isinstance(expected["all_required_observations_current"], bool):
                    raise InvariantError(f"{label}.expected.all_required_observations_current must be boolean")
                canonical_expected["all_required_observations_current"] = expected["all_required_observations_current"]
            canonical["contact_tolerance_m"] = _finite_number(
                assertion.get("contact_tolerance_m", tolerances["contact_m"]),
                f"{label}.contact_tolerance_m",
            )
            if canonical["contact_tolerance_m"] < 0.0:
                raise InvariantError(f"{label}.contact_tolerance_m must be non-negative")
            canonical["expected"] = canonical_expected
        elif assertion_type == "chain":
            canonical["from_link"] = _string(assertion.get("from_link"), f"{label}.from_link")
            canonical["to_link"] = _string(assertion.get("to_link"), f"{label}.to_link")
            expected = assertion.get("expected")
            if not isinstance(expected, dict) or not any(key in expected for key in ("links", "joints")):
                raise InvariantError(f"{label}.expected must contain links and/or joints")
            canonical["expected"] = {}
            if "links" in expected:
                canonical["expected"]["links"] = _string_list(expected["links"], f"{label}.expected.links")
            if "joints" in expected:
                canonical["expected"]["joints"] = _string_list(expected["joints"], f"{label}.expected.joints")
        elif assertion_type == "affected_links":
            canonical["joint"] = _string(assertion.get("joint"), f"{label}.joint")
            canonical["expected_links"] = _string_list(assertion.get("expected_links"), f"{label}.expected_links")
        elif assertion_type == "frame_semantics":
            canonical["frame"] = _string(assertion.get("frame"), f"{label}.frame")
            expected = assertion.get("expected")
            if not isinstance(expected, dict):
                raise InvariantError(f"{label}.expected must be an object")
            canonical["expected"] = {}
            if "semantic_type" in expected:
                canonical["expected"]["semantic_type"] = _string(expected["semantic_type"], f"{label}.expected.semantic_type")
            if "parent_frame" in expected:
                parent = expected["parent_frame"]
                if parent is not None and (not isinstance(parent, str) or not parent):
                    raise InvariantError(f"{label}.expected.parent_frame must be null or a non-empty string")
                canonical["expected"]["parent_frame"] = parent
            if not canonical["expected"]:
                raise InvariantError(f"{label}.expected must contain semantic_type and/or parent_frame")
        elif assertion_type == "geometry_aabb":
            canonical["geometry_frame"] = _string(assertion.get("geometry_frame"), f"{label}.geometry_frame")
            expected = assertion.get("expected")
            if not isinstance(expected, dict):
                raise InvariantError(f"{label}.expected must be an object")
            canonical_expected = {
                "min_xyz_m": _vector(expected.get("min_xyz_m"), 3, f"{label}.expected.min_xyz_m"),
                "max_xyz_m": _vector(expected.get("max_xyz_m"), 3, f"{label}.expected.max_xyz_m"),
            }
            if any(canonical_expected["min_xyz_m"][axis] > canonical_expected["max_xyz_m"][axis] for axis in range(3)):
                raise InvariantError(f"{label}.expected AABB min must not exceed max")
            canonical["expected"] = canonical_expected
        elif assertion_type == "self_collision_status":
            expected = assertion.get("expected")
            if expected not in {"collision", "collision_free", "indeterminate"}:
                raise InvariantError(f"{label}.expected must be collision, collision_free, or indeterminate")
            canonical["expected"] = expected
        assertions.append(canonical)
    return {
        "schema_version": SCHEMA_VERSION,
        "robot": robot,
        "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()},
        "default_tolerances": tolerances,
        "poses": poses,
        "world_scene": scene_binding,
        "observation": observation_binding,
        "assertions": assertions,
    }


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else round(value, 12)


def _clean_vector(values: Iterable[float]) -> list[float]:
    return [_clean(value) for value in values]


def _quaternion_xyzw(transform: list[list[float]]) -> list[float]:
    trace = transform[0][0] + transform[1][1] + transform[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [(transform[2][1] - transform[1][2]) / scale, (transform[0][2] - transform[2][0]) / scale, (transform[1][0] - transform[0][1]) / scale, 0.25 * scale]
    elif transform[0][0] > transform[1][1] and transform[0][0] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[0][0] - transform[1][1] - transform[2][2]) * 2.0
        quaternion = [0.25 * scale, (transform[0][1] + transform[1][0]) / scale, (transform[0][2] + transform[2][0]) / scale, (transform[2][1] - transform[1][2]) / scale]
    elif transform[1][1] > transform[2][2]:
        scale = math.sqrt(1.0 + transform[1][1] - transform[0][0] - transform[2][2]) * 2.0
        quaternion = [(transform[0][1] + transform[1][0]) / scale, 0.25 * scale, (transform[1][2] + transform[2][1]) / scale, (transform[0][2] - transform[2][0]) / scale]
    else:
        scale = math.sqrt(1.0 + transform[2][2] - transform[0][0] - transform[1][1]) * 2.0
        quaternion = [(transform[0][2] + transform[2][0]) / scale, (transform[1][2] + transform[2][1]) / scale, 0.25 * scale, (transform[1][0] - transform[0][1]) / scale]
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    quaternion = [value / magnitude for value in quaternion]
    if quaternion[3] < 0.0:
        quaternion = [-value for value in quaternion]
    return _clean_vector(quaternion)


def _quaternion_error_deg(expected: list[float], actual: list[float]) -> float:
    expected_norm = math.sqrt(sum(value * value for value in expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if expected_norm <= 1e-15 or actual_norm <= 1e-15:
        raise InvariantError("quaternion magnitude must be non-zero")
    cosine = abs(sum(expected[index] * actual[index] for index in range(4)) / (expected_norm * actual_norm))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, cosine))))


def _axis_error_deg(expected: list[float], actual: list[float]) -> float:
    expected_norm = math.sqrt(sum(value * value for value in expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    if expected_norm <= 1e-15 or actual_norm <= 1e-15:
        raise InvariantError("axis magnitude must be non-zero")
    cosine = sum(expected[index] * actual[index] for index in range(3)) / (expected_norm * actual_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def verify_invariant_contract(
    model: Any,
    contract: dict[str, Any],
    srdf: dict[str, Any] | None = None,
    package_map_path: Path | None = None,
    world_scene: Any | None = None,
    observation_resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tolerances = contract["default_tolerances"]
    geometry_cache: dict[str, dict[str, dict[str, Any]]] = {}
    collision_cache: dict[tuple[str, float], dict[str, Any]] = {}
    scene_collision_cache: dict[tuple[str, float], dict[str, Any]] = {}
    observation_collision_cache: dict[float, dict[str, Any]] = {}
    frame_semantics = model.frame_semantics()
    results: list[dict[str, Any]] = []

    def tolerance(assertion: dict[str, Any], key: str) -> float:
        field = ASSERTION_TOLERANCE_FIELDS[key]
        value = assertion.get(field, tolerances[key])
        number = _finite_number(value, f"assertion {assertion['id']} {field}")
        if number < 0.0:
            raise InvariantError(f"assertion {assertion['id']} tolerance must be non-negative")
        return number

    for assertion in contract["assertions"]:
        assertion_type = assertion["type"]
        pose_name = assertion.get("pose")
        pose = contract["poses"].get(pose_name, {})
        result: dict[str, Any] = {
            "id": assertion["id"],
            "type": assertion_type,
            "pose": pose_name or "pose_independent",
            "status": "failed",
            "source_urdf_sha256": model.sha256,
            "source_world_scene_sha256": None if world_scene is None else world_scene.sha256,
            "source_observation_log_sha256": None if observation_resolved is None else observation_resolved["report"]["observation_log"]["sha256"],
        }
        try:
            if assertion_type == "frame_pose":
                transform = model.transform(assertion["from"], assertion["to"], pose)
                actual = {
                    "translation_xyz_m": _clean_vector(transform[index][3] for index in range(3)),
                    "quaternion_xyzw": _quaternion_xyzw(transform),
                }
                expected = assertion["expected"]
                translation_error = math.sqrt(sum((actual["translation_xyz_m"][index] - expected["translation_xyz_m"][index]) ** 2 for index in range(3)))
                rotation_error = _quaternion_error_deg(expected["quaternion_xyzw"], actual["quaternion_xyzw"])
                thresholds = {"translation_m": tolerance(assertion, "translation_m"), "rotation_deg": tolerance(assertion, "rotation_deg")}
                passed = translation_error <= thresholds["translation_m"] and rotation_error <= thresholds["rotation_deg"]
                result.update({"expected": expected, "actual": actual, "metrics": {"translation_error_m": translation_error, "rotation_error_deg": rotation_error}, "tolerances": thresholds, "transform": f"{assertion['from']}_from_{assertion['to']}"})
            elif assertion_type == "scene_transform":
                if world_scene is None:
                    raise InvariantError("scene_transform requires a supplied world scene")
                transform = world_scene.transform(assertion["from"], assertion["to"], model, pose)
                actual = {
                    "translation_xyz_m": _clean_vector(transform[index][3] for index in range(3)),
                    "quaternion_xyzw": _quaternion_xyzw(transform),
                }
                expected = assertion["expected"]
                translation_error = math.sqrt(sum(
                    (actual["translation_xyz_m"][index] - expected["translation_xyz_m"][index]) ** 2
                    for index in range(3)
                ))
                rotation_error = _quaternion_error_deg(expected["quaternion_xyzw"], actual["quaternion_xyzw"])
                thresholds = {
                    "translation_m": tolerance(assertion, "translation_m"),
                    "rotation_deg": tolerance(assertion, "rotation_deg"),
                }
                passed = translation_error <= thresholds["translation_m"] and rotation_error <= thresholds["rotation_deg"]
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": {"translation_error_m": translation_error, "rotation_error_deg": rotation_error},
                    "tolerances": thresholds,
                    "transform": f"{assertion['from']}_from_{assertion['to']}",
                    "snapshot_id": world_scene.snapshot["id"],
                })
            elif assertion_type == "actuation_declarations":
                expected = assertion["expected"]
                actual: dict[str, Any] = {}
                if "ros2_control_systems" in expected:
                    actual["ros2_control_systems"] = sorted(model.actuation["ros2_control_systems"])
                if "legacy_transmissions" in expected:
                    actual["legacy_transmissions"] = sorted(model.actuation["legacy_transmissions"])
                for field, interface_key in (
                    ("joint_command_interfaces", "command_interfaces"),
                    ("joint_state_interfaces", "state_interfaces"),
                ):
                    if field not in expected:
                        continue
                    actual[field] = {}
                    for joint in expected[field]:
                        interfaces = {
                            interface
                            for binding in model.actuation["joint_bindings"][joint]["ros2_control"]
                            for interface in binding[interface_key]
                        }
                        actual[field][joint] = sorted(interfaces)
                passed = actual == expected
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "comparison": "exact declarations; interface arrays compared as unordered sets",
                    "epistemic_scope": model.actuation["epistemic_scope"],
                })
            elif assertion_type == "frame_distance":
                transform = model.transform(assertion["from"], assertion["to"], pose)
                actual_distance = math.sqrt(sum(transform[index][3] ** 2 for index in range(3)))
                error = abs(actual_distance - assertion["expected_m"])
                threshold = tolerance(assertion, "distance_m")
                passed = error <= threshold
                result.update({"expected": {"distance_m": assertion["expected_m"]}, "actual": {"distance_m": _clean(actual_distance)}, "metrics": {"absolute_error_m": error}, "tolerances": {"distance_m": threshold}})
            elif assertion_type == "declared_mass_properties":
                properties = model.mass_properties(pose, assertion["frame"], assertion["subtree_root"])
                expected = assertion["expected"]
                actual: dict[str, Any] = {}
                metrics: dict[str, float] = {}
                thresholds: dict[str, float] = {}
                checks: list[bool] = []
                if "status" in expected:
                    actual["status"] = properties["status"]
                    checks.append(actual["status"] == expected["status"])
                if "declared_mass_kg" in expected:
                    actual["declared_mass_kg"] = properties.get("declared_mass_kg")
                    if actual["declared_mass_kg"] is None:
                        checks.append(False)
                    else:
                        error = abs(actual["declared_mass_kg"] - expected["declared_mass_kg"])
                        threshold = tolerance(assertion, "mass_kg")
                        metrics["absolute_mass_error_kg"] = error
                        thresholds["mass_kg"] = threshold
                        checks.append(error <= threshold)
                if "center_of_mass_xyz_m" in expected:
                    actual["center_of_mass_xyz_m"] = properties.get("center_of_mass_in_expressed_frame_m")
                    if actual["center_of_mass_xyz_m"] is None:
                        checks.append(False)
                    else:
                        error = math.sqrt(sum((actual["center_of_mass_xyz_m"][axis] - expected["center_of_mass_xyz_m"][axis]) ** 2 for axis in range(3)))
                        threshold = tolerance(assertion, "center_of_mass_m")
                        metrics["center_of_mass_error_m"] = error
                        thresholds["center_of_mass_m"] = threshold
                        checks.append(error <= threshold)
                inertia_field = "inertia_about_center_of_mass_matrix_3x3_kg_m2"
                if inertia_field in expected:
                    inertia = properties.get("inertia_about_center_of_mass_in_expressed_frame_kg_m2")
                    actual[inertia_field] = None if inertia is None else inertia["matrix_3x3_rowmajor"]
                    if actual[inertia_field] is None:
                        checks.append(False)
                    else:
                        error = max(abs(actual[inertia_field][row][column] - expected[inertia_field][row][column]) for row in range(3) for column in range(3))
                        threshold = tolerance(assertion, "inertia_kg_m2")
                        metrics["maximum_inertia_component_error_kg_m2"] = error
                        thresholds["inertia_kg_m2"] = threshold
                        checks.append(error <= threshold)
                for field in ("missing_inertial_links", "invalid_or_incomplete_inertial_links"):
                    if field in expected:
                        actual[field] = sorted(properties["coverage"][field])
                        checks.append(actual[field] == expected[field])
                passed = all(checks)
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": metrics,
                    "tolerances": thresholds,
                    "selection": properties["selection"],
                    "expressed_in_frame": properties["expressed_in_frame"],
                    "epistemic_scope": properties["epistemic_scope"],
                    "physical_world_completeness": properties["coverage"]["physical_world_completeness"],
                })
            elif assertion_type == "joint_axis":
                actual_axis = model.axis(assertion["joint"], assertion["frame"], pose)
                error = _axis_error_deg(assertion["expected_unit_vector"], actual_axis)
                threshold = tolerance(assertion, "axis_deg")
                passed = error <= threshold
                result.update({"expected": {"unit_vector": assertion["expected_unit_vector"]}, "actual": {"unit_vector": actual_axis}, "metrics": {"angular_error_deg": error}, "tolerances": {"axis_deg": threshold}, "expressed_in_frame": assertion["frame"]})
            elif assertion_type == "static_gravity_loads":
                loads = model.static_gravity_loads(
                    pose,
                    assertion["gravity_vector_xyz_m_s2"],
                    assertion["gravity_frame"],
                    assertion["subtree_root"],
                )
                expected = assertion["expected"]
                actual: dict[str, Any] = {}
                metrics: dict[str, Any] = {}
                thresholds: dict[str, float] = {}
                checks: list[bool] = []
                if "status" in expected:
                    actual["status"] = loads["status"]
                    checks.append(actual["status"] == expected["status"])
                threshold = tolerance(assertion, "generalized_effort")
                for field, load_field in (
                    ("generalized_gravity_forces", "generalized_gravity_force"),
                    ("ideal_static_holding_efforts", "ideal_static_holding_effort"),
                ):
                    if field not in expected:
                        continue
                    actual[field] = {}
                    metrics[f"absolute_errors_{field}"] = {}
                    thresholds["generalized_effort"] = threshold
                    driver_loads = loads.get("independent_driver_loads") or {}
                    for joint, expected_value in expected[field].items():
                        actual_value = driver_loads.get(joint, {}).get(load_field)
                        actual[field][joint] = actual_value
                        if actual_value is None:
                            checks.append(False)
                            metrics[f"absolute_errors_{field}"][joint] = None
                        else:
                            error = abs(actual_value - expected_value)
                            metrics[f"absolute_errors_{field}"][joint] = error
                            checks.append(error <= threshold)
                for field in ("missing_inertial_links", "invalid_or_incomplete_inertial_links"):
                    if field in expected:
                        actual[field] = sorted(loads["coverage"][field])
                        checks.append(actual[field] == expected[field])
                passed = all(checks)
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": metrics,
                    "tolerances": thresholds,
                    "selection": loads["selection"],
                    "gravity": loads["gravity"],
                    "sign_convention": loads["sign_convention"],
                    "epistemic_scope": loads["epistemic_scope"],
                    "physical_world_completeness": loads["coverage"]["physical_world_completeness"],
                })
            elif assertion_type == "scene_gravity_loads":
                if world_scene is None:
                    raise InvariantError("scene_gravity_loads requires a supplied world scene")
                converted = world_scene.gravity_in_robot_root()
                loads = None
                actual_status = "not_provided"
                if converted["status"] == "computed":
                    loads = model.static_gravity_loads(
                        pose,
                        converted["vector_in_robot_root_xyz_m_s2"],
                        model.root_link,
                    )
                    actual_status = loads["status"]
                expected = assertion["expected"]
                actual: dict[str, Any] = {}
                metrics: dict[str, Any] = {}
                thresholds: dict[str, float] = {}
                checks: list[bool] = []
                if "status" in expected:
                    actual["status"] = actual_status
                    checks.append(actual_status == expected["status"])
                if "gravity_in_robot_root_xyz_m_s2" in expected:
                    actual_vector = converted.get("vector_in_robot_root_xyz_m_s2")
                    actual["gravity_in_robot_root_xyz_m_s2"] = actual_vector
                    if actual_vector is None:
                        checks.append(False)
                        metrics["maximum_gravity_component_error_m_s2"] = None
                    else:
                        error = max(abs(actual_vector[index] - expected["gravity_in_robot_root_xyz_m_s2"][index]) for index in range(3))
                        threshold = tolerance(assertion, "gravity_m_s2")
                        metrics["maximum_gravity_component_error_m_s2"] = error
                        thresholds["gravity_m_s2"] = threshold
                        checks.append(error <= threshold)
                effort_threshold = tolerance(assertion, "generalized_effort")
                for field, load_field in (
                    ("generalized_gravity_forces", "generalized_gravity_force"),
                    ("ideal_static_holding_efforts", "ideal_static_holding_effort"),
                ):
                    if field not in expected:
                        continue
                    actual[field] = {}
                    metrics[f"absolute_errors_{field}"] = {}
                    thresholds["generalized_effort"] = effort_threshold
                    driver_loads = {} if loads is None else (loads.get("independent_driver_loads") or {})
                    for joint, expected_value in expected[field].items():
                        actual_value = driver_loads.get(joint, {}).get(load_field)
                        actual[field][joint] = actual_value
                        if actual_value is None:
                            metrics[f"absolute_errors_{field}"][joint] = None
                            checks.append(False)
                        else:
                            error = abs(actual_value - expected_value)
                            metrics[f"absolute_errors_{field}"][joint] = error
                            checks.append(error <= effort_threshold)
                passed = all(checks)
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": metrics,
                    "tolerances": thresholds,
                    "scene_gravity": converted,
                    "snapshot_id": world_scene.snapshot["id"],
                    "epistemic_scope": "declared scene/root/gravity convention and URDF inertials only; not physical-world or hardware proof",
                })
            elif assertion_type == "robot_environment_collision":
                if world_scene is None:
                    raise InvariantError("robot_environment_collision requires a supplied world scene")
                contact_tolerance = assertion.get("contact_tolerance_m", tolerance(assertion, "contact_m"))
                cache_key = (pose_name, contact_tolerance)
                if cache_key not in scene_collision_cache:
                    scene_collision_cache[cache_key] = world_scene.robot_environment_collisions(
                        model,
                        pose,
                        package_map_path,
                        contact_tolerance,
                    )
                collision = scene_collision_cache[cache_key]
                expected = assertion["expected"]
                actual: dict[str, Any] = {}
                metrics: dict[str, Any] = {}
                thresholds: dict[str, float] = {"contact_m": contact_tolerance}
                checks: list[bool] = []
                if "status" in expected:
                    actual["status"] = collision["status"]
                    checks.append(actual["status"] == expected["status"])
                if "minimum_separation_status" in expected:
                    actual["minimum_separation_status"] = collision["minimum_separation"]["status"]
                    checks.append(actual["minimum_separation_status"] == expected["minimum_separation_status"])
                if "minimum_separation_m" in expected:
                    actual_value = collision["minimum_separation"].get("distance_m")
                    actual["minimum_separation_m"] = actual_value
                    if actual_value is None:
                        metrics["absolute_minimum_separation_error_m"] = None
                        checks.append(False)
                    else:
                        error = abs(actual_value - expected["minimum_separation_m"])
                        threshold = tolerance(assertion, "distance_m")
                        metrics["absolute_minimum_separation_error_m"] = error
                        thresholds["distance_m"] = threshold
                        checks.append(error <= threshold)
                if "collision_pairs" in expected:
                    actual["collision_pairs"] = sorted(
                        [
                            {
                                "robot_geometry": record["robot_geometry"],
                                "environment_geometry": record["environment_geometry"],
                            }
                            for record in collision["pair_results"]
                            if record["status"] == "collision"
                        ],
                        key=lambda pair: (pair["robot_geometry"], pair["environment_geometry"]),
                    )
                    checks.append(actual["collision_pairs"] == expected["collision_pairs"])
                if "indeterminate_pair_count" in expected:
                    actual["indeterminate_pair_count"] = collision["coverage"]["indeterminate_pair_count"]
                    checks.append(actual["indeterminate_pair_count"] == expected["indeterminate_pair_count"])
                passed = all(checks)
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": metrics,
                    "tolerances": thresholds,
                    "coverage": collision["coverage"],
                    "snapshot_id": world_scene.snapshot["id"],
                    "epistemic_scope": collision["epistemic_scope"],
                })
            elif assertion_type == "observation_readiness":
                if observation_resolved is None:
                    raise InvariantError("observation_readiness requires supplied observation artifacts")
                report = observation_resolved["report"]
                readiness = report["readiness"]
                actual_source = {
                    "status": report["status"],
                    "all_required_observations_current": readiness["all_required_observations_current"],
                    "nominal_world_state_computable": readiness["nominal_world_state_computable"],
                    "declaration_fallback_used": readiness["declaration_fallback_used"],
                    "declaration_fallback_entities": sorted(readiness["declaration_fallback_entities"]),
                }
                actual = {field: actual_source[field] for field in assertion["expected"]}
                passed = actual == assertion["expected"]
                result.update({
                    "expected": assertion["expected"],
                    "actual": actual,
                    "query": report["query"],
                    "selection_method": report["selection_method"],
                    "epistemic_scope": report["epistemic_scope"],
                })
            elif assertion_type == "observation_transform":
                if observation_resolved is None or world_scene is None:
                    raise InvariantError("observation_transform requires supplied observation artifacts and world scene")
                if not observation_resolved["nominal_computable"]:
                    raise InvariantError("nominal observed world state is unavailable")
                transform = world_scene.transform(
                    assertion["from"],
                    assertion["to"],
                    model,
                    observation_resolved["joint_pose"],
                    world_from_robot_root=observation_resolved["world_from_robot_root"],
                    world_from_objects=observation_resolved["world_from_objects"],
                )
                actual = {
                    "translation_xyz_m": _clean_vector(transform[index][3] for index in range(3)),
                    "quaternion_xyzw": _quaternion_xyzw(transform),
                }
                expected = assertion["expected"]
                translation_error = math.sqrt(sum(
                    (actual["translation_xyz_m"][index] - expected["translation_xyz_m"][index]) ** 2
                    for index in range(3)
                ))
                rotation_error = _quaternion_error_deg(expected["quaternion_xyzw"], actual["quaternion_xyzw"])
                thresholds = {
                    "translation_m": tolerance(assertion, "translation_m"),
                    "rotation_deg": tolerance(assertion, "rotation_deg"),
                }
                passed = translation_error <= thresholds["translation_m"] and rotation_error <= thresholds["rotation_deg"]
                result.update({
                    "expected": expected,
                    "actual": actual,
                    "metrics": {"translation_error_m": translation_error, "rotation_error_deg": rotation_error},
                    "tolerances": thresholds,
                    "transform": f"{assertion['from']}_from_{assertion['to']}",
                    "observation_query_id": observation_resolved["report"]["query"]["query_id"],
                    "all_required_observations_current": observation_resolved["all_required_current"],
                })
            elif assertion_type == "observation_collision":
                if observation_resolved is None or world_scene is None:
                    raise InvariantError("observation_collision requires supplied observation artifacts and world scene")
                contact_tolerance = assertion["contact_tolerance_m"]
                if observation_resolved["nominal_computable"]:
                    if contact_tolerance not in observation_collision_cache:
                        observation_collision_cache[contact_tolerance] = world_scene.robot_environment_collisions(
                            model,
                            observation_resolved["joint_pose"],
                            package_map_path,
                            contact_tolerance,
                            world_from_robot_root=observation_resolved["world_from_robot_root"],
                            world_from_objects=observation_resolved["world_from_objects"],
                        )
                    collision = observation_collision_cache[contact_tolerance]
                    analysis_status = (
                        "computed_from_current_observations"
                        if observation_resolved["all_required_current"]
                        else "computed_nominally_with_declaration_fallback"
                    )
                    nominal_status = collision["status"]
                else:
                    collision = None
                    analysis_status = "not_computed"
                    nominal_status = "not_computed"
                actual_source = {
                    "nominal_status": nominal_status,
                    "analysis_status": analysis_status,
                    "all_required_observations_current": observation_resolved["all_required_current"],
                }
                actual = {field: actual_source[field] for field in assertion["expected"]}
                passed = actual == assertion["expected"]
                result.update({
                    "expected": assertion["expected"],
                    "actual": actual,
                    "tolerances": {"contact_m": contact_tolerance},
                    "coverage": None if collision is None else collision["coverage"],
                    "physical_collision_status": "not_established",
                    "safety_conclusion": "not_established",
                })
            elif assertion_type == "chain":
                actual_chain = model.chain(assertion["from_link"], assertion["to_link"])
                actual = {
                    key: ([step["joint"] for step in actual_chain["steps"]] if key == "joints" else actual_chain[key])
                    for key in assertion["expected"]
                }
                passed = actual == assertion["expected"]
                result.update({"expected": assertion["expected"], "actual": actual})
            elif assertion_type == "affected_links":
                actual_links = sorted(model.affected_by_joint(assertion["joint"])["affected_links"])
                expected_links = sorted(assertion["expected_links"])
                passed = actual_links == expected_links
                result.update({"expected": {"links": expected_links}, "actual": {"links": actual_links}, "comparison": "unordered_exact_set"})
            elif assertion_type == "frame_semantics":
                if assertion["frame"] not in frame_semantics:
                    raise InvariantError(f"unknown frame {assertion['frame']!r}")
                semantic = frame_semantics[assertion["frame"]]
                actual = {
                    key: semantic["type" if key == "semantic_type" else key]
                    for key in assertion["expected"]
                }
                passed = actual == assertion["expected"]
                result.update({"expected": assertion["expected"], "actual": actual})
            elif assertion_type == "geometry_aabb":
                if pose_name not in geometry_cache:
                    geometry_cache[pose_name] = model.geometry_analysis(pose, True, package_map_path)[0]
                analysis = geometry_cache[pose_name]
                if assertion["geometry_frame"] not in analysis:
                    raise InvariantError(f"unknown geometry frame {assertion['geometry_frame']!r}")
                record = analysis[assertion["geometry_frame"]]
                if record["status"] != "measured":
                    raise InvariantError(f"geometry {assertion['geometry_frame']!r} is not measured: {record.get('reason')}")
                actual = {
                    "min_xyz_m": record["bounds_in_root_frame_at_pose"]["min_xyz_m"],
                    "max_xyz_m": record["bounds_in_root_frame_at_pose"]["max_xyz_m"],
                }
                maximum_error = max(abs(actual[key][axis] - assertion["expected"][key][axis]) for key in ("min_xyz_m", "max_xyz_m") for axis in range(3))
                threshold = tolerance(assertion, "aabb_m")
                passed = maximum_error <= threshold
                result.update({"expected": assertion["expected"], "actual": actual, "metrics": {"maximum_coordinate_error_m": maximum_error}, "tolerances": {"aabb_m": threshold}, "expressed_in_frame": model.root_link})
            elif assertion_type == "self_collision_status":
                contact_tolerance = tolerance(assertion, "contact_m")
                cache_key = (pose_name, contact_tolerance)
                if cache_key not in collision_cache:
                    canonical = model.canonical(
                        pose,
                        pose_name,
                        package_map_path=package_map_path,
                        srdf=srdf,
                        workspace_samples=0,
                        surface_collisions=True,
                        contact_tolerance_m=contact_tolerance,
                        inspect_mesh_kinds={"collision"},
                    )
                    collision_cache[cache_key] = canonical["collision_surface"]
                collision = collision_cache[cache_key]
                actual_status = collision["self_collision_status"]
                passed = actual_status == assertion["expected"]
                result.update({"expected": {"self_collision_status": assertion["expected"]}, "actual": {"self_collision_status": actual_status}, "tolerances": {"contact_m": contact_tolerance}, "collision_pair_count": collision["collision_pair_count"], "indeterminate_candidate_count": collision["indeterminate_candidate_count"]})
            else:
                raise InvariantError(f"unsupported assertion type {assertion_type!r}")
            result["status"] = "passed" if passed else "failed"
        except (InvariantError, KeyError, OSError, ValueError) as error:
            result.update({"status": "failed", "error": str(error)})
        results.append(result)
    passed_count = sum(result["status"] == "passed" for result in results)
    return {
        "schema_version": REPORT_VERSION,
        "status": "passed" if passed_count == len(results) else "failed",
        "meaning": "all declared project spatial invariants must pass before an edit is accepted",
        "robot": model.name,
        "source_urdf_sha256": model.sha256,
        "world_scene": (
            None
            if world_scene is None
            else {"scene_id": world_scene.scene_id, "snapshot_id": world_scene.snapshot["id"], "sha256": world_scene.sha256}
        ),
        "observation": contract.get("observation"),
        "contract_source": contract["source"],
        "assertion_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "results": results,
    }
