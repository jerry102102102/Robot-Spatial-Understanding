#!/usr/bin/env python3
"""Executable, pose-independent articulation grammar for supported URDF trees."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARTICULATION_SCHEMA = "robot-spatial-articulation-grammar.v1"
EVALUATION_SCHEMA = "robot-spatial-articulation-evaluation.v1"
VERIFICATION_SCHEMA = "robot-spatial-articulation-verification.v1"
COMPARISON_SCHEMA = "robot-spatial-articulation-comparison.v1"
LAW_SCHEMA = "robot-spatial-canonical-kinematic-law.v1"
CORRESPONDENCE_SCHEMA = "robot-spatial-articulation-correspondence.v1"
EPSILON = 1e-12


class ArticulationError(ValueError):
    """An invalid articulation grammar, binding, or evaluation request."""


def _clean(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(float(value), 12)


def _clean_vector(values: Iterable[float]) -> list[float]:
    return [_clean(float(value)) for value in values]


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [_clean(sum(left[row][inner] * right[inner][column] for inner in range(4))) for column in range(4)]
        for row in range(4)
    ]


def _translation(vector: list[float]) -> list[list[float]]:
    result = _identity()
    for index in range(3):
        result[index][3] = float(vector[index])
    return result


def _rpy_matrix(rpy: list[float]) -> list[list[float]]:
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


def _origin_matrix(xyz: list[float], rpy: list[float]) -> list[list[float]]:
    result = _rpy_matrix(rpy)
    for index in range(3):
        result[index][3] = float(xyz[index])
    return result


def _axis_angle(axis: list[float], angle: float) -> list[list[float]]:
    norm = math.sqrt(sum(component * component for component in axis))
    if norm <= EPSILON:
        raise ArticulationError("rotation axis must be non-zero")
    x, y, z = [component / norm for component in axis]
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _quaternion_xyzw(transform: list[list[float]]) -> list[float]:
    m00, m11, m22 = transform[0][0], transform[1][1], transform[2][2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (transform[2][1] - transform[1][2]) / scale
        y = (transform[0][2] - transform[2][0]) / scale
        z = (transform[1][0] - transform[0][1]) / scale
        w = 0.25 * scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        x = 0.25 * scale
        y = (transform[0][1] + transform[1][0]) / scale
        z = (transform[0][2] + transform[2][0]) / scale
        w = (transform[2][1] - transform[1][2]) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        x = (transform[0][1] + transform[1][0]) / scale
        y = 0.25 * scale
        z = (transform[1][2] + transform[2][1]) / scale
        w = (transform[0][2] - transform[2][0]) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        x = (transform[0][2] + transform[2][0]) / scale
        y = (transform[1][2] + transform[2][1]) / scale
        z = 0.25 * scale
        w = (transform[1][0] - transform[0][1]) / scale
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return _clean_vector([x, y, z, w])


def _pose_record(transform: list[list[float]]) -> dict[str, Any]:
    return {
        "translation_xyz_m": _clean_vector(transform[index][3] for index in range(3)),
        "quaternion_xyzw": _quaternion_xyzw(transform),
        "matrix_4x4_rowmajor": [[_clean(value) for value in row] for row in transform],
    }


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _matrix_from_record(record: Any, label: str) -> list[list[float]]:
    if not isinstance(record, dict):
        raise ArticulationError(f"{label} must be an object")
    matrix = record.get("matrix_4x4_rowmajor")
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ArticulationError(f"{label}.matrix_4x4_rowmajor must be 4x4")
    try:
        result = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError) as error:
        raise ArticulationError(f"{label}.matrix_4x4_rowmajor must contain numbers") from error
    if any(not math.isfinite(value) for row in result for value in row):
        raise ArticulationError(f"{label}.matrix_4x4_rowmajor must contain finite numbers")
    return result


def _default_for_domain(domain: dict[str, Any]) -> float:
    lower, upper = domain["minimum"], domain["maximum"]
    if lower is not None and 0.0 < float(lower):
        return float(lower)
    if upper is not None and 0.0 > float(upper):
        return float(upper)
    return 0.0


def _joint_local_attachments(model: Any) -> dict[str, list[list[float]]]:
    attachments: dict[str, list[list[float]]] = {}
    for link in model.links.values():
        for collection in ("visuals", "collisions", "frames"):
            for geometry in link.get(collection, []):
                attachments[geometry["frame"]] = geometry.get("origin_matrix") or _origin_matrix(
                    geometry["origin_xyz_m"], geometry["origin_rpy_rad"]
                )
        inertial = link.get("inertial")
        if inertial is not None:
            attachments[inertial["frame"]] = inertial.get("origin_matrix") or _origin_matrix(
                inertial["origin_xyz_m"], inertial["origin_rpy_rad"]
            )
    return attachments


def articulation_law_projection(grammar: dict[str, Any]) -> dict[str, Any]:
    """Return the source-binding-free executable law used for identity and comparison."""
    variables: dict[str, Any] = {}
    for name, variable in sorted(grammar["independent_variables"].items()):
        variables[name] = {
            key: value
            for key, value in variable.items()
            if key not in {"structural_causality", "epistemic_scope"}
        }
    position_rules = {
        name: {key: value for key, value in rule.items() if key != "equation_cnl"}
        for name, rule in sorted(grammar["joint_position_rules"].items())
    }
    operators = {
        name: {
            key: value
            for key, value in operator.items()
            if key not in {"composition_rule", "pre_motion_frame_is_affected_by_own_motion"}
        }
        for name, operator in sorted(grammar["joint_operators"].items())
    }
    derivations = {
        name: {
            key: value
            for key, value in derivation.items()
            if key not in {"composition_tokens", "expression_cnl"}
        }
        for name, derivation in sorted(grammar["frame_derivations"].items())
    }
    return {
        "schema_version": LAW_SCHEMA,
        "coordinate_contract": grammar["coordinate_contract"],
        "independent_variables": variables,
        "joint_position_rules": position_rules,
        "joint_operators": operators,
        "frame_derivations": derivations,
        "evaluation_contract": grammar["evaluation_contract"],
        "coverage": grammar["coverage"],
    }


def build_articulation_grammar(model: Any, source_binding: dict[str, Any]) -> dict[str, Any]:
    """Build a typed, executable grammar from one validated RobotModel."""
    driver_names = sorted(
        name for name, joint in model.joints.items()
        if joint.type != "fixed" and joint.mimic is None
    )
    independent_variables: dict[str, Any] = {}
    for driver in driver_names:
        contract = model.independent_driver_contract(driver)
        domain = contract["feasible_domain"]
        independent_variables[driver] = {
            "variable_id": f"q/{driver}",
            "joint_type": contract["joint_type"],
            "unit": contract["unit"],
            "default_value": _clean(_default_for_domain(domain)),
            "feasible_domain": domain,
            "physical_joints_driven": contract["physical_joints_driven"],
            "structural_causality": model.affected_by_joint(driver),
        }

    joint_position_rules: dict[str, Any] = {}
    operators: dict[str, Any] = {}
    for name, joint in sorted(model.joints.items()):
        if joint.type == "fixed":
            position_rule = {"type": "constant", "value": 0.0, "unit": None}
        else:
            driver, multiplier, offset, chain = model._mimic_affine_from_driver(name)
            position_rule = {
                "type": "independent_variable" if name == driver else "affine_driver_dependency",
                "driver_variable": f"q/{driver}",
                "driver_joint": driver,
                "multiplier": _clean(multiplier),
                "offset": _clean(offset),
                "unit": "m" if joint.type == "prismatic" else "rad",
                "mimic_chain_from_physical_joint_to_driver": chain,
                "equation_cnl": f"q[{name}] = {_clean(multiplier)} * q[{driver}] + {_clean(offset)}",
            }
        joint_position_rules[name] = position_rule
        if joint.type == "fixed":
            motion = {"type": "identity", "position_rule_ref": f"joint_position_rule/{name}"}
        elif joint.type in {"revolute", "continuous"}:
            motion = {
                "type": "rotation_about_axis",
                "axis_xyz_in_pre_motion_frame": _clean_vector(joint.axis),
                "angle_source": f"joint_position_rule/{name}",
                "angle_unit": "rad",
            }
        else:
            motion = {
                "type": "translation_along_axis",
                "axis_xyz_in_pre_motion_frame": _clean_vector(joint.axis),
                "distance_source": f"joint_position_rule/{name}",
                "distance_unit": "m",
            }
        operators[name] = {
            "operator_id": f"joint_operator/{name}",
            "joint": name,
            "joint_type": joint.type,
            "parent_link": joint.parent,
            "child_link": joint.child,
            "constant_parent_from_pre_motion": _pose_record(joint.origin),
            "motion_operator": motion,
            "post_motion_from_child_zero": _pose_record(
                getattr(joint, "post_motion", _identity())
            ),
            "composition_rule": (
                "parent_from_child(q) = parent_from_joint_pre_motion "
                "x joint_motion(q_joint) x joint_post_motion_from_child_zero"
            ),
            "pre_motion_frame": f"joint/{name}",
            "pre_motion_frame_is_affected_by_own_motion": False,
        }

    semantics = model.frame_semantics()
    attachments = _joint_local_attachments(model)
    frame_derivations: dict[str, Any] = {}
    for frame_name, semantic in sorted(semantics.items()):
        frame_type = semantic["type"]
        if frame_type == "link":
            attachment_link = frame_name
            terminal = _identity()
        elif frame_type == "joint_pre_motion":
            owner_joint = model.joints[str(semantic["owner"])]
            attachment_link = owner_joint.parent
            terminal = owner_joint.origin
        else:
            attachment_link = str(semantic["owner"])
            terminal = attachments[frame_name]
        path = model.chain(model.root_link, attachment_link)
        operator_refs = [f"joint_operator/{step['joint']}" for step in path["steps"]]
        dependencies = sorted({
            str(joint_position_rules[step["joint"]].get("driver_joint"))
            for step in path["steps"]
            if joint_position_rules[step["joint"]]["type"] != "constant"
        })
        tokens: list[str] = ["I"]
        for step in path["steps"]:
            tokens.extend([f"C[{step['joint']}]", f"M[{step['joint']}](q[{step['joint']}])"])
        tokens.append(f"A[{frame_name}]")
        frame_derivations[frame_name] = {
            "frame": frame_name,
            "semantic_type": frame_type,
            "owner": semantic["owner"],
            "parent_frame": semantic["parent_frame"],
            "attachment_link_before_terminal_constant": attachment_link,
            "ordered_operator_refs": operator_refs,
            "terminal_constant_attachment": _pose_record(terminal),
            "independent_driver_dependencies": dependencies,
            "composition_tokens": tokens,
            "expression_cnl": f"{model.root_link}_from_{frame_name}(q) = " + " x ".join(tokens),
        }

    core = {
        "schema_version": ARTICULATION_SCHEMA,
        "source_binding": source_binding,
        "coordinate_contract": {
            "root_frame": model.root_link,
            "transform_notation": "A_from_B is the pose of B expressed in A",
            "composition_order": "left-to-right parent-to-child homogeneous 4x4 matrix multiplication",
            "matrix_storage": "row-major 4x4",
            "joint_axis_frame": "joint pre-motion frame",
            "length_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "xyzw",
        },
        "language_contract": {
            "links": "typed spatial entities (nouns)",
            "joint_operators": "typed parameterized relations (verbs)",
            "frame_derivations": "ordered operator composition (syntax)",
            "driver_bindings": "variable assignment for one evaluated pose",
        },
        "independent_variables": independent_variables,
        "joint_position_rules": joint_position_rules,
        "joint_operators": operators,
        "frame_derivations": frame_derivations,
        "evaluation_contract": {
            "inputs": "finite positions for independent driver joints; omitted drivers use each declared feasible default",
            "dependent_joint_policy": "mimic joints are evaluated from affine rules; supplied dependent/fixed values must agree",
            "domain_policy": "reject independent values outside the mimic-constrained feasible domain",
            "result": "root_from_frame(q) for selected or all frames plus resolved physical joint positions",
        },
        "layer_contract": {
            "articulation_grammar": "pose-independent parameterized law",
            "forward_kinematics": "evaluation of that law at one driver binding",
            "geometric_jacobian": "pose-conditioned first derivative of target motion",
            "counterfactual_motion_atlas": "finite signed endpoint evaluations around one baseline",
        },
        "coverage": {
            "link_count": len(model.links),
            "physical_joint_count": len(model.joints),
            "independent_driver_count": len(independent_variables),
            "mimic_joint_count": sum(joint.mimic is not None for joint in model.joints.values()),
            "fixed_joint_count": sum(joint.type == "fixed" for joint in model.joints.values()),
            "frame_derivation_count": len(frame_derivations),
            "all_supported_frames_have_derivations": set(frame_derivations) == set(semantics),
        },
        "epistemic_scope": (
            "exact executable kinematic law for the validated supported source tree, its normalized joint anchors, "
            "axes, limits, and supported dependency equations; it is not a symbolic dynamics model, trajectory, swept volume, "
            "closed-loop constraint system, controller/hardware model, calibration proof, or physical observation"
        ),
    }
    law_projection = articulation_law_projection(core)
    law_digest = hashlib.sha256(_json_bytes(law_projection)).hexdigest()
    core["law_identity"] = {
        "schema_version": LAW_SCHEMA,
        "canonical_law_id": f"kinematic-law-{law_digest[:20]}",
        "canonical_law_sha256": law_digest,
        "source_binding_excluded": True,
        "identifier_policy": "typed identifiers are semantic; use an explicit bijective correspondence when source names differ",
        "normalization": "pre_constant x typed_motion(parameter) x post_constant for every tree edge",
    }
    digest = hashlib.sha256(_json_bytes(core)).hexdigest()
    return {
        **core,
        "grammar_id": f"articulation-{digest[:20]}",
        "grammar_input_sha256": digest,
    }


def write_articulation_grammar(path: Path, model: Any, source_binding: dict[str, Any]) -> dict[str, Any]:
    grammar = build_articulation_grammar(model, source_binding)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(grammar))
    return grammar


def read_articulation_grammar(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArticulationError(f"cannot read articulation grammar {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArticulationError("articulation grammar root must be an object")
    return value


def _typed_identifier_sets(grammar: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    joints = set(grammar["joint_operators"])
    links = {str(grammar["coordinate_contract"]["root_frame"])}
    for operator in grammar["joint_operators"].values():
        links.update((str(operator["parent_link"]), str(operator["child_link"])))
    frames = set(grammar["frame_derivations"])
    return links, joints, frames


def _mapping_contract(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    correspondence: dict[str, Any] | None,
    reference_sha256: str,
    candidate_sha256: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str, str | None]:
    reference_links, reference_joints, reference_frames = _typed_identifier_sets(reference)
    candidate_links, candidate_joints, candidate_frames = _typed_identifier_sets(candidate)
    if correspondence is None:
        if (reference_links, reference_joints, reference_frames) != (
            candidate_links,
            candidate_joints,
            candidate_frames,
        ):
            raise ArticulationError(
                "grammar identifiers differ; provide a digest-bound robot-spatial-articulation-correspondence.v1"
            )
        return (
            {name: name for name in candidate_links},
            {name: name for name in candidate_joints},
            {name: name for name in candidate_frames},
            "exact_typed_identifiers",
            None,
        )
    if correspondence.get("schema_version") != CORRESPONDENCE_SCHEMA:
        raise ArticulationError(f"correspondence schema must be {CORRESPONDENCE_SCHEMA!r}")
    if correspondence.get("reference_grammar_sha256") != reference_sha256:
        raise ArticulationError("correspondence reference_grammar_sha256 does not bind the supplied reference grammar")
    if correspondence.get("candidate_grammar_sha256") != candidate_sha256:
        raise ArticulationError("correspondence candidate_grammar_sha256 does not bind the supplied candidate grammar")
    mapping = correspondence.get("candidate_to_reference")
    if not isinstance(mapping, dict):
        raise ArticulationError("correspondence.candidate_to_reference must be an object")

    def complete_map(kind: str, candidate_names: set[str], reference_names: set[str]) -> dict[str, str]:
        value = mapping.get(kind)
        if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            raise ArticulationError(f"correspondence candidate_to_reference.{kind} must be a string map")
        if set(value) != candidate_names:
            raise ArticulationError(
                f"correspondence {kind} keys must cover every candidate identifier; "
                f"missing={sorted(candidate_names - set(value))}, unexpected={sorted(set(value) - candidate_names)}"
            )
        if set(value.values()) != reference_names or len(set(value.values())) != len(value):
            raise ArticulationError(f"correspondence {kind} must be a bijection onto every reference identifier")
        return dict(value)

    link_map = complete_map("links", candidate_links, reference_links)
    joint_map = complete_map("joints", candidate_joints, reference_joints)
    explicit_frames = mapping.get("frames", {})
    if not isinstance(explicit_frames, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in explicit_frames.items()
    ):
        raise ArticulationError("correspondence candidate_to_reference.frames must be a string map")
    frame_map: dict[str, str] = {}
    for frame in candidate_frames:
        if frame in candidate_links:
            frame_map[frame] = link_map[frame]
        elif frame.startswith("joint/") and frame.removeprefix("joint/") in candidate_joints:
            frame_map[frame] = f"joint/{joint_map[frame.removeprefix('joint/')]}"
        elif frame in explicit_frames:
            frame_map[frame] = explicit_frames[frame]
        else:
            raise ArticulationError(f"correspondence lacks non-derived frame mapping for {frame!r}")
    unexpected_frames = sorted(set(explicit_frames) - candidate_frames)
    if unexpected_frames:
        raise ArticulationError(f"correspondence has unexpected frame mappings: {unexpected_frames}")
    if set(frame_map.values()) != reference_frames or len(set(frame_map.values())) != len(frame_map):
        raise ArticulationError("correspondence frames must form a bijection onto every reference frame")
    digest = hashlib.sha256(_json_bytes(correspondence)).hexdigest()
    return link_map, joint_map, frame_map, "explicit_digest_bound_typed_identifier_correspondence", digest


def _mapped_law_projection(
    grammar: dict[str, Any],
    link_map: dict[str, str],
    joint_map: dict[str, str],
    frame_map: dict[str, str],
) -> dict[str, Any]:
    source = articulation_law_projection(grammar)

    def joint_rule_ref(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("joint_position_rule/"):
            name = value.removeprefix("joint_position_rule/")
            return f"joint_position_rule/{joint_map[name]}"
        return value

    coordinate = dict(source["coordinate_contract"])
    coordinate["root_frame"] = link_map[str(coordinate["root_frame"])]
    variables: dict[str, Any] = {}
    for candidate_name, variable in source["independent_variables"].items():
        mapped = json.loads(json.dumps(variable))
        mapped["variable_id"] = f"q/{joint_map[candidate_name]}"
        mapped["physical_joints_driven"] = sorted(joint_map[name] for name in mapped["physical_joints_driven"])
        for constraint in mapped["feasible_domain"]["constraints"]:
            constraint["joint"] = joint_map[constraint["joint"]]
            constraint["mimic_chain"] = [joint_map[name] for name in constraint["mimic_chain"]]
        mapped["feasible_domain"]["constraints"] = sorted(
            mapped["feasible_domain"]["constraints"], key=lambda item: item["joint"]
        )
        variables[joint_map[candidate_name]] = mapped
    position_rules: dict[str, Any] = {}
    for candidate_name, rule in source["joint_position_rules"].items():
        mapped = json.loads(json.dumps(rule))
        if "driver_joint" in mapped:
            mapped["driver_joint"] = joint_map[mapped["driver_joint"]]
        if "driver_variable" in mapped:
            mapped["driver_variable"] = f"q/{joint_map[mapped['driver_variable'].removeprefix('q/')]}"
        if "mimic_chain_from_physical_joint_to_driver" in mapped:
            mapped["mimic_chain_from_physical_joint_to_driver"] = [
                joint_map[name] for name in mapped["mimic_chain_from_physical_joint_to_driver"]
            ]
        position_rules[joint_map[candidate_name]] = mapped
    operators: dict[str, Any] = {}
    for candidate_name, operator in source["joint_operators"].items():
        mapped = json.loads(json.dumps(operator))
        reference_name = joint_map[candidate_name]
        mapped["operator_id"] = f"joint_operator/{reference_name}"
        mapped["joint"] = reference_name
        mapped["parent_link"] = link_map[mapped["parent_link"]]
        mapped["child_link"] = link_map[mapped["child_link"]]
        mapped["pre_motion_frame"] = f"joint/{reference_name}"
        mapped["motion_operator"] = {
            key: joint_rule_ref(value) for key, value in mapped["motion_operator"].items()
        }
        operators[reference_name] = mapped
    derivations: dict[str, Any] = {}
    for candidate_frame, derivation in source["frame_derivations"].items():
        mapped = json.loads(json.dumps(derivation))
        reference_frame = frame_map[candidate_frame]
        mapped["frame"] = reference_frame
        semantic_type = mapped["semantic_type"]
        if semantic_type == "joint_pre_motion":
            mapped["owner"] = joint_map[mapped["owner"]]
        else:
            mapped["owner"] = link_map[mapped["owner"]]
        if mapped["parent_frame"] is not None:
            mapped["parent_frame"] = frame_map[mapped["parent_frame"]]
        mapped["attachment_link_before_terminal_constant"] = link_map[
            mapped["attachment_link_before_terminal_constant"]
        ]
        mapped["ordered_operator_refs"] = [
            f"joint_operator/{joint_map[value.removeprefix('joint_operator/')]}"
            for value in mapped["ordered_operator_refs"]
        ]
        mapped["independent_driver_dependencies"] = sorted(
            joint_map[name] for name in mapped["independent_driver_dependencies"]
        )
        derivations[reference_frame] = mapped
    return {
        "schema_version": LAW_SCHEMA,
        "coordinate_contract": coordinate,
        "independent_variables": dict(sorted(variables.items())),
        "joint_position_rules": dict(sorted(position_rules.items())),
        "joint_operators": dict(sorted(operators.items())),
        "frame_derivations": dict(sorted(derivations.items())),
        "evaluation_contract": source["evaluation_contract"],
        "coverage": source["coverage"],
    }


def _first_difference(reference: Any, candidate: Any, path: str = "$.") -> dict[str, Any] | None:
    if type(reference) is not type(candidate):
        return {"path": path, "reference": reference, "candidate": candidate, "reason": "type_mismatch"}
    if isinstance(reference, dict):
        if set(reference) != set(candidate):
            return {
                "path": path,
                "reference_keys": sorted(reference),
                "candidate_keys": sorted(candidate),
                "reason": "key_set_mismatch",
            }
        for key in sorted(reference):
            difference = _first_difference(reference[key], candidate[key], f"{path}{key}.")
            if difference is not None:
                return difference
        return None
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            return {"path": path, "reference_length": len(reference), "candidate_length": len(candidate), "reason": "length_mismatch"}
        for index, (left, right) in enumerate(zip(reference, candidate)):
            difference = _first_difference(left, right, f"{path}[{index}].")
            if difference is not None:
                return difference
        return None
    if reference != candidate:
        return {"path": path, "reference": reference, "candidate": candidate, "reason": "value_mismatch"}
    return None


def compare_articulation_grammars(
    reference_path: Path,
    candidate_path: Path,
    correspondence_path: Path | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ArticulationError("comparison tolerance must be finite and positive")
    reference = read_articulation_grammar(reference_path)
    candidate = read_articulation_grammar(candidate_path)
    for label, grammar in (("reference", reference), ("candidate", candidate)):
        if grammar.get("schema_version") != ARTICULATION_SCHEMA:
            raise ArticulationError(f"{label} grammar schema must be {ARTICULATION_SCHEMA!r}")
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    correspondence = None
    if correspondence_path is not None:
        try:
            correspondence = json.loads(correspondence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArticulationError(f"cannot read correspondence: {error}") from error
        if not isinstance(correspondence, dict):
            raise ArticulationError("correspondence root must be an object")
    link_map, joint_map, frame_map, mode, correspondence_sha = _mapping_contract(
        reference,
        candidate,
        correspondence,
        reference_sha,
        candidate_sha,
    )
    reference_projection = articulation_law_projection(reference)
    candidate_projection = _mapped_law_projection(candidate, link_map, joint_map, frame_map)
    structural_difference = _first_difference(reference_projection, candidate_projection)
    issues: list[dict[str, Any]] = []
    if structural_difference is not None:
        issues.append({"check": "canonical_law_projection", **structural_difference})
    maximum_error = 0.0
    frame_evaluations = 0
    probe_count = 0
    if structural_difference is None:
        inverse_joint_map = {reference_name: candidate_name for candidate_name, reference_name in joint_map.items()}
        for probe_index, reference_pose in enumerate(_probe_values(reference)):
            candidate_pose = {
                inverse_joint_map[reference_driver]: value
                for reference_driver, value in reference_pose.items()
            }
            reference_result = evaluate_articulation_grammar(
                reference, reference_pose, pose_name=f"cross_representation_probe_{probe_index}"
            )
            candidate_result = evaluate_articulation_grammar(
                candidate, candidate_pose, pose_name=f"cross_representation_probe_{probe_index}"
            )
            probe_count += 1
            for candidate_frame, reference_frame in frame_map.items():
                reference_matrix = reference_result["frames"][reference_frame]["root_from_frame"]["matrix_4x4_rowmajor"]
                candidate_matrix = candidate_result["frames"][candidate_frame]["root_from_frame"]["matrix_4x4_rowmajor"]
                error = max(
                    abs(float(reference_matrix[row][column]) - float(candidate_matrix[row][column]))
                    for row in range(4)
                    for column in range(4)
                )
                maximum_error = max(maximum_error, error)
                frame_evaluations += 1
                if error > tolerance:
                    issues.append({
                        "check": f"execution.probe_{probe_index}.frame.{reference_frame}",
                        "matrix_absolute_error": error,
                    })
    reference_projection_sha = hashlib.sha256(_json_bytes(reference_projection)).hexdigest()
    mapped_candidate_projection_sha = hashlib.sha256(_json_bytes(candidate_projection)).hexdigest()
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "equivalent" if not issues else "different",
        "comparison_mode": mode,
        "reference": {
            "path": str(reference_path.resolve()),
            "artifact_sha256": reference_sha,
            "grammar_id": reference.get("grammar_id"),
            "law_identity": reference.get("law_identity"),
            "source_binding": reference.get("source_binding"),
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "artifact_sha256": candidate_sha,
            "grammar_id": candidate.get("grammar_id"),
            "law_identity": candidate.get("law_identity"),
            "source_binding": candidate.get("source_binding"),
        },
        "correspondence": {
            "provided": correspondence is not None,
            "path": None if correspondence_path is None else str(correspondence_path.resolve()),
            "sha256": correspondence_sha,
            "candidate_to_reference": {"links": link_map, "joints": joint_map, "frames": frame_map},
        },
        "canonical_comparison": {
            "reference_projection_sha256": reference_projection_sha,
            "mapped_candidate_projection_sha256": mapped_candidate_projection_sha,
            "exact_projection_match": structural_difference is None,
        },
        "execution_crosscheck": {
            "probe_count": probe_count,
            "all_frame_evaluation_count": frame_evaluations,
            "matrix_tolerance": tolerance,
            "maximum_matrix_absolute_error": _clean(maximum_error),
        },
        "issues": issues,
        "epistemic_scope": (
            "exact common-law and unseen-binding agreement for the two normalized supported source trees under the explicit typed identifier contract; "
            "not evidence for omitted source semantics, dynamics, closed loops, hardware, calibration, or physical truth"
        ),
    }


def _resolve_positions(grammar: dict[str, Any], supplied: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    variables = grammar.get("independent_variables")
    rules = grammar.get("joint_position_rules")
    if not isinstance(variables, dict) or not isinstance(rules, dict):
        raise ArticulationError("grammar must contain independent_variables and joint_position_rules objects")
    unknown = sorted(set(supplied) - set(rules))
    if unknown:
        raise ArticulationError(f"pose contains joints absent from the grammar: {unknown}")
    drivers: dict[str, float] = {}
    for driver, record in sorted(variables.items()):
        if not isinstance(record, dict):
            raise ArticulationError(f"independent variable {driver!r} must be an object")
        try:
            value = float(supplied.get(driver, record["default_value"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ArticulationError(f"independent variable {driver!r} has no valid value/default") from error
        if not math.isfinite(value):
            raise ArticulationError(f"independent variable {driver!r} must be finite")
        domain = record.get("feasible_domain")
        if not isinstance(domain, dict):
            raise ArticulationError(f"independent variable {driver!r} has no feasible_domain")
        lower, upper = domain.get("minimum"), domain.get("maximum")
        if lower is not None and value < float(lower) - EPSILON:
            raise ArticulationError(f"independent variable {driver!r} value {value} is below feasible minimum {lower}")
        if upper is not None and value > float(upper) + EPSILON:
            raise ArticulationError(f"independent variable {driver!r} value {value} is above feasible maximum {upper}")
        drivers[driver] = value
    resolved: dict[str, float] = {}
    for joint, rule in sorted(rules.items()):
        if not isinstance(rule, dict):
            raise ArticulationError(f"joint position rule {joint!r} must be an object")
        rule_type = rule.get("type")
        if rule_type == "constant":
            value = float(rule.get("value", 0.0))
        elif rule_type in {"independent_variable", "affine_driver_dependency"}:
            driver = rule.get("driver_joint")
            if driver not in drivers:
                raise ArticulationError(f"joint position rule {joint!r} references unknown driver {driver!r}")
            value = float(rule.get("multiplier")) * drivers[str(driver)] + float(rule.get("offset"))
        else:
            raise ArticulationError(f"joint position rule {joint!r} has unsupported type {rule_type!r}")
        if not math.isfinite(value):
            raise ArticulationError(f"joint position rule {joint!r} produced a non-finite value")
        if joint in supplied and abs(float(supplied[joint]) - value) > 1e-9:
            raise ArticulationError(
                f"supplied dependent/fixed joint {joint!r} value {supplied[joint]} disagrees with grammar value {value}"
            )
        resolved[joint] = _clean(value)
    return drivers, resolved


def evaluate_articulation_grammar(
    grammar: dict[str, Any],
    supplied_positions: dict[str, float],
    target_frames: Iterable[str] | None = None,
    pose_name: str = "supplied",
) -> dict[str, Any]:
    """Execute only the grammar artifact; no URDF parser or RobotModel is used."""
    if grammar.get("schema_version") != ARTICULATION_SCHEMA:
        raise ArticulationError(
            f"unsupported articulation schema {grammar.get('schema_version')!r}; expected {ARTICULATION_SCHEMA!r}"
        )
    coordinate = grammar.get("coordinate_contract")
    operators = grammar.get("joint_operators")
    derivations = grammar.get("frame_derivations")
    if not isinstance(coordinate, dict) or not isinstance(operators, dict) or not isinstance(derivations, dict):
        raise ArticulationError("grammar is missing coordinate_contract, joint_operators, or frame_derivations")
    root = coordinate.get("root_frame")
    if not isinstance(root, str) or not root:
        raise ArticulationError("coordinate_contract.root_frame must be a non-empty string")
    drivers, resolved = _resolve_positions(grammar, supplied_positions)
    selected = sorted(derivations) if target_frames is None else list(dict.fromkeys(target_frames))
    unknown = sorted(set(selected) - set(derivations))
    if unknown:
        raise ArticulationError(f"unknown target frames in grammar: {unknown}")
    frames: dict[str, Any] = {}
    for frame in selected:
        derivation = derivations[frame]
        if not isinstance(derivation, dict):
            raise ArticulationError(f"frame derivation {frame!r} must be an object")
        transform = _identity()
        current_link = root
        trace: list[dict[str, Any]] = []
        refs = derivation.get("ordered_operator_refs")
        if not isinstance(refs, list):
            raise ArticulationError(f"frame derivation {frame!r}.ordered_operator_refs must be an array")
        for index, operator_ref in enumerate(refs):
            if not isinstance(operator_ref, str) or not operator_ref.startswith("joint_operator/"):
                raise ArticulationError(f"frame derivation {frame!r} has invalid operator reference {operator_ref!r}")
            joint = operator_ref.removeprefix("joint_operator/")
            operator = operators.get(joint)
            if not isinstance(operator, dict):
                raise ArticulationError(f"frame derivation {frame!r} references missing operator {operator_ref!r}")
            if operator.get("operator_id") != operator_ref:
                raise ArticulationError(f"operator {joint!r} ID does not match reference {operator_ref!r}")
            if operator.get("parent_link") != current_link:
                raise ArticulationError(
                    f"frame derivation {frame!r} operator {joint!r} starts at {operator.get('parent_link')!r}, expected {current_link!r}"
                )
            constant = _matrix_from_record(
                operator.get("constant_parent_from_pre_motion"),
                f"joint_operators.{joint}.constant_parent_from_pre_motion",
            )
            post_motion_record = operator.get("post_motion_from_child_zero", _pose_record(_identity()))
            post_motion = _matrix_from_record(
                post_motion_record,
                f"joint_operators.{joint}.post_motion_from_child_zero",
            )
            motion_record = operator.get("motion_operator")
            if not isinstance(motion_record, dict):
                raise ArticulationError(f"joint operator {joint!r}.motion_operator must be an object")
            motion_type = motion_record.get("type")
            position = resolved.get(joint)
            if position is None:
                raise ArticulationError(f"joint operator {joint!r} has no position rule")
            if motion_type == "identity":
                motion = _identity()
            elif motion_type == "rotation_about_axis":
                axis = motion_record.get("axis_xyz_in_pre_motion_frame")
                if not isinstance(axis, list) or len(axis) != 3:
                    raise ArticulationError(f"joint operator {joint!r} rotation axis must have three components")
                motion = _axis_angle([float(value) for value in axis], position)
            elif motion_type == "translation_along_axis":
                axis = motion_record.get("axis_xyz_in_pre_motion_frame")
                if not isinstance(axis, list) or len(axis) != 3:
                    raise ArticulationError(f"joint operator {joint!r} translation axis must have three components")
                motion = _translation([float(value) * position for value in axis])
            else:
                raise ArticulationError(f"joint operator {joint!r} has unsupported motion type {motion_type!r}")
            transform = _matmul(_matmul(_matmul(transform, constant), motion), post_motion)
            current_link = operator.get("child_link")
            trace.append({
                "index": index,
                "operator_ref": operator_ref,
                "joint_position": position,
                "joint_position_unit": grammar["joint_position_rules"][joint].get("unit"),
                "constant_parent_from_pre_motion": operator["constant_parent_from_pre_motion"],
                "motion_operator": motion_record,
                "post_motion_from_child_zero": post_motion_record,
                "result_root_from_child": _pose_record(transform),
            })
        expected_attachment = derivation.get("attachment_link_before_terminal_constant")
        if current_link != expected_attachment:
            raise ArticulationError(
                f"frame derivation {frame!r} ends at link {current_link!r}, expected {expected_attachment!r}"
            )
        terminal = _matrix_from_record(
            derivation.get("terminal_constant_attachment"),
            f"frame_derivations.{frame}.terminal_constant_attachment",
        )
        transform = _matmul(transform, terminal)
        frames[frame] = {
            "root_from_frame": _pose_record(transform),
            "independent_driver_dependencies": derivation.get("independent_driver_dependencies", []),
            "operator_trace": trace,
        }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "grammar_binding": {
            "grammar_id": grammar.get("grammar_id"),
            "grammar_input_sha256": grammar.get("grammar_input_sha256"),
            "law_identity": grammar.get("law_identity"),
            "source_binding": grammar.get("source_binding"),
        },
        "pose": {
            "name": pose_name,
            "supplied_joint_positions": {name: _clean(float(value)) for name, value in sorted(supplied_positions.items())},
            "independent_driver_positions": {name: _clean(value) for name, value in sorted(drivers.items())},
            "resolved_physical_joint_positions": resolved,
        },
        "reference_frame": root,
        "frame_count": len(frames),
        "frames": frames,
        "meaning": "pose-independent articulation law evaluated at this explicit variable binding",
        "epistemic_scope": grammar.get("epistemic_scope"),
    }


def _probe_values(grammar: dict[str, Any]) -> list[dict[str, float]]:
    variables = grammar["independent_variables"]
    baseline = {name: float(record["default_value"]) for name, record in sorted(variables.items())}
    probes = [baseline]
    for name, record in sorted(variables.items()):
        domain = record["feasible_domain"]
        lower, upper = domain["minimum"], domain["maximum"]
        value = baseline[name]
        nominal = 0.37 if record["unit"] == "rad" else 0.037
        if lower is not None and upper is not None:
            lower_value, upper_value = float(lower), float(upper)
            value = lower_value + 0.37 * (upper_value - lower_value)
        elif upper is not None:
            value = min(value + nominal, float(upper) - min(nominal, 1e-6))
        elif lower is not None:
            value = max(value - nominal, float(lower) + min(nominal, 1e-6))
        else:
            value += nominal
        probe = dict(baseline)
        probe[name] = _clean(value)
        probes.append(probe)
    return probes


def verify_articulation_grammar(
    path: Path,
    model: Any,
    source_binding: dict[str, Any],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Regenerate the law and cross-check its standalone evaluator against RobotModel FK."""
    issues: list[dict[str, Any]] = []
    try:
        grammar = read_articulation_grammar(path)
    except ArticulationError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "grammar_path": str(path.resolve()),
            "issues": [{"check": "grammar.read", "message": str(error)}],
        }
    grammar_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = build_articulation_grammar(model, source_binding)
    if grammar.get("source_binding") != source_binding:
        issues.append({"check": "grammar.source_binding", "message": "source binding differs from the supplied source artifact"})
    if grammar != expected:
        issues.append({"check": "grammar.regeneration", "message": "grammar does not exactly match deterministic regeneration"})
    max_matrix_error = 0.0
    verified_frames = 0
    probes = _probe_values(expected)
    try:
        for probe_index, probe in enumerate(probes):
            evaluated = evaluate_articulation_grammar(grammar, probe, pose_name=f"verification_probe_{probe_index}")
            oracle_frames, oracle_pose = model.world_frames(probe)
            if evaluated["pose"]["resolved_physical_joint_positions"] != {
                name: _clean(value) for name, value in sorted(oracle_pose.items())
            }:
                issues.append({
                    "check": f"evaluation.probe_{probe_index}.joint_positions",
                    "message": "grammar joint bindings differ from source-model dependency resolution",
                })
            if set(evaluated["frames"]) != set(oracle_frames):
                issues.append({
                    "check": f"evaluation.probe_{probe_index}.frame_set",
                    "message": "grammar and source-model frame sets differ",
                })
                continue
            for frame, oracle_matrix in oracle_frames.items():
                actual_matrix = evaluated["frames"][frame]["root_from_frame"]["matrix_4x4_rowmajor"]
                error = max(
                    abs(float(actual_matrix[row][column]) - float(oracle_matrix[row][column]))
                    for row in range(4)
                    for column in range(4)
                )
                max_matrix_error = max(max_matrix_error, error)
                verified_frames += 1
                if error > tolerance:
                    issues.append({
                        "check": f"evaluation.probe_{probe_index}.frame.{frame}",
                        "message": f"matrix error {error} exceeds tolerance {tolerance}",
                    })
    except (ArticulationError, KeyError, TypeError, ValueError) as error:
        issues.append({"check": "grammar.execution", "message": str(error)})
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "grammar_path": str(path.resolve()),
        "grammar_sha256": grammar_sha256,
        "grammar_id": grammar.get("grammar_id"),
        "source_binding": source_binding,
        "probe_count": len(probes),
        "verified_frame_evaluation_count": verified_frames,
        "matrix_tolerance": tolerance,
        "maximum_matrix_absolute_error": _clean(max_matrix_error),
        "checks": [
            "exact deterministic grammar regeneration",
            "standalone typed-AST execution",
            "mimic-constrained joint binding agreement",
            "all-frame FK matrix agreement over deterministic fresh probes",
        ],
        "independent_oracle": False,
        "issues": issues,
        "epistemic_scope": (
            "regeneration and a separate grammar executor are compared with this implementation's source-format FK; "
            "use the independent articulation oracle for parser/FK implementation diversity"
        ),
    }
