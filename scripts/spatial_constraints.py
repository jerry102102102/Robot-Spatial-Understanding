#!/usr/bin/env python3
"""Executable supplemental constraints over a standalone articulation grammar."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from spatial_articulation import (
    ARTICULATION_SCHEMA,
    ArticulationError,
    evaluate_articulation_grammar,
    read_articulation_grammar,
)


SPEC_SCHEMA = "robot-spatial-constraint-spec.v1"
GRAPH_SCHEMA = "robot-spatial-constraint-graph.v1"
EVALUATION_SCHEMA = "robot-spatial-constraint-evaluation.v1"
SOLUTION_SCHEMA = "robot-spatial-constraint-solution.v1"
VERIFICATION_SCHEMA = "robot-spatial-constraint-verification.v1"
EPSILON = 1e-12
Matrix = list[list[float]]
Vector = list[float]


class ConstraintError(ValueError):
    """A malformed constraint contract or failed constraint operation."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: float, digits: int = 12) -> float:
    return 0.0 if abs(value) < 0.5 * 10 ** (-digits) else round(float(value), digits)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConstraintError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ConstraintError(f"{label} must be a finite number")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ConstraintError(f"{label} must be positive")
    return result


def _vector(value: Any, label: str, length: int = 3) -> Vector:
    if not isinstance(value, list) or len(value) != length:
        raise ConstraintError(f"{label} must be an array of {length} finite numbers")
    return [_finite(component, f"{label}[{index}]") for index, component in enumerate(value)]


def _norm(vector: Vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector, right: Vector) -> Vector:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _normalize(value: Any, label: str) -> Vector:
    vector = _vector(value, label)
    length = _norm(vector)
    if length <= EPSILON:
        raise ConstraintError(f"{label} must be non-zero")
    normalized = [component / length for component in vector]
    if abs(length - 1.0) > 1e-9:
        raise ConstraintError(f"{label} must be unit length; got norm {length}")
    return [_clean(component) for component in normalized]


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


def _rotate(transform: Matrix, vector: Vector) -> Vector:
    return [sum(transform[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _quaternion_matrix(quaternion_xyzw: Vector) -> Matrix:
    x, y, z, w = quaternion_xyzw
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), 0.0],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), 0.0],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_quaternion_xyzw(transform: Matrix) -> Vector:
    trace = transform[0][0] + transform[1][1] + transform[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (transform[2][1] - transform[1][2]) / scale
        y = (transform[0][2] - transform[2][0]) / scale
        z = (transform[1][0] - transform[0][1]) / scale
    elif transform[0][0] > transform[1][1] and transform[0][0] > transform[2][2]:
        scale = math.sqrt(max(0.0, 1.0 + transform[0][0] - transform[1][1] - transform[2][2])) * 2.0
        x = 0.25 * scale
        y = (transform[0][1] + transform[1][0]) / scale
        z = (transform[0][2] + transform[2][0]) / scale
        w = (transform[2][1] - transform[1][2]) / scale
    elif transform[1][1] > transform[2][2]:
        scale = math.sqrt(max(0.0, 1.0 + transform[1][1] - transform[0][0] - transform[2][2])) * 2.0
        x = (transform[0][1] + transform[1][0]) / scale
        y = 0.25 * scale
        z = (transform[1][2] + transform[2][1]) / scale
        w = (transform[0][2] - transform[2][0]) / scale
    else:
        scale = math.sqrt(max(0.0, 1.0 + transform[2][2] - transform[0][0] - transform[1][1])) * 2.0
        x = (transform[0][2] + transform[2][0]) / scale
        y = (transform[1][2] + transform[2][1]) / scale
        z = 0.25 * scale
        w = (transform[1][0] - transform[0][1]) / scale
    quaternion = [x, y, z, w]
    length = _norm(quaternion)
    if length <= EPSILON:
        return [0.0, 0.0, 0.0, 1.0]
    quaternion = [component / length for component in quaternion]
    if quaternion[3] < 0.0:
        quaternion = [-component for component in quaternion]
    return [_clean(component) for component in quaternion]


def _pose_record(transform: Matrix) -> dict[str, Any]:
    return {
        "translation_xyz_m": [_clean(transform[index][3]) for index in range(3)],
        "quaternion_xyzw": _matrix_quaternion_xyzw(transform),
        "matrix_4x4_rowmajor": [[_clean(component) for component in row] for row in transform],
    }


def _matrix_from_pose_record(record: Any, label: str) -> Matrix:
    if not isinstance(record, dict):
        raise ConstraintError(f"{label} must be a pose object")
    matrix = record.get("matrix_4x4_rowmajor")
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ConstraintError(f"{label}.matrix_4x4_rowmajor must be 4x4")
    result = [[_finite(value, f"{label}.matrix[{row}][{column}]") for column, value in enumerate(values)] for row, values in enumerate(matrix)]
    if any(abs(result[3][index] - expected) > 1e-9 for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ConstraintError(f"{label} is not a homogeneous rigid transform")
    return result


def _canonical_transform(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"translation_xyz_m", "quaternion_xyzw"}:
        raise ConstraintError(f"{label} must contain exactly translation_xyz_m and quaternion_xyzw")
    translation = _vector(value["translation_xyz_m"], f"{label}.translation_xyz_m")
    quaternion = _vector(value["quaternion_xyzw"], f"{label}.quaternion_xyzw", 4)
    length = _norm(quaternion)
    if abs(length - 1.0) > 1e-9:
        raise ConstraintError(f"{label}.quaternion_xyzw must be unit length; got norm {length}")
    if quaternion[3] < 0.0:
        quaternion = [-component for component in quaternion]
    transform = _quaternion_matrix(quaternion)
    for index in range(3):
        transform[index][3] = translation[index]
    return _pose_record(transform)


def _rotation_vector(transform: Matrix) -> Vector:
    x, y, z, w = _matrix_quaternion_xyzw(transform)
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= EPSILON:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, w))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return [angle * x / vector_norm, angle * y / vector_norm, angle * z / vector_norm]


def _perpendicular_basis(axis: Vector) -> tuple[Vector, Vector]:
    seed = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.8 else [0.0, 1.0, 0.0]
    first = _cross(axis, seed)
    length = _norm(first)
    first = [component / length for component in first]
    second = _cross(axis, first)
    return first, second


def _axis_alignment_vector(reference: Vector, candidate: Vector) -> Vector:
    cross = _cross(candidate, reference)
    sine = min(1.0, _norm(cross))
    cosine = max(-1.0, min(1.0, _dot(candidate, reference)))
    angle = math.atan2(sine, cosine)
    if sine <= EPSILON:
        if cosine >= 0.0:
            return [0.0, 0.0, 0.0]
        first, _ = _perpendicular_basis(reference)
        return [math.pi * component for component in first]
    return [angle * component / sine for component in cross]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConstraintError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConstraintError(f"{label} root must be an object")
    return value


def read_constraint_graph(path: Path) -> dict[str, Any]:
    graph = _read_json(path, "constraint graph")
    if graph.get("schema_version") != GRAPH_SCHEMA:
        raise ConstraintError(f"constraint graph must use schema_version {GRAPH_SCHEMA}")
    return graph


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or ".." in value.split("/"):
        raise ConstraintError(f"{label} must be a non-empty relative typed identifier")
    return value


def _exact_fields(record: Any, required: set[str], optional: set[str], label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ConstraintError(f"{label} must be an object")
    missing = required - set(record)
    unknown = set(record) - required - optional
    if missing or unknown:
        raise ConstraintError(f"{label} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return record


def _normalize_tolerances(value: Any, label: str) -> dict[str, float]:
    record = _exact_fields(value, {"translation_m", "rotation_rad"}, set(), label)
    return {
        "translation_m": _positive(record["translation_m"], f"{label}.translation_m"),
        "rotation_rad": _positive(record["rotation_rad"], f"{label}.rotation_rad"),
    }


def _frame_operator_refs(grammar: dict[str, Any], frame: str, attachments: dict[str, dict[str, Any]]) -> list[str]:
    base_frame = attachments[frame]["parent_frame"] if frame in attachments else frame
    derivation = grammar["frame_derivations"][base_frame]
    return [str(reference) for reference in derivation["ordered_operator_refs"]]


def _tree_path(grammar: dict[str, Any], frame_a: str, frame_b: str, attachments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = _frame_operator_refs(grammar, frame_a, attachments)
    right = _frame_operator_refs(grammar, frame_b, attachments)
    common = 0
    while common < min(len(left), len(right)) and left[common] == right[common]:
        common += 1
    return {
        "from_frame": frame_a,
        "to_frame": frame_b,
        "common_prefix_operator_count": common,
        "from_branch_reverse_operator_refs": list(reversed(left[common:])),
        "to_branch_forward_operator_refs": right[common:],
    }


def build_constraint_graph(
    grammar: dict[str, Any],
    grammar_sha256: str,
    spec: dict[str, Any],
    spec_sha256: str,
) -> dict[str, Any]:
    if grammar.get("schema_version") != ARTICULATION_SCHEMA:
        raise ConstraintError(f"embedded grammar must use schema_version {ARTICULATION_SCHEMA}")
    _exact_fields(
        spec,
        {"schema_version", "constraint_set_id", "grammar_sha256", "attachments", "constraints"},
        set(),
        "constraint spec",
    )
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ConstraintError(f"constraint spec must use schema_version {SPEC_SCHEMA}")
    constraint_set_id = _identifier(spec.get("constraint_set_id"), "constraint_set_id")
    if spec.get("grammar_sha256") != grammar_sha256:
        raise ConstraintError("constraint spec grammar_sha256 does not bind the exact grammar artifact")
    frame_derivations = grammar.get("frame_derivations")
    if not isinstance(frame_derivations, dict) or not frame_derivations:
        raise ConstraintError("grammar frame_derivations must be a non-empty object")

    raw_attachments = spec.get("attachments")
    if not isinstance(raw_attachments, list):
        raise ConstraintError("constraint spec attachments must be an array")
    attachment_records: list[dict[str, Any]] = []
    attachment_map: dict[str, dict[str, Any]] = {}
    allowed_attachment_roles = {"constraint_anchor", "mount", "tcp", "measurement_point", "joint_anchor"}
    for index, raw in enumerate(raw_attachments):
        label = f"attachments[{index}]"
        record = _exact_fields(
            raw,
            {"attachment_id", "parent_frame", "semantic_role", "parent_from_attachment"},
            set(),
            label,
        )
        attachment_id = _identifier(record["attachment_id"], f"{label}.attachment_id")
        frame_id = f"attachment/{attachment_id}"
        if frame_id in attachment_map or frame_id in frame_derivations:
            raise ConstraintError(f"duplicate or colliding attachment frame {frame_id!r}")
        parent_frame = _identifier(record["parent_frame"], f"{label}.parent_frame")
        if parent_frame not in frame_derivations:
            raise ConstraintError(f"{label}.parent_frame {parent_frame!r} is absent from grammar frames")
        role = record["semantic_role"]
        if role not in allowed_attachment_roles:
            raise ConstraintError(f"{label}.semantic_role is unsupported: {role!r}")
        normalized = {
            "attachment_id": attachment_id,
            "frame_id": frame_id,
            "parent_frame": parent_frame,
            "semantic_role": role,
            "parent_from_attachment": _canonical_transform(
                record["parent_from_attachment"], f"{label}.parent_from_attachment"
            ),
        }
        attachment_records.append(normalized)
        attachment_map[frame_id] = normalized
    attachment_records.sort(key=lambda record: record["attachment_id"])
    known_frames = set(frame_derivations) | set(attachment_map)

    raw_constraints = spec.get("constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        raise ConstraintError("constraint spec constraints must be a non-empty array")
    normalized_constraints: list[dict[str, Any]] = []
    constraint_ids: set[str] = set()
    frame_constraints: list[dict[str, Any]] = []
    coordinate_constraints: list[dict[str, Any]] = []
    physical_joints = set(grammar.get("joint_position_rules", {}))
    for index, raw in enumerate(raw_constraints):
        label = f"constraints[{index}]"
        if not isinstance(raw, dict):
            raise ConstraintError(f"{label} must be an object")
        constraint_type = raw.get("type")
        constraint_id = _identifier(raw.get("constraint_id"), f"{label}.constraint_id")
        if constraint_id in constraint_ids:
            raise ConstraintError(f"duplicate constraint_id {constraint_id!r}")
        constraint_ids.add(constraint_id)
        if constraint_type == "kinematic_pair":
            required = {"constraint_id", "type", "role", "frame_a", "frame_b", "joint_type", "tolerances"}
            joint_type = raw.get("joint_type")
            if joint_type in {"revolute", "continuous", "prismatic"}:
                required |= {"axis_xyz_in_a", "axis_xyz_in_b"}
            record = _exact_fields(raw, required, set(), label)
            role = record["role"]
            if role not in {"loop_closure", "assembly_constraint", "calibration_assertion"}:
                raise ConstraintError(f"{label}.role is unsupported for kinematic_pair")
            if joint_type not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ConstraintError(f"{label}.joint_type is unsupported: {joint_type!r}")
            frame_a = _identifier(record["frame_a"], f"{label}.frame_a")
            frame_b = _identifier(record["frame_b"], f"{label}.frame_b")
            if frame_a not in known_frames or frame_b not in known_frames or frame_a == frame_b:
                raise ConstraintError(f"{label} must reference two distinct known frames")
            normalized = {
                "constraint_id": constraint_id,
                "constraint_ref": f"constraint/{constraint_id}",
                "type": constraint_type,
                "role": role,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "joint_type": joint_type,
                "tolerances": _normalize_tolerances(record["tolerances"], f"{label}.tolerances"),
                "residual_contract": {
                    "fixed": "three origin errors in frame_a plus the shortest three-component orientation log",
                    "revolute_or_continuous": "three origin errors plus two axis-alignment angular errors; twist about the aligned axis is free",
                    "prismatic": "two translation errors perpendicular to the axis plus the full three-component orientation log; translation along the axis is free",
                }["revolute_or_continuous" if joint_type in {"revolute", "continuous"} else joint_type],
            }
            if joint_type in {"revolute", "continuous", "prismatic"}:
                normalized["axis_xyz_in_a"] = _normalize(record["axis_xyz_in_a"], f"{label}.axis_xyz_in_a")
                normalized["axis_xyz_in_b"] = _normalize(record["axis_xyz_in_b"], f"{label}.axis_xyz_in_b")
            frame_constraints.append(normalized)
        elif constraint_type == "point_distance":
            record = _exact_fields(
                raw,
                {"constraint_id", "type", "role", "frame_a", "frame_b", "distance_m", "tolerance_m"},
                set(),
                label,
            )
            if record["role"] not in {"loop_closure", "cable_length", "assembly_constraint"}:
                raise ConstraintError(f"{label}.role is unsupported for point_distance")
            frame_a = _identifier(record["frame_a"], f"{label}.frame_a")
            frame_b = _identifier(record["frame_b"], f"{label}.frame_b")
            if frame_a not in known_frames or frame_b not in known_frames or frame_a == frame_b:
                raise ConstraintError(f"{label} must reference two distinct known frames")
            distance = _finite(record["distance_m"], f"{label}.distance_m")
            if distance < 0.0:
                raise ConstraintError(f"{label}.distance_m must be non-negative")
            normalized = {
                "constraint_id": constraint_id,
                "constraint_ref": f"constraint/{constraint_id}",
                "type": constraint_type,
                "role": record["role"],
                "frame_a": frame_a,
                "frame_b": frame_b,
                "distance_m": distance,
                "tolerance_m": _positive(record["tolerance_m"], f"{label}.tolerance_m"),
                "residual_contract": "euclidean distance between frame origins minus distance_m",
            }
            frame_constraints.append(normalized)
        elif constraint_type == "coordinate_linear":
            record = _exact_fields(
                raw,
                {"constraint_id", "type", "role", "terms", "offset", "tolerance"},
                set(),
                label,
            )
            if record["role"] not in {"mechanical_coupling", "calibration_assertion"}:
                raise ConstraintError(f"{label}.role is unsupported for coordinate_linear")
            terms = record["terms"]
            if not isinstance(terms, list) or not terms:
                raise ConstraintError(f"{label}.terms must be a non-empty array")
            normalized_terms: list[dict[str, Any]] = []
            term_joints: set[str] = set()
            for term_index, term in enumerate(terms):
                term_label = f"{label}.terms[{term_index}]"
                term_record = _exact_fields(term, {"joint", "coefficient"}, set(), term_label)
                joint = _identifier(term_record["joint"], f"{term_label}.joint")
                if joint not in physical_joints:
                    raise ConstraintError(f"{term_label}.joint {joint!r} is absent from grammar joint rules")
                if joint in term_joints:
                    raise ConstraintError(f"{label}.terms repeats joint {joint!r}")
                term_joints.add(joint)
                coefficient = _finite(term_record["coefficient"], f"{term_label}.coefficient")
                if abs(coefficient) <= EPSILON:
                    raise ConstraintError(f"{term_label}.coefficient must be non-zero")
                normalized_terms.append({"joint": joint, "coefficient": coefficient})
            normalized_terms.sort(key=lambda term: term["joint"])
            normalized = {
                "constraint_id": constraint_id,
                "constraint_ref": f"constraint/{constraint_id}",
                "type": constraint_type,
                "role": record["role"],
                "terms": normalized_terms,
                "offset": _finite(record["offset"], f"{label}.offset"),
                "tolerance": _positive(record["tolerance"], f"{label}.tolerance"),
                "residual_contract": "sum(coefficient * resolved_physical_joint_position) + offset = 0",
            }
            coordinate_constraints.append(normalized)
        else:
            raise ConstraintError(f"{label}.type is unsupported: {constraint_type!r}")
        normalized_constraints.append(normalized)
    normalized_constraints.sort(key=lambda record: record["constraint_id"])
    frame_constraints.sort(key=lambda record: record["constraint_id"])
    coordinate_constraints.sort(key=lambda record: record["constraint_id"])

    tree_edges = [
        {
            "edge_type": "tree_kinematic_edge",
            "joint": joint,
            "operator_ref": operator["operator_id"],
            "parent_link": operator["parent_link"],
            "child_link": operator["child_link"],
            "joint_type": operator["joint_type"],
        }
        for joint, operator in sorted(grammar["joint_operators"].items())
    ]
    attachment_edges = [
        {
            "edge_type": "rigid_attachment_edge",
            "parent_frame": record["parent_frame"],
            "child_frame": record["frame_id"],
            "semantic_role": record["semantic_role"],
        }
        for record in attachment_records
    ]
    constraint_edges = [
        {
            "edge_type": "frame_constraint_edge",
            "constraint_ref": record["constraint_ref"],
            "constraint_type": record["type"],
            "role": record["role"],
            "frame_a": record["frame_a"],
            "frame_b": record["frame_b"],
        }
        for record in frame_constraints
    ]
    cycle_records = [
        {
            "cycle_id": f"cycle/{record['constraint_id']}",
            "closure_constraint_ref": record["constraint_ref"],
            "tree_path": _tree_path(grammar, record["frame_a"], record["frame_b"], attachment_map),
            "meaning": "the spanning-tree path plus this supplemental constraint forms one declared mechanism loop or cross-branch relation",
        }
        for record in frame_constraints
    ]
    core = {
        "schema_version": GRAPH_SCHEMA,
        "constraint_set_id": constraint_set_id,
        "source_binding": {
            "articulation_grammar_sha256": grammar_sha256,
            "constraint_spec_sha256": spec_sha256,
            "source_formats": [grammar.get("source_binding", {}).get("source_format")],
            "supplemental_constraints_are_asserted": True,
        },
        "articulation_grammar": grammar,
        "attachments": attachment_records,
        "constraints": normalized_constraints,
        "structural_graph": {
            "root_frame": grammar["coordinate_contract"]["root_frame"],
            "spanning_tree_edges": tree_edges,
            "rigid_attachment_edges": attachment_edges,
            "frame_constraint_edges": constraint_edges,
            "coordinate_constraint_refs": [record["constraint_ref"] for record in coordinate_constraints],
            "declared_cycle_records": cycle_records,
            "tree_is_parameterization_not_complete_mechanism": bool(frame_constraints or coordinate_constraints),
        },
        "executable_contract": {
            "pose_input": "independent articulation driver positions; omitted drivers use grammar defaults",
            "frame_composition": "root_from_attachment = root_from_parent_frame x parent_from_attachment",
            "constraint_result": "typed signed residual components with declared units and tolerances",
            "local_mobility": "independent driver count minus numerical rank of the normalized residual Jacobian at one explicit pose",
            "solver": "damped Gauss-Newton over explicitly selected independent variables; all other drivers remain fixed",
        },
        "coverage": {
            "attachment_count": len(attachment_records),
            "constraint_count": len(normalized_constraints),
            "frame_constraint_count": len(frame_constraints),
            "coordinate_constraint_count": len(coordinate_constraints),
            "declared_cycle_count": len(cycle_records),
            "supported_constraint_types": ["coordinate_linear", "kinematic_pair", "point_distance"],
            "supported_kinematic_pairs": ["continuous", "fixed", "prismatic", "revolute"],
        },
        "epistemic_scope": (
            "exact execution of the embedded spanning-tree articulation law plus explicitly asserted supplemental rigid attachments, "
            "kinematic-pair, point-distance, and linear coordinate constraints; local rank/mobility is pose-conditioned numerical evidence, "
            "not a global configuration-space proof, dynamics/contact model, compliant solver, controller, hardware, calibration, or physical observation"
        ),
    }
    digest = hashlib.sha256(_json_bytes(core)).hexdigest()
    return {
        **core,
        "constraint_graph_id": f"constraint-graph-{digest[:20]}",
        "constraint_graph_sha256": digest,
    }


def write_constraint_graph(path: Path, grammar_path: Path, spec_path: Path) -> dict[str, Any]:
    grammar = read_articulation_grammar(grammar_path)
    spec = _read_json(spec_path, "constraint spec")
    graph = build_constraint_graph(grammar, _sha256(grammar_path), spec, _sha256(spec_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(graph))
    return graph


def _component(name: str, value: float, unit: str, tolerance: float) -> dict[str, Any]:
    normalized = abs(value) / tolerance
    return {
        "name": name,
        "value": _clean(value),
        "unit": unit,
        "tolerance": tolerance,
        "normalized_abs": _clean(normalized),
        "satisfied": normalized <= 1.0 + 1e-12,
    }


def _constraint_components(
    constraint: dict[str, Any],
    frame_matrices: dict[str, Matrix],
    physical_positions: dict[str, float],
) -> list[dict[str, Any]]:
    constraint_type = constraint["type"]
    if constraint_type == "coordinate_linear":
        value = constraint["offset"] + sum(
            term["coefficient"] * physical_positions[term["joint"]] for term in constraint["terms"]
        )
        return [_component("coordinate_linear_equation", value, "joint_coordinate", constraint["tolerance"])]
    frame_a = frame_matrices[constraint["frame_a"]]
    frame_b = frame_matrices[constraint["frame_b"]]
    a_from_b = _matmul(_inverse_rigid(frame_a), frame_b)
    translation = [a_from_b[index][3] for index in range(3)]
    if constraint_type == "point_distance":
        value = _norm(translation) - constraint["distance_m"]
        return [_component("origin_distance_minus_target", value, "m", constraint["tolerance_m"])]
    tolerances = constraint["tolerances"]
    joint_type = constraint["joint_type"]
    if joint_type == "fixed":
        rotation = _rotation_vector(a_from_b)
        return [
            *[_component(f"origin_{axis}", translation[index], "m", tolerances["translation_m"]) for index, axis in enumerate("xyz")],
            *[_component(f"orientation_log_{axis}", rotation[index], "rad", tolerances["rotation_rad"]) for index, axis in enumerate("xyz")],
        ]
    axis_a = constraint["axis_xyz_in_a"]
    axis_b_in_a = _rotate(a_from_b, constraint["axis_xyz_in_b"])
    first, second = _perpendicular_basis(axis_a)
    if joint_type in {"revolute", "continuous"}:
        alignment = _axis_alignment_vector(axis_a, axis_b_in_a)
        return [
            *[_component(f"origin_{axis}", translation[index], "m", tolerances["translation_m"]) for index, axis in enumerate("xyz")],
            _component("axis_alignment_u", _dot(alignment, first), "rad", tolerances["rotation_rad"]),
            _component("axis_alignment_v", _dot(alignment, second), "rad", tolerances["rotation_rad"]),
        ]
    rotation = _rotation_vector(a_from_b)
    return [
        _component("translation_perpendicular_u", _dot(translation, first), "m", tolerances["translation_m"]),
        _component("translation_perpendicular_v", _dot(translation, second), "m", tolerances["translation_m"]),
        *[_component(f"orientation_log_{axis}", rotation[index], "rad", tolerances["rotation_rad"]) for index, axis in enumerate("xyz")],
    ]


def _evaluate_core(graph: dict[str, Any], supplied_positions: dict[str, float], pose_name: str) -> dict[str, Any]:
    if graph.get("schema_version") != GRAPH_SCHEMA:
        raise ConstraintError(f"graph must use schema_version {GRAPH_SCHEMA}")
    grammar = graph.get("articulation_grammar")
    if not isinstance(grammar, dict) or grammar.get("schema_version") != ARTICULATION_SCHEMA:
        raise ConstraintError("constraint graph has no supported embedded articulation grammar")
    try:
        articulation = evaluate_articulation_grammar(grammar, supplied_positions, None, pose_name)
    except ArticulationError as error:
        raise ConstraintError(str(error)) from error
    frame_matrices = {
        frame: _matrix_from_pose_record(record["root_from_frame"], f"articulation.frames.{frame}.root_from_frame")
        for frame, record in articulation["frames"].items()
    }
    attachment_results: dict[str, Any] = {}
    for attachment in graph.get("attachments", []):
        parent = attachment["parent_frame"]
        if parent not in frame_matrices:
            raise ConstraintError(f"attachment parent frame {parent!r} is absent from embedded grammar evaluation")
        parent_from_attachment = _matrix_from_pose_record(
            attachment["parent_from_attachment"],
            f"attachments.{attachment['attachment_id']}.parent_from_attachment",
        )
        root_from_attachment = _matmul(frame_matrices[parent], parent_from_attachment)
        frame_id = attachment["frame_id"]
        frame_matrices[frame_id] = root_from_attachment
        attachment_results[frame_id] = {
            "parent_frame": parent,
            "semantic_role": attachment["semantic_role"],
            "root_from_frame": _pose_record(root_from_attachment),
        }
    constraint_results: list[dict[str, Any]] = []
    normalized_vector: list[float] = []
    component_order: list[str] = []
    for constraint in graph.get("constraints", []):
        components = _constraint_components(
            constraint,
            frame_matrices,
            articulation["pose"]["resolved_physical_joint_positions"],
        )
        normalized_vector.extend(component["value"] / component["tolerance"] for component in components)
        component_order.extend(f"{constraint['constraint_ref']}/{component['name']}" for component in components)
        maximum = max((component["normalized_abs"] for component in components), default=0.0)
        constraint_results.append({
            "constraint_id": constraint["constraint_id"],
            "constraint_ref": constraint["constraint_ref"],
            "type": constraint["type"],
            "role": constraint["role"],
            "satisfied": all(component["satisfied"] for component in components),
            "maximum_normalized_abs": _clean(maximum),
            "residual_l2_normalized": _clean(math.sqrt(sum((component["value"] / component["tolerance"]) ** 2 for component in components))),
            "components": components,
        })
    maximum = max((abs(value) for value in normalized_vector), default=0.0)
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "satisfied" if all(result["satisfied"] for result in constraint_results) else "violated",
        "constraint_graph_binding": {
            "constraint_graph_id": graph.get("constraint_graph_id"),
            "constraint_graph_sha256": graph.get("constraint_graph_sha256"),
            "source_binding": graph.get("source_binding"),
        },
        "pose": articulation["pose"],
        "reference_frame": articulation["reference_frame"],
        "attachments": attachment_results,
        "constraint_count": len(constraint_results),
        "residual_component_count": len(normalized_vector),
        "maximum_normalized_abs": _clean(maximum),
        "constraints": constraint_results,
        "_normalized_residual_vector": normalized_vector,
        "_normalized_residual_component_order": component_order,
        "meaning": "the spanning-tree articulation binding evaluated against every asserted supplemental constraint",
        "epistemic_scope": graph.get("epistemic_scope"),
    }


def _driver_defaults(grammar: dict[str, Any], supplied: dict[str, float]) -> dict[str, float]:
    variables = grammar["independent_variables"]
    return {
        name: float(supplied.get(name, record["default_value"]))
        for name, record in sorted(variables.items())
    }


def _step_bounds(record: dict[str, Any]) -> tuple[float | None, float | None]:
    domain = record["feasible_domain"]
    return (
        None if domain["minimum"] is None else float(domain["minimum"]),
        None if domain["maximum"] is None else float(domain["maximum"]),
    )


def _finite_difference_jacobian(
    graph: dict[str, Any],
    driver_positions: dict[str, float],
    variable_order: list[str],
    pose_name: str,
    step: float = 1e-6,
) -> tuple[list[list[float]], list[str]]:
    grammar = graph["articulation_grammar"]
    baseline = _evaluate_core(graph, driver_positions, pose_name)
    component_order = baseline["_normalized_residual_component_order"]
    columns: list[list[float]] = []
    for variable in variable_order:
        value = driver_positions[variable]
        lower, upper = _step_bounds(grammar["independent_variables"][variable])
        plus_ok = upper is None or value + step <= upper + EPSILON
        minus_ok = lower is None or value - step >= lower - EPSILON
        if plus_ok and minus_ok:
            plus = dict(driver_positions)
            minus = dict(driver_positions)
            plus[variable] = value + step
            minus[variable] = value - step
            right = _evaluate_core(graph, plus, pose_name)["_normalized_residual_vector"]
            left = _evaluate_core(graph, minus, pose_name)["_normalized_residual_vector"]
            column = [(a - b) / (2.0 * step) for a, b in zip(right, left)]
        elif plus_ok:
            plus = dict(driver_positions)
            plus[variable] = value + step
            right = _evaluate_core(graph, plus, pose_name)["_normalized_residual_vector"]
            base = baseline["_normalized_residual_vector"]
            column = [(a - b) / step for a, b in zip(right, base)]
        elif minus_ok:
            minus = dict(driver_positions)
            minus[variable] = value - step
            left = _evaluate_core(graph, minus, pose_name)["_normalized_residual_vector"]
            base = baseline["_normalized_residual_vector"]
            column = [(a - b) / step for a, b in zip(base, left)]
        else:
            raise ConstraintError(f"cannot perturb independent variable {variable!r} inside its feasible domain")
        columns.append(column)
    rows = [
        [columns[column][row] for column in range(len(columns))]
        for row in range(len(component_order))
    ]
    return rows, component_order


def _matrix_rank(matrix: list[list[float]], relative_tolerance: float = 1e-8) -> tuple[int, float]:
    if not matrix or not matrix[0]:
        return 0, 0.0
    work = [row[:] for row in matrix]
    row_count, column_count = len(work), len(work[0])
    scale = max(abs(value) for row in work for value in row)
    threshold = relative_tolerance * max(1.0, scale)
    rank = 0
    for column in range(column_count):
        pivot = max(range(rank, row_count), key=lambda row: abs(work[row][column]), default=rank)
        if abs(work[pivot][column]) <= threshold:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for remaining_column in range(column, column_count):
            work[rank][remaining_column] /= pivot_value
        for row in range(row_count):
            if row == rank:
                continue
            factor = work[row][column]
            if abs(factor) <= threshold:
                continue
            for remaining_column in range(column, column_count):
                work[row][remaining_column] -= factor * work[rank][remaining_column]
        rank += 1
        if rank == row_count:
            break
    return rank, threshold


def evaluate_constraint_graph(
    graph: dict[str, Any],
    supplied_positions: dict[str, float],
    pose_name: str = "supplied",
    analyze_local_mobility: bool = True,
) -> dict[str, Any]:
    result = _evaluate_core(graph, supplied_positions, pose_name)
    normalized_vector = result.pop("_normalized_residual_vector")
    component_order = result.pop("_normalized_residual_component_order")
    if analyze_local_mobility:
        grammar = graph["articulation_grammar"]
        drivers = _driver_defaults(grammar, supplied_positions)
        variable_order = sorted(drivers)
        jacobian, jacobian_component_order = _finite_difference_jacobian(
            graph, drivers, variable_order, pose_name
        )
        if jacobian_component_order != component_order:
            raise ConstraintError("constraint residual component order changed during local analysis")
        rank, threshold = _matrix_rank(jacobian)
        result["local_constraint_analysis"] = {
            "analysis_type": "pose_conditioned_normalized_residual_finite_difference_jacobian",
            "independent_variable_order": variable_order,
            "normalized_residual_component_order": component_order,
            "normalized_residual_vector": [_clean(value) for value in normalized_vector],
            "normalized_jacobian_rowmajor": [[_clean(value) for value in row] for row in jacobian],
            "finite_difference_step": 1e-6,
            "rank_relative_tolerance": 1e-8,
            "rank_absolute_threshold": _clean(threshold),
            "local_constraint_rank": rank,
            "tree_independent_variable_count": len(variable_order),
            "local_mobility_estimate": len(variable_order) - rank,
            "singularity_warning": (
                "rank is local and may change at singular configurations; it is not a global mechanism DOF proof"
            ),
        }
    return result


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-18:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def _clamp_driver(grammar: dict[str, Any], name: str, value: float) -> float:
    lower, upper = _step_bounds(grammar["independent_variables"][name])
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def solve_constraint_graph(
    graph: dict[str, Any],
    supplied_positions: dict[str, float],
    solve_for: Iterable[str],
    pose_name: str = "constraint-solution",
    max_iterations: int = 80,
    damping: float = 1e-8,
) -> dict[str, Any]:
    grammar = graph.get("articulation_grammar")
    if not isinstance(grammar, dict):
        raise ConstraintError("constraint graph has no embedded articulation grammar")
    variables = set(grammar["independent_variables"])
    selected = list(dict.fromkeys(solve_for))
    if not selected:
        raise ConstraintError("solve_for must name at least one independent variable")
    unknown = sorted(set(selected) - variables)
    if unknown:
        raise ConstraintError(f"solve_for contains non-independent variables: {unknown}")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ConstraintError("max_iterations must be a positive integer")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ConstraintError("damping must be finite and positive")
    positions = _driver_defaults(grammar, supplied_positions)
    seed = dict(positions)
    trace: list[dict[str, Any]] = []
    termination = "maximum_iterations"
    converged = False
    for iteration in range(max_iterations + 1):
        core = _evaluate_core(graph, positions, pose_name)
        residual = core["_normalized_residual_vector"]
        objective = 0.5 * sum(value * value for value in residual)
        maximum = max((abs(value) for value in residual), default=0.0)
        if maximum <= 1.0 + 1e-12:
            converged = True
            termination = "all_declared_tolerances_satisfied"
            trace.append({
                "iteration": iteration,
                "objective": _clean(objective),
                "maximum_normalized_abs": _clean(maximum),
                "accepted_scale": 0.0,
                "maximum_step": 0.0,
            })
            break
        jacobian, _ = _finite_difference_jacobian(graph, positions, selected, pose_name)
        dimension = len(selected)
        normal = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
        gradient = [0.0 for _ in range(dimension)]
        for row, residual_value in zip(jacobian, residual):
            for left in range(dimension):
                gradient[left] += row[left] * residual_value
                for right in range(dimension):
                    normal[left][right] += row[left] * row[right]
        for index in range(dimension):
            normal[index][index] += damping
        step = _solve_linear_system(normal, [-value for value in gradient])
        if step is None or any(not math.isfinite(value) for value in step):
            termination = "singular_normal_equations"
            break
        accepted_scale = 0.0
        accepted_positions: dict[str, float] | None = None
        for trial in range(16):
            scale = 0.5**trial
            candidate = dict(positions)
            for name, delta in zip(selected, step):
                candidate[name] = _clamp_driver(grammar, name, positions[name] + scale * delta)
            candidate_residual = _evaluate_core(graph, candidate, pose_name)["_normalized_residual_vector"]
            candidate_objective = 0.5 * sum(value * value for value in candidate_residual)
            if candidate_objective < objective:
                accepted_scale = scale
                accepted_positions = candidate
                break
        maximum_step = max((abs(accepted_scale * value) for value in step), default=0.0)
        trace.append({
            "iteration": iteration,
            "objective": _clean(objective),
            "maximum_normalized_abs": _clean(maximum),
            "accepted_scale": _clean(accepted_scale),
            "maximum_step": _clean(maximum_step),
        })
        if accepted_positions is None:
            termination = "line_search_failed"
            break
        positions = accepted_positions
        if maximum_step <= 1e-12:
            termination = "step_stagnation"
            break
    evaluation = evaluate_constraint_graph(graph, positions, pose_name, True)
    return {
        "schema_version": SOLUTION_SCHEMA,
        "status": "converged" if converged else "not_converged",
        "constraint_graph_binding": evaluation["constraint_graph_binding"],
        "solve_for": selected,
        "fixed_independent_variables": sorted(variables - set(selected)),
        "seed_independent_driver_positions": {name: _clean(value) for name, value in sorted(seed.items())},
        "solved_independent_driver_positions": {
            name: _clean(value) for name, value in sorted(positions.items())
        },
        "termination": termination,
        "iteration_count": max(0, len(trace) - (1 if converged else 0)),
        "trace": trace,
        "evaluation": evaluation,
        "epistemic_scope": (
            "one local numerical solution from the supplied seed with explicitly selected solve variables; "
            "not proof of uniqueness, global completeness, singularity avoidance, dynamics, stability, collision safety, or hardware feasibility"
        ),
    }


def verify_constraint_graph(
    grammar_path: Path,
    spec_path: Path,
    graph_path: Path,
    supplied_positions: dict[str, float] | None = None,
    pose_name: str = "verification",
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    try:
        actual = read_constraint_graph(graph_path)
        expected = build_constraint_graph(
            read_articulation_grammar(grammar_path),
            _sha256(grammar_path),
            _read_json(spec_path, "constraint spec"),
            _sha256(spec_path),
        )
        if _json_bytes(actual) != _json_bytes(expected):
            issues.append({"check": "exact_regeneration", "message": "constraint graph differs from exact regeneration"})
        execution = evaluate_constraint_graph(actual, supplied_positions or {}, pose_name, True)
    except (ConstraintError, ArticulationError) as error:
        issues.append({"check": "read_regenerate_execute", "message": str(error)})
        execution = None
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "graph_path": str(graph_path.resolve()),
        "graph_artifact_sha256": _sha256(graph_path),
        "grammar_artifact_sha256": _sha256(grammar_path),
        "constraint_spec_sha256": _sha256(spec_path),
        "exact_regeneration_match": not issues,
        "execution_status_at_verification_pose": None if execution is None else execution["status"],
        "local_constraint_analysis": None if execution is None else execution.get("local_constraint_analysis"),
        "issues": issues,
        "meaning": (
            "pass proves artifact binding, exact regeneration, standalone execution, and local-analysis reproducibility; "
            "the verification pose may legitimately violate the asserted mechanism constraints"
        ),
    }
