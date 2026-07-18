#!/usr/bin/env python3
"""Compile and query a proof-carrying concept language for robot structure."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable


CONCEPT_SCHEMA = "robot-spatial-concept-graph.v1"
QUERY_SCHEMA = "robot-spatial-concept-query.v1"
ANSWER_SCHEMA = "robot-spatial-concept-answer.v1"
VERIFICATION_SCHEMA = "robot-spatial-concept-verification.v1"
LANGUAGE_VERSION = "RSC-LANG/1"


class ConceptError(ValueError):
    """An invalid concept graph, query, binding, or context artifact."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ConceptError(f"cannot read {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConceptError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConceptError(f"{label} must contain one JSON object")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConceptError(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _artifact(
    canonical: dict[str, Any],
    artifact_name: str,
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    record = canonical.get("artifacts", {}).get(artifact_name)
    if record is None:
        if value is not None:
            raise ConceptError(f"{artifact_name} value supplied without canonical artifact binding")
        return None
    if not isinstance(record, dict) or not isinstance(value, dict):
        raise ConceptError(f"canonical {artifact_name} artifact and value must both be objects")
    return record


def _source_binding(
    canonical: dict[str, Any],
    articulation: dict[str, Any],
    constraint_graph: dict[str, Any] | None,
    configuration_atlas: dict[str, Any] | None,
) -> dict[str, Any]:
    source = canonical.get("source", {})
    robot = canonical.get("robot", {})
    articulation_record = _artifact(canonical, "articulation_grammar", articulation)
    assert articulation_record is not None
    binding: dict[str, Any] = {
        "robot_name": robot.get("name"),
        "root_link": robot.get("root_link"),
        "urdf_semantic_sha256": source.get("semantic_sha256"),
        "articulation_grammar": {
            "schema_version": articulation.get("schema_version"),
            "grammar_id": articulation.get("grammar_id"),
            "artifact_sha256": articulation_record.get("sha256"),
            "canonical_law_sha256": articulation.get("law_identity", {}).get("canonical_law_sha256"),
        },
        "constraint_graph": None,
        "configuration_atlas": None,
    }
    if not all(isinstance(binding[key], str) and binding[key] for key in ("robot_name", "root_link", "urdf_semantic_sha256")):
        raise ConceptError("canonical robot/source binding is incomplete")
    if constraint_graph is not None:
        record = _artifact(canonical, "constraint_graph", constraint_graph)
        assert record is not None
        binding["constraint_graph"] = {
            "schema_version": constraint_graph.get("schema_version"),
            "constraint_graph_id": constraint_graph.get("constraint_graph_id"),
            "artifact_sha256": record.get("sha256"),
            "semantic_sha256": constraint_graph.get("constraint_graph_sha256"),
        }
    elif canonical.get("artifacts", {}).get("constraint_graph") is not None:
        raise ConceptError("canonical binds a constraint graph but its value was not supplied")
    if configuration_atlas is not None:
        record = _artifact(canonical, "configuration_atlas", configuration_atlas)
        assert record is not None
        binding["configuration_atlas"] = {
            "schema_version": configuration_atlas.get("schema_version"),
            "configuration_atlas_id": configuration_atlas.get("configuration_atlas_id"),
            "artifact_sha256": record.get("sha256"),
            "semantic_sha256": configuration_atlas.get("configuration_atlas_sha256"),
        }
    elif canonical.get("artifacts", {}).get("configuration_atlas") is not None:
        raise ConceptError("canonical binds a configuration atlas but its value was not supplied")
    return binding


def _cnl_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _Builder:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.clauses: dict[str, dict[str, Any]] = {}

    def entity(self, entity_id: str, concept_types: Iterable[str], source_paths: Iterable[str]) -> None:
        if not isinstance(entity_id, str) or "/" not in entity_id:
            raise ConceptError(f"invalid typed entity ID {entity_id!r}")
        types = sorted(set(concept_types))
        paths = sorted(set(source_paths))
        existing = self.entities.get(entity_id)
        if existing is None:
            self.entities[entity_id] = {
                "entity_id": entity_id,
                "concept_types": types,
                "source_paths": paths,
            }
            return
        existing["concept_types"] = sorted(set(existing["concept_types"]) | set(types))
        existing["source_paths"] = sorted(set(existing["source_paths"]) | set(paths))

    def clause(
        self,
        predicate: str,
        subject: str,
        object_value: Any,
        *,
        modality: str,
        scope: str,
        exact: bool,
        source_type: str,
        source_paths: Iterable[str],
        rule: str,
        premise_clause_ids: Iterable[str] = (),
        cnl: str,
    ) -> str:
        if subject not in self.entities:
            raise ConceptError(f"clause subject is not a declared entity: {subject}")
        body = {
            "predicate": predicate,
            "subject": subject,
            "object": copy.deepcopy(object_value),
            "modality": modality,
            "scope": scope,
            "evidence": {
                "exact": exact,
                "source_type": source_type,
                "source_paths": sorted(set(source_paths)),
            },
            "proof": {
                "rule": rule,
                "premise_clause_ids": list(premise_clause_ids),
            },
            "cnl": " ".join(cnl.split()),
        }
        clause_id = f"concept_clause/{predicate}/{_sha256_bytes(_canonical_bytes(body))[:20]}"
        record = {"clause_id": clause_id, **body}
        existing = self.clauses.get(clause_id)
        if existing is not None and existing != record:
            raise ConceptError(f"concept clause ID collision at {clause_id}")
        self.clauses[clause_id] = record
        return clause_id


def _tree_projection(
    canonical: dict[str, Any],
    builder: _Builder,
    robot_entity: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    links = canonical.get("links")
    joints = canonical.get("joints")
    if not isinstance(links, dict) or not isinstance(joints, dict) or not links:
        raise ConceptError("canonical links and joints must be non-empty objects")
    root = canonical["robot"]["root_link"]
    if root not in links:
        raise ConceptError(f"canonical root link {root!r} is absent")

    children: dict[str, list[tuple[str, str]]] = {name: [] for name in links}
    parent: dict[str, tuple[str, str]] = {}
    edges: list[dict[str, Any]] = []
    edge_clauses: dict[tuple[str, str], str] = {}
    for link_name in sorted(links):
        builder.entity(f"link/{link_name}", ["link", "rigid_body"], [f"model.json#/links/{link_name}"])
        builder.entity(f"frame/{link_name}", ["frame", "link_frame"], [f"model.json#/frames/{link_name}"])
    for joint_name, joint in sorted(joints.items()):
        if not isinstance(joint, dict):
            raise ConceptError(f"joint {joint_name!r} must be an object")
        parent_link = joint.get("parent_link")
        child_link = joint.get("child_link")
        joint_type = joint.get("type")
        if parent_link not in links or child_link not in links:
            raise ConceptError(f"joint {joint_name!r} references an unknown link")
        if child_link in parent:
            raise ConceptError(f"link {child_link!r} has multiple incoming joints")
        children[parent_link].append((joint_name, child_link))
        parent[child_link] = (joint_name, parent_link)
        joint_entity = f"joint/{joint_name}"
        builder.entity(joint_entity, ["joint", f"joint_type/{joint_type}"], [f"model.json#/joints/{joint_name}"])
        pre_motion = joint.get("pre_motion_frame")
        if isinstance(pre_motion, str):
            builder.entity(f"frame/{pre_motion}", ["frame", "joint_pre_motion_frame"], [f"model.json#/frames/{pre_motion}"])
        object_value = {
            "parent_link": f"link/{parent_link}",
            "child_link": f"link/{child_link}",
            "joint_type": joint_type,
            "pre_motion_frame": f"frame/{pre_motion}" if isinstance(pre_motion, str) else None,
        }
        clause_id = builder.clause(
            "connects_parent_to_child",
            joint_entity,
            object_value,
            modality="declared_exact",
            scope="pose_independent_tree_law",
            exact=True,
            source_type="validated_urdf_tree",
            source_paths=[f"model.json#/joints/{joint_name}"],
            rule="urdf_joint_edge",
            cnl=(
                f"JOINT {joint_entity} CONNECTS parent link/{parent_link} TO child "
                f"link/{child_link} AS {joint_type}; direction is parent-to-child."
            ),
        )
        edge_clauses[(parent_link, child_link)] = clause_id
        edges.append({
            "joint": joint_entity,
            "parent_link": f"link/{parent_link}",
            "child_link": f"link/{child_link}",
            "joint_type": joint_type,
            "supporting_clause_id": clause_id,
        })
    for records in children.values():
        records.sort()
    if set(parent) != set(links) - {root}:
        raise ConceptError("canonical topology is not one complete rooted tree")

    root_clause = builder.clause(
        "has_root_link",
        robot_entity,
        f"link/{root}",
        modality="declared_exact",
        scope="pose_independent_tree_law",
        exact=True,
        source_type="validated_urdf_tree",
        source_paths=["model.json#/robot/root_link"],
        rule="validated_unique_tree_root",
        cnl=f"ROBOT {robot_entity} HAS ROOT link/{root}.",
    )

    branch_points = sorted(name for name, records in children.items() if len(records) > 1)
    leaves = sorted(name for name, records in children.items() if not records)
    for link_name in branch_points:
        child_entities = [f"link/{child}" for _, child in children[link_name]]
        builder.clause(
            "is_structural_branch_point",
            f"link/{link_name}",
            child_entities,
            modality="derived_exact",
            scope="pose_independent_tree_topology",
            exact=True,
            source_type="tree_graph_derivation",
            source_paths=["model.json#/joints"],
            rule="out_degree_greater_than_one",
            premise_clause_ids=[edge_clauses[(link_name, child)] for _, child in children[link_name]],
            cnl=f"LINK link/{link_name} IS A STRUCTURAL BRANCH POINT with children {_cnl_value(child_entities)}.",
        )
    for link_name in leaves:
        builder.clause(
            "is_structural_leaf",
            f"link/{link_name}",
            [],
            modality="derived_exact",
            scope="pose_independent_tree_topology",
            exact=True,
            source_type="tree_graph_derivation",
            source_paths=["model.json#/joints"],
            rule="out_degree_equals_zero_over_complete_link_set",
            cnl=(
                f"LINK link/{link_name} IS A STRUCTURAL LEAF in the URDF tree; "
                "this does not by itself assert end-effector semantics."
            ),
        )

    for ancestor in sorted(links):
        queue: deque[tuple[str, list[str], list[str]]] = deque(
            (child, [joint], [edge_clauses[(ancestor, child)]])
            for joint, child in children[ancestor]
        )
        while queue:
            descendant, joint_path, premises = queue.popleft()
            builder.clause(
                "is_descendant_of",
                f"link/{descendant}",
                {
                    "ancestor": f"link/{ancestor}",
                    "ordered_joint_path": [f"joint/{joint}" for joint in joint_path],
                },
                modality="derived_exact",
                scope="pose_independent_tree_topology",
                exact=True,
                source_type="tree_transitive_closure",
                source_paths=["model.json#/joints"],
                rule="directed_parent_child_transitive_closure",
                premise_clause_ids=premises,
                cnl=(
                    f"LINK link/{descendant} IS DESCENDANT OF link/{ancestor} THROUGH joints "
                    f"{_cnl_value([f'joint/{joint}' for joint in joint_path])}."
                ),
            )
            for next_joint, next_child in children[descendant]:
                queue.append((
                    next_child,
                    [*joint_path, next_joint],
                    [*premises, edge_clauses[(descendant, next_child)]],
                ))

    boundaries = {root, *branch_points, *leaves}
    segments: list[dict[str, Any]] = []
    for start in sorted(boundaries):
        for first_joint, first_child in children[start]:
            segment_links = [start, first_child]
            segment_joints = [first_joint]
            current = first_child
            while current not in boundaries and len(children[current]) == 1:
                next_joint, next_child = children[current][0]
                segment_joints.append(next_joint)
                segment_links.append(next_child)
                current = next_child
            segment_index = len(segments)
            entity_id = f"serial_segment/{canonical['robot']['name']}/{segment_index:04d}"
            builder.entity(entity_id, ["maximal_serial_segment", "topology_abstraction"], ["model.json#/joints"])
            record = {
                "segment_entity": entity_id,
                "start_link": f"link/{start}",
                "end_link": f"link/{current}",
                "ordered_links": [f"link/{name}" for name in segment_links],
                "ordered_joints": [f"joint/{name}" for name in segment_joints],
                "start_boundary": "root" if start == root else "branch_point",
                "end_boundary": "leaf" if current in leaves else "branch_point",
            }
            clause_id = builder.clause(
                "is_maximal_serial_segment",
                entity_id,
                record,
                modality="derived_exact",
                scope="pose_independent_tree_topology",
                exact=True,
                source_type="tree_graph_derivation",
                source_paths=["model.json#/joints"],
                rule="maximal_path_between_root_branch_or_leaf_boundaries",
                premise_clause_ids=[
                    edge_clauses[(segment_links[index], segment_links[index + 1])]
                    for index in range(len(segment_joints))
                ],
                cnl=(
                    f"SERIAL SEGMENT {entity_id} RUNS FROM link/{start} TO link/{current} "
                    f"VIA joints {_cnl_value(record['ordered_joints'])}."
                ),
            )
            record["supporting_clause_id"] = clause_id
            segments.append(record)

    return {
        "root_link": f"link/{root}",
        "root_supporting_clause_id": root_clause,
        "links": [f"link/{name}" for name in sorted(links)],
        "joints": [f"joint/{name}" for name in sorted(joints)],
        "edges": edges,
        "branch_points": [f"link/{name}" for name in branch_points],
        "structural_leaves": [f"link/{name}" for name in leaves],
        "maximal_serial_segments": segments,
        "coverage": {
            "complete_link_set": True,
            "complete_joint_set": True,
            "complete_parent_child_tree": True,
            "structural_leaf_is_not_end_effector_assertion": True,
        },
    }, edge_clauses


def _articulation_projection(
    articulation: dict[str, Any],
    builder: _Builder,
) -> dict[str, Any]:
    grammar_id = articulation.get("grammar_id")
    variables = articulation.get("independent_variables")
    rules = articulation.get("joint_position_rules")
    operators = articulation.get("joint_operators")
    derivations = articulation.get("frame_derivations")
    if not isinstance(grammar_id, str) or not all(isinstance(value, dict) for value in (variables, rules, operators, derivations)):
        raise ConceptError("articulation grammar is malformed")
    drivers: list[dict[str, Any]] = []
    for driver, variable in sorted(variables.items()):
        entity_id = f"articulation_variable/{grammar_id}/{driver}"
        builder.entity(entity_id, ["independent_driver", "configuration_coordinate"], [f"articulation-grammar.json#/independent_variables/{driver}"])
        domain = {
            "driver": driver,
            "unit": variable.get("unit"),
            "joint_type": variable.get("joint_type"),
            "default_value": variable.get("default_value"),
            "feasible_domain": variable.get("feasible_domain"),
        }
        domain_clause = builder.clause(
            "has_feasible_driver_domain",
            entity_id,
            domain,
            modality="derived_exact",
            scope="pose_independent_articulation_law",
            exact=True,
            source_type="executable_articulation_grammar",
            source_paths=[f"articulation-grammar.json#/independent_variables/{driver}"],
            rule="mimic_constrained_driver_domain",
            cnl=f"DRIVER {entity_id} HAS FEASIBLE DOMAIN {_cnl_value(domain)}.",
        )
        driven_joint_clauses: list[str] = []
        for physical_joint in variable.get("physical_joints_driven", []):
            joint_entity = f"joint/{physical_joint}"
            builder.entity(joint_entity, ["joint"], [f"articulation-grammar.json#/joint_position_rules/{physical_joint}"])
            rule = rules.get(physical_joint, {})
            driven_joint_clauses.append(builder.clause(
                "drives_physical_joint",
                entity_id,
                {
                    "physical_joint": joint_entity,
                    "equation_cnl": rule.get("equation_cnl"),
                    "multiplier": rule.get("multiplier"),
                    "offset": rule.get("offset"),
                    "unit": rule.get("unit"),
                },
                modality="derived_exact",
                scope="pose_independent_articulation_law",
                exact=True,
                source_type="executable_articulation_grammar",
                source_paths=[f"articulation-grammar.json#/joint_position_rules/{physical_joint}"],
                rule="physical_joint_position_dependency",
                premise_clause_ids=[domain_clause],
                cnl=(
                    f"DRIVER {entity_id} DRIVES {joint_entity} BY "
                    f"{_cnl_value(rule.get('equation_cnl'))}."
                ),
            ))
        affected_frames: list[str] = []
        affected_frame_clauses: list[str] = []
        for frame_name, derivation in sorted(derivations.items()):
            dependencies = derivation.get("independent_driver_dependencies", [])
            if driver not in dependencies:
                continue
            frame_entity = f"frame/{frame_name}"
            builder.entity(frame_entity, ["frame"], [f"articulation-grammar.json#/frame_derivations/{frame_name}"])
            affected_frames.append(frame_entity)
            affected_frame_clauses.append(builder.clause(
                "can_change_pose_of_frame_relative_to_root",
                entity_id,
                {
                    "frame": frame_entity,
                    "other_independent_drivers_held_fixed": True,
                    "root_frame": articulation["coordinate_contract"]["root_frame"],
                },
                modality="derived_exact",
                scope="pose_independent_structural_causality",
                exact=True,
                source_type="executable_articulation_dependency_graph",
                source_paths=[f"articulation-grammar.json#/frame_derivations/{frame_name}/independent_driver_dependencies"],
                rule="driver_occurs_in_ordered_frame_composition",
                premise_clause_ids=driven_joint_clauses,
                cnl=(
                    f"DRIVER {entity_id} CAN CHANGE POSE OF {frame_entity} RELATIVE TO ROOT when "
                    "other independent drivers are held fixed."
                ),
            ))
        drivers.append({
            "driver_entity": entity_id,
            "driver_name": driver,
            "domain": domain,
            "physical_joints_driven": [f"joint/{name}" for name in variable.get("physical_joints_driven", [])],
            "affected_frames": affected_frames,
            "domain_clause_id": domain_clause,
            "drives_clause_ids": driven_joint_clauses,
            "affects_clause_ids": affected_frame_clauses,
        })

    frame_laws: list[dict[str, Any]] = []
    for frame_name, derivation in sorted(derivations.items()):
        frame_entity = f"frame/{frame_name}"
        builder.entity(frame_entity, ["frame", f"frame_semantics/{derivation.get('semantic_type')}"], [f"articulation-grammar.json#/frame_derivations/{frame_name}"])
        value = {
            "root_frame": articulation["coordinate_contract"]["root_frame"],
            "expression_cnl": derivation.get("expression_cnl"),
            "ordered_operator_refs": derivation.get("ordered_operator_refs"),
            "independent_driver_dependencies": [
                f"articulation_variable/{grammar_id}/{name}"
                for name in derivation.get("independent_driver_dependencies", [])
            ],
        }
        clause_id = builder.clause(
            "has_ordered_pose_composition_law",
            frame_entity,
            value,
            modality="derived_exact",
            scope="pose_independent_articulation_law",
            exact=True,
            source_type="standalone_executable_articulation_ast",
            source_paths=[f"articulation-grammar.json#/frame_derivations/{frame_name}"],
            rule="ordered_root_to_frame_operator_composition",
            cnl=f"FRAME {frame_entity} HAS POSE LAW {_cnl_value(derivation.get('expression_cnl'))}.",
        )
        frame_laws.append({"frame": frame_entity, **value, "supporting_clause_id": clause_id})

    operator_records: list[dict[str, Any]] = []
    for joint_name, operator in sorted(operators.items()):
        joint_entity = f"joint/{joint_name}"
        builder.entity(joint_entity, ["joint"], [f"articulation-grammar.json#/joint_operators/{joint_name}"])
        value = {
            "operator_ref": operator.get("operator_id"),
            "joint_type": operator.get("joint_type"),
            "parent_link": f"link/{operator.get('parent_link')}",
            "child_link": f"link/{operator.get('child_link')}",
            "motion_operator": operator.get("motion_operator"),
            "composition_rule": operator.get("composition_rule"),
        }
        clause_id = builder.clause(
            "has_typed_motion_operator",
            joint_entity,
            value,
            modality="derived_exact",
            scope="pose_independent_articulation_law",
            exact=True,
            source_type="standalone_executable_articulation_ast",
            source_paths=[f"articulation-grammar.json#/joint_operators/{joint_name}"],
            rule="normalized_joint_operator",
            cnl=f"JOINT {joint_entity} HAS MOTION OPERATOR {_cnl_value(value)}.",
        )
        operator_records.append({"joint": joint_entity, **value, "supporting_clause_id": clause_id})
    return {
        "grammar_id": grammar_id,
        "root_frame": articulation["coordinate_contract"]["root_frame"],
        "drivers": drivers,
        "joint_operators": operator_records,
        "frame_laws": frame_laws,
        "coverage": {
            "all_independent_drivers": True,
            "all_physical_joint_operators": True,
            "all_supported_frame_laws": articulation.get("coverage", {}).get("all_supported_frames_have_derivations") is True,
            "negative_driver_frame_effect_answerable_from_complete_dependency_sets": articulation.get("coverage", {}).get("all_supported_frames_have_derivations") is True,
        },
    }


def _semantic_projection(canonical: dict[str, Any], builder: _Builder) -> dict[str, Any]:
    semantics = canonical.get("semantics", {})
    frames = semantics.get("frames", {}) if isinstance(semantics, dict) else {}
    roles: list[dict[str, Any]] = []
    if isinstance(frames, dict):
        for frame_name, annotation in sorted(frames.items()):
            if not isinstance(annotation, dict):
                continue
            entity = f"frame/{frame_name}"
            builder.entity(entity, ["frame"], [f"model.json#/semantics/frames/{frame_name}"])
            for role in sorted(annotation.get("roles", [])):
                clause_id = builder.clause(
                    "has_asserted_semantic_role",
                    entity,
                    {"role": role, "meaning": annotation.get("meaning")},
                    modality="project_asserted",
                    scope="project_semantics",
                    exact=False,
                    source_type="project_semantic_annotation",
                    source_paths=[f"model.json#/semantics/frames/{frame_name}"],
                    rule="copy_explicit_project_annotation_without_name_inference",
                    cnl=f"PROJECT ASSERTS FRAME {entity} HAS ROLE {_cnl_value(role)}.",
                )
                roles.append({"frame": entity, "role": role, "meaning": annotation.get("meaning"), "supporting_clause_id": clause_id})
    return {
        "status": semantics.get("status") if isinstance(semantics, dict) else "not_provided",
        "asserted_frame_roles": roles,
        "name_based_role_inference_permitted": False,
    }


def _drivers_for_constraint_value(
    value: Any,
    articulation: dict[str, Any],
    attachment_parents: dict[str, str],
    key: str | None = None,
) -> set[str]:
    variables = articulation["independent_variables"]
    rules = articulation["joint_position_rules"]
    derivations = articulation["frame_derivations"]
    drivers: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in {"tolerances", "parent_from_attachment", "axis_xyz_in_a", "axis_xyz_in_b", "point_xyz_in_a", "point_xyz_in_b"}:
                continue
            if child_key in {"coefficients", "terms"} and isinstance(child, dict):
                for name in child:
                    if name in variables:
                        drivers.add(name)
                    elif name in rules:
                        driver = rules[name].get("driver_joint")
                        if isinstance(driver, str):
                            drivers.add(driver)
            drivers.update(_drivers_for_constraint_value(child, articulation, attachment_parents, child_key))
    elif isinstance(value, list):
        for child in value:
            drivers.update(_drivers_for_constraint_value(child, articulation, attachment_parents, key))
    elif isinstance(value, str):
        if key in {"frame", "frame_a", "frame_b", "parent_frame"}:
            frame = attachment_parents.get(value, value)
            derivation = derivations.get(frame)
            if isinstance(derivation, dict):
                drivers.update(derivation.get("independent_driver_dependencies", []))
        elif key in {"joint", "joint_a", "joint_b", "driver", "coordinate"}:
            if value in variables:
                drivers.add(value)
            elif value in rules:
                driver = rules[value].get("driver_joint")
                if isinstance(driver, str):
                    drivers.add(driver)
    return drivers


def _mechanism_projection(
    constraint_graph: dict[str, Any] | None,
    articulation: dict[str, Any],
    builder: _Builder,
    robot_entity: str,
) -> dict[str, Any]:
    if constraint_graph is None:
        return {
            "status": "not_provided",
            "tree_is_complete_mechanism": "not_established_for_closed_or_coupled_mechanisms",
            "constraints": [],
            "attachments": [],
        }
    graph_id = constraint_graph.get("constraint_graph_id")
    if not isinstance(graph_id, str):
        raise ConceptError("constraint graph ID is missing")
    graph_entity = f"constraint_graph/{graph_id}"
    builder.entity(graph_entity, ["constraint_graph", "mechanism_relation_layer"], ["constraint-graph.json#/"])
    tree_parameterization = bool(
        constraint_graph.get("structural_graph", {}).get("tree_is_parameterization_not_complete_mechanism")
    )
    parameterization_clause = builder.clause(
        "uses_tree_as_coordinate_parameterization",
        graph_entity,
        {
            "robot": robot_entity,
            "tree_is_parameterization_not_complete_mechanism": tree_parameterization,
        },
        modality="supplemental_asserted_structure",
        scope="pose_independent_mechanism_model",
        exact=False,
        source_type="digest_bound_supplemental_constraint_spec",
        source_paths=["constraint-graph.json#/structural_graph"],
        rule="explicit_supplemental_mechanism_layer",
        cnl=(
            f"MECHANISM {graph_entity} USES THE URDF TREE AS A COORDINATE PARAMETERIZATION; "
            f"tree_not_complete={_cnl_value(tree_parameterization)}."
        ),
    )
    attachment_parents: dict[str, str] = {}
    attachments: list[dict[str, Any]] = []
    for attachment in constraint_graph.get("attachments", []):
        attachment_id = attachment.get("attachment_id")
        frame_id = attachment.get("frame_id")
        parent_frame = attachment.get("parent_frame")
        entity = f"attachment/{graph_id}/{attachment_id}"
        builder.entity(entity, ["rigid_attachment_frame", "mechanism_anchor"], [f"constraint-graph.json#/attachments/{attachment_id}"])
        if isinstance(frame_id, str) and isinstance(parent_frame, str):
            attachment_parents[frame_id] = parent_frame
        value = {
            "frame_id": frame_id,
            "parent_frame": f"frame/{parent_frame}",
            "semantic_role": attachment.get("semantic_role"),
            "parent_from_attachment": attachment.get("parent_from_attachment"),
        }
        clause_id = builder.clause(
            "declares_rigid_attachment",
            entity,
            value,
            modality="supplemental_asserted_structure",
            scope="pose_independent_mechanism_model",
            exact=False,
            source_type="digest_bound_supplemental_constraint_spec",
            source_paths=[f"constraint-graph.json#/attachments/{attachment_id}"],
            rule="copy_declared_rigid_attachment",
            premise_clause_ids=[parameterization_clause],
            cnl=f"MECHANISM DECLARES ATTACHMENT {entity} AS {_cnl_value(value)}.",
        )
        attachments.append({"attachment_entity": entity, **value, "supporting_clause_id": clause_id})

    constraints: list[dict[str, Any]] = []
    for constraint in constraint_graph.get("constraints", []):
        constraint_id = constraint.get("constraint_id")
        entity = f"constraint/{graph_id}/{constraint_id}"
        builder.entity(entity, ["mechanism_constraint", f"constraint_type/{constraint.get('type')}"], [f"constraint-graph.json#/constraints/{constraint_id}"])
        dependencies = sorted(_drivers_for_constraint_value(constraint, articulation, attachment_parents))
        value = {
            "constraint_id": constraint_id,
            "type": constraint.get("type"),
            "role": constraint.get("role"),
            "relation": constraint,
            "driver_dependencies": [f"articulation_variable/{articulation['grammar_id']}/{name}" for name in dependencies],
        }
        relation_clause = builder.clause(
            "requires_mechanism_relation",
            entity,
            value,
            modality="supplemental_asserted_relation",
            scope="pose_independent_mechanism_model",
            exact=False,
            source_type="digest_bound_supplemental_constraint_spec",
            source_paths=[f"constraint-graph.json#/constraints/{constraint_id}"],
            rule="copy_typed_asserted_mechanism_constraint",
            premise_clause_ids=[parameterization_clause],
            cnl=(
                f"MECHANISM REQUIRES CONSTRAINT {entity} TYPE {_cnl_value(constraint.get('type'))} "
                f"ROLE {_cnl_value(constraint.get('role'))}."
            ),
        )
        dependency_clauses: list[str] = []
        for driver in dependencies:
            driver_entity = f"articulation_variable/{articulation['grammar_id']}/{driver}"
            dependency_clauses.append(builder.clause(
                "constraint_depends_on_driver",
                entity,
                driver_entity,
                modality="derived_exact_from_asserted_relation",
                scope="pose_independent_mechanism_model",
                exact=True,
                source_type="constraint_frame_and_coordinate_dependency_analysis",
                source_paths=[
                    f"constraint-graph.json#/constraints/{constraint_id}",
                    "articulation-grammar.json#/frame_derivations",
                ],
                rule="union_frame_derivation_and_coordinate_dependencies",
                premise_clause_ids=[relation_clause],
                cnl=f"CONSTRAINT {entity} DEPENDS ON DRIVER {driver_entity}.",
            ))
        constraints.append({
            "constraint_entity": entity,
            **value,
            "relation_clause_id": relation_clause,
            "dependency_clause_ids": dependency_clauses,
        })
    return {
        "status": "supplemental_mechanism_relations_bound",
        "constraint_graph_entity": graph_entity,
        "tree_is_coordinate_parameterization": tree_parameterization,
        "parameterization_clause_id": parameterization_clause,
        "attachments": attachments,
        "constraints": constraints,
        "asserted_relations_are_physical_truth": False,
    }


def _configuration_projection(
    atlas: dict[str, Any] | None,
    builder: _Builder,
) -> dict[str, Any]:
    if atlas is None:
        return {
            "status": "not_provided",
            "charts": [],
            "global_branch_topology_certified": False,
        }
    atlas_id = atlas.get("configuration_atlas_id")
    if not isinstance(atlas_id, str):
        raise ConceptError("configuration atlas ID is missing")
    atlas_entity = f"configuration_atlas/{atlas_id}"
    builder.entity(atlas_entity, ["finite_configuration_witness_atlas"], ["configuration-atlas.json#/"])
    status_clause = builder.clause(
        "has_finite_declared_sampling_status",
        atlas_entity,
        {"status": atlas.get("status"), "coverage": atlas.get("coverage")},
        modality="finite_computed_evidence",
        scope="finite_declared_configuration_sampling",
        exact=True,
        source_type="reexecutable_configuration_atlas",
        source_paths=["configuration-atlas.json#/status", "configuration-atlas.json#/coverage"],
        rule="copy_verified_finite_sampling_coverage",
        cnl=(
            f"ATLAS {atlas_entity} HAS FINITE DECLARED STATUS {_cnl_value(atlas.get('status'))}; "
            "this is not global topology certification."
        ),
    )
    charts: list[dict[str, Any]] = []
    for chart in atlas.get("charts", []):
        chart_id = chart.get("chart_id")
        chart_entity = f"configuration_chart/{atlas_id}/{chart_id}"
        builder.entity(chart_entity, ["one_parameter_configuration_chart"], [f"configuration-atlas.json#/charts/{chart_id}"])
        chart_value = {
            "parameter_driver": chart.get("parameter_driver"),
            "parameter_values": chart.get("parameter_values"),
            "solve_for": chart.get("solve_for"),
            "driver_scales": chart.get("driver_scales"),
            "minimum_solutions_per_sample": chart.get("minimum_solutions_per_sample"),
            "coverage": chart.get("coverage"),
        }
        chart_clause = builder.clause(
            "declares_finite_configuration_chart",
            chart_entity,
            chart_value,
            modality="finite_computed_evidence",
            scope="finite_declared_configuration_sampling",
            exact=True,
            source_type="reexecutable_configuration_atlas",
            source_paths=[f"configuration-atlas.json#/charts/{chart_id}"],
            rule="copy_explicit_chart_contract_and_outcomes",
            premise_clause_ids=[status_clause],
            cnl=f"CONFIGURATION CHART {chart_entity} DECLARES {_cnl_value(chart_value)}.",
        )
        nodes: list[dict[str, Any]] = []
        for sample in chart.get("samples", []):
            sample_index = int(sample.get("sample_index"))
            for solution_index, node in enumerate(sample.get("solutions", [])):
                node_entity = f"configuration_node/{atlas_id}/{chart_id}/{sample_index:04d}/{solution_index:04d}"
                builder.entity(node_entity, ["configuration_witness", "constraint_satisfying_binding"], [
                    f"configuration-atlas.json#/charts/{chart_id}/samples/{sample_index}/solutions/{solution_index}"
                ])
                node_value = {
                    "parameter_driver": sample.get("parameter_driver"),
                    "parameter_value": sample.get("parameter_value"),
                    "independent_driver_positions": node.get("independent_driver_positions"),
                    "constraint_status": node.get("constraint_status"),
                    "maximum_normalized_abs": node.get("maximum_normalized_abs"),
                    "full_constraint_rank": node.get("full_constraint_jacobian", {}).get("numerical_rank"),
                    "chart_passive_rank": node.get("chart_passive_jacobian", {}).get("numerical_rank"),
                    "singularity_witness": node.get("singularity_witness"),
                }
                node_clause = builder.clause(
                    "is_executable_satisfying_configuration_witness",
                    node_entity,
                    node_value,
                    modality="finite_computed_evidence",
                    scope="finite_declared_configuration_sampling",
                    exact=True,
                    source_type="reexecuted_constraint_solution_node",
                    source_paths=[
                        f"configuration-atlas.json#/charts/{chart_id}/samples/{sample_index}/solutions/{solution_index}"
                    ],
                    rule="local_solve_then_exact_constraint_reevaluation",
                    premise_clause_ids=[chart_clause],
                    cnl=f"CONFIGURATION NODE {node_entity} IS A FINITE SATISFYING WITNESS {_cnl_value(node_value)}.",
                )
                nodes.append({"node_entity": node_entity, **node_value, "supporting_clause_id": node_clause})
        components: list[dict[str, Any]] = []
        for component_index, component in enumerate(chart.get("witness_components", [])):
            component_entity = f"configuration_component/{atlas_id}/{chart_id}/{component_index:04d}"
            builder.entity(component_entity, ["finite_proximity_component"], [
                f"configuration-atlas.json#/charts/{chart_id}/witness_components/{component_index}"
            ])
            raw_nodes = component.get("node_ids", [])
            normalized_nodes = [
                f"configuration_node/{atlas_id}/{node_id.removeprefix('configuration_node/')}"
                for node_id in raw_nodes
            ]
            component_value = {
                "node_entities": normalized_nodes,
                "meaning": "connected component of only the finite declared proximity graph",
            }
            component_clause = builder.clause(
                "groups_finite_proximity_witnesses",
                component_entity,
                component_value,
                modality="finite_computed_evidence",
                scope="finite_declared_configuration_sampling",
                exact=True,
                source_type="finite_configuration_proximity_graph",
                source_paths=[f"configuration-atlas.json#/charts/{chart_id}/witness_components/{component_index}"],
                rule="connected_components_of_stored_proximity_edges",
                premise_clause_ids=[chart_clause],
                cnl=(
                    f"FINITE PROXIMITY COMPONENT {component_entity} GROUPS NODES "
                    f"{_cnl_value(normalized_nodes)}; it is not a certified global branch."
                ),
            )
            components.append({
                "component_entity": component_entity,
                **component_value,
                "supporting_clause_id": component_clause,
            })
        charts.append({
            "chart_entity": chart_entity,
            **chart_value,
            "chart_clause_id": chart_clause,
            "nodes": nodes,
            "finite_proximity_components": components,
        })
    return {
        "status": atlas.get("status"),
        "atlas_entity": atlas_entity,
        "status_clause_id": status_clause,
        "charts": charts,
        "global_branch_topology_certified": False,
        "certified_singularity": False,
        "finite_proximity_component_is_global_branch": False,
    }


def _entity_references(value: Any, known_entities: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        if value in known_entities:
            found.add(value)
    elif isinstance(value, dict):
        for child in value.values():
            found.update(_entity_references(child, known_entities))
    elif isinstance(value, list):
        for child in value:
            found.update(_entity_references(child, known_entities))
    return found


def build_concept_graph(
    canonical: dict[str, Any],
    articulation: dict[str, Any],
    constraint_graph: dict[str, Any] | None = None,
    configuration_atlas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binding = _source_binding(canonical, articulation, constraint_graph, configuration_atlas)
    builder = _Builder()
    robot_entity = f"robot/{canonical['robot']['name']}"
    builder.entity(robot_entity, ["robot", "articulated_system"], ["model.json#/robot"])
    topology, _edge_clauses = _tree_projection(canonical, builder, robot_entity)
    articulation_projection = _articulation_projection(articulation, builder)
    semantics = _semantic_projection(canonical, builder)
    mechanism = _mechanism_projection(constraint_graph, articulation, builder, robot_entity)
    configuration = _configuration_projection(configuration_atlas, builder)

    entities = [builder.entities[key] for key in sorted(builder.entities)]
    clauses = [builder.clauses[key] for key in sorted(builder.clauses)]
    clause_ids = set(builder.clauses)
    known_entities = set(builder.entities)
    for clause in clauses:
        missing_premises = set(clause["proof"]["premise_clause_ids"]) - clause_ids
        if missing_premises:
            raise ConceptError(f"clause {clause['clause_id']} has missing premises {sorted(missing_premises)}")

    by_predicate: dict[str, list[str]] = {}
    by_subject: dict[str, list[str]] = {}
    by_entity: dict[str, list[str]] = {entity: [] for entity in known_entities}
    for clause in clauses:
        clause_id = clause["clause_id"]
        by_predicate.setdefault(clause["predicate"], []).append(clause_id)
        by_subject.setdefault(clause["subject"], []).append(clause_id)
        references = {clause["subject"]} | _entity_references(clause["object"], known_entities)
        for entity in references:
            by_entity[entity].append(clause_id)

    concept_graph_id = (
        f"concept_graph/{canonical['robot']['name']}/"
        f"{_sha256_bytes(_canonical_bytes(binding))[:16]}"
    )
    body = {
        "schema_version": CONCEPT_SCHEMA,
        "concept_graph_id": concept_graph_id,
        "source_binding": binding,
        "language_contract": {
            "language_version": LANGUAGE_VERSION,
            "clause_form": "typed subject + predicate + typed or literal object + modality + scope + evidence + proof + controlled natural language",
            "closed_world_domains": [
                "validated URDF link/joint tree",
                "articulation grammar independent drivers, physical joint operators, and supported frame derivations",
            ],
            "open_world_domains": [
                "semantic roles not explicitly annotated",
                "supplemental mechanism relations not declared",
                "physical construction, calibration, environment, runtime, hardware, and safety",
                "global configuration branches, topology, reachability, and certified singularities",
            ],
            "negative_answer_rule": "return a negative only when the queried relation belongs to an explicitly complete closed-world projection; otherwise return unknown",
        },
        "ontology_contract": {
            "entity_identity_is_typed": True,
            "link_frame_and_link_body_are_distinct": True,
            "structural_leaf_is_not_end_effector": True,
            "asserted_semantics_are_not_inferred_geometry": True,
            "tree_parameterization_is_not_complete_closed_mechanism": True,
            "finite_witness_component_is_not_certified_global_branch": True,
            "clause_modalities": [
                "declared_exact",
                "derived_exact",
                "project_asserted",
                "supplemental_asserted_structure",
                "supplemental_asserted_relation",
                "derived_exact_from_asserted_relation",
                "finite_computed_evidence",
            ],
        },
        "entities": entities,
        "clauses": clauses,
        "indexes": {
            "by_predicate": {key: sorted(value) for key, value in sorted(by_predicate.items())},
            "by_subject": {key: sorted(value) for key, value in sorted(by_subject.items())},
            "by_entity": {key: sorted(value) for key, value in sorted(by_entity.items())},
        },
        "projections": {
            "topology": topology,
            "articulation": articulation_projection,
            "project_semantics": semantics,
            "mechanism": mechanism,
            "configuration": configuration,
        },
        "query_contract": {
            "schema_version": QUERY_SCHEMA,
            "intents": {
                "structural_summary": {},
                "trace_kinematic_path": {"from_link": "typed or unique bare link", "to_link": "typed or unique bare link"},
                "explain_driver_effect": {"driver": "typed or unique bare driver", "target_frame": "optional typed or unique bare frame"},
                "explain_frame_pose_law": {"frame": "typed or unique bare frame"},
                "explain_constraint": {"constraint": "typed or unique bare constraint"},
                "compare_configuration_nodes": {"node_a": "typed node", "node_b": "typed node"},
                "describe_entity": {"entity": "exact typed or unique bare entity"},
            },
        },
        "coverage": {
            "entity_count": len(entities),
            "clause_count": len(clauses),
            "predicate_count": len(by_predicate),
            "tree_link_count": len(topology["links"]),
            "tree_joint_count": len(topology["joints"]),
            "independent_driver_count": len(articulation_projection["drivers"]),
            "constraint_count": len(mechanism["constraints"]),
            "configuration_chart_count": len(configuration["charts"]),
            "exact_regeneration_supported": True,
        },
        "epistemic_scope": (
            "proof-carrying symbolic abstraction of the exact bound canonical tree and articulation law, explicit project assertions, "
            "explicit supplemental mechanism relations, and optional finite configuration witnesses; it supports structural queries "
            "and exact negative answers only over declared complete closed-world projections. It is not an ontology of inferred function, "
            "a physical observation, a dynamics/controller model, a global configuration-space proof, or a safety certificate."
        ),
    }
    body["concept_graph_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def render_concept_language(graph: dict[str, Any]) -> str:
    lines = [
        f"LANGUAGE {LANGUAGE_VERSION}",
        f"CONCEPT_GRAPH {graph['concept_graph_id']}",
        f"CONCEPT_GRAPH_SHA256 {graph['concept_graph_sha256']}",
        f"SOURCE_BINDING {_cnl_value(graph['source_binding'])}",
        "NEGATIVE_RULE Answer false only inside an explicitly complete closed-world projection; otherwise answer unknown.",
        "IDENTITY_RULE link/X, frame/X, joint/X, and other typed IDs are never interchangeable.",
        "BOUNDARY_RULE A structural leaf is not an end effector; an asserted constraint is not physical truth; a finite component is not a certified global branch.",
    ]
    for clause in graph["clauses"]:
        evidence = clause["evidence"]
        lines.append(
            f"CLAUSE {clause['clause_id']} | {clause['cnl']} | "
            f"modality={clause['modality']} scope={clause['scope']} "
            f"evidence={evidence['source_type']} exact={str(evidence['exact']).lower()}"
        )
    lines.append("END_CONCEPT_GRAPH")
    return "\n".join(lines) + "\n"


def write_concept_graph(
    graph_path: Path,
    language_path: Path,
    canonical: dict[str, Any],
    articulation: dict[str, Any],
    constraint_graph: dict[str, Any] | None = None,
    configuration_atlas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph = build_concept_graph(canonical, articulation, constraint_graph, configuration_atlas)
    language = render_concept_language(graph)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    language_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_bytes(_json_bytes(graph))
    language_path.write_text(language, encoding="utf-8")
    return graph


def _load_bound_artifact(
    context_directory: Path,
    canonical: dict[str, Any],
    artifact_name: str,
    required: bool,
) -> dict[str, Any] | None:
    record = canonical.get("artifacts", {}).get(artifact_name)
    if record is None:
        if required:
            raise ConceptError(f"model.json has no {artifact_name} artifact")
        return None
    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise ConceptError(f"model.json {artifact_name} artifact binding is malformed")
    path = context_directory / record["path"]
    actual_sha = _sha256_path(path)
    if actual_sha != record["sha256"]:
        raise ConceptError(
            f"{artifact_name} artifact digest mismatch: expected {record['sha256']}, got {actual_sha}"
        )
    return _read_json(path, artifact_name)


def build_concept_graph_from_context(context_directory: Path) -> dict[str, Any]:
    canonical = _read_json(context_directory / "model.json", "canonical model")
    articulation = _load_bound_artifact(context_directory, canonical, "articulation_grammar", True)
    assert articulation is not None
    constraint_graph = _load_bound_artifact(context_directory, canonical, "constraint_graph", False)
    configuration_atlas = _load_bound_artifact(context_directory, canonical, "configuration_atlas", False)
    return build_concept_graph(canonical, articulation, constraint_graph, configuration_atlas)


def write_concept_graph_from_context(
    context_directory: Path,
    graph_path: Path,
    language_path: Path,
) -> dict[str, Any]:
    canonical = _read_json(context_directory / "model.json", "canonical model")
    articulation = _load_bound_artifact(context_directory, canonical, "articulation_grammar", True)
    assert articulation is not None
    constraint_graph = _load_bound_artifact(context_directory, canonical, "constraint_graph", False)
    configuration_atlas = _load_bound_artifact(context_directory, canonical, "configuration_atlas", False)
    return write_concept_graph(
        graph_path,
        language_path,
        canonical,
        articulation,
        constraint_graph,
        configuration_atlas,
    )


def _validate_concept_graph(graph: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "concept_graph_id",
        "source_binding",
        "language_contract",
        "ontology_contract",
        "entities",
        "clauses",
        "indexes",
        "projections",
        "query_contract",
        "coverage",
        "epistemic_scope",
        "concept_graph_sha256",
    }
    _expect_keys(graph, expected, "concept graph")
    if graph["schema_version"] != CONCEPT_SCHEMA:
        raise ConceptError(f"concept graph must use {CONCEPT_SCHEMA}")
    digest_body = {key: value for key, value in graph.items() if key != "concept_graph_sha256"}
    expected_digest = _sha256_bytes(_canonical_bytes(digest_body))
    if graph["concept_graph_sha256"] != expected_digest:
        raise ConceptError("concept graph semantic digest is invalid")
    language_contract = graph["language_contract"]
    if not isinstance(language_contract, dict) or set(language_contract) != {
        "language_version", "clause_form", "closed_world_domains", "open_world_domains", "negative_answer_rule"
    }:
        raise ConceptError("concept graph language contract is malformed")
    if language_contract["language_version"] != LANGUAGE_VERSION:
        raise ConceptError(f"concept graph language must use {LANGUAGE_VERSION}")
    ontology_contract = graph["ontology_contract"]
    if not isinstance(ontology_contract, dict) or set(ontology_contract) != {
        "entity_identity_is_typed",
        "link_frame_and_link_body_are_distinct",
        "structural_leaf_is_not_end_effector",
        "asserted_semantics_are_not_inferred_geometry",
        "tree_parameterization_is_not_complete_closed_mechanism",
        "finite_witness_component_is_not_certified_global_branch",
        "clause_modalities",
    }:
        raise ConceptError("concept graph ontology contract is malformed")
    boundary_flags = [
        ontology_contract["entity_identity_is_typed"],
        ontology_contract["link_frame_and_link_body_are_distinct"],
        ontology_contract["structural_leaf_is_not_end_effector"],
        ontology_contract["asserted_semantics_are_not_inferred_geometry"],
        ontology_contract["tree_parameterization_is_not_complete_closed_mechanism"],
        ontology_contract["finite_witness_component_is_not_certified_global_branch"],
    ]
    if boundary_flags != [True] * len(boundary_flags):
        raise ConceptError("concept graph ontology boundary flags must remain explicit")
    expected_modalities = [
        "declared_exact",
        "derived_exact",
        "project_asserted",
        "supplemental_asserted_structure",
        "supplemental_asserted_relation",
        "derived_exact_from_asserted_relation",
        "finite_computed_evidence",
    ]
    if ontology_contract["clause_modalities"] != expected_modalities:
        raise ConceptError("concept graph clause modality vocabulary is malformed")
    query_contract = graph["query_contract"]
    if (
        not isinstance(query_contract, dict)
        or set(query_contract) != {"schema_version", "intents"}
        or query_contract["schema_version"] != QUERY_SCHEMA
        or not isinstance(query_contract["intents"], dict)
    ):
        raise ConceptError("concept graph query contract is malformed")
    entities = graph["entities"]
    clauses = graph["clauses"]
    if not isinstance(entities, list) or not isinstance(clauses, list):
        raise ConceptError("concept graph entities and clauses must be arrays")
    entity_ids: set[str] = set()
    ordered_entity_ids: list[str] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ConceptError(f"entity {index} must be an object")
        _expect_keys(entity, {"entity_id", "concept_types", "source_paths"}, f"entity {index}")
        entity_id = entity["entity_id"]
        if not isinstance(entity_id, str) or entity_id in entity_ids:
            raise ConceptError(f"invalid or duplicate entity ID at entity {index}")
        concept_types = entity["concept_types"]
        source_paths = entity["source_paths"]
        if (
            not isinstance(concept_types, list)
            or not concept_types
            or not all(isinstance(value, str) and value for value in concept_types)
            or concept_types != sorted(set(concept_types))
        ):
            raise ConceptError(f"entity {entity_id!r} concept types are malformed")
        if (
            not isinstance(source_paths, list)
            or not all(isinstance(value, str) and value for value in source_paths)
            or source_paths != sorted(set(source_paths))
        ):
            raise ConceptError(f"entity {entity_id!r} source paths are malformed")
        entity_ids.add(entity_id)
        ordered_entity_ids.append(entity_id)
    if ordered_entity_ids != sorted(ordered_entity_ids):
        raise ConceptError("concept graph entities are not in canonical order")
    clause_ids: set[str] = set()
    ordered_clause_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, clause in enumerate(clauses):
        if not isinstance(clause, dict):
            raise ConceptError(f"clause {index} must be an object")
        _expect_keys(
            clause,
            {"clause_id", "predicate", "subject", "object", "modality", "scope", "evidence", "proof", "cnl"},
            f"clause {index}",
        )
        if clause["subject"] not in entity_ids:
            raise ConceptError(f"clause {index} references unknown subject {clause['subject']!r}")
        clause_id = clause["clause_id"]
        if not isinstance(clause_id, str) or clause_id in clause_ids:
            raise ConceptError(f"invalid or duplicate clause ID at clause {index}")
        if not isinstance(clause["predicate"], str) or not clause["predicate"]:
            raise ConceptError(f"clause {index} predicate is malformed")
        if not isinstance(clause["modality"], str) or clause["modality"] not in graph["ontology_contract"]["clause_modalities"]:
            raise ConceptError(f"clause {clause_id!r} modality is not declared")
        if not isinstance(clause["scope"], str) or not clause["scope"]:
            raise ConceptError(f"clause {clause_id!r} scope is malformed")
        evidence = clause["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"exact", "source_type", "source_paths"}:
            raise ConceptError(f"clause {clause_id!r} evidence is malformed")
        if not isinstance(evidence["exact"], bool) or not isinstance(evidence["source_type"], str):
            raise ConceptError(f"clause {clause_id!r} evidence types are malformed")
        if (
            not isinstance(evidence["source_paths"], list)
            or not all(isinstance(value, str) and value for value in evidence["source_paths"])
            or evidence["source_paths"] != sorted(set(evidence["source_paths"]))
        ):
            raise ConceptError(f"clause {clause_id!r} evidence source paths are malformed")
        if not isinstance(clause["cnl"], str) or not clause["cnl"] or clause["cnl"] != " ".join(clause["cnl"].split()):
            raise ConceptError(f"clause {clause_id!r} controlled language is malformed")
        body = {key: value for key, value in clause.items() if key != "clause_id"}
        expected_id = f"concept_clause/{clause['predicate']}/{_sha256_bytes(_canonical_bytes(body))[:20]}"
        if clause_id != expected_id:
            raise ConceptError(f"clause {clause_id!r} content digest is invalid")
        clause_ids.add(clause_id)
        ordered_clause_ids.append(clause_id)
        by_id[clause_id] = clause
    if ordered_clause_ids != sorted(ordered_clause_ids):
        raise ConceptError("concept graph clauses are not in canonical order")
    for clause in clauses:
        proof = clause["proof"]
        if not isinstance(proof, dict) or set(proof) != {"rule", "premise_clause_ids"}:
            raise ConceptError(f"clause {clause['clause_id']} proof is malformed")
        if not isinstance(proof["rule"], str) or not proof["rule"]:
            raise ConceptError(f"clause {clause['clause_id']} proof rule is malformed")
        if (
            not isinstance(proof["premise_clause_ids"], list)
            or not all(isinstance(value, str) for value in proof["premise_clause_ids"])
            or len(proof["premise_clause_ids"]) != len(set(proof["premise_clause_ids"]))
        ):
            raise ConceptError(f"clause {clause['clause_id']} premise list is malformed")
        missing = set(proof["premise_clause_ids"]) - clause_ids
        if missing:
            raise ConceptError(f"clause {clause['clause_id']} has missing premises {sorted(missing)}")
    indexes = graph["indexes"]
    if not isinstance(indexes, dict) or set(indexes) != {"by_predicate", "by_subject", "by_entity"}:
        raise ConceptError("concept graph indexes are malformed")
    for index_name, mapping in indexes.items():
        if not isinstance(mapping, dict):
            raise ConceptError(f"concept graph index {index_name} must be an object")
        for key, ids in mapping.items():
            if not isinstance(key, str) or not isinstance(ids, list) or set(ids) - clause_ids:
                raise ConceptError(f"concept graph index {index_name}/{key} is invalid")

    expected_by_predicate: dict[str, list[str]] = {}
    expected_by_subject: dict[str, list[str]] = {}
    expected_by_entity: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
    for clause in clauses:
        clause_id = clause["clause_id"]
        expected_by_predicate.setdefault(clause["predicate"], []).append(clause_id)
        expected_by_subject.setdefault(clause["subject"], []).append(clause_id)
        references = {clause["subject"]} | _entity_references(clause["object"], entity_ids)
        for entity_id in references:
            expected_by_entity[entity_id].append(clause_id)
    expected_indexes = {
        "by_predicate": {key: sorted(value) for key, value in sorted(expected_by_predicate.items())},
        "by_subject": {key: sorted(value) for key, value in sorted(expected_by_subject.items())},
        "by_entity": {key: sorted(value) for key, value in sorted(expected_by_entity.items())},
    }
    if indexes != expected_indexes:
        raise ConceptError("concept graph indexes do not exactly match the clauses")

    projections = graph["projections"]
    if not isinstance(projections, dict) or set(projections) != {
        "topology", "articulation", "project_semantics", "mechanism", "configuration"
    }:
        raise ConceptError("concept graph projections are malformed")

    def validate_projection_references(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}/{key}"
                if key.endswith("_clause_id"):
                    if child is not None and (not isinstance(child, str) or child not in clause_ids):
                        raise ConceptError(f"projection clause reference {child_path} is invalid")
                elif key.endswith("_clause_ids"):
                    if (
                        not isinstance(child, list)
                        or not all(isinstance(item, str) and item in clause_ids for item in child)
                        or len(child) != len(set(child))
                    ):
                        raise ConceptError(f"projection clause references {child_path} are invalid")
                validate_projection_references(child, child_path)
        elif isinstance(value, list):
            for item_index, child in enumerate(value):
                validate_projection_references(child, f"{path}/{item_index}")

    validate_projection_references(projections, "projections")

    def require_support(
        clause_id: str,
        predicate: str,
        subject: str,
        object_value: Any,
        label: str,
    ) -> None:
        support = by_id.get(clause_id)
        if (
            support is None
            or support["predicate"] != predicate
            or support["subject"] != subject
            or support["object"] != object_value
        ):
            raise ConceptError(f"{label} disagrees with its supporting clause")

    topology = projections["topology"]
    required_topology = {
        "root_link", "root_supporting_clause_id", "links", "joints", "edges", "branch_points",
        "structural_leaves", "maximal_serial_segments", "coverage",
    }
    if not isinstance(topology, dict) or set(topology) != required_topology:
        raise ConceptError("topology projection is malformed")
    for key, prefix in (("links", "link/"), ("joints", "joint/"), ("branch_points", "link/"), ("structural_leaves", "link/")):
        values = topology[key]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(value not in entity_ids or not value.startswith(prefix) for value in values)
        ):
            raise ConceptError(f"topology {key} is not an exact typed entity set")
    if topology["root_link"] not in topology["links"]:
        raise ConceptError("topology root link is absent from its complete link set")
    edge_keys: set[tuple[str, str, str]] = set()
    for edge in topology["edges"]:
        if not isinstance(edge, dict) or set(edge) != {
            "parent_link", "child_link", "joint", "joint_type", "supporting_clause_id"
        }:
            raise ConceptError("topology edge is malformed")
        edge_key = (edge["parent_link"], edge["child_link"], edge["joint"])
        if edge_key in edge_keys:
            raise ConceptError(f"duplicate topology edge {edge_key}")
        edge_keys.add(edge_key)
        if edge["parent_link"] not in topology["links"] or edge["child_link"] not in topology["links"]:
            raise ConceptError(f"topology edge {edge_key} has an unknown link")
        if edge["joint"] not in topology["joints"]:
            raise ConceptError(f"topology edge {edge_key} has an unknown joint")
        support = by_id.get(edge["supporting_clause_id"])
        if not isinstance(support, dict) or not isinstance(support.get("object"), dict) or (
            support["predicate"] != "connects_parent_to_child"
            or support["subject"] != edge["joint"]
            or support["object"].get("parent_link") != edge["parent_link"]
            or support["object"].get("child_link") != edge["child_link"]
            or support["object"].get("joint_type") != edge["joint_type"]
        ):
            raise ConceptError(f"topology edge {edge_key} disagrees with its supporting clause")
    if len(topology["edges"]) != len(topology["links"]) - 1 or len(topology["joints"]) != len(topology["edges"]):
        raise ConceptError("topology projection does not encode one complete tree")
    children: dict[str, list[str]] = {link: [] for link in topology["links"]}
    incoming: dict[str, int] = {link: 0 for link in topology["links"]}
    for edge in topology["edges"]:
        children[edge["parent_link"]].append(edge["child_link"])
        incoming[edge["child_link"]] += 1
    if incoming[topology["root_link"]] != 0 or any(
        count != (0 if link == topology["root_link"] else 1)
        for link, count in incoming.items()
    ):
        raise ConceptError("topology projection has invalid rooted-tree parent counts")
    reached: set[str] = set()
    queue = deque([topology["root_link"]])
    while queue:
        link = queue.popleft()
        if link in reached:
            raise ConceptError("topology projection contains a cycle")
        reached.add(link)
        queue.extend(children[link])
    if reached != set(topology["links"]):
        raise ConceptError("topology projection is not connected from its root")
    expected_branches = sorted(link for link, values in children.items() if len(values) > 1)
    expected_leaves = sorted(link for link, values in children.items() if not values)
    if topology["branch_points"] != expected_branches or topology["structural_leaves"] != expected_leaves:
        raise ConceptError("topology branch or leaf projection disagrees with the complete tree")
    root_support = by_id.get(topology["root_supporting_clause_id"])
    if (
        not isinstance(root_support, dict)
        or root_support["predicate"] != "has_root_link"
        or root_support["object"] != topology["root_link"]
    ):
        raise ConceptError("topology root disagrees with its supporting clause")

    for segment in topology["maximal_serial_segments"]:
        required_segment = {
            "segment_entity", "start_link", "end_link", "ordered_links", "ordered_joints",
            "start_boundary", "end_boundary", "supporting_clause_id",
        }
        if not isinstance(segment, dict) or set(segment) != required_segment:
            raise ConceptError("serial-segment projection is malformed")
        object_value = {key: value for key, value in segment.items() if key != "supporting_clause_id"}
        require_support(
            segment["supporting_clause_id"],
            "is_maximal_serial_segment",
            segment["segment_entity"],
            object_value,
            f"serial segment {segment.get('segment_entity')!r}",
        )
        if segment["segment_entity"] not in entity_ids:
            raise ConceptError("serial-segment projection references an unknown entity")
        ordered_links = segment["ordered_links"]
        ordered_joints = segment["ordered_joints"]
        if (
            not isinstance(ordered_links, list)
            or not isinstance(ordered_joints, list)
            or len(ordered_links) != len(ordered_joints) + 1
            or ordered_links[0] != segment["start_link"]
            or ordered_links[-1] != segment["end_link"]
        ):
            raise ConceptError("serial-segment ordered path is malformed")
        for index, joint in enumerate(ordered_joints):
            if (ordered_links[index], ordered_links[index + 1], joint) not in edge_keys:
                raise ConceptError("serial-segment ordered path is not a tree path")

    articulation_projection = projections["articulation"]
    if not isinstance(articulation_projection, dict) or set(articulation_projection) != {
        "grammar_id", "root_frame", "drivers", "joint_operators", "frame_laws", "coverage"
    }:
        raise ConceptError("articulation projection is malformed")
    for driver in articulation_projection["drivers"]:
        required_driver = {
            "driver_entity", "driver_name", "domain", "physical_joints_driven", "affected_frames",
            "domain_clause_id", "drives_clause_ids", "affects_clause_ids",
        }
        if not isinstance(driver, dict) or set(driver) != required_driver:
            raise ConceptError("articulation driver projection is malformed")
        driver_entity = driver["driver_entity"]
        if driver_entity not in entity_ids:
            raise ConceptError(f"articulation driver {driver_entity!r} is not a declared entity")
        require_support(
            driver["domain_clause_id"], "has_feasible_driver_domain", driver_entity,
            driver["domain"], f"articulation driver {driver_entity!r} domain",
        )
        driven: list[str] = []
        for clause_id in driver["drives_clause_ids"]:
            clause = by_id[clause_id]
            object_value = clause["object"]
            if (
                clause["predicate"] != "drives_physical_joint"
                or clause["subject"] != driver_entity
                or not isinstance(object_value, dict)
                or object_value.get("physical_joint") not in entity_ids
            ):
                raise ConceptError(f"articulation driver {driver_entity!r} has an invalid driven-joint clause")
            driven.append(object_value["physical_joint"])
        if driver["physical_joints_driven"] != driven:
            raise ConceptError(f"articulation driver {driver_entity!r} driven-joint projection is inconsistent")
        affected: list[str] = []
        for clause_id in driver["affects_clause_ids"]:
            clause = by_id[clause_id]
            object_value = clause["object"]
            if (
                clause["predicate"] != "can_change_pose_of_frame_relative_to_root"
                or clause["subject"] != driver_entity
                or not isinstance(object_value, dict)
                or object_value.get("frame") not in entity_ids
            ):
                raise ConceptError(f"articulation driver {driver_entity!r} has an invalid frame-effect clause")
            affected.append(object_value["frame"])
        if driver["affected_frames"] != affected:
            raise ConceptError(f"articulation driver {driver_entity!r} affected-frame projection is inconsistent")
    for law in articulation_projection["frame_laws"]:
        if not isinstance(law, dict) or set(law) != {
            "frame", "root_frame", "expression_cnl", "ordered_operator_refs",
            "independent_driver_dependencies", "supporting_clause_id",
        }:
            raise ConceptError("articulation frame-law projection is malformed")
        object_value = {key: value for key, value in law.items() if key not in {"frame", "supporting_clause_id"}}
        require_support(
            law["supporting_clause_id"], "has_ordered_pose_composition_law", law["frame"],
            object_value, f"frame law {law.get('frame')!r}",
        )
    for operator in articulation_projection["joint_operators"]:
        if not isinstance(operator, dict) or set(operator) != {
            "joint", "operator_ref", "joint_type", "parent_link", "child_link", "motion_operator",
            "composition_rule", "supporting_clause_id",
        }:
            raise ConceptError("articulation joint-operator projection is malformed")
        object_value = {key: value for key, value in operator.items() if key not in {"joint", "supporting_clause_id"}}
        require_support(
            operator["supporting_clause_id"], "has_typed_motion_operator", operator["joint"],
            object_value, f"joint operator {operator.get('joint')!r}",
        )

    semantic_projection = projections["project_semantics"]
    if not isinstance(semantic_projection, dict) or set(semantic_projection) != {
        "status", "asserted_frame_roles", "name_based_role_inference_permitted"
    } or semantic_projection["name_based_role_inference_permitted"] is not False:
        raise ConceptError("project-semantics projection is malformed")
    for role in semantic_projection["asserted_frame_roles"]:
        if not isinstance(role, dict) or set(role) != {"frame", "role", "meaning", "supporting_clause_id"}:
            raise ConceptError("asserted semantic role projection is malformed")
        require_support(
            role["supporting_clause_id"], "has_asserted_semantic_role", role["frame"],
            {"role": role["role"], "meaning": role["meaning"]},
            f"asserted semantic role for {role.get('frame')!r}",
        )

    mechanism = projections["mechanism"]
    if not isinstance(mechanism, dict) or not isinstance(mechanism.get("constraints"), list):
        raise ConceptError("mechanism projection is malformed")
    if mechanism.get("status") == "not_provided":
        if set(mechanism) != {"status", "tree_is_complete_mechanism", "constraints", "attachments"}:
            raise ConceptError("absent mechanism projection is malformed")
    else:
        required_mechanism = {
            "status", "constraint_graph_entity", "tree_is_coordinate_parameterization",
            "parameterization_clause_id", "attachments", "constraints",
            "asserted_relations_are_physical_truth",
        }
        if set(mechanism) != required_mechanism or mechanism["asserted_relations_are_physical_truth"] is not False:
            raise ConceptError("bound mechanism projection is malformed")
        for attachment in mechanism["attachments"]:
            if not isinstance(attachment, dict) or set(attachment) != {
                "attachment_entity", "frame_id", "parent_frame", "semantic_role",
                "parent_from_attachment", "supporting_clause_id",
            }:
                raise ConceptError("mechanism attachment projection is malformed")
            object_value = {key: value for key, value in attachment.items() if key not in {"attachment_entity", "supporting_clause_id"}}
            require_support(
                attachment["supporting_clause_id"], "declares_rigid_attachment", attachment["attachment_entity"],
                object_value, f"mechanism attachment {attachment.get('attachment_entity')!r}",
            )
        for constraint in mechanism["constraints"]:
            if not isinstance(constraint, dict) or set(constraint) != {
                "constraint_entity", "constraint_id", "type", "role", "relation", "driver_dependencies",
                "relation_clause_id", "dependency_clause_ids",
            }:
                raise ConceptError("mechanism constraint projection is malformed")
            constraint_entity = constraint["constraint_entity"]
            relation_value = {
                key: constraint[key]
                for key in ("constraint_id", "type", "role", "relation", "driver_dependencies")
            }
            require_support(
                constraint["relation_clause_id"], "requires_mechanism_relation", constraint_entity,
                relation_value, f"mechanism constraint {constraint_entity!r}",
            )
            dependency_objects: list[str] = []
            for clause_id in constraint["dependency_clause_ids"]:
                clause = by_id[clause_id]
                if clause["predicate"] != "constraint_depends_on_driver" or clause["subject"] != constraint_entity:
                    raise ConceptError(f"mechanism constraint {constraint_entity!r} dependency clause is malformed")
                dependency_objects.append(clause["object"])
            if dependency_objects != constraint["driver_dependencies"]:
                raise ConceptError(f"mechanism constraint {constraint_entity!r} dependency projection is inconsistent")

    configuration = projections["configuration"]
    if not isinstance(configuration, dict) or not isinstance(configuration.get("charts"), list):
        raise ConceptError("configuration projection is malformed")
    if configuration.get("status") == "not_provided":
        if set(configuration) != {"status", "charts", "global_branch_topology_certified"}:
            raise ConceptError("absent configuration projection is malformed")
    else:
        required_configuration = {
            "status", "atlas_entity", "status_clause_id", "charts", "global_branch_topology_certified",
            "certified_singularity", "finite_proximity_component_is_global_branch",
        }
        if set(configuration) != required_configuration or any(
            configuration[key] is not False
            for key in (
                "global_branch_topology_certified", "certified_singularity",
                "finite_proximity_component_is_global_branch",
            )
        ):
            raise ConceptError("bound configuration projection is malformed")
        status_support = by_id.get(configuration["status_clause_id"])
        if (
            not isinstance(status_support, dict)
            or status_support["predicate"] != "has_finite_declared_sampling_status"
            or status_support["subject"] != configuration["atlas_entity"]
            or not isinstance(status_support["object"], dict)
            or status_support["object"].get("status") != configuration["status"]
        ):
            raise ConceptError("configuration status projection disagrees with its supporting clause")
        all_node_entities: set[str] = set()
        for chart in configuration["charts"]:
            required_chart = {
                "chart_entity", "parameter_driver", "parameter_values", "solve_for", "driver_scales",
                "minimum_solutions_per_sample", "coverage", "chart_clause_id", "nodes",
                "finite_proximity_components",
            }
            if not isinstance(chart, dict) or set(chart) != required_chart:
                raise ConceptError("configuration chart projection is malformed")
            chart_value = {
                key: chart[key]
                for key in (
                    "parameter_driver", "parameter_values", "solve_for", "driver_scales",
                    "minimum_solutions_per_sample", "coverage",
                )
            }
            require_support(
                chart["chart_clause_id"], "declares_finite_configuration_chart", chart["chart_entity"],
                chart_value, f"configuration chart {chart.get('chart_entity')!r}",
            )
            for node in chart["nodes"]:
                required_node = {
                    "node_entity", "parameter_driver", "parameter_value", "independent_driver_positions",
                    "constraint_status", "maximum_normalized_abs", "full_constraint_rank", "chart_passive_rank",
                    "singularity_witness", "supporting_clause_id",
                }
                if not isinstance(node, dict) or set(node) != required_node:
                    raise ConceptError("configuration node projection is malformed")
                node_value = {key: value for key, value in node.items() if key not in {"node_entity", "supporting_clause_id"}}
                require_support(
                    node["supporting_clause_id"], "is_executable_satisfying_configuration_witness",
                    node["node_entity"], node_value, f"configuration node {node.get('node_entity')!r}",
                )
                all_node_entities.add(node["node_entity"])
        for chart in configuration["charts"]:
            for component in chart["finite_proximity_components"]:
                if not isinstance(component, dict) or set(component) != {
                    "component_entity", "node_entities", "meaning", "supporting_clause_id"
                }:
                    raise ConceptError("configuration component projection is malformed")
                component_value = {
                    "node_entities": component["node_entities"],
                    "meaning": component["meaning"],
                }
                require_support(
                    component["supporting_clause_id"], "groups_finite_proximity_witnesses",
                    component["component_entity"], component_value,
                    f"configuration component {component.get('component_entity')!r}",
                )
                if not set(component["node_entities"]).issubset(all_node_entities):
                    raise ConceptError("configuration component references an unknown node")

    coverage = graph["coverage"]
    expected_coverage = {
        "entity_count": len(entities),
        "clause_count": len(clauses),
        "predicate_count": len(expected_by_predicate),
        "tree_link_count": len(topology["links"]),
        "tree_joint_count": len(topology["joints"]),
        "independent_driver_count": len(projections["articulation"]["drivers"]),
        "constraint_count": len(projections["mechanism"]["constraints"]),
        "configuration_chart_count": len(projections["configuration"]["charts"]),
        "exact_regeneration_supported": True,
    }
    if coverage != expected_coverage:
        raise ConceptError("concept graph coverage does not exactly match its contents")


def read_concept_graph(path: Path) -> dict[str, Any]:
    graph = _read_json(path, "concept graph")
    _validate_concept_graph(graph)
    return graph


def verify_concept_graph(
    context_directory: Path,
    graph_path: Path,
    language_path: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    try:
        stored = read_concept_graph(graph_path)
    except ConceptError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "exact_graph_regeneration_match": False,
            "exact_language_regeneration_match": False,
            "validated_clause_count": 0,
            "issues": [{"check": "stored_graph_structure", "message": str(error)}],
        }
    expected = build_concept_graph_from_context(context_directory)
    expected_language = render_concept_language(expected)
    graph_match = _json_bytes(stored) == _json_bytes(expected)
    if not graph_match:
        issues.append({"check": "exact_graph_regeneration", "message": "stored concept graph differs from exact regeneration"})
    try:
        stored_language = language_path.read_text(encoding="utf-8")
    except OSError as error:
        stored_language = ""
        issues.append({"check": "language_read", "message": str(error)})
    language_match = stored_language == expected_language
    if not language_match:
        issues.append({"check": "exact_language_regeneration", "message": "stored controlled language differs from exact regeneration"})
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "concept_graph_id": stored["concept_graph_id"],
        "concept_graph_sha256": stored["concept_graph_sha256"],
        "exact_graph_regeneration_match": graph_match,
        "exact_language_regeneration_match": language_match,
        "validated_entity_count": len(stored["entities"]),
        "validated_clause_count": len(stored["clauses"]),
        "issues": issues,
        "epistemic_scope": "exact regeneration validates the symbolic abstraction against the same bound source artifacts; it does not independently validate those artifacts or establish physical truth",
    }


def _resolve_entity(graph: dict[str, Any], supplied: str, prefixes: tuple[str, ...] | None = None) -> str:
    entities = [record["entity_id"] for record in graph["entities"]]
    candidates = entities
    if prefixes is not None:
        candidates = [entity for entity in candidates if entity.startswith(prefixes)]
    if supplied in candidates:
        return supplied
    bare_matches = [entity for entity in candidates if entity.rsplit("/", 1)[-1] == supplied]
    if len(bare_matches) == 1:
        return bare_matches[0]
    if not bare_matches:
        raise ConceptError(f"unknown typed entity or unique bare name {supplied!r}")
    raise ConceptError(f"ambiguous bare entity name {supplied!r}; candidates={sorted(bare_matches)}")


def _clauses(graph: dict[str, Any], clause_ids: Iterable[str]) -> list[dict[str, Any]]:
    by_id = {clause["clause_id"]: clause for clause in graph["clauses"]}
    selected: set[str] = set()
    queue = deque(clause_ids)
    while queue:
        clause_id = queue.popleft()
        if clause_id in selected:
            continue
        if clause_id not in by_id:
            raise ConceptError(f"query selected unknown clause {clause_id}")
        selected.add(clause_id)
        queue.extend(by_id[clause_id]["proof"]["premise_clause_ids"])
    return [by_id[clause_id] for clause_id in sorted(selected)]


def _path_query(graph: dict[str, Any], start: str, target: str) -> tuple[dict[str, Any], list[str]]:
    topology = graph["projections"]["topology"]
    adjacency: dict[str, list[tuple[str, dict[str, Any], str]]] = {link: [] for link in topology["links"]}
    for edge in topology["edges"]:
        adjacency[edge["parent_link"]].append((edge["child_link"], edge, "parent_to_child"))
        adjacency[edge["child_link"]].append((edge["parent_link"], edge, "child_to_parent"))
    queue: deque[tuple[str, list[dict[str, Any]]]] = deque([(start, [])])
    visited = {start}
    while queue:
        current, steps = queue.popleft()
        if current == target:
            clause_ids = [step["supporting_clause_id"] for step in steps]
            return {
                "from_link": start,
                "to_link": target,
                "ordered_steps": steps,
                "joint_count": len(steps),
                "unique_path_in_tree": True,
            }, clause_ids
        for neighbor, edge, direction in sorted(adjacency[current], key=lambda item: (item[0], item[1]["joint"])):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, [*steps, {
                "from_link": current,
                "to_link": neighbor,
                "joint": edge["joint"],
                "joint_type": edge["joint_type"],
                "traversal_direction": direction,
                "supporting_clause_id": edge["supporting_clause_id"],
            }]))
    raise ConceptError(f"no tree path between {start} and {target}")


def _driver_record(graph: dict[str, Any], entity: str) -> dict[str, Any]:
    for record in graph["projections"]["articulation"]["drivers"]:
        if record["driver_entity"] == entity:
            return record
    raise ConceptError(f"driver projection has no record for {entity}")


def _node_record(graph: dict[str, Any], entity: str) -> tuple[dict[str, Any], str | None]:
    for chart in graph["projections"]["configuration"]["charts"]:
        for node in chart["nodes"]:
            if node["node_entity"] == entity:
                component = next((
                    component["component_entity"]
                    for component in chart["finite_proximity_components"]
                    if entity in component["node_entities"]
                ), None)
                return node, component
    raise ConceptError(f"configuration projection has no node {entity}")


def query_concept_graph(graph: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    _validate_concept_graph(graph)
    _expect_keys(query, {"schema_version", "query_id", "intent", "parameters"}, "concept query")
    if query["schema_version"] != QUERY_SCHEMA:
        raise ConceptError(f"concept query must use {QUERY_SCHEMA}")
    if not isinstance(query["query_id"], str) or not query["query_id"]:
        raise ConceptError("concept query_id must be a non-empty string")
    intent = query["intent"]
    parameters = query["parameters"]
    if not isinstance(parameters, dict):
        raise ConceptError("concept query parameters must be an object")
    clause_ids: list[str] = []
    unknowns: list[str] = []

    if intent == "structural_summary":
        _expect_keys(parameters, set(), "structural_summary parameters")
        topology = graph["projections"]["topology"]
        articulation = graph["projections"]["articulation"]
        mechanism = graph["projections"]["mechanism"]
        configuration = graph["projections"]["configuration"]
        clause_ids = [topology["root_supporting_clause_id"]]
        clause_ids.extend(segment["supporting_clause_id"] for segment in topology["maximal_serial_segments"])
        clause_ids.extend(driver["domain_clause_id"] for driver in articulation["drivers"])
        if mechanism.get("parameterization_clause_id"):
            clause_ids.append(mechanism["parameterization_clause_id"])
        if configuration.get("status_clause_id"):
            clause_ids.append(configuration["status_clause_id"])
        answer = {
            "robot": graph["source_binding"]["robot_name"],
            "root_link": topology["root_link"],
            "link_count": len(topology["links"]),
            "joint_count": len(topology["joints"]),
            "branch_points": topology["branch_points"],
            "structural_leaves": topology["structural_leaves"],
            "maximal_serial_segments": topology["maximal_serial_segments"],
            "independent_drivers": [record["driver_entity"] for record in articulation["drivers"]],
            "supplemental_mechanism_status": mechanism["status"],
            "constraint_count": len(mechanism["constraints"]),
            "configuration_evidence_status": configuration["status"],
        }
        answer_cnl = (
            f"Robot {answer['robot']} has root {answer['root_link']}, {answer['link_count']} links, "
            f"{answer['joint_count']} joints, {len(answer['independent_drivers'])} independent drivers, "
            f"and {answer['constraint_count']} supplemental constraints. Structural leaves are not automatically end effectors."
        )
    elif intent == "trace_kinematic_path":
        _expect_keys(parameters, {"from_link", "to_link"}, "trace_kinematic_path parameters")
        start = _resolve_entity(graph, parameters["from_link"], ("link/",))
        target = _resolve_entity(graph, parameters["to_link"], ("link/",))
        answer, clause_ids = _path_query(graph, start, target)
        answer_cnl = f"The unique URDF-tree path from {start} to {target} traverses {answer['joint_count']} joints."
    elif intent == "explain_driver_effect":
        allowed = {"driver", "target_frame"}
        if set(parameters) not in ({"driver"}, allowed):
            raise ConceptError("explain_driver_effect parameters require driver and optional target_frame")
        driver = _resolve_entity(graph, parameters["driver"], ("articulation_variable/",))
        record = _driver_record(graph, driver)
        clause_ids = [record["domain_clause_id"], *record["drives_clause_ids"]]
        target = None
        target_changes: bool | None = None
        if "target_frame" in parameters:
            target = _resolve_entity(graph, parameters["target_frame"], ("frame/",))
            target_changes = target in record["affected_frames"]
            if target_changes:
                index = record["affected_frames"].index(target)
                clause_ids.append(record["affects_clause_ids"][index])
            elif not graph["projections"]["articulation"]["coverage"]["negative_driver_frame_effect_answerable_from_complete_dependency_sets"]:
                target_changes = None
                unknowns.append("driver-to-frame negative is outside a complete frame dependency projection")
        else:
            clause_ids.extend(record["affects_clause_ids"])
        answer = {
            "driver": driver,
            "domain": record["domain"],
            "physical_joints_driven": record["physical_joints_driven"],
            "affected_frames_relative_to_root": record["affected_frames"],
            "other_independent_drivers_held_fixed": True,
            "target_frame": target,
            "target_pose_can_change_relative_to_root": target_changes,
        }
        if target is None:
            answer_cnl = f"Driver {driver} drives {record['physical_joints_driven']} and can change {len(record['affected_frames'])} frame poses relative to root."
        elif target_changes is True:
            answer_cnl = f"Driver {driver} can change pose of {target} relative to root because it occurs in that frame's ordered articulation composition."
        elif target_changes is False:
            answer_cnl = f"Driver {driver} cannot change pose of {target} relative to root within the complete supported articulation dependency graph while other independent drivers stay fixed."
        else:
            answer_cnl = f"Whether driver {driver} can change {target} is unknown under the available coverage."
    elif intent == "explain_frame_pose_law":
        _expect_keys(parameters, {"frame"}, "explain_frame_pose_law parameters")
        frame = _resolve_entity(graph, parameters["frame"], ("frame/",))
        law = next((record for record in graph["projections"]["articulation"]["frame_laws"] if record["frame"] == frame), None)
        if law is None:
            raise ConceptError(f"no supported pose law for {frame}")
        clause_ids = [law["supporting_clause_id"]]
        answer = law
        answer_cnl = f"Frame {frame} is computed by the ordered pose law {law['expression_cnl']}."
    elif intent == "explain_constraint":
        _expect_keys(parameters, {"constraint"}, "explain_constraint parameters")
        constraint = _resolve_entity(graph, parameters["constraint"], ("constraint/",))
        record = next((record for record in graph["projections"]["mechanism"]["constraints"] if record["constraint_entity"] == constraint), None)
        if record is None:
            raise ConceptError(f"no supplemental constraint record for {constraint}")
        clause_ids = [record["relation_clause_id"], *record["dependency_clause_ids"]]
        answer = record
        answer_cnl = (
            f"Constraint {constraint} is an asserted {record['type']} relation with role {record['role']} "
            f"and driver dependencies {record['driver_dependencies']}; assertion is not physical observation."
        )
    elif intent == "compare_configuration_nodes":
        _expect_keys(parameters, {"node_a", "node_b"}, "compare_configuration_nodes parameters")
        node_a_entity = _resolve_entity(graph, parameters["node_a"], ("configuration_node/",))
        node_b_entity = _resolve_entity(graph, parameters["node_b"], ("configuration_node/",))
        node_a, component_a = _node_record(graph, node_a_entity)
        node_b, component_b = _node_record(graph, node_b_entity)
        positions_a = node_a["independent_driver_positions"]
        positions_b = node_b["independent_driver_positions"]
        drivers = sorted(set(positions_a) | set(positions_b))
        deltas = {
            driver: positions_b.get(driver) - positions_a.get(driver)
            for driver in drivers
            if isinstance(positions_a.get(driver), (int, float)) and isinstance(positions_b.get(driver), (int, float))
        }
        clause_ids = [node_a["supporting_clause_id"], node_b["supporting_clause_id"]]
        same_component = component_a is not None and component_a == component_b
        answer = {
            "node_a": node_a_entity,
            "node_b": node_b_entity,
            "driver_position_delta_b_minus_a": deltas,
            "full_constraint_rank_a": node_a["full_constraint_rank"],
            "full_constraint_rank_b": node_b["full_constraint_rank"],
            "chart_passive_rank_a": node_a["chart_passive_rank"],
            "chart_passive_rank_b": node_b["chart_passive_rank"],
            "finite_proximity_component_a": component_a,
            "finite_proximity_component_b": component_b,
            "same_finite_proximity_component": same_component,
            "same_global_branch": "not_established",
        }
        answer_cnl = (
            f"Nodes {node_a_entity} and {node_b_entity} have driver deltas {_cnl_value(deltas)}; "
            f"same finite proximity component={str(same_component).lower()}, while same global branch is not established."
        )
    elif intent == "describe_entity":
        _expect_keys(parameters, {"entity"}, "describe_entity parameters")
        entity = _resolve_entity(graph, parameters["entity"])
        clause_ids = graph["indexes"]["by_entity"].get(entity, [])
        entity_record = next(record for record in graph["entities"] if record["entity_id"] == entity)
        answer = {
            "entity": entity_record,
            "direct_relation_clause_ids": clause_ids,
        }
        answer_cnl = f"Entity {entity} has concept types {entity_record['concept_types']} and {len(clause_ids)} directly indexed relations."
    else:
        raise ConceptError(f"unsupported concept query intent {intent!r}")

    support = _clauses(graph, clause_ids)
    return {
        "schema_version": ANSWER_SCHEMA,
        "query_id": query["query_id"],
        "intent": intent,
        "status": "answered" if not unknowns else "answered_with_unknown_boundary",
        "answer": answer,
        "answer_cnl": answer_cnl,
        "supporting_clauses": support,
        "unknowns": unknowns,
        "source_binding": graph["source_binding"],
        "concept_graph": {
            "concept_graph_id": graph["concept_graph_id"],
            "concept_graph_sha256": graph["concept_graph_sha256"],
        },
        "epistemic_scope": graph["epistemic_scope"],
    }


def query_concept_graph_files(graph_path: Path, query_path: Path) -> dict[str, Any]:
    return query_concept_graph(read_concept_graph(graph_path), _read_json(query_path, "concept query"))
