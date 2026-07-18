#!/usr/bin/env python3
"""Compile and query project-declared robot function, capability, and affordance knowledge."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable


SPEC_SCHEMA = "robot-spatial-function-affordance-spec.v1"
MODEL_SCHEMA = "robot-spatial-functional-model.v1"
QUERY_SCHEMA = "robot-spatial-functional-query.v1"
ANSWER_SCHEMA = "robot-spatial-functional-answer.v1"
VERIFICATION_SCHEMA = "robot-spatial-functional-verification.v1"


class FunctionalError(ValueError):
    """An invalid function/affordance specification, model, query, or binding."""


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
        raise FunctionalError(f"cannot read {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FunctionalError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FunctionalError(f"{label} must contain one JSON object")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FunctionalError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _typed_id(value: Any, prefix: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(f"{prefix}/") or value == f"{prefix}/":
        raise FunctionalError(f"{label} must be a typed {prefix}/... ID")
    return value


def _strings(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FunctionalError(f"{label} must be a list of non-empty strings")
    if nonempty and not value:
        raise FunctionalError(f"{label} must not be empty")
    if len(value) != len(set(value)):
        raise FunctionalError(f"{label} must not contain duplicates")
    return list(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FunctionalError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _clause_id(body: dict[str, Any]) -> str:
    return f"functional_clause/{body['predicate']}/{_sha256_bytes(_canonical_bytes(body))[:20]}"


class _Builder:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.clauses: dict[str, dict[str, Any]] = {}

    def entity(self, entity_id: str, entity_types: Iterable[str], source_refs: Iterable[str]) -> None:
        if not isinstance(entity_id, str) or "/" not in entity_id:
            raise FunctionalError(f"invalid typed functional entity ID {entity_id!r}")
        types = sorted(set(entity_types))
        refs = sorted(set(source_refs))
        existing = self.entities.get(entity_id)
        if existing is None:
            self.entities[entity_id] = {
                "entity_id": entity_id,
                "entity_types": types,
                "source_refs": refs,
            }
            return
        existing["entity_types"] = sorted(set(existing["entity_types"]) | set(types))
        existing["source_refs"] = sorted(set(existing["source_refs"]) | set(refs))

    def clause(
        self,
        predicate: str,
        subject: str,
        object_value: Any,
        *,
        modality: str,
        exact: bool,
        source_type: str,
        source_refs: Iterable[str],
        rule: str,
        premise_clause_ids: Iterable[str] = (),
        concept_premise_clause_ids: Iterable[str] = (),
        cnl: str,
    ) -> str:
        if subject not in self.entities:
            raise FunctionalError(f"functional clause subject is not declared: {subject}")
        body = {
            "predicate": predicate,
            "subject": subject,
            "object": copy.deepcopy(object_value),
            "modality": modality,
            "evidence": {
                "exact": exact,
                "source_type": source_type,
                "source_refs": sorted(set(source_refs)),
            },
            "proof": {
                "rule": rule,
                "premise_clause_ids": list(premise_clause_ids),
                "concept_premise_clause_ids": list(concept_premise_clause_ids),
            },
            "cnl": " ".join(cnl.split()),
        }
        clause_id = _clause_id(body)
        record = {"clause_id": clause_id, **body}
        existing = self.clauses.get(clause_id)
        if existing is not None and existing != record:
            raise FunctionalError(f"functional clause ID collision at {clause_id}")
        self.clauses[clause_id] = record
        return clause_id


def _artifact_binding(canonical: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = canonical.get("artifacts", {}).get(name)
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
        raise FunctionalError(f"canonical artifact binding {name!r} is malformed")
    return value


def _source_binding(
    canonical: dict[str, Any],
    concept_graph: dict[str, Any],
    concept_graph_artifact_sha256: str,
    spec_sha256: str,
) -> dict[str, Any]:
    articulation = _artifact_binding(canonical, "articulation_grammar")
    if articulation is None:
        raise FunctionalError("functional model requires a bound articulation grammar")
    constraint = _artifact_binding(canonical, "constraint_graph")
    configuration = _artifact_binding(canonical, "configuration_atlas")
    semantic_sha = canonical.get("source", {}).get("semantic_sha256")
    if not isinstance(semantic_sha, str):
        raise FunctionalError("canonical URDF semantic digest is missing")
    return {
        "function_spec_sha256": spec_sha256,
        "urdf_semantic_sha256": semantic_sha,
        "articulation_grammar_sha256": articulation["sha256"],
        "constraint_graph_sha256": None if constraint is None else constraint["sha256"],
        "configuration_atlas_sha256": None if configuration is None else configuration["sha256"],
        "concept_graph": {
            "concept_graph_id": concept_graph.get("concept_graph_id"),
            "concept_graph_sha256": concept_graph.get("concept_graph_sha256"),
            "artifact_sha256": concept_graph_artifact_sha256,
        },
    }


def _validate_declared_binding(spec: dict[str, Any], actual: dict[str, Any]) -> None:
    binding = spec.get("source_binding")
    expected_keys = {
        "urdf_semantic_sha256",
        "articulation_grammar_sha256",
        "constraint_graph_sha256",
        "configuration_atlas_sha256",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise FunctionalError("function spec source_binding is malformed")
    for key in expected_keys:
        if binding[key] != actual[key]:
            raise FunctionalError(
                f"function spec {key} mismatch: expected bound value {binding[key]!r}, got {actual[key]!r}"
            )


def _concept_clause_map(concept_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clauses = concept_graph.get("clauses")
    if not isinstance(clauses, list):
        raise FunctionalError("concept graph clauses are missing")
    return {clause["clause_id"]: clause for clause in clauses}


def _validate_concept_input(canonical: dict[str, Any], concept_graph: dict[str, Any]) -> None:
    if concept_graph.get("schema_version") != "robot-spatial-concept-graph.v1":
        raise FunctionalError("functional model requires robot-spatial-concept-graph.v1")
    stored_digest = concept_graph.get("concept_graph_sha256")
    body = {key: value for key, value in concept_graph.items() if key != "concept_graph_sha256"}
    if stored_digest != _sha256_bytes(_canonical_bytes(body)):
        raise FunctionalError("bound concept graph semantic digest is invalid")
    binding = concept_graph.get("source_binding")
    if not isinstance(binding, dict):
        raise FunctionalError("bound concept graph source binding is missing")
    semantic_sha = canonical.get("source", {}).get("semantic_sha256")
    articulation = _artifact_binding(canonical, "articulation_grammar")
    if binding.get("urdf_semantic_sha256") != semantic_sha:
        raise FunctionalError("concept graph URDF semantic binding does not match canonical model")
    if articulation is None or binding.get("articulation_grammar", {}).get("artifact_sha256") != articulation["sha256"]:
        raise FunctionalError("concept graph articulation binding does not match canonical model")
    for artifact_name, binding_name in (
        ("constraint_graph", "constraint_graph"),
        ("configuration_atlas", "configuration_atlas"),
    ):
        artifact = _artifact_binding(canonical, artifact_name)
        concept_record = binding.get(binding_name)
        if artifact is None and concept_record is not None:
            raise FunctionalError(f"concept graph unexpectedly binds {binding_name}")
        if artifact is not None and (
            not isinstance(concept_record, dict)
            or concept_record.get("artifact_sha256") != artifact["sha256"]
        ):
            raise FunctionalError(f"concept graph {binding_name} binding does not match canonical model")
    if not isinstance(concept_graph.get("entities"), list) or not isinstance(concept_graph.get("projections"), dict):
        raise FunctionalError("bound concept graph entities/projections are malformed")
    _concept_clause_map(concept_graph)


def _concept_closure(
    clause_ids: Iterable[str],
    clauses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: set[str] = set()
    queue = deque(clause_ids)
    while queue:
        clause_id = queue.popleft()
        if clause_id in selected:
            continue
        clause = clauses.get(clause_id)
        if clause is None:
            raise FunctionalError(f"functional requirement references missing concept clause {clause_id}")
        selected.add(clause_id)
        queue.extend(clause["proof"]["premise_clause_ids"])
    return [clauses[clause_id] for clause_id in sorted(selected)]


def _requirement_result(
    requirement: dict[str, Any],
    concept_graph: dict[str, Any],
) -> dict[str, Any]:
    _expect_keys(requirement, {"requirement_id", "type", "parameters"}, "enabling requirement")
    requirement_id = _typed_id(requirement["requirement_id"], "requirement", "requirement_id")
    requirement_type = requirement["type"]
    parameters = requirement["parameters"]
    if not isinstance(requirement_type, str) or not isinstance(parameters, dict):
        raise FunctionalError(f"requirement {requirement_id!r} type/parameters are malformed")
    entities = {record["entity_id"] for record in concept_graph["entities"]}
    clauses = _concept_clause_map(concept_graph)
    projection = concept_graph["projections"]
    concept_ids: list[str] = []
    exact = True
    modality = "derived_exact"
    status = "not_satisfied_exact_closed_world"
    explanation = "the complete bound structural projection does not satisfy this requirement"
    closure_basis = "complete bound concept-graph projection"

    if requirement_type == "entity_exists":
        _expect_keys(parameters, {"entity"}, f"requirement {requirement_id} parameters")
        entity = parameters["entity"]
        if not isinstance(entity, str) or "/" not in entity:
            raise FunctionalError(f"requirement {requirement_id} entity must be a typed concept entity ID")
        closure_basis = "complete concept entity inventory"
        satisfied = entity in entities
    elif requirement_type == "kinematic_path_exists":
        _expect_keys(parameters, {"from_link", "to_link"}, f"requirement {requirement_id} parameters")
        start = _typed_id(parameters["from_link"], "link", f"requirement {requirement_id} from_link")
        target = _typed_id(parameters["to_link"], "link", f"requirement {requirement_id} to_link")
        closure_basis = "complete canonical tree links and edges"
        topology = projection["topology"]
        adjacency: dict[str, list[tuple[str, str]]] = {link: [] for link in topology["links"]}
        for edge in topology["edges"]:
            adjacency[edge["parent_link"]].append((edge["child_link"], edge["supporting_clause_id"]))
            adjacency[edge["child_link"]].append((edge["parent_link"], edge["supporting_clause_id"]))
        if start not in adjacency or target not in adjacency:
            satisfied = False
        else:
            queue: deque[tuple[str, list[str]]] = deque([(start, [])])
            visited = {start}
            satisfied = False
            while queue:
                current, path_ids = queue.popleft()
                if current == target:
                    satisfied = True
                    concept_ids = path_ids
                    break
                for neighbor, clause_id in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, [*path_ids, clause_id]))
    elif requirement_type in {"driver_drives_joint", "driver_affects_frame"}:
        expected_parameters = (
            {"driver", "joint"} if requirement_type == "driver_drives_joint" else {"driver", "frame"}
        )
        _expect_keys(parameters, expected_parameters, f"requirement {requirement_id} parameters")
        driver = _typed_id(
            parameters["driver"],
            "articulation_variable",
            f"requirement {requirement_id} driver",
        )
        closure_basis = "complete articulation driver projection"
        record = next((item for item in projection["articulation"]["drivers"] if item["driver_entity"] == driver), None)
        satisfied = False
        if record is not None and requirement_type == "driver_drives_joint":
            target = _typed_id(parameters["joint"], "joint", f"requirement {requirement_id} joint")
            for clause_id in record["drives_clause_ids"]:
                if clauses[clause_id]["object"]["physical_joint"] == target:
                    concept_ids = [clause_id]
                    satisfied = True
                    break
        elif record is not None:
            target = _typed_id(parameters["frame"], "frame", f"requirement {requirement_id} frame")
            for clause_id in record["affects_clause_ids"]:
                if clauses[clause_id]["object"]["frame"] == target:
                    concept_ids = [clause_id]
                    satisfied = True
                    break
    elif requirement_type == "frame_has_asserted_role":
        _expect_keys(parameters, {"frame", "role"}, f"requirement {requirement_id} parameters")
        _typed_id(parameters["frame"], "frame", f"requirement {requirement_id} frame")
        _text(parameters["role"], f"requirement {requirement_id} role")
        closure_basis = "open-world project semantic role assertions"
        matching = [
            role for role in projection["project_semantics"]["asserted_frame_roles"]
            if role["frame"] == parameters["frame"] and role["role"] == parameters["role"]
        ]
        satisfied = bool(matching)
        status = "not_established_open_world"
        explanation = "the required project semantic role is not asserted"
        exact = False
        modality = "project_asserted"
        if matching:
            concept_ids = [matching[0]["supporting_clause_id"]]
    elif requirement_type == "constraint_declared":
        _expect_keys(parameters, {"constraint"}, f"requirement {requirement_id} parameters")
        if not isinstance(parameters["constraint"], str) or "/" not in parameters["constraint"]:
            raise FunctionalError(f"requirement {requirement_id} constraint must be a typed constraint entity ID")
        closure_basis = "open-world supplemental mechanism constraint declarations"
        matching = [
            item for item in projection["mechanism"]["constraints"]
            if item["constraint_entity"] == parameters["constraint"]
        ]
        satisfied = bool(matching)
        status = "not_established_open_world"
        explanation = "the supplemental mechanism relation is not declared"
        exact = False
        modality = "supplemental_asserted_relation"
        if matching:
            concept_ids = [matching[0]["relation_clause_id"]]
    elif requirement_type == "finite_configuration_witness_exists":
        _expect_keys(parameters, {"chart"}, f"requirement {requirement_id} parameters")
        if not isinstance(parameters["chart"], str) or "/" not in parameters["chart"]:
            raise FunctionalError(f"requirement {requirement_id} chart must be a typed configuration chart ID")
        closure_basis = "finite declared configuration chart witnesses"
        matching = [
            chart for chart in projection["configuration"]["charts"]
            if chart["chart_entity"] == parameters["chart"]
        ]
        satisfied = bool(matching and matching[0]["nodes"])
        status = "not_established_open_world"
        explanation = "no finite satisfying witness is available in the named chart"
        modality = "finite_computed_evidence"
        if satisfied:
            concept_ids = [matching[0]["chart_clause_id"], matching[0]["nodes"][0]["supporting_clause_id"]]
    else:
        raise FunctionalError(f"requirement {requirement_id!r} uses unsupported type {requirement_type!r}")

    if satisfied:
        status = "satisfied"
        explanation = "the named requirement is grounded in the bound concept graph"
    return {
        "requirement_id": requirement_id,
        "type": requirement_type,
        "parameters": copy.deepcopy(parameters),
        "status": status,
        "satisfied": satisfied,
        "evidence": {
            "exact": exact,
            "modality": modality,
            "concept_clause_ids": sorted(set(concept_ids)),
            "closure_basis": closure_basis,
            "explanation": explanation,
        },
    }


def build_functional_model(
    canonical: dict[str, Any],
    concept_graph: dict[str, Any],
    concept_graph_artifact_sha256: str,
    spec: dict[str, Any],
    spec_sha256: str,
) -> dict[str, Any]:
    _validate_concept_input(canonical, concept_graph)
    expected_spec_keys = {
        "schema_version",
        "function_set_id",
        "source_binding",
        "object_types",
        "components",
        "functions",
        "conditions",
        "effects",
        "capabilities",
        "affordances",
        "inventory_completeness",
    }
    _expect_keys(spec, expected_spec_keys, "function/affordance spec")
    if spec["schema_version"] != SPEC_SCHEMA:
        raise FunctionalError(f"function spec must use {SPEC_SCHEMA}")
    function_set_id = _typed_id(spec["function_set_id"], "function_set", "function_set_id")
    for field in expected_spec_keys - {"schema_version", "function_set_id", "source_binding"}:
        if not isinstance(spec[field], list):
            raise FunctionalError(f"function spec {field} must be an array")

    binding = _source_binding(canonical, concept_graph, concept_graph_artifact_sha256, spec_sha256)
    _validate_declared_binding(spec, binding)
    builder = _Builder()
    builder.entity(function_set_id, ["functional_knowledge_model"], ["function-spec.json#/"])
    concept_entities = {record["entity_id"] for record in concept_graph["entities"]}

    object_types: list[dict[str, Any]] = []
    object_type_ids: set[str] = set()
    for index, raw in enumerate(spec["object_types"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"object type {index} must be an object")
        _expect_keys(raw, {"object_type_id", "meaning"}, f"object type {index}")
        entity_id = _typed_id(raw["object_type_id"], "object_type", f"object type {index}")
        if entity_id in object_type_ids:
            raise FunctionalError(f"duplicate object type {entity_id}")
        object_type_ids.add(entity_id)
        meaning = _text(raw["meaning"], f"object type {entity_id} meaning")
        builder.entity(entity_id, ["declared_object_type"], [f"function-spec.json#/object_types/{index}"])
        clause_id = builder.clause(
            "declares_object_type", entity_id, {"meaning": meaning},
            modality="project_asserted_type", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/object_types/{index}"],
            rule="copy_explicit_project_object_type",
            cnl=f"PROJECT DECLARES OBJECT TYPE {entity_id}: {meaning}.",
        )
        object_types.append({"object_type_id": entity_id, "meaning": meaning, "supporting_clause_id": clause_id})

    components: list[dict[str, Any]] = []
    component_ids: set[str] = set()
    for index, raw in enumerate(spec["components"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"component {index} must be an object")
        _expect_keys(raw, {"component_id", "members", "meaning"}, f"component {index}")
        component_id = _typed_id(raw["component_id"], "component", f"component {index}")
        if component_id in component_ids:
            raise FunctionalError(f"duplicate component {component_id}")
        members = sorted(_strings(raw["members"], f"component {component_id} members", nonempty=True))
        unknown = sorted(set(members) - concept_entities)
        if unknown:
            raise FunctionalError(f"component {component_id} references unknown concept entities {unknown}")
        meaning = _text(raw["meaning"], f"component {component_id} meaning")
        component_ids.add(component_id)
        builder.entity(component_id, ["project_declared_component"], [f"function-spec.json#/components/{index}"])
        clause_id = builder.clause(
            "declares_component_membership", component_id,
            {"members": members, "meaning": meaning},
            modality="project_asserted_component", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/components/{index}"],
            rule="copy_explicit_project_component_grouping",
            cnl=f"PROJECT DECLARES COMPONENT {component_id} WITH MEMBERS {json.dumps(members)}.",
        )
        components.append({"component_id": component_id, "members": members, "meaning": meaning, "supporting_clause_id": clause_id})

    provider_ids = concept_entities | component_ids
    functions: list[dict[str, Any]] = []
    function_ids: set[str] = set()
    for index, raw in enumerate(spec["functions"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"function {index} must be an object")
        _expect_keys(raw, {"function_id", "provided_by", "verb", "object_types", "purpose"}, f"function {index}")
        function_id = _typed_id(raw["function_id"], "function", f"function {index}")
        if function_id in function_ids:
            raise FunctionalError(f"duplicate function {function_id}")
        providers = sorted(_strings(raw["provided_by"], f"function {function_id} provided_by", nonempty=True))
        unknown_providers = sorted(set(providers) - provider_ids)
        if unknown_providers:
            raise FunctionalError(f"function {function_id} has unknown providers {unknown_providers}")
        targets = sorted(_strings(raw["object_types"], f"function {function_id} object_types"))
        unknown_types = sorted(set(targets) - object_type_ids)
        if unknown_types:
            raise FunctionalError(f"function {function_id} has unknown object types {unknown_types}")
        verb = _text(raw["verb"], f"function {function_id} verb")
        purpose = _text(raw["purpose"], f"function {function_id} purpose")
        function_ids.add(function_id)
        builder.entity(function_id, ["project_declared_function"], [f"function-spec.json#/functions/{index}"])
        value = {"provided_by": providers, "verb": verb, "object_types": targets, "purpose": purpose}
        clause_id = builder.clause(
            "declares_component_function", function_id, value,
            modality="project_asserted_function", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/functions/{index}"],
            rule="copy_explicit_project_function",
            cnl=f"PROJECT DECLARES FUNCTION {function_id} VERB {verb} PROVIDED BY {json.dumps(providers)}.",
        )
        functions.append({"function_id": function_id, **value, "supporting_clause_id": clause_id})

    conditions: list[dict[str, Any]] = []
    condition_ids: set[str] = set()
    allowed_truth_sources = {
        "runtime_observation_required",
        "planner_verification_required",
        "operator_confirmation_required",
        "project_assumption",
    }
    for index, raw in enumerate(spec["conditions"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"condition {index} must be an object")
        _expect_keys(raw, {"condition_id", "predicate", "arguments", "truth_source", "meaning"}, f"condition {index}")
        condition_id = _typed_id(raw["condition_id"], "condition", f"condition {index}")
        if condition_id in condition_ids:
            raise FunctionalError(f"duplicate condition {condition_id}")
        truth_source = raw["truth_source"]
        if truth_source not in allowed_truth_sources:
            raise FunctionalError(f"condition {condition_id} has unsupported truth_source {truth_source!r}")
        arguments = _strings(raw["arguments"], f"condition {condition_id} arguments")
        predicate = _text(raw["predicate"], f"condition {condition_id} predicate")
        meaning = _text(raw["meaning"], f"condition {condition_id} meaning")
        condition_ids.add(condition_id)
        builder.entity(condition_id, ["declared_precondition"], [f"function-spec.json#/conditions/{index}"])
        value = {"predicate": predicate, "arguments": arguments, "truth_source": truth_source, "meaning": meaning}
        clause_id = builder.clause(
            "declares_action_condition", condition_id, value,
            modality="project_asserted_condition", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/conditions/{index}"],
            rule="copy_explicit_project_condition_without_evaluating_runtime_truth",
            cnl=f"PROJECT DECLARES CONDITION {condition_id}: {predicate}({', '.join(arguments)}); truth source {truth_source}.",
        )
        conditions.append({"condition_id": condition_id, **value, "supporting_clause_id": clause_id})

    effects: list[dict[str, Any]] = []
    effect_ids: set[str] = set()
    for index, raw in enumerate(spec["effects"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"effect {index} must be an object")
        _expect_keys(raw, {"effect_id", "predicate", "arguments", "meaning"}, f"effect {index}")
        effect_id = _typed_id(raw["effect_id"], "effect", f"effect {index}")
        if effect_id in effect_ids:
            raise FunctionalError(f"duplicate effect {effect_id}")
        arguments = _strings(raw["arguments"], f"effect {effect_id} arguments")
        predicate = _text(raw["predicate"], f"effect {effect_id} predicate")
        meaning = _text(raw["meaning"], f"effect {effect_id} meaning")
        effect_ids.add(effect_id)
        builder.entity(effect_id, ["declared_intended_effect"], [f"function-spec.json#/effects/{index}"])
        value = {"predicate": predicate, "arguments": arguments, "meaning": meaning, "observed_effect": False}
        clause_id = builder.clause(
            "declares_intended_action_effect", effect_id, value,
            modality="project_asserted_effect", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/effects/{index}"],
            rule="copy_explicit_intended_effect_without_claiming_execution",
            cnl=f"PROJECT DECLARES INTENDED EFFECT {effect_id}: {predicate}({', '.join(arguments)}); not observed execution.",
        )
        effects.append({"effect_id": effect_id, **value, "supporting_clause_id": clause_id})

    capabilities: list[dict[str, Any]] = []
    capability_ids: set[str] = set()
    all_structural_clause_ids: set[str] = set()
    for index, raw in enumerate(spec["capabilities"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"capability {index} must be an object")
        _expect_keys(
            raw,
            {"capability_id", "provided_by", "realizes_functions", "enabling_requirements", "condition_refs", "limitations"},
            f"capability {index}",
        )
        capability_id = _typed_id(raw["capability_id"], "capability", f"capability {index}")
        if capability_id in capability_ids:
            raise FunctionalError(f"duplicate capability {capability_id}")
        providers = sorted(_strings(raw["provided_by"], f"capability {capability_id} provided_by", nonempty=True))
        realized = sorted(_strings(raw["realizes_functions"], f"capability {capability_id} realizes_functions", nonempty=True))
        condition_refs = sorted(_strings(raw["condition_refs"], f"capability {capability_id} condition_refs"))
        limitations = [_text(value, f"capability {capability_id} limitation") for value in _strings(raw["limitations"], f"capability {capability_id} limitations")]
        if set(providers) - provider_ids:
            raise FunctionalError(f"capability {capability_id} has unknown providers {sorted(set(providers) - provider_ids)}")
        if set(realized) - function_ids:
            raise FunctionalError(f"capability {capability_id} has unknown functions {sorted(set(realized) - function_ids)}")
        if set(condition_refs) - condition_ids:
            raise FunctionalError(f"capability {capability_id} has unknown conditions {sorted(set(condition_refs) - condition_ids)}")
        raw_requirements = raw["enabling_requirements"]
        if not isinstance(raw_requirements, list) or not raw_requirements:
            raise FunctionalError(f"capability {capability_id} must declare at least one enabling requirement")
        requirement_results = [_requirement_result(requirement, concept_graph) for requirement in raw_requirements]
        requirement_ids = [item["requirement_id"] for item in requirement_results]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise FunctionalError(f"capability {capability_id} has duplicate requirement IDs")
        capability_ids.add(capability_id)
        builder.entity(capability_id, ["project_declared_capability"], [f"function-spec.json#/capabilities/{index}"])
        declaration_value = {
            "provided_by": providers,
            "realizes_functions": realized,
            "condition_refs": condition_refs,
            "limitations": limitations,
        }
        declaration_clause = builder.clause(
            "declares_robot_capability", capability_id, declaration_value,
            modality="project_asserted_capability", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/capabilities/{index}"],
            rule="copy_explicit_project_capability",
            cnl=f"PROJECT DECLARES CAPABILITY {capability_id} PROVIDED BY {json.dumps(providers)}.",
        )
        requirement_clause_ids: list[str] = []
        for requirement_index, result in enumerate(requirement_results):
            evidence = result["evidence"]
            modality = evidence["modality"] if result["satisfied"] else (
                "derived_exact" if result["status"] == "not_satisfied_exact_closed_world" else "open_world_unknown"
            )
            concept_ids = evidence["concept_clause_ids"]
            all_structural_clause_ids.update(clause["clause_id"] for clause in _concept_closure(concept_ids, _concept_clause_map(concept_graph)))
            requirement_clause_ids.append(builder.clause(
                "has_grounded_enabling_requirement", capability_id, result,
                modality=modality,
                exact=bool(evidence["exact"] and result["status"] != "not_established_open_world"),
                source_type="concept_graph_requirement_evaluation",
                source_refs=[f"function-spec.json#/capabilities/{index}/enabling_requirements/{requirement_index}"],
                rule="evaluate_typed_requirement_against_bound_concept_projection",
                premise_clause_ids=[declaration_clause],
                concept_premise_clause_ids=concept_ids,
                cnl=(
                    f"CAPABILITY {capability_id} REQUIREMENT {result['requirement_id']} STATUS {result['status']}; "
                    "satisfaction does not verify physical execution."
                ),
            ))
        grounding_status = (
            "all_declared_requirements_grounded"
            if all(item["satisfied"] for item in requirement_results)
            else "one_or_more_declared_requirements_not_grounded"
        )
        capabilities.append({
            "capability_id": capability_id,
            **declaration_value,
            "requirements": requirement_results,
            "grounding_status": grounding_status,
            "physical_capability_verified": False,
            "declaration_clause_id": declaration_clause,
            "requirement_clause_ids": requirement_clause_ids,
        })

    affordances: list[dict[str, Any]] = []
    affordance_ids: set[str] = set()
    for index, raw in enumerate(spec["affordances"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"affordance {index} must be an object")
        _expect_keys(
            raw,
            {"affordance_id", "offered_by", "action_verb", "target_object_types", "capability_refs", "precondition_refs", "effect_refs", "meaning"},
            f"affordance {index}",
        )
        affordance_id = _typed_id(raw["affordance_id"], "affordance", f"affordance {index}")
        if affordance_id in affordance_ids:
            raise FunctionalError(f"duplicate affordance {affordance_id}")
        offered_by = sorted(_strings(raw["offered_by"], f"affordance {affordance_id} offered_by", nonempty=True))
        targets = sorted(_strings(raw["target_object_types"], f"affordance {affordance_id} targets", nonempty=True))
        capability_refs = sorted(_strings(raw["capability_refs"], f"affordance {affordance_id} capabilities", nonempty=True))
        preconditions = sorted(_strings(raw["precondition_refs"], f"affordance {affordance_id} preconditions"))
        effect_refs = sorted(_strings(raw["effect_refs"], f"affordance {affordance_id} effects", nonempty=True))
        for label, values, known in (
            ("provider", offered_by, provider_ids),
            ("target object type", targets, object_type_ids),
            ("capability", capability_refs, capability_ids),
            ("precondition", preconditions, condition_ids),
            ("effect", effect_refs, effect_ids),
        ):
            if set(values) - known:
                raise FunctionalError(f"affordance {affordance_id} has unknown {label} refs {sorted(set(values) - known)}")
        action_verb = _text(raw["action_verb"], f"affordance {affordance_id} action_verb")
        meaning = _text(raw["meaning"], f"affordance {affordance_id} meaning")
        affordance_ids.add(affordance_id)
        builder.entity(affordance_id, ["project_declared_relational_affordance"], [f"function-spec.json#/affordances/{index}"])
        value = {
            "offered_by": offered_by,
            "action_verb": action_verb,
            "target_object_types": targets,
            "capability_refs": capability_refs,
            "precondition_refs": preconditions,
            "effect_refs": effect_refs,
            "meaning": meaning,
            "relational_contract": "actor + action + target type + intended effect under named conditions",
            "current_preconditions_satisfied": "not_evaluated",
            "physical_executability": "not_established",
        }
        premises = [
            next(item["declaration_clause_id"] for item in capabilities if item["capability_id"] == ref)
            for ref in capability_refs
        ]
        premises.extend(
            next(item["supporting_clause_id"] for item in conditions if item["condition_id"] == ref)
            for ref in preconditions
        )
        premises.extend(
            next(item["supporting_clause_id"] for item in effects if item["effect_id"] == ref)
            for ref in effect_refs
        )
        clause_id = builder.clause(
            "declares_relational_affordance", affordance_id, value,
            modality="project_asserted_affordance", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/affordances/{index}"],
            rule="compose_declared_actor_action_target_effect_relation",
            premise_clause_ids=premises,
            cnl=(
                f"PROJECT DECLARES AFFORDANCE {affordance_id}: {json.dumps(offered_by)} MAY {action_verb} "
                f"{json.dumps(targets)} ONLY IF NAMED PRECONDITIONS HOLD; physical execution is not established."
            ),
        )
        affordances.append({"affordance_id": affordance_id, **value, "supporting_clause_id": clause_id})

    completeness: list[dict[str, Any]] = []
    allowed_inventories = {"functions", "capabilities", "affordances"}
    declared_complete_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(spec["inventory_completeness"]):
        if not isinstance(raw, dict):
            raise FunctionalError(f"inventory completeness record {index} must be an object")
        _expect_keys(raw, {"subject", "inventories", "scope"}, f"inventory completeness {index}")
        subject = raw["subject"]
        if subject not in provider_ids:
            raise FunctionalError(f"inventory completeness references unknown subject {subject!r}")
        inventories = sorted(_strings(raw["inventories"], f"inventory completeness {subject}", nonempty=True))
        if set(inventories) - allowed_inventories:
            raise FunctionalError(f"inventory completeness {subject} has unknown inventories")
        duplicate_pairs = sorted(
            (subject, inventory)
            for inventory in inventories
            if (subject, inventory) in declared_complete_pairs
        )
        if duplicate_pairs:
            raise FunctionalError(f"inventory completeness repeats subject/inventory declarations {duplicate_pairs}")
        declared_complete_pairs.update((subject, inventory) for inventory in inventories)
        scope = _text(raw["scope"], f"inventory completeness {subject} scope")
        clause_id = builder.clause(
            "declares_project_inventory_completeness", function_set_id,
            {"subject": subject, "inventories": inventories, "scope": scope},
            modality="project_asserted_inventory_completeness", exact=False,
            source_type="project_function_affordance_spec",
            source_refs=[f"function-spec.json#/inventory_completeness/{index}"],
            rule="copy_explicit_project_inventory_scope",
            cnl=(
                f"PROJECT DECLARES {json.dumps(inventories)} INVENTORY COMPLETE FOR {subject} ONLY WITHIN SCOPE {scope}; "
                "absence is not physical impossibility."
            ),
        )
        completeness.append({"subject": subject, "inventories": inventories, "scope": scope, "supporting_clause_id": clause_id})

    entities = [builder.entities[key] for key in sorted(builder.entities)]
    clauses = [builder.clauses[key] for key in sorted(builder.clauses)]
    known_entities = {record["entity_id"] for record in entities} | concept_entities
    by_predicate: dict[str, list[str]] = {}
    by_subject: dict[str, list[str]] = {}
    by_entity: dict[str, list[str]] = {entity: [] for entity in known_entities}

    def entity_refs(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, str) and value in known_entities:
            found.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                found.update(entity_refs(child))
        elif isinstance(value, list):
            for child in value:
                found.update(entity_refs(child))
        return found

    for clause in clauses:
        clause_id = clause["clause_id"]
        by_predicate.setdefault(clause["predicate"], []).append(clause_id)
        by_subject.setdefault(clause["subject"], []).append(clause_id)
        for entity in {clause["subject"]} | entity_refs(clause["object"]):
            by_entity[entity].append(clause_id)
    structural_clauses = _concept_closure(all_structural_clause_ids, _concept_clause_map(concept_graph))
    if not capabilities:
        status = "no_capabilities_declared"
    elif all(item["grounding_status"] == "all_declared_requirements_grounded" for item in capabilities):
        status = "all_declared_capabilities_structurally_grounded"
    else:
        status = "one_or_more_declared_capabilities_not_grounded"
    body = {
        "schema_version": MODEL_SCHEMA,
        "functional_model_id": (
            f"functional_model/{function_set_id.removeprefix('function_set/')}/"
            f"{_sha256_bytes(_canonical_bytes(binding))[:16]}"
        ),
        "function_set_id": function_set_id,
        "source_binding": binding,
        "ontology_contract": {
            "component_function_capability_affordance_are_distinct": True,
            "affordance_is_relational_actor_action_target_effect": True,
            "structural_grounding_is_not_physical_capability_verification": True,
            "intended_effect_is_not_observed_effect": True,
            "inventory_absence_is_physical_impossibility": False,
            "condition_and_effect_arguments_are_declared_symbolic_tokens": True,
            "modalities": [
                "project_asserted_type",
                "project_asserted_component",
                "project_asserted_function",
                "project_asserted_capability",
                "project_asserted_condition",
                "project_asserted_effect",
                "project_asserted_affordance",
                "project_asserted_inventory_completeness",
                "derived_exact",
                "project_asserted",
                "supplemental_asserted_relation",
                "finite_computed_evidence",
                "open_world_unknown",
            ],
        },
        "entities": entities,
        "clauses": clauses,
        "structural_evidence_clauses": structural_clauses,
        "indexes": {
            "by_predicate": {key: sorted(value) for key, value in sorted(by_predicate.items())},
            "by_subject": {key: sorted(value) for key, value in sorted(by_subject.items())},
            "by_entity": {key: sorted(value) for key, value in sorted(by_entity.items())},
        },
        "projections": {
            "object_types": sorted(object_types, key=lambda item: item["object_type_id"]),
            "components": sorted(components, key=lambda item: item["component_id"]),
            "functions": sorted(functions, key=lambda item: item["function_id"]),
            "conditions": sorted(conditions, key=lambda item: item["condition_id"]),
            "effects": sorted(effects, key=lambda item: item["effect_id"]),
            "capabilities": sorted(capabilities, key=lambda item: item["capability_id"]),
            "affordances": sorted(affordances, key=lambda item: item["affordance_id"]),
            "inventory_completeness": sorted(completeness, key=lambda item: (item["subject"], item["inventories"])),
        },
        "query_contract": {
            "schema_version": QUERY_SCHEMA,
            "intents": {
                "describe_component": {"component": "typed component ID"},
                "explain_function": {"function": "typed function ID"},
                "explain_capability": {"capability": "typed capability ID"},
                "explain_affordance": {"affordance": "typed affordance ID"},
                "what_is_entity_for": {"entity": "typed concept or component ID"},
                "can_perform_action": {
                    "offered_by": "typed provider ID",
                    "action_verb": "exact declared verb",
                    "target_object_type": "typed object_type ID",
                },
            },
        },
        "status": status,
        "coverage": {
            "component_count": len(components),
            "function_count": len(functions),
            "capability_count": len(capabilities),
            "affordance_count": len(affordances),
            "condition_count": len(conditions),
            "effect_count": len(effects),
            "requirement_count": sum(len(item["requirements"]) for item in capabilities),
            "satisfied_requirement_count": sum(
                1 for item in capabilities for requirement in item["requirements"] if requirement["satisfied"]
            ),
            "all_declared_capabilities_structurally_grounded": status in {
                "no_capabilities_declared",
                "all_declared_capabilities_structurally_grounded",
            },
        },
        "epistemic_scope": (
            "project-declared component, function, capability, condition, intended-effect, and relational-affordance knowledge "
            "grounded against the exact bound structural concept graph. Structural requirement satisfaction is deterministic "
            "within that represented model, but function/capability/affordance intent remains asserted; runtime preconditions, "
            "observed effects, physical executability, hardware behavior, and safety are not established."
        ),
    }
    body["functional_model_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def write_functional_model(
    output_path: Path,
    canonical: dict[str, Any],
    concept_graph: dict[str, Any],
    concept_graph_artifact_sha256: str,
    spec_path: Path,
) -> dict[str, Any]:
    spec = _read_json(spec_path, "function/affordance spec")
    model = build_functional_model(
        canonical,
        concept_graph,
        concept_graph_artifact_sha256,
        spec,
        _sha256_path(spec_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_json_bytes(model))
    return model


def _load_context(context_directory: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    canonical = _read_json(context_directory / "model.json", "canonical model")
    concept_record = canonical.get("artifacts", {}).get("concept_graph")
    if not isinstance(concept_record, dict) or not isinstance(concept_record.get("path"), str) or not isinstance(concept_record.get("sha256"), str):
        raise FunctionalError("canonical model has no valid concept graph artifact binding")
    concept_path = context_directory / concept_record["path"]
    actual_sha = _sha256_path(concept_path)
    if actual_sha != concept_record["sha256"]:
        raise FunctionalError("bound concept graph artifact digest mismatch")
    concept_graph = _read_json(concept_path, "concept graph")
    return canonical, concept_graph, actual_sha


def write_functional_model_from_context(
    context_directory: Path,
    spec_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    canonical, concept_graph, artifact_sha = _load_context(context_directory)
    return write_functional_model(output_path, canonical, concept_graph, artifact_sha, spec_path)


def _validate_functional_model_structure(model: dict[str, Any]) -> None:
    expected = {
        "schema_version", "functional_model_id", "function_set_id", "source_binding", "ontology_contract",
        "entities", "clauses", "structural_evidence_clauses", "indexes", "projections", "query_contract",
        "status", "coverage", "epistemic_scope", "functional_model_sha256",
    }
    _expect_keys(model, expected, "functional model")
    if model["schema_version"] != MODEL_SCHEMA:
        raise FunctionalError(f"functional model must use {MODEL_SCHEMA}")
    body = {key: value for key, value in model.items() if key != "functional_model_sha256"}
    if model["functional_model_sha256"] != _sha256_bytes(_canonical_bytes(body)):
        raise FunctionalError("functional model semantic digest is invalid")
    if not isinstance(model["entities"], list) or not isinstance(model["clauses"], list):
        raise FunctionalError("functional model entities/clauses must be arrays")
    entity_ids: set[str] = set()
    for index, entity in enumerate(model["entities"]):
        if not isinstance(entity, dict):
            raise FunctionalError(f"functional entity {index} is malformed")
        _expect_keys(entity, {"entity_id", "entity_types", "source_refs"}, f"functional entity {index}")
        if not isinstance(entity["entity_id"], str) or "/" not in entity["entity_id"]:
            raise FunctionalError(f"functional entity {index} has an invalid typed ID")
        if entity["entity_types"] != sorted(_strings(entity["entity_types"], f"functional entity {index} types", nonempty=True)):
            raise FunctionalError(f"functional entity {index} types are not canonical")
        if entity["source_refs"] != sorted(_strings(entity["source_refs"], f"functional entity {index} source refs", nonempty=True)):
            raise FunctionalError(f"functional entity {index} source refs are not canonical")
        if entity["entity_id"] in entity_ids:
            raise FunctionalError(f"duplicate functional entity {entity['entity_id']!r}")
        entity_ids.add(entity["entity_id"])
    if [item["entity_id"] for item in model["entities"]] != sorted(entity_ids):
        raise FunctionalError("functional entities are not in canonical order")
    clause_ids: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for index, clause in enumerate(model["clauses"]):
        if not isinstance(clause, dict):
            raise FunctionalError(f"functional clause {index} is malformed")
        _expect_keys(
            clause,
            {"clause_id", "predicate", "subject", "object", "modality", "evidence", "proof", "cnl"},
            f"functional clause {index}",
        )
        clause_id = clause["clause_id"]
        if clause_id in clause_ids or clause["subject"] not in entity_ids:
            raise FunctionalError(f"invalid functional clause identity/subject at {index}")
        if clause_id != _clause_id({key: value for key, value in clause.items() if key != "clause_id"}):
            raise FunctionalError(f"functional clause {clause_id!r} content digest is invalid")
        if not isinstance(clause["predicate"], str) or not clause["predicate"]:
            raise FunctionalError(f"functional clause {clause_id!r} predicate is invalid")
        if not isinstance(clause["modality"], str) or not isinstance(clause["cnl"], str):
            raise FunctionalError(f"functional clause {clause_id!r} modality/CNL is invalid")
        evidence = clause["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"exact", "source_type", "source_refs"}:
            raise FunctionalError(f"functional clause {clause_id!r} evidence is malformed")
        if not isinstance(evidence["exact"], bool) or not isinstance(evidence["source_type"], str):
            raise FunctionalError(f"functional clause {clause_id!r} evidence types are malformed")
        if evidence["source_refs"] != sorted(_strings(evidence["source_refs"], f"functional clause {clause_id} source refs", nonempty=True)):
            raise FunctionalError(f"functional clause {clause_id!r} source refs are not canonical")
        clause_ids.add(clause_id)
        by_id[clause_id] = clause
    if [item["clause_id"] for item in model["clauses"]] != sorted(clause_ids):
        raise FunctionalError("functional clauses are not in canonical order")
    structural_ids: set[str] = set()
    structural_by_id: dict[str, dict[str, Any]] = {}
    for clause in model["structural_evidence_clauses"]:
        clause_id = clause.get("clause_id") if isinstance(clause, dict) else None
        if not isinstance(clause_id, str) or clause_id in structural_ids:
            raise FunctionalError("structural evidence clauses are malformed or duplicated")
        body = {key: value for key, value in clause.items() if key != "clause_id"}
        expected_id = f"concept_clause/{clause.get('predicate')}/{_sha256_bytes(_canonical_bytes(body))[:20]}"
        if clause_id != expected_id:
            raise FunctionalError(f"structural evidence clause {clause_id!r} content digest is invalid")
        structural_ids.add(clause_id)
        structural_by_id[clause_id] = clause
    if [item["clause_id"] for item in model["structural_evidence_clauses"]] != sorted(structural_ids):
        raise FunctionalError("structural evidence clauses are not in canonical order")
    for clause in model["clauses"]:
        proof = clause["proof"]
        if not isinstance(proof, dict) or set(proof) != {
            "rule", "premise_clause_ids", "concept_premise_clause_ids"
        }:
            raise FunctionalError(f"functional clause {clause['clause_id']} proof is malformed")
        for field in ("premise_clause_ids", "concept_premise_clause_ids"):
            values = proof[field]
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise FunctionalError(f"functional clause {clause['clause_id']} proof {field} is malformed")
            if len(values) != len(set(values)):
                raise FunctionalError(f"functional clause {clause['clause_id']} proof {field} has duplicates")
        if set(proof["premise_clause_ids"]) - clause_ids:
            raise FunctionalError(f"functional clause {clause['clause_id']} has missing functional premises")
        if set(proof["concept_premise_clause_ids"]) - structural_ids:
            raise FunctionalError(f"functional clause {clause['clause_id']} has missing structural premises")
    for clause in model["structural_evidence_clauses"]:
        proof = clause.get("proof")
        if not isinstance(proof, dict) or not isinstance(proof.get("premise_clause_ids"), list):
            raise FunctionalError(f"structural evidence clause {clause['clause_id']} proof is malformed")
        missing = set(proof["premise_clause_ids"]) - structural_ids
        if missing:
            raise FunctionalError(f"structural evidence clause {clause['clause_id']} has missing recursive premises")

    def reject_cycles(records: dict[str, dict[str, Any]], label: str) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(clause_id: str) -> None:
            if clause_id in visiting:
                raise FunctionalError(f"{label} proof graph contains a cycle at {clause_id}")
            if clause_id in visited:
                return
            visiting.add(clause_id)
            for premise in records[clause_id]["proof"]["premise_clause_ids"]:
                visit(premise)
            visiting.remove(clause_id)
            visited.add(clause_id)

        for clause_id in records:
            visit(clause_id)

    reject_cycles(by_id, "functional")
    reject_cycles(structural_by_id, "structural")
    indexes = model["indexes"]
    if not isinstance(indexes, dict) or set(indexes) != {"by_predicate", "by_subject", "by_entity"}:
        raise FunctionalError("functional indexes are malformed")
    for name, mapping in indexes.items():
        if not isinstance(mapping, dict):
            raise FunctionalError(f"functional index {name} must be an object")
        for key, ids in mapping.items():
            if not isinstance(key, str) or not isinstance(ids, list) or set(ids) - clause_ids:
                raise FunctionalError(f"functional index {name}/{key} is invalid")
            if ids != sorted(set(ids)):
                raise FunctionalError(f"functional index {name}/{key} is not canonical")
    if not entity_ids.issubset(indexes["by_entity"]):
        raise FunctionalError("functional by_entity index omits declared functional entities")
    indexed_entities = set(indexes["by_entity"])

    def indexed_refs(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value in indexed_entities else set()
        if isinstance(value, dict):
            return set().union(*(indexed_refs(child) for child in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(indexed_refs(child) for child in value)) if value else set()
        return set()

    expected_by_predicate: dict[str, list[str]] = {}
    expected_by_subject: dict[str, list[str]] = {}
    expected_by_entity: dict[str, list[str]] = {entity: [] for entity in indexed_entities}
    for clause in model["clauses"]:
        clause_id = clause["clause_id"]
        expected_by_predicate.setdefault(clause["predicate"], []).append(clause_id)
        expected_by_subject.setdefault(clause["subject"], []).append(clause_id)
        for entity in {clause["subject"]} | indexed_refs(clause["object"]):
            expected_by_entity[entity].append(clause_id)
    exact_indexes = {
        "by_predicate": {key: sorted(value) for key, value in sorted(expected_by_predicate.items())},
        "by_subject": {key: sorted(value) for key, value in sorted(expected_by_subject.items())},
        "by_entity": {key: sorted(value) for key, value in sorted(expected_by_entity.items())},
    }
    if indexes != exact_indexes:
        raise FunctionalError("functional indexes do not exactly match the clauses")
    projections = model["projections"]
    if not isinstance(projections, dict) or set(projections) != {
        "object_types", "components", "functions", "conditions", "effects", "capabilities", "affordances",
        "inventory_completeness",
    }:
        raise FunctionalError("functional projections are malformed")

    projection_ids: dict[str, set[str]] = {}

    def projection_records(collection: str, id_key: str, prefix: str) -> list[dict[str, Any]]:
        records = projections[collection]
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise FunctionalError(f"functional projection {collection} must be an array of objects")
        ids = [_typed_id(record.get(id_key), prefix, f"{collection} {id_key}") for record in records]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise FunctionalError(f"functional projection {collection} is duplicated or not canonical")
        if set(ids) - entity_ids:
            raise FunctionalError(f"functional projection {collection} references missing functional entities")
        projection_ids[collection] = set(ids)
        return records

    def supporting_clause(
        clause_id: Any,
        predicate: str,
        subject: str,
        object_value: Any,
        label: str,
    ) -> dict[str, Any]:
        clause = by_id.get(clause_id)
        if clause is None:
            raise FunctionalError(f"{label} references missing supporting clause {clause_id!r}")
        if clause["predicate"] != predicate or clause["subject"] != subject or clause["object"] != object_value:
            raise FunctionalError(f"{label} does not match its supporting clause")
        return clause

    simple_contracts = (
        ("object_types", "object_type_id", "object_type", "declares_object_type"),
        ("components", "component_id", "component", "declares_component_membership"),
        ("functions", "function_id", "function", "declares_component_function"),
        ("conditions", "condition_id", "condition", "declares_action_condition"),
        ("effects", "effect_id", "effect", "declares_intended_action_effect"),
        ("affordances", "affordance_id", "affordance", "declares_relational_affordance"),
    )
    for collection, id_key, prefix, predicate in simple_contracts:
        for record in projection_records(collection, id_key, prefix):
            _expect_keys(
                record,
                {
                    "object_types": {"object_type_id", "meaning", "supporting_clause_id"},
                    "components": {"component_id", "members", "meaning", "supporting_clause_id"},
                    "functions": {"function_id", "provided_by", "verb", "object_types", "purpose", "supporting_clause_id"},
                    "conditions": {"condition_id", "predicate", "arguments", "truth_source", "meaning", "supporting_clause_id"},
                    "effects": {"effect_id", "predicate", "arguments", "meaning", "observed_effect", "supporting_clause_id"},
                    "affordances": {
                        "affordance_id", "offered_by", "action_verb", "target_object_types", "capability_refs",
                        "precondition_refs", "effect_refs", "meaning", "relational_contract",
                        "current_preconditions_satisfied", "physical_executability", "supporting_clause_id",
                    },
                }[collection],
                f"functional projection {collection}/{record[id_key]}",
            )
            object_value = {
                key: value for key, value in record.items() if key not in {id_key, "supporting_clause_id"}
            }
            clause = supporting_clause(
                record["supporting_clause_id"],
                predicate,
                record[id_key],
                object_value,
                f"functional projection {collection}/{record[id_key]}",
            )
            if collection == "effects" and record["observed_effect"] is not False:
                raise FunctionalError("intended effect projection must not claim an observed effect")
            if collection == "affordances":
                if record["current_preconditions_satisfied"] != "not_evaluated" or record["physical_executability"] != "not_established":
                    raise FunctionalError("affordance projection overclaims precondition truth or physical execution")
                expected_premises = {
                    next(item["declaration_clause_id"] for item in projections["capabilities"] if item["capability_id"] == ref)
                    for ref in record["capability_refs"]
                }
                expected_premises.update(
                    next(item["supporting_clause_id"] for item in projections["conditions"] if item["condition_id"] == ref)
                    for ref in record["precondition_refs"]
                )
                expected_premises.update(
                    next(item["supporting_clause_id"] for item in projections["effects"] if item["effect_id"] == ref)
                    for ref in record["effect_refs"]
                )
                if set(clause["proof"]["premise_clause_ids"]) != expected_premises:
                    raise FunctionalError(f"affordance {record[id_key]} proof premises do not match its declared relation")

    capabilities = projection_records("capabilities", "capability_id", "capability")
    for record in capabilities:
        _expect_keys(
            record,
            {
                "capability_id", "provided_by", "realizes_functions", "condition_refs", "limitations",
                "requirements", "grounding_status", "physical_capability_verified", "declaration_clause_id",
                "requirement_clause_ids",
            },
            f"functional capability {record['capability_id']}",
        )
        declaration_value = {
            key: record[key] for key in ("provided_by", "realizes_functions", "condition_refs", "limitations")
        }
        supporting_clause(
            record["declaration_clause_id"],
            "declares_robot_capability",
            record["capability_id"],
            declaration_value,
            f"functional capability {record['capability_id']}",
        )
        requirements = record["requirements"]
        if not isinstance(requirements, list) or not requirements:
            raise FunctionalError(f"functional capability {record['capability_id']} has no requirement results")
        if not isinstance(record["requirement_clause_ids"], list) or len(record["requirement_clause_ids"]) != len(requirements):
            raise FunctionalError(f"functional capability {record['capability_id']} requirement clause count mismatch")
        requirement_ids: set[str] = set()
        for requirement, clause_id in zip(requirements, record["requirement_clause_ids"]):
            if not isinstance(requirement, dict):
                raise FunctionalError(f"functional capability {record['capability_id']} has malformed requirement")
            _expect_keys(requirement, {"requirement_id", "type", "parameters", "status", "satisfied", "evidence"}, "capability requirement result")
            requirement_id = _typed_id(requirement["requirement_id"], "requirement", "capability requirement_id")
            if requirement_id in requirement_ids:
                raise FunctionalError(f"functional capability {record['capability_id']} repeats {requirement_id}")
            requirement_ids.add(requirement_id)
            if requirement["status"] == "satisfied" and requirement["satisfied"] is not True:
                raise FunctionalError(f"requirement {requirement_id} status/satisfaction mismatch")
            if requirement["status"] != "satisfied" and requirement["satisfied"] is not False:
                raise FunctionalError(f"requirement {requirement_id} status/satisfaction mismatch")
            evidence = requirement["evidence"]
            _expect_keys(evidence, {"exact", "modality", "concept_clause_ids", "closure_basis", "explanation"}, f"requirement {requirement_id} evidence")
            clause = supporting_clause(
                clause_id,
                "has_grounded_enabling_requirement",
                record["capability_id"],
                requirement,
                f"requirement {requirement_id}",
            )
            if clause["proof"]["concept_premise_clause_ids"] != evidence["concept_clause_ids"]:
                raise FunctionalError(f"requirement {requirement_id} structural premises do not match its evidence")
        expected_grounding = (
            "all_declared_requirements_grounded"
            if all(requirement["satisfied"] for requirement in requirements)
            else "one_or_more_declared_requirements_not_grounded"
        )
        if record["grounding_status"] != expected_grounding or record["physical_capability_verified"] is not False:
            raise FunctionalError(f"functional capability {record['capability_id']} grounding or physical boundary is invalid")

    inventory = projections["inventory_completeness"]
    if not isinstance(inventory, list) or not all(isinstance(record, dict) for record in inventory):
        raise FunctionalError("inventory completeness projection is malformed")
    seen_complete: set[tuple[str, str]] = set()
    for record in inventory:
        _expect_keys(record, {"subject", "inventories", "scope", "supporting_clause_id"}, "inventory completeness projection")
        supporting_clause(
            record["supporting_clause_id"],
            "declares_project_inventory_completeness",
            model["function_set_id"],
            {key: record[key] for key in ("subject", "inventories", "scope")},
            f"inventory completeness {record['subject']}",
        )
        for inventory_name in record["inventories"]:
            pair = (record["subject"], inventory_name)
            if pair in seen_complete:
                raise FunctionalError(f"inventory completeness repeats {pair}")
            seen_complete.add(pair)

    expected_status = (
        "no_capabilities_declared"
        if not capabilities
        else (
            "all_declared_capabilities_structurally_grounded"
            if all(item["grounding_status"] == "all_declared_requirements_grounded" for item in capabilities)
            else "one_or_more_declared_capabilities_not_grounded"
        )
    )
    if model["status"] != expected_status:
        raise FunctionalError("functional model status does not match capability grounding")
    expected_coverage = {
        "component_count": len(projections["components"]),
        "function_count": len(projections["functions"]),
        "capability_count": len(projections["capabilities"]),
        "affordance_count": len(projections["affordances"]),
        "condition_count": len(projections["conditions"]),
        "effect_count": len(projections["effects"]),
        "requirement_count": sum(len(item["requirements"]) for item in projections["capabilities"]),
        "satisfied_requirement_count": sum(
            1 for item in projections["capabilities"] for requirement in item["requirements"] if requirement["satisfied"]
        ),
        "all_declared_capabilities_structurally_grounded": model["status"] in {
            "no_capabilities_declared",
            "all_declared_capabilities_structurally_grounded",
        },
    }
    if model["coverage"] != expected_coverage:
        raise FunctionalError("functional model coverage does not match its contents")

    expected_ontology = {
        "component_function_capability_affordance_are_distinct": True,
        "affordance_is_relational_actor_action_target_effect": True,
        "structural_grounding_is_not_physical_capability_verification": True,
        "intended_effect_is_not_observed_effect": True,
        "inventory_absence_is_physical_impossibility": False,
        "condition_and_effect_arguments_are_declared_symbolic_tokens": True,
        "modalities": [
            "project_asserted_type",
            "project_asserted_component",
            "project_asserted_function",
            "project_asserted_capability",
            "project_asserted_condition",
            "project_asserted_effect",
            "project_asserted_affordance",
            "project_asserted_inventory_completeness",
            "derived_exact",
            "project_asserted",
            "supplemental_asserted_relation",
            "finite_computed_evidence",
            "open_world_unknown",
        ],
    }
    if model["ontology_contract"] != expected_ontology:
        raise FunctionalError("functional ontology contract is invalid")
    if any(clause["modality"] not in expected_ontology["modalities"] for clause in model["clauses"]):
        raise FunctionalError("functional clause uses a modality outside the ontology contract")
    expected_query_contract = {
        "schema_version": QUERY_SCHEMA,
        "intents": {
            "describe_component": {"component": "typed component ID"},
            "explain_function": {"function": "typed function ID"},
            "explain_capability": {"capability": "typed capability ID"},
            "explain_affordance": {"affordance": "typed affordance ID"},
            "what_is_entity_for": {"entity": "typed concept or component ID"},
            "can_perform_action": {
                "offered_by": "typed provider ID",
                "action_verb": "exact declared verb",
                "target_object_type": "typed object_type ID",
            },
        },
    }
    if model["query_contract"] != expected_query_contract:
        raise FunctionalError("functional query contract is invalid")
    _typed_id(model["function_set_id"], "function_set", "functional model function_set_id")
    binding = model["source_binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "function_spec_sha256", "urdf_semantic_sha256", "articulation_grammar_sha256",
        "constraint_graph_sha256", "configuration_atlas_sha256", "concept_graph",
    }:
        raise FunctionalError("functional model source binding is malformed")
    for key in ("function_spec_sha256", "urdf_semantic_sha256", "articulation_grammar_sha256"):
        if not isinstance(binding[key], str) or len(binding[key]) != 64:
            raise FunctionalError(f"functional model source binding {key} is malformed")
    for key in ("constraint_graph_sha256", "configuration_atlas_sha256"):
        if binding[key] is not None and (not isinstance(binding[key], str) or len(binding[key]) != 64):
            raise FunctionalError(f"functional model source binding {key} is malformed")
    if not isinstance(binding["concept_graph"], dict) or set(binding["concept_graph"]) != {
        "concept_graph_id", "concept_graph_sha256", "artifact_sha256",
    }:
        raise FunctionalError("functional model concept graph binding is malformed")
    expected_model_id = (
        f"functional_model/{model['function_set_id'].removeprefix('function_set/')}/"
        f"{_sha256_bytes(_canonical_bytes(binding))[:16]}"
    )
    if model["functional_model_id"] != expected_model_id:
        raise FunctionalError("functional model ID does not match its source binding")


def _validate_functional_model(model: dict[str, Any]) -> None:
    try:
        _validate_functional_model_structure(model)
    except FunctionalError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, StopIteration) as error:
        raise FunctionalError(f"functional model structure is malformed: {error}") from error


def read_functional_model(path: Path) -> dict[str, Any]:
    model = _read_json(path, "functional model")
    _validate_functional_model(model)
    return model


def _functional_closure(model: dict[str, Any], clause_ids: Iterable[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    functional_by_id = {clause["clause_id"]: clause for clause in model["clauses"]}
    structural_by_id = {clause["clause_id"]: clause for clause in model["structural_evidence_clauses"]}
    selected: set[str] = set()
    structural_selected: set[str] = set()
    queue = deque(clause_ids)
    while queue:
        clause_id = queue.popleft()
        if clause_id in selected:
            continue
        clause = functional_by_id.get(clause_id)
        if clause is None:
            raise FunctionalError(f"query selected missing functional clause {clause_id}")
        selected.add(clause_id)
        queue.extend(clause["proof"]["premise_clause_ids"])
        structural_selected.update(clause["proof"]["concept_premise_clause_ids"])
    structural_queue = deque(structural_selected)
    while structural_queue:
        clause_id = structural_queue.popleft()
        clause = structural_by_id.get(clause_id)
        if clause is None:
            raise FunctionalError(f"query selected missing structural clause {clause_id}")
        for premise in clause["proof"]["premise_clause_ids"]:
            if premise not in structural_selected:
                structural_selected.add(premise)
                structural_queue.append(premise)
    return (
        [functional_by_id[clause_id] for clause_id in sorted(selected)],
        [structural_by_id[clause_id] for clause_id in sorted(structural_selected)],
    )


def _resolve(model: dict[str, Any], supplied: str, prefixes: tuple[str, ...] | None = None) -> str:
    functional_entities = [item["entity_id"] for item in model["entities"]]
    structural_entities = list(model["indexes"]["by_entity"])
    candidates = sorted(set(functional_entities) | set(structural_entities))
    if prefixes is not None:
        candidates = [value for value in candidates if value.startswith(prefixes)]
    if supplied in candidates:
        return supplied
    matches = [value for value in candidates if value.rsplit("/", 1)[-1] == supplied]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FunctionalError(f"unknown functional or structural entity {supplied!r}")
    raise FunctionalError(f"ambiguous bare entity {supplied!r}; candidates={matches}")


def query_functional_model(model: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    _validate_functional_model(model)
    _expect_keys(query, {"schema_version", "query_id", "intent", "parameters"}, "functional query")
    if query["schema_version"] != QUERY_SCHEMA:
        raise FunctionalError(f"functional query must use {QUERY_SCHEMA}")
    query_id = _text(query["query_id"], "functional query_id")
    intent = query["intent"]
    parameters = query["parameters"]
    if not isinstance(intent, str) or not isinstance(parameters, dict):
        raise FunctionalError("functional query intent/parameters are malformed")
    projections = model["projections"]
    selected: list[str] = []
    unknowns: list[str] = []

    def one(collection: str, key: str, entity_id: str) -> dict[str, Any]:
        record = next((item for item in projections[collection] if item[key] == entity_id), None)
        if record is None:
            raise FunctionalError(f"projection {collection} has no record for {entity_id}")
        return record

    if intent == "describe_component":
        _expect_keys(parameters, {"component"}, "describe_component parameters")
        entity = _resolve(model, parameters["component"], ("component/",))
        component = one("components", "component_id", entity)
        functions = [item for item in projections["functions"] if entity in item["provided_by"]]
        capabilities = [item for item in projections["capabilities"] if entity in item["provided_by"]]
        affordances = [item for item in projections["affordances"] if entity in item["offered_by"]]
        selected = [component["supporting_clause_id"]]
        selected += [item["supporting_clause_id"] for item in functions]
        selected += [item["declaration_clause_id"] for item in capabilities]
        selected += [item["supporting_clause_id"] for item in affordances]
        answer = {
            "component": component,
            "functions": functions,
            "capabilities": capabilities,
            "affordances": affordances,
        }
        answer_cnl = f"Component {entity} has {len(functions)} declared functions, {len(capabilities)} capabilities, and {len(affordances)} relational affordances."
    elif intent == "explain_function":
        _expect_keys(parameters, {"function"}, "explain_function parameters")
        entity = _resolve(model, parameters["function"], ("function/",))
        record = one("functions", "function_id", entity)
        selected = [record["supporting_clause_id"]]
        answer = record
        answer_cnl = f"Function {entity} is project-declared, not inferred from structure or names."
    elif intent == "explain_capability":
        _expect_keys(parameters, {"capability"}, "explain_capability parameters")
        entity = _resolve(model, parameters["capability"], ("capability/",))
        record = one("capabilities", "capability_id", entity)
        selected = [record["declaration_clause_id"], *record["requirement_clause_ids"]]
        answer = record
        unknowns = [
            "runtime condition truth is not evaluated",
            "physical capability, controller execution, hardware behavior, and safety are not established",
        ]
        answer_cnl = f"Capability {entity} has grounding status {record['grounding_status']}; this is not physical capability verification."
    elif intent == "explain_affordance":
        _expect_keys(parameters, {"affordance"}, "explain_affordance parameters")
        entity = _resolve(model, parameters["affordance"], ("affordance/",))
        record = one("affordances", "affordance_id", entity)
        selected = [record["supporting_clause_id"]]
        answer = record
        unknowns = [
            "current precondition satisfaction is not evaluated",
            "the intended effect is not an observed effect",
            "physical executability and safety are not established",
        ]
        answer_cnl = f"Affordance {entity} is a project-declared actor-action-target-effect relation conditional on named preconditions."
    elif intent == "what_is_entity_for":
        _expect_keys(parameters, {"entity"}, "what_is_entity_for parameters")
        entity = _resolve(model, parameters["entity"])
        components = [item for item in projections["components"] if entity == item["component_id"] or entity in item["members"]]
        providers = {entity, *(item["component_id"] for item in components)}
        functions = [item for item in projections["functions"] if providers & set(item["provided_by"])]
        capabilities = [item for item in projections["capabilities"] if providers & set(item["provided_by"])]
        affordances = [item for item in projections["affordances"] if providers & set(item["offered_by"])]
        selected = [item["supporting_clause_id"] for item in components]
        selected += [item["supporting_clause_id"] for item in functions]
        selected += [item["declaration_clause_id"] for item in capabilities]
        selected += [item["supporting_clause_id"] for item in affordances]
        answer = {
            "entity": entity,
            "declared_components": [item["component_id"] for item in components],
            "declared_functions": [item["function_id"] for item in functions],
            "declared_capabilities": [item["capability_id"] for item in capabilities],
            "declared_affordances": [item["affordance_id"] for item in affordances],
            "name_based_inference_used": False,
        }
        if not selected:
            unknowns.append("no function inventory for this entity is established by the available project specification")
        answer_cnl = f"Entity {entity} has only the explicitly returned project function knowledge; no role was guessed from its name."
    elif intent == "can_perform_action":
        _expect_keys(parameters, {"offered_by", "action_verb", "target_object_type"}, "can_perform_action parameters")
        provider = _resolve(model, parameters["offered_by"])
        action = _text(parameters["action_verb"], "can_perform_action action_verb")
        target_type = _resolve(model, parameters["target_object_type"], ("object_type/",))
        matches = [
            item for item in projections["affordances"]
            if provider in item["offered_by"] and item["action_verb"] == action and target_type in item["target_object_types"]
        ]
        if matches:
            selected = [item["supporting_clause_id"] for item in matches]
            capability_records = [
                one("capabilities", "capability_id", capability_id)
                for item in matches for capability_id in item["capability_refs"]
            ]
            selected += [clause_id for item in capability_records for clause_id in item["requirement_clause_ids"]]
            capabilities_by_id = {item["capability_id"]: item for item in capability_records}
            grounded_matches = [
                item["affordance_id"]
                for item in matches
                if all(
                    capabilities_by_id[capability_id]["grounding_status"] == "all_declared_requirements_grounded"
                    for capability_id in item["capability_refs"]
                )
            ]
            conclusion = (
                "declared_possible_if_preconditions_hold"
                if grounded_matches
                else "declared_affordance_with_ungrounded_capability_requirements"
            )
            answer = {
                "conclusion": conclusion,
                "matching_affordances": [item["affordance_id"] for item in matches],
                "structurally_grounded_matching_affordances": grounded_matches,
                "precondition_refs": sorted({ref for item in matches for ref in item["precondition_refs"]}),
                "effect_refs": sorted({ref for item in matches for ref in item["effect_refs"]}),
                "capability_grounding": {
                    item["capability_id"]: item["grounding_status"] for item in capability_records
                },
                "current_preconditions_satisfied": "not_evaluated",
                "physical_executability": "not_established",
            }
            unknowns = ["runtime/planner/operator preconditions must be evaluated before action execution"]
            if not grounded_matches:
                unknowns.append("no matching affordance has all referenced capability requirements structurally grounded")
                answer_cnl = f"The project declares an affordance for {provider} to {action} {target_type}, but no match has all capability requirements structurally grounded; execution is not established."
            else:
                answer_cnl = f"The project model declares that {provider} may {action} {target_type} if all named preconditions hold; execution is not verified."
        else:
            complete = next((
                item for item in projections["inventory_completeness"]
                if item["subject"] == provider and "affordances" in item["inventories"]
            ), None)
            if complete is not None:
                selected = [complete["supporting_clause_id"]]
                conclusion = "not_declared_in_complete_project_inventory"
                answer_cnl = f"No matching affordance is declared for {provider} within the explicitly complete project inventory; physical impossibility is not established."
            else:
                conclusion = "unknown_not_in_incomplete_inventory"
                unknowns.append("the affordance inventory is not declared complete for this provider")
                answer_cnl = f"No matching affordance is present, but the project inventory is incomplete; answer unknown."
            answer = {
                "conclusion": conclusion,
                "matching_affordances": [],
                "physical_impossibility": "not_established",
            }
    else:
        raise FunctionalError(f"unsupported functional query intent {intent!r}")
    functional_clauses, structural_clauses = _functional_closure(model, selected)
    return {
        "schema_version": ANSWER_SCHEMA,
        "status": "answered" if not (intent == "can_perform_action" and answer.get("conclusion") == "unknown_not_in_incomplete_inventory") else "unknown",
        "query_id": query_id,
        "intent": intent,
        "functional_model": {
            "functional_model_id": model["functional_model_id"],
            "functional_model_sha256": model["functional_model_sha256"],
        },
        "answer": answer,
        "answer_cnl": answer_cnl,
        "supporting_clauses": functional_clauses,
        "structural_supporting_clauses": structural_clauses,
        "unknowns": unknowns,
        "source_binding": model["source_binding"],
        "epistemic_scope": model["epistemic_scope"],
    }


def query_functional_model_files(model_path: Path, query_path: Path) -> dict[str, Any]:
    model = read_functional_model(model_path)
    query = _read_json(query_path, "functional query")
    return query_functional_model(model, query)


def build_functional_model_from_context(context_directory: Path, spec_path: Path) -> dict[str, Any]:
    canonical, concept_graph, artifact_sha = _load_context(context_directory)
    spec = _read_json(spec_path, "function/affordance spec")
    return build_functional_model(canonical, concept_graph, artifact_sha, spec, _sha256_path(spec_path))


def verify_functional_model(
    context_directory: Path,
    spec_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    try:
        stored = read_functional_model(model_path)
    except FunctionalError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "exact_regeneration_match": False,
            "issues": [{"check": "stored_model_structure", "message": str(error)}],
        }
    try:
        expected = build_functional_model_from_context(context_directory, spec_path)
    except FunctionalError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "functional_model_id": stored["functional_model_id"],
            "exact_regeneration_match": False,
            "issues": [{"check": "regeneration", "message": str(error)}],
        }
    exact = _json_bytes(stored) == _json_bytes(expected)
    if not exact:
        issues.append({"check": "exact_regeneration", "message": "stored functional model differs from exact regeneration"})
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "functional_model_id": stored["functional_model_id"],
        "functional_model_sha256": stored["functional_model_sha256"],
        "exact_regeneration_match": exact,
        "validated_functional_clause_count": len(stored["clauses"]),
        "validated_structural_evidence_clause_count": len(stored["structural_evidence_clauses"]),
        "all_declared_capabilities_structurally_grounded": stored["coverage"]["all_declared_capabilities_structurally_grounded"],
        "issues": issues,
        "epistemic_scope": (
            "exact regeneration verifies project functional assertions and structural requirement grounding against the same "
            "bound concept graph; it does not validate runtime preconditions, observed effects, physical capability, hardware, or safety"
        ),
    }
