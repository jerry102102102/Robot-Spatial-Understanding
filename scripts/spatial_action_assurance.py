#!/usr/bin/env python3
"""Compile and query replayable evidence for one declared robot action instance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from spatial_functional import FunctionalError, query_functional_model, read_functional_model


BUNDLE_SCHEMA = "robot-spatial-action-evidence-bundle.v1"
SOURCE_SCHEMA = "robot-spatial-action-evidence-source.v1"
MODEL_SCHEMA = "robot-spatial-action-assurance.v1"
QUERY_SCHEMA = "robot-spatial-action-assurance-query.v1"
ANSWER_SCHEMA = "robot-spatial-action-assurance-answer.v1"
VERIFICATION_SCHEMA = "robot-spatial-action-assurance-verification.v1"

CONDITION_EVIDENCE_TYPE = {
    "runtime_observation_required": "runtime_observation",
    "planner_verification_required": "planner_verification",
    "operator_confirmation_required": "operator_confirmation",
    "project_assumption": "project_assumption",
}
TRUTH_VALUES = {"true", "false", "unknown"}
LIFECYCLE_CONTRACT = {
    "goal_response": {
        "subject_ref": "lifecycle/goal_response",
        "predicate": "goal_response",
        "values": {"accepted", "rejected"},
    },
    "action_status": {
        "subject_ref": "lifecycle/action_status",
        "predicate": "action_status",
        "values": {"accepted", "executing", "canceling", "succeeded", "aborted", "canceled"},
    },
    "action_result": {
        "subject_ref": "lifecycle/action_result",
        "predicate": "action_result",
        "values": {"succeeded", "aborted", "canceled"},
    },
}
EVIDENCE_TYPES = set(CONDITION_EVIDENCE_TYPE.values()) | {"effect_observation", *LIFECYCLE_CONTRACT}


class ActionAssuranceError(ValueError):
    """An invalid action-evidence bundle, evidence source, assurance model, or query."""


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
        raise ActionAssuranceError(f"cannot read {path}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActionAssuranceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ActionAssuranceError(f"{label} must contain one JSON object")
    return value


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ActionAssuranceError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActionAssuranceError(f"{label} must be a non-empty string")
    return value


def _typed_id(value: Any, prefix: str, label: str) -> str:
    result = _text(value, label)
    if not result.startswith(f"{prefix}/") or result.endswith("/"):
        raise ActionAssuranceError(f"{label} must use typed prefix {prefix}/")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ActionAssuranceError(f"{label} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, label)


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ActionAssuranceError(f"{label} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise ActionAssuranceError(f"{label} must not contain duplicates")
    return list(value)


def _string_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in value.items()
    ):
        raise ActionAssuranceError(f"{label} must map non-empty strings to non-empty strings")
    return dict(value)


def _clock(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ActionAssuranceError(f"{label} must be an object")
    _expect_keys(value, {"domain", "unit", "epoch"}, label)
    result = {key: _text(value[key], f"{label}.{key}") for key in ("domain", "unit", "epoch")}
    if result["unit"] != "nanoseconds":
        raise ActionAssuranceError(f"{label}.unit must be 'nanoseconds'")
    return result


def _resolve_under(base: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ActionAssuranceError(f"{label} must be relative to the evidence bundle")
    root = base.resolve()
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise ActionAssuranceError(f"{label} must resolve to a regular non-symlink file")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ActionAssuranceError(f"{label} escapes the evidence bundle directory") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ActionAssuranceError(f"{label} must resolve to a regular non-symlink file")
    return resolved


def _normalize_record(
    raw: Any,
    source_id: str,
    source_binding: dict[str, Any],
    producer: dict[str, str],
    index: int,
) -> dict[str, Any]:
    label = f"evidence source {source_id} record {index}"
    if not isinstance(raw, dict):
        raise ActionAssuranceError(f"{label} must be an object")
    _expect_keys(
        raw,
        {
            "record_id",
            "evidence_type",
            "subject_ref",
            "predicate",
            "bindings",
            "value",
            "observed_at_ns",
            "valid_until_ns",
            "claim_scope",
            "limitations",
        },
        label,
    )
    record_id = _typed_id(raw["record_id"], "evidence", f"{label}.record_id")
    evidence_type = _text(raw["evidence_type"], f"{label}.evidence_type")
    if evidence_type not in EVIDENCE_TYPES:
        raise ActionAssuranceError(f"{label} has unsupported evidence_type {evidence_type!r}")
    subject_ref = _text(raw["subject_ref"], f"{label}.subject_ref")
    predicate = _text(raw["predicate"], f"{label}.predicate")
    bindings = _string_map(raw["bindings"], f"{label}.bindings")
    value = _text(raw["value"], f"{label}.value")
    observed_at_ns = _nonnegative_int(raw["observed_at_ns"], f"{label}.observed_at_ns")
    valid_until_ns = _optional_nonnegative_int(raw["valid_until_ns"], f"{label}.valid_until_ns")
    if valid_until_ns is not None and valid_until_ns < observed_at_ns:
        raise ActionAssuranceError(f"{label}.valid_until_ns precedes observed_at_ns")
    limitations = _strings(raw["limitations"], f"{label}.limitations")
    claim_scope = _text(raw["claim_scope"], f"{label}.claim_scope")
    if evidence_type in set(CONDITION_EVIDENCE_TYPE.values()) | {"effect_observation"}:
        if value not in TRUTH_VALUES:
            raise ActionAssuranceError(f"{label}.value must be true, false, or unknown")
        expected_prefix = "condition/" if evidence_type != "effect_observation" else "effect/"
        if not subject_ref.startswith(expected_prefix):
            raise ActionAssuranceError(f"{label}.subject_ref must use {expected_prefix}")
    else:
        contract = LIFECYCLE_CONTRACT[evidence_type]
        if subject_ref != contract["subject_ref"] or predicate != contract["predicate"]:
            raise ActionAssuranceError(f"{label} does not match the {evidence_type} lifecycle contract")
        if value not in contract["values"]:
            raise ActionAssuranceError(f"{label}.value is invalid for {evidence_type}")
        if set(bindings) != {"action_instance"}:
            raise ActionAssuranceError(f"{label}.bindings must contain exactly action_instance")
    return {
        "record_id": record_id,
        "evidence_type": evidence_type,
        "subject_ref": subject_ref,
        "predicate": predicate,
        "bindings": dict(sorted(bindings.items())),
        "value": value,
        "observed_at_ns": observed_at_ns,
        "valid_until_ns": valid_until_ns,
        "claim_scope": claim_scope,
        "limitations": sorted(limitations),
        "source": {
            "source_id": source_id,
            "producer": producer,
            "artifact_path": source_binding["path"],
            "artifact_sha256": source_binding["sha256"],
        },
    }


def _load_sources(
    bundle_path: Path,
    references: Any,
    bundle_clock: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(references, list) or not references:
        raise ActionAssuranceError("action evidence bundle evidence_sources must be a non-empty array")
    bindings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    paths: set[str] = set()
    record_ids: set[str] = set()
    for index, reference in enumerate(references):
        label = f"evidence source reference {index}"
        if not isinstance(reference, dict):
            raise ActionAssuranceError(f"{label} must be an object")
        _expect_keys(reference, {"source_id", "path", "sha256"}, label)
        source_id = _typed_id(reference["source_id"], "evidence_source", f"{label}.source_id")
        relative = _text(reference["path"], f"{label}.path")
        expected_sha = _text(reference["sha256"], f"{label}.sha256")
        if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
            raise ActionAssuranceError(f"{label}.sha256 must be lowercase SHA-256")
        if source_id in source_ids or relative in paths:
            raise ActionAssuranceError("action evidence bundle repeats an evidence source ID or path")
        source_ids.add(source_id)
        paths.add(relative)
        path = _resolve_under(bundle_path.parent, relative, f"{label}.path")
        actual_sha = _sha256_path(path)
        if actual_sha != expected_sha:
            raise ActionAssuranceError(
                f"evidence source digest mismatch for {relative!r}; expected {expected_sha}, got {actual_sha}"
            )
        source = _read_json(path, "action evidence source")
        _expect_keys(source, {"schema_version", "source_id", "clock", "producer", "records"}, f"evidence source {source_id}")
        if source["schema_version"] != SOURCE_SCHEMA or source["source_id"] != source_id:
            raise ActionAssuranceError(f"evidence source {relative!r} schema or source_id mismatch")
        source_clock = _clock(source["clock"], f"evidence source {source_id}.clock")
        if source_clock != bundle_clock:
            raise ActionAssuranceError(f"evidence source {source_id} clock does not match the bundle clock")
        producer = source["producer"]
        if not isinstance(producer, dict):
            raise ActionAssuranceError(f"evidence source {source_id}.producer must be an object")
        _expect_keys(producer, {"producer_id", "producer_type"}, f"evidence source {source_id}.producer")
        normalized_producer = {
            "producer_id": _text(producer["producer_id"], f"evidence source {source_id}.producer_id"),
            "producer_type": _text(producer["producer_type"], f"evidence source {source_id}.producer_type"),
        }
        if not isinstance(source["records"], list) or not source["records"]:
            raise ActionAssuranceError(f"evidence source {source_id}.records must be a non-empty array")
        binding = {"source_id": source_id, "path": relative, "sha256": actual_sha}
        bindings.append(binding)
        for record_index, raw_record in enumerate(source["records"]):
            record = _normalize_record(raw_record, source_id, binding, normalized_producer, record_index)
            if record["record_id"] in record_ids:
                raise ActionAssuranceError(f"duplicate evidence record ID {record['record_id']}")
            record_ids.add(record["record_id"])
            records.append(record)
    return (
        sorted(bindings, key=lambda item: item["source_id"]),
        sorted(records, key=lambda item: (item["observed_at_ns"], item["record_id"])),
    )


def _functional_basis(functional_model: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    projections = functional_model["projections"]
    affordance = next(
        (item for item in projections["affordances"] if item["affordance_id"] == action["affordance_id"]),
        None,
    )
    if affordance is None:
        raise ActionAssuranceError(f"functional model has no affordance {action['affordance_id']}")
    if action["offered_by"] not in affordance["offered_by"]:
        raise ActionAssuranceError("action offered_by is not a provider of the selected affordance")
    if action["action_verb"] != affordance["action_verb"]:
        raise ActionAssuranceError("action verb does not match the selected affordance")
    if action["target_object_type"] not in affordance["target_object_types"]:
        raise ActionAssuranceError("action target object type does not match the selected affordance")
    condition_by_id = {item["condition_id"]: item for item in projections["conditions"]}
    effect_by_id = {item["effect_id"]: item for item in projections["effects"]}
    capability_by_id = {item["capability_id"]: item for item in projections["capabilities"]}
    conditions = [condition_by_id[identifier] for identifier in affordance["precondition_refs"]]
    effects = [effect_by_id[identifier] for identifier in affordance["effect_refs"]]
    required_tokens = {"actor", "target"}
    for record in [*conditions, *effects]:
        required_tokens.update(record["arguments"])
    if set(action["argument_bindings"]) != required_tokens:
        raise ActionAssuranceError(
            "action argument_bindings must contain exactly actor, target, and every selected condition/effect token; "
            f"expected={sorted(required_tokens)}, got={sorted(action['argument_bindings'])}"
        )
    if action["argument_bindings"]["actor"] != action["offered_by"]:
        raise ActionAssuranceError("action actor binding must equal offered_by")
    if action["argument_bindings"]["target"] != action["target_instance_id"]:
        raise ActionAssuranceError("action target binding must equal target_instance_id")
    try:
        action_answer = query_functional_model(functional_model, {
            "schema_version": "robot-spatial-functional-query.v1",
            "query_id": f"action-assurance/{action['action_instance_id'].removeprefix('action_instance/')}",
            "intent": "can_perform_action",
            "parameters": {
                "offered_by": action["offered_by"],
                "action_verb": action["action_verb"],
                "target_object_type": action["target_object_type"],
            },
        })
    except FunctionalError as error:
        raise ActionAssuranceError(f"cannot query bound functional model: {error}") from error
    if affordance["affordance_id"] not in action_answer["answer"].get("matching_affordances", []):
        raise ActionAssuranceError("selected affordance is absent from the bound action query result")
    capabilities = [capability_by_id[identifier] for identifier in affordance["capability_refs"]]
    return {
        "affordance": {
            key: affordance[key]
            for key in (
                "affordance_id",
                "offered_by",
                "action_verb",
                "target_object_types",
                "capability_refs",
                "precondition_refs",
                "effect_refs",
                "meaning",
                "supporting_clause_id",
            )
        },
        "conditions": [
            {key: item[key] for key in ("condition_id", "predicate", "arguments", "truth_source", "meaning", "supporting_clause_id")}
            for item in conditions
        ],
        "effects": [
            {key: item[key] for key in ("effect_id", "predicate", "arguments", "meaning", "supporting_clause_id")}
            for item in effects
        ],
        "capabilities": [
            {
                "capability_id": item["capability_id"],
                "grounding_status": item["grounding_status"],
                "physical_capability_verified": item["physical_capability_verified"],
                "declaration_clause_id": item["declaration_clause_id"],
                "requirement_clause_ids": item["requirement_clause_ids"],
            }
            for item in capabilities
        ],
        "action_query": {
            "conclusion": action_answer["answer"]["conclusion"],
            "matching_affordances": action_answer["answer"].get("matching_affordances", []),
            "structurally_grounded_matching_affordances": action_answer["answer"].get(
                "structurally_grounded_matching_affordances", []
            ),
            "capability_grounding": action_answer["answer"].get("capability_grounding", {}),
            "functional_clause_ids": [item["clause_id"] for item in action_answer["supporting_clauses"]],
            "structural_clause_ids": [item["clause_id"] for item in action_answer["structural_supporting_clauses"]],
        },
    }


def _expected_bindings(symbols: Iterable[str], action_bindings: dict[str, str]) -> dict[str, str]:
    return {symbol: action_bindings[symbol] for symbol in symbols}


def _truth_evaluation(
    declared: dict[str, Any],
    records: list[dict[str, Any]],
    expected_type: str,
    expected_bindings: dict[str, str],
    reference_time_ns: int,
    maximum_age_ns: int | None,
    identity_key: str,
) -> dict[str, Any]:
    subject_id = declared[identity_key]
    same_subject = [record for record in records if record["subject_ref"] == subject_id]
    same_predicate = [record for record in same_subject if record["predicate"] == declared["predicate"]]
    same_bindings = [record for record in same_predicate if record["bindings"] == expected_bindings]
    correct_type = [record for record in same_bindings if record["evidence_type"] == expected_type]
    eligible: list[dict[str, Any]] = []
    ignored = {"future": [], "expired": [], "stale": [], "wrong_evidence_type": [], "binding_mismatch": []}
    ignored["wrong_evidence_type"] = [
        record["record_id"] for record in same_bindings if record["evidence_type"] != expected_type
    ]
    ignored["binding_mismatch"] = [
        record["record_id"] for record in same_predicate if record["bindings"] != expected_bindings
    ]
    for record in correct_type:
        if record["observed_at_ns"] > reference_time_ns:
            ignored["future"].append(record["record_id"])
        elif record["valid_until_ns"] is not None and record["valid_until_ns"] < reference_time_ns:
            ignored["expired"].append(record["record_id"])
        elif maximum_age_ns is not None and reference_time_ns - record["observed_at_ns"] > maximum_age_ns:
            ignored["stale"].append(record["record_id"])
        else:
            eligible.append(record)
    selected: list[dict[str, Any]] = []
    if eligible:
        latest_time = max(record["observed_at_ns"] for record in eligible)
        selected = [record for record in eligible if record["observed_at_ns"] == latest_time]
        values = {record["value"] for record in selected}
        if len(values) > 1:
            status = "unknown_conflicting_latest_evidence"
            truth = "unknown"
        else:
            truth = next(iter(values))
            status = {"true": "satisfied", "false": "not_satisfied", "unknown": "unknown_reported"}[truth]
    else:
        truth = "unknown"
        if ignored["wrong_evidence_type"]:
            status = "unknown_wrong_evidence_type"
        elif ignored["binding_mismatch"]:
            status = "unknown_binding_mismatch"
        elif ignored["expired"]:
            status = "unknown_expired_evidence"
        elif ignored["stale"]:
            status = "unknown_stale_evidence"
        elif ignored["future"]:
            status = "unknown_future_only"
        else:
            status = "unknown_missing_evidence"
    return {
        identity_key: subject_id,
        "predicate": declared["predicate"],
        "declared_arguments": declared["arguments"],
        "bound_arguments": expected_bindings,
        "required_evidence_type": expected_type,
        "reference_time_ns": reference_time_ns,
        "maximum_age_ns": maximum_age_ns,
        "status": status,
        "truth": truth,
        "selected_record_ids": sorted(record["record_id"] for record in selected),
        "selected_records": sorted(selected, key=lambda item: item["record_id"]),
        "ignored_record_ids": {key: sorted(value) for key, value in ignored.items()},
        "evidence_artifact_integrity_verified": True,
        "producer_truthfulness_verified": False,
    }


def _lifecycle_projection(
    records: list[dict[str, Any]],
    action: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    expected_bindings = {"action_instance": action["action_instance_id"]}
    lifecycle = [
        record
        for record in records
        if record["evidence_type"] in LIFECYCLE_CONTRACT and record["bindings"] == expected_bindings
    ]
    eligible = [record for record in lifecycle if record["observed_at_ns"] <= action["evaluation_time_ns"]]
    future = [record["record_id"] for record in lifecycle if record["observed_at_ns"] > action["evaluation_time_ns"]]
    eligible.sort(key=lambda item: (item["observed_at_ns"], item["record_id"]))
    issues: list[dict[str, Any]] = []

    def distinct_values(kind: str) -> set[str]:
        return {record["value"] for record in eligible if record["evidence_type"] == kind}

    goal_records = [record for record in eligible if record["evidence_type"] == "goal_response"]
    status_records = [record for record in eligible if record["evidence_type"] == "action_status"]
    result_records = [record for record in eligible if record["evidence_type"] == "action_result"]
    if len(distinct_values("goal_response")) > 1:
        issues.append({"code": "conflicting_goal_responses", "record_ids": [item["record_id"] for item in goal_records]})
    if len(distinct_values("action_result")) > 1:
        issues.append({"code": "conflicting_action_results", "record_ids": [item["record_id"] for item in result_records]})
    goal_value = goal_records[-1]["value"] if goal_records and len(distinct_values("goal_response")) == 1 else None
    result_value = result_records[-1]["value"] if result_records and len(distinct_values("action_result")) == 1 else None
    if goal_value == "rejected" and (status_records or result_records):
        issues.append({"code": "lifecycle_after_rejected_goal", "record_ids": [item["record_id"] for item in [*status_records, *result_records]]})
    if policy["require_goal_acceptance_before_status"] and (status_records or result_records):
        accepted_times = [item["observed_at_ns"] for item in goal_records if item["value"] == "accepted"]
        if not accepted_times:
            issues.append({"code": "status_or_result_without_goal_acceptance", "record_ids": [item["record_id"] for item in [*status_records, *result_records]]})
        else:
            acceptance_time = min(accepted_times)
            earlier = [item["record_id"] for item in [*status_records, *result_records] if item["observed_at_ns"] < acceptance_time]
            if earlier:
                issues.append({"code": "status_or_result_precedes_goal_acceptance", "record_ids": earlier})
    allowed_next = {
        "accepted": {"accepted", "executing", "canceling", "succeeded", "aborted", "canceled"},
        "executing": {"executing", "canceling", "succeeded", "aborted", "canceled"},
        "canceling": {"canceling", "canceled", "aborted"},
        "succeeded": {"succeeded"},
        "aborted": {"aborted"},
        "canceled": {"canceled"},
    }
    for before, after in zip(status_records, status_records[1:]):
        if after["value"] not in allowed_next[before["value"]]:
            issues.append({
                "code": "invalid_observed_status_transition",
                "record_ids": [before["record_id"], after["record_id"]],
            })
    terminal_statuses = [item for item in status_records if item["value"] in {"succeeded", "aborted", "canceled"}]
    if policy["require_terminal_result_status_match"] and result_value is not None and terminal_statuses:
        if terminal_statuses[-1]["value"] != result_value:
            issues.append({
                "code": "terminal_status_result_mismatch",
                "record_ids": [terminal_statuses[-1]["record_id"], result_records[-1]["record_id"]],
            })
    execution_statuses = [
        item for item in status_records if item["value"] in {"executing", "canceling", "succeeded", "aborted", "canceled"}
    ]
    execution_started_at_ns = execution_statuses[0]["observed_at_ns"] if execution_statuses else None
    if issues:
        lifecycle_status = "inconsistent_lifecycle_evidence"
    elif goal_value == "rejected":
        lifecycle_status = "goal_rejected"
    elif goal_value != "accepted":
        lifecycle_status = "goal_response_not_observed"
    elif result_value is not None:
        lifecycle_status = f"result_{result_value}"
    elif status_records:
        lifecycle_status = f"status_{status_records[-1]['value']}"
    else:
        lifecycle_status = "goal_accepted_no_execution_status"
    return {
        "status": lifecycle_status,
        "consistency": "passed" if not issues else "failed",
        "issues": issues,
        "goal_response": goal_value or "not_observed",
        "latest_observed_status": status_records[-1]["value"] if status_records else "not_observed",
        "terminal_result": result_value or "not_observed",
        "execution_started_observed": execution_started_at_ns is not None,
        "execution_started_at_ns": execution_started_at_ns,
        "selected_records": eligible,
        "selected_record_ids": [item["record_id"] for item in eligible],
        "future_record_ids": sorted(future),
        "action_server_reports_are_independent_physical_verification": False,
    }


def _derive_assurance(derivation_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    action = derivation_input["action_instance"]
    policy = derivation_input["evidence_policy"]
    basis = derivation_input["functional_basis"]
    records = derivation_input["evidence_records"]
    condition_evaluations = []
    for condition in basis["conditions"]:
        expected_type = CONDITION_EVIDENCE_TYPE[condition["truth_source"]]
        condition_evaluations.append(_truth_evaluation(
            condition,
            records,
            expected_type,
            _expected_bindings(condition["arguments"], action["argument_bindings"]),
            action["decision_time_ns"],
            policy["maximum_age_ns"][expected_type],
            "condition_id",
        ))
    false_count = sum(item["status"] == "not_satisfied" for item in condition_evaluations)
    satisfied_count = sum(item["status"] == "satisfied" for item in condition_evaluations)
    unknown_count = len(condition_evaluations) - false_count - satisfied_count
    if false_count:
        precondition_status = "one_or_more_not_satisfied"
    elif unknown_count:
        precondition_status = "incomplete_evidence"
    else:
        precondition_status = "all_satisfied"
    structurally_grounded = (
        basis["affordance"]["affordance_id"]
        in basis["action_query"]["structurally_grounded_matching_affordances"]
    )
    if not structurally_grounded:
        readiness = "not_ready_ungrounded_capability_requirements"
    elif precondition_status == "one_or_more_not_satisfied":
        readiness = "not_ready_declared_precondition_false"
    elif precondition_status == "incomplete_evidence":
        readiness = "not_ready_missing_stale_conflicting_or_invalid_evidence"
    else:
        readiness = "ready_under_declared_model_and_evidence"
    lifecycle = _lifecycle_projection(records, action, policy)
    effect_evaluations = []
    for effect in basis["effects"]:
        evaluation = _truth_evaluation(
            effect,
            records,
            "effect_observation",
            _expected_bindings(effect["arguments"], action["argument_bindings"]),
            action["evaluation_time_ns"],
            None,
            "effect_id",
        )
        if evaluation["selected_records"]:
            selected_time = evaluation["selected_records"][0]["observed_at_ns"]
            start = lifecycle["execution_started_at_ns"]
            if start is None:
                temporal_relation = "execution_start_not_observed"
            elif selected_time < start:
                temporal_relation = "before_observed_execution_start"
            else:
                temporal_relation = "at_or_after_observed_execution_start"
        else:
            temporal_relation = "no_selected_effect_record"
        evaluation["temporal_relation_to_execution"] = temporal_relation
        evaluation["counts_as_post_execution_effect_evidence"] = (
            temporal_relation == "at_or_after_observed_execution_start"
        )
        evaluation["caused_by_action"] = "not_established"
        effect_evaluations.append(evaluation)
    post_effects = [item for item in effect_evaluations if item["counts_as_post_execution_effect_evidence"]]
    if len(post_effects) == len(effect_evaluations) and all(item["truth"] == "true" for item in post_effects):
        effect_status = "all_declared_effects_observed_true_after_execution_started"
    elif any(item["truth"] == "false" for item in post_effects):
        effect_status = "one_or_more_declared_effects_observed_false_after_execution_started"
    else:
        effect_status = "incomplete_or_temporally_unlinked_effect_evidence"
    terminal = lifecycle["terminal_result"]
    if lifecycle["consistency"] == "failed":
        outcome = "inconsistent_lifecycle_evidence"
    elif terminal == "succeeded":
        if effect_status == "all_declared_effects_observed_true_after_execution_started":
            outcome = "action_server_reported_success_and_all_declared_effects_observed_after_execution_started"
        elif effect_status == "one_or_more_declared_effects_observed_false_after_execution_started":
            outcome = "action_server_reported_success_but_declared_effect_observation_false"
        else:
            outcome = "action_server_reported_success_effect_evidence_incomplete_or_temporally_unlinked"
    elif terminal in {"aborted", "canceled"}:
        outcome = f"action_server_reported_{terminal}"
    else:
        outcome = "no_terminal_action_result_observed"
    discrepancies = list(lifecycle["issues"])
    if lifecycle["goal_response"] == "accepted" and readiness != "ready_under_declared_model_and_evidence":
        discrepancies.append({
            "code": "goal_accepted_without_complete_declared_readiness_evidence",
            "readiness_conclusion": readiness,
        })
    if terminal == "succeeded" and effect_status != "all_declared_effects_observed_true_after_execution_started":
        discrepancies.append({"code": "reported_success_without_complete_positive_declared_effect_evidence"})
    projections = {
        "declared_action": {
            "action_instance_id": action["action_instance_id"],
            "affordance_id": action["affordance_id"],
            "offered_by": action["offered_by"],
            "action_verb": action["action_verb"],
            "target_object_type": action["target_object_type"],
            "target_instance_id": action["target_instance_id"],
            "argument_bindings": action["argument_bindings"],
            "functional_action_conclusion": basis["action_query"]["conclusion"],
            "structurally_grounded": structurally_grounded,
            "capability_grounding": basis["action_query"]["capability_grounding"],
            "physical_executability": "not_established",
        },
        "preconditions": condition_evaluations,
        "precondition_summary": {
            "status": precondition_status,
            "declared_count": len(condition_evaluations),
            "satisfied_count": satisfied_count,
            "not_satisfied_count": false_count,
            "unknown_count": unknown_count,
            "decision_time_ns": action["decision_time_ns"],
        },
        "readiness": {
            "conclusion": readiness,
            "authorization_to_dispatch": "not_provided",
            "physical_executability": "not_established",
            "safety": "not_established",
        },
        "lifecycle": lifecycle,
        "effects": effect_evaluations,
        "effect_summary": {
            "status": effect_status,
            "declared_count": len(effect_evaluations),
            "post_execution_evidence_count": len(post_effects),
            "causal_attribution": "not_established",
        },
        "outcome": {
            "conclusion": outcome,
            "reported_terminal_result": terminal,
            "causal_success": "not_established",
            "physical_world_truth": "not_established",
            "safety": "not_established",
        },
        "discrepancies": discrepancies,
    }
    selected_ids = {
        record_id
        for evaluation in [*condition_evaluations, *effect_evaluations]
        for record_id in evaluation["selected_record_ids"]
    } | set(lifecycle["selected_record_ids"])
    coverage = {
        "evidence_source_count": len({item["source"]["source_id"] for item in records}),
        "evidence_record_count": len(records),
        "selected_evidence_record_count": len(selected_ids),
        "declared_precondition_count": len(condition_evaluations),
        "satisfied_precondition_count": satisfied_count,
        "unknown_precondition_count": unknown_count,
        "declared_effect_count": len(effect_evaluations),
        "post_execution_effect_evidence_count": len(post_effects),
        "ready_under_declared_model_and_evidence": readiness == "ready_under_declared_model_and_evidence",
        "terminal_result_observed": terminal != "not_observed",
        "all_declared_effects_observed_true_after_execution_started": (
            effect_status == "all_declared_effects_observed_true_after_execution_started"
        ),
    }
    return projections, coverage


def _normalize_bundle(
    functional_model_path: Path,
    bundle_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        functional_model = read_functional_model(functional_model_path)
    except FunctionalError as error:
        raise ActionAssuranceError(f"cannot read functional model: {error}") from error
    bundle = _read_json(bundle_path, "action evidence bundle")
    _expect_keys(
        bundle,
        {
            "schema_version",
            "bundle_id",
            "functional_model_binding",
            "clock",
            "action_instance",
            "evidence_policy",
            "evidence_sources",
        },
        "action evidence bundle",
    )
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ActionAssuranceError(f"action evidence bundle must use {BUNDLE_SCHEMA}")
    bundle_id = _typed_id(bundle["bundle_id"], "action_evidence_bundle", "bundle_id")
    binding = bundle["functional_model_binding"]
    if not isinstance(binding, dict):
        raise ActionAssuranceError("functional_model_binding must be an object")
    _expect_keys(
        binding,
        {"functional_model_id", "functional_model_sha256", "functional_model_artifact_sha256"},
        "functional_model_binding",
    )
    actual_artifact_sha = _sha256_path(functional_model_path)
    expected_binding = {
        "functional_model_id": functional_model["functional_model_id"],
        "functional_model_sha256": functional_model["functional_model_sha256"],
        "functional_model_artifact_sha256": actual_artifact_sha,
    }
    if binding != expected_binding:
        raise ActionAssuranceError(
            f"functional model binding mismatch; expected={expected_binding}, supplied={binding}"
        )
    clock = _clock(bundle["clock"], "action evidence bundle clock")
    raw_action = bundle["action_instance"]
    if not isinstance(raw_action, dict):
        raise ActionAssuranceError("action_instance must be an object")
    _expect_keys(
        raw_action,
        {
            "action_instance_id",
            "affordance_id",
            "offered_by",
            "action_verb",
            "target_object_type",
            "target_instance_id",
            "argument_bindings",
            "requested_at_ns",
            "decision_time_ns",
            "evaluation_time_ns",
        },
        "action_instance",
    )
    action = {
        "action_instance_id": _typed_id(raw_action["action_instance_id"], "action_instance", "action_instance_id"),
        "affordance_id": _typed_id(raw_action["affordance_id"], "affordance", "action affordance_id"),
        "offered_by": _text(raw_action["offered_by"], "action offered_by"),
        "action_verb": _text(raw_action["action_verb"], "action action_verb"),
        "target_object_type": _typed_id(raw_action["target_object_type"], "object_type", "action target_object_type"),
        "target_instance_id": _typed_id(raw_action["target_instance_id"], "object_instance", "action target_instance_id"),
        "argument_bindings": dict(sorted(_string_map(raw_action["argument_bindings"], "action argument_bindings").items())),
        "requested_at_ns": _nonnegative_int(raw_action["requested_at_ns"], "action requested_at_ns"),
        "decision_time_ns": _nonnegative_int(raw_action["decision_time_ns"], "action decision_time_ns"),
        "evaluation_time_ns": _nonnegative_int(raw_action["evaluation_time_ns"], "action evaluation_time_ns"),
    }
    if not action["offered_by"].startswith(("component/", "link/", "frame/")):
        raise ActionAssuranceError("action offered_by must be a typed component/link/frame provider")
    if not action["requested_at_ns"] <= action["decision_time_ns"] <= action["evaluation_time_ns"]:
        raise ActionAssuranceError("action times must satisfy requested_at <= decision_time <= evaluation_time")
    raw_policy = bundle["evidence_policy"]
    if not isinstance(raw_policy, dict):
        raise ActionAssuranceError("evidence_policy must be an object")
    _expect_keys(
        raw_policy,
        {"maximum_age_ns", "require_goal_acceptance_before_status", "require_terminal_result_status_match"},
        "evidence_policy",
    )
    maximum_age = raw_policy["maximum_age_ns"]
    if not isinstance(maximum_age, dict) or set(maximum_age) != set(CONDITION_EVIDENCE_TYPE.values()):
        raise ActionAssuranceError(
            f"evidence_policy.maximum_age_ns must contain exactly {sorted(CONDITION_EVIDENCE_TYPE.values())}"
        )
    policy = {
        "maximum_age_ns": {
            key: _nonnegative_int(maximum_age[key], f"maximum_age_ns.{key}")
            for key in sorted(maximum_age)
        },
        "require_goal_acceptance_before_status": raw_policy["require_goal_acceptance_before_status"],
        "require_terminal_result_status_match": raw_policy["require_terminal_result_status_match"],
    }
    if not isinstance(policy["require_goal_acceptance_before_status"], bool) or not isinstance(
        policy["require_terminal_result_status_match"], bool
    ):
        raise ActionAssuranceError("evidence_policy lifecycle controls must be booleans")
    source_bindings, records = _load_sources(bundle_path, bundle["evidence_sources"], clock)
    for record in records:
        if record["evidence_type"] in LIFECYCLE_CONTRACT and record["bindings"]["action_instance"] != action["action_instance_id"]:
            raise ActionAssuranceError(
                f"lifecycle evidence {record['record_id']} binds a different action instance"
            )
    normalized_bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "functional_model_binding": expected_binding,
        "clock": clock,
        "action_instance": action,
        "evidence_policy": policy,
        "evidence_sources": source_bindings,
    }
    basis = _functional_basis(functional_model, action)
    return functional_model, normalized_bundle, basis, source_bindings, records


def build_action_assurance(functional_model_path: Path, bundle_path: Path) -> dict[str, Any]:
    functional_model, bundle, basis, source_bindings, records = _normalize_bundle(functional_model_path, bundle_path)
    derivation_input = {
        "action_instance": bundle["action_instance"],
        "evidence_policy": bundle["evidence_policy"],
        "functional_basis": basis,
        "evidence_records": records,
    }
    projections, coverage = _derive_assurance(derivation_input)
    bundle_sha = _sha256_path(bundle_path)
    identity_digest = _sha256_bytes(_canonical_bytes({
        "functional_model_sha256": functional_model["functional_model_sha256"],
        "bundle_sha256": bundle_sha,
        "action_instance_id": bundle["action_instance"]["action_instance_id"],
    }))[:16]
    body = {
        "schema_version": MODEL_SCHEMA,
        "assurance_id": (
            f"action_assurance/{bundle['action_instance']['action_instance_id'].removeprefix('action_instance/')}/"
            f"{identity_digest}"
        ),
        "functional_model_binding": bundle["functional_model_binding"],
        "action_evidence_binding": {
            "bundle_id": bundle["bundle_id"],
            "bundle_sha256": bundle_sha,
            "evidence_sources": source_bindings,
        },
        "clock": bundle["clock"],
        "derivation_input": derivation_input,
        "projections": projections,
        "coverage": coverage,
        "query_contract": {
            "schema_version": QUERY_SCHEMA,
            "intents": {
                "summarize_action": {},
                "explain_precondition": {"condition": "typed condition ID"},
                "explain_effect": {"effect": "typed effect ID"},
                "explain_lifecycle": {},
                "explain_evidence": {"evidence": "typed evidence record ID"},
                "why_not_ready": {},
            },
        },
        "provenance_contract": {
            "evidence_sources_are_content_bound_entities": True,
            "action_instance_is_an_activity": True,
            "producers_are_responsible_agents_not_truth_oracles": True,
            "condition_evidence_is_selected_at_decision_time": True,
            "lifecycle_and_effect_evidence_are_selected_at_evaluation_time": True,
            "goal_result_and_effect_observation_are_distinct": True,
            "reported_success_is_physical_or_causal_proof": False,
            "readiness_is_dispatch_authorization": False,
        },
        "epistemic_scope": (
            "digest-bound replay of one project-declared action contract, time-qualified condition reports, observed action-server "
            "lifecycle reports, and declared-effect observations. Artifact integrity and deterministic selection are verified; producer "
            "truthfulness, clock synchronization, calibration, physical executability, causal attribution, and safety are not established."
        ),
    }
    body["assurance_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def write_action_assurance(functional_model_path: Path, bundle_path: Path, output_path: Path) -> dict[str, Any]:
    model = build_action_assurance(functional_model_path, bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_json_bytes(model))
    return model


def _validate_model_structure(model: dict[str, Any]) -> None:
    _expect_keys(
        model,
        {
            "schema_version",
            "assurance_id",
            "functional_model_binding",
            "action_evidence_binding",
            "clock",
            "derivation_input",
            "projections",
            "coverage",
            "query_contract",
            "provenance_contract",
            "epistemic_scope",
            "assurance_sha256",
        },
        "action assurance model",
    )
    if model["schema_version"] != MODEL_SCHEMA:
        raise ActionAssuranceError(f"action assurance model must use {MODEL_SCHEMA}")
    _typed_id(model["assurance_id"], "action_assurance", "assurance_id")
    supplied_sha = _text(model["assurance_sha256"], "assurance_sha256")
    body = {key: value for key, value in model.items() if key != "assurance_sha256"}
    expected_sha = _sha256_bytes(_canonical_bytes(body))
    if supplied_sha != expected_sha:
        raise ActionAssuranceError(
            f"action assurance semantic digest mismatch; expected {expected_sha}, got {supplied_sha}"
        )
    _clock(model["clock"], "action assurance clock")
    binding = model["functional_model_binding"]
    if not isinstance(binding, dict):
        raise ActionAssuranceError("functional_model_binding must be an object")
    _expect_keys(
        binding,
        {"functional_model_id", "functional_model_sha256", "functional_model_artifact_sha256"},
        "functional_model_binding",
    )
    action_binding = model["action_evidence_binding"]
    if not isinstance(action_binding, dict):
        raise ActionAssuranceError("action_evidence_binding must be an object")
    _expect_keys(action_binding, {"bundle_id", "bundle_sha256", "evidence_sources"}, "action_evidence_binding")
    derivation_input = model["derivation_input"]
    if not isinstance(derivation_input, dict):
        raise ActionAssuranceError("derivation_input must be an object")
    _expect_keys(
        derivation_input,
        {"action_instance", "evidence_policy", "functional_basis", "evidence_records"},
        "derivation_input",
    )
    records = derivation_input["evidence_records"]
    if not isinstance(records, list):
        raise ActionAssuranceError("derivation_input.evidence_records must be an array")
    record_ids = [record.get("record_id") for record in records if isinstance(record, dict)]
    if len(record_ids) != len(records) or len(set(record_ids)) != len(record_ids):
        raise ActionAssuranceError("embedded evidence records are malformed or duplicate")
    expected_projections, expected_coverage = _derive_assurance(derivation_input)
    if model["projections"] != expected_projections:
        raise ActionAssuranceError("action assurance projections do not match embedded derivation inputs")
    if model["coverage"] != expected_coverage:
        raise ActionAssuranceError("action assurance coverage does not match embedded derivation inputs")
    query_contract = model["query_contract"]
    if not isinstance(query_contract, dict) or query_contract.get("schema_version") != QUERY_SCHEMA:
        raise ActionAssuranceError("action assurance query contract is malformed")


def read_action_assurance(path: Path) -> dict[str, Any]:
    model = _read_json(path, "action assurance model")
    try:
        _validate_model_structure(model)
    except ActionAssuranceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ActionAssuranceError(f"action assurance model structure is malformed: {error}") from error
    return model


def query_action_assurance(model: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    _validate_model_structure(model)
    _expect_keys(query, {"schema_version", "query_id", "intent", "parameters"}, "action assurance query")
    if query["schema_version"] != QUERY_SCHEMA:
        raise ActionAssuranceError(f"action assurance query must use {QUERY_SCHEMA}")
    query_id = _text(query["query_id"], "action assurance query_id")
    intent = _text(query["intent"], "action assurance query intent")
    parameters = query["parameters"]
    if not isinstance(parameters, dict):
        raise ActionAssuranceError("action assurance query parameters must be an object")
    projections = model["projections"]
    supporting_records: list[dict[str, Any]] = []
    functional_support: dict[str, Any] = {}
    unknowns = [
        "producer truthfulness, calibration, and clock synchronization are not verified",
        "physical executability, causal attribution, and safety are not established",
    ]
    if intent == "summarize_action":
        _expect_keys(parameters, set(), "summarize_action parameters")
        answer = {
            "declared_action": projections["declared_action"],
            "precondition_summary": projections["precondition_summary"],
            "readiness": projections["readiness"],
            "lifecycle": {
                key: projections["lifecycle"][key]
                for key in (
                    "status",
                    "consistency",
                    "goal_response",
                    "latest_observed_status",
                    "terminal_result",
                    "execution_started_observed",
                    "execution_started_at_ns",
                )
            },
            "effect_summary": projections["effect_summary"],
            "outcome": projections["outcome"],
            "discrepancies": projections["discrepancies"],
        }
        supporting_records = model["derivation_input"]["evidence_records"]
        functional_support = model["derivation_input"]["functional_basis"]
        answer_cnl = (
            f"Action {projections['declared_action']['action_instance_id']} readiness is "
            f"{projections['readiness']['conclusion']}; lifecycle is {projections['lifecycle']['status']}; "
            f"outcome evidence is {projections['outcome']['conclusion']}."
        )
    elif intent == "explain_precondition":
        _expect_keys(parameters, {"condition"}, "explain_precondition parameters")
        condition = _text(parameters["condition"], "condition")
        matches = [item for item in projections["preconditions"] if item["condition_id"] == condition or item["condition_id"].rsplit("/", 1)[-1] == condition]
        if len(matches) != 1:
            raise ActionAssuranceError(f"condition query must resolve uniquely; matches={[item['condition_id'] for item in matches]}")
        answer = matches[0]
        supporting_records = answer["selected_records"]
        functional_support = next(
            item for item in model["derivation_input"]["functional_basis"]["conditions"]
            if item["condition_id"] == answer["condition_id"]
        )
        answer_cnl = f"Condition {answer['condition_id']} is {answer['status']} at decision time {answer['reference_time_ns']}."
    elif intent == "explain_effect":
        _expect_keys(parameters, {"effect"}, "explain_effect parameters")
        effect = _text(parameters["effect"], "effect")
        matches = [item for item in projections["effects"] if item["effect_id"] == effect or item["effect_id"].rsplit("/", 1)[-1] == effect]
        if len(matches) != 1:
            raise ActionAssuranceError(f"effect query must resolve uniquely; matches={[item['effect_id'] for item in matches]}")
        answer = matches[0]
        supporting_records = answer["selected_records"]
        functional_support = next(
            item for item in model["derivation_input"]["functional_basis"]["effects"]
            if item["effect_id"] == answer["effect_id"]
        )
        answer_cnl = (
            f"Effect {answer['effect_id']} is {answer['status']} with temporal relation "
            f"{answer['temporal_relation_to_execution']}; causation is not established."
        )
    elif intent == "explain_lifecycle":
        _expect_keys(parameters, set(), "explain_lifecycle parameters")
        answer = projections["lifecycle"]
        supporting_records = answer["selected_records"]
        answer_cnl = f"Observed action-server lifecycle status is {answer['status']} with consistency {answer['consistency']}."
    elif intent == "explain_evidence":
        _expect_keys(parameters, {"evidence"}, "explain_evidence parameters")
        evidence = _text(parameters["evidence"], "evidence")
        matches = [
            item for item in model["derivation_input"]["evidence_records"]
            if item["record_id"] == evidence or item["record_id"].rsplit("/", 1)[-1] == evidence
        ]
        if len(matches) != 1:
            raise ActionAssuranceError(f"evidence query must resolve uniquely; matches={[item['record_id'] for item in matches]}")
        answer = matches[0]
        supporting_records = matches
        answer_cnl = (
            f"Evidence {answer['record_id']} is a {answer['evidence_type']} report from "
            f"{answer['source']['producer']['producer_id']}; artifact integrity is verified, source truth is not."
        )
    elif intent == "why_not_ready":
        _expect_keys(parameters, set(), "why_not_ready parameters")
        blockers = []
        if not projections["declared_action"]["structurally_grounded"]:
            blockers.append({
                "type": "ungrounded_capability_requirements",
                "capability_grounding": projections["declared_action"]["capability_grounding"],
            })
        blockers.extend(
            {
                "type": "precondition",
                "condition_id": item["condition_id"],
                "status": item["status"],
                "selected_record_ids": item["selected_record_ids"],
            }
            for item in projections["preconditions"]
            if item["status"] != "satisfied"
        )
        answer = {
            "readiness_conclusion": projections["readiness"]["conclusion"],
            "blockers": blockers,
            "authorization_to_dispatch": "not_provided",
        }
        selected = {record_id for blocker in blockers for record_id in blocker.get("selected_record_ids", [])}
        supporting_records = [
            item for item in model["derivation_input"]["evidence_records"] if item["record_id"] in selected
        ]
        functional_support = model["derivation_input"]["functional_basis"]
        answer_cnl = (
            "No declared readiness blocker is present, but this model still does not authorize dispatch."
            if not blockers
            else f"Action is not ready because {len(blockers)} declared structural or evidence blocker(s) remain."
        )
    else:
        raise ActionAssuranceError(f"unsupported action assurance query intent {intent!r}")
    return {
        "schema_version": ANSWER_SCHEMA,
        "status": "answered",
        "query_id": query_id,
        "intent": intent,
        "assurance": {
            "assurance_id": model["assurance_id"],
            "assurance_sha256": model["assurance_sha256"],
        },
        "answer": answer,
        "answer_cnl": answer_cnl,
        "supporting_evidence_records": supporting_records,
        "functional_support": functional_support,
        "unknowns": unknowns,
        "epistemic_scope": model["epistemic_scope"],
    }


def query_action_assurance_files(model_path: Path, query_path: Path) -> dict[str, Any]:
    model = read_action_assurance(model_path)
    query = _read_json(query_path, "action assurance query")
    return query_action_assurance(model, query)


def verify_action_assurance(
    functional_model_path: Path,
    bundle_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    try:
        stored = read_action_assurance(model_path)
    except ActionAssuranceError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "exact_regeneration_match": False,
            "issues": [{"check": "stored_model_structure", "message": str(error)}],
        }
    try:
        expected = build_action_assurance(functional_model_path, bundle_path)
    except ActionAssuranceError as error:
        return {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "failed",
            "assurance_id": stored["assurance_id"],
            "exact_regeneration_match": False,
            "issues": [{"check": "regeneration", "message": str(error)}],
        }
    exact = _json_bytes(stored) == _json_bytes(expected)
    issues = [] if exact else [{"check": "exact_regeneration", "message": "stored assurance differs from exact regeneration"}]
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if exact else "failed",
        "assurance_id": stored["assurance_id"],
        "assurance_sha256": stored["assurance_sha256"],
        "exact_regeneration_match": exact,
        "validated_evidence_source_count": stored["coverage"]["evidence_source_count"],
        "validated_evidence_record_count": stored["coverage"]["evidence_record_count"],
        "readiness_conclusion": stored["projections"]["readiness"]["conclusion"],
        "outcome_conclusion": stored["projections"]["outcome"]["conclusion"],
        "issues": issues,
        "epistemic_scope": (
            "exact regeneration verifies functional binding, evidence artifact digests, time-policy selection, and deterministic "
            "assurance derivation; it does not validate producer truthfulness, physical causation, execution safety, or hardware state"
        ),
    }
