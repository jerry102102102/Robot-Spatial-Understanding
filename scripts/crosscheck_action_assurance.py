#!/usr/bin/env python3
"""Dependency-free randomized cross-check for the public action-assurance CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).with_name("robot_spatial.py")
CLOCK = {"domain": "oracle_monotonic", "unit": "nanoseconds", "epoch": "oracle-start"}
ACTOR = "component/tool"
TARGET = "object_instance/oracle-target"
ACTION = "action_instance/oracle-action"
CONDITION_BINDINGS = {"actor": ACTOR, "target": TARGET}
LIFECYCLE_BINDINGS = {"action_instance": ACTION}


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def run_cli(*arguments: object) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"public CLI failed ({process.returncode}): {' '.join(str(value) for value in arguments)}\n"
            f"stdout={process.stdout[:1200]}\nstderr={process.stderr[:1200]}"
        )
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("public CLI did not return one JSON object")
    return value


def make_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="action_oracle">
  <link name="base"/>
  <link name="tool"/>
  <joint name="tool_hinge" type="revolute">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="5" velocity="2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def functional_spec(context: Path, *, grounded: bool) -> dict[str, Any]:
    canonical = json.loads((context / "model.json").read_text(encoding="utf-8"))
    requirement_entity = "link/tool" if grounded else "link/not_present"
    return {
        "schema_version": "robot-spatial-function-affordance-spec.v1",
        "function_set_id": f"function_set/action_oracle_{'grounded' if grounded else 'ungrounded'}",
        "source_binding": {
            "urdf_semantic_sha256": canonical["source"]["semantic_sha256"],
            "articulation_grammar_sha256": canonical["artifacts"]["articulation_grammar"]["sha256"],
            "constraint_graph_sha256": None,
            "configuration_atlas_sha256": None,
        },
        "object_types": [{
            "object_type_id": "object_type/workpiece",
            "meaning": "Project-declared oracle target type.",
        }],
        "components": [{
            "component_id": ACTOR,
            "members": ["link/tool", "joint/tool_hinge"],
            "meaning": "Project-declared oracle component.",
        }],
        "functions": [{
            "function_id": "function/manipulate_workpiece",
            "provided_by": [ACTOR],
            "verb": "manipulate",
            "object_types": ["object_type/workpiece"],
            "purpose": "Exercise the action-evidence oracle.",
        }],
        "conditions": [
            {
                "condition_id": "condition/target_visible",
                "predicate": "target_visible",
                "arguments": ["actor", "target"],
                "truth_source": "runtime_observation_required",
                "meaning": "Runtime observer reports the target visible.",
            },
            {
                "condition_id": "condition/plan_valid",
                "predicate": "plan_valid",
                "arguments": ["actor", "target"],
                "truth_source": "planner_verification_required",
                "meaning": "Planner reports the candidate valid.",
            },
        ],
        "effects": [{
            "effect_id": "effect/workpiece_moved",
            "predicate": "workpiece_moved_by",
            "arguments": ["target", "actor"],
            "meaning": "Observer reports the declared workpiece-moved predicate.",
        }],
        "capabilities": [{
            "capability_id": "capability/tool_motion",
            "provided_by": [ACTOR],
            "realizes_functions": ["function/manipulate_workpiece"],
            "enabling_requirements": [{
                "requirement_id": "requirement/required_entity",
                "type": "entity_exists",
                "parameters": {"entity": requirement_entity},
            }],
            "condition_refs": ["condition/target_visible"],
            "limitations": ["Oracle structure does not establish physical execution or safety."],
        }],
        "affordances": [{
            "affordance_id": "affordance/move_workpiece",
            "offered_by": [ACTOR],
            "action_verb": "move",
            "target_object_types": ["object_type/workpiece"],
            "capability_refs": ["capability/tool_motion"],
            "precondition_refs": ["condition/target_visible", "condition/plan_valid"],
            "effect_refs": ["effect/workpiece_moved"],
            "meaning": "Project-declared conditional oracle affordance.",
        }],
        "inventory_completeness": [{
            "subject": ACTOR,
            "inventories": ["functions", "capabilities", "affordances"],
            "scope": "Oracle project specification only.",
        }],
    }


def condition_record(
    identifier: str,
    condition: str,
    evidence_type: str,
    value: str,
    time_ns: int,
    *,
    bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    predicate = {"target_visible": "target_visible", "plan_valid": "plan_valid"}[condition]
    return {
        "record_id": f"evidence/{identifier}",
        "evidence_type": evidence_type,
        "subject_ref": f"condition/{condition}",
        "predicate": predicate,
        "bindings": bindings or CONDITION_BINDINGS,
        "value": value,
        "observed_at_ns": time_ns,
        "valid_until_ns": None,
        "claim_scope": "Oracle condition report only.",
        "limitations": ["The producer is not treated as a truth oracle."],
    }


def lifecycle_record(identifier: str, kind: str, value: str, time_ns: int) -> dict[str, Any]:
    subject, predicate = {
        "goal_response": ("lifecycle/goal_response", "goal_response"),
        "action_status": ("lifecycle/action_status", "action_status"),
        "action_result": ("lifecycle/action_result", "action_result"),
    }[kind]
    return {
        "record_id": f"evidence/{identifier}",
        "evidence_type": kind,
        "subject_ref": subject,
        "predicate": predicate,
        "bindings": LIFECYCLE_BINDINGS,
        "value": value,
        "observed_at_ns": time_ns,
        "valid_until_ns": None,
        "claim_scope": "Oracle action-server report only.",
        "limitations": ["Protocol state is not physical proof."],
    }


def effect_record(value: str, time_ns: int) -> dict[str, Any]:
    return {
        "record_id": "evidence/effect",
        "evidence_type": "effect_observation",
        "subject_ref": "effect/workpiece_moved",
        "predicate": "workpiece_moved_by",
        "bindings": CONDITION_BINDINGS,
        "value": value,
        "observed_at_ns": time_ns,
        "valid_until_ns": None,
        "claim_scope": "Oracle effect report only.",
        "limitations": ["Temporal succession does not establish causation."],
    }


def scenario(case_index: int, rng: random.Random) -> tuple[str, bool, list[dict[str, Any]]]:
    variant = case_index % 8
    jitter = rng.randint(0, 3)
    visible = condition_record("visible", "target_visible", "runtime_observation", "true", 80 + jitter)
    plan = condition_record("plan", "plan_valid", "planner_verification", "true", 85 + jitter)
    accepted = lifecycle_record("goal", "goal_response", "accepted", 110 + jitter)
    executing = lifecycle_record("executing", "action_status", "executing", 120 + jitter)
    status_success = lifecycle_record("status", "action_status", "succeeded", 160 + jitter)
    result_success = lifecycle_record("result", "action_result", "succeeded", 165 + jitter)
    if variant == 0:
        return "ready_success_effect_true", True, [visible, plan, accepted, executing, status_success, result_success, effect_record("true", 175 + jitter)]
    if variant == 1:
        visible["value"] = "false"
        return "false_precondition_success_effect_false", True, [visible, plan, accepted, executing, status_success, result_success, effect_record("false", 175 + jitter)]
    if variant == 2:
        visible["observed_at_ns"] = 40
        return "stale_precondition_goal_only", True, [visible, plan, accepted]
    if variant == 3:
        conflict = dict(visible)
        conflict["record_id"] = "evidence/visible-conflict"
        conflict["value"] = "false"
        return "conflicting_latest_evidence", True, [visible, conflict, plan, accepted]
    if variant == 4:
        status_success["value"] = "aborted"
        return "terminal_mismatch", True, [visible, plan, accepted, executing, status_success, result_success, effect_record("true", 175 + jitter)]
    if variant == 5:
        return "effect_before_execution", True, [visible, plan, effect_record("true", 115 + jitter), accepted, executing, status_success, result_success]
    if variant == 6:
        future = condition_record("visible-future", "target_visible", "runtime_observation", "true", 105 + jitter)
        return "future_condition_evidence", True, [future, plan, accepted]
    return "ungrounded_positive_reports", False, [visible, plan, accepted, executing, status_success, result_success, effect_record("true", 175 + jitter)]


def select_truth(
    records: list[dict[str, Any]],
    subject: str,
    predicate: str,
    evidence_type: str,
    reference_time: int,
    maximum_age: int | None,
) -> tuple[str, str]:
    same_predicate = [item for item in records if item["subject_ref"] == subject and item["predicate"] == predicate]
    same_binding = [item for item in same_predicate if item["bindings"] == CONDITION_BINDINGS]
    correct_type = [item for item in same_binding if item["evidence_type"] == evidence_type]
    eligible = [
        item for item in correct_type
        if item["observed_at_ns"] <= reference_time
        and (maximum_age is None or reference_time - item["observed_at_ns"] <= maximum_age)
    ]
    if eligible:
        latest = max(item["observed_at_ns"] for item in eligible)
        values = {item["value"] for item in eligible if item["observed_at_ns"] == latest}
        if len(values) > 1:
            return "unknown_conflicting_latest_evidence", "unknown"
        truth = next(iter(values))
        return {"true": "satisfied", "false": "not_satisfied", "unknown": "unknown_reported"}[truth], truth
    if any(item["evidence_type"] != evidence_type for item in same_binding):
        return "unknown_wrong_evidence_type", "unknown"
    if any(item["bindings"] != CONDITION_BINDINGS for item in same_predicate):
        return "unknown_binding_mismatch", "unknown"
    if any(reference_time - item["observed_at_ns"] > maximum_age for item in correct_type if item["observed_at_ns"] <= reference_time and maximum_age is not None):
        return "unknown_stale_evidence", "unknown"
    if any(item["observed_at_ns"] > reference_time for item in correct_type):
        return "unknown_future_only", "unknown"
    return "unknown_missing_evidence", "unknown"


def independent_expected(records: list[dict[str, Any]], grounded: bool) -> dict[str, Any]:
    visible_status, _ = select_truth(
        records, "condition/target_visible", "target_visible", "runtime_observation", 100, 30
    )
    plan_status, _ = select_truth(
        records, "condition/plan_valid", "plan_valid", "planner_verification", 100, 30
    )
    statuses = [visible_status, plan_status]
    if not grounded:
        readiness = "not_ready_ungrounded_capability_requirements"
    elif "not_satisfied" in statuses:
        readiness = "not_ready_declared_precondition_false"
    elif any(value != "satisfied" for value in statuses):
        readiness = "not_ready_missing_stale_conflicting_or_invalid_evidence"
    else:
        readiness = "ready_under_declared_model_and_evidence"

    eligible = sorted(
        [
            item for item in records
            if item["evidence_type"] in {"goal_response", "action_status", "action_result"}
            and item["bindings"] == LIFECYCLE_BINDINGS
            and item["observed_at_ns"] <= 200
        ],
        key=lambda item: (item["observed_at_ns"], item["record_id"]),
    )
    goals = [item for item in eligible if item["evidence_type"] == "goal_response"]
    action_statuses = [item for item in eligible if item["evidence_type"] == "action_status"]
    results = [item for item in eligible if item["evidence_type"] == "action_result"]
    goal = goals[-1]["value"] if goals and len({item["value"] for item in goals}) == 1 else None
    result = results[-1]["value"] if results and len({item["value"] for item in results}) == 1 else None
    lifecycle_issue = False
    if (action_statuses or results) and not any(item["value"] == "accepted" for item in goals):
        lifecycle_issue = True
    terminal_statuses = [item for item in action_statuses if item["value"] in {"succeeded", "aborted", "canceled"}]
    if result is not None and terminal_statuses and terminal_statuses[-1]["value"] != result:
        lifecycle_issue = True
    if lifecycle_issue:
        lifecycle = "inconsistent_lifecycle_evidence"
    elif goal == "rejected":
        lifecycle = "goal_rejected"
    elif goal != "accepted":
        lifecycle = "goal_response_not_observed"
    elif result is not None:
        lifecycle = f"result_{result}"
    elif action_statuses:
        lifecycle = f"status_{action_statuses[-1]['value']}"
    else:
        lifecycle = "goal_accepted_no_execution_status"
    execution_records = [
        item for item in action_statuses
        if item["value"] in {"executing", "canceling", "succeeded", "aborted", "canceled"}
    ]
    execution_start = execution_records[0]["observed_at_ns"] if execution_records else None
    effect_status, effect_truth = select_truth(
        records, "effect/workpiece_moved", "workpiece_moved_by", "effect_observation", 200, None
    )
    selected_effects = [
        item for item in records
        if item["subject_ref"] == "effect/workpiece_moved"
        and item["predicate"] == "workpiece_moved_by"
        and item["evidence_type"] == "effect_observation"
        and item["bindings"] == CONDITION_BINDINGS
        and item["observed_at_ns"] <= 200
    ]
    effect_time = max((item["observed_at_ns"] for item in selected_effects), default=None)
    effect_post = effect_time is not None and execution_start is not None and effect_time >= execution_start
    if effect_post and effect_status == "satisfied" and effect_truth == "true":
        effect_summary = "all_declared_effects_observed_true_after_execution_started"
    elif effect_post and effect_truth == "false":
        effect_summary = "one_or_more_declared_effects_observed_false_after_execution_started"
    else:
        effect_summary = "incomplete_or_temporally_unlinked_effect_evidence"
    if lifecycle_issue:
        outcome = "inconsistent_lifecycle_evidence"
    elif result == "succeeded":
        outcome = {
            "all_declared_effects_observed_true_after_execution_started": "action_server_reported_success_and_all_declared_effects_observed_after_execution_started",
            "one_or_more_declared_effects_observed_false_after_execution_started": "action_server_reported_success_but_declared_effect_observation_false",
            "incomplete_or_temporally_unlinked_effect_evidence": "action_server_reported_success_effect_evidence_incomplete_or_temporally_unlinked",
        }[effect_summary]
    elif result in {"aborted", "canceled"}:
        outcome = f"action_server_reported_{result}"
    else:
        outcome = "no_terminal_action_result_observed"
    return {
        "condition_statuses": {
            "condition/target_visible": visible_status,
            "condition/plan_valid": plan_status,
        },
        "readiness": readiness,
        "lifecycle": lifecycle,
        "effect_summary": effect_summary,
        "outcome": outcome,
    }


def build_functional_models(root: Path) -> dict[bool, Path]:
    urdf = root / "oracle.urdf"
    context = root / "context"
    make_urdf(urdf)
    run_cli("export", urdf, "--workspace-samples", 0, "--out", context)
    result: dict[bool, Path] = {}
    for grounded in (True, False):
        label = "grounded" if grounded else "ungrounded"
        spec_path = root / f"functional-{label}.json"
        model_path = root / f"functional-model-{label}.json"
        write_json(spec_path, functional_spec(context, grounded=grounded))
        run_cli("functional-model", context, spec_path, "--out", model_path)
        result[grounded] = model_path
    return result


def run_crosscheck(cases: int, seed: int) -> dict[str, Any]:
    if cases < 1:
        raise ValueError("cases must be positive")
    rng = random.Random(seed)
    failures: list[dict[str, Any]] = []
    checks = 0
    queries = 0
    with tempfile.TemporaryDirectory(prefix="action-assurance-oracle-") as temporary:
        root = Path(temporary)
        functional_models = build_functional_models(root)
        for case_index in range(cases):
            label, grounded, records = scenario(case_index, rng)
            case_dir = root / f"case-{case_index:03d}"
            case_dir.mkdir()
            functional_model = functional_models[grounded]
            functional = json.loads(functional_model.read_text(encoding="utf-8"))
            source_path = case_dir / "evidence-source.json"
            write_json(source_path, {
                "schema_version": "robot-spatial-action-evidence-source.v1",
                "source_id": f"evidence_source/oracle-{case_index}",
                "clock": CLOCK,
                "producer": {"producer_id": "oracle/generator", "producer_type": "independent_test_oracle"},
                "records": records,
            })
            bundle_path = case_dir / "evidence-bundle.json"
            write_json(bundle_path, {
                "schema_version": "robot-spatial-action-evidence-bundle.v1",
                "bundle_id": f"action_evidence_bundle/oracle-{case_index}",
                "functional_model_binding": {
                    "functional_model_id": functional["functional_model_id"],
                    "functional_model_sha256": functional["functional_model_sha256"],
                    "functional_model_artifact_sha256": sha256(functional_model),
                },
                "clock": CLOCK,
                "action_instance": {
                    "action_instance_id": ACTION,
                    "affordance_id": "affordance/move_workpiece",
                    "offered_by": ACTOR,
                    "action_verb": "move",
                    "target_object_type": "object_type/workpiece",
                    "target_instance_id": TARGET,
                    "argument_bindings": CONDITION_BINDINGS,
                    "requested_at_ns": 50,
                    "decision_time_ns": 100,
                    "evaluation_time_ns": 200,
                },
                "evidence_policy": {
                    "maximum_age_ns": {
                        "operator_confirmation": 30,
                        "planner_verification": 30,
                        "project_assumption": 1000,
                        "runtime_observation": 30,
                    },
                    "require_goal_acceptance_before_status": True,
                    "require_terminal_result_status_match": True,
                },
                "evidence_sources": [{
                    "source_id": f"evidence_source/oracle-{case_index}",
                    "path": source_path.name,
                    "sha256": sha256(source_path),
                }],
            })
            assurance_path = case_dir / "action-assurance.json"
            try:
                run_cli("action-assurance", functional_model, bundle_path, "--out", assurance_path)
                assurance = json.loads(assurance_path.read_text(encoding="utf-8"))
                expected = independent_expected(records, grounded)
                actual = {
                    "condition_statuses": {
                        item["condition_id"]: item["status"]
                        for item in assurance["projections"]["preconditions"]
                    },
                    "readiness": assurance["projections"]["readiness"]["conclusion"],
                    "lifecycle": assurance["projections"]["lifecycle"]["status"],
                    "effect_summary": assurance["projections"]["effect_summary"]["status"],
                    "outcome": assurance["projections"]["outcome"]["conclusion"],
                }
                checks += 5
                if actual != expected:
                    failures.append({"case": case_index, "scenario": label, "expected": expected, "actual": actual})
                query_path = case_dir / "query.json"
                write_json(query_path, {
                    "schema_version": "robot-spatial-action-assurance-query.v1",
                    "query_id": f"oracle/{case_index}",
                    "intent": "summarize_action",
                    "parameters": {},
                })
                query = run_cli("query-action-assurance", assurance_path, query_path, "--compact")
                queries += 1
                checks += 4
                query_actual = {
                    "readiness": query["answer"]["readiness"]["conclusion"],
                    "lifecycle": query["answer"]["lifecycle"]["status"],
                    "effect_summary": query["answer"]["effect_summary"]["status"],
                    "outcome": query["answer"]["outcome"]["conclusion"],
                }
                query_expected = {key: expected[key] for key in query_actual}
                if query_actual != query_expected:
                    failures.append({
                        "case": case_index,
                        "scenario": label,
                        "phase": "public_query",
                        "expected": query_expected,
                        "actual": query_actual,
                    })
                verification = run_cli(
                    "verify-action-assurance", functional_model, bundle_path, "--model", assurance_path
                )
                checks += 1
                if verification.get("status") != "passed" or not verification.get("exact_regeneration_match"):
                    failures.append({"case": case_index, "scenario": label, "phase": "verification", "actual": verification})
            except Exception as error:  # independent harness must preserve the failing case
                failures.append({"case": case_index, "scenario": label, "phase": "exception", "message": str(error)})
    return {
        "schema_version": "robot-spatial-action-assurance-crosscheck.v1",
        "status": "passed" if not failures else "failed",
        "seed": seed,
        "case_count": cases,
        "scenario_family_count": 8,
        "public_query_count": queries,
        "assertion_count": checks,
        "failures": failures,
        "exclusions": [
            "does not import production action-assurance, functional, concept, or URDF modules",
            "does not validate evidence-producer truthfulness, clock synchronization, calibration, physical causation, hardware state, or safety",
            "validates public CLI transcription, time selection, readiness, lifecycle/effect separation, query projection, and exact regeneration only",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    try:
        result = run_crosscheck(arguments.cases, arguments.seed)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        result = {
            "schema_version": "robot-spatial-action-assurance-crosscheck.v1",
            "status": "failed",
            "seed": arguments.seed,
            "case_count": arguments.cases,
            "failures": [{"phase": "harness", "message": str(error)}],
        }
    serialized = json_bytes(result)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_bytes(serialized)
    sys.stdout.buffer.write(serialized)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
