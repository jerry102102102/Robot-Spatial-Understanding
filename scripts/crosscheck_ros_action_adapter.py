#!/usr/bin/env python3
"""Independently cross-check ROS action capture normalization over generated cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from spatial_functional import write_functional_model_from_context


SCRIPT_DIR = Path(__file__).resolve().parent
ROBOT_SPATIAL = SCRIPT_DIR / "robot_spatial.py"
ADAPTER = SCRIPT_DIR / "ros_action_adapter.py"
FIXTURE = SCRIPT_DIR / "tests" / "fixtures" / "mimic_branch.urdf"
CONFIG_SCHEMA = "robot-spatial-ros-action-adapter-config.v1"
CAPTURE_SCHEMA = "robot-spatial-ros-action-capture.v1"
SOURCE_SCHEMA = "robot-spatial-action-evidence-source.v1"
CLOCK = {"domain": "oracle_monotonic", "unit": "nanoseconds", "epoch": "oracle-run"}
STATUS_VALUE = {1: "accepted", 2: "executing", 3: "canceling", 4: "succeeded", 5: "canceled", 6: "aborted"}


class CrosscheckError(RuntimeError):
    """Cross-check setup or assertion failure."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _functional_spec(context: Path) -> dict[str, Any]:
    canonical = json.loads((context / "model.json").read_text(encoding="utf-8"))
    graph = json.loads((context / "concept-graph.json").read_text(encoding="utf-8"))
    driver = graph["projections"]["articulation"]["drivers"][0]["driver_entity"]
    return {
        "schema_version": "robot-spatial-function-affordance-spec.v1",
        "function_set_id": "function_set/action_oracle",
        "source_binding": {
            "urdf_semantic_sha256": canonical["source"]["semantic_sha256"],
            "articulation_grammar_sha256": canonical["artifacts"]["articulation_grammar"]["sha256"],
            "constraint_graph_sha256": None,
            "configuration_atlas_sha256": None,
        },
        "object_types": [{
            "object_type_id": "object_type/graspable",
            "meaning": "Oracle-declared target type.",
        }],
        "components": [{
            "component_id": "component/gripper",
            "members": ["link/driver_link", "link/follower_link", "joint/driver", "joint/follower"],
            "meaning": "Oracle-declared paired mechanism.",
        }],
        "functions": [{
            "function_id": "function/retain_object",
            "provided_by": ["component/gripper"],
            "verb": "retain",
            "object_types": ["object_type/graspable"],
            "purpose": "Oracle-declared retention intent.",
        }],
        "conditions": [
            {
                "condition_id": "condition/target_between_fingers",
                "predicate": "target_between_fingers",
                "arguments": ["actor", "target"],
                "truth_source": "runtime_observation_required",
                "meaning": "Oracle condition.",
            },
            {
                "condition_id": "condition/plan_approved",
                "predicate": "plan_approved",
                "arguments": ["actor", "target"],
                "truth_source": "planner_verification_required",
                "meaning": "Oracle planner condition.",
            },
        ],
        "effects": [{
            "effect_id": "effect/target_retained",
            "predicate": "target_retained_by",
            "arguments": ["target", "actor"],
            "meaning": "Oracle intended effect.",
        }],
        "capabilities": [{
            "capability_id": "capability/coordinated_closure",
            "provided_by": ["component/gripper"],
            "realizes_functions": ["function/retain_object"],
            "enabling_requirements": [
                {
                    "requirement_id": "requirement/driver_affects_left",
                    "type": "driver_affects_frame",
                    "parameters": {"driver": driver, "frame": "frame/driver_link"},
                },
                {
                    "requirement_id": "requirement/driver_affects_right",
                    "type": "driver_affects_frame",
                    "parameters": {"driver": driver, "frame": "frame/follower_link"},
                },
            ],
            "condition_refs": ["condition/target_between_fingers"],
            "limitations": ["Oracle structure does not prove execution."],
        }],
        "affordances": [{
            "affordance_id": "affordance/grasp",
            "offered_by": ["component/gripper"],
            "action_verb": "grasp",
            "target_object_types": ["object_type/graspable"],
            "capability_refs": ["capability/coordinated_closure"],
            "precondition_refs": ["condition/target_between_fingers", "condition/plan_approved"],
            "effect_refs": ["effect/target_retained"],
            "meaning": "Oracle action relation.",
        }],
        "inventory_completeness": [{
            "subject": "component/gripper",
            "inventories": ["functions", "capabilities", "affordances"],
            "scope": "Oracle fixture only.",
        }],
    }


def _prepare_model(root: Path) -> tuple[Path, dict[str, Any]]:
    context = root / "context"
    result = subprocess.run(
        [
            sys.executable,
            str(ROBOT_SPATIAL),
            "export",
            str(FIXTURE),
            "--workspace-samples",
            "0",
            "--out",
            str(context),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise CrosscheckError(f"cannot prepare oracle functional context: {result.stderr}")
    spec_path = root / "functional-spec.json"
    model_path = root / "functional-model.json"
    _write(spec_path, _functional_spec(context))
    model = write_functional_model_from_context(context, spec_path, model_path)
    return model_path, model


def _config(model_path: Path, model: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "adapter_id": "oracle_action",
        "clock": CLOCK,
        "functional_model_binding": {
            "functional_model_id": model["functional_model_id"],
            "functional_model_sha256": model["functional_model_sha256"],
            "functional_model_artifact_sha256": _sha_path(model_path),
        },
        "ros_action": {
            "name": "/oracle/grasp",
            "type": "example_interfaces/action/Fibonacci",
            "status_topic": "/oracle/grasp/_action/status",
        },
        "action_instance": {
            "action_instance_id": "action_instance/oracle-action",
            "affordance_id": "affordance/grasp",
            "offered_by": "component/gripper",
            "action_verb": "grasp",
            "target_object_type": "object_type/graspable",
            "target_instance_id": "object_instance/oracle-target",
            "argument_bindings": {
                "actor": "component/gripper",
                "target": "object_instance/oracle-target",
            },
        },
        "evidence_policy": {
            "maximum_age_ns": {
                "operator_confirmation": 100,
                "planner_verification": 100,
                "project_assumption": 1000,
                "runtime_observation": 100,
            },
            "require_goal_acceptance_before_status": True,
            "require_terminal_result_status_match": True,
        },
        "policies": {
            "reject_multiple_status_publishers": True,
            "reject_conflicting_same_time_status": True,
        },
    }


def _record(sequence: int, kind: str, timestamp: int, payload: dict[str, Any], publisher: str | None = None) -> dict[str, Any]:
    return {
        "record_id": f"ros_action_record/oracle/{sequence:06d}",
        "sequence": sequence,
        "kind": kind,
        "event_timestamp_ns": timestamp,
        "publisher_id": publisher,
        "payload": payload,
    }


def _base_capture(config_sha: str, goal_uuid: str, goal_payload: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CAPTURE_SCHEMA,
        "capture_id": "oracle-capture",
        "adapter_config_sha256": config_sha,
        "clock": CLOCK,
        "interval": {
            "started_at_ns": 100,
            "requested_at_ns": 105,
            "decision_time_ns": 130,
            "evaluation_time_ns": 240,
            "ended_at_ns": 240,
            "termination_reason": "oracle_case",
        },
        "source": {
            "transport": "synthetic_fixture",
            "reference": "independent randomized oracle",
            "ros_distro": None,
            "status_publisher_identity_visibility": "oracle fixture",
            "service_server_identity_visibility": "unavailable",
            "feedback_publisher_identity_visibility": "unavailable",
        },
        "action": {
            "name": "/oracle/grasp",
            "type": "example_interfaces/action/Fibonacci",
            "goal_uuid": goal_uuid,
            "goal_payload": goal_payload,
            "goal_payload_sha256": _sha_bytes(_canonical_bytes(goal_payload)),
        },
        "client": {"node_name": "oracle_client", "use_sim_time": False},
        "records": records,
    }


def _valid_records(
    goal_uuid: str,
    *,
    accepted: bool,
    statuses: list[int],
    result: int | None,
    publisher: str | None = "rmw:oracle-server",
    include_unknown_other: bool = False,
    duplicate_first_status: bool = False,
    jitter: int = 0,
) -> list[dict[str, Any]]:
    records = [
        _record(1, "send_goal_request", 135 + jitter, {}),
        _record(2, "goal_response", 140 + jitter, {
            "accepted": accepted,
            "server_acceptance_timestamp_ns": 138 + jitter if accepted else 0,
        }),
    ]
    sequence = 3
    timestamp = 150 + jitter
    if include_unknown_other:
        records.append(_record(sequence, "status_array", timestamp, {"statuses": [
            {"goal_uuid": goal_uuid, "accepted_at_ns": 138 + jitter, "status_code": 0},
            {"goal_uuid": "f" * 32, "accepted_at_ns": 120, "status_code": 2},
        ]}, publisher))
        sequence += 1
        timestamp += 10
    for index, status in enumerate(statuses):
        records.append(_record(sequence, "status_array", timestamp, {"statuses": [{
            "goal_uuid": goal_uuid,
            "accepted_at_ns": 138 + jitter,
            "status_code": status,
        }]}, publisher))
        sequence += 1
        if duplicate_first_status and index == 0:
            records.append(_record(sequence, "status_array", timestamp, {"statuses": [{
                "goal_uuid": goal_uuid,
                "accepted_at_ns": 138 + jitter,
                "status_code": status,
            }]}, publisher))
            sequence += 1
        timestamp += 10
    if result is not None:
        records.append(_record(sequence, "get_result_request", timestamp, {}))
        sequence += 1
        records.append(_record(sequence, "result_response", timestamp + 5, {
            "goal_uuid": goal_uuid,
            "status_code": result,
            "result": {"oracle_token": result, "values": [jitter, len(statuses)]},
        }))
    return records


def _case(family: str, config_sha: str, rng: random.Random, index: int) -> tuple[dict[str, Any], bool, list[tuple[str, str, int]], dict[str, int]]:
    goal_uuid = f"{rng.getrandbits(128):032x}"
    goal_payload = {"order": rng.randint(1, 20), "case": index}
    jitter = rng.randint(0, 5)
    expected: list[tuple[str, str, int]] = []
    counters = {"unknown": 0, "other": 0, "duplicate": 0}
    valid = True
    if family == "success":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2, 4], result=4, jitter=jitter)
    elif family == "aborted":
        records = _valid_records(goal_uuid, accepted=True, statuses=[1, 2, 6], result=6, jitter=jitter)
    elif family == "canceled":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2, 3, 5], result=5, jitter=jitter)
    elif family == "accepted_only":
        records = _valid_records(goal_uuid, accepted=True, statuses=[], result=None, jitter=jitter)
    elif family == "rejected":
        records = _valid_records(goal_uuid, accepted=False, statuses=[], result=None, jitter=jitter)
    elif family == "unknown_other":
        records = _valid_records(
            goal_uuid, accepted=True, statuses=[2], result=None, include_unknown_other=True, jitter=jitter
        )
        counters.update({"unknown": 1, "other": 1})
    elif family == "duplicate":
        records = _valid_records(
            goal_uuid, accepted=True, statuses=[2], result=None, duplicate_first_status=True, jitter=jitter
        )
        counters["duplicate"] = 1
    elif family == "identity_missing":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2], result=None, publisher=None, jitter=jitter)
    elif family == "multiple_publishers":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2, 4], result=None, jitter=jitter)
        records[-1]["publisher_id"] = "rmw:second-server"
        valid = False
    elif family == "conflicting_status":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2], result=None, jitter=jitter)
        records[-1]["payload"]["statuses"].append({
            "goal_uuid": goal_uuid,
            "accepted_at_ns": 138 + jitter,
            "status_code": 4,
        })
        valid = False
    elif family == "missing_goal_response":
        records = [_record(1, "send_goal_request", 135 + jitter, {})]
        valid = False
    elif family == "nonterminal_result":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2], result=2, jitter=jitter)
        valid = False
    elif family == "result_without_request":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2], result=4, jitter=jitter)
        records.pop(-2)
        for sequence, record in enumerate(records, start=1):
            record["sequence"] = sequence
            record["record_id"] = f"ros_action_record/oracle/{sequence:06d}"
        valid = False
    elif family == "config_digest":
        records = _valid_records(goal_uuid, accepted=True, statuses=[], result=None, jitter=jitter)
        valid = False
    elif family == "goal_digest":
        records = _valid_records(goal_uuid, accepted=True, statuses=[], result=None, jitter=jitter)
        valid = False
    elif family == "after_evaluation":
        records = _valid_records(goal_uuid, accepted=True, statuses=[2], result=None, jitter=jitter)
        records[-1]["event_timestamp_ns"] = 241
        valid = False
    else:
        raise CrosscheckError(f"unknown oracle family {family}")
    capture = _base_capture(config_sha, goal_uuid, goal_payload, records)
    if family == "config_digest":
        capture["adapter_config_sha256"] = "0" * 64
    if family == "goal_digest":
        capture["action"]["goal_payload_sha256"] = "0" * 64
    if valid:
        response = next(record for record in records if record["kind"] == "goal_response")
        expected.append(("goal_response", "accepted" if response["payload"]["accepted"] else "rejected", response["event_timestamp_ns"]))
        seen_status: set[tuple[int, int]] = set()
        for record in records:
            if record["kind"] != "status_array":
                continue
            for status in record["payload"]["statuses"]:
                if status["goal_uuid"] != goal_uuid or status["status_code"] == 0:
                    continue
                key = (record["event_timestamp_ns"], status["status_code"])
                if key in seen_status:
                    continue
                seen_status.add(key)
                expected.append(("action_status", STATUS_VALUE[status["status_code"]], record["event_timestamp_ns"]))
        for record in records:
            if record["kind"] == "result_response":
                expected.append(("action_result", STATUS_VALUE[record["payload"]["status_code"]], record["event_timestamp_ns"]))
        expected.sort(key=lambda item: (item[2], item[0], item[1]))
    return capture, valid, expected, counters


def crosscheck(case_count: int, seed: int) -> dict[str, Any]:
    if case_count <= 0:
        raise CrosscheckError("--cases must be positive")
    families = [
        "success",
        "aborted",
        "canceled",
        "accepted_only",
        "rejected",
        "unknown_other",
        "duplicate",
        "identity_missing",
        "multiple_publishers",
        "conflicting_status",
        "missing_goal_response",
        "nonterminal_result",
        "result_without_request",
        "config_digest",
        "goal_digest",
        "after_evaluation",
    ]
    rng = random.Random(seed)
    assertions = 0
    results: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_path, model = _prepare_model(root)
        config_path = root / "config.json"
        _write(config_path, _config(model_path, model))
        config_sha = _sha_path(config_path)
        for index in range(case_count):
            family = families[index % len(families)]
            family_counts[family] += 1
            case_root = root / f"case-{index:04d}"
            capture, valid, expected, counters = _case(family, config_sha, rng, index)
            capture_path = case_root / "capture.json"
            source_path = case_root / "evidence" / "ros-action.json"
            bundle_path = case_root / "bundle.json"
            report_path = case_root / "report.json"
            _write(capture_path, capture)
            process = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER),
                    "normalize",
                    str(model_path),
                    "--config",
                    str(config_path),
                    "--capture",
                    str(capture_path),
                    "--evidence-source",
                    str(source_path),
                    "--bundle",
                    str(bundle_path),
                    "--report",
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assertions += 1
            if valid and process.returncode != 0:
                raise CrosscheckError(f"case {index} {family} unexpectedly failed: {process.stderr}")
            if not valid and process.returncode == 0:
                raise CrosscheckError(f"case {index} {family} unexpectedly normalized")
            if not valid:
                assertions += 1
                if any(path.exists() for path in (source_path, bundle_path, report_path)):
                    raise CrosscheckError(f"case {index} {family} wrote output on rejection")
                results.append({"case": index, "family": family, "expected": "rejected", "status": "passed"})
                continue
            source = json.loads(source_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            actual = sorted(
                ((
                    record["evidence_type"],
                    record["value"],
                    record["observed_at_ns"],
                ) for record in source["records"]),
                key=lambda item: (item[2], item[0], item[1]),
            )
            assertions += 1
            if source["schema_version"] != SOURCE_SCHEMA:
                raise CrosscheckError(f"case {index} emitted wrong evidence source schema")
            assertions += 1
            if actual != expected:
                raise CrosscheckError(f"case {index} lifecycle mismatch expected={expected}, actual={actual}")
            status_report = report["status_normalization"]
            for key, report_key in (
                ("unknown", "unknown_target_status_count"),
                ("other", "ignored_other_goal_status_count"),
                ("duplicate", "exact_duplicate_status_count"),
            ):
                assertions += 1
                if status_report[report_key] != counters[key]:
                    raise CrosscheckError(
                        f"case {index} {report_key} expected {counters[key]}, got {status_report[report_key]}"
                    )
            assertions += 3
            if report["feedback"]["promoted_to_condition_or_effect_evidence"] is not False:
                raise CrosscheckError(f"case {index} promoted feedback")
            if "not established" not in report["epistemic_boundary"]["physical_and_causal"]:
                raise CrosscheckError(f"case {index} overclaimed physical or causal truth")
            if _sha_path(source_path) != report["outputs"]["lifecycle_evidence_source"]["sha256"]:
                raise CrosscheckError(f"case {index} source digest mismatch")
            results.append({
                "case": index,
                "family": family,
                "expected": "normalized",
                "lifecycle_record_count": len(actual),
                "status": "passed",
            })
    return {
        "schema_version": "robot-spatial-ros-action-adapter-crosscheck.v1",
        "status": "passed",
        "seed": seed,
        "case_count": case_count,
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "assertion_count": assertions,
        "cases": results,
        "oracle_independence": {
            "imports_ros_action_adapter": False,
            "expected_lifecycle_derived_directly_from_raw_capture": True,
            "adapter_exercised_through_cli_subprocess": True,
            "valid_and_invalid_cases": True,
        },
        "limitations": [
            "Synthetic offline traces do not verify live ROS middleware, QoS delivery, or an action server.",
            "The cross-check validates deterministic capture normalization, not producer truthfulness or physical execution.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = crosscheck(args.cases, args.seed)
        output = args.out.expanduser().resolve()
        if output.exists():
            raise CrosscheckError(f"output path already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_json_bytes(report))
        print(_json_bytes({
            "status": report["status"],
            "report": str(output),
            "sha256": _sha_path(output),
            "case_count": report["case_count"],
            "family_count": report["family_count"],
            "assertion_count": report["assertion_count"],
        }).decode("utf-8"), end="")
        return 0
    except (CrosscheckError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
