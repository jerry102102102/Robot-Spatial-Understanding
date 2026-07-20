"""Bridge simulation predicate results into the existing action-assurance evidence source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import SchemaError
from .report import AssuranceReport
from .util import load_structured, require_list, require_mapping, require_string, sha256_json, write_json


ACTION_MAP_SCHEMA = "robot-spatial-simulation-action-map.v1"
ACTION_EVIDENCE_SOURCE_SCHEMA = "robot-spatial-action-evidence-source.v1"


def _time_values(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "time_s" and isinstance(child, (int, float)):
                result.append(float(child))
            elif key == "time_s" and isinstance(child, list):
                result.extend(float(item) for item in child if isinstance(item, (int, float)))
            else:
                result.extend(_time_values(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_time_values(child))
    return result


def build_action_evidence_source(report: AssuranceReport, mapping_path: str | Path) -> dict[str, Any]:
    mapping = require_mapping(load_structured(Path(mapping_path)), "simulation action map")
    if mapping.get("schema_version") != ACTION_MAP_SCHEMA:
        raise SchemaError(f"simulation action map schema must be {ACTION_MAP_SCHEMA!r}")
    source_id = require_string(mapping.get("source_id"), "simulation action map.source_id")
    if not source_id.startswith("evidence_source/"):
        raise SchemaError("simulation action map.source_id must use evidence_source/")
    producer = require_mapping(mapping.get("producer"), "simulation action map.producer")
    if set(producer) != {"producer_id", "producer_type"}:
        raise SchemaError("simulation action map.producer must contain exactly producer_id and producer_type")
    require_string(producer["producer_id"], "producer.producer_id")
    require_string(producer["producer_type"], "producer.producer_type")
    action_instance = require_string(mapping.get("action_instance_id"), "simulation action map.action_instance_id")
    if not action_instance.startswith("action_instance/"):
        raise SchemaError("simulation action map.action_instance_id must use action_instance/")
    by_id = {predicate["predicate_id"]: predicate for predicate in report.data["predicates"]}
    effects = require_list(mapping.get("effects"), "simulation action map.effects")
    if not effects:
        raise SchemaError("simulation action map.effects must not be empty")
    mapping_digest = sha256_json(mapping)
    clock_binding = report.data["bindings"]["clock"]
    clock = {
        "domain": require_string(clock_binding.get("domain"), "report clock.domain"),
        "unit": "nanoseconds",
        "epoch": require_string(clock_binding.get("clock_id"), "report clock.clock_id"),
    }
    evaluation_time_s = float(report.data["bindings"]["interval"]["end_s"])
    evaluation_time_ns = int(round(evaluation_time_s * 1_000_000_000))
    records: list[dict[str, Any]] = []
    used_predicates: set[str] = set()
    for index, raw in enumerate(effects):
        effect = require_mapping(raw, f"simulation action map.effects[{index}]")
        if set(effect) != {"predicate_id", "effect_id", "predicate", "bindings"}:
            raise SchemaError(
                f"simulation action map.effects[{index}] must contain predicate_id, effect_id, predicate, and bindings"
            )
        predicate_id = require_string(effect["predicate_id"], f"effects[{index}].predicate_id")
        if predicate_id in used_predicates:
            raise SchemaError(f"simulation action map repeats predicate {predicate_id!r}")
        used_predicates.add(predicate_id)
        if predicate_id not in by_id:
            raise SchemaError(f"simulation action map references absent predicate {predicate_id!r}")
        effect_id = require_string(effect["effect_id"], f"effects[{index}].effect_id")
        if not effect_id.startswith("effect/"):
            raise SchemaError(f"effects[{index}].effect_id must use effect/")
        bindings = require_mapping(effect["bindings"], f"effects[{index}].bindings")
        if not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in bindings.items()):
            raise SchemaError(f"effects[{index}].bindings must map non-empty strings to non-empty strings")
        predicate_result = by_id[predicate_id]
        truth_value = {
            "supported": "true",
            "refuted": "false",
            "unknown": "unknown",
            "conflicting": "unknown",
        }[predicate_result["status"]]
        observed_times = _time_values(predicate_result["evidence"])
        observed_time_s = max(observed_times) if observed_times else evaluation_time_s
        observed_time_ns = min(evaluation_time_ns, int(round(observed_time_s * 1_000_000_000)))
        limitations = list(predicate_result["limitations"])
        limitations.extend(
            [
                f"simulation_run_manifest_sha256={report.data['bindings']['run_manifest_sha256']}",
                f"simulation_report_sha256={report.digest}",
                f"simulation_predicate_evidence_sha256={predicate_result['evidence_sha256']}",
                f"simulation_action_map_sha256={mapping_digest}",
                "A simulation effect observation does not establish real-hardware truth or that the action caused the effect.",
            ]
        )
        records.append(
            {
                "record_id": f"evidence/simulation/{index}/{predicate_id}",
                "evidence_type": "effect_observation",
                "subject_ref": effect_id,
                "predicate": require_string(effect["predicate"], f"effects[{index}].predicate"),
                "bindings": dict(sorted(bindings.items())),
                "value": truth_value,
                "observed_at_ns": observed_time_ns,
                "valid_until_ns": evaluation_time_ns,
                "claim_scope": (
                    f"simulation predicate {predicate_id} for {action_instance}; bounded to report {report.digest}"
                ),
                "limitations": sorted(set(limitations)),
            }
        )
    return {
        "schema_version": ACTION_EVIDENCE_SOURCE_SCHEMA,
        "source_id": source_id,
        "clock": clock,
        "producer": dict(producer),
        "records": records,
    }


def write_action_evidence_source(
    report_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = AssuranceReport.load(report_path)
    source = build_action_evidence_source(report, mapping_path)
    write_json(Path(output_path), source)
    return source
