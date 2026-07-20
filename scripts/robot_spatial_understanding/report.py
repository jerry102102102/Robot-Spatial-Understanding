"""Layered simulation assurance reports and human-readable explanations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import IntegrityError, SchemaError
from .predicates import PredicateEngine, PredicateResult
from .simulation import SimulationRun
from .task import TaskSpec
from .util import ensure_new_directory, load_json, require_mapping, sha256_json, write_json


REPORT_SCHEMA = "robot-spatial-simulation-assurance-report.v1"


def _aggregate(statuses: list[str]) -> str:
    if not statuses:
        return "unknown"
    if "conflicting" in statuses:
        return "conflicting"
    if "refuted" in statuses:
        return "refuted"
    if "unknown" in statuses:
        return "unknown"
    return "supported"


@dataclass(frozen=True)
class AssuranceReport:
    """Digest-bound result with protocol, execution, effect, and truth boundaries separated."""

    data: dict[str, Any]

    @classmethod
    def evaluate(cls, run: SimulationRun, task: TaskSpec) -> "AssuranceReport":
        engine = PredicateEngine(run, task)
        results = engine.evaluate()
        goal_status = engine.goal_status()
        failure_status = engine.failure_status()
        physical_status = (
            "refuted"
            if failure_status == "supported"
            else goal_status
        )
        events = run.events()
        action_events = [event for event in events if event.get("type") in {"action_status", "action_result", "controller_status"}]
        terminal = action_events[-1] if action_events else None
        protocol_status = "unknown"
        protocol_summary = "No controller or action lifecycle report was supplied."
        if terminal is not None:
            reported = str(terminal.get("status", terminal.get("state", "unknown"))).lower()
            protocol_status = "reported"
            protocol_summary = f"The latest supplied controller/action report is {reported!r}; this is not physical-success evidence."
        trajectory_types = {
            "joint_within_tolerance",
            "joint_position_in_range",
            "joint_velocity_below_threshold",
            "frame_within_pose_tolerance",
            "frame_position_within_tolerance",
            "base_reached_goal",
            "collision_free_over_interval",
            "path_stayed_within_corridor",
            "inserted_to_depth",
        }
        trajectory_results = [result.status for result in results.values() if result.predicate_type in trajectory_types]
        task_effect_types = {
            "contact_sustained",
            "object_above_height",
            "object_follows_frame_for_duration",
            "object_inside_region",
            "object_grasped",
            "object_released_in_region",
            "inserted_to_depth",
            "deformable_keypoints_in_region",
            "deformable_shape_within_tolerance",
        }
        effect_results = [result.status for result in results.values() if result.predicate_type in task_effect_types]
        model_validation = run.manifest.get("model_validation")
        geometry_status = "unknown"
        geometry_summary = "The run does not bind a verified model-validation artifact."
        if isinstance(model_validation, dict) and model_validation.get("status") == "passed":
            geometry_status = "supported"
            geometry_summary = "The run binds a passed model-validation artifact; its scope remains the declared model checks."
        unknowns: list[dict[str, Any]] = []
        for result in results.values():
            if result.status in {"unknown", "conflicting"}:
                unknowns.append(
                    {
                        "predicate_id": result.predicate_id,
                        "status": result.status,
                        "missing_evidence": result.missing_evidence,
                    }
                )
        for channel in task.required_channels:
            if not run.channel_available(channel):
                unknowns.append({"channel": channel, "status": "unavailable"})
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "report_id": f"report/{run.manifest['run_id']}/{task.task_id}",
            "bindings": {
                "run_id": run.manifest["run_id"],
                "run_manifest_sha256": run.digest,
                "completeness_sha256": run.completeness["completeness_sha256"],
                "task_id": task.task_id,
                "task_spec_sha256": task.digest,
                "simulator": run.manifest["simulator"],
                "adapter": run.manifest["adapter"],
                "interval": run.manifest["interval"],
                "clock": run.manifest["clock"],
            },
            "completeness": {
                "status": run.completeness["status"],
                "channel_statuses": {
                    name: channel["status"] for name, channel in run.completeness["channels"].items()
                },
            },
            "predicates": [results[predicate["predicate_id"]].to_dict() for predicate in task.predicates],
            "verdict": {
                "goal_status": goal_status,
                "failure_condition_status": failure_status,
                "simulation_bounded_physical_success": physical_status,
                "confirmed_success": physical_status == "supported",
            },
            "layers": {
                "model_geometry_validity": {"status": geometry_status, "summary": geometry_summary},
                "controller_action_protocol": {
                    "status": protocol_status,
                    "summary": protocol_summary,
                    "latest_report": terminal,
                },
                "trajectory_execution": {
                    "status": _aggregate(trajectory_results),
                    "summary": "Aggregated only from declared trajectory, pose, path, and collision predicates.",
                },
                "observed_task_effects": {
                    "status": _aggregate(effect_results),
                    "summary": "Aggregated only from declared effect predicates and their evidence.",
                },
                "simulation_bounded_physical_success": {
                    "status": physical_status,
                    "summary": "Bounded to the task spec, simulator, model/assets, recorded channels, sampling, thresholds, and interval.",
                },
                "causation": {
                    "status": "unknown",
                    "summary": "Temporal order alone does not establish causation; no bound counterfactual replay was supplied.",
                },
                "authorization": {
                    "status": "unknown",
                    "summary": "A simulation result does not establish organizational or dispatch authorization.",
                },
                "safety": {
                    "status": "unknown",
                    "summary": "A sampled simulation episode is not a hardware safety certificate.",
                },
            },
            "unknowns": unknowns,
            "boundaries": {
                "official_oracle_used_for_prediction": False,
                "simulation_only": True,
                "hardware_truth_established": False,
                "causation_established": False,
                "authorization_established": False,
                "safety_established": False,
            },
        }
        report["report_sha256"] = sha256_json(report)
        return cls(report)

    @classmethod
    def load(cls, path: str | Path, *, verify_digest: bool = True) -> "AssuranceReport":
        data = require_mapping(load_json(Path(path)), "assurance report")
        if data.get("schema_version") != REPORT_SCHEMA:
            raise SchemaError(f"report schema must be {REPORT_SCHEMA!r}")
        expected = sha256_json({key: value for key, value in data.items() if key != "report_sha256"})
        if verify_digest and data.get("report_sha256") != expected:
            raise IntegrityError("assurance report digest mismatch")
        return cls(data)

    @property
    def digest(self) -> str:
        return str(self.data["report_sha256"])

    def write(self, output: str | Path) -> Path:
        directory = Path(output)
        ensure_new_directory(directory)
        path = directory / "report.json"
        write_json(path, self.data)
        return path

    def to_markdown(self) -> str:
        verdict = self.data["verdict"]
        lines = [
            "# Simulation Assurance Report",
            "",
            f"- Run: `{self.data['bindings']['run_id']}`",
            f"- Task: `{self.data['bindings']['task_id']}`",
            f"- Report digest: `{self.data['report_sha256']}`",
            f"- Goal: **{verdict['goal_status']}**",
            f"- Simulation-bounded physical success: **{verdict['simulation_bounded_physical_success']}**",
            "",
            "## Evidence layers",
            "",
            "| Layer | Status | Meaning |",
            "| --- | --- | --- |",
        ]
        for name, layer in self.data["layers"].items():
            lines.append(f"| `{name}` | `{layer['status']}` | {layer['summary']} |")
        lines.extend(["", "## Predicate results", ""])
        for predicate in self.data["predicates"]:
            lines.extend(
                [
                    f"### `{predicate['predicate_id']}` — {predicate['status']}",
                    "",
                    predicate["summary"],
                    "",
                    f"Evidence digest: `{predicate['evidence_sha256']}`",
                ]
            )
            if predicate["missing_evidence"]:
                lines.append("")
                lines.append("Missing or conflicting evidence:")
                lines.append("")
                for item in predicate["missing_evidence"]:
                    lines.append(f"- `{item}`")
            lines.append("")
        lines.extend(
            [
                "## Non-guarantees",
                "",
                "- The official benchmark oracle was not used to produce this prediction.",
                "- This report does not establish real-hardware truth, causation, authorization, or safety.",
                "- Every supported claim is limited to the bound simulator, assets, clock, streams, interval, and thresholds.",
                "",
            ]
        )
        return "\n".join(lines)

    def explain_to(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_markdown(), encoding="utf-8")
        return output
