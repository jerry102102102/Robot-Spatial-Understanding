"""Controlled simulation replay comparison for bounded causal-contribution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import EvidenceError, IntegrityError
from .report import AssuranceReport
from .simulation import SimulationRun
from .task import TaskSpec
from .util import sha256_json, write_json


COUNTERFACTUAL_SCHEMA = "robot-spatial-counterfactual-assurance.v1"
STATE_CHANNELS = ("joint_state", "pose", "odometry", "contact", "collision", "deformable")


def _jsonable(value: np.ndarray) -> Any:
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def initial_state_fingerprint(run: SimulationRun) -> str:
    state: dict[str, Any] = {}
    for channel in STATE_CHANNELS:
        if not run.channel_available(channel):
            continue
        arrays = run.stream(channel)
        if len(arrays["time_s"]) == 0:
            continue
        sample_count = len(arrays["time_s"])
        channel_state: dict[str, Any] = {}
        for name, values in arrays.items():
            if values.ndim > 0 and values.shape[0] == sample_count and name not in {
                "joint_ids",
                "entity_ids",
                "sensor_ids",
            }:
                channel_state[name] = _jsonable(values[0])
            else:
                channel_state[name] = _jsonable(values)
        state[channel] = channel_state
    return sha256_json(state)


@dataclass(frozen=True)
class CounterfactualAssurance:
    data: dict[str, Any]

    @classmethod
    def compare(
        cls,
        action_run: SimulationRun,
        control_run: SimulationRun,
        task: TaskSpec,
    ) -> "CounterfactualAssurance":
        compatibility_fields = ("simulator", "seed", "timestep_s", "clock", "robot", "world", "conventions")
        mismatches = [
            field
            for field in compatibility_fields
            if action_run.manifest.get(field) != control_run.manifest.get(field)
        ]
        if mismatches:
            raise IntegrityError(f"counterfactual runs differ in required bindings: {mismatches}")
        action_initial = initial_state_fingerprint(action_run)
        control_initial = initial_state_fingerprint(control_run)
        if action_initial != control_initial:
            raise IntegrityError("counterfactual runs do not have the same normalized initial-state fingerprint")
        action_intervention = action_run.manifest.get("intervention", {})
        control_intervention = control_run.manifest.get("intervention", {})
        if action_intervention == control_intervention:
            raise EvidenceError("counterfactual runs must declare different interventions")
        if control_intervention.get("type") not in {"no_op", "controlled_perturbation"}:
            raise EvidenceError("control run intervention.type must be no_op or controlled_perturbation")
        action_report = AssuranceReport.evaluate(action_run, task)
        control_report = AssuranceReport.evaluate(control_run, task)
        action_status = action_report.data["verdict"]["simulation_bounded_physical_success"]
        control_status = control_report.data["verdict"]["simulation_bounded_physical_success"]
        if "conflicting" in {action_status, control_status}:
            contribution = "conflicting"
            summary = "The controlled comparison contains conflicting outcome evidence."
        elif "unknown" in {action_status, control_status}:
            contribution = "unknown"
            summary = "The controlled comparison lacks sufficient evidence for one or both outcomes."
        elif action_status == "supported" and control_status == "refuted":
            contribution = "supported_under_controlled_simulation"
            summary = "The declared effect occurred in the action replay and not in the matched control replay."
        elif action_status == "supported" and control_status == "supported":
            contribution = "not_supported"
            summary = "The effect also occurred in the control replay, so this intervention comparison does not support action contribution."
        else:
            contribution = "not_supported"
            summary = "The action replay did not establish the declared effect under the task policy."
        data: dict[str, Any] = {
            "schema_version": COUNTERFACTUAL_SCHEMA,
            "task_id": task.task_id,
            "task_spec_sha256": task.digest,
            "action": {
                "run_id": action_run.manifest["run_id"],
                "run_manifest_sha256": action_run.digest,
                "intervention": action_intervention,
                "report_sha256": action_report.digest,
                "outcome": action_status,
            },
            "control": {
                "run_id": control_run.manifest["run_id"],
                "run_manifest_sha256": control_run.digest,
                "intervention": control_intervention,
                "report_sha256": control_report.digest,
                "outcome": control_status,
            },
            "matched_bindings": {field: action_run.manifest.get(field) for field in compatibility_fields},
            "initial_state_sha256": action_initial,
            "causal_contribution": {
                "status": contribution,
                "summary": summary,
                "scope": "controlled deterministic simulation replay only",
            },
            "limitations": [
                "This comparison supports at most contribution under the matched simulator intervention; it does not prove real-world causation, exclude hidden simulator differences, or establish safety."
            ],
        }
        data["counterfactual_sha256"] = sha256_json(data)
        return cls(data)

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        write_json(output, self.data)
        return output
