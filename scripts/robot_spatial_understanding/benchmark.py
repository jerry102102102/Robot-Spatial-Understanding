"""Oracle-isolated benchmark runner and reproducible classification metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import IntegrityError, OracleIsolationError, SchemaError
from .report import AssuranceReport
from .simulation import SimulationRun
from .task import TaskSpec, VALID_STATUSES
from .util import (
    ensure_new_directory,
    is_relative_to,
    load_structured,
    require_list,
    require_mapping,
    require_string,
    safe_relative_path,
    sha256_json,
    write_json,
)


BENCHMARK_SCHEMA = "robot-spatial-benchmark-suite.v1"
REFERENCE_SCHEMA = "robot-spatial-reference-result.v1"
BENCHMARK_REPORT_SCHEMA = "robot-spatial-benchmark-report.v1"
LABELS = ("supported", "refuted", "unknown", "conflicting")


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total == 0:
        return {"count": successes, "total": total, "estimate": None, "lower_95": None, "upper_95": None}
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "count": successes,
        "total": total,
        "estimate": probability,
        "lower_95": max(0.0, center - margin),
        "upper_95": min(1.0, center + margin),
    }


def _classification_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual, predicted in pairs:
        if actual not in VALID_STATUSES or predicted not in VALID_STATUSES:
            raise SchemaError("benchmark labels must be supported, refuted, unknown, or conflicting")
        matrix[actual][predicted] += 1
    per_label: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in LABELS if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = None if precision is None or recall is None or precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        if f1 is not None:
            f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    correct = sum(matrix[label][label] for label in LABELS)
    total = len(pairs)
    supported_true_positive = matrix["supported"]["supported"]
    supported_false_positive = sum(matrix[actual]["supported"] for actual in LABELS if actual != "supported")
    supported_negative = sum(matrix[actual][predicted] for actual in LABELS if actual != "supported" for predicted in LABELS)
    return {
        "count": total,
        "confusion_matrix": matrix,
        "accuracy": _wilson(correct, total),
        "per_label": per_label,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "confirmed_success_precision": _wilson(
            supported_true_positive,
            supported_true_positive + supported_false_positive,
        ),
        "confirmed_success_false_positive_rate": _wilson(
            supported_false_positive,
            supported_negative,
        ),
    }


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    run_path: Path
    task_path: Path
    reference_path: Path


@dataclass
class BenchmarkSuite:
    path: Path
    data: dict[str, Any]
    cases: list[BenchmarkCase]

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSuite":
        suite_path = Path(path).resolve()
        data = require_mapping(load_structured(suite_path), "benchmark suite")
        if data.get("schema_version") != BENCHMARK_SCHEMA:
            raise SchemaError(f"benchmark schema must be {BENCHMARK_SCHEMA!r}")
        require_string(data.get("suite_id"), "benchmark.suite_id")
        root = suite_path.parent
        cases: list[BenchmarkCase] = []
        seen: set[str] = set()
        for index, raw in enumerate(require_list(data.get("cases"), "benchmark.cases")):
            value = require_mapping(raw, f"benchmark.cases[{index}]")
            case_id = require_string(value.get("case_id"), f"benchmark.cases[{index}].case_id")
            if case_id in seen:
                raise SchemaError(f"duplicate benchmark case_id {case_id!r}")
            seen.add(case_id)
            run_path = safe_relative_path(root, require_string(value.get("run"), f"case {case_id}.run"), f"case {case_id}.run")
            task_path = safe_relative_path(root, require_string(value.get("task"), f"case {case_id}.task"), f"case {case_id}.task")
            reference_path = safe_relative_path(root, require_string(value.get("reference"), f"case {case_id}.reference"), f"case {case_id}.reference")
            if is_relative_to(reference_path, run_path):
                raise OracleIsolationError(f"case {case_id!r} stores its reference result inside the candidate run")
            cases.append(BenchmarkCase(case_id, run_path, task_path, reference_path))
        if not cases:
            raise SchemaError("benchmark suite must contain at least one case")
        return cls(suite_path, data, cases)

    def run(self, output: str | Path) -> dict[str, Any]:
        output_path = Path(output)
        ensure_new_directory(output_path)
        prediction_records: list[dict[str, Any]] = []
        try:
            # Phase 1: predictions. No reference path is opened in this loop.
            for case in self.cases:
                run = SimulationRun.load(case.run_path)
                task = TaskSpec.load(case.task_path)
                report = AssuranceReport.evaluate(run, task)
                case_directory = output_path / "predictions" / case.case_id
                report.write(case_directory)
                prediction_records.append(
                    {
                        "case_id": case.case_id,
                        "run_manifest_sha256": run.digest,
                        "task_spec_sha256": task.digest,
                        "report_path": str((case_directory / "report.json").relative_to(output_path)),
                        "report": report,
                    }
                )

            # Phase 2: references are revealed only after every prediction is immutable on disk.
            predicate_pairs: list[tuple[str, str]] = []
            verdict_pairs: list[tuple[str, str]] = []
            case_results: list[dict[str, Any]] = []
            for case, prediction in zip(self.cases, prediction_records):
                reference = require_mapping(load_structured(case.reference_path), f"reference {case.case_id}")
                if reference.get("schema_version") != REFERENCE_SCHEMA:
                    raise SchemaError(f"reference {case.case_id!r} schema must be {REFERENCE_SCHEMA!r}")
                if reference.get("case_id") != case.case_id:
                    raise IntegrityError(f"reference case_id mismatch for {case.case_id!r}")
                if reference.get("run_manifest_sha256") != prediction["run_manifest_sha256"]:
                    raise IntegrityError(f"reference run digest mismatch for {case.case_id!r}")
                if reference.get("task_spec_sha256") != prediction["task_spec_sha256"]:
                    raise IntegrityError(f"reference task digest mismatch for {case.case_id!r}")
                predicted_report = prediction["report"].data
                predicted_predicates = {value["predicate_id"]: value["status"] for value in predicted_report["predicates"]}
                reference_predicates = require_mapping(reference.get("predicates"), f"reference {case.case_id}.predicates")
                unscored_predicates = {
                    require_string(value, f"reference {case.case_id}.unscored_predicates")
                    for value in require_list(reference.get("unscored_predicates", []), f"reference {case.case_id}.unscored_predicates")
                }
                if set(reference_predicates) & unscored_predicates:
                    raise IntegrityError(f"reference predicate cannot be both scored and unscored for {case.case_id!r}")
                if set(reference_predicates) | unscored_predicates != set(predicted_predicates):
                    raise IntegrityError(f"reference predicate inventory mismatch for {case.case_id!r}")
                for predicate_id in sorted(reference_predicates):
                    actual = require_string(reference_predicates[predicate_id], f"reference predicate {predicate_id}")
                    predicted = predicted_predicates[predicate_id]
                    predicate_pairs.append((actual, predicted))
                reference_verdict = require_string(reference.get("verdict"), f"reference {case.case_id}.verdict")
                predicted_verdict = predicted_report["verdict"]["simulation_bounded_physical_success"]
                verdict_pairs.append((reference_verdict, predicted_verdict))
                case_results.append(
                    {
                        "case_id": case.case_id,
                        "report_sha256": predicted_report["report_sha256"],
                        "reference_sha256": sha256_json(reference),
                        "predicate_metrics": _classification_metrics(
                            [(reference_predicates[predicate_id], predicted_predicates[predicate_id]) for predicate_id in sorted(reference_predicates)]
                        ),
                        "reference_verdict": reference_verdict,
                        "predicted_verdict": predicted_verdict,
                        "verdict_agreement": reference_verdict == predicted_verdict,
                        "scored_predicates": sorted(reference_predicates),
                        "unscored_predicates": sorted(unscored_predicates),
                    }
                )
            result: dict[str, Any] = {
                "schema_version": BENCHMARK_REPORT_SCHEMA,
                "suite_id": self.data["suite_id"],
                "suite_sha256": sha256_json(self.data),
                "case_count": len(self.cases),
                "oracle_isolation": {
                    "official_labels_available_during_prediction": False,
                    "prediction_phase_completed_before_reference_load": True,
                    "reference_paths_outside_candidate_runs": True,
                },
                "predicate_metrics": _classification_metrics(predicate_pairs),
                "episode_metrics": _classification_metrics(verdict_pairs),
                "cases": case_results,
                "limitations": [
                    "Agreement measures the declared benchmark cases, versions, seeds, adapters, and official labels only; it is not proof of universal robot understanding or hardware safety."
                ],
            }
            result["benchmark_report_sha256"] = sha256_json(result)
            write_json(output_path / "benchmark-report.json", result)
            return result
        except Exception:
            # Preserve predictions for audit, but mark the partial directory so it cannot be mistaken for a completed score.
            write_json(
                output_path / "INCOMPLETE.json",
                {"status": "incomplete", "reason": "benchmark scoring did not complete; inspect the raised error"},
            )
            raise
