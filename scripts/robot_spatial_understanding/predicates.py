"""Generic evidence predicates over normalized simulator streams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .errors import EvidenceError, SchemaError
from .simulation import SimulationRun
from .task import TaskSpec, VALID_STATUSES
from .util import (
    euclidean,
    finite_number,
    pair_matches,
    quaternion_angle,
    relative_pose,
    rotate_vector,
    require_list,
    require_mapping,
    require_string,
    sha256_json,
)


EVIDENCE_SCHEMA = "robot-spatial-simulation-predicate-evidence.v1"
CONFLICT_ISSUES = frozenset({"invalid_time", "out_of_order", "conflicting_duplicate", "invalid_quaternion"})
INCOMPLETE_ISSUES = frozenset(
    {"no_samples", "gap", "incomplete_start_coverage", "incomplete_end_coverage", "missing_values"}
)


@dataclass(frozen=True)
class PredicateResult:
    predicate_id: str
    predicate_type: str
    status: str
    summary: str
    evidence: list[dict[str, Any]]
    missing_evidence: list[str]
    limitations: list[str]
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "predicate_id": self.predicate_id,
            "type": self.predicate_type,
            "status": self.status,
            "summary": self.summary,
            "evidence": self.evidence,
            "missing_evidence": self.missing_evidence,
            "limitations": self.limitations,
            "evidence_sha256": self.evidence_sha256,
        }


def _result(
    predicate: dict[str, Any],
    status: str,
    summary: str,
    *,
    evidence: list[dict[str, Any]] | None = None,
    missing: list[str] | None = None,
    limitations: list[str] | None = None,
) -> PredicateResult:
    if status not in VALID_STATUSES:
        raise EvidenceError(f"invalid predicate result status {status!r}")
    body = {
        "schema_version": EVIDENCE_SCHEMA,
        "predicate_id": predicate["predicate_id"],
        "type": predicate["type"],
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
        "missing_evidence": missing or [],
        "limitations": limitations
        or ["This predicate is established only inside the declared simulator, streams, interval, and thresholds."],
    }
    return PredicateResult(
        predicate_id=body["predicate_id"],
        predicate_type=body["type"],
        status=body["status"],
        summary=body["summary"],
        evidence=body["evidence"],
        missing_evidence=body["missing_evidence"],
        limitations=body["limitations"],
        evidence_sha256=sha256_json(body),
    )


class PredicateEngine:
    """Evaluate task predicates without importing simulator reward or success labels."""

    def __init__(self, run: SimulationRun, task: TaskSpec):
        if run.manifest["task_id"] != task.task_id:
            raise EvidenceError(
                f"run task_id {run.manifest['task_id']!r} does not match task spec {task.task_id!r}"
            )
        self.run = run
        self.task = task
        self.results: dict[str, PredicateResult] = {}
        self._predicates = {predicate["predicate_id"]: predicate for predicate in task.predicates}

    def evaluate(self) -> dict[str, PredicateResult]:
        missing_required = [channel for channel in self.task.required_channels if not self.run.channel_available(channel)]
        if missing_required:
            # Individual predicates still run so the report names exact missing dependencies.
            pass
        deferred: list[dict[str, Any]] = []
        conjunctions: list[dict[str, Any]] = []
        for predicate in self.task.predicates:
            if predicate["type"] == "evidence_conjunction":
                conjunctions.append(predicate)
            elif predicate["type"] in {"object_grasped", "object_released_in_region"}:
                deferred.append(predicate)
            else:
                self.results[predicate["predicate_id"]] = self._evaluate_one(predicate)
        for predicate in deferred:
            self.results[predicate["predicate_id"]] = self._evaluate_one(predicate)
        for predicate in conjunctions:
            self.results[predicate["predicate_id"]] = self._evaluate_one(predicate)
        return dict(self.results)

    def goal_status(self) -> str:
        if not self.results:
            self.evaluate()
        return self._expression_status(self.task.data["goal"])

    def failure_status(self) -> str | None:
        if "failure" not in self.task.data:
            return None
        if not self.results:
            self.evaluate()
        return self._expression_status(self.task.data["failure"])

    def _expression_status(self, expression: Any) -> str:
        node = require_mapping(expression, "task expression")
        if "predicate" in node:
            return self.results[str(node["predicate"])].status
        if "not" in node:
            status = self._expression_status(node["not"])
            return {"supported": "refuted", "refuted": "supported"}.get(status, status)
        for operator in ("all", "any"):
            if operator not in node:
                continue
            statuses = [
                self.results[child].status if isinstance(child, str) else self._expression_status(child)
                for child in node[operator]
            ]
            if operator == "all":
                if "refuted" in statuses:
                    return "refuted"
                if "conflicting" in statuses:
                    return "conflicting"
                if "unknown" in statuses:
                    return "unknown"
                return "supported"
            if "supported" in statuses:
                return "supported"
            if "conflicting" in statuses:
                return "conflicting"
            if "unknown" in statuses:
                return "unknown"
            return "refuted"
        raise SchemaError("invalid task expression")

    def _entity(self, value: Any, label: str) -> str:
        identifier = require_string(value, label)
        return str(self.task.data["entities"].get(identifier, identifier))

    def _number(self, value: Any, label: str) -> float:
        """Resolve a literal or a raw per-episode measurement bound to the run."""

        if not isinstance(value, dict):
            return finite_number(value, label)
        reference = require_mapping(value, label)
        measurement_id = require_string(reference.get("measurement"), f"{label}.measurement")
        measurements = require_mapping(self.run.manifest["world"].get("measurements", {}), "run.world.measurements")
        if measurement_id not in measurements:
            raise EvidenceError(f"run measurement {measurement_id!r} is absent")
        measured = finite_number(measurements[measurement_id], f"run.world.measurements.{measurement_id}")
        scale = finite_number(reference.get("scale", 1.0), f"{label}.scale")
        offset = finite_number(reference.get("offset", 0.0), f"{label}.offset")
        return measured * scale + offset

    def _window(self, predicate: dict[str, Any]) -> tuple[float, float]:
        declared = require_mapping(predicate.get("window", {}), f"{predicate['predicate_id']}.window")
        interval = self.run.manifest["interval"]
        start = finite_number(declared.get("start_s", interval["start_s"]), "predicate.window.start_s")
        end = finite_number(declared.get("end_s", interval["end_s"]), "predicate.window.end_s")
        if start < interval["start_s"] or end > interval["end_s"] or end < start:
            raise SchemaError(
                f"predicate {predicate['predicate_id']!r} window must lie inside the run interval"
            )
        return start, end

    def _guard(self, predicate: dict[str, Any], channels: Iterable[str], *, continuous: bool) -> PredicateResult | None:
        conflicting: list[str] = []
        missing: list[str] = []
        for channel in channels:
            if not self.run.channel_available(channel):
                missing.append(f"channel/{channel}: unavailable")
                continue
            report = self.run.channel_completeness(channel)
            for issue in report["issues"]:
                if issue["type"] in CONFLICT_ISSUES:
                    conflicting.append(f"channel/{channel}:{issue['type']}")
                elif continuous and issue["type"] in INCOMPLETE_ISSUES:
                    missing.append(f"channel/{channel}:{issue['type']}")
        if conflicting:
            return _result(
                predicate,
                "conflicting",
                "The required stream is internally inconsistent.",
                missing=conflicting + missing,
            )
        if missing:
            return _result(
                predicate,
                "unknown",
                "The required stream is absent or incomplete for this claim.",
                missing=missing,
            )
        return None

    def _evaluate_one(self, predicate: dict[str, Any]) -> PredicateResult:
        handlers: dict[str, Callable[[dict[str, Any]], PredicateResult]] = {
            "joint_within_tolerance": self._joint_within_tolerance,
            "joint_position_in_range": self._joint_position_in_range,
            "joint_velocity_below_threshold": self._joint_velocity_below_threshold,
            "frame_within_pose_tolerance": self._frame_within_pose_tolerance,
            "frame_position_within_tolerance": self._frame_position_within_tolerance,
            "frame_position_in_bounds": self._frame_position_in_bounds,
            "frame_velocity_below_threshold": self._frame_velocity_below_threshold,
            "base_reached_goal": self._frame_within_pose_tolerance,
            "collision_free_over_interval": self._collision_free,
            "path_stayed_within_corridor": self._path_stayed_within_corridor,
            "contact_sustained": self._contact_sustained,
            "contact_force_at_terminal": self._contact_force_at_terminal,
            "object_above_height": self._object_above_height,
            "object_follows_frame_for_duration": self._object_follows_frame,
            "object_inside_region": self._object_inside_region,
            "object_grasped": self._object_grasped,
            "object_released_in_region": self._object_released,
            "inserted_to_depth": self._inserted_to_depth,
            "evidence_conjunction": self._evidence_conjunction,
            "deformable_keypoints_in_region": self._deformable_keypoints_in_region,
            "deformable_shape_within_tolerance": self._deformable_shape_within_tolerance,
        }
        return handlers[predicate["type"]](predicate)

    def _joint_within_tolerance(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["joint_state"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        targets = require_mapping(parameters.get("targets"), "joint_within_tolerance.targets")
        tolerance = finite_number(parameters.get("tolerance", 1e-3), "joint_within_tolerance.tolerance")
        start, end = self._window(predicate)
        stream = self.run.stream("joint_state")
        times = stream["time_s"]
        rows = np.flatnonzero(np.logical_and(times >= start, times <= end))
        if rows.size == 0:
            return _result(predicate, "unknown", "No joint sample exists in the evaluation window.", missing=["joint_state sample in window"])
        row = int(rows[-1])
        age = end - float(times[row])
        max_age = finite_number(parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5), "max_sample_age_s")
        if age > max_age:
            return _result(predicate, "unknown", "The terminal joint sample is stale.", missing=[f"terminal joint sample age {age:.9g}s exceeds {max_age:.9g}s"])
        joint_ids = [str(value) for value in stream["joint_ids"]]
        evidence_values: dict[str, Any] = {}
        passed = True
        for joint_role, raw_target in targets.items():
            joint = self._entity(joint_role, f"joint target {joint_role}")
            if joint not in joint_ids:
                return _result(predicate, "unknown", f"Joint {joint!r} is absent from the stream.", missing=[f"joint/{joint}"])
            column = joint_ids.index(joint)
            if not bool(stream["position_present"][row, column]):
                return _result(predicate, "unknown", f"Joint {joint!r} has no terminal position.", missing=[f"joint/{joint}/position"])
            actual = float(stream["position"][row, column])
            target = finite_number(raw_target, f"target for {joint}")
            error = abs(actual - target)
            evidence_values[joint] = {"actual": actual, "target": target, "absolute_error": error}
            passed = passed and error <= tolerance
        status = "supported" if passed else "refuted"
        return _result(
            predicate,
            status,
            f"Terminal joint targets are {'within' if passed else 'outside'} the declared tolerance.",
            evidence=[
                {
                    "channel": "joint_state",
                    "sample_index": row,
                    "time_s": float(times[row]),
                    "tolerance": tolerance,
                    "values": evidence_values,
                    "source_sha256": self.run.manifest["channels"]["joint_state"]["sha256"],
                }
            ],
        )

    def _terminal_joint_sample(
        self,
        predicate: dict[str, Any],
        parameters: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], int] | PredicateResult:
        guard = self._guard(predicate, ["joint_state"], continuous=False)
        if guard:
            return guard
        start, end = self._window(predicate)
        stream = self.run.stream("joint_state")
        rows = np.flatnonzero(np.logical_and(stream["time_s"] >= start, stream["time_s"] <= end))
        if rows.size == 0:
            return _result(
                predicate,
                "unknown",
                "No joint sample exists in the evaluation window.",
                missing=["joint_state sample in window"],
            )
        row = int(rows[-1])
        age = end - float(stream["time_s"][row])
        max_age = finite_number(
            parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5),
            "max_sample_age_s",
        )
        if age > max_age:
            return _result(
                predicate,
                "unknown",
                "The terminal joint sample is stale.",
                missing=[f"terminal joint sample age {age:.9g}s exceeds {max_age:.9g}s"],
            )
        return stream, row

    def _joint_position_in_range(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        sample = self._terminal_joint_sample(predicate, parameters)
        if isinstance(sample, PredicateResult):
            return sample
        stream, row = sample
        ranges = require_mapping(parameters.get("ranges"), "joint_position_in_range.ranges")
        if not ranges:
            raise SchemaError("joint_position_in_range.ranges must not be empty")
        joint_ids = [str(value) for value in stream["joint_ids"]]
        values: dict[str, Any] = {}
        passed = True
        for role, raw_range in ranges.items():
            joint = self._entity(role, f"joint range {role}")
            if joint not in joint_ids:
                return _result(predicate, "unknown", f"Joint {joint!r} is absent from the stream.", missing=[f"joint/{joint}"])
            column = joint_ids.index(joint)
            if not bool(stream["position_present"][row, column]):
                return _result(predicate, "unknown", f"Joint {joint!r} has no terminal position.", missing=[f"joint/{joint}/position"])
            if isinstance(raw_range, dict):
                minimum = finite_number(raw_range.get("minimum"), f"range minimum for {joint}")
                maximum = finite_number(raw_range.get("maximum"), f"range maximum for {joint}")
            else:
                bounds = require_list(raw_range, f"range for {joint}")
                if len(bounds) != 2:
                    raise SchemaError(f"range for {joint} must contain [minimum, maximum]")
                minimum = finite_number(bounds[0], f"range minimum for {joint}")
                maximum = finite_number(bounds[1], f"range maximum for {joint}")
            if maximum < minimum:
                raise SchemaError(f"range maximum for {joint} must be at least the minimum")
            actual = float(stream["position"][row, column])
            within = minimum <= actual <= maximum
            values[joint] = {"actual": actual, "minimum": minimum, "maximum": maximum, "within": within}
            passed = passed and within
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal joint positions are {'inside' if passed else 'outside'} the declared ranges.",
            evidence=[{
                "channel": "joint_state",
                "sample_index": row,
                "time_s": float(stream["time_s"][row]),
                "values": values,
                "source_sha256": self.run.manifest["channels"]["joint_state"]["sha256"],
            }],
        )

    def _joint_velocity_below_threshold(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        sample = self._terminal_joint_sample(predicate, parameters)
        if isinstance(sample, PredicateResult):
            return sample
        stream, row = sample
        roles = require_list(parameters.get("joints"), "joint_velocity_below_threshold.joints")
        if not roles:
            raise SchemaError("joint_velocity_below_threshold.joints must not be empty")
        threshold = finite_number(
            parameters.get("maximum_abs_velocity"),
            "joint_velocity_below_threshold.maximum_abs_velocity",
        )
        if threshold < 0.0:
            raise SchemaError("joint_velocity_below_threshold.maximum_abs_velocity must be non-negative")
        joint_ids = [str(value) for value in stream["joint_ids"]]
        values: dict[str, float] = {}
        for role in roles:
            joint = self._entity(role, f"velocity joint {role}")
            if joint not in joint_ids:
                return _result(predicate, "unknown", f"Joint {joint!r} is absent from the stream.", missing=[f"joint/{joint}"])
            column = joint_ids.index(joint)
            if not bool(stream["velocity_present"][row, column]):
                return _result(predicate, "unknown", f"Joint {joint!r} has no terminal velocity.", missing=[f"joint/{joint}/velocity"])
            values[joint] = abs(float(stream["velocity"][row, column]))
        maximum = max(values.values())
        passed = maximum <= threshold
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal joint velocity is {'within' if passed else 'above'} the declared maximum absolute velocity.",
            evidence=[{
                "channel": "joint_state",
                "sample_index": row,
                "time_s": float(stream["time_s"][row]),
                "absolute_velocities": values,
                "maximum_observed": maximum,
                "maximum_allowed": threshold,
                "source_sha256": self.run.manifest["channels"]["joint_state"]["sha256"],
            }],
        )

    def _pose_rows(self, predicate: dict[str, Any], channel: str, entity: str) -> tuple[np.ndarray, np.ndarray, int]:
        stream = self.run.stream(channel)
        ids = [str(value) for value in stream["entity_ids"]]
        if entity not in ids:
            raise EvidenceError(f"entity {entity!r} is absent from channel {channel!r}")
        column = ids.index(entity)
        start, end = self._window(predicate)
        rows = np.flatnonzero(
            np.logical_and.reduce((stream["time_s"] >= start, stream["time_s"] <= end, stream["present"][:, column]))
        )
        return stream["time_s"], rows, column

    def _frame_within_pose_tolerance(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        channel = str(parameters.get("channel", "pose"))
        guard = self._guard(predicate, [channel], continuous=False)
        if guard:
            return guard
        entity = self._entity(parameters.get("entity", parameters.get("frame")), "pose predicate entity")
        try:
            times, rows, column = self._pose_rows(predicate, channel, entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"{channel}/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No pose sample exists for the entity in the evaluation window.", missing=[f"{channel}/{entity}/pose"])
        row = int(rows[-1])
        _, end = self._window(predicate)
        max_age = finite_number(parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5), "max_sample_age_s")
        if end - float(times[row]) > max_age:
            return _result(predicate, "unknown", "The terminal pose sample is stale.", missing=[f"{channel}/{entity}/fresh_pose"])
        target = require_mapping(parameters.get("target"), "pose target")
        stream = self.run.stream(channel)
        actual_position = [float(value) for value in stream["position_m"][row, column]]
        actual_quaternion = [float(value) for value in stream["quaternion_xyzw"][row, column]]
        target_entity: str | None = None
        if "entity" in target:
            target_entity = self._entity(target["entity"], "pose target entity")
            entity_ids = [str(value) for value in stream["entity_ids"]]
            if target_entity not in entity_ids:
                return _result(
                    predicate,
                    "unknown",
                    f"Target entity {target_entity!r} is absent from the pose stream.",
                    missing=[f"{channel}/{target_entity}"],
                )
            target_column = entity_ids.index(target_entity)
            if not bool(stream["present"][row, target_column]):
                return _result(
                    predicate,
                    "unknown",
                    f"Target entity {target_entity!r} has no pose at the evaluated sample.",
                    missing=[f"{channel}/{target_entity}/pose at sample {row}"],
                )
            target_position = [float(value) for value in stream["position_m"][row, target_column]]
            target_quaternion = [float(value) for value in stream["quaternion_xyzw"][row, target_column]]
            target_evidence: dict[str, Any] = {
                "entity": target_entity,
                "position_m": target_position,
                "quaternion_xyzw": target_quaternion,
            }
        else:
            target_position = require_list(target.get("position_m"), "pose target.position_m")
            target_quaternion = require_list(target.get("quaternion_xyzw"), "pose target.quaternion_xyzw")
            target_evidence = target
        position_error = euclidean(actual_position, target_position)
        orientation_error = quaternion_angle(actual_quaternion, target_quaternion)
        position_tolerance = finite_number(parameters.get("position_tolerance_m", 1e-3), "position_tolerance_m")
        orientation_tolerance = finite_number(parameters.get("orientation_tolerance_rad", math.radians(0.1)), "orientation_tolerance_rad")
        passed = position_error <= position_tolerance and orientation_error <= orientation_tolerance
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal pose is {'within' if passed else 'outside'} the declared position and orientation tolerances.",
            evidence=[
                {
                    "channel": channel,
                    "entity": entity,
                    "sample_index": row,
                    "time_s": float(times[row]),
                    "actual": {"position_m": actual_position, "quaternion_xyzw": actual_quaternion},
                    "target": target_evidence,
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "position_tolerance_m": position_tolerance,
                    "orientation_tolerance_rad": orientation_tolerance,
                    "source_sha256": self.run.manifest["channels"][channel]["sha256"],
                }
            ],
        )

    def _frame_position_within_tolerance(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        channel = str(parameters.get("channel", "pose"))
        guard = self._guard(predicate, [channel], continuous=False)
        if guard:
            return guard
        entity = self._entity(parameters.get("entity", parameters.get("frame")), "position predicate entity")
        try:
            times, rows, column = self._pose_rows(predicate, channel, entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"{channel}/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No position sample exists for the entity in the evaluation window.", missing=[f"{channel}/{entity}/position"])
        row = int(rows[-1])
        _, end = self._window(predicate)
        max_age = finite_number(parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5), "max_sample_age_s")
        if end - float(times[row]) > max_age:
            return _result(predicate, "unknown", "The terminal position sample is stale.", missing=[f"{channel}/{entity}/fresh_position"])
        target = require_mapping(parameters.get("target"), "position target")
        stream = self.run.stream(channel)
        actual = [float(value) for value in stream["position_m"][row, column]]
        if "entity" in target:
            target_entity = self._entity(target["entity"], "position target entity")
            entity_ids = [str(value) for value in stream["entity_ids"]]
            if target_entity not in entity_ids:
                return _result(predicate, "unknown", f"Target entity {target_entity!r} is absent from the pose stream.", missing=[f"{channel}/{target_entity}"])
            target_column = entity_ids.index(target_entity)
            if not bool(stream["present"][row, target_column]):
                return _result(predicate, "unknown", f"Target entity {target_entity!r} has no position at the evaluated sample.", missing=[f"{channel}/{target_entity}/position at sample {row}"])
            target_position = [float(value) for value in stream["position_m"][row, target_column]]
            target_evidence: dict[str, Any] = {"entity": target_entity, "position_m": target_position}
        else:
            target_position = [finite_number(value, "position target component") for value in require_list(target.get("position_m"), "position target.position_m")]
            if len(target_position) != 3:
                raise SchemaError("position target.position_m must contain three values")
            target_evidence = {"position_m": target_position}
        axes_raw = require_list(parameters.get("axes", ["x", "y", "z"]), "frame position axes")
        axis_lookup = {"x": 0, "y": 1, "z": 2}
        if not axes_raw or any(str(axis) not in axis_lookup for axis in axes_raw) or len(set(map(str, axes_raw))) != len(axes_raw):
            raise SchemaError("frame position axes must be a non-empty unique subset of x, y, z")
        axes = [str(axis) for axis in axes_raw]
        axis_indices = [axis_lookup[axis] for axis in axes]
        error_m = euclidean([actual[index] for index in axis_indices], [target_position[index] for index in axis_indices])
        tolerance = finite_number(parameters.get("position_tolerance_m", 1e-3), "position_tolerance_m")
        exclusive = bool(parameters.get("exclusive", False))
        passed = error_m < tolerance if exclusive else error_m <= tolerance
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal position is {'within' if passed else 'outside'} the declared tolerance.",
            evidence=[{
                "channel": channel,
                "entity": entity,
                "sample_index": row,
                "time_s": float(times[row]),
                "actual_position_m": actual,
                "target": target_evidence,
                "axes": axes,
                "position_error_m": error_m,
                "position_tolerance_m": tolerance,
                "exclusive": exclusive,
                "source_sha256": self.run.manifest["channels"][channel]["sha256"],
            }],
        )

    def _frame_position_in_bounds(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "frame_position_in_bounds.entity")
        reference_raw = parameters.get("reference")
        reference = self._entity(reference_raw, "frame_position_in_bounds.reference") if reference_raw is not None else None
        coordinate_frame = str(parameters.get("coordinate_frame", "world"))
        if coordinate_frame not in {"world", "reference"}:
            raise SchemaError("frame_position_in_bounds.coordinate_frame must be world or reference")
        if coordinate_frame == "reference" and reference is None:
            raise SchemaError("reference coordinates require a reference entity")
        stream = self.run.stream("pose")
        entity_ids = [str(value) for value in stream["entity_ids"]]
        missing = [name for name in (entity, reference) if name is not None and name not in entity_ids]
        if missing:
            return _result(predicate, "unknown", "A bounded-position entity is absent.", missing=[f"pose/{name}" for name in missing])
        entity_column = entity_ids.index(entity)
        reference_column = entity_ids.index(reference) if reference is not None else None
        start, end = self._window(predicate)
        mask = np.logical_and.reduce(
            (stream["time_s"] >= start, stream["time_s"] <= end, stream["present"][:, entity_column])
        )
        if reference_column is not None:
            mask = np.logical_and(mask, stream["present"][:, reference_column])
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return _result(predicate, "unknown", "No synchronized bounded-position sample exists.", missing=[f"pose/{entity}+{reference}"])
        row = int(rows[-1])
        max_age = finite_number(parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5), "max_sample_age_s")
        if end - float(stream["time_s"][row]) > max_age:
            return _result(predicate, "unknown", "The terminal bounded-position sample is stale.", missing=[f"pose/{entity}/fresh_position"])
        entity_position = [float(value) for value in stream["position_m"][row, entity_column]]
        if reference_column is None:
            coordinates = entity_position
        else:
            reference_position = [float(value) for value in stream["position_m"][row, reference_column]]
            if coordinate_frame == "world":
                coordinates = [actual - origin for actual, origin in zip(entity_position, reference_position)]
            else:
                coordinates, _ = relative_pose(
                    reference_position,
                    stream["quaternion_xyzw"][row, reference_column],
                    entity_position,
                    stream["quaternion_xyzw"][row, entity_column],
                )
        bounds = require_mapping(parameters.get("bounds"), "frame_position_in_bounds.bounds")
        axis_lookup = {"x": 0, "y": 1, "z": 2}
        if not bounds or any(str(axis) not in axis_lookup for axis in bounds):
            raise SchemaError("frame_position_in_bounds.bounds must use one or more x, y, z keys")
        checks: dict[str, Any] = {}
        passed = True
        for raw_axis, raw_bound in bounds.items():
            axis = str(raw_axis)
            value = float(coordinates[axis_lookup[axis]])
            bound = require_mapping(raw_bound, f"bounds.{axis}")
            if "minimum_m" not in bound and "maximum_m" not in bound:
                raise SchemaError(f"bounds.{axis} requires minimum_m and/or maximum_m")
            axis_passed = True
            resolved: dict[str, Any] = {"value_m": value}
            if "minimum_m" in bound:
                try:
                    minimum = self._number(bound["minimum_m"], f"bounds.{axis}.minimum_m")
                except EvidenceError as error:
                    return _result(predicate, "unknown", str(error), missing=[str(error)])
                inclusive = bool(bound.get("minimum_inclusive", True))
                axis_passed = axis_passed and (value >= minimum if inclusive else value > minimum)
                resolved.update({"minimum_m": minimum, "minimum_inclusive": inclusive})
            if "maximum_m" in bound:
                try:
                    maximum = self._number(bound["maximum_m"], f"bounds.{axis}.maximum_m")
                except EvidenceError as error:
                    return _result(predicate, "unknown", str(error), missing=[str(error)])
                inclusive = bool(bound.get("maximum_inclusive", True))
                axis_passed = axis_passed and (value <= maximum if inclusive else value < maximum)
                resolved.update({"maximum_m": maximum, "maximum_inclusive": inclusive})
            resolved["passed"] = axis_passed
            checks[axis] = resolved
            passed = passed and axis_passed
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal position is {'inside' if passed else 'outside'} the declared component bounds.",
            evidence=[{
                "channel": "pose",
                "sample_index": row,
                "time_s": float(stream["time_s"][row]),
                "entity": entity,
                "reference": reference,
                "coordinate_frame": coordinate_frame,
                "coordinates_m": [float(value) for value in coordinates],
                "checks": checks,
                "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
            }],
        )

    def _frame_velocity_below_threshold(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "frame_velocity_below_threshold.entity")
        try:
            times, rows, column = self._pose_rows(predicate, "pose", entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"pose/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No velocity-bearing pose sample exists.", missing=[f"pose/{entity}"])
        row = int(rows[-1])
        stream = self.run.stream("pose")
        _, end = self._window(predicate)
        max_age = finite_number(
            parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5),
            "max_sample_age_s",
        )
        if end - float(times[row]) > max_age:
            return _result(
                predicate,
                "unknown",
                "The terminal frame-velocity sample is stale.",
                missing=[f"pose/{entity}/fresh_velocity"],
            )
        thresholds = {
            "linear": parameters.get("maximum_linear_speed_mps"),
            "angular": parameters.get("maximum_angular_speed_radps"),
        }
        if all(value is None for value in thresholds.values()):
            raise SchemaError("frame_velocity_below_threshold requires a linear and/or angular threshold")
        evidence: dict[str, Any] = {
            "channel": "pose",
            "entity": entity,
            "sample_index": row,
            "time_s": float(times[row]),
            "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
        }
        passed = True
        for kind, raw_threshold in thresholds.items():
            if raw_threshold is None:
                continue
            array_name = f"{kind}_velocity_" + ("mps" if kind == "linear" else "radps")
            mask_name = f"{kind}_velocity_present"
            if array_name not in stream or mask_name not in stream or not bool(stream[mask_name][row, column]):
                return _result(predicate, "unknown", f"Terminal {kind} velocity is absent.", missing=[f"pose/{entity}/{array_name}"])
            vector = [float(value) for value in stream[array_name][row, column]]
            speed = float(np.linalg.norm(vector))
            threshold = finite_number(raw_threshold, f"maximum_{kind}_speed")
            evidence[f"{kind}_velocity"] = vector
            evidence[f"{kind}_speed"] = speed
            evidence[f"maximum_{kind}_speed"] = threshold
            passed = passed and speed <= threshold
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal frame velocity is {'within' if passed else 'above'} the declared thresholds.",
            evidence=[evidence],
        )

    def _pair_rows(self, predicate: dict[str, Any], channel: str, pair: list[Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        stream = self.run.stream(channel)
        start, end = self._window(predicate)
        rows = [
            index
            for index, (time_s, body_a, body_b) in enumerate(zip(stream["time_s"], stream["body_a"], stream["body_b"]))
            if start <= float(time_s) <= end
            and pair_matches(str(body_a), str(body_b), [self._entity(pair[0], "pair[0]"), self._entity(pair[1], "pair[1]")])
        ]
        return stream, np.asarray(rows, dtype=np.int64)

    def _collision_free(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        pair = parameters.get("pair")
        allowed_raw = require_list(parameters.get("allowed_pairs", []), "collision_free_over_interval.allowed_pairs")
        ignored_raw = require_list(parameters.get("ignored_pairs", []), "collision_free_over_interval.ignored_pairs")

        def configured_pairs(raw_pairs: list[Any], label: str) -> list[list[str]]:
            pairs: list[list[str]] = []
            for raw_pair in raw_pairs:
                values = require_list(raw_pair, label)
                if len(values) != 2:
                    raise SchemaError(f"{label} must contain exactly two entity roles")
                pairs.append([
                    self._entity(values[0], f"{label}[0]"),
                    self._entity(values[1], f"{label}[1]"),
                ])
            return pairs

        allowed_pairs = configured_pairs(allowed_raw, "allowed collision pair")
        ignored_pairs = configured_pairs(ignored_raw, "ignored collision pair")
        guard = self._guard(predicate, ["collision"], continuous=False)
        if guard:
            return guard
        stream = self.run.stream("collision")
        start, end = self._window(predicate)
        rows: list[int] = []
        for index, time_s in enumerate(stream["time_s"]):
            if not start <= float(time_s) <= end:
                continue
            if pair is not None and not pair_matches(
                str(stream["body_a"][index]),
                str(stream["body_b"][index]),
                [self._entity(pair[0], "collision pair[0]"), self._entity(pair[1], "collision pair[1]")],
            ):
                continue
            rows.append(index)
        active_rows = [row for row in rows if bool(stream["active"][row])]
        exempt_rows = [
            row
            for row in active_rows
            if any(pair_matches(str(stream["body_a"][row]), str(stream["body_b"][row]), expected) for expected in allowed_pairs + ignored_pairs)
        ]
        active_rows = [row for row in active_rows if row not in exempt_rows]
        if active_rows:
            return _result(
                predicate,
                "refuted",
                "At least one declared collision was active in the interval.",
                evidence=[
                    {
                        "channel": "collision",
                        "sample_indices": active_rows,
                        "time_s": [float(stream["time_s"][row]) for row in active_rows],
                        "pairs": [
                            [str(stream["body_a"][row]), str(stream["body_b"][row])] for row in active_rows
                        ],
                        "source_sha256": self.run.manifest["channels"]["collision"]["sha256"],
                    }
                ],
            )
        continuous_guard = self._guard(predicate, ["collision"], continuous=True)
        if continuous_guard:
            return continuous_guard
        return _result(
            predicate,
            "supported",
            "No active declared collision was recorded in the complete interval.",
            evidence=[
                {
                    "channel": "collision",
                    "sample_indices": rows,
                    "interval": {"start_s": start, "end_s": end},
                    "pair_filter": pair,
                    "allowed_pairs": allowed_pairs,
                    "ignored_pairs": ignored_pairs,
                    "exempt_active_sample_indices": exempt_rows,
                    "source_sha256": self.run.manifest["channels"]["collision"]["sha256"],
                }
            ],
            limitations=[
                "Collision freedom covers only the simulator collision channel, declared bodies, sampling rate, and interval; it is not continuous real-world safety proof."
            ],
        )

    def _contact_sustained(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["contact"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        pair = require_list(parameters.get("pair"), "contact_sustained.pair")
        stream, rows = self._pair_rows(predicate, "contact", pair)
        if rows.size == 0:
            continuous_guard = self._guard(predicate, ["contact"], continuous=True)
            if continuous_guard:
                return continuous_guard
            return _result(predicate, "refuted", "The requested body pair never had an active contact sample.")
        minimum = finite_number(parameters.get("minimum_duration_s", 0.0), "contact_sustained.minimum_duration_s")
        minimum_force = finite_number(parameters.get("minimum_normal_force_n", 0.0), "minimum_normal_force_n")
        if minimum_force > 0.0 and "normal_force_present" in stream:
            missing_force_rows = [
                int(row)
                for row in rows
                if bool(stream["active"][row]) and not bool(stream["normal_force_present"][row])
            ]
            if missing_force_rows:
                return _result(
                    predicate,
                    "unknown",
                    "Active contact exists, but required normal-force evidence is missing.",
                    missing=[f"contact.normal_force_n at samples {missing_force_rows}"],
                )
        longest = 0.0
        segment_start: float | None = None
        selected: list[int] = []
        current: list[int] = []
        had_active = False
        for row in rows:
            force_ok = (
                "normal_force_present" not in stream
                or not bool(stream["normal_force_present"][row])
                or float(stream["normal_force_n"][row]) >= minimum_force
            )
            if bool(stream["active"][row]) and force_ok:
                had_active = True
                if segment_start is None:
                    segment_start = float(stream["time_s"][row])
                    current = []
                current.append(int(row))
                duration = float(stream["time_s"][row]) - segment_start
                if duration >= longest:
                    longest = duration
                    selected = list(current)
            else:
                segment_start = None
                current = []
        if had_active and longest >= minimum:
            return _result(
                predicate,
                "supported",
                "The requested contact pair remained active for the declared duration.",
                evidence=[
                    {
                        "channel": "contact",
                        "sample_indices": selected,
                        "longest_duration_s": longest,
                        "minimum_duration_s": minimum,
                        "minimum_normal_force_n": minimum_force,
                        "source_sha256": self.run.manifest["channels"]["contact"]["sha256"],
                    }
                ],
            )
        continuous_guard = self._guard(predicate, ["contact"], continuous=True)
        if continuous_guard:
            return continuous_guard
        return _result(
            predicate,
            "refuted",
            "Contact existed, but not for the declared sustained duration.",
            evidence=[{"channel": "contact", "longest_duration_s": longest, "minimum_duration_s": minimum}],
        )

    def _contact_force_at_terminal(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        direction_raw = parameters.get("direction")
        channels = ["contact"] + (["pose"] if direction_raw is not None else [])
        guard = self._guard(predicate, channels, continuous=False)
        if guard:
            return guard
        pair = require_list(parameters.get("pair"), "contact_force_at_terminal.pair")
        if len(pair) != 2:
            raise SchemaError("contact_force_at_terminal.pair must contain two entities")
        contact, rows = self._pair_rows(predicate, "contact", pair)
        if rows.size == 0:
            return _result(predicate, "unknown", "No terminal contact sample exists for the declared pair.", missing=[f"contact/{pair}"])
        row = int(rows[-1])
        _, end = self._window(predicate)
        max_age = finite_number(parameters.get("max_sample_age_s", self.run.manifest["timestep_s"] * 2.5), "max_sample_age_s")
        age = end - float(contact["time_s"][row])
        if age > max_age:
            return _result(predicate, "unknown", "The terminal contact sample is stale.", missing=[f"contact/{pair}/fresh_sample"])
        minimum_force = finite_number(parameters.get("minimum_force_n", 0.0), "minimum_force_n")
        if "normal_force_present" not in contact or not bool(contact["normal_force_present"][row]):
            return _result(predicate, "unknown", "Terminal contact-force magnitude is absent.", missing=[f"contact/{pair}/normal_force_n"])
        force = float(contact["normal_force_n"][row])
        active = bool(contact["active"][row])
        passed = active and force >= minimum_force
        evidence: dict[str, Any] = {
            "channel": "contact",
            "sample_index": row,
            "time_s": float(contact["time_s"][row]),
            "pair": [self._entity(pair[0], "pair[0]"), self._entity(pair[1], "pair[1]")],
            "active": active,
            "force_magnitude_n": force,
            "minimum_force_n": minimum_force,
            "source_sha256": self.run.manifest["channels"]["contact"]["sha256"],
        }
        if direction_raw is not None:
            direction = require_mapping(direction_raw, "contact_force_at_terminal.direction")
            direction_entity = self._entity(direction.get("entity"), "contact direction entity")
            local_axis = np.asarray(require_list(direction.get("local_axis"), "contact direction local_axis"), dtype=np.float64)
            if local_axis.shape != (3,) or not np.isfinite(local_axis).all() or np.linalg.norm(local_axis) <= 1e-12:
                raise SchemaError("contact direction local_axis must be a finite non-zero three-vector")
            if "present" not in contact or not bool(contact["present"][row]):
                return _result(predicate, "unknown", "Terminal contact-force vector is absent.", missing=[f"contact/{pair}/force_n"])
            pose = self.run.stream("pose")
            entity_ids = [str(value) for value in pose["entity_ids"]]
            if direction_entity not in entity_ids:
                return _result(predicate, "unknown", "Contact direction entity pose is absent.", missing=[f"pose/{direction_entity}"])
            column = entity_ids.index(direction_entity)
            synchronized = np.flatnonzero(
                np.logical_and(
                    np.isclose(pose["time_s"], contact["time_s"][row], rtol=0.0, atol=1e-12),
                    pose["present"][:, column],
                )
            )
            if synchronized.size == 0:
                return _result(predicate, "unknown", "No synchronized pose exists for the force direction.", missing=[f"pose/{direction_entity}/at_contact_time"])
            pose_row = int(synchronized[-1])
            local_axis = local_axis / np.linalg.norm(local_axis)
            world_axis = np.asarray(rotate_vector(pose["quaternion_xyzw"][pose_row, column], local_axis), dtype=np.float64)
            force_vector = np.asarray(contact["force_n"][row], dtype=np.float64)
            force_norm = float(np.linalg.norm(force_vector))
            angle: float | None = None
            if force_norm > 1e-12:
                cosine = float(np.clip(np.dot(world_axis, force_vector / force_norm), -1.0, 1.0))
                angle = float(math.acos(cosine))
            maximum_angle = finite_number(direction.get("maximum_angle_rad"), "maximum_angle_rad")
            passed = passed and angle is not None and angle <= maximum_angle
            evidence.update(
                {
                    "direction_entity": direction_entity,
                    "local_axis": [float(value) for value in local_axis],
                    "world_axis": [float(value) for value in world_axis],
                    "force_vector_n": [float(value) for value in force_vector],
                    "angle_rad": angle,
                    "maximum_angle_rad": maximum_angle,
                    "pose_source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
                }
            )
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Terminal contact force is {'inside' if passed else 'outside'} the declared magnitude/direction bounds.",
            evidence=[evidence],
        )

    def _object_above_height(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "object_above_height.entity")
        axis_name = str(parameters.get("axis", "z"))
        if axis_name not in {"x", "y", "z"}:
            raise SchemaError("object_above_height.axis must be x, y, or z")
        axis = {"x": 0, "y": 1, "z": 2}[axis_name]
        duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        try:
            times, rows, column = self._pose_rows(predicate, "pose", entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"pose/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No object pose exists in the evaluation window.", missing=[f"pose/{entity}"])
        pose_stream = self.run.stream("pose")
        heights = pose_stream["position_m"][rows, column, axis]
        has_absolute = "minimum_m" in parameters
        has_delta = "minimum_delta_m" in parameters
        if has_absolute == has_delta:
            raise SchemaError("object_above_height requires exactly one of minimum_m or minimum_delta_m")
        initial_height: float | None = None
        if has_delta:
            initial_rows = np.flatnonzero(pose_stream["present"][:, column])
            if initial_rows.size == 0:
                return _result(predicate, "unknown", "No initial object pose is available.", missing=[f"pose/{entity}/initial"])
            initial_height = float(pose_stream["position_m"][int(initial_rows[0]), column, axis])
            threshold = initial_height + finite_number(parameters["minimum_delta_m"], "object_above_height.minimum_delta_m")
        else:
            threshold = finite_number(parameters["minimum_m"], "object_above_height.minimum_m")
        longest = 0.0
        start_time: float | None = None
        selected: list[int] = []
        current: list[int] = []
        any_below = False
        for row, height in zip(rows, heights):
            if float(height) >= threshold:
                if start_time is None:
                    start_time = float(times[row])
                    current = []
                current.append(int(row))
                observed_duration = float(times[row]) - start_time
                if observed_duration >= longest:
                    longest = observed_duration
                    selected = list(current)
            else:
                any_below = True
                start_time = None
                current = []
        passed = longest >= duration if duration > 0.0 else float(heights[-1]) >= threshold
        if not passed:
            continuous_guard = self._guard(predicate, ["pose"], continuous=duration > 0.0)
            if continuous_guard:
                return continuous_guard
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Object height is {'above' if passed else 'not above'} the declared threshold for the required duration.",
            evidence=[
                {
                    "channel": "pose",
                    "entity": entity,
                    "axis": axis_name,
                    "terminal_height_m": float(heights[-1]),
                    "minimum_m": threshold,
                    "initial_height_m": initial_height,
                    "minimum_delta_m": parameters.get("minimum_delta_m"),
                    "longest_duration_s": longest,
                    "minimum_duration_s": duration,
                    "sample_indices": selected or [int(rows[-1])],
                    "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
                }
            ],
        )

    def _object_inside_region(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "object_inside_region.entity")
        region = require_mapping(parameters.get("region"), "object_inside_region.region")
        duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        try:
            times, rows, column = self._pose_rows(predicate, "pose", entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"pose/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No object pose exists in the evaluation window.", missing=[f"pose/{entity}"])
        positions = self.run.stream("pose")["position_m"][rows, column]
        inside = [self._inside_region(position, region) for position in positions]
        longest, selected = self._longest_boolean_duration(times, rows, inside)
        passed = longest >= duration if duration > 0.0 else bool(inside[-1])
        if not passed and duration > 0.0:
            continuous_guard = self._guard(predicate, ["pose"], continuous=True)
            if continuous_guard:
                return continuous_guard
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Object is {'inside' if passed else 'outside'} the declared region under the duration policy.",
            evidence=[
                {
                    "channel": "pose",
                    "entity": entity,
                    "terminal_position_m": [float(value) for value in positions[-1]],
                    "region": region,
                    "longest_duration_s": longest,
                    "minimum_duration_s": duration,
                    "sample_indices": selected or [int(rows[-1])],
                    "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
                }
            ],
        )

    @staticmethod
    def _inside_region(position: Iterable[float], region: dict[str, Any]) -> bool:
        point = [float(value) for value in position]
        region_type = region.get("type")
        if region_type == "aabb":
            minimum = require_list(region.get("min_m"), "region.min_m")
            maximum = require_list(region.get("max_m"), "region.max_m")
            if len(minimum) != 3 or len(maximum) != 3:
                raise SchemaError("AABB region bounds must contain three values")
            return all(float(low) <= value <= float(high) for value, low, high in zip(point, minimum, maximum))
        if region_type == "sphere":
            center = require_list(region.get("center_m"), "region.center_m")
            radius = finite_number(region.get("radius_m"), "region.radius_m")
            return euclidean(point, center) <= radius
        raise SchemaError("region.type must be aabb or sphere")

    @staticmethod
    def _longest_boolean_duration(
        times: np.ndarray, rows: np.ndarray, values: Iterable[bool]
    ) -> tuple[float, list[int]]:
        longest = 0.0
        start: float | None = None
        current: list[int] = []
        selected: list[int] = []
        for row, value in zip(rows, values):
            if value:
                if start is None:
                    start = float(times[row])
                    current = []
                current.append(int(row))
                duration = float(times[row]) - start
                if duration >= longest:
                    longest = duration
                    selected = list(current)
            else:
                start = None
                current = []
        return longest, selected

    def _object_follows_frame(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=True)
        if guard:
            return guard
        parameters = predicate["parameters"]
        reference = self._entity(parameters.get("reference"), "object_follows.reference")
        target = self._entity(parameters.get("target"), "object_follows.target")
        stream = self.run.stream("pose")
        ids = [str(value) for value in stream["entity_ids"]]
        if reference not in ids or target not in ids:
            missing = [f"pose/{entity}" for entity in (reference, target) if entity not in ids]
            return _result(predicate, "unknown", "Reference or target pose is absent.", missing=missing)
        reference_column, target_column = ids.index(reference), ids.index(target)
        start, end = self._window(predicate)
        rows = np.flatnonzero(
            np.logical_and.reduce(
                (
                    stream["time_s"] >= start,
                    stream["time_s"] <= end,
                    stream["present"][:, reference_column],
                    stream["present"][:, target_column],
                )
            )
        )
        if rows.size < 2:
            return _result(predicate, "unknown", "At least two synchronized reference/target poses are required.", missing=[f"synchronized pose/{reference}+{target}"])
        relative_positions: list[list[float]] = []
        relative_quaternions: list[list[float]] = []
        for row in rows:
            position, quaternion = relative_pose(
                stream["position_m"][row, reference_column],
                stream["quaternion_xyzw"][row, reference_column],
                stream["position_m"][row, target_column],
                stream["quaternion_xyzw"][row, target_column],
            )
            relative_positions.append(position)
            relative_quaternions.append(quaternion)
        expected = parameters.get("expected_relative_pose")
        position_tolerance = finite_number(parameters.get("position_tolerance_m", 0.01), "position_tolerance_m")
        orientation_tolerance = finite_number(parameters.get("orientation_tolerance_rad", math.radians(5.0)), "orientation_tolerance_rad")
        minimum_duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        best_local_rows: list[int] = []
        best_position_errors: list[float] = []
        best_orientation_errors: list[float] = []
        if expected is not None:
            expected_pose = require_mapping(expected, "expected_relative_pose")
            baseline_position = [float(value) for value in require_list(expected_pose.get("position_m"), "expected_relative_pose.position_m")]
            baseline_quaternion = [float(value) for value in require_list(expected_pose.get("quaternion_xyzw"), "expected_relative_pose.quaternion_xyzw")]
            expectation_source = "task spec"
            position_errors = [euclidean(value, baseline_position) for value in relative_positions]
            orientation_errors = [quaternion_angle(value, baseline_quaternion) for value in relative_quaternions]
            valid = [
                position <= position_tolerance and orientation <= orientation_tolerance
                for position, orientation in zip(position_errors, orientation_errors)
            ]
            _, selected = self._longest_boolean_duration(stream["time_s"], rows, valid)
            best_local_rows = [list(rows).index(row) for row in selected]
            best_position_errors = [position_errors[index] for index in best_local_rows]
            best_orientation_errors = [orientation_errors[index] for index in best_local_rows]
        else:
            expectation_source = "start of longest stable synchronized segment"
            # A grasp can begin after reaching. Search for the longest consecutive segment whose
            # relative pose remains within tolerance of that segment's first sample.
            for start_index in range(len(rows)):
                candidate_rows: list[int] = []
                candidate_position_errors: list[float] = []
                candidate_orientation_errors: list[float] = []
                for end_index in range(start_index, len(rows)):
                    position_error = euclidean(relative_positions[end_index], relative_positions[start_index])
                    orientation_error = quaternion_angle(relative_quaternions[end_index], relative_quaternions[start_index])
                    if position_error > position_tolerance or orientation_error > orientation_tolerance:
                        break
                    candidate_rows.append(end_index)
                    candidate_position_errors.append(position_error)
                    candidate_orientation_errors.append(orientation_error)
                if not candidate_rows:
                    continue
                candidate_duration = float(
                    stream["time_s"][rows[candidate_rows[-1]]] - stream["time_s"][rows[candidate_rows[0]]]
                )
                best_duration = 0.0 if not best_local_rows else float(
                    stream["time_s"][rows[best_local_rows[-1]]] - stream["time_s"][rows[best_local_rows[0]]]
                )
                if candidate_duration >= best_duration:
                    best_local_rows = candidate_rows
                    best_position_errors = candidate_position_errors
                    best_orientation_errors = candidate_orientation_errors
        selected_rows = [int(rows[index]) for index in best_local_rows]
        duration = 0.0 if len(selected_rows) < 2 else float(
            stream["time_s"][selected_rows[-1]] - stream["time_s"][selected_rows[0]]
        )
        passed = duration >= minimum_duration
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Target {'maintained' if passed else 'did not maintain'} the declared relative pose to the reference.",
            evidence=[
                {
                    "channel": "pose",
                    "reference": reference,
                    "target": target,
                    "sample_indices": selected_rows,
                    "duration_s": duration,
                    "minimum_duration_s": minimum_duration,
                    "maximum_position_drift_m": max(best_position_errors, default=None),
                    "position_tolerance_m": position_tolerance,
                    "maximum_orientation_drift_rad": max(best_orientation_errors, default=None),
                    "orientation_tolerance_rad": orientation_tolerance,
                    "relative_pose_expectation_source": expectation_source,
                    "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
                }
            ],
        )

    def _path_stayed_within_corridor(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        channel = str(parameters.get("channel", "odometry"))
        guard = self._guard(predicate, [channel], continuous=True)
        if guard:
            return guard
        entity = self._entity(parameters.get("entity"), "path corridor entity")
        try:
            times, rows, column = self._pose_rows(predicate, channel, entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"{channel}/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No path samples exist in the interval.", missing=[f"{channel}/{entity}/path"])
        corridor = require_mapping(parameters.get("corridor"), "path corridor")
        positions = self.run.stream(channel)["position_m"][rows, column]
        if corridor.get("type") == "aabb":
            inside = [self._inside_region(position, corridor) for position in positions]
            distances = [0.0 if value else math.inf for value in inside]
        elif corridor.get("type") == "polyline_xy":
            points = require_list(corridor.get("points_m"), "corridor.points_m")
            if len(points) < 2:
                raise SchemaError("polyline_xy corridor requires at least two points")
            half_width = finite_number(corridor.get("half_width_m"), "corridor.half_width_m")
            distances = [self._polyline_distance_xy(position, points) for position in positions]
            inside = [distance <= half_width for distance in distances]
        else:
            raise SchemaError("corridor.type must be aabb or polyline_xy")
        passed = all(inside)
        violating = [int(row) for row, value in zip(rows, inside) if not value]
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Recorded path {'stayed inside' if passed else 'left'} the declared corridor.",
            evidence=[
                {
                    "channel": channel,
                    "entity": entity,
                    "sample_indices": [int(row) for row in rows],
                    "violating_sample_indices": violating,
                    "maximum_polyline_distance_m": None if any(math.isinf(value) for value in distances) else max(distances),
                    "corridor": corridor,
                    "source_sha256": self.run.manifest["channels"][channel]["sha256"],
                }
            ],
        )

    @staticmethod
    def _polyline_distance_xy(position: Iterable[float], points: list[Any]) -> float:
        x, y = [float(value) for value in list(position)[:2]]
        best = math.inf
        for left, right in zip(points, points[1:]):
            ax, ay = [float(value) for value in require_list(left, "corridor point")[:2]]
            bx, by = [float(value) for value in require_list(right, "corridor point")[:2]]
            dx, dy = bx - ax, by - ay
            denominator = dx * dx + dy * dy
            ratio = 0.0 if denominator <= 1e-15 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / denominator))
            closest_x, closest_y = ax + ratio * dx, ay + ratio * dy
            best = min(best, math.hypot(x - closest_x, y - closest_y))
        return best

    def _inserted_to_depth(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["pose"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "inserted entity")
        reference = self._entity(parameters.get("reference"), "insertion reference")
        axis = np.asarray(require_list(parameters.get("axis"), "insertion axis"), dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-12:
            raise SchemaError("insertion axis must be a finite non-zero three-vector")
        axis = axis / np.linalg.norm(axis)
        stream = self.run.stream("pose")
        ids = [str(value) for value in stream["entity_ids"]]
        if entity not in ids or reference not in ids:
            return _result(predicate, "unknown", "Insertion entity or reference pose is absent.", missing=[f"pose/{entity}", f"pose/{reference}"])
        entity_column, reference_column = ids.index(entity), ids.index(reference)
        start, end = self._window(predicate)
        rows = np.flatnonzero(
            np.logical_and.reduce(
                (
                    stream["time_s"] >= start,
                    stream["time_s"] <= end,
                    stream["present"][:, entity_column],
                    stream["present"][:, reference_column],
                )
            )
        )
        if rows.size == 0:
            return _result(predicate, "unknown", "No synchronized insertion pose exists.", missing=[f"pose/{entity}+{reference}"])
        row = int(rows[-1])
        delta = stream["position_m"][row, entity_column] - stream["position_m"][row, reference_column]
        depth = float(np.dot(delta, axis))
        lateral = float(np.linalg.norm(delta - depth * axis))
        minimum_depth = finite_number(parameters.get("minimum_depth_m"), "minimum_depth_m")
        maximum_lateral = finite_number(parameters.get("maximum_lateral_error_m", math.inf), "maximum_lateral_error_m")
        passed = depth >= minimum_depth and lateral <= maximum_lateral
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Insertion is {'deep and aligned enough' if passed else 'outside the declared depth/alignment bounds'}.",
            evidence=[
                {
                    "channel": "pose",
                    "sample_index": row,
                    "time_s": float(stream["time_s"][row]),
                    "entity": entity,
                    "reference": reference,
                    "axis": [float(value) for value in axis],
                    "depth_m": depth,
                    "minimum_depth_m": minimum_depth,
                    "lateral_error_m": lateral,
                    "maximum_lateral_error_m": maximum_lateral,
                    "source_sha256": self.run.manifest["channels"]["pose"]["sha256"],
                }
            ],
        )

    def _deformable_rows(
        self, predicate: dict[str, Any], entity: str
    ) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
        stream = self.run.stream("deformable")
        entities = [str(value) for value in stream["entity_ids"]]
        if entity not in entities:
            raise EvidenceError(f"deformable entity {entity!r} is absent")
        column = entities.index(entity)
        start, end = self._window(predicate)
        rows = np.flatnonzero(
            np.logical_and(
                np.logical_and(stream["time_s"] >= start, stream["time_s"] <= end),
                np.any(stream["keypoint_present"][:, column], axis=1),
            )
        )
        return stream, rows, column

    def _deformable_keypoints_in_region(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        guard = self._guard(predicate, ["deformable"], continuous=duration > 0.0)
        if guard:
            return guard
        entity = self._entity(parameters.get("entity"), "deformable entity")
        region = require_mapping(parameters.get("region"), "deformable region")
        minimum_fraction = finite_number(parameters.get("minimum_fraction", 1.0), "minimum_fraction")
        if not 0.0 <= minimum_fraction <= 1.0:
            raise SchemaError("minimum_fraction must lie in [0, 1]")
        try:
            stream, rows, column = self._deformable_rows(predicate, entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"deformable/{entity}"])
        if rows.size == 0:
            return _result(
                predicate,
                "unknown",
                "No deformable keypoint sample exists.",
                missing=[f"deformable/{entity}/keypoints"],
            )
        fractions: list[float] = []
        passes: list[bool] = []
        counts: list[int] = []
        for row in rows:
            present = stream["keypoint_present"][row, column]
            points = stream["keypoints_m"][row, column][present]
            fraction = sum(self._inside_region(point, region) for point in points) / len(points)
            fractions.append(fraction)
            passes.append(fraction >= minimum_fraction)
            counts.append(int(len(points)))
        longest, selected = self._longest_boolean_duration(stream["time_s"], rows, passes)
        passed = longest >= duration if duration > 0.0 else passes[-1]
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Observed deformable keypoint fraction is {'inside' if passed else 'not inside'} the declared region threshold.",
            evidence=[
                {
                    "channel": "deformable",
                    "entity": entity,
                    "sample_indices": selected or [int(rows[-1])],
                    "terminal_fraction": fractions[-1],
                    "minimum_fraction": minimum_fraction,
                    "terminal_keypoint_count": counts[-1],
                    "longest_duration_s": longest,
                    "minimum_duration_s": duration,
                    "region": region,
                    "source_sha256": self.run.manifest["channels"]["deformable"]["sha256"],
                }
            ],
            limitations=[
                "The result covers supplied keypoints only; unobserved surface regions, topology, self-intersection, material state, and continuous deformation remain unestablished."
            ],
        )

    def _deformable_shape_within_tolerance(self, predicate: dict[str, Any]) -> PredicateResult:
        guard = self._guard(predicate, ["deformable"], continuous=False)
        if guard:
            return guard
        parameters = predicate["parameters"]
        entity = self._entity(parameters.get("entity"), "deformable entity")
        expected_raw = require_list(parameters.get("expected_keypoints_m"), "expected_keypoints_m")
        expected = np.asarray(expected_raw, dtype=np.float64)
        if expected.ndim != 2 or expected.shape[1] != 3 or not np.isfinite(expected).all():
            raise SchemaError("expected_keypoints_m must be a finite N x 3 array")
        try:
            stream, rows, column = self._deformable_rows(predicate, entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"deformable/{entity}"])
        if rows.size == 0:
            return _result(
                predicate,
                "unknown",
                "No deformable keypoint sample exists.",
                missing=[f"deformable/{entity}/keypoints"],
            )
        row = int(rows[-1])
        present = stream["keypoint_present"][row, column]
        actual = stream["keypoints_m"][row, column][present]
        if actual.shape != expected.shape:
            return _result(
                predicate,
                "unknown",
                "Observed and expected keypoint inventories differ.",
                missing=[f"expected {len(expected)} ordered keypoints, observed {len(actual)}"],
            )
        errors = np.linalg.norm(actual - expected, axis=1)
        rmse = float(math.sqrt(float(np.mean(errors * errors))))
        maximum = float(np.max(errors))
        rmse_tolerance = finite_number(parameters.get("rmse_tolerance_m"), "rmse_tolerance_m")
        maximum_tolerance = finite_number(
            parameters.get("maximum_point_error_m", rmse_tolerance), "maximum_point_error_m"
        )
        passed = rmse <= rmse_tolerance and maximum <= maximum_tolerance
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Observed ordered deformable keypoints are {'within' if passed else 'outside'} the declared shape tolerances.",
            evidence=[
                {
                    "channel": "deformable",
                    "entity": entity,
                    "sample_index": row,
                    "keypoint_count": len(actual),
                    "rmse_m": rmse,
                    "rmse_tolerance_m": rmse_tolerance,
                    "maximum_point_error_m": maximum,
                    "maximum_point_error_tolerance_m": maximum_tolerance,
                    "source_sha256": self.run.manifest["channels"]["deformable"]["sha256"],
                }
            ],
            limitations=[
                "Ordered keypoint agreement is not complete mesh, topology, material, strain-energy, or physical compliance validation."
            ],
        )

    def _dependency_results(self, predicate: dict[str, Any], names: Iterable[str]) -> tuple[list[PredicateResult], PredicateResult | None]:
        parameters = predicate["parameters"]
        dependencies: list[PredicateResult] = []
        missing: list[str] = []
        for name in names:
            reference = parameters.get(name)
            if reference is None:
                missing.append(name)
                continue
            if reference not in self.results:
                missing.append(f"predicate/{reference}")
                continue
            dependencies.append(self.results[reference])
        if missing:
            return dependencies, _result(
                predicate,
                "unknown",
                "Composite predicate dependencies are missing.",
                missing=missing,
            )
        return dependencies, None

    def _evidence_conjunction(self, predicate: dict[str, Any]) -> PredicateResult:
        references = require_list(predicate["parameters"].get("predicates"), "evidence_conjunction.predicates")
        if not references:
            raise SchemaError("evidence_conjunction.predicates must not be empty")
        missing = [str(reference) for reference in references if str(reference) not in self.results]
        if missing:
            return _result(
                predicate,
                "unknown",
                "Conjunction dependencies are missing.",
                missing=[f"predicate/{reference}" for reference in missing],
            )
        dependencies = [self.results[str(reference)] for reference in references]
        statuses = [dependency.status for dependency in dependencies]
        if "refuted" in statuses:
            status = "refuted"
        elif "conflicting" in statuses:
            status = "conflicting"
        elif "unknown" in statuses:
            status = "unknown"
        else:
            status = "supported"
        return _result(
            predicate,
            status,
            {
                "supported": "Every declared evidence dependency is supported.",
                "refuted": "At least one declared evidence dependency is refuted.",
                "unknown": "At least one declared evidence dependency is unknown.",
                "conflicting": "At least one declared evidence dependency conflicts.",
            }[status],
            evidence=[{
                "type": "predicate_composition",
                "operator": "all",
                "requirements": [
                    {
                        "predicate_id": dependency.predicate_id,
                        "status": dependency.status,
                        "evidence_sha256": dependency.evidence_sha256,
                    }
                    for dependency in dependencies
                ],
            }],
        )

    def _object_grasped(self, predicate: dict[str, Any]) -> PredicateResult:
        parameters = predicate["parameters"]
        contact_references = parameters.get("contact_predicates")
        if contact_references is None:
            contact_references = [parameters.get("contact_predicate")]
        contact_references = require_list(contact_references, "object_grasped.contact_predicates")
        references = contact_references + [
            parameters.get("gripper_predicate"),
            parameters.get("follows_predicate"),
            parameters.get("lift_predicate"),
        ]
        missing = [str(reference) if reference is not None else "unspecified dependency" for reference in references if reference is None or reference not in self.results]
        if missing:
            return _result(predicate, "unknown", "Composite predicate dependencies are missing.", missing=[f"predicate/{item}" for item in missing])
        dependencies = [self.results[str(reference)] for reference in references]
        statuses = [dependency.status for dependency in dependencies]
        if "conflicting" in statuses:
            status = "conflicting"
        elif "refuted" in statuses:
            status = "refuted"
        elif "unknown" in statuses:
            status = "unknown"
        else:
            status = "supported"
        return _result(
            predicate,
            status,
            {
                "supported": "Contact, gripper state, relative following, and lift evidence jointly support a grasp.",
                "refuted": "At least one required grasp condition was refuted; contact alone is not promoted to a grasp.",
                "unknown": "At least one required grasp condition lacks evidence.",
                "conflicting": "At least one required grasp condition has conflicting evidence.",
            }[status],
            evidence=[
                {
                    "type": "predicate_composition",
                    "requirements": [
                        {"predicate_id": dependency.predicate_id, "status": dependency.status, "evidence_sha256": dependency.evidence_sha256}
                        for dependency in dependencies
                    ],
                }
            ],
            limitations=[
                "A supported grasp is bounded to the declared simulator contact, joint, pose, lift, sampling, and tolerance evidence; it is not a real-world force-closure certificate."
            ],
        )

    def _object_released(self, predicate: dict[str, Any]) -> PredicateResult:
        dependencies, error = self._dependency_results(predicate, ("inside_predicate", "contact_predicate"))
        if error:
            return error
        inside, contact = dependencies
        gripper_reference = predicate["parameters"].get("gripper_predicate")
        gripper = self.results.get(gripper_reference) if gripper_reference else None
        if inside.status == "conflicting" or contact.status == "conflicting" or (gripper and gripper.status == "conflicting"):
            status = "conflicting"
        elif inside.status == "refuted" or contact.status == "supported" or (gripper and gripper.status == "refuted"):
            status = "refuted"
        elif inside.status == "unknown" or contact.status == "unknown" or (gripper and gripper.status == "unknown"):
            status = "unknown"
        else:
            status = "supported"
        evidence_dependencies = [inside, contact] + ([gripper] if gripper else [])
        return _result(
            predicate,
            status,
            {
                "supported": "The object is inside the release region and sustained tool contact is absent under the declared gripper policy.",
                "refuted": "The release-region, contact, or gripper evidence refutes a completed release.",
                "unknown": "Release cannot be established because required evidence is missing.",
                "conflicting": "Release cannot be established because required evidence conflicts.",
            }[status],
            evidence=[
                {
                    "type": "predicate_composition",
                    "requirements": [
                        {"predicate_id": dependency.predicate_id, "status": dependency.status, "evidence_sha256": dependency.evidence_sha256}
                        for dependency in evidence_dependencies
                    ],
                }
            ],
        )
