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
        for predicate in self.task.predicates:
            if predicate["type"] in {"object_grasped", "object_released_in_region"}:
                deferred.append(predicate)
            else:
                self.results[predicate["predicate_id"]] = self._evaluate_one(predicate)
        for predicate in deferred:
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
            "frame_within_pose_tolerance": self._frame_within_pose_tolerance,
            "base_reached_goal": self._frame_within_pose_tolerance,
            "collision_free_over_interval": self._collision_free,
            "path_stayed_within_corridor": self._path_stayed_within_corridor,
            "contact_sustained": self._contact_sustained,
            "object_above_height": self._object_above_height,
            "object_follows_frame_for_duration": self._object_follows_frame,
            "object_inside_region": self._object_inside_region,
            "object_grasped": self._object_grasped,
            "object_released_in_region": self._object_released,
            "inserted_to_depth": self._inserted_to_depth,
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
        threshold = finite_number(parameters.get("minimum_m"), "object_above_height.minimum_m")
        duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        try:
            times, rows, column = self._pose_rows(predicate, "pose", entity)
        except EvidenceError as error:
            return _result(predicate, "unknown", str(error), missing=[f"pose/{entity}"])
        if rows.size == 0:
            return _result(predicate, "unknown", "No object pose exists in the evaluation window.", missing=[f"pose/{entity}"])
        heights = self.run.stream("pose")["position_m"][rows, column, axis]
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
        if expected is None:
            baseline_position = relative_positions[0]
            baseline_quaternion = relative_quaternions[0]
            expectation_source = "first synchronized sample"
        else:
            expected_pose = require_mapping(expected, "expected_relative_pose")
            baseline_position = [float(value) for value in require_list(expected_pose.get("position_m"), "expected_relative_pose.position_m")]
            baseline_quaternion = [float(value) for value in require_list(expected_pose.get("quaternion_xyzw"), "expected_relative_pose.quaternion_xyzw")]
            expectation_source = "task spec"
        position_errors = [euclidean(value, baseline_position) for value in relative_positions]
        orientation_errors = [quaternion_angle(value, baseline_quaternion) for value in relative_quaternions]
        position_tolerance = finite_number(parameters.get("position_tolerance_m", 0.01), "position_tolerance_m")
        orientation_tolerance = finite_number(parameters.get("orientation_tolerance_rad", math.radians(5.0)), "orientation_tolerance_rad")
        minimum_duration = finite_number(parameters.get("minimum_duration_s", 0.0), "minimum_duration_s")
        duration = float(stream["time_s"][rows[-1]] - stream["time_s"][rows[0]])
        passed = (
            duration >= minimum_duration
            and max(position_errors) <= position_tolerance
            and max(orientation_errors) <= orientation_tolerance
        )
        return _result(
            predicate,
            "supported" if passed else "refuted",
            f"Target {'maintained' if passed else 'did not maintain'} the declared relative pose to the reference.",
            evidence=[
                {
                    "channel": "pose",
                    "reference": reference,
                    "target": target,
                    "sample_indices": [int(row) for row in rows],
                    "duration_s": duration,
                    "minimum_duration_s": minimum_duration,
                    "maximum_position_drift_m": max(position_errors),
                    "position_tolerance_m": position_tolerance,
                    "maximum_orientation_drift_rad": max(orientation_errors),
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

    def _object_grasped(self, predicate: dict[str, Any]) -> PredicateResult:
        dependencies, error = self._dependency_results(
            predicate,
            ("contact_predicate", "gripper_predicate", "follows_predicate", "lift_predicate"),
        )
        if error:
            return error
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
