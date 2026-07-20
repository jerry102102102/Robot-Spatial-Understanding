"""Declarative task contract with no simulator-specific success code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SchemaError
from .util import load_structured, require_list, require_mapping, require_string, sha256_json


TASK_SCHEMA = "robot-spatial-task-spec.v1"
PREDICATE_TYPES = frozenset(
    {
        "joint_within_tolerance",
        "frame_within_pose_tolerance",
        "base_reached_goal",
        "collision_free_over_interval",
        "path_stayed_within_corridor",
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
)
VALID_STATUSES = frozenset({"supported", "refuted", "unknown", "conflicting"})


def _validate_expression(expression: Any, predicate_ids: set[str], label: str) -> None:
    node = require_mapping(expression, label)
    operators = [operator for operator in ("all", "any", "not", "predicate") if operator in node]
    if len(operators) != 1:
        raise SchemaError(f"{label} must contain exactly one of all, any, not, or predicate")
    operator = operators[0]
    if operator == "predicate":
        predicate_id = require_string(node[operator], f"{label}.predicate")
        if predicate_id not in predicate_ids:
            raise SchemaError(f"{label} references unknown predicate {predicate_id!r}")
        return
    if operator == "not":
        _validate_expression(node[operator], predicate_ids, f"{label}.not")
        return
    children = require_list(node[operator], f"{label}.{operator}")
    if not children:
        raise SchemaError(f"{label}.{operator} must not be empty")
    for index, child in enumerate(children):
        if isinstance(child, str):
            if child not in predicate_ids:
                raise SchemaError(f"{label}.{operator}[{index}] references unknown predicate {child!r}")
        else:
            _validate_expression(child, predicate_ids, f"{label}.{operator}[{index}]")


@dataclass(frozen=True)
class TaskSpec:
    """Validated versioned task declaration."""

    path: Path
    data: dict[str, Any]
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        task_path = Path(path)
        data = require_mapping(load_structured(task_path), "task spec")
        if data.get("schema_version") != TASK_SCHEMA:
            raise SchemaError(f"task schema must be {TASK_SCHEMA!r}")
        require_string(data.get("task_id"), "task.task_id")
        entities = require_mapping(data.get("entities"), "task.entities")
        if not entities:
            raise SchemaError("task.entities must not be empty")
        for role, entity in entities.items():
            require_string(str(role), "task entity role")
            require_string(entity, f"task.entities.{role}")
        requirements = require_mapping(data.get("requirements", {}), "task.requirements")
        required_channels = require_list(requirements.get("channels", []), "task.requirements.channels")
        for index, channel in enumerate(required_channels):
            require_string(channel, f"task.requirements.channels[{index}]")
        predicates = require_list(data.get("predicates"), "task.predicates")
        if not predicates:
            raise SchemaError("task.predicates must not be empty")
        predicate_ids: set[str] = set()
        for index, raw in enumerate(predicates):
            predicate = require_mapping(raw, f"task.predicates[{index}]")
            predicate_id = require_string(predicate.get("predicate_id"), f"task.predicates[{index}].predicate_id")
            if predicate_id in predicate_ids:
                raise SchemaError(f"duplicate predicate_id {predicate_id!r}")
            predicate_ids.add(predicate_id)
            predicate_type = require_string(predicate.get("type"), f"task.predicates[{index}].type")
            if predicate_type not in PREDICATE_TYPES:
                raise SchemaError(
                    f"unsupported predicate type {predicate_type!r}; expected one of {sorted(PREDICATE_TYPES)}"
                )
            require_mapping(predicate.get("parameters", {}), f"task.predicates[{index}].parameters")
            window = require_mapping(predicate.get("window", {}), f"task.predicates[{index}].window")
            for boundary in ("start_s", "end_s"):
                if boundary in window and not isinstance(window[boundary], (int, float)):
                    raise SchemaError(f"task.predicates[{index}].window.{boundary} must be numeric")
        _validate_expression(data.get("goal"), predicate_ids, "task.goal")
        if "failure" in data:
            _validate_expression(data["failure"], predicate_ids, "task.failure")
        termination = require_mapping(data.get("termination", {}), "task.termination")
        mode = termination.get("mode", "episode_end")
        if mode not in {"episode_end", "first_goal", "declared_event"}:
            raise SchemaError("task.termination.mode must be episode_end, first_goal, or declared_event")
        normalized = dict(data)
        normalized.setdefault("requirements", {"channels": []})
        normalized.setdefault("termination", {"mode": "episode_end"})
        normalized.setdefault(
            "claim_boundaries",
            {
                "simulation_only": True,
                "causation_requires_counterfactual": True,
                "authorization_established": False,
                "safety_established": False,
            },
        )
        digest = sha256_json(normalized)
        return cls(task_path, normalized, digest)

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def predicates(self) -> list[dict[str, Any]]:
        return list(self.data["predicates"])

    @property
    def required_channels(self) -> list[str]:
        return list(self.data.get("requirements", {}).get("channels", []))
