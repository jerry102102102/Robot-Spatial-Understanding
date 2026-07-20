"""Conservative deformable-state predicates shared by optional simulator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import EvidenceError, SchemaError
from .simulation import SimulationRun


DEFORMABLE_STATE_SCHEMA = "robot-spatial-deformable-state.v1"


@dataclass(frozen=True)
class DeformableStateSummary:
    entity: str
    sample_index: int
    keypoint_count: int
    aabb_min_m: list[float]
    aabb_max_m: list[float]
    centroid_m: list[float]

    @classmethod
    def from_run(cls, run: SimulationRun, entity: str, sample_index: int = -1) -> "DeformableStateSummary":
        if not run.channel_available("deformable"):
            raise EvidenceError("deformable channel is unavailable")
        stream = run.stream("deformable")
        entities = [str(value) for value in stream["entity_ids"]]
        if entity not in entities:
            raise EvidenceError(f"deformable entity {entity!r} is absent")
        column = entities.index(entity)
        row = sample_index if sample_index >= 0 else len(stream["time_s"]) - 1
        if row < 0 or row >= len(stream["time_s"]):
            raise SchemaError("deformable sample index is out of range")
        present = stream["keypoint_present"][row, column]
        points = stream["keypoints_m"][row, column][present]
        if points.size == 0:
            raise EvidenceError("deformable sample has no observed keypoints")
        return cls(
            entity=entity,
            sample_index=row,
            keypoint_count=int(points.shape[0]),
            aabb_min_m=[float(value) for value in np.min(points, axis=0)],
            aabb_max_m=[float(value) for value in np.max(points, axis=0)],
            centroid_m=[float(value) for value in np.mean(points, axis=0)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEFORMABLE_STATE_SCHEMA,
            "entity": self.entity,
            "sample_index": self.sample_index,
            "keypoint_count": self.keypoint_count,
            "aabb_min_m": self.aabb_min_m,
            "aabb_max_m": self.aabb_max_m,
            "centroid_m": self.centroid_m,
            "limitations": [
                "This summary covers supplied keypoints only; it does not prove topology, material parameters, continuous surface coverage, or unobserved deformation."
            ],
        }
