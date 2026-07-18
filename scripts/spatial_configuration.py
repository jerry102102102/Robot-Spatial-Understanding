#!/usr/bin/env python3
"""Finite configuration-space witnesses for digest-bound constrained mechanisms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from spatial_constraints import (
    GRAPH_SCHEMA,
    ConstraintError,
    evaluate_constraint_graph,
    read_constraint_graph,
    solve_constraint_graph,
)


SPEC_SCHEMA = "robot-spatial-configuration-atlas-spec.v1"
ATLAS_SCHEMA = "robot-spatial-configuration-atlas.v1"
VERIFICATION_SCHEMA = "robot-spatial-configuration-atlas-verification.v1"
EPSILON = 1e-12


class ConfigurationError(ValueError):
    """A malformed configuration-atlas contract or failed atlas operation."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} root must be an object")
    return value


def _exact_fields(record: Any, required: set[str], optional: set[str], label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ConfigurationError(f"{label} must be an object")
    missing = required - set(record)
    unknown = set(record) - required - optional
    if missing or unknown:
        raise ConfigurationError(
            f"{label} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return record


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or ".." in value.split("/"):
        raise ConfigurationError(f"{label} must be a non-empty relative typed identifier")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ConfigurationError(f"{label} must be a finite number")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ConfigurationError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _canonical_number(value: float) -> float:
    return 0.0 if abs(value) < 5e-13 else round(float(value), 12)


def _driver_domain(grammar: dict[str, Any], name: str) -> tuple[float | None, float | None]:
    domain = grammar["independent_variables"][name]["feasible_domain"]
    return domain["minimum"], domain["maximum"]


def _inside_domain(grammar: dict[str, Any], name: str, value: float) -> bool:
    lower, upper = _driver_domain(grammar, name)
    return (lower is None or value >= float(lower) - EPSILON) and (
        upper is None or value <= float(upper) + EPSILON
    )


def _periodic_drivers(graph: dict[str, Any]) -> set[str]:
    grammar = graph["articulation_grammar"]
    coordinate_joints = {
        term["joint"]
        for constraint in graph.get("constraints", [])
        if constraint["type"] == "coordinate_linear"
        for term in constraint["terms"]
    }
    result: set[str] = set()
    for driver, variable in grammar["independent_variables"].items():
        if variable["joint_type"] != "continuous":
            continue
        if coordinate_joints.isdisjoint(variable["physical_joints_driven"]):
            result.add(driver)
    return result


def _wrapped_delta(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _configuration_distance(
    left: dict[str, float],
    right: dict[str, float],
    order: list[str],
    scales: dict[str, float],
    periodic: set[str],
) -> float:
    squared = 0.0
    for name in order:
        delta = (
            _wrapped_delta(left[name], right[name])
            if name in periodic
            else left[name] - right[name]
        )
        squared += (delta / scales[name]) ** 2
    return math.sqrt(squared)


def _jacobi_eigenvalues_symmetric(matrix: list[list[float]]) -> list[float]:
    size = len(matrix)
    if size == 0:
        return []
    work = [row[:] for row in matrix]
    for _ in range(max(32, 64 * size * size)):
        p, q = 0, 0
        maximum = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                value = abs(work[row][column])
                if value > maximum:
                    maximum = value
                    p, q = row, column
        if maximum <= 1e-14:
            break
        app, aqq, apq = work[p][p], work[q][q], work[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine, sine = math.cos(angle), math.sin(angle)
        for index in range(size):
            if index in {p, q}:
                continue
            aip, aiq = work[index][p], work[index][q]
            work[index][p] = work[p][index] = cosine * aip - sine * aiq
            work[index][q] = work[q][index] = sine * aip + cosine * aiq
        work[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        work[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        work[p][q] = work[q][p] = 0.0
    return sorted((max(0.0, work[index][index]) for index in range(size)), reverse=True)


def _singular_diagnostics(
    matrix: list[list[float]],
    column_count: int,
    relative_tolerance: float,
) -> dict[str, Any]:
    gram = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
    for row in matrix:
        for left in range(column_count):
            for right in range(column_count):
                gram[left][right] += row[left] * row[right]
    singular_values = [math.sqrt(value) for value in _jacobi_eigenvalues_symmetric(gram)]
    maximum = max(singular_values, default=0.0)
    threshold = relative_tolerance * max(1.0, maximum)
    rank = sum(value > threshold for value in singular_values)
    positive = [value for value in singular_values if value > threshold]
    condition = None
    if singular_values:
        minimum = singular_values[-1]
        condition = None if minimum <= threshold else maximum / minimum
    return {
        "singular_values_descending": [_canonical_number(value) for value in singular_values],
        "relative_tolerance": relative_tolerance,
        "absolute_threshold": _canonical_number(threshold),
        "numerical_rank": rank,
        "nullity": column_count - rank,
        "largest_singular_value": None if not singular_values else _canonical_number(maximum),
        "smallest_singular_value": None if not singular_values else _canonical_number(singular_values[-1]),
        "smallest_resolved_singular_value": None if not positive else _canonical_number(min(positive)),
        "condition_number": None if condition is None else _canonical_number(condition),
        "condition_number_infinite_or_unresolved": condition is None and bool(singular_values),
    }


def _normalize_spec(
    graph: dict[str, Any],
    graph_artifact_sha256: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    _exact_fields(
        spec,
        {
            "schema_version",
            "atlas_id",
            "constraint_graph_sha256",
            "singular_value_relative_tolerance",
            "charts",
        },
        set(),
        "configuration atlas spec",
    )
    if spec["schema_version"] != SPEC_SCHEMA:
        raise ConfigurationError(f"configuration atlas spec must use schema_version {SPEC_SCHEMA}")
    if spec["constraint_graph_sha256"] != graph_artifact_sha256:
        raise ConfigurationError(
            "configuration atlas spec constraint_graph_sha256 does not bind the exact graph artifact"
        )
    atlas_id = _identifier(spec["atlas_id"], "configuration atlas spec atlas_id")
    relative_tolerance = _positive(
        spec["singular_value_relative_tolerance"],
        "configuration atlas spec singular_value_relative_tolerance",
    )
    grammar = graph["articulation_grammar"]
    driver_names = sorted(grammar["independent_variables"])
    charts = spec["charts"]
    if not isinstance(charts, list) or not charts:
        raise ConfigurationError("configuration atlas spec charts must be a non-empty array")
    normalized_charts: list[dict[str, Any]] = []
    chart_ids: set[str] = set()
    for chart_index, chart in enumerate(charts):
        label = f"configuration atlas spec charts[{chart_index}]"
        _exact_fields(
            chart,
            {
                "chart_id",
                "parameter_driver",
                "parameter_values",
                "solve_for",
                "driver_scales",
                "seeds",
                "solution_merge_tolerance_normalized",
                "continuation_edge_max_distance_normalized",
                "minimum_solutions_per_sample",
            },
            set(),
            label,
        )
        chart_id = _identifier(chart["chart_id"], f"{label}.chart_id")
        if chart_id in chart_ids:
            raise ConfigurationError(f"configuration atlas spec has duplicate chart_id {chart_id!r}")
        chart_ids.add(chart_id)
        parameter = chart["parameter_driver"]
        if parameter not in driver_names:
            raise ConfigurationError(f"{label}.parameter_driver is not an independent driver")
        values = chart["parameter_values"]
        if not isinstance(values, list) or len(values) < 2:
            raise ConfigurationError(f"{label}.parameter_values must contain at least two values")
        normalized_values = [_finite(value, f"{label}.parameter_values[{index}]") for index, value in enumerate(values)]
        if any(right <= left for left, right in zip(normalized_values, normalized_values[1:])):
            raise ConfigurationError(f"{label}.parameter_values must be strictly increasing")
        for value in normalized_values:
            if not _inside_domain(grammar, parameter, value):
                raise ConfigurationError(
                    f"{label}.parameter value {value} is outside the feasible domain of {parameter!r}"
                )
        solve_for = chart["solve_for"]
        if (
            not isinstance(solve_for, list)
            or not solve_for
            or any(not isinstance(name, str) for name in solve_for)
            or len(set(solve_for)) != len(solve_for)
        ):
            raise ConfigurationError(f"{label}.solve_for must be a unique non-empty driver array")
        if parameter in solve_for or set(solve_for) != set(driver_names) - {parameter}:
            raise ConfigurationError(
                f"{label}.solve_for must contain every independent driver except parameter_driver"
            )
        scales = chart["driver_scales"]
        if not isinstance(scales, dict) or set(scales) != set(driver_names):
            raise ConfigurationError(f"{label}.driver_scales must name every independent driver exactly")
        normalized_scales = {
            name: _positive(scales[name], f"{label}.driver_scales.{name}")
            for name in driver_names
        }
        seeds = chart["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ConfigurationError(f"{label}.seeds must be a non-empty array")
        normalized_seeds: list[dict[str, Any]] = []
        seed_ids: set[str] = set()
        for seed_index, seed in enumerate(seeds):
            seed_label = f"{label}.seeds[{seed_index}]"
            _exact_fields(seed, {"seed_id", "joints"}, set(), seed_label)
            seed_id = _identifier(seed["seed_id"], f"{seed_label}.seed_id")
            if seed_id in seed_ids:
                raise ConfigurationError(f"{label} has duplicate seed_id {seed_id!r}")
            seed_ids.add(seed_id)
            joints = seed["joints"]
            if not isinstance(joints, dict) or set(joints) != set(driver_names):
                raise ConfigurationError(f"{seed_label}.joints must bind every independent driver exactly")
            normalized_joints = {
                name: _finite(joints[name], f"{seed_label}.joints.{name}")
                for name in driver_names
            }
            for name, value in normalized_joints.items():
                if not _inside_domain(grammar, name, value):
                    raise ConfigurationError(
                        f"{seed_label}.joints.{name} is outside its feasible domain"
                    )
            normalized_seeds.append({"seed_id": seed_id, "joints": normalized_joints})
        normalized_charts.append({
            "chart_id": chart_id,
            "parameter_driver": parameter,
            "parameter_values": normalized_values,
            "solve_for": solve_for,
            "driver_scales": normalized_scales,
            "seeds": normalized_seeds,
            "solution_merge_tolerance_normalized": _positive(
                chart["solution_merge_tolerance_normalized"],
                f"{label}.solution_merge_tolerance_normalized",
            ),
            "continuation_edge_max_distance_normalized": _positive(
                chart["continuation_edge_max_distance_normalized"],
                f"{label}.continuation_edge_max_distance_normalized",
            ),
            "minimum_solutions_per_sample": _positive_integer(
                chart["minimum_solutions_per_sample"],
                f"{label}.minimum_solutions_per_sample",
            ),
        })
    return {
        "schema_version": SPEC_SCHEMA,
        "atlas_id": atlas_id,
        "constraint_graph_sha256": graph_artifact_sha256,
        "singular_value_relative_tolerance": relative_tolerance,
        "charts": normalized_charts,
    }


def _deduplicate_seed_candidates(
    candidates: list[dict[str, Any]],
    driver_order: list[str],
    scales: dict[str, float],
    periodic: set[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            tuple(candidate["joints"][name] for name in driver_order),
            candidate["seed_refs"],
        ),
    )
    unique: list[dict[str, Any]] = []
    for candidate in ordered:
        match = next(
            (
                record
                for record in unique
                if _configuration_distance(
                    record["joints"], candidate["joints"], driver_order, scales, periodic
                ) <= tolerance
            ),
            None,
        )
        if match is None:
            unique.append({"joints": candidate["joints"], "seed_refs": list(candidate["seed_refs"])})
        else:
            match["seed_refs"] = sorted(set(match["seed_refs"]) | set(candidate["seed_refs"]))
    return unique


def _node_analysis(
    evaluation: dict[str, Any],
    solve_for: list[str],
    relative_tolerance: float,
) -> dict[str, Any]:
    local = evaluation["local_constraint_analysis"]
    order = local["independent_variable_order"]
    full = local["normalized_jacobian_rowmajor"]
    indices = [order.index(name) for name in solve_for]
    passive = [[row[index] for index in indices] for row in full]
    return {
        "full_constraint_jacobian": {
            "column_order": order,
            **_singular_diagnostics(full, len(order), relative_tolerance),
        },
        "chart_passive_jacobian": {
            "column_order": solve_for,
            **_singular_diagnostics(passive, len(solve_for), relative_tolerance),
        },
        "local_constraint_analysis": local,
    }


def _connected_components(node_ids: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        left, right = edge["from_node"], edge["to_node"]
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    unseen = set(node_ids)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[str] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def build_configuration_atlas(
    graph: dict[str, Any],
    graph_artifact_sha256: str,
    spec: dict[str, Any],
    spec_sha256: str,
) -> dict[str, Any]:
    if graph.get("schema_version") != GRAPH_SCHEMA:
        raise ConfigurationError(f"constraint graph must use schema_version {GRAPH_SCHEMA}")
    normalized_spec = _normalize_spec(graph, graph_artifact_sha256, spec)
    grammar = graph["articulation_grammar"]
    driver_order = sorted(grammar["independent_variables"])
    periodic = _periodic_drivers(graph)
    chart_records: list[dict[str, Any]] = []
    total_attempts = 0
    total_converged = 0
    total_nodes = 0
    total_singular = 0
    complete = True

    for chart in normalized_spec["charts"]:
        samples: list[dict[str, Any]] = []
        previous_nodes: list[dict[str, Any]] = []
        chart_attempts = 0
        chart_converged = 0
        for sample_index, parameter_value in enumerate(chart["parameter_values"]):
            candidates = []
            for seed in chart["seeds"]:
                joints = dict(seed["joints"])
                joints[chart["parameter_driver"]] = parameter_value
                candidates.append({
                    "joints": joints,
                    "seed_refs": [f"explicit_seed/{seed['seed_id']}"],
                })
            for node in previous_nodes:
                joints = dict(node["independent_driver_positions"])
                joints[chart["parameter_driver"]] = parameter_value
                candidates.append({
                    "joints": joints,
                    "seed_refs": [f"continuation_node/{node['node_id']}"],
                })
            candidates = _deduplicate_seed_candidates(
                candidates,
                driver_order,
                chart["driver_scales"],
                periodic,
                chart["solution_merge_tolerance_normalized"],
            )
            attempts: list[dict[str, Any]] = []
            converged_candidates: list[dict[str, Any]] = []
            for attempt_index, candidate in enumerate(candidates):
                solution = solve_constraint_graph(
                    graph,
                    candidate["joints"],
                    chart["solve_for"],
                    f"{normalized_spec['atlas_id']}/{chart['chart_id']}/sample-{sample_index:04d}",
                )
                chart_attempts += 1
                attempt_id = f"attempt/{chart['chart_id']}/{sample_index:04d}/{attempt_index:04d}"
                attempt = {
                    "attempt_id": attempt_id,
                    "seed_refs": candidate["seed_refs"],
                    "seed_independent_driver_positions": solution["seed_independent_driver_positions"],
                    "status": solution["status"],
                    "termination": solution["termination"],
                    "iteration_count": solution["iteration_count"],
                    "maximum_normalized_abs": solution["evaluation"]["maximum_normalized_abs"],
                }
                attempts.append(attempt)
                if solution["status"] != "converged":
                    continue
                chart_converged += 1
                positions = {
                    name: float(value)
                    for name, value in solution["solved_independent_driver_positions"].items()
                }
                evaluation = evaluate_constraint_graph(
                    graph,
                    positions,
                    f"{normalized_spec['atlas_id']}/{chart['chart_id']}/sample-{sample_index:04d}",
                    True,
                )
                converged_candidates.append({
                    "attempt_id": attempt_id,
                    "seed_refs": candidate["seed_refs"],
                    "positions": positions,
                    "evaluation": evaluation,
                })
            ordered_candidates = sorted(
                converged_candidates,
                key=lambda candidate: tuple(candidate["positions"][name] for name in driver_order),
            )
            clusters: list[dict[str, Any]] = []
            for candidate in ordered_candidates:
                match = next(
                    (
                        cluster
                        for cluster in clusters
                        if _configuration_distance(
                            cluster["positions"],
                            candidate["positions"],
                            driver_order,
                            chart["driver_scales"],
                            periodic,
                        ) <= chart["solution_merge_tolerance_normalized"]
                    ),
                    None,
                )
                if match is None:
                    clusters.append({
                        "positions": candidate["positions"],
                        "evaluation": candidate["evaluation"],
                        "attempt_ids": [candidate["attempt_id"]],
                        "seed_refs": list(candidate["seed_refs"]),
                    })
                else:
                    match["attempt_ids"].append(candidate["attempt_id"])
                    match["seed_refs"] = sorted(set(match["seed_refs"]) | set(candidate["seed_refs"]))
            nodes: list[dict[str, Any]] = []
            for solution_index, cluster in enumerate(clusters):
                node_id = f"configuration_node/{chart['chart_id']}/{sample_index:04d}/{solution_index:04d}"
                analysis = _node_analysis(
                    cluster["evaluation"],
                    chart["solve_for"],
                    normalized_spec["singular_value_relative_tolerance"],
                )
                nodes.append({
                    "node_id": node_id,
                    "sample_index": sample_index,
                    "parameter_driver": chart["parameter_driver"],
                    "parameter_value": _canonical_number(parameter_value),
                    "independent_driver_positions": {
                        name: _canonical_number(cluster["positions"][name])
                        for name in driver_order
                    },
                    "constraint_status": cluster["evaluation"]["status"],
                    "maximum_normalized_abs": cluster["evaluation"]["maximum_normalized_abs"],
                    "supporting_attempt_ids": sorted(cluster["attempt_ids"]),
                    "source_seed_refs": sorted(cluster["seed_refs"]),
                    **analysis,
                })
            if len(nodes) < chart["minimum_solutions_per_sample"]:
                complete = False
            samples.append({
                "sample_index": sample_index,
                "parameter_driver": chart["parameter_driver"],
                "parameter_value": _canonical_number(parameter_value),
                "attempt_count": len(attempts),
                "converged_attempt_count": sum(attempt["status"] == "converged" for attempt in attempts),
                "unique_solution_count": len(nodes),
                "minimum_solutions_required": chart["minimum_solutions_per_sample"],
                "coverage_status": (
                    "met" if len(nodes) >= chart["minimum_solutions_per_sample"] else "below_required_minimum"
                ),
                "attempts": attempts,
                "solutions": nodes,
            })
            previous_nodes = nodes

        all_nodes = [node for sample in samples for node in sample["solutions"]]
        max_full_rank = max(
            (node["full_constraint_jacobian"]["numerical_rank"] for node in all_nodes),
            default=0,
        )
        max_passive_rank = max(
            (node["chart_passive_jacobian"]["numerical_rank"] for node in all_nodes),
            default=0,
        )
        for node in all_nodes:
            node["singularity_witness"] = {
                "maximum_full_rank_observed_in_chart": max_full_rank,
                "maximum_passive_rank_observed_in_chart": max_passive_rank,
                "mechanism_rank_drop_candidate": (
                    node["full_constraint_jacobian"]["numerical_rank"] < max_full_rank
                ),
                "chart_parameterization_rank_drop_candidate": (
                    node["chart_passive_jacobian"]["numerical_rank"] < max_passive_rank
                ),
                "meaning": (
                    "finite numerical witness relative to ranks observed in this declared chart; "
                    "not a certified singularity classification"
                ),
            }
        edges: list[dict[str, Any]] = []
        for left_sample, right_sample in zip(samples, samples[1:]):
            for left in left_sample["solutions"]:
                for right in right_sample["solutions"]:
                    distance = _configuration_distance(
                        left["independent_driver_positions"],
                        right["independent_driver_positions"],
                        driver_order,
                        chart["driver_scales"],
                        periodic,
                    )
                    if distance <= chart["continuation_edge_max_distance_normalized"]:
                        edges.append({
                            "edge_type": "adjacent_sample_configuration_proximity_witness",
                            "from_node": left["node_id"],
                            "to_node": right["node_id"],
                            "normalized_configuration_distance": _canonical_number(distance),
                        })
        for sample in samples:
            singular_nodes = [
                node
                for node in sample["solutions"]
                if node["singularity_witness"]["mechanism_rank_drop_candidate"]
                or node["singularity_witness"]["chart_parameterization_rank_drop_candidate"]
            ]
            for left_index, left in enumerate(singular_nodes):
                for right in singular_nodes[left_index + 1 :]:
                    distance = _configuration_distance(
                        left["independent_driver_positions"],
                        right["independent_driver_positions"],
                        driver_order,
                        chart["driver_scales"],
                        periodic,
                    )
                    if distance <= chart["continuation_edge_max_distance_normalized"]:
                        edges.append({
                            "edge_type": "same_sample_singularity_slice_proximity_witness",
                            "from_node": left["node_id"],
                            "to_node": right["node_id"],
                            "normalized_configuration_distance": _canonical_number(distance),
                        })
        components = _connected_components(
            [node["node_id"] for node in all_nodes],
            edges,
        )
        singular_count = sum(
            node["singularity_witness"]["mechanism_rank_drop_candidate"]
            or node["singularity_witness"]["chart_parameterization_rank_drop_candidate"]
            for node in all_nodes
        )
        chart_record = {
            **{key: value for key, value in chart.items() if key != "seeds"},
            "seed_contracts": chart["seeds"],
            "periodic_driver_metric": sorted(periodic),
            "samples": samples,
            "configuration_proximity_edges": sorted(
                edges,
                key=lambda edge: (edge["from_node"], edge["to_node"], edge["edge_type"]),
            ),
            "witness_components": [
                {
                    "component_id": f"configuration_component/{chart['chart_id']}/{index:04d}",
                    "node_ids": component,
                }
                for index, component in enumerate(components)
            ],
            "observed_rank_reference": {
                "maximum_full_constraint_rank": max_full_rank,
                "maximum_chart_passive_rank": max_passive_rank,
            },
            "coverage": {
                "parameter_sample_count": len(samples),
                "attempt_count": chart_attempts,
                "converged_attempt_count": chart_converged,
                "unique_solution_node_count": len(all_nodes),
                "samples_meeting_minimum": sum(sample["coverage_status"] == "met" for sample in samples),
                "samples_below_minimum": sum(sample["coverage_status"] != "met" for sample in samples),
                "singularity_candidate_node_count": singular_count,
                "proximity_edge_count": len(edges),
                "witness_component_count": len(components),
            },
        }
        chart_records.append(chart_record)
        total_attempts += chart_attempts
        total_converged += chart_converged
        total_nodes += len(all_nodes)
        total_singular += singular_count

    core = {
        "schema_version": ATLAS_SCHEMA,
        "atlas_id": normalized_spec["atlas_id"],
        "status": "complete_for_declared_sampling" if complete else "partial_for_declared_sampling",
        "source_binding": {
            "constraint_graph_artifact_sha256": graph_artifact_sha256,
            "constraint_graph_id": graph.get("constraint_graph_id"),
            "constraint_graph_sha256": graph.get("constraint_graph_sha256"),
            "configuration_atlas_spec_sha256": spec_sha256,
        },
        "constraint_graph": graph,
        "exploration_contract": normalized_spec,
        "charts": chart_records,
        "coverage": {
            "chart_count": len(chart_records),
            "parameter_sample_count": sum(len(chart["samples"]) for chart in chart_records),
            "solve_attempt_count": total_attempts,
            "converged_attempt_count": total_converged,
            "unique_solution_node_count": total_nodes,
            "singularity_candidate_node_count": total_singular,
            "all_declared_sample_minima_met": complete,
        },
        "epistemic_scope": (
            "finite multi-seed local-solve witnesses over explicit one-parameter charts; solution nodes are exact constraint evaluations "
            "within declared tolerances, while proximity edges, components, rank-drop labels, branch coverage, and singularity interpretation "
            "are finite numerical evidence—not exhaustive global configuration-space topology, certified branch enumeration, a global solver, "
            "dynamics/contact/compliance, calibration, hardware truth, or safety"
        ),
    }
    digest = hashlib.sha256(_json_bytes(core)).hexdigest()
    return {
        **core,
        "configuration_atlas_id": f"configuration-atlas-{digest[:20]}",
        "configuration_atlas_sha256": digest,
    }


def write_configuration_atlas(path: Path, graph_path: Path, spec_path: Path) -> dict[str, Any]:
    graph = read_constraint_graph(graph_path)
    spec = _read_json(spec_path, "configuration atlas spec")
    atlas = build_configuration_atlas(graph, _sha256(graph_path), spec, _sha256(spec_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(atlas))
    return atlas


def read_configuration_atlas(path: Path) -> dict[str, Any]:
    atlas = _read_json(path, "configuration atlas")
    if atlas.get("schema_version") != ATLAS_SCHEMA:
        raise ConfigurationError(f"configuration atlas must use schema_version {ATLAS_SCHEMA}")
    return atlas


def verify_configuration_atlas(
    graph_path: Path,
    spec_path: Path,
    atlas_path: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    try:
        actual = read_configuration_atlas(atlas_path)
        expected = build_configuration_atlas(
            read_constraint_graph(graph_path),
            _sha256(graph_path),
            _read_json(spec_path, "configuration atlas spec"),
            _sha256(spec_path),
        )
        exact = _json_bytes(actual) == _json_bytes(expected)
        if not exact:
            issues.append({
                "check": "exact_regeneration",
                "message": "configuration atlas differs from exact regeneration",
            })
        node_count = 0
        violated_nodes: list[str] = []
        graph = actual.get("constraint_graph")
        if not isinstance(graph, dict):
            raise ConfigurationError("configuration atlas has no embedded constraint graph")
        for chart in actual.get("charts", []):
            for sample in chart.get("samples", []):
                for node in sample.get("solutions", []):
                    node_count += 1
                    evaluation = evaluate_constraint_graph(
                        graph,
                        node["independent_driver_positions"],
                        f"verification/{node['node_id']}",
                        True,
                    )
                    if evaluation["status"] != "satisfied":
                        violated_nodes.append(node["node_id"])
        if violated_nodes:
            issues.append({
                "check": "node_execution",
                "message": "stored configuration nodes violate embedded constraints",
                "node_ids": violated_nodes,
            })
    except (ConfigurationError, ConstraintError, KeyError, TypeError, IndexError) as error:
        exact = False
        node_count = 0
        issues.append({"check": "read_regenerate_execute", "message": str(error)})
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not issues else "failed",
        "atlas_path": str(atlas_path.resolve()),
        "atlas_artifact_sha256": _sha256(atlas_path),
        "constraint_graph_artifact_sha256": _sha256(graph_path),
        "configuration_atlas_spec_sha256": _sha256(spec_path),
        "exact_regeneration_match": exact,
        "executed_configuration_node_count": node_count,
        "issues": issues,
        "meaning": (
            "pass proves exact deterministic regeneration and standalone execution of every stored witness node; "
            "it does not prove exhaustive branch coverage, global topology, certified singularities, or physical truth"
        ),
    }
