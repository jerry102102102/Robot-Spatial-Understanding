#!/usr/bin/env python3
"""Independent analytic oracle for finite spherical-loop configuration atlases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "robot-spatial-configuration-atlas-independent-oracle.v1"


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def run_cli(script: Path, arguments: list[str], allowed: tuple[int, ...] = (0,)) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode not in allowed:
        raise RuntimeError(
            f"CLI returned {completed.returncode}: {' '.join(arguments)}\n{completed.stderr}\n{completed.stdout}"
        )
    return json.loads(completed.stdout if completed.stdout.strip() else completed.stderr)


def spherical_urdf(case_id: str) -> str:
    return f'''<robot name="{case_id}">
  <link name="ground"/><link name="x_rotor"/><link name="z_rotor"/><link name="end_rotor"/>
  <joint name="q_a" type="revolute"><parent link="ground"/><child link="x_rotor"/>
    <axis xyz="1 0 0"/><limit lower="-1.2" upper="1.2" effort="1" velocity="1"/></joint>
  <joint name="q_b" type="continuous"><parent link="x_rotor"/><child link="z_rotor"/><axis xyz="0 0 1"/></joint>
  <joint name="q_c" type="continuous"><parent link="z_rotor"/><child link="end_rotor"/><axis xyz="1 0 0"/></joint>
</robot>
'''


def constraint_spec(grammar_sha256: str) -> dict[str, Any]:
    identity = {
        "translation_xyz_m": [0.0, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    return {
        "schema_version": "robot-spatial-constraint-spec.v1",
        "constraint_set_id": "analytic-spherical-three-r-loop",
        "grammar_sha256": grammar_sha256,
        "attachments": [
            {
                "attachment_id": "ground_axis",
                "parent_frame": "ground",
                "semantic_role": "joint_anchor",
                "parent_from_attachment": identity,
            },
            {
                "attachment_id": "end_axis",
                "parent_frame": "end_rotor",
                "semantic_role": "joint_anchor",
                "parent_from_attachment": identity,
            },
        ],
        "constraints": [{
            "constraint_id": "terminal_z_axis_closure",
            "type": "kinematic_pair",
            "role": "loop_closure",
            "frame_a": "attachment/ground_axis",
            "frame_b": "attachment/end_axis",
            "joint_type": "revolute",
            "axis_xyz_in_a": [0.0, 0.0, 1.0],
            "axis_xyz_in_b": [0.0, 0.0, 1.0],
            "tolerances": {"translation_m": 1e-8, "rotation_rad": 1e-8},
        }],
    }


def atlas_spec(graph_sha256: str, amplitude: float) -> dict[str, Any]:
    return {
        "schema_version": "robot-spatial-configuration-atlas-spec.v1",
        "atlas_id": "analytic-spherical-loop-witnesses",
        "constraint_graph_sha256": graph_sha256,
        "singular_value_relative_tolerance": 1e-7,
        "charts": [{
            "chart_id": "drive-q-a",
            "parameter_driver": "q_a",
            "parameter_values": [-amplitude, 0.0, amplitude],
            "solve_for": ["q_b", "q_c"],
            "driver_scales": {"q_a": 1.0, "q_b": 1.0, "q_c": 1.0},
            "seeds": [
                {"seed_id": "branch-zero", "joints": {"q_a": 0.0, "q_b": 0.0, "q_c": 0.0}},
                {"seed_id": "singular-mid", "joints": {"q_a": 0.0, "q_b": math.pi / 2.0, "q_c": 0.0}},
                {"seed_id": "branch-pi", "joints": {"q_a": 0.0, "q_b": math.pi, "q_c": 0.0}},
            ],
            "solution_merge_tolerance_normalized": 1e-5,
            "continuation_edge_max_distance_normalized": 2.0,
            "minimum_solutions_per_sample": 2,
        }],
    }


def wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def terminal_z_axis(a: float, b: float, c: float) -> list[float]:
    """Analytic R_x(a) R_z(b) R_x(c) applied to the terminal z axis."""
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_b, cos_b = math.sin(b), math.cos(b)
    sin_c, cos_c = math.sin(c), math.cos(c)
    return [
        sin_b * sin_c,
        -cos_a * cos_b * sin_c - sin_a * cos_c,
        -sin_a * cos_b * sin_c + cos_a * cos_c,
    ]


def alignment_error(positions: dict[str, float]) -> float:
    axis = terminal_z_axis(positions["q_a"], positions["q_b"], positions["q_c"])
    return math.sqrt(axis[0] * axis[0] + axis[1] * axis[1])


def nearest_branch(
    solutions: list[dict[str, Any]],
    target_b: float,
) -> dict[str, float]:
    return min(
        (node["independent_driver_positions"] for node in solutions),
        key=lambda positions: abs(wrap(positions["q_b"] - target_b)),
    )


def check_case(
    script: Path,
    root: Path,
    index: int,
    amplitude: float,
    analytic_tolerance: float,
) -> dict[str, Any]:
    case_id = f"spherical-atlas-{index:04d}"
    source = root / "tree.urdf"
    grammar = root / "grammar.json"
    constraint_source = root / "constraints.json"
    graph = root / "constraint-graph.json"
    atlas_source = root / "configuration-atlas-spec.json"
    atlas_path = root / "configuration-atlas.json"
    source.write_text(spherical_urdf(case_id), encoding="utf-8")
    run_cli(script, ["articulation-grammar", str(source), "--out", str(grammar)])
    constraint_source.write_text(
        dump(constraint_spec(hashlib.sha256(grammar.read_bytes()).hexdigest())),
        encoding="utf-8",
    )
    run_cli(script, ["constraint-graph", str(grammar), str(constraint_source), "--out", str(graph)])
    atlas_source.write_text(
        dump(atlas_spec(hashlib.sha256(graph.read_bytes()).hexdigest(), amplitude)),
        encoding="utf-8",
    )
    generated = run_cli(
        script,
        ["configuration-atlas", str(graph), str(atlas_source), "--out", str(atlas_path)],
    )
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    verified = run_cli(script, [
        "verify-configuration-atlas",
        str(graph),
        str(atlas_source),
        "--atlas",
        str(atlas_path),
    ])
    chart = atlas["charts"][0]
    failures: list[str] = []
    maximum_axis_error = 0.0
    for sample in chart["samples"]:
        for node in sample["solutions"]:
            maximum_axis_error = max(
                maximum_axis_error,
                alignment_error(node["independent_driver_positions"]),
            )
    if maximum_axis_error > analytic_tolerance:
        failures.append("analytic_terminal_axis_alignment")
    generic_branch_errors: list[float] = []
    for sample in (chart["samples"][0], chart["samples"][2]):
        parameter = sample["parameter_value"]
        zero = nearest_branch(sample["solutions"], 0.0)
        pi = nearest_branch(sample["solutions"], math.pi)
        errors = [
            abs(wrap(zero["q_b"])),
            abs(zero["q_c"] + parameter),
            abs(abs(wrap(pi["q_b"])) - math.pi),
            abs(pi["q_c"] - parameter),
        ]
        generic_branch_errors.extend(errors)
        if max(errors) > analytic_tolerance:
            failures.append(f"analytic_two_branch_law_sample_{sample['sample_index']}")
    singular_sample = chart["samples"][1]
    singular_by_b = sorted(
        singular_sample["solutions"],
        key=lambda node: node["independent_driver_positions"]["q_b"],
    )
    if len(singular_by_b) < 3:
        failures.append("singular_slice_seed_coverage")
    mid = nearest_branch(singular_by_b, math.pi / 2.0)
    if abs(mid["q_c"]) > analytic_tolerance:
        failures.append("analytic_singular_slice_mid_solution")
    zero_node = min(
        singular_by_b,
        key=lambda node: abs(wrap(node["independent_driver_positions"]["q_b"])),
    )
    mid_node = min(
        singular_by_b,
        key=lambda node: abs(wrap(node["independent_driver_positions"]["q_b"] - math.pi / 2.0)),
    )
    if zero_node["full_constraint_jacobian"]["numerical_rank"] != 1:
        failures.append("analytic_rank_one_at_b_zero")
    if mid_node["full_constraint_jacobian"]["numerical_rank"] != 2:
        failures.append("analytic_rank_two_at_b_mid")
    if not zero_node["singularity_witness"]["mechanism_rank_drop_candidate"]:
        failures.append("relative_rank_drop_candidate")
    if generated["sampling_status"] != "complete_for_declared_sampling":
        failures.append("declared_sample_minimum")
    if verified["status"] != "passed" or not verified["exact_regeneration_match"]:
        failures.append("exact_regeneration_and_node_execution")
    return {
        "case_id": case_id,
        "amplitude_rad": amplitude,
        "maximum_analytic_terminal_axis_alignment_error": maximum_axis_error,
        "maximum_generic_branch_law_error_rad": max(generic_branch_errors, default=0.0),
        "zero_slice_unique_solution_count": singular_sample["unique_solution_count"],
        "rank_at_zero_branch_singular_slice": zero_node["full_constraint_jacobian"]["numerical_rank"],
        "rank_at_mid_singular_slice": mid_node["full_constraint_jacobian"]["numerical_rank"],
        "atlas_node_count": atlas["coverage"]["unique_solution_node_count"],
        "executed_verification_node_count": verified["executed_configuration_node_count"],
        "failures": sorted(set(failures)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--analytic-tolerance", type=float, default=2e-6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    if not math.isfinite(args.analytic_tolerance) or args.analytic_tolerance <= 0.0:
        parser.error("--analytic-tolerance must be finite and positive")
    rng = random.Random(args.seed)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="robot-spatial-configuration-oracle-") as temp_dir:
        root = Path(temp_dir)
        for index in range(args.cases):
            case_root = root / f"case-{index:04d}"
            case_root.mkdir()
            amplitude = round(rng.uniform(0.25, 0.9), 6)
            cases.append(check_case(args.script, case_root, index, amplitude, args.analytic_tolerance))
    failures = [
        {"case_id": case["case_id"], "failures": case["failures"]}
        for case in cases
        if case["failures"]
    ]
    report = {
        "schema_version": SCHEMA,
        "status": "passed" if not failures else "failed",
        "seed": args.seed,
        "analytic_tolerance": args.analytic_tolerance,
        "coverage": {
            "case_count": len(cases),
            "nonplanar_spherical_three_r_loop_count": len(cases),
            "generic_two_branch_sample_count": 2 * len(cases),
            "singular_slice_sample_count": len(cases),
            "analytic_rank_contrast_count": len(cases),
            "exact_regeneration_and_node_execution_count": len(cases),
        },
        "maximum_analytic_terminal_axis_alignment_error": max(
            case["maximum_analytic_terminal_axis_alignment_error"] for case in cases
        ),
        "maximum_generic_branch_law_error_rad": max(
            case["maximum_generic_branch_law_error_rad"] for case in cases
        ),
        "failure_count": len(failures),
        "failures": failures,
        "cases": cases,
        "independence": (
            "this script imports no production parser, articulation, constraint, configuration, matrix, Jacobian, rank, residual, or solver implementation; "
            "it calls only the public CLI and independently evaluates R_x(a) R_z(b) R_x(c), its two analytic closure branches, and its rank contrast"
        ),
        "epistemic_scope": (
            "randomized amplitudes of one nonplanar spherical three-revolute closure family with explicit finite samples and seeds; "
            "not proof of exhaustive configuration-space coverage, arbitrary mechanism topology, certified singularities, physical assembly, dynamics, contact, hardware, or safety"
        ),
    }
    serialized = dump(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
