#!/usr/bin/env python3
"""Independent analytic oracle for supplemental loop and coordinate constraints."""

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


SCHEMA = "robot-spatial-constraint-independent-oracle.v1"


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def run_cli(script: Path, arguments: list[str], allowed: tuple[int, ...] = (0,)) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
    )
    if completed.returncode not in allowed:
        raise RuntimeError(
            f"CLI returned {completed.returncode}: {' '.join(arguments)}\n{completed.stderr}\n{completed.stdout}"
        )
    stream = completed.stdout if completed.stdout.strip() else completed.stderr
    return json.loads(stream)


def write_pose(path: Path, joints: dict[str, float], name: str) -> None:
    path.write_text(dump({"pose_name": name, "joints": joints}), encoding="utf-8")


def fourbar_urdf(case_id: str, crank_length: float, ground_length: float) -> str:
    return f'''<robot name="{case_id}">
  <link name="ground"/><link name="crank"/><link name="coupler"/><link name="rocker"/>
  <joint name="input" type="revolute"><parent link="ground"/><child link="crank"/>
    <axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8" effort="1" velocity="1"/></joint>
  <joint name="coupler_relative" type="continuous"><parent link="crank"/><child link="coupler"/>
    <origin xyz="{crank_length:.12g} 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="rocker" type="continuous"><parent link="ground"/><child link="rocker"/>
    <origin xyz="{ground_length:.12g} 0 0"/><axis xyz="0 0 1"/></joint>
</robot>
'''


def fourbar_spec(grammar_sha: str, crank_length: float, ground_length: float) -> dict[str, Any]:
    return {
        "schema_version": "robot-spatial-constraint-spec.v1",
        "constraint_set_id": "oracle-fourbar",
        "grammar_sha256": grammar_sha,
        "attachments": [
            {
                "attachment_id": "coupler_tip",
                "parent_frame": "coupler",
                "semantic_role": "constraint_anchor",
                "parent_from_attachment": {
                    "translation_xyz_m": [ground_length, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "attachment_id": "rocker_tip",
                "parent_frame": "rocker",
                "semantic_role": "constraint_anchor",
                "parent_from_attachment": {
                    "translation_xyz_m": [crank_length, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ],
        "constraints": [
            {
                "constraint_id": "loop_joint",
                "type": "kinematic_pair",
                "role": "loop_closure",
                "frame_a": "attachment/coupler_tip",
                "frame_b": "attachment/rocker_tip",
                "joint_type": "revolute",
                "axis_xyz_in_a": [0.0, 0.0, 1.0],
                "axis_xyz_in_b": [0.0, 0.0, 1.0],
                "tolerances": {"translation_m": 1e-8, "rotation_rad": 1e-8},
            }
        ],
    }


def fourbar_points(crank_length: float, ground_length: float, pose: dict[str, float]) -> tuple[list[float], list[float]]:
    input_angle = pose["input"]
    coupler_angle = input_angle + pose["coupler_relative"]
    rocker_angle = pose["rocker"]
    coupler_tip = [
        crank_length * math.cos(input_angle) + ground_length * math.cos(coupler_angle),
        crank_length * math.sin(input_angle) + ground_length * math.sin(coupler_angle),
    ]
    rocker_tip = [
        ground_length + crank_length * math.cos(rocker_angle),
        crank_length * math.sin(rocker_angle),
    ]
    return coupler_tip, rocker_tip


def point_error(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def check_fourbar(
    script: Path,
    root: Path,
    index: int,
    rng: random.Random,
    analytic_tolerance: float,
) -> dict[str, Any]:
    case_id = f"fourbar-{index:04d}"
    crank_length = round(rng.uniform(0.45, 1.35), 6)
    ground_length = round(rng.uniform(1.1, 2.8), 6)
    angle = rng.uniform(-1.2, 1.2)
    if abs(angle) < 0.3:
        angle += 0.5 if angle >= 0.0 else -0.5
    source = root / "tree.urdf"
    grammar = root / "grammar.json"
    spec = root / "constraints.json"
    graph = root / "graph.json"
    source.write_text(fourbar_urdf(case_id, crank_length, ground_length), encoding="utf-8")
    run_cli(script, ["articulation-grammar", str(source), "--out", str(grammar)])
    spec.write_text(
        dump(fourbar_spec(hashlib.sha256(grammar.read_bytes()).hexdigest(), crank_length, ground_length)),
        encoding="utf-8",
    )
    compiled = run_cli(script, ["constraint-graph", str(grammar), str(spec), "--out", str(graph)])
    valid = {"input": angle, "coupler_relative": -angle, "rocker": angle}
    valid_pose = root / "valid.json"
    write_pose(valid_pose, valid, f"{case_id}/valid")
    valid_result = run_cli(script, ["evaluate-constraints", str(graph), "--pose", str(valid_pose)])
    left, right = fourbar_points(crank_length, ground_length, valid)
    valid_analytic_error = point_error(left, right)

    invalid = dict(valid)
    invalid["coupler_relative"] += 0.19
    invalid_pose = root / "invalid.json"
    write_pose(invalid_pose, invalid, f"{case_id}/invalid")
    invalid_result = run_cli(
        script,
        ["evaluate-constraints", str(graph), "--pose", str(invalid_pose), "--no-local-analysis"],
        (1,),
    )
    invalid_left, invalid_right = fourbar_points(crank_length, ground_length, invalid)
    invalid_analytic_error = point_error(invalid_left, invalid_right)

    seed = {
        "input": angle,
        "coupler_relative": -angle + rng.uniform(-0.12, 0.12),
        "rocker": angle + rng.uniform(-0.12, 0.12),
    }
    seed_pose = root / "seed.json"
    write_pose(seed_pose, seed, f"{case_id}/seed")
    solution = run_cli(script, [
        "solve-constraints",
        str(graph),
        "--pose",
        str(seed_pose),
        "--solve-for",
        "coupler_relative",
        "--solve-for",
        "rocker",
    ])
    solved = solution["solved_independent_driver_positions"]
    solved_left, solved_right = fourbar_points(crank_length, ground_length, solved)
    solved_analytic_error = point_error(solved_left, solved_right)
    verification = run_cli(script, [
        "verify-constraint-graph",
        str(grammar),
        str(spec),
        "--graph",
        str(graph),
        "--pose",
        str(valid_pose),
    ])
    analysis = valid_result["local_constraint_analysis"]
    failures: list[str] = []
    if compiled["coverage"]["declared_cycle_count"] != 1:
        failures.append("declared_cycle_count")
    if valid_result["status"] != "satisfied" or valid_analytic_error > analytic_tolerance:
        failures.append("valid_pose")
    if analysis["local_constraint_rank"] != 2 or analysis["local_mobility_estimate"] != 1:
        failures.append("local_rank_or_mobility")
    if invalid_result["status"] != "violated" or invalid_analytic_error <= 1e-3:
        failures.append("invalid_pose_negative_control")
    if solution["status"] != "converged" or solved_analytic_error > analytic_tolerance:
        failures.append("local_solver")
    if abs(solved["input"] - angle) > 1e-12:
        failures.append("fixed_input_changed")
    if verification["status"] != "passed":
        failures.append("artifact_verification")
    return {
        "case_id": case_id,
        "crank_length_m": crank_length,
        "ground_length_m": ground_length,
        "input_angle_rad": angle,
        "valid_analytic_closure_error_m": valid_analytic_error,
        "invalid_analytic_closure_error_m": invalid_analytic_error,
        "solved_analytic_closure_error_m": solved_analytic_error,
        "local_constraint_rank": analysis["local_constraint_rank"],
        "local_mobility_estimate": analysis["local_mobility_estimate"],
        "solver_iterations": solution["iteration_count"],
        "failures": failures,
    }


def coupled_urdf(case_id: str) -> str:
    return f'''<robot name="{case_id}">
  <link name="root"/><link name="left"/><link name="right"/>
  <joint name="left_angle" type="revolute"><parent link="root"/><child link="left"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/></joint>
  <joint name="right_angle" type="revolute"><parent link="root"/><child link="right"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/></joint>
</robot>
'''


def check_coordinate(
    script: Path,
    root: Path,
    index: int,
    rng: random.Random,
    analytic_tolerance: float,
) -> dict[str, Any]:
    case_id = f"coordinate-{index:04d}"
    left_coefficient = rng.choice((-2.0, -1.5, 0.75, 1.25, 2.0))
    right_coefficient = rng.choice((-2.25, -1.0, 0.5, 1.5, 2.5))
    left_value = rng.uniform(-0.8, 0.8)
    right_value = rng.uniform(-0.8, 0.8)
    offset = -(left_coefficient * left_value + right_coefficient * right_value)
    source = root / "tree.urdf"
    grammar = root / "grammar.json"
    spec = root / "constraints.json"
    graph = root / "graph.json"
    source.write_text(coupled_urdf(case_id), encoding="utf-8")
    run_cli(script, ["articulation-grammar", str(source), "--out", str(grammar)])
    spec.write_text(dump({
        "schema_version": "robot-spatial-constraint-spec.v1",
        "constraint_set_id": "oracle-coordinate",
        "grammar_sha256": hashlib.sha256(grammar.read_bytes()).hexdigest(),
        "attachments": [],
        "constraints": [{
            "constraint_id": "linear_coupling",
            "type": "coordinate_linear",
            "role": "mechanical_coupling",
            "terms": [
                {"joint": "left_angle", "coefficient": left_coefficient},
                {"joint": "right_angle", "coefficient": right_coefficient},
            ],
            "offset": offset,
            "tolerance": 1e-9,
        }],
    }), encoding="utf-8")
    run_cli(script, ["constraint-graph", str(grammar), str(spec), "--out", str(graph)])
    valid = {"left_angle": left_value, "right_angle": right_value}
    valid_pose = root / "valid.json"
    write_pose(valid_pose, valid, f"{case_id}/valid")
    valid_result = run_cli(script, ["evaluate-constraints", str(graph), "--pose", str(valid_pose)])
    valid_equation = left_coefficient * left_value + right_coefficient * right_value + offset

    invalid = dict(valid)
    invalid["right_angle"] += 0.2
    invalid_pose = root / "invalid.json"
    write_pose(invalid_pose, invalid, f"{case_id}/invalid")
    invalid_result = run_cli(
        script,
        ["evaluate-constraints", str(graph), "--pose", str(invalid_pose), "--no-local-analysis"],
        (1,),
    )
    invalid_equation = left_coefficient * invalid["left_angle"] + right_coefficient * invalid["right_angle"] + offset

    seed = {"left_angle": left_value, "right_angle": right_value + rng.uniform(-0.25, 0.25)}
    seed_pose = root / "seed.json"
    write_pose(seed_pose, seed, f"{case_id}/seed")
    solution = run_cli(script, [
        "solve-constraints",
        str(graph),
        "--pose",
        str(seed_pose),
        "--solve-for",
        "right_angle",
    ])
    solved = solution["solved_independent_driver_positions"]
    solved_equation = left_coefficient * solved["left_angle"] + right_coefficient * solved["right_angle"] + offset
    analysis = valid_result["local_constraint_analysis"]
    failures: list[str] = []
    if valid_result["status"] != "satisfied" or abs(valid_equation) > analytic_tolerance:
        failures.append("valid_equation")
    if analysis["local_constraint_rank"] != 1 or analysis["local_mobility_estimate"] != 1:
        failures.append("local_rank_or_mobility")
    if invalid_result["status"] != "violated" or abs(invalid_equation) <= 1e-3:
        failures.append("invalid_negative_control")
    if solution["status"] != "converged" or abs(solved_equation) > analytic_tolerance:
        failures.append("coordinate_solver")
    if abs(solved["left_angle"] - left_value) > 1e-12:
        failures.append("fixed_coordinate_changed")
    return {
        "case_id": case_id,
        "coefficients": [left_coefficient, right_coefficient],
        "valid_analytic_equation_error": valid_equation,
        "invalid_analytic_equation_error": invalid_equation,
        "solved_analytic_equation_error": solved_equation,
        "local_constraint_rank": analysis["local_constraint_rank"],
        "local_mobility_estimate": analysis["local_mobility_estimate"],
        "solver_iterations": solution["iteration_count"],
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--fourbar-cases", type=int, default=32)
    parser.add_argument("--coordinate-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--analytic-tolerance", type=float, default=2e-8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.fourbar_cases < 0 or args.coordinate_cases < 0 or not args.fourbar_cases + args.coordinate_cases:
        parser.error("at least one positive case count is required")
    if not math.isfinite(args.analytic_tolerance) or args.analytic_tolerance <= 0.0:
        parser.error("--analytic-tolerance must be finite and positive")
    rng = random.Random(args.seed)
    fourbars: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="robot-spatial-constraint-oracle-") as temp_dir:
        root = Path(temp_dir)
        for index in range(args.fourbar_cases):
            case_root = root / f"fourbar-{index:04d}"
            case_root.mkdir()
            fourbars.append(check_fourbar(args.script, case_root, index, rng, args.analytic_tolerance))
        for index in range(args.coordinate_cases):
            case_root = root / f"coordinate-{index:04d}"
            case_root.mkdir()
            coordinates.append(check_coordinate(args.script, case_root, index, rng, args.analytic_tolerance))
    failures = [
        {"case_id": case["case_id"], "failures": case["failures"]}
        for case in [*fourbars, *coordinates]
        if case["failures"]
    ]
    report = {
        "schema_version": SCHEMA,
        "status": "passed" if not failures else "failed",
        "seed": args.seed,
        "analytic_tolerance": args.analytic_tolerance,
        "coverage": {
            "total_case_count": len(fourbars) + len(coordinates),
            "fourbar_loop_case_count": len(fourbars),
            "coordinate_coupling_case_count": len(coordinates),
            "valid_pose_count": len(fourbars) + len(coordinates),
            "invalid_negative_control_count": len(fourbars) + len(coordinates),
            "local_solver_case_count": len(fourbars) + len(coordinates),
            "artifact_verification_case_count": len(fourbars),
            "fourbar_expected_local_rank": 2,
            "fourbar_expected_local_mobility": 1,
            "coordinate_expected_local_rank": 1,
            "coordinate_expected_local_mobility": 1,
        },
        "maximum_valid_fourbar_closure_error_m": max(
            (case["valid_analytic_closure_error_m"] for case in fourbars), default=0.0
        ),
        "maximum_solved_fourbar_closure_error_m": max(
            (case["solved_analytic_closure_error_m"] for case in fourbars), default=0.0
        ),
        "maximum_valid_coordinate_error": max(
            (abs(case["valid_analytic_equation_error"]) for case in coordinates), default=0.0
        ),
        "maximum_solved_coordinate_error": max(
            (abs(case["solved_analytic_equation_error"]) for case in coordinates), default=0.0
        ),
        "failure_count": len(failures),
        "failures": failures,
        "cases": {"fourbar": fourbars, "coordinate": coordinates},
        "independence": (
            "this script imports no production parser, articulation, constraint, matrix, rank, residual, or solver implementation; "
            "it generates sources/specs, calls only the public CLI, and recomputes planar closure and coordinate equations analytically"
        ),
        "epistemic_scope": (
            "generated planar parallelogram loops and linear joint couplings under the documented supplemental schema; "
            "not proof for arbitrary mechanisms, global solution completeness, singular configurations, dynamics, contact, hardware, or physical truth"
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
