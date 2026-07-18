#!/usr/bin/env python3
"""Independent source generator and FK oracle for URDF/SDF/MJCF common articulation laws."""

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


SCHEMA = "robot-spatial-cross-representation-independent-oracle.v1"
EPSILON = 1e-12
Matrix = list[list[float]]
Vector = list[float]


def identity() -> Matrix:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [[sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)] for row in range(4)]


def translation(vector: Vector) -> Matrix:
    result = identity()
    result[0][3], result[1][3], result[2][3] = vector
    return result


def axis_angle(axis: Vector, angle: float) -> Matrix:
    x, y, z = axis
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def fmt(values: Vector) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def clean(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(value, 12)


def run_cli(script: Path, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"CLI failed ({completed.returncode}): {' '.join(arguments)}\n{completed.stderr}\n{completed.stdout}")
    return json.loads(completed.stdout)


def random_law(index: int, rng: random.Random, link_count: int) -> dict[str, Any]:
    axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    kinds = ("fixed", "revolute", "continuous", "prismatic")
    edges: list[dict[str, Any]] = []
    for child in range(1, link_count):
        parent = rng.randrange(child)
        kind = kinds[(index + child) % len(kinds)]
        origin = [round(rng.uniform(-0.45, 0.45), 3) for _ in range(3)]
        axis = list(axes[(index + 2 * child) % len(axes)])
        if kind == "revolute":
            lower, upper = round(rng.uniform(-1.5, -0.2), 3), round(rng.uniform(0.2, 1.5), 3)
        elif kind == "prismatic":
            lower, upper = round(rng.uniform(-0.2, 0.0), 3), round(rng.uniform(0.05, 0.4), 3)
        else:
            lower = upper = None
        edges.append({
            "index": child - 1,
            "parent": parent,
            "child": child,
            "type": kind,
            "origin": origin,
            "post": [0.0, 0.0, 0.0],
            "axis": axis,
            "lower": lower,
            "upper": upper,
        })
    return {"case_id": f"common-{index:04d}", "link_count": link_count, "edges": edges}


def names(prefix: str, law: dict[str, Any]) -> tuple[list[str], dict[int, str]]:
    links = [f"{prefix}_link_{index}" for index in range(law["link_count"])]
    joints = {edge["index"]: f"{prefix}_joint_{edge['index']}" for edge in law["edges"]}
    return links, joints


def urdf_source(law: dict[str, Any]) -> tuple[str, list[str], dict[int, str]]:
    links, joints = names("u", law)
    lines = [f'<robot name="{law["case_id"]}_urdf">']
    lines.extend(f'  <link name="{name}"/>' for name in links)
    for edge in law["edges"]:
        name = joints[edge["index"]]
        lines.extend([
            f'  <joint name="{name}" type="{edge["type"]}">',
            f'    <parent link="{links[edge["parent"]]}"/><child link="{links[edge["child"]]}"/>',
            f'    <origin xyz="{fmt(edge["origin"])}" rpy="0 0 0"/>',
        ])
        if edge["type"] != "fixed":
            lines.append(f'    <axis xyz="{fmt(edge["axis"])}"/>')
            if edge["type"] in {"revolute", "prismatic"}:
                lines.append(f'    <limit lower="{edge["lower"]}" upper="{edge["upper"]}" effort="1" velocity="1"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n", links, joints


def sdf_source(law: dict[str, Any], same_names: bool = False) -> tuple[str, list[str], dict[int, str]]:
    links, joints = names("u" if same_names else "s", law)
    lines = [f'<sdf version="1.11"><model name="{law["case_id"]}_sdf">']
    child_edge = {edge["child"]: edge for edge in law["edges"]}
    for index, link in enumerate(links):
        if index == 0:
            lines.append(f'  <link name="{link}"/>')
        else:
            edge = child_edge[index]
            if any(abs(value) > EPSILON for value in edge["post"]):
                zero = [edge["origin"][axis] + edge["post"][axis] for axis in range(3)]
                lines.append(f'  <link name="{link}"><pose relative_to="{links[edge["parent"]]}">{fmt(zero)} 0 0 0</pose></link>')
            else:
                lines.append(f'  <link name="{link}"><pose relative_to="{joints[edge["index"]]}">0 0 0 0 0 0</pose></link>')
    for edge in law["edges"]:
        name = joints[edge["index"]]
        lines.extend([
            f'  <joint name="{name}" type="{edge["type"]}">',
            f'    <parent>{links[edge["parent"]]}</parent><child>{links[edge["child"]]}</child>',
            f'    <pose relative_to="{links[edge["parent"]]}">{fmt(edge["origin"])} 0 0 0</pose>',
        ])
        if edge["type"] != "fixed":
            lines.append(f'    <axis><xyz expressed_in="{name}">{fmt(edge["axis"])}</xyz>')
            if edge["type"] in {"revolute", "prismatic"}:
                lines.append(f'      <limit><lower>{edge["lower"]}</lower><upper>{edge["upper"]}</upper></limit>')
            lines.append("    </axis>")
        lines.append("  </joint>")
    lines.append("</model></sdf>")
    return "\n".join(lines) + "\n", links, joints


def mjcf_source(law: dict[str, Any], same_names: bool = False) -> tuple[str, list[str], dict[int, str]]:
    prefix = "u" if same_names else "m"
    links, desired_joints = names(prefix, law)
    edge_by_child = {edge["child"]: edge for edge in law["edges"]}
    children: dict[int, list[int]] = {index: [] for index in range(law["link_count"])}
    for edge in law["edges"]:
        children[edge["parent"]].append(edge["child"])
    actual_joints: dict[int, str] = {}

    def body(index: int, indent: str) -> list[str]:
        if index == 0:
            opening = f'{indent}<body name="{links[index]}">'
        else:
            edge = edge_by_child[index]
            zero = [edge["origin"][axis] + edge["post"][axis] for axis in range(3)]
            opening = f'{indent}<body name="{links[index]}" pos="{fmt(zero)}">'
        result = [opening]
        if index != 0:
            edge = edge_by_child[index]
            if edge["type"] == "fixed":
                actual_joints[edge["index"]] = f"fixed__{links[edge['parent']]}__{links[edge['child']]}"
            else:
                actual_joints[edge["index"]] = desired_joints[edge["index"]]
                raw_type = "slide" if edge["type"] == "prismatic" else "hinge"
                anchor = [-value for value in edge["post"]]
                attributes = [
                    f'name="{desired_joints[edge["index"]]}"',
                    f'type="{raw_type}"',
                    f'pos="{fmt(anchor)}"',
                    f'axis="{fmt(edge["axis"])}"',
                ]
                if edge["type"] in {"revolute", "prismatic"}:
                    attributes.extend(['limited="true"', f'range="{edge["lower"]} {edge["upper"]}"'])
                result.append(f'{indent}  <joint {" ".join(attributes)}/>')
        for child in sorted(children[index]):
            result.extend(body(child, indent + "  "))
        result.append(f"{indent}</body>")
        return result

    lines = [f'<mujoco model="{law["case_id"]}_mjcf"><compiler coordinate="local" angle="radian"/><worldbody>']
    lines.extend(body(0, "  "))
    lines.append("</worldbody></mujoco>")
    return "\n".join(lines) + "\n", links, actual_joints


def correspondence(reference: Path, candidate: Path, reference_links: list[str], candidate_links: list[str], reference_joints: dict[int, str], candidate_joints: dict[int, str]) -> dict[str, Any]:
    return {
        "schema_version": "robot-spatial-articulation-correspondence.v1",
        "reference_grammar_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "candidate_grammar_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "candidate_to_reference": {
            "links": dict(zip(candidate_links, reference_links)),
            "joints": {candidate_joints[index]: reference_joints[index] for index in sorted(reference_joints)},
            "frames": {},
        },
    }


def independent_frames(law: dict[str, Any], pose: dict[int, float]) -> tuple[dict[int, Matrix], dict[int, Matrix]]:
    link_frames: dict[int, Matrix] = {0: identity()}
    joint_frames: dict[int, Matrix] = {}
    children: dict[int, list[dict[str, Any]]] = {index: [] for index in range(law["link_count"])}
    for edge in law["edges"]:
        children[edge["parent"]].append(edge)

    def descend(parent: int) -> None:
        for edge in children[parent]:
            pre = matmul(link_frames[parent], translation(edge["origin"]))
            joint_frames[edge["index"]] = pre
            value = pose.get(edge["index"], 0.0)
            if edge["type"] in {"revolute", "continuous"}:
                motion = axis_angle(edge["axis"], value)
            elif edge["type"] == "prismatic":
                motion = translation([component * value for component in edge["axis"]])
            else:
                motion = identity()
            link_frames[edge["child"]] = matmul(matmul(pre, motion), translation(edge["post"]))
            descend(edge["child"])

    descend(0)
    return link_frames, joint_frames


def max_error(left: Matrix, right: Matrix) -> float:
    return max(abs(float(left[row][column]) - float(right[row][column])) for row in range(4) for column in range(4))


def check_case(script: Path, root: Path, law: dict[str, Any], formats: tuple[str, ...], poses: int, rng: random.Random, tolerance: float) -> dict[str, Any]:
    source_builders = {"urdf": urdf_source, "sdf": sdf_source, "mjcf": mjcf_source}
    grammar_paths: dict[str, Path] = {}
    source_names: dict[str, tuple[list[str], dict[int, str]]] = {}
    failures: list[dict[str, Any]] = []
    for source_format in formats:
        source_text, link_names, joint_names = source_builders[source_format](law)
        extension = {"urdf": ".urdf", "sdf": ".sdf", "mjcf": ".xml"}[source_format]
        source_path = root / f"{source_format}{extension}"
        grammar_path = root / f"{source_format}.json"
        source_path.write_text(source_text)
        run_cli(script, ["articulation-grammar", str(source_path), "--out", str(grammar_path)])
        verification = run_cli(script, ["verify-articulation-grammar", str(source_path), "--grammar", str(grammar_path)])
        if verification["status"] != "passed":
            failures.append({"check": f"{source_format}.verification", "report": verification})
        grammar_paths[source_format] = grammar_path
        source_names[source_format] = (link_names, joint_names)
    reference_format = formats[0]
    for candidate_format in formats[1:]:
        reference_links, reference_joints = source_names[reference_format]
        candidate_links, candidate_joints = source_names[candidate_format]
        mapping_path = root / f"{candidate_format}-mapping.json"
        mapping_path.write_text(json.dumps(correspondence(
            grammar_paths[reference_format],
            grammar_paths[candidate_format],
            reference_links,
            candidate_links,
            reference_joints,
            candidate_joints,
        ), indent=2, sort_keys=True) + "\n")
        comparison = run_cli(script, [
            "compare-articulation-grammars",
            str(grammar_paths[reference_format]),
            str(grammar_paths[candidate_format]),
            "--correspondence",
            str(mapping_path),
            "--tolerance",
            str(tolerance),
        ])
        if comparison["status"] != "equivalent":
            failures.append({"check": f"{reference_format}_vs_{candidate_format}", "report": comparison})

    maximum = 0.0
    frame_checks = 0
    for pose_index in range(poses):
        indexed_pose: dict[int, float] = {}
        for edge in law["edges"]:
            if edge["type"] == "fixed":
                continue
            if edge["lower"] is not None:
                value = rng.uniform(edge["lower"], edge["upper"])
            else:
                value = rng.uniform(-1.0, 1.0)
            indexed_pose[edge["index"]] = value
        expected_links, expected_joints = independent_frames(law, indexed_pose)
        for source_format in formats:
            link_names, joint_names = source_names[source_format]
            binding = {
                joint_names[index]: value
                for index, value in indexed_pose.items()
            }
            pose_path = root / f"pose-{source_format}-{pose_index}.json"
            pose_path.write_text(json.dumps({"pose_name": f"oracle/{pose_index}", "joints": binding}))
            evaluated = run_cli(script, ["evaluate-articulation", str(grammar_paths[source_format]), "--pose", str(pose_path)])
            for link_index, expected in expected_links.items():
                actual = evaluated["frames"][link_names[link_index]]["root_from_frame"]["matrix_4x4_rowmajor"]
                error = max_error(actual, expected)
                maximum = max(maximum, error)
                frame_checks += 1
                if error > tolerance:
                    failures.append({"check": f"{source_format}.pose_{pose_index}.link_{link_index}", "error": error})
            for joint_index, expected in expected_joints.items():
                actual = evaluated["frames"][f"joint/{joint_names[joint_index]}"]["root_from_frame"]["matrix_4x4_rowmajor"]
                error = max_error(actual, expected)
                maximum = max(maximum, error)
                frame_checks += 1
                if error > tolerance:
                    failures.append({"check": f"{source_format}.pose_{pose_index}.joint_{joint_index}", "error": error})
    return {
        "case_id": law["case_id"],
        "formats": list(formats),
        "link_count": law["link_count"],
        "joint_count": len(law["edges"]),
        "non_identity_post_motion_joint_count": sum(any(abs(value) > EPSILON for value in edge["post"]) for edge in law["edges"]),
        "pose_count": poses,
        "independent_all_frame_checks": frame_checks,
        "maximum_matrix_absolute_error": clean(maximum),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--post-anchor-cases", type=int, default=24)
    parser.add_argument("--poses-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=420731)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.cases, args.post_anchor_cases) < 0 or args.poses_per_case <= 0 or args.tolerance <= 0.0:
        raise SystemExit("case counts must be non-negative; poses and tolerance must be positive")
    rng = random.Random(args.seed)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        generated = Path(temp_dir)
        for index in range(args.cases):
            law = random_law(index, rng, rng.randint(2, 7))
            case_root = generated / law["case_id"]
            case_root.mkdir()
            results.append(check_case(args.script, case_root, law, ("urdf", "sdf", "mjcf"), args.poses_per_case, rng, args.tolerance))
        for index in range(args.post_anchor_cases):
            law = random_law(args.cases + index, rng, rng.randint(2, 7))
            law["case_id"] = f"post-anchor-{index:04d}"
            for edge in law["edges"]:
                edge["post"] = (
                    [0.0, 0.0, 0.0]
                    if edge["type"] == "fixed"
                    else [round(rng.uniform(-0.2, 0.2), 3) for _ in range(3)]
                )
            case_root = generated / law["case_id"]
            case_root.mkdir()
            results.append(check_case(args.script, case_root, law, ("sdf", "mjcf"), args.poses_per_case, rng, args.tolerance))
    failures = [
        {"case_id": result["case_id"], **failure}
        for result in results
        for failure in result["failures"]
    ]
    report = {
        "schema_version": SCHEMA,
        "status": "passed" if not failures else "failed",
        "seed": args.seed,
        "production_cli": str(args.script.resolve()),
        "production_imports_used": False,
        "oracle_implementation": "independent random common-law source generation plus analytic pre-motion/motion/post-motion FK",
        "coverage": {
            "urdf_sdf_mjcf_common_case_count": args.cases,
            "sdf_mjcf_non_identity_post_motion_case_count": args.post_anchor_cases,
            "total_case_count": len(results),
            "pose_count": sum(result["pose_count"] for result in results),
            "independent_all_frame_evaluation_count": sum(result["independent_all_frame_checks"] for result in results),
            "non_identity_post_motion_joint_count": sum(result["non_identity_post_motion_joint_count"] for result in results),
            "fixed_revolute_continuous_prismatic": args.cases > 0,
            "branched_and_serial_trees": args.cases > 1,
            "different_typed_identifiers_with_digest_bound_correspondence": args.cases > 0,
        },
        "tolerance": args.tolerance,
        "maximum_matrix_absolute_error": max((result["maximum_matrix_absolute_error"] for result in results), default=0.0),
        "failure_count": len(failures),
        "failures": failures,
        "cases": results,
        "epistemic_scope": (
            "independent agreement for generated supported common tree laws, typed correspondence, joint anchors, and unseen-pose all-frame FK; "
            "not validation of arbitrary external source files, unsupported format constructs, dynamics, closed loops, hardware, or physical truth"
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
