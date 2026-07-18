#!/usr/bin/env python3
"""Independent parser/FK oracle for robot-spatial-articulation-grammar.v1.

This file intentionally imports no production parser, matrix, grammar, or FK
module. It invokes only the public CLI and recomputes expected laws and poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SCHEMA = "robot-spatial-independent-articulation-crosscheck.v1"
GRAMMAR_SCHEMA = "robot-spatial-articulation-grammar.v1"
EPSILON = 1e-12


def clean(value: float) -> float:
    return 0.0 if abs(value) < EPSILON else round(float(value), 12)


def vec(raw: str | None, default: list[float]) -> list[float]:
    if raw is None:
        return list(default)
    values = [float(value) for value in raw.split()]
    if len(values) != 3:
        raise ValueError(f"expected three-vector, got {raw!r}")
    return values


def identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[row][inner] * right[inner][column] for inner in range(4)) for column in range(4)] for row in range(4)]


def origin(xyz: list[float], rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
        [-sp, cp * sr, cp * cr, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation(axis: list[float], angle: float) -> list[list[float]]:
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = [value / norm for value in axis]
    cosine, sine, one_minus = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0],
        [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0],
        [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation(vector: list[float]) -> list[list[float]]:
    result = identity()
    for index in range(3):
        result[index][3] = vector[index]
    return result


def parse_origin(element: ET.Element) -> tuple[list[float], list[float], list[list[float]]]:
    child = element.find("origin")
    xyz = vec(None if child is None else child.get("xyz"), [0.0, 0.0, 0.0])
    rpy = vec(None if child is None else child.get("rpy"), [0.0, 0.0, 0.0])
    return xyz, rpy, origin(xyz, rpy)


def parse_model(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_bytes())
    links: dict[str, Any] = {}
    frames: dict[str, dict[str, Any]] = {}
    for link_element in root.findall("link"):
        name = str(link_element.get("name"))
        links[name] = {"visuals": [], "collisions": [], "inertial": None}
        for collection, tag in (("visuals", "visual"), ("collisions", "collision")):
            for index, item in enumerate(link_element.findall(tag)):
                xyz, rpy, matrix = parse_origin(item)
                frame = f"{tag}/{name}/{index}"
                links[name][collection].append({"frame": frame, "xyz": xyz, "rpy": rpy, "matrix": matrix})
        inertial = link_element.find("inertial")
        if inertial is not None:
            xyz, rpy, matrix = parse_origin(inertial)
            links[name]["inertial"] = {"frame": f"inertial/{name}", "xyz": xyz, "rpy": rpy, "matrix": matrix}
    joints: dict[str, Any] = {}
    child_joint: dict[str, str] = {}
    children: dict[str, list[str]] = {name: [] for name in links}
    for element in root.findall("joint"):
        name, joint_type = str(element.get("name")), str(element.get("type"))
        parent = str(element.find("parent").get("link"))
        child = str(element.find("child").get("link"))
        xyz, rpy, matrix = parse_origin(element)
        axis_element = element.find("axis")
        axis = vec(None if axis_element is None else axis_element.get("xyz"), [1.0, 0.0, 0.0])
        if joint_type != "fixed":
            norm = math.sqrt(sum(value * value for value in axis))
            axis = [value / norm for value in axis]
        limit_element = element.find("limit")
        limit = {
            key: None if limit_element is None or limit_element.get(key) is None else float(limit_element.get(key))
            for key in ("lower", "upper")
        }
        mimic_element = element.find("mimic")
        mimic = None if mimic_element is None else {
            "joint": str(mimic_element.get("joint")),
            "multiplier": float(mimic_element.get("multiplier", "1")),
            "offset": float(mimic_element.get("offset", "0")),
        }
        joints[name] = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "origin": matrix,
            "axis": axis,
            "limit": limit,
            "mimic": mimic,
        }
        child_joint[child] = name
        children[parent].append(name)
    roots = sorted(set(links) - set(child_joint))
    if len(roots) != 1:
        raise ValueError(f"independent oracle requires one root, got {roots}")
    root_link = roots[0]
    frames[root_link] = {"type": "link", "owner": root_link, "parent": None}
    for name, joint in joints.items():
        frames[f"joint/{name}"] = {"type": "joint_pre_motion", "owner": name, "parent": joint["parent"]}
        frames[joint["child"]] = {"type": "link", "owner": joint["child"], "parent": f"joint/{name}"}
    attachments: dict[str, list[list[float]]] = {}
    for link_name, link in links.items():
        for collection, frame_type in (("visuals", "visual"), ("collisions", "collision")):
            for item in link[collection]:
                frames[item["frame"]] = {"type": frame_type, "owner": link_name, "parent": link_name}
                attachments[item["frame"]] = item["matrix"]
        if link["inertial"] is not None:
            item = link["inertial"]
            frames[item["frame"]] = {"type": "inertial", "owner": link_name, "parent": link_name}
            attachments[item["frame"]] = item["matrix"]
    return {
        "name": root.get("name") or "unnamed_robot",
        "root": root_link,
        "links": links,
        "joints": joints,
        "child_joint": child_joint,
        "children": children,
        "frames": frames,
        "attachments": attachments,
    }


def affine(model: dict[str, Any], joint_name: str) -> tuple[str, float, float, list[str]]:
    current, multiplier, offset = joint_name, 1.0, 0.0
    chain = [current]
    while model["joints"][current]["mimic"] is not None:
        mimic = model["joints"][current]["mimic"]
        offset = multiplier * mimic["offset"] + offset
        multiplier *= mimic["multiplier"]
        current = mimic["joint"]
        chain.append(current)
    return current, multiplier, offset, chain


def driver_contract(model: dict[str, Any], driver: str) -> dict[str, Any]:
    lower, upper = -math.inf, math.inf
    physical: list[str] = []
    constraints: list[dict[str, Any]] = []
    for name, joint in sorted(model["joints"].items()):
        if joint["type"] == "fixed":
            continue
        source, multiplier, offset, chain = affine(model, name)
        if source != driver:
            continue
        physical.append(name)
        declared_lower = None if joint["type"] == "continuous" else joint["limit"]["lower"]
        declared_upper = None if joint["type"] == "continuous" else joint["limit"]["upper"]
        constraints.append({
            "joint": name,
            "joint_type": joint["type"],
            "affine_position_from_driver": {"multiplier": clean(multiplier), "offset": clean(offset)},
            "mimic_chain": chain,
            "declared_lower": declared_lower,
            "declared_upper": declared_upper,
        })
        if abs(multiplier) <= EPSILON:
            continue
        if declared_lower is not None:
            boundary = (declared_lower - offset) / multiplier
            if multiplier > 0.0:
                lower = max(lower, boundary)
            else:
                upper = min(upper, boundary)
        if declared_upper is not None:
            boundary = (declared_upper - offset) / multiplier
            if multiplier > 0.0:
                upper = min(upper, boundary)
            else:
                lower = max(lower, boundary)
    return {
        "minimum": None if not math.isfinite(lower) else clean(lower),
        "maximum": None if not math.isfinite(upper) else clean(upper),
        "minimum_unbounded": not math.isfinite(lower),
        "maximum_unbounded": not math.isfinite(upper),
        "constraints": constraints,
        "physical": physical,
    }


def resolve(model: dict[str, Any], supplied: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    active: set[str] = set()

    def one(name: str) -> float:
        if name in result:
            return result[name]
        if name in active:
            raise ValueError("mimic cycle")
        active.add(name)
        joint = model["joints"][name]
        if joint["type"] == "fixed":
            value = 0.0
        elif joint["mimic"] is not None:
            mimic = joint["mimic"]
            value = mimic["multiplier"] * one(mimic["joint"]) + mimic["offset"]
        else:
            value = supplied.get(name, 0.0)
        active.remove(name)
        result[name] = clean(value)
        return value

    for name in model["joints"]:
        one(name)
    return result


def world_frames(model: dict[str, Any], supplied: dict[str, float]) -> tuple[dict[str, list[list[float]]], dict[str, float]]:
    positions = resolve(model, supplied)
    frames = {model["root"]: identity()}

    def descend(link: str) -> None:
        for joint_name in sorted(model["children"][link]):
            joint = model["joints"][joint_name]
            pre = multiply(frames[link], joint["origin"])
            frames[f"joint/{joint_name}"] = pre
            value = positions[joint_name]
            if joint["type"] in {"revolute", "continuous"}:
                motion = rotation(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion = translation([value * component for component in joint["axis"]])
            else:
                motion = identity()
            frames[joint["child"]] = multiply(pre, motion)
            descend(joint["child"])

    descend(model["root"])
    for frame, local in model["attachments"].items():
        owner = model["frames"][frame]["owner"]
        frames[frame] = multiply(frames[owner], local)
    return frames, positions


def root_path(model: dict[str, Any], link: str) -> list[str]:
    result: list[str] = []
    current = link
    while current != model["root"]:
        joint_name = model["child_joint"][current]
        result.append(joint_name)
        current = model["joints"][joint_name]["parent"]
    return list(reversed(result))


def max_error(left: list[list[float]], right: list[list[float]]) -> float:
    return max(abs(float(left[row][column]) - float(right[row][column])) for row in range(4) for column in range(4))


def matrix_from(record: dict[str, Any]) -> list[list[float]]:
    return [[float(value) for value in row] for row in record["matrix_4x4_rowmajor"]]


def feasible_probes(model: dict[str, Any], count: int, rng: random.Random) -> list[dict[str, float]]:
    drivers = sorted(
        name for name, joint in model["joints"].items()
        if joint["type"] != "fixed" and joint["mimic"] is None
    )
    contracts = {name: driver_contract(model, name) for name in drivers}
    probes: list[dict[str, float]] = []
    for probe_index in range(count):
        pose: dict[str, float] = {}
        for name in drivers:
            domain = contracts[name]
            lower, upper = domain["minimum"], domain["maximum"]
            if lower is None and upper is None:
                value = rng.uniform(-1.0, 1.0)
            elif lower is None:
                value = float(upper) - rng.uniform(0.05, 0.8)
            elif upper is None:
                value = float(lower) + rng.uniform(0.05, 0.8)
            elif abs(float(upper) - float(lower)) <= EPSILON:
                value = float(lower)
            else:
                fraction = (probe_index + 1) / (count + 1)
                value = float(lower) + fraction * (float(upper) - float(lower))
            pose[name] = clean(value)
        probes.append(pose)
    return probes


def random_vector(rng: random.Random, scale: float) -> str:
    return " ".join(f"{rng.uniform(-scale, scale):.12g}" for _ in range(3))


def random_urdf(case_index: int, rng: random.Random) -> str:
    multiplier = rng.choice([-1.25, -0.6, 0.55, 1.1])
    nested_multiplier = rng.choice([-0.8, 0.7])
    offset = rng.uniform(-0.15, 0.15)
    nested_offset = rng.uniform(-0.08, 0.08)
    yaw_type = "continuous" if case_index % 4 == 0 else "revolute"
    yaw_limit = "" if yaw_type == "continuous" else '<limit lower="-1.4" upper="1.2" effort="8" velocity="2"/>'
    axis = rng.choice(["0 0 1", "0 1 0", "1 0 0", "1 1 0"])
    return f'''<robot name="oracle_case_{case_index}">
      <link name="base"><visual><origin xyz="{random_vector(rng, 0.1)}" rpy="{random_vector(rng, 0.2)}"/><geometry><box size="0.2 0.3 0.1"/></geometry></visual></link>
      <link name="arm"><inertial><origin xyz="{random_vector(rng, 0.2)}" rpy="{random_vector(rng, 0.2)}"/><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
      <link name="carriage"><collision><origin xyz="{random_vector(rng, 0.1)}" rpy="{random_vector(rng, 0.2)}"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
      <link name="tool"/><link name="branch"/><link name="nested"/><link name="sensor"/>
      <joint name="yaw" type="{yaw_type}"><parent link="base"/><child link="arm"/><origin xyz="{random_vector(rng, 0.4)}" rpy="{random_vector(rng, 0.4)}"/><axis xyz="{axis}"/>{yaw_limit}</joint>
      <joint name="slide" type="prismatic"><parent link="arm"/><child link="carriage"/><origin xyz="{random_vector(rng, 0.3)}" rpy="{random_vector(rng, 0.4)}"/><axis xyz="1 -0.5 0.25"/><limit lower="-0.2" upper="0.8" effort="4" velocity="1"/></joint>
      <joint name="mount" type="fixed"><parent link="carriage"/><child link="tool"/><origin xyz="{random_vector(rng, 0.2)}" rpy="{random_vector(rng, 0.3)}"/></joint>
      <joint name="follower" type="revolute"><parent link="base"/><child link="branch"/><origin xyz="{random_vector(rng, 0.4)}" rpy="{random_vector(rng, 0.4)}"/><axis xyz="0 0 1"/><limit lower="-0.9" upper="0.9" effort="3" velocity="1"/><mimic joint="yaw" multiplier="{multiplier}" offset="{offset:.12g}"/></joint>
      <joint name="nested_follower" type="revolute"><parent link="branch"/><child link="nested"/><origin xyz="{random_vector(rng, 0.2)}" rpy="{random_vector(rng, 0.3)}"/><axis xyz="0 1 0"/><limit lower="-0.8" upper="0.8" effort="3" velocity="1"/><mimic joint="follower" multiplier="{nested_multiplier}" offset="{nested_offset:.12g}"/></joint>
      <joint name="sensor_mount" type="fixed"><parent link="arm"/><child link="sensor"/><origin xyz="{random_vector(rng, 0.2)}" rpy="{random_vector(rng, 0.3)}"/></joint>
    </robot>'''


def run_cli(script: Path, arguments: list[str]) -> tuple[int, str, str]:
    result = subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_case(
    script: Path,
    urdf: Path,
    case_id: str,
    pose_count: int,
    rng: random.Random,
    tolerance: float,
) -> dict[str, Any]:
    model = parse_model(urdf)
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        grammar_path = root / "grammar.json"
        code, stdout, stderr = run_cli(script, ["articulation-grammar", str(urdf), "--out", str(grammar_path)])
        if code != 0:
            return {"case_id": case_id, "failures": [{"check": "generate", "message": stderr}], "pose_count": 0, "frame_checks": 0, "max_error": None}
        grammar = json.loads(grammar_path.read_text())
        if grammar.get("schema_version") != GRAMMAR_SCHEMA:
            failures.append({"check": "schema", "actual": grammar.get("schema_version")})
        core = {key: value for key, value in grammar.items() if key not in {"grammar_id", "grammar_input_sha256"}}
        digest = hashlib.sha256((json.dumps(core, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()).hexdigest()
        if grammar.get("grammar_input_sha256") != digest or grammar.get("grammar_id") != f"articulation-{digest[:20]}":
            failures.append({"check": "grammar_identity_digest"})

        drivers = sorted(
            name for name, joint in model["joints"].items()
            if joint["type"] != "fixed" and joint["mimic"] is None
        )
        if sorted(grammar.get("independent_variables", {})) != drivers:
            failures.append({"check": "independent_driver_set", "expected": drivers})
        for driver in drivers:
            expected = driver_contract(model, driver)
            actual = grammar["independent_variables"][driver]
            if actual["feasible_domain"] != {key: expected[key] for key in ("minimum", "maximum", "minimum_unbounded", "maximum_unbounded", "constraints")}:
                failures.append({"check": f"driver.{driver}.domain"})
            if actual["physical_joints_driven"] != expected["physical"]:
                failures.append({"check": f"driver.{driver}.physical_joints"})
        for name, joint in sorted(model["joints"].items()):
            rule = grammar["joint_position_rules"][name]
            if joint["type"] == "fixed":
                if rule != {"type": "constant", "value": 0.0, "unit": None}:
                    failures.append({"check": f"joint.{name}.fixed_rule"})
            else:
                driver, multiplier, offset, chain = affine(model, name)
                expected_type = "independent_variable" if name == driver else "affine_driver_dependency"
                if (
                    rule["type"] != expected_type
                    or rule["driver_joint"] != driver
                    or abs(rule["multiplier"] - multiplier) > tolerance
                    or abs(rule["offset"] - offset) > tolerance
                    or rule["mimic_chain_from_physical_joint_to_driver"] != chain
                ):
                    failures.append({"check": f"joint.{name}.position_rule"})
            operator = grammar["joint_operators"][name]
            if operator["parent_link"] != joint["parent"] or operator["child_link"] != joint["child"]:
                failures.append({"check": f"joint.{name}.edge"})
            if max_error(matrix_from(operator["constant_parent_from_pre_motion"]), joint["origin"]) > tolerance:
                failures.append({"check": f"joint.{name}.constant_transform"})
            expected_motion = "identity" if joint["type"] == "fixed" else ("translation_along_axis" if joint["type"] == "prismatic" else "rotation_about_axis")
            if operator["motion_operator"]["type"] != expected_motion:
                failures.append({"check": f"joint.{name}.motion_type"})

        for frame, semantic in sorted(model["frames"].items()):
            derivation = grammar["frame_derivations"].get(frame)
            if derivation is None:
                failures.append({"check": f"frame.{frame}.missing_derivation"})
                continue
            if semantic["type"] == "link":
                attachment_link, terminal = frame, identity()
            elif semantic["type"] == "joint_pre_motion":
                owner = model["joints"][semantic["owner"]]
                attachment_link, terminal = owner["parent"], owner["origin"]
            else:
                attachment_link, terminal = semantic["owner"], model["attachments"][frame]
            expected_refs = [f"joint_operator/{name}" for name in root_path(model, attachment_link)]
            if derivation["ordered_operator_refs"] != expected_refs:
                failures.append({"check": f"frame.{frame}.operator_path"})
            if max_error(matrix_from(derivation["terminal_constant_attachment"]), terminal) > tolerance:
                failures.append({"check": f"frame.{frame}.terminal"})

        matrix_max = 0.0
        frame_checks = 0
        probes = feasible_probes(model, pose_count, rng)
        for probe_index, probe in enumerate(probes):
            pose_path = root / f"pose-{probe_index}.json"
            pose_path.write_text(json.dumps({"pose_name": f"oracle/{case_id}/{probe_index}", "joints": probe}))
            code, stdout, stderr = run_cli(script, ["evaluate-articulation", str(grammar_path), "--pose", str(pose_path)])
            if code != 0:
                failures.append({"check": f"pose.{probe_index}.evaluate", "message": stderr})
                continue
            evaluated = json.loads(stdout)
            expected_frames, expected_positions = world_frames(model, probe)
            actual_positions = evaluated["pose"]["resolved_physical_joint_positions"]
            if (
                set(actual_positions) != set(expected_positions)
                or any(abs(float(actual_positions[name]) - float(expected_positions[name])) > tolerance for name in expected_positions)
            ):
                failures.append({"check": f"pose.{probe_index}.resolved_positions"})
            if set(evaluated["frames"]) != set(expected_frames):
                failures.append({"check": f"pose.{probe_index}.frame_set"})
                continue
            for frame, expected_matrix in expected_frames.items():
                error = max_error(matrix_from(evaluated["frames"][frame]["root_from_frame"]), expected_matrix)
                matrix_max = max(matrix_max, error)
                frame_checks += 1
                if error > tolerance:
                    failures.append({"check": f"pose.{probe_index}.frame.{frame}", "error": error})
    return {
        "case_id": case_id,
        "robot_name": model["name"],
        "link_count": len(model["links"]),
        "joint_count": len(model["joints"]),
        "driver_count": len(drivers),
        "mimic_count": sum(joint["mimic"] is not None for joint in model["joints"].values()),
        "pose_count": pose_count,
        "frame_checks": frame_checks,
        "max_error": clean(matrix_max),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("robot_spatial.py"))
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--poses-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=731942)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--real-urdf", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cases < 0 or args.poses_per_case <= 0 or args.tolerance <= 0.0:
        raise SystemExit("cases must be non-negative; poses and tolerance must be positive")
    rng = random.Random(args.seed)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        generated = Path(temp_dir)
        for index in range(args.cases):
            path = generated / f"case-{index:04d}.urdf"
            path.write_text(random_urdf(index, rng))
            results.append(check_case(args.script, path, f"random-{index:04d}", args.poses_per_case, rng, args.tolerance))
    for index, path in enumerate(args.real_urdf):
        results.append(check_case(args.script, path.resolve(), f"real-{index:02d}-{path.stem}", args.poses_per_case, rng, args.tolerance))
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
        "oracle_implementation": "independent ElementTree parsing, mimic/domain algebra, homogeneous transforms, FK, and frame derivation",
        "coverage": {
            "random_case_count": args.cases,
            "real_urdf_count": len(args.real_urdf),
            "total_case_count": len(results),
            "pose_count": sum(result["pose_count"] for result in results),
            "driver_count": sum(result["driver_count"] for result in results),
            "mimic_joint_count": sum(result["mimic_count"] for result in results),
            "all_frame_evaluation_count": sum(result["frame_checks"] for result in results),
            "positive_and_negative_nested_mimic": args.cases > 0,
            "fixed_revolute_continuous_prismatic": args.cases > 0,
            "link_joint_pre_motion_visual_collision_inertial_frames": args.cases > 0,
        },
        "tolerance": args.tolerance,
        "maximum_matrix_absolute_error": max((result["max_error"] or 0.0 for result in results), default=0.0),
        "failure_count": len(failures),
        "failures": failures,
        "cases": results,
        "epistemic_scope": (
            "independent agreement for enumerated supported URDF tree laws, limits/mimic equations, frame derivations, "
            "and unseen-pose FK; not closed-loop constraints, dynamics, trajectories, swept volume, control/hardware, "
            "calibration, observation truth, or safety"
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
