from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_DIR / "robot_spatial.py"
FIXTURES = Path(__file__).parent / "fixtures"


class SpatialConstraintTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
        )

    def write_pose(self, path: Path, joints: dict[str, float], name: str = "test") -> None:
        path.write_text(json.dumps({"pose_name": name, "joints": joints}) + "\n")

    def build_fourbar(
        self,
        root: Path,
        constraints: list[dict] | None = None,
        attachments: list[dict] | None = None,
    ) -> tuple[Path, Path, Path]:
        grammar = root / "grammar.json"
        self.run_cli("articulation-grammar", FIXTURES / "fourbar_tree.urdf", "--out", grammar)
        if attachments is None:
            attachments = [
                {
                    "attachment_id": "coupler_tip",
                    "parent_frame": "coupler",
                    "semantic_role": "constraint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [2.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "attachment_id": "rocker_tip",
                    "parent_frame": "rocker",
                    "semantic_role": "constraint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [1.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ]
        if constraints is None:
            constraints = [{
                "constraint_id": "fourbar_closure",
                "type": "kinematic_pair",
                "role": "loop_closure",
                "frame_a": "attachment/coupler_tip",
                "frame_b": "attachment/rocker_tip",
                "joint_type": "revolute",
                "axis_xyz_in_a": [0.0, 0.0, 1.0],
                "axis_xyz_in_b": [0.0, 0.0, 1.0],
                "tolerances": {"translation_m": 1e-8, "rotation_rad": 1e-8},
            }]
        spec = root / "constraints.json"
        spec.write_text(json.dumps({
            "schema_version": "robot-spatial-constraint-spec.v1",
            "constraint_set_id": "fixture-fourbar",
            "grammar_sha256": hashlib.sha256(grammar.read_bytes()).hexdigest(),
            "attachments": attachments,
            "constraints": constraints,
        }, indent=2, sort_keys=True) + "\n")
        graph = root / "constraint-graph.json"
        result = json.loads(self.run_cli("constraint-graph", grammar, spec, "--out", graph).stdout)
        self.assertEqual(result["status"], "generated")
        return grammar, spec, graph

    def test_fourbar_spanning_tree_and_full_mechanism_are_distinct_and_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar, spec, graph_path = self.build_fourbar(root)
            graph = json.loads(graph_path.read_text())
            self.assertTrue(graph["structural_graph"]["tree_is_parameterization_not_complete_mechanism"])
            self.assertEqual(graph["coverage"]["declared_cycle_count"], 1)
            cycle = graph["structural_graph"]["declared_cycle_records"][0]
            self.assertEqual(cycle["closure_constraint_ref"], "constraint/fourbar_closure")
            self.assertEqual(
                cycle["tree_path"]["from_branch_reverse_operator_refs"],
                ["joint_operator/coupler_angle", "joint_operator/crank_angle"],
            )
            self.assertEqual(
                cycle["tree_path"]["to_branch_forward_operator_refs"],
                ["joint_operator/rocker_angle"],
            )

            angle = math.pi / 3.0
            valid_pose = root / "valid.json"
            self.write_pose(valid_pose, {
                "crank_angle": angle,
                "coupler_angle": -angle,
                "rocker_angle": angle,
            }, "fourbar-valid")
            evaluated = json.loads(self.run_cli(
                "evaluate-constraints", graph_path, "--pose", valid_pose
            ).stdout)
            self.assertEqual(evaluated["status"], "satisfied")
            self.assertLessEqual(evaluated["maximum_normalized_abs"], 1.0)
            analysis = evaluated["local_constraint_analysis"]
            self.assertEqual(analysis["tree_independent_variable_count"], 3)
            self.assertEqual(analysis["local_constraint_rank"], 2)
            self.assertEqual(analysis["local_mobility_estimate"], 1)

            invalid_pose = root / "invalid.json"
            self.write_pose(invalid_pose, {
                "crank_angle": 0.7,
                "coupler_angle": 0.0,
                "rocker_angle": 0.2,
            }, "fourbar-invalid")
            failed = self.run_cli(
                "evaluate-constraints", graph_path, "--pose", invalid_pose, check=False
            )
            self.assertEqual(failed.returncode, 1)
            violated = json.loads(failed.stdout)
            self.assertEqual(violated["status"], "violated")
            self.assertGreater(violated["maximum_normalized_abs"], 1.0)

            verified = json.loads(self.run_cli(
                "verify-constraint-graph", grammar, spec, "--graph", graph_path, "--pose", valid_pose
            ).stdout)
            self.assertEqual(verified["status"], "passed")
            self.assertEqual(verified["execution_status_at_verification_pose"], "satisfied")

    def test_fourbar_local_solver_closes_loop_with_one_input_fixed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, _spec, graph = self.build_fourbar(root)
            seed = root / "seed.json"
            input_angle = 0.8
            self.write_pose(seed, {
                "crank_angle": input_angle,
                "coupler_angle": -0.55,
                "rocker_angle": 0.55,
            }, "fourbar-seed")
            solution = json.loads(self.run_cli(
                "solve-constraints",
                graph,
                "--pose",
                seed,
                "--solve-for",
                "coupler_angle",
                "--solve-for",
                "rocker_angle",
            ).stdout)
            self.assertEqual(solution["status"], "converged", solution)
            solved = solution["solved_independent_driver_positions"]
            self.assertAlmostEqual(solved["crank_angle"], input_angle, places=12)
            self.assertAlmostEqual(solved["coupler_angle"], -input_angle, places=6)
            self.assertAlmostEqual(solved["rocker_angle"], input_angle, places=6)
            self.assertEqual(solution["evaluation"]["status"], "satisfied")
            self.assertLessEqual(solution["evaluation"]["maximum_normalized_abs"], 1.0)

    def test_coordinate_coupling_and_distance_constraints_are_typed_and_ranked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            constraints = [{
                "constraint_id": "differential",
                "type": "coordinate_linear",
                "role": "mechanical_coupling",
                "terms": [
                    {"joint": "crank_angle", "coefficient": 1.0},
                    {"joint": "coupler_angle", "coefficient": 2.0},
                ],
                "offset": 0.0,
                "tolerance": 1e-9,
            }]
            _grammar, _spec, graph = self.build_fourbar(root, constraints, [])
            pose = root / "coupled.json"
            self.write_pose(pose, {
                "crank_angle": 0.4,
                "coupler_angle": -0.2,
                "rocker_angle": 0.17,
            })
            result = json.loads(self.run_cli("evaluate-constraints", graph, "--pose", pose).stdout)
            self.assertEqual(result["status"], "satisfied")
            self.assertEqual(result["local_constraint_analysis"]["local_constraint_rank"], 1)
            self.assertEqual(result["local_constraint_analysis"]["local_mobility_estimate"], 2)
            self.assertEqual(result["constraints"][0]["components"][0]["unit"], "joint_coordinate")

            distance_root = root / "distance"
            distance_root.mkdir()
            attachments = [
                {
                    "attachment_id": "crank_tip",
                    "parent_frame": "crank",
                    "semantic_role": "measurement_point",
                    "parent_from_attachment": {
                        "translation_xyz_m": [1.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "attachment_id": "rocker_tip",
                    "parent_frame": "rocker",
                    "semantic_role": "measurement_point",
                    "parent_from_attachment": {
                        "translation_xyz_m": [1.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ]
            distance = [{
                "constraint_id": "rod_length",
                "type": "point_distance",
                "role": "cable_length",
                "frame_a": "attachment/crank_tip",
                "frame_b": "attachment/rocker_tip",
                "distance_m": 2.0,
                "tolerance_m": 1e-8,
            }]
            _grammar, _spec, distance_graph = self.build_fourbar(distance_root, distance, attachments)
            distance_pose = distance_root / "pose.json"
            self.write_pose(distance_pose, {"crank_angle": 0.0, "coupler_angle": 0.3, "rocker_angle": 0.0})
            distance_result = json.loads(self.run_cli(
                "evaluate-constraints", distance_graph, "--pose", distance_pose
            ).stdout)
            self.assertEqual(distance_result["status"], "satisfied")
            self.assertEqual(distance_result["constraints"][0]["type"], "point_distance")

    def test_constraint_graph_is_standalone_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar, spec, graph = self.build_fourbar(root)
            standalone = root / "standalone.json"
            standalone.write_bytes(graph.read_bytes())
            grammar.unlink()
            spec.unlink()
            angle = math.pi / 4.0
            pose = root / "pose.json"
            self.write_pose(pose, {
                "crank_angle": angle,
                "coupler_angle": -angle,
                "rocker_angle": angle,
            })
            result = json.loads(self.run_cli(
                "evaluate-constraints", standalone, "--pose", pose
            ).stdout)
            self.assertEqual(result["status"], "satisfied")

            grammar, spec, graph = self.build_fourbar(root / "tamper")
            tampered = json.loads(graph.read_text())
            tampered["coverage"]["constraint_count"] = 999
            graph.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
            verification = self.run_cli(
                "verify-constraint-graph", grammar, spec, "--graph", graph, check=False
            )
            self.assertEqual(verification.returncode, 1)
            report = json.loads(verification.stdout)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["exact_regeneration_match"])

    def test_export_and_prepare_publish_constraint_graph_context_with_failure_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, spec, _graph = self.build_fourbar(root / "build")

            exported = root / "exported"
            evaluation_key = root / "private" / "constraint-key.jsonl"
            response = json.loads(self.run_cli(
                "export",
                FIXTURES / "fourbar_tree.urdf",
                "--constraint-spec",
                spec,
                "--out",
                exported,
                "--workspace-samples",
                "0",
                "--generate-evaluation",
                "--evaluation-key-out",
                evaluation_key,
            ).stdout)
            self.assertEqual(response["status"], "exported")
            model = json.loads((exported / "model.json").read_text())
            graph_record = model["artifacts"]["constraint_graph"]
            self.assertEqual(graph_record["evaluation"]["status"], "satisfied")
            self.assertTrue(graph_record["structural_graph"]["tree_is_parameterization_not_complete_mechanism"])
            context = json.loads((exported / "agent-context.json").read_text())
            graph_id = graph_record["constraint_graph_id"]
            entities = context["artifacts"]["entity_index"]
            entity_index = json.loads((exported / entities["path"]).read_text())
            self.assertIn(f"constraint_graph/{graph_id}", entity_index["by_entity_id"])
            self.assertIn(f"constraint/{graph_id}/fourbar_closure", entity_index["by_entity_id"])
            facts = [json.loads(line) for line in (exported / "facts.jsonl").read_text().splitlines()]
            predicates = {fact["predicate"] for fact in facts}
            self.assertIn("tree_is_parameterization_not_complete_mechanism", predicates)
            self.assertIn("has_pose_conditioned_numerical_local_mobility_estimate", predicates)
            self.assertIn("constraint_graph", context["artifacts"])
            evaluation_manifest = json.loads((exported / "evaluation" / "manifest.json").read_text())
            self.assertGreaterEqual(
                evaluation_manifest["capability_counts"]["supplemental_mechanism_understanding"],
                4,
            )
            self.assertTrue(evaluation_key.is_file())

            prepared = root / "prepared"
            prepared_response = json.loads(self.run_cli(
                "prepare",
                FIXTURES / "fourbar_tree.urdf",
                "--constraint-spec",
                spec,
                "--out",
                prepared,
                "--workspace-samples",
                "0",
            ).stdout)
            self.assertEqual(prepared_response["status"], "prepared")
            self.assertIn("constraint_graph", prepared_response["artifacts"])
            source_manifest = json.loads((prepared / "source-manifest.json").read_text())
            self.assertIsNotNone(source_manifest["supplemental_constraints"])

            invalid_pose = root / "invalid-export-pose.json"
            self.write_pose(invalid_pose, {
                "crank_angle": 0.7,
                "coupler_angle": 0.0,
                "rocker_angle": 0.2,
            }, "invalid-export")
            invalid_export = root / "invalid-export"
            failed = self.run_cli(
                "export",
                FIXTURES / "fourbar_tree.urdf",
                "--constraint-spec",
                spec,
                "--pose",
                invalid_pose,
                "--out",
                invalid_export,
                "--workspace-samples",
                "0",
                check=False,
            )
            self.assertEqual(failed.returncode, 1)
            failed_response = json.loads(failed.stdout)
            self.assertEqual(failed_response["status"], "exported_with_violated_constraints")
            self.assertTrue((invalid_export / "constraint-evaluation.json").is_file())
            self.assertEqual(
                json.loads((invalid_export / "constraint-evaluation.json").read_text())["status"],
                "violated",
            )

    def test_fixed_and_prismatic_pair_residual_manifolds_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attachments = [
                {
                    "attachment_id": "left_anchor",
                    "parent_frame": "crank",
                    "semantic_role": "joint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [1.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "attachment_id": "right_anchor",
                    "parent_frame": "rocker",
                    "semantic_role": "joint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [-1.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ]
            zero = root / "zero.json"
            self.write_pose(zero, {
                "crank_angle": 0.0,
                "coupler_angle": 0.0,
                "rocker_angle": 0.0,
            })
            for pair_type, expected_components in (("fixed", 6), ("prismatic", 5)):
                with self.subTest(pair_type=pair_type):
                    pair_root = root / pair_type
                    pair_root.mkdir()
                    constraint = {
                        "constraint_id": f"{pair_type}_closure",
                        "type": "kinematic_pair",
                        "role": "assembly_constraint",
                        "frame_a": "attachment/left_anchor",
                        "frame_b": "attachment/right_anchor",
                        "joint_type": pair_type,
                        "tolerances": {"translation_m": 1e-9, "rotation_rad": 1e-9},
                    }
                    if pair_type == "prismatic":
                        constraint["axis_xyz_in_a"] = [1.0, 0.0, 0.0]
                        constraint["axis_xyz_in_b"] = [1.0, 0.0, 0.0]
                    _grammar, _spec, graph = self.build_fourbar(
                        pair_root, [constraint], attachments
                    )
                    result = json.loads(self.run_cli(
                        "evaluate-constraints", graph, "--pose", zero
                    ).stdout)
                    self.assertEqual(result["status"], "satisfied")
                    self.assertEqual(result["residual_component_count"], expected_components)
                    perturbed = pair_root / "perturbed.json"
                    self.write_pose(perturbed, {
                        "crank_angle": 0.2,
                        "coupler_angle": 0.0,
                        "rocker_angle": -0.1,
                    })
                    violation = self.run_cli(
                        "evaluate-constraints", graph, "--pose", perturbed, check=False
                    )
                    self.assertEqual(violation.returncode, 1)
                    self.assertEqual(json.loads(violation.stdout)["status"], "violated")

    def test_invalid_supplemental_contracts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar = root / "grammar.json"
            self.run_cli("articulation-grammar", FIXTURES / "fourbar_tree.urdf", "--out", grammar)
            base = {
                "schema_version": "robot-spatial-constraint-spec.v1",
                "constraint_set_id": "invalid",
                "grammar_sha256": hashlib.sha256(grammar.read_bytes()).hexdigest(),
                "attachments": [],
                "constraints": [{
                    "constraint_id": "coupling",
                    "type": "coordinate_linear",
                    "role": "mechanical_coupling",
                    "terms": [{"joint": "crank_angle", "coefficient": 1.0}],
                    "offset": 0.0,
                    "tolerance": 1e-9,
                }],
            }
            cases = []
            wrong_digest = json.loads(json.dumps(base))
            wrong_digest["grammar_sha256"] = "0" * 64
            cases.append((wrong_digest, "does not bind"))
            unknown_joint = json.loads(json.dumps(base))
            unknown_joint["constraints"][0]["terms"][0]["joint"] = "missing"
            cases.append((unknown_joint, "absent from grammar"))
            zero_coefficient = json.loads(json.dumps(base))
            zero_coefficient["constraints"][0]["terms"][0]["coefficient"] = 0.0
            cases.append((zero_coefficient, "non-zero"))
            for index, (spec_value, message) in enumerate(cases):
                with self.subTest(index=index):
                    spec = root / f"invalid-{index}.json"
                    graph = root / f"invalid-{index}-graph.json"
                    spec.write_text(json.dumps(spec_value) + "\n")
                    result = self.run_cli(
                        "constraint-graph", grammar, spec, "--out", graph, check=False
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, json.loads(result.stderr)["error"])


if __name__ == "__main__":
    unittest.main()
