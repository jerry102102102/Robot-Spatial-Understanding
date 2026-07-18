from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_DIR / "robot_spatial.py"
FIXTURES = Path(__file__).parent / "fixtures"
TWO_DOF = FIXTURES / "two_dof.urdf"
MIMIC = FIXTURES / "mimic_branch.urdf"
FIXED_TREE = FIXTURES / "fixed_tree.urdf"


class ArticulationGrammarTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
        )

    def generate(self, urdf: Path, path: Path) -> dict:
        result = self.run_cli("articulation-grammar", urdf, "--out", path)
        response = json.loads(result.stdout)
        self.assertEqual(response["status"], "generated")
        return json.loads(path.read_text())

    def test_standalone_unseen_pose_evaluation_and_all_frame_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar_path = root / "grammar.json"
            grammar = self.generate(TWO_DOF, grammar_path)
            self.assertEqual(grammar["schema_version"], "robot-spatial-articulation-grammar.v1")
            self.assertEqual(grammar["law_identity"]["schema_version"], "robot-spatial-canonical-kinematic-law.v1")
            self.assertTrue(grammar["law_identity"]["source_binding_excluded"])
            self.assertEqual(sorted(grammar["independent_variables"]), ["shoulder", "slide"])
            self.assertEqual(grammar["coverage"]["frame_derivation_count"], 12)
            self.assertTrue(grammar["coverage"]["all_supported_frames_have_derivations"])
            tool = grammar["frame_derivations"]["tool0"]
            self.assertEqual(
                tool["ordered_operator_refs"],
                ["joint_operator/shoulder", "joint_operator/slide", "joint_operator/tool_mount"],
            )
            self.assertEqual(tool["independent_driver_dependencies"], ["shoulder", "slide"])
            self.assertNotIn(
                "joint_operator/shoulder",
                grammar["frame_derivations"]["joint/shoulder"]["ordered_operator_refs"],
            )
            self.assertEqual(
                grammar["joint_operators"]["shoulder"]["post_motion_from_child_zero"]["matrix_4x4_rowmajor"],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            )

            unseen = root / "unseen.json"
            unseen.write_text(json.dumps({"pose_name": "unseen", "joints": {"shoulder": 0.37, "slide": 0.23}}))
            evaluated = json.loads(self.run_cli(
                "evaluate-articulation",
                grammar_path,
                "--pose",
                unseen,
                "--target",
                "tool0",
            ).stdout)
            direct = json.loads(self.run_cli(
                "transform",
                TWO_DOF,
                "--pose",
                unseen,
                "--from",
                "base_link",
                "--to",
                "tool0",
            ).stdout)
            self.assertEqual(
                evaluated["frames"]["tool0"]["root_from_frame"]["matrix_4x4_rowmajor"],
                direct["matrix_4x4_rowmajor"],
            )
            self.assertEqual(evaluated["query_evidence"]["method"], "standalone_typed_articulation_ast_execution")
            self.assertNotIn("urdf", evaluated["query_evidence"])

            verified = json.loads(self.run_cli(
                "verify-articulation-grammar",
                TWO_DOF,
                "--grammar",
                grammar_path,
            ).stdout)
            self.assertEqual(verified["status"], "passed")
            self.assertEqual(verified["probe_count"], 3)
            self.assertEqual(verified["verified_frame_evaluation_count"], 36)
            self.assertLessEqual(verified["maximum_matrix_absolute_error"], 1e-12)

    def test_negative_mimic_equation_domain_and_dependency_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar_path = root / "grammar.json"
            grammar = self.generate(MIMIC, grammar_path)
            variable = grammar["independent_variables"]["driver"]
            self.assertEqual(variable["feasible_domain"]["minimum"], -0.2)
            self.assertEqual(variable["feasible_domain"]["maximum"], 0.6)
            self.assertEqual(variable["physical_joints_driven"], ["driver", "follower"])
            follower = grammar["joint_position_rules"]["follower"]
            self.assertEqual(follower["type"], "affine_driver_dependency")
            self.assertEqual(follower["driver_joint"], "driver")
            self.assertEqual(follower["multiplier"], -0.5)
            self.assertEqual(follower["offset"], 0.1)
            self.assertEqual(follower["mimic_chain_from_physical_joint_to_driver"], ["follower", "driver"])
            self.assertEqual(
                grammar["frame_derivations"]["follower_link"]["independent_driver_dependencies"],
                ["driver"],
            )
            pose = root / "pose.json"
            pose.write_text(json.dumps({"pose_name": "fresh", "joints": {"driver": 0.4}}))
            result = json.loads(self.run_cli("evaluate-articulation", grammar_path, "--pose", pose).stdout)
            self.assertAlmostEqual(result["pose"]["resolved_physical_joint_positions"]["follower"], -0.1)
            self.assertEqual(
                result["frames"]["joint/follower"]["operator_trace"],
                [],
                "the follower's own pre-motion frame must exclude follower motion",
            )
            self.assertEqual(
                result["frames"]["follower_link"]["operator_trace"][0]["operator_ref"],
                "joint_operator/follower",
            )

    def test_standalone_evaluator_rejects_domain_unknown_and_dependent_conflicts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar_path = root / "grammar.json"
            self.generate(MIMIC, grammar_path)
            cases = [
                ({"driver": 0.7}, "above feasible maximum"),
                ({"driver": 0.0, "ghost": 1.0}, "absent from the grammar"),
                ({"driver": 0.0, "follower": 0.0}, "disagrees with grammar value"),
            ]
            for index, (joints, message) in enumerate(cases):
                pose = root / f"invalid-{index}.json"
                pose.write_text(json.dumps({"pose_name": f"invalid-{index}", "joints": joints}))
                result = self.run_cli("evaluate-articulation", grammar_path, "--pose", pose, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, json.loads(result.stderr)["error"])

    def test_verifier_rejects_operator_derivation_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grammar_path = root / "grammar.json"
            baseline = self.generate(TWO_DOF, grammar_path)
            mutations = []

            axis = json.loads(json.dumps(baseline))
            axis["joint_operators"]["shoulder"]["motion_operator"]["axis_xyz_in_pre_motion_frame"] = [1.0, 0.0, 0.0]
            mutations.append((axis, "grammar.regeneration"))

            order = json.loads(json.dumps(baseline))
            order["frame_derivations"]["tool0"]["ordered_operator_refs"] = list(reversed(
                order["frame_derivations"]["tool0"]["ordered_operator_refs"]
            ))
            mutations.append((order, "grammar.execution"))

            source = json.loads(json.dumps(baseline))
            source["source_binding"]["urdf_semantic_sha256"] = "0" * 64
            mutations.append((source, "grammar.source_binding"))

            for index, (mutated, expected_check) in enumerate(mutations):
                path = root / f"tampered-{index}.json"
                path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
                result = self.run_cli(
                    "verify-articulation-grammar",
                    TWO_DOF,
                    "--grammar",
                    path,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"], "failed")
                self.assertTrue(any(issue["check"] == expected_check for issue in report["issues"]), report)

    def test_grammar_is_byte_deterministic_and_export_is_context_grounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first, second = root / "first.json", root / "second.json"
            self.generate(TWO_DOF, first)
            self.generate(TWO_DOF, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), hashlib.sha256(second.read_bytes()).hexdigest())

            context = root / "context"
            key = root / "private" / "key.jsonl"
            response = json.loads(self.run_cli(
                "export",
                TWO_DOF,
                "--pose",
                FIXTURES / "bent_pose.json",
                "--workspace-samples",
                "0",
                "--generate-evaluation",
                "--evaluation-key-out",
                key,
                "--out",
                context,
            ).stdout)
            grammar_path = context / "articulation-grammar.json"
            self.assertEqual(Path(response["articulation_grammar"]).resolve(), grammar_path.resolve())
            model = json.loads((context / "model.json").read_text())
            artifact = model["artifacts"]["articulation_grammar"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(grammar_path.read_bytes()).hexdigest())
            agent = json.loads((context / "agent-context.json").read_text())
            counts = agent["statistics"]["entity_type_counts"]
            self.assertEqual(counts["articulation_grammar"], 1)
            self.assertEqual(counts["articulation_variable"], 2)
            self.assertEqual(counts["articulation_operator"], 3)
            self.assertEqual(counts["articulation_derivation"], 12)
            self.assertEqual(agent["artifacts"]["articulation_grammar"]["sha256"], artifact["sha256"])
            facts = [json.loads(line) for line in (context / "facts.jsonl").read_text().splitlines()]
            predicates = {record["predicate"] for record in facts}
            self.assertIn("binds_pose_independent_joint_laws_and_frame_compositions", predicates)
            self.assertIn("has_typed_parameterized_joint_operator", predicates)
            self.assertIn("has_ordered_root_from_frame_composition", predicates)
            self.assertIn("separates_source_binding_from_canonical_kinematic_law_identity", predicates)
            questions = [json.loads(line) for line in (context / "evaluation" / "questions.jsonl").read_text().splitlines()]
            grammar_questions = [
                record for record in questions
                if record["capability"] == "articulation_grammar_understanding"
            ]
            self.assertEqual(len(grammar_questions), 6)

    def test_fixed_tree_generates_six_grammar_questions_without_variables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = root / "context"
            key = root / "private" / "key.jsonl"
            self.run_cli(
                "export",
                FIXED_TREE,
                "--workspace-samples",
                "0",
                "--generate-evaluation",
                "--evaluation-key-out",
                key,
                "--out",
                context,
            )
            grammar = json.loads((context / "articulation-grammar.json").read_text())
            self.assertEqual(grammar["independent_variables"], {})
            self.assertEqual(grammar["coverage"]["fixed_joint_count"], 1)
            questions = [
                json.loads(line)
                for line in (context / "evaluation" / "questions.jsonl").read_text().splitlines()
            ]
            grammar_questions = [
                record for record in questions
                if record["capability"] == "articulation_grammar_understanding"
            ]
            self.assertEqual(len(grammar_questions), 6)
            tasks = {record["task"] for record in grammar_questions}
            self.assertIn("report_zero_independent_variable_contract", tasks)
            self.assertIn("report_joint_position_equation", tasks)
            self.assertIn("explain_typed_joint_operator", tasks)


if __name__ == "__main__":
    unittest.main()
