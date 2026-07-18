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
URDF = FIXTURES / "two_dof.urdf"
POSE = FIXTURES / "bent_pose.json"
PACKAGE_MAP = FIXTURES / "package_map.json"


class CounterfactualMotionAtlasTests(unittest.TestCase):
    def generate(self, output: Path, *, urdf: Path = URDF, pose: Path = POSE, inspect_meshes: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "motion-atlas",
            str(urdf),
            "--pose",
            str(pose),
            "--out",
            str(output),
        ]
        if inspect_meshes:
            command.extend(["--inspect-meshes", "--package-map", str(PACKAGE_MAP)])
        return subprocess.run(command, check=True, capture_output=True, text=True)

    def verify_command(self, atlas: Path, *, urdf: Path = URDF, pose: Path = POSE, inspect_meshes: bool = True) -> list[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "verify-motion-atlas",
            str(urdf),
            "--pose",
            str(pose),
            "--atlas",
            str(atlas),
        ]
        if inspect_meshes:
            command.extend(["--inspect-meshes", "--package-map", str(PACKAGE_MAP)])
        return command

    def test_export_integrates_motion_context_facts_evaluation_and_verifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "context"
            key = root / "private" / "key.jsonl"
            completed = subprocess.run([
                sys.executable,
                str(SCRIPT),
                "export",
                str(URDF),
                "--pose",
                str(POSE),
                "--inspect-meshes",
                "--package-map",
                str(PACKAGE_MAP),
                "--motion-atlas",
                "--workspace-samples",
                "0",
                "--generate-evaluation",
                "--evaluation-key-out",
                str(key),
                "--out",
                str(output),
            ], check=True, capture_output=True, text=True)
            response = json.loads(completed.stdout)
            atlas_path = output / "motion-atlas" / "manifest.json"
            self.assertEqual(Path(response["motion_atlas"]).resolve(), atlas_path.resolve())
            atlas = json.loads(atlas_path.read_text())
            self.assertEqual(atlas["schema_version"], "robot-spatial-motion-atlas.v1")
            self.assertEqual(atlas["coverage"]["independent_drivers"], ["shoulder", "slide"])
            self.assertEqual(atlas["coverage"]["available_signed_endpoint_count"], 4)
            shoulder = atlas["drivers"]["shoulder"]
            self.assertFalse(shoulder["endpoints"]["plus"]["causality_check"]["pre_motion_frame_changed"])
            self.assertEqual(shoulder["endpoints"]["plus"]["causality_check"]["unexpected_changed_frames"], [])
            self.assertFalse(shoulder["endpoints"]["plus"]["link_frame_deltas"]["base_link"]["frame_changed"])
            self.assertTrue(shoulder["endpoints"]["plus"]["link_frame_deltas"]["tool0"]["frame_changed"])
            view = shoulder["views"]["isometric"]
            self.assertEqual(view["screen"]["fit_scope"], "baseline_and_all_available_signed_endpoints_for_this_driver")
            self.assertTrue(any(record["frame_name"] == "tool0" for record in view["motion_vectors"]))
            for driver in atlas["drivers"].values():
                for motion_view in driver["views"].values():
                    svg_path = atlas_path.parent / motion_view["artifact"]["path"]
                    self.assertEqual(hashlib.sha256(svg_path.read_bytes()).hexdigest(), motion_view["artifact"]["sha256"])
                    svg = svg_path.read_text()
                    for entity_id in motion_view["expected_svg_entity_ids"]:
                        self.assertIn(f'data-entity-id="{entity_id}"', svg)
            context = json.loads((output / "agent-context.json").read_text())
            self.assertEqual(context["statistics"]["entity_type_counts"]["motion_atlas"], 1)
            self.assertEqual(context["statistics"]["entity_type_counts"]["motion_driver"], 2)
            self.assertEqual(context["statistics"]["entity_type_counts"]["motion_view"], 8)
            facts = [json.loads(line) for line in (output / "facts.jsonl").read_text().splitlines()]
            predicates = {fact["predicate"] for fact in facts}
            self.assertIn("binds_counterfactual_joint_causes_to_finite_endpoint_effects", predicates)
            self.assertIn("has_signed_finite_counterfactual_endpoint_effect", predicates)
            self.assertIn("has_shared_screen_counterfactual_projection_contract", predicates)
            questions = [json.loads(line) for line in (output / "evaluation" / "questions.jsonl").read_text().splitlines()]
            motion_questions = [record for record in questions if record["capability"] == "counterfactual_motion_understanding"]
            self.assertEqual(len(motion_questions), 6)
            verification = subprocess.run(self.verify_command(atlas_path), check=True, capture_output=True, text=True)
            report = json.loads(verification.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["verified_driver_count"], 2)
            self.assertEqual(report["verified_view_count"], 8)

    def test_mimic_constraints_make_limit_endpoint_one_sided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose = root / "pose.json"
            pose.write_text(json.dumps({"pose_name": "mimic_lower", "joints": {"driver": -0.2}}))
            output = root / "atlas"
            self.generate(output, urdf=FIXTURES / "mimic_branch.urdf", pose=pose, inspect_meshes=False)
            atlas = json.loads((output / "manifest.json").read_text())
            driver = atlas["drivers"]["driver"]
            self.assertEqual(driver["feasible_interval"]["minimum"], -0.2)
            self.assertEqual(driver["feasible_interval"]["maximum"], 0.6)
            self.assertEqual(driver["physical_joints_driven"], ["driver", "follower"])
            self.assertEqual(driver["endpoints"]["minus"]["status"], "unavailable_at_feasible_limit")
            self.assertEqual(driver["endpoints"]["minus"]["applied_delta"], 0.0)
            self.assertEqual(driver["endpoints"]["plus"]["status"], "applied_nominal_step")
            self.assertAlmostEqual(driver["endpoints"]["plus"]["physical_joint_positions"]["driver"], -0.1)
            self.assertAlmostEqual(driver["endpoints"]["plus"]["physical_joint_positions"]["follower"], 0.15)
            self.assertEqual(atlas["coverage"]["available_signed_endpoint_count"], 1)
            result = subprocess.run(
                self.verify_command(output / "manifest.json", urdf=FIXTURES / "mimic_branch.urdf", pose=pose, inspect_meshes=False),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_motion_atlas_is_byte_deterministic_across_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first, second = root / "first", root / "second"
            self.generate(first)
            self.generate(second)
            first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes(), relative)

    def test_verifier_rejects_tampering_missing_ids_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "atlas"
            self.generate(output)
            atlas_path = output / "manifest.json"
            atlas = json.loads(atlas_path.read_text())
            view = atlas["drivers"]["shoulder"]["views"]["front"]
            svg_path = output / view["artifact"]["path"]
            svg_path.write_text(svg_path.read_text() + "<!-- tampered -->\n")
            result = subprocess.run(self.verify_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertTrue(any(issue["check"].endswith("artifact.sha256") for issue in report["issues"]))

            self.generate(output)
            atlas = json.loads(atlas_path.read_text())
            view = atlas["drivers"]["shoulder"]["views"]["front"]
            svg_path = output / view["artifact"]["path"]
            entity_id = view["expected_svg_entity_ids"][0]
            svg_path.write_text(svg_path.read_text().replace(f'data-entity-id="{entity_id}"', 'data-entity-id="removed/entity"', 1))
            view["artifact"]["sha256"] = hashlib.sha256(svg_path.read_bytes()).hexdigest()
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verify_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertTrue(any(issue["check"].endswith("svg_entity_ids") for issue in report["issues"]))

            self.generate(output)
            atlas = json.loads(atlas_path.read_text())
            atlas["source_binding"]["urdf_semantic_sha256"] = "0" * 64
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verify_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(any(issue["check"] == "manifest.source_binding" for issue in json.loads(result.stdout)["issues"]))

            self.generate(output)
            atlas = json.loads(atlas_path.read_text())
            atlas["drivers"]["shoulder"]["views"]["front"]["artifact"]["path"] = "../../outside.svg"
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verify_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(any(issue["check"].endswith("artifact.path") for issue in json.loads(result.stdout)["issues"]))


if __name__ == "__main__":
    unittest.main()
