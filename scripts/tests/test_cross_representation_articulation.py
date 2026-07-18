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


class CrossRepresentationArticulationTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
        )

    def compile(self, source: Path, output: Path) -> dict:
        result = json.loads(self.run_cli("articulation-grammar", source, "--out", output).stdout)
        self.assertEqual(result["status"], "generated")
        return json.loads(output.read_text())

    def correspondence(
        self,
        path: Path,
        reference_path: Path,
        candidate_path: Path,
        links: dict[str, str],
        joints: dict[str, str],
        frames: dict[str, str] | None = None,
    ) -> None:
        path.write_text(json.dumps({
            "schema_version": "robot-spatial-articulation-correspondence.v1",
            "reference_grammar_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "candidate_grammar_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "candidate_to_reference": {
                "links": links,
                "joints": joints,
                "frames": frames or {},
            },
        }, indent=2, sort_keys=True) + "\n")

    def test_differently_named_urdf_sdf_mjcf_compile_to_one_mapped_law(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            urdf_path, sdf_path, mjcf_path = root / "urdf.json", root / "sdf.json", root / "mjcf.json"
            urdf = self.compile(FIXTURES / "cross_law.urdf", urdf_path)
            sdf = self.compile(FIXTURES / "cross_law.sdf", sdf_path)
            mjcf = self.compile(FIXTURES / "cross_law.xml", mjcf_path)
            self.assertEqual(
                [urdf["source_binding"]["source_format"], sdf["source_binding"]["source_format"], mjcf["source_binding"]["source_format"]],
                ["urdf", "sdf", "mjcf"],
            )
            self.assertEqual(len({
                urdf["law_identity"]["canonical_law_sha256"],
                sdf["law_identity"]["canonical_law_sha256"],
                mjcf["law_identity"]["canonical_law_sha256"],
            }), 3, "name-sensitive law identity must not silently equate different identifiers")

            candidates = [
                (
                    sdf_path,
                    {"chassis": "base_link", "segment": "arm_link", "effector": "tool_link"},
                    {"spin": "yaw_joint", "slide": "extension_joint"},
                ),
                (
                    mjcf_path,
                    {"root_body": "base_link", "middle_body": "arm_link", "end_body": "tool_link"},
                    {"hinge_a": "yaw_joint", "slide_b": "extension_joint"},
                ),
            ]
            for index, (candidate_path, link_map, joint_map) in enumerate(candidates):
                mapping_path = root / f"mapping-{index}.json"
                self.correspondence(mapping_path, urdf_path, candidate_path, link_map, joint_map)
                comparison = json.loads(self.run_cli(
                    "compare-articulation-grammars",
                    urdf_path,
                    candidate_path,
                    "--correspondence",
                    mapping_path,
                ).stdout)
                self.assertEqual(comparison["status"], "equivalent", comparison)
                self.assertEqual(
                    comparison["comparison_mode"],
                    "explicit_digest_bound_typed_identifier_correspondence",
                )
                self.assertTrue(comparison["canonical_comparison"]["exact_projection_match"])
                self.assertEqual(comparison["execution_crosscheck"]["probe_count"], 3)
                self.assertEqual(comparison["execution_crosscheck"]["all_frame_evaluation_count"], 15)
                self.assertLessEqual(comparison["execution_crosscheck"]["maximum_matrix_absolute_error"], 1e-12)

            for source, grammar_path, expected_format in (
                (FIXTURES / "cross_law.urdf", urdf_path, "urdf"),
                (FIXTURES / "cross_law.sdf", sdf_path, "sdf"),
                (FIXTURES / "cross_law.xml", mjcf_path, "mjcf"),
            ):
                verification = json.loads(self.run_cli(
                    "verify-articulation-grammar",
                    source,
                    "--grammar",
                    grammar_path,
                ).stdout)
                self.assertEqual(verification["status"], "passed")
                self.assertEqual(verification["query_evidence"]["source_format"], expected_format)

    def test_pre_motion_post_motion_anchor_is_executable_and_exact_across_sdf_mjcf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdf_source = root / "anchor.sdf"
            sdf_source.write_text(
                '<sdf version="1.11"><model name="anchor">'
                '<link name="root"/><link name="child"><pose>1 0 0 0 0 0</pose></link>'
                '<joint name="hinge" type="continuous"><parent>root</parent><child>child</child>'
                '<pose>0.2 0 0 0 0 0</pose><axis><xyz>0 0 1</xyz></axis></joint>'
                '</model></sdf>\n'
            )
            mjcf_source = root / "anchor.xml"
            mjcf_source.write_text(
                '<mujoco model="anchor"><compiler angle="radian"/><worldbody><body name="root">'
                '<body name="child" pos="1 0 0"><joint name="hinge" type="hinge" pos="0.2 0 0" axis="0 0 1"/>'
                '</body></body></worldbody></mujoco>\n'
            )
            sdf_grammar_path, mjcf_grammar_path = root / "sdf.json", root / "mjcf.json"
            sdf_grammar = self.compile(sdf_source, sdf_grammar_path)
            mjcf_grammar = self.compile(mjcf_source, mjcf_grammar_path)
            self.assertEqual(
                sdf_grammar["law_identity"]["canonical_law_sha256"],
                mjcf_grammar["law_identity"]["canonical_law_sha256"],
            )
            operator = sdf_grammar["joint_operators"]["hinge"]
            self.assertEqual(operator["constant_parent_from_pre_motion"]["translation_xyz_m"], [1.2, 0.0, 0.0])
            self.assertEqual(operator["post_motion_from_child_zero"]["translation_xyz_m"], [-0.2, 0.0, 0.0])
            pose = root / "pose.json"
            pose.write_text(json.dumps({"pose_name": "quarter-turn", "joints": {"hinge": math.pi / 2}}))
            evaluation = json.loads(self.run_cli(
                "evaluate-articulation",
                sdf_grammar_path,
                "--pose",
                pose,
                "--target",
                "child",
            ).stdout)
            translation = evaluation["frames"]["child"]["root_from_frame"]["translation_xyz_m"]
            self.assertAlmostEqual(translation[0], 1.2, places=11)
            self.assertAlmostEqual(translation[1], -0.2, places=11)
            comparison = json.loads(self.run_cli(
                "compare-articulation-grammars",
                sdf_grammar_path,
                mjcf_grammar_path,
            ).stdout)
            self.assertEqual(comparison["status"], "equivalent")
            self.assertEqual(comparison["comparison_mode"], "exact_typed_identifiers")

    def test_comparison_requires_digest_bound_bijective_correspondence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            urdf_path, sdf_path = root / "urdf.json", root / "sdf.json"
            self.compile(FIXTURES / "cross_law.urdf", urdf_path)
            self.compile(FIXTURES / "cross_law.sdf", sdf_path)
            missing = self.run_cli(
                "compare-articulation-grammars", urdf_path, sdf_path, check=False
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("identifiers differ", json.loads(missing.stderr)["error"])

            mapping = root / "mapping.json"
            self.correspondence(
                mapping,
                urdf_path,
                sdf_path,
                {"chassis": "base_link", "segment": "arm_link", "effector": "tool_link"},
                {"spin": "yaw_joint", "slide": "extension_joint"},
            )
            value = json.loads(mapping.read_text())
            value["candidate_grammar_sha256"] = "0" * 64
            mapping.write_text(json.dumps(value))
            tampered = self.run_cli(
                "compare-articulation-grammars",
                urdf_path,
                sdf_path,
                "--correspondence",
                mapping,
                check=False,
            )
            self.assertEqual(tampered.returncode, 2)
            self.assertIn("does not bind", json.loads(tampered.stderr)["error"])

    def test_ambiguous_or_unsupported_source_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = [
                (
                    "nested.sdf",
                    '<sdf version="1.11"><model name="outer"><model name="inner"><link name="x"/></model></model></sdf>',
                    "nested or included",
                ),
                (
                    "equality.xml",
                    '<mujoco><worldbody><body name="root"/></worldbody><equality><joint joint1="a" joint2="b"/></equality></mujoco>',
                    "equality constraints",
                ),
                (
                    "compound.xml",
                    '<mujoco><worldbody><body name="root"><body name="child"><joint name="a"/><joint name="b"/></body></body></worldbody></mujoco>',
                    "multiple joints",
                ),
            ]
            for filename, source, expected in cases:
                path = root / filename
                path.write_text(source)
                result = self.run_cli("articulation-grammar", path, "--out", root / f"{filename}.json", check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, json.loads(result.stderr)["error"])


if __name__ == "__main__":
    unittest.main()
