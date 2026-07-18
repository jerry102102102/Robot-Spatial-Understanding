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
SEMANTICS = FIXTURES / "semantics.json"
PACKAGE_MAP = FIXTURES / "package_map.json"


class SemanticRenderAtlasTests(unittest.TestCase):
    def export(self, output: Path, *, inspect_meshes: bool = True, evaluation_key: Path | None = None) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "export",
            str(URDF),
            "--pose",
            str(POSE),
            "--semantics",
            str(SEMANTICS),
            "--render",
            "--out",
            str(output),
        ]
        if inspect_meshes:
            command.extend(["--inspect-meshes", "--package-map", str(PACKAGE_MAP)])
        if evaluation_key is not None:
            command.extend(["--generate-evaluation", "--evaluation-key-out", str(evaluation_key)])
        return subprocess.run(command, check=True, capture_output=True, text=True)

    def verifier_command(self, atlas: Path, *, inspect_meshes: bool = True) -> list[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "verify-render",
            str(URDF),
            "--pose",
            str(POSE),
            "--semantics",
            str(SEMANTICS),
            "--atlas",
            str(atlas),
        ]
        if inspect_meshes:
            command.extend(["--inspect-meshes", "--package-map", str(PACKAGE_MAP)])
        return command

    def test_export_binds_numeric_projection_svg_entities_context_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "context"
            response = json.loads(self.export(output, evaluation_key=root / "private" / "key.jsonl").stdout)
            self.assertEqual(response["status"], "exported")
            atlas_path = output / "render-atlas" / "manifest.json"
            self.assertEqual(Path(response["render_atlas"]).resolve(), atlas_path.resolve())
            atlas = json.loads(atlas_path.read_text())
            model = json.loads((output / "model.json").read_text())
            self.assertEqual(atlas["schema_version"], "robot-spatial-render-atlas.v1")
            self.assertEqual(atlas["source_binding"]["urdf_semantic_sha256"], model["source"]["semantic_sha256"])
            self.assertEqual(atlas["coverage"]["view_count"], 4)
            self.assertTrue(atlas["coverage"]["complete_for_declared_geometry"])
            self.assertEqual(set(atlas["views"]), {"front", "side", "top", "isometric"})
            front = atlas["views"]["front"]
            self.assertEqual(front["projection"]["root_xyz_to_uv_matrix_2x3"], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            tool = next(record for record in front["link_frames"] if record["entity_id"] == "frame/tool0")
            root_xyz = model["frames"]["tool0"]["world_from_frame"]["translation_xyz_m"]
            self.assertEqual(tool["origin_root_xyz_m"], root_xyz)
            self.assertEqual(tool["projected_uv_m"], [root_xyz[0], root_xyz[2]])
            screen = front["screen"]
            expected_pixel = [
                screen["width_px"] / 2.0 + (tool["projected_uv_m"][0] - screen["center_uv_m"][0]) * screen["scale_px_per_m"],
                screen["plot_rect_xywh_px"][1] + screen["plot_rect_xywh_px"][3] / 2.0 - (tool["projected_uv_m"][1] - screen["center_uv_m"][1]) * screen["scale_px_per_m"],
            ]
            for observed, expected in zip(tool["pixel_xy"], expected_pixel):
                self.assertAlmostEqual(observed, expected, places=5)
            for view_id, view in atlas["views"].items():
                svg_path = atlas_path.parent / view["artifact"]["path"]
                self.assertEqual(hashlib.sha256(svg_path.read_bytes()).hexdigest(), view["artifact"]["sha256"])
                svg = svg_path.read_text()
                self.assertIn('data-entity-id="frame/tool0"', svg)
                self.assertIn('data-entity-id="joint/shoulder"', svg)
                self.assertIn(view_id.capitalize() if view_id != "isometric" else "Isometric", svg)
            artifact = model["artifacts"]["semantic_render_atlas"]
            self.assertEqual(artifact["manifest_sha256"], hashlib.sha256(atlas_path.read_bytes()).hexdigest())
            self.assertTrue(model["capabilities"]["semantic_render_atlas"]["generated"])
            cards = [json.loads(line) for line in (output / "entity-cards.jsonl").read_text().splitlines()]
            atlas_cards = [card for card in cards if card["entity_type"] in {"render_atlas", "render_view"}]
            self.assertEqual(len(atlas_cards), 5)
            isometric_card = next(card for card in atlas_cards if card["entity_id"].endswith("/isometric"))
            self.assertGreater(isometric_card["trust"]["bound_fact_count"], 1)
            questions = [json.loads(line) for line in (output / "evaluation" / "questions.jsonl").read_text().splitlines()]
            visual_questions = [record for record in questions if record["capability"] == "semantic_visual_grounding"]
            self.assertEqual(len(visual_questions), 5)
            verification = subprocess.run(self.verifier_command(atlas_path), check=True, capture_output=True, text=True)
            report = json.loads(verification.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["issue_count"], 0)

    def test_render_atlas_is_byte_deterministic_across_output_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first, second = root / "first", root / "second"
            self.export(first)
            self.export(second)
            relative_files = [
                Path("render-atlas/manifest.json"),
                Path("render-atlas/views/front.svg"),
                Path("render-atlas/views/side.svg"),
                Path("render-atlas/views/top.svg"),
                Path("render-atlas/views/isometric.svg"),
            ]
            for relative in relative_files:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes(), relative)

    def test_uninspected_mesh_is_explicitly_unrendered_and_verifiable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "context"
            self.export(output, inspect_meshes=False)
            atlas_path = output / "render-atlas" / "manifest.json"
            atlas = json.loads(atlas_path.read_text())
            self.assertFalse(atlas["coverage"]["complete_for_declared_geometry"])
            self.assertEqual(atlas["coverage"]["unrendered_geometry_frames"], ["collision/slider_link/0"])
            for view in atlas["views"].values():
                self.assertNotIn("frame/collision/slider_link/0", {record["entity_id"] for record in view["geometry"]})
            result = subprocess.run(self.verifier_command(atlas_path, inspect_meshes=False), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "passed")

    def test_verifier_rejects_tampered_svg_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "context"
            self.export(output)
            atlas_path = output / "render-atlas" / "manifest.json"
            atlas = json.loads(atlas_path.read_text())
            front_path = atlas_path.parent / atlas["views"]["front"]["artifact"]["path"]
            front_path.write_text(front_path.read_text() + "<!-- tampered -->\n")
            result = subprocess.run(self.verifier_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any(issue["check"] == "views.front.artifact.sha256" for issue in report["issues"]))

            self.export(output)
            atlas = json.loads(atlas_path.read_text())
            front_path = atlas_path.parent / atlas["views"]["front"]["artifact"]["path"]
            front_svg = front_path.read_text()
            self.assertIn('data-entity-id="frame/tool0"', front_svg)
            front_path.write_text(front_svg.replace('data-entity-id="frame/tool0"', 'data-entity-id="removed/tool0"', 1))
            atlas["views"]["front"]["artifact"]["sha256"] = hashlib.sha256(front_path.read_bytes()).hexdigest()
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verifier_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertTrue(any(issue["check"] == "views.front.svg_entity_ids" for issue in report["issues"]))

            self.export(output)
            atlas = json.loads(atlas_path.read_text())
            atlas["source_binding"]["urdf_semantic_sha256"] = "0" * 64
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verifier_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertTrue(any(issue["check"] == "manifest.source_binding" for issue in report["issues"]))

            self.export(output)
            atlas = json.loads(atlas_path.read_text())
            atlas["combined_overview"]["path"] = "../../outside.svg"
            atlas_path.write_text(json.dumps(atlas, indent=2, sort_keys=True) + "\n")
            result = subprocess.run(self.verifier_command(atlas_path), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertTrue(any(issue["check"] == "combined_overview.path" for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
