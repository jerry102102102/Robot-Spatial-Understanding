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
sys.path.insert(0, str(SCRIPT_DIR))

from spatial_configuration import ConfigurationError, _singular_diagnostics, build_configuration_atlas
from spatial_constraints import read_constraint_graph


class SpatialConfigurationAtlasTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def build_spherical_loop(
        self,
        root: Path,
        minimum_solutions: int = 2,
    ) -> tuple[Path, Path, Path, Path, Path]:
        grammar = root / "grammar.json"
        self.run_cli(
            "articulation-grammar",
            FIXTURES / "spherical_loop_tree.urdf",
            "--out",
            grammar,
        )
        constraint_spec = root / "constraints.json"
        constraint_spec.write_text(json.dumps({
            "schema_version": "robot-spatial-constraint-spec.v1",
            "constraint_set_id": "spherical-three-r-loop",
            "grammar_sha256": hashlib.sha256(grammar.read_bytes()).hexdigest(),
            "attachments": [
                {
                    "attachment_id": "ground_axis",
                    "parent_frame": "ground",
                    "semantic_role": "joint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [0.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "attachment_id": "end_axis",
                    "parent_frame": "end_rotor",
                    "semantic_role": "joint_anchor",
                    "parent_from_attachment": {
                        "translation_xyz_m": [0.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ],
            "constraints": [{
                "constraint_id": "align_terminal_z_axes",
                "type": "kinematic_pair",
                "role": "loop_closure",
                "frame_a": "attachment/ground_axis",
                "frame_b": "attachment/end_axis",
                "joint_type": "revolute",
                "axis_xyz_in_a": [0.0, 0.0, 1.0],
                "axis_xyz_in_b": [0.0, 0.0, 1.0],
                "tolerances": {"translation_m": 1e-8, "rotation_rad": 1e-8},
            }],
        }, indent=2, sort_keys=True) + "\n")
        graph = root / "constraint-graph.json"
        self.run_cli("constraint-graph", grammar, constraint_spec, "--out", graph)
        atlas_spec = root / "configuration-atlas-spec.json"
        atlas_spec.write_text(json.dumps({
            "schema_version": "robot-spatial-configuration-atlas-spec.v1",
            "atlas_id": "spherical-loop-finite-witnesses",
            "constraint_graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
            "singular_value_relative_tolerance": 1e-7,
            "charts": [{
                "chart_id": "drive-q-a",
                "parameter_driver": "q_a",
                "parameter_values": [-0.6, 0.0, 0.6],
                "solve_for": ["q_b", "q_c"],
                "driver_scales": {"q_a": 1.0, "q_b": 1.0, "q_c": 1.0},
                "seeds": [
                    {"seed_id": "branch-zero", "joints": {"q_a": 0.0, "q_b": 0.0, "q_c": 0.0}},
                    {"seed_id": "singular-slice-mid", "joints": {"q_a": 0.0, "q_b": math.pi / 2.0, "q_c": 0.0}},
                    {"seed_id": "branch-pi", "joints": {"q_a": 0.0, "q_b": math.pi, "q_c": 0.0}},
                ],
                "solution_merge_tolerance_normalized": 1e-5,
                "continuation_edge_max_distance_normalized": 2.0,
                "minimum_solutions_per_sample": minimum_solutions,
            }],
        }, indent=2, sort_keys=True) + "\n")
        atlas = root / "configuration-atlas.json"
        return grammar, constraint_spec, graph, atlas_spec, atlas

    @staticmethod
    def nodes(atlas: dict) -> list[dict]:
        return [
            node
            for chart in atlas["charts"]
            for sample in chart["samples"]
            for node in sample["solutions"]
        ]

    def test_nonplanar_spherical_loop_exposes_two_branches_and_singular_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, _constraint_spec, graph, spec, atlas_path = self.build_spherical_loop(root)
            generated = self.run_cli("configuration-atlas", graph, spec, "--out", atlas_path)
            result = json.loads(generated.stdout)
            self.assertEqual(result["status"], "generated", result)
            atlas = json.loads(atlas_path.read_text())
            self.assertEqual(atlas["status"], "complete_for_declared_sampling")
            self.assertTrue(atlas["coverage"]["all_declared_sample_minima_met"])

            samples = atlas["charts"][0]["samples"]
            for sample in (samples[0], samples[2]):
                self.assertGreaterEqual(sample["unique_solution_count"], 2)
                parameter = sample["parameter_value"]
                branch_zero = min(
                    sample["solutions"],
                    key=lambda node: abs(((node["independent_driver_positions"]["q_b"] + math.pi) % (2 * math.pi)) - math.pi),
                )["independent_driver_positions"]
                branch_pi = min(
                    sample["solutions"],
                    key=lambda node: abs(abs(((node["independent_driver_positions"]["q_b"] + math.pi) % (2 * math.pi)) - math.pi) - math.pi),
                )["independent_driver_positions"]
                self.assertAlmostEqual(branch_zero["q_c"], -parameter, places=5)
                self.assertAlmostEqual(branch_pi["q_c"], parameter, places=5)

            singular_sample = samples[1]
            self.assertGreaterEqual(singular_sample["unique_solution_count"], 3)
            singular_b_values = sorted(
                node["independent_driver_positions"]["q_b"]
                for node in singular_sample["solutions"]
            )
            self.assertTrue(any(abs(value - math.pi / 2.0) < 1e-6 for value in singular_b_values))
            witnesses = [
                node for node in self.nodes(atlas)
                if node["singularity_witness"]["mechanism_rank_drop_candidate"]
                or node["singularity_witness"]["chart_parameterization_rank_drop_candidate"]
            ]
            self.assertTrue(witnesses)
            self.assertTrue(all(node["constraint_status"] == "satisfied" for node in self.nodes(atlas)))
            self.assertIn("finite", atlas["epistemic_scope"])

    def test_exact_regeneration_execution_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, _constraint_spec, graph, spec, atlas_path = self.build_spherical_loop(root)
            self.run_cli("configuration-atlas", graph, spec, "--out", atlas_path)
            report_path = root / "verification.json"
            verified = json.loads(self.run_cli(
                "verify-configuration-atlas",
                graph,
                spec,
                "--atlas",
                atlas_path,
                "--out",
                report_path,
            ).stdout)
            self.assertEqual(verified["status"], "passed", verified)
            self.assertTrue(verified["exact_regeneration_match"])
            self.assertEqual(
                verified["executed_configuration_node_count"],
                len(self.nodes(json.loads(atlas_path.read_text()))),
            )
            tampered = json.loads(atlas_path.read_text())
            tampered["charts"][0]["samples"][0]["solutions"][0]["independent_driver_positions"]["q_c"] += 0.2
            atlas_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
            failed = self.run_cli(
                "verify-configuration-atlas",
                graph,
                spec,
                "--atlas",
                atlas_path,
                check=False,
            )
            self.assertEqual(failed.returncode, 1)
            failure = json.loads(failed.stdout)
            self.assertEqual(failure["status"], "failed")
            self.assertFalse(failure["exact_regeneration_match"])

    def test_partial_status_and_strict_digest_bound_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, _constraint_spec, graph_path, spec_path, atlas_path = self.build_spherical_loop(root, 10)
            partial = self.run_cli(
                "configuration-atlas", graph_path, spec_path, "--out", atlas_path, check=False
            )
            self.assertEqual(partial.returncode, 1)
            self.assertEqual(json.loads(partial.stdout)["status"], "generated_partial")

            graph = read_constraint_graph(graph_path)
            spec = json.loads(spec_path.read_text())
            spec["constraint_graph_sha256"] = "0" * 64
            with self.assertRaisesRegex(ConfigurationError, "does not bind"):
                build_configuration_atlas(graph, hashlib.sha256(graph_path.read_bytes()).hexdigest(), spec, "spec")

    def test_malformed_chart_contracts_and_atlas_structure_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, _constraint_spec, graph_path, spec_path, atlas_path = self.build_spherical_loop(root)
            graph = read_constraint_graph(graph_path)
            graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            baseline = json.loads(spec_path.read_text())
            variants = []
            extra_root = json.loads(json.dumps(baseline))
            extra_root["unknown"] = True
            variants.append(extra_root)
            repeated_parameter = json.loads(json.dumps(baseline))
            repeated_parameter["charts"][0]["parameter_values"] = [0.0, 0.0]
            variants.append(repeated_parameter)
            incomplete_solve = json.loads(json.dumps(baseline))
            incomplete_solve["charts"][0]["solve_for"] = ["q_b"]
            variants.append(incomplete_solve)
            incomplete_scales = json.loads(json.dumps(baseline))
            incomplete_scales["charts"][0]["driver_scales"].pop("q_c")
            variants.append(incomplete_scales)
            incomplete_seed = json.loads(json.dumps(baseline))
            incomplete_seed["charts"][0]["seeds"][0]["joints"].pop("q_c")
            variants.append(incomplete_seed)
            boolean_minimum = json.loads(json.dumps(baseline))
            boolean_minimum["charts"][0]["minimum_solutions_per_sample"] = True
            variants.append(boolean_minimum)
            for invalid in variants:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ConfigurationError):
                        build_configuration_atlas(graph, graph_sha, invalid, "spec")

            self.run_cli("configuration-atlas", graph_path, spec_path, "--out", atlas_path)
            malformed = json.loads(atlas_path.read_text())
            malformed["charts"] = None
            atlas_path.write_text(json.dumps(malformed, indent=2, sort_keys=True) + "\n")
            failed = self.run_cli(
                "verify-configuration-atlas",
                graph_path,
                spec_path,
                "--atlas",
                atlas_path,
                check=False,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(json.loads(failed.stdout)["status"], "failed")

    def test_singular_value_diagnostics_are_dependency_free_and_rank_revealing(self):
        full = _singular_diagnostics([[3.0, 0.0], [0.0, 4.0]], 2, 1e-9)
        self.assertEqual(full["numerical_rank"], 2)
        self.assertEqual(full["singular_values_descending"], [4.0, 3.0])
        rank_one = _singular_diagnostics([[1.0, 2.0], [2.0, 4.0]], 2, 1e-9)
        self.assertEqual(rank_one["numerical_rank"], 1)
        self.assertEqual(rank_one["nullity"], 1)
        self.assertTrue(rank_one["condition_number_infinite_or_unresolved"])

    def test_export_publishes_atlas_facts_cards_and_retrieval_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _grammar, constraint_spec, _graph, atlas_spec, _atlas = self.build_spherical_loop(root)
            export_dir = root / "export"
            exported = json.loads(self.run_cli(
                "export",
                FIXTURES / "spherical_loop_tree.urdf",
                "--out",
                export_dir,
                "--constraint-spec",
                constraint_spec,
                "--configuration-atlas-spec",
                atlas_spec,
                "--workspace-samples",
                0,
                "--generate-evaluation",
                "--evaluation-key-out",
                root / "private-answer-key.jsonl",
            ).stdout)
            self.assertEqual(exported["status"], "exported", exported)
            self.assertTrue((export_dir / "configuration-atlas.json").is_file())
            model = json.loads((export_dir / "model.json").read_text())
            artifact = model["artifacts"]["configuration_atlas"]
            self.assertEqual(artifact["status"], "complete_for_declared_sampling")
            self.assertTrue(model["capabilities"]["finite_configuration_atlas"]["generated"])
            cards = [json.loads(line) for line in (export_dir / "entity-cards.jsonl").read_text().splitlines()]
            card_types = {card["entity_type"] for card in cards}
            self.assertTrue({
                "configuration_atlas",
                "configuration_chart",
                "configuration_node",
                "configuration_component",
            }.issubset(card_types))
            facts = [json.loads(line) for line in (export_dir / "facts.jsonl").read_text().splitlines()]
            self.assertIn(
                "is_executable_constraint_satisfying_configuration_witness",
                {fact["predicate"] for fact in facts},
            )
            atlas_entity = next(
                card["entity_id"] for card in cards if card["entity_type"] == "configuration_atlas"
            )
            retrieved = json.loads(self.run_cli(
                "retrieve", export_dir, "--entity", atlas_entity
            ).stdout)
            self.assertEqual(retrieved["entity_card"]["entity_id"], atlas_entity)
            manifest = json.loads((export_dir / "agent-context.json").read_text())
            self.assertIn("configuration_atlas", manifest["artifacts"])
            self.assertTrue(any(
                route.get("tool") == "configuration-atlas"
                for route in manifest["question_router"]
            ))
            evaluation_manifest = json.loads((export_dir / "evaluation" / "manifest.json").read_text())
            self.assertGreaterEqual(
                evaluation_manifest["capability_counts"]["finite_configuration_space_understanding"],
                3,
            )

            prepared_dir = root / "prepared"
            prepared = json.loads(self.run_cli(
                "prepare",
                FIXTURES / "spherical_loop_tree.urdf",
                "--out",
                prepared_dir,
                "--constraint-spec",
                constraint_spec,
                "--configuration-atlas-spec",
                atlas_spec,
                "--workspace-samples",
                0,
            ).stdout)
            self.assertEqual(prepared["status"], "prepared")
            self.assertTrue((prepared_dir / "context" / "configuration-atlas.json").is_file())
            source_manifest = json.loads((prepared_dir / "source-manifest.json").read_text())
            self.assertEqual(
                source_manifest["configuration_atlas"]["sha256"],
                hashlib.sha256(atlas_spec.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
