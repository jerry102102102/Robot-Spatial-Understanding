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

from spatial_concepts import ConceptError, QUERY_SCHEMA, query_concept_graph, read_concept_graph


class SpatialConceptGraphTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def export(self, root: Path, urdf: Path, *extra: object) -> Path:
        context = root / "context"
        result = json.loads(self.run_cli(
            "export",
            urdf,
            "--workspace-samples",
            "0",
            *extra,
            "--out",
            context,
        ).stdout)
        self.assertEqual(result["status"], "exported", result)
        self.assertEqual(Path(result["concept_graph"]), (context / "concept-graph.json").resolve())
        return context

    @staticmethod
    def query(graph: dict, query_id: str, intent: str, parameters: dict) -> dict:
        return query_concept_graph(graph, {
            "schema_version": QUERY_SCHEMA,
            "query_id": query_id,
            "intent": intent,
            "parameters": parameters,
        })

    @staticmethod
    def rehash(graph: dict) -> dict:
        body = {key: value for key, value in graph.items() if key != "concept_graph_sha256"}
        graph["concept_graph_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return graph

    def test_export_builds_proof_language_queries_and_exact_negative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.export(root, FIXTURES / "two_dof.urdf")
            graph = read_concept_graph(context / "concept-graph.json")
            self.assertEqual(graph["schema_version"], "robot-spatial-concept-graph.v1")
            self.assertEqual(graph["coverage"]["tree_link_count"], 4)
            self.assertEqual(graph["coverage"]["independent_driver_count"], 2)
            self.assertTrue(graph["language_contract"]["negative_answer_rule"].startswith("return a negative"))

            summary = self.query(graph, "summary", "structural_summary", {})
            self.assertEqual(summary["answer"]["root_link"], "link/base_link")
            self.assertEqual(summary["answer"]["structural_leaves"], ["link/tool0"])
            self.assertEqual(len(summary["answer"]["maximal_serial_segments"]), 1)

            path = self.query(graph, "path", "trace_kinematic_path", {
                "from_link": "tool0",
                "to_link": "base_link",
            })
            self.assertEqual(path["answer"]["joint_count"], 3)
            self.assertTrue(all(
                step["traversal_direction"] == "child_to_parent"
                for step in path["answer"]["ordered_steps"]
            ))
            self.assertEqual(len(path["supporting_clauses"]), 3)

            positive = self.query(graph, "positive", "explain_driver_effect", {
                "driver": "shoulder",
                "target_frame": "tool0",
            })
            self.assertTrue(positive["answer"]["target_pose_can_change_relative_to_root"])
            self.assertTrue(any(
                clause["predicate"] == "can_change_pose_of_frame_relative_to_root"
                for clause in positive["supporting_clauses"]
            ))

            negative = self.query(graph, "negative", "explain_driver_effect", {
                "driver": "slide",
                "target_frame": "base_link",
            })
            self.assertFalse(negative["answer"]["target_pose_can_change_relative_to_root"])
            self.assertEqual(negative["status"], "answered")
            self.assertEqual(negative["unknowns"], [])

            law = self.query(graph, "law", "explain_frame_pose_law", {"frame": "tool0"})
            self.assertEqual(
                law["answer"]["ordered_operator_refs"],
                ["joint_operator/shoulder", "joint_operator/slide", "joint_operator/tool_mount"],
            )

            model = json.loads((context / "model.json").read_text())
            agent = json.loads((context / "agent-context.json").read_text())
            self.assertEqual(model["capabilities"]["proof_carrying_spatial_concept_graph"]["generated"], True)
            self.assertEqual(agent["artifacts"]["concept_graph"]["concept_graph_sha256"], graph["concept_graph_sha256"])
            self.assertEqual(agent["statistics"]["entity_type_counts"]["concept_graph"], 1)
            concept_cards = [
                json.loads(line)
                for line in (context / "entity-cards.jsonl").read_text().splitlines()
                if json.loads(line)["entity_type"] == "concept_graph"
            ]
            self.assertEqual(concept_cards[0]["entity_id"], graph["concept_graph_id"])
            self.assertIn("query-concepts", {query["command"] for query in concept_cards[0]["tool_queries"]})

    def test_branch_and_mimic_are_concepts_not_name_guesses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.export(root, FIXTURES / "mimic_branch.urdf")
            graph = read_concept_graph(context / "concept-graph.json")
            topology = graph["projections"]["topology"]
            self.assertEqual(topology["branch_points"], ["link/base"])
            self.assertEqual(topology["structural_leaves"], ["link/driver_link", "link/follower_link"])
            self.assertEqual(len(topology["maximal_serial_segments"]), 2)

            effect = self.query(graph, "mimic-effect", "explain_driver_effect", {"driver": "driver"})
            self.assertEqual(
                effect["answer"]["physical_joints_driven"],
                ["joint/driver", "joint/follower"],
            )
            self.assertIn("frame/driver_link", effect["answer"]["affected_frames_relative_to_root"])
            self.assertIn("frame/follower_link", effect["answer"]["affected_frames_relative_to_root"])
            self.assertNotIn("frame/joint/driver", effect["answer"]["affected_frames_relative_to_root"])

            for leaf in topology["structural_leaves"]:
                description = self.query(graph, f"describe-{leaf}", "describe_entity", {"entity": leaf})
                leaf_clauses = [
                    clause for clause in description["supporting_clauses"]
                    if clause["predicate"] == "is_structural_leaf"
                ]
                self.assertEqual(len(leaf_clauses), 1)
                self.assertIn("does not by itself assert end-effector", leaf_clauses[0]["cnl"])

    def build_closed_context(self, root: Path) -> Path:
        urdf = FIXTURES / "spherical_loop_tree.urdf"
        grammar = root / "grammar.json"
        self.run_cli("articulation-grammar", urdf, "--out", grammar)
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
        graph_path = root / "constraint-graph.json"
        self.run_cli("constraint-graph", grammar, constraint_spec, "--out", graph_path)
        atlas_spec = root / "atlas-spec.json"
        atlas_spec.write_text(json.dumps({
            "schema_version": "robot-spatial-configuration-atlas-spec.v1",
            "atlas_id": "concept-test-atlas",
            "constraint_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "singular_value_relative_tolerance": 1e-7,
            "charts": [{
                "chart_id": "drive-q-a",
                "parameter_driver": "q_a",
                "parameter_values": [-0.6, 0.0, 0.6],
                "solve_for": ["q_b", "q_c"],
                "driver_scales": {"q_a": 1.0, "q_b": 1.0, "q_c": 1.0},
                "seeds": [
                    {"seed_id": "zero", "joints": {"q_a": 0.0, "q_b": 0.0, "q_c": 0.0}},
                    {"seed_id": "mid", "joints": {"q_a": 0.0, "q_b": math.pi / 2.0, "q_c": 0.0}},
                    {"seed_id": "pi", "joints": {"q_a": 0.0, "q_b": math.pi, "q_c": 0.0}},
                ],
                "solution_merge_tolerance_normalized": 1e-5,
                "continuation_edge_max_distance_normalized": 2.0,
                "minimum_solutions_per_sample": 2,
            }],
        }, indent=2, sort_keys=True) + "\n")
        return self.export(
            root,
            urdf,
            "--constraint-spec",
            constraint_spec,
            "--configuration-atlas-spec",
            atlas_spec,
        )

    def test_constraints_and_finite_nodes_preserve_assertion_and_topology_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.build_closed_context(root)
            graph = read_concept_graph(context / "concept-graph.json")
            constraint = self.query(graph, "constraint", "explain_constraint", {
                "constraint": "align_terminal_z_axes",
            })
            self.assertEqual(constraint["answer"]["type"], "kinematic_pair")
            self.assertEqual(constraint["answer"]["role"], "loop_closure")
            self.assertEqual(len(constraint["answer"]["driver_dependencies"]), 3)
            relation = next(
                clause for clause in constraint["supporting_clauses"]
                if clause["predicate"] == "requires_mechanism_relation"
            )
            self.assertFalse(relation["evidence"]["exact"])
            self.assertEqual(relation["modality"], "supplemental_asserted_relation")

            configuration = graph["projections"]["configuration"]
            nodes = configuration["charts"][0]["nodes"]
            compared = self.query(graph, "compare", "compare_configuration_nodes", {
                "node_a": nodes[0]["node_entity"],
                "node_b": nodes[1]["node_entity"],
            })
            self.assertEqual(compared["answer"]["same_global_branch"], "not_established")
            self.assertIn("finite proximity component", compared["answer_cnl"])
            self.assertFalse(configuration["global_branch_topology_certified"])
            self.assertFalse(configuration["certified_singularity"])

    def test_standalone_regeneration_tamper_and_query_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = self.export(root, FIXTURES / "two_dof.urdf")
            regenerated = root / "regenerated.json"
            language = root / "regenerated.rsl"
            self.run_cli(
                "concept-graph",
                context,
                "--out",
                regenerated,
                "--language-out",
                language,
            )
            self.assertEqual(regenerated.read_bytes(), (context / "concept-graph.json").read_bytes())
            self.assertEqual(language.read_bytes(), (context / "concept-language.rsl").read_bytes())
            passed = json.loads(self.run_cli(
                "verify-concept-graph",
                context,
                "--concept",
                regenerated,
                "--language",
                language,
            ).stdout)
            self.assertEqual(passed["status"], "passed", passed)

            malformed = json.loads(regenerated.read_text())
            malformed["clauses"][0]["cnl"] = "tampered"
            regenerated.write_text(json.dumps(malformed, indent=2, sort_keys=True) + "\n")
            failed_query = root / "query.json"
            failed_query.write_text(json.dumps({
                "schema_version": QUERY_SCHEMA,
                "query_id": "tamper",
                "intent": "structural_summary",
                "parameters": {},
            }))
            failed = self.run_cli("query-concepts", regenerated, failed_query, check=False)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("digest is invalid", json.loads(failed.stderr)["error"])

            regenerated.write_bytes((context / "concept-graph.json").read_bytes())
            language.write_text(language.read_text() + "TAMPER\n")
            failed_verify = self.run_cli(
                "verify-concept-graph",
                context,
                "--concept",
                regenerated,
                "--language",
                language,
                check=False,
            )
            self.assertEqual(failed_verify.returncode, 1)
            self.assertFalse(json.loads(failed_verify.stdout)["exact_language_regeneration_match"])

            graph = read_concept_graph(regenerated)
            with self.assertRaisesRegex(ConceptError, "fields mismatch"):
                query_concept_graph(graph, {
                    "schema_version": QUERY_SCHEMA,
                    "query_id": "extra",
                    "intent": "structural_summary",
                    "parameters": {},
                    "unexpected": True,
                })

            original = json.loads(regenerated.read_text())
            indexed_entity = next(
                entity for entity, clause_ids in original["indexes"]["by_entity"].items()
                if clause_ids
            )
            index_tamper = json.loads(json.dumps(original))
            index_tamper["indexes"]["by_entity"][indexed_entity].pop()
            regenerated.write_text(json.dumps(self.rehash(index_tamper), indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ConceptError, "indexes do not exactly match"):
                read_concept_graph(regenerated)

            coverage_tamper = json.loads(json.dumps(original))
            coverage_tamper["coverage"]["clause_count"] += 1
            regenerated.write_text(json.dumps(self.rehash(coverage_tamper), indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ConceptError, "coverage does not exactly match"):
                read_concept_graph(regenerated)

            projection_tamper = json.loads(json.dumps(original))
            first_edge = projection_tamper["projections"]["topology"]["edges"][0]
            first_edge["parent_link"] = first_edge["child_link"]
            regenerated.write_text(json.dumps(self.rehash(projection_tamper), indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ConceptError, "disagrees with its supporting clause"):
                read_concept_graph(regenerated)

            effect_tamper = json.loads(json.dumps(original))
            effect_tamper["projections"]["articulation"]["drivers"][0]["affected_frames"].pop()
            regenerated.write_text(json.dumps(self.rehash(effect_tamper), indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ConceptError, "affected-frame projection is inconsistent"):
                read_concept_graph(regenerated)


if __name__ == "__main__":
    unittest.main()
