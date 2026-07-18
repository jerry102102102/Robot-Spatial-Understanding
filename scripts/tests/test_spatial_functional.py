from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_DIR / "robot_spatial.py"
ORACLE = SCRIPT_DIR / "crosscheck_functional_model.py"
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(SCRIPT_DIR))

from spatial_functional import (  # noqa: E402
    FunctionalError,
    QUERY_SCHEMA,
    build_functional_model_from_context,
    query_functional_model,
    read_functional_model,
    verify_functional_model,
    write_functional_model_from_context,
)
import spatial_evaluation_suite  # noqa: E402


class SpatialFunctionalModelTests(unittest.TestCase):
    def run_cli(self, *arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(value) for value in arguments)],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def export(self, root: Path, name: str = "base", *extra: object, check: bool = True) -> tuple[Path, subprocess.CompletedProcess[str]]:
        context = root / name
        result = self.run_cli(
            "export",
            FIXTURES / "mimic_branch.urdf",
            "--workspace-samples",
            "0",
            *extra,
            "--out",
            context,
            check=check,
        )
        return context, result

    @staticmethod
    def spec(context: Path, *, complete: bool = True) -> dict:
        canonical = json.loads((context / "model.json").read_text(encoding="utf-8"))
        graph = json.loads((context / "concept-graph.json").read_text(encoding="utf-8"))
        driver = graph["projections"]["articulation"]["drivers"][0]["driver_entity"]
        return {
            "schema_version": "robot-spatial-function-affordance-spec.v1",
            "function_set_id": "function_set/test_gripper",
            "source_binding": {
                "urdf_semantic_sha256": canonical["source"]["semantic_sha256"],
                "articulation_grammar_sha256": canonical["artifacts"]["articulation_grammar"]["sha256"],
                "constraint_graph_sha256": None,
                "configuration_atlas_sha256": None,
            },
            "object_types": [{
                "object_type_id": "object_type/graspable",
                "meaning": "A project-declared target type, not a geometry-based classification.",
            }],
            "components": [{
                "component_id": "component/gripper",
                "members": ["link/driver_link", "link/follower_link", "joint/driver", "joint/follower"],
                "meaning": "The explicitly grouped paired mechanism.",
            }],
            "functions": [{
                "function_id": "function/retain_object",
                "provided_by": ["component/gripper"],
                "verb": "retain",
                "object_types": ["object_type/graspable"],
                "purpose": "Intended object retention after separate execution checks.",
            }],
            "conditions": [
                {
                    "condition_id": "condition/target_between_fingers",
                    "predicate": "target_between_fingers",
                    "arguments": ["actor", "target"],
                    "truth_source": "runtime_observation_required",
                    "meaning": "The target is observed between the fingers.",
                },
                {
                    "condition_id": "condition/plan_approved",
                    "predicate": "plan_approved",
                    "arguments": ["actor", "target"],
                    "truth_source": "planner_verification_required",
                    "meaning": "A separate planner approves the candidate action.",
                },
            ],
            "effects": [{
                "effect_id": "effect/target_retained",
                "predicate": "target_retained_by",
                "arguments": ["target", "actor"],
                "meaning": "The target is intended to remain relative to the gripper.",
            }],
            "capabilities": [{
                "capability_id": "capability/coordinated_closure",
                "provided_by": ["component/gripper"],
                "realizes_functions": ["function/retain_object"],
                "enabling_requirements": [
                    {
                        "requirement_id": "requirement/driver_affects_left",
                        "type": "driver_affects_frame",
                        "parameters": {"driver": driver, "frame": "frame/driver_link"},
                    },
                    {
                        "requirement_id": "requirement/driver_affects_right",
                        "type": "driver_affects_frame",
                        "parameters": {"driver": driver, "frame": "frame/follower_link"},
                    },
                    {
                        "requirement_id": "requirement/shared_tree",
                        "type": "kinematic_path_exists",
                        "parameters": {"from_link": "link/driver_link", "to_link": "link/follower_link"},
                    },
                ],
                "condition_refs": ["condition/target_between_fingers"],
                "limitations": ["Structure does not establish force closure, payload, runtime, hardware, or safety."],
            }],
            "affordances": [{
                "affordance_id": "affordance/grasp",
                "offered_by": ["component/gripper"],
                "action_verb": "grasp",
                "target_object_types": ["object_type/graspable"],
                "capability_refs": ["capability/coordinated_closure"],
                "precondition_refs": ["condition/target_between_fingers", "condition/plan_approved"],
                "effect_refs": ["effect/target_retained"],
                "meaning": "A conditional actor-action-target-effect relation.",
            }],
            "inventory_completeness": (
                [{
                    "subject": "component/gripper",
                    "inventories": ["functions", "capabilities", "affordances"],
                    "scope": "Only this project specification.",
                }]
                if complete
                else []
            ),
        }

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def query(model: dict, query_id: str, intent: str, parameters: dict) -> dict:
        return query_functional_model(model, {
            "schema_version": QUERY_SCHEMA,
            "query_id": query_id,
            "intent": intent,
            "parameters": parameters,
        })

    @staticmethod
    def rehash(model: dict) -> dict:
        body = {key: value for key, value in model.items() if key != "functional_model_sha256"}
        model["functional_model_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return model

    def test_export_context_and_queries_preserve_modal_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _ = self.export(root)
            spec_path = root / "functional-spec.json"
            self.write_json(spec_path, self.spec(base))
            private_key = root / "private" / "answers.jsonl"
            context, process = self.export(
                root,
                "functional",
                "--functional-spec",
                spec_path,
                "--generate-evaluation",
                "--evaluation-key-out",
                private_key,
            )
            response = json.loads(process.stdout)
            self.assertEqual(response["status"], "exported")

            model = read_functional_model(context / "functional-model.json")
            self.assertEqual(model["status"], "all_declared_capabilities_structurally_grounded")
            self.assertEqual(model["coverage"]["satisfied_requirement_count"], 3)
            capability = self.query(model, "capability", "explain_capability", {"capability": "coordinated_closure"})
            self.assertFalse(capability["answer"]["physical_capability_verified"])
            self.assertEqual(len(capability["structural_supporting_clauses"]), 7)
            self.assertTrue(all(requirement["status"] == "satisfied" for requirement in capability["answer"]["requirements"]))

            affordance = self.query(model, "affordance", "explain_affordance", {"affordance": "grasp"})
            self.assertEqual(affordance["answer"]["current_preconditions_satisfied"], "not_evaluated")
            self.assertEqual(affordance["answer"]["physical_executability"], "not_established")
            self.assertIn("the intended effect is not an observed effect", affordance["unknowns"])

            possible = self.query(model, "possible", "can_perform_action", {
                "offered_by": "gripper", "action_verb": "grasp", "target_object_type": "graspable",
            })
            self.assertEqual(possible["answer"]["conclusion"], "declared_possible_if_preconditions_hold")
            self.assertEqual(possible["answer"]["current_preconditions_satisfied"], "not_evaluated")
            self.assertEqual(possible["answer"]["physical_executability"], "not_established")

            absent = self.query(model, "absent", "can_perform_action", {
                "offered_by": "gripper", "action_verb": "weld", "target_object_type": "graspable",
            })
            self.assertEqual(absent["answer"]["conclusion"], "not_declared_in_complete_project_inventory")
            self.assertEqual(absent["answer"]["physical_impossibility"], "not_established")

            agent = json.loads((context / "agent-context.json").read_text(encoding="utf-8"))
            self.assertIn("functional_model", agent["artifacts"])
            self.assertIn("query-functions#task_relevant_functional_and_structural_proof_closure", agent["load_order"])
            self.assertEqual(agent["statistics"]["entity_type_counts"]["functional_affordance"], 1)
            cards = [json.loads(line) for line in (context / "entity-cards.jsonl").read_text().splitlines()]
            gripper = next(card for card in cards if card["entity_id"] == "component/gripper")
            self.assertEqual(gripper["trust"]["classification"], "project_asserted_with_digest_bound_proof_model")
            self.assertIn("query-functions", {query["command"] for query in gripper["tool_queries"]})
            evaluation = json.loads((context / "evaluation" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation["capability_counts"]["function_affordance_understanding"], 5)
            questions = [
                json.loads(line)
                for line in (context / "evaluation" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {question["task"] for question in questions if question["capability"] == "function_affordance_understanding"},
                {
                    "apply_declared_action_contract_without_claiming_execution",
                    "apply_inventory_completeness_without_claiming_physical_impossibility",
                    "explain_explicit_component_function_without_name_inference",
                    "explain_relational_affordance_preconditions_and_intended_effects",
                    "separate_capability_declaration_structural_grounding_and_physical_truth",
                },
            )
            functional_questions = {
                question["task"]: question
                for question in questions
                if question["capability"] == "function_affordance_understanding"
            }
            component_contract = functional_questions[
                "explain_explicit_component_function_without_name_inference"
            ]["submission_contract"]["record"]["answer"]
            self.assertEqual(
                set(component_contract),
                {
                    "component_id",
                    "members",
                    "meaning",
                    "function_ids",
                    "capability_ids",
                    "affordance_ids",
                    "name_or_geometry_inference_used",
                },
            )
            requirement_contract = functional_questions[
                "separate_capability_declaration_structural_grounding_and_physical_truth"
            ]["submission_contract"]["record"]["answer"]["requirements"][0]
            self.assertEqual(
                set(requirement_contract),
                {
                    "requirement_id",
                    "type",
                    "status",
                    "satisfied",
                    "modality",
                    "closure_basis",
                    "concept_clause_ids",
                },
            )
            self.assertTrue(private_key.is_file())

    def test_open_world_exact_negative_and_incomplete_inventory_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, _ = self.export(root)
            spec = self.spec(context, complete=False)
            spec["capabilities"][0]["enabling_requirements"].extend([
                {
                    "requirement_id": "requirement/missing_entity",
                    "type": "entity_exists",
                    "parameters": {"entity": "link/not_declared"},
                },
                {
                    "requirement_id": "requirement/unasserted_role",
                    "type": "frame_has_asserted_role",
                    "parameters": {"frame": "frame/driver_link", "role": "tool_center_point"},
                },
            ])
            spec_path = root / "functional-spec.json"
            self.write_json(spec_path, spec)
            model = build_functional_model_from_context(context, spec_path)
            results = {item["requirement_id"]: item for item in model["projections"]["capabilities"][0]["requirements"]}
            self.assertEqual(results["requirement/missing_entity"]["status"], "not_satisfied_exact_closed_world")
            self.assertEqual(results["requirement/missing_entity"]["evidence"]["closure_basis"], "complete concept entity inventory")
            self.assertEqual(results["requirement/unasserted_role"]["status"], "not_established_open_world")
            self.assertFalse(model["coverage"]["all_declared_capabilities_structurally_grounded"])

            declared_but_ungrounded = self.query(model, "ungrounded", "can_perform_action", {
                "offered_by": "gripper", "action_verb": "grasp", "target_object_type": "graspable",
            })
            self.assertEqual(
                declared_but_ungrounded["answer"]["conclusion"],
                "declared_affordance_with_ungrounded_capability_requirements",
            )
            self.assertEqual(declared_but_ungrounded["answer"]["structurally_grounded_matching_affordances"], [])

            unknown = self.query(model, "unknown", "can_perform_action", {
                "offered_by": "gripper", "action_verb": "weld", "target_object_type": "graspable",
            })
            self.assertEqual(unknown["status"], "unknown")
            self.assertEqual(unknown["answer"]["conclusion"], "unknown_not_in_incomplete_inventory")

    def test_cli_compile_query_and_exact_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, _ = self.export(root)
            spec_path = root / "functional-spec.json"
            model_path = root / "functional-model.json"
            query_path = root / "query.json"
            self.write_json(spec_path, self.spec(context))
            compiled = json.loads(self.run_cli("functional-model", context, spec_path, "--out", model_path).stdout)
            self.assertEqual(compiled["status"], "generated")
            self.assertEqual(compiled["grounding_status"], "all_declared_capabilities_structurally_grounded")
            self.write_json(query_path, {
                "schema_version": QUERY_SCHEMA,
                "query_id": "why",
                "intent": "what_is_entity_for",
                "parameters": {"entity": "link/driver_link"},
            })
            answer = json.loads(self.run_cli("query-functions", model_path, query_path, "--compact").stdout)
            self.assertEqual(answer["answer"]["declared_components"], ["component/gripper"])
            self.assertFalse(answer["answer"]["name_based_inference_used"])
            verified = json.loads(self.run_cli(
                "verify-functional-model", context, spec_path, "--model", model_path,
            ).stdout)
            self.assertEqual(verified["status"], "passed")

            coherent_tamper = json.loads(model_path.read_text(encoding="utf-8"))
            coherent_tamper["epistemic_scope"] += " tampered"
            self.rehash(coherent_tamper)
            self.write_json(model_path, coherent_tamper)
            failed = verify_functional_model(context, spec_path, model_path)
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed["exact_regeneration_match"])

    def test_rehashed_internal_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, _ = self.export(root)
            spec_path = root / "functional-spec.json"
            model_path = root / "functional-model.json"
            self.write_json(spec_path, self.spec(context))
            write_functional_model_from_context(context, spec_path, model_path)
            original = json.loads(model_path.read_text(encoding="utf-8"))

            tamper_cases = []
            projection = copy.deepcopy(original)
            projection["projections"]["capabilities"][0]["physical_capability_verified"] = True
            tamper_cases.append(projection)
            index = copy.deepcopy(original)
            index["indexes"]["by_subject"].pop("capability/coordinated_closure")
            tamper_cases.append(index)
            coverage = copy.deepcopy(original)
            coverage["coverage"]["requirement_count"] += 1
            tamper_cases.append(coverage)
            structural = copy.deepcopy(original)
            structural["structural_evidence_clauses"][0]["cnl"] += " altered"
            tamper_cases.append(structural)
            projection_clause = copy.deepcopy(original)
            projection_clause["projections"]["functions"][0]["purpose"] = "A different unbound purpose."
            tamper_cases.append(projection_clause)

            for index, tampered in enumerate(tamper_cases):
                path = root / f"tampered-{index}.json"
                self.write_json(path, self.rehash(tampered))
                with self.subTest(index=index), self.assertRaises(FunctionalError):
                    read_functional_model(path)

    def test_invalid_binding_duplicate_completeness_and_no_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context, _ = self.export(root)
            spec = self.spec(context)
            bad_binding = copy.deepcopy(spec)
            bad_binding["source_binding"]["urdf_semantic_sha256"] = "0" * 64
            path = root / "bad-binding.json"
            self.write_json(path, bad_binding)
            with self.assertRaisesRegex(FunctionalError, "urdf_semantic_sha256 mismatch"):
                build_functional_model_from_context(context, path)

            duplicate = copy.deepcopy(spec)
            duplicate["inventory_completeness"].append(copy.deepcopy(duplicate["inventory_completeness"][0]))
            path = root / "duplicate.json"
            self.write_json(path, duplicate)
            with self.assertRaisesRegex(FunctionalError, "repeats subject/inventory"):
                build_functional_model_from_context(context, path)

            no_capabilities = copy.deepcopy(spec)
            no_capabilities["capabilities"] = []
            no_capabilities["affordances"] = []
            no_capabilities["inventory_completeness"] = []
            path = root / "no-capabilities.json"
            self.write_json(path, no_capabilities)
            model = build_functional_model_from_context(context, path)
            self.assertEqual(model["status"], "no_capabilities_declared")
            self.assertTrue(model["coverage"]["all_declared_capabilities_structurally_grounded"])

    def test_prepare_provenance_and_raw_evaluation_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bootstrap, _ = self.export(root)
            spec_path = root / "functional-spec.json"
            self.write_json(spec_path, self.spec(bootstrap))
            prepared = root / "prepared"
            response = json.loads(self.run_cli(
                "prepare",
                FIXTURES / "mimic_branch.urdf",
                "--workspace-samples",
                "0",
                "--functional-spec",
                spec_path,
                "--out",
                prepared,
            ).stdout)
            self.assertEqual(response["status"], "prepared")
            self.assertEqual(response["artifacts"]["functional_model"]["path"], "context/functional-model.json")
            source_manifest = json.loads((prepared / "source-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(source_manifest["functional_knowledge"]["sha256"], hashlib.sha256(spec_path.read_bytes()).hexdigest())
            compiled = json.loads((prepared / "context" / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(compiled["source_compilation"]["functional_knowledge"]["sha256"], source_manifest["functional_knowledge"]["sha256"])

            task_root = root / "raw-task"
            source_root = task_root / "source"
            source_root.mkdir(parents=True)
            (source_root / "robot.urdf").write_bytes((FIXTURES / "mimic_branch.urdf").read_bytes())
            (source_root / "functional-spec.json").write_bytes(spec_path.read_bytes())
            task = {
                "schema_version": "robot-spatial-raw-source-task.v1",
                "workflow": "direct",
                "input_format": "urdf",
                "entrypoint": "source/robot.urdf",
                "export_options": {"functional_spec": "source/functional-spec.json", "workspace_samples": 0},
            }
            spatial_evaluation_suite._validate_raw_task_spec(task, task_root, "task")
            copied = spatial_evaluation_suite._copy_raw_source_tree(source_root, task_root / "public-source")
            self.assertIn(task_root / "public-source" / "functional-spec.json", copied)
            (source_root / "functional-model.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(spatial_evaluation_suite.EvaluationSuiteError, "generated context artifact"):
                spatial_evaluation_suite._copy_raw_source_tree(source_root, task_root / "forbidden-copy")

    def test_dependency_free_oracle_smoke(self):
        process = subprocess.run(
            [sys.executable, str(ORACLE), "--cases", "3", "--seed", "9231"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(process.stdout)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["independence"]["production_modules_imported"], [])
        self.assertEqual(report["counts"]["requirement_result_count"], 21)
        self.assertEqual(report["counts"]["query_count"], 9)


if __name__ == "__main__":
    unittest.main()
