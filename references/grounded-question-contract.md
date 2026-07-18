# Grounded natural-language spatial question contract

## Purpose

Natural language is the user interface, not the source of geometric truth. Interpret the user's wording, then ground every spatial claim in a typed fact or deterministic query result. Answer in the user's language while preserving frame, pose, units, direction, and uncertainty.

## Interpretation protocol

For each question:

1. Identify the requested relation: compositional structure, identity, project-declared component function/capability/affordance, time-qualified action readiness/lifecycle/effect evidence, tree or supplemental topology, constraint satisfaction/local mobility, finite branch/rank witness, causality, robot-local transform, scene-bound transform, axis, instantaneous motion, declared mass properties, root-convention or world-scene gravity loads, embedded actuation declarations, geometry, self/environment collision, workspace, or semantic intent.
2. Resolve every name to a typed entity. Do not silently choose between `link/X`, `frame/X`, and snapshot-bound `robot_frame/X`; ask or state both interpretations when the distinction changes the answer.
3. Determine whether the answer is pose- and snapshot-independent. If not, name the explicit pose and, for world questions, the bound `scene_id`, `snapshot.id`, and scene digest.
4. Route symbolic structural composition first to `query-concepts`; route function, capability, condition, intended-effect, affordance, and declared action-possibility questions to `query-functions`; route one bound action instance's readiness, selected condition evidence, goal/status/result lifecycle, declared-effect observation, and discrepancies to `query-action-assurance`; route atomic retrieval or numeric evaluation to `retrieve`, `chain`, `constraint-graph`, `evaluate-constraints`, `solve-constraints`, `configuration-atlas`, `verify-configuration-atlas`, `affects`, `transform`, `axis`, `jacobian`, `mass-properties`, `gravity-loads`, `actuation`, `bounds`, `surface-collisions`, `scene-transform`, `scene-gravity-loads`, `scene-collisions`, or `workspace`. Do not perform proof closure, evidence time selection, graph closure, transform, constraint residual, rank, center-of-mass, inertia, gravity-load, or collision arithmetic in prose.
5. Classify each claim with the controlled vocabulary below. Do not invent a synonym when producing machine-readable output.
6. Answer the question directly, then state the minimum frame/pose/unit/evidence qualifiers needed to prevent a false interpretation.

The deterministic CLI adds `robot-spatial-query-evidence.v1` to fresh query results. Preserve its `query_id`, source URDF digest, method, parameters, and epistemic scope with the answer. For retrieved facts, preserve `fact_id` and the fact's evidence record.

`query-concepts` instead returns `robot-spatial-concept-answer.v1`. Preserve its concept-graph ID/digest, strict query ID/intent, selected projection coverage, answer status, unknowns, and the complete returned recursive proof closure. A clause modality is part of the claim: `supplemental_asserted_relation` must not be paraphrased as deterministic or physical truth, while `derived_exact_from_asserted_relation` is exact only conditional on the asserted relation. Return `false` only when the query is inside an explicitly complete closed-world tree/articulation projection; undeclared roles, supplemental relations, physical behavior, global configuration topology, and safety remain unknown.

`query-functions` returns `robot-spatial-functional-answer.v1`. Preserve its functional-model ID/digest, query ID/intent, project assertions, typed requirement statuses and closure bases, functional and structural proof closures, condition truth sources, intended-effect status, inventory scope, unknowns, and physical-execution boundary. `declared_possible_if_preconditions_hold` is conditional project knowledge, not a runtime or physical yes. `not_declared_in_complete_project_inventory` is a scoped negative declaration, not physical impossibility. Without explicit completeness, return `unknown_not_in_incomplete_inventory`.

`query-action-assurance` returns `robot-spatial-action-assurance-answer.v1`. Preserve its assurance ID/digest, action instance and exact bindings, clock, decision/evaluation times, selected and ignored evidence record IDs, condition selection status, bounded readiness, dispatch-authorization boundary, lifecycle consistency, goal/status/result separation, execution-start observation, effect timing, discrepancies, producer/source provenance, and causal/physical/safety unknowns. `ready_under_declared_model_and_evidence` is not authorization. `result_succeeded` is a protocol report, not physical success. A positive effect observation after observed execution start does not establish causation.

## Grounded answer shape

Use prose for ordinary interaction. Use this auditable shape when the user requests machine-readable output or when running a free-form benchmark:

```json
{
  "schema_version": "robot-spatial-grounded-answer.v1",
  "interpretation": {
    "intent": "kinematic_causality",
    "entities": ["joint/wrist", "frame/camera"],
    "pose": "inspection"
  },
  "answer": "Yes. Changing wrist can change the camera frame pose.",
  "claims": [
    {
      "claim": "joint/wrist can change frame/camera",
      "status": "exact",
      "evidence_refs": ["fact-...", "query-..."]
    }
  ],
  "unknowns": []
}
```

Evidence references are auditable only when the corresponding context pack or complete query output is preserved. A fluent sentence without that binding is an explanation, not verified understanding.

## Controlled status vocabulary

Machine-readable answers use these exact tokens:

- `exact`: a deterministic result establishes the claim within the represented model.
- `asserted`: a project annotation or planning-semantic file explicitly declares the claim.
- `sampled`: the claim is supported only by the recorded finite sample.
- `indeterminate`: the requested relation is meaningful, but the available evidence is incomplete or the analysis was not run.
- `not_provided`: an optional evidence layer such as SRDF or semantic annotations is absent.
- `not_established`: the available description does not explicitly establish a requested semantic role or intent. A suggestive name is not enough.
- `unsupported`: the current deterministic engine or representation cannot evaluate the requested operation or content type.

These tokens are not interchangeable. In particular, use `not_established` for an undeclared TCP role, not `unsupported`; use `unsupported` when the engine lacks a needed capability; and use `indeterminate` when the capability exists but required evidence is incomplete. A benchmark question may narrow a field to a smaller explicit subset, such as `provided|not_provided` or `asserted|not_established`; obey that field-level enum exactly.

## Ambiguity and absence

- If no pose is supplied for a pose-dependent question, use the exported pose only when the question permits it; otherwise request a pose.
- If “direction” could mean a URDF-local joint axis or that axis expressed in another frame, state the distinction and use `axis` for the requested observer frame.
- If the user asks whether two parts collide and surface analysis is absent or incomplete, answer `indeterminate`, not “no.”
- If the user asks how many closed-chain branches or singularities exist, require an explicit configuration atlas before making even a finite claim. Report exact samples/seeds/scales/thresholds and deficient samples; call stored modes finite witnesses, components proximity components, and rank changes candidates. Do not convert finite coverage into a global count or certificate.
- If the user asks where the robot is in the room/world and no scene root mount is supplied, answer `not_provided`; robot-local `world_from_frame` is not physical world evidence.
- If the user asks about current surroundings, first state that a supplied scene is one static snapshot. `captured_at` and `valid_until` are declarations, not proof that the snapshot is current. Omitted objects are unknown.
- If the user asks whether the robot is collision-free with the environment, require all declared robot/environment pairs to be resolved. Preserve `indeterminate` for overlapping unsupported cylinders, unmeasured geometry, incomplete solids, or partial coverage. A disjoint AABB proves one pair is collision-free but gives only a clearance lower bound.
- If the user asks what a component is “for,” require an explicit functional model and use `query-functions`. Distinguish the project-declared component grouping and purpose from structural evidence that only grounds its enabling requirements. Without a model, report `not_provided`; never answer from the link/component name, topology, or mesh appearance.
- If the user asks whether the robot “can” perform an action, report the matched relational affordance, actor/provider, exact action verb, target object type, named preconditions and truth sources, intended effects, and capability grounding. Unless separate execution evidence exists, state `current_preconditions_satisfied: not_evaluated` and `physical_executability: not_established`.
- If the user asks whether one action is ready “now,” require a bound action assurance. Name its clock and decision time; report every condition's required evidence type, selected status, selected record IDs, and stale/future/conflict exclusions. Preserve `authorization_to_dispatch: not_provided`, `physical_executability: not_established`, and `safety: not_established` even when all declared preconditions are satisfied.
- If the user asks whether an attempted action succeeded, report goal response, latest status, terminal result, lifecycle consistency, observed execution-start time, declared-effect observation and its time relation, and all discrepancies. Never convert an action-server `succeeded` result into physical or causal success. Never convert temporal succession into causation.
- If an action is missing, check exact provider-specific inventory completeness. A complete project inventory permits only `not_declared_in_complete_project_inventory`; an incomplete inventory requires unknown. Neither status implies physical impossibility.
- If the user asks for mass, center of mass, or inertia, say that the values cover URDF-declared inertials, name the selected tree, pose, and expression frame, and report missing/invalid inertial links. Do not infer mass from visual/collision geometry or treat absence as zero physical mass.
- If the user asks for gravity force/torque or holding effort, name the pose, selected tree, gravity vector and frame, independent driver, sign convention, and N versus N·m unit. Report missing/invalid inertials. Call it a declared gravity-only static model, not inverse dynamics or actuator feasibility.
- If gravity comes from a world scene, also name the source scene frame, transformed root-frame vector, root mounting, snapshot, and scene digest. Do not mix it with the canonical root-frame gravity convention.
- If the user asks what can be commanded, sensed, controlled, or actuated, use `actuation` and say exactly what the expanded URDF declares. Do not turn a command interface, plugin string, mechanical reduction, `calculate_dynamics` parameter, or dynamics attribute into a runtime/hardware claim. Absence of an embedded declaration is `not_declared_in_expanded_urdf`, not proof of an unactuated robot.
- If a semantic role or intent is absent from URDF/SRDF/project assertions, report `not_established` by the available description; do not turn absence into a physical-world fact. Reserve `unsupported` for an operation or representation the engine cannot evaluate.

## Free-form evaluation

Do not derive all evaluation wording from the same generator that created the context. Author paraphrased, compositional, negative, ambiguous, and epistemic-boundary questions independently. Bind private expected claims to deterministic facts/query outputs, require evidence references, and include controls where a plausible but ungrounded answer must fail. Report the exact source, pose, capability, runtime, and isolation boundary.
