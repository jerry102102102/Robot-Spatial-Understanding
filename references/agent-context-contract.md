# Agent spatial context contract

## Purpose

Do not present a language model with raw URDF as if XML syntax were spatial understanding. `export` compiles one expanded URDF into a pose-independent articulation grammar, a proof-carrying concept graph, one evaluated pose, declared inertial/geometry state, optional function/affordance, world/observation/motion layers, SRDF, semantics, and invariants. The artifacts are digest-bound in the rules-first manifest; six context layers expose them progressively:

1. `agent-context.json`: small rules-first manifest;
2. `concept-graph.json` queried through `query-concepts`: typed compositional clauses and minimal proof closures; `concept-language.rsl` is its optional whole-robot controlled rendering;
3. optional `functional-model.json` queried through `query-functions`: project-declared function/capability/affordance clauses plus recursive structural proof closures;
4. `entity-cards.jsonl`: one controlled-language and structured card per typed entity;
5. `facts.jsonl`: provenance-bearing atomic relations;
6. `model.json`: complete canonical machine representation.

This order is deliberate. A model should load rules and unresolved boundaries first, one entity card second, only its bound facts third, and the full model only when a deterministic query cannot answer the task more narrowly.

## Identity and non-collapse rules

The manifest schema is `robot-spatial-agent-context.v1`. Every entity uses a typed ID:

- `robot/<name>`
- `link/<name>`
- `joint/<name>`
- `frame/<exact-frame-name>`
- `group/<name>` and `srdf_group/<name>`
- `end_effector/<name>` and `srdf_end_effector/<name>`
- `invariant/<assertion-id>`
- `ros2_control_system/<system-name>`
- `transmission/<transmission-name>`
- `actuator/<transmission-name>/<actuator-name>`
- `control_sensor/<system-name>/<sensor-name>` and `control_gpio/<system-name>/<gpio-name>`
- `robot_instance/<instance-id>`
- `scene_frame/<name>`
- `robot_frame/<exact-URDF-frame-name>`
- `scene_object/<object-id>`
- `scene_geometry/<object-id>/<geometry-id>`
- `robot_geometry/<exact-URDF-geometry-frame-name>`
- `observation_log/<log-id>`
- `ros_capture/<capture-id>`
- `render_atlas/<render-id>`
- `render_view/<render-id>/<front|side|top|isometric>`
- `motion_atlas/<motion-id>`
- `motion_driver/<motion-id>/<independent-driver>`
- `motion_view/<motion-id>/<independent-driver>/<front|side|top|isometric>`
- `articulation_grammar/<grammar-id>`
- `articulation_variable/<grammar-id>/<independent-driver>`
- `articulation_operator/<grammar-id>/<physical-joint>`
- `articulation_derivation/<grammar-id>/<exact-frame-name>`
- `constraint_graph/<constraint-graph-id>`
- `attachment/<constraint-graph-id>/<attachment-id>`
- `constraint/<constraint-graph-id>/<constraint-id>`
- `configuration_atlas/<configuration-atlas-id>`
- `configuration_chart/<configuration-atlas-id>/<chart-id>`
- `configuration_node/<configuration-atlas-id>/<chart-id>/<sample-index>/<solution-index>`
- `configuration_component/<configuration-atlas-id>/<chart-id>/<component-index>`
- `concept_graph/<robot-name>/<binding-digest>`
- `serial_segment/<robot-name>/<segment-index>` inside the concept graph
- `functional_model/<function-set-name>/<binding-digest>`
- `object_type/<project-object-type>`
- `component/<project-component>`
- `function/<project-function>`
- `condition/<project-condition>`
- `effect/<project-effect>`
- `capability/<project-capability>`
- `affordance/<project-affordance>`

`link/tool0` is the URDF graph node. `frame/tool0` is the coordinate frame attached to that node. They are related but not interchangeable. The retrieval interface rejects an ambiguous bare name rather than guessing which typed entity the caller meant.

When a world scene is bound, `frame/tool0` remains robot-local while `robot_frame/tool0` is snapshot-bound. `robot_geometry/collision/tool/0` likewise names the mounted geometry used in robot/environment collision evidence. Scene identities never erase or rename their robot-local counterparts.

When observations are bound, the observation-log card preserves clock/query policy, sample selections, ages, future-sample exclusions, effective source layers, and nominal analysis. It does not replace the static scene card. An effective pose sourced from `static_scene_declaration` remains a scene hypothesis, not an observation.

When a v2 log comes from the ROS adapter, a separate `ros_capture/<capture-id>` card preserves capture/config digests, partial-joint assembly method, TF path reconstruction/age policy, clock limitation, visible authority policy, and its normalized log identity. It does not replace either the raw capture or observation card. A publisher GID or topic proves only transport attribution visible to the capture, not sensor truth.

When `--render` is used, `render_atlas/<render-id>` preserves exact model/pose/input binding and geometry coverage. Each `render_view/<render-id>/<view>` card preserves its root-XYZ-to-UV basis, UV-to-pixel mapping, projected frame origins, joint edges, geometry hull/depth records, SVG entity IDs, and artifact digest. These views are deterministic encodings of the canonical model, not independent geometry evidence or camera observations. Read [render-atlas-contract.md](render-atlas-contract.md) before using them.

When `--motion-atlas` is used, `motion_atlas/<motion-id>` preserves exact model/baseline-pose/step binding and coverage. Each `motion_driver/<motion-id>/<driver>` preserves its mimic-constrained feasible interval, driven physical joints, structural affected frames, signed endpoint status, all-frame SE(3) deltas, geometry endpoint AABBs, and four view identities. Each `motion_view` preserves one shared fit for baseline and available endpoints plus typed motion vectors. These are exact finite FK endpoint encodings, not a trajectory, swept volume, dynamics result, hardware observation, or safety proof. Read [motion-atlas-contract.md](motion-atlas-contract.md) before using them.

Every export includes `articulation_grammar/<grammar-id>`. Its grammar card keeps exact source/import binding separate from the source-binding-free canonical law identity; variable cards preserve independent domains and dependency constraints; operator cards preserve the pre-motion × motion × post-motion law and physical-joint equation; derivation cards preserve exact ordered root-to-frame composition. These are the general pose-independent law. FK, Jacobian, and motion-atlas records are respectively one evaluation, a local derivative, and finite samples. For cross-representation claims, preserve the digest-bound typed correspondence and comparison report. Read [articulation-grammar-contract.md](articulation-grammar-contract.md) and [cross-representation-contract.md](cross-representation-contract.md) before making a general or cross-format motion claim.

When `--constraint-spec` is used, `constraint_graph/<graph-id>` states that the tree is a coordinate parameterization and binds its supplemental mechanism relations. `attachment/<graph-id>/<attachment-id>` cards preserve rigid typed anchors; `constraint/<graph-id>/<constraint-id>` cards preserve asserted relation semantics and the export-pose typed residual separately. The graph card exposes local residual-Jacobian rank/mobility only as pose-conditioned numerical evidence. Read [constraint-graph-contract.md](constraint-graph-contract.md) before calling a tree pose mechanism-valid or reporting closed-chain mobility.

When `--configuration-atlas-spec` is also used, `configuration_atlas/<atlas-id>` preserves exact graph/spec binding, declared sampling status, coverage, and epistemic scope. Chart cards preserve parameter values, complete seeds, scales, merge/edge thresholds, minimum solution count, and observed rank reference. Node cards preserve executable satisfying bindings, typed residual maximum, and full/passive numerical singular diagnostics. Component cards preserve only finite declared proximity connectivity. Read [configuration-atlas-contract.md](configuration-atlas-contract.md) before using these records for branch or singularity language.

Every URDF export also writes `concept-graph.json` and `concept-language.rsl`. The graph composes the validated tree, articulation grammar, explicit project semantics, optional constraint graph, and optional configuration atlas into typed clauses with modality, evidence, proof premises, controlled language, and closed/open-world rules. Use `query-concepts` for structural summaries, unique tree paths, branch/leaf/serial abstractions, driver effects, frame laws, asserted constraint dependencies, and finite-node comparisons. It returns only the relevant recursive proof closure. Exact negatives are allowed only for complete validated tree or supported articulation projections; all undeclared semantic, physical, runtime, global-configuration, and safety absences remain unknown. Read [concept-language-contract.md](concept-language-contract.md) before treating this layer as a knowledge representation.

When `--functional-spec` is used, `functional-model.json` binds project-declared object types, components, functions, conditions, intended effects, capabilities, and relational affordances to the exact concept graph. Functional cards use `project_asserted_with_digest_bound_proof_model`, never an exact-URDF trust label. `query-functions` returns recursive functional premises plus every required structural concept premise. A satisfied typed requirement establishes a match inside represented structure only. It does not establish current preconditions, physical capability, observed effects, hardware execution, impossibility, or safety. When this artifact is absent, `unresolved_claims` explicitly forbids inferring function from names or geometry. Read [function-affordance-contract.md](function-affordance-contract.md).

## Entity cards

Each `robot-spatial-entity-card.v1` record contains:

- typed identity and entity class;
- `summary_cnl`, a deterministic controlled-natural-language index;
- structured `data` copied or derived from the canonical model;
- task-specific `tool_queries` for relationships not already exported;
- exact bound `fact_ids`;
- a trust classification and evidence-source counts.

The card is optimized for model comprehension but is not an independent oracle. Numeric or semantic claims should cite bound facts or a fresh deterministic tool result. Joint cards distinguish declaration and exported pose. Articulation cards distinguish the general variable law, physical operator, and composition syntax. Motion-driver cards distinguish a finite controlled sample. Link, scene, observation, and actuation cards preserve their respective identities and epistemic layers without upgrading declarations to runtime or physical truth.

## Fact index and integrity

`entity-index.json` uses `robot-spatial-entity-index.v1` and maps typed IDs and bare names to card byte ranges. Keeping it outside the manifest prevents the rules-first layer from growing linearly with robot size. `fact-index.json` uses `robot-spatial-fact-index.v1` and maps every fact ID, subject, predicate, and entity to records in `facts.jsonl`, with byte offsets and lengths. The manifest binds the guide, cards, both indexes, and facts by SHA-256.

Run:

```bash
python3 scripts/robot_spatial.py retrieve work/robot-spatial --entity joint/shoulder --compact
python3 scripts/robot_spatial.py retrieve work/robot-spatial --entity frame/tool0 --predicate has_pose --evidence exact
python3 scripts/robot_spatial.py retrieve work/robot-spatial --fact-id fact-0123456789abcdefabcd
python3 scripts/robot_spatial.py retrieve work/robot-spatial --list-entities
```

Retrieval validates every indexed artifact digest before using offsets. A modified cards, facts, or index file is an error. Entity filtering starts from the card's bound fact IDs, then optionally filters exact predicate, exported-pose qualifier, and exact/nonexact evidence class.

Use `--compact` when the result will be placed directly in model context. It changes only JSON formatting, not content or verification.

## Epistemic language

`evidence.exact=true` means the fact is deterministic within its declared source representation and tolerance: URDF declaration, validated graph derivation, forward kinematics, analytic geometry, measured mesh, or an explicitly reported verification engine.

`evidence.exact=false` does not mean false. It means one of:

- project/user assertion, such as TCP role;
- supplemental mechanism assertion, such as a loop-closure pair or coordinate coupling;
- heuristic interpretation, such as compact/elongated shape language;
- finite-sample observation, such as workspace bounds.

Read `evidence.source_type` and qualifiers before wording the answer. Absence, `not_provided`, `not_requested`, `not_inspected`, and `indeterminate` are unknown/incomplete states, never negative facts. `unavailable_at_feasible_limit` means no nonzero endpoint exists in that signed direction; it is not a missing record. `unresolved_claims` separates unmeasured geometry, missing counterfactual motion, unavailable signed endpoints, trajectory/dynamics/hardware motion, actual gravity/mass completeness, scene currency/completeness/calibration, observation freshness/source truth, and actuation runtime/hardware claims. Root-convention gravity is not physical mounting evidence; scene-bound gravity is conditional on the declared mount and snapshot. `current` observation status proves only compliance with the exact age policy. An omitted object is not evidence of empty space. These boundaries appear before detailed geometry, kinematics, or declared physics.

## Answer contract

Every spatial answer should identify:

1. evaluated pose, or that the relation is pose-independent;
2. reference and target frames with transform direction;
3. length/angle units and quaternion order when numeric;
4. supporting fact IDs or fresh deterministic tool output;
5. asserted, heuristic, sampled, indeterminate, or unsupported boundaries.

For an observation-conditioned answer, additionally identify the log/query digests, clock domain, query time, selected sample IDs and ages, maximum-age limits, required objects, fallback entities, and whether every required observation is current.

For a ROS-normalized answer, additionally identify capture/config digests, header-versus-receipt timestamp policy, component/TF-edge age limits, TF target path, publisher-identity visibility, and whether any conflict or stale path prevented normalization. Do not describe a matching clock-domain string as verified synchronization.

For a view-grounded answer, additionally identify render ID/input digest, pose binding, view basis, root/UV/pixel units and mapping, geometry coverage, projection support, and SVG digest. Use `verify-render` before treating pixels and numeric records as consistent. Do not infer visibility, occlusion, perspective, calibrated-camera pixels, or physical truth from the semantic hulls.

For a finite-motion answer, additionally identify motion ID/input digest, baseline pose, independent driver, physical mimic joints driven, signed requested and applied step with units, endpoint status, affected frame, root-frame and optional shared-screen displacement, and SVG digest. Use `verify-motion-atlas` before treating endpoint and view records as consistent. Distinguish this finite result from the infinitesimal Jacobian and from any unknown intermediate path.

For a general articulation answer, identify grammar ID/input/artifact digests, whether the relation is pose-independent, driver domain, physical joint equation, typed operator, ordered frame derivation, units, and tree-only boundary. For an evaluated answer, also state the supplied driver binding and resolved mimic values. Use `evaluate-articulation` for a new binding, `verify-articulation-grammar` for regeneration/FK consistency, and the independent oracle for implementation diversity.

For a supplemental-mechanism answer, identify graph/spec/grammar digests, the fact that the tree is a parameterization, attachment and constraint IDs, explicit pose, typed residual/tolerance/status, and whether the relation itself is asserted. Report numerical local mobility as pose-conditioned and singularity-sensitive, never as global DOF. For a local solution, state seed, fixed drivers, solved drivers, convergence/residuals, and possible branch dependence.

For a finite configuration-space answer, identify atlas/graph/spec digests, chart and parameter sample, complete explicit seeds, solved/fixed drivers, scales, merge/edge thresholds, declared minimum and actual coverage, exact node IDs/bindings/residuals, and full/passive rank diagnostics. Say “finite proximity component” and “rank-drop candidate relative to the maximum observed in this chart”; do not say global branch, connected component, or certified singularity without separate proof.

For a structural concept answer, identify concept graph ID/semantic digest, query ID/intent, the exact typed entities, selected projection coverage, and every supporting clause modality. Preserve the distinction between declared/derived, asserted, derived-from-assertion, and finite evidence. State why a negative is permitted by a complete closed-world projection; otherwise return unknown. Use a numeric CLI for transforms, axes, distances, Jacobians, collision, or observations rather than treating the symbolic clause text as a numeric oracle.

For a functional answer, identify functional model/spec/concept bindings, query ID/intent, exact provider/component/target IDs, project-asserted functions or affordances, every requirement status/modality/closure basis, named condition truth sources, intended effects, inventory-completeness scope, and physical-execution unknowns. Use `verify-functional-model` before trusting a copied model. Never transform a structural proof into proof that the intended function is physically realized.

Do not subtract translations to obtain a relative transform, collapse robot-local and scene-bound frames, infer axes from rendering, treat projection overlap as 3D collision, infer mass from meshes, average link origins to obtain center of mass, turn AABB overlap into collision, treat two finite endpoints or their AABB union as a swept volume, infer TCP/flange roles from names, use a future sample for an earlier query, call a static fallback observed, or treat a sampled workspace as complete reachability. For scene questions, state scene/snapshot digest binding, pose, typed frame direction, object coverage, tolerance, and whether each result is exact, lower-bound-only, or indeterminate. For observation questions, state temporal binding/readiness and keep nominal geometry separate from physical truth and safety. For gravity questions, state vector/frame, pose, sign, units, coverage, and gravity-only boundary. For actuation questions, say “declares” and withhold runtime conclusions.

## Scope

This context pack improves grounding, compositional structural and explicit functional reasoning, retrieval, identity discipline, explicit ignorance, and provenance. The concept graph does not infer component function or affordances. The optional functional model transcribes project declarations and grounds named structural requirements but does not infer undeclared roles, action plans, runtime state, physical truth, action success, or global configuration topology. The context pack also does not by itself prove that an unseen model can reason correctly. That claim still requires a fresh-agent blind evaluation with private keys, varied robot families, unchanged/collateral controls for edits, and task-specific deterministic outcomes.
