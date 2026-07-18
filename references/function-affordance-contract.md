# Function and affordance contract

This contract adds explicit, proof-carrying function knowledge above the structural concept graph.

It answers questions such as “what is this component for?”, “what structural relations enable this capability?”, and “what action does this robot afford on this type of object?” without guessing from link names, mesh shapes, or topology.

## Contents

1. [Layer model](#layer-model)
2. [Core distinctions](#core-distinctions)
3. [Input artifact](#input-artifact)
4. [Typed identities](#typed-identities)
5. [Source binding](#source-binding)
6. [Declaration records](#declaration-records)
7. [Structural requirement language](#structural-requirement-language)
8. [Requirement status and modality](#requirement-status-and-modality)
9. [Compiled functional model](#compiled-functional-model)
10. [Query contract](#query-contract)
11. [Negative and unknown answers](#negative-and-unknown-answers)
12. [Proof closure](#proof-closure)
13. [Verification](#verification)
14. [Agent-context integration](#agent-context-integration)
15. [Evaluation](#evaluation)
16. [Authoring workflow](#authoring-workflow)
17. [Failure cases](#failure-cases)
18. [Epistemic boundary](#epistemic-boundary)

## Layer model

The representation has four relevant layers:

1. URDF and articulation artifacts describe typed bodies, joints, frames, axes, limits, and pose-independent kinematic laws.
2. `concept-graph.json` composes those artifacts into proof-carrying structural concepts.
3. A project-owned function/affordance specification declares intended function knowledge.
4. `functional-model.json` binds those declarations to the exact concept graph and evaluates their typed structural requirements.

The compiler never promotes layer 3 declarations into source-derived physical facts.

The compiler never infers layer 3 from layers 1 or 2.

The functional layer is standalone instead of being inserted into the concept graph. This avoids a digest cycle and preserves a clean distinction between exact structural knowledge and project-declared intent.

## Core distinctions

### Component

A component is a project-declared grouping of exact concept entities.

Membership is asserted even when every member ID is exact.

Topology alone does not prove that a set of links is one functional component.

### Function

A function states an intended contribution or purpose, such as retaining an object or sensing a workspace region.

A function is not an action instance and is not inferred from a component name.

### Capability

A capability states that a provider is intended to realize one or more declared functions, subject to named enabling requirements and conditions.

The compiler may establish that its structural requirements are grounded.

It does not establish that the real robot can successfully execute the capability.

### Condition

A condition is a named symbolic predicate plus a declared truth source.

Its arguments are symbolic tokens. They are not automatically concept entities or runtime bindings.

The compiler records the condition but does not evaluate its current truth.

### Effect

An effect is a declared intended postcondition.

It is never an observed effect merely because it appears in the specification.

### Affordance

An affordance is a relational contract:

`actor/provider + action + target object type + intended effect under named conditions`

An affordance is not a property stored in an object in isolation.

It is not an unconditional permission or safety certificate.

## Input artifact

The exact schema version is:

`robot-spatial-function-affordance-spec.v1`

The top-level object contains exactly:

- `schema_version`
- `function_set_id`
- `source_binding`
- `object_types`
- `components`
- `functions`
- `conditions`
- `effects`
- `capabilities`
- `affordances`
- `inventory_completeness`

Unknown fields are rejected.

All arrays are explicit. Use an empty array when a category is intentionally not declared.

## Typed identities

Use typed, stable IDs:

- `function_set/<id>`
- `object_type/<id>`
- `component/<id>`
- `function/<id>`
- `condition/<id>`
- `effect/<id>`
- `capability/<id>`
- `affordance/<id>`
- `requirement/<id>`

Structural references must use the exact typed IDs from `concept-graph.json`, for example:

- `link/tool_link`
- `joint/finger_joint`
- `frame/tool_link`
- `articulation_variable/<grammar-id>/<driver>`

Do not substitute bare names when authoring a specification.

## Source binding

`source_binding` contains exactly:

- `urdf_semantic_sha256`
- `articulation_grammar_sha256`
- `constraint_graph_sha256`
- `configuration_atlas_sha256`

Use `null` for an optional constraint graph or configuration atlas that is not present.

The compiler rejects any mismatch.

The compiled model additionally binds:

- the function specification artifact digest;
- the concept graph semantic digest;
- the concept graph artifact digest;
- the concept graph typed ID.

This prevents a function declaration authored for one mechanism revision from silently grounding against another.

## Declaration records

### Object type

Required fields:

- `object_type_id`
- `meaning`

Object types are project vocabulary. A declared type is not a geometry classifier.

### Component

Required fields:

- `component_id`
- `members`
- `meaning`

Every member must exist in the bound concept graph.

### Function

Required fields:

- `function_id`
- `provided_by`
- `verb`
- `object_types`
- `purpose`

Providers may be exact structural concept entities or declared components.

### Condition

Required fields:

- `condition_id`
- `predicate`
- `arguments`
- `truth_source`
- `meaning`

Allowed truth sources are:

- `runtime_observation_required`
- `planner_verification_required`
- `operator_confirmation_required`
- `project_assumption`

These labels identify who or what must establish truth. They do not establish truth themselves.

### Effect

Required fields:

- `effect_id`
- `predicate`
- `arguments`
- `meaning`

The compiler always emits `observed_effect: false`.

### Capability

Required fields:

- `capability_id`
- `provided_by`
- `realizes_functions`
- `enabling_requirements`
- `condition_refs`
- `limitations`

Every capability must include at least one typed enabling requirement.

Limitations should name important excluded truths, such as friction, force closure, payload, controller readiness, calibration, or hardware health.

### Affordance

Required fields:

- `affordance_id`
- `offered_by`
- `action_verb`
- `target_object_types`
- `capability_refs`
- `precondition_refs`
- `effect_refs`
- `meaning`

At least one target type, capability, effect, and provider is required.

### Inventory completeness

Required fields:

- `subject`
- `inventories`
- `scope`

Allowed inventory names are:

- `functions`
- `capabilities`
- `affordances`

Completeness is project-scoped. Declare the scope narrowly and honestly.

The same subject/inventory pair may be declared only once.

## Structural requirement language

The v1 compiler supports these requirement types:

| Type | Parameters | Closed/open-world behavior |
|---|---|---|
| `entity_exists` | `entity` | exact closed-world concept entity inventory |
| `kinematic_path_exists` | `from_link`, `to_link` | exact closed-world canonical tree |
| `driver_drives_joint` | `driver`, `joint` | exact closed-world articulation projection |
| `driver_affects_frame` | `driver`, `frame` | exact closed-world articulation projection |
| `frame_has_asserted_role` | `frame`, `role` | open-world project assertion |
| `constraint_declared` | `constraint` | open-world supplemental assertion |
| `finite_configuration_witness_exists` | `chart` | finite computed evidence only |

Each requirement has exactly:

- `requirement_id`
- `type`
- `parameters`

The compiler validates typed parameter syntax before evaluating the requirement.

## Requirement status and modality

Each result contains:

- `status`
- `satisfied`
- `evidence.exact`
- `evidence.modality`
- `evidence.concept_clause_ids`
- `evidence.closure_basis`
- `evidence.explanation`

Possible statuses are:

- `satisfied`
- `not_satisfied_exact_closed_world`
- `not_established_open_world`

A finite configuration witness can satisfy a requirement, but it remains finite computed evidence rather than a global proof.

The capability-level status is:

- `all_declared_requirements_grounded`, or
- `one_or_more_declared_requirements_not_grounded`.

The model-level status additionally distinguishes:

- `no_capabilities_declared`.

## Compiled functional model

The exact schema version is:

`robot-spatial-functional-model.v1`

The model contains:

- digest-bound source bindings;
- typed functional entities;
- content-addressed functional clauses;
- recursive structural evidence clauses copied from the concept graph;
- exact lookup indexes;
- normalized projections;
- coverage counts;
- ontology and query contracts;
- an epistemic scope statement;
- a semantic digest.

Functional clause IDs are derived from canonical clause content.

Structural concept clause IDs are revalidated after embedding.

The structural evidence set contains the recursive premise closure needed by capability requirements, not merely the leaf clauses named by a requirement.

## Query contract

Queries use:

`robot-spatial-functional-query.v1`

The query object contains exactly:

- `schema_version`
- `query_id`
- `intent`
- `parameters`

Supported intents are:

- `describe_component`
- `explain_function`
- `explain_capability`
- `explain_affordance`
- `what_is_entity_for`
- `can_perform_action`

`can_perform_action` takes:

- `offered_by`
- `action_verb`
- `target_object_type`

A positive match returns:

`declared_possible_if_preconditions_hold`

This conclusion requires at least one matching affordance whose referenced capabilities all have grounded declared structural requirements.

If affordance declarations match but none has all capability requirements grounded, the conclusion is:

`declared_affordance_with_ungrounded_capability_requirements`

Both forms also return unevaluated preconditions, intended effect references, per-capability grounding, and `physical_executability: not_established`.

## Negative and unknown answers

When no affordance matches and the provider's affordance inventory is explicitly complete, the answer is:

`not_declared_in_complete_project_inventory`

When no affordance matches and completeness is absent, the answer is:

`unknown_not_in_incomplete_inventory`

Neither answer establishes physical impossibility.

Missing function knowledge for an entity is never filled by name-based inference.

## Proof closure

Every answer contains:

- `supporting_clauses`
- `structural_supporting_clauses`

The first list contains the recursive functional premise closure.

The second list contains the recursive concept-graph premise closure reached from the selected functional clauses.

Preserve clause modality when translating an answer into natural language.

Do not cite a supporting structural clause as proof of the project function assertion itself.

## Verification

`verify-functional-model` performs:

1. semantic digest validation;
2. typed entity and content-addressed clause validation;
3. functional and structural proof reference validation;
4. proof-cycle rejection;
5. exact index recomputation;
6. projection-to-clause consistency checks;
7. capability status and coverage recomputation;
8. source/spec-bound exact regeneration;
9. byte-for-byte canonical comparison.

Exact regeneration verifies consistency with the same project inputs.

It is not an independent physical or functional oracle.

## Agent-context integration

When a functional model is present, `agent-context.json` adds:

- a `functional_model` artifact binding;
- a `query-functions` load-order step;
- a question-router entry;
- typed component/function/capability/condition/effect/affordance/object-type cards;
- explicit unresolved physical-execution and safety claims.

When absent, the context explicitly says not to infer purpose from names.

The functional card trust class is:

`project_asserted_with_digest_bound_proof_model`

This is intentionally distinct from exact URDF facts.

## Evaluation

A generated evaluation adds five functional competencies when a model is present:

1. explicit component/function comprehension without name inference;
2. capability declaration versus structural grounding versus physical truth;
3. relational affordance, precondition source, and intended-effect semantics;
4. positive action-contract composition with grounded-versus-ungrounded capability conclusions;
5. complete/incomplete inventory boundary without physical-impossibility overclaim.

Each public functional question includes the exact answer-object field contract with typed placeholders. The contract reveals no expected values and does not prescribe array length; it prevents semantically equivalent but grader-incompatible field renaming.

Generated-context suites copy `functional-model.json` as a public representation artifact.

Raw-source suites may include the project function specification but must reject a precompiled `functional-model.json`.

## Authoring workflow

1. Export or prepare the robot without a functional spec.
2. Read `model.json`, `concept-graph.json`, and relevant `query-concepts` answers.
3. Copy exact semantic and artifact digests into `source_binding`.
4. Declare object vocabulary and project components.
5. Declare functions before capabilities.
6. Declare every runtime, planner, operator, or project-assumption condition explicitly.
7. Declare intended effects without execution language.
8. Add typed structural requirements to each capability.
9. Add relational affordances that reference the capability, conditions, effects, and target types.
10. Declare inventory completeness only where the project can defend it.
11. Re-export with `--functional-spec`.
12. Inspect every unsatisfied or open-world requirement.
13. Run `query-functions` for the intended user questions.
14. Run `verify-functional-model` before publishing or benchmarking the artifact.

## Failure cases

The compiler rejects:

- wrong schema versions;
- unknown or extra fields;
- duplicate IDs;
- duplicate list entries;
- unknown providers, members, functions, conditions, effects, capabilities, or target types;
- malformed typed requirement parameters;
- unsupported requirement types or truth sources;
- capabilities without requirements;
- affordances without providers, targets, capabilities, or effects;
- repeated inventory-completeness coverage;
- source binding mismatches;
- malformed concept evidence;
- internally inconsistent or tampered compiled models.

An export with one or more ungrounded capability requirements writes the diagnostic artifacts but exits non-zero with:

`exported_with_ungrounded_functional_requirements`

## Epistemic boundary

The functional model can establish:

- exactly what the project declared;
- exactly which typed structural requirements match the bound concept graph;
- exactly which recursive structural clauses support a satisfied requirement;
- exactly whether an affordance is absent from a declared complete project inventory.

It cannot establish by itself:

- current runtime condition truth;
- object classification in a camera or world scene;
- physical force closure or contact mechanics;
- payload capacity;
- collision-free execution;
- controller configuration or readiness;
- hardware health, calibration, or connectivity;
- action success;
- an observed effect;
- safety;
- physical impossibility.

Add separate observation, planning, dynamics, control, hardware, execution, and safety evidence layers before making those claims. For one declared action instance, [action-assurance-contract.md](action-assurance-contract.md) supplies the digest-bound time-selection, action-server lifecycle, declared-effect observation, discrepancy, and causal-boundary layer. It still does not establish physical execution or safety.
