# Robot Spatial Concept Language Contract

Use this contract when the task asks how an AI should be shown robot structure, why one joint affects a frame, where branches or serial segments are, how a loop relation changes the meaning of the URDF tree, or which symbolic conclusions are exact, asserted, finite, or unknown.

## Contents

1. [Purpose and layer](#purpose-and-layer)
2. [Artifact and source binding](#artifact-and-source-binding)
3. [Typed ontology and identity](#typed-ontology-and-identity)
4. [Clause and proof contract](#clause-and-proof-contract)
5. [Closed-world and open-world reasoning](#closed-world-and-open-world-reasoning)
6. [Compiled structural concepts](#compiled-structural-concepts)
7. [Executable query AST](#executable-query-ast)
8. [Controlled-language rendering](#controlled-language-rendering)
9. [Verification](#verification)
10. [AI use order](#ai-use-order)
11. [Boundary](#boundary)

## Purpose and layer

Raw URDF XML is authoring syntax. `model.json` is canonical machine state. `articulation-grammar.json` is the pose-independent executable kinematic law. Supplemental constraint and configuration artifacts describe additional mechanism relations and finite witnesses. None alone is an efficient compositional language for an AI.

Every export therefore compiles:

- `concept-graph.json`, schema `robot-spatial-concept-graph.v1`;
- `concept-language.rsl`, language `RSC-LANG/1`.

The graph provides typed entities, composable clauses, evidence modality, proof premises, complete projections, and a strict query interface. The `.rsl` file renders the same clauses as one controlled symbolic text. It is meant to give an AI the equivalent of a small formal language for the robot rather than another prose summary.

The concept layer does not replace numeric tools. It tells the AI which deterministic program or artifact defines a relation, how concepts compose, and when a negative answer is justified.

## Artifact and source binding

`source_binding` names:

- robot and root-link identity;
- exact URDF semantic digest;
- articulation grammar schema, ID, artifact digest, and source-binding-free law digest;
- optional constraint graph schema, ID, artifact digest, and semantic digest;
- optional configuration atlas schema, ID, artifact digest, and semantic digest.

`concept_graph_sha256` is the canonical semantic digest of the complete graph excluding that digest field. The exported model and agent manifest separately bind the exact `concept-graph.json` bytes and `concept-language.rsl` bytes.

Preserve both levels:

- semantic digest identifies graph meaning;
- artifact SHA-256 detects byte changes in the exported context.

Do not accept a graph copied from another model merely because entity names match.

## Typed ontology and identity

Concept entities always use typed IDs. Important examples are:

```text
robot/<robot-name>
link/<link-name>
joint/<joint-name>
frame/<exact-frame-name>
articulation_variable/<grammar-id>/<driver>
constraint/<constraint-graph-id>/<constraint-id>
configuration_node/<atlas-id>/<chart-id>/<sample-index>/<solution-index>
serial_segment/<robot-name>/<segment-index>
```

Identity is semantic, not string similarity:

- `link/tool0` is a rigid-body graph node;
- `frame/tool0` is a coordinate frame;
- `joint/tool0` would be a separate joint if declared;
- a structural leaf is not automatically an end effector;
- a filename or name token never establishes component function.

Every entity record contains `concept_types` and source JSON-pointer paths. Entity types are indexes; numeric and relational claims still require clauses or fresh deterministic query output.

## Clause and proof contract

Every clause contains exactly:

```json
{
  "clause_id": "concept_clause/<predicate>/<content-digest>",
  "predicate": "can_change_pose_of_frame_relative_to_root",
  "subject": "articulation_variable/<grammar-id>/shoulder",
  "object": {},
  "modality": "derived_exact",
  "scope": "pose_independent_structural_causality",
  "evidence": {
    "exact": true,
    "source_type": "executable_articulation_dependency_graph",
    "source_paths": []
  },
  "proof": {
    "rule": "driver_occurs_in_ordered_frame_composition",
    "premise_clause_ids": []
  },
  "cnl": "DRIVER ... CAN CHANGE POSE OF ..."
}
```

The clause ID binds all other clause fields. Changing its object, modality, evidence, proof, or controlled sentence without changing the ID is invalid.

Modalities are deliberately distinct:

| Modality | Meaning |
| --- | --- |
| `declared_exact` | Exact transcription from the validated supported source model |
| `derived_exact` | Exact graph or articulation consequence inside the declared representation |
| `project_asserted` | Explicit project semantic intent; not inferred from names |
| `supplemental_asserted_structure` | Explicit supplemental mechanism structure |
| `supplemental_asserted_relation` | Explicit mechanism constraint; not observed physical truth |
| `derived_exact_from_asserted_relation` | Exact dependency consequence conditional on an asserted relation |
| `finite_computed_evidence` | Re-executable finite sampling or node evidence |

A proof closure is the selected clauses plus every recursive `premise_clause_id`. `query-concepts` returns this minimal closure so an AI can explain why a result follows without loading every model record.

## Closed-world and open-world reasoning

The concept language has one mandatory negative rule:

> Return false only inside an explicitly complete closed-world projection; otherwise return unknown.

Closed-world domains in version 1 are:

- the complete validated URDF link and joint tree;
- every articulation independent driver;
- every physical joint operator in the supported grammar;
- every supported frame derivation when grammar coverage says all are present.

Examples of justified exact negatives:

- a declared link has no outgoing joint and is therefore a structural leaf;
- a driver does not occur in a frame's complete ordered derivation and therefore cannot change that frame relative to the root while other independent drivers are fixed;
- a joint is not on the unique path between two links in the complete validated tree.

Open-world domains include:

- undeclared semantic roles and component function;
- undeclared loop, contact, compliance, or coupling relations;
- physical construction, calibration, payload, environment, runtime, and hardware behavior;
- global configuration branches, topology, connectivity, reachability, and certified singularities;
- safety.

Absence in an open-world domain is not false. It is `unknown`, `not_provided`, `not_established`, or `unsupported`, depending on the missing layer.

## Compiled structural concepts

### Topology

The topology projection compiles:

- unique root;
- every typed parent-to-child joint edge;
- transitive `is_descendant_of` relations with ordered joint paths;
- exact branch points from link out-degree greater than one;
- exact structural leaves from out-degree zero;
- maximal serial segments between root, branch, and leaf boundaries;
- complete link, joint, and tree coverage flags.

Maximal serial segments are abstractions over exact edges. They do not invent modules, arms, fingers, or end effectors.

### Articulation

The articulation projection compiles:

- every independent driver and complete mimic-constrained feasible domain;
- every physical joint driven and its affine position equation;
- every frame whose root-relative pose may change under that driver with other independent drivers fixed;
- every normalized typed joint motion operator;
- every ordered root-to-frame pose composition law.

`can_change_pose_of_frame_relative_to_root` is structural causality. It does not claim a command occurred, a controller exists, a collision-free trajectory exists, or hardware moved.

### Project semantics

Only explicit annotations create `has_asserted_semantic_role` clauses. These clauses use `project_asserted` and `exact=false`. A frame named `tool0`, `flange`, `camera`, or `base` has no semantic role unless a project artifact declares it.

### Supplemental mechanism

When a constraint graph is present, the concept graph records:

- that the URDF tree may be only a coordinate parameterization;
- rigid attachment frames;
- typed asserted mechanism constraints;
- driver dependencies derived from each constraint's frames and coordinates.

The constraint relation remains asserted. Its driver dependency can be an exact consequence of the asserted relation plus the exact articulation grammar. Do not collapse these modalities.

### Finite configuration evidence

When a configuration atlas is present, the concept graph records:

- chart contract and declared sampling coverage;
- executable satisfying configuration witnesses;
- full and passive numerical ranks and candidate labels;
- finite proximity components.

It also records explicit false boundaries:

- `global_branch_topology_certified=false`;
- `certified_singularity=false`;
- `finite_proximity_component_is_global_branch=false`.

## Executable query AST

Every query has exactly:

```json
{
  "schema_version": "robot-spatial-concept-query.v1",
  "query_id": "why-shoulder-moves-tool",
  "intent": "explain_driver_effect",
  "parameters": {
    "driver": "shoulder",
    "target_frame": "tool0"
  }
}
```

Supported intents are:

| Intent | Required parameters | Result |
| --- | --- | --- |
| `structural_summary` | none | Root, counts, branches, leaves, segments, drivers, mechanism and configuration status |
| `trace_kinematic_path` | `from_link`, `to_link` | Unique tree path, ordered joints, and traversal directions |
| `explain_driver_effect` | `driver`; optional `target_frame` | Domain, driven physical joints, affected frames, and exact positive/negative when covered |
| `explain_frame_pose_law` | `frame` | Ordered operator composition and driver dependencies |
| `explain_constraint` | `constraint` | Asserted relation, type, role, and derived driver dependencies |
| `compare_configuration_nodes` | `node_a`, `node_b` | Driver deltas, ranks, finite component membership, and global-branch unknown |
| `describe_entity` | `entity` | Typed entity record and directly indexed relation proof closure |

Typed IDs are preferred. A bare suffix is accepted only when it resolves uniquely within the required entity class. Ambiguity is an error.

Example:

```bash
python3 scripts/robot_spatial.py query-concepts \
  work/context/concept-graph.json work/query.json \
  --out work/concept-answer.json
```

An answer binds graph ID/digest, query ID/intent, controlled answer, structured answer, minimal supporting clauses, unknown boundaries, source binding, and epistemic scope.

## Controlled-language rendering

`concept-language.rsl` is a deterministic rendering, not an independent source. It begins with:

```text
LANGUAGE RSC-LANG/1
CONCEPT_GRAPH ...
CONCEPT_GRAPH_SHA256 ...
SOURCE_BINDING ...
NEGATIVE_RULE ...
IDENTITY_RULE ...
BOUNDARY_RULE ...
```

Each remaining line contains one clause, modality, scope, evidence type, and exact flag. Use it when a whole-robot symbolic overview is more useful than a targeted query. For large robots, prefer `query-concepts` so the context window receives only the relevant proof closure.

## Verification

Run:

```bash
python3 scripts/robot_spatial.py verify-concept-graph work/context \
  --concept work/context/concept-graph.json \
  --language work/context/concept-language.rsl \
  --out work/concept-verification.json
```

The verifier:

1. validates graph schema, semantic digest, clause content IDs, typed subjects, proof premises, exact indexes/coverage, rooted-tree closure, and projection-to-clause consistency for topology, articulation, semantics, mechanism, and configuration records;
2. verifies each bound articulation, constraint, and atlas artifact SHA-256 from `model.json`;
3. regenerates the graph from the exact context;
4. requires exact canonical graph equality;
5. regenerates and requires byte-identical controlled language.

This verifies deterministic abstraction against the same bound artifacts. It is not an independent URDF parser, kinematics oracle, physical inspection, or global configuration proof. Use `crosscheck_concept_graph.py` for dependency-free randomized structural implementation diversity, and the existing source-format, constraint, geometry, or configuration oracles for their respective numeric layers.

## AI use order

1. Read `agent-context.json` for identity, epistemic, and unresolved boundaries.
2. Map the user's structural question to one strict concept-query intent.
3. Run `query-concepts` and read the structured answer plus proof closure.
4. Preserve clause modality in the final wording.
5. Use `retrieve` for entity-local facts or fresh numeric tools for a pose, transform, axis, Jacobian, collision, or observation not answered symbolically.
6. Run `verify-concept-graph` before high-assurance structural claims or after any source edit.

Never answer a numeric transform from the concept language alone. Never answer an open-world absence as false. Never replace an asserted semantic role with a naming guess.

## Boundary

Version 1 is a deterministic symbolic abstraction and query layer over already supported robot-spatial artifacts. It does not provide:

- free-form natural-language semantic parsing;
- automatic functional part classification;
- learned affordances or task capability inference;
- arbitrary first-order logic, temporal planning, or action precondition/effect reasoning;
- proof that supplemental assertions match a physical mechanism;
- global configuration-space enumeration or certification;
- dynamics, controller execution, hardware observation, or safety.

These are explicit future layers. The concept graph is valuable because it makes current knowledge compositional and current ignorance representable, not because it hides either boundary.
