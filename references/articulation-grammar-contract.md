# Articulation Grammar Contract

Use this contract when the question is about the robot's general motion law rather than one exported pose or two sampled endpoints.

## Purpose

`robot-spatial-articulation-grammar.v1` is a pose-independent, executable representation of supported URDF, SDF, or canonical MJCF tree kinematics. It gives an AI four distinct kinds of language:

- links and frames are typed spatial entities, analogous to nouns;
- joint operators are typed parameterized relations, analogous to verbs;
- root-to-frame derivations are ordered operator compositions, analogous to syntax;
- independent driver positions are variable bindings that produce one evaluated pose.

Do not substitute a pose-conditioned artifact for this law:

| Layer | Meaning |
|---|---|
| articulation grammar | general parameterized kinematic law |
| supplemental constraint graph | asserted full-mechanism relations over tree coordinates |
| forward kinematics | evaluation of the law at one driver binding |
| geometric Jacobian | first derivative of target motion at one binding |
| counterfactual motion atlas | finite signed endpoint evaluations around one baseline |

## Artifact binding

The export always writes `articulation-grammar.json`. Bind every use to:

- `schema_version`;
- `grammar_id` and `grammar_input_sha256`;
- artifact SHA-256;
- exact source format, byte and semantic SHA-256 values, and import contract;
- robot name and root frame;
- coordinate contract.

The grammar is deterministic. Identical semantic input and supported declarations produce identical JSON bytes.

Keep `law_identity` separate from `source_binding`. The canonical law hash excludes source provenance but retains typed identifiers and every executable field. Differently named sources require the explicit digest-bound correspondence in [cross-representation-contract.md](cross-representation-contract.md).

## Coordinate and composition semantics

`A_from_B` means the pose of B expressed in A. Matrices are homogeneous 4×4, row-major, and compose left to right from parent to child.

For a physical joint `j`:

```text
parent_from_child(q)
  = parent_from_joint_pre_motion
  × joint_motion(q[j])
  × joint_post_motion_from_child_zero
```

The first and third factors are normalized source-declared joint-anchor constants. URDF makes the third factor identity; SDF joint frames and MJCF joint anchors may not. The middle factor is one typed motion operator:

- fixed: identity;
- revolute or continuous: rotation about the normalized declared axis by `q[j]` radians;
- prismatic: translation along the normalized declared axis by `q[j]` meters.

The axis is expressed in the pre-motion joint frame. A joint's own pre-motion frame includes the constant origin but excludes that joint's motion.

## Variables and mimic equations

`independent_variables` contains only non-fixed, non-mimic driver joints. Each record declares:

- variable identity and unit;
- feasible default;
- complete feasible domain;
- every physical joint driven by the variable;
- structural affected links and frames.

`joint_position_rules` contains one typed rule for every physical joint:

- `constant` for fixed joints;
- `independent_variable` for a driver;
- `affine_driver_dependency` for a mimic follower.

Mimic chains are flattened into the exact affine equation:

```text
q[physical_joint] = multiplier × q[independent_driver] + offset
```

The driver domain is the intersection of every applicable revolute/prismatic position limit after transforming follower limits through the affine equation. This handles nested mimic chains, negative multipliers, one-sided bounds, and constant zero-multiplier followers. Continuous joints contribute no declared position bounds, but a non-continuous mimic follower may still bound their driver.

`minimum: null` or `maximum: null` means unbounded in that direction. It does not mean zero or unknown.

The standalone evaluator accepts independent driver values. It may also receive a complete physical-joint pose, but every supplied fixed or dependent value must agree with the grammar-derived value. Unknown joints, non-finite values, contradictions, and values outside the feasible domain are rejected.

## Joint operators

Each `joint_operators/<joint>` record contains:

- typed parent and child links;
- constant parent-from-pre-motion transform;
- typed motion operator and axis;
- position-rule reference;
- composition rule;
- explicit own-pre-motion-frame causality.

Treat the typed fields as executable authority. `composition_rule` and `equation_cnl` are controlled-language explanations of the same data, not a separate symbolic algebra source.

## Frame derivations

Every supported link, joint pre-motion, visual, collision, and inertial frame has one `frame_derivations` record. It contains:

- exact semantic type, owner, and parent frame;
- link reached before its terminal constant attachment;
- ordered joint-operator references from root;
- terminal constant attachment;
- independent driver dependencies;
- controlled-language composition tokens and expression.

Derivation construction follows these rules:

- root link: no joint operators and identity terminal;
- non-root link: every root-to-link joint operator, including the incoming joint motion;
- joint pre-motion frame: operators only through the parent link, then that joint's constant origin; its own motion is absent;
- visual/collision/inertial frame: operators through the owning link, then the declared local origin.

This distinction is essential. Coincident zero-pose matrices do not make two frame identities interchangeable.

## Commands

Generate an artifact directly from URDF, SDF, or canonical MJCF:

```bash
python3 scripts/robot_spatial.py articulation-grammar robot.urdf \
  --out work/articulation-grammar.json

python3 scripts/robot_spatial.py articulation-grammar robot.sdf \
  --format auto --out work/sdf-articulation-grammar.json
```

Every normal `export` also generates `articulation-grammar.json`, typed context cards, provenance facts, and six grammar-understanding evaluation questions.

The same export compiles the grammar into `concept-graph.json` and `concept-language.rsl`. That layer adds tree closure, branch/leaf/maximal-serial abstractions, driver-to-frame causal clauses, strict queries, and proof closure without replacing the executable grammar. Use `query-concepts` to explain a structural composition; use `evaluate-articulation` for a new numeric binding. Read [concept-language-contract.md](concept-language-contract.md).

Evaluate a new pose using only the grammar artifact:

```bash
python3 scripts/robot_spatial.py evaluate-articulation \
  work/articulation-grammar.json \
  --pose new-pose.json \
  --target tool0 \
  --out work/tool0-at-new-pose.json
```

This command does not parse the URDF. Its operator trace exposes every constant and motion step used to obtain the frame result.

Regenerate and verify against the source-format implementation:

```bash
python3 scripts/robot_spatial.py verify-articulation-grammar robot.urdf \
  --grammar work/articulation-grammar.json \
  --out work/articulation-verification.json
```

The verifier checks exact deterministic regeneration, standalone AST execution, mimic binding agreement, frame-set agreement, and every frame matrix over deterministic fresh driver probes. This is an internal consistency verifier, not implementation diversity.

Compare two source-neutral laws and unseen executions:

```bash
python3 scripts/robot_spatial.py compare-articulation-grammars \
  work/urdf-articulation-grammar.json \
  work/sdf-articulation-grammar.json \
  --correspondence work/sdf-to-urdf.json \
  --out work/cross-representation-report.json
```

Read [cross-representation-contract.md](cross-representation-contract.md) before interpreting this result.

For stronger evidence, run `scripts/crosscheck_articulation_grammar.py`. It parses URDF, resolves mimic equations, evaluates FK, and checks generated grammar laws without importing production parser, matrix, grammar, or FK code.

## Agent load order

For a general articulation question:

1. read `agent-context.json` and its unresolved boundaries;
2. retrieve `articulation_grammar/<grammar-id>`;
3. retrieve the relevant `articulation_variable/<grammar-id>/<driver>`;
4. retrieve `articulation_operator/<grammar-id>/<physical-joint>`;
5. retrieve `articulation_derivation/<grammar-id>/<frame>`;
6. follow bound fact IDs;
7. run `evaluate-articulation` for a new binding;
8. run `verify-articulation-grammar` when integrity matters;
9. compare source-binding-free laws with `compare-articulation-grammars` before claiming cross-representation agreement;
10. if a supplemental graph exists, load and evaluate it before calling the tree pose a valid full-mechanism configuration;
11. use the independent oracle before claiming cross-implementation agreement.

State whether the answer is pose-independent or evaluated. For evaluated results, state the driver binding, resolved mimic positions, root/reference frame, target frame, meters/radians, quaternion order, and grammar digest.

## Epistemic boundary

The grammar is exact for supported fixed, revolute, continuous, and prismatic joints in the validated source tree, including normalized joint anchors, axes, limits, and supported dependency relationships. The full context/export pipeline remains URDF-specific; SDF/MJCF support currently targets articulation-law compilation, evaluation, verification, and comparison.

It does not establish:

- planar or floating-joint semantics;
- supplemental constraints inside the grammar itself; use the separately bound layer in [constraint-graph-contract.md](constraint-graph-contract.md);
- inverse kinematics or reachability of an arbitrary target;
- time, interpolation, trajectories, swept volumes, or continuous collision;
- velocity, acceleration, effort, dynamics, contacts, friction, or compliance;
- controller, transmission, plugin, actuator, sensor, or hardware behavior;
- calibration, observation truth, world placement, or physical safety.

URDF tree structure is the grammar boundary, not the complete-mechanism boundary. For parallel mechanisms or closed chains, use [constraint-graph-contract.md](constraint-graph-contract.md); never infer missing constraints from geometry or names. Its current evaluator and solver still do not prove global configuration-space structure or physical truth.
