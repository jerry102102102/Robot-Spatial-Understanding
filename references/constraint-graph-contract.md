# Supplemental Constraint Graph Contract

Use this contract when a tree-shaped articulation grammar is only a coordinate parameterization of a mechanism with loop closures, parallel branches, rigid attachment points, distance relations, or coordinate couplings.

## Why this is a separate layer

URDF requires a kinematic tree, while real mechanisms may contain relations that close a cycle or reduce the independent mobility of that tree. Do not invent those relations from names, coincident zero-pose geometry, or common mechanism patterns.

The representation therefore has two bound layers:

1. `robot-spatial-articulation-grammar.v1` supplies executable tree coordinates and every root-to-frame law.
2. `robot-spatial-constraint-graph.v1` supplies explicitly asserted attachments and constraint residuals over that law.

This matches the source-model distinction described by the [SDFormat model-kinematics specification](https://sdformat.org/tutorials/specification/spec_model_kinematics/) and the explicit equality-constraint approach in the [MuJoCo XML reference](https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html). URDF+'s motivation likewise documents the tree limitation and need for loop-capable mechanism semantics in [arXiv:2411.19753](https://arxiv.org/abs/2411.19753).

The supplement is declared mechanism intent. Digest binding makes it reproducible; it does not make it an observation of assembled hardware.

## Source contract

`robot-spatial-constraint-spec.v1` contains exactly:

```json
{
  "schema_version": "robot-spatial-constraint-spec.v1",
  "constraint_set_id": "parallel-linkage-v1",
  "grammar_sha256": "<sha256 of exact articulation-grammar.json bytes>",
  "attachments": [],
  "constraints": []
}
```

The exact grammar digest prevents a constraint set from being applied to a different tree law. The normal authoring order is:

1. compile the exact articulation grammar;
2. author or update the spec with that artifact SHA-256;
3. compile and verify the constraint graph;
4. evaluate an explicit pose before claiming it is a valid mechanism configuration.

Every attachment parent must be a frame already declared by the grammar. Version 1 does not allow attachments to be nested under other attachments.

## Rigid attachments

Each attachment declares:

- unique `attachment_id`;
- exact grammar `parent_frame`;
- one role: `constraint_anchor`, `mount`, `tcp`, `measurement_point`, or `joint_anchor`;
- `parent_from_attachment` with meters and an xyzw unit quaternion.

The executable rule is:

```text
root_from_attachment
  = root_from_parent_frame
  × parent_from_attachment
```

An attachment is a typed point/frame used by a constraint. Its semantic role is asserted, not inferred.

## Supported constraints

### Kinematic pair

`kinematic_pair` relates `frame_a` and `frame_b` as one of:

- `fixed`: three translational plus three rotational residuals;
- `revolute` or `continuous`: coincident origins plus two axis-alignment residuals, leaving twist about the axis free;
- `prismatic`: two transverse-translation residuals plus three rotational residuals, leaving translation along the axis free.

Axes are unit vectors expressed in the named attachment frames. Translation tolerances are meters; rotation tolerances are radians.

### Point distance

`point_distance` evaluates:

```text
norm(origin(frame_b) - origin(frame_a)) - distance_m
```

It has one signed meter residual and an explicit `tolerance_m`.

### Linear coordinate coupling

`coordinate_linear` evaluates resolved physical-joint coordinates after mimic rules:

```text
sum(coefficient_i × q[joint_i]) + offset = 0
```

It has one signed `joint_coordinate` residual. All terms in one equation must be dimensionally meaningful; version 1 records but does not symbolically prove dimensional homogeneity.

Unknown roles, pair types, constraint types, frames, joints, extra fields, non-finite values, zero coefficients, non-unit axes/quaternions, and non-positive tolerances are rejected.

## Structural meaning

The graph preserves three edge classes:

- spanning-tree joint-operator edges;
- rigid attachment edges;
- supplemental frame-constraint edges.

For every frame constraint it records the exact tree path between endpoints plus the closure edge. Coordinate constraints are separately indexed. If any frame or coordinate constraint is present:

```json
"tree_is_parameterization_not_complete_mechanism": true
```

This flag is epistemically important. A valid tree pose may violate the full mechanism.

If no graph was supplied, report supplemental constraints as `not_provided` or unknown. Never convert absence of a supplement into proof that no loop or coupling exists.

## Evaluation

`evaluate-constraints` first evaluates the embedded standalone articulation grammar, composes attachments, then emits every typed signed residual component with:

- value and unit;
- declared tolerance;
- normalized absolute value;
- component and constraint satisfaction;
- complete pose binding and graph binding.

A constraint is satisfied only when all of its components are within tolerance. The graph is satisfied only when every constraint is satisfied.

The result also computes a normalized residual finite-difference Jacobian with respect to every independent tree driver. Its numerical rank yields:

```text
local_mobility_estimate
  = tree_independent_variable_count
  - local_constraint_rank
```

Always call this a pose-conditioned numerical local mobility estimate. Rank can change at singular configurations. It is not proof of global mechanism DOF, branch count, global reachability, or configuration-space topology.

When the task needs finite evidence across poses, do not repeat ad hoc local solves and then narrate a branch. Use the digest-bound one-parameter charts, multi-seed records, node execution, full/passive rank diagnostics, and explicit proximity semantics in [configuration-atlas-contract.md](configuration-atlas-contract.md).

## Local solving

`solve-constraints` uses damped Gauss–Newton with line search and driver-domain clamps. The caller must provide:

- an explicit seed pose;
- every independent driver that the solver may change.

All unselected drivers remain fixed. Report the seed, fixed variables, solved variables, trace, convergence status, final typed residuals, and potential branch dependence.

Convergence proves only that this local numerical process reached the declared tolerances from that seed. Failure does not prove infeasibility; convergence does not prove uniqueness, global completeness, collision freedom, dynamics, or physical assembly.

## Commands

```bash
python3 scripts/robot_spatial.py articulation-grammar robot.urdf \
  --out work/articulation-grammar.json

python3 scripts/robot_spatial.py constraint-graph \
  work/articulation-grammar.json constraints.json \
  --out work/constraint-graph.json

python3 scripts/robot_spatial.py evaluate-constraints \
  work/constraint-graph.json --pose pose.json \
  --out work/constraint-evaluation.json

python3 scripts/robot_spatial.py solve-constraints \
  work/constraint-graph.json --pose seed.json \
  --solve-for passive_joint_a --solve-for passive_joint_b \
  --out work/constraint-solution.json

python3 scripts/robot_spatial.py verify-constraint-graph \
  work/articulation-grammar.json constraints.json \
  --graph work/constraint-graph.json --pose pose.json \
  --out work/constraint-verification.json
```

For a URDF context pack, pass `--constraint-spec constraints.json` to `export` or `prepare`. The pack adds `constraint_graph/`, `attachment/`, and `constraint/` cards plus asserted/dependent fact labels. A violated export pose still emits the diagnostic artifacts but exits non-zero with `exported_with_violated_constraints`.

The context's concept graph preserves the modality boundary explicitly: the supplemental relation and attachment are assertions, while the set of drivers referenced by their exact frames/coordinates is a deterministic derivation conditional on those assertions. `query-concepts` returns both clauses and their proof relation. It never upgrades the supplement to observed mechanism or hardware truth. Read [concept-language-contract.md](concept-language-contract.md).

To add finite configuration witnesses, also pass `--configuration-atlas-spec configuration-atlas-spec.json`. It must bind the exact generated constraint graph artifact. This adds atlas/chart/node/component cards and exits non-zero if any declared per-sample solution minimum is unmet.

## Verification and independent evidence

`verify-constraint-graph` checks:

- exact deterministic graph regeneration from the bound grammar and spec;
- both source digests;
- standalone graph execution;
- reproducibility of the pose-conditioned local analysis.

This is internal consistency, not implementation diversity.

`crosscheck_constraint_graph.py` is the dependency-free independent oracle. It imports no production parser, articulation, constraint, matrix, rank, residual, or solver implementation. It generates raw planar parallelogram and linear-coupling cases, calls only the public CLI, and independently recomputes analytic closure/coupling errors. Preserve its seed, case counts, negative controls, solver coverage, maximum errors, and exclusions.

## Boundary

Version 1 supports rigid attachments; fixed/revolute/continuous/prismatic pair residuals; point distance; linear coordinate coupling; numerical local rank/mobility; and a local equality solver.

It does not support nested attachments, planar/floating pair constraints, gears with backlash, inequality constraints, joint stops beyond the embedded grammar domains, compliant constraints, contact complementarity, friction, force closure, dynamics, time integration, controller synthesis, global branch enumeration, certified global solving, uncertainty, calibration, or hardware verification.

Preserve three distinct claims:

1. the supplemental relation was asserted;
2. the typed residual was deterministically evaluated from that assertion and tree law;
3. physical truth remains not established.
