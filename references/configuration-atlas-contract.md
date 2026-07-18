# Finite Configuration Atlas Contract

Use this contract when one pose-local constraint evaluation is insufficient and the task asks about multiple satisfying configurations, finite branch witnesses, rank changes, or singularity candidates of a constrained mechanism.

## Layer and claim

The representation has three ordered mechanism layers:

1. `robot-spatial-articulation-grammar.v1` is the executable spanning-tree coordinate law.
2. `robot-spatial-constraint-graph.v1` adds asserted loop, attachment, distance, and coordinate relations.
3. `robot-spatial-configuration-atlas.v1` records finite multi-seed local-solve witnesses over explicit one-parameter charts.

The atlas does not convert URDF into a globally certified configuration manifold. It makes the finite exploration contract, seeds, solver attempts, satisfying nodes, numerical rank diagnostics, merge metric, and proximity graph visible to the AI.

Preserve this distinction:

- a node is an executable satisfying configuration witness within declared constraint tolerances;
- a connected component is a component of the finite declared proximity graph;
- a rank-drop label is relative to the maximum numerical rank observed in that declared chart;
- none of these is exhaustive branch enumeration, a topological component certificate, or a certified singularity.

## Digest-bound input

`robot-spatial-configuration-atlas-spec.v1` contains exactly:

```json
{
  "schema_version": "robot-spatial-configuration-atlas-spec.v1",
  "atlas_id": "parallel-linkage-finite-witnesses",
  "constraint_graph_sha256": "<sha256 of exact constraint-graph.json bytes>",
  "singular_value_relative_tolerance": 1e-7,
  "charts": []
}
```

The graph digest binds the exact standalone constraint artifact, including its articulation law, asserted supplement, tolerances, and provenance. A spec for a different graph is rejected.

The authoring order is:

1. compile and verify the exact articulation grammar;
2. compile and verify the exact constraint graph;
3. author the atlas spec with the graph artifact SHA-256;
4. generate the atlas;
5. verify exact regeneration and every stored node;
6. read chart/node/component cards before making a finite configuration-space claim.

## One-parameter chart contract

Every chart contains exactly:

```json
{
  "chart_id": "drive-input",
  "parameter_driver": "input",
  "parameter_values": [-0.6, 0.0, 0.6],
  "solve_for": ["passive_a", "passive_b"],
  "driver_scales": {
    "input": 1.0,
    "passive_a": 1.0,
    "passive_b": 1.0
  },
  "seeds": [
    {
      "seed_id": "assembly-mode-a",
      "joints": {
        "input": 0.0,
        "passive_a": 0.0,
        "passive_b": 0.0
      }
    }
  ],
  "solution_merge_tolerance_normalized": 1e-5,
  "continuation_edge_max_distance_normalized": 0.5,
  "minimum_solutions_per_sample": 1
}
```

Rules:

- `chart_id` and seed IDs are unique typed relative identifiers.
- `parameter_driver` is one independent articulation driver.
- `parameter_values` contains at least two finite strictly increasing values inside that driver's feasible domain.
- `solve_for` contains every other independent driver exactly once. Version 1 intentionally supports only explicit one-parameter charts.
- `driver_scales` names every independent driver exactly and supplies positive units-to-normalized-distance scales.
- every seed binds every independent driver and lies inside every feasible domain;
- merge and edge thresholds are positive normalized distances;
- `minimum_solutions_per_sample` is a positive declared coverage expectation, not an assertion that exactly that many global solutions exist.

Seed coverage is supplied exploration intent. The tool cannot know whether the seeds cover every basin. Include analytically known assembly modes, mirrored configurations, and suspected singular slices when they matter. Missing a solution from all seeds is unknown, not proof of absence.

## Exploration algorithm

For each parameter sample, the atlas:

1. overwrites the parameter value in every explicit seed;
2. adds every previous-sample node as a continuation seed with the new parameter value;
3. deduplicates seeds under the declared normalized configuration metric;
4. runs the constraint graph's damped local solver over `solve_for` only;
5. re-evaluates every converged result against all typed constraint tolerances;
6. merges satisfying configurations within the declared normalized merge tolerance;
7. records every attempt, convergence result, unique node, source seed, and support attempt.

The parameter driver remains fixed. Local failure does not prove infeasibility. Convergence does not prove uniqueness.

For continuous drivers, distance uses shortest wrapped angular difference unless that driver's physical joint participates in a `coordinate_linear` constraint. A linear coordinate equation can distinguish turns, so wrapping would erase declared semantics. Other drivers use ordinary difference. Every coordinate difference is divided by its declared scale before Euclidean distance is computed.

## Node execution and numerical diagnostics

Every configuration node records:

- exact chart/sample/parameter identity;
- all independent driver positions;
- constraint status and maximum normalized residual;
- supporting attempts and source seeds;
- the complete pose-conditioned local constraint analysis;
- singular values, rank, nullity, threshold, and condition diagnostics for the full normalized constraint Jacobian;
- the same diagnostics for the passive Jacobian restricted to `solve_for`.

Residual rows are normalized by their declared typed tolerances before finite differences. The dependency-free implementation computes eigenvalues of `JᵀJ` with a symmetric Jacobi iteration and derives singular values. The numerical rank threshold is:

```text
singular_value_relative_tolerance × max(1, largest_singular_value)
```

The full Jacobian diagnoses the local constraint manifold in all tree coordinates. The passive Jacobian diagnoses whether this chosen parameterization can locally solve the remaining coordinates. A passive rank drop can be a chart failure without a mechanism rank drop.

For each chart, a node is labeled:

- `mechanism_rank_drop_candidate` when its full rank is below the maximum full rank observed in that chart;
- `chart_parameterization_rank_drop_candidate` when its passive rank is below the maximum passive rank observed in that chart.

These are finite relative witnesses. A missed regular sample, poor scaling, finite-difference error, or tolerance choice can change the reference. Certification requires a separate symbolic, interval, algebraic, or otherwise validated global method.

## Proximity graph

The atlas adds an undirected proximity witness edge when:

- nodes are in adjacent parameter samples and their normalized configuration distance is within the declared edge threshold; or
- nodes are in the same sample, both carry a rank-drop candidate label, and their distance is within the threshold.

Connected components are computed over only these stored edges. They help an AI navigate likely continuations and singular slices. They do not establish true connected components of the constraint manifold. A large threshold can merge distinct modes; a small threshold or coarse samples can split one mode.

## Coverage and status

Each sample reports attempts, converged attempts, unique solution count, required minimum, and `met` or `below_required_minimum`.

Atlas status is:

- `complete_for_declared_sampling` when every explicit sample meets its declared minimum;
- `partial_for_declared_sampling` otherwise.

“Complete” applies only to the finite declared sample/minimum contract. It never means globally complete. Export still emits a partial atlas but exits non-zero as `exported_with_incomplete_configuration_atlas`. Missing nodes must remain visible.

## Commands

```bash
python3 scripts/robot_spatial.py configuration-atlas \
  work/constraint-graph.json configuration-atlas-spec.json \
  --out work/configuration-atlas.json

python3 scripts/robot_spatial.py verify-configuration-atlas \
  work/constraint-graph.json configuration-atlas-spec.json \
  --atlas work/configuration-atlas.json \
  --out work/configuration-atlas-verification.json

python3 scripts/robot_spatial.py export robot.urdf \
  --constraint-spec constraints.json \
  --configuration-atlas-spec configuration-atlas-spec.json \
  --workspace-samples 0 --out work/context
```

`--configuration-atlas-spec` requires `--constraint-spec` for `export` and `prepare`. The context adds:

- `configuration_atlas/<atlas-id>`;
- `configuration_chart/<atlas-id>/<chart-id>`;
- `configuration_node/<atlas-id>/<chart-id>/<sample-index>/<solution-index>`;
- `configuration_component/<atlas-id>/<chart-id>/<component-index>`.

Use `retrieve` on exact typed IDs. Do not ask the model to infer branches from filenames, node ordering, or an SVG.

When the atlas is included in an export or preparation, the concept graph compiles chart contracts, satisfying-node records, ranks/candidates, and finite proximity components as `finite_computed_evidence`. `compare_configuration_nodes` reports finite component membership but fixes `same_global_branch` to `not_established`; the graph also carries explicit false certification boundaries. This makes the correct finite/global distinction machine-queryable rather than dependent on prose. Read [concept-language-contract.md](concept-language-contract.md).

## Verification and independent oracle

`verify-configuration-atlas`:

- rebuilds the atlas from the exact graph and exact spec;
- requires byte-equivalent canonical JSON regeneration;
- executes every stored node against the embedded standalone constraint graph;
- reports every violation and exact binding digest.

This proves deterministic internal consistency only.

`crosscheck_configuration_atlas.py` imports no production parser, articulation, constraint, configuration, matrix, Jacobian, rank, residual, or solver implementation. It generates non-planar spherical three-revolute closures and independently evaluates:

```text
R = R_x(a) R_z(b) R_x(c)
```

The two generic analytic closure branches are `b=0, c=-a` and `b=π, c=a`. At `a=0, c=0`, every `b` is a satisfying singular slice; the independent analytic Jacobian has rank one at `b=0` and rank two at `b=π/2`. Preserve randomized amplitudes, seed, case count, maximum branch/alignment errors, rank contrast, node-execution count, and exclusions.

## Boundary

Version 1 is a dependency-free finite witness atlas for equality-constrained mechanisms. It does not provide adaptive meshing, deflation, homotopy completeness, interval certification, algebraic component decomposition, inequality/contact manifolds, collision-aware continuation, dynamics, stability, bifurcation certification, uncertainty, calibration, hardware verification, or safety.

Do not say “the mechanism has N branches” from N stored components. Say “the declared chart produced N finite proximity components under these samples, seeds, scales, and thresholds.”
