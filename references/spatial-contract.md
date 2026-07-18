# Spatial contract

## Contents

- Meaning of transforms
- Static world snapshots and root mounting
- Xacro provenance and frame semantics
- Canonical, geometry, and fact artifacts
- Declared mass, center of mass, and inertia
- Static gravity loads and holding-effort sign semantics
- Embedded actuation and control declarations
- Semantic annotations and SRDF semantics
- Pose-independent articulation grammar
- Proof-carrying spatial concept language
- Project function, capability, and affordance language
- Supplemental constraint graph and finite configuration atlas
- Project spatial invariants and edit acceptance
- Instantaneous motion and sampled workspace
- Triangle surface distance and solid collision
- Evaluation and understanding evidence
- Independent-oracle validation
- Engine boundary and acceptance questions

## Meaning of transforms

The canonical transform is named `reference_from_target`. It maps target-frame coordinates into the reference frame:

`p_reference = reference_from_target @ p_target`

The CLI command `transform --from A --to B` therefore emits `A_from_B`. Matrices are 4×4 row-major homogeneous transforms. Translation uses meters. Joint positions use radians for revolute/continuous joints and meters for prismatic joints. Quaternions are explicitly `xyzw`.

## Static world snapshots and root mounting

The canonical URDF `world_from_frame` namespace is robot-local: its `world` is the URDF root used as a mathematical origin. It is not a claim about the physical world. Supply a separate `robot-spatial-world-scene.v1` file to mount the URDF root under an acyclic scene-frame graph, declare world-expressed gravity, and place external objects for one identified static snapshot.

Snapshot-bound transforms use `scene_frame/X`, `robot_frame/X`, `robot_geometry/X`, `scene_object/X`, and `scene_geometry/X/Y`. `scene-transform --from A --to B` preserves the same `A_from_B` direction as robot-local transforms. Scene queries bind the URDF semantic digest, scene file SHA-256, `scene_id`, `snapshot.id`, pose, and parameters. Read [world-scene-contract.md](world-scene-contract.md) before using these records.

## Xacro provenance

Never treat Xacro source as concrete URDF. Run `expand-xacro` through the project/ROS `xacro` executable with explicit `name:=value` mappings. The command validates the resulting URDF and writes `<output>.meta.json` containing input/output SHA-256 digests, mappings, executable path, diagnostics, and structural counts.

## Frame semantics

- `<link>` is the moving URDF link frame after its parent joint motion.
- `joint/<name>` is the parent-to-joint origin before joint motion. A URDF joint axis is expressed here.
- `visual/<link>/<index>` is the visual geometry origin relative to the link.
- `collision/<link>/<index>` is the collision geometry origin relative to the link.
- `inertial/<link>` is the URDF-declared center-of-mass and inertia-coordinate frame relative to the link. It is not a geometry-derived centroid.

These frames must not be collapsed even when their transforms happen to be identical.

## Canonical artifact

`model.json` uses schema `robot-spatial.v2` and contains:

- source identity and SHA-256 digest;
- declared link and joint semantics;
- resolved joint pose, including mimic relationships;
- explicit frame types and parent relationships;
- `world_from_frame` for every supported frame;
- world-expressed joint axes;
- analytic geometric Jacobians and optional sampled workspace observations for explicitly declared target frames;
- validated link inertials and pose-conditioned whole-tree declared mass properties;
- pose-conditioned declared-model static gravity loads under an explicit root-frame convention;
- an optional bound static world scene with root mounting, world-conditioned gravity loads, typed scene/object geometry, and robot/environment collision evidence;
- embedded ros2_control, legacy transmission/actuator, interface, hardware-plugin, and joint-dynamics declarations;
- limitations and validation warnings.

When `--invariants` is supplied, the canonical artifact also embeds `invariant_validation`, and `artifacts.invariant_report` points to the standalone deterministic report. This layer is project intent, not a property inferred from the current URDF.

## Geometry evidence

Primitive geometry is measured analytically. `--inspect-meshes` opens and measures every declared visual and collision STL/OBJ mesh. `--inspect-mesh-kind visual` or `--inspect-mesh-kind collision` selects one representation and implies inspection; repeat the option to select both. This allows collision STL to be measured even when unrelated visual DAE is unsupported, without claiming that the DAE was inspected. For `package://` URIs, pass a JSON map such as:

```json
{"my_robot_description": "/workspace/src/my_robot_description"}
```

Each measured geometry record contains geometry-frame and pose-conditioned root-frame AABBs, surface area, trustworthy volume when the triangle surface is watertight with consistent winding, vertex-covariance principal axes, six extrema landmarks, and an explicitly heuristic shape class. Mesh source units are unknown; the declared URDF mesh scale is applied exactly.

`capabilities.mesh_content_inspection` records `requested_kinds`, counts and completeness under `by_kind`, `complete_for_requested_kinds`, and `complete_for_all_declared_meshes`. A selective pass is complete only for its requested representation. Every unselected mesh remains `not_inspected`, so downstream agents must not generalize collision-shape evidence into a claim about render/visual geometry.

`collision_broadphase` compares measured collision AABBs. It is complete only when every declared collision geometry was measured. A listed pair is a conservative candidate and does not prove surface contact.

When requested, `collision_surface` uses exact AABB-distance rejection at the contact tolerance followed by deterministic BVH branch-and-bound over triangle-pair distance. The portable exact triangle representations are:

- every declared STL/OBJ triangle after URDF scale and pose;
- an analytic box boundary represented by 12 triangles.

The engine does not tessellate analytic cylinders or spheres for collision, because that would turn an approximation into an apparently exact answer. A candidate containing either shape is `indeterminate`. The report always records the contact tolerance, witness points, triangle indices, search counts, representation status, and SRDF policy annotation.

Surface distance and solid collision are different claims. Intersecting or tolerance-close surfaces establish contact, but positive surface distance can coexist with collision when one solid completely contains another. Containment is classified with generalized winding solid angle only when both participating surfaces are watertight and consistently oriented. Open or inconsistently wound candidates with positive surface distance remain `indeterminate`. Same-link geometry pairs are intentionally excluded from robot self-collision, matching the broad-phase contract.

`scene.svg` provides a compact front/side/top/isometric overview. With `--render`, `render-atlas/manifest.json` additionally binds four standalone SVGs to the URDF semantic digest, resolved pose, geometry points/status, explicit root-to-UV bases, fitted UV-to-pixel mappings, typed frame/joint/geometry IDs, artifact digests, and coverage. Mesh-vertex and box convex hulls are exact for their loaded/declaration support; cylinder/sphere hulls use explicit deterministic boundary samples. The atlas does not compute visibility, occlusion, perspective, calibrated-camera pixels, or physical truth. Run `verify-render` to regenerate and verify view/numeric consistency. Read [render-atlas-contract.md](render-atlas-contract.md) before making view-grounded claims.

With `--motion-atlas`, `motion-atlas/manifest.json` binds one finite signed counterfactual record per independent movable joint to the same URDF semantic digest and resolved baseline pose. It intersects driver and full-chain mimic limits, records nominal/clipped/unavailable endpoint status, holds other independent drivers fixed, recomputes mimic followers, exposes all-frame SE(3) and measured-geometry endpoint deltas, and overlays baseline/minus/plus in one fitted screen per driver and view. Run `verify-motion-atlas` to regenerate and verify semantic, numeric, path, digest, and typed-SVG-identity consistency. The two endpoint states and their AABB union do not establish interpolation, swept volume, continuous collision, time, dynamics, controller response, hardware motion, or safety. Read [motion-atlas-contract.md](motion-atlas-contract.md) before making finite-motion claims.

`context.md` is a lossy, compact projection for language-model context. Use JSON or a direct CLI query for exact values.

`facts.jsonl` uses `robot-spatial-fact.v1`. Each line is an independently retrievable subject–predicate–object fact with pose qualifiers, source type, source digest, and an `exact` flag. Semantic roles, shape labels, and sampled workspace summaries are deliberately marked non-exact; URDF declarations, FK, analytic Jacobians, measured bounds, tree derivations, triangle-surface distances, and resolved collision statuses are marked exact within the explicitly reported representation and tolerance boundary.

Every URDF export also writes `concept-graph.json` (`robot-spatial-concept-graph.v1`) and `concept-language.rsl` (`RSC-LANG/1`). The graph is a proof-carrying, typed symbolic abstraction over the canonical tree, articulation grammar, explicit semantic assertions, optional constraint graph, and optional finite configuration atlas. It materializes root/branch/leaf/maximal-serial abstractions, tree transitive closure, driver domains and causal frame dependencies, ordered frame laws, assertion-conditioned mechanism dependencies, and finite configuration evidence. Each clause has a content-derived ID, modality, scope, evidence paths, proof rule/premises, and controlled-language rendering. `query-concepts` accepts a strict AST and returns the minimal recursive proof closure; it does not use an LLM to recompute topology. `verify-concept-graph` requires exact regeneration from the bound context and byte-identical RSL. Exact negative answers are limited to explicitly complete validated tree/articulation projections. Undeclared roles, supplemental relations, physical truth, runtime behavior, safety, global branches/topology, and certified singularities remain unknown. Read [concept-language-contract.md](concept-language-contract.md).

With `--functional-spec`, `functional-model.json` (`robot-spatial-functional-model.v1`) compiles explicit project object types, components, functions, conditions, intended effects, capabilities, and relational affordances after the exact concept graph exists. Every capability has typed enabling requirements evaluated against a declared closure basis: complete entity inventory, complete tree topology, complete articulation dependencies, open-world semantic/constraint assertions, or finite configuration witnesses. A query returns recursive functional clauses plus the required concept-clause proof closure. Project intent remains asserted even when structural requirements are exactly satisfied. Conditions retain their runtime/planner/operator/assumption truth source; intended effects remain unobserved; action possibility remains conditional; inventory absence never becomes physical impossibility. `verify-functional-model` validates internal content/index/projection/proof consistency and exact source/spec regeneration. Read [function-affordance-contract.md](function-affordance-contract.md).

With `--constraint-spec`, `constraint-graph.json` binds the exact articulation grammar and asserted supplemental spec. Context adds `constraint_graph/`, `attachment/`, and `constraint/` identities. The graph topology and typed residual evaluation are deterministic; the attachment/constraint intent facts remain non-exact assertions. The export pose must satisfy all declared tolerances or export exits non-zero after writing diagnostics. Read [constraint-graph-contract.md](constraint-graph-contract.md).

With `--configuration-atlas-spec`, `configuration-atlas.json` additionally binds the exact constraint graph and explicit finite chart contract. Context adds atlas/chart/node/component identities. Nodes are re-executable satisfying witnesses; singular values, rank drops, proximity edges, and components are finite numerical evidence under declared samples/seeds/scales/thresholds. Missing declared minimum solutions make export non-zero; complete declared sampling is still not exhaustive global topology. Read [configuration-atlas-contract.md](configuration-atlas-contract.md).

## Declared mass, center of mass, and inertia

`mass-properties` consumes URDF `<inertial>` declarations; it does not infer mass from visual or collision geometry. The inertial origin is the declared center of mass and the coordinate frame in which the six symmetric tensor components are expressed. Validation reports missing components, requires positive mass, checks that the tensor is positive semidefinite, and checks the rigid-body triangle inequality on its principal moments.

For a whole tree or `--subtree-root`, the query transforms each valid inertial frame to the requested `--frame` at the stated pose, rotates each tensor, computes the mass-weighted center of mass, and shifts every tensor to that center with the parallel-axis theorem. Output schema `robot-spatial-mass-properties.v1` includes:

- selected links and subtree root;
- pose and expression frame;
- every per-link declared contribution;
- aggregate declared mass, center of mass, inertia tensor, and principal moments;
- missing and invalid/incomplete inertial links;
- stable query evidence and an explicit physical-world boundary.

`status: computed` means the aggregate is exact for all valid inertials declared in the selected URDF tree. If any selected declaration is invalid or incomplete, status is `indeterminate` and all aggregate numeric fields are null; the engine never hides the defect by summing only the remaining valid links. If no selected link declares inertial properties, status is `not_provided`. None of these states means every selected link has a physical mass model. An absent `<inertial>` is declaration absence, not proof that the real component is massless. Even when every link has an inertial, URDF alone cannot establish payload, calibration-to-hardware, parameter identification quality, or agreement with the built robot. Use the phrase “declared mass properties,” preserve `physical_world_completeness: not_established`, and do not promote this kinematic aggregation to a dynamics simulation.

## Static gravity loads

`gravity-loads` consumes the same validated inertials plus the stated pose and an explicit gravity acceleration vector. `--gravity-frame` names the frame whose orientation expresses the three `--gravity` components; the default query and canonical export use `[0, 0, -9.80665]` m/s² expressed in the URDF root. This is a coordinate convention, not knowledge of how the physical robot is mounted in the world.

When a bound scene declares gravity, `scene-gravity-loads` first rotates that vector through the validated scene-frame graph and mounted root transform, then runs the same declared-model statics. Keep this result distinct from the root-convention result. If scene gravity is absent, the scene-conditioned value is `not_provided`; no default is substituted.

For every valid selected link inertial, the engine applies `F=m*g` at the FK-resolved declared center of mass. Each ancestor revolute/continuous joint receives `axis · ((p_com-p_joint) × F)`; each ancestor prismatic joint receives `axis · F`. Mimic followers are folded into their independent driver with the chain rule `dq_follower/dq_driver`, including nested multipliers. A subtree query includes only the selected descendant masses but retains loads transmitted through its upstream joints.

Output schema `robot-spatial-static-gravity-loads.v1` distinguishes two signs:

- `generalized_gravity_force`: force or torque exerted by the modeled gravity field along positive independent-joint motion;
- `ideal_static_holding_effort`: the exact opposite required for equilibrium in this gravity-only model.

Revolute/continuous values use N·m and prismatic values use N. The reported potential energy is relative to the root origin; its absolute zero is coordinate-dependent, while its derivative gives the generalized gravity force. The result records per-link and per-physical-joint contributions, independent-driver aggregation, declared joint effort-limit comparison, missing/invalid inertial coverage, and query provenance.

Invalid or incomplete selected inertials make all aggregate driver loads `indeterminate`; no selected valid inertial gives `not_provided`. Missing inertials remain explicit coverage gaps and are never silently treated as zero physical mass or zero load. `computed` is exact only for the declared gravity-only model. It excludes actual world-to-root mounting, unmodeled payload/cabling/tooling, contacts, friction/damping, velocity, acceleration, Coriolis/centrifugal terms, motor/gear efficiency, backlash, elasticity, controller behavior, and hardware feasibility. Even a modeled load below a declared URDF effort magnitude does not prove a real actuator can hold it.

## Embedded actuation and control declarations

`actuation` parses and reference-validates declarations embedded in the expanded URDF:

- ros2_control systems and their type;
- declared hardware plugin string and `<param>` values;
- joint, sensor, and GPIO command/state interfaces and interface parameters;
- legacy transmissions, joint hardware interfaces, actuators, and numeric mechanical reductions;
- standard URDF joint damping/friction values and nonstandard `<dynamics>` attributes preserved as uninterpreted strings.

`robot-spatial-actuation-declarations.v1` binds each kinematic joint to its embedded systems/interfaces and legacy transmissions and reports movable-joint declaration coverage. Typed context entities include `ros2_control_system/<name>`, `transmission/<name>`, `actuator/<transmission>/<name>`, and system-qualified control sensors/GPIOs. `actuation --joint`, `--system`, or `--transmission` returns a provenance-bound narrow query.

These records answer only “what this expanded XML declares.” They do not load plugins, parse external controller YAML/launch graphs, contact controller_manager, claim interfaces, connect hardware, establish mechanical-reduction convention/calibration, or prove that commands will execute. A system parameter such as `calculate_dynamics=true` remains a transcribed string, not evidence that dynamics are actually available or correct. Absence of an embedded declaration is also not proof that the physical robot is unactuated; control may live outside the URDF.

## Semantic annotations

URDF does not identify planning intent. Add an optional JSON file when roles matter:

```json
{
  "schema_version": "robot-semantics.v1",
  "frames": {
    "base_link": {"roles": ["base", "planning"], "meaning": "Cell robot base"},
    "slider_link": {"roles": ["flange"]},
    "tool0": {"roles": ["tcp"], "meaning": "Welding process point"}
  },
  "groups": {
    "manipulator": {
      "joints": ["shoulder", "slide"],
      "base_frame": "base_link",
      "tip_frame": "tool0"
    }
  },
  "end_effectors": {
    "welder": {"mount_frame": "slider_link", "tcp_frame": "tool0"}
  }
}
```

Every named frame and joint is validated against the URDF. Roles are user/project assertions, not facts inferred by this engine.

## SRDF semantics

Pass `--srdf robot.srdf` to validate MoveIt groups, directed base-to-tip chains, subgroup expansion, group states, end effectors, passive joints, virtual joints, and disabled-collision pairs against the URDF. Use `--pose-name group/state` to condition FK and geometry artifacts on an SRDF group state.

SRDF is planning semantics, not new physical geometry. A disabled-collision pair means the planner may ignore that pair; it does not mean their AABBs or surfaces cannot overlap. `collision_surface.self_collision_status` and every pair result remain physical. When SRDF is present, `srdf_policy_filtered_self_collision_status` plus enabled/disabled counts separately report the planning-policy view. Group-scoped `passive_joint` declarations are validated as both passive annotations and group joint membership; named-pose assignments to mimic followers must agree with the URDF mimic relation. JSON semantic annotations remain useful for roles SRDF does not state explicitly, such as the intended TCP or human-facing frame meaning.

## Instantaneous motion and sampled workspace

Every URDF export writes `articulation-grammar.json` using `robot-spatial-articulation-grammar.v1`; the articulation compiler also accepts the strict SDF and canonical MJCF subsets. This is the pose-independent executable law: independent variables with constrained domains, supported physical-joint dependency rules, typed pre-motion × motion × post-motion operators, and one ordered derivation for every supported frame. Each artifact separates exact source/import binding from `robot-spatial-canonical-kinematic-law.v1` identity. `evaluate-articulation` executes only that artifact at a new driver binding; `verify-articulation-grammar` regenerates it and compares all frames with source-format FK over fresh deterministic probes; `compare-articulation-grammars` requires exact mapped law equality and unseen all-frame agreement. Read [articulation-grammar-contract.md](articulation-grammar-contract.md) and [cross-representation-contract.md](cross-representation-contract.md) before generalizing one format, pose, Jacobian, or finite motion-atlas endpoint into a universal relationship.

The motion layers are intentionally non-interchangeable: articulation grammar is the general law, FK is one evaluation, Jacobian is a local derivative, and motion atlas is a finite endpoint sample.

The mechanism layers are also non-interchangeable: the articulation grammar is the executable spanning-tree coordinate law, a supplemental constraint graph reduces valid configurations through asserted relations, and a configuration atlas stores finite multi-seed satisfying/rank/proximity witnesses over declared charts. A tree-valid pose is not necessarily mechanism-valid. Numerical local mobility is not global DOF; a finite proximity component is not a global branch certificate.

`jacobian` emits `robot-geometric-jacobian.v1`. Rows are linear x/y/z followed by angular x/y/z. Columns are independent joint drivers along the root-to-target path. Each column records the physical joints that contributed; mimic derivatives are accumulated under the ultimate independent source joint. The target is the origin of any explicit link, pre-motion joint, visual, collision, or inertial frame. The target twist is relative to the URDF root, while vector components may be re-expressed in any known frame orientation.

The analytic Jacobian is exact for the supported tree model at the stated pose. It is local: it answers the instantaneous direction and magnitude of target motion caused by joint rates, not global reachability or collision freedom.

`motion-atlas` emits `robot-spatial-motion-atlas.v1`. It answers a different causal question: for a stated finite positive and negative driver displacement at one baseline, which exact endpoint poses result? Each independent driver record includes its physical mimic joints, structural affected frames, feasible interval, signed endpoint status, complete resolved pose, all-frame delta, geometry endpoint AABB delta, and four shared-screen projections. The driver's own pre-motion frame remains upstream and unchanged. Structurally affected frames may remain numerically stationary at a particular endpoint due to geometric coincidence or cancellation.

This finite contract is neither an infinitesimal derivative nor a path. The Jacobian is the rate-to-twist oracle at one point; the motion atlas is exact FK at two controlled endpoints. Neither alone proves reachability, collision freedom, or executable motion.

`workspace` emits `robot-sampled-workspace.v1`. It samples finite independent joint ranges using a deterministic center/extrema/Halton sequence. Revolute and prismatic drivers require complete URDF position limits; continuous joints use one canonical cycle `[-π, π]`. Driver ranges are intersected with every applicable mimic follower limit through the full affine mimic chain. The output includes the exact evaluated sample count, sample digest, observed target-origin AABB, observed radial range, and component ranges for the target's x/y/z axes.

Every workspace record is approximate. The observed AABB bounds only the evaluated samples: it does not prove that all interior points are reachable, that every reachable point lies inside, or that any sampled pose is collision-free. Use `workspace --include-samples` for a direct query or `export --include-workspace-samples` when individual observations must be independently audited.

## Project spatial invariants and edit acceptance

URDF and SRDF describe the current model, but neither says which relationships are protected design intent. Add a project-owned `robot-spatial-invariants.v1` contract to preserve explicit relative frame poses, frame distances, signed joint axes, ordered chains, causal subtrees, frame identity/parentage, pose-conditioned declared subtree mass/COM/inertia, pose-conditioned static gravity loads under an explicit vector/frame, embedded actuation declarations, geometry AABBs, and self-collision status.

Run `check-invariants` before and after every authoring-source edit. The output is `robot-spatial-invariant-report.v1`; every assertion records expected and actual values, pose, numeric errors, tolerances, source digests, and pass/fail state. Mass and gravity assertions still report `physical_world_completeness: not_established`; actuation assertions preserve declarations without promoting them to runtime truth. The command exits non-zero on any failure. `export --invariants` embeds the same evidence into context and facts and also exits non-zero while retaining diagnostic artifacts.

The contract must be authored from approved project intent, not captured blindly from whatever the current URDF happens to contain. Updating expected values solely because an edit failed removes the protection. See [invariant-contract.md](invariant-contract.md) for the complete schema and workflow.

## Triangle surface distance and solid collision

Use `surface-distance` for a direct geometry-frame pair query and `surface-collisions` for pose-conditioned robot self-collision. `export --surface-collisions` includes the same result in `model.json`, `context.md`, facts, and generated evaluation. It automatically inspects collision meshes, not unrelated visual meshes; package mappings are still required for `package://` URIs. An explicit `--inspect-meshes` or visual selection may request additional geometry independently.

The decision sequence is:

1. exact root-frame AABB distance rejects pairs farther apart than `contact_tolerance_m`;
2. overlapping AABBs and non-overlapping AABBs within contact tolerance become candidates;
3. exact supported triangle surfaces are searched with deterministic BVHs;
4. distance at or below `contact_tolerance_m` is contact/collision;
5. otherwise, closed consistently wound surface components are tested for containment;
6. unsupported or topologically incomplete unresolved candidates remain `indeterminate`.

An SRDF-disabled pair stays physically `collision` when the geometry says so. The policy fields explain whether a planner may ignore it; they never overwrite the measured spatial fact. Use `srdf_policy_filtered_self_collision_status` for the aggregate policy-filtered view and `self_collision_status` for the aggregate physical view. This robot-local self-collision engine is discrete at one pose. It does not compute swept-volume/continuous collision, penetration depth, a contact manifold, or dynamics. Robot/environment collision is a separate static-snapshot layer described by [world-scene-contract.md](world-scene-contract.md).

## Evaluation and understanding evidence

`export --generate-evaluation` creates a deterministic, bilingual competency suite from the canonical artifact, facts, concept graph, and optional functional model. `spatial_evaluation.py generate --action-assurance` can add one exact functional-model-bound action assurance after export. The public `questions.jsonl` contains prompts and submission shapes without expected values or evidence locators. The separately located private `answer-key.jsonl` contains expected answers, numeric tolerances, provenance, and exactness. Public `answer-template.jsonl` fixes the candidate submission format, and `manifest.json` records capability counts and grader controls without revealing the key path. When a functional model is present, five questions cover explicit component purpose without name inference, capability assertion versus structural grounding versus physical truth, relational preconditions/intended effects, positive grounded-versus-ungrounded action conclusions, and complete/incomplete inventory boundaries. When an action assurance is supplied, five more questions cover decision-time readiness, action-server lifecycle versus physical execution, effect observation versus causation, discrepancies, and evidence provenance/epistemic scope.

Keep the key outside every filesystem and context surface available to the candidate; merely placing it in another folder is not a security boundary. Give the candidate the robot sources, this skill, public manifest, questions, and answer template. Grade one-record-per-question JSONL with `spatial_evaluation.py verify`. The report includes total and per-capability accuracy plus missing, malformed, duplicate, unexpected, and numerically incorrect answers. Quaternion comparison accepts the equivalent `q` and `-q` forms. The generator self-check requires a perfect synthetic submission to pass and a one-missing-answer submission to fail.

Read-only competency does not establish edit competence. Use `spatial_edit_evaluation.py` for a blind authoring task whose private key pins the baseline and dependency digests, authorized XML attributes/elements, authorized invariant fields/membership, evaluator-owned poses, and required topology, frame-pose, joint-axis, or geometry-AABB outcomes. For supported leaf-branch add/remove, complete-subtree add/remove, and complete-subtree reparent changes, require `spatial_graph_edit.py` to compile a typed, baseline-bound change set and require the evaluator to reproduce the candidate URDF from it. Require complete typed edges for every structural edit. The grader removes or restores allow-listed changes and compares the rest of the semantic XML tree and canonical invariant contract exactly, then runs the full edited invariant contract. Use `spatial_edit_suite.py` to aggregate independently keyed edit categories without weakening task-level gates. See [edit-evaluation-contract.md](edit-evaluation-contract.md) and [graph-change-contract.md](graph-change-contract.md) for the schemas, controls, security boundary, and current coverage limit.

Capabilities cover topology, frame semantics, explicit project function/capability/affordance semantics, pose transforms, joint axes, kinematic causality, analytic instantaneous motion, finite configuration/rank/proximity witnesses, finite signed counterfactual endpoint motion, declared mass/center-of-mass/inertia aggregation, gravity-only static loads, embedded actuation/control declaration grounding, measured geometry, semantic/SRDF grounding, project-intent invariants, triangle-surface/solid collision, timestamp-policy selection, deterministic ROS JointState/TF capture normalization, model/scene/capture/observation layer separation, nominal observed-world transforms/collision, and the epistemic limits of declarations, finite endpoints, observations, AABB candidates, and sampled workspaces. `agent-context.json`, typed entity cards, and byte-indexed provenance facts present these capabilities through progressive disclosure without collapsing identity or silently upgrading modeled, functionally asserted, structurally grounded, declared, finite-configuration, finite-counterfactual, transported, observed, heuristic, sampled, or unknown states. See [agent-context-contract.md](agent-context-contract.md), [function-affordance-contract.md](function-affordance-contract.md), [configuration-atlas-contract.md](configuration-atlas-contract.md), [motion-atlas-contract.md](motion-atlas-contract.md), [ros-observation-adapter-contract.md](ros-observation-adapter-contract.md), and [temporal-observation-contract.md](temporal-observation-contract.md). A score measures only generated tasks for the artifact and query. It is evidence of operational consistency, not proof of general intelligence, sensor truth, physical validity beyond the engine boundary, or robustness to unseen robot families.

## Independent-oracle validation

Use `crosscheck_articulation_grammar.py` as the independent parser/FK oracle for the general joint law. It must not import production parser, matrix, grammar, or FK code. Preserve randomized tree/mimic/limit coverage, real-robot cases, unseen driver bindings, matrix tolerances, rejection controls, and explicit tree-only boundaries.

Use `crosscheck_constraint_graph.py` as the dependency-free independent oracle for supplemental mechanism execution. It imports no production parser, articulation, constraint, matrix, rank, residual, or solver implementation; generates planar parallelogram loops and linear coordinate couplings; calls only the public CLI; and independently recomputes analytic closure/coupling errors. Preserve seed, case counts, valid and violated controls, local-rank/mobility expectations, solver coverage, maximum errors, and exclusions.

Use `crosscheck_configuration_atlas.py` as the dependency-free independent oracle for finite configuration witnesses. It imports no production parser, articulation, constraint, configuration, matrix, Jacobian, rank, residual, or solver implementation; generates non-planar spherical three-revolute closures; calls only the public CLI; and independently evaluates `R_x(a)R_z(b)R_x(c)`, its two generic closure branches, singular-slice rank contrast, exact regeneration, and every stored node. Preserve randomized amplitudes, seed, cases, errors, rank contrast, node count, and exclusions.

Use `crosscheck_yourdfpy.py` to generate deterministic feasible joint poses and compare every URDF link-frame FK matrix with yourdfpy. The report records URDF SHA-256, optional upstream revision, engine versions, joint sampling ranges, mimic constraints, every supplied/resolved pose, translation and rotation tolerances, maximum errors, and uncovered domains.

Use `crosscheck_mass_properties.py` in an isolated yourdfpy/NumPy environment to independently parse inertials, transform them at deterministic feasible poses, and compare aggregate mass, center of mass, the complete inertia tensor, and the missing-inertial link set. Agreement validates only the declared selected tree and poses. It does not validate hardware mass, payload, parameter identification, contact dynamics, or unmodeled constraints.

Use `crosscheck_gravity_loads.py` in an isolated yourdfpy/NumPy environment to compute independent FK-based gravitational potential energy and obtain each independent driver's generalized gravity force as the central finite difference `-dU/dq`. This uses a separate parser/FK engine and a numerical energy derivative instead of the candidate's analytic force/moment projection. Preserve the root-frame gravity vector, pose seed/count, finite-difference step, per-driver units, tolerances, maximum errors, and discrepancies. Agreement validates only the enumerated declared inertials, poses, mimic behavior, and gravity convention; it does not validate actual mounting, payload, contact, full dynamics, actuation, control, or hardware.

Use `crosscheck_trimesh.py` on an exported model whose meshes were inspected. It independently reopens each mesh through trimesh and compares local bounds, surface area, processed watertight classification, and volume when the candidate marked volume trustworthy. It verifies mesh digests before comparison. Keep every optional oracle isolated from the core standard-library implementation so agreement is meaningful.

Use `crosscheck_fcl.py` in an isolated Python containing `python-fcl` and NumPy. It compares arbitrary triangle-pair distances and randomly transformed box collision/distance results against the Flexible Collision Library, plus explicit closed-box containment cases. Preserve the deterministic seed, case counts, tolerances, version, maximum errors, and discrepancy records. FCL agreement validates only those generated cases; it does not extend portable runtime support to cylinders, spheres, other mesh formats, or continuous collision.

Use `crosscheck_render_atlas.py` as an independent analytic oracle for the semantic render atlas. It generates randomized fixed, revolute, and prismatic chains with rotated box geometry, invokes only the public CLI, and independently recomputes FK, box support points, orthographic projection, convex hulls, depth intervals, screen fitting, pixel coordinates, SVG digests, and typed entity IDs without importing renderer internals. Preserve the seed, case and entity counts, separate world/projection and rounded-pixel tolerances, maximum numeric error, discrepancies, and CLI failures. Agreement validates the deterministic numeric/view contract for the generated cases; it does not make the atlas an independent physical, visibility, occlusion, perspective-camera, or photorealistic oracle.

Use `crosscheck_motion_atlas.py` as an independent analytic oracle for the counterfactual motion atlas. It generates randomized branched fixed, revolute, prismatic, and positive/negative-multiplier mimic models; invokes only the public CLI; and independently recomputes driver/follower feasible intervals, nominal/clipped/unavailable endpoints, branched FK, all-frame SE(3) deltas, causal boundaries, shared orthographic screen fitting, motion vectors, SVG digests, and typed entity IDs without importing any production parser, FK, motion, or rendering module. Preserve seed, case/driver/view counts, endpoint-status counts, tolerances, discrepancies, and CLI failures. Agreement validates only the generated finite endpoint and view contracts, not any intermediate trajectory, dynamics, physical motion, or safety.

Use `crosscheck_concept_graph.py` as the dependency-free independent structural oracle. It generates randomized serial, branched, fixed, revolute, prismatic, and mimic URDF trees; calls only the public CLI; and independently parses raw XML to recompute roots, complete typed edge sets, branch points, structural leaves, maximal serial segments, transitive paths, mimic-driven physical joints, and driver-affected link/pre-motion frames. It also checks exact-negative controls and the explicit leaf/end-effector and finite/global boundary clauses. Preserve seed, case/topology/mimic/query counts, discrepancies, CLI failures, and exclusions. Agreement validates the generated symbolic structure and query behavior, not physical truth, function, affordances, dynamics, or global constrained configuration space.

Use `crosscheck_functional_model.py` as the dependency-free independent function-grounding oracle. It must generate randomized raw URDF trees and project-owned function/affordance specs, call only public export/query/verify commands, independently parse XML plus public concept artifacts, and recompute typed entity/path/driver requirement outcomes, complete/incomplete inventory conclusions, and physical-boundary fields without importing production functional, concept, or URDF code. Preserve seed, case/requirement/query counts, exact-negative/open-world controls, discrepancies, failures, and exclusions. Agreement validates only project-declaration transcription, represented structural grounding, and query boundary behavior—not whether the declared function is physically real or executable.

Use `crosscheck_action_assurance.py` as the dependency-free independent action-evidence oracle. It must generate raw URDF plus project-owned functional declarations, call only public export/functional/action compile-query-verify commands, create randomized condition/lifecycle/effect reports, and independently recompute decision-time evidence selection, readiness, lifecycle consistency, effect timing, and outcome conclusions without importing production action-assurance, functional, concept, or URDF code. Preserve seed, case/scenario/query/assertion counts, discrepancies, failures, and exclusions. Agreement validates only content binding, deterministic time selection, bounded evidence accounting, public query projection, and exact regeneration—not producer truthfulness, clock synchronization, physical causation, hardware state, authorization, or safety.

Real URDF files may contain recoverable schema mistakes. If `<geometry>` contains exactly one recognized primitive plus misplaced non-geometry elements, retain the primitive and emit a validation warning naming ignored tags. Continue to reject missing, zero, or multiple recognized primitives. Never turn recovery into silent acceptance.

## Engine boundary

This standard-library engine is a kinematic, proof-carrying structural-concept, project-declared function/affordance, time-qualified action-evidence accounting, supplemental equality-constraint, finite-configuration-witness, finite-counterfactual-endpoint, URDF-declared mass-property and gravity-only static-load, embedded actuation-declaration, SRDF-semantic, STL/OBJ measurement, timestamp-selection, ROS-capture normalization, and discrete collision oracle for its enumerated representations. The concept layer supports typed compositional structure, proof closure, and closed-world negatives over complete tree/articulation projections, but it does not infer function, affordances, action plans, physical truth, or missing relations. The functional layer compiles explicit intent and deterministically grounds named requirements; it does not prove runtime condition truth, physical capability, observed effects, action success, hardware behavior, impossibility, or safety. The action-assurance layer content-binds one functional action instance, selects condition evidence at decision time, distinguishes goal/status/result protocol reports, and relates declared-effect observations to observed execution start; it does not authorize dispatch or prove producer truthfulness, clock synchronization, physical execution, causation, or safety. The constraint layer supports asserted rigid attachments, typed pair/distance/linear residuals, pose-conditioned rank/mobility, and local solving. The configuration layer supports explicit one-parameter samples, multi-seed and continuation solving, normalized merging/proximity, and observed full/passive rank candidates, but no exhaustive branch enumeration, certified singularities/global topology, inequality/contact/compliance, dynamics, calibration, or hardware truth. The motion layer supports exact finite signed FK endpoints but no interpolation or swept-volume claim. Scene and observation claims remain bound to declared snapshots/logs. It is not a trajectory generator, rigid-body dynamics engine, controller/runtime/hardware verifier, action dispatcher, physical action-execution verifier, clock synchronizer, probabilistic/continuous-collision oracle, multi-robot world, occupancy mapper, contact-manifold engine, globally complete constrained planner, or complete-reachability/configuration-space oracle. Preserve typed identities, clause modality/proofs, units, transform/force directions, constraint tolerances/status, chart samples/seeds/scales/thresholds, finite-step semantics, temporal/gravity conventions, digests, representation status, and provenance when extending it.

## Acceptance questions

A compatible implementation must answer these without XML mental arithmetic:

1. What is B relative to A at a named joint pose?
2. What direction does a joint axis point when expressed in frame A?
3. Which joint connects two links and what motion type does it permit?
4. How do visual and collision origins differ for a link?
5. Which exact spatial facts changed after a URDF edit?
6. Which links and geometry can move when joint J changes?
7. Which ordered joints connect the base to a requested tool link?
8. At this pose, which independent joint rate moves or rotates the TCP in each root-frame direction?
9. What target positions and orientations were actually observed under a reproducible, finite joint-space sample, and what does that sample not prove?
10. Which collision pairs are physically intersecting, contained, separated, or indeterminate at this pose, under what triangle representation and tolerance, and does SRDF policy suppress only planning or the geometry result itself?
11. Which project-owned spatial invariants passed or failed after an edit, at what pose and tolerance, and did the edit preserve approved frame, geometry, causal, and collision intent?
12. At this pose, what URDF-declared mass, center of mass, and inertia belong to the whole tree or a selected subtree, in which frame, and which links remain physically unmodeled because no valid inertial was declared?
13. Under an explicit gravity vector and frame at this pose, what generalized gravity force acts on each independent joint, what opposite ideal static holding effort balances it, and which inertial coverage gaps prevent a physical-world claim?
14. Which ros2_control systems, command/state interfaces, sensors, GPIOs, legacy transmissions, actuators, mechanical reductions, and joint dynamics values are embedded in the expanded URDF, and which runtime/hardware conclusions remain unestablished?
15. At query time `t`, which past joint/root/object samples were selected, how old were they, were future samples excluded, which values fell back to static declarations, what nominal transforms/collisions follow, and which physical-world or safety claims remain unestablished?
16. In each semantic atlas view, where do a named frame, joint edge, and geometry project in root, UV, depth, and pixel coordinates; which model, pose, and geometry bytes bind that view; and why is the view corroborating access to the same canonical model rather than independent visual or physical evidence?
17. For one independent driver at a baseline pose, which physical mimic joints and structural frames can it affect; what exact minus/plus finite endpoints are nominal, clipped, or unavailable; how does a named downstream frame move in root and shared-screen coordinates; which upstream frames stay fixed; and why do these endpoints not establish a path, swept volume, dynamics, hardware motion, or safety?
18. What are the root, branch points, structural leaves, and maximal serial segments; which proof clauses establish them; why is a structural leaf not automatically an end effector; and when is an exact negative causal answer permitted rather than unknown?
19. What is one project-declared component for, which exact concept entities are grouped into it, and which function, capability, and affordance clauses support the answer without name or geometry inference?
20. Which typed structural requirements ground a declared capability, which are exact closed-world, open-world, or finite evidence, and why does complete structural grounding still not prove physical execution or safety?
21. For a declared affordance, who is the actor/provider, what action targets which object type, which named preconditions and truth sources apply, what effects are merely intended, and what remains unknown before execution?
22. When an action is missing, is the relevant project affordance inventory explicitly complete or incomplete, what exact conclusion follows, and why is physical impossibility never established by inventory absence?
23. For one action instance at its declared decision time, which exact evidence records satisfy, falsify, or fail to establish every precondition, and why does a ready result still not authorize dispatch or establish physical executability or safety?
24. What goal response, status sequence, terminal result, and execution-start observation were recorded; are they internally consistent; and why does an action-server `succeeded` result not independently verify physical execution?
25. Which declared effects were observed before or after observed execution start, which discrepancies remain, and why do post-execution observation and reported success still not establish causal success or physical-world truth?
