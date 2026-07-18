# Counterfactual motion atlas contract

## Purpose

The static semantic render atlas makes links, frames, joint edges, and geometry visually addressable at one pose. It does not by itself make joint causality perceptually obvious. Generate a `robot-spatial-motion-atlas.v1` when the question is “what changes if this joint moves?”, “which parts stay fixed?”, “in which signed direction does a frame move?”, or “how does a mimic follower participate?”

Run either:

```bash
python3 scripts/robot_spatial.py export robot.urdf --pose pose.json \
  --motion-atlas --motion-angular-step-rad 0.1 --motion-linear-step-m 0.01 \
  --out work/context

python3 scripts/robot_spatial.py motion-atlas robot.urdf --pose pose.json \
  --motion-angular-step-rad 0.1 --motion-linear-step-m 0.01 \
  --out work/motion-atlas
```

The atlas is a second, causal encoding of the canonical kinematic model. It is not an independent physical-motion oracle.

## Artifact and identity

`motion-atlas/manifest.json` binds:

- the exact URDF semantic digest and root frame;
- the fully resolved baseline pose and its digest;
- angular and linear finite-step policy;
- every independent movable driver;
- mimic-constrained feasible intervals and physical joints driven;
- signed endpoint availability, clipping, joint positions, and exact FK consequences;
- four shared-screen views and their standalone SVG digests;
- the explicit epistemic boundary.

Standalone views are written as:

```text
motion-atlas/
├── manifest.json
└── drivers/<independent-driver>/<front|side|top|isometric>.svg
```

Agent entities are:

- `motion_atlas/<motion-id>`
- `motion_driver/<motion-id>/<independent-driver>`
- `motion_view/<motion-id>/<independent-driver>/<view>`

SVG entities use `motion_sample/<motion-id>/<driver>/<baseline|minus|plus>/<frame|joint>/<name>` and `motion_vector/<motion-id>/<driver>/<minus|plus>/frame/<name>`. Never shorten them to bare names when binding evidence.

## Independent drivers and mimic causality

The atlas creates one driver record for every non-fixed, non-mimic joint. A mimic joint is not perturbed independently. Instead, its full affine chain is reduced to:

```text
q_physical = multiplier * q_driver + offset
```

The driver interval is the intersection of every declared driver and mimic-follower position constraint. Negative multipliers reverse lower/upper contributions. Missing one-sided limits remain one-sided; continuous joints contribute no position bound. `physical_joints_driven` lists the driver and all mimic followers controlled by it.

Structural causality comes from the URDF tree. `affected_frames` lists frames whose root pose may change while every other independent driver stays fixed. The driver's own `joint/<driver>` pre-motion frame is upstream of its motion and must remain unchanged. Structural “may change” is not the same as numerically changed at one finite endpoint; exact cancellations or coincident origins can leave an affected frame stationary.

## Signed finite endpoints

For each driver, `minus` requests `-nominal_step` and `plus` requests `+nominal_step`. Angular joints use radians and prismatic joints use meters. Each endpoint is one of:

- `applied_nominal_step`: the full requested step is feasible;
- `clipped_to_feasible_limit`: a nonzero smaller step reaches the exact feasible bound;
- `unavailable_at_feasible_limit`: the baseline is already at that signed bound, so no endpoint is fabricated.

All other independent drivers are held at their resolved baseline positions. Mimic followers are recomputed, not held fixed. Available endpoints contain the complete resolved joint pose and exact forward kinematics.

This is a controlled finite counterfactual, not a derivative. Use the Jacobian for infinitesimal rate-to-twist questions. Use the motion atlas for a stated finite signed displacement. If the step changes, the motion-atlas identity changes.

## Frame and geometry effects

Every available endpoint records all-frame and link-only deltas. Root-origin displacement is expressed in the root frame. The relative transform named `baseline_frame_from_endpoint_frame` is:

```text
inverse(root_from_baseline_frame) * root_from_endpoint_frame
```

Its translation is expressed in the baseline frame. Its quaternion, axis, and angle describe endpoint orientation relative to baseline. `causality_check` compares numeric changes with structural affected frames, asserts that the pre-motion frame stayed fixed, and exposes any unexpected change instead of hiding it.

For every measured geometry frame, the atlas records baseline and endpoint root-frame AABBs, their center displacement, and the union of only those two endpoint AABBs. `endpoint_union_is_continuous_swept_volume` is always false. The union is useful for endpoint comparison but says nothing about intermediate occupancy or continuous collision.

## Shared-screen views

Each driver has front, side, top, and isometric orthographic views. Baseline and every available signed endpoint are fitted together once per driver. The projection basis, combined UV bounds, screen center, scale, and UV-to-pixel mapping are explicit in the manifest. Therefore a baseline-to-endpoint pixel vector is meaningful within one driver/view; pixels from different drivers or independently fitted artifacts must not be subtracted.

The SVG convention is:

- gray: baseline;
- blue: minus endpoint;
- red: plus endpoint;
- arrow: projected frame-origin displacement;
- ring at a stationary projected origin: orientation changed without a visible origin displacement in that view.

Depth, occlusion, hidden-surface visibility, perspective, and camera calibration are not inferred. A small or zero projected vector can be caused by view direction; consult the root-frame displacement and other views.

## Load order and question routing

For a causal motion question:

1. read `agent-context.json` identity, evidence, and unresolved-claim rules;
2. retrieve `motion_atlas/<motion-id>` for source, pose, policy, and coverage;
3. retrieve `motion_driver/<motion-id>/<driver>` for feasible interval, mimic structure, structural causality, and endpoint deltas;
4. retrieve one `motion_view/<motion-id>/<driver>/<view>` and inspect its SVG only when a projected direction or cross-grounding question requires it;
5. run `verify-motion-atlas` before relying on endpoint/view consistency;
6. state driver, baseline, signed applied step, units, endpoint status, affected frame, coordinate frame, and finite-endpoint boundary in the answer.

Do not start from an SVG and infer a joint, frame, axis, or causal relationship from color or visual proximity. The typed manifest record is the binding layer.

## Verification and failure handling

Run the verifier with the exact same URDF, pose, geometry inspection policy, package map, mesh kinds, and step values:

```bash
python3 scripts/robot_spatial.py verify-motion-atlas robot.urdf --pose pose.json \
  --motion-angular-step-rad 0.1 --motion-linear-step-m 0.01 \
  --atlas work/context/motion-atlas/manifest.json \
  --out work/motion-verification.json
```

The verifier regenerates the manifest, compares model/pose/policy/endpoint/view semantics, confines every artifact path to the atlas, checks each SVG digest, and checks every expected typed entity ID. A nonzero exit invalidates the atlas for reasoning. Do not repair a failed verification by editing the manifest or SVG; regenerate from the pinned source and policy.

For high assurance, run `crosscheck_motion_atlas.py`. It does not import production parser, FK, motion-atlas, or rendering modules. It generates randomized branched revolute/prismatic/fixed/mimic URDFs and independently recomputes feasible intervals, signed endpoint status, FK, SE(3) deltas, structural boundaries, shared projection, pixel mapping, SVG identity coverage, and digests.

## Epistemic boundary

The atlas establishes exact finite endpoint consequences under the supported URDF tree, declared geometry, baseline pose, and perturbation policy. It does not establish:

- any interpolation or path between baseline and endpoint;
- time, duration, velocity, acceleration, jerk, effort, dynamics, friction, damping, or contact response;
- a swept volume, continuous collision result, reachability proof, or safe trajectory;
- controller execution, actuator capability, calibration, hardware motion, or physical-world truth.

Use endpoint language: “under a +0.1 rad finite model counterfactual, the declared frame endpoint changes by …”. Do not say “the robot will follow this path” or “this motion is safe.”
