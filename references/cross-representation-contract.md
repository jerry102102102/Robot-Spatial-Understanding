# Cross-representation kinematic law contract

Use this contract when deciding whether two URDF, SDF, or MJCF sources describe the same supported kinematic structure.

## Semantic objective

Source syntax is provenance, not the robot law. Compile every supported source into `robot-spatial-articulation-grammar.v1`, then keep two identities separate:

- `grammar_id` and `grammar_input_sha256` bind one exact source artifact and import contract;
- `law_identity.canonical_law_sha256` hashes the source-binding-free typed law projection.

The common edge law is:

```text
parent_from_child(q)
  = parent_from_joint_pre_motion
  × joint_motion(q_joint)
  × joint_post_motion_from_child_zero
```

URDF makes the post-motion constant identity. SDF joint frames and MJCF joint anchors may require a non-identity post-motion constant. Omitting that factor changes the mechanism.

## Source semantics and authority

The importer follows the official format semantics:

- [SDFormat pose/frame semantics](https://sdformat.org/tutorials/specification/pose_frame_semantics/1.7/) define link poses relative to the model by default, joint poses relative to the child link by default, and `axis/xyz@expressed_in`.
- [SDFormat joint specification](https://sdformat.org/spec/1.11/joint/) defines supported joint pose, axis, and limit fields.
- [MuJoCo modeling coordinates](https://mujoco.readthedocs.io/en/stable/modeling.html) define local body and element frames.
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html) defines body-local joint `pos`, `axis`, hinge/slide motion, limits, compiler angle units, and multiple-joint ordering.

These links specify format semantics; they are not runtime or physical validation.

## Strict import boundary

`articulation-grammar --format auto` detects `<robot>`, `<sdf>`, or `<mujoco>`.

Supported SDF subset:

- one flat model and no world;
- one tree of fixed, revolute, continuous, or prismatic joints;
- recursive `pose@relative_to`, explicit frames, and `axis/xyz@expressed_in`;
- link, joint-pre-motion, visual, collision, inertial, and supported declared frames.

Reject nested/includes, non-tree or multi-DOF joints, legacy `use_parent_model_frame`, and unnormalized mimic semantics. Do not silently discard them.

Supported MJCF subset:

- compiled canonical local-coordinate MJCF with no includes/default classes or meta-elements;
- exactly one welded top body;
- zero or one named hinge/slide joint per descendant body;
- welded bodies, local quaternion frames, explicit joint ranges and compiler angle conversion;
- body, joint-pre-motion, named site/geom, and explicit inertial frames.

Reject equality constraints, free/ball joints, multiple joints on one body, defaults/includes, `attach`, `frame`, `replicate`, and non-quaternion orientation encodings. Canonicalize a richer MJCF through the official compiler and save it before import. Rejection is evidence that the common law would otherwise be incomplete.

## Identifier correspondence

Typed names are semantic identities. Never infer that two differently named links or joints correspond merely because their strings or geometry look similar.

When names differ, provide `robot-spatial-articulation-correspondence.v1`:

```json
{
  "schema_version": "robot-spatial-articulation-correspondence.v1",
  "reference_grammar_sha256": "...",
  "candidate_grammar_sha256": "...",
  "candidate_to_reference": {
    "links": {"candidate_base": "reference_base"},
    "joints": {"candidate_joint": "reference_joint"},
    "frames": {}
  }
}
```

Link and joint maps must be complete bijections. Link frames and `joint/<name>` frames are derived automatically. Every other frame requires an explicit mapping. Both grammar digests must match before the correspondence is used.

## Commands

```bash
python3 scripts/robot_spatial.py articulation-grammar robot.sdf \
  --format auto --out work/sdf-grammar.json

python3 scripts/robot_spatial.py articulation-grammar robot.xml \
  --format mjcf --out work/mjcf-grammar.json

python3 scripts/robot_spatial.py verify-articulation-grammar robot.sdf \
  --grammar work/sdf-grammar.json --out work/sdf-verification.json

python3 scripts/robot_spatial.py compare-articulation-grammars \
  work/urdf-grammar.json work/sdf-grammar.json \
  --correspondence work/sdf-to-urdf.json \
  --out work/cross-representation-report.json
```

The comparison requires exact equality of mapped canonical law projections and independently executes all mapped frames over deterministic unseen driver probes. A pass establishes the enumerated common supported tree law, not every construct in either source language.

For implementation-level evidence, run the production-independent generator/FK oracle:

```bash
python3 scripts/crosscheck_cross_representation.py \
  --cases 48 --post-anchor-cases 24 --poses-per-case 3 \
  --out work/cross-representation-oracle.json
```

It does not import the production parser, grammar, matrix, mapping, or evaluator implementation. It generates differently named URDF/SDF/MJCF trees, derives expected link and joint frames analytically, exercises fixed/revolute/continuous/prismatic joints, serial and branched trees, and adds SDF/MJCF cases with non-identity post-motion joint anchors. Preserve counts, maximum matrix error, failures, seed, and tolerance.

For agent-level evidence, a `raw_sources` suite task may declare `articulation_sources` and `articulation_comparisons` as defined in [evaluation-suite-contract.md](evaluation-suite-contract.md). Publish only raw sources, digest-bound typed correspondence, questions, and an empty template. The candidate must generate and verify every grammar and comparison under its own task-local work root; publishing an evaluator-generated grammar or comparison invalidates the blind claim.

## Agent answer contract

For every cross-representation claim, report:

- both artifact digests and source formats;
- both import contracts and rejected/omitted boundaries;
- exact identifier mode or correspondence digest;
- canonical projection hashes;
- probe count, all-frame evaluation count, tolerance, and maximum matrix error;
- whether the result is `equivalent` or `different`;
- the first structural difference when different.

Do not call two files equivalent from zero-pose appearance, matching names, mesh similarity, or one end-effector transform. Do not upgrade a common kinematic-law pass into equality of dynamics, contacts, actuators, sensors, plugins, constraints, controllers, calibration, or physical robots.
