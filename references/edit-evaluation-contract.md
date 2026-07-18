# Blind edit evaluation contract

## What this proves

`spatial_edit_evaluation.py` tests a narrower and stronger claim than question answering: given an explicit spatial change request, can a candidate produce authoring artifacts that realize the requested transform without making any other semantic source or project-intent change?

A passing report is evidence only for the task's declared robot, source digests, authorized attributes, invariant fields, poses, transforms, and tolerances. It is not evidence of unrestricted URDF authoring ability, physical safety, controller compatibility, manufacturability, or dynamics correctness.

## Public/private separation

Give the candidate:

- the baseline URDF and its referenced local geometry;
- the baseline `robot-spatial-invariants.v1` contract;
- package map and other source dependencies;
- the public `robot-spatial-edit-task.v1`;
- a public change-set template when `require_graph_change_set` is enabled;
- this skill.

Do not give the candidate `robot-spatial-edit-key.v1`. Keep the key outside every filesystem, tool, prompt context, retrieval source, and generated archive available to the candidate. Directory naming alone is not a security boundary.

The public task identifies the robot, natural-language request, input paths, immutable dependencies, required submission paths, workflow, and public acceptance criteria. The private key pins baseline SHA-256 digests and defines the exact source allowlist, exact invariant allowlist, required spatial outcomes, and tolerances.

## Current private-key fields

`baseline` contains `urdf_sha256`, `invariants_sha256`, `robot`, and `root_link`. When `task.inputs.protected_files` lists package maps, meshes, calibration files, or other immutable dependencies, `baseline.protected_file_sha256` must contain exactly the same path IDs and their digests. Baseline integrity fails before candidate grading if any protected dependency changes.

Each `authorized_urdf_changes` entry selects one named top-level entity such as a `<joint>` or `<link>`, a direct-child path with explicit zero-based indices, and one numeric three-vector attribute. This covers `origin.xyz`, `origin.rpy`, `axis.xyz`, and nested primitive dimensions such as `collision/geometry/box.size`. `expected_numeric_vector` is the required value. The legacy `joint` + `child_tag` selector remains accepted. The grader replaces the candidate attribute with its baseline value and then requires the entire canonical XML tree to equal the baseline. XML formatting whitespace and attribute order are ignored; element order, tags, attributes, non-whitespace text, and every unapproved value are protected.

`authorized_urdf_element_additions`, `authorized_urdf_element_removals`, and `authorized_urdf_element_replacements` define the exact permitted top-level `<link>`/`<joint>` semantic delta. An addition or replacement includes its complete candidate `element_xml`; numeric XML values are compared semantically, while tags, child order, names, and nonnumeric values remain exact. A removal names the pinned-baseline element. A replacement requires exactly one same-named element on both sides. After approved elements and attributes are removed or restored on their respective sides, the complete canonical XML trees must match.

Each `authorized_invariant_changes` entry selects one assertion ID and a nested `field_path`, then supplies the approved `expected_value`. `authorized_invariant_additions` contains complete assertions, and `authorized_invariant_removals` contains exact assertion IDs. The grader removes the approved membership delta, restores approved fields, and requires the complete canonical contract to equal the baseline. This protects assertion IDs, types, poses, frames, tolerances, expected values, assertion membership, and top-level schema fields not explicitly authorized.

Set `require_graph_change_set` for a supported structural edit. The candidate must submit a typed `robot-spatial-graph-change-set.v1`; the evaluator compiles it again from the digest-pinned baseline and requires the complete semantic XML tree to equal the submitted URDF. See [graph-change-contract.md](graph-change-contract.md).

Each `required_spatial_outcomes` entry owns its joint values and explicit pose name. Supported types are:

- `frame_pose`: explicit `from` and `to`, translation, quaternion, and translation/rotation tolerances. Transform direction is `from_from_to`: pose of `to` expressed in `from`.
- `joint_axis`: joint, expression frame, signed unit vector, and angular tolerance. Opposite directions differ by 180 degrees.
- `geometry_aabb`: geometry frame, root-frame minimum/maximum coordinates, and AABB tolerance.
- `topology`: exact robot root plus the complete unordered sets of link and joint names. Add `expected.edges` to require one complete `{joint, type, parent_link, child_link}` record for every joint; require this for structural edits because node-name sets alone do not prove parentage or joint type.

## Acceptance checks

The report schema is `robot-spatial-edit-report.v1`. An edit passes only if every emitted check passes:

1. Public baseline URDF, invariant contract, and protected dependencies match the private SHA-256 digests.
2. Candidate URDF is a supported, connected, single-root model.
3. Robot name and root link remain unchanged.
4. When required, the submitted typed graph change set deterministically reproduces the candidate URDF from the pinned baseline.
5. Every authorized XML attribute and top-level element membership change has its requested value/content.
6. No other semantic XML node, attribute, or text changed.
7. Every evaluator-owned spatial outcome matches at its explicit pose and tolerance.
8. Candidate invariant contract is valid for the edited robot.
9. Every authorized invariant field and assertion membership change has its approved value/content.
10. No other canonical invariant field changed.
11. Every updated and protected invariant passes on the edited model.

Exit code `0` means accepted, `1` means rejected, and `2` means the evaluation inputs are structurally invalid.

## Multi-task suites

Use `spatial_edit_suite.py` with a public `robot-spatial-edit-suite.v1` manifest and private `robot-spatial-edit-suite-key.v1` key manifest. Each task retains its own isolated key, category, submission paths, and complete task report. The suite reports total and per-category pass rates and exits non-zero when `minimum_pass_rate` is not met. Use `1.0` for a high-assurance gate; a lower research threshold must never hide which individual tasks failed.

For a generalization claim, vary robot identity, names, topology shape, attachment joint type, descendant joint composition, mount rotation, pose, and invariant consequences. Repeating one baseline with different numbers establishes parameter coverage, not structural generalization. Run passing, unchanged, and collateral controls for every task and publish only answer-free aggregate evidence.

## Required controls

Do not trust a new task definition until all three controls have been exercised:

- Passing control: the minimal correct URDF and approved invariant update must pass every check.
- Unchanged control: resubmitting the baseline must fail graph reproduction when required, requested XML value/element, required spatial outcome, and approved invariant value/membership checks.
- Collateral-damage control: a submission with the correct target plus one unapproved change must fail the XML allowlist and, when covered by intent, the protected invariant gate.

Keep control submissions and the private key outside the public candidate package. A public control summary may expose statuses and check IDs, but it must not introduce secret expectations beyond the public task.

## Extension boundary

The attribute selector intentionally supports numeric three-vector attributes reached through child elements below one named top-level entity. It covers joint-origin translation/RPY rotation, joint-axis direction, geometry origin, mesh scale, and three-vector primitive dimensions when paired with a supported outcome. Typed graph edits add/remove one fixed leaf, insert a complete connected subtree with fixed/revolute/continuous/prismatic typed joints and optional complete link payloads, remove one exact complete non-root subtree, or reparent one existing complete subtree through its unchanged attachment joint.

The evaluator's exact element and invariant allowlists cover these structural deltas, but they do not add semantic validation for opaque XML extensions. Add a schema version and deterministic normalization before evaluating alternate rotation encodings, scalar attributes, joint mimic/safety/calibration/dynamics metadata, renaming/reference rewrites, mesh-content replacement, material/transmission/controller changes, Xacro authoring edits, SRDF migration, or dynamic parameters. Do not simulate broader coverage with loose XML comparison or post-hoc invariant rewrites.
