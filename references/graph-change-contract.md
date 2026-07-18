# Typed graph change contract

## Purpose

`spatial_graph_edit.py` turns a small, explicit robot-graph intent into deterministic URDF. The change set is the reasoning artifact: it names graph nodes and edges, states the parent transform and removal preconditions, and binds itself to one exact baseline. The compiled URDF remains the interoperable authoring artifact.

This version uses schema `robot-spatial-graph-change-set.v1` and emits `robot-spatial-graph-edit-report.v1`.

## Common fields

Every change set contains:

- `change_set_id`: stable identifier for the requested graph edit;
- `robot`: exact URDF robot name;
- `baseline_urdf_sha256`: SHA-256 of the only baseline to which the operations may apply;
- `operations`: one or more uniquely identified typed operations, applied in order.

The compiler validates the baseline before applying anything, validates the intermediate tree before every subsequent operation, and parses the final output again before reporting success. The report records source, change-set, and output digests; every operation; complete topology before/after; added/removed link and joint sets; and changed typed edges.

## `add_leaf_link`

The operation requires:

```json
{
  "operation_id": "add-camera-leaf",
  "type": "add_leaf_link",
  "parent_link": "tool0",
  "new_link": "camera_link",
  "new_joint": "camera_mount",
  "joint_type": "fixed",
  "origin": {
    "xyz_m": [0.1, 0.0, 0.05],
    "rpy_rad": [0.0, 0.0, 1.5707963267948966]
  }
}
```

The parent must exist; both new names must be unused; all transform values must be finite; and the joint type must be `fixed`. The compiler creates one empty `<link>` and one fixed `<joint>` whose origin is `parent_link_from_new_link` using URDF `xyz` meters and fixed-axis `rpy` radians.

An empty link is a coordinate frame, not a physical sensor or body. Use `add_subtree` when a new link needs an exact visual, collision, or inertial payload or a movable attachment. Transmissions, controllers, and extension-specific semantics remain outside this operation.

## `remove_leaf_link`

The operation requires:

```json
{
  "operation_id": "remove-camera-leaf",
  "type": "remove_leaf_link",
  "link": "camera_link",
  "expected_parent_link": "tool0",
  "expected_parent_joint": "camera_mount"
}
```

The named link must exist, must not be the root, must have no child joints, and must have exactly the stated parent link and parent joint. The top-level link and joint declarations must be unique. Removal is rejected if another XML element outside the removal set contains either name as a complete attribute value or complete text value. On success, the complete leaf link element and its parent joint element are removed.

## `add_subtree`

The operation inserts one connected tree with exactly one attachment to an existing parent:

```json
{
  "operation_id": "add-gimbal-tree",
  "type": "add_subtree",
  "root_link": "gimbal_base",
  "expected_parent_link": "tool0",
  "links": [
    {"name": "gimbal_base"},
    {
      "name": "camera_link",
      "element_xml": "<link name=\"camera_link\"><collision><geometry><box size=\"0.1 0.04 0.06\"/></geometry></collision></link>"
    }
  ],
  "joints": [
    {
      "name": "gimbal_mount",
      "joint_type": "fixed",
      "parent_link": "tool0",
      "child_link": "gimbal_base",
      "origin": {"xyz_m": [0.1, 0.0, 0.05], "rpy_rad": [0.0, 0.0, 0.0]}
    },
    {
      "name": "camera_yaw",
      "joint_type": "revolute",
      "parent_link": "gimbal_base",
      "child_link": "camera_link",
      "origin": {"xyz_m": [0.0, 0.0, 0.1], "rpy_rad": [0.0, 0.0, 0.0]},
      "axis_xyz": [0.0, 0.0, 1.0],
      "limit": {"lower": -1.2, "upper": 1.2, "effort": 2.0, "velocity": 1.5}
    }
  ]
}
```

`root_link` must be one of the new links, `expected_parent_link` must already exist, and all new link and joint names must be unused in their namespaces. There must be exactly one incoming joint per new link, exactly one joint from `expected_parent_link` to `root_link`, and every other parent must be a new link in the same declared tree. Disconnected branches, multiple parents, and cycles are rejected before compilation.

Omit `element_xml` to create an empty link, or provide exactly one `<link>` whose `name` matches the declared link. The complete link element is preserved, while direct joint declarations inside it are forbidden. This permits visual, collision, and inertial payloads that the core URDF engine can validate; other nested extensions remain opaque XML and need their own validator before making semantic claims about them.

New joints are typed as `fixed`, `revolute`, `continuous`, or `prismatic`. Every joint explicitly declares parent, child, and origin. A movable joint also requires a nonzero axis plus finite effort and velocity limits; revolute and prismatic joints additionally require ordered lower/upper limits. The compiler constructs these joint elements rather than accepting raw joint XML, validates the resulting complete tree, and verifies that its compiled subtree membership exactly equals the declaration.

## `remove_subtree`

The operation removes one complete non-root subtree, including its incoming attachment joint:

```json
{
  "operation_id": "remove-gimbal-tree",
  "type": "remove_subtree",
  "root_link": "gimbal_base",
  "expected_parent_link": "tool0",
  "expected_parent_joint": "gimbal_mount",
  "expected_subtree": {
    "links": ["gimbal_base", "camera_link"],
    "joints": ["gimbal_mount", "camera_yaw"]
  }
}
```

The root must exist and cannot be the robot root. The current attachment link and joint must exactly match the precondition. `expected_subtree.links` must enumerate the root and every descendant; `expected_subtree.joints` must enumerate the incoming attachment joint and every descendant joint. Missing or extra members reject the operation. Every named top-level element must be unique.

Before removal, the compiler scans every XML element outside the removal set and rejects complete attribute or text values equal to any removed link or joint name. This catches common URDF extension references such as `<gazebo reference="camera_link">` and transmission `<joint><name>camera_yaw</name></joint>`. It does not parse embedded expressions or token lists; projects that encode references inside compound strings need a domain-specific reference validator. After removal, the compiler validates the remaining connected tree.

## `reparent_subtree`

The operation moves an existing attachment joint and every descendant below its child link:

```json
{
  "operation_id": "reparent-slider",
  "type": "reparent_subtree",
  "joint": "slide",
  "child_link": "slider_link",
  "expected_parent_link": "arm_link",
  "expected_joint_type": "prismatic",
  "new_parent_link": "base_link",
  "new_origin": {
    "xyz_m": [0.0, 1.0, 0.0],
    "rpy_rad": [0.0, 0.0, 0.0]
  },
  "expected_subtree": {
    "links": ["slider_link", "tool0"],
    "joints": ["slide", "tool_mount"]
  }
}
```

`expected_subtree.links` contains the child root and all descendants. `expected_subtree.joints` contains the incoming attachment joint and every descendant joint. Both sets must exactly match the current graph. The attachment joint name, child, current parent, and type are explicit preconditions. The new parent must exist, differ from the current parent, and remain outside the moved subtree. `new_origin` is `new_parent_link_from_joint/<joint>` in meters and fixed-axis RPY radians.

The compiler replaces only the attachment joint's `<parent>` and `<origin>`. It preserves joint type, child, axis, limits, mimic/calibration/dynamics declarations, every link payload, and all descendant joints. It then validates the complete tree and reports the changed edge. Recompute every pose, chain, causal set, joint axis, geometry bound, collision result, and invariant affected by the new ancestry.

## Evaluation requirements

A structural blind-edit task should require all of the following:

1. the candidate change set compiles from the private-key-pinned public baseline;
2. the compiler output is semantically identical to the submitted URDF;
3. exact top-level link/joint additions, removals, or replacements match the private element allowlist;
4. an evaluator-owned `topology` outcome matches the complete root/link/joint graph and every typed parent-child edge for structural edits;
5. frame-pose outcomes verify new edge transforms when applicable;
6. invariant additions, removals, or field changes explicitly capture the approved project-intent delta;
7. every remaining and updated invariant passes on the edited model;
8. unchanged and collateral-damage controls fail for the intended reasons.

## Boundary

This contract now expresses fixed-leaf add/remove, complete subtree add/remove, and complete-subtree reparenting. `add_subtree` can carry a complete link element, but only link identity and the core engine's supported visual/collision/inertial semantics are validated; arbitrary extension payload semantics are not. New joints do not yet express mimic, safety, calibration, dynamics, or extension-specific metadata. Sequential operations are accepted and every intermediate tree is validated, but current benchmark evidence does not establish multi-operation transaction generality.

The contract still does not express renaming with reference rewrites, closed loops, planar/floating joints, Xacro-source edits, mesh-content replacement, typed inertial/dynamic-property mutation, transmissions, controllers, Gazebo/plugin semantics, or SRDF migration. Stop at this boundary instead of encoding those requests as unconstrained XML patches. Extend the schema with typed preconditions, deterministic compilation, domain-specific reference validation, semantic outcomes, and negative controls first.
