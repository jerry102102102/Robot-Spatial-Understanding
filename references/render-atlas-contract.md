# Semantic render atlas contract

## Contents

1. Purpose and trust boundary
2. Artifact set
3. Binding and identity
4. Coordinate and screen contracts
5. Geometry fidelity and coverage
6. Agent load order
7. Verification
8. Failure handling
9. Unsupported claims

## Purpose and trust boundary

`robot-spatial-render-atlas.v1` gives an AI two synchronized encodings of one pose-conditioned robot model:

- standalone SVG views with typed `data-entity-id` attributes;
- numeric projection records for the same link origins, joint edges, geometry hulls, depth intervals, and highlighted frame axes.

The atlas is derived from the same canonical URDF geometry and forward-kinematics implementation as `model.json`. It makes visual grounding inspectable and reproducible; it is not an independent geometry engine, sensor image, calibrated camera, or physical-world observation.

## Artifact set

`export --render` writes:

```text
scene.svg
render-atlas/
├── manifest.json
└── views/
    ├── front.svg
    ├── side.svg
    ├── top.svg
    └── isometric.svg
```

`scene.svg` is a backward-compatible four-panel overview. `render-atlas/manifest.json` is the canonical visual-grounding artifact. Each standalone view is content-bound by SHA-256.

The manifest contains:

- `render_id` and `render_input_sha256`;
- exact URDF byte and semantic digests;
- pose name, complete resolved joint positions, and pose-binding digest;
- root/UV/pixel coordinate conventions;
- declared versus rendered geometry coverage;
- four view records;
- an explicit epistemic scope.

## Binding and identity

The render input digest binds:

1. robot name and root frame;
2. URDF byte and semantic digests;
3. resolved pose name and joint positions;
4. every renderable root-frame geometry point;
5. every declared geometry measurement status;
6. highlighted semantic frames.

Use these typed identities:

- atlas: `render_atlas/<render-id>`;
- view: `render_view/<render-id>/<front|side|top|isometric>`;
- geometry and link frame: `frame/<exact-URDF-frame-name>`;
- kinematic edge: `joint/<joint-name>`;
- drawn frame axis: `frame_axis/<frame-name>/<x|y|z>`.

An SVG element's `data-entity-id` must match the corresponding manifest record exactly. A view remains valid only for the bound model, resolved pose, inspection policy, and semantic highlight set.

## Coordinate and screen contracts

Every view declares an orthographic 2×3 matrix `M`:

```text
[u, v]^T = M × [x, y, z]^T
```

The standard view bases are:

| View | `u` | `v` | depth coordinate |
| --- | --- | --- | --- |
| front | `+X` | `+Z` | `+Y` |
| side | `+Y` | `+Z` | `+X` |
| top | `+X` | `+Y` | `+Z` |
| isometric | `(X-Y)/√2` | `-(X+Y)/√6 + 2Z/√6` | `(X+Y+Z)/√3` |

The screen record declares `center_uv_m` and `scale_px_per_m`. Pixel origin is top-left and pixel Y points down:

```text
pixel_x = width / 2 + (u - center_u) × scale
pixel_y = plot_top + plot_height / 2 - (v - center_v) × scale
```

Do not compare pixel positions across independently fitted views without using each view's mapping. Pixel distance is not a metric distance unless converted by the declared scale, and projection still discards depth.

## Geometry fidelity and coverage

Each geometry record states its projection support:

| Geometry | Support | Hull interpretation |
| --- | --- | --- |
| loaded STL/OBJ mesh | every transformed mesh vertex | exact convex hull of the loaded vertex set |
| box | eight transformed corners | exact declared-box projection hull |
| cylinder | two 32-sample boundary rings | deterministic curved-boundary approximation |
| sphere | three 24-sample great circles | deterministic curved-boundary approximation |

The atlas renders convex hulls, not visible triangle silhouettes. It does not perform surface rasterization, hidden-surface removal, depth ordering, lighting, material rendering, or perspective.

Check `coverage.complete_for_declared_geometry` and `unrendered_geometry_frames`. An unsupported or uninspected mesh stays explicitly absent. Visible collision geometry never proves visual-geometry coverage, and vice versa.

`depth_interval_m` preserves the lost orthographic coordinate for each geometry. It supports reasoning about projected overlap versus depth separation, but it is not an occlusion result.

## Agent load order

When an image helps answer a spatial question:

1. read `agent-context.json` for identity and unresolved claims;
2. retrieve `render_atlas/<render-id>` for binding and coverage;
3. retrieve exactly one `render_view/<render-id>/<view>` card;
4. follow its bound facts for projection, frame, edge, or geometry records;
5. inspect the corresponding SVG;
6. use `transform`, `bounds`, or other numeric queries when the view alone loses depth or exact surface information;
7. run `verify-render` when integrity or view/numeric consistency matters.

Never infer a transform, joint axis, distance, collision, or physical-world claim from pixel appearance alone.

## Verification

Generate and verify with the same inputs:

```bash
python3 scripts/robot_spatial.py export robot.urdf \
  --pose pose.json --semantics semantics.json \
  --inspect-meshes --package-map package-map.json \
  --render --out work/context

python3 scripts/robot_spatial.py verify-render robot.urdf \
  --pose pose.json --semantics semantics.json \
  --inspect-meshes --package-map package-map.json \
  --atlas work/context/render-atlas/manifest.json \
  --out work/render-verification.json
```

`verify-render` regenerates the four semantic projections and verifies:

- source/pose/input binding;
- projection matrices and fitted screen mappings;
- geometry hulls, depth intervals, frame origins, joint edges, and frame axes;
- standalone SVG digests;
- presence of every expected typed SVG entity ID;
- combined overview digest when present.

A pass proves deterministic reproduction from the same canonical implementation. Use a separate oracle or renderer to claim cross-engine agreement.

## Failure handling

| Failure | Meaning | Required action |
| --- | --- | --- |
| model or pose binding differs | atlas belongs to another input | regenerate; do not relabel |
| incomplete geometry coverage | one or more shapes were not measured | provide package map/inspection kind or keep the omission explicit |
| view numeric record differs | manifest was changed or generated by another contract | reject until reviewed and regenerated |
| SVG digest differs | rendered artifact changed after manifest creation | reject and regenerate |
| typed SVG ID missing | visual cannot be mapped back to its entity record | reject and fix renderer/artifact |
| curved primitive support is sampled | hull is not exact analytic silhouette | preserve approximate evidence label |

## Unsupported claims

Version 1 does not establish:

- photorealistic appearance or visible-surface segmentation;
- self-occlusion, environment occlusion, or depth-buffer ordering;
- perspective, lens distortion, camera intrinsics/extrinsics, or sensor pixels;
- exact analytic cylinder/sphere silhouettes;
- scene-object or observation-conditioned atlas rendering;
- independent collision, transform, or mesh truth;
- physical robot/world agreement or safety.

Extend the schema and evaluator before making those claims.
