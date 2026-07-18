# ROS project preparation contract

## Purpose

Use `robot_spatial.py prepare` when the robot is an authoring project rather than one already-expanded URDF. The command compiles a concrete, provenance-bound input pack without requiring a hand-written package map:

```bash
python3 scripts/robot_spatial.py prepare workspace/src/robot_config/config/robot.urdf.xacro \
  --workspace-root workspace/src \
  --xacro-bin /opt/ros/bin/xacro \
  --arg variant:=production \
  --scene workspace/world-scene.json \
  --observations workspace/observations.json \
  --observation-query workspace/observation-query.json \
  --constraint-spec workspace/mechanism-constraints.json \
  --configuration-atlas-spec workspace/configuration-atlas-spec.json \
  --functional-spec workspace/function-affordance-spec.json \
  --inspect-mesh-kind collision \
  --out work/robot-prepared
```

The output path must not exist. A failed run removes only the directory it created, so a partial context is never mistaken for a completed preparation.

## Resolution

`prepare` finds the entrypoint's closest `package.xml`, scans each explicit workspace root, ensures the entry package itself is included, and rejects duplicate package names. With no explicit root it scans the entry package's parent so sibling source packages can be found. It ignores common generated workspace directories (`build`, `install`, and `log`) and VCS/cache directories. The scan is source-workspace discovery, not an installed ROS overlay resolver.

For Xacro, the command supplies a temporary minimal `ament_index_python` lookup layer backed by the discovered package map. This resolves and logs `$(find package_name)` calls without modifying the source tree. Explicit `--arg name:=value` mappings remain ordered and are recorded. After expansion, non-semantic XML comments are removed and package-owned absolute `filename`/`url` values are rewritten to equivalent `package://` URIs. This makes the resolved URDF and query IDs stable when the same package trees are relocated. The raw Xacro-output digest and every rewrite remain in expansion metadata. For URDF, it copies the source byte-for-byte to `resolved.urdf` and validates it.

The command does not guess missing Xacro arguments, select robot variants from filenames, execute launch files, source shell setup scripts, or expand SRDF Xacro. Provide every authoring choice that lacks a source default.

## Output

- `prepare.json`: top-level status, input format, source and resolved-URDF digests, package usage, context digests, and next action.
- `resolved.urdf`: validated concrete model used by every query.
- `resolved.urdf.meta.json`: Xacro runtime, mapping, input/output, and validation provenance when Xacro was used.
- `package-map.json`: discovered package name to absolute source directory mapping for local mesh resolution.
- `source-manifest.json`: conservative content manifest for the entry package and packages actually observed through Xacro lookup or `package://` URIs.
- `context/`: the normal progressive-disclosure agent pack, including the always-generated pose-independent `articulation-grammar.json`, proof-carrying `concept-graph.json`, controlled `concept-language.rsl`, and optional `functional-model.json`, each present artifact bound in `prepare.json` by SHA-256.

The concept graph is compiled only after the canonical model, articulation grammar, and any supplied constraint/configuration layers are final. It therefore has the same resolved-source identity as the rest of the prepared context. `prepare.json` records both concept artifact digests and directs an agent to query the graph before loading cards, facts, or the complete model. Run `verify-concept-graph` against `context/` before trusting a copied graph/language pair; verification requires exact graph regeneration and byte-identical controlled language. Read [concept-language-contract.md](concept-language-contract.md).

When `--scene` is supplied, `prepare` validates the scene against the resolved robot identity, binds its SHA-256 and snapshot ID into `prepare.json`, the source manifest/compilation record, and the generated context, and propagates package resolution to scene and robot meshes. The scene remains a separate source artifact; it is not merged into or inferred from Xacro.

When `--observations` and `--observation-query` are supplied, both are required together with `--scene`. `prepare` accepts normalized observation-log v1 or v2, verifies its exact model/scene binding, records both input SHA-256 digests and IDs in `prepare.json`, the source manifest, and the compilation record, and exports the same time-selected state and epistemic boundaries into `context/`. A v2 ROS-normalized log also yields a `ros_capture/` provenance card. Run ROS capture normalization against the resolved concrete URDF before `prepare`; the normalized log and query remain separate immutable source artifacts and do not become URDF or static-scene declarations.

When `--constraint-spec` is supplied, its digest must bind the articulation grammar deterministically generated from the resolved URDF. Preparation records the supplemental source in the source manifest/compilation record and publishes `context/constraint-graph.json` plus its export-pose evaluation, cards, and facts. Because the spec binds exact grammar bytes, compile the same resolved source grammar before authoring the spec. A violated preparation pose fails and removes the partial preparation root. Read [constraint-graph-contract.md](constraint-graph-contract.md).

When `--configuration-atlas-spec` is supplied, `--constraint-spec` is required and the atlas spec must bind the exact graph artifact preparation will regenerate. Preparation records the raw atlas spec in its source manifest/compilation record and publishes `context/configuration-atlas.json` plus atlas/chart/node/component cards and facts. It fails and removes the partial root if any sample misses its declared minimum. This means incomplete declared sampling is visible; it does not make complete declared sampling a global branch/topology proof. Read [configuration-atlas-contract.md](configuration-atlas-contract.md).

When `--functional-spec` is supplied, its source binding must match the resolved URDF semantic digest, generated articulation grammar artifact digest, and any supplied constraint/configuration artifacts. Preparation records the raw specification in `source-manifest.json` and `robot-spatial-source-compilation.v1`, then publishes `context/functional-model.json` plus functional typed cards and a `query-functions` route. The model is compiled after the concept graph so every satisfied capability requirement carries a recursive structural proof closure. Component membership, function, capability, conditions, intended effects, and affordances remain project assertions. A grounded structural requirement is not physical execution evidence. Read [function-affordance-contract.md](function-affordance-contract.md).

When `--render` is supplied, the generated `context/` contains both `scene.svg` and the digest-bound `render-atlas/` directory. The atlas binds the resolved concrete URDF semantic digest, selected pose, measured/unmeasured geometry status, semantic highlights, standalone view SVGs, and numeric projection records. Run `verify-render` against the preparation root's `resolved.urdf` with its `package-map.json` and identical pose/semantics/inspection options before claiming view/numeric consistency. Read [render-atlas-contract.md](render-atlas-contract.md).

When `--motion-atlas` is supplied, the generated `context/` also contains a digest-bound `motion-atlas/` directory. It binds the resolved concrete URDF semantic digest, fully resolved baseline pose, angular/linear step policy, independent drivers, full-chain mimic limits, signed finite endpoint states, all-frame/geometry endpoint deltas, shared-screen views, and standalone SVG digests. Run `verify-motion-atlas` against the preparation root's `resolved.urdf` with its `package-map.json` and identical pose/inspection/step options before claiming endpoint/view consistency. Preparation does not turn these endpoints into an interpolated trajectory, swept volume, dynamics result, controller behavior, hardware motion, or safety proof. Read [motion-atlas-contract.md](motion-atlas-contract.md).

Read `prepare.json` first to confirm resolution, then `context/agent-context.json`, then issue one task-relevant `query-concepts` request for structure and one `query-functions` request when explicit function knowledge is relevant. Load entity cards, facts, or `model.json` only when the returned proof closures are insufficient. The context manifest and canonical model contain a `robot-spatial-source-compilation.v1` record binding the source manifest, package map, entrypoint, mappings, resolved URDF, Xacro resolution, and optional raw functional specification.

## Provenance boundary

Package tree identity hashes sorted relative paths, file/symlink kinds, symlink targets, and file contents; machine-local package root paths are excluded from the tree digest. The manifest intentionally hashes the complete source trees of used packages because Xacro can compute include/YAML paths dynamically and a static include parser would under-report the dependency closure.

This is a conservative source binding, not proof that every hashed file influenced the model. Conversely, runtime code outside the source packages—most importantly the Xacro executable and its Python dependencies—is not captured by package tree hashes. The Xacro launcher file digest is recorded, but reproducibility still requires the recorded runtime environment or an independently pinned container.

An expanded URDF is authoritative for the supported declared model only after validation. Package discovery does not establish semantic roles, function, affordances, calibration correctness, full dynamics, controller/plugin/runtime/hardware behavior, collision safety, or physical agreement with a built robot. A supplied functional specification establishes project intent and deterministic requirement grounding only; it does not establish runtime conditions, observed effects, physical capability, impossibility, or safety. A supplied world scene establishes only a digest-bound declared static snapshot; preparation does not prove its source labels, currency, completeness, physical calibration, or omitted objects. A bound observation pair establishes only deterministic sample selection and age/fallback status under its declared clock policy; it does not establish source truth, clock synchronization, physical completeness, uncertainty-bounded geometry, or safety. Embedded control records may be transcribed from the resolved URDF, but external runtime and hardware remain outside project preparation.
