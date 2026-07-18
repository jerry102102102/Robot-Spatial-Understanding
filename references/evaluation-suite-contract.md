# Multi-robot read-only evaluation suite contract

## Purpose

Use `spatial_evaluation_suite.py` when one generated evaluation is too narrow to support a generalization claim. The suite binds multiple independently sourced robot inputs, question sets, templates, and private keys under one aggregate gate.

Choose one candidate input mode for the whole suite:

- `generated_context` evaluates comprehension and retrieval from an evaluator-generated canonical representation.
- `raw_sources` evaluates autonomous source-to-representation understanding: the candidate receives raw URDF or Xacro inputs, optional project function/affordance specs, optional action-evidence bundles/sources, optional declared URDF/SDF/MJCF articulation bundles with typed correspondences, optional digest-bound supplemental constraint specs, and any declared raw world-scene/observation sources, but no canonical model, generated grammar/concept/functional/action-assurance/constraint graph/evaluation/solution/comparison, context, entity cards, facts, indexes, or rendered scene.

Neither mode tests raw physical perception, full dynamics, controller/runtime/hardware operation, unrestricted world knowledge, or safe editing. Static-gravity, actuation, world-scene, temporal, visual, counterfactual, and action-evidence layers remain limited to their declared inputs. Action-assurance questions test content binding, decision/evaluation-time selection, bounded readiness, observed goal/status/result lifecycle, declared-effect observations, discrepancies, and epistemic boundaries; they do not prove producer truthfulness, clock synchronization, dispatch authorization, physical execution, causation, hardware state, or safety. Articulation questions cover digest-bound pose-independent tree laws. Supplemental-constraint questions cover only supplied attachment/relation assertions, typed residuals at enumerated poses, pose-conditioned numerical local mobility, and declared local solves. Configuration-atlas questions additionally cover explicit finite one-parameter charts, multi-seed satisfying nodes, proximity components, and observed rank-drop candidates; they do not prove exhaustive branches, global DOF/topology, certified singularities, uniqueness, assembly, calibration, compliance, contact, dynamics, hardware, or physical truth.

## Build configuration

The evaluator creates `robot-spatial-evaluation-suite-build.v1` outside candidate context. In `generated_context` mode, present `articulation-grammar.json`, `concept-graph.json`, `concept-language.rsl`, `functional-model.json`, constraint graph/evaluation, `configuration-atlas.json`, and present `render-atlas/` and `motion-atlas/` directories are copied and digest-bound alongside the required context artifacts. Concept questions may require a strict `robot-spatial-concept-query.v1`; functional questions may require `robot-spatial-functional-query.v1`. Both must preserve returned modalities and proof closures.

```json
{
  "schema_version": "robot-spatial-evaluation-suite-build.v1",
  "suite_id": "real-robots-v1",
  "candidate_input": "generated_context",
  "tasks": [
    {
      "task_id": "panda",
      "robot_family": "articulated_manipulator",
      "context_dir": "/evaluator/exports/panda",
      "evaluation_dir": "/evaluator/exports/panda/evaluation",
      "answer_key": "/evaluator/keys/panda.jsonl",
      "source": {"repository": "https://example/repo", "commit": "pinned", "sha256": "..."}
    }
  ]
}
```

Run:

```bash
python3 scripts/spatial_evaluation_suite.py build build.json \
  --public-out candidate-readable/real-robots-v1 \
  --private-out evaluator-only/real-robots-v1
```

The roots must be disjoint and neither may contain the other. They must be absent or empty. The builder copies the progressive context, a location-sanitized canonical model, public evaluation artifacts, and optional scene into one task directory. It copies keys only into the private root. Every public file receives a SHA-256 entry; the private manifest binds the exact public manifest and every answer key.

Before involving a candidate, run the private synthetic control:

```bash
python3 scripts/spatial_evaluation_suite.py self-check \
  candidate-readable/real-robots-v1/manifest.json \
  evaluator-only/real-robots-v1/manifest.json
```

It materializes submissions only inside an automatically removed temporary directory, requires a perfect key-derived suite to pass, and requires the same suite with one answer removed to fail. This validates suite wiring and gates, not model competence.

Machine-local URDF, SRDF, invariant, and mesh locations are replaced with `sha256:<content-digest>` when their record already carries that digest. The spatial-truth digest likewise excludes only those content-bound locations. Relocation therefore preserves truth identity, while a structural or numeric change still changes it.

## Raw source mode

To test the full source-to-representation path, set `candidate_input` to `raw_sources` and replace `context_dir` with `source_dir` plus a constrained `candidate_task`:

```json
{
  "task_id": "held-out-arm",
  "robot_family": "held_out_serial_arm",
  "source_dir": "/evaluator/raw/held-out-arm",
  "evaluation_dir": "/evaluator/ground-truth/held-out-arm/evaluation",
  "answer_key": "/evaluator/keys/held-out-arm.jsonl",
  "candidate_task": {
    "schema_version": "robot-spatial-raw-source-task.v1",
    "input_format": "urdf",
    "entrypoint": "source/robot.urdf",
    "export_options": {
      "pose": "source/pose.json",
      "scene": "source/world-scene.json",
      "observations": "source/observations.json",
      "observation_query": "source/observation-query.json",
      "functional_spec": "source/function-affordance-spec.json",
      "motion_atlas": true,
      "motion_angular_step_rad": 0.1,
      "motion_linear_step_m": 0.01,
      "render": true,
      "workspace_samples": 0
    },
    "action_assurances": [
      {
        "assurance_id": "attempt-17",
        "functional_model_source": "exported_functional_model",
        "evidence_bundle": "source/action-evidence-bundle.json",
        "output": "action-assurance.json"
      }
    ]
  }
}
```

Version 1 accepts either an expanded `.urdf` entrypoint or an executable `.urdf.xacro`/`.xacro` entrypoint. A URDF task must not contain executable Xacro elements. A Xacro task must contain executable Xacro elements and declare its expansion separately from export options. Path-valued `scene`, `observations`, `observation_query`, `pose`, `semantics`, `invariants`, `functional_spec`, `constraint_spec`, `configuration_atlas_spec`, `package_map`, and `srdf` options must resolve inside the copied `source/` tree. `observations` and `observation_query` must appear together and require `scene`. Both `action_assurances` and `ros_action_adapters` require `functional_spec`. A direct assurance record names the exact exported functional model, one raw bundle under `source/`, and a unique JSON output basename. A ROS action adapter record instead names raw config/capture/supplemental sources and distinct lifecycle-source, bundle, report, and assurance basenames under candidate work. For example, add this evaluator-only build metadata at suite and task level:

```json
{
  "runtime_requirements": {
    "xacro": {
      "executable": "xacro",
      "version": "2.1.1",
      "provision": "evaluator"
    }
  },
  "candidate_task": {
    "schema_version": "robot-spatial-raw-source-task.v1",
    "input_format": "xacro",
    "entrypoint": "source/robot.urdf.xacro",
    "expansion": {
      "executable": "xacro",
      "mappings": ["namespace:=held_"],
      "output": "expanded.urdf"
    },
    "export_options": {
      "pose": "source/pose.json",
      "render": false,
      "workspace_samples": 0
    }
  }
}
```

The executable is a portable command token, not a machine-local path. The evaluator tells the isolated candidate how that token is provisioned; the builder publishes only the requirement. Expansion output must be a `.urdf` basename and is created under that task's submission directory. Mappings are ordered `name:=value` strings. The task expansion executable must match the suite runtime requirement. Path-valued export options must resolve inside the copied `source/` directory. Boolean, numeric, string, and repeated mesh-kind options are type-checked before publication. Every source file, `task.json`, and evaluation artifact is digest-bound.

For a cross-representation articulation task, keep the normal URDF/Xacro entrypoint for the full context pipeline and declare an additional strict articulation bundle:

```json
{
  "articulation_sources": [
    {"source_id": "urdf", "format": "urdf", "path": "source/robot.urdf"},
    {"source_id": "sdf", "format": "sdf", "path": "source/robot.sdf"},
    {"source_id": "mjcf", "format": "mjcf", "path": "source/robot.xml"}
  ],
  "articulation_comparisons": [
    {
      "reference": "urdf",
      "candidate": "sdf",
      "correspondence": "source/sdf-to-urdf-correspondence.json"
    },
    {
      "reference": "urdf",
      "candidate": "mjcf",
      "correspondence": "source/mjcf-to-urdf-correspondence.json"
    }
  ]
}
```

At least one articulation source is required when the bundle is present. Comparisons are optional, but when supplied they must be non-empty and use at least two distinct declared sources. Source IDs are unique; formats are exactly `urdf`, `sdf`, or `mjcf`; every path resolves under `source/`; comparison pairs are unique. The correspondence is raw typed source intent, not a generated equivalence result. It must bind the exact deterministic grammar digests that the candidate will regenerate from the published bytes.

For a supplemental-mechanism task, declare one or more graphs over named articulation sources:

```json
{
  "export_options": {
    "constraint_spec": "source/mechanism-constraints.json",
    "workspace_samples": 0
  },
  "articulation_sources": [
    {"source_id": "tree", "format": "urdf", "path": "source/mechanism-tree.urdf"}
  ],
  "constraint_graphs": [
    {
      "graph_id": "full-mechanism",
      "articulation_source": "tree",
      "spec": "source/mechanism-constraints.json"
    }
  ]
}
```

Every graph ID is unique, its source must be declared, and its spec must stay under `source/`. The spec is raw asserted mechanism intent and binds the exact deterministic grammar digest that the candidate must regenerate. The candidate compiles and verifies each graph in task-local work, evaluates explicit poses, and uses the local solver only with an explicit seed and solved-variable list. Questions must keep tree-variable count separate from local mobility, asserted semantics separate from deterministic residuals, and local rank separate from global DOF.

For a finite configuration-space task, add a raw atlas spec and name the declared graph it binds:

```json
{
  "export_options": {
    "constraint_spec": "source/mechanism-constraints.json",
    "configuration_atlas_spec": "source/configuration-atlas-spec.json",
    "workspace_samples": 0
  },
  "configuration_atlases": [
    {
      "atlas_id": "finite-assembly-witnesses",
      "constraint_graph": "full-mechanism",
      "spec": "source/configuration-atlas-spec.json"
    }
  ]
}
```

Every atlas ID is unique, every named graph must exist, and the raw spec must stay under `source/`. At most one declared atlas targets one graph in version 1. The candidate regenerates the named graph first, generates the atlas in task-local work, verifies exact regeneration and all stored nodes, and answers from exact chart/node/component records. Questions must preserve parameter samples, seeds, scales, merge/edge thresholds, minimum coverage, and deficient samples. A stored node is a satisfying witness; a proximity component is not a branch certificate; a rank drop is relative to the maximum observed in the declared chart and is not a certified singularity.

To test raw ROS-to-observation understanding instead of giving the candidate a pre-normalized log, add this task block and omit `observations`/`observation_query` from `export_options`:

```json
{
  "ros_observation_adapter": {
    "config": "source/adapter-config.json",
    "capture": "source/ros-capture.json",
    "observation_query": "source/observation-query.json",
    "output_filename": "observations.json",
    "report_filename": "normalization-report.json"
  }
}
```

The three inputs must be digest-bound files under `source/`; the two outputs must be distinct JSON basenames. `export_options.scene` remains required. The candidate creates both outputs only under a new task-local submission/work directory, then adds the generated observation path and declared query path to `export` or `prepare`. This mode measures autonomous config/capture validation, partial-joint assembly, TF path reconstruction, and context generation without publishing evaluator-normalized observations. It still does not test live subscription unless the evaluation separately provisions and records ROS 2 runtime behavior.

To test raw ROS-action-to-assurance understanding instead of publishing a prebuilt lifecycle evidence source or action bundle, add this block:

```json
{
  "ros_action_adapters": [
    {
      "adapter_id": "grasp-run-42",
      "functional_model_source": "exported_functional_model",
      "config": "source/ros-action-config.json",
      "capture": "source/ros-action-capture.json",
      "supplemental_sources": [
        {
          "source": "source/condition-effect-evidence.json",
          "output": "condition-effect-evidence.json"
        }
      ],
      "evidence_source_output": "ros-action-evidence.json",
      "bundle_output": "generated-action-bundle.json",
      "report_output": "ros-action-normalization-report.json",
      "assurance_output": "ros-action-assurance.json"
    }
  ]
}
```

This block requires `export_options.functional_spec`. Config, capture, and every supplemental source must be raw files under `source/`. Each supplemental record declares a source path and a distinct JSON basename to copy byte-for-byte into candidate work, placing it under the generated bundle directory without mutating the public source tree. Lifecycle source, bundle, report, and assurance outputs are distinct JSON basenames and may not collide across adapters, with `action_assurances`, or with ROS observation outputs.

The candidate normalizes the immutable capture against the exact freshly exported functional model, compiles the generated bundle, verifies action assurance, and answers through its query contract. The candidate never runs `execute-capture`; an evaluation must not dispatch a robot action. Questions should independently test goal UUID/payload/config/capture binding, client receipt versus server acceptance time, status-code mapping, other-goal and unknown-status handling, duplicate/conflicting status behavior, publisher-identity visibility, feedback/result non-promotion, lifecycle consistency, condition/effect timing, and the absence of physical, causal, authorization, and safety proof. Offline normalization does not demonstrate live ROS discovery, QoS, service, or hardware integration.

For a ROS source workspace, add `"workflow": "prepare"` and one or more confined workspace roots such as `"workspace_roots": ["source/workspace"]`. Keep `input_format`, entrypoint, expansion runtime/mappings, and export options. In this workflow the Xacro expansion output must be `resolved.urdf`, and `package_map` is forbidden because `prepare` generates it; confined scene/log/query paths remain allowed and must be propagated to preparation. The candidate runs one `prepare` command into a new task-local preparation root and answers from its `context/`. The default workflow is `direct`, which preserves the expanded-URDF behavior above and does not accept workspace roots.

The builder rejects symlinks and generated context artifacts anywhere in the raw source tree, including `agent-context.json`, cards/indexes/facts, `model.json`, `scene.svg`, `articulation-grammar.json`, `concept-graph.json`, `concept-language.rsl`, `functional-model.json`, `action-assurance.json`, generated ROS action lifecycle sources/bundles/normalization reports, `articulation-comparison.json`, `constraint-graph.json`, `constraint-evaluation.json`, `constraint-solution.json`, and `configuration-atlas.json`. It also rejects generated `render-atlas/` and `motion-atlas/` directories. Raw ROS action config, capture, and separately typed condition/effect sources remain permitted inputs. This makes the public claim auditable: the candidate had raw source assertions, questions, and an empty template—not evaluator-generated spatial truth or expected outputs. A question may publish an exact answer-object field contract containing typed placeholders; placeholders reveal neither expected values nor array lengths and prevent interface-only grading failures caused by arbitrary field renaming.

A raw project function/affordance spec is permitted because it is project-owned source intent. A precompiled `functional-model.json` is forbidden because it contains evaluator-generated normalization, requirement results, proof clauses, indexes, and structural closure. The candidate must compile it after regenerating the exact bound concept graph. Functional questions should independently cover component grouping/purpose without name inference, capability declaration versus typed structural grounding versus physical truth, relational preconditions and intended effects, and complete/incomplete inventory negatives without physical-impossibility overclaim.

Raw action-evidence bundles and their referenced source records are permitted because they are source evidence, not derived assurance conclusions. Their functional-model binding must match the exact deterministic model regenerated from the public URDF and functional spec. A precompiled `action-assurance.json` is forbidden because it contains evaluator-derived evidence selection, readiness, lifecycle, effect, discrepancy, and coverage projections. The candidate compiles and verifies it under task-local work. Action-evidence questions independently cover condition evidence selection at decision time, goal/status/result lifecycle at evaluation time, effect timing, discrepancies, content provenance, and the explicit absence of dispatch, physical, causal, and safety proof.

Raw ROS action captures are transport evidence rather than normalized lifecycle conclusions. Their config and goal-payload digests must bind the exact deterministic functional model and action instance. The evaluator may provide separately typed condition/effect sources, but must not publish the normalized lifecycle source, generated action bundle, normalization report, or assurance model. The candidate copies supplemental sources byte-for-byte into task-local work, runs offline normalization, and keeps result/feedback payloads distinct from effect observations.

For `workflow: prepare`, the candidate passes the declared entrypoint, workspace roots, mappings, runtime, and export options to `prepare`; direct mode validates/expands then exports. Every export creates an articulation grammar, concept graph, and controlled concept language. When `functional_spec` is present, it also creates and verifies a functional model after the concept graph. When `ros_action_adapters` is present, the candidate copies declared supplemental sources into candidate work, normalizes each raw capture against that exact generated model, compiles and verifies the generated assurance, and never dispatches. When `action_assurances` is present, the candidate compiles each raw evidence bundle against that exact generated functional model, verifies exact regeneration/evidence-source digests, and uses `query-action-assurance`. When `articulation_sources` is present, the candidate compiles and verifies every source and runs declared comparisons. When `constraint_graphs` is present, it compiles and verifies each graph, evaluates named poses, and locally solves only declared variables/seeds. When `configuration_atlases` is present, it generates and verifies each digest-bound finite atlas after its graph. The candidate queries concept questions through `query-concepts`, cites supporting clause IDs/modalities, and uses exact negatives only inside the declared complete tree/articulation projections. It queries function questions through `query-functions`, preserving assertion, requirement, precondition, intended-effect, completeness, and physical-execution boundaries. It queries action evidence through `query-action-assurance`, preserving decision/evaluation times, goal UUID/payload binding, selected/ignored record reasons, server/client time and authority visibility, protocol-versus-physical boundaries, result/feedback-versus-effect boundaries, effect-versus-causation boundaries, discrepancies, and lack of dispatch authorization. The spanning tree, symbolic concept abstraction, project functional assertion, raw action transport, normalized lifecycle evidence, action assurance, supplemental mechanism, FK snapshot, local Jacobian, numerical constraint rank, finite configuration atlas, and finite motion atlas are distinct evidence layers. Observation normalization still occurs before final export/preparation. The candidate must not run `--generate-evaluation`.

## Candidate protocol

Give the candidate only:

- the public suite directory;
- the `understand-robot-spatial` skill;
- a separate empty submissions root.

Use a fresh task/context. Do not pass prior conclusions, expected values, evaluator prompts, private paths, report paths, or authoring traces. Require the candidate to stay inside the public root and submissions root. In `raw_sources` mode it first validates/expands source, then constructs fresh artifacts from `task.json`; an articulation bundle adds grammar generation/verification/comparison, and a constraint bundle adds graph compilation/verification/evaluation/local solving. No evaluator-generated context, grammar, graph, residual, solution, comparison report, or answer is present.

Filesystem separation is mandatory for a strong blind claim. A fresh agent that merely receives instructions not to inspect a shared filesystem is instruction-isolated, not security-isolated. Report that distinction.

## Verification

Run outside candidate visibility:

```bash
python3 scripts/spatial_evaluation_suite.py verify \
  candidate-readable/real-robots-v1/manifest.json \
  evaluator-only/real-robots-v1/manifest.json \
  --submissions-root candidate-submissions \
  --per-task-accuracy 1.0 \
  --minimum-task-pass-rate 1.0 \
  --minimum-overall-accuracy 1.0 \
  --report evaluator-only/report.json
```

Before grading, the verifier checks:

- exact public-manifest binding from the private manifest;
- SHA-256 of every public artifact and private key;
- exact task-ID equality between public and private manifests;
- question, answer-template, and key ID equality;
- unique IDs, schemas, question counts, capability counts, and spatial-truth binding;
- missing, malformed, duplicate, and unexpected candidate answers.

The report may contain expected answers in failure details. Keep it private. A task passes only at its configured accuracy and with no malformed, duplicate, or unexpected record. The suite then applies both task-pass-rate and total-question accuracy gates.

Create a publishable answer-free result only from the evaluator side:

```bash
python3 scripts/spatial_evaluation_suite.py summarize evaluator-only/report.json \
  --out public-evidence/result-summary.json
```

The summary keeps aggregate, family, task, and capability accuracy; submission digests; and failure-category counts. It removes expected answers, actual answers, question IDs, and detailed comparison reasons. Review and add the real isolation level separately before publishing it.

## Evidence language

State exact input mode, workflow, task families, source/import subsets, grammar/spec/graph/correspondence digests, poses/seeds/solved variables, residual tolerances, local-rank results, negative controls, comparison probes, scene/observation bindings, question coverage, thresholds, runtime, and isolation. A perfect raw constraint-bundle result adds evidence only that the candidate compiled the enumerated asserted mechanism graph, distinguished the spanning tree from the mechanism, executed the enumerated residuals, reported local mobility correctly, and performed the declared local solve. It does not prove global DOF/configuration space, uniqueness, arbitrary mechanism generalization, physical assembly, calibration, compliance, contact, dynamics, control, hardware, or safety. Never tune questions or expose keys merely to restore a passing score.
