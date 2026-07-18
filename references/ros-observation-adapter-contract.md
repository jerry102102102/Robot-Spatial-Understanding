# ROS 2 observation adapter contract

## Contents

1. [Purpose and layer boundary](#purpose-and-layer-boundary)
2. [Workflow](#workflow)
3. [Adapter config](#adapter-config)
4. [Immutable capture](#immutable-capture)
5. [JointState assembly](#jointstate-assembly)
6. [TF reconstruction](#tf-reconstruction)
7. [Authority and clock rules](#authority-and-clock-rules)
8. [Output and provenance](#output-and-provenance)
9. [Live capture and rosbag replay](#live-capture-and-rosbag-replay)
10. [Failure behavior](#failure-behavior)
11. [Version 1 boundary](#version-1-boundary)

## Purpose and layer boundary

`ros_observation_adapter.py` converts ROS 2 transport records into the canonical temporal layer. It never asks a language model to interpret raw `sensor_msgs/msg/JointState` or `tf2_msgs/msg/TFMessage` data.

Keep five artifacts separate:

1. URDF/Xacro is the declared mechanism.
2. `robot-spatial-world-scene.v1` is the declared static workcell.
3. `robot-spatial-ros-adapter-config.v1` states topic, frame, joint-name, clock, age, and authority policy.
4. `robot-spatial-ros-capture.v1` preserves messages and transport metadata without interpreting them as physical truth.
5. `robot-spatial-observation-log.v2` contains deterministic normalized samples consumed by spatial queries.

The config and capture byte digests are embedded in the v2 log. Changing either input creates a different observation artifact. A topic, publisher GID, header stamp, TF frame, or `current` result remains a source report; it does not prove sensor correctness, calibration, synchronization, completeness, or safety.

## Workflow

Probe the current environment:

```bash
python3 scripts/ros_observation_adapter.py probe
```

Create an exact model/scene-bound config:

```bash
python3 scripts/ros_observation_adapter.py make-config resolved.urdf \
  --scene world-scene.json \
  --adapter-id cell_a \
  --clock-domain ros_time \
  --ros-reference-frame world \
  --scene-parent-frame world \
  --robot-root-frame base_link \
  --joint-map shoulder_pan_joint=arm_shoulder_pan_joint \
  --object-frame pallet=pallet_tf \
  --maximum-joint-component-age-ns 50000000 \
  --maximum-tf-edge-age-ns 50000000 \
  --out adapter-config.json
```

Capture ROS messages in a sourced ROS 2 environment:

```bash
python3 scripts/ros_observation_adapter.py capture \
  --config adapter-config.json \
  --capture-id run_42 \
  --duration-sec 10 \
  --source-reference "commissioning run 42" \
  --out ros-capture.json
```

Normalize without requiring ROS:

```bash
python3 scripts/ros_observation_adapter.py normalize resolved.urdf \
  --scene world-scene.json \
  --config adapter-config.json \
  --capture ros-capture.json \
  --out observations.json \
  --report normalization-report.json
```

Then use `observations.json` with the normal `observe-*`, `export`, `prepare`, and invariant commands. Never use `normalization-report.json` as a substitute for the v2 log; the report explains reconstruction, while the log is the canonical query input.

## Adapter config

The config binds the exact robot semantic digest and scene digest. Its main fields are:

- `clock`: one asserted domain, nanosecond unit, and optional epoch;
- `topics`: disjoint JointState, dynamic TF, and static TF topic lists;
- `frames.ros_reference_frame`: ROS frame treated as coincident with `frames.scene_parent_frame`;
- `frames.robot_root_frame`: ROS TF target corresponding to the URDF root instance;
- `frames.objects`: scene-object ID to ROS TF target mapping;
- `joint_mapping`: exact independent URDF driver joint to ROS joint-name mapping;
- `policies.timestamp_source`: `message_header` or explicit `message_header_or_receipt` fallback;
- maximum joint-component and dynamic-TF-edge ages;
- mandatory conflict and parent-switch rejection.

`make-config` covers every independent movable URDF driver. Mimic followers are derived from their driver and need no ROS mapping. Unmapped ROS joint names are retained in the audit report as ignored; they never become model joints by name similarity.

The v1 adapter requires these policies to remain true:

- `reject_multiple_publishers_per_joint`;
- `reject_multiple_publishers_per_child`;
- `reject_parent_switches`.

Do not weaken them to make a capture pass. Split the capture, fix the ROS graph, or define a future schema with explicit arbitration semantics.

## Immutable capture

The live command records:

- one capture interval in the config clock domain;
- node `use_sim_time` when visible;
- topic, receipt time, and `MessageInfo.publisher_gid` when supplied by the RMW implementation;
- JointState header time, names, and positions;
- each TF parent, child, header time, pose, and static/dynamic class;
- ROS distribution and a human source reference when provided.

Every record and transform has a unique ID. Receipt times must fall inside the declared interval. A nonzero header timestamp later than receipt time is rejected because the config asserts one clock domain. The adapter does not infer clock offsets.

`capture` writes nothing when no subscribed message is received. It does not normalize in the callback, so an invalid or ambiguous raw capture remains inspectable and can fail deterministically later.

## JointState assembly

ROS JointState messages may contain only part of the robot. The normalizer maintains the latest value of each explicitly mapped independent driver and emits a complete snapshot only when:

- every driver has a value at or before the event time;
- every component age is at most `maximum_component_age_ns`;
- the assembled pose satisfies URDF joint limits and mimic semantics;
- one joint is not attributed to multiple visible authorities;
- two records do not report conflicting values for the same joint and source timestamp.

Exact duplicate values are counted. Conflicting same-time values are rejected. A partial or stale component event is reported but does not create a falsely fresh joint snapshot. The snapshot source references the exact capture and config digests plus its component record IDs in the normalization report.

## TF reconstruction

TF is reconstructed independently of `tf2_ros` so the same capture produces the same result offline. The algorithm:

1. validates unit quaternions and canonical frame IDs;
2. builds one child-to-parent topology;
3. rejects cycles, parent changes, static/dynamic reuse of one child, conflicting duplicate transforms, and multiple visible authorities per child;
4. finds the structural path from `ros_reference_frame` to each configured root/object target;
5. considers only timestamps of dynamic edges on that target path;
6. selects the latest sample for every required edge at or before each event time;
7. rejects a composite event when any required dynamic edge is missing or older than `maximum_dynamic_edge_age_ns`;
8. composes the directed transform with zero-order hold and emits one pose relative to the mapped scene frame.

Unrelated TF traffic cannot refresh a target pose. Static-only paths are timestamped when the complete relevant static path has been received. Static transforms are treated as timeless within reconstruction, but the resulting observation sample still ages normally under the later observation query.

The reference-to-scene mapping is an explicit identity assertion. If ROS `map` is not physically coincident with the declared scene frame, add the required transform to TF or change the scene/config; never hide a calibration transform in prose.

## Authority and clock rules

Publisher identity is evidence only when the capture transport exposes it. Live rclpy capture requests `MessageInfo.publisher_gid`. Some rosbag replay paths expose only a topic-level surrogate; the report and v2 provenance mark that visibility limitation.

A unique authority does not prove that the publisher is correct. Rejection prevents silent arbitration, not malicious or misconfigured data.

`message_header` rejects zero or missing header timestamps. `message_header_or_receipt` is an explicit degraded mode and records every receipt fallback. Receipt fallback changes what the timestamp means; state it in every downstream time claim.

Matching clock-domain strings establish only schema consistency. They do not measure NTP/PTP synchronization, ROS `/clock` agreement, publisher latency, transport delay, or clock jumps. `normalization.clock_policy.synchronization_verified` therefore remains false.

## Output and provenance

The output uses `robot-spatial-observation-log.v2`. It preserves all v1 streams and adds `robot-spatial-ros-normalization-provenance.v1`:

- adapter and capture IDs;
- config and capture SHA-256;
- joint assembly and TF reconstruction method;
- timestamp-source policy and explicit unverified synchronization;
- authority-conflict policy and publisher-identity visibility;
- TF maximum edge age and false interpolation/extrapolation/future-consumption flags.

`export` turns this into both `observation_log/<log-id>` and `ros_capture/<capture-id>` entity cards plus a digest-bound provenance fact. Load the ROS capture card before interpreting assembled joint or TF poses.

`normalization-report.json` additionally contains per-component ages, component record IDs, TF path edges, selected edge timestamps, skipped stale/missing events, ignored ROS joint names, authorities, and output counts. It may be large and is an audit artifact rather than the primary AI retrieval surface.

## Live capture and rosbag replay

Live capture requires a sourced ROS 2 Python environment containing `rclpy`, `sensor_msgs`, and `tf2_msgs`. `probe` reports this honestly. The deterministic normalizer remains available without those packages.

To evaluate a rosbag without adding a bag-decoder dependency, capture the replayed ROS graph:

```bash
# terminal A, sourced ROS environment
python3 scripts/ros_observation_adapter.py capture --config adapter-config.json \
  --capture-id bag_replay_01 --duration-sec 30 --out bag-replay-capture.json

# terminal B, same ROS_DOMAIN_ID and clock policy
ros2 bag play my_bag --clock
```

This validates the same subscription surface used live. It does not preserve original publisher GIDs when the replay process replaces them; authority visibility must say so. Direct rosbag2/MCAP decoding is not implemented in v1.

## Failure behavior

`make-config`, `capture`, and `normalize` require new output paths. A validation error exits nonzero and does not overwrite an existing artifact. Normalize prevalidates both output names; a successful observation log is accompanied by a separate audit report.

Typical failures and required response:

| Failure | Meaning | Correct response |
| --- | --- | --- |
| config/capture digest mismatch | capture was made under different bytes | restore exact config or recapture |
| model/scene binding mismatch | semantic source changed | regenerate config and capture evidence |
| clock mismatch or future header | time basis is inconsistent | repair time configuration; do not shift timestamps silently |
| multiple joint/TF authorities | arbitration is ambiguous | remove duplicate publisher or create explicit future arbitration contract |
| TF parent switch/cycle | topology is not one stable tree | split runs or repair TF graph |
| stale/missing TF edge | target pose is not current enough | retain missing/stale result; do not refresh from unrelated TF |
| incomplete/stale joint components | no complete model state exists | wait for all drivers or change the explicitly reviewed age policy |
| joint-limit violation | report conflicts with bound URDF semantics | inspect model variant, units, name mapping, and source |

## Version 1 boundary

The adapter supports ROS 2 `JointState`, `/tf`, and `/tf_static` capture plus deterministic offline normalization for one tree-structured robot and declared scene objects. It does not implement direct rosbag2/MCAP decoding, DDS security validation, original rosbag publisher identity recovery, TF interpolation, velocity/effort state, covariance propagation, clock synchronization measurement, latency compensation, changing TF topology, source arbitration, dynamic object creation, multi-robot TF ownership, sensor calibration, data association, occupancy maps, continuous collision, contact state, controller feedback semantics, or safety decisions.

Live capture code is interface-implemented but must be verified in an actual sourced ROS 2 environment before claiming ROS integration. Normalizer tests do not prove RMW QoS compatibility, message delivery, replay timing, or hardware state correctness.
