# ManiSkill PickCube live fixtures

These are small normalized numeric fixtures captured from ManiSkill `PickCube-v1` seed 2 with
ManiSkill 3.0.1, SAPIEN 3.0.3, `physx_cpu`, `pd_joint_pos`, and a fixed 100-step horizon. They do not
contain upstream meshes, textures, URDFs, videos, observations, rewards, `info`, success fields, or
official evaluator output.

Exact report-digest regeneration is scoped to the recorded evaluator dependency versions. Across
supported Python/NumPy versions, regression tests require identical run/task bindings, predicate
statuses, and verdicts; tiny backend-level floating-point differences remain visible in regenerated
measurements instead of being hidden by rounding.

| Fixture | Robot Spatial verdict | Run manifest | Report |
| --- | --- | --- | --- |
| `success-seed-002` | `supported` | `7aabe8e249ff9aa7d29452a5199c731868899a854e1a31df76775d0ee839c7fb` | `52ffea49c8a53aba7faa91def01ca10f87420882be1acfa5d733fea2c1b34382` |
| `failure-seed-002-no-op` | `refuted` | `d776d55c62b4d92a57c2f17041bee152523dc2ae733b27a28a7c125353c9fd21` | `6e16b43c32603989817ebda24fe1960ac6fdc5e053081b0c206dad226cb8c93a` |

The official reference is intentionally absent from each candidate directory. The external
100-case benchmark record binds these run/report digests to independently replayed official
references.
