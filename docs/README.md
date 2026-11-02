# KinoWeave documentation

KinoWeave plans coordinated motions for fixed robot arms and saves them on a
shared timeline. Documentation is maintained in English.

## Start here

| Goal | Guide |
| --- | --- |
| Install the core or optional features | [Installation and supported scope](installation.md) |
| Plan, validate, and replay saved files | [Command-line workflow](cli.md) |
| See a plan in a browser | [Standalone visualization](visualization.md) |
| Run two UR5 models with meshes | [Two-UR5 walkthrough](visualization.md#two-ur5-arms) |
| Define robots and a workspace | [URDF robots and scene YAML](scene-authoring.md) |
| Understand capabilities and the relationship to cuArms | [Frequently asked questions](faq.md) |

## Python API and file formats

| Topic | Guide |
| --- | --- |
| Models, units, poses, results, and JSON | [Data contracts](contracts.md) |
| Planner, search choices, budgets, and result statuses | [Coordinated planning](planning.md) |
| PRMs, geometry caching, and batched interference | [Roadmaps and GPU interference tables](roadmaps.md) |
| CUDA FK, IK, and single-robot geometry | [CUDA geometry backend](curobo.md) |
| Time-law evaluation and CPU validation | [CPU reference](reference.md) |
| Kinematic playback and execution adapter boundaries | [Playback and adapter contracts](playback.md) |
| Runnable source examples | [Examples](../examples/README.md) |
| Licenses and robot asset attribution | [Third-party assets](third-party-assets.md) |

The core requires Python 3.11 or newer and has no runtime dependencies.
The optional visualization and CUDA environments use Python 3.12. Collision
validation is sampled; playback evaluates kinematics without simulating
contact or controller tracking.
