# Installation and supported scope

KinoWeave is a development release. Install a source checkout or a built
wheel with Python 3.11 or newer:

```bash
python -m pip install .
# Alternatively, install a wheel obtained from the project:
python -m pip install /path/to/kinoweave-0.1.0.dev0-py3-none-any.whl
kinoweave --version
```

The core has no runtime dependencies. It can create and read problems, plan
using the CPU reference, independently validate saved GPU results, and record
kinematic playback without NumPy, Torch, cuRobo, or simulator SDKs. Installing
the core wheel does not install a GPU runtime. The optional Linux x86_64 /
Python 3.12 [CUDA environment](curobo.md) has a separate dependency lock.

`pip install '.[viz]'` adds the standalone browser viewer. Add `'.[urdf]'`
for URDF and scene-YAML import, or `'.[viz,urdf]'` for both. The source
checkout's locked Python 3.12 [visualization environment](../environments/viz/pyproject.toml)
provides both extras without Torch or CUDA. See the
[visualization walkthrough](visualization.md) and [scene format](scene-authoring.md).

The supported host platform is Linux x86_64. The core supports Python 3.11
through 3.14; the locked visualization and CUDA environments use Python 3.12.
Other platforms and Python versions are outside the supported configuration.

## Supported scope

| Component | Scope |
| --- | --- |
| Native model | Fixed-base trees with explicit joint names/order, base/link transforms, limits, and link-attached collision spheres. |
| CPU reference/player | Fixed, revolute, continuous, prismatic, and mimic joints. Continuous angles remain unwrapped. |
| CUDA backend | Fixed, bounded revolute/prismatic joints with signed coordinate axes; unsupported features are rejected. |
| Static environment | Sphere and oriented box obstacles. |
| Planning | Individual PRMs, shared-tick quintic-stop motions, waits, prioritized or LaCAM-style search, and small finite product A*. |
| Result exchange | Strict schema v1 JSON, input identity, provenance, diagnostic settings, and shared-time trajectories. |
| Manipulation | Placed/held geometry, CPU/CUDA state conflict tables, task progress, atomic bimanual skills, fixed-skeleton multi-arm search, and independent CPU verification. |
| Refinement | Explicit completion/distance objectives and bounded anytime single-/multi-arm shortcuts with independent acceptance checks. |
| Task scheduling | Finite grounded alternatives, symbolic/resource ordering, retry feedback, shared geometry and verified incumbents. |
| Playback | SDK-free kinematic snapshots, arbitrary seeks, read-back observations, and file reports. |
| Visualization | Optional Viser playback of native robot frames/meshes and separate manipulation playback with object states. |
| Scene authoring | Optional expanded URDF 1.0 subset, explicit planning YAML, and multi-instance scene YAML. |

The initial application scope is two fixed arms, including the supplied UR5
scene. The [eight-UR5 showcase](eight-arm-demo.md) adds an experimental fixed
inspection cell with real link meshes. General larger fleets, dense industrial cells, arbitrary mesh collision,
general automatic sphere fitting, MJCF/USD import, continuous task/grasp
synthesis, mobile bases, dynamics, and physical control are outside the supported
scope. URDF support is limited to the [documented subset](scene-authoring.md#supported-urdf-subset).
The memory player requires at least one independent joint per robot.
Represent static bodies as obstacles.

The [experimental scaling example](performance.md) exercises two, four and
eight Cartesian tool proxies. Its functional scope is separate from the
supported real-arm geometry and deployment scope above.

Collision results have level `sampled`. Whole-interval analytic checks cover
position/velocity/acceleration limits for the declared time law; collision
sampling does not establish continuous-time safety. Model geometry and limits
are caller-supplied. Converting an existing simulator model requires an
adapter that preserves and checks the required model semantics.

Botrail, MuJoCo, Isaac Sim, and other runtimes are optional integration targets.
The scene, playback, and physical execution interfaces are separate; shipping
these interfaces does not imply a native SDK connection is implemented.

## Runtime errors

An unavailable or unsupported backend returns `unsupported`. Invalid scenes,
resource limits, finite graph exhaustion, and independent verification failure
have distinct [planning statuses](planning.md). GPU environments must expose
a compatible NVIDIA device. The reference backend is selected explicitly;
there is no automatic GPU-to-CPU planning fallback.
