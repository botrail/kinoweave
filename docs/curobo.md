# CUDA geometry backend

`kinoweave.backends.curobo.CuroboBackend` evaluates one robot against itself
and static obstacles. It provides batched forward kinematics (FK), inverse
kinematics (IK), world-frame collision spheres, and discrete collision masks.
It runs independently of Botrail, MuJoCo, and Isaac Sim.

## Install

The optional environment in this checkout targets Linux x86_64 and Python 3.12:

```bash
uv sync --project environments/cuda12 --frozen
uv run --project environments/cuda12 --frozen python examples/gpu_geometry.py
```

The lock pins cuRobo V2 commit
`78fd485fa82d9b9a063fb4985e371814587e666a`, PyTorch `2.9.0+cu128`,
Warp `1.11.0`, and CUDA Core `0.7.0`. Use an NVIDIA GPU and a driver compatible
with this locked stack. Installing arbitrary newer dependencies is outside
the supported configuration. The adapter accepts
cuRobo API version `0.8.0.post1.dev43`; use the lock to select its source revision.

An accessible NVIDIA CUDA device is required. The backend raises
`BackendUnavailableError` if its dependencies or CUDA access are missing.
The core package still installs without PyTorch, cuRobo, or a simulator SDK.

## Evaluate a robot

```python
from dataclasses import replace

from kinoweave import PlanningProblem, Pose
from kinoweave.backends.curobo import CuroboBackend
from kinoweave.io import load

problem = load("examples/data/two_sliders.problem.json", expected=PlanningProblem)
problem = replace(problem, config=replace(problem.config, backend="curobo", device="cuda:0"))

with CuroboBackend(problem, "left") as backend:
    fk = backend.forward([[0.0], [0.3]])
    masks = backend.collision([[0.0], [0.3]])
    print(fk.spheres)  # CUDA tensor, world xyzr, original sphere order
    print(masks.valid)  # limits AND no self/static collision
    ik = backend.inverse("carriage", (Pose(position=(0.0, -0.2, 0.5)),))
    print(ik.positions[ik.success])
```

Inputs accept `[D]`, `[B,D]`, or `[B,H,D]`, where D is the number of
independent joints, B is batch size, and H is the number of configurations
per batch. H has no timing or interpolation meaning. Inputs use
`RobotModel.joint_order`, or an explicitly supplied `joint_names` permutation.
Boolean, complex, non-finite, empty, and incorrectly shaped batches are rejected.

All output tensors stay on the selected CUDA device and use float32 geometry.
Converting outputs with `.cpu()`, `.tolist()`, or `.item()` synchronizes with
the host. Outputs own their data and remain valid after another query.

| Operation | Output |
| --- | --- |
| `forward(q)` | `FKBatch`: world positions `[B,H,L,3]`, xyzw quaternions `[B,H,L,4]`, spheres `[B,H,S,4]`, and link names |
| `collision(q)` | `CollisionBatch`: self/static collision and position-limit masks `[B,H]`; `valid` combines them |
| `inverse(link, goals)` | `IKBatch`: joint positions `[B,D]` in model order, success `[B]`, position error in m, orientation error in rad |

FK exports exactly the input links and spheres. Robot bases and joint origins
follow the [core coordinate contract](contracts.md). Upstream wxyz quaternions
are converted at the boundary. IK goals are world-frame `Pose` records. Returned
IK candidates are checked again for joint limits, self/static collisions,
and FK pose error; use only rows whose `success` mask is true. A failed mask
does not prove that no IK solution exists.

CUDA Graph is enabled for IK by default. Set `use_cuda_graph=False` on
construction to disable it. A backend owns reusable native buffers and
cached IK solvers; use one instance per concurrent worker and close it with
a context manager. `backend.info` and `backend.dependency_versions` identify
the stack actually used.

## Supported geometry and limits

The backend supports fixed-base trees containing fixed, bounded revolute, and bounded
prismatic joints, with signed X/Y/Z axes and explicit velocity/acceleration
limits. Continuous joints, mimic joints, arbitrary moving axes, zero-width
moving bounds, and models with no independent joints raise
`UnsupportedModelError`. Values outside the float32 representable range,
including nonzero geometry that underflows to zero, are rejected.

Collision geometry is the supplied link spheres and static oriented boxes
or spheres. Same-link spheres never form self pairs. Only explicitly supplied
allowed link pairs are excluded from self checking; adjacent links are not
automatically ignored. Collision masks include contact at clearance
`<= collision_margin + collision_tolerance`, subject to float32 arithmetic.

cuRobo evaluates box and self-collision activation costs. These costs are not
signed clearances. Analytic CUDA guards include contact at the cost boundary;
static sphere obstacles use analytic sphere distances directly. Sphere
obstacles are screened after native IK, so the optimizer may return an
unsuccessful candidate even if another valid solution exists.

The pinned loader cannot construct an empty self-pair table. For a model with
one collision link, the adapter creates an internal fixed helper with an
identical sphere. Its occupied union is unchanged, it is absent from outputs,
and synthetic pairs are never queried. When all original pairs are excluded,
self checking is disabled in both the query and IK paths.

The generated URDF is kinematic only: dynamics loading is disabled and no
torque, inertia, or tracking claim is made. The backend does not check other
robot instances, motion between configurations, velocities, accelerations,
or an entire trajectory. `CollisionBatch.valid` is not a plan validation
report. [GPU interference tables](roadmaps.md) check pairs of robot motions;
[Planner](planning.md) connects those tables to coordinated search. The [CPU reference](reference.md) can check
multi-robot intervals and trajectories without GPU dependencies.
