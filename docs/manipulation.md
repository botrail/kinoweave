# Single-arm manipulation

For coordinated tasks and atomic two-arm skills, see
[multi-arm manipulation](multi-arm-manipulation.md).

KinoWeave can connect configured skills through collision-checked free motion
and generate a single-arm Pick & Place plan. Movable objects have explicit
sphere geometry and are either placed in the world or attached to named robot
links. The CPU reference and optional CUDA backend evaluate mode-specific PRM
masks. Every returned solution undergoes independent CPU verification.

The constructor requires one scene robot and a fixed, ordered skeleton of
single-arm skills. Skills supply their own joint traces, grasp transforms,
effects, and contact windows. This constructor does not perform automatic
grasp generation, skeleton generation, or cost optimization.
The [data contract guide](manipulation-contracts.md) describes these inputs.

## Run the demo

From an installed source checkout:

```bash
python examples/single_arm_pick_place.py --output /tmp/pick-place
```

The example uses a synthetic Cartesian gripper with three prismatic joints.
It approaches a part on one table, picks it up, carries it above a central
obstacle, places it on another table, and returns to its initial position.
The robot and part use explicit collision spheres. This is a kinematic proxy
demonstration, without physical grasp or actuator modeling.

For standalone browser playback:

```bash
uv run --project environments/viz --frozen python examples/single_arm_pick_place.py --output /tmp/pick-place --view
```

Open the printed loopback URL (default port 8090). Play, Loop, Reset, Speed,
and the time slider control the display. The inspector and object label show
whether the part is placed or held. No simulator is required.

To generate a plan on CUDA, use the [qualified GPU environment](curobo.md):

```bash
uv run --project environments/cuda12 --frozen python examples/single_arm_pick_place.py --backend curobo --output /tmp/pick-place-gpu
```

View its saved files using the Python example below in the visualization
environment. CUDA planning does not require Viser, and viewing does not
require CUDA. The motion-only `kinoweave view` command accepts native
motion-only documents; use `ManipulationViewer` for manipulation documents.

## Construct a plan

```python
from kinoweave.io import dump, load
from kinoweave.manipulation import ManipulationProblem, SingleArmPlanner
from kinoweave.roadmap import PRMConfig

problem = load("problem.json", expected=ManipulationProblem)
planner = SingleArmPlanner(problem, prm=PRMConfig(num_samples=128, neighbors=12))
result = planner.plan()  # optional samples=(q1, q2, ...) adds explicit anchors
dump(result, "result.json")
print(result.status, planner.statistics)
```

One static PRM includes native start/goal states, every skill entry/exit, and
seeded samples. Each mode adds object-dependent node and edge masks to this
same graph. Edges use the native shared tick and stopped quintic law in both
directions. A held object moves with its named link; an obstacle can invalidate
a carried-object edge even when the robot alone clears it.

The constructor tries supplied skill options in input order, connects their
boundaries, checks the skill trace, and returns the first independently
verified complete plan. Fixed timing is preserved. A stretch skill uses the
larger of its source duration and minimum duration; no duration optimization
is performed. Every free-motion/skill interval is explicit. The robot's native
`goal` and the complete object `goal_mode` must both be reached.

Free-motion anchors must be valid against native self/static geometry.
In-skill contact permissions do not exempt PRM nodes or free-motion edges.
The constructor does not invent alternative skill motions when a supplied
trace collides. `no_solution_in_discretization` is limited to the supplied
graph/options. `max_states` and the scene time limit bound construction;
the time limit is checked cooperatively between batches, without preempting
backend initialization, an in-flight native call, or final verification.

## Geometry and mode masks

```python
from kinoweave.manipulation import ModeEvaluator, ModeGeometry

geometry = ModeGeometry(problem)
frame = geometry.snapshot(problem.initial_mode, {"gantry": problem.scene.robots[0].start})
assert frame.valid

with ModeEvaluator(problem, batch_size=64, max_cache_bytes=32 * 1024 * 1024) as evaluator:
    mask = evaluator.mask(planner.roadmap.graph, problem.initial_mode)
    print(mask.node_flags, mask.edge_flags, evaluator.cache_stats)
```

`ModeGeometry` supplies independent CPU FK and complete-scene collision queries.
Placed poses use `T_world_object`; attachments use
`T_world_link @ T_link_object`. Both holders of a dual grasp must produce
matching world poses within `GeometryPolicy` position/orientation tolerances.

Local masks include placed objects and objects held exclusively by the
selected robot. Other robots and their held objects are omitted for later
joint conflict checking. Dual-held objects require joint configurations and
are rejected by local masks. Complete-scene verification includes all bodies.

CUDA masks cache owned robot FK/self/static rows independently of the object
mode. `batch_size`, `sphere_tile`, `max_pair_bytes`, `max_cache_bytes`, and
`max_masks` bound batches, estimated sphere-pair scratch space, retained tensor
payloads, and mask count. These are not caps on total CUDA allocator memory.
`max_cache_bytes=0` disables tensor retention. Floating-point precision changes
invalidate cached geometry/masks. Requests for unsupported CUDA models or an
unavailable CUDA stack fail explicitly; no CPU planning fallback is used.

## Verification

```python
from kinoweave.manipulation import validate_manipulation, verify_manipulation

report = validate_manipulation(problem, result)
checked = verify_manipulation(problem, result)
assert checked.status == "success"
```

Verification reconstructs source traces and scheduled clocks rather than
trusting imported validation reports. It checks binding, precedence, explicit
robot occupancy, endpoints, native joint limits, mode effects, object goals,
collision geometry, and contact permissions. Quintic rate extrema are checked
analytically, including mimic joints. Linear velocity jumps and nonzero
start/end velocities are rejected under finite acceleration limits.

Collision sampling includes trace knots, effects, contact boundaries, and
subdivisions no wider than `validation_time_step`. Both sides of internal
boundaries and intermediate effects sharing a timestamp are checked. Each
transition must preserve the object's world pose; changing attachments cannot
teleport it. `max_samples` bounds distinct sampled timestamps, including
boundaries. This is sampled collision verification, without a continuous-time
swept-volume or dynamics guarantee.

An attachment permits contact only between its object and its declared
robot/link. Explicit windows permit only their named body pair over the
scheduled interval. Internal start/end boundaries use one-sided permissions;
plan endpoints use their in-domain side. Authorized contacts still reject
clearance below `-contact_tolerance`; they cannot waive deep penetration.
Other collisions use the scene's positive margin and collision tolerance.

A declared support must be a horizontal box top face, optionally yawed.
All object-sphere bottoms must stay above the support plane; at least one
touching sphere's projected disk must fit inside the top face, within
`support_tolerance`. This conservative geometric predicate does not prove
force balance or stability. Omit `support` for a world-fixed placement that
does not claim a verified support surface. Sphere/tilted-box supports are
rejected. Tolerances default to `1e-6` in metres/radians through `GeometryPolicy`.

## View a saved manipulation plan

Run this with the optional `viz` extra:

```python
from kinoweave.io import load
from kinoweave.manipulation import ManipulationProblem, ManipulationResult
from kinoweave.visualization import ManipulationViewer

problem = load("/tmp/pick-place-gpu/problem.json", expected=ManipulationProblem)
result = load("/tmp/pick-place-gpu/result.json", expected=ManipulationResult)
with ManipulationViewer(problem, result, port=8090) as viewer:
    print(viewer.url)
    viewer.run()
```

The viewer independently revalidates the plan on CPU before opening its
server, including imported `success` results. `viewer.seek(t)` and
`viewer.frame` expose the evaluated time, mode, and world geometry. It displays
collision proxies, link skeletons, obstacles, and object state. Supply
`visuals=load_visuals(manifest_path, problem.scene)` to add hash-checked robot
link meshes, as in the [eight-UR5 demo](eight-arm-demo.md).
Simulator and controller adapters remain separate optional integrations.
