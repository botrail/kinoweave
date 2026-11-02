# Roadmaps and GPU interference tables

KinoWeave builds individual joint-space PRMs and checks synchronous candidate
motions on CUDA. It provides a minimum-tick single-robot path and a compact
boolean compatibility table for two robots. [Planner](planning.md) combines
these building blocks into independently verified multi-robot trajectories.

The [locked CUDA environment](curobo.md) is required for GPU evaluation.
Botrail, MuJoCo, Isaac Sim, and other runtimes remain optional integrations.

```bash
uv sync --project environments/cuda12 --frozen
uv run --project environments/cuda12 --frozen python examples/gpu_roadmap.py
```

The example samples individual roadmaps around a static sphere, validates
their paths with the CPU reference, and creates a table of candidate motion
pairs. Its JSON output contains geometry results and actual backend metadata,
not a successful coordinated `PlanResult`.

## Build a roadmap

```python
from dataclasses import replace

from kinoweave import PlanningProblem
from kinoweave.collision import GpuCollisionEvaluator
from kinoweave.io import load
from kinoweave.roadmap import PRMConfig, build_prm
from kinoweave.validation import validate_trajectory

problem = load("examples/data/retreat.problem.json", expected=PlanningProblem)
problem = replace(problem, config=replace(problem.config, backend="curobo", device="cuda:0"))

with GpuCollisionEvaluator(problem) as evaluator:
    roadmap = build_prm(evaluator, "alpha", config=PRMConfig(num_samples=64, neighbors=8))
    path = roadmap.shortest_path()  # vertex indices, or None
    trajectory = roadmap.trajectory()  # one track, or None
    if trajectory is not None:
        single = replace(problem, robots=(problem.robots[0],))
        report = validate_trajectory(single, trajectory)
```

Sampling uses the problem's seed and uniform declared independent-joint
position ranges. The metric divides each coordinate by its range. The graph
connects nearest neighbors, connects start and goal to their neighbors, and
also tests a direct start-to-goal connection. Only self/static-clear samples
are retained. Other robots are handled later through motion compatibility,
and are not frozen as obstacles during individual roadmap construction.

Every graph edge takes exactly `PlannerConfig.time_step`. A straight joint
connection is split into enough stop-to-stop intervals to satisfy the
quintic velocity and acceleration bounds. Every resulting interval is
checked in **both directions**, including endpoints and interior samples.
Intermediate vertices are inserted only after the entire connection passes.
Connections are checked in batches controlled by `connection_batch_size`.

`PRMConfig.max_segment_steps` limits subdivisions per connection; a connection
requiring more is omitted. `max_vertices` bounds the graph; exceeding it raises
an error. `num_samples` counts random attempts, not guaranteed retained nodes.
Optional `samples=((...), ...)` replaces random sampling with explicit
configurations, useful for repeatable scenes. Duplicate samples are merged.

Sampling requires bounded independent joints with nonzero finite ranges.
GPU evaluation has the [cuRobo model restrictions](curobo.md). For explicit
CPU experiments, use `ReferenceMotionEvaluator` with
`backend="reference", device="cpu"`; there is no automatic CPU fallback.

`Roadmap.graph` is an undirected `JointGraph`. `shortest_path()` minimizes
the number of equal-duration edges. If start equals goal, `trajectory()`
returns one hold interval. No path means only that this finite graph is
disconnected. A returned one-robot trajectory should be checked against its
intended scene before use; it does not cover the other robots.

## Check motion pairs

```python
from kinoweave.collision import Motion

first = (
    Motion(start=(-1, 0), end=(-1, 0)),  # wait
    Motion(start=(-1, 0), end=(0, 0)),
)
second = (
    Motion(start=(1, 0), end=(1, 0)),
    Motion(start=(1, 0), end=(0, 0)),
)
with GpuCollisionEvaluator(problem) as evaluator:
    valid = evaluator.motion_validity("alpha", first)
    table = evaluator.interference("alpha", first, "beta", second)
    print(table.compatible)
```

Each `Motion` uses its model's explicit joint order and the common `time_step`.
The duration and quintic law are identical on both sides. Waits and goal holds
are ordinary motions with equal start/end configurations. Sampling includes
both endpoints and uses spacing no larger than `validation_time_step`.

`first_valid` and `second_valid` include analytic position/velocity/acceleration
checks and sampled self/static collision checks. A `compatible[i][j]` value
is true only when both motions are individually valid and their spheres do
not collide at any shared sample time. It checks exactly the two named robots;
additional robot instances are not included in that table.

cuRobo computes robot geometry and self/static masks. PyTorch evaluates joint
interpolation on CUDA in float64, then geometry in float32, and reduces sphere
distances on CUDA. Only reduced booleans return to the host. Contact at
`clearance <= collision_margin + collision_tolerance` is included. Position
limits use the core's `joint_tolerance`, and rate limits use the same analytic
checks as the CPU reference. Numerical failures raise errors.

`reference_interference(problem, first_robot, first, second_robot, second)`
provides an independent scalar CPU oracle. CPU/GPU values near a boundary
remain subject to their different floating-point precision. Both use sampled
collision checks; neither table proves continuous-time safety.

## Bound work and reuse geometry

```python
from kinoweave.collision import GeometryCache, GpuTiling

cache = GeometryCache(max_bytes=16 * 1024 * 1024)
tiling = GpuTiling(
    motion_batch_size=16,
    time_batch_size=8,
    pair_batch_size=64,
    sphere_batch_size=16,
    max_pair_bytes=1024 * 1024,
)
with GpuCollisionEvaluator(problem, tiling=tiling, cache=cache) as evaluator:
    table = evaluator.interference("alpha", first, "beta", second)
    print(cache.stats)
```

Candidate pairs, time samples, and both sphere axes are tiled. The full
candidate/time/sphere-pair distance array is never materialized. Pair tiles
shrink automatically to the `max_pair_bytes` workspace estimate.
`max_geometry_bytes` limits the conservative payload estimate per geometry
batch; if one time tile cannot fit, reduce `time_batch_size` or increase that
budget. `max_table_pairs` limits the number of booleans in the host table.
Large tables should be queried as smaller candidate sets.

These limits bound this evaluator's tensor-payload estimates. Native cuRobo
scratch, the CUDA context, and PyTorch allocator reservations are additional;
the limits are not a cap on total device VRAM. The evaluator exposes
`max_pair_tile_bytes` and `max_geometry_tile_bytes` for the largest estimates
actually used.

The cache is an LRU of owned CUDA sphere tensors and self/static masks, with
an exact tensor-payload byte cap. `max_bytes=0` disables storage. Statistics
count hits, misses, evictions, resident payload bytes, and entries. `clear()`
releases entries and preserves cumulative counters.

Cache keys include the whole problem digest (models, revisions, base poses,
geometry, environment, limits, seed, timing and margins), supported backend
versions/source revision, device, precision and Torch matmul mode, interpolation, motion endpoints,
and the sample slice. A changed immutable problem needs a new evaluator;
sharing the same cache is safe because changed inputs use different keys.
Changes to tiling do not change the mathematical query.

An evaluator is not thread-safe. Use one per concurrent worker and close it
with a context manager. It clears an internally owned cache on close; an
external cache remains owned by its caller and can be shared sequentially
with another evaluator.
