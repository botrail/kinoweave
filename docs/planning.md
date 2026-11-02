# Coordinated planning

`Planner` coordinates robot motions on individual joint graphs. Geometry is
evaluated with the selected backend, search runs on the CPU, and the returned
shared-time trajectory is independently checked with the CPU reference.
Botrail, MuJoCo, Isaac Sim, and other execution platforms remain optional.

## Plan from a problem

Install the [locked CUDA environment](curobo.md), then:

```python
from dataclasses import replace

from kinoweave import Planner, PlanningProblem, SearchConfig
from kinoweave.io import dump, load
from kinoweave.roadmap import PRMConfig

problem = load("examples/data/retreat.problem.json", expected=PlanningProblem)
problem = replace(problem, config=replace(problem.config, backend="curobo", device="cuda:0"))

with Planner(
    problem, prm=PRMConfig(num_samples=64, neighbors=8), search=SearchConfig(algorithm="lacam")
) as planner:
    result = planner.plan()
dump(problem, "problem.json")
dump(result, "result.json")
print(result.status, result.makespan, result.termination_reason)
```

`plan()` constructs individual PRMs on first use and reuses them on subsequent
calls. `planner.roadmaps` exposes generated roadmaps. Its `last_graphs` property
contains the graph set bound during the most recent call. The problem is
immutable; construct another planner for a changed scene or configuration.

You can supply `plan(graphs)` with one `JointGraph` or matching `Roadmap` per
robot instead. Robot names, joint order, start/goal coordinates, and roadmap
problem/timing identity are checked. Each edge takes one common `time_step`.
Supplied edges are checked on demand for analytic joint limits and sampled
self/static/inter-robot collisions; graph indices are never used as collision
identities across robots.

`Planner` is not thread-safe. Use a context manager and one instance per
concurrent worker. Optional `tiling=GpuTiling(...)` and `cache=GeometryCache(...)`
configure the [GPU work budgets and geometry cache](roadmaps.md).

## Run the retreat example

```bash
uv run --project environments/cuda12 --frozen python examples/coordinated_planning.py --output /tmp/kinoweave-demo
```

The example uses a supplied T-shaped graph: one robot must retreat while the
other passes. It saves a matching `problem.json` and `result.json`, and writes
the result to stdout. Native SDK diagnostics go to stderr. The graph restricts
candidate motion; the scene does not contain physical corridor walls.

For a CPU-only experiment, select it explicitly:

```bash
uv run python examples/coordinated_planning.py --backend reference --output /tmp/kinoweave-cpu-demo
```

`backend="reference", device="cpu"` is also supported by `Planner`. Missing
GPU dependencies or an unavailable CUDA device return `unsupported`; a GPU
request is not switched to CPU planning. CPU final verification of a GPU plan
is a separate, intentional verification step.

## Select a search method

Pass a `SearchConfig` when constructing the planner or to an individual
`plan(..., search=...)` call.

| Algorithm | Behavior |
| --- | --- |
| `lacam` | Default: lazy constraints over joint configurations, with priority inheritance and backtracking to propose successors. |
| `prioritized` | Fixed-order space-time planning; previously planned trajectories remain reserved at their goals. |
| `auto` | Try prioritized planning, then lazy constraint search if needed. |
| `joint_astar` | Finite product A* using the same geometric predicates; useful for small correctness cases. |

The lazy search adapts the two-level structure of
[LaCAM](https://arxiv.org/abs/2211.13432) and priority inheritance from
[the author's implementation](https://github.com/Kei18/lacam). Each joint
configuration owns a queue of partial next-motion constraints. A PIBT-inspired
generator reserves a candidate and recursively asks robots whose waiting
geometry blocks it to move. Failed branches roll back. Constraints eventually
specify every robot, so heuristic proposal failures do not discard the
remaining fully specified compatible successors.

This implementation uses individual robot graphs and geometric motion
compatibility. Original shared-grid PIBT liveness assumptions do not transfer
to these models. Lazy search returns a first solution and makes no optimality
claim. Product A* minimizes tick count in the specified finite, sampled graph;
the overall automatic method returns its first verified solution.

Waits are available at every vertex. A robot at its individual goal can leave
and return before the joint goal is reached. All robots remain present until
global completion. In the fixed-priority baseline, a lower-priority robot
must also be able to hold its goal while higher-priority reserved motions
finish. `priority_order=("beta", "alpha")` changes that baseline's order.

With `exact_fallback=True` (the default), two-robot lazy searches reserve part
of the remaining wall budget for a product A* complement if the lazy phase
times out. Set it to false to run the lazy method alone. Explicit
`prioritized` runs only that baseline; an incomplete baseline can be followed
by the other methods through `auto`.

## Budgets and results

`PlannerConfig.time_limit` is one **soft** wall budget for backend setup,
roadmap preparation, all search phases, and verification. `auto` gives the
prioritized phase up to 25% of the remaining budget. When the two-robot exact
complement is enabled, the lazy phase gets up to 80% of its remaining budget.
Fallback uses the same evaluator, deadline, and aggregate resource counters.

| SearchConfig field | Meaning |
| --- | --- |
| `max_states` | Aggregate state creation budget across search phases. |
| `max_constraints` | Aggregate creation budget for lazy partial constraints. |
| `priority_horizon` | Maximum ticks considered by each prioritized space-time search. |
| `max_query_pairs` | Maximum motion pairs per compatibility query block. |
| `max_cached_pairs` | Resident boolean compatibility cells; zero disables this cache. |
| `max_cached_options` | Resident per-vertex motion-option entries. |
| `max_validation_samples` | Maximum samples in final CPU trajectory verification. |

Compatibility queries reuse bounded CPU boolean blocks over the GPU geometry
cache. A backend result with foreign problem, motion, timing, or backend
identity is rejected. Changing query blocks or disabling caches does not
truncate the mathematical candidate set.

A result is `success` only after all tracks are reconstructed and CPU
verification passes. Its provenance identifies the problem digest and actual
geometry backend/device. JSON diagnostics include search settings, PRM settings
when generated, graph digests/counts, attempted algorithms, counters, and the
termination stage. These diagnostics preserve schema v1 without changing the
input digest of existing problem documents.

`phase_timings` adds wall-time measurements for backend setup, roadmap work,
graph binding, endpoint checks, search, reconstruction, and final CPU validation.
Search includes backend queries and synchronization. These measurements stop
before result assembly and are not isolated GPU kernel times. GPU memory
estimates cover evaluator tensor payloads, not the entire CUDA context or
native SDK allocations; see [work budgets](roadmaps.md#bound-work-and-reuse-geometry).

`no_solution_in_discretization` is returned after finite lazy/product search
exhaustion or a provably disconnected individual graph. A time, state,
constraint, or prioritized horizon limit returns `timeout`. In particular,
`prioritized_incomplete` reports an incomplete baseline, even if wall time
remains; it is not a proof that the supplied graphs have no joint solution.
Malformed bindings or failed evaluations return `invalid_problem`, unsupported
backends/models return `unsupported`, and rejected CPU verification returns
`validation_failed` without a trajectory.

Budget checks run between bounded operations. One SDK call or final CPU
verification can overrun the deadline. A verification that completes late and
passes still returns `success` with `time_limit_after_validation`; an expired
budget before verification returns `timeout`.

All successful reports currently have level **`sampled`** for collision
checks. Analytic joint-rate checks cover each whole quintic interval, but
collision sampling does not prove continuous-time safety. Supply geometry
and limits appropriate to your robot. Manipulation modes, attachments,
simulator SDK adapters, and physical control tracking are not provided.
