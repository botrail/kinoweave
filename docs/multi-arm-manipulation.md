# Multi-arm manipulation

`MultiArmPlanner` coordinates free motion, waits, and configured single- or
two-arm skills through a fixed schedule skeleton. Its search state tracks
roadmap positions, completed skills, ongoing source clocks, and object mode.
It returns the first complete plan that passes independent CPU verification.
The core works without simulator SDKs or other runtime dependencies.

The initial search returns a first solution. Apply
[objective evaluation and anytime refinement](refinement.md) afterward to
improve free motion while retaining configured skills.
Use [task schedule search](task-scheduling.md) to select between configured
assignments, parameters and resource orders and retry failed skeletons.

## Run the handover demo

From an installed source checkout:

```bash
python examples/two_arm_handover.py --output /tmp/handover
```

Two synthetic Cartesian tools pick up a part, meet for an atomic two-arm
handover, and place it on another table. The object passes through single-,
dual-, and single-holder modes. Both robots return to their native goal
configurations. The explicit sphere models are kinematic proxies, without
grasp-force, actuator, or physical robot accuracy claims.

For standalone playback:

```bash
uv run --project environments/viz --frozen python examples/two_arm_handover.py --output /tmp/handover --view
```

Open the printed URL, normally `http://127.0.0.1:8091`. Seek, Play, Loop,
Reset, and Speed use the original plan clocks. The object label identifies
its current holders. No simulator or CUDA installation is needed to view it.

Generate on CUDA with the [qualified environment](curobo.md):

```bash
uv run --project environments/cuda12 --frozen python examples/two_arm_handover.py --backend curobo --output /tmp/handover-gpu
```

Use `ManipulationViewer` in the visualization environment to open the saved
manipulation input/result, as in the [saved-plan example](manipulation.md#view-a-saved-manipulation-plan).

## Python API

```python
from kinoweave.io import dump, load
from kinoweave.manipulation import (
    ManipulationProblem,
    ManipulationSearchConfig,
    MultiArmPlanner,
)
from kinoweave.roadmap import PRMConfig

problem = load("problem.json", expected=ManipulationProblem)
planner = MultiArmPlanner(
    problem,
    prm=PRMConfig(num_samples=128, neighbors=12),
    search_config=ManipulationSearchConfig(algorithm="lacam", max_states=10_000),
)
result = planner.plan()
dump(result, "result.json")
print(result.status, planner.statistics)
```

Each robot gets one reusable static PRM, seeded with all of its skill entry
and exit configurations. `plan(samples={"alpha": (q1, q2)})` adds anchors.
Alternatively, `plan(graphs)` accepts a tuple of native `JointGraph` records,
one per robot, with exact native start/goal configurations, joint order,
unique configuration vertices, and explicit skill anchors. Missing skill
anchors disable that option. Invalid native edges are filtered before search.

Inputs use the existing [manipulation contracts](manipulation-contracts.md).
Skill traces, effects, contacts, grasp transforms, participant names, and
precedence come from the input. The planner does not generate grasps, choose
task assignments, change precedence, or optimize the declared cost.

## Task progress and skill timing

A skill becomes available when all participants are at the selected entry
configurations, all predecessors are complete, and its preconditions match.
A two-arm choice commits both robots to the same option and start time.
The search cannot combine one participant's skill with the other's wait or
motion. Ongoing skills have one forced continuation and cannot be interrupted.
Independent skills can overlap; other arms can move or retreat while a skill
is running. Completion advances the task state and applies the configured
per-object effects at their original source times.

Free motions use one shared scene tick and the native stopped quintic law.
An ongoing skill keeps its complete source trace, including nonuniform knots
and nonzero velocity across tick boundaries. It is never split into fresh
stopped-quintic fragments. With `timing="fixed"`, a nonintegral duration is
preserved and followed by an explicit hold to the next tick boundary. The
participants remain reserved through that hold. With `timing="stretch"`, the
whole source clock can slow uniformly to a tick boundary satisfying the
minimum duration. Contacts and effects use that same scaled clock.

Waiting dual holders stay coupled. Independent free motion by dual holders
is excluded; provide a bimanual skill to move the shared object. Overlapping
dual-holder groups involving more than two robots are unsupported. Goal
robots remain available to yield until every skill and native goal is complete.

## Conflict tables and search

Before recursive selection at a state, the planner gathers every candidate's
geometry and builds a complete unary/pair compatibility table. It checks
robot/robot, robot/object, object/object, and static-environment conflicts.
Each object is represented once: held objects follow their holders; a placed
object that may change belongs to its next action's deciding participant;
objects with no remaining action form static background geometry. This lets
concurrent decisions move objects without treating their old placements as
permanent obstacles.

CUDA uses cached native robot FK, on-device object transforms, and batched,
tiled candidate-pair distances. Dual-grasp closure, support, effect continuity,
and scoped contact penetration limits are checked as well. The CPU backend
provides the same sampled predicate. The table is transferred as Boolean
results. PIBT-style priority inheritance and rollback use table lookups;
they do not launch collision kernels recursively.

Skill choices rank ahead of free motion, then graph distance guides each
robot toward its next eligible entry or goal. Precedence-blocked robots use
a distance-to-home heuristic. Lazy constraint prefixes retain alternative
choices when heuristic completion fails. Fully constrained prefixes can
select every legal joint combination, including synchronized bundles.
`algorithm="exhaustive"` enumerates the same candidate model breadth-first
and is intended for small reference problems. The lazy planner makes no
optimality or shared-grid PIBT liveness claim.

`ManipulationSearchConfig` bounds states, constraint prefixes, candidates per
robot, skill ticks, complete state-table cells, query batch size, table-cache
retention, estimated state geometry, and sphere-pair scratch space. Reaching
a resource/time limit returns `timeout`. Full exhaustion returns
`no_solution_in_discretization`, limited to these finite candidates, tick
alignment, and sampled geometry. It does not prove continuous-space infeasibility.
The time limit is cooperative between batches; native calls and independent
final verification cannot be preempted. Geometry/cache byte limits describe
specific payloads and estimates, not total process or CUDA allocator memory.

## Verification and scope

Every candidate solution is reconstructed into original skill executions and
explicit per-robot connectors/holds, then checked with
`validate_manipulation` on CPU. Imported evidence is not trusted. Native and
manipulation JSON formats remain schema version 1. The motion-only CLI and
execution adapters keep accepting their original formats.

Collision checking samples aligned motions and actual skill slices, with
knots, contact boundaries, effects, and both sides of boundaries included.
Closing step permissions cover the endpoint; the next step checks its own
opening permissions, and final verification checks the reconstructed complete
timeline. Whole-skill enclosing volumes, continuous collision guarantees,
dynamics, tracking, continuous grasp/IK synthesis, and paper benchmark
performance are not provided. Botrail, MuJoCo, and Isaac Sim
remain optional integration targets.
