# Objective evaluation and anytime refinement

KinoWeave can improve a verified single- or multi-arm manipulation plan after
initial construction. It proposes free-motion shortcuts and shorter shared
free phases, filters complete candidates on CPU or CUDA, and independently
verifies each accepted plan on CPU. Every accepted update strictly lowers
the declared objective. Skills retain their selected options and source traces.

## Refine the handover demo

```bash
python examples/two_arm_handover.py --refine --output /tmp/handover-refined
```

The example saves `initial-result.json`, the best `result.json`, and
`refinement.json` containing before/after metrics, accepted cost history,
settings, backend identity, statistics, and termination reason.
`--refinement-seconds` sets the refinement budget; `--refinement-iterations`
sets the iteration limit. A budget exit can still return an improved,
independently verified plan.

To refine and display it without a simulator:

```bash
uv run --project environments/viz --frozen python examples/two_arm_handover.py --refine --output /tmp/handover-refined --view
```

For CUDA filtering, run the example in the [qualified GPU environment](curobo.md)
with `--backend curobo --refine`, then open the saved result using
[ManipulationViewer](manipulation.md#view-a-saved-manipulation-plan).

## Evaluate the objective

```python
from kinoweave.io import load
from kinoweave.manipulation import evaluate_objective

problem = load("run/problem.json")
result = load("run/result.json")
value = evaluate_objective(problem, result)
print(value.total, value.makespan, value.completion, value.distance)
```

`ManipulationCost` in the input supplies the nonnegative weights and distance
metric. The evaluator reports all components separately and computes:

`J = w_makespan * C_max + w_completion * sum(C_i) + w_distance * D`

For each robot, `C_i` includes the end of every assigned skill, even a
stationary skill, and the last joint motion before it remains stationary.
Unnecessary terminal goal holds do not increase `C_i`. `C_max` is the latest
robot completion; `schedule_duration` is the stored plan's complete timeline.
These can differ when the plan includes extra terminal padding. The evaluator
does not reward merely deleting that storage padding. Native result timing
and schema fields are unchanged.

| Distance metric | Definition | Unit |
| --- | --- | --- |
| `joint_l1` | Sum of absolute independent-joint changes along every source segment. | Raw joint units |
| `joint_l2` | Sum of Euclidean independent-joint segment lengths. | Raw joint units |
| `tool_path` | Sampled world-position path length of one named link per robot. | m |
| `sphere_volume_path` | Sum of each robot collision sphere's center-path length multiplied by `4*pi*r^3/3`. | m^4 |

Joint distances are exact for the declared piecewise joint-space paths.
Continuous angles remain unwrapped. Angular and prismatic joint components
are not normalized; weights must reflect the intended units and experiment.
Tool cost measures translation, without an orientation penalty. Sphere costs
use robot spheres only, without volume normalization or held-object spheres.
Overlapping proxies contribute separately.

FK distances use joint-parameter subdivisions controlled by
`MetricConfig.max_joint_step`, including mimic-joint variation, rather than
time samples. Constant segments and forward-collinear stops are canonicalized
before sampling, with a normalized-direction tolerance of `1e-12`. Changing
the clock or adding stops on the same joint-space line does not manufacture
a distance improvement. FK lengths remain numerical polyline approximations,
without a certified arc-length error bound. `max_samples` bounds FK queries.

For tool metrics, an unbranched model defaults to its sole leaf. Select links
explicitly when the model has multiple leaves:

```python
from kinoweave.manipulation import MetricConfig

metric = MetricConfig(
    max_joint_step=0.025,
    tool_links=(("alpha", "tool0"), ("beta", "tool0")),
)
value = evaluate_objective(problem, result, config=metric)
```

Objective records include per-robot values, units, resolved links, metric
settings, and input/plan digests. Keep package versions and these settings
fixed for comparisons. `evaluate_objective` checks structural binding; it
does not independently certify collision or rate safety.

## Keep the best verified plan

```python
from kinoweave.io import dump
from kinoweave.manipulation import ManipulationRefiner, RefinementConfig

refiner = ManipulationRefiner(
    problem,
    config=RefinementConfig(time_limit=10, max_iterations=32, seed=0),
)


def updated(update):
    print(update.sample.elapsed, update.sample.objective.total)
    dump(update.result, "best-result.json")


report = refiner.refine(result, on_improvement=updated)
dump(report.result, "result.json")
print(report.status, report.termination_reason)
```

The seed is freshly verified on CPU even if it already claims `success`.
The first callback contains that verified seed; subsequent callbacks contain
strict improvements. Returning `False` stops cooperatively.
`refiner.best_result` exposes the latest verified incumbent. The convenience
function `refine_manipulation(problem, result, config=...)` returns the same
report without a callback. History entries contain accepted costs and elapsed
refinement times, beginning with the seed at zero.

Seed verification and initial measurement precede the refinement clock and
are reported as initialization time. The refinement budget includes backend
setup, proposal generation, scoring, filtering, callbacks, and acceptance
verification. Limits are checked between operations; an in-flight native call
or CPU verification cannot be preempted. A verified improvement completing
after a soft deadline is retained before returning. `max_iterations=0` or
`time_limit=0` returns the freshly checked seed.

Reports distinguish `completed`, `budget_exhausted`, `cancelled`, `unsupported`,
and `invalid_seed`. Check both status and reason. No improvement, resource
limits, cancellation, or unavailable CUDA preserve the verified incumbent.
An invalid seed returns a failed result without a plan. If initial metric
measurement exhausts its sample budget, the verified seed remains available,
with no before/after cost assertion. An unavailable CUDA request does not
switch to CPU optimization; the report identifies the failure explicitly.

## Candidate construction and acceptance

Shortcuts join stopped free-motion knots with a direct stopped quintic. They
may retain the original interval or finish earlier with explicit goal holds.
Single-arm changes leave other clocks untouched. Composite changes are built
from raw edits before rejecting individual inter-robot collisions: two changes
may work together even when neither works alone. A bounded greedy pass can
fuse disjoint promising windows, then rechecks the entire composition.

Shared free phases can be shortened in parallel or by prioritized movements.
Deleting shared stationary time shifts all later activities together. No
deleted interval may contain an active skill. Selected skills, durations,
relative contacts/effects, and source time laws are preserved within the
contracts' arithmetic tolerance. Refinement can leave the search's shared
tick grid. Accepted candidates may trim unused terminal holds.

CUDA filtering batches complete candidate timelines on aligned samples,
reusing robot FK and evaluating held objects, dual grasps, supports, contacts,
and all robot/object/environment pairs. Different timeline grids are grouped
separately. Analytic rate checks and structural binding precede filtering;
independent CPU verification is required before acceptance. Objective ranking
and final verification run on CPU, including for CUDA requests.

`RefinementConfig` bounds iterations, candidate windows/count, composite size,
batch size, estimated batch geometry, pair scratch, and validation samples.
FK metric sampling has a separate `MetricConfig` budget. Tensor-cache settings
can be passed to `ManipulationRefiner` as `max_cache_bytes=`. These caps describe
specific payloads/estimates, without bounding total process or CUDA allocator
memory. The candidate neighborhood is finite and bounded; exhaustion does not
certify global or local optimality over continuous space.

Collision verification remains sampled. Whole-skill certified envelopes,
continuous collision guarantees, dynamics, tracking, new grasp/skill options,
task-catalog search, and paper benchmark performance are outside
this refinement API. It operates independently of Botrail, MuJoCo, or Isaac Sim.
The separate [schedule planner](task-scheduling.md) can select configured
alternatives and invoke refinement for each feasible skeleton.
