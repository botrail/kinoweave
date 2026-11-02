# Data contracts

KinoWeave's data records describe planning problems, trajectories, plan
results, and runtime observations without importing GPU or simulator SDKs.
Constructors validate their structure and consistency. Creating or importing
a record does not plan a motion or certify that a trajectory is collision-free.

## Load and save a problem

Install the source checkout with `pip install .` before running these examples.

```python
from kinoweave import PlanningProblem, PlanProvenance, PlanResult
from kinoweave.io import document_digest, dump, load

problem = load("examples/data/two_sliders.problem.json", expected=PlanningProblem)
dump(problem, "problem.json")

# Record an unstarted request, without claiming that planning succeeded.
result = PlanResult(
    provenance=PlanProvenance(
        problem_digest=document_digest(problem),
        config=problem.config,
    ),
)
dump(result, "result.json")
restored = load("result.json", expected=PlanResult)
restored.check_problem(problem)
assert restored.status == "not_run"
```

The [two-slider example](../examples/data/two_sliders.problem.json) is a
data-exchange example. It defines two instances of one model, a static box,
joint limits, and starting and target configurations. The
[round-trip script](../examples/contract_roundtrip.py) runs without a planner.

## Coordinates and models

All records use metres, radians, seconds, and right-handed frames. A `Pose`
maps local coordinates into its parent frame. Its quaternion is **xyzw** and
must have unit norm within `1e-6`; it is not automatically normalized.
Robot base poses and obstacle poses are relative to the world. Joint origins
are relative to their parent link; link-sphere centres are link-local.

`RobotModel` defines a single rooted link tree, joints, collision spheres,
an explicit independent `joint_order`, optional allowed self-collision pairs,
and a model revision. Instances refer to models by name. Robots and static
obstacles share a scene-name namespace.

Supported joint records are `fixed`, `revolute`, `continuous`, and `prismatic`.
Every moving joint needs explicit positive velocity and acceleration limits.
Revolute and prismatic joints also require lower and upper position bounds.
Continuous coordinates are **unwrapped**, and their position bounds must be null.
Axes of moving joints must be unit vectors.

Fixed and mimic joints are excluded from `joint_order`. A mimic joint names
a compatible source and follows `multiplier * source + offset`; cycles and
unknown sources are rejected. `RobotModel.joint_values()` resolves those
coordinates without computing forward kinematics. Initial and target states
are checked against all position limits, including derived mimic coordinates.

Robots have explicit collision spheres. Static obstacles currently support
`Sphere` and `Box`, selected by a required JSON `kind`. Box sizes are full
side lengths. These records do not claim to cover an underlying visual mesh.
Meshes, free/ball joints, movable-object modes, and manipulation events are
not part of schema version 1.

## Planning configuration

`PlanningProblem.config` is a `PlannerConfig` record:

| Field | Default | Meaning |
| --- | --- | --- |
| `backend` | `"curobo"` | Geometry backend; use `"reference"` for CPU planning. |
| `device` | `"cuda:0"` | Backend device; pair the reference backend with `"cpu"`. |
| `seed` | `0` | Non-negative sampling/search seed. |
| `time_limit` | `10.0` | Positive soft planning budget in seconds. |
| `time_step` | `0.1` | Positive common trajectory tick interval in seconds. |
| `validation_time_step` | `0.01` | Positive maximum collision sample spacing in seconds. |
| `collision_margin` | `0.01` | Non-negative collision clearance margin in metres. |
| `collision_tolerance` | `1e-6` | Non-negative additional collision tolerance in metres. |
| `joint_tolerance` | `1e-6` | Non-negative position tolerance in each joint's native unit. |

The dataclass defaults request CUDA. Select `reference` and `cpu` explicitly
for dependency-free planning. The CLI's `example` command produces a CPU
problem by default. See [planning budgets](planning.md#budgets-and-results)
for deadline behavior and search-specific resource settings.

## Shared-time trajectories

`TrajectoryBundle` contains a shared zero-based `times` array and one
`RobotTrajectory` per robot. Every track has a joint-name order and position
rows for **every** knot, including end holds. Knot times must strictly
increase at uniform intervals, with relative tolerance `1e-9`.

The sole interpolation identifier is `quintic_stop`. For an interval
`[t0, t1]`, its meaning is:

```text
u = (t - t0) / (t1 - t0)
q(t) = q0 + (q1 - q0) * (10*u^3 - 15*u^4 + 6*u^5)
```

Velocity and acceleration are zero at each knot. Continuous-joint coordinates
are used as supplied, without shortest-angle wrapping. The representation
preserves the time law. `TrajectoryBundle.evaluate(time)` returns every
robot's positions, velocities, and accelerations in its track's joint order.
Times outside `[0, duration]` are rejected. Continuous coordinates remain
unwrapped. The [CPU reference](reference.md) provides analytic derivative-limit
checks, interval evaluation, and sampled collision validation.

## Plan results and validation

`PlanResult.status` is one of:

| Status | Meaning |
| --- | --- |
| `not_run` | No planning has run; no trajectory or backend execution is claimed. |
| `success` | A trajectory and passed validation evidence are supplied. |
| `timeout` | No validated solution was returned within the time or search budget; inspect `termination_reason`. |
| `no_solution_in_discretization` | The solver exhausted its specified finite search space. |
| `invalid_problem` | The request was rejected as invalid. |
| `unsupported` | A requested model, shape, constraint, or operation is unsupported. |
| `validation_failed` | A candidate failed final validation; failed evidence is required. |

Only `success` carries a trajectory. It requires provenance identifying the
problem digest, configuration, and actual backend/version/device, plus a
passed report containing `joint_limits`, `collisions`, and `endpoints` checks.
A solver that times out after finding a validated solution returns `success`
and records its stopping reason separately. Search exhaustion is a solver's
responsibility and is not inferred by the data layer.

Validation reports distinguish `not_run`, `passed`, and `failed`. Performed
checks identify a `sampled` or `conservative` level and their collision
margin and numeric tolerance. Sampled checks require a positive time step.
An imported report is evidence asserted by its producer, not a substitute for
an independent verifier. The data layer does not independently prove either level.

`PlanResult.check_problem(problem)` checks the problem digest, configuration,
robot coverage, joint order, knot interval, endpoints, and knot-position
limits. Endpoint and position comparisons use `joint_tolerance` in each
joint's native unit. This binding check does not evaluate interval velocities,
accelerations, or collisions. Use `validation.validate_trajectory(problem,
trajectory)` to recompute those checks from the trajectory itself.

## Runtime boundaries

`kinoweave.adapters` defines three optional structural protocols:

| Protocol | Responsibility |
| --- | --- |
| `SceneAdapter.snapshot()` | Export a configured scene and goals as a planning problem. |
| `PlaybackAdapter.load()` / `set_time()` | Replay every robot at a common reference time without physics or retiming. |
| `ExecutionAdapter.run()` | Own reset, actuator mapping, control, stepping, observation, and shutdown. |

An implementation declares `AdapterCapabilities` for its shapes, joint types,
mimic support, and supported roles. Loading or running a plan requires a
successful result bound to the intended problem. Botrail, MuJoCo, and Isaac
Sim implementations are not included; none is a required dependency.

The [memory player](playback.md) implements native scene snapshots and kinematic
playback. Shared conformance helpers check declared model support and reload
plans with independent CPU sampled validation. Physical execution remains a
separate adapter role.

`ExecutionReport` is a separate document. It identifies the plan-result digest,
runtime/version/device, execution mode, time step, observations, optional
per-joint tracking errors, and contacts. Observations retain both runtime
time and corresponding trajectory-reference time. Runtime time must increase;
reference time cannot reverse. Per-joint errors state `rad` or `m` rather than
combining linear and angular errors into one number. Contacts are observations
and do not by themselves classify an execution as valid or invalid.

`ExecutionReport.check_plan(problem, plan)` binds the result digest, reference
time range, robot coverage, canonical joint order, and tracking-error units.
It does not judge physical tracking or require a complete run.
`adapters.check_playback_report(...)` additionally compares kinematic samples
with the original time law, including velocities when supplied.

## JSON compatibility and errors

The three root formats are `kinoweave.problem`, `kinoweave.result`, and
`kinoweave.execution`. Every JSON document must explicitly include `format`,
integer `schema_version: 1`, and a `conventions` object. Fields with defaults
may be omitted; serialization writes them explicitly.

Unknown fields, shape kinds, formats, and versions are rejected. Duplicate
object keys, non-finite numbers, incorrect dimensions, boolean numbers, and
string-to-number coercions are also rejected. `ContractError` extends
`ValueError` and exposes `path` and `message`; model-reference errors include
the relevant robot, joint, or link name.

Native records are frozen dataclasses. Numeric arrays passed as lists are
copied to immutable tuples. `dump` / `load` use UTF-8 files, and `dumps` /
`loads` exchange strings. `expected=...` checks the root document type.

`document_digest()` computes SHA-256 over normalized fields, including defaults,
using sorted JSON object keys and preserving array order. Whitespace in the
input file does not affect it. Geometry, model revision, and configuration
changes do. A digest links records to an input; it is not a signature or a
validation certificate.
