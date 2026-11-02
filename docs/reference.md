# CPU reference motion, validation, and search

The CPU reference evaluates the shared-time trajectory contract, checks
supplied collision geometry, and searches small, explicit joint graphs.
It uses Python's double-precision arithmetic and has no third-party runtime
dependencies. It runs independently of the GPU backend and simulators.

## Evaluate motion

```python
from kinoweave.trajectory import evaluate_segment, minimum_duration

state = evaluate_segment((0.0,), (0.3,), duration=2.0, time=1.0)
print(state.positions, state.velocities, state.accelerations)
```

Every interval uses the `quintic_stop` law. Writing `u=t/T` and `d=q1-q0`:

```text
q(t)   = q0 + d (10u³ - 15u⁴ + 6u⁵)
dq(t)  = d (30u²(1-u)²) / T
ddq(t) = d (60u(1-u)(1-2u)) / T²
max |dq|  = (15/8) |d| / T
max |ddq| = (10/sqrt(3)) |d| / T²
```

`minimum_duration(model, start, end)` returns the smallest duration required
by all velocity and acceleration limits, including mimic joints. It returns
zero for a hold and does not check position limits or collisions.
`segment_limit_issues(...)` checks endpoint position extrema and analytic
derivative peaks. The quintic blend is monotonic, so bounded and affine mimic
joint positions stay between their endpoint values. `joint_tolerance` applies
to positions; derivative comparisons use only a `1e-12` relative arithmetic
guard. Rates are checked per joint in m/s or rad/s and m/s² or rad/s².

`TrajectoryBundle.evaluate(time)` evaluates every track at one common time
and returns `{robot_name: MotionState}`. Every knot is a stop. Waits have
zero derivatives, continuous coordinates remain unwrapped, and times outside
the interval or trajectory are rejected. Inputs and computed outputs must
be finite; numeric evaluation failures do not imply clear geometry.

## Inspect geometry

```python
from kinoweave import PlanningProblem
from kinoweave.geometry.reference import ReferenceCollisionChecker
from kinoweave.io import load

problem = load("examples/data/retreat.problem.json", expected=PlanningProblem)
checker = ReferenceCollisionChecker(problem)
states = {robot.name: robot.start for robot in problem.robots}
fk = checker.forward("alpha", states["alpha"])
hits = checker.collisions(states)
```

CPU FK supports every core joint kind, arbitrary unit axes, fixed origins,
continuous coordinates, mimic chains, and instance base poses. `ReferenceFK`
returns world-frame link `Pose` records and spheres in original input order.
The checker includes self collisions, static spheres and oriented boxes,
and every pair of robots. All robots must be supplied, including waiting and
arrived robots. Same-link sphere pairs and explicitly allowed self-link pairs
are excluded; adjacent links are otherwise checked.

Clearance is signed surface separation. A collision means
`clearance <= collision_margin + collision_tolerance`, including contact.
Spheres are exactly the supplied model geometry, with no mesh approximation
or automatic geometry generation. Geometry queries do not check joint limits.

## Validate intervals and saved trajectories

```python
from kinoweave.validation import validate_interval, validate_trajectory

start = {robot.name: robot.start for robot in problem.robots}
end = {robot.name: robot.goal for robot in problem.robots}
report = validate_interval(problem, start, end, duration=2.0)
print(report.status, report.issues)
# For a complete TrajectoryBundle:
# report = validate_trajectory(problem, trajectory)
```

Both functions check analytic position/velocity/acceleration limits and
sampled self/static/inter-robot collisions. A candidate interval can start
and end at arbitrary configurations. Full trajectory validation additionally
checks robot coverage, joint order, the problem's knot interval, and start/goal
agreement. It reconstructs motion from the stored time law and does not trust
search verdicts or an imported validation report.
For an imported `PlanResult`, also call `result.check_problem(problem)` to
verify its input digest and configuration before validating its trajectory.

Sampling subdivides **each interval**, includes every knot, and uses a spacing
no larger than `PlannerConfig.validation_time_step`. The report records that
maximum requested spacing. All tracks cover the same total duration, so a
robot's terminal hold remains occupied while others move. A failed report
can stop at the first collision and identifies its time and sphere pair.
Missing tracks, numerical errors, and a sampling budget overrun cannot pass.

The default `max_samples=100_000` limits samples across the entire checked
trajectory. Supply a larger explicit budget when needed. These reports have
level **`sampled`**: a thin obstacle can fall between samples. Analytic rate
checks cover the entire interval; collision sampling does not establish
continuous-time safety or verify geometry absent from the input.

## Search a small supplied graph

Run the reference example:

```bash
python examples/reference_coordination.py > result.json
```

It loads two XY robots and supplies a T-shaped graph. One must use its
retreat branch to exchange endpoints. The graph restricts available motions;
the scene does not contain physical corridor walls. The output is a real
`PlanResult` with CPU provenance and a recomputed sampled validation report.

```python
from kinoweave.search.reference import JointGraph, reference_plan

vertices = ((-1, 0), (0, 0), (1, 0), (0, 1))
edges = ((0, 1), (1, 2), (1, 3))
graphs = (
    JointGraph(
        robot="alpha", joint_names=("x", "y"), vertices=vertices, edges=edges, start=0, goal=2
    ),
    JointGraph(
        robot="beta", joint_names=("x", "y"), vertices=vertices, edges=edges, start=2, goal=0
    ),
)
result = reference_plan(problem, graphs, max_states=100_000)
```

Select `backend="reference", device="cpu"` explicitly. There is no automatic
CPU fallback from a GPU request. Each undirected graph needs a nonempty joint
order matching its robot and explicit start/goal vertices matching the problem.
Every edge takes one `time_step` and is checked with `validate_interval`.
Edges that exceed limits are rejected; the search does not silently retime or
subdivide them. Provide intermediate vertices or a larger common interval.

A wait is available at every vertex. A robot may leave its own goal before
the joint goal is reached. Breadth-first search returns a minimum-tick path
in this finite graph product under the sampled predicate. When all robots
already start at their goals, schema v1 represents the result as one shared
hold interval. The final trajectory is recomputed and validated before a
successful result is returned.

`no_solution_in_discretization` is returned only after exhaustion of the
finite product, and does not prove impossibility in continuous space.
`time_limit` or the discovered-state cap produces `timeout`; numerical failures
produce `invalid_problem`, and rejected final verification produces
`validation_failed`. Wall time is checked between transitions; validation of
one transition or the final trajectory can overrun that soft deadline.

This reference search is intended for small correctness examples. It does
not build PRMs or generate GPU interference tables. Use [Planner](planning.md)
for PRM preparation and prioritized or LaCAM-style coordination.
