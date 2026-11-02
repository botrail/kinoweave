# Manipulation data contracts

`kinoweave.manipulation` provides dependency-free records for movable objects,
grasp states, configured skills, schedule skeletons, and timed plan candidates.
It supports strict JSON exchange and structural/symbolic consistency checks.
The separate [manipulation API](manipulation.md) adds mode geometry,
independent CPU verification, single-arm construction, and object playback.
The motion-only `Planner`, CLI, and execution adapters accept their original
formats; use the manipulation API for these records.

The new formats are `kinoweave.manipulation.problem` and
`kinoweave.manipulation.result`, each with schema version 1. They are separate
from the existing motion-only formats. Existing `kinoweave.problem`,
`kinoweave.result`, and `kinoweave.execution` documents retain their fields,
time law, and semantic digests.

## Read and write an input

Install the core from a source checkout, then run the contract example:

```bash
python examples/manipulation_contract.py --output /tmp/transfer-contract
```

It creates a synthetic two-arm transfer input and a `not_run` result. It does
not solve a manipulation problem or certify a physically executable transfer.
The input is also supplied as
[manipulation_transfer.problem.json](../examples/data/manipulation_transfer.problem.json).

```python
from kinoweave.io import document_digest, dump, load
from kinoweave import PlanProvenance
from kinoweave.manipulation import ManipulationProblem, ManipulationResult

problem = load("examples/data/manipulation_transfer.problem.json", expected=ManipulationProblem)
result = ManipulationResult(
    provenance=PlanProvenance(
        problem_digest=document_digest(problem),
        config=problem.scene.config,
    )
)
result.check_problem(problem)
dump(result, "result.json")
assert result.status == "not_run"
```

Both formats use the standard `load`, `loads`, `dump`, `dumps`, and
`document_digest` functions. Unknown fields, duplicate keys, unknown versions,
nonfinite values, and implicit numeric coercions are rejected. Construction
freezes input sequences. No optional SDK is imported.

## Scene, objects, and modes

`ManipulationProblem.scene` is an existing `PlanningProblem`: it supplies the
robot models, instance base poses, initial/final joint states, static
obstacles, and configuration. Movable objects are declared separately and
must not also be registered as static obstacles.

| Record | Meaning |
| --- | --- |
| `RigidObject` | A named object, revision, and explicit object-local `ObjectSphere` geometry. |
| `PlacedObject` | An object pose in the world and an optional static-obstacle support name. |
| `GraspAttachment` | A robot/link reference and `T_link_object` pose. |
| `HeldObject` | An object held by one or two distinct robots. |
| `ManipulationMode` | A named set of object states, with no duplicate object entries. |

Initial and goal modes must each cover every declared object exactly once.
A state is explicitly placed or held; it cannot contain both placement and
attachment fields. All references are checked against the scene. Units,
right-handed frames, and xyzw quaternions follow the [native contracts](contracts.md).
An empty object set is allowed for tasks such as welding or inspection.

The `support` name records an intended support relationship. A two-holder
state records both attachment transforms. Neither proves that an object is
supported or that the two robot frames agree geometrically.

## Skills and schedule skeletons

A `Skill` is one configured action instance, with a unique name, one or two
participating robot names, a positive `minimum_duration`, and one or more
named `SkillOption` records. Its `kind` is a descriptive label such as
`pick`, `place`, `handover`, or `carry`; it does not create effects or select
a controller automatically.

Each option declares:

- A `JointTrace` covering exactly the participants, using each native model's
  complete independent-joint order and position-valid knots.
- Per-object preconditions, expressed as placed or held states.
- Time-ordered `ModeTransition` records containing explicit before/after
  states of the same object. Their chain must agree with the preconditions.
- Optional link/body-specific `ContactWindow` records within the trace.

Transitions update only their named objects. This allows skills on different
robots and different objects to run independently. There is no need to
enumerate every combination of global modes in the input.

`ScheduleSkeleton` contains configured skills and `ActionOrder(before, after)`
dependencies. Action names must exist, and the graph must be acyclic. Skills
sharing a robot or a referenced manipulation object require an explicit
ordering path. Joint object access is represented by a single two-robot
skill. Array order alone does not imply precedence. Split-action handover
exchange constraints are not currently represented.

Contact bodies distinguish a specific robot link, a movable object, or a
static obstacle. Contact windows cannot authorize a non-participating robot,
refer to an undeclared object condition, or exceed the source trace. They
are scoped permissions interpreted by the geometric verifier, with a separate
penetration cap. They do not provide evidence that contact occurred correctly.

## Time and costs

`JointTrace` has an explicit, strictly increasing relative `times` array
starting at zero and complete robot position tracks. Unlike motion-only
`TrajectoryBundle`, its timestamps may be nonuniform. Its interpolation is
explicitly `linear` or `quintic_stop`. Construction records the time law;
the separate verifier checks rates and sampled geometry. Moving linear
corners and nonzero endpoint velocities fail finite-acceleration verification.

A skill with `timing="fixed"` preserves its option's duration. With
`timing="stretch"`, a scheduled skill may take longer, scaling the entire
source clock uniformly, including effects and contact windows. It may not
be sped up, and its scheduled duration must meet `minimum_duration`.

`ManipulationCost` declares nonnegative makespan, sum-of-completion-times,
and distance weights; at least one must be positive. The distance metric
is explicit (`joint_l1`, `joint_l2`, `tool_path`, or `sphere_volume_path`).
This record selects the objective used by the separate
[evaluator and refiner](refinement.md). Metrics and units must match the
intended benchmark before comparing performance.

## Timed candidates and binding

`ManipulationPlan` contains `ScheduledMotion` connectors/holds,
`ScheduledSkill` selections, a makespan, and a final mode. Motions contain
one robot's local trace; skill selections refer to input option names.
Independent local traces preserve their original interpolation when another
robot has an unrelated knot.

`ManipulationResult.check_problem(problem)` checks:

- Exact input digest and configuration identity.
- Coverage of every skeleton skill, option identity, duration constraints,
  and precedence (`before.end <= after.start`).
- Explicit activity for every robot from zero to makespan, with no overlap
  or implicit gaps, matching joint boundaries, and native start/goal states.
- Selected option preconditions, time-scaled effects, and the resulting
  symbolic final/goal object states.

Adjacent schedule endpoints must be exactly shared; reuse the same timestamp
for an activity's end and its successor's start. Source trace duration
comparisons allow a relative `1e-9` arithmetic tolerance. At a shared
timestamp, dependency order places a predecessor's final effect before the
successor's precondition. Mode names are descriptive; state matching uses
the declared per-object records, without geometric pose approximation.

Use `status="candidate"` for an unverified timed plan. Binding a candidate
does not change its status or validation report. `success` requires an
identified backend and passed evidence for `joint_limits`, `collisions`,
`endpoints`, `precedence`, `mode_transitions`, `skill_traces`, `contacts`, and
`object_goals`. As with native reports, supplied evidence is a producer's
assertion; the data classes do not independently certify it.

The [geometric verifier](manipulation.md#verification) checks sampled collision,
attachment closure, support, mode-transition continuity, and joint rates.
Continuous collision guarantees, whole-skill swept envelopes, dynamics, and
controller tracking are outside that verification. Motion-only CLI planning,
validation, playback, and viewing reject manipulation documents.

The separate `kinoweave.manipulation.tasks` schema-v1 root defines finite
task alternatives and dependencies. See [task scheduling](task-scheduling.md)
for catalog semantics and selected problem/result binding.
