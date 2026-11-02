# Playback and adapter contracts

`MemoryPlayback` is a small SDK-free scene and kinematic player for verifying
the runtime boundary. It loads a successful plan, updates every robot at one
reference time, and records read-back states. It uses the CPU reference for
world link poses and collision spheres. Botrail, MuJoCo, and Isaac Sim remain
optional integrations; no simulator is needed for this player.

## Load and seek a saved plan

```python
from kinoweave import PlanningProblem, PlanResult
from kinoweave.adapters import MemoryPlayback, take_snapshot
from kinoweave.io import load

problem = load("problem.json", expected=PlanningProblem)
plan = load("result.json", expected=PlanResult)

with MemoryPlayback(problem) as player:
    initial_scene = take_snapshot(player)
    player.load(problem, plan)
    observation = player.set_time(0.25)
    frame = player.frame
    current_scene = take_snapshot(player)
```

The player is bound to its configured problem. Loading checks the input
digest, configuration, track coverage/order, endpoints, and declared collision
conditions. It also recomputes CPU analytic joint limits and sampled collision
validation. A producer's `success` label alone is insufficient. Coarser producer
sampling than the problem requires is rejected.

`set_time(t)` accepts any finite time in `[0, plan.makespan]`, including backwards
seeks. It evaluates the original `quintic_stop` law, without retiming, clamping,
or angle wrapping. `reset()` seeks to zero. Every robot remains present during
waits and terminal holds. Load and seek commit a complete frame atomically;
a failure leaves the previous frame intact. Returned frames are immutable
snapshots that remain valid after later seeks.

`frame.observation` uses canonical problem robot names and independent joint
order. `frame.robots` contains runtime joint slots, positions, velocities,
accelerations, world-frame link poses, and spheres. Mimic offsets affect
positions only; derivatives follow the multiplier. Fixed joints have zero
coordinates and derivatives.

`snapshot()` exports the current independent positions as new planning start
states, preserving goals, models, base poses, obstacles, and settings. The
new input digest changes when the start changes. To load a changed problem,
construct another player. Closing prevents further use of the live player;
already returned frame records remain usable.

## Bind runtime names and slots

By default, the memory runtime stores all model joints in the model's joint
declaration order. You can explicitly permute and rename its slots:

```python
from kinoweave.adapters import RobotBinding

binding = RobotBinding(
    robot="arm",
    runtime_robot="cell/arm_1",
    joint_names=("wrist", "shoulder"),
    runtime_joint_names=("joint_6", "joint_1"),
)
```

Supply one binding per robot through `MemoryPlayback(problem, bindings=(...))`.
`joint_names` lists planning model joints in runtime slot order. Empty
`runtime_joint_names` defaults to the same names. The memory runtime requires
every model joint exactly once, including fixed and mimic joints, and unique
runtime robot/slot names. Observations are reconstructed from these slots into
the canonical independent joint order.

This slot layout belongs to the memory runtime. A concrete SDK adapter can
use its own joint/actuator mapping while returning the common observation
contract. Models with no independent joint are rejected by this player because
schema v1 observations require a nonempty independent joint state. Static
obstacles do not need a robot binding.

## Record and verify playback

```python
from kinoweave.adapters import check_playback_report
from kinoweave.io import dump

with MemoryPlayback(problem) as player:
    player.load(problem, plan)
    report = player.record(time_step=0.25)

report.check_plan(problem, plan)
check_playback_report(problem, plan, report)
dump(report, "playback.json")
```

Recording starts a new forward session at reference time zero, includes every
knot and the final time, and samples the intervals without changing their time
law. The requested spacing is rounded down by subdividing each knot interval
an integer number of times. `report.time_step` is the resulting nominal spacing;
the observation timestamps retain the actual knot times and their floating-point
precision. The default sample budget is 100,000, checked before recording starts.

Each sample is committed and read back before the next. A failed sample leaves
the last committed frame and produces no complete report. Free backwards seeks
are not appended to a forward report. The memory clock is the reference clock,
so each recorded `time` equals `reference_time`; no wall-clock pacing occurs.

The report uses `mode="kinematic_playback"` and identifies the memory CPU
runtime, even when the original plan was computed on CUDA. `check_plan` checks
identity, reference-time range, coverage/order, and error units. It allows
partial runs and does not assess physical tracking. `check_playback_report`
checks observed positions and available velocities at their reference times.
Its position tolerance defaults to `joint_tolerance`; velocity tolerance
defaults to `1e-9` in each joint's native rate unit. Use zero tolerances when
exact CPU read-back agreement is required.

The memory runtime does not simulate dynamics or sense contacts. Empty contact
and tracking-error lists do not establish physical tracking or contact safety.
Its load-time collision check is `sampled`, including when the imported plan
asserts a stronger validation level.

## Run the file exchange example

```bash
uv run python examples/coordinated_planning.py --backend reference --output /tmp/kinoweave-demo
uv run python examples/playback_plan.py /tmp/kinoweave-demo/problem.json /tmp/kinoweave-demo/result.json --step 0.25 --output /tmp/kinoweave-demo/playback.json
```

The same playback command can read a GPU-produced problem/result pair in an
environment without Torch, cuRobo, or simulator SDKs.

## Implement another runtime adapter

The three runtime-checkable structural protocols remain separate:

| Protocol | Role |
| --- | --- |
| `SceneAdapter` | Export a configured scene and current start states through `snapshot()`. |
| `PlaybackAdapter` | Load a successful matching plan and seek all robots with `set_time()`. |
| `ExecutionAdapter` | Own reset, actuator mapping, simulation/control clock, stepping, observations, and shutdown through `run()`. |

Declare `AdapterCapabilities` for supported roles, shapes, joint types, and
mimic behavior. `take_snapshot(adapter)` verifies the returned native problem
against declared support. `check_capabilities(problem, capabilities, role)`
rejects unsupported roles or scene features with a named error.
`validate_plan_for_adapter(...)` provides binding and independent CPU validation
for a playback or execution adapter before transfer.

Structural protocol checks verify available members, not behavioral correctness.
Concrete adapters must validate model revisions, units, base/link frames,
geometry, names, and runtime bindings. Kinematic playback preserves reference
time; physical execution must record its actual runtime clock and corresponding
reference time, plus measured tracking errors and contacts when supported.

`MemoryPlayback` declares scene and playback support, has no physical `run()`
method, and rejects the execution role. It verifies this common boundary;
it does not qualify a Botrail, MuJoCo, or Isaac Sim connection.
