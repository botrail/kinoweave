# Experimental scaling and measurement

The supported application remains fixed-arm coordination, including the
two-UR5 example. KinoWeave also includes an experimental synthetic inspection
example for two, four or eight tools and longer task lists. This checks task
coverage, scheduling, cache reuse and verification as the problem grows. Its
Cartesian tool spheres do not model complete industrial arms or grasp physics.

## Run a larger standalone example

From an installed source checkout:

```bash
python examples/many_arm_inspection.py --robots 8 --tasks 40 --topology paired --output /tmp/eight-tools
```

Each task has a configured approach, target dwell and return trace. The
producer tries a serial assignment and a balanced assignment, preserving every
target and each robot's home goal. The selected plan is independently checked
on CPU. `domain.json`, the selected `problem.json` and `result.json`, and the
attempt/cost sidecar `schedule.json` are saved together.

Two layouts answer different questions:

| Layout | Meaning |
| --- | --- |
| `paired` | Separated workcells, each with two tools sharing two target regions. Useful for checking parallel work and resource growth. It does not demonstrate coordination in one dense eight-arm cell. |
| `shared` | All tool homes surround one central target region. Irreversible configured skills can conflict, making assignment and start-time search harder. An alternative may time out while the first verified plan is retained. |

Use `--robots 2`, `4` or `8`, `--tasks`, `--seed`, and `--seconds` to select a
case and budget. The example's default layout is `shared`. A result describes
the finite configured traces and sampled checks; increasing robot count does
not expand the supported real-model or physical execution claims.

For standalone browser playback:

```bash
uv run --project environments/viz --frozen python examples/many_arm_inspection.py --robots 8 --tasks 10 --topology paired --output /tmp/eight-tools --view
```

For CUDA planning, use the [qualified environment](curobo.md) and pass
`--backend curobo`. The visualization environment stays separate; it can
[open the saved manipulation pair](manipulation.md#view-a-saved-manipulation-plan)
without CUDA. No Botrail, MuJoCo or Isaac Sim installation is required.

## Interpret costs and timings

Execution makespan measures the planned work's duration. Planning latency
measures the computation needed to find it. A shorter execution schedule does
not imply faster planning or GPU acceleration. Compare the complete outcome
and failure counts as well as costs on successful trials.

`SchedulePlanner.statistics` includes `first_verified_seconds`, recorded
immediately after independent acceptance verification, and `timing_seconds`
for generation, backend initialization, roadmaps, fixed-skeleton planning,
acceptance CPU verification, objective measurement, optional refinement,
callbacks and feedback. The fixed-skeleton phase already includes its own CPU
verification; the acceptance phase is the additional scheduler check. These
are named phases rather than overlapping estimates of kernel time.

First-verified time includes preceding failed attempts and initial setup/PRM
construction. The first measured cost callback follows objective evaluation.
Cold-process import, CUDA context or instrumentation setup must be reported
separately when excluded from the planner clock. Native calls, verification
and bounded metric evaluation may overrun cooperative time limits.

Use identical geometry, task catalog, sampled graphs, seeds, validation,
objective, limits and hardware when comparing implementations. Report warmup,
offline roadmap reuse, timing exclusions and thread settings. GPU memory
allocated/reserved by PyTorch and process peak RSS cover different scopes;
neither is total device-memory usage. Tiny proxy cases may spend more time on
CUDA setup and transfer than on geometry work.

## Relationship to published benchmarks

The synthetic example is separate from
[xECBS's original queries and published results](https://github.com/yoraish/xecbs_icaps24_experiments)
and the [MRMG assembly benchmark](https://github.com/vhartman/multirobot-pathplanning-benchmark).
Original model revisions, collision geometry/exclusions, task grounding,
objective, validation and timing must be qualified before a matched comparison.
Source FK agreement alone does not qualify collision geometry.

xECBS sums joint L1 path lengths over robots, excluding waits. MRMG uses a
synchronized multi-robot edge cost: the longest per-robot displacement plus
one percent of the sum of displacements. Its per-robot norm is part of the
source configuration; the current upstream RAI implementation selects
Euclidean distance. See the pinned upstream
[cost implementation](https://github.com/vhartman/multirobot-pathplanning-benchmark/blob/60bc37fef0980cf5948925daaaed6fca478d2bb4/src/multi_robot_multi_goal_planning/problems/core/configuration.py)
and [RAI configuration](https://github.com/vhartman/multirobot-pathplanning-benchmark/blob/60bc37fef0980cf5948925daaaed6fca478d2bb4/src/multi_robot_multi_goal_planning/problems/rai_base_env.py).
Neither makespan nor summed L1 can silently replace that synchronized cost.
The current [refinement objective](refinement.md) is a separate API.

KinoWeave does not currently claim to reproduce cuArms' published benchmark
performance or full task solver. Published baseline CSVs are reference data,
not evidence of a local baseline execution. Exact RoboBallet episode/target
data cannot be replaced by synthetic inspection tasks.
