"""Counterfactual and geometric witnesses for the eight-arm example; SDK-free."""

import math
from dataclasses import asdict
from itertools import combinations

from kinoweave import PlanProvenance
from kinoweave.io import document_digest
from kinoweave.manipulation import (
    ManipulationPlan,
    ManipulationResult,
    ManipulationTimeline,
    ModeGeometry,
    ScheduledSkill,
    validate_manipulation,
)
from kinoweave.manipulation.roadmap import fingerprint


def independent_result(problem, result=None):
    """ASAP schedule of the same closed skill traces, ignoring robot/robot geometry.

    This is deliberately a candidate, never a successful/verified result.
    Only the demo's equal-duration, one-arm, home-to-home skills are supported.
    """
    definitions = {s.name: s for s in problem.skeleton.skills}
    selected = {s.skill: s.option for s in result.plan.skills} if result else {}
    ends = {r.name: 0.0 for r in problem.scene.robots}
    starts = {r.name: r.start for r in problem.scene.robots}
    scheduled = {}
    for name, ancestors in problem.skeleton.ancestors().items():
        skill = definitions[name]
        if len(skill.robots) != 1 or len(skill.options) != 1:
            raise ValueError("comparison requires one-arm skills with one source trace")
        robot = skill.robots[0]
        option = next(
            o for o in skill.options if o.name == selected.get(name, skill.options[0].name)
        )
        track = option.trace.tracks[0]
        if (
            len(option.trace.tracks) != 1
            or track.robot != robot
            or track.positions[0] != starts[robot]
            or track.positions[-1] != starts[robot]
            or skill.minimum_duration != option.trace.duration
        ):
            raise ValueError("comparison requires unchanged home-to-home source clocks")
        start = max(ends[robot], *(scheduled[a].end for a in ancestors), 0.0)
        ends[robot] = start + option.trace.duration
        scheduled[name] = ScheduledSkill(
            skill=name, option=option.name, start=start, end=ends[robot]
        )
    if len(set(ends.values())) != 1:
        raise ValueError("comparison requires equal per-arm workload durations")
    candidate = ManipulationResult(
        status="candidate",
        provenance=PlanProvenance(
            problem_digest=document_digest(problem), config=problem.scene.config
        ),
        plan=ManipulationPlan(
            makespan=max(ends.values()),
            motions=(),
            skills=tuple(scheduled.values()),
            final_mode=problem.goal_mode,
        ),
    )
    candidate.check_problem(problem)
    return candidate


def closest_pair(first, second):
    """Exact minimum native sphere clearance between two distinct robot frames."""
    clearance, a, b = min(
        (math.dist(a.center, b.center) - a.radius - b.radius, a.index, b.index)
        for a in first.spheres
        for b in second.spheres
    )
    return clearance, first.spheres[a], second.spheres[b]


def coordination_report(problem, result, metadata):
    """Require an invalid ASAP baseline, connected eight-arm conflicts, and yielding.

    Baseline witnesses are sampled at declared weld opening/midpoint/closing.
    They prove conflicts exist, not an exhaustive set or first collision time.
    The planned result's independent full CPU proof is performed by the caller.
    """
    result.check_problem(problem)
    baseline = independent_result(problem, result)
    timeline, geometry = ManipulationTimeline(problem, baseline), ModeGeometry(problem)
    schedule = {s.skill: s for s in baseline.plan.skills}
    definitions = {s.name: s for s in problem.skeleton.skills}
    times = set()
    for target in metadata["targets"]:
        execution = schedule[target["skill"]]
        trace = definitions[target["skill"]].options[0].trace
        begin, end = target["weld_source"]
        for source in (begin, (begin + end) / 2, end):
            times.add(execution.start + source / trace.duration * (execution.end - execution.start))
    witnesses = {}
    threshold = geometry.threshold
    for time in sorted(times):
        frames = {r: geometry.fk(r, q) for r, q in timeline.configurations(time).items()}
        for first, second in combinations(sorted(frames), 2):
            clearance, a, b = closest_pair(frames[first], frames[second])
            if clearance <= threshold and (
                (first, second) not in witnesses
                or clearance < witnesses[first, second]["clearance_m"]
            ):
                witnesses[first, second] = {
                    "robots": [first, second],
                    "time": time,
                    "clearance_m": clearance,
                    "first": asdict(a),
                    "second": asdict(b),
                }
    names = {r.name for r in problem.scene.robots}
    reached = {min(names)}
    while True:
        expanded = reached | {r for pair in witnesses if reached.intersection(pair) for r in pair}
        if expanded == reached:
            break
        reached = expanded
    if reached != names:
        raise ValueError("baseline conflict graph must connect all eight arms")
    validation = validate_manipulation(problem, baseline)
    if validation.status != "failed" or not any(i.code == "collision" for i in validation.issues):
        raise ValueError("the uncoordinated baseline must fail collision verification")
    independent = {s.skill: s for s in baseline.plan.skills}
    delays = [
        {"skill": s.skill, "delay_s": s.start - independent[s.skill].start}
        for s in result.plan.skills
        if s.start > independent[s.skill].start
    ]
    if not delays:
        raise ValueError("coordinated execution must differ from the colliding baseline")
    waiting = {r: 0.0 for r in names}
    for robot in names:
        end = 0.0
        for skill in sorted(result.plan.skills, key=lambda s: s.start):
            if definitions[skill.skill].robots == (robot,):
                waiting[robot] += skill.start - end
                end = skill.end
    return {
        "problem_digest": document_digest(problem),
        "plan_digest": fingerprint(result.plan),
        "baseline_status": validation.status,
        "baseline_validation": asdict(validation),
        "baseline_duration_s": baseline.makespan,
        "planned_duration_s": result.makespan,
        "witness_sampling": "declared weld opening/midpoint/closing; not exhaustive",
        "conflict_graph_connected": True,
        "conflicting_pairs": [list(pair) for pair in sorted(witnesses)],
        "witnesses": [witnesses[pair] for pair in sorted(witnesses)],
        "delayed_starts": delays,
        "pending_wait_s_by_robot": dict(sorted(waiting.items())),
    }
