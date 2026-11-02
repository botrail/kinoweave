"""Stopped-knot free-motion edits and globally shared idle-time removal."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations, pairwise

from .._contracts import ContractError
from ..trajectory import RobotTrajectory, minimum_duration
from .models import JointTrace
from .results import ScheduledMotion
from .timeline import Activity, evaluate_trace


@dataclass(frozen=True, slots=True)
class Shortcut:
    robot: str
    start: float
    end: float
    first: tuple[float, ...]
    last: tuple[float, ...]
    duration: float
    delay: float = 0.0


@dataclass(frozen=True, slots=True)
class Proposal:
    edits: tuple[Shortcut, ...]
    kind: str
    cut: tuple[float, float] | None = None


def compatible(edits):
    return all(
        a.robot != b.robot or a.end <= b.start or b.end <= a.start
        for a, b in combinations(edits, 2)
    )


def _source(activity, time):
    for t in activity.trace.times:
        if activity.absolute(t) == time:
            return t
    return activity.clock(time)


def _clip(motion, begin, end):
    if begin == motion.start and end == motion.end:
        return motion
    activity = Activity("clip", motion.start, motion.end, motion.trace)
    a, b = _source(activity, begin), _source(activity, end)
    times = (a, *(t for t in motion.trace.times if a < t < b), b)
    track = motion.trace.tracks[0]
    trace = JointTrace(
        times=tuple(t - a for t in times),
        interpolation=motion.trace.interpolation,
        tracks=(
            replace(
                track,
                positions=tuple(
                    evaluate_trace(motion.trace, t)[track.robot].positions for t in times
                ),
            ),
        ),
    )
    return replace(motion, start=begin, end=end, trace=trace)


def at_rest(timeline, robot, time):
    for activity in timeline.by_robot[robot]:
        if not activity.start <= time <= activity.end:
            continue
        trace = activity.trace
        source = _source(activity, time)
        if trace.interpolation == "quintic_stop" and source in trace.times:
            continue
        track = next(t for t in trace.tracks if t.robot == robot)
        if any(
            a <= source <= b and track.positions[i] != track.positions[i + 1]
            for i, (a, b) in enumerate(pairwise(trace.times))
        ):
            return False
    return True


def _runs(plan, robot):
    runs = []
    for motion in sorted(
        (m for m in plan.motions if m.trace.tracks[0].robot == robot), key=lambda m: m.start
    ):
        if (
            runs
            and runs[-1][-1].end == motion.start
            and runs[-1][-1].trace.tracks[0].positions[-1] == motion.trace.tracks[0].positions[0]
        ):
            runs[-1].append(motion)
        else:
            runs.append([motion])
    for run in runs:
        values = {}
        for motion in run:
            activity = Activity("run", motion.start, motion.end, motion.trace)
            for time, q in zip(motion.trace.times, motion.trace.tracks[0].positions, strict=True):
                values[activity.absolute(time)] = q
        yield tuple(sorted(values.items()))


def _windows(points, maximum):
    count = 0
    for width in range(len(points) - 1, 0, -1):
        for i in range(len(points) - width):
            yield points[i], points[i + width]
            count += 1
            if count >= maximum:
                return


def _move(problem, edit):
    robot = next(r for r in problem.scene.robots if r.name == edit.robot)
    model = next(m for m in problem.scene.models if m.name == robot.model)
    result = []
    moving_start = edit.start + edit.delay
    moving_end = (
        edit.end
        if edit.delay + edit.duration == edit.end - edit.start
        else moving_start + edit.duration
    )
    if not edit.start <= moving_start <= moving_end <= edit.end:
        raise ContractError("shortcut", "unrepresentable edit times")
    for a, b, first, last in (
        (edit.start, moving_start, edit.first, edit.first),
        (moving_start, moving_end, edit.first, edit.last),
        (moving_end, edit.end, edit.last, edit.last),
    ):
        if a == b:
            continue
        trace = JointTrace(
            times=(0, b - a),
            interpolation="quintic_stop",
            tracks=(
                RobotTrajectory(
                    robot=edit.robot, joint_names=model.joint_order, positions=(first, last)
                ),
            ),
        )
        result.append(
            ScheduledMotion(start=a, end=b, trace=trace, kind="wait" if first == last else "move")
        )
    return result


def apply_proposal(problem, plan, proposal):
    if not compatible(proposal.edits):
        raise ContractError("shortcut", "overlapping same-robot windows")
    for edit in proposal.edits:
        if any(
            edit.robot
            in next(s.robots for s in problem.skeleton.skills if s.name == scheduled.skill)
            and scheduled.start < edit.end
            and scheduled.end > edit.start
            for scheduled in plan.skills
        ):
            raise ContractError("shortcut", "cannot edit skill executions")
    motions = []
    for motion in plan.motions:
        spans = [(motion.start, motion.end)]
        robot = motion.trace.tracks[0].robot
        for edit in proposal.edits:
            if edit.robot != robot:
                continue
            remaining = []
            for a, b in spans:
                if edit.start >= b or edit.end <= a:
                    remaining.append((a, b))
                else:
                    if a < edit.start:
                        remaining.append((a, edit.start))
                    if edit.end < b:
                        remaining.append((edit.end, b))
            spans = remaining
        motions.extend(_clip(motion, a, b) for a, b in spans)
    for edit in proposal.edits:
        motions.extend(_move(problem, edit))
    result = replace(
        plan,
        motions=tuple(sorted(motions, key=lambda m: (m.start, m.end, m.trace.tracks[0].robot))),
    )
    if proposal.cut is not None:
        result = _cut(problem, result, *proposal.cut)
    return result


def _cut(problem, plan, begin, end):
    if not 0 <= begin < end <= plan.makespan:
        raise ContractError("shortcut.cut", "invalid cut")
    if any(s.start < end and s.end > begin for s in plan.skills):
        raise ContractError("shortcut.cut", "cannot delete skill time")
    motions = []
    for motion in plan.motions:
        a, b = motion.start, motion.end
        if a < end and b > begin:
            if motion.trace.tracks[0].positions.count(motion.trace.tracks[0].positions[0]) != len(
                motion.trace.times
            ):
                raise ContractError("shortcut.cut", "deleted time must be stationary")
            if a < begin:
                motions.append(_clip(motion, a, begin))
            if end < b:
                motions.append(_clip(motion, end, b))
        else:
            motions.append(motion)
    delta = end - begin
    old_times = {
        0.0,
        plan.makespan,
        *[t for m in motions for t in (m.start, m.end)],
        *[t for s in plan.skills for t in (s.start, s.end)],
    }
    definitions = {s.name: s for s in problem.skeleton.skills}
    mapped = {}
    correction = 0.0
    previous = None
    for old in sorted(old_times):
        new = old if old <= begin else old - delta + correction
        if previous is not None and new <= previous and old != end:
            new = math.nextafter(previous, math.inf)
        for skill in plan.skills:
            if skill.end == old:
                minimum = definitions[skill.skill].minimum_duration
                start = mapped[skill.start]
                while new - start < minimum:
                    new = math.nextafter(new, math.inf)
        if not math.isfinite(new):
            raise ContractError("shortcut.cut", "unrepresentable translated time")
        if old > begin:
            correction = new - (old - delta)
        mapped[old] = new
        previous = new
    return replace(
        plan,
        makespan=mapped[plan.makespan],
        motions=tuple(replace(m, start=mapped[m.start], end=mapped[m.end]) for m in motions),
        skills=tuple(replace(s, start=mapped[s.start], end=mapped[s.end]) for s in plan.skills),
    )


def trim_tail(plan, time):
    if time >= plan.makespan:
        return plan
    motions = tuple(_clip(m, m.start, min(m.end, time)) for m in plan.motions if m.start < time)
    return replace(plan, makespan=time, motions=motions)


def proposals(problem, result, timeline, config, rng):
    models = {m.name: m for m in problem.scene.models}
    robots = problem.scene.robots
    pools = {}
    for robot in robots:
        model = models[robot.model]
        pool = []
        remaining = config.windows_per_robot
        runs = list(_runs(result.plan, robot.name))
        # One wide window from each run, then smaller alternatives round-robin.
        generators = [iter(_windows(run, config.windows_per_robot)) for run in runs]
        while remaining and generators:
            following = []
            for generator in generators:
                try:
                    (a, qa), (b, qb) = next(generator)
                except StopIteration:
                    continue
                following.append(generator)
                if not at_rest(timeline, robot.name, a) or not at_rest(timeline, robot.name, b):
                    continue
                minimum = minimum_duration(model, qa, qb)
                fast = (
                    max(minimum, config.minimum_duration) * config.duration_safety
                    if qa != qb
                    else 0.0
                )
                for duration in dict.fromkeys((min(b - a, fast), b - a)):
                    pool.append(Shortcut(robot.name, a, b, qa, qb, duration))
                remaining -= 1
                if not remaining:
                    break
            generators = following
        pools[robot.name] = pool
    result_proposals = []
    seen = set()

    def add(proposal):
        if proposal not in seen:
            seen.add(proposal)
            result_proposals.append(proposal)

    # Shared free phases permit clock compression without changing skill traces.
    cursor = 0.0
    blocks = []
    for skill in sorted(result.plan.skills, key=lambda s: s.start):
        if skill.start > cursor:
            blocks.append((cursor, skill.start))
        cursor = max(cursor, skill.end)
    if cursor < result.makespan:
        blocks.append((cursor, result.makespan))
    for a, b in blocks:
        if not all(at_rest(timeline, r.name, a) and at_rest(timeline, r.name, b) for r in robots):
            continue
        first, last = timeline.configurations(a), timeline.configurations(b)
        durations = {
            r.name: (
                max(
                    minimum_duration(models[r.model], first[r.name], last[r.name]),
                    config.minimum_duration,
                )
                * config.duration_safety
                if first[r.name] != last[r.name]
                else 0.0
            )
            for r in robots
        }
        parallel = max(durations.values())
        if parallel < b - a:
            edits = tuple(
                Shortcut(r.name, a, b, first[r.name], last[r.name], parallel) for r in robots
            )
            add(Proposal(edits, "compression", (a + parallel, b)))
        serial = math.fsum(durations.values())
        if serial < b - a:
            for order in (robots, robots[::-1]):
                elapsed = 0.0
                edits = []
                for r in order:
                    edits.append(
                        Shortcut(
                            r.name, a, b, first[r.name], last[r.name], durations[r.name], elapsed
                        )
                    )
                    elapsed += durations[r.name]
                add(Proposal(tuple(edits), "prioritized", (a + serial, b)))
    # Generate composites from raw edits, even if a single edit collides or
    # leaves makespan unchanged. Joint improvements need not work individually.
    names = [name for name, pool in pools.items() if pool]
    for a, b in combinations(names, 2):
        for first in pools[a]:
            second = next(
                (
                    e
                    for e in pools[b]
                    if e.start == first.start
                    and e.end == first.end
                    and (e.duration == e.end - e.start)
                    == (first.duration == first.end - first.start)
                ),
                None,
            )
            if second is not None:
                add(Proposal((first, second), "composite"))
                if len(result_proposals) >= config.max_candidates // 2:
                    break
    for _ in range(config.max_candidates * 2):
        if len(names) < 2 or len(result_proposals) >= config.max_candidates // 2:
            break
        size = rng.randint(2, min(len(names), config.max_composite_arms))
        chosen = rng.sample(names, size)
        add(Proposal(tuple(rng.choice(pools[n]) for n in chosen), "composite"))
    for pool in zip(*[pools[name] for name in names], strict=False) if names else ():
        for edit in pool:
            add(Proposal((edit,), "single"))
    # zip stops at the shortest pool; include remaining proposals without
    # materializing an unbounded Cartesian product.
    for name in names:
        for edit in pools[name]:
            add(Proposal((edit,), "single"))
    return tuple(result_proposals[: config.max_candidates])
