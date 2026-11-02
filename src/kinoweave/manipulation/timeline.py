"""Evaluate original local time laws and both sides of manipulation events."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise

from .._contracts import ContractError, value_as
from ..trajectory import MotionState, evaluate_segment
from .models import JointTrace, ManipulationMode


def evaluate_trace(trace, time, *, side="after"):
    time = value_as(time, float, "time")
    if side not in ("before", "after") or not 0 <= time <= trace.duration:
        raise ContractError("trace.time", "invalid time or boundary side")
    index = (
        (bisect_left(trace.times, time) - 1)
        if side == "before"
        else (bisect_right(trace.times, time) - 1)
    )
    index = min(max(index, 0), len(trace.times) - 2)
    begin, end = trace.times[index : index + 2]
    elapsed = time - begin
    duration = end - begin
    result = {}
    for track in trace.tracks:
        first, last = track.positions[index : index + 2]
        if trace.interpolation == "quintic_stop":
            state = evaluate_segment(first, last, duration, elapsed)
        else:
            u = elapsed / duration
            state = MotionState(
                positions=tuple(a + (b - a) * u for a, b in zip(first, last, strict=True)),
                velocities=tuple((b - a) / duration for a, b in zip(first, last, strict=True)),
                accelerations=(0,) * len(first),
            )
        result[track.robot] = state
    return result


@dataclass(frozen=True, slots=True)
class Activity:
    name: str
    start: float
    end: float
    trace: JointTrace

    def clock(self, time):
        if time == self.start:
            return 0.0
        if time == self.end:
            return self.trace.duration
        return self.trace.duration * ((time - self.start) / (self.end - self.start))

    def absolute(self, time):
        value = (
            self.end
            if time == self.trace.duration
            else self.start + (self.end - self.start) * (time / self.trace.duration)
        )
        if not math.isfinite(value) or not self.start <= value <= self.end:
            raise ContractError("timeline", "source time cannot be represented in the schedule")
        if 0 < time < self.trace.duration and not self.start < value < self.end:
            raise ContractError("timeline", "interior source time collapsed onto an endpoint")
        return value


class _Timeline:
    def _init(self, problem, mode, activities, executions, makespan):
        self.problem, self.initial_mode, self.activities, self.makespan = (
            problem,
            mode,
            tuple(activities),
            makespan,
        )
        self.events = []
        self.windows = []
        rank = {name: i for i, name in enumerate(problem.skeleton.ancestors())}
        for skill, option, activity in executions:
            source_times = sorted(
                set(
                    (
                        *option.trace.times,
                        *(c.time for c in option.transitions),
                        *(t for w in option.contacts for t in (w.start, w.end)),
                    )
                )
            )
            absolute_times = tuple(activity.absolute(t) for t in source_times)
            if any(b <= a for a, b in pairwise(absolute_times)):
                raise ContractError(
                    "timeline", "distinct source boundaries collapsed in the schedule"
                )
            for i, change in enumerate(option.transitions):
                self.events.append((activity.absolute(change.time), rank[skill.name], i, change))
            for window in option.contacts:
                self.windows.append(
                    (
                        activity.absolute(window.start),
                        activity.absolute(window.end),
                        (window.first, window.second),
                    )
                )
        self.events.sort(key=lambda event: event[:3])
        self.by_robot = {
            r.name: sorted(
                (a for a in self.activities if any(t.robot == r.name for t in a.trace.tracks)),
                key=lambda a: a.start,
            )
            for r in problem.scene.robots
        }
        boundaries = {0.0, makespan}
        for activity in self.activities:
            times = tuple(activity.absolute(t) for t in activity.trace.times)
            if any(b <= a for a, b in pairwise(times)):
                raise ContractError("timeline", "distinct source knots collapsed in the schedule")
            boundaries.update(times)
        boundaries.update(event[0] for event in self.events)
        for begin, end, _pair in self.windows:
            boundaries.update((begin, end))
        self.boundaries = tuple(sorted(boundaries))

    def states(self, time, *, side="after"):
        time = value_as(time, float, "time")
        if side not in ("before", "after") or not 0 <= time <= self.makespan:
            raise ContractError("timeline.time", "invalid time or boundary side")
        result = {}
        for robot, activities in self.by_robot.items():
            choices = (
                [a for a in activities if a.start < time <= a.end]
                if side == "before"
                else [a for a in activities if a.start <= time < a.end]
            )
            if not choices:
                choices = (
                    activities[:1]
                    if time == 0
                    else activities[-1:]
                    if time == self.makespan
                    else []
                )
            if len(choices) != 1:
                raise ContractError("timeline", "missing/ambiguous robot activity")
            activity = choices[0]
            state = evaluate_trace(activity.trace, activity.clock(time), side=side)[robot]
            scale = activity.trace.duration / (activity.end - activity.start)
            result[robot] = MotionState(
                positions=state.positions,
                velocities=tuple(v * scale for v in state.velocities),
                accelerations=tuple(a * scale * scale for a in state.accelerations),
            )
        return result

    def mode(self, time, *, side="after"):
        time = value_as(time, float, "time")
        if side not in ("before", "after") or not 0 <= time <= self.makespan:
            raise ContractError("timeline.time", "invalid time or boundary side")
        states = {s.object: s for s in self.initial_mode.states}
        for at, _rank, _index, change in self.events:
            if at > time or (side == "before" and at == time):
                break
            if states.get(change.before.object) != change.before:
                raise ContractError(
                    "timeline.mode", "transition before state does not match current mode"
                )
            states[change.after.object] = change.after
        return ManipulationMode(name=f"at {time:g} {side}", states=tuple(states.values()))

    def contacts(self, time, *, side="after"):
        time = value_as(time, float, "time")
        if side not in ("before", "after") or not 0 <= time <= self.makespan:
            raise ContractError("timeline.time", "invalid time or boundary side")
        # At the plan endpoints there is only one in-domain neighborhood.
        if time == 0:
            side = "after"
        elif time == self.makespan:
            side = "before"
        return tuple(
            pair
            for start, end, pair in self.windows
            if (start < time <= end if side == "before" else start <= time < end)
        )

    def configurations(self, time, *, side="after"):
        return {name: state.positions for name, state in self.states(time, side=side).items()}


class ManipulationTimeline(_Timeline):
    """A structurally bound plan; geometry validation remains separate."""

    def __init__(self, problem, result):
        result.check_problem(problem)
        if result.plan is None:
            raise ContractError("timeline", "a timed candidate or solution is required")
        definitions = {s.name: s for s in problem.skeleton.skills}
        activities = [
            Activity(f"motion[{i}]", m.start, m.end, m.trace)
            for i, m in enumerate(result.plan.motions)
        ]
        executions = []
        for scheduled in result.plan.skills:
            skill = definitions[scheduled.skill]
            option = next(o for o in skill.options if o.name == scheduled.option)
            activity = Activity(skill.name, scheduled.start, scheduled.end, option.trace)
            activities.append(activity)
            executions.append((skill, option, activity))
        self._init(problem, problem.initial_mode, activities, executions, result.plan.makespan)


def skill_timeline(problem, mode, skill, option, duration):
    """Internal full-configuration skill check for the single-arm constructor."""
    if set(skill.robots) != {r.name for r in problem.scene.robots}:
        raise ContractError("skill", "standalone verification needs every robot configuration")
    if duration < skill.minimum_duration or duration <= 0:
        raise ContractError("skill.duration", "invalid duration")
    current = {s.object: s for s in mode.states}
    if any(current.get(s.object) != s for s in option.preconditions):
        raise ContractError("skill", "precondition mismatch")
    result = _Timeline()
    activity = Activity(skill.name, 0, duration, option.trace)
    result._init(problem, mode, [activity], [(skill, option, activity)], duration)
    result.states(0)
    return result
