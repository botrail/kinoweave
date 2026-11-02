"""Finite task-progress states and source-clock candidate steps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from .._contracts import Contract, ContractError
from ..trajectory import evaluate_segment
from .models import HeldObject, ManipulationMode
from .timeline import evaluate_trace


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationSearchConfig(Contract):
    algorithm: Literal["lacam", "exhaustive"] = "lacam"
    max_states: int = 10_000
    max_constraints: int = 100_000
    max_candidates_per_robot: int = 128
    max_skill_ticks: int = 1000
    max_state_pairs: int = 65_536
    max_query_pairs: int = 256
    max_state_bytes: int = 64 * 1024 * 1024
    max_cached_tables: int = 8
    max_cached_table_pairs: int = 131_072
    pair_batch_size: int = 32
    sphere_tile: int = 16
    max_pair_bytes: int = 4 * 1024 * 1024
    max_validation_samples: int = 100_000

    def _validate(self):
        for name in (
            "max_states",
            "max_constraints",
            "max_candidates_per_robot",
            "max_skill_ticks",
            "max_state_pairs",
            "max_query_pairs",
            "max_state_bytes",
            "pair_batch_size",
            "sphere_tile",
            "max_pair_bytes",
        ):
            self._require(getattr(self, name) > 0, name, "must be positive")
        for name in ("max_cached_tables", "max_cached_table_pairs"):
            self._require(getattr(self, name) >= 0, name, "must be non-negative")
        self._require(
            self.max_validation_samples >= 2, "max_validation_samples", "need two samples"
        )
        self._require(self.max_pair_bytes >= 128, "max_pair_bytes", "must fit one pair")


@dataclass(frozen=True, slots=True)
class RunningSkill:
    skill: int
    option: int
    phase: int


@dataclass(frozen=True, slots=True)
class TaskState:
    vertices: tuple[int, ...]
    running: tuple[RunningSkill | None, ...]
    completed: frozenset[int]
    mode: ManipulationMode


def canonical_mode(problem, states):
    return ManipulationMode(name="search", states=tuple(states[o.name] for o in problem.objects))


def skill_clock(skill, option, dt, maximum):
    duration = max(option.trace.duration, skill.minimum_duration)
    ratio = duration / dt
    if not math.isfinite(ratio) or ratio > maximum:
        from ..search.space import SearchLimit

        raise SearchLimit("skill_tick_limit", "skill_preparation")
    ticks = max(1, math.ceil(ratio))
    if skill.timing == "stretch":
        duration = ticks * dt
    if not math.isfinite(duration):
        raise ContractError("skill.duration", "unrepresentable duration")
    return duration, ticks


@dataclass(frozen=True, slots=True)
class TaskAction:
    """One indivisible choice, possibly shared by both participants of a skill."""

    robots: tuple[int, ...]
    targets: tuple[int, ...]
    kind: Literal["move", "wait", "skill"]
    skill: int = -1
    option: int = -1
    phase: int = 0


class ActionClock:
    def __init__(self, space, state, action):
        self.space, self.state, self.action = space, state, action
        self.problem, self.dt = space.problem, space.dt
        self.definition = self.option = None
        self.duration, self.ticks = self.dt, 1
        self.offset = 0.0
        self.end_offset = self.dt
        self.events = []
        boundaries = {0.0, self.dt}
        if action.kind == "skill":
            self.definition = space.skills[action.skill]
            self.option = self.definition.options[action.option]
            self.duration, self.ticks = space.clocks[action.skill, action.option]
            self.offset = action.phase * self.dt
            self.end_offset = (action.phase + 1) * self.dt
            source = self.option.trace.duration
            source_times = sorted(
                set(
                    (
                        *self.option.trace.times,
                        *(e.time for e in self.option.transitions),
                        *(t for w in self.option.contacts for t in (w.start, w.end)),
                    )
                )
            )
            scaled = tuple(t / source * self.duration for t in source_times)
            if any(b <= a for a, b in pairwise(scaled)):
                raise ContractError("skill.clock", "distinct source boundaries collapsed")
            for value in scaled:
                if self.offset <= value <= self.end_offset:
                    relative = self.dt if value == self.end_offset else value - self.offset
                    if not 0 <= relative <= self.dt:
                        raise ContractError("skill.clock", "unrepresentable phase boundary")
                    boundaries.add(relative)
            for i, event in enumerate(self.option.transitions):
                at = event.time / source * self.duration
                if (at > self.offset or (action.phase == 0 and at == 0)) and at <= self.end_offset:
                    relative = self.dt if at == self.end_offset else at - self.offset
                    self.events.append((relative, action.skill, i, event))
        self.boundaries = tuple(sorted(boundaries))

    def configurations(self, time, side="after"):
        action, space = self.action, self.space
        if action.kind != "skill":
            return {
                space.names[r]: evaluate_segment(
                    space.graphs[r].vertices[self.state.vertices[r]],
                    space.graphs[r].vertices[target],
                    self.dt,
                    time,
                ).positions
                for r, target in zip(action.robots, action.targets, strict=True)
            }
        elapsed = min(self.duration, self.end_offset if time == self.dt else self.offset + time)
        source = (
            self.option.trace.duration
            if elapsed == self.duration
            else elapsed / self.duration * self.option.trace.duration
        )
        return {
            name: state.positions
            for name, state in evaluate_trace(self.option.trace, source, side=side).items()
        }

    def mode(self, time, side="after"):
        states = {s.object: s for s in self.state.mode.states}
        for at, _rank, _index, change in self.events:
            if at > time or (side == "before" and at == time):
                break
            if states[change.before.object] != change.before:
                raise ContractError("skill.mode", "effect precondition mismatch")
            states[change.after.object] = change.after
        return canonical_mode(self.problem, states)

    def contacts(self, time, side="after"):
        if self.option is None:
            return ()
        elapsed = self.end_offset if time == self.dt else self.offset + time
        if elapsed > self.duration:
            return ()
        # Closing permissions cover the current step's endpoint. The next
        # step rechecks its opening side using its own selected permissions.
        if time == self.dt:
            side = "before"
        source = (
            self.option.trace.duration
            if elapsed == self.duration
            else elapsed / self.duration * self.option.trace.duration
        )
        return tuple(
            (w.first, w.second)
            for w in self.option.contacts
            if (w.start < source <= w.end if side == "before" else w.start <= source < w.end)
        )


def object_owners(space, state):
    """Assign every object once; a changing object follows its deciding bundle."""
    owners = {}
    for obj in state.mode.states:
        if isinstance(obj, HeldObject):
            owners[obj.object] = space.names.index(obj.grasps[0].robot)
        else:
            pending = [
                i
                for i in space.topological
                if i not in state.completed and obj.object in space.skills[i].objects
            ]
            owners[obj.object] = (
                space.names.index(space.skills[pending[0]].robots[0]) if pending else None
            )
    return owners
