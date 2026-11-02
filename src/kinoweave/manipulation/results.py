"""Timed manipulation candidates and structural binding, separate from verification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .._contracts import Contract
from ..models import Conventions, Diagnostic, PlanProvenance, ValidationReport
from ..results import PlanStatus
from .models import JointTrace, ManipulationMode, ManipulationProblem, same_duration

MANIPULATION_CHECKS = frozenset(
    (
        "joint_limits",
        "collisions",
        "endpoints",
        "precedence",
        "mode_transitions",
        "skill_traces",
        "contacts",
        "object_goals",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduledMotion(Contract):
    start: float
    end: float
    trace: JointTrace
    kind: Literal["move", "wait"] = "move"

    def _validate(self):
        self._require(0 <= self.start < self.end, "end", "expected 0 <= start < end")
        self._require(len(self.trace.tracks) == 1, "trace", "a connector has exactly one robot")
        self._require(
            same_duration(self.end - self.start, self.trace.duration),
            "trace.times",
            "connector timestamps must preserve the source trace duration",
        )
        if self.kind == "wait":
            rows = self.trace.tracks[0].positions
            self._require(all(row == rows[0] for row in rows), "trace", "waits cannot move")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduledSkill(Contract):
    skill: str
    option: str
    start: float
    end: float

    def _validate(self):
        self._require(0 <= self.start < self.end, "end", "expected 0 <= start < end")


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationPlan(Contract):
    """Independent local traces avoid changing a skill when another robot has a knot."""

    makespan: float
    motions: tuple[ScheduledMotion, ...]
    skills: tuple[ScheduledSkill, ...]
    final_mode: ManipulationMode

    def _validate(self):
        self._require(self.makespan > 0, "makespan", "must be positive")
        self._require(bool(self.skills), "skills", "at least one scheduled skill is required")
        self._unique(tuple(s.skill for s in self.skills), "skills")
        self._require(
            all(s.end <= self.makespan for s in (*self.motions, *self.skills)),
            "makespan",
            "an activity exceeds the plan duration",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationResult(Contract):
    status: PlanStatus | Literal["candidate"] = "not_run"
    plan: ManipulationPlan | None = None
    provenance: PlanProvenance | None = None
    validation: ValidationReport = field(default_factory=ValidationReport)
    diagnostics: tuple[Diagnostic, ...] = ()
    termination_reason: str | None = None
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.manipulation.result"] = "kinoweave.manipulation.result"
    schema_version: Literal[1] = 1

    def _validate(self):
        if self.status in ("candidate", "success"):
            self._require(self.plan is not None, "plan", "a timed plan is required")
            self._require(self.provenance is not None, "provenance", "input identity is required")
        else:
            self._require(self.plan is None, "plan", "only candidate or success carries a plan")
        if self.status == "success":
            self._require(
                self.provenance is not None and self.provenance.backend is not None,
                "provenance.backend",
                "success must identify the executed backend",
            )
            self._require(
                self.validation.status == "passed"
                and MANIPULATION_CHECKS <= set(self.validation.checks),
                "validation",
                "success requires complete manipulation validation evidence",
            )
        else:
            self._require(self.validation.status != "passed", "validation", "no validated solution")
        if self.status in ("not_run", "candidate"):
            self._require(
                self.validation.status == "not_run", "validation", "validation has not run"
            )
        if self.status == "not_run":
            self._require(
                self.provenance is None or self.provenance.backend is None,
                "provenance.backend",
                "not_run cannot claim backend execution",
            )
        if self.status == "validation_failed":
            self._require(
                self.validation.status == "failed", "validation", "failure evidence needed"
            )

    @property
    def makespan(self):
        return self.plan.makespan if self.plan is not None else None

    def check_problem(self, problem: ManipulationProblem):
        """Check identity, schedule, trace boundaries, and symbolic effects only.

        No collision, grasp geometry, support, swept volume, derivative, or
        tracking checks are performed. A bound candidate remains unverified.
        """
        from ..io import document_digest

        self._require(
            type(problem) is ManipulationProblem, "problem", "expected ManipulationProblem"
        )
        self._require(self.provenance is not None, "provenance", "input identity is required")
        assert self.provenance is not None
        self._require(
            self.provenance.problem_digest == document_digest(problem),
            "provenance.problem_digest",
            "result belongs to a different problem",
        )
        self._require(
            self.provenance.config == problem.scene.config, "provenance.config", "config mismatch"
        )
        if self.plan is None:
            return
        plan = self.plan
        selected = {s.skill: s for s in plan.skills}
        definitions = {s.name: s for s in problem.skeleton.skills}
        self._require(
            selected.keys() == definitions.keys(), "plan.skills", "skill coverage mismatch"
        )
        for order in problem.skeleton.precedence:
            self._require(
                selected[order.before].end <= selected[order.after].start,
                "plan.skills",
                "schedule violates precedence",
            )
        intervals = {robot.name: [] for robot in problem.scene.robots}
        for motion in plan.motions:
            problem.check_trace(motion.trace)
            track = motion.trace.tracks[0]
            intervals[track.robot].append((motion.start, motion.end, track.positions))
        events = []
        rank = {name: i for i, name in enumerate(problem.skeleton.ancestors())}
        for name, scheduled in selected.items():
            skill = definitions[name]
            options = {o.name: o for o in skill.options}
            self._require(scheduled.option in options, "plan.skills.option", "unknown option")
            option = options[scheduled.option]
            duration = scheduled.end - scheduled.start
            self._require(
                duration >= skill.minimum_duration,
                "plan.skills",
                "skill is shorter than its minimum duration",
            )
            if skill.timing == "fixed":
                self._require(
                    same_duration(duration, option.trace.duration),
                    "plan.skills",
                    "fixed skill timing must be preserved",
                )
            else:
                self._require(
                    duration >= option.trace.duration
                    or same_duration(duration, option.trace.duration),
                    "plan.skills",
                    "a stretch skill cannot be sped up",
                )
            for track in option.trace.tracks:
                intervals[track.robot].append((scheduled.start, scheduled.end, track.positions))
            # Dependency order disambiguates a predecessor's final effect and a
            # successor's initial precondition at the same exact timestamp.
            events.append((scheduled.start, rank[name], 0, option.preconditions))
            for i, change in enumerate(option.transitions, 1):
                time = (
                    scheduled.end
                    if change.time == option.trace.duration
                    else scheduled.start + duration * (change.time / option.trace.duration)
                )
                self._require(
                    math.isfinite(time) and scheduled.start <= time <= scheduled.end,
                    "plan.skills",
                    "transition time is not representable within the scheduled interval",
                )
                events.append((time, rank[name], i, change))
        tolerance = problem.scene.config.joint_tolerance

        def matching(first, second):
            return all(abs(a - b) <= tolerance for a, b in zip(first, second, strict=True))

        for robot in problem.scene.robots:
            segments = sorted(intervals[robot.name], key=lambda item: (item[0], item[1]))
            self._require(bool(segments), "plan", f"robot {robot.name!r} has no activity")
            self._require(
                segments[0][0] == 0 and segments[-1][1] == plan.makespan,
                "plan",
                "each robot needs explicit activity/holds over the entire plan",
            )
            self._require(
                matching(segments[0][2][0], robot.start)
                and matching(segments[-1][2][-1], robot.goal),
                "plan",
                "robot start/goal mismatch",
            )
            for first, second in zip(segments, segments[1:], strict=False):
                self._require(
                    first[1] == second[0],
                    "plan",
                    "robot activities overlap or leave an implicit gap",
                )
                self._require(
                    matching(first[2][-1], second[2][0]), "plan", "discontinuous activity boundary"
                )
        states = {s.object: s for s in problem.initial_mode.states}
        for _time, _rank, index, event in sorted(events, key=lambda item: item[:3]):
            if index == 0:
                self._require(
                    all(states.get(s.object) == s for s in event),
                    "plan.skills",
                    "skill precondition does not match the current object state",
                )
            else:
                self._require(
                    states.get(event.before.object) == event.before,
                    "plan.skills",
                    "transition before state does not match the current mode",
                )
                states[event.after.object] = event.after
        self._require(
            {s.object: s for s in plan.final_mode.states} == states,
            "plan.final_mode",
            "final mode does not match scheduled effects",
        )
        self._require(
            states == {s.object: s for s in problem.goal_mode.states},
            "plan.final_mode",
            "object goal mode was not reached",
        )
