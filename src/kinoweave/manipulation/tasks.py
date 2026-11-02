"""Finite, explicitly grounded task alternatives, independent of a scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .._contracts import Contract
from ..models import Conventions, PlanningProblem
from .models import (
    ActionOrder,
    ManipulationCost,
    ManipulationMode,
    ManipulationProblem,
    RigidObject,
    ScheduleSkeleton,
    Skill,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskAlternative(Contract):
    name: str
    skills: tuple[Skill, ...]
    precedence: tuple[ActionOrder, ...] = ()
    parameters: tuple[tuple[str, str], ...] = ()

    def _validate(self):
        ScheduleSkeleton(skills=self.skills, precedence=self.precedence)
        self._unique(tuple(k for k, _ in self.parameters), "parameters")


@dataclass(frozen=True, slots=True, kw_only=True)
class Task(Contract):
    name: str
    alternatives: tuple[TaskAlternative, ...]

    def _validate(self):
        self._require(
            bool(self.alternatives), "alternatives", "at least one grounded alternative required"
        )
        self._unique(tuple(a.name for a in self.alternatives), "alternatives")


def task_ancestors(tasks, precedence):
    pending = {t.name: set() for t in tasks}
    for edge in precedence:
        if edge.before not in pending or edge.after not in pending:
            from .._contracts import ContractError

            raise ContractError("tasks.precedence", "unknown task")
        pending[edge.after].add(edge.before)
    result = {}
    while pending:
        ready = sorted(n for n, p in pending.items() if p <= result.keys())
        if not ready:
            from .._contracts import ContractError

            raise ContractError("tasks.precedence", "cycle in task dependencies")
        for name in ready:
            parents = pending.pop(name)
            result[name] = frozenset(parents.union(*(result[p] for p in parents)))
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDomain(Contract):
    name: str
    scene: PlanningProblem
    objects: tuple[RigidObject, ...]
    initial_mode: ManipulationMode
    goal_mode: ManipulationMode
    tasks: tuple[Task, ...]
    precedence: tuple[ActionOrder, ...] = ()
    cost: ManipulationCost = field(default_factory=ManipulationCost)
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.manipulation.tasks"] = "kinoweave.manipulation.tasks"
    schema_version: Literal[1] = 1

    def _validate(self):
        self._require(bool(self.tasks), "tasks", "at least one task required")
        self._unique(tuple(t.name for t in self.tasks), "tasks")
        self._unique(tuple((p.before, p.after) for p in self.precedence), "precedence")
        task_ancestors(self.tasks, self.precedence)
        for task in self.tasks:
            for alternative in task.alternatives:
                self.problem(
                    ScheduleSkeleton(skills=alternative.skills, precedence=alternative.precedence)
                )

    def problem(self, skeleton):
        """Bind a concrete skeleton to the domain's unchanged scene and goals."""
        return ManipulationProblem(
            scene=self.scene,
            objects=self.objects,
            initial_mode=self.initial_mode,
            goal_mode=self.goal_mode,
            skeleton=skeleton,
            cost=self.cost,
            conventions=self.conventions,
        )

    def anchors(self):
        values = {r.name: [] for r in self.scene.robots}
        for task in self.tasks:
            for alternative in task.alternatives:
                for skill in alternative.skills:
                    for option in skill.options:
                        for track in option.trace.tracks:
                            values[track.robot].extend((track.positions[0], track.positions[-1]))
        return {name: tuple(dict.fromkeys(rows)) for name, rows in values.items()}


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskChoice(Contract):
    task: str
    alternative: str
    skill_options: tuple[tuple[str, str], ...] = ()

    def _validate(self):
        self._unique(tuple(n for n, _ in self.skill_options), "skill_options")


@dataclass(frozen=True, slots=True, kw_only=True)
class SkeletonCandidate(Contract):
    domain_digest: str
    choices: tuple[TaskChoice, ...]
    order: tuple[str, ...]
    problem: ManipulationProblem
    producer: str = "finite"

    @property
    def identity(self):
        from .roadmap import fingerprint

        return fingerprint(self.problem.skeleton)
