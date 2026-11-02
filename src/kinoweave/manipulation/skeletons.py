"""Lazy finite assignment, parameter and resource-order enumeration."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal, Protocol
from urllib.parse import quote

from .._contracts import Contract, ContractError, value_as
from ..io import document_digest
from ..search.space import SearchLimit
from .models import ActionOrder, ScheduleSkeleton
from .tasks import SkeletonCandidate, TaskChoice, TaskDomain


@dataclass(frozen=True, slots=True, kw_only=True)
class SkeletonSearchConfig(Contract):
    strategy: Literal["input", "distance"] = "input"
    max_candidates: int = 128
    max_expansions: int = 100_000
    max_skills: int = 256
    max_frontier: int = 4096
    seed: int = 0

    def _validate(self):
        for name in ("max_candidates", "max_expansions", "max_skills", "max_frontier"):
            self._require(getattr(self, name) > 0, name, "must be positive")
        self._require(self.seed >= 0, "seed", "must be non-negative")


class SkeletonProducer(Protocol):
    def __iter__(self) -> Iterator[SkeletonCandidate]: ...
    def feedback(self, attempt) -> None: ...


def scoped_skill(task, skill):
    """Collision-free namespacing, including literal slashes and percent signs."""
    return f"{quote(task, safe='')}/{quote(skill, safe='')}"


def _ground(domain, choices):
    choices = value_as(choices, tuple[TaskChoice, ...], "choices")
    by_task = {c.task: c for c in choices}
    if len(by_task) != len(choices) or by_task.keys() != {t.name for t in domain.tasks}:
        raise ContractError("choices", "must select every task exactly once")
    skills = {}
    base_edges = set()
    task_skills = {}
    normalized = []
    for task in domain.tasks:
        choice = by_task[task.name]
        alternative = next((a for a in task.alternatives if a.name == choice.alternative), None)
        if alternative is None:
            raise ContractError("choices.alternative", "unknown alternative")
        options = dict(choice.skill_options)
        if not options and all(len(s.options) == 1 for s in alternative.skills):
            options = {s.name: s.options[0].name for s in alternative.skills}
        if options.keys() != {s.name for s in alternative.skills}:
            raise ContractError("choices.skill_options", "select one option for every local skill")
        task_skills[task.name] = []
        for skill in alternative.skills:
            option = next((o for o in skill.options if o.name == options[skill.name]), None)
            if option is None:
                raise ContractError("choices.skill_options", "unknown skill option")
            name = scoped_skill(task.name, skill.name)
            skills[name] = replace(skill, name=name, options=(option,))
            task_skills[task.name].append(name)
        normalized.append(
            replace(
                choice, skill_options=tuple((s.name, options[s.name]) for s in alternative.skills)
            )
        )
        base_edges.update(
            (scoped_skill(task.name, e.before), scoped_skill(task.name, e.after))
            for e in alternative.precedence
        )
    for edge in domain.precedence:
        # A task dependency covers the entire selected fragments. Keeping
        # explicit barriers is conservative and makes its meaning unambiguous.
        base_edges.update((a, b) for a in task_skills[edge.before] for b in task_skills[edge.after])
    return skills, base_edges, tuple(normalized)


def compile_skeleton(domain, choices, order, *, producer="external"):
    """Compile and symbolically validate a catalog selection and linearization."""
    domain = value_as(domain, TaskDomain, "domain")
    order = value_as(order, tuple[str, ...], "order")
    skills, edges, choices = _ground(domain, choices)
    if len(order) != len(skills) or set(order) != skills.keys():
        raise ContractError("order", "must cover every selected skill exactly once")
    rank = {name: i for i, name in enumerate(order)}
    if any(rank[a] >= rank[b] for a, b in edges):
        raise ContractError("order", "violates task or fragment precedence")
    states = {s.object: s for s in domain.initial_mode.states}
    resources = {}
    for name in order:
        skill = skills[name]
        option = skill.options[0]
        if any(states.get(s.object) != s for s in option.preconditions):
            raise ContractError("order.preconditions", "symbolic precondition mismatch")
        for change in option.transitions:
            if states.get(change.before.object) != change.before:
                raise ContractError("order.effects", "symbolic effect mismatch")
            states[change.after.object] = change.after
        for resource in (
            *(("robot", r) for r in skill.robots),
            *(("object", o) for o in skill.objects),
        ):
            if resource in resources:
                edges.add((resources[resource], name))
            resources[resource] = name
    if states != {s.object: s for s in domain.goal_mode.states}:
        raise ContractError(
            "order.goals", "selected alternatives do not reach symbolic object goals"
        )
    skeleton = ScheduleSkeleton(
        skills=tuple(skills[n] for n in sorted(skills)),
        precedence=tuple(ActionOrder(before=a, after=b) for a, b in sorted(edges)),
    )
    return SkeletonCandidate(
        domain_digest=document_digest(domain),
        choices=choices,
        order=order,
        problem=domain.problem(skeleton),
        producer=producer,
    )


def check_candidate(domain, candidate):
    """Untrusted producers cannot replace tasks, geometry, goals or skill traces."""
    value_as(candidate, SkeletonCandidate, "candidate")
    if candidate.domain_digest != document_digest(domain):
        raise ContractError("candidate.domain_digest", "candidate belongs to another task domain")
    compiled = compile_skeleton(
        domain, candidate.choices, candidate.order, producer=candidate.producer
    )
    if candidate.problem != compiled.problem:
        raise ContractError(
            "candidate.problem", "does not match the declared task catalog selection"
        )
    return compiled


class FiniteSkeletonProducer:
    """Bounded DFS over concrete alternatives and symbolically enabled orders.

    Only shared resources gain scheduling edges; independent skills may run
    concurrently. Nonconflicting linearizations with the same DAG are deduplicated.
    Geometric failure feedback is recorded without asserting geometric infeasibility.
    """

    def __init__(self, domain, *, config=None):
        self.domain = value_as(domain, TaskDomain, "domain")
        self.config = value_as(
            config or SkeletonSearchConfig(), SkeletonSearchConfig, "skeleton_search"
        )
        self.statistics = {}
        self.last_feedback = None

    def feedback(self, attempt):
        self.last_feedback = attempt
        self.statistics["feedback_received"] = self.statistics.get("feedback_received", 0) + 1

    def __iter__(self):
        return self.candidates()

    def candidates(self, *, check_budget=lambda: None):
        cfg, domain = self.config, self.domain
        self.statistics = {
            "expansions": 0,
            "generated": 0,
            "symbolic_rejected": 0,
            "duplicates": 0,
            "feedback_received": 0,
        }
        if sum(max(len(a.skills) for a in t.alternatives) for t in domain.tasks) > cfg.max_skills:
            raise SearchLimit("skeleton_skill_limit", "skeleton_generation")
        seen = set()
        rng = random.Random(cfg.seed)

        def expand():
            check_budget()
            if self.statistics["expansions"] >= cfg.max_expansions:
                raise SearchLimit("skeleton_expansion_limit", "skeleton_generation")
            self.statistics["expansions"] += 1

        def choices_for(task):
            for alternative in task.alternatives:
                # product is lazy; do not materialize joint parameter combinations.
                for options in product(*(s.options for s in alternative.skills)):
                    expand()
                    yield TaskChoice(
                        task=task.name,
                        alternative=alternative.name,
                        skill_options=tuple(
                            (s.name, o.name)
                            for s, o in zip(alternative.skills, options, strict=True)
                        ),
                    )

        # Explicit stack avoids Python recursion depth as task counts grow.
        stack = [((), iter(choices_for(domain.tasks[0])))]
        while stack:
            expand()
            prefix, iterator = stack[-1]
            try:
                choice = next(iterator)
            except StopIteration:
                stack.pop()
                continue
            selected = (*prefix, choice)
            if len(selected) < len(domain.tasks):
                stack.append((selected, iter(choices_for(domain.tasks[len(selected)]))))
                continue
            skills, base_edges, _ = _ground(domain, selected)
            parents = {n: set() for n in skills}
            for a, b in base_edges:
                parents[b].add(a)
            initial = {s.object: s for s in domain.initial_mode.states}
            positions = {r.name: r.start for r in domain.scene.robots}
            frontier = [((), initial, positions)]
            while frontier:
                expand()
                order, states, at = frontier.pop()
                if len(order) == len(skills):
                    if states != {s.object: s for s in domain.goal_mode.states}:
                        self.statistics["symbolic_rejected"] += 1
                        continue
                    candidate = compile_skeleton(domain, selected, order, producer="finite")
                    if candidate.identity in seen:
                        self.statistics["duplicates"] += 1
                        continue
                    if self.statistics["generated"] >= cfg.max_candidates:
                        raise SearchLimit("skeleton_candidate_limit", "skeleton_generation")
                    seen.add(candidate.identity)
                    self.statistics["generated"] += 1
                    yield candidate
                    continue
                ready = [n for n in skills if n not in order and parents[n] <= set(order)]
                if cfg.strategy == "distance":
                    tie = {n: rng.random() for n in ready}
                    ready.sort(
                        key=lambda n: (
                            sum(
                                math.dist(at[t.robot], t.positions[0])
                                for t in skills[n].options[0].trace.tracks
                            ),
                            tie[n],
                            n,
                        )
                    )
                for name in reversed(ready):
                    option = skills[name].options[0]
                    if any(states.get(s.object) != s for s in option.preconditions):
                        self.statistics["symbolic_rejected"] += 1
                        continue
                    target = dict(states)
                    for effect in option.transitions:
                        target[effect.after.object] = effect.after
                    moved = dict(at)
                    moved.update((t.robot, t.positions[-1]) for t in option.trace.tracks)
                    if len(frontier) >= cfg.max_frontier:
                        raise SearchLimit("skeleton_frontier_limit", "skeleton_generation")
                    frontier.append(((*order, name), target, moved))
