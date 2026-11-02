"""Lazy task-progress search; recursive selection uses only prefetched tables."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from itertools import product


@dataclass(slots=True)
class _Node:
    state: object
    parent: _Node | None
    choices: tuple | None
    ages: tuple
    order: tuple
    constraints: deque


def complete(space, node, prepared, prefix):
    """PIBT-style inheritance, atomic bundles and rollback, without geometry calls."""
    assigned = space.selection(prepared, prefix)
    if assigned is None:
        return None
    rank = {r: i for i, r in enumerate(node.order)}
    table = prepared.table

    def assign(robot):
        space.check("task_pibt")
        if robot in assigned:
            return True
        for candidate in space.preferred(node.state, prepared, robot):
            choice = table.actions[candidate]
            if any(r in assigned and assigned[r] != candidate for r in choice.robots):
                continue
            if not table.valid[candidate] or any(
                not table.compatible(candidate, i) for i in set(assigned.values())
            ):
                continue
            saved = assigned.copy()
            assigned.update((r, candidate) for r in choice.robots)
            blockers = [
                r
                for r in node.order
                if r not in assigned and not table.compatible(candidate, prepared.holds[r])
            ]
            success = True
            for other in sorted(blockers, key=rank.get):
                if other not in assigned:
                    space.stats["inheritance_calls"] += 1
                    if not assign(other):
                        success = False
                        break
            if success:
                return True
            assigned.clear()
            assigned.update(saved)
            space.stats["backtracks"] += 1
        return False

    for robot in node.order:
        if robot not in assigned and not assign(robot):
            return None
    return tuple(assigned[r] for r in range(len(space.names)))


def search(space):
    rng = random.Random(space.problem.scene.config.seed)
    ties = tuple(rng.random() for _ in space.names)

    def node(state, parent=None, choices=None):
        space.budget.consume("states")
        space.budget.consume("constraints")
        ages = tuple(
            (parent.ages[r] + 1 if parent else space.distance(state, r))
            if not (
                len(state.completed) == len(space.skills)
                and state.vertices[r] == space.graphs[r].goal
            )
            else 0
            for r in range(len(space.names))
        )
        order = tuple(
            sorted(
                range(len(space.names)),
                key=lambda r: (-ages[r], -space.distance(state, r), ties[r]),
            )
        )
        return _Node(state, parent, choices, ages, order, deque([()]))

    def path(current):
        result = []
        while current.parent is not None:
            result.append((current.parent.state, current.choices))
            current = current.parent
        return tuple(reversed(result))

    root = node(space.start)
    visited = {root.state}
    if space.config.algorithm == "exhaustive":
        frontier = deque([root])
        while frontier:
            space.check("task_exhaustive")
            current = frontier.popleft()
            if space.goal(current.state):
                return path(current)
            prepared = space.prepare(current.state)
            space.stats["expanded"] += 1
            for choices in product(*prepared.options):
                space.check("task_exhaustive")
                space.budget.consume("constraints")
                space.stats["successor_attempts"] += 1
                target = space.advance(current.state, prepared, choices)
                if target is not None and target not in visited:
                    visited.add(target)
                    frontier.append(
                        node(target, current, tuple(prepared.table.actions[i] for i in choices))
                    )
        return None
    stack = [root]
    while stack:
        space.check("task_lacam")
        current = stack[-1]
        if space.goal(current.state):
            return path(current)
        if not current.constraints:
            stack.pop()
            continue
        prepared = space.prepare(current.state)
        prefix = current.constraints.popleft()
        space.stats["expanded"] += 1
        if space.selection(prepared, prefix) is None:
            continue
        if len(prefix) < len(space.names):
            robot = current.order[len(prefix)]
            for candidate in space.preferred(current.state, prepared, robot):
                space.budget.consume("constraints")
                current.constraints.append((*prefix, candidate))
        space.stats["successor_attempts"] += 1
        choices = complete(space, current, prepared, prefix)
        if choices is None:
            continue
        target = space.advance(current.state, prepared, choices)
        if target is None or target in visited:
            continue
        visited.add(target)
        child = node(target, current, tuple(prepared.table.actions[i] for i in choices))
        stack.append(child)
    return None
