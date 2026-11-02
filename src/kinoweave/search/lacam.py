"""Geometry-based lazy constraint search with a PIBT-inspired configuration generator.

The two-level search follows the LaCAM idea (Okumura, AAAI 2023). Graphs
belong to individual robots; all conflicts use synchronous geometric motions.
No shared-grid PIBT liveness or optimality guarantee is assumed.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from .space import CoordinationSpace, SearchOutcome


@dataclass(slots=True)
class _Node:
    state: tuple[int, ...]
    parent: _Node | None
    ages: tuple[float, ...]
    order: tuple[int, ...]
    constraints: deque


def _preferred(space, state, robot, rng):
    values = list(space.options(robot, state[robot]))
    rng.shuffle(values)
    values.sort(key=lambda target: space.distances[robot][target])
    return values


def _partial_valid(space, state, fixed):
    for i, target in fixed.items():
        if target not in space.options(i, state[i]):
            return False
        for j, other in fixed.items():
            if j < i and not space.pair(i, state[i], target, j, state[j], other):
                return False
    return True


def _complete(space, node, fixed, rng):
    """Reserve a move, inherit priority into geometric blockers, rollback on failure."""
    assigned = dict(fixed)
    rank = {robot: k for k, robot in enumerate(node.order)}

    def assign(robot):
        space.budget.check("lacam_search")
        if robot in assigned:
            return True
        for target in _preferred(space, node.state, robot, rng):
            if any(
                not space.pair(robot, node.state[robot], target, j, node.state[j], other)
                for j, other in assigned.items()
            ):
                continue
            snapshot = assigned.copy()
            assigned[robot] = target
            blockers = [
                j
                for j in node.order
                if j not in assigned
                and not space.pair(
                    robot, node.state[robot], target, j, node.state[j], node.state[j]
                )
            ]
            success = True
            for other in sorted(blockers, key=rank.get):
                if other not in assigned:
                    space.budget.stats["inheritance_calls"] += 1
                    if not assign(other):
                        success = False
                        break
            if success:
                return True
            assigned.clear()
            assigned.update(snapshot)
            space.budget.stats["backtracks"] += 1
        return False

    for robot in node.order:
        if robot not in assigned and not assign(robot):
            return None
    target = tuple(assigned[i] for i in range(len(node.state)))
    return target if space.transition(node.state, target) else None


def lacam_search(space: CoordinationSpace) -> SearchOutcome:
    """First-solution DFS over configurations with per-node lazy constraint queues.

    All legal next-vertex prefixes are enumerated without candidate truncation.
    A fully constrained compatible successor bypasses heuristic incompleteness.
    Exhaustion is meaningful only when no resource/deadline limit interrupted it.
    """
    budget = space.budget
    budget.check("lacam_search")
    rng = random.Random(space.problem.config.seed)
    tie = tuple(rng.random() for _ in space.graphs)

    def make_node(state, parent):
        budget.consume("states")
        budget.consume("constraints")
        ages = tuple(
            (0 if state[i] == space.goal[i] else parent.ages[i] + 1)
            if parent is not None
            else space.distances[i][state[i]]
            for i in range(len(state))
        )
        order = tuple(
            sorted(
                range(len(state)), key=lambda i: (-ages[i], -space.distances[i][state[i]], tie[i])
            )
        )
        return _Node(state, parent, ages, order, deque([()]))

    root = make_node(space.start, None)
    visited = {root.state: root}
    stack = [root]
    while stack:
        budget.check("lacam_search")
        node = stack[-1]
        if node.state == space.goal:
            path = []
            while node is not None:
                path.append(node.state)
                node = node.parent
            return SearchOutcome(tuple(reversed(path)), "goal_reached")
        if not node.constraints:
            stack.pop()
            continue
        prefix = node.constraints.popleft()
        budget.stats["expanded"] += 1
        fixed = dict(zip(node.order, prefix, strict=False))
        if not _partial_valid(space, node.state, fixed):
            continue
        # Retain every partial assignment even when heuristic completion fails.
        if len(prefix) < len(node.state):
            robot = node.order[len(prefix)]
            for target in _preferred(space, node.state, robot, rng):
                budget.consume("constraints")
                node.constraints.append((*prefix, target))
        budget.stats["successor_attempts"] += 1
        successor = _complete(space, node, fixed, rng)
        if successor is None or successor in visited:
            continue
        child = make_node(successor, node)
        visited[successor] = child
        stack.append(child)
    return SearchOutcome(None, "exhausted_graph_product", exhausted=True)
