"""Finite product A* used as an exact, GPU-backed two-robot complement."""

import heapq
from itertools import count, product

from .space import CoordinationSpace, SearchOutcome


def joint_astar_search(space: CoordinationSpace) -> SearchOutcome:
    budget = space.budget
    budget.check("joint_astar_search")
    budget.consume("states")
    parent = {space.start: None}
    cost = {space.start: 0}
    tie = count()
    frontier = [(space.distance(space.start), 0, next(tie), space.start)]
    while frontier:
        budget.check("joint_astar_search")
        _, g, _, current = heapq.heappop(frontier)
        if g != cost[current]:
            continue
        budget.stats["expanded"] += 1
        if current == space.goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return SearchOutcome(tuple(reversed(path)), "goal_reached")
        for candidate in product(*(space.options(i, v) for i, v in enumerate(current))):
            budget.check("joint_astar_search")
            if cost.get(candidate, float("inf")) <= g + 1:
                continue
            budget.stats["successor_attempts"] += 1
            if not space.transition(current, candidate):
                continue
            if candidate not in cost:
                budget.consume("states")
            cost[candidate] = g + 1
            parent[candidate] = current
            heapq.heappush(
                frontier, (g + 1 + space.distance(candidate), g + 1, next(tie), candidate)
            )
    return SearchOutcome(None, "exhausted_graph_product", exhausted=True)
