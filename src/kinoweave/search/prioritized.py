"""Fixed-priority space-time planning with persistent goal occupancy."""

from collections import deque

from .space import CoordinationSpace, SearchOutcome


def prioritized_search(space: CoordinationSpace) -> SearchOutcome:
    budget = space.budget
    budget.check("prioritized_search")
    names = tuple(g.robot for g in space.graphs)
    order = (
        tuple(names.index(name) for name in space.config.priority_order)
        if space.config.priority_order
        else tuple(range(len(names)))
    )
    reserved = {}

    def at(path, tick):
        return path[min(tick, len(path) - 1)]

    def safe(robot, source, target, tick):
        return all(
            space.pair(robot, source, target, j, at(path, tick), at(path, tick + 1))
            for j, path in reserved.items()
        )

    for robot in order:
        budget.check("prioritized_search")
        start = (space.start[robot], 0)
        budget.consume("states")
        parents = {start: None}
        queue = deque([start])
        found = None
        while queue:
            budget.check("prioritized_search")
            current = queue.popleft()
            vertex, tick = current
            budget.stats["expanded"] += 1
            if vertex == space.goal[robot]:
                until = max((len(path) - 1 for path in reserved.values()), default=tick)
                if all(safe(robot, vertex, vertex, t) for t in range(tick, until + 1)):
                    found = current
                    break
            if tick >= space.config.priority_horizon:
                continue
            targets = sorted(
                space.options(robot, vertex), key=lambda t: (space.distances[robot][t], t)
            )
            for target in targets:
                budget.check("prioritized_search")
                candidate = (target, tick + 1)
                if candidate in parents or not safe(robot, vertex, target, tick):
                    continue
                budget.consume("states")
                parents[candidate] = current
                queue.append(candidate)
        if found is None:
            return SearchOutcome(None, "prioritized_incomplete")
        path = []
        while found is not None:
            path.append(found[0])
            found = parents[found]
        reserved[robot] = tuple(reversed(path))
    length = max(len(path) for path in reserved.values())
    path = tuple(tuple(at(reserved[i], tick) for i in range(len(names))) for tick in range(length))
    return SearchOutcome(path, "goal_reached")
