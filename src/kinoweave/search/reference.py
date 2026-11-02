"""Exhaustive breadth-first search over a small finite product of joint graphs."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from itertools import product

from .._contracts import Contract, ContractError, value_as
from ..io import document_digest
from ..models import BackendInfo, Diagnostic, PlanningProblem, PlanProvenance, ValidationReport
from ..results import PlanResult
from ..trajectory import RobotTrajectory, TrajectoryBundle
from ..validation import validate_interval, validate_trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class JointGraph(Contract):
    """Explicit undirected graph; each edge takes exactly one problem time step.

    A wait is available at every vertex, including goals. Graph construction
    and edge subdivision/retiming are deliberately left to the caller.
    """

    robot: str
    joint_names: tuple[str, ...]
    vertices: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[int, int], ...]
    start: int
    goal: int

    def _validate(self) -> None:
        self._require(
            bool(self.joint_names) and bool(self.vertices),
            "vertices",
            "nonempty joints and vertices required",
        )
        self._unique(self.joint_names, "joint_names")
        for row in self.vertices:
            self._require(len(row) == len(self.joint_names), "vertices", "joint count mismatch")
        self._require(
            0 <= self.start < len(self.vertices) and 0 <= self.goal < len(self.vertices),
            "start",
            "start and goal must be vertex indices",
        )
        seen = set()
        for a, b in self.edges:
            self._require(
                0 <= a < len(self.vertices) and 0 <= b < len(self.vertices) and a != b,
                "edges",
                "edges must connect distinct valid vertices",
            )
            pair = frozenset((a, b))
            self._require(pair not in seen, "edges", "duplicate undirected edge")
            seen.add(pair)


def reference_plan(
    problem: PlanningProblem, graphs: tuple[JointGraph, ...], *, max_states: int = 100_000
) -> PlanResult:
    """Find a minimum-tick plan in a supplied finite discretization on CPU.

    Exhaustion refers only to this graph product and the configured sampled
    predicate. Budgets are checked between transitions; this is not a hard
    real-time solver. CPU selection must be explicit, never a GPU fallback.
    """
    began = time.monotonic()
    max_states = value_as(max_states, int, "max_states")
    if max_states < 1:
        raise ContractError("max_states", "must be positive")
    if problem.config.backend != "reference" or problem.config.device != "cpu":
        raise ContractError("config", "explicitly select backend='reference', device='cpu'")
    provenance = PlanProvenance(
        problem_digest=document_digest(problem),
        config=problem.config,
        backend=BackendInfo(name="reference", version="0.1.0.dev0", device="cpu"),
    )

    def finish(status, reason, *, trajectory=None, validation=None, message=None):
        return PlanResult(
            status=status,
            provenance=provenance,
            trajectory=trajectory,
            validation=validation or ValidationReport(),
            termination_reason=reason,
            diagnostics=(
                Diagnostic(code=reason, message=message or reason, stage="reference_search"),
            ),
        )

    by_robot = {graph.robot: graph for graph in graphs}
    models = {model.name: model for model in problem.models}
    if len(by_robot) != len(graphs) or by_robot.keys() != {r.name for r in problem.robots}:
        return finish(
            "invalid_problem", "graph_binding", message="graphs must cover every robot once"
        )
    ordered = tuple(by_robot[robot.name] for robot in problem.robots)
    for robot, graph in zip(problem.robots, ordered, strict=True):
        if graph.joint_names != models[robot.model].joint_order:
            return finish(
                "invalid_problem", "graph_binding", message=f"{robot.name}: joint order mismatch"
            )
        for index, target in ((graph.start, robot.start), (graph.goal, robot.goal)):
            if any(
                abs(a - b) > problem.config.joint_tolerance
                for a, b in zip(graph.vertices[index], target, strict=True)
            ):
                return finish(
                    "invalid_problem", "graph_binding", message=f"{robot.name}: start/goal mismatch"
                )
    neighbors = []
    for graph in ordered:
        adjacency = [{i} for i in range(len(graph.vertices))]
        for a, b in graph.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        neighbors.append(tuple(tuple(sorted(row)) for row in adjacency))
    start, goal = tuple(g.start for g in ordered), tuple(g.goal for g in ordered)

    def configurations(state):
        return {g.robot: g.vertices[i] for g, i in zip(ordered, state, strict=True)}

    start_values = configurations(start)
    initial = validate_interval(problem, start_values, start_values, problem.config.time_step)
    if initial.status != "passed":
        return finish("invalid_problem", "invalid_start", validation=initial)
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        if time.monotonic() - began >= problem.config.time_limit:
            return finish("timeout", "time_limit")
        current = frontier.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            if len(path) == 1:
                path.append(path[0])  # schema v1 represents at least one interval
            times = tuple(i * problem.config.time_step for i in range(len(path)))
            if not all(math.isfinite(t) for t in times):
                return finish("invalid_problem", "numeric_range")
            trajectory = TrajectoryBundle(
                times=times,
                tracks=tuple(
                    RobotTrajectory(
                        robot=g.robot,
                        joint_names=g.joint_names,
                        positions=tuple(g.vertices[state[k]] for state in path),
                    )
                    for k, g in enumerate(ordered)
                ),
            )
            report = validate_trajectory(problem, trajectory)
            if report.status != "passed":
                return finish("validation_failed", "final_validation", validation=report)
            result = finish(
                "success",
                "goal_reached",
                trajectory=trajectory,
                validation=report,
                message=f"expanded={expanded}, discovered={len(parent)}",
            )
            result.check_problem(problem)
            return result
        expanded += 1
        for candidate in product(*(neighbors[k][node] for k, node in enumerate(current))):
            if candidate in parent:
                continue
            if time.monotonic() - began >= problem.config.time_limit:
                return finish("timeout", "time_limit")
            report = validate_interval(
                problem,
                configurations(current),
                configurations(candidate),
                problem.config.time_step,
            )
            if report.status != "passed":
                if any(issue.code == "evaluation_failed" for issue in report.issues):
                    return finish("invalid_problem", "evaluation_failed", validation=report)
                continue
            if len(parent) >= max_states:
                return finish("timeout", "state_limit")
            parent[candidate] = current
            frontier.append(candidate)
    return finish(
        "no_solution_in_discretization",
        "exhausted_graph_product",
        message=f"expanded={expanded}, discovered={len(parent)}",
    )
