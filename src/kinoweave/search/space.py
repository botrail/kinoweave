"""Finite joint graphs, bounded compatibility queries, and shared search budgets."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Literal

from .._contracts import Contract, ContractError, value_as
from ..collision import CoordinationEvaluator, InterferenceTable, Motion
from ..io import document_digest
from .reference import JointGraph


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchConfig(Contract):
    algorithm: Literal["lacam", "prioritized", "auto", "joint_astar"] = "lacam"
    max_states: int = 100_000
    max_constraints: int = 1_000_000
    priority_horizon: int = 256
    priority_order: tuple[str, ...] = ()
    max_query_pairs: int = 4096
    max_cached_pairs: int = 65_536
    max_cached_options: int = 1024
    max_validation_samples: int = 100_000
    exact_fallback: bool = True

    def _validate(self):
        for name in (
            "max_states",
            "max_constraints",
            "priority_horizon",
            "max_query_pairs",
            "max_cached_options",
        ):
            self._require(getattr(self, name) > 0, name, "must be positive")
        self._require(self.max_cached_pairs >= 0, "max_cached_pairs", "must be non-negative")
        self._require(
            self.max_validation_samples >= 2,
            "max_validation_samples",
            "at least two samples required",
        )
        self._unique(self.priority_order, "priority_order")


class SearchLimit(RuntimeError):
    def __init__(self, reason: str, stage: str):
        self.reason, self.stage = reason, stage
        super().__init__(f"{stage}: {reason}")


class Budget:
    """One soft wall deadline and aggregate resource counters across all phases."""

    def __init__(self, seconds: float, config: SearchConfig):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.phase_deadline = None
        self.config = config
        self.stage = "initialization"
        self.stats = {
            "states": 0,
            "constraints": 0,
            "expanded": 0,
            "successor_attempts": 0,
            "inheritance_calls": 0,
            "backtracks": 0,
            "pair_queries": 0,
            "pair_cache_hits": 0,
            "peak_cached_pairs": 0,
        }

    def elapsed(self):
        return time.monotonic() - self.started

    def check(self, stage: str):
        self.stage = stage
        now = time.monotonic()
        if now >= self.deadline:
            raise SearchLimit("time_limit", stage)
        if self.phase_deadline is not None and now >= self.phase_deadline:
            raise SearchLimit("phase_time_limit", stage)

    def reserve_phase(self, fraction: float | None):
        self.phase_deadline = (
            None
            if fraction is None
            else time.monotonic() + max(0, self.deadline - time.monotonic()) * fraction
        )

    def consume(self, kind: str):
        maximum = self.config.max_states if kind == "states" else self.config.max_constraints
        if self.stats[kind] >= maximum:
            raise SearchLimit("state_limit" if kind == "states" else "constraint_limit", self.stage)
        self.stats[kind] += 1


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    path: tuple[tuple[int, ...], ...] | None
    reason: str
    exhausted: bool = False


class CoordinationSpace:
    def __init__(
        self,
        evaluator: CoordinationEvaluator,
        graphs: tuple[JointGraph, ...],
        config: SearchConfig,
        budget: Budget,
    ):
        self.evaluator, self.problem = evaluator, evaluator.problem
        self.config, self.budget = config, budget
        self.digest = document_digest(self.problem)
        by_robot = {g.robot: g for g in graphs}
        if len(by_robot) != len(graphs) or by_robot.keys() != {r.name for r in self.problem.robots}:
            raise ContractError("graphs", "must cover every robot exactly once")
        self.graphs = tuple(by_robot[r.name] for r in self.problem.robots)
        models = {m.name: m for m in self.problem.models}
        for robot, graph in zip(self.problem.robots, self.graphs, strict=True):
            if graph.joint_names != models[robot.model].joint_order:
                raise ContractError(f"graphs.{robot.name}", "joint order does not match")
            for node, expected in ((graph.start, robot.start), (graph.goal, robot.goal)):
                if any(
                    abs(a - b) > self.problem.config.joint_tolerance
                    for a, b in zip(graph.vertices[node], expected, strict=True)
                ):
                    raise ContractError(f"graphs.{robot.name}", "start/goal does not match")
        self.start = tuple(g.start for g in self.graphs)
        self.goal = tuple(g.goal for g in self.graphs)
        self.adjacency, self.distances = [], []
        for graph in self.graphs:
            budget.check("graph_preparation")
            rows = [{i} for i in range(len(graph.vertices))]
            for a, b in graph.edges:
                rows[a].add(b)
                rows[b].add(a)
            self.adjacency.append(tuple(tuple(sorted(row)) for row in rows))
            distances = [float("inf")] * len(rows)
            distances[graph.goal] = 0
            queue = deque([graph.goal])
            while queue:
                node = queue.popleft()
                for other in rows[node]:
                    if distances[other] == float("inf"):
                        distances[other] = distances[node] + 1
                        queue.append(other)
            self.distances.append(tuple(distances))
        self._options = OrderedDict()
        self._pairs = OrderedDict()
        self._pair_cells = 0

    def motion(self, robot: int, start: int, end: int):
        graph = self.graphs[robot]
        return Motion(start=graph.vertices[start], end=graph.vertices[end])

    def options(self, robot: int, source: int) -> tuple[int, ...]:
        key = (robot, source)
        if key in self._options:
            self._options.move_to_end(key)
            return self._options[key]
        self.budget.check(self.budget.stage)
        targets = self.adjacency[robot][source]
        flags = value_as(
            self.evaluator.motion_validity(
                self.graphs[robot].robot, tuple(self.motion(robot, source, t) for t in targets)
            ),
            tuple[bool, ...],
            "motion_validity",
        )
        self.budget.check(self.budget.stage)
        if len(flags) != len(targets):
            raise ContractError("motion_validity", "verdict count mismatch")
        result = tuple(t for t, flag in zip(targets, flags, strict=True) if flag)
        if len(self._options) >= self.config.max_cached_options:
            self._options.popitem(last=False)
        self._options[key] = result
        return result

    def pair(self, i, source_i, target_i, j, source_j, target_j) -> bool:
        if i > j:
            return self.pair(j, source_j, target_j, i, source_i, target_i)
        first, second = self.options(i, source_i), self.options(j, source_j)
        if target_i not in first or target_j not in second:
            return False
        # Cache bounded rectangular blocks of outgoing motions, never vertex IDs as collisions.
        maximum = self.config.max_query_pairs
        tiling = getattr(self.evaluator, "tiling", None)
        if tiling is not None:
            maximum = min(maximum, tiling.max_table_pairs)
        width = min(len(second), maximum)
        height = min(len(first), max(1, maximum // width))
        a, b = first.index(target_i), second.index(target_j)
        first_block = first[(a // height) * height : (a // height + 1) * height]
        second_block = second[(b // width) * width : (b // width + 1) * width]
        key = (i, source_i, first_block, j, source_j, second_block)
        if key in self._pairs:
            self.budget.stats["pair_cache_hits"] += 1
            self._pairs.move_to_end(key)
            compatible = self._pairs[key]
        else:
            self.budget.check(self.budget.stage)
            first_motions = tuple(self.motion(i, source_i, t) for t in first_block)
            second_motions = tuple(self.motion(j, source_j, t) for t in second_block)
            table = value_as(
                self.evaluator.interference(
                    self.graphs[i].robot, first_motions, self.graphs[j].robot, second_motions
                ),
                InterferenceTable,
                "interference",
            )
            self.budget.stats["pair_queries"] += 1
            self.budget.check(self.budget.stage)
            if (
                table.problem_digest != self.digest
                or table.first_robot != self.graphs[i].robot
                or table.second_robot != self.graphs[j].robot
                or table.first_motions != first_motions
                or table.second_motions != second_motions
                or table.backend != self.evaluator.info
                or table.duration != self.problem.config.time_step
                or table.sample_time_step != self.problem.config.validation_time_step
            ):
                raise ContractError(
                    "interference", "table does not match the requested problem and motions"
                )
            compatible = table.compatible
            cells = len(first_block) * len(second_block)
            if cells <= self.config.max_cached_pairs:
                while self._pair_cells + cells > self.config.max_cached_pairs:
                    _, old = self._pairs.popitem(last=False)
                    self._pair_cells -= len(old) * len(old[0])
                self._pairs[key] = compatible
                self._pair_cells += cells
                self.budget.stats["peak_cached_pairs"] = max(
                    self.budget.stats["peak_cached_pairs"], self._pair_cells
                )
        return compatible[first_block.index(target_i)][second_block.index(target_j)]

    def transition(self, source, target):
        if any(
            t not in self.options(i, s) for i, (s, t) in enumerate(zip(source, target, strict=True))
        ):
            return False
        return all(
            self.pair(i, source[i], target[i], j, source[j], target[j])
            for i in range(len(source))
            for j in range(i + 1, len(source))
        )

    def distance(self, state):
        return max(self.distances[i][v] for i, v in enumerate(state))
