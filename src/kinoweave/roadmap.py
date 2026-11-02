"""Seeded bounded-joint PRM construction with checked stop-to-stop subdivisions."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ._contracts import Contract, ContractError, value_as
from .collision import Motion, MotionEvaluator
from .io import document_digest
from .models import BackendInfo
from .search.reference import JointGraph
from .trajectory import RobotTrajectory, TrajectoryBundle


@dataclass(frozen=True, slots=True, kw_only=True)
class PRMConfig(Contract):
    num_samples: int = 128
    neighbors: int = 12
    max_segment_steps: int = 1000
    max_vertices: int = 10_000
    connection_batch_size: int = 32

    def _validate(self) -> None:
        self._require(self.num_samples >= 0, "num_samples", "must be non-negative")
        for name in ("neighbors", "max_segment_steps", "max_vertices", "connection_batch_size"):
            self._require(getattr(self, name) > 0, name, "must be positive")
        self._require(
            self.num_samples + 2 <= self.max_vertices,
            "max_vertices",
            "sample count exceeds vertex budget",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Roadmap(Contract):
    problem_digest: str
    graph: JointGraph
    time_step: float
    backend: BackendInfo
    attempted_samples: int
    accepted_samples: int
    candidate_connections: int
    accepted_connections: int

    def _validate(self) -> None:
        self._require(self.time_step > 0, "time_step", "must be positive")
        self._require(
            0 <= self.accepted_samples <= self.attempted_samples,
            "accepted_samples",
            "invalid sample counts",
        )
        self._require(
            0 <= self.accepted_connections <= self.candidate_connections,
            "accepted_connections",
            "invalid connection counts",
        )

    def shortest_path(self) -> tuple[int, ...] | None:
        """Minimum-tick single-robot path, ignoring other robot instances."""
        graph = self.graph
        adjacency = [[] for _ in graph.vertices]
        for a, b in graph.edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        parent = {graph.start: None}
        frontier = deque([graph.start])
        while frontier:
            current = frontier.popleft()
            if current == graph.goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return tuple(reversed(path))
            for neighbor in sorted(adjacency[current]):
                if neighbor not in parent:
                    parent[neighbor] = current
                    frontier.append(neighbor)
        return None

    def trajectory(self) -> TrajectoryBundle | None:
        """One track only; callers must separately validate it for its intended scene."""
        path = self.shortest_path()
        if path is None:
            return None
        if len(path) == 1:
            path = (*path, path[0])
        return TrajectoryBundle(
            times=tuple(i * self.time_step for i in range(len(path))),
            tracks=(
                RobotTrajectory(
                    robot=self.graph.robot,
                    joint_names=self.graph.joint_names,
                    positions=tuple(self.graph.vertices[i] for i in path),
                ),
            ),
        )


def _subdivide(model, start, end, duration, maximum):
    first, last = model.joint_values(start), model.joint_values(end)
    ratio = 1.0
    for joint in model.joints:
        if joint.limits is None:
            continue
        delta = abs(last[joint.name] - first[joint.name])
        ratio = max(
            ratio,
            (delta / duration) * (15 / 8) / joint.limits.velocity,
            (delta / duration / duration) * (10 / math.sqrt(3)) / joint.limits.acceleration,
        )
    ratio = value_as(ratio, float, "subdivision")
    steps = math.ceil(ratio)
    if steps > maximum:
        return None
    points = tuple(
        tuple(a + (b - a) * (i / steps) for a, b in zip(start, end, strict=True))
        for i in range(steps)
    ) + (end,)
    return tuple(Motion(start=a, end=b) for a, b in zip(points, points[1:], strict=False))


def build_prm(
    evaluator: MotionEvaluator,
    robot: str,
    *,
    config: PRMConfig | None = None,
    samples: tuple[tuple[float, ...], ...] | None = None,
    check_budget: Callable[[], None] | None = None,
) -> Roadmap:
    """Build a finite k-nearest PRM and connect explicit start/goal states.

    Sampling is uniform in declared position ranges, seeded by the problem.
    Supplied samples replace random sampling. No other robot is treated as a
    static obstacle. Every accepted connection is subdivided into one-tick
    stops and checked in both directions by the chosen evaluator.
    """
    config = config or PRMConfig()
    problem = evaluator.problem
    check_budget = check_budget or (lambda: None)

    def checked(motions):
        check_budget()
        flags = value_as(
            evaluator.motion_validity(robot, motions), tuple[bool, ...], "motion_validity"
        )
        if len(flags) != len(motions):
            raise ContractError("motion_validity", "verdict count must match the candidate count")
        check_budget()
        return flags

    instances = [r for r in problem.robots if r.name == robot]
    if not instances:
        raise ContractError("robot", f"unknown robot {robot!r}")
    instance = instances[0]
    model = next(m for m in problem.models if m.name == instance.model)
    by_name = {j.name: j for j in model.joints}
    bounds = []
    for name in model.joint_order:
        limits = by_name[name].limits
        if limits is None or limits.lower is None or limits.lower == limits.upper:
            raise ContractError(
                f"joints.{name}", "PRM sampling requires nonzero finite position ranges"
            )
        span = value_as(limits.upper - limits.lower, float, f"joints.{name}.range")
        bounds.append((limits.lower, limits.upper, span))
    if not bounds:
        raise ContractError("joint_order", "at least one independent joint required")
    if samples is None:
        rng = random.Random(problem.config.seed)
        samples = tuple(
            tuple(rng.uniform(a, b) for a, b, _ in bounds) for _ in range(config.num_samples)
        )
    else:
        samples = value_as(samples, tuple[tuple[float, ...], ...], "samples")
        if len(samples) + 2 > config.max_vertices:
            raise ContractError("samples", "vertex budget exceeded")
    if any(len(row) != len(bounds) for row in samples):
        raise ContractError("samples", "joint count mismatch")
    endpoints = (
        Motion(start=instance.start, end=instance.start),
        Motion(start=instance.goal, end=instance.goal),
    )
    valid = checked(endpoints)
    for label, flag in zip(("start", "goal"), valid, strict=True):
        if not flag:
            raise ContractError(
                f"robots.{robot}.{label}", "self/static collision or invalid joint limits"
            )
    vertices = list(dict.fromkeys((instance.start, instance.goal)))
    goal = vertices.index(instance.goal)
    sample_flags = checked(tuple(Motion(start=p, end=p) for p in samples))
    accepted_samples = 0
    for point, flag in zip(samples, sample_flags, strict=True):
        if flag and point not in vertices:
            vertices.append(point)
            accepted_samples += 1
    initial_count = len(vertices)

    def distance(a, b):
        return math.hypot(
            *((x - y) / span for x, y, (_, _, span) in zip(a, b, bounds, strict=True))
        )

    connections = set()
    for i, point in enumerate(vertices):
        check_budget()
        nearest = sorted(
            (j for j in range(initial_count) if j != i),
            key=lambda j: (distance(point, vertices[j]), j),
        )[: config.neighbors]
        connections.update(tuple(sorted((i, j))) for j in nearest)
    if goal != 0:
        connections.add((0, goal))
    edges = set()
    indices = {p: i for i, p in enumerate(vertices)}
    accepted_connections = 0

    def check_connections(batch):
        nonlocal accepted_connections
        candidates = tuple(
            m
            for motions in batch
            for m in (*motions, *(Motion(start=m.end, end=m.start) for m in reversed(motions)))
        )
        flags = checked(candidates)
        offset = 0
        for motions in batch:
            size = 2 * len(motions)
            valid = all(flags[offset : offset + size])
            offset += size
            if not valid:
                continue
            additions = list(dict.fromkeys(m.end for m in motions if m.end not in indices))
            if len(vertices) + len(additions) > config.max_vertices:
                raise ContractError("max_vertices", "checked subdivision exceeds vertex budget")
            for point in additions:
                indices[point] = len(vertices)
                vertices.append(point)
            for motion in motions:
                i, j = indices[motion.start], indices[motion.end]
                if i != j:
                    edges.add(tuple(sorted((i, j))))
            accepted_connections += 1

    pending = []
    for a, b in sorted(connections):
        check_budget()
        motions = _subdivide(
            model, vertices[a], vertices[b], problem.config.time_step, config.max_segment_steps
        )
        if motions is None:
            continue
        pending.append(motions)
        if len(pending) >= config.connection_batch_size:
            check_connections(pending)
            pending = []
    if pending:
        check_connections(pending)
    graph = JointGraph(
        robot=robot,
        joint_names=model.joint_order,
        vertices=tuple(vertices),
        edges=tuple(sorted(edges)),
        start=0,
        goal=goal,
    )
    return Roadmap(
        problem_digest=document_digest(problem),
        graph=graph,
        time_step=problem.config.time_step,
        backend=evaluator.info,
        attempted_samples=len(samples),
        accepted_samples=accepted_samples,
        candidate_connections=len(connections),
        accepted_connections=accepted_connections,
    )
