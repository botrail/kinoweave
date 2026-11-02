"""Explicit robot-completion and clock-invariant motion-distance objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from .._contracts import Contract, ContractError, value_as
from ..io import document_digest
from ..search.space import SearchLimit
from .geometry import ModeGeometry
from .roadmap import fingerprint
from .timeline import ManipulationTimeline


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricConfig(Contract):
    max_joint_step: float = 0.05
    max_samples: int = 100_000
    tool_links: tuple[tuple[str, str], ...] = ()

    def _validate(self):
        self._require(self.max_joint_step > 0, "max_joint_step", "must be positive")
        self._require(self.max_samples >= 2, "max_samples", "need at least two samples")
        self._unique(tuple(r for r, _ in self.tool_links), "tool_links")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectiveValue(Contract):
    problem_digest: str
    plan_digest: str
    total: float
    makespan: float
    completion: float
    distance: float
    schedule_duration: float
    robot_completion: tuple[tuple[str, float], ...]
    robot_distance: tuple[tuple[str, float], ...]
    distance_metric: str
    distance_unit: str
    metric_config: MetricConfig
    resolved_tools: tuple[tuple[str, str], ...] = ()
    geometry_samples: int = 0

    def _validate(self):
        for name in ("total", "makespan", "completion", "distance", "schedule_duration"):
            self._require(getattr(self, name) >= 0, name, "must be non-negative")


def _sum(values):
    try:
        return value_as(math.fsum(values), float, "objective")
    except OverflowError as error:
        raise ContractError("objective", "cost overflow") from error


def _tools(problem, config):
    robots = {r.name: r for r in problem.scene.robots}
    models = {m.name: m for m in problem.scene.models}
    tools = dict(config.tool_links)
    if not tools.keys() <= robots.keys():
        raise ContractError("metric.tool_links", "unknown robot")
    for name, link in tools.items():
        if link not in models[robots[name].model].links:
            raise ContractError("metric.tool_links", "unknown link")
    if problem.cost.distance_metric == "tool_path":
        for robot in problem.scene.robots:
            if robot.name not in tools:
                model = models[robot.model]
                parents = {j.parent for j in model.joints}
                leaves = [link for link in model.links if link not in parents]
                if len(leaves) != 1:
                    raise ContractError(
                        "metric.tool_links", f"{robot.name}: select a link for a branched model"
                    )
                tools[robot.name] = leaves[0]
    return tools


def _measure(problem, result, config, *, geometry=None, check=lambda: None):
    """Internal bound-plan measurement; no collision validation is implied."""
    timeline = ManipulationTimeline(problem, result)
    tools = _tools(problem, config)
    geometry = geometry or ModeGeometry(problem)
    models = geometry.models
    completions = {r.name: 0.0 for r in problem.scene.robots}
    distances = {r.name: [] for r in problem.scene.robots}
    metric = problem.cost.distance_metric
    samples = 0
    definitions = {s.name: s for s in problem.skeleton.skills}
    for skill in result.plan.skills:
        for robot in definitions[skill.skill].robots:
            completions[robot] = max(completions[robot], skill.end)

    def length(robot, first, last):
        nonlocal samples
        if first == last:
            return 0.0
        if metric == "joint_l1":
            return _sum(abs(a - b) for a, b in zip(first, last, strict=True))
        if metric == "joint_l2":
            return value_as(math.dist(first, last), float, "objective.distance")
        model = models[geometry.robots[robot].model]
        a, b = model.joint_values(first), model.joint_values(last)
        ratio = max(abs(b[n] - a[n]) for n in a) / config.max_joint_step
        if not math.isfinite(ratio) or ratio > config.max_samples - samples - 1:
            raise SearchLimit("metric_sample_limit", "objective")
        count = max(1, math.ceil(ratio))
        samples += count + 1
        points = []
        weights = None
        for i in range(count + 1):
            check()
            q = (
                first
                if i == 0
                else last
                if i == count
                else tuple(a + (b - a) * (i / count) for a, b in zip(first, last, strict=True))
            )
            fk = geometry.fk(robot, q)
            if metric == "tool_path":
                points.append((fk.poses[fk.link_names.index(tools[robot])].position,))
                weights = (1.0,)
            else:
                points.append(tuple(s.center for s in fk.spheres))
                weights = tuple(
                    value_as(
                        (4 * math.pi / 3) * s.radius * s.radius * s.radius, float, "sphere_volume"
                    )
                    for s in fk.spheres
                )
                if any(w <= 0 for w in weights):
                    raise ContractError("objective.sphere_volume", "volume underflow")
        return _sum(
            w * math.dist(x, y)
            for a, b in pairwise(points)
            for w, x, y in zip(weights, a, b, strict=True)
        )

    for robot in problem.scene.robots:
        previous = robot.start
        path = [previous]

        def accumulate(first, last, *, name=robot.name, path=path):
            if first == last:
                return
            if metric in ("joint_l1", "joint_l2"):
                distances[name].append(length(name, first, last))
                return
            # A new stop on the same joint-space line changes timing, not
            # geometry. Canonicalize such lines before choosing FK samples.
            while len(path) >= 2:
                a, b = path[-2:]
                u = tuple(y - x for x, y in zip(a, b, strict=True))
                v = tuple(y - x for x, y in zip(b, last, strict=True))
                su, sv = max(map(abs, u)), max(map(abs, v))
                if (
                    not su
                    or not sv
                    or any(abs(x / su - y / sv) > 1e-12 for x, y in zip(u, v, strict=True))
                ):
                    break
                path.pop()
            path.append(last)

        for activity in timeline.by_robot[robot.name]:
            check()
            track = next(t for t in activity.trace.tracks if t.robot == robot.name)
            if previous != track.positions[0]:
                completions[robot.name] = max(completions[robot.name], activity.start)
                accumulate(previous, track.positions[0])
            for i, (first, last) in enumerate(pairwise(track.positions)):
                if first != last:
                    completions[robot.name] = max(
                        completions[robot.name], activity.absolute(activity.trace.times[i + 1])
                    )
                    accumulate(first, last)
            previous = track.positions[-1]
        if metric not in ("joint_l1", "joint_l2"):
            distances[robot.name].extend(length(robot.name, a, b) for a, b in pairwise(path))
    completion = _sum(completions.values())
    makespan = max(completions.values())
    robot_distance = tuple((r.name, _sum(distances[r.name])) for r in problem.scene.robots)
    distance = _sum(v for _, v in robot_distance)
    total = _sum(
        (
            problem.cost.makespan * makespan,
            problem.cost.completion * completion,
            problem.cost.distance * distance,
        )
    )
    return ObjectiveValue(
        problem_digest=document_digest(problem),
        plan_digest=fingerprint(result.plan),
        total=total,
        makespan=makespan,
        completion=completion,
        distance=distance,
        schedule_duration=result.makespan,
        robot_completion=tuple(completions.items()),
        robot_distance=robot_distance,
        distance_metric=metric,
        distance_unit={
            "joint_l1": "joint_units",
            "joint_l2": "joint_units",
            "tool_path": "m",
            "sphere_volume_path": "m^4",
        }[metric],
        metric_config=config,
        resolved_tools=tuple(tools.items()),
        geometry_samples=samples,
    )


def evaluate_objective(problem, result, *, config=None):
    """Measure a structurally bound plan; imported safety evidence is not verified.

    Completion includes every assigned skill, even stationary skills, and the
    last motion before the robot remains stationary. Joint distances are exact
    for source polylines. FK distances use clock-independent joint-parameter
    samples, not temporal samples; sphere volumes are summed without normalization.
    """
    config = value_as(config or MetricConfig(), MetricConfig, "metric_config")
    try:
        return _measure(problem, result, config)
    except SearchLimit as error:
        raise ContractError("objective", error.reason) from error
