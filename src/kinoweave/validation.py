"""CPU evaluation of candidate intervals and independently reconstructed trajectories."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise

from ._contracts import ContractError, value_as
from .geometry.reference import ReferenceCollisionChecker
from .models import Diagnostic, PlanningProblem, ValidationReport
from .trajectory import RobotTrajectory, TrajectoryBundle, sample_times, segment_limit_issues


def _report(problem, checks, issues):
    return ValidationReport(
        status="failed" if issues else "passed",
        level="sampled",
        checks=tuple(checks),
        sample_time_step=problem.config.validation_time_step,
        collision_margin=problem.config.collision_margin,
        collision_tolerance=problem.config.collision_tolerance,
        issues=tuple(issues),
    )


def _tracks(problem: PlanningProblem, trajectory: TrajectoryBundle):
    tracks = {track.robot: track for track in trajectory.tracks}
    if tracks.keys() != {robot.name for robot in problem.robots}:
        raise ContractError("trajectory.tracks", "must cover every robot exactly once")
    models = {model.name: model for model in problem.models}
    for robot in problem.robots:
        if tracks[robot.name].joint_names != models[robot.model].joint_order:
            raise ContractError(
                f"trajectory.tracks.{robot.name}.joint_names", "joint order does not match"
            )
    return tracks, models


def _sample_trajectory(trajectory, step, max_samples):
    max_samples = value_as(max_samples, int, "max_samples")
    if max_samples < 2:
        raise ContractError("max_samples", "at least two samples required")
    count = 1
    previous = 0.0
    yield previous
    for start, end in pairwise(trajectory.times):
        offsets = sample_times(end - start, step, max_samples=max_samples - count + 1)
        count += len(offsets) - 1
        for index, offset in enumerate(offsets[1:], 1):
            time = end if index == len(offsets) - 1 else start + offset
            if time <= previous:
                raise ContractError("time", "sample spacing is below floating-point resolution")
            yield time
            previous = time


def _validate(
    problem: PlanningProblem,
    trajectory: TrajectoryBundle,
    *,
    endpoints: bool,
    max_samples: int,
) -> ValidationReport:
    checks = ["structure"]
    issues = []
    try:
        tracks, models = _tracks(problem, trajectory)
        if endpoints and not math.isclose(
            trajectory.time_step, problem.config.time_step, rel_tol=1e-9
        ):
            raise ContractError("trajectory.times", "time step does not match the problem")
    except ContractError as error:
        return _report(
            problem,
            checks,
            [Diagnostic(code="trajectory_structure", message=str(error), path=error.path)],
        )
    if endpoints:
        checks.append("endpoints")
        for robot in problem.robots:
            for label, actual, expected in (
                ("start", tracks[robot.name].positions[0], robot.start),
                ("goal", tracks[robot.name].positions[-1], robot.goal),
            ):
                if any(
                    abs(a - b) > problem.config.joint_tolerance
                    for a, b in zip(actual, expected, strict=True)
                ):
                    issues.append(
                        Diagnostic(
                            code="endpoint_mismatch",
                            message=f"{robot.name}: {label} does not match",
                            path=robot.name,
                        )
                    )
    checks.extend(("joint_limits", "velocity_limits", "acceleration_limits"))
    try:
        for robot in problem.robots:
            track = tracks[robot.name]
            for i, (start, end) in enumerate(pairwise(track.positions)):
                for issue in segment_limit_issues(
                    models[robot.model],
                    start,
                    end,
                    trajectory.times[i + 1] - trajectory.times[i],
                    joint_tolerance=problem.config.joint_tolerance,
                ):
                    issues.append(
                        Diagnostic(
                            code=issue.code,
                            message=f"{robot.name} interval {i}: {issue.message}",
                            path=f"{robot.name}.intervals[{i}].{issue.path}",
                        )
                    )
        checks.append("collisions")
        checker = ReferenceCollisionChecker(problem)
        for time in _sample_trajectory(
            trajectory, problem.config.validation_time_step, max_samples
        ):
            states = trajectory.evaluate(time)
            collisions = checker.collisions(
                {name: state.positions for name, state in states.items()}
            )
            if collisions:
                hit = collisions[0]
                issues.append(
                    Diagnostic(
                        code="collision",
                        message=(
                            f"t={time:.17g}: {hit.first} vs {hit.second}, "
                            f"clearance={hit.clearance:.17g}"
                        ),
                        path=f"time[{time:.17g}]",
                    )
                )
                break
    except (ContractError, OverflowError) as error:
        issues.append(
            Diagnostic(
                code="evaluation_failed",
                message=str(error),
                path=getattr(error, "path", "geometry"),
            )
        )
    return _report(problem, checks, issues)


def validate_trajectory(
    problem: PlanningProblem, trajectory: TrajectoryBundle, *, max_samples: int = 100_000
) -> ValidationReport:
    """Check binding, analytic limits, and sampled collisions from the saved time law.

    Every knot and every robot is included, including all waits and terminal
    occupancy. Collisions remain sampled evidence, not a continuous proof.
    No search cache or imported validation report is consulted.
    """
    return _validate(problem, trajectory, endpoints=True, max_samples=max_samples)


def validate_interval(
    problem: PlanningProblem,
    start: Mapping[str, tuple[float, ...]],
    end: Mapping[str, tuple[float, ...]],
    duration: float,
    *,
    max_samples: int = 100_000,
) -> ValidationReport:
    """Check a synchronous candidate edge, including stationary robots.

    Endpoints need not be the problem's start/goal and duration need not be
    its knot interval. All robots and their declared joint order are required.
    """
    names = {robot.name for robot in problem.robots}
    if set(start) != names or set(end) != names:
        raise ContractError("configurations", "must cover every robot exactly once")
    models = {model.name: model for model in problem.models}
    trajectory = TrajectoryBundle(
        times=(0.0, duration),
        tracks=tuple(
            RobotTrajectory(
                robot=robot.name,
                joint_names=models[robot.model].joint_order,
                positions=(start[robot.name], end[robot.name]),
            )
            for robot in problem.robots
        ),
    )
    return _validate(problem, trajectory, endpoints=False, max_samples=max_samples)
