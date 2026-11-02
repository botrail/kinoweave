"""Shared-time quintic-stop trajectories and analytic motion limits."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from ._contracts import Contract, ContractError, value_as
from .models import Diagnostic, RobotModel


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionState(Contract):
    """Positions and time derivatives in the caller's independent joint order."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]

    def _validate(self) -> None:
        self._require(
            len(self.positions) == len(self.velocities) == len(self.accelerations),
            "positions",
            "joint counts must match",
        )


def _duration(duration: float) -> float:
    duration = value_as(duration, float, "duration")
    if duration <= 0:
        raise ContractError("duration", "must be positive")
    return duration


def evaluate_segment(start, end, duration: float, time: float) -> MotionState:
    """Evaluate a stop-to-stop quintic interval, without wrapping or extrapolation."""
    start = value_as(start, tuple[float, ...], "start")
    end = value_as(end, tuple[float, ...], "end")
    if len(start) != len(end):
        raise ContractError("end", "joint counts must match")
    duration = _duration(duration)
    time = value_as(time, float, "time")
    if not 0 <= time <= duration:
        raise ContractError("time", "must be within the interval")
    if time == 0 or time == duration:
        zero = (0.0,) * len(start)
        return MotionState(
            positions=start if time == 0 else end, velocities=zero, accelerations=zero
        )
    u = time / duration
    v = 1 - u
    # Symmetry avoids loss of precision in the position polynomial near u=1.
    w = min(u, v)
    blend = w**3 * (10 + w * (-15 + 6 * w))
    blend = blend if u <= 0.5 else 1 - blend
    speed = 30 * u**2 * v**2 / duration
    acceleration = 60 * u * v * (1 - 2 * u) / duration / duration
    delta = tuple(b - a for a, b in zip(start, end, strict=True))
    return MotionState(
        positions=tuple(a + d * blend for a, d in zip(start, delta, strict=True)),
        velocities=tuple(d * speed if d else 0.0 for d in delta),
        accelerations=tuple(d * acceleration if d else 0.0 for d in delta),
    )


def minimum_duration(model: RobotModel, start, end) -> float:
    """Minimum quintic duration for every moving joint, including mimic chains.

    This checks derivative limits only; position limits and collisions are
    separate. A stationary interval needs no minimum duration (returns zero).
    """
    first, last = model.joint_values(start), model.joint_values(end)
    required = 0.0
    for joint in model.joints:
        if joint.limits is None:
            continue
        delta = abs(last[joint.name] - first[joint.name])
        speed_time = (delta / joint.limits.velocity) * (15 / 8)
        acceleration_time = (math.sqrt(delta) / math.sqrt(joint.limits.acceleration)) * math.sqrt(
            10 / math.sqrt(3)
        )
        required = max(required, speed_time, acceleration_time)
    return value_as(required, float, "minimum_duration")


def segment_limit_issues(
    model: RobotModel, start, end, duration: float, *, joint_tolerance: float = 0.0
) -> tuple[Diagnostic, ...]:
    """Analytic position extrema and derivative peaks, without temporal sampling."""
    duration = _duration(duration)
    tolerance = value_as(joint_tolerance, float, "joint_tolerance")
    if tolerance < 0:
        raise ContractError("joint_tolerance", "must be non-negative")
    first, last = model.joint_values(start), model.joint_values(end)
    issues = []
    for joint in model.joints:
        limit = joint.limits
        if limit is None:
            continue
        a, b = first[joint.name], last[joint.name]
        delta = value_as(abs(b - a), float, f"joints.{joint.name}.displacement")
        if limit.lower is not None and not (
            limit.lower - tolerance <= min(a, b) and max(a, b) <= limit.upper + tolerance
        ):
            issues.append(
                Diagnostic(
                    code="position_limit",
                    message=f"{joint.name}: position bound exceeded",
                    path=joint.name,
                )
            )
        velocity = value_as((delta / duration) * (15 / 8), float, "peak_velocity")
        acceleration = value_as(
            (delta / duration / duration) * (10 / math.sqrt(3)), float, "peak_acceleration"
        )
        for code, actual, maximum in (
            ("velocity_limit", velocity, limit.velocity),
            ("acceleration_limit", acceleration, limit.acceleration),
        ):
            # Only a relative round-off guard; position tolerance is not a rate tolerance.
            if actual > maximum and not math.isclose(actual, maximum, rel_tol=1e-12, abs_tol=0):
                issues.append(
                    Diagnostic(
                        code=code,
                        message=f"{joint.name}: peak {actual:.17g} exceeds {maximum:.17g}",
                        path=joint.name,
                    )
                )
    return tuple(issues)


def sample_times(
    duration: float, max_step: float, *, max_samples: int = 100_000
) -> tuple[float, ...]:
    """Include both endpoints, using equal subdivisions no wider than max_step."""
    duration, max_step = _duration(duration), _duration(max_step)
    max_samples = value_as(max_samples, int, "max_samples")
    ratio = duration / max_step
    if max_samples < 2 or not math.isfinite(ratio) or ratio > max_samples - 1:
        raise ContractError(
            "max_samples", "sampling budget exceeded (at least two samples required)"
        )
    count = max(1, math.ceil(ratio))
    times = tuple(duration * (i / count) for i in range(count)) + (duration,)
    if any(b <= a for a, b in pairwise(times)):
        raise ContractError("time", "sample spacing is below floating-point resolution")
    return times


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotTrajectory(Contract):
    robot: str
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]

    def _validate(self) -> None:
        self._require(bool(self.joint_names), "joint_names", "at least one joint is required")
        self._unique(self.joint_names, "joint_names")
        self._require(len(self.positions) >= 2, "positions", "at least two knots are required")
        for index, row in enumerate(self.positions):
            self._require(
                len(row) == len(self.joint_names),
                f"positions[{index}]",
                f"expected joint order {self.joint_names!r}",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryBundle(Contract):
    """All tracks use the same zero-based, uniform knot times.

    ``quintic_stop`` specifies q(u)=q0+(q1-q0)*(10*u**3-15*u**4+6*u**5)
    on each interval. Velocity and acceleration are zero at every knot.
    Continuous joints use explicit unwrapped coordinates, with no implicit
    shortest-angle interpolation. Every track includes stationary end holds.
    """

    times: tuple[float, ...]
    tracks: tuple[RobotTrajectory, ...]
    interpolation: Literal["quintic_stop"] = "quintic_stop"

    def _validate(self) -> None:
        self._require(len(self.times) >= 2, "times", "at least two knots are required")
        self._require(self.times[0] == 0, "times[0]", "the shared clock must start at zero")
        step = self.times[1] - self.times[0]
        self._require(step > 0, "times", "times must be strictly increasing")
        for index in range(1, len(self.times)):
            interval = self.times[index] - self.times[index - 1]
            self._require(interval > 0, f"times[{index}]", "times must be strictly increasing")
            self._require(
                math.isclose(interval, step, rel_tol=1e-9, abs_tol=0),
                f"times[{index}]",
                "v1 requires uniform knot intervals",
            )
        self._require(bool(self.tracks), "tracks", "at least one robot track is required")
        self._unique(tuple(track.robot for track in self.tracks), "tracks")
        for track in self.tracks:
            self._require(
                len(track.positions) == len(self.times),
                f"tracks.{track.robot}.positions",
                "every track must cover all shared knot times",
            )

    @property
    def duration(self) -> float:
        return self.times[-1]

    @property
    def time_step(self) -> float:
        return self.times[1]

    def evaluate(self, time: float) -> dict[str, MotionState]:
        """Evaluate every robot at one shared time in [0, duration]."""
        time = value_as(time, float, "time")
        if not 0 <= time <= self.duration:
            raise ContractError("time", "must be within the trajectory")
        index = min(bisect_right(self.times, time) - 1, len(self.times) - 2)
        duration = self.times[index + 1] - self.times[index]
        return {
            track.robot: evaluate_segment(
                track.positions[index],
                track.positions[index + 1],
                duration,
                time - self.times[index],
            )
            for track in self.tracks
        }
