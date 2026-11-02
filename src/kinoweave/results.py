"""Plan and execution evidence, kept separate from the planning algorithm."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from ._contracts import Contract
from .models import (
    BackendInfo,
    Conventions,
    Diagnostic,
    PlanningProblem,
    PlanProvenance,
    ValidationReport,
)
from .trajectory import TrajectoryBundle

PlanStatus = Literal[
    "not_run",
    "success",
    "timeout",
    "no_solution_in_discretization",
    "invalid_problem",
    "unsupported",
    "validation_failed",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanResult(Contract):
    status: PlanStatus = "not_run"
    trajectory: TrajectoryBundle | None = None
    provenance: PlanProvenance | None = None
    validation: ValidationReport = field(default_factory=ValidationReport)
    diagnostics: tuple[Diagnostic, ...] = ()
    termination_reason: str | None = None
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.result"] = "kinoweave.result"
    schema_version: Literal[1] = 1

    def _validate(self) -> None:
        if self.status == "success":
            self._require(
                self.trajectory is not None, "trajectory", "success requires a trajectory"
            )
            self._require(self.provenance is not None, "provenance", "success requires provenance")
            assert self.provenance is not None
            self._require(
                self.provenance.backend is not None,
                "provenance.backend",
                "success must identify the backend and device actually used",
            )
            self._require(
                self.validation.status == "passed", "validation", "success needs validation"
            )
            required = {"joint_limits", "collisions", "endpoints"}
            self._require(
                required <= set(self.validation.checks),
                "validation.checks",
                f"success requires checks {sorted(required)!r}",
            )
        else:
            self._require(
                self.trajectory is None, "trajectory", "only success returns a trajectory"
            )
            self._require(
                self.validation.status != "passed",
                "validation",
                "no validated solution was returned",
            )
        if self.status == "not_run":
            self._require(self.validation.status == "not_run", "validation", "planning has not run")
            self._require(
                self.provenance is None or self.provenance.backend is None,
                "provenance.backend",
                "not_run cannot claim that a backend ran",
            )
        if self.status == "validation_failed":
            self._require(
                self.validation.status == "failed", "validation", "failure evidence required"
            )

    @property
    def makespan(self) -> float | None:
        return self.trajectory.duration if self.trajectory is not None else None

    def check_problem(self, problem: PlanningProblem) -> None:
        """Check identity, ordering, and knot states; not collision/dynamic validation.

        Adapters must perform this binding check before using a returned plan.
        Use validation.validate_trajectory for analytic limits and sampled collisions.
        """
        from .io import document_digest

        self._require(self.provenance is not None, "provenance", "problem binding needs provenance")
        assert self.provenance is not None
        self._require(
            self.provenance.problem_digest == document_digest(problem),
            "provenance.problem_digest",
            "plan belongs to a different problem",
        )
        self._require(
            self.provenance.config == problem.config,
            "provenance.config",
            "config does not match",
        )
        if self.trajectory is None:
            return
        self._require(
            math.isclose(self.trajectory.time_step, problem.config.time_step, rel_tol=1e-9),
            "trajectory.times",
            "time step does not match the problem",
        )
        robots = {robot.name: robot for robot in problem.robots}
        self._require(
            {track.robot for track in self.trajectory.tracks} == robots.keys(),
            "trajectory.tracks",
            "tracks must cover every robot in the problem exactly once",
        )
        models = {model.name: model for model in problem.models}
        tolerance = problem.config.joint_tolerance
        for track in self.trajectory.tracks:
            robot = robots[track.robot]
            model = models[robot.model]
            self._require(
                track.joint_names == model.joint_order,
                f"trajectory.tracks.{track.robot}.joint_names",
                "joint order does not match the model",
            )
            for endpoint, target in (
                (track.positions[0], robot.start),
                (track.positions[-1], robot.goal),
            ):
                self._require(
                    all(abs(a - b) <= tolerance for a, b in zip(endpoint, target, strict=True)),
                    f"trajectory.tracks.{track.robot}.positions",
                    "start or goal does not match",
                )
            for row in track.positions:
                values = model.joint_values(row)
                for joint in model.joints:
                    if joint.limits is not None and joint.limits.lower is not None:
                        value = values[joint.name]
                        assert joint.limits.upper is not None
                        self._require(
                            joint.limits.lower - tolerance
                            <= value
                            <= joint.limits.upper + tolerance,
                            f"trajectory.tracks.{track.robot}.positions",
                            f"joint {joint.name!r} is outside its position limits",
                        )


@dataclass(frozen=True, slots=True, kw_only=True)
class JointState(Contract):
    robot: str
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...] | None = None

    def _validate(self) -> None:
        self._require(bool(self.joint_names), "joint_names", "at least one joint is required")
        self._unique(self.joint_names, "joint_names")
        self._require(
            len(self.positions) == len(self.joint_names), "positions", "joint count mismatch"
        )
        self._require(
            self.velocities is None or len(self.velocities) == len(self.joint_names),
            "velocities",
            "joint count mismatch",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation(Contract):
    """Actual runtime time and the corresponding reference trajectory time."""

    time: float
    reference_time: float
    robots: tuple[JointState, ...]

    def _validate(self) -> None:
        self._require(self.time >= 0, "time", "must be non-negative")
        self._require(self.reference_time >= 0, "reference_time", "must be non-negative")
        self._require(bool(self.robots), "robots", "at least one observed robot is required")
        self._unique(tuple(robot.robot for robot in self.robots), "robots")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingError(Contract):
    robot: str
    joint: str
    unit: Literal["rad", "m"]
    max_position_error: float

    def _validate(self) -> None:
        self._require(self.max_position_error >= 0, "max_position_error", "must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class Contact(Contract):
    time: float
    first: str
    second: str

    def _validate(self) -> None:
        self._require(self.time >= 0, "time", "must be non-negative")
        self._require(self.first != self.second, "second", "contact bodies must differ")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionReport(Contract):
    """Runtime observations; this report does not certify a plan's geometry."""

    plan_digest: str
    runtime: BackendInfo
    mode: Literal["kinematic_playback", "control_tracking"]
    observations: tuple[Observation, ...]
    time_step: float
    tracking_errors: tuple[TrackingError, ...] = ()
    contacts: tuple[Contact, ...] = ()
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.execution"] = "kinoweave.execution"
    schema_version: Literal[1] = 1

    def _validate(self) -> None:
        self._require(
            re.fullmatch(r"[0-9a-f]{64}", self.plan_digest) is not None,
            "plan_digest",
            "expected a lowercase SHA-256 digest of the plan result",
        )
        self._require(self.time_step > 0, "time_step", "must be positive")
        self._require(bool(self.observations), "observations", "actual observations are required")
        signature = {state.robot: state.joint_names for state in self.observations[0].robots}
        previous = None
        for observation in self.observations:
            self._require(
                {state.robot: state.joint_names for state in observation.robots} == signature,
                "observations",
                "robot names and joint order must remain consistent",
            )
            if previous is not None:
                self._require(
                    observation.time > previous.time
                    and observation.reference_time >= previous.reference_time,
                    "observations",
                    "runtime times must increase; reference time cannot reverse",
                )
            previous = observation
        self._unique(
            tuple((error.robot, error.joint) for error in self.tracking_errors),
            "tracking_errors",
        )
        for error in self.tracking_errors:
            self._require(
                error.robot in signature and error.joint in signature[error.robot],
                "tracking_errors",
                f"unknown observed joint {(error.robot, error.joint)!r}",
            )

    def check_plan(self, problem: PlanningProblem, plan: PlanResult) -> None:
        """Bind observations to a successful plan and its canonical robot/joint names.

        This checks identity, reference-time range, coverage, and error units.
        It does not compare observed motion with the time law or certify
        physical tracking. A report may describe only part of a run.
        """
        from .io import document_digest

        self._require(plan.status == "success", "plan", "a successful plan is required")
        plan.check_problem(problem)
        self._require(
            self.plan_digest == document_digest(plan),
            "plan_digest",
            "report belongs to a different plan result",
        )
        assert plan.trajectory is not None
        models = {model.name: model for model in problem.models}
        signature = {robot.name: models[robot.model].joint_order for robot in problem.robots}
        for index, observation in enumerate(self.observations):
            self._require(
                {state.robot: state.joint_names for state in observation.robots} == signature,
                f"observations[{index}].robots",
                "robot coverage and joint order must match the problem",
            )
            self._require(
                observation.reference_time <= plan.trajectory.duration,
                f"observations[{index}].reference_time",
                "outside the plan duration",
            )
        kinds = {
            (robot.name, joint.name): joint.kind
            for robot in problem.robots
            for joint in models[robot.model].joints
        }
        for error in self.tracking_errors:
            expected = "m" if kinds[(error.robot, error.joint)] == "prismatic" else "rad"
            self._require(
                error.unit == expected,
                "tracking_errors",
                f"{error.robot}/{error.joint}: expected {expected}",
            )
