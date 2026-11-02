"""SDK-independent checks shared by optional scene and runtime adapters."""

from __future__ import annotations

from typing import Literal

from .._contracts import ContractError, value_as
from ..models import PlanningProblem, ValidationReport
from ..results import ExecutionReport, PlanResult
from ..validation import validate_trajectory
from .protocols import AdapterCapabilities, SceneAdapter

AdapterRole = Literal["scene_import", "playback", "execution"]


class UnsupportedCapabilityError(ContractError):
    """The requested role or model feature is not declared by this adapter."""


def check_capabilities(
    problem: PlanningProblem, capabilities: AdapterCapabilities, role: AdapterRole
) -> None:
    """Reject unsupported roles/features instead of silently dropping scene data."""
    problem = value_as(problem, PlanningProblem, "problem")
    capabilities = value_as(capabilities, AdapterCapabilities, "capabilities")
    role = value_as(role, AdapterRole, "role")
    if not getattr(capabilities, role):
        raise UnsupportedCapabilityError(f"capabilities.{role}", "requested role is unsupported")
    for model in problem.models:
        if model.collision_spheres and "sphere" not in capabilities.shapes:
            raise UnsupportedCapabilityError(
                f"models.{model.name}.collision_spheres", "sphere geometry is unsupported"
            )
        for joint in model.joints:
            path = f"models.{model.name}.joints.{joint.name}"
            if joint.kind not in capabilities.joint_types:
                raise UnsupportedCapabilityError(path, f"{joint.kind} joints are unsupported")
            if joint.mimic is not None and not capabilities.mimic:
                raise UnsupportedCapabilityError(path, "mimic joints are unsupported")
    for obstacle in problem.obstacles:
        if obstacle.shape.kind not in capabilities.shapes:
            raise UnsupportedCapabilityError(
                f"obstacles.{obstacle.name}", f"{obstacle.shape.kind} geometry is unsupported"
            )


def take_snapshot(adapter: SceneAdapter) -> PlanningProblem:
    """Read an adapter's native contract snapshot and check its declared coverage."""
    capabilities = value_as(adapter.capabilities, AdapterCapabilities, "capabilities")
    if not capabilities.scene_import:
        raise UnsupportedCapabilityError("capabilities.scene_import", "scene export is unsupported")
    problem = value_as(adapter.snapshot(), PlanningProblem, "snapshot")
    check_capabilities(problem, capabilities, "scene_import")
    return problem


def validate_plan_for_adapter(
    problem: PlanningProblem,
    plan: PlanResult,
    capabilities: AdapterCapabilities,
    *,
    role: AdapterRole = "playback",
    max_samples: int = 100_000,
) -> ValidationReport:
    """Check a successful bound plan and recompute CPU sampled validation.

    This verifies declared input support and the core time law, not an external
    runtime's model conversion or control tracking.
    """
    check_capabilities(problem, capabilities, role)
    plan = value_as(plan, PlanResult, "plan")
    if plan.status != "success":
        raise ContractError("plan.status", "a successful plan is required")
    plan.check_problem(problem)
    if (
        plan.validation.collision_margin != problem.config.collision_margin
        or plan.validation.collision_tolerance != problem.config.collision_tolerance
    ):
        raise ContractError("plan.validation", "collision conditions do not match the problem")
    if (
        plan.validation.level == "sampled"
        and plan.validation.sample_time_step > problem.config.validation_time_step
    ):
        raise ContractError(
            "plan.validation.sample_time_step",
            "producer sampling is coarser than the problem requires",
        )
    assert plan.trajectory is not None
    report = validate_trajectory(problem, plan.trajectory, max_samples=max_samples)
    if report.status != "passed":
        issue = report.issues[0]
        raise ContractError(
            f"plan.trajectory.{issue.path or issue.code}",
            f"CPU revalidation failed: {issue.message}",
        )
    return report


def check_playback_report(
    problem: PlanningProblem,
    plan: PlanResult,
    report: ExecutionReport,
    *,
    position_tolerance: float | None = None,
    velocity_tolerance: float = 1e-9,
) -> None:
    """Compare kinematic observations to the original time law at reference times.

    Positions are always checked; velocities are checked when supplied. This
    verifies recorded samples, not unrecorded times, model fidelity, or physics.
    Tolerances use each joint's native position/rate units and no retiming.
    """
    report = value_as(report, ExecutionReport, "report")
    report.check_plan(problem, plan)
    if report.mode != "kinematic_playback":
        raise ContractError("report.mode", "kinematic playback evidence is required")
    position_tolerance = value_as(
        problem.config.joint_tolerance if position_tolerance is None else position_tolerance,
        float,
        "position_tolerance",
    )
    velocity_tolerance = value_as(velocity_tolerance, float, "velocity_tolerance")
    if position_tolerance < 0 or velocity_tolerance < 0:
        raise ContractError("tolerance", "must be non-negative")
    assert plan.trajectory is not None
    for index, observation in enumerate(report.observations):
        expected = plan.trajectory.evaluate(observation.reference_time)
        for state in observation.robots:
            target = expected[state.robot]
            if any(
                abs(a - b) > position_tolerance
                for a, b in zip(state.positions, target.positions, strict=True)
            ):
                raise ContractError(
                    f"observations[{index}].{state.robot}.positions",
                    "does not match the original time law",
                )
            if state.velocities is not None and any(
                abs(a - b) > velocity_tolerance
                for a, b in zip(state.velocities, target.velocities, strict=True)
            ):
                raise ContractError(
                    f"observations[{index}].{state.robot}.velocities",
                    "does not match the original time law",
                )
