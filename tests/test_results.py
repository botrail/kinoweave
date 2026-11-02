from dataclasses import replace

import pytest

from kinoweave import (
    BackendInfo,
    Contact,
    ContractError,
    Diagnostic,
    ExecutionReport,
    JointState,
    Observation,
    PlanResult,
    TrackingError,
    TrajectoryBundle,
    ValidationReport,
)
from kinoweave.adapters import AdapterCapabilities
from kinoweave.io import document_digest, dumps, loads


def test_trajectory_carries_shared_time_law_and_every_robot(result):
    trajectory = result.trajectory
    assert trajectory.duration == trajectory.time_step == 0.1
    assert trajectory.interpolation == "quintic_stop"
    assert result.makespan == 0.1
    assert {track.robot for track in trajectory.tracks} == {"left", "right"}
    assert loads(dumps(result)) == result


@pytest.mark.parametrize(
    "times",
    [
        (0,),
        (1, 2),
        (0, 0),
        (0, -1),
        (0, 0.1, 0.3),
        (0, 1e-15, 3e-15),
        (0, float("inf")),
    ],
)
def test_invalid_shared_clocks_are_rejected(result, times):
    with pytest.raises(ContractError, match="times"):
        replace(result.trajectory, times=times)


def test_tracks_cannot_be_missing_knots_or_duplicate_robots(result):
    track = result.trajectory.tracks[0]
    with pytest.raises(ContractError, match="duplicate"):
        replace(result.trajectory, tracks=(track, track))
    with pytest.raises(ContractError, match="shared knot"):
        replace(result.trajectory, tracks=(replace(track, positions=((0,), (0.2,), (0.3,))),))
    with pytest.raises(ContractError, match="joint order"):
        replace(track, positions=((0, 0), (0.3,)))


def test_success_requires_named_validation_checks_and_actual_backend(result):
    with pytest.raises(ContractError, match="trajectory"):
        PlanResult(status="success")
    with pytest.raises(ContractError, match="success needs validation"):
        replace(result, validation=ValidationReport())
    with pytest.raises(ContractError, match="success requires checks"):
        replace(result, validation=replace(result.validation, checks=("collisions",)))
    with pytest.raises(ContractError, match="actually used"):
        replace(result, provenance=replace(result.provenance, backend=None))
    with pytest.raises(ContractError, match="only success"):
        replace(result, status="timeout")


def test_not_run_and_validation_failure_are_distinct():
    assert PlanResult().makespan is None
    issue = Diagnostic(code="collision", message="left and right meet", stage="validation")
    report = ValidationReport(
        status="failed",
        level="sampled",
        checks=("collisions",),
        sample_time_step=0.01,
        collision_margin=0.01,
        collision_tolerance=1e-6,
        issues=(issue,),
    )
    assert PlanResult(status="validation_failed", validation=report).status == "validation_failed"
    with pytest.raises(ContractError, match="failure evidence"):
        PlanResult(status="validation_failed")
    with pytest.raises(ContractError, match="not_run cannot"):
        ValidationReport(checks=("collisions",))


def test_result_binding_rejects_stale_input_and_incorrect_joint_names(problem, result):
    result.check_problem(problem)
    with pytest.raises(ContractError, match="different problem"):
        result.check_problem(replace(problem, scene_revision="2"))
    with pytest.raises(ContractError, match="config does not match"):
        replace(
            result,
            provenance=replace(
                result.provenance,
                config=replace(problem.config, seed=8),
            ),
        ).check_problem(problem)
    left, right = result.trajectory.tracks
    with pytest.raises(ContractError, match="joint order"):
        replace(
            result,
            trajectory=replace(
                result.trajectory,
                tracks=(
                    replace(left, joint_names=("wrong_joint",)),
                    right,
                ),
            ),
        ).check_problem(problem)
    with pytest.raises(ContractError, match="cover every robot"):
        replace(result, trajectory=replace(result.trajectory, tracks=(left,))).check_problem(
            problem
        )


def test_result_binding_rejects_changed_endpoints_intermediate_limits_and_retiming(problem, result):
    left, right = result.trajectory.tracks
    with pytest.raises(ContractError, match="start or goal"):
        replace(
            result,
            trajectory=replace(
                result.trajectory,
                tracks=(
                    replace(left, positions=((0,), (0.2,))),
                    right,
                ),
            ),
        ).check_problem(problem)
    with pytest.raises(ContractError, match="outside its position limits"):
        replace(
            result,
            trajectory=TrajectoryBundle(
                times=(0, 0.1, 0.2),
                tracks=(
                    replace(left, positions=(left.positions[0], (2,), left.positions[-1])),
                    replace(right, positions=(right.positions[0], (0,), right.positions[-1])),
                ),
            ),
        ).check_problem(problem)
    with pytest.raises(ContractError, match="time step"):
        replace(result, trajectory=replace(result.trajectory, times=(0, 0.2))).check_problem(
            problem
        )


def execution_report(result) -> ExecutionReport:
    return ExecutionReport(
        plan_digest=document_digest(result),
        runtime=BackendInfo(name="observation_fixture", version="1", device="cpu"),
        mode="control_tracking",
        time_step=0.01,
        observations=(
            Observation(
                time=4,
                reference_time=0,
                robots=(JointState(robot="left", joint_names=("slide",), positions=(0.01,)),),
            ),
            Observation(
                time=4.01,
                reference_time=0.01,
                robots=(JointState(robot="left", joint_names=("slide",), positions=(0.02,)),),
            ),
        ),
        tracking_errors=(
            TrackingError(robot="left", joint="slide", unit="m", max_position_error=0.01),
        ),
        contacts=(Contact(time=4.01, first="left/carriage", second="table"),),
    )


def test_execution_observations_keep_runtime_clock_and_reference_clock_separate(result):
    report = execution_report(result)
    restored = loads(dumps(report), expected=ExecutionReport)
    assert restored == report
    assert restored.observations[0].time == 4
    assert restored.observations[0].reference_time == 0
    assert restored.tracking_errors[0].unit == "m"
    # A physical observation report is not a plan validation certificate.
    with pytest.raises(ContractError, match="expected PlanResult"):
        loads(dumps(report), expected=PlanResult)


def test_execution_rejects_reversed_clocks_and_inconsistent_observed_joint_order(result):
    report = execution_report(result)
    with pytest.raises(ContractError, match="runtime times"):
        replace(report, observations=tuple(reversed(report.observations)))
    with pytest.raises(ContractError, match="joint order must remain"):
        second = replace(
            report.observations[1],
            robots=(replace(report.observations[1].robots[0], joint_names=("different",)),),
        )
        replace(report, observations=(report.observations[0], second))
    with pytest.raises(ContractError, match="unknown observed joint"):
        replace(report, tracking_errors=(replace(report.tracking_errors[0], joint="missing"),))


def test_adapter_capabilities_are_platform_independent_and_explicit():
    capabilities = AdapterCapabilities(
        playback=True, shapes=("sphere", "box"), joint_types=("prismatic",)
    )
    assert capabilities.playback and not capabilities.execution
    with pytest.raises(ContractError, match="shapes"):
        replace(capabilities, shapes=("mesh",))
    with pytest.raises(ContractError, match="duplicate"):
        replace(capabilities, shapes=("sphere", "sphere"))
