import math
from dataclasses import replace

import pytest

from kinoweave import ContractError, JointLimits, Mimic, RobotTrajectory, TrajectoryBundle
from kinoweave.trajectory import (
    evaluate_segment,
    minimum_duration,
    sample_times,
    segment_limit_issues,
)


def test_quintic_values_derivatives_and_exact_stops():
    state = evaluate_segment([1, 2], [3, -2], 2, 0.5)
    assert state.positions == pytest.approx((1.20703125, 1.5859375))
    assert state.velocities == pytest.approx((1.0546875, -2.109375))
    assert state.accelerations == pytest.approx((2.8125, -5.625))
    for time, expected in ((0, (1, 2)), (2, (3, -2))):
        endpoint = evaluate_segment([1, 2], [3, -2], 2, time)
        assert endpoint.positions == expected
        assert endpoint.velocities == endpoint.accelerations == (0, 0)
    wait = evaluate_segment([2], [2], 3, 1)
    assert wait.positions == (2,) and wait.velocities == wait.accelerations == (0,)


def test_derivatives_match_finite_differences():
    time, h = 0.73, 1e-5
    a = evaluate_segment([0.4], [-1.2], 2.3, time - h)
    b = evaluate_segment([0.4], [-1.2], 2.3, time)
    c = evaluate_segment([0.4], [-1.2], 2.3, time + h)
    assert (c.positions[0] - a.positions[0]) / (2 * h) == pytest.approx(b.velocities[0], abs=1e-8)
    assert (c.velocities[0] - a.velocities[0]) / (2 * h) == pytest.approx(
        b.accelerations[0], abs=1e-8
    )


@pytest.mark.parametrize(
    "duration,time", [(0, 0), (-1, 0), (1, -0.1), (1, 1.1), (1, True), (1, float("nan"))]
)
def test_interval_rejects_invalid_times(duration, time):
    with pytest.raises(ContractError):
        evaluate_segment([0], [1], duration, time)


def test_continuous_coordinates_are_unwrapped_and_knots_are_stops():
    trajectory = TrajectoryBundle(
        times=(0, 1, 2),
        tracks=(
            RobotTrajectory(
                robot="arm",
                joint_names=("turn",),
                positions=((0,), (4 * math.pi,), (4 * math.pi,)),
            ),
        ),
    )
    assert trajectory.evaluate(0.5)["arm"].positions == pytest.approx((2 * math.pi,))
    assert trajectory.evaluate(1)["arm"].velocities == (0,)
    assert trajectory.evaluate(1.7)["arm"].positions == (4 * math.pi,)
    with pytest.raises(ContractError):
        trajectory.evaluate(2.01)


def test_analytic_peaks_find_violations_between_endpoint_samples(problem):
    model = problem.models[0]
    issues = segment_limit_issues(model, (0,), (0.3,), 0.1)
    assert {i.code for i in issues} == {"velocity_limit", "acceleration_limit"}
    required = minimum_duration(model, (0,), (0.3,))
    assert required == pytest.approx(math.sqrt(0.3 * 10 / math.sqrt(3)))
    assert segment_limit_issues(model, (0,), (0.3,), required) == ()
    assert segment_limit_issues(model, (0,), (0.3,), required * 0.999)
    assert minimum_duration(model, (0,), (0,)) == 0


def test_mimic_limits_and_zero_multiplier_are_checked(problem):
    model = problem.models[0]
    follower = replace(
        model.joints[0],
        name="follower",
        parent="carriage",
        child="finger",
        mimic=Mimic(joint="slide", multiplier=-2, offset=0.1),
        limits=JointLimits(lower=-1, upper=1, velocity=0.1, acceleration=0.2),
    )
    model = replace(model, links=(*model.links, "finger"), joints=(*model.joints, follower))
    assert minimum_duration(model, (0,), (0.2,)) == pytest.approx(7.5)
    issues = segment_limit_issues(model, (0,), (0.2,), 2)
    assert any(i.code == "velocity_limit" and i.path == "follower" for i in issues)
    assert any(i.code == "position_limit" for i in segment_limit_issues(model, (0,), (0.6,), 20))
    model = replace(
        model, joints=(model.joints[0], replace(follower, mimic=Mimic(joint="slide", multiplier=0)))
    )
    assert minimum_duration(model, (0,), (0.2,)) < 2


def test_sampling_includes_endpoint_without_exceeding_step():
    assert sample_times(1, 0.3) == (0, 0.25, 0.5, 0.75, 1)
    assert sample_times(0.1, 1) == (0, 0.1)
    with pytest.raises(ContractError, match="budget"):
        sample_times(1, 0.001, max_samples=100)
    with pytest.raises(ContractError):
        sample_times(1, 0)


def test_nonfinite_computed_motion_fails_closed(problem):
    with pytest.raises(ContractError):
        evaluate_segment([-1e308], [1e308], 1, 0.5)
    with pytest.raises(ContractError):
        segment_limit_issues(problem.models[0], (-1e308,), (1e308,), 1)
    wait = evaluate_segment([1], [1], 1e-300, 5e-301)
    assert wait.velocities == wait.accelerations == (0,)
