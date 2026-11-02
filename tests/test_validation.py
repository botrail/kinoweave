from dataclasses import replace

from kinoweave import Obstacle, Pose, RobotTrajectory, Sphere, TrajectoryBundle
from kinoweave.validation import validate_interval, validate_trajectory


def bundle(problem, rows):
    return TrajectoryBundle(
        times=tuple(i * problem.config.time_step for i in range(len(rows[0]))),
        tracks=tuple(
            RobotTrajectory(robot=r.name, joint_names=problem.models[0].joint_order, positions=p)
            for r, p in zip(problem.robots, rows, strict=True)
        ),
    )


def test_mid_interval_collision_and_sequential_waits(retreat_problem):
    problem = retreat_problem
    # Swap directly: endpoints are clear but the middle of both motions coincides.
    trajectory = bundle(problem, (((-1, 0), (1, 0)), ((1, 0), (-1, 0))))
    report = validate_trajectory(problem, trajectory)
    assert report.status == "failed" and any(i.code == "collision" for i in report.issues)
    # B goes to the bay; A passes, B leaves after A has passed.
    trajectory = bundle(
        problem,
        (
            ((-1, 0), (-1, 0), (0, 0), (1, 0), (1, 0)),
            ((1, 0), (0, 0), (0, 1), (0, 0), (-1, 0)),
        ),
    )
    report = validate_trajectory(problem, trajectory)
    assert report.status == "passed" and report.level == "sampled"
    assert {"velocity_limits", "acceleration_limits", "collisions", "endpoints"} <= set(
        report.checks
    )


def test_terminal_robot_is_not_removed_from_collision_check(retreat_problem):
    a, b = retreat_problem.robots
    problem = replace(retreat_problem, robots=(replace(a, start=(0, 0), goal=(0, 0)), b))
    trajectory = bundle(problem, (((0, 0), (0, 0)), ((1, 0), (-1, 0))))
    assert any(i.code == "collision" for i in validate_trajectory(problem, trajectory).issues)


def test_every_knot_is_checked_even_when_sampling_step_is_larger(retreat_problem):
    problem = replace(
        retreat_problem, config=replace(retreat_problem.config, validation_time_step=10)
    )
    trajectory = bundle(problem, (((-1, 0), (0, 0), (1, 0)), ((1, 0), (0, 0), (-1, 0))))
    report = validate_trajectory(problem, trajectory)
    assert any(i.code == "collision" and "t=1:" in i.message for i in report.issues)


def test_derivative_limits_are_analytic_when_only_endpoints_are_sampled(problem, result):
    problem = replace(problem, config=replace(problem.config, validation_time_step=10))
    report = validate_trajectory(problem, result.trajectory)
    assert {i.code for i in report.issues} >= {"velocity_limit", "acceleration_limit"}


def test_binding_mismatch_and_position_bound_are_reported(retreat_problem):
    problem = retreat_problem
    valid = bundle(problem, (((-1, 0), (1, 0)), ((1, 0), (-1, 0))))
    bad_order = replace(
        valid, tracks=(replace(valid.tracks[0], joint_names=("y", "x")), valid.tracks[1])
    )
    report = validate_trajectory(problem, bad_order)
    assert report.status == "failed" and report.checks == ("structure",)
    missing = replace(valid, tracks=valid.tracks[:1])
    assert validate_trajectory(problem, missing).status == "failed"
    changed = replace(
        valid, tracks=(replace(valid.tracks[0], positions=((-3, 0), (1, 0))), valid.tracks[1])
    )
    issues = {i.code for i in validate_trajectory(problem, changed).issues}
    assert {"position_limit", "endpoint_mismatch"} <= issues


def test_wait_checks_static_obstacles_and_sampling_budget(retreat_problem):
    problem = replace(
        retreat_problem,
        obstacles=(
            Obstacle(name="block", shape=Sphere(radius=0.2), pose=Pose(position=(-1, 0, 0))),
        ),
    )
    states = {r.name: r.start for r in problem.robots}
    assert validate_interval(problem, states, states, 1).status == "failed"
    clean = retreat_problem
    states = {r.name: r.start for r in clean.robots}
    report = validate_interval(clean, states, states, 1, max_samples=2)
    assert report.status == "failed" and any(i.code == "evaluation_failed" for i in report.issues)


def test_collision_sampling_is_not_a_continuous_proof(retreat_problem):
    single = replace(
        retreat_problem,
        robots=retreat_problem.robots[:1],
        config=replace(retreat_problem.config, time_step=2, validation_time_step=10),
        obstacles=(Obstacle(name="thin", shape=Sphere(radius=0.01), pose=Pose()),),
    )
    trajectory = bundle(single, (((-1, 0), (1, 0)),))
    sparse = validate_trajectory(single, trajectory)
    dense = validate_trajectory(
        replace(single, config=replace(single.config, validation_time_step=0.01)), trajectory
    )
    assert sparse.status == "passed" and sparse.level == "sampled"
    assert dense.status == "failed"
