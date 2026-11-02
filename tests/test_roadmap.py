from dataclasses import replace

import pytest

from kinoweave import Box, ContractError, Obstacle, Pose, Sphere
from kinoweave.collision import ReferenceMotionEvaluator
from kinoweave.roadmap import PRMConfig, build_prm
from kinoweave.validation import validate_trajectory


def test_subdivision_satisfies_one_tick_limits(problem):
    problem = replace(problem, config=replace(problem.config, time_step=0.5))
    roadmap = build_prm(ReferenceMotionEvaluator(problem), "left", config=PRMConfig(num_samples=0))
    path = roadmap.shortest_path()
    assert len(path) == 8  # acceleration requires 7 separate stop-to-stop intervals
    assert len(roadmap.graph.edges) == 7
    trajectory = roadmap.trajectory()
    assert trajectory.duration == 3.5
    single = replace(problem, robots=problem.robots[:1])
    assert validate_trajectory(single, trajectory).status == "passed"


def test_prm_goes_around_static_obstacle(retreat_problem):
    problem = replace(
        retreat_problem,
        robots=retreat_problem.robots[:1],
        obstacles=(Obstacle(name="block", shape=Sphere(radius=0.2), pose=Pose()),),
    )
    roadmap = build_prm(
        ReferenceMotionEvaluator(problem),
        "alpha",
        config=PRMConfig(num_samples=0, neighbors=2),
        samples=((-1, 1), (1, 1)),
    )
    path = roadmap.shortest_path()
    assert path is not None and any(roadmap.graph.vertices[i][1] > 0 for i in path)
    assert validate_trajectory(problem, roadmap.trajectory()).status == "passed"


def test_seeded_sampling_is_reproducible_and_connects_start_goal(retreat_problem):
    config = PRMConfig(num_samples=12, neighbors=3)
    evaluator = ReferenceMotionEvaluator(retreat_problem)
    first = build_prm(evaluator, "alpha", config=config)
    second = build_prm(evaluator, "alpha", config=config)
    assert first == second
    assert first == build_prm(evaluator, "alpha", config=replace(config, connection_batch_size=1))
    assert first.graph.vertices[first.graph.start] == retreat_problem.robots[0].start
    assert first.graph.vertices[first.graph.goal] == retreat_problem.robots[0].goal
    other = replace(retreat_problem, config=replace(retreat_problem.config, seed=8))
    assert build_prm(ReferenceMotionEvaluator(other), "alpha", config=config).graph != first.graph


def test_disconnected_prm_does_not_claim_continuous_impossibility(retreat_problem):
    problem = replace(
        retreat_problem,
        obstacles=(Obstacle(name="wall", shape=Box(size=(0.4, 6, 1)), pose=Pose()),),
    )
    roadmap = build_prm(
        ReferenceMotionEvaluator(problem),
        "alpha",
        config=PRMConfig(num_samples=0),
        samples=((-1, 1), (1, 1)),
    )
    assert roadmap.shortest_path() is None and roadmap.trajectory() is None


def test_invalid_endpoints_and_resource_limits_are_explicit(retreat_problem, problem):
    blocked = replace(
        retreat_problem,
        obstacles=(
            Obstacle(name="blocked_start", shape=Sphere(radius=1), pose=Pose(position=(-1, 0, 0))),
        ),
    )
    with pytest.raises(ContractError, match="start"):
        build_prm(ReferenceMotionEvaluator(blocked), "alpha", config=PRMConfig(num_samples=0))
    short = replace(problem, config=replace(problem.config, time_step=0.5))
    with pytest.raises(ContractError, match="vertex budget"):
        build_prm(
            ReferenceMotionEvaluator(short), "left", config=PRMConfig(num_samples=0, max_vertices=5)
        )
    limited = build_prm(
        ReferenceMotionEvaluator(short),
        "left",
        config=PRMConfig(num_samples=0, max_segment_steps=1),
    )
    assert limited.shortest_path() is None


def test_equal_start_goal_returns_one_hold(retreat_problem):
    robot = replace(retreat_problem.robots[0], goal=retreat_problem.robots[0].start)
    problem = replace(retreat_problem, robots=(robot,))
    roadmap = build_prm(
        ReferenceMotionEvaluator(problem), robot.name, config=PRMConfig(num_samples=0)
    )
    assert roadmap.shortest_path() == (0,)
    trajectory = roadmap.trajectory()
    assert trajectory.duration == problem.config.time_step
    assert trajectory.tracks[0].positions == (robot.start, robot.start)


def test_sampling_rejects_unbounded_and_wrong_dimension(problem):
    from kinoweave import JointLimits

    model = problem.models[0]
    model = replace(
        model,
        joints=(
            replace(
                model.joints[0], kind="continuous", limits=JointLimits(velocity=1, acceleration=1)
            ),
        ),
    )
    with pytest.raises(ContractError, match="position ranges"):
        build_prm(ReferenceMotionEvaluator(replace(problem, models=(model,))), "left")
    with pytest.raises(ContractError, match="joint count"):
        build_prm(ReferenceMotionEvaluator(problem), "left", samples=((0, 1),))


def test_missing_candidate_verdicts_cannot_be_accepted_as_clear(problem):
    class BrokenEvaluator(ReferenceMotionEvaluator):
        def motion_validity(self, robot, motions):
            if any(m.start != m.end for m in motions):
                return ()
            return super().motion_validity(robot, motions)

    with pytest.raises(ContractError, match="verdict count"):
        build_prm(BrokenEvaluator(problem), "left", config=PRMConfig(num_samples=0))
