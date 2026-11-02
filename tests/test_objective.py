import math
from dataclasses import replace

import pytest

from kinoweave import ContractError, Joint, Pose
from kinoweave.io import document_digest
from kinoweave.manipulation import (
    ManipulationResult,
    MetricConfig,
    evaluate_objective,
)


@pytest.mark.parametrize(
    "metric,expected,unit",
    [
        ("joint_l1", 8, "joint_units"),
        ("joint_l2", 4 * math.sqrt(2), "joint_units"),
        ("tool_path", 4 * math.sqrt(2), "m"),
        ("sphere_volume_path", (4 * math.pi / 3) * 0.1**3 * 4 * math.sqrt(2), "m^4"),
    ],
)
def test_all_distance_metrics_and_clock_invariance(refinement_cases, metric, expected, unit):
    problem, result = refinement_cases(metric=metric, weights=(2, 3, 0.5))
    score = evaluate_objective(problem, result)
    assert score.makespan == 4.5 and score.completion == 9
    assert score.distance == pytest.approx(expected)
    assert score.total == pytest.approx(2 * 4.5 + 3 * 9 + 0.5 * expected)
    assert score.distance_unit == unit
    stretched = replace(
        result.plan,
        makespan=8.5,
        motions=tuple(
            replace(m, end=8.5, trace=replace(m.trace, times=tuple(t * 2 for t in m.trace.times)))
            for m in result.plan.motions
        ),
    )
    candidate = ManipulationResult(status="candidate", plan=stretched, provenance=result.provenance)
    later = evaluate_objective(problem, candidate)
    assert later.distance == score.distance
    assert later.geometry_samples == score.geometry_samples
    assert later.makespan == 8.5


def test_completion_includes_stationary_skills_but_excludes_terminal_holds(refinement_cases):
    from kinoweave.manipulation import ActionOrder, ScheduledMotion, ScheduledSkill

    problem, result = refinement_cases(weights=(2, 3, 0.5))
    finish = replace(
        problem.skeleton.skills[1],
        name="finish_beta",
        options=(
            replace(
                problem.skeleton.skills[1].options[0],
                trace=replace(
                    problem.skeleton.skills[1].options[0].trace,
                    tracks=(
                        replace(
                            problem.skeleton.skills[1].options[0].trace.tracks[0],
                            positions=((1, 3), (1, 3)),
                        ),
                    ),
                ),
            ),
        ),
    )
    problem = replace(
        problem,
        skeleton=replace(
            problem.skeleton,
            skills=(*problem.skeleton.skills, finish),
            precedence=(ActionOrder(before="beta", after="finish_beta"),),
        ),
    )
    holds = []
    for robot, a, b in (("alpha", 4.5, 6), ("beta", 4.5, 5), ("beta", 5.5, 6)):
        original = next(m for m in result.plan.motions if m.trace.tracks[0].robot == robot)
        last = original.trace.tracks[0].positions[-1]
        trace = replace(
            original.trace,
            times=(0, b - a),
            tracks=(replace(original.trace.tracks[0], positions=(last, last)),),
        )
        holds.append(ScheduledMotion(start=a, end=b, trace=trace, kind="wait"))
    plan = replace(
        result.plan,
        makespan=6,
        motions=(*result.plan.motions, *holds),
        skills=(
            *result.plan.skills,
            ScheduledSkill(skill=finish.name, option="inspect", start=5, end=5.5),
        ),
    )
    bound = ManipulationResult(
        status="candidate",
        plan=plan,
        provenance=replace(result.provenance, problem_digest=document_digest(problem)),
    )
    score = evaluate_objective(problem, bound)
    assert dict(score.robot_completion) == {"alpha": 4.5, "beta": 5.5}
    assert score.makespan == 5.5 and score.schedule_duration == 6
    assert score.completion == 10 and score.total == 45


def test_tool_binding_requires_unambiguous_leaf_and_metric_budgets(refinement_cases):
    problem, result = refinement_cases(metric="tool_path")
    model = problem.scene.models[0]
    branched = replace(
        model,
        links=(*model.links, "camera"),
        joints=(
            *model.joints,
            Joint(
                name="camera_fixed",
                kind="fixed",
                parent="carriage",
                child="camera",
                origin=Pose(position=(0, 0, 0.1)),
            ),
        ),
    )
    problem = replace(problem, scene=replace(problem.scene, models=(branched,)))
    result = replace(
        result, provenance=replace(result.provenance, problem_digest=document_digest(problem))
    )
    with pytest.raises(ContractError, match="select a link"):
        evaluate_objective(problem, result)
    config = MetricConfig(tool_links=(("alpha", "tool"), ("beta", "tool")))
    assert evaluate_objective(problem, result, config=config).distance == pytest.approx(
        4 * math.sqrt(2)
    )
    with pytest.raises(ContractError, match="metric_sample_limit"):
        evaluate_objective(problem, result, config=replace(config, max_samples=2))
    with pytest.raises(ContractError, match="unknown link"):
        evaluate_objective(problem, result, config=MetricConfig(tool_links=(("alpha", "absent"),)))


def test_makespan_is_latest_robot_completion_without_global_wait_padding(refinement_cases):
    problem, result = refinement_cases()
    # Explicitly append native goal holds; the objective is independent of
    # the caller's extra storage/playback duration.
    from kinoweave.manipulation import ScheduledMotion

    extra = []
    for m in result.plan.motions:
        track = m.trace.tracks[0]
        extra.append(
            ScheduledMotion(
                start=4.5,
                end=10,
                kind="wait",
                trace=replace(
                    m.trace,
                    times=(0, 5.5),
                    tracks=(replace(track, positions=(track.positions[-1],) * 2),),
                ),
            )
        )
    candidate = ManipulationResult(
        status="candidate",
        provenance=result.provenance,
        plan=replace(result.plan, makespan=10, motions=(*result.plan.motions, *extra)),
    )
    value = evaluate_objective(problem, candidate)
    assert value.total == 4.5 and value.schedule_duration == 10


def test_rotating_tool_and_volume_paths_use_geometric_samples(refinement_cases):
    from kinoweave import JointLimits

    problem, result = refinement_cases(metric="tool_path")
    model = problem.scene.models[0]
    model = replace(
        model,
        links=("base", "tool", "tip"),
        joint_order=("theta",),
        joints=(
            Joint(
                name="theta",
                kind="revolute",
                parent="base",
                child="tool",
                axis=(0, 0, 1),
                limits=JointLimits(lower=-4, upper=4, velocity=10, acceleration=30),
            ),
            Joint(
                name="tip_fixed",
                kind="fixed",
                parent="tool",
                child="tip",
                origin=Pose(position=(1, 0, 0)),
            ),
        ),
        collision_spheres=(replace(model.collision_spheres[0], center=(1, 0, 0)),),
    )
    robot = replace(
        problem.scene.robots[0],
        start=(0,),
        goal=(math.pi,),
        base_pose=Pose(position=(2, 3, 1), quaternion=(0, 0, math.sqrt(0.5), math.sqrt(0.5))),
    )
    skill = problem.skeleton.skills[0]
    option = skill.options[0]
    trace = replace(
        option.trace,
        tracks=(replace(option.trace.tracks[0], joint_names=("theta",), positions=((0,), (0,))),),
    )
    skill = replace(skill, options=(replace(option, trace=trace),))
    problem = replace(
        problem,
        scene=replace(problem.scene, models=(model,), robots=(robot,)),
        skeleton=replace(problem.skeleton, skills=(skill,)),
    )
    motion = result.plan.motions[0]
    motion = replace(
        motion,
        trace=replace(
            motion.trace,
            times=(0, 4),
            tracks=(
                replace(
                    motion.trace.tracks[0], joint_names=("theta",), positions=((0,), (math.pi,))
                ),
            ),
        ),
    )
    result = replace(
        result,
        plan=replace(result.plan, motions=(motion,), skills=(result.plan.skills[0],)),
        provenance=replace(result.provenance, problem_digest=document_digest(problem)),
    )
    fine = evaluate_objective(problem, result)
    coarse = evaluate_objective(problem, result, config=MetricConfig(max_joint_step=4))
    assert fine.distance == pytest.approx(math.pi, rel=2e-4)
    assert coarse.distance == pytest.approx(2)
    assert fine.resolved_tools == ((robot.name, "tip"),)
    split = replace(
        motion,
        trace=replace(
            motion.trace,
            times=(0, 1, 2, 3, 4),
            tracks=(
                replace(
                    motion.trace.tracks[0], positions=tuple((i * math.pi / 4,) for i in range(5))
                ),
            ),
        ),
    )
    split_result = replace(result, plan=replace(result.plan, motions=(split,)))
    split_value = evaluate_objective(problem, split_result)
    assert split_value.distance == fine.distance
    assert split_value.geometry_samples == fine.geometry_samples
    # Rotating the named link origin itself has zero translational tool cost.
    assert (
        evaluate_objective(
            problem, result, config=MetricConfig(tool_links=((robot.name, "tool"),))
        ).distance
        == 0
    )
    problem = replace(problem, cost=replace(problem.cost, distance_metric="sphere_volume_path"))
    result = replace(
        result, provenance=replace(result.provenance, problem_digest=document_digest(problem))
    )
    assert evaluate_objective(problem, result).distance == pytest.approx(
        (4 * math.pi / 3) * 0.1**3 * math.pi, rel=2e-4
    )
