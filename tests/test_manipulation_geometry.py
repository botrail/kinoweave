import math
import subprocess
import sys
from dataclasses import replace

import pytest

from examples.single_arm_pick_place import build_problem
from kinoweave import ContractError, Pose, RobotTrajectory
from kinoweave.io import document_digest, dumps, loads
from kinoweave.manipulation import (
    MANIPULATION_CHECKS,
    ContactBody,
    GraspAttachment,
    HeldObject,
    JointTrace,
    ManipulationMode,
    ManipulationTimeline,
    ModeEvaluator,
    ModeGeometry,
    SingleArmPlanner,
    evaluate_skill,
    evaluate_trace,
    validate_manipulation,
    verify_manipulation,
)
from kinoweave.roadmap import PRMConfig
from kinoweave.search.reference import JointGraph


@pytest.fixture(scope="module")
def pick_place():
    problem = build_problem()
    planner = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0, neighbors=8))
    result = planner.plan()
    assert result.status == "success", result
    return problem, result, planner


def held_mode(problem):
    return ManipulationMode(
        name="held", states=(problem.skeleton.skills[0].options[0].transitions[0].after,)
    )


def test_generated_plan_has_connectors_skills_and_independent_evidence(pick_place):
    problem, result, planner = pick_place
    assert len(result.plan.motions) == 3
    assert [s.skill for s in result.plan.skills] == ["pick", "place"]
    assert planner.roadmap.graph.vertices and planner.statistics["hits"] > 0
    assert result.provenance.backend.device == "cpu"
    assert MANIPULATION_CHECKS <= set(result.validation.checks)
    assert validate_manipulation(problem, result).status == "passed"
    restored = loads(dumps(result))
    restored.check_problem(problem)
    timeline = ManipulationTimeline(problem, restored)
    assert timeline.configurations(0)["gantry"] == problem.scene.robots[0].start
    assert timeline.configurations(result.makespan)["gantry"] == problem.scene.robots[0].goal
    assert timeline.mode(result.makespan).states == problem.goal_mode.states
    for scheduled in result.plan.skills:
        skill = next(s for s in problem.skeleton.skills if s.name == scheduled.skill)
        trace = skill.options[0].trace
        time = scheduled.start + 0.37 * trace.duration
        assert timeline.configurations(time)["gantry"] == pytest.approx(
            evaluate_trace(trace, 0.37 * trace.duration)["gantry"].positions
        )


def test_placed_held_masks_and_shared_fk(pick_place):
    problem, _, _ = pick_place
    # The robot clears the pillar here; its carried object would touch it.
    graph = JointGraph(
        robot="gantry",
        joint_names=("x", "y", "z"),
        vertices=((0, 0, 0.45), (0, 0, 0.46)),
        edges=((0, 1),),
        start=0,
        goal=1,
    )
    with ModeEvaluator(problem, batch_size=2) as evaluator:
        placed = evaluator.mask(graph, problem.initial_mode)
        misses = evaluator.cache_stats["misses"]
        carried = evaluator.mask(graph, held_mode(problem))
        assert placed.node_flags == (True, True)
        assert placed.edge_flags == (True,)
        assert carried.node_flags == (False, True)
        assert carried.edge_flags == (False,)
        assert placed.path(graph, 0, 1) == (0, 1)
        assert carried.path(graph, 0, 1) is None
        assert evaluator.cache_stats["misses"] == misses
        assert evaluator.mask(graph, problem.initial_mode) is placed
        with pytest.raises(ContractError, match="another graph"):
            placed.path(replace(graph, vertices=((0, 0, 0.6), (0, 0, 0.7))), 0, 1)
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.mask(graph, problem.initial_mode)


def test_mask_rejects_collision_inside_edge_with_valid_endpoints():
    problem = build_problem()
    problem = replace(
        problem, scene=replace(problem.scene, config=replace(problem.scene.config, time_step=2))
    )
    graph = JointGraph(
        robot="gantry",
        joint_names=("x", "y", "z"),
        vertices=((-0.2, 0, 0.45), (0.2, 0, 0.45)),
        edges=((0, 1),),
        start=0,
        goal=1,
    )
    with ModeEvaluator(problem) as evaluator:
        placed = evaluator.mask(graph, problem.initial_mode)
        held = evaluator.mask(graph, held_mode(problem))
        assert placed.node_flags == held.node_flags == (True, True)
        assert placed.edge_flags == (True,)
        assert held.edge_flags == (False,)


def test_object_fk_composes_base_link_grasp_and_object_local_geometry():
    problem = build_problem()
    h = math.sqrt(0.5)
    robot = replace(
        problem.scene.robots[0], base_pose=Pose(position=(2, 3, 1), quaternion=(0, 0, h, h))
    )
    problem = replace(problem, scene=replace(problem.scene, robots=(robot,)))
    obj = replace(
        problem.objects[0], spheres=(replace(problem.objects[0].spheres[0], center=(0.2, 0, 0)),)
    )
    problem = replace(problem, objects=(obj,))
    state = HeldObject(
        object="part",
        grasps=(
            GraspAttachment(
                robot="gantry",
                link="tool",
                pose=Pose(position=(0.1, 0, 0), quaternion=(0, 0, h, h)),
            ),
        ),
    )
    geometry = ModeGeometry(problem)
    fk = geometry.fk("gantry", (0.3, 0.4, 0.5))
    frame, issues = geometry.object_frame(
        state, {"gantry": dict(zip(fk.link_names, fk.poses, strict=True))}
    )
    assert not issues
    assert frame.pose.position == pytest.approx((1.6, 3.4, 1.5))
    assert frame.spheres[0].center == pytest.approx((1.4, 3.4, 1.5))


def test_support_and_contact_are_scoped_and_penetration_is_capped():
    problem = build_problem()
    geometry = ModeGeometry(problem)
    q = {"gantry": (-0.3, 0, 0.16)}
    pair = (
        ContactBody(kind="robot", name="gantry", link="tool"),
        ContactBody(kind="object", name="part"),
    )
    assert not geometry.snapshot(problem.initial_mode, q).valid
    assert geometry.snapshot(problem.initial_mode, q, contacts=(pair,)).valid
    wrong_link = (replace(pair[0], link="base"), pair[1])
    assert not geometry.snapshot(problem.initial_mode, q, contacts=(wrong_link,)).valid
    # Move a fingertip into the object. Named contact permission cannot hide depth.
    assert not geometry.snapshot(
        problem.initial_mode, {"gantry": (-0.31, 0, 0.16)}, contacts=(pair,)
    ).valid
    for z in (0.07, 0.09):
        state = replace(problem.initial_mode.states[0], pose=Pose(position=(-0.3, 0, z)))
        mode = replace(problem.initial_mode, states=(state,))
        assert any(i.code == "object_support" for i in geometry.snapshot(mode, q).issues)
    source, *rest = problem.scene.obstacles
    tilted = replace(
        source, pose=replace(source.pose, quaternion=(math.sin(0.01), 0, 0, math.cos(0.01)))
    )
    tilted_problem = replace(problem, scene=replace(problem.scene, obstacles=(tilted, *rest)))
    assert any(
        i.code == "unsupported_support"
        for i in ModeGeometry(tilted_problem).snapshot(tilted_problem.initial_mode, q).issues
    )


def test_dual_grasp_closure_and_local_boundary():
    from kinoweave.backends._description import UnsupportedModelError

    problem = build_problem()
    first = problem.scene.robots[0]
    second = replace(first, name="second", base_pose=Pose(position=(0.2, 0, 0)))
    problem = replace(problem, scene=replace(problem.scene, robots=(first, second)))
    state = held_mode(problem).states[0]
    other = GraspAttachment(robot="second", link="tool", pose=Pose(position=(-0.2, 0, -0.08)))
    dual = replace(state, grasps=(*state.grasps, other))
    mode = ManipulationMode(name="dual", states=(dual,))
    q = {"gantry": (-0.3, 0, 0.55), "second": (-0.3, 0, 0.55)}
    geometry = ModeGeometry(problem)
    assert not any(i.code == "grasp_closure" for i in geometry.snapshot(mode, q).issues)
    q["second"] = (-0.29, 0, 0.55)
    assert any(i.code == "grasp_closure" for i in geometry.snapshot(mode, q).issues)
    with pytest.raises(UnsupportedModelError):
        geometry.snapshot(mode, {"gantry": q["gantry"]}, local_robot="gantry")
    # A foreign held object does not become an immobile obstacle in a local mask.
    foreign = replace(state, grasps=(other,))
    foreign_mode = replace(mode, states=(foreign,))
    assert geometry.snapshot(foreign_mode, {"gantry": q["gantry"]}, local_robot="gantry").valid


def test_skill_contacts_boundaries_rates_and_teleportation():
    problem = build_problem()
    pick = problem.skeleton.skills[0]
    option = pick.options[0]
    assert evaluate_skill(problem, problem.initial_mode, pick, option).validation.status == "passed"
    no_permission = replace(option, contacts=())
    broken = replace(pick, options=(no_permission,))
    p = replace(
        problem, skeleton=replace(problem.skeleton, skills=(broken, problem.skeleton.skills[1]))
    )
    assert evaluate_skill(p, p.initial_mode, broken, no_permission).validation.status == "failed"
    # The original attachment is continuous; the replacement moves the part 1 cm at grasp.
    old = option.transitions[0]
    bad_grasp = replace(old.after.grasps[0], pose=Pose(position=(0, 0.01, -0.08)))
    shifted = replace(old.after, grasps=(bad_grasp,))
    bad_option = replace(option, transitions=(replace(old, after=shifted),))
    bad_skill = replace(pick, options=(bad_option,))
    p = replace(
        problem, skeleton=replace(problem.skeleton, skills=(bad_skill, problem.skeleton.skills[1]))
    )
    edge = evaluate_skill(p, p.initial_mode, bad_skill, bad_option)
    assert edge.validation.status == "failed"
    assert any(i.code == "object_teleport" for i in edge.validation.issues)
    linear = replace(option, trace=replace(option.trace, interpolation="linear"))
    linear_skill = replace(pick, options=(linear,))
    p = replace(
        problem,
        skeleton=replace(problem.skeleton, skills=(linear_skill, problem.skeleton.skills[1])),
    )
    report = evaluate_skill(p, p.initial_mode, linear_skill, linear).validation
    assert any(i.code == "linear_velocity_jump" for i in report.issues)
    with pytest.raises(ContractError, match="belong"):
        evaluate_skill(problem, problem.initial_mode, linear_skill, linear)


def test_reverification_rejects_claims_and_resource_exhaustion(pick_place):
    problem, result, _ = pick_place
    report = validate_manipulation(problem, result, max_samples=2)
    assert report.status == "failed"
    assert any(i.code == "manipulation_evaluation" for i in report.issues)
    # Existing success assertions are ignored when geometry changes (and identity is rebound).
    pillar = replace(
        problem.scene.obstacles[2],
        pose=Pose(position=(0, 0, 0.4)),
        shape=replace(problem.scene.obstacles[2].shape, size=(1.4, 1.4, 0.8)),
    )
    p = replace(
        problem, scene=replace(problem.scene, obstacles=(*problem.scene.obstacles[:2], pillar))
    )
    claimed = replace(
        result, provenance=replace(result.provenance, problem_digest=document_digest(p))
    )
    checked = verify_manipulation(p, claimed)
    assert checked.status == "validation_failed" and checked.plan is None
    timeline = ManipulationTimeline(problem, result)
    for side in ("before", "after"):
        with pytest.raises(ContractError):
            timeline.contacts(-1, side=side)
    scheduled = result.plan.skills[0]
    option = problem.skeleton.skills[0].options[0]
    contact_end = scheduled.start + option.contacts[0].end
    pair = (option.contacts[0].first, option.contacts[0].second)
    assert pair in timeline.contacts(contact_end, side="before")
    assert pair not in timeline.contacts(contact_end, side="after")


def test_absolute_schedule_rejects_collapsed_source_boundary(pick_place):
    problem, result, _ = pick_place
    pick, place = problem.skeleton.skills
    option = pick.options[0]
    track = option.trace.tracks[0]
    option = replace(
        option,
        trace=replace(
            option.trace,
            times=(0, 1e-20, 1, 2.5),
            tracks=(replace(track, positions=(track.positions[0], *track.positions)),),
        ),
    )
    pick = replace(pick, options=(option,))
    problem = replace(problem, skeleton=replace(problem.skeleton, skills=(pick, place)))
    claimed = replace(
        result, provenance=replace(result.provenance, problem_digest=document_digest(problem))
    )
    claimed.check_problem(problem)  # Structural exchange still accepts the explicit trace.
    checked = validate_manipulation(problem, claimed)
    assert checked.status == "failed"
    assert "collapsed" in checked.issues[0].message


def test_constructor_limits_and_nonuniform_trace():
    problem = build_problem()
    planner = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0), max_states=1)
    assert planner.plan().status == "timeout"
    tiny = replace(
        problem,
        scene=replace(problem.scene, config=replace(problem.scene.config, time_limit=1e-15)),
    )
    assert SingleArmPlanner(tiny).plan().status == "timeout"
    from examples.manipulation_contract import build_problem as build_multi

    assert SingleArmPlanner(build_multi()).plan().status == "unsupported"
    trace = JointTrace(
        times=(0, 1, 4),
        interpolation="quintic_stop",
        tracks=(RobotTrajectory(robot="r", joint_names=("x",), positions=((0,), (1,), (3,))),),
    )
    assert evaluate_trace(trace, 2.5)["r"].positions == (2,)
    assert evaluate_trace(trace, 1, side="before")["r"].velocities == (0,)
    assert evaluate_trace(trace, 1, side="after")["r"].velocities == (0,)


def test_constructor_backtracks_options_and_reports_finite_failure():
    problem = build_problem()
    pick, place = problem.skeleton.skills
    original = pick.options[0]
    track = original.trace.tracks[0]
    stranded = replace(
        original,
        name="stranded",
        trace=replace(
            original.trace,
            tracks=(replace(track, positions=(*track.positions[:-1], track.positions[-2])),),
        ),
        contacts=tuple(replace(c, end=original.trace.duration) for c in original.contacts),
    )
    # Grasping without lifting is a valid skill, but its exit has no free carry edge.
    pick = replace(pick, options=(stranded, original))
    problem = replace(problem, skeleton=replace(problem.skeleton, skills=(pick, place)))
    assert (
        evaluate_skill(problem, problem.initial_mode, pick, stranded).validation.status == "passed"
    )
    result = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan()
    assert result.status == "success" and result.plan.skills[0].option == original.name
    pick = replace(pick, options=(stranded,))
    problem = replace(problem, skeleton=replace(problem.skeleton, skills=(pick, place)))
    assert (
        SingleArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan().status
        == "no_solution_in_discretization"
    )


def test_stretch_skill_uses_minimum_duration_and_scales_event_clock():
    problem = build_problem()
    pick, place = problem.skeleton.skills
    pick = replace(pick, timing="stretch", minimum_duration=5)
    problem = replace(problem, skeleton=replace(problem.skeleton, skills=(pick, place)))
    edge = evaluate_skill(problem, problem.initial_mode, pick, pick.options[0])
    assert edge.duration == 5 and edge.validation.status == "passed"
    result = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan()
    assert result.status == "success"
    scheduled = result.plan.skills[0]
    assert scheduled.end - scheduled.start == 5
    timeline = ManipulationTimeline(problem, result)
    event_time = scheduled.start + 2
    assert not isinstance(timeline.mode(event_time, side="before").states[0], HeldObject)
    assert isinstance(timeline.mode(event_time, side="after").states[0], HeldObject)


def test_cpu_path_does_not_import_optional_sdks():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        blocked = {'torch','numpy','curobo','warp','viser','mujoco','isaacsim','botrail'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from examples.single_arm_pick_place import build_problem
from kinoweave.manipulation import SingleArmPlanner
from kinoweave.roadmap import PRMConfig
assert SingleArmPlanner(build_problem(), prm=PRMConfig(num_samples=0)).plan().status == 'success'
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
