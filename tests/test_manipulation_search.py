import subprocess
import sys
from collections import deque
from dataclasses import replace
from itertools import product

import pytest

from examples.two_arm_handover import build_problem
from kinoweave import (
    Joint,
    JointLimits,
    LinkSphere,
    PlannerConfig,
    PlanningProblem,
    Pose,
    RobotInstance,
    RobotModel,
    RobotTrajectory,
)
from kinoweave.io import dumps, loads
from kinoweave.manipulation import (
    HeldObject,
    JointTrace,
    ManipulationMode,
    ManipulationProblem,
    ManipulationSearchConfig,
    ManipulationTimeline,
    ModeEvaluator,
    MultiArmPlanner,
    ScheduleSkeleton,
    Skill,
    SkillOption,
    evaluate_trace,
    validate_manipulation,
)
from kinoweave.manipulation.search import _Node, complete, search
from kinoweave.manipulation.space import TaskSpace
from kinoweave.manipulation.task import ActionClock
from kinoweave.roadmap import PRMConfig
from kinoweave.search.reference import JointGraph
from kinoweave.search.space import Budget, SearchConfig


def task_problem(scene, skills):
    return ManipulationProblem(
        scene=scene,
        objects=(),
        initial_mode=ManipulationMode(name="initial", states=()),
        goal_mode=ManipulationMode(name="goal", states=()),
        skeleton=ScheduleSkeleton(skills=skills),
    )


def holds(scene, duration=0.5):
    return tuple(
        Skill(
            name=r.name,
            robots=(r.name,),
            minimum_duration=duration,
            options=(
                SkillOption(
                    name="hold",
                    trace=JointTrace(
                        times=(0, duration),
                        interpolation="quintic_stop",
                        tracks=(
                            RobotTrajectory(
                                robot=r.name,
                                joint_names=scene.models[0].joint_order,
                                positions=(r.goal, r.goal),
                            ),
                        ),
                    ),
                ),
            ),
        )
        for r in scene.robots
    )


def make_space(problem, graphs, evaluator, **options):
    config = ManipulationSearchConfig(**options)
    return TaskSpace(
        evaluator,
        graphs,
        config,
        Budget(
            120, SearchConfig(max_states=config.max_states, max_constraints=config.max_constraints)
        ),
    )


@pytest.fixture(scope="module")
def handover():
    problem = build_problem()
    planner = MultiArmPlanner(problem, prm=PRMConfig(num_samples=0, neighbors=8))
    result = planner.plan()
    assert result.status == "success", result
    graphs = tuple(r.graph for r in planner.roadmaps)
    with ModeEvaluator(problem) as evaluator:
        space = make_space(problem, graphs, evaluator)
        path = search(space)
        yield problem, result, planner, space, path


def test_handover_full_plan_modes_and_terminal_occupancy(handover):
    problem, result, planner, space, path = handover
    assert [s.skill for s in result.plan.skills] == ["pick_alpha", "handover", "place_beta"]
    assert planner.statistics["prefetch_calls"] > 0
    assert planner.statistics["prefetched_pairs"] > 0
    assert result.provenance.backend.device == "cpu"
    assert validate_manipulation(problem, result).status == "passed"
    restored = loads(dumps(result))
    restored.check_problem(problem)
    timeline = ManipulationTimeline(problem, restored)
    scheduled = result.plan.skills[1]
    state = timeline.mode(scheduled.start + 1.25).states[0]
    assert isinstance(state, HeldObject) and len(state.grasps) == 2
    assert {g.robot for g in state.grasps} == {"alpha", "beta"}
    assert timeline.mode(result.makespan).states == problem.goal_mode.states
    assert all(
        q == r.goal
        for q, r in zip(
            timeline.configurations(result.makespan).values(), problem.scene.robots, strict=True
        )
    )
    # Alpha returns while beta is still carrying/placing, rather than a globally serial plan.
    place = result.plan.skills[2]
    assert any(
        m.trace.tracks[0].robot == "alpha" and m.start < place.end and m.end > place.start
        for m in result.plan.motions
    )
    assert len(path) * space.dt == result.makespan


def test_bimanual_choice_is_atomic_and_ongoing_cannot_be_interrupted(handover):
    _, _, _, space, path = handover
    state = next(s for s, choices in path if any(a.skill == 1 and a.phase == 0 for a in choices))
    prepared = space.prepare(state)
    start = next(i for i, a in enumerate(prepared.table.actions) if a.skill == 1 and a.phase == 0)
    hold = prepared.holds[1]
    assert space.selection(prepared, (start, hold)) is None
    assigned = space.selection(prepared, (start,))
    assert assigned == {0: start, 1: start}
    next_state = space.advance(state, prepared, (start, start))
    assert next_state.running[0] == next_state.running[1]
    ongoing = space.prepare(next_state)
    assert ongoing.options[0] == ongoing.options[1] and len(ongoing.options[0]) == 1
    assert ongoing.table.actions[ongoing.options[0][0]].phase == 1


def test_prefetched_prefixes_match_full_product_without_geometry_in_recursion(
    handover, monkeypatch
):
    _, _, _, space, path = handover
    state = next(s for s, choices in path if any(a.skill == 1 and a.phase == 0 for a in choices))
    prepared = space.prepare(state)
    node = _Node(state, None, None, (0, 0), (1, 0), deque())
    expected = set()
    for choices in product(*prepared.options):
        target = space.advance(state, prepared, choices)
        if target is not None:
            expected.add(target)
    calls = space.stats["prefetch_calls"]

    def forbidden(*args, **kwargs):
        raise AssertionError("geometry called inside recursive selection")

    monkeypatch.setattr(space.builder, "build", forbidden)
    monkeypatch.setattr(space.evaluator, "_device_geometry", forbidden)
    monkeypatch.setattr(space.evaluator.cpu, "fk", forbidden)
    observed = set()
    for prefix in product(*(prepared.options[r] for r in node.order)):
        choices = complete(space, node, prepared, prefix)
        if choices is not None:
            observed.add(space.advance(state, prepared, choices))
    assert observed == expected and expected
    assert space.stats["prefetch_calls"] == calls


def test_joint_tables_match_independent_full_scene_oracle(handover):
    _, _, _, space, path = handover
    states = [
        path[0][0],
        next(s for s, _ in path if any(r is not None and r.skill == 0 for r in s.running)),
        next(s for s, choices in path if any(a.skill == 1 and a.phase == 0 for a in choices)),
    ]
    for state in states:
        prepared = space.prepare(state)
        for choices in product(*prepared.options):
            actions = tuple(dict.fromkeys(prepared.table.actions[i] for i in choices))
            participants = [r for a in actions for r in a.robots]
            if sorted(participants) != list(range(len(space.names))):
                assert space.selection(prepared, choices) is None
                continue
            clocks = [ActionClock(space, state, a) for a in actions]
            expected = True
            for time, side in prepared.table.points:
                configurations = {}
                mode = {s.object: s for s in state.mode.states}
                contacts = []
                for clock in clocks:
                    configurations.update(clock.configurations(time, side))
                    contacts.extend(clock.contacts(time, "after" if time == 0 else side))
                changes = sorted(
                    (
                        e
                        for c in clocks
                        for e in c.events
                        if e[0] < time or (e[0] == time and side == "after")
                    ),
                    key=lambda e: e[:3],
                )
                for _, _, _, change in changes:
                    mode[change.after.object] = change.after
                whole = replace(state.mode, states=tuple(mode.values()))
                if not space.evaluator.cpu.snapshot(
                    whole, configurations, contacts=tuple(contacts)
                ).valid:
                    expected = False
                    break
            actual = all(prepared.table.valid[i] for i in choices) and all(
                prepared.table.compatible(a, b) for a, b in product(choices, choices)
            )
            assert actual == expected


@pytest.mark.parametrize("case", ["retreat", "corridor"])
def test_lazy_and_exhaustive_agree_with_task_progress(coordination_cases, case):
    scene, graphs = coordination_cases[case]
    problem = task_problem(scene, holds(scene))
    results = []
    for algorithm in ("lacam", "exhaustive"):
        planner = MultiArmPlanner(
            problem, search_config=ManipulationSearchConfig(algorithm=algorithm)
        )
        results.append(planner.plan(graphs))
        if case == "retreat":
            assert planner.statistics["inheritance_calls"] > 0 or algorithm == "exhaustive"
    assert {r.status for r in results} == (
        {"success"} if case == "retreat" else {"no_solution_in_discretization"}
    )
    if case == "retreat":
        assert results[0].plan.skills and results[0].validation.status == "passed"


def independent_durations():
    model = RobotModel(
        name="slider",
        links=("base", "tool"),
        joint_order=("x",),
        joints=(
            Joint(
                name="x",
                kind="prismatic",
                parent="base",
                child="tool",
                axis=(1, 0, 0),
                limits=JointLimits(lower=-1, upper=2, velocity=10, acceleration=30),
            ),
        ),
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.03),),
    )
    scene = PlanningProblem(
        name="independent_source_clocks",
        models=(model,),
        robots=tuple(
            RobotInstance(
                name=name,
                model=model.name,
                start=(0,),
                goal=(1,),
                base_pose=Pose(position=(0, y, 0)),
            )
            for name, y in (("alpha", 0), ("beta", 3))
        ),
        config=PlannerConfig(
            backend="reference", device="cpu", time_step=1, validation_time_step=0.1, time_limit=120
        ),
    )
    skills = tuple(
        Skill(
            name=r.name,
            robots=(r.name,),
            minimum_duration=duration,
            options=(
                SkillOption(
                    name="source",
                    trace=JointTrace(
                        times=(0, duration),
                        interpolation="quintic_stop",
                        tracks=(
                            RobotTrajectory(
                                robot=r.name, joint_names=("x",), positions=((0,), (1,))
                            ),
                        ),
                    ),
                ),
            ),
        )
        for r, duration in zip(scene.robots, (1.4, 3.2), strict=True)
    )
    graphs = tuple(
        JointGraph(
            robot=r.name,
            joint_names=("x",),
            vertices=((0,), (1,)),
            edges=((0, 1),),
            start=0,
            goal=1,
        )
        for r in scene.robots
    )
    return task_problem(scene, skills), graphs


def test_independent_nonintegral_skills_preserve_clocks_and_explicit_padding():
    problem, graphs = independent_durations()
    result = MultiArmPlanner(problem).plan(graphs)
    assert result.status == "success", result
    assert [(s.start, s.end) for s in result.plan.skills] == [(0, 1.4), (0, 3.2)]
    assert result.makespan == 4
    timeline = ManipulationTimeline(problem, result)
    source = problem.skeleton.skills[0].options[0].trace
    assert timeline.configurations(0.7)["alpha"] == evaluate_trace(source, 0.7)["alpha"].positions
    assert any(m.start == 1.4 and m.end == 2 and m.kind == "wait" for m in result.plan.motions)
    assert timeline.states(1.1)["beta"].velocities[0] > 0
    # A stretch policy is allowed to slow the whole clock to a tick boundary.
    stretched = replace(problem.skeleton.skills[0], timing="stretch", minimum_duration=2.2)
    p = replace(
        problem, skeleton=replace(problem.skeleton, skills=(stretched, problem.skeleton.skills[1]))
    )
    result = MultiArmPlanner(p).plan(graphs)
    assert result.status == "success" and result.plan.skills[0].end == 3


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"max_states": 1}, "state_limit"),
        ({"max_constraints": 1}, "constraint_limit"),
        ({"max_candidates_per_robot": 1}, "candidate_limit"),
        ({"max_state_pairs": 1}, "state_pair_limit"),
        ({"max_state_bytes": 1}, "state_geometry_limit"),
        ({"max_skill_ticks": 1}, "skill_tick_limit"),
    ],
)
def test_resource_limits_are_not_reported_as_exhaustion(changes, reason):
    problem, graphs = independent_durations()
    result = MultiArmPlanner(problem, search_config=ManipulationSearchConfig(**changes)).plan(
        graphs
    )
    assert result.status == "timeout" and result.termination_reason == reason


def test_table_retention_and_graph_binding(coordination_cases):
    scene, graphs = coordination_cases["retreat"]
    problem = task_problem(scene, holds(scene))
    for count in (0, 1):
        planner = MultiArmPlanner(
            problem,
            search_config=ManipulationSearchConfig(
                max_cached_tables=count, max_cached_table_pairs=32
            ),
        )
        result = planner.plan(graphs)
        assert result.status == "success"
        assert planner.statistics["peak_cached_table_pairs"] <= 32
    bad = replace(graphs[0], start=1)
    assert MultiArmPlanner(problem).plan((bad, graphs[1])).status == "invalid_problem"
    tiny = replace(problem, scene=replace(scene, config=replace(scene.config, time_limit=1e-15)))
    assert MultiArmPlanner(tiny).plan(graphs).status == "timeout"


def test_core_search_runs_with_optional_sdks_blocked():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        blocked={'torch','numpy','curobo','warp','viser','mujoco','isaacsim','botrail'}
        if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from examples.two_arm_handover import build_problem
from kinoweave.manipulation import MultiArmPlanner
from kinoweave.roadmap import PRMConfig
r=MultiArmPlanner(build_problem(),prm=PRMConfig(num_samples=0)).plan()
assert r.status=='success',r
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)


def test_running_skill_forces_yield_and_rejects_incompatible_simultaneous_start(retreat_problem):
    scene = replace(
        retreat_problem,
        robots=(
            retreat_problem.robots[0],
            replace(retreat_problem.robots[1], start=(0, 0), goal=(0, 0)),
        ),
        config=replace(retreat_problem.config, time_step=1, time_limit=120),
    )
    alpha, beta = scene.robots
    skill_a = Skill(
        name="cross",
        robots=(alpha.name,),
        minimum_duration=3,
        options=(
            SkillOption(
                name="source",
                trace=JointTrace(
                    times=(0, 3),
                    interpolation="quintic_stop",
                    tracks=(
                        RobotTrajectory(
                            robot=alpha.name,
                            joint_names=("x", "y"),
                            positions=(alpha.start, alpha.goal),
                        ),
                    ),
                ),
            ),
        ),
    )
    skill_b = holds(replace(scene, robots=(beta,)), duration=2)[0]
    problem = task_problem(scene, (skill_a, skill_b))
    graphs = (
        JointGraph(
            robot=alpha.name,
            joint_names=("x", "y"),
            vertices=(alpha.start, alpha.goal),
            edges=((0, 1),),
            start=0,
            goal=1,
        ),
        JointGraph(
            robot=beta.name,
            joint_names=("x", "y"),
            vertices=((0, 0), (0, 1)),
            edges=((0, 1),),
            start=0,
            goal=0,
        ),
    )
    planner = MultiArmPlanner(problem)
    result = planner.plan(graphs)
    assert result.status == "success", result
    assert planner.statistics["backtracks"] > 0
    assert any(
        m.trace.tracks[0].robot == beta.name and (0, 1) in m.trace.tracks[0].positions
        for m in result.plan.motions
    )
    assert validate_manipulation(problem, result).status == "passed"


def test_multiple_background_objects_and_mode_order_do_not_change_goal():
    from kinoweave.manipulation import ObjectSphere, PlacedObject, RigidObject

    problem, graphs = independent_durations()
    objects = tuple(
        RigidObject(name=name, spheres=(ObjectSphere(center=(0, 0, 0), radius=0.02),))
        for name in ("first", "second")
    )
    states = tuple(
        PlacedObject(object=o.name, pose=Pose(position=(10 + i, 0, 0)))
        for i, o in enumerate(objects)
    )
    problem = replace(
        problem,
        objects=objects,
        initial_mode=ManipulationMode(name="i", states=states),
        goal_mode=ManipulationMode(name="g", states=states[::-1]),
    )
    result = MultiArmPlanner(problem).plan(graphs)
    assert result.status == "success", result
    assert validate_manipulation(problem, result).status == "passed"


def test_decimal_tick_effect_is_applied_once_and_bad_sample_types_are_rejected():
    from kinoweave.manipulation import (
        GraspAttachment,
        ModeTransition,
        ObjectSphere,
        PlacedObject,
        RigidObject,
    )

    problem, graphs = independent_durations()
    robot = replace(problem.scene.robots[0], goal=(0,))
    scene = replace(
        problem.scene, robots=(robot,), config=replace(problem.scene.config, time_step=0.1)
    )
    placed = PlacedObject(object="part", pose=Pose(position=(0.1, 0, 0)))
    held = HeldObject(
        object="part", grasps=(GraspAttachment(robot=robot.name, link="tool", pose=placed.pose),)
    )
    skill = Skill(
        name="attach",
        robots=(robot.name,),
        minimum_duration=0.4,
        options=(
            SkillOption(
                name="source",
                preconditions=(placed,),
                trace=JointTrace(
                    times=(0, 0.4),
                    interpolation="quintic_stop",
                    tracks=(
                        RobotTrajectory(
                            robot=robot.name, joint_names=("x",), positions=((0,), (0,))
                        ),
                    ),
                ),
                transitions=(ModeTransition(time=0.3, before=placed, after=held),),
            ),
        ),
    )
    problem = ManipulationProblem(
        scene=scene,
        objects=(RigidObject(name="part", spheres=(ObjectSphere(center=(0, 0, 0), radius=0.02),)),),
        initial_mode=ManipulationMode(name="i", states=(placed,)),
        goal_mode=ManipulationMode(name="g", states=(held,)),
        skeleton=ScheduleSkeleton(skills=(skill,)),
    )
    graph = replace(graphs[0], vertices=((0,),), edges=(), start=0, goal=0)
    result = MultiArmPlanner(problem).plan((graph,))
    assert result.status == "success", result
    timeline = ManipulationTimeline(problem, result)
    assert timeline.mode(0.3, side="before").states == (placed,)
    assert timeline.mode(0.3, side="after").states == (held,)
    assert MultiArmPlanner(problem).plan(samples=[]).status == "invalid_problem"
