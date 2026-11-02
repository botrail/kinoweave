import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from kinoweave import (
    ContractError,
    Joint,
    JointLimits,
    Mimic,
    Planner,
    PlanResult,
    Pose,
    SearchConfig,
    TrackingError,
)
from kinoweave.adapters import (
    AdapterCapabilities,
    ExecutionAdapter,
    MemoryPlayback,
    PlaybackAdapter,
    RobotBinding,
    SceneAdapter,
    UnsupportedCapabilityError,
    check_capabilities,
    check_playback_report,
    take_snapshot,
    validate_plan_for_adapter,
)
from kinoweave.io import document_digest, dumps, loads
from kinoweave.search.reference import JointGraph


@pytest.fixture
def playback_case(coordination_cases):
    problem, graphs = coordination_cases["retreat"]
    with Planner(problem, search=SearchConfig(exact_fallback=False)) as planner:
        plan = planner.plan(graphs)
    assert plan.status == "success"
    return problem, plan


@pytest.fixture
def mimic_case(problem):
    model = problem.models[0]
    follower = Joint(
        name="follower",
        kind="prismatic",
        parent="carriage",
        child="finger",
        axis=(0, 1, 0),
        limits=JointLimits(lower=-2, upper=2, velocity=2, acceleration=4),
        mimic=Mimic(joint="slide", multiplier=-2, offset=0.1),
    )
    mount = Joint(
        name="mount", kind="fixed", parent="finger", child="tip", origin=Pose(position=(0.1, 0, 0))
    )
    model = replace(
        model, links=(*model.links, "finger", "tip"), joints=(*model.joints, follower, mount)
    )
    problem = replace(
        problem,
        models=(model,),
        config=replace(problem.config, time_step=2, validation_time_step=0.1),
    )
    graphs = tuple(
        JointGraph(
            robot=r.name,
            joint_names=model.joint_order,
            vertices=(r.start, r.goal),
            edges=((0, 1),),
            start=0,
            goal=1,
        )
        for r in problem.robots
    )
    with Planner(problem) as planner:
        plan = planner.plan(graphs)
    assert plan.status == "success"
    return problem, plan


def test_roles_snapshot_and_loaded_scene_binding(playback_case):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        assert isinstance(player, SceneAdapter) and isinstance(player, PlaybackAdapter)
        assert not isinstance(player, ExecutionAdapter) and not player.capabilities.execution
        assert take_snapshot(player) == problem
        with pytest.raises(RuntimeError, match="load"):
            player.set_time(0)
        player.load(problem, plan)
        assert player.verification.status == "passed"
        assert player.plan_digest == document_digest(plan)
        observation = player.set_time(2.5)
        snapshot = take_snapshot(player)
        assert tuple(r.start for r in snapshot.robots) == tuple(
            r.positions for r in observation.robots
        )
        assert tuple(r.goal for r in snapshot.robots) == tuple(r.goal for r in problem.robots)
        assert snapshot.models == problem.models and snapshot.obstacles == problem.obstacles
        assert document_digest(snapshot) != document_digest(problem)
        assert player.reset().reference_time == 0
        assert take_snapshot(player) == problem
    with pytest.raises(RuntimeError, match="closed"):
        player.set_time(0)


def test_independent_joint_readback_uses_runtime_slots_and_mimic_derivatives(mimic_case):
    problem, plan = mimic_case
    bindings = (
        RobotBinding(
            robot="right", runtime_robot="platform/B", joint_names=("slide", "mount", "follower")
        ),
        RobotBinding(
            robot="left",
            runtime_robot="platform/A",
            joint_names=("mount", "follower", "slide"),
            runtime_joint_names=("fixed_slot", "finger_drive", "rail_drive"),
        ),
    )
    with MemoryPlayback(problem, bindings=bindings) as player:
        player.load(problem, plan)
        observed = player.set_time(0.5)
        assert tuple(r.robot for r in observed.robots) == ("left", "right")
        assert observed.robots[0].joint_names == ("slide",)
        assert observed.robots[0].positions == pytest.approx((0.0310546875,))
        assert observed.robots[0].velocities == pytest.approx((0.158203125,))
        left = player.frame.robots[0]
        assert left.runtime_robot == "platform/A"
        assert left.joint_names == ("fixed_slot", "finger_drive", "rail_drive")
        assert left.positions == pytest.approx((0, 0.037890625, 0.0310546875))
        assert left.velocities == pytest.approx((0, -0.31640625, 0.158203125))
        assert left.accelerations == pytest.approx((0, -0.84375, 0.421875))
        assert left.link_poses[left.link_names.index("tip")].position == pytest.approx(
            (-0.1689453125, -0.162109375, 0.5)
        )
        assert left.spheres[0].center == pytest.approx((-0.2689453125, -0.2, 0.5))


def test_all_tracks_and_runtime_geometry_commit_atomically(playback_case, monkeypatch):
    import kinoweave.adapters.memory as memory

    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        before = player.frame
        original = memory.forward

        def failed(model, robot, positions):
            if robot.name == "beta":
                raise ContractError("runtime", "injected second-robot failure")
            return original(model, robot, positions)

        monkeypatch.setattr(memory, "forward", failed)
        with pytest.raises(ContractError, match="second-robot"):
            player.set_time(0.5)
        assert player.frame is before


@pytest.mark.parametrize("time", [-0.1, 4.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_seeks_preserve_the_previous_frame(playback_case, time):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        player.set_time(1.25)
        before = player.frame
        with pytest.raises(ContractError):
            player.set_time(time)
        assert player.frame is before


def test_arbitrary_seeks_keep_owned_snapshots_and_exact_knot_stops(playback_case):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        player.set_time(2.75)
        saved = player.frame
        player.set_time(0.25)
        assert saved.observation.reference_time == 2.75
        with pytest.raises(FrozenInstanceError):
            saved.robots[0].positions = ()
        for time in plan.trajectory.times:
            observation = player.set_time(time)
            assert all(all(v == 0 for v in r.velocities) for r in observation.robots)
        player.set_time(plan.makespan)
        assert tuple(r.positions for r in player.frame.observation.robots) == tuple(
            r.goal for r in problem.robots
        )
        player.load(problem, plan)
        assert player.frame.observation.reference_time == 0


def test_plan_handoff_and_recording_preserve_identity_and_time_law(playback_case):
    problem, plan = playback_case
    # Exchange and track ordering must not become positional robot mapping.
    plan = loads(
        dumps(
            replace(
                plan,
                trajectory=replace(plan.trajectory, tracks=tuple(reversed(plan.trajectory.tracks))),
            )
        ),
        expected=PlanResult,
    )
    digest = document_digest(plan)
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        report = player.record(0.3)
        assert report.mode == "kinematic_playback" and report.runtime == player.info
        assert report.runtime.device == "cpu" and report.time_step == 0.25
        assert len(report.observations) == 17
        assert set(plan.trajectory.times) <= set(o.reference_time for o in report.observations)
        assert all(o.time == o.reference_time for o in report.observations)
        assert report.observations[-1].reference_time == plan.makespan
        assert player.frame.observation == report.observations[-1]
        report.check_plan(problem, plan)
        check_playback_report(problem, plan, report, position_tolerance=0, velocity_tolerance=0)
        assert loads(dumps(report)) == report
        assert document_digest(plan) == digest


def test_unsupported_roles_and_model_features_are_explicit(mimic_case):
    problem, plan = mimic_case
    capabilities = MemoryPlayback.capabilities
    with pytest.raises(UnsupportedCapabilityError, match="execution"):
        validate_plan_for_adapter(problem, plan, capabilities, role="execution")
    with pytest.raises(UnsupportedCapabilityError, match="mimic"):
        check_capabilities(problem, replace(capabilities, mimic=False), "playback")
    with pytest.raises(UnsupportedCapabilityError, match="box"):
        check_capabilities(problem, replace(capabilities, shapes=("sphere",)), "playback")
    with pytest.raises(UnsupportedCapabilityError, match="fixed"):
        check_capabilities(problem, replace(capabilities, joint_types=("prismatic",)), "playback")

    class NoScene:
        capabilities = AdapterCapabilities()

        def snapshot(self):
            raise AssertionError("unsupported role must be rejected before calling")

    with pytest.raises(UnsupportedCapabilityError):
        take_snapshot(NoScene())


def test_wrong_or_unverified_plans_cannot_replace_loaded_state(playback_case, result):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        player.set_time(1.5)
        before = player.frame
        for candidate in (
            PlanResult(),
            result,
            replace(plan, validation=replace(plan.validation, collision_margin=0.1)),
            replace(plan, validation=replace(plan.validation, sample_time_step=0.2)),
        ):
            with pytest.raises(ContractError):
                player.load(problem, candidate)
            assert player.frame is before and player.plan_digest == document_digest(plan)
        with pytest.raises(ContractError, match="runtime scene"):
            player.load(replace(problem, scene_revision="other"), plan)
        assert player.frame is before


def test_schema_success_alone_is_not_independent_motion_validation(problem, result):
    with MemoryPlayback(problem) as player:
        with pytest.raises(ContractError, match="CPU revalidation"):
            player.load(problem, result)
        assert player.verification.status == "not_run" and player.plan_digest is None


def test_binding_requires_every_joint_and_distinct_runtime_names(mimic_case):
    problem, _ = mimic_case
    names = ("slide", "follower", "mount")
    with pytest.raises(ContractError, match="every model joint"):
        MemoryPlayback(
            problem,
            bindings=(
                RobotBinding(robot="left", runtime_robot="A", joint_names=("slide",)),
                RobotBinding(robot="right", runtime_robot="B", joint_names=names),
            ),
        )
    with pytest.raises(ContractError, match="unique"):
        MemoryPlayback(
            problem,
            bindings=tuple(
                RobotBinding(robot=r.name, runtime_robot="same", joint_names=names)
                for r in problem.robots
            ),
        )
    with pytest.raises(ContractError, match="every robot"):
        MemoryPlayback(problem, bindings=())
    with pytest.raises(ContractError):
        RobotBinding(
            robot="r",
            runtime_robot="R",
            joint_names=("a", "b"),
            runtime_joint_names=("same", "same"),
        )


def test_zero_dof_scene_limitation_is_explicit(problem):
    from kinoweave import LinkSphere

    model = replace(
        problem.models[0],
        links=("base",),
        joints=(),
        joint_order=(),
        collision_spheres=(LinkSphere(link="base", center=(0, 0, 0), radius=0.1),),
    )
    static = replace(
        problem,
        models=(model,),
        robots=tuple(replace(r, start=(), goal=()) for r in problem.robots),
    )
    with pytest.raises(UnsupportedCapabilityError, match="independent joint"):
        MemoryPlayback(static)


def test_record_budget_failure_happens_before_seeking(playback_case):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        player.set_time(1.5)
        before = player.frame
        for step, maximum in ((0, 100), (-1, 100), (float("nan"), 100), (0.1, 2), (0.1, True)):
            with pytest.raises(ContractError):
                player.record(step, max_samples=maximum)
            assert player.frame is before


def test_report_binding_and_kinematic_conformance_are_separate(playback_case):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        report = player.record(0.5)
    with pytest.raises(ContractError, match="plan_digest"):
        report.check_plan(problem, replace(plan, termination_reason="different result"))
    last = report.observations[-1]
    with pytest.raises(ContractError, match="duration"):
        replace(
            report,
            observations=(
                *report.observations[:-1],
                replace(last, reference_time=plan.makespan + 1),
            ),
        ).check_plan(problem, plan)
    shifted = replace(
        report,
        observations=tuple(
            replace(
                o,
                robots=tuple(
                    replace(r, positions=tuple(q + 0.01 for q in r.positions)) for r in o.robots
                ),
            )
            for o in report.observations
        ),
    )
    shifted.check_plan(problem, plan)
    with pytest.raises(ContractError, match="time law"):
        check_playback_report(problem, plan, shifted)
    controlled = replace(shifted, mode="control_tracking")
    controlled.check_plan(problem, plan)  # physical observations need not exactly equal references
    with pytest.raises(ContractError, match="kinematic"):
        check_playback_report(problem, plan, controlled)
    wrong_units = replace(
        report,
        tracking_errors=(
            TrackingError(robot="alpha", joint="x", unit="rad", max_position_error=0),
        ),
    )
    with pytest.raises(ContractError, match="expected m"):
        wrong_units.check_plan(problem, plan)


def test_reports_cannot_drop_robots_or_change_canonical_joint_order(playback_case):
    problem, plan = playback_case
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        report = player.record(1)
    missing = replace(
        report, observations=tuple(replace(o, robots=o.robots[:1]) for o in report.observations)
    )
    with pytest.raises(ContractError, match="coverage"):
        missing.check_plan(problem, plan)
    reordered = replace(
        report,
        observations=tuple(
            replace(
                o,
                robots=tuple(
                    replace(
                        r,
                        joint_names=tuple(reversed(r.joint_names)),
                        positions=tuple(reversed(r.positions)),
                        velocities=tuple(reversed(r.velocities)),
                    )
                    for r in o.robots
                ),
            )
            for o in report.observations
        ),
    )
    with pytest.raises(ContractError, match="joint order"):
        reordered.check_plan(problem, plan)


def test_continuous_joint_playback_does_not_wrap_coordinates(problem):
    model = problem.models[0]
    model = replace(
        model,
        joints=(
            replace(
                model.joints[0],
                kind="continuous",
                axis=(0, 0, 1),
                limits=JointLimits(velocity=3, acceleration=8),
            ),
        ),
    )
    robot = replace(problem.robots[0], start=(0,), goal=(4 * math.pi,), base_pose=Pose())
    problem = replace(
        problem,
        models=(model,),
        robots=(robot,),
        obstacles=(),
        config=replace(problem.config, time_step=10, validation_time_step=0.1),
    )
    graph = JointGraph(
        robot=robot.name,
        joint_names=model.joint_order,
        vertices=(robot.start, robot.goal),
        edges=((0, 1),),
        start=0,
        goal=1,
    )
    with Planner(problem) as planner:
        plan = planner.plan((graph,))
    assert plan.status == "success"
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        assert player.set_time(5).robots[0].positions == pytest.approx((2 * math.pi,))
        assert player.set_time(10).robots[0].positions == pytest.approx((4 * math.pi,))


def test_saved_plan_playback_example(playback_case, tmp_path):
    from pathlib import Path

    from kinoweave import ExecutionReport
    from kinoweave.io import dump, load

    problem, plan = playback_case
    dump(problem, tmp_path / "problem.json")
    dump(plan, tmp_path / "plan.json")
    process = subprocess.run(
        [
            sys.executable,
            "examples/playback_plan.py",
            str(tmp_path / "problem.json"),
            str(tmp_path / "plan.json"),
            "--step",
            "0.3",
            "--output",
            str(tmp_path / "playback.json"),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    report = loads(process.stdout, expected=ExecutionReport)
    assert report == load(tmp_path / "playback.json", expected=ExecutionReport)
    check_playback_report(problem, plan, report)
