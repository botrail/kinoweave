import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from examples.manipulation_contract import build_problem
from kinoweave import (
    BackendInfo,
    Box,
    ContractError,
    Obstacle,
    PlanProvenance,
    PlanResult,
    Pose,
    RobotTrajectory,
    ValidationReport,
)
from kinoweave.cli import main as cli_main
from kinoweave.io import document_digest, dump, dumps, load, loads
from kinoweave.manipulation import (
    MANIPULATION_CHECKS,
    ActionOrder,
    ContactBody,
    GraspAttachment,
    HeldObject,
    JointTrace,
    ManipulationCost,
    ManipulationMode,
    ManipulationPlan,
    ManipulationResult,
    ModeTransition,
    ObjectSphere,
    PlacedObject,
    RigidObject,
    ScheduledMotion,
    ScheduledSkill,
    ScheduleSkeleton,
    Skill,
    SkillOption,
)


@pytest.fixture
def manipulation():
    return build_problem()


def trace(robot, first, last, duration):
    return JointTrace(
        times=(0, duration),
        tracks=(
            RobotTrajectory(robot=robot, joint_names=("slide",), positions=((first,), (last,))),
        ),
    )


def bind(problem, plan):
    return ManipulationResult(
        status="candidate",
        plan=plan,
        provenance=PlanProvenance(
            problem_digest=document_digest(problem), config=problem.scene.config
        ),
    )


@pytest.fixture
def candidate(manipulation):
    plan = ManipulationPlan(
        makespan=2.5,
        motions=(
            ScheduledMotion(start=1.5, end=2, trace=trace("alpha", 0.4, 0, 0.5)),
            ScheduledMotion(start=2, end=2.5, trace=trace("alpha", 0, 0, 0.5), kind="wait"),
            ScheduledMotion(start=0, end=1, trace=trace("beta", 0, -0.4, 1)),
        ),
        skills=tuple(
            ScheduledSkill(skill=skill.name, option="default", start=i * 0.5, end=(i + 1) * 0.5)
            for i, skill in enumerate(manipulation.skeleton.skills)
        ),
        final_mode=manipulation.goal_mode,
    )
    return bind(manipulation, plan)


def test_native_manipulation_roundtrip_and_symbolic_binding(manipulation, candidate, tmp_path):
    assert (
        load(Path(__file__).parents[1] / "examples/data/manipulation_transfer.problem.json")
        == manipulation
    )
    candidate.check_problem(manipulation)
    assert candidate.status == "candidate" and candidate.validation.status == "not_run"
    for document in (manipulation, ManipulationResult(), candidate):
        path = tmp_path / f"{document.format}-{getattr(document, 'status', 'input')}.json"
        dump(document, path)
        restored = load(path, expected=type(document))
        assert restored == document and dumps(restored) == dumps(document)
        assert document_digest(restored) == document_digest(document)
    assert candidate.makespan == 2.5
    with pytest.raises(ContractError, match="expected PlanResult"):
        loads(dumps(candidate), expected=PlanResult)


def test_old_wire_digests_and_closed_native_fields_are_unchanged():
    expected = {
        "retreat.problem.json": "b5ba42a016d9774ab19fe3b5e7d2ac2bc38238a8bd3a13917c30bbac152b8be3",
        "two_sliders.problem.json": (
            "b27ddefb6283783a874144051014744b5ff9b3ca78b5d8030dc10bb0450abaf0"
        ),
    }
    for name, digest in expected.items():
        data = json.loads((Path(__file__).parents[1] / "examples/data" / name).read_text())
        assert document_digest(loads(json.dumps(data))) == digest
        data["objects"] = []
        with pytest.raises(ContractError, match="unknown field"):
            loads(json.dumps(data))
    assert document_digest(PlanResult()) == (
        "227c11c67f6499fd6aa0fab1fa3e9db2e8893755145f1f40d9434af396118de0"
    )


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1"])
def test_new_format_rejects_unknown_or_coerced_versions(manipulation, version):
    data = json.loads(dumps(manipulation))
    data["schema_version"] = version
    with pytest.raises(ContractError, match="schema_version"):
        loads(json.dumps(data))


@pytest.mark.parametrize("field", ["format", "schema_version", "conventions"])
def test_new_format_needs_envelope(manipulation, field):
    data = json.loads(dumps(manipulation))
    del data[field]
    with pytest.raises(ContractError, match=field):
        loads(json.dumps(data))


def test_new_union_discriminators_fields_and_scalar_types_are_strict(manipulation):
    data = json.loads(dumps(manipulation))
    data["initial_mode"]["states"][0]["kind"] = "floating"
    with pytest.raises(ContractError, match="kind"):
        loads(json.dumps(data))
    data = json.loads(dumps(manipulation))
    data["objects"][0]["mass"] = 1
    with pytest.raises(ContractError, match="objects.*mass"):
        loads(json.dumps(data))
    data = json.loads(dumps(manipulation))
    data["objects"][0]["spheres"][0]["radius"] = True
    with pytest.raises(ContractError, match="radius"):
        loads(json.dumps(data))
    for value in (float("nan"), float("inf")):
        with pytest.raises(ContractError, match="finite"):
            ObjectSphere(center=(0, 0, 0), radius=value)


def test_states_require_unique_complete_ownership_and_real_references(manipulation):
    part = manipulation.initial_mode.states[0]
    with pytest.raises(ContractError, match="duplicate"):
        ManipulationMode(name="bad", states=(part, part))
    with pytest.raises(ContractError, match="cover every object"):
        replace(manipulation, initial_mode=ManipulationMode(name="empty", states=()))
    grasp = GraspAttachment(robot="alpha", link="tool")
    with pytest.raises(ContractError, match="duplicate"):
        HeldObject(object="part", grasps=(grasp, grasp))
    with pytest.raises(ContractError, match="one or two"):
        HeldObject(object="part", grasps=())
    with pytest.raises(ContractError, match="unknown robot link"):
        replace(
            manipulation,
            initial_mode=ManipulationMode(
                name="bad",
                states=(HeldObject(object="part", grasps=(replace(grasp, link="missing"),)),),
            ),
        )
    with pytest.raises(ContractError, match="support"):
        replace(
            manipulation,
            goal_mode=ManipulationMode(name="bad", states=(replace(part, support="x"),)),
        )
    with pytest.raises(ContractError, match="duplicate"):
        replace(
            manipulation,
            scene=replace(
                manipulation.scene, obstacles=(Obstacle(name="part", shape=Box(size=(1, 1, 1))),)
            ),
        )


def test_transitions_contacts_and_option_timing_are_explicit(manipulation):
    skill = manipulation.skeleton.skills[0]
    option = skill.options[0]
    change = option.transitions[0]
    with pytest.raises(ContractError, match="object must not change"):
        replace(change, after=replace(change.after, object="other"))
    with pytest.raises(ContractError, match="preceding effect"):
        replace(option, preconditions=())
    with pytest.raises(ContractError, match="time-ordered"):
        replace(option, transitions=(replace(change, time=1),))
    with pytest.raises(ContractError, match="simultaneous"):
        replace(
            option,
            transitions=(
                change,
                ModeTransition(time=change.time, before=change.after, after=change.before),
            ),
        )
    with pytest.raises(ContractError, match="window exceeds"):
        replace(option, contacts=(replace(option.contacts[0], end=1),))
    with pytest.raises(ContractError, match="require a link"):
        ContactBody(kind="robot", name="alpha")
    with pytest.raises(ContractError, match="non-participating"):
        replace(
            skill,
            options=(
                replace(
                    option,
                    contacts=(
                        replace(
                            option.contacts[0],
                            first=ContactBody(kind="robot", name="beta", link="tool"),
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ContractError, match="minimum duration"):
        replace(skill, minimum_duration=1)


def test_skill_trace_joint_identity_and_positions_are_checked(manipulation):
    skill = manipulation.skeleton.skills[0]
    option = skill.options[0]

    def problem_with(track):
        changed = replace(option, trace=replace(option.trace, tracks=(track,)))
        return replace(
            manipulation,
            skeleton=replace(
                manipulation.skeleton,
                skills=(replace(skill, options=(changed,)), *manipulation.skeleton.skills[1:]),
            ),
        )

    track = option.trace.tracks[0]
    with pytest.raises(ContractError, match="joint order"):
        problem_with(replace(track, joint_names=("wrong",)))
    with pytest.raises(ContractError, match="position limits"):
        problem_with(replace(track, positions=((2,), (2,))))
    with pytest.raises(ContractError, match="strictly increase"):
        replace(option.trace, times=(0, 0))
    with pytest.raises(ContractError, match="cover every knot"):
        replace(option.trace, times=(0, 0.1, 0.5))
    nonuniform = JointTrace(
        times=(0, 0.1, 0.5), tracks=(replace(track, positions=((0,), (0.1,), (0,))),)
    )
    assert nonuniform.duration == 0.5 and nonuniform.interpolation == "linear"


def test_dependency_cycles_unknown_actions_and_unordered_resources_fail(manipulation):
    skeleton = manipulation.skeleton
    with pytest.raises(ContractError, match="cycle"):
        replace(
            skeleton,
            precedence=(*skeleton.precedence, ActionOrder(before="place_beta", after="pick_alpha")),
        )
    with pytest.raises(ContractError, match="unknown action"):
        replace(
            skeleton,
            precedence=(*skeleton.precedence, ActionOrder(before="missing", after="pick_alpha")),
        )
    with pytest.raises(ContractError, match="explicit order"):
        replace(skeleton, precedence=skeleton.precedence[1:])
    with pytest.raises(ContractError, match="duplicate"):
        replace(skeleton, skills=(*skeleton.skills, skeleton.skills[0]))


def test_independent_robot_object_actions_can_overlap(manipulation):
    states = tuple(PlacedObject(object=name, pose=Pose()) for name in ("a", "b"))
    skills = tuple(
        Skill(
            name=f"inspect_{robot}",
            robots=(robot,),
            minimum_duration=1,
            options=(SkillOption(name="one", trace=trace(robot, 0, 0, 1), preconditions=(state,)),),
        )
        for robot, state in zip(("alpha", "beta"), states, strict=True)
    )
    mode = ManipulationMode(name="stationary", states=states)
    problem = replace(
        manipulation,
        objects=tuple(
            RigidObject(name=s.object, spheres=(ObjectSphere(center=(0, 0, 0), radius=0.01),))
            for s in states
        ),
        initial_mode=mode,
        goal_mode=mode,
        skeleton=ScheduleSkeleton(skills=skills),
    )
    plan = ManipulationPlan(
        makespan=1,
        motions=(),
        skills=tuple(ScheduledSkill(skill=s.name, option="one", start=0, end=1) for s in skills),
        final_mode=mode,
    )
    bind(problem, plan).check_problem(problem)
    with pytest.raises(ContractError, match="explicit order"):
        ScheduleSkeleton(
            skills=(
                skills[0],
                replace(
                    skills[1], options=(replace(skills[1].options[0], preconditions=(states[0],)),)
                ),
            )
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: replace(p, motions=p.motions[1:]), "implicit gap"),
        (lambda p: replace(p, motions=(p.motions[0], p.motions[2])), "entire plan"),
        (lambda p: replace(p, skills=p.skills[:-1]), "coverage"),
        (
            lambda p: replace(p, skills=(replace(p.skills[0], option="missing"), *p.skills[1:])),
            "unknown option",
        ),
        (
            lambda p: replace(
                p, skills=(replace(p.skills[0], start=0.25, end=0.75), *p.skills[1:])
            ),
            "precedence",
        ),
        (
            lambda p: replace(p, skills=(replace(p.skills[0], end=0.25), *p.skills[1:])),
            "minimum duration",
        ),
        (
            lambda p: replace(
                p,
                motions=(
                    replace(p.motions[0], trace=trace("alpha", 0.4, 0.1, 0.5)),
                    *p.motions[1:],
                ),
            ),
            "discontinuous",
        ),
    ],
)
def test_candidate_schedule_binding_rejects_inconsistent_activity(
    manipulation, candidate, mutation, match
):
    with pytest.raises(ContractError, match=match):
        bind(manipulation, mutation(candidate.plan)).check_problem(manipulation)


def test_identity_symbolic_preconditions_and_object_goals_are_bound(manipulation, candidate):
    with pytest.raises(ContractError, match="different problem"):
        candidate.check_problem(replace(manipulation, cost=ManipulationCost(distance=1)))
    initial = manipulation.initial_mode.states[0]
    changed = replace(
        manipulation,
        initial_mode=replace(manipulation.initial_mode, states=(replace(initial, pose=Pose()),)),
    )
    with pytest.raises(ContractError, match="precondition"):
        bind(changed, candidate.plan).check_problem(changed)
    changed = replace(manipulation, goal_mode=manipulation.initial_mode)
    with pytest.raises(ContractError, match="object goal"):
        bind(changed, candidate.plan).check_problem(changed)
    with pytest.raises(ContractError, match="scheduled effects"):
        bind(
            manipulation, replace(candidate.plan, final_mode=manipulation.initial_mode)
        ).check_problem(manipulation)


def test_stretched_bimanual_skill_effects_precede_successor_at_shared_boundary(
    manipulation, candidate
):
    skills = manipulation.skeleton.skills
    problem = replace(
        manipulation,
        skeleton=replace(
            manipulation.skeleton,
            skills=(
                *skills[:2],
                replace(skills[2], timing="stretch", minimum_duration=0.8),
                *skills[3:],
            ),
        ),
    )
    plan = replace(
        candidate.plan,
        makespan=3,
        skills=tuple(
            replace(s, end=2)
            if s.skill == "handover"
            else replace(s, start=s.start + 0.5, end=s.end + 0.5)
            if s.start >= 1.5
            else s
            for s in candidate.plan.skills
        ),
        motions=tuple(
            replace(m, start=m.start + 0.5, end=m.end + 0.5) if m.start >= 1.5 else m
            for m in candidate.plan.motions
        ),
    )
    bind(problem, plan).check_problem(problem)
    with pytest.raises(ContractError, match="fixed skill timing"):
        bind(manipulation, plan).check_problem(manipulation)


def test_candidate_is_not_validation_and_native_success_checks_are_insufficient(
    manipulation, candidate
):
    candidate.check_problem(manipulation)
    assert candidate.validation.status == "not_run"
    report = ValidationReport(
        status="passed",
        level="sampled",
        checks=("joint_limits", "collisions", "endpoints"),
        sample_time_step=0.01,
        collision_margin=0.01,
        collision_tolerance=1e-6,
    )
    provenance = replace(
        candidate.provenance, backend=BackendInfo(name="test_fixture", version="1", device="cpu")
    )
    with pytest.raises(ContractError, match="complete manipulation"):
        replace(candidate, status="success", provenance=provenance, validation=report)
    asserted = replace(
        candidate,
        status="success",
        provenance=provenance,
        validation=replace(report, checks=tuple(sorted(MANIPULATION_CHECKS))),
    )
    assert asserted.status == "success"  # Producer assertion only; no verifier has run here.
    with pytest.raises(ContractError, match="backend execution"):
        ManipulationResult(provenance=provenance)


def test_sequences_are_owned_and_digest_includes_new_semantics(manipulation):
    times = [0, 1]
    rows = [[0], [0.1]]
    value = JointTrace(
        times=times, tracks=[RobotTrajectory(robot="alpha", joint_names=["slide"], positions=rows)]
    )
    times.append(2)
    rows[0][0] = 0.9
    assert value.times == (0, 1) and value.tracks[0].positions[0] == (0,)
    original = document_digest(manipulation)
    obj = manipulation.objects[0]
    changed = replace(
        manipulation, objects=(replace(obj, spheres=(replace(obj.spheres[0], radius=0.03),)),)
    )
    assert document_digest(changed) != original
    with pytest.raises(ContractError, match="positive"):
        ManipulationCost(makespan=0)


def test_explicit_waits_and_tiny_trace_clocks_cannot_be_reinterpreted():
    with pytest.raises(ContractError, match="waits cannot move"):
        ScheduledMotion(start=0, end=1, trace=trace("alpha", 0, 0.1, 1), kind="wait")
    with pytest.raises(ContractError, match="preserve the source"):
        ScheduledMotion(start=0, end=1e-13, trace=trace("alpha", 0, 0, 1e-20))


def test_parallel_per_object_effects_do_not_replace_unrelated_state(manipulation):
    initial, final, skills = [], [], []
    for robot, obj in (("alpha", "a"), ("beta", "b")):
        before = PlacedObject(object=obj, pose=Pose())
        after = HeldObject(object=obj, grasps=(GraspAttachment(robot=robot, link="tool"),))
        initial.append(before)
        final.append(after)
        skills.append(
            Skill(
                name=f"pick_{robot}",
                robots=(robot,),
                minimum_duration=1,
                options=(
                    SkillOption(
                        name="one",
                        trace=trace(robot, 0, 0, 1),
                        preconditions=(before,),
                        transitions=(ModeTransition(time=0.5, before=before, after=after),),
                    ),
                ),
            )
        )
    goal = ManipulationMode(name="both_held", states=final)
    problem = replace(
        manipulation,
        objects=tuple(replace(manipulation.objects[0], name=obj) for obj in ("a", "b")),
        initial_mode=ManipulationMode(name="both_placed", states=initial),
        goal_mode=goal,
        skeleton=ScheduleSkeleton(skills=skills),
    )
    plan = ManipulationPlan(
        makespan=1,
        motions=(),
        skills=tuple(
            ScheduledSkill(skill=s.name, option="one", start=0, end=1) for s in reversed(skills)
        ),
        final_mode=goal,
    )
    bind(problem, plan).check_problem(problem)


def test_motion_only_cli_rejects_manipulation_input(manipulation, tmp_path, capsys):
    dump(manipulation, tmp_path / "input.json")
    target = tmp_path / "run"
    assert cli_main(["plan", str(tmp_path / "input.json"), "--output", str(target)]) == 2
    assert "expected PlanningProblem" in capsys.readouterr().err
    assert not target.exists()


def test_sdk_free_subprocess_can_load_and_bind_manipulation(manipulation, candidate, tmp_path):
    dump(manipulation, tmp_path / "input.json")
    dump(candidate, tmp_path / "candidate.json")
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','torch','warp','curobo','viser','yourdfpy',
                                      'trimesh','yaml','mujoco','isaacsim','botrail'}:
            raise AssertionError('optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from kinoweave.io import load, dumps, loads
from kinoweave.manipulation import ManipulationProblem, ManipulationResult
p = load(sys.argv[1], expected=ManipulationProblem)
r = load(sys.argv[2], expected=ManipulationResult)
r.check_problem(p)
assert loads(dumps(p)) == p
assert r.status == 'candidate' and r.validation.status == 'not_run'
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(tmp_path / "input.json"),
            str(tmp_path / "candidate.json"),
        ],
        check=True,
    )
