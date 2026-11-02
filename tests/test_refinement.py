import subprocess
import sys
from dataclasses import replace

import pytest

from kinoweave import Box, Obstacle, Pose
from kinoweave.io import document_digest
from kinoweave.manipulation import (
    ManipulationRefiner,
    RefinementConfig,
    evaluate_objective,
    refine_manipulation,
    validate_manipulation,
)
from kinoweave.manipulation.refinement_geometry import RefinementGeometry
from kinoweave.manipulation.shortcuts import Proposal, Shortcut, apply_proposal


def edits(result):
    return tuple(
        Shortcut(
            m.trace.tracks[0].robot,
            m.start,
            m.end,
            m.trace.tracks[0].positions[0],
            m.trace.tracks[0].positions[-1],
            m.end - m.start,
        )
        for m in result.plan.motions
    )


def test_anytime_shortening_is_monotone_reverified_and_input_is_immutable(refinement_cases):
    problem, seed = refinement_cases()
    initial = seed.plan
    updates = []
    refiner = ManipulationRefiner(
        problem, config=RefinementConfig(time_limit=30, max_iterations=2, max_candidates=16)
    )
    report = refiner.refine(seed, on_improvement=lambda u: updates.append(u))
    assert report.result.status == "success" and report.after.total < report.before.total
    assert report.result.makespan < seed.makespan
    assert report.before == evaluate_objective(problem, seed)
    assert report.after == evaluate_objective(problem, report.result)
    assert all(
        b.objective.total < a.objective.total
        for a, b in zip(report.history, report.history[1:], strict=False)
    )
    for update in updates:
        assert validate_manipulation(problem, update.result).status == "passed"
    assert refiner.best_result is report.result
    assert report.statistics["accepted"] > 0 and report.statistics["cpu_verified"] > 0
    assert seed.plan is initial and seed.plan.makespan == 4.5


def test_cooperative_shortcuts_work_even_when_both_single_edits_collide(
    refinement_cases, monkeypatch
):
    problem, seed = refinement_cases(cooperative=True, weights=(0, 0, 1))
    a, b = edits(seed)
    proposals = (Proposal((a,), "single"), Proposal((b,), "single"), Proposal((a, b), "composite"))
    candidates = tuple(replace(seed, plan=apply_proposal(problem, seed.plan, p)) for p in proposals)
    assert [validate_manipulation(problem, c).status for c in candidates] == [
        "failed",
        "failed",
        "passed",
    ]
    monkeypatch.setattr("kinoweave.manipulation.refinement.proposals", lambda *args: proposals)
    report = refine_manipulation(
        problem, seed, config=RefinementConfig(time_limit=30, max_iterations=1)
    )
    assert report.statistics["geometry_rejected"] == 2
    assert report.after.total < report.before.total and report.history[-1].kind == "composite"
    assert validate_manipulation(problem, report.result).status == "passed"


def test_greedy_disjoint_fusion_is_checked_as_a_complete_plan(refinement_cases, monkeypatch):
    problem, seed = refinement_cases(weights=(0, 0, 1))
    a, b = edits(seed)
    monkeypatch.setattr(
        "kinoweave.manipulation.refinement.proposals",
        lambda *args: (Proposal((a,), "single"), Proposal((b,), "single")),
    )
    report = refine_manipulation(
        problem, seed, config=RefinementConfig(time_limit=30, max_iterations=1)
    )
    assert report.history[-1].kind == "greedy"
    assert report.before.distance == 8 and report.after.distance == 4
    assert validate_manipulation(problem, report.result).status == "passed"


def test_colliding_improvement_is_rejected_and_seed_survives(refinement_cases, monkeypatch):
    problem, seed = refinement_cases(cooperative=True, weights=(0, 0, 1))
    monkeypatch.setattr(
        "kinoweave.manipulation.refinement.proposals",
        lambda *args: (Proposal((edits(seed)[0],), "single"),),
    )
    report = refine_manipulation(problem, seed, config=RefinementConfig(time_limit=30))
    assert report.termination_reason == "no_improvement_in_candidates"
    assert report.result.plan == seed.plan and report.after == report.before
    assert report.statistics["geometry_rejected"] == 1


@pytest.mark.parametrize(
    "config,reason",
    [
        (RefinementConfig(time_limit=0), "time_limit"),
        (RefinementConfig(max_iterations=0), "iteration_limit"),
        (RefinementConfig(max_geometry_bytes=1, time_limit=30), "geometry_limit"),
    ],
)
def test_budget_exit_retains_verified_incumbent(refinement_cases, config, reason):
    problem, seed = refinement_cases()
    report = refine_manipulation(problem, seed, config=config)
    assert report.status == "budget_exhausted" and report.termination_reason == reason
    assert report.result.status == "success" and report.result.plan == seed.plan
    assert validate_manipulation(problem, report.result).status == "passed"


def test_cancellation_and_late_atomic_verification_keep_the_improvement(
    refinement_cases, monkeypatch
):
    problem, seed = refinement_cases()
    refiner = ManipulationRefiner(
        problem, config=RefinementConfig(time_limit=30, max_candidates=16)
    )
    cancelled = refiner.refine(
        seed, on_improvement=lambda u: False if u.sample.iteration > 0 else None
    )
    assert cancelled.status == "cancelled" and cancelled.after.total < cancelled.before.total
    assert refiner.best_result == cancelled.result
    now = [0.0]
    monkeypatch.setattr("kinoweave.manipulation.refinement.time.monotonic", lambda: now[0])

    def elapsed(update):
        if update.sample.iteration:
            now[0] = 100

    timed = refiner.refine(seed, on_improvement=elapsed)
    assert timed.termination_reason == "time_limit" and timed.status == "budget_exhausted"
    assert timed.statistics["accepted"] == 1 and timed.after.total < timed.before.total


def test_forged_success_is_reverified_before_any_proposal(refinement_cases, monkeypatch):
    problem, seed = refinement_cases()
    obstacle = Obstacle(name="cover", shape=Box(size=(8, 8, 8)), pose=Pose())
    problem = replace(problem, scene=replace(problem.scene, obstacles=(obstacle,)))
    seed = replace(
        seed, provenance=replace(seed.provenance, problem_digest=document_digest(problem))
    )

    def forbidden(*args):
        raise AssertionError("invalid seed reached proposal generation")

    monkeypatch.setattr("kinoweave.manipulation.refinement.proposals", forbidden)
    report = refine_manipulation(problem, seed)
    assert report.status == "invalid_seed" and report.result.status == "validation_failed"
    assert report.result.plan is None and not report.history


def test_full_plan_geometry_batch_accepts_different_time_grids(refinement_cases):
    problem, seed = refinement_cases(cooperative=True, weights=(0, 0, 1))
    a, b = edits(seed)
    choices = (Proposal((a,), "single"), Proposal((b,), "single"), Proposal((a, b), "composite"))
    candidates = tuple(replace(seed, plan=apply_proposal(problem, seed.plan, p)) for p in choices)
    engine = RefinementGeometry(problem, RefinementConfig(), None, lambda: None)
    try:
        assert engine.check_many(candidates) == (False, False, True)
        assert engine.stats["geometry_candidates"] == 3
    finally:
        engine.close()


def test_handover_refinement_preserves_skills_and_object_effects():
    from examples.two_arm_handover import build_problem
    from kinoweave.manipulation import ManipulationTimeline, MultiArmPlanner
    from kinoweave.roadmap import PRMConfig

    problem = build_problem()
    seed = MultiArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan()
    report = refine_manipulation(
        problem, seed, config=RefinementConfig(time_limit=30, max_iterations=2, max_candidates=16)
    )
    assert report.after.makespan < seed.makespan
    timeline = ManipulationTimeline(problem, report.result)
    original = ManipulationTimeline(problem, seed)
    for first, last in zip(seed.plan.skills, report.result.plan.skills, strict=True):
        assert first.skill == last.skill and first.option == last.option
        assert last.end - last.start == pytest.approx(first.end - first.start)
        for u in (0.2, 0.5, 0.8):
            a = original.configurations(first.start + u * (first.end - first.start))
            b = timeline.configurations(last.start + u * (last.end - last.start))
            skill = next(s for s in problem.skeleton.skills if s.name == first.skill)
            for robot in skill.robots:
                assert b[robot] == pytest.approx(a[robot])
    assert timeline.mode(report.result.makespan).states == problem.goal_mode.states
    assert validate_manipulation(problem, report.result).status == "passed"


def test_cpu_refinement_does_not_import_optional_sdks(refinement_cases, tmp_path):
    from kinoweave.io import dump

    problem, seed = refinement_cases()
    dump(problem, tmp_path / "problem.json")
    dump(seed, tmp_path / "result.json")
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        blocked={'numpy','torch','warp','curobo','viser','mujoco','botrail','isaacsim'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from kinoweave.io import load
from kinoweave.manipulation import RefinementConfig,refine_manipulation
p,r=load(sys.argv[1]),load(sys.argv[2])
report=refine_manipulation(p,r,config=RefinementConfig(time_limit=30,max_iterations=1,max_candidates=16))
assert report.after.total<report.before.total
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "problem.json"),
            str(tmp_path / "result.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_unavailable_cuda_keeps_seed_without_cpu_optimization(refinement_cases, monkeypatch):
    from kinoweave.backends.curobo import BackendUnavailableError

    problem, seed = refinement_cases(backend="curobo")

    def unavailable(*args, **kwargs):
        raise BackendUnavailableError("test CUDA unavailable")

    monkeypatch.setattr("kinoweave.backends.curobo.CuroboBackend", unavailable)
    report = refine_manipulation(problem, seed)
    assert report.status == "unsupported" and report.backend is None
    assert report.statistics["accepted"] == 0 and report.result.plan == seed.plan
    assert report.result.provenance.backend.name == "reference"


def test_metric_budget_failure_retains_independently_verified_seed(refinement_cases):
    from kinoweave.manipulation import MetricConfig

    problem, seed = refinement_cases(metric="tool_path")
    report = refine_manipulation(problem, seed, metric_config=MetricConfig(max_samples=2))
    assert (
        report.status == "budget_exhausted" and report.termination_reason == "metric_sample_limit"
    )
    assert report.result.status == "success" and report.result.plan == seed.plan
    assert report.before is None and not report.history
