import json
import subprocess
import sys
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

from kinoweave import ContractError, Planner, PlanResult, SearchConfig
from kinoweave.collision import ReferenceMotionEvaluator
from kinoweave.io import dumps, loads
from kinoweave.roadmap import PRMConfig
from kinoweave.search.reference import reference_plan
from kinoweave.validation import validate_trajectory


def test_example_saves_matching_problem_and_result(tmp_path):
    from pathlib import Path

    from kinoweave import PlanningProblem
    from kinoweave.io import load

    process = subprocess.run(
        [
            sys.executable,
            "examples/coordinated_planning.py",
            "--backend",
            "reference",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    problem = load(tmp_path / "problem.json", expected=PlanningProblem)
    result = loads(process.stdout, expected=PlanResult)
    assert result == load(tmp_path / "result.json", expected=PlanResult)
    assert result.status == "success"
    result.check_problem(problem)
    assert validate_trajectory(problem, result.trajectory).status == "passed"


def details(result, code):
    return json.loads(next(d.message for d in result.diagnostics if d.code == code))


@pytest.mark.parametrize("case", ["independent", "crossing", "retreat", "goal_retreat", "reroute"])
@pytest.mark.parametrize("algorithm", ["lacam", "joint_astar"])
def test_coordinated_trajectories_are_independently_validated(coordination_cases, case, algorithm):
    problem, graphs = coordination_cases[case]
    with Planner(
        problem, search=SearchConfig(algorithm=algorithm, exact_fallback=False)
    ) as planner:
        result = planner.plan(graphs)
    assert result.status == "success", result.diagnostics
    assert result.provenance.backend.device == "cpu"
    assert validate_trajectory(problem, result.trajectory).status == "passed"
    result.check_problem(problem)
    restored = loads(dumps(result), expected=PlanResult)
    restored.check_problem(problem)
    assert restored == result
    assert details(result, "search_attempts")[-1]["algorithm"] == algorithm
    if case == "independent":
        assert result.makespan == 2
    if case == "goal_retreat":
        track = result.trajectory.tracks[0]
        assert track.positions[0] == track.positions[-1] == (0, 0)
        assert any(p != (0, 0) for p in track.positions)


def test_prioritized_failure_is_incomplete_and_auto_can_retreat(coordination_cases):
    problem, graphs = coordination_cases["retreat"]
    with Planner(problem) as planner:
        baseline = planner.plan(
            graphs, search=SearchConfig(algorithm="prioritized", priority_horizon=6)
        )
        result = planner.plan(graphs, search=SearchConfig(algorithm="auto", priority_horizon=6))
    assert baseline.status == "timeout" and baseline.termination_reason == "prioritized_incomplete"
    assert result.status == "success"
    assert [a["algorithm"] for a in details(result, "search_attempts")] == ["prioritized", "lacam"]
    assert details(result, "search_stats")["inheritance_calls"] > 0


def test_priority_order_and_future_goal_occupancy(coordination_cases):
    problem, graphs = coordination_cases["goal_retreat"]
    with Planner(problem) as planner:
        blocked = planner.plan(
            graphs, search=SearchConfig(algorithm="prioritized", priority_horizon=5)
        )
        result = planner.plan(
            graphs,
            search=SearchConfig(
                algorithm="prioritized", priority_order=("beta", "alpha"), priority_horizon=5
            ),
        )
    assert blocked.status == "timeout"
    assert result.status == "success" and result.makespan == 2
    assert result.trajectory.tracks[0].positions == ((0, 0), (0, 1), (0, 0))


@pytest.mark.parametrize("algorithm", ["lacam", "joint_astar"])
def test_finite_exhaustion_and_budget_limits_are_distinct(coordination_cases, algorithm):
    problem, graphs = coordination_cases["corridor"]
    with Planner(problem) as planner:
        result = planner.plan(
            graphs, search=SearchConfig(algorithm=algorithm, exact_fallback=False)
        )
        limited = planner.plan(
            graphs, search=SearchConfig(algorithm=algorithm, max_states=1, exact_fallback=False)
        )
    assert result.status == "no_solution_in_discretization"
    assert limited.status == "timeout" and limited.termination_reason == "state_limit"


def test_constraint_cap_and_small_pair_tiles_do_not_claim_no_solution(coordination_cases):
    problem, graphs = coordination_cases["retreat"]
    with Planner(problem) as planner:
        capped = planner.plan(graphs, search=SearchConfig(max_constraints=1, exact_fallback=False))
        first = planner.plan(
            graphs, search=SearchConfig(max_query_pairs=1, max_cached_pairs=0, exact_fallback=False)
        )
        second = planner.plan(graphs, search=SearchConfig(exact_fallback=False))
    assert capped.status == "timeout" and capped.termination_reason == "constraint_limit"
    assert first.status == second.status == "success" and first.trajectory == second.trajectory
    assert details(first, "search_stats")["peak_cached_pairs"] == 0


def test_final_verification_rejects_forged_safe_tables(coordination_cases):
    problem, graphs = coordination_cases["crossing"]

    class Unsafe(ReferenceMotionEvaluator):
        def interference(self, *args):
            table = super().interference(*args)
            return replace(
                table,
                compatible=tuple(
                    tuple(True for _ in table.second_motions) for _ in table.first_motions
                ),
            )

    with Planner(problem, evaluator=Unsafe(problem)) as planner:
        result = planner.plan(graphs)
    assert result.status == "validation_failed" and result.trajectory is None
    assert any(issue.code == "collision" for issue in result.validation.issues)


def test_foreign_tables_and_graph_bindings_are_rejected(coordination_cases):
    problem, graphs = coordination_cases["retreat"]

    class Foreign(ReferenceMotionEvaluator):
        def interference(self, *args):
            return replace(super().interference(*args), problem_digest="0" * 64)

    with Planner(problem, evaluator=Foreign(problem)) as planner:
        assert planner.plan(graphs).status == "invalid_problem"
    with Planner(problem) as planner:
        assert planner.plan(graphs[:1]).status == "invalid_problem"
        assert (
            planner.plan((replace(graphs[0], joint_names=("y", "x")), graphs[1])).status
            == "invalid_problem"
        )
        assert (
            planner.plan(graphs, search=SearchConfig(priority_order=("missing",))).status
            == "invalid_problem"
        )
    with pytest.raises(RuntimeError, match="closed"):
        planner.plan(graphs)


def test_preparation_is_reused_and_backend_selection_is_explicit(coordination_cases):
    problem, _ = coordination_cases["independent"]
    with Planner(problem, prm=PRMConfig(num_samples=0)) as planner:
        first = planner.plan()
        roadmaps = planner.roadmaps
        second = planner.plan()
        assert planner.roadmaps is roadmaps
    assert first.status == second.status == "success" and first.trajectory == second.trajectory
    wrong = replace(problem, config=replace(problem.config, backend="unknown"))
    assert Planner(wrong).plan().status == "unsupported"
    with pytest.raises(ContractError):
        Planner(problem, search=False)


def test_shared_deadline_also_covers_preparation(coordination_cases, monkeypatch):
    import kinoweave.search.space as space

    problem, graphs = coordination_cases["retreat"]
    clock = [0.0]
    monkeypatch.setattr(space, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Slow(ReferenceMotionEvaluator):
        def motion_validity(self, *args):
            result = super().motion_validity(*args)
            clock[0] += problem.config.time_limit + 1
            return result

    with Planner(problem, evaluator=Slow(problem)) as planner:
        result = planner.plan()
    assert result.status == "timeout" and result.termination_reason == "time_limit"
    assert (
        next(d.stage for d in result.diagnostics if d.code == "termination")
        == "roadmap_preparation"
    )


def test_already_finished_robots_return_a_verified_hold(coordination_cases):
    problem, graphs = coordination_cases["independent"]
    problem = replace(problem, robots=tuple(replace(r, goal=r.start) for r in problem.robots))
    graphs = tuple(replace(g, goal=g.start) for g in graphs)
    with Planner(problem) as planner:
        result = planner.plan(graphs)
    assert result.status == "success" and result.makespan == problem.config.time_step
    assert all(t.positions[0] == t.positions[-1] for t in result.trajectory.tracks)


def test_verified_success_reports_soft_deadline_overrun(coordination_cases, monkeypatch):
    import kinoweave.planner as module
    import kinoweave.search.space as space

    problem, graphs = coordination_cases["independent"]
    clock = [0.0]
    monkeypatch.setattr(space, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = module.validate_trajectory

    def late_verification(*args, **kwargs):
        report = original(*args, **kwargs)
        clock[0] = problem.config.time_limit + 1
        return report

    monkeypatch.setattr(module, "validate_trajectory", late_verification)
    with Planner(problem) as planner:
        result = planner.plan(graphs)
    assert result.status == "success" and result.termination_reason == "time_limit_after_validation"
    assert result.validation.status == "passed"


def test_two_robot_fallback_keeps_the_same_budget_and_evaluator(coordination_cases, monkeypatch):
    import kinoweave.planner as module
    from kinoweave.search.space import SearchLimit

    problem, graphs = coordination_cases["retreat"]
    seen = []

    def interrupted(space):
        seen.append(space.budget)
        raise SearchLimit("phase_time_limit", "lacam_search")

    original = module.joint_astar_search

    def fallback(space):
        assert space.budget is seen[0]
        return original(space)

    monkeypatch.setattr(module, "lacam_search", interrupted)
    monkeypatch.setattr(module, "joint_astar_search", fallback)
    with Planner(problem) as planner:
        result = planner.plan(graphs)
    assert result.status == "success"
    assert [a["algorithm"] for a in details(result, "search_attempts")] == ["lacam", "joint_astar"]


def test_lacam_matches_reference_exhaustion_on_all_64_tiny_graph_pairs(coordination_cases):
    problem, graphs = coordination_cases["retreat"]
    edges = graphs[0].edges
    subsets = tuple(
        tuple(edge for k, edge in enumerate(edges) if mask & (1 << k)) for mask in range(8)
    )
    with Planner(problem, search=SearchConfig(exact_fallback=False)) as planner:
        for first, second in product(subsets, repeat=2):
            pair = (replace(graphs[0], edges=first), replace(graphs[1], edges=second))
            expected = reference_plan(problem, pair)
            actual = planner.plan(pair)
            assert actual.status == expected.status, (first, second, actual.diagnostics)
