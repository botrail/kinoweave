import math
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinoweave import ContractError, PlanResult, ValidationReport
from kinoweave.io import dumps, loads
from kinoweave.search.reference import JointGraph, reference_plan

VERTICES = ((-1, 0), (0, 0), (1, 0), (0, 1))
EDGES = ((0, 1), (1, 2), (1, 3))


def graphs(edges=EDGES):
    return tuple(
        JointGraph(
            robot=name,
            joint_names=("x", "y"),
            vertices=VERTICES,
            edges=edges,
            start=start,
            goal=goal,
        )
        for name, start, goal in (("alpha", 0, 2), ("beta", 2, 0))
    )


def test_retreat_plan_has_known_minimum_and_roundtrips(retreat_problem):
    result = reference_plan(retreat_problem, graphs())
    assert result.status == "success" and result.makespan == 4
    assert (
        result.provenance.backend.name == "reference" and result.provenance.backend.device == "cpu"
    )
    assert any((0, 1) in track.positions for track in result.trajectory.tracks)
    result.check_problem(retreat_problem)
    restored = loads(dumps(result), expected=PlanResult)
    restored.check_problem(retreat_problem)
    assert restored == result


def test_corridor_without_bay_has_no_solution_in_the_graph(retreat_problem):
    result = reference_plan(retreat_problem, graphs(EDGES[:2]))
    assert result.status == "no_solution_in_discretization"
    assert result.termination_reason == "exhausted_graph_product"
    assert result.trajectory is None


def test_already_arrived_robot_can_leave_goal_and_return(retreat_problem):
    alpha, beta = retreat_problem.robots
    problem = replace(
        retreat_problem,
        robots=(
            replace(alpha, start=(0, 0), goal=(0, 0)),
            replace(beta, start=(-1, 0), goal=(1, 0)),
        ),
    )
    a, b = graphs()
    candidates = (
        replace(a, start=1, goal=1, edges=((1, 3),)),
        replace(b, start=0, goal=2, edges=EDGES[:2]),
    )
    result = reference_plan(problem, candidates)
    assert result.status == "success" and result.makespan == 2
    assert result.trajectory.tracks[0].positions == ((0, 0), (0, 1), (0, 0))
    blocked = (replace(candidates[0], edges=()), candidates[1])
    assert reference_plan(problem, blocked).status == "no_solution_in_discretization"


def test_budget_exhaustion_is_not_a_proof_of_no_solution(retreat_problem, monkeypatch):
    result = reference_plan(retreat_problem, graphs(), max_states=1)
    assert result.status == "timeout" and result.termination_reason == "state_limit"
    import kinoweave.search.reference as reference

    clock = iter((0, 20))
    monkeypatch.setattr(reference, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    result = reference.reference_plan(retreat_problem, graphs())
    assert result.status == "timeout" and result.termination_reason == "time_limit"


def test_final_validator_does_not_trust_transition_verdicts(retreat_problem, monkeypatch):
    import kinoweave.search.reference as reference

    fake = ValidationReport(
        status="passed",
        level="sampled",
        checks=("collisions",),
        sample_time_step=0.1,
        collision_margin=0,
        collision_tolerance=1e-9,
    )
    monkeypatch.setattr(reference, "validate_interval", lambda *args, **kwargs: fake)
    result = reference.reference_plan(retreat_problem, graphs(((0, 2),)))
    assert result.status == "validation_failed" and result.trajectory is None
    assert any(i.code == "collision" for i in result.validation.issues)


def test_all_at_goal_returns_a_shared_hold(retreat_problem):
    problem = replace(
        retreat_problem, robots=tuple(replace(r, goal=r.start) for r in retreat_problem.robots)
    )
    candidates = tuple(replace(g, goal=g.start) for g in graphs())
    result = reference_plan(problem, candidates)
    assert result.status == "success" and result.makespan == 1
    assert all(track.positions[0] == track.positions[1] for track in result.trajectory.tracks)


def test_graph_binding_and_explicit_cpu_selection(retreat_problem):
    assert reference_plan(retreat_problem, graphs()[:1]).status == "invalid_problem"
    a, b = graphs()
    assert (
        reference_plan(retreat_problem, (replace(a, joint_names=("y", "x")), b)).status
        == "invalid_problem"
    )
    assert reference_plan(retreat_problem, (replace(a, start=1), b)).status == "invalid_problem"
    with pytest.raises(ContractError, match="explicitly select"):
        reference_plan(
            replace(
                retreat_problem,
                config=replace(retreat_problem.config, backend="curobo", device="cuda:0"),
            ),
            graphs(),
        )
    with pytest.raises(ContractError):
        replace(a, edges=((0, 1), (1, 0)))


def test_infeasible_timing_is_not_silently_retimed(retreat_problem):
    problem = replace(
        retreat_problem,
        robots=retreat_problem.robots[:1],
        config=replace(retreat_problem.config, time_step=0.1),
    )
    assert reference_plan(problem, graphs()[:1]).status == "no_solution_in_discretization"


def exact_tick_distance(first_edges, second_edges):
    # Independent all-pairs closure with exact relative-line closest approach.
    # For these XY prismatic robots both motions share the same monotonic blend,
    # so minimizing in blend-space checks the entire continuous interval.
    states = tuple((a, b) for a, b in product(range(4), repeat=2) if a != b)
    adjacency = []
    for edges in (first_edges, second_edges):
        rows = [{i} for i in range(4)]
        for a, b in edges:
            rows[a].add(b)
            rows[b].add(a)
        adjacency.append(rows)
    distance = [[math.inf] * len(states) for _ in states]
    for i, a in enumerate(states):
        distance[i][i] = 0
        for j, b in enumerate(states):
            if b[0] not in adjacency[0][a[0]] or b[1] not in adjacency[1][a[1]]:
                continue
            r0 = tuple(x - y for x, y in zip(VERTICES[a[0]], VERTICES[a[1]], strict=True))
            r1 = tuple(x - y for x, y in zip(VERTICES[b[0]], VERTICES[b[1]], strict=True))
            delta = tuple(y - x for x, y in zip(r0, r1, strict=True))
            norm2 = sum(d * d for d in delta)
            s = (
                max(0, min(1, -sum(x * d for x, d in zip(r0, delta, strict=True)) / norm2))
                if norm2
                else 0
            )
            separation = math.hypot(*(x + s * d for x, d in zip(r0, delta, strict=True)))
            if separation > 0.36 + 1e-9:
                distance[i][j] = min(distance[i][j], 1)
    for k in range(len(states)):
        for i in range(len(states)):
            for j in range(len(states)):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance[states.index((0, 2))][states.index((2, 0))]


def test_all_64_pairs_of_tiny_graphs_match_independent_closure(retreat_problem):
    subsets = [tuple(edge for k, edge in enumerate(EDGES) if mask & (1 << k)) for mask in range(8)]
    a, b = graphs()
    for first, second in product(subsets, repeat=2):
        expected = exact_tick_distance(first, second)
        result = reference_plan(
            retreat_problem, (replace(a, edges=first), replace(b, edges=second))
        )
        if math.isfinite(expected):
            assert result.status == "success" and result.makespan == expected
        else:
            assert result.status == "no_solution_in_discretization"


def test_reference_example_outputs_a_real_validated_result(retreat_problem):
    process = subprocess.run(
        [sys.executable, "examples/reference_coordination.py"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    result = loads(process.stdout, expected=PlanResult)
    assert result.status == "success" and result.validation.level == "sampled"
    result.check_problem(retreat_problem)
