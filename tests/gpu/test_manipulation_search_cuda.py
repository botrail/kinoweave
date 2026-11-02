import pytest

from examples.two_arm_handover import build_problem
from kinoweave.io import dump
from kinoweave.manipulation import (
    ManipulationSearchConfig,
    ModeEvaluator,
    MultiArmPlanner,
    validate_manipulation,
)
from kinoweave.manipulation.search import search
from kinoweave.manipulation.space import TaskSpace
from kinoweave.roadmap import PRMConfig
from kinoweave.search.space import Budget, SearchConfig

pytestmark = pytest.mark.cuda


def test_cuda_joint_handover_and_eager_tables_match_cpu(cuda, tmp_path):
    cpu_problem = build_problem()
    cpu_planner = MultiArmPlanner(cpu_problem, prm=PRMConfig(num_samples=0, neighbors=8))
    cpu_result = cpu_planner.plan()
    assert cpu_result.status == "success"
    graphs = tuple(m.graph for m in cpu_planner.roadmaps)
    problem = build_problem(backend="curobo", device="cuda:0")
    planner = MultiArmPlanner(
        problem,
        search_config=ManipulationSearchConfig(
            pair_batch_size=2, sphere_tile=1, max_pair_bytes=4096
        ),
    )
    result = planner.plan(graphs)
    assert result.status == "success", result
    assert result.provenance.backend.name == "curobo"
    assert validate_manipulation(problem, result).status == "passed"
    assert planner.statistics["gpu_pair_batches"] > 0
    assert planner.statistics["max_pair_tile_bytes"] <= 4096
    assert planner.statistics["geometry_native_calls"] > 0
    dump(problem, tmp_path / "problem.json")
    dump(result, tmp_path / "result.json")
    config = ManipulationSearchConfig()
    with ModeEvaluator(cpu_problem) as cpu, ModeEvaluator(problem) as gpu:
        cpu_space = TaskSpace(cpu, graphs, config, Budget(120, SearchConfig()))
        gpu_space = TaskSpace(gpu, graphs, config, Budget(120, SearchConfig()))
        path = search(cpu_space)
        selected = [path[0][0]]
        selected.extend(
            s
            for s, _ in path
            if any(
                run is not None and cpu_space.skills[run.skill].name == "handover"
                for run in s.running
            )
        )
        for state in selected:
            a, b = cpu_space.prepare(state), gpu_space.prepare(state)
            assert a.options == b.options and a.holds == b.holds
            assert a.table.actions == b.table.actions and a.table.points == b.table.points
            assert a.table.valid == b.table.valid
            assert a.table.pairs == b.table.pairs
