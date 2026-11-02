from dataclasses import replace
from pathlib import Path

import pytest

from kinoweave import (
    Joint,
    JointLimits,
    LinkSphere,
    Planner,
    Pose,
    RobotInstance,
    RobotModel,
    SearchConfig,
)
from kinoweave.collision import GeometryCache, GpuTiling
from kinoweave.io import dumps
from kinoweave.roadmap import PRMConfig
from kinoweave.search.reference import JointGraph
from kinoweave.validation import validate_trajectory

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("case", ["independent", "crossing", "retreat", "goal_retreat", "reroute"])
@pytest.mark.parametrize("algorithm", ["lacam", "joint_astar"])
def test_gpu_backed_coordination_cases(coordination_cases, cuda, case, algorithm):
    problem, graphs = coordination_cases[case]
    problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_limit=30)
    )
    with Planner(
        problem, search=SearchConfig(algorithm=algorithm, exact_fallback=False)
    ) as planner:
        result = planner.plan(graphs)
    assert result.status == "success", result.diagnostics
    assert (
        result.provenance.backend.name == "curobo" and result.provenance.backend.device == "cuda:0"
    )
    result.check_problem(problem)
    assert validate_trajectory(problem, result.trajectory).status == "passed"
    if case == "independent":
        assert result.makespan == 2


def test_gpu_prioritized_failure_then_lazy_retreat(coordination_cases, cuda):
    problem, graphs = coordination_cases["retreat"]
    problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_limit=30)
    )
    with Planner(problem) as planner:
        baseline = planner.plan(
            graphs, search=SearchConfig(algorithm="prioritized", priority_horizon=6)
        )
        result = planner.plan(graphs, search=SearchConfig(algorithm="auto", priority_horizon=6))
    assert baseline.status == "timeout" and result.status == "success"
    assert validate_trajectory(problem, result.trajectory).status == "passed"


def test_gpu_planner_prepares_and_reuses_prms(coordination_cases, cuda):
    problem, _ = coordination_cases["independent"]
    problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_limit=30)
    )
    with Planner(problem, prm=PRMConfig(num_samples=12, neighbors=3)) as planner:
        first = planner.plan()
        roadmaps = planner.roadmaps
        second = planner.plan()
        assert planner.roadmaps is roadmaps and len(roadmaps) == 2
    assert first.status == second.status == "success"
    assert first.trajectory == second.trajectory
    assert validate_trajectory(problem, first.trajectory).status == "passed"


def test_gpu_search_tiling_and_cache_do_not_change_the_plan(coordination_cases, cuda):
    problem, graphs = coordination_cases["retreat"]
    problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_limit=30)
    )
    trajectories = []
    for tiling, capacity, settings in (
        (
            GpuTiling(
                motion_batch_size=1,
                time_batch_size=2,
                pair_batch_size=1,
                sphere_batch_size=1,
                max_table_pairs=1,
            ),
            0,
            SearchConfig(exact_fallback=False, max_query_pairs=1, max_cached_pairs=0),
        ),
        (GpuTiling(), 1_000_000, SearchConfig(exact_fallback=False)),
    ):
        with Planner(
            problem, tiling=tiling, cache=GeometryCache(capacity), search=settings
        ) as planner:
            result = planner.plan(graphs)
            assert result.status == "success", result.diagnostics
            trajectories.append(result.trajectory)
    assert trajectories[0] == trajectories[1]


@pytest.mark.parametrize("algorithm", ["lacam", "prioritized"])
def test_heterogeneous_fixed_arms_return_a_verified_gpu_plan(
    retreat_problem, cuda, algorithm, record_testsuite_property
):
    def model(name, dof):
        joints = tuple(
            Joint(
                name=f"j{i}",
                kind="revolute",
                parent=f"l{i}",
                child=f"l{i + 1}",
                origin=Pose(position=(0.16, 0, 0.05)),
                axis=((0, 0, 1), (0, 1, 0), (1, 0, 0))[i % 3],
                limits=JointLimits(lower=-1, upper=1, velocity=3, acceleration=8),
            )
            for i in range(dof)
        )
        return RobotModel(
            name=name,
            links=tuple(f"l{i}" for i in range(dof + 1)),
            joints=joints,
            joint_order=tuple(j.name for j in reversed(joints)),
            collision_spheres=tuple(
                LinkSphere(link=f"l{i}", center=(0, 0, 0), radius=0.03) for i in (2, 4, dof)
            ),
        )

    models = (model("seven", 7), model("six", 6))
    robots = (
        RobotInstance(
            name="alpha",
            model="seven",
            start=(0,) * 7,
            goal=(0.2,) * 7,
            base_pose=Pose(position=(0, -1, 0)),
        ),
        RobotInstance(
            name="beta",
            model="six",
            start=(0.2,) * 6,
            goal=(0,) * 6,
            base_pose=Pose(position=(0, 1, 0), quaternion=(0, 0, 1, 0)),
        ),
    )
    problem = replace(
        retreat_problem,
        models=models,
        robots=robots,
        config=replace(retreat_problem.config, backend="curobo", device="cuda:0", time_limit=30),
    )
    graphs = tuple(
        JointGraph(
            robot=r.name,
            joint_names=m.joint_order,
            vertices=(r.start, (0.1,) * len(m.joint_order), r.goal),
            edges=((0, 1), (1, 2)),
            start=0,
            goal=2,
        )
        for r, m in zip(robots, models, strict=True)
    )
    with Planner(
        problem, search=SearchConfig(algorithm=algorithm, exact_fallback=False)
    ) as planner:
        result = planner.plan(graphs)
    assert result.status == "success" and result.makespan == 2, result.diagnostics
    assert validate_trajectory(problem, result.trajectory).status == "passed"
    if algorithm == "lacam":
        record_testsuite_property("m4_arms_problem", dumps(problem))
        record_testsuite_property("m4_arms_result", dumps(result))


def test_gpu_coordinated_example_records_a_real_result(cuda, record_testsuite_property):
    import runpy

    module = runpy.run_path(str(Path(__file__).parents[2] / "examples/coordinated_planning.py"))
    problem, result = module["run"]("curobo")
    assert result.status == "success" and result.provenance.backend.device == "cuda:0"
    assert validate_trajectory(problem, result.trajectory).status == "passed"
    record_testsuite_property("m4_problem", dumps(problem))
    record_testsuite_property("m4_result", dumps(result))
