from dataclasses import replace
from pathlib import Path

import pytest

from kinoweave import (
    BackendInfo,
    PlanningProblem,
    PlanProvenance,
    PlanResult,
    RobotTrajectory,
    TrajectoryBundle,
    ValidationReport,
)
from kinoweave.io import document_digest, load
from kinoweave.search.reference import JointGraph


@pytest.fixture
def retreat_problem() -> PlanningProblem:
    return load(
        Path(__file__).parents[1] / "examples/data/retreat.problem.json", expected=PlanningProblem
    )


@pytest.fixture
def coordination_cases(retreat_problem):
    def graphs(first, second, edges_a, edges_b=None, start_a=0, goal_a=2, start_b=2, goal_b=0):
        return (
            JointGraph(
                robot="alpha",
                joint_names=("x", "y"),
                vertices=first,
                edges=edges_a,
                start=start_a,
                goal=goal_a,
            ),
            JointGraph(
                robot="beta",
                joint_names=("x", "y"),
                vertices=second,
                edges=edges_a if edges_b is None else edges_b,
                start=start_b,
                goal=goal_b,
            ),
        )

    line = ((-1, 0), (0, 0), (1, 0))
    tee = (*line, (0, 1))
    path = ((0, 1), (1, 2))
    branch = (*path, (1, 3))
    result = {
        "retreat": (retreat_problem, graphs(tee, tee, branch)),
        "corridor": (retreat_problem, graphs(line, line, path)),
    }
    alpha, beta = retreat_problem.robots
    cross = replace(retreat_problem, robots=(alpha, replace(beta, start=(0, -1), goal=(0, 1))))
    result["crossing"] = (cross, graphs(line, ((0, -1), (0, 0), (0, 1)), path, start_b=0, goal_b=2))
    from kinoweave import Pose

    free = replace(
        retreat_problem, robots=(alpha, replace(beta, base_pose=Pose(position=(0, 3, 0))))
    )
    result["independent"] = (free, graphs(line, line, path))
    evacuate = replace(
        retreat_problem,
        robots=(
            replace(alpha, start=(0, 0), goal=(0, 0)),
            replace(beta, start=(-1, 0), goal=(1, 0)),
        ),
    )
    result["goal_retreat"] = (
        evacuate,
        graphs(tee, tee, ((1, 3),), path, start_a=1, goal_a=1, start_b=0, goal_b=2),
    )
    diamond = ((-1, 0), (0, 1), (1, 0), (0, -1))
    ring = ((0, 1), (1, 2), (0, 3), (3, 2))
    result["reroute"] = (retreat_problem, graphs(diamond, diamond, ring))
    return result


def pytest_addoption(parser):
    parser.addoption("--run-cuda", action="store_true", help="Require real CUDA geometry tests")
    parser.addoption("--run-viz", action="store_true", help="Require Viser and browser tests")
    parser.addoption("--run-assets", action="store_true", help="Require optional URDF/YAML tests")


@pytest.fixture
def viz(request):
    if not request.config.getoption("--run-viz"):
        pytest.skip("optional Viser/browser tests require --run-viz")
    import viser

    return viser


@pytest.fixture(scope="session")
def assets(request):
    if not (request.config.getoption("--run-assets") or request.config.getoption("--run-viz")):
        pytest.skip("optional URDF/YAML tests require --run-assets or --run-viz")
    import yourdfpy

    return yourdfpy


@pytest.fixture(scope="session")
def cuda(request):
    if not request.config.getoption("--run-cuda"):
        pytest.skip("real CUDA tests require --run-cuda and the qualified GPU environment")
    import torch

    assert torch.cuda.is_available(), "--run-cuda requires accessible CUDA, not a CPU fallback"
    torch.zeros(1, device="cuda").sum().item()
    return torch


@pytest.fixture
def problem() -> PlanningProblem:
    return load(
        Path(__file__).parents[1] / "examples/data/two_sliders.problem.json",
        expected=PlanningProblem,
    )


@pytest.fixture
def result(problem: PlanningProblem) -> PlanResult:
    # Synthetic schema fixture only: it does not represent a computed or verified motion.
    return PlanResult(
        status="success",
        trajectory=TrajectoryBundle(
            times=(0, problem.config.time_step),
            tracks=tuple(
                RobotTrajectory(
                    robot=robot.name,
                    joint_names=problem.models[0].joint_order,
                    positions=(robot.start, robot.goal),
                )
                for robot in problem.robots
            ),
        ),
        provenance=PlanProvenance(
            problem_digest=document_digest(problem),
            config=problem.config,
            backend=BackendInfo(name="schema_fixture", version="1", device="cpu"),
        ),
        validation=ValidationReport(
            status="passed",
            level="sampled",
            checks=("joint_limits", "collisions", "endpoints"),
            sample_time_step=problem.config.validation_time_step,
            collision_margin=problem.config.collision_margin,
            collision_tolerance=problem.config.collision_tolerance,
        ),
    )
