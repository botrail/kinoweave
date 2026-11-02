"""Solve a supplied tiny graph with a retreat bay, independently of GPU/simulation."""

from pathlib import Path

from kinoweave import PlanningProblem
from kinoweave.io import dumps, load
from kinoweave.search.reference import JointGraph, reference_plan


def main():
    problem = load(Path(__file__).parent / "data/retreat.problem.json", expected=PlanningProblem)
    vertices = ((-1, 0), (0, 0), (1, 0), (0, 1))
    edges = ((0, 1), (1, 2), (1, 3))
    graphs = (
        JointGraph(
            robot="alpha", joint_names=("x", "y"), vertices=vertices, edges=edges, start=0, goal=2
        ),
        JointGraph(
            robot="beta", joint_names=("x", "y"), vertices=vertices, edges=edges, start=2, goal=0
        ),
    )
    result = reference_plan(problem, graphs)
    if result.status != "success":
        raise RuntimeError(f"Reference example failed: {result.status}, {result.diagnostics}")
    print(dumps(result))


if __name__ == "__main__":
    main()
