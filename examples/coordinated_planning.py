"""Coordinate two robots through a retreat bay and emit a validated PlanResult."""

import argparse
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from kinoweave import Planner, PlanningProblem, SearchConfig
from kinoweave.io import dump, dumps, load
from kinoweave.search.reference import JointGraph


def run(backend="curobo"):
    problem = load(Path(__file__).parent / "data/retreat.problem.json", expected=PlanningProblem)
    problem = replace(
        problem,
        config=replace(
            problem.config,
            backend=backend,
            device="cuda:0" if backend == "curobo" else "cpu",
            time_limit=30,
        ),
    )
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
    with Planner(problem, search=SearchConfig(algorithm="auto", priority_horizon=8)) as planner:
        result = planner.plan(graphs)
    if result.status != "success":
        raise RuntimeError(f"Coordination failed: {result.status}, {result.diagnostics}")
    result.check_problem(problem)
    return problem, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("curobo", "reference"), default="curobo")
    parser.add_argument(
        "--output", type=Path, help="Directory for matching problem.json and result.json"
    )
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        problem, result = run(args.backend)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        dump(problem, args.output / "problem.json")
        dump(result, args.output / "result.json")
    print(dumps(result))


if __name__ == "__main__":
    main()
