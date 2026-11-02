"""Compile, plan, and optionally view two UR5 models in a shared workspace."""

import argparse
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from kinoweave import Planner, SearchConfig
from kinoweave.assets import load_scene, load_visuals, write_scene
from kinoweave.io import dump, dumps
from kinoweave.roadmap import PRMConfig


def run(output, backend="reference", independent=False):
    filename = "two_ur5_independent.yaml" if independent else "two_ur5.yaml"
    bundle = load_scene(Path(__file__).parent / "scenes" / filename)
    problem = replace(
        bundle.problem,
        config=replace(
            bundle.problem.config,
            backend=backend,
            device="cpu" if backend == "reference" else "cuda:0",
        ),
    )
    bundle = replace(bundle, problem=problem)
    write_scene(bundle, output)
    with Planner(
        problem, search=SearchConfig(algorithm="lacam"), prm=PRMConfig(num_samples=0)
    ) as planner:
        plan = planner.plan()
    dump(plan, Path(output) / "result.json")
    return problem, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--independent", action="store_true", help="Use separated bases without interference"
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        problem, plan = run(args.output, args.backend, args.independent)
    print(dumps(plan), flush=True)
    if plan.status != "success":
        return 1
    if args.view:
        from kinoweave.visualization import ViserViewer

        with ViserViewer(
            problem,
            plan,
            visuals=load_visuals(args.output / "assets.manifest.json", problem),
            port=args.port,
            autoplay=True,
        ) as viewer:
            print(f"KinoWeave viewer: {viewer.url}", file=sys.stderr, flush=True)
            viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
