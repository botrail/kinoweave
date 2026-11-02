"""File-based planning, independent validation, and SDK-free playback."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

from . import __version__
from ._contracts import ContractError
from .adapters import MemoryPlayback, check_playback_report, validate_plan_for_adapter
from .io import dump, dumps, load, loads
from .models import PlanningProblem
from .planner import Planner
from .results import PlanResult
from .roadmap import PRMConfig
from .search.space import SearchConfig


def _backend(problem, backend):
    if backend is None:
        return problem
    return replace(
        problem,
        config=replace(
            problem.config, backend=backend, device="cpu" if backend == "reference" else "cuda:0"
        ),
    )


def _run(args):
    if args.command == "import-scene":
        from .assets import load_scene, write_scene

        scene = load_scene(args.scene)
        write_scene(scene, args.output)
        return scene.problem, 0
    if args.command == "example":
        problem = loads(
            files("kinoweave").joinpath("data/two_arms.problem.json").read_text(encoding="utf-8"),
            expected=PlanningProblem,
        )
        problem = _backend(problem, args.backend)
        dump(problem, args.output)
        return problem, 0

    problem = load(args.problem, expected=PlanningProblem)
    if args.command == "plan":
        original = problem
        problem = _backend(problem, args.backend)
        search = SearchConfig(algorithm=args.algorithm)
        prm = PRMConfig(num_samples=args.samples, neighbors=args.neighbors)
        asset_path = args.problem.parent / "assets.manifest.json"
        if asset_path.is_file():
            from .assets.manifest import copy_assets

            copy_assets(asset_path, original, args.output, problem)
        with Planner(
            problem,
            search=search,
            prm=prm,
        ) as planner:
            result = planner.plan()
        args.output.mkdir(parents=True, exist_ok=True)
        dump(problem, args.output / "problem.json")
        dump(result, args.output / "result.json")
        return result, 0 if result.status == "success" else 1

    plan = load(args.result, expected=PlanResult)
    if args.command == "view":
        from .assets import load_visuals
        from .visualization import ViserViewer

        asset_path = args.assets or args.problem.parent / "assets.manifest.json"
        visuals = (
            load_visuals(asset_path, problem)
            if args.assets is not None or asset_path.is_file()
            else None
        )
        with ViserViewer(
            problem,
            plan,
            host=args.host,
            port=args.port,
            fps=args.fps,
            autoplay=args.autoplay,
            visuals=visuals,
            max_validation_samples=args.max_samples,
        ) as viewer:
            print(f"KinoWeave viewer: {viewer.url}", file=sys.stderr, flush=True)
            viewer.run()
        return {"status": "closed"}, 0
    if args.command == "validate":
        verification = validate_plan_for_adapter(
            problem, plan, MemoryPlayback.capabilities, max_samples=args.max_samples
        )
        return asdict(verification), 0

    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        report = player.record(args.step, max_samples=args.max_samples)
    check_playback_report(problem, plan, report)
    if args.output is not None:
        dump(report, args.output)
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"KinoWeave {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    scene = commands.add_parser(
        "import-scene", help="Compile optional URDF/YAML into portable problem and link assets"
    )
    scene.add_argument("scene", type=Path)
    scene.add_argument("--output", required=True, type=Path)
    example = commands.add_parser("example", help="Save a self-contained two-arm problem")
    example.add_argument("--output", required=True, type=Path)
    example.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    plan = commands.add_parser("plan", help="Build PRMs and save the effective problem and result")
    plan.add_argument("problem", type=Path)
    plan.add_argument(
        "--output", required=True, type=Path, help="Directory for problem/result JSON"
    )
    plan.add_argument(
        "--backend", choices=("reference", "curobo"), help="Explicit backend override"
    )
    plan.add_argument(
        "--algorithm", choices=("lacam", "prioritized", "auto", "joint_astar"), default="lacam"
    )
    plan.add_argument("--samples", type=int, default=128)
    plan.add_argument("--neighbors", type=int, default=12)
    validate = commands.add_parser("validate", help="Independently revalidate a successful plan")
    playback = commands.add_parser("playback", help="Record SDK-free kinematic playback")
    view = commands.add_parser("view", help="Show saved plans in an optional Viser browser viewer")
    for command in (validate, playback, view):
        command.add_argument("problem", type=Path)
        command.add_argument("result", type=Path)
        command.add_argument("--max-samples", type=int, default=100_000)
    playback.add_argument("--step", type=float, default=0.25, help="Maximum nominal spacing [s]")
    playback.add_argument("--output", type=Path)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--fps", type=float, default=30)
    view.add_argument("--autoplay", action="store_true")
    view.add_argument(
        "--assets", type=Path, help="Link visuals; defaults to sibling assets.manifest.json"
    )
    args = parser.parse_args(argv)
    try:
        # SDK diagnostics must not corrupt the machine-readable stdout document.
        with redirect_stdout(sys.stderr):
            result, code = _run(args)
        print(
            json.dumps(result, sort_keys=True, allow_nan=False)
            if isinstance(result, dict)
            else dumps(result)
        )
        return code
    except (ContractError, OSError, ValueError) as error:
        print(f"kinoweave: {error}", file=sys.stderr)
        return 2
