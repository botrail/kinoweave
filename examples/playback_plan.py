"""Replay a saved common-time plan in memory and emit kinematic observations."""

import argparse
from pathlib import Path

from kinoweave import PlanningProblem, PlanResult
from kinoweave.adapters import MemoryPlayback, check_playback_report
from kinoweave.io import dump, dumps, load


def run(problem_path, plan_path, *, time_step=0.25):
    problem = load(problem_path, expected=PlanningProblem)
    plan = load(plan_path, expected=PlanResult)
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        report = player.record(time_step)
    check_playback_report(problem, plan, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--step", type=float, default=0.25, help="Maximum nominal observation spacing in seconds"
    )
    parser.add_argument("--output", type=Path, help="Save the kinematic playback report")
    args = parser.parse_args()
    report = run(args.problem, args.plan, time_step=args.step)
    if args.output is not None:
        dump(report, args.output)
    print(dumps(report))


if __name__ == "__main__":
    main()
