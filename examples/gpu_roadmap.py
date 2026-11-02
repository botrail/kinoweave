"""Build two individual PRMs and a GPU interference table, without joint search."""

import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

from kinoweave import Obstacle, PlanningProblem, Pose, Sphere
from kinoweave.collision import GeometryCache, GpuCollisionEvaluator, GpuTiling, Motion
from kinoweave.io import load
from kinoweave.roadmap import PRMConfig, build_prm
from kinoweave.validation import validate_trajectory


def run():
    problem = load(Path(__file__).parent / "data/retreat.problem.json", expected=PlanningProblem)
    problem = replace(
        problem,
        config=replace(
            problem.config, backend="curobo", device="cuda:0", validation_time_step=0.125
        ),
        obstacles=(Obstacle(name="block", shape=Sphere(radius=0.2), pose=Pose()),),
    )
    cache = GeometryCache(4 * 1024 * 1024)
    tiling = GpuTiling(max_pair_bytes=64 * 1024)
    with GpuCollisionEvaluator(problem, tiling=tiling, cache=cache) as evaluator:
        roadmaps = tuple(
            build_prm(evaluator, robot.name, config=PRMConfig(num_samples=24, neighbors=5))
            for robot in problem.robots
        )
        candidates = []
        records = {}
        for robot, roadmap in zip(problem.robots, roadmaps, strict=True):
            trajectory = roadmap.trajectory()
            if trajectory is None:
                raise RuntimeError(f"{robot.name}: no path in this sampled roadmap")
            report = validate_trajectory(replace(problem, robots=(robot,)), trajectory)
            if report.status != "passed":
                raise RuntimeError(f"{robot.name}: CPU verification failed: {report.issues}")
            rows = trajectory.tracks[0].positions
            candidates.append(
                (
                    Motion(start=rows[0], end=rows[0]),
                    *(Motion(start=a, end=b) for a, b in pairwise(rows)),
                    Motion(start=rows[-1], end=rows[-1]),
                )
            )
            records[robot.name] = {
                "vertices": len(roadmap.graph.vertices),
                "edges": len(roadmap.graph.edges),
                "path": rows,
                "duration": trajectory.duration,
                "validation": asdict(report),
            }
        table = evaluator.interference("alpha", candidates[0], "beta", candidates[1])
        warm = evaluator.interference("alpha", candidates[0], "beta", candidates[1])
        if warm != table:
            raise RuntimeError("Cached geometry changed the interference table")
        return {
            "stage": "individual_roadmaps_and_interference",
            "backend": asdict(evaluator.info),
            "problem_digest": evaluator.digest,
            "roadmaps": records,
            "table": asdict(table),
            "cache": asdict(cache.stats),
            "max_pair_tile_bytes": evaluator.max_pair_tile_bytes,
            "max_geometry_tile_bytes": evaluator.max_geometry_tile_bytes,
        }


def main():
    # Keep native SDK diagnostics separate from the machine-readable result.
    with redirect_stdout(sys.stderr):
        result = run()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
