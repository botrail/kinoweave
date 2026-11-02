"""Evaluate one slider on CUDA; this does not plan or coordinate the two robots."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from kinoweave import PlanningProblem, Pose
from kinoweave.backends.curobo import CuroboBackend
from kinoweave.io import load


def main():
    problem = load(
        Path(__file__).parent / "data/two_sliders.problem.json", expected=PlanningProblem
    )
    problem = replace(problem, config=replace(problem.config, backend="curobo", device="cuda:0"))
    robot = problem.robots[0]
    with CuroboBackend(problem, robot.name) as backend:
        fk = backend.forward([robot.start, robot.goal])
        collision = backend.collision([robot.start, robot.goal])
        tip = fk.link_names.index("carriage")
        goal = Pose(
            position=tuple(fk.positions[1, 0, tip].tolist()),
            quaternion=tuple(fk.quaternions[1, 0, tip].tolist()),
        )
        ik = backend.inverse("carriage", (goal,))
        if not ik.success.all().item():
            raise RuntimeError(
                f"IK goal failed: q={ik.positions.tolist()}, "
                f"position_error={ik.position_error.tolist()}, "
                f"orientation_error={ik.orientation_error.tolist()}"
            )
        print(
            json.dumps(
                {
                    "backend": asdict(backend.info),
                    "dependencies": backend.dependency_versions,
                    "robot": robot.name,
                    "joint_names": backend.joint_names,
                    "link_names": fk.link_names,
                    "world_positions": fk.positions.tolist(),
                    "world_spheres": fk.spheres.tolist(),
                    "static_valid": collision.valid.tolist(),
                    "ik_positions": ik.positions.tolist(),
                    "ik_success": ik.success.tolist(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
