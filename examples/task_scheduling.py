"""Retry grounded task choices and improve a serial inspection to parallel work."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kinoweave import (
    Box,
    Joint,
    JointLimits,
    LinkSphere,
    Obstacle,
    PlannerConfig,
    PlanningProblem,
    Pose,
    RobotInstance,
    RobotModel,
    RobotTrajectory,
)
from kinoweave.io import dump
from kinoweave.manipulation import (
    JointTrace,
    ManipulationMode,
    ManipulationSearchConfig,
    ScheduleConfig,
    SchedulePlanner,
    Skill,
    SkillOption,
    Task,
    TaskAlternative,
    TaskDomain,
)
from kinoweave.roadmap import PRMConfig


def build_domain(*, backend="reference", device="cpu"):
    homes = {"alpha": (-0.5, -0.3, 0.4), "beta": (-0.5, 0.3, 0.4)}
    links = ("base", "x_carriage", "y_carriage", "tool")
    model = RobotModel(
        name="cartesian_tool",
        links=links,
        joint_order=("x", "y", "z"),
        joints=tuple(
            Joint(
                name=name,
                kind="prismatic",
                parent=links[i],
                child=links[i + 1],
                axis=axis,
                limits=JointLimits(lower=lo, upper=hi, velocity=1, acceleration=2),
            )
            for i, (name, axis, lo, hi) in enumerate(
                (
                    ("x", (1, 0, 0), -0.7, 0.7),
                    ("y", (0, 1, 0), -0.7, 0.7),
                    ("z", (0, 0, 1), 0.05, 0.8),
                )
            )
        ),
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.02),),
    )
    scene = PlanningProblem(
        name="inspection_tasks",
        models=(model,),
        robots=tuple(
            RobotInstance(name=name, model=model.name, start=q, goal=q) for name, q in homes.items()
        ),
        obstacles=(
            Obstacle(
                name="pillar",
                pose=Pose(position=(-0.15, 0, 0.15)),
                shape=Box(size=(0.1, 0.1, 0.3)),
            ),
        ),
        config=PlannerConfig(
            backend=backend,
            device=device,
            time_step=0.5,
            validation_time_step=0.025,
            collision_margin=0.005,
            seed=7,
        ),
    )

    def alternative(task, robot):
        upper = (-0.3 if task == "A" else 0, homes[robot][1], 0.55)
        target = (-0.15 if task == "A" else 0.15, 0, 0.4)
        options = []
        # The first concrete parameter choice enters the pillar. It is
        # structurally valid, but must fail sampled geometric planning.
        for name in ("low", "upper") if (task, robot) == ("A", "alpha") else ("upper",):
            approach = (-0.15, 0, 0.15) if name == "low" else upper
            options.append(
                SkillOption(
                    name=name,
                    trace=JointTrace(
                        times=(0, 1.5, 3, 4.5, 6),
                        interpolation="quintic_stop",
                        tracks=(
                            RobotTrajectory(
                                robot=robot,
                                joint_names=("x", "y", "z"),
                                positions=(homes[robot], approach, target, upper, homes[robot]),
                            ),
                        ),
                    ),
                )
            )
        return TaskAlternative(
            name=robot,
            parameters=(("assigned_robot", robot),),
            skills=(
                Skill(
                    name="inspect",
                    kind="inspect",
                    robots=(robot,),
                    options=tuple(options),
                    minimum_duration=6,
                ),
            ),
        )

    return TaskDomain(
        name="two_inspections",
        scene=scene,
        objects=(),
        initial_mode=ManipulationMode(name="initial", states=()),
        goal_mode=ManipulationMode(name="goal", states=()),
        tasks=tuple(
            Task(name=task, alternatives=tuple(alternative(task, robot) for robot in homes))
            for task in ("A", "B")
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--candidate-seconds", type=float, default=5)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("task-scheduling-output"))
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    domain = build_domain(
        backend=args.backend, device="cuda:0" if args.backend == "curobo" else "cpu"
    )
    refinement = None
    if args.refine:
        from kinoweave.manipulation import RefinementConfig

        refinement = RefinementConfig(time_limit=2, max_iterations=1, max_candidates=16)
    planner = SchedulePlanner(
        domain,
        config=ScheduleConfig(
            time_limit=args.seconds, candidate_time_limit=args.candidate_seconds, max_attempts=16
        ),
        search_config=ManipulationSearchConfig(max_states=512),
        refinement_config=refinement,
        prm=PRMConfig(num_samples=0),
    )
    report = planner.plan()
    args.output.mkdir(parents=True, exist_ok=True)
    dump(domain, args.output / "domain.json")
    details = asdict(report)
    details.pop("problem")
    details.pop("result")
    # History is a compact sidecar; the final typed pair is stored separately.
    for update in details["history"]:
        update.pop("problem")
        update.pop("result")
    (args.output / "schedule.json").write_text(json.dumps(details, indent=2) + "\n")
    if report.result is not None:
        dump(report.problem, args.output / "problem.json")
        dump(report.result, args.output / "result.json")
    print(
        json.dumps(
            {
                "status": report.status,
                "termination_reason": report.termination_reason,
                "solution": report.result.status if report.result else None,
                "attempts": len(report.attempts),
                "improvements": [u.objective.total for u in report.history],
                "statistics": report.statistics,
            },
            indent=2,
        ),
        flush=True,
    )
    if report.result is None:
        raise SystemExit(1)
    if args.view:
        from kinoweave.visualization import ManipulationViewer

        with ManipulationViewer(
            report.problem, report.result, host=args.host, port=args.port
        ) as viewer:
            print(f"Open {viewer.url}", flush=True)
            viewer.run()


if __name__ == "__main__":
    main()
