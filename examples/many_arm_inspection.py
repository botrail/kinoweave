"""Experimental 2/4/8-tool scheduling in shared or paired synthetic workspaces."""

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from kinoweave import (
    Joint,
    JointLimits,
    LinkSphere,
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
    TaskChoice,
    TaskDomain,
    compile_skeleton,
)
from kinoweave.roadmap import PRMConfig


def build_domain(robots=4, tasks=10, *, seed=0, backend="reference", topology="shared"):
    if robots not in (2, 4, 8) or not 1 <= tasks <= 100:
        raise ValueError("use 2/4/8 robots and 1..100 tasks")
    if topology not in ("shared", "paired"):
        raise ValueError("unknown topology")
    rng = random.Random(seed)
    links = ("base", "x_carriage", "y_carriage", "tool")
    model = RobotModel(
        name="cartesian_proxy",
        links=links,
        joint_order=("x", "y", "z"),
        joints=tuple(
            Joint(
                name=name,
                kind="prismatic",
                parent=links[i],
                child=links[i + 1],
                axis=axis,
                limits=JointLimits(
                    lower=-1 if i < 2 else 0.05, upper=1, velocity=2, acceleration=8
                ),
            )
            for i, (name, axis) in enumerate(
                (
                    ("x", (1, 0, 0)),
                    ("y", (0, 1, 0)),
                    ("z", (0, 0, 1)),
                )
            )
        ),
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.02),),
    )
    homes = tuple(
        (0.5 * math.cos(2 * math.pi * i / robots), 0.5 * math.sin(2 * math.pi * i / robots), 0.4)
        if topology == "shared"
        else (-0.5, -0.3 if i % 2 == 0 else 0.3, 0.4)
        for i in range(robots)
    )
    scene = PlanningProblem(
        name=f"synthetic_{topology}_{robots}_{tasks}",
        models=(model,),
        robots=tuple(
            RobotInstance(
                name=f"r{i}",
                model=model.name,
                start=q,
                goal=q,
                base_pose=Pose(position=(2 * (i // 2), 0, 0)) if topology == "paired" else Pose(),
            )
            for i, q in enumerate(homes)
        ),
        config=PlannerConfig(
            backend=backend,
            device="cuda:0" if backend == "curobo" else "cpu",
            seed=seed,
            time_limit=30,
            time_step=0.5,
            validation_time_step=0.1,
            collision_margin=0.005,
        ),
    )
    catalog = []
    for j in range(tasks):
        # Every target is a configured local pose, not an IK or physical weld.
        round_index = j if topology == "shared" else j // (robots // 2)
        target = ((-0.15 if round_index % 2 == 0 else 0.15), rng.uniform(-0.025, 0.025), 0.4)
        pod = j % (robots // 2)
        participants = range(robots) if topology == "shared" else (2 * pod, 2 * pod + 1)
        alternatives = []
        for i in participants:
            home = homes[i]
            upper = (
                (home[0] * 0.65, home[1] * 0.65, 0.55)
                if topology == "shared"
                else (-0.3 if target[0] < 0 else 0, home[1], 0.55)
            )
            trace = JointTrace(
                times=(0, 1, 2, 2.5, 3.5, 4.5),
                interpolation="quintic_stop",
                tracks=(
                    RobotTrajectory(
                        robot=f"r{i}",
                        joint_names=model.joint_order,
                        positions=(home, upper, target, target, upper, home),
                    ),
                ),
            )
            alternatives.append(
                TaskAlternative(
                    name=f"r{i}",
                    skills=(
                        Skill(
                            name="inspect",
                            robots=(f"r{i}",),
                            kind="inspect",
                            minimum_duration=4.5,
                            options=(SkillOption(name="configured", trace=trace),),
                        ),
                    ),
                )
            )
        catalog.append(Task(name=f"target{j:03d}", alternatives=tuple(alternatives)))
    return TaskDomain(
        name=scene.name,
        scene=scene,
        objects=(),
        tasks=tuple(catalog),
        initial_mode=ManipulationMode(name="initial", states=()),
        goal_mode=ManipulationMode(name="goal", states=()),
    )


class InspectionProducer:
    """Two bounded catalog choices: serial per cell, then balanced assignment."""

    def __init__(self, domain, *, fixed=False):
        self.domain, self.fixed = domain, fixed
        self.attempts = []

    def __iter__(self):
        for balanced in (False,) if self.fixed else (False, True):
            counts = {}
            choices = []
            for task in self.domain.tasks:
                names = tuple(a.name for a in task.alternatives)
                key = names
                index = counts.get(key, 0) % len(names) if balanced else 0
                counts[key] = counts.get(key, 0) + 1
                choices.append(TaskChoice(task=task.name, alternative=names[index]))
            yield compile_skeleton(
                self.domain,
                tuple(choices),
                tuple(f"{t.name}/inspect" for t in self.domain.tasks),
                producer="balanced" if balanced else "serial",
            )

    def feedback(self, attempt):
        self.attempts.append(attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robots", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--topology", choices=("shared", "paired"), default="shared")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("many-arm-output"))
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    domain = build_domain(
        args.robots, args.tasks, topology=args.topology, seed=args.seed, backend=args.backend
    )
    report = SchedulePlanner(
        domain,
        prm=PRMConfig(num_samples=0),
        config=ScheduleConfig(time_limit=args.seconds, candidate_time_limit=args.seconds / 2),
        search_config=ManipulationSearchConfig(max_states=4000, max_constraints=40000),
    ).plan(InspectionProducer(domain))
    args.output.mkdir(parents=True, exist_ok=True)
    dump(domain, args.output / "domain.json")
    record = asdict(report)
    record.pop("problem")
    record.pop("result")
    for update in record["history"]:
        update.pop("problem")
        update.pop("result")
    (args.output / "schedule.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report.status,
                "history": [h.objective.total for h in report.history],
                "statistics": report.statistics,
            },
            indent=2,
        ),
        flush=True,
    )
    if report.result is None:
        raise SystemExit(1)
    dump(report.problem, args.output / "problem.json")
    dump(report.result, args.output / "result.json")
    if args.view:
        from kinoweave.visualization import ManipulationViewer

        with ManipulationViewer(report.problem, report.result, port=8080) as viewer:
            print(f"Open {viewer.url}", flush=True)
            viewer.run()


if __name__ == "__main__":
    main()
