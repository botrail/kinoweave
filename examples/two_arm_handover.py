"""Plan a synthetic two-arm Pick, synchronized Handover, and Place."""

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
    ActionOrder,
    ContactBody,
    ContactWindow,
    GraspAttachment,
    HeldObject,
    JointTrace,
    ManipulationMode,
    ManipulationProblem,
    ModeTransition,
    MultiArmPlanner,
    ObjectSphere,
    PlacedObject,
    RigidObject,
    ScheduleSkeleton,
    Skill,
    SkillOption,
)
from kinoweave.roadmap import PRMConfig


def build_problem(*, backend="reference", device="cpu"):
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
        name="two_arm_handover",
        models=(model,),
        robots=tuple(
            RobotInstance(name=name, model=model.name, start=(x, -0.25, 0.5), goal=(x, -0.25, 0.5))
            for name, x in (("alpha", -0.45), ("beta", 0.45))
        ),
        obstacles=(
            Obstacle(
                name="source", pose=Pose(position=(-0.3, 0, 0)), shape=Box(size=(0.4, 0.4, 0.1))
            ),
            Obstacle(
                name="destination", pose=Pose(position=(0.3, 0, 0)), shape=Box(size=(0.4, 0.4, 0.1))
            ),
            Obstacle(
                name="pillar", pose=Pose(position=(0, 0, 0.17)), shape=Box(size=(0.12, 0.2, 0.34))
            ),
        ),
        config=PlannerConfig(
            backend=backend,
            device=device,
            time_step=0.5,
            validation_time_step=0.025,
            collision_margin=0.005,
            time_limit=120,
            seed=7,
        ),
    )
    source = PlacedObject(object="part", pose=Pose(position=(-0.3, 0, 0.08)), support="source")
    target = PlacedObject(object="part", pose=Pose(position=(0.3, 0, 0.08)), support="destination")
    a = GraspAttachment(robot="alpha", link="tool", pose=Pose(position=(0.05, 0, 0)))
    b = GraspAttachment(robot="beta", link="tool", pose=Pose(position=(-0.05, 0, 0)))
    alpha = HeldObject(object="part", grasps=(a,))
    both = HeldObject(object="part", grasps=(a, b))
    beta = HeldObject(object="part", grasps=(b,))
    obj = ContactBody(kind="object", name="part")

    def skill(name, robots, times, positions, precondition, changes, support=None, window=None):
        contacts = [
            ContactWindow(
                first=ContactBody(kind="robot", name=r, link="tool"),
                second=obj,
                start=0,
                end=times[-1],
            )
            for r in robots
        ]
        if support:
            contacts.append(
                ContactWindow(
                    first=obj,
                    second=ContactBody(kind="obstacle", name=support),
                    start=window[0],
                    end=window[1],
                )
            )
        return Skill(
            name=name,
            kind=name,
            robots=robots,
            minimum_duration=times[-1],
            options=(
                SkillOption(
                    name="default",
                    trace=JointTrace(
                        times=times,
                        interpolation="quintic_stop",
                        tracks=tuple(
                            RobotTrajectory(robot=r, joint_names=model.joint_order, positions=q)
                            for r, q in zip(robots, positions, strict=True)
                        ),
                    ),
                    preconditions=(precondition,),
                    transitions=tuple(
                        ModeTransition(time=t, before=x, after=y) for t, x, y in changes
                    ),
                    contacts=tuple(contacts),
                ),
            ),
        )

    pick = skill(
        "pick_alpha",
        ("alpha",),
        (0, 1, 2.5),
        (((-0.35, 0, 0.3), (-0.35, 0, 0.08), (-0.35, 0, 0.45)),),
        source,
        ((1, source, alpha),),
        "source",
        (0.75, 1.25),
    )
    handover = skill(
        "handover",
        ("alpha", "beta"),
        (0, 1, 1.5, 2.5),
        (
            ((-0.05, 0, 0.45), (-0.05, 0, 0.45), (-0.05, 0, 0.45), (-0.15, 0, 0.45)),
            ((0.15, 0, 0.45), (0.05, 0, 0.45), (0.05, 0, 0.45), (0.05, 0, 0.45)),
        ),
        alpha,
        ((1, alpha, both), (1.5, both, beta)),
    )
    place = skill(
        "place_beta",
        ("beta",),
        (0, 1.5, 2.5),
        (((0.35, 0, 0.45), (0.35, 0, 0.08), (0.35, 0, 0.3)),),
        beta,
        ((1.5, beta, target),),
        "destination",
        (1.25, 1.75),
    )
    return ManipulationProblem(
        scene=scene,
        objects=(RigidObject(name="part", spheres=(ObjectSphere(center=(0, 0, 0), radius=0.03),)),),
        initial_mode=ManipulationMode(name="source", states=(source,)),
        goal_mode=ManipulationMode(name="destination", states=(target,)),
        skeleton=ScheduleSkeleton(
            skills=(pick, handover, place),
            precedence=(
                ActionOrder(before="pick_alpha", after="handover"),
                ActionOrder(before="handover", after="place_beta"),
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=Path("/tmp/two-arm-handover"))
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--refine", action="store_true", help="Improve the verified plan before saving/viewing"
    )
    parser.add_argument("--refinement-seconds", type=float, default=10.0)
    parser.add_argument("--refinement-iterations", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    device = args.device or ("cpu" if args.backend == "reference" else "cuda:0")
    problem = build_problem(backend=args.backend, device=device)
    planner = MultiArmPlanner(problem, prm=PRMConfig(num_samples=0, neighbors=8))
    result = planner.plan()
    args.output.mkdir(parents=True, exist_ok=True)
    dump(problem, args.output / "problem.json")
    refinement = None
    if args.refine and result.status == "success":
        from kinoweave.manipulation import RefinementConfig, refine_manipulation

        dump(result, args.output / "initial-result.json")
        refinement = refine_manipulation(
            problem,
            result,
            config=RefinementConfig(
                time_limit=args.refinement_seconds, max_iterations=args.refinement_iterations
            ),
        )
        result = refinement.result
        details = asdict(refinement)
        details.pop("result")
        (args.output / "refinement.json").write_text(json.dumps(details, indent=2) + "\n")
    dump(result, args.output / "result.json")
    print(
        json.dumps(
            {
                "status": result.status,
                "makespan": result.makespan,
                "statistics": planner.statistics,
                **(
                    {
                        "refinement": {
                            "status": refinement.status,
                            "termination_reason": refinement.termination_reason,
                            "before": refinement.before.total if refinement.before else None,
                            "after": refinement.after.total if refinement.after else None,
                            "accepted": refinement.statistics["accepted"],
                        }
                    }
                    if refinement
                    else {}
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    if result.status != "success":
        print(result.termination_reason, result.validation.issues, result.diagnostics, flush=True)
        raise SystemExit(1)
    if args.view:
        from kinoweave.visualization import ManipulationViewer

        with ManipulationViewer(problem, result, host=args.host, port=args.port) as viewer:
            print(f"Open {viewer.url}", flush=True)
            viewer.run()


if __name__ == "__main__":
    main()
