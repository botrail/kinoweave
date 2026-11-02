"""Generate a verified Cartesian-gripper Pick & Place, with optional Viser playback."""

import argparse
import json
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
    ObjectSphere,
    PlacedObject,
    RigidObject,
    ScheduleSkeleton,
    SingleArmPlanner,
    Skill,
    SkillOption,
)
from kinoweave.roadmap import PRMConfig


def build_problem(*, backend="reference", device="cpu"):
    """Synthetic kinematic proxy; not a model of physical grasp forces."""
    links = ("base", "x_carriage", "y_carriage", "tool")
    model = RobotModel(
        name="cartesian_gripper",
        links=links,
        joint_order=("x", "y", "z"),
        joints=tuple(
            Joint(
                name=name,
                kind="prismatic",
                parent=links[i],
                child=links[i + 1],
                axis=axis,
                limits=JointLimits(lower=lower, upper=upper, velocity=1, acceleration=2),
            )
            for i, (name, axis, lower, upper) in enumerate(
                (
                    ("x", (1, 0, 0), -0.7, 0.7),
                    ("y", (0, 1, 0), -0.7, 0.7),
                    ("z", (0, 0, 1), 0.1, 0.8),
                )
            )
        ),
        collision_spheres=(
            LinkSphere(link="tool", center=(0, 0, 0), radius=0.02),
            LinkSphere(link="tool", center=(-0.04, 0, -0.08), radius=0.01),
            LinkSphere(link="tool", center=(0.04, 0, -0.08), radius=0.01),
        ),
    )
    home = (-0.45, -0.25, 0.5)
    scene = PlanningProblem(
        name="single_arm_pick_place",
        models=(model,),
        robots=(RobotInstance(name="gantry", model=model.name, start=home, goal=home),),
        obstacles=(
            Obstacle(
                name="source", pose=Pose(position=(-0.3, 0, 0)), shape=Box(size=(0.4, 0.4, 0.1))
            ),
            Obstacle(
                name="destination", pose=Pose(position=(0.3, 0, 0)), shape=Box(size=(0.4, 0.4, 0.1))
            ),
            Obstacle(
                name="pillar", pose=Pose(position=(0, 0, 0.17)), shape=Box(size=(0.15, 0.2, 0.34))
            ),
        ),
        config=PlannerConfig(
            backend=backend,
            device=device,
            seed=7,
            time_limit=120,
            time_step=0.5,
            validation_time_step=0.025,
            collision_margin=0.005,
        ),
    )
    source = PlacedObject(object="part", pose=Pose(position=(-0.3, 0, 0.08)), support="source")
    target = PlacedObject(object="part", pose=Pose(position=(0.3, 0, 0.08)), support="destination")
    held = HeldObject(
        object="part",
        grasps=(
            GraspAttachment(
                robot="gantry",
                link="tool",
                pose=Pose(position=(0, 0, -0.08)),
            ),
        ),
    )
    obj = ContactBody(kind="object", name="part")
    tool = ContactBody(kind="robot", name="gantry", link="tool")

    def skill(name, times, positions, before, after, change, tool_window, support_window, support):
        return Skill(
            name=name,
            kind=name,
            robots=("gantry",),
            minimum_duration=times[-1],
            options=(
                SkillOption(
                    name="vertical",
                    preconditions=(before,),
                    trace=JointTrace(
                        times=times,
                        interpolation="quintic_stop",
                        tracks=(
                            RobotTrajectory(
                                robot="gantry", joint_names=model.joint_order, positions=positions
                            ),
                        ),
                    ),
                    transitions=(ModeTransition(time=change, before=before, after=after),),
                    contacts=(
                        ContactWindow(
                            first=tool, second=obj, start=tool_window[0], end=tool_window[1]
                        ),
                        ContactWindow(
                            first=obj,
                            second=ContactBody(kind="obstacle", name=support),
                            start=support_window[0],
                            end=support_window[1],
                        ),
                    ),
                ),
            ),
        )

    pick = skill(
        "pick",
        (0, 1, 2.5),
        ((-0.3, 0, 0.30), (-0.3, 0, 0.16), (-0.3, 0, 0.55)),
        source,
        held,
        1,
        (0.5, 1.25),
        (0.75, 1.25),
        "source",
    )
    place = skill(
        "place",
        (0, 1.5, 2.5),
        ((0.3, 0, 0.55), (0.3, 0, 0.16), (0.3, 0, 0.30)),
        held,
        target,
        1.5,
        (1.25, 2.0),
        (1.25, 1.75),
        "destination",
    )
    return ManipulationProblem(
        scene=scene,
        objects=(RigidObject(name="part", spheres=(ObjectSphere(center=(0, 0, 0), radius=0.03),)),),
        initial_mode=ManipulationMode(name="source", states=(source,)),
        goal_mode=ManipulationMode(name="destination", states=(target,)),
        skeleton=ScheduleSkeleton(
            skills=(pick, place), precedence=(ActionOrder(before="pick", after="place"),)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("out/pick-place"))
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    device = args.device or ("cpu" if args.backend == "reference" else "cuda:0")
    problem = build_problem(backend=args.backend, device=device)
    planner = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0, neighbors=8))
    result = planner.plan()
    args.output.mkdir(parents=True, exist_ok=True)
    dump(problem, args.output / "problem.json")
    dump(result, args.output / "result.json")
    print(
        json.dumps(
            {
                "status": result.status,
                "makespan": result.makespan,
                "statistics": planner.statistics,
            },
            indent=2,
        ),
        flush=True,
    )
    if result.status != "success":
        print(result.termination_reason, result.diagnostics, flush=True)
        raise SystemExit(1)
    if args.view:
        from kinoweave.visualization import ManipulationViewer

        with ManipulationViewer(problem, result, host=args.host, port=args.port) as viewer:
            print(f"Open {viewer.url}", flush=True)
            viewer.run()


if __name__ == "__main__":
    main()
