"""Create and round-trip a symbolic two-arm transfer input; no planning is run."""

import argparse
from pathlib import Path

from kinoweave import (
    Joint,
    JointLimits,
    LinkSphere,
    PlannerConfig,
    PlanningProblem,
    PlanProvenance,
    Pose,
    RobotInstance,
    RobotModel,
    RobotTrajectory,
)
from kinoweave.io import document_digest, dump, load
from kinoweave.manipulation import (
    ActionOrder,
    ContactBody,
    ContactWindow,
    GraspAttachment,
    HeldObject,
    JointTrace,
    ManipulationMode,
    ManipulationProblem,
    ManipulationResult,
    ModeTransition,
    ObjectSphere,
    PlacedObject,
    RigidObject,
    ScheduleSkeleton,
    Skill,
    SkillOption,
)


def build_problem():
    model = RobotModel(
        name="rail_arm",
        links=("base", "tool"),
        joints=(
            Joint(
                name="slide",
                kind="prismatic",
                parent="base",
                child="tool",
                axis=(1, 0, 0),
                limits=JointLimits(lower=-1, upper=1, velocity=1, acceleration=2),
            ),
        ),
        joint_order=("slide",),
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.04),),
    )
    scene = PlanningProblem(
        name="symbolic_two_arm_transfer",
        models=(model,),
        robots=tuple(
            RobotInstance(
                name=name,
                model=model.name,
                base_pose=Pose(position=(x, 0, 0)),
                start=(0,),
                goal=(0,),
            )
            for name, x in (("alpha", -0.5), ("beta", 0.5))
        ),
        config=PlannerConfig(backend="reference", device="cpu", time_step=0.25),
    )
    placed = PlacedObject(object="part", pose=Pose(position=(-0.5, 0, 0)))
    target = PlacedObject(object="part", pose=Pose(position=(0.3, 0, 0)))
    alpha_grasp = GraspAttachment(robot="alpha", link="tool")
    beta_grasp = GraspAttachment(robot="beta", link="tool", pose=Pose(position=(-0.2, 0, 0)))
    alpha = HeldObject(object="part", grasps=(alpha_grasp,))
    both = HeldObject(object="part", grasps=(alpha_grasp, beta_grasp))
    beta = HeldObject(object="part", grasps=(beta_grasp,))

    def skill(name, robots, endpoints, precondition, transitions=(), kind="carry"):
        return Skill(
            name=name,
            kind=kind,
            robots=robots,
            minimum_duration=0.5,
            options=(
                SkillOption(
                    name="default",
                    trace=JointTrace(
                        times=(0, 0.5),
                        tracks=tuple(
                            RobotTrajectory(robot=r, joint_names=("slide",), positions=q)
                            for r, q in zip(robots, endpoints, strict=True)
                        ),
                    ),
                    preconditions=(precondition,),
                    transitions=transitions,
                    contacts=tuple(
                        ContactWindow(
                            first=ContactBody(kind="robot", name=r, link="tool"),
                            second=ContactBody(kind="object", name="part"),
                            start=0,
                            end=0.5,
                        )
                        for r in robots
                    ),
                ),
            ),
        )

    skills = (
        skill(
            "pick_alpha",
            ("alpha",),
            (((0,), (0,)),),
            placed,
            (ModeTransition(time=0.25, before=placed, after=alpha),),
            kind="pick",
        ),
        skill("carry_alpha", ("alpha",), (((0,), (0.4,)),), alpha),
        skill(
            "handover",
            ("alpha", "beta"),
            (((0.4,), (0.4,)), ((-0.4,), (-0.4,))),
            alpha,
            (
                ModeTransition(time=0.25, before=alpha, after=both),
                ModeTransition(time=0.5, before=both, after=beta),
            ),
            kind="handover",
        ),
        skill("carry_beta", ("beta",), (((-0.4,), (0,)),), beta),
        skill(
            "place_beta",
            ("beta",),
            (((0,), (0,)),),
            beta,
            (ModeTransition(time=0.25, before=beta, after=target),),
            kind="place",
        ),
    )
    return ManipulationProblem(
        scene=scene,
        objects=(RigidObject(name="part", spheres=(ObjectSphere(center=(0, 0, 0), radius=0.02),)),),
        initial_mode=ManipulationMode(name="initial", states=(placed,)),
        goal_mode=ManipulationMode(name="goal", states=(target,)),
        skeleton=ScheduleSkeleton(
            skills=skills,
            precedence=tuple(
                ActionOrder(before=first.name, after=second.name)
                for first, second in zip(skills, skills[1:], strict=False)
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    problem = build_problem()
    result = ManipulationResult(
        provenance=PlanProvenance(
            problem_digest=document_digest(problem), config=problem.scene.config
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dump(problem, args.output / "problem.json")
    dump(result, args.output / "result.json")
    restored = load(args.output / "result.json", expected=ManipulationResult)
    restored.check_problem(load(args.output / "problem.json", expected=ManipulationProblem))
    print(f"Wrote manipulation contracts to {args.output}; planning status: {restored.status}")


if __name__ == "__main__":
    main()
