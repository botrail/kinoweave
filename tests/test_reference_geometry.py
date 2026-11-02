import math
from dataclasses import replace

import pytest

from kinoweave import (
    Box,
    ContractError,
    Joint,
    JointLimits,
    LinkSphere,
    Mimic,
    Obstacle,
    Pose,
    RobotInstance,
    RobotModel,
    Sphere,
)
from kinoweave.geometry.reference import (
    ReferenceCollisionChecker,
    WorldSphere,
    forward,
    obstacle_clearance,
    sphere_clearance,
)


def test_arbitrary_axis_fixed_mimic_and_unordered_tree():
    limits = JointLimits(lower=-5, upper=5, velocity=10, acceleration=10)
    joints = (
        Joint(
            name="fixed",
            kind="fixed",
            parent="finger",
            child="tip",
            origin=Pose(position=(1, 0, 0)),
        ),
        Joint(
            name="follower",
            kind="prismatic",
            parent="slide",
            child="finger",
            axis=(0, 1, 0),
            limits=limits,
            mimic=Mimic(joint="move", multiplier=-2, offset=0.5),
        ),
        Joint(
            name="move",
            kind="prismatic",
            parent="rotated",
            child="slide",
            axis=(2**-0.5, 2**-0.5, 0),
            limits=limits,
        ),
        Joint(
            name="turn",
            kind="continuous",
            parent="root",
            child="rotated",
            axis=(0, 0, 1),
            limits=JointLimits(velocity=1, acceleration=1),
        ),
    )
    model = RobotModel(
        name="tree",
        links=("tip", "root", "finger", "slide", "rotated"),
        joints=joints,
        joint_order=("move", "turn"),
        collision_spheres=(LinkSphere(link="tip", center=(0, 1, 0), radius=0.1),),
    )
    robot = RobotInstance(
        name="arm", model="tree", start=(0, 0), goal=(1, 1), base_pose=Pose(position=(3, 4, 5))
    )
    state = forward(model, robot, (2**0.5, 4 * math.pi + math.pi / 2))
    # Local slide=(1,1), follower=(0,0.5-2sqrt(2)), fixed=(1,0,0), sphere=(0,1,0).
    local_y = 2.5 - 2 * math.sqrt(2)
    assert state.spheres[0].center == pytest.approx((3 - local_y, 6, 5), abs=1e-12)
    assert state.link_names == model.links


def test_oriented_box_signed_distance_and_sphere_contact():
    sphere = WorldSphere(robot="r", link="l", index=0, center=(1.1, 0.2, 0), radius=0.05)
    box = Obstacle(
        name="box", shape=Box(size=(0.2, 2, 0.2)), pose=Pose(quaternion=(0, 0, 2**-0.5, 2**-0.5))
    )
    assert obstacle_clearance(sphere, box) == pytest.approx(math.sqrt(0.02) - 0.05)
    assert obstacle_clearance(replace(sphere, center=(0.8, 0, 0)), box) == pytest.approx(-0.15)
    first = replace(sphere, center=(0, 0, 0), radius=0.125)
    second = replace(first, robot="s", center=(0.25, 0, 0))
    assert sphere_clearance(first, second) == 0
    obstacle = Obstacle(name="ball", shape=Sphere(radius=0.125), pose=Pose(position=(0.25, 0, 0)))
    assert obstacle_clearance(first, obstacle) == 0


def test_self_ignores_same_link_and_explicit_allowed_pairs(problem):
    model = replace(
        problem.models[0],
        collision_spheres=(
            LinkSphere(link="base", center=(0, 0, 0), radius=0.1),
            LinkSphere(link="carriage", center=(0, 0, 0), radius=0.1),
            LinkSphere(link="carriage", center=(0, 0, 0), radius=0.1),
        ),
    )
    single = replace(problem, models=(model,), robots=problem.robots[:1], obstacles=())
    checker = ReferenceCollisionChecker(single)
    assert len(checker.collisions({"left": (0,)})) == 2
    allowed = replace(model, allowed_collisions=(("carriage", "base"),))
    assert not ReferenceCollisionChecker(replace(single, models=(allowed,))).collisions(
        {"left": (0,)}
    )


def test_every_robot_is_required_and_inter_robot_contact_counts(problem):
    robots = tuple(replace(r, base_pose=Pose()) for r in problem.robots)
    problem = replace(
        problem,
        robots=robots,
        obstacles=(),
        config=replace(problem.config, collision_margin=0, collision_tolerance=0),
    )
    checker = ReferenceCollisionChecker(problem)
    assert checker.collisions({"left": (0,), "right": (0.1,)})
    assert not checker.collisions({"left": (0,), "right": (0.11,)})
    with pytest.raises(ContractError, match="every robot"):
        checker.collisions({"left": (0,)})
    with pytest.raises(ContractError):
        checker.forward("left", (True,))


def test_geometry_overflow_is_an_error():
    first = WorldSphere(robot="a", link="l", index=0, center=(-1e308, 0, 0), radius=1)
    with pytest.raises(ContractError):
        sphere_clearance(first, replace(first, robot="b", center=(1e308, 0, 0)))
