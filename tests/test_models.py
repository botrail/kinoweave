from dataclasses import FrozenInstanceError, replace

import pytest

from kinoweave import (
    Box,
    ContractError,
    Joint,
    JointLimits,
    LinkSphere,
    Mimic,
    PlannerConfig,
    Pose,
    RobotModel,
    Sphere,
)


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf")])
def test_scalar_inputs_do_not_coerce_strings_bools_or_nonfinite_values(value):
    with pytest.raises(ContractError, match="radius"):
        Sphere(radius=value)


def test_pose_and_shape_contracts_validate_dimensions_and_conventions():
    with pytest.raises(ContractError, match="unit xyzw"):
        Pose(quaternion=(1, 0, 0, 1))
    with pytest.raises(ContractError, match="position"):
        Pose(position=(0, 0))
    with pytest.raises(ContractError, match="positive"):
        Box(size=(1, 0, 1))
    with pytest.raises(ContractError, match="positive"):
        Sphere(radius=-1)


def test_inputs_are_copied_to_immutable_tuples():
    position = [1, 2, 3]
    pose = Pose(position=position)
    position[0] = 99
    assert pose.position == (1.0, 2.0, 3.0)
    with pytest.raises(FrozenInstanceError):
        pose.position = (0, 0, 0)


def test_names_references_and_position_limits_are_checked(problem):
    left, right = problem.robots
    with pytest.raises(ContractError, match="duplicate"):
        replace(problem, robots=(left, replace(right, name=left.name)))
    with pytest.raises(ContractError, match="unknown model"):
        replace(problem, robots=(replace(left, model="missing"), right))
    with pytest.raises(ContractError, match="left.goal.*outside"):
        replace(problem, robots=(replace(left, goal=(2,)), right))
    with pytest.raises(ContractError, match="left.start.*joint order"):
        replace(problem, robots=(replace(left, start=(0, 0)), right))
    with pytest.raises(ContractError, match="duplicate"):
        replace(problem, obstacles=(replace(problem.obstacles[0], name=left.name),))


def test_joint_order_geometry_and_topology_are_checked(problem):
    model = problem.models[0]
    joint = model.joints[0]
    with pytest.raises(ContractError, match="joint_order"):
        replace(model, joint_order=("missing",))
    with pytest.raises(ContractError, match="unknown link"):
        replace(model, collision_spheres=(LinkSphere(link="missing", center=(0, 0, 0), radius=1),))
    with pytest.raises(ContractError, match="unknown link"):
        replace(model, joints=(replace(joint, parent="missing"),))
    with pytest.raises(ContractError, match="one root|cycle"):
        replace(
            model, joints=(joint, Joint(name="back", kind="fixed", parent="carriage", child="base"))
        )
    with pytest.raises(ContractError, match="multiple parents"):
        replace(
            model, joints=(joint, replace(joint, name="second")), joint_order=("slide", "second")
        )


def mimic_model() -> RobotModel:
    return RobotModel(
        name="gripper",
        links=("base", "left", "right", "tool"),
        joint_order=("opening",),
        joints=(
            Joint(
                name="opening",
                kind="prismatic",
                parent="base",
                child="left",
                limits=JointLimits(lower=0, upper=1, velocity=1, acceleration=1),
            ),
            Joint(
                name="follower",
                kind="prismatic",
                parent="left",
                child="right",
                limits=JointLimits(lower=-1, upper=0, velocity=1, acceleration=1),
                mimic=Mimic(joint="opening", multiplier=-1),
            ),
            Joint(name="mount", kind="fixed", parent="right", child="tool"),
        ),
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.01),),
    )


def test_fixed_and_mimic_joints_are_not_independent_coordinates():
    model = mimic_model()
    assert model.joint_values((0.4,)) == {"opening": 0.4, "follower": -0.4, "mount": 0.0}
    with pytest.raises(ContractError, match="joint_order"):
        replace(model, joint_order=("opening", "follower"))
    with pytest.raises(ContractError, match="unknown mimic source"):
        replace(
            model,
            joints=(
                model.joints[0],
                replace(model.joints[1], mimic=Mimic(joint="missing")),
                model.joints[2],
            ),
        )
    with pytest.raises(ContractError, match="mimic cycle"):
        replace(
            model,
            joint_order=(),
            joints=(
                replace(model.joints[0], mimic=Mimic(joint="follower")),
                *model.joints[1:],
            ),
        )


def test_mimic_joint_limits_are_checked_in_instance_states(problem):
    model = mimic_model()
    follower = replace(
        model.joints[1], limits=JointLimits(lower=-0.1, upper=0, velocity=1, acceleration=1)
    )
    model = replace(model, joints=(model.joints[0], follower, model.joints[2]))
    robot = replace(problem.robots[0], model=model.name, start=(0,), goal=(0.5,))
    with pytest.raises(ContractError, match="follower.*outside"):
        replace(problem, models=(model,), robots=(robot,))


def test_continuous_joint_coordinates_are_unwrapped(problem):
    model = problem.models[0]
    joint = replace(
        model.joints[0],
        kind="continuous",
        limits=JointLimits(velocity=1, acceleration=1),
    )
    model = replace(model, joints=(joint,))
    robot = replace(problem.robots[0], start=(7,), goal=(8,))
    unwrapped = replace(problem, models=(model,), robots=(robot,))
    assert unwrapped.robots[0].start == (7.0,)
    with pytest.raises(ContractError, match="unbounded"):
        replace(joint, limits=JointLimits(lower=-1, upper=1, velocity=1, acceleration=1))


@pytest.mark.parametrize(
    "setting",
    [
        {"time_step": 0},
        {"validation_time_step": -1},
        {"seed": True},
        {"seed": -1},
        {"collision_margin": -0.1},
        {"joint_tolerance": -1},
    ],
)
def test_invalid_planning_settings_are_rejected(setting):
    with pytest.raises(ContractError):
        PlannerConfig(**setting)
