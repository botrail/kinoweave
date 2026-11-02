from dataclasses import replace

import pytest

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
    Sphere,
)
from kinoweave.backends.curobo import CuroboBackend

pytestmark = pytest.mark.cuda


def test_cpu_reference_matches_cuda_geometry_and_masks(stage, cuda):
    import numpy as np

    from kinoweave.geometry.reference import ReferenceCollisionChecker

    rows = np.random.default_rng(23).uniform(-0.8, 0.8, (64, 3)).astype("float32")
    checker = ReferenceCollisionChecker(stage)
    expected = []
    with CuroboBackend(stage, "stage") as backend:
        actual = backend.forward(rows)
        for index, row in enumerate(rows):
            values = tuple(float(x) for x in row)
            cpu = checker.forward("stage", values)
            np.testing.assert_allclose(
                actual.spheres[index, 0].cpu(),
                [[*s.center, s.radius] for s in cpu.spheres],
                atol=2e-6,
            )
            expected.append(not checker.collisions({"stage": values}))
        assert backend.collision(rows).valid[:, 0].tolist() == expected


def test_standalone_gpu_example(cuda, capsys):
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).parents[2] / "examples/gpu_geometry.py"), run_name="__main__")
    assert '"ik_success": [\n    true\n  ]' in capsys.readouterr().out


@pytest.fixture
def stage(cuda):
    model = RobotModel(
        name="cartesian_stage",
        links=("base", "x", "y", "tool"),
        joints=(
            Joint(
                name="slide_x",
                kind="prismatic",
                parent="base",
                child="x",
                axis=(1, 0, 0),
                limits=JointLimits(lower=-1, upper=1, velocity=1, acceleration=2),
            ),
            Joint(
                name="slide_y",
                kind="prismatic",
                parent="x",
                child="y",
                axis=(0, 1, 0),
                origin=Pose(position=(0.2, 0, 0.1)),
                limits=JointLimits(lower=-1, upper=1, velocity=2, acceleration=3),
            ),
            Joint(
                name="turn",
                kind="revolute",
                parent="y",
                child="tool",
                axis=(0, 0, 1),
                origin=Pose(position=(0.1, 0, 0)),
                limits=JointLimits(lower=-3, upper=3, velocity=3, acceleration=4),
            ),
        ),
        joint_order=("turn", "slide_y", "slide_x"),
        collision_spheres=(
            LinkSphere(link="tool", center=(0.1, 0, 0.5), radius=0.05),
            LinkSphere(link="base", center=(0, 0, 0.6), radius=0.05),
            LinkSphere(link="tool", center=(0.15, 0, 0.5), radius=0.03),
        ),
    )
    return PlanningProblem(
        name="gpu_stage",
        models=(model,),
        robots=(
            RobotInstance(
                name="stage",
                model=model.name,
                start=(0.2, 0.3, 0.2),
                goal=(0.1, 0.4, 0.5),
                base_pose=Pose(position=(1, -0.5, 0.25), quaternion=(0, 0, 2**-0.5, 2**-0.5)),
            ),
        ),
        obstacles=(
            Obstacle(
                name="box",
                shape=Box(size=(0.1, 0.1, 0.1)),
                pose=Pose(position=(1, -0.1, 0.85), quaternion=(0, 0, 0.2588190451, 0.9659258263)),
            ),
            Obstacle(
                name="sphere", shape=Sphere(radius=0.02), pose=Pose(position=(0.6, 0.4, 0.85))
            ),
        ),
        config=PlannerConfig(backend="curobo", device="cuda:0", collision_margin=0.01),
    )


def reference(model, instance, row):
    # Independent CPU implementation for qualification only, not a production FK backend.
    import numpy as np
    from scipy.spatial.transform import Rotation

    values = dict(zip(model.joint_order, row, strict=True))
    root = next(link for link in model.links if link not in {joint.child for joint in model.joints})
    base = np.eye(4)
    base[:3, :3] = Rotation.from_quat(instance.base_pose.quaternion).as_matrix()
    base[:3, 3] = instance.base_pose.position
    transforms = {root: base}
    pending = list(model.joints)
    while pending:
        for joint in pending[:]:
            if joint.parent not in transforms:
                continue
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_quat(joint.origin.quaternion).as_matrix()
            origin[:3, 3] = joint.origin.position
            motion = np.eye(4)
            if joint.kind == "revolute":
                motion[:3, :3] = Rotation.from_rotvec(
                    np.array(joint.axis) * values[joint.name]
                ).as_matrix()
            elif joint.kind == "prismatic":
                motion[:3, 3] = np.array(joint.axis) * values[joint.name]
            transforms[joint.child] = transforms[joint.parent] @ origin @ motion
            pending.remove(joint)
    spheres = []
    for sphere in model.collision_spheres:
        transform = transforms[sphere.link]
        spheres.append([*(transform[:3, :3] @ sphere.center + transform[:3, 3]), sphere.radius])
    return transforms, np.array(spheres)


def test_world_fk_spheres_ordering_and_persistent_output(stage, cuda):
    import numpy as np
    from scipy.spatial.transform import Rotation

    rows = [[0.2, 0.3, 0.1], [-0.5, -0.2, 0.4], [0, 0, -0.4]]
    with CuroboBackend(stage, "stage") as backend:
        result = backend.forward(rows)
        assert result.positions.is_cuda and result.spheres.dtype == cuda.float32
        assert result.sphere_links == ("tool", "base", "tool")
        for index, row in enumerate(rows):
            transforms, spheres = reference(stage.models[0], stage.robots[0], row)
            np.testing.assert_allclose(result.spheres[index, 0].cpu(), spheres, atol=2e-6)
            for link_index, link in enumerate(result.link_names):
                np.testing.assert_allclose(
                    result.positions[index, 0, link_index].cpu(), transforms[link][:3, 3], atol=2e-6
                )
                observed = Rotation.from_quat(
                    result.quaternions[index, 0, link_index].cpu()
                ).as_matrix()
                np.testing.assert_allclose(observed, transforms[link][:3, :3], atol=2e-6)
        snapshot = result.spheres.clone()
        backend.forward([0, 0, 0])
        assert cuda.equal(result.spheres, snapshot)
        reordered = backend.forward([[0.1, 0.3, 0.2]], joint_names=("slide_x", "slide_y", "turn"))
        assert cuda.allclose(reordered.spheres, result.spheres[:1], atol=2e-6)


def test_self_boxes_spheres_and_limits_are_checked(stage, cuda):
    with CuroboBackend(stage, "stage") as backend:
        result = backend.collision(
            [[0, 0, 0], [0, 0, -0.4], [0, 0.4, 0.2], [0, 0.4, 0.5], [0, 0, 1.2]]
        )
        assert result.world_collision[:, 0].tolist() == [True, False, False, True, False]
        assert result.self_collision[:, 0].tolist() == [False, True, False, False, False]
        assert result.within_limits[:, 0].tolist() == [True, True, True, True, False]
        assert result.valid[:, 0].tolist() == [False, False, True, False, False]


def test_allowed_self_pairs_are_preserved(stage, cuda):
    model = replace(stage.models[0], allowed_collisions=(("base", "tool"),))
    problem = replace(stage, models=(model,))
    with CuroboBackend(problem, "stage") as backend:
        assert not backend.collision([0, 0, -0.4]).self_collision.item()
    clean = replace(problem, obstacles=())
    with CuroboBackend(clean, "stage", use_cuda_graph=False) as backend:
        fk = backend.forward([0, 0, -0.4])
        tip = fk.link_names.index("tool")
        goal = Pose(
            position=tuple(fk.positions[0, 0, tip].tolist()),
            quaternion=tuple(fk.quaternions[0, 0, tip].tolist()),
        )
        assert backend.inverse("tool", (goal,)).success.item()


def test_single_sphere_model_has_no_fake_self_pairs(problem, cuda):
    problem = replace(problem, config=replace(problem.config, backend="curobo", device="cuda:0"))
    with CuroboBackend(problem, "left") as backend:
        out = backend.forward([[0], [0.3]])
        assert out.spheres.shape == (2, 1, 1, 4)
        assert out.spheres[..., 3].flatten().tolist() == pytest.approx([0.05, 0.05])
        assert not backend.collision([[0], [0.3]]).self_collision.any().item()
        assert "_kinoweave_sphere_helper" not in out.link_names


@pytest.mark.parametrize("margin", [0.0, 0.01])
def test_self_collision_margin_threshold(stage, cuda, margin):
    clean = replace(stage, obstacles=(), config=replace(stage.config, collision_margin=margin))
    # World/base rotations cannot change clearance between these two link spheres.
    threshold = margin + clean.config.collision_tolerance
    with CuroboBackend(clean, "stage") as backend:
        offsets = [-0.4 + 0.1 + threshold + delta for delta in (-0.0001, 0.0001)]
        assert backend.collision([[0, 0, x] for x in offsets]).self_collision[:, 0].tolist() == [
            True,
            False,
        ]


def test_default_torch_dtype_does_not_change_geometry(stage, cuda):
    original = cuda.get_default_dtype()
    try:
        cuda.set_default_dtype(cuda.float64)
        with CuroboBackend(stage, "stage") as backend:
            assert backend.forward([0, 0, 0]).positions.dtype == cuda.float32
            assert backend.collision([0, 0, 0]).world_collision.item()
    finally:
        cuda.set_default_dtype(original)


def test_rotated_boxes_and_spheres_against_cpu_distances(stage, cuda):
    import numpy as np
    from scipy.spatial.transform import Rotation

    rows = np.random.default_rng(9).uniform(-0.8, 0.8, (256, 3)).astype("float32")
    expected_world = []
    expected_self = []
    threshold = stage.config.collision_margin + stage.config.collision_tolerance
    for row in rows:
        _, spheres = reference(stage.models[0], stage.robots[0], row)
        hit = False
        for obstacle in stage.obstacles:
            delta = spheres[:, :3] - obstacle.pose.position
            if isinstance(obstacle.shape, Sphere):
                clearance = np.linalg.norm(delta, axis=-1) - obstacle.shape.radius - spheres[:, 3]
            else:
                local = delta @ Rotation.from_quat(obstacle.pose.quaternion).as_matrix()
                excess = np.abs(local) - np.array(obstacle.shape.size) / 2
                signed = np.linalg.norm(np.maximum(excess, 0), axis=-1) + np.minimum(
                    np.max(excess, axis=-1), 0
                )
                clearance = signed - spheres[:, 3]
            hit = hit or bool(np.any(clearance <= threshold))
        expected_world.append(hit)
        expected_self.append(
            any(
                np.linalg.norm(spheres[i, :3] - spheres[j, :3]) - spheres[i, 3] - spheres[j, 3]
                <= threshold
                for i, a in enumerate(stage.models[0].collision_spheres)
                for j, b in enumerate(stage.models[0].collision_spheres)
                if i < j and a.link != b.link
            )
        )
    with CuroboBackend(stage, "stage") as backend:
        result = backend.collision(rows)
        assert result.world_collision[:, 0].tolist() == expected_world
        assert result.self_collision[:, 0].tolist() == expected_self


def test_native_joint_limits_keep_names_and_values(stage, cuda):
    with CuroboBackend(stage, "stage") as backend:
        limits = backend._kinematics.get_joint_limits()
        source = {j.name: j.limits for j in stage.models[0].joints}
        for i, name in enumerate(limits.joint_names):
            value = source[name]
            assert limits.position[:, i].tolist() == pytest.approx([value.lower, value.upper])
            assert limits.velocity[:, i].tolist() == pytest.approx(
                [-value.velocity, value.velocity]
            )
            assert limits.acceleration[:, i].tolist() == pytest.approx(
                [-value.acceleration, value.acceleration]
            )


@pytest.mark.parametrize("shape", [Box(size=(0.25, 0.25, 0.25)), Sphere(radius=0.125)])
def test_exact_contact_with_zero_margin_and_tolerance(problem, cuda, shape):
    model = replace(
        problem.models[0],
        collision_spheres=(
            LinkSphere(link="base", center=(0, 0, 0), radius=0.125),
            LinkSphere(link="carriage", center=(0, 0, 0), radius=0.125),
        ),
    )
    instance = replace(problem.robots[0], base_pose=Pose())
    problem = replace(
        problem,
        models=(model,),
        robots=(instance,),
        obstacles=(Obstacle(name="touch", shape=shape, pose=Pose()),),
        config=replace(
            problem.config,
            backend="curobo",
            device="cuda:0",
            collision_margin=0,
            collision_tolerance=0,
        ),
    )
    with CuroboBackend(problem, instance.name) as backend:
        out = backend.collision([[0.25 - 1e-4], [0.25], [0.25 + 1e-4]])
        assert out.self_collision[:, 0].tolist() == [True, True, False]
    # Exclude the stationary sphere, which otherwise always intersects the obstacle.
    model = replace(model, collision_spheres=model.collision_spheres[1:])
    with CuroboBackend(replace(problem, models=(model,)), instance.name) as backend:
        out = backend.collision([[0.25 - 1e-4], [0.25], [0.25 + 1e-4]])
        assert out.world_collision[:, 0].tolist() == [True, True, False]


def test_horizon_batches_and_bad_inputs(stage, cuda):
    with CuroboBackend(stage, "stage") as backend:
        assert backend.forward([[[0.2, 0.3, 0.1], [0.3, 0.2, 0.4]]]).positions.shape[:2] == (1, 2)
        rows = [[0, 0, 0], [0, 0.4, 0.2]]
        flat = backend.collision(rows).valid
        horizon = backend.collision([rows]).valid
        assert cuda.equal(flat.flatten(), horizon.flatten())
        for bad in ([True, False, True], [float("nan"), 0, 0], [0, 0], []):
            with pytest.raises(ValueError):
                backend.forward(bad)
        with pytest.raises(ValueError, match="joint set"):
            backend.forward([0, 0, 0], joint_names=("turn", "turn", "slide_x"))
    with pytest.raises(RuntimeError, match="closed"):
        backend.forward([0, 0, 0])


@pytest.mark.parametrize("graph", [False, True])
def test_native_ik_world_targets_and_static_revalidation(stage, cuda, graph):
    clean = replace(stage, obstacles=())
    with CuroboBackend(clean, "stage", use_cuda_graph=graph) as backend:
        expected = backend.forward([[0.3, 0.2, 0.4], [-0.2, -0.1, 0.2]])
        tip = expected.link_names.index("tool")
        goals = tuple(
            Pose(position=tuple(p), quaternion=tuple(q))
            for p, q in zip(
                expected.positions[:, 0, tip].cpu().tolist(),
                expected.quaternions[:, 0, tip].cpu().tolist(),
                strict=True,
            )
        )
        solved = backend.inverse("tool", goals)
        assert solved.joint_names == stage.models[0].joint_order
        assert solved.success.tolist() == [True, True]
        assert (solved.position_error <= 1e-3).all().item()
        assert (solved.orientation_error <= 1e-2).all().item()
        actual = backend.forward(solved.positions)
        assert cuda.allclose(actual.positions[:, 0, tip], expected.positions[:, 0, tip], atol=1e-3)
        again = backend.inverse("tool", tuple(reversed(goals)))
        assert again.success.all().item()
        assert cuda.allclose(
            backend.forward(again.positions).positions[:, 0, tip],
            expected.positions[:, 0, tip].flip(0),
            atol=1e-3,
        )
        unreachable = backend.inverse("tool", (Pose(position=(100, 100, 100)),))
        assert not unreachable.success.any().item()
    # Sphere obstacles are screened exactly after native IK, without mesh approximation.
    blocked = replace(
        clean, obstacles=(Obstacle(name="blocked", shape=Sphere(radius=1), pose=goals[0]),)
    )
    with CuroboBackend(blocked, "stage", use_cuda_graph=False) as backend:
        assert not backend.inverse("tool", (goals[0],)).success.item()


def test_seven_dof_rotated_tree_against_independent_cpu(stage, cuda):
    import numpy as np
    from scipy.spatial.transform import Rotation

    joints = tuple(
        Joint(
            name=f"j{i}",
            kind="revolute",
            parent=f"l{i}",
            child=f"l{i + 1}",
            axis=((0, 0, -1), (0, 1, 0), (1, 0, 0))[i % 3],
            origin=Pose(position=(0.15, 0, 0.1), quaternion=(0.5, 0.5, 0.5, 0.5)),
            limits=JointLimits(lower=-2, upper=2, velocity=1 + i / 10, acceleration=2 + i / 10),
        )
        for i in range(7)
    )
    branch = Joint(
        name="fixed_branch",
        kind="fixed",
        parent="l3",
        child="branch",
        origin=Pose(position=(0.1, 0.2, -0.1), quaternion=(0, 2**-0.5, 0, 2**-0.5)),
    )
    model = RobotModel(
        name="seven",
        links=(*tuple(f"l{i}" for i in range(8)), "branch"),
        joints=(*joints, branch),
        joint_order=tuple(j.name for j in reversed(joints)),
        collision_spheres=(
            LinkSphere(link="l0", center=(0, 0, 0), radius=0.01),
            LinkSphere(link="l7", center=(0.05, 0, 0), radius=0.01),
            LinkSphere(link="branch", center=(0.02, -0.01, 0.03), radius=0.01),
        ),
    )
    instance = RobotInstance(
        name="arm",
        model=model.name,
        start=(0,) * 7,
        goal=(0.1,) * 7,
        base_pose=Pose(position=(0.3, -0.4, 0.2), quaternion=(0.5, 0.5, 0.5, 0.5)),
    )
    problem = replace(stage, models=(model,), robots=(instance,), obstacles=())
    rows = np.random.default_rng(7).uniform(-1, 1, (32, 7)).astype("float32")
    with CuroboBackend(problem, "arm") as backend:
        out = backend.forward(rows)
        for i, row in enumerate(rows):
            transforms, spheres = reference(model, instance, row)
            np.testing.assert_allclose(out.spheres[i, 0].cpu(), spheres, atol=3e-6)
            for k, name in enumerate(out.link_names):
                np.testing.assert_allclose(
                    out.positions[i, 0, k].cpu(), transforms[name][:3, 3], atol=3e-6
                )
                np.testing.assert_allclose(
                    Rotation.from_quat(out.quaternions[i, 0, k].cpu()).as_matrix(),
                    transforms[name][:3, :3],
                    atol=3e-6,
                )
