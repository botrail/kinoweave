"""Qualify the imported model against source URDF and actual CUDA output."""

from dataclasses import replace
from pathlib import Path

import pytest

from kinoweave import Planner, SearchConfig
from kinoweave.assets import load_scene
from kinoweave.backends.curobo import CuroboBackend
from kinoweave.geometry.reference import forward
from kinoweave.io import dumps
from kinoweave.roadmap import PRMConfig

pytestmark = pytest.mark.cuda


def test_imported_real_ur5_fk_and_coordination_on_cuda(cuda, assets, record_testsuite_property):
    import numpy as np
    import trimesh
    import yourdfpy

    root = Path(__file__).parents[2]
    bundle = load_scene(root / "examples/scenes/two_ur5.yaml")
    problem = replace(
        bundle.problem, config=replace(bundle.problem.config, backend="curobo", device="cuda:0")
    )
    model = problem.models[0]
    source = yourdfpy.URDF.load(
        root / "examples/assets/ur5/source/robots/ur_description/urdf/ur5_robot.urdf",
        load_meshes=False,
    )
    for robot in problem.robots:
        q = np.array(
            [
                tuple(a + (b - a) * blend for a, b in zip(robot.start, robot.goal, strict=True))
                for blend in (0, 0.25, 0.5, 0.75, 1)
            ]
        )
        with CuroboBackend(problem, robot.name) as backend:
            output = backend.forward(q)
            positions = output.positions[:, 0].cpu().numpy()
            quaternions = output.quaternions[:, 0].cpu().numpy()
            spheres = output.spheres[:, 0].cpu().numpy()
            for i, joints in enumerate(q):
                state = forward(model, robot, tuple(float(x) for x in joints))
                x, y, z, w = robot.base_pose.quaternion
                base = trimesh.transformations.quaternion_matrix((w, x, y, z))
                base[:3, 3] = robot.base_pose.position
                source.update_cfg(dict(zip(model.joint_order, joints, strict=True)))
                for index, link in enumerate(output.link_names):
                    expected = base @ source.get_transform(link, "world")
                    np.testing.assert_allclose(positions[i, index], expected[:3, 3], atol=2e-5)
                    expected_wxyz = trimesh.transformations.quaternion_from_matrix(expected)
                    expected_xyzw = expected_wxyz[[1, 2, 3, 0]]
                    assert abs(float(quaternions[i, index] @ expected_xyzw)) == pytest.approx(
                        1, abs=2e-5
                    )
                np.testing.assert_allclose(
                    spheres[i, :, :3], [s.center for s in state.spheres], atol=2e-5
                )
                np.testing.assert_allclose(
                    spheres[i, :, 3], [s.radius for s in state.spheres], atol=2e-6
                )
    with Planner(
        problem, search=SearchConfig(algorithm="lacam"), prm=PRMConfig(num_samples=0)
    ) as planner:
        plan = planner.plan()
    assert plan.status == "success" and plan.provenance.backend.device == "cuda:0"
    assert plan.makespan == 7 and plan.validation.status == "passed"
    alpha = next(t for t in plan.trajectory.tracks if t.robot == "alpha")
    assert alpha.positions[2] == alpha.positions[0]
    record_testsuite_property("ur5_problem", dumps(problem))
    record_testsuite_property("ur5_plan", dumps(plan))
