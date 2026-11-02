import io
import shutil
from pathlib import Path

import pytest

from kinoweave import Planner, SearchConfig
from kinoweave.assets import load_scene, write_scene
from kinoweave.geometry.reference import forward
from kinoweave.roadmap import PRMConfig
from kinoweave.trajectory import RobotTrajectory, TrajectoryBundle
from kinoweave.validation import validate_trajectory

pytestmark = pytest.mark.assets


@pytest.fixture(scope="module")
def ur5_problem(assets):
    root = Path(__file__).parents[1]
    if not (root / "examples/assets/ur5/source/LICENSE").is_file():
        pytest.fail("Fetch pinned assets with python examples/fetch_ur5.py before --run-assets")
    return load_scene(root / "examples/scenes/two_ur5.yaml")


@pytest.fixture(scope="module")
def ur5_plan(ur5_problem):
    with Planner(
        ur5_problem.problem, search=SearchConfig(algorithm="lacam"), prm=PRMConfig(num_samples=0)
    ) as planner:
        plan = planner.plan()
    assert plan.status == "success"
    return plan


def test_real_urdf_fk_matches_native_with_both_base_frames(ur5_problem):
    import numpy as np
    import trimesh
    import yourdfpy

    root = Path(__file__).parents[1] / "examples/assets/ur5/source"
    source = yourdfpy.URDF.load(
        root / "robots/ur_description/urdf/ur5_robot.urdf", load_meshes=False
    )
    problem = ur5_problem.problem
    model = problem.models[0]
    rng = np.random.default_rng(7)
    for robot in problem.robots:
        x, y, z, w = robot.base_pose.quaternion
        base = trimesh.transformations.quaternion_matrix((w, x, y, z))
        base[:3, 3] = robot.base_pose.position
        for blend in (0, 0.5, 1, *rng.uniform(size=8)):
            q = tuple(a + (b - a) * blend for a, b in zip(robot.start, robot.goal, strict=True))
            source.update_cfg(dict(zip(model.joint_order, q, strict=True)))
            native = forward(model, robot, q)
            for link, pose in zip(native.link_names, native.poses, strict=True):
                expected = base @ source.get_transform(link, "world")
                np.testing.assert_allclose(pose.position, expected[:3, 3], atol=1e-12)
                x, y, z, w = pose.quaternion
                np.testing.assert_allclose(
                    trimesh.transformations.quaternion_matrix((w, x, y, z))[:3, :3],
                    expected[:3, :3],
                    atol=1e-12,
                )


def test_real_models_need_coordination_and_return_verified_retreat(ur5_problem, ur5_plan):
    problem, plan = ur5_problem.problem, ur5_plan
    naive = TrajectoryBundle(
        times=(0, 1, 2, 3, 4),
        tracks=tuple(
            RobotTrajectory(
                robot=r.name,
                joint_names=problem.models[0].joint_order,
                positions=tuple(
                    tuple(a + (b - a) * i / 4 for a, b in zip(r.start, r.goal, strict=True))
                    for i in range(5)
                ),
            )
            for r in problem.robots
        ),
    )
    assert any(issue.code == "collision" for issue in validate_trajectory(problem, naive).issues)
    assert validate_trajectory(problem, plan.trajectory).status == "passed"
    assert plan.makespan == 7
    alpha = next(t for t in plan.trajectory.tracks if t.robot == "alpha")
    assert alpha.positions[1][0] > alpha.positions[2][0]  # a real retreat, before forward progress
    assert alpha.positions[0] == alpha.positions[2]


@pytest.mark.viz
def test_real_meshes_reach_the_browser(viz, ur5_problem, ur5_plan, tmp_path):
    import trimesh
    from playwright.sync_api import expect, sync_playwright

    from kinoweave.assets import load_visuals
    from kinoweave.visualization import ViserViewer

    write_scene(ur5_problem, tmp_path)
    visuals = load_visuals(tmp_path / "assets.manifest.json", ur5_problem.problem)
    assert len(visuals) == 7
    assert all(
        trimesh.load_scene(io.BytesIO(data), file_type="glb").geometry for data in visuals.values()
    )
    with (
        ViserViewer(ur5_problem.problem, ur5_plan, visuals=visuals, port=0) as viewer,
        sync_playwright() as playwright,
    ):
        assert len(viewer._meshes) == 14
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(viewer.url)
            expect(page.get_by_text("Shared-time robot playback")).to_be_visible()
            viewer.seek(2.5)
            expect(page.get_by_text("Reference time: 2.500 s", exact=False)).to_be_visible()
            if viewer.show_goals.value:
                page.get_by_text("Goal outlines", exact=True).click()
            page.wait_for_timeout(1000)  # allow GLB material/geometry uploads to finish
            page.screenshot(path=tmp_path / "viser-ur5.png")
            assert not errors
        finally:
            browser.close()
