import shutil
import threading
from dataclasses import replace

import pytest

from kinoweave import ContractError, Planner, SearchConfig
from kinoweave.visualization import ViserViewer

pytestmark = pytest.mark.viz


@pytest.fixture
def viewed_case(coordination_cases):
    problem, graphs = coordination_cases["retreat"]
    with Planner(problem, search=SearchConfig(algorithm="auto", priority_horizon=8)) as planner:
        plan = planner.plan(graphs)
    assert plan.status == "success"
    return problem, plan


def test_native_frames_drive_scene_nodes_and_display_clock(viz, viewed_case):
    problem, plan = viewed_case
    with ViserViewer(problem, plan, port=0) as viewer:
        observed = viewer.seek(0.5)
        assert observed.robots[1].positions == (0.5, 0.0)
        beta = viewer.frame.robots[1]
        for link, pose in zip(beta.link_names, beta.link_poses, strict=True):
            assert tuple(viewer._links["beta"][link].position) == pose.position
            x, y, z, w = pose.quaternion
            assert tuple(viewer._links["beta"][link].wxyz) == (w, x, y, z)
        assert tuple(viewer._spheres["beta"][0].position) == beta.spheres[0].center
        viewer.playing.value = True
        viewer.speed.value = 2.0
        viewer.tick(0.25)
        assert viewer.frame.observation.reference_time == 1.0
        viewer.tick((plan.makespan - 1) / 2)
        assert viewer.frame.observation.reference_time == 0.0
        assert viewer.playing.value is True
        viewer.loop.value = False
        viewer.tick(10)
        assert viewer.frame.observation.reference_time == plan.makespan
        assert viewer.playing.value is False
        viewer.seek(0.5)
        before = viewer.frame
        with pytest.raises(ContractError):
            viewer.seek(plan.makespan + 1)
        assert viewer.frame is before
    with pytest.raises(RuntimeError, match="closed"):
        viewer.seek(0)


def test_nonidentity_base_orientation_is_applied_once(viz, viewed_case):
    from kinoweave import Pose
    from kinoweave.io import document_digest

    problem, plan = viewed_case
    problem = replace(
        problem,
        robots=tuple(
            replace(r, base_pose=Pose(position=(2, 3, 1), quaternion=(0, 0, 1, 0)))
            for r in problem.robots
        ),
    )
    plan = replace(
        plan,
        provenance=replace(
            plan.provenance, problem_digest=document_digest(problem), config=problem.config
        ),
    )
    with ViserViewer(problem, plan, port=0) as viewer:
        viewer.seek(0.5)
        handle = viewer._links["beta"]["tool"]
        assert tuple(handle.position) == pytest.approx((1.5, 3, 1))
        assert tuple(handle.wxyz) == (0, 0, 0, 1)


def test_browser_can_seek_toggle_and_play(viz, viewed_case, tmp_path):
    from playwright.sync_api import expect, sync_playwright

    problem, plan = viewed_case
    with ViserViewer(problem, plan, port=0) as viewer, sync_playwright() as playwright:
        thread = threading.Thread(target=viewer.run, daemon=True)
        thread.start()
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(viewer.url)
            expect(page.get_by_text("Shared-time robot playback")).to_be_visible()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).to_be_visible()
            slider = page.get_by_role("slider").first
            slider.focus()
            slider.press("Home")
            slider.press("ArrowRight", delay=10)
            expect(page.get_by_text("Reference time: 0.001 s", exact=False)).to_be_visible()
            assert viewer.frame.observation.reference_time == pytest.approx(0.001)
            page.get_by_text("Link axes", exact=True).click()
            expect(page.locator("canvas[data-engine]")).to_be_visible()
            page.screenshot(path=tmp_path / "viser-json.png")
            page.get_by_role("button", name="Reset", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).to_be_visible()
            page.get_by_text("Play", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).not_to_be_visible()
            assert viewer.frame.observation.reference_time > 0
            page.get_by_text("Play", exact=True).click()
        finally:
            browser.close()
            viewer.close()
            thread.join(timeout=5)
            assert not thread.is_alive()
