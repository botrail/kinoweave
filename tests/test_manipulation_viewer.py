import shutil
import threading

import pytest

from examples.single_arm_pick_place import build_problem
from kinoweave import ContractError
from kinoweave.manipulation import HeldObject, SingleArmPlanner
from kinoweave.roadmap import PRMConfig
from kinoweave.visualization import ManipulationViewer

pytestmark = pytest.mark.viz


@pytest.fixture(scope="module")
def viewed_manipulation():
    p = build_problem()
    result = SingleArmPlanner(p, prm=PRMConfig(num_samples=0, neighbors=8)).plan()
    assert result.status == "success"
    return p, result


def test_mode_frames_object_nodes_and_clock(viz, viewed_manipulation):
    problem, result = viewed_manipulation
    with ManipulationViewer(problem, result, port=0) as viewer:
        pick = result.plan.skills[0]
        frame = viewer.seek(pick.end)
        assert isinstance(frame.mode.states[0], HeldObject)
        assert frame.geometry.objects[0].pose.position == pytest.approx((-0.3, 0, 0.47))
        assert tuple(viewer._objects["part"][0].position) == pytest.approx((-0.3, 0, 0.47))
        final = viewer.seek(result.makespan)
        assert final.mode.states == problem.goal_mode.states
        assert final.geometry.objects[0].pose.position == pytest.approx((0.3, 0, 0.08))
        assert tuple(viewer._objects["part"][0].position) == pytest.approx((0.3, 0, 0.08))
        with pytest.raises(ContractError):
            viewer.seek(result.makespan + 1)
        assert viewer.frame is final
        viewer.seek(0)
        viewer.playing.value = True
        viewer.speed.value = 2
        viewer.tick(0.25)
        assert viewer.frame.time == 0.5
        viewer.tick((result.makespan - 0.5) / 2)
        assert viewer.frame.time == 0
        viewer.loop.value = False
        viewer.tick(100)
        assert viewer.frame.time == result.makespan and not viewer.playing.value
    with pytest.raises(RuntimeError):
        viewer.seek(0)


def test_browser_manipulation_seek_play_and_object_modes(viz, viewed_manipulation, tmp_path):
    from playwright.sync_api import expect, sync_playwright

    problem, result = viewed_manipulation
    with ManipulationViewer(problem, result, port=0) as viewer, sync_playwright() as playwright:
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
            expect(page.get_by_text("Manipulation playback", exact=False)).to_be_visible()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).to_be_visible()
            slider = page.get_by_role("slider").first
            slider.focus()
            slider.press("Home")
            slider.press("ArrowRight")
            expect(page.get_by_text("Reference time: 0.001 s", exact=False)).to_be_visible()
            assert viewer.frame.time == pytest.approx(0.001)
            viewer.seek(result.plan.skills[0].end)
            expect(page.get_by_text("held by gantry", exact=True)).to_be_visible()
            expect(page.locator("canvas[data-engine]")).to_be_visible()
            page.screenshot(path=tmp_path / "manipulation-held.png")
            viewer.seek(result.makespan)
            expect(page.get_by_text("placed", exact=True)).to_be_visible()
            page.get_by_role("button", name="Reset", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).to_be_visible()
            page.get_by_text("Play", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).not_to_be_visible()
            page.get_by_text("Play", exact=True).click()
            assert viewer.frame.time > 0
        finally:
            browser.close()
            viewer.close()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("refine", [False, True])
def test_browser_two_arm_handover_shows_dual_and_released_modes(viz, tmp_path, refine):
    from playwright.sync_api import expect, sync_playwright

    from examples.two_arm_handover import build_problem
    from kinoweave.manipulation import MultiArmPlanner

    problem = build_problem()
    result = MultiArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan()
    assert result.status == "success"
    if refine:
        from kinoweave.manipulation import RefinementConfig, refine_manipulation

        report = refine_manipulation(
            problem,
            result,
            config=RefinementConfig(time_limit=30, max_iterations=1, max_candidates=16),
        )
        assert report.after.total < report.before.total
        result = report.result
    with ManipulationViewer(problem, result, port=0) as viewer, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(viewer.url)
            expect(page.get_by_text("Manipulation playback", exact=False)).to_be_visible()
            handover = next(s for s in result.plan.skills if s.skill == "handover")
            frame = viewer.seek(handover.start + 1.25)
            assert len(frame.mode.states[0].grasps) == 2
            assert frame.geometry.objects[0].pose.position == pytest.approx((0, 0, 0.45))
            assert tuple(viewer._objects["part"][0].position) == pytest.approx((0, 0, 0.45))
            expect(page.get_by_text("held by alpha, beta", exact=True)).to_be_visible()
            expect(page.locator("canvas[data-engine]")).to_be_visible()
            filename = "two-arm-handover-refined.png" if refine else "two-arm-handover.png"
            page.screenshot(path=tmp_path / filename)
            viewer.seek(handover.start + 1.75)
            expect(page.get_by_text("held by beta", exact=True)).to_be_visible()
            viewer.seek(result.makespan)
            expect(page.get_by_text("placed", exact=True)).to_be_visible()
            assert viewer.frame.mode.states == problem.goal_mode.states
        finally:
            browser.close()
