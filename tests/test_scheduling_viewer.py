import shutil
import threading

import pytest

from examples.task_scheduling import build_domain
from kinoweave.manipulation import ManipulationSearchConfig, SchedulePlanner
from kinoweave.roadmap import PRMConfig
from kinoweave.visualization import ManipulationViewer

pytestmark = pytest.mark.viz


def test_browser_selected_parallel_task_schedule(viz, tmp_path):
    from playwright.sync_api import expect, sync_playwright

    report = SchedulePlanner(
        build_domain(),
        prm=PRMConfig(num_samples=0),
        search_config=ManipulationSearchConfig(max_states=512),
    ).plan()
    assert report.result.makespan == 6
    with (
        ManipulationViewer(report.problem, report.result, port=0) as viewer,
        sync_playwright() as p,
    ):
        thread = threading.Thread(target=viewer.run, daemon=True)
        thread.start()
        browser = p.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(viewer.url)
            expect(page.get_by_text("Manipulation playback", exact=False)).to_be_visible()
            viewer.seek(3)
            expect(page.get_by_text("Reference time: 3.000 s", exact=False)).to_be_visible()
            configurations = viewer.timeline_data.configurations(viewer.frame.time)
            assert configurations["alpha"] == pytest.approx((-0.15, 0, 0.4))
            assert configurations["beta"] == pytest.approx((0.15, 0, 0.4))
            expect(page.locator("canvas[data-engine]")).to_be_visible()
            page.wait_for_timeout(1800)
            page.screenshot(path=tmp_path / "parallel-tasks.png", animations="disabled")
            page.get_by_role("button", name="Reset", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).to_be_visible()
            page.get_by_text("Play", exact=True).click()
            expect(page.get_by_text("Reference time: 0.000 s", exact=False)).not_to_be_visible()
        finally:
            browser.close()
            viewer.close()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_browser_eight_tool_paired_schedule(viz, tmp_path):
    from playwright.sync_api import expect, sync_playwright

    from examples.many_arm_inspection import InspectionProducer, build_domain

    domain = build_domain(8, 8, topology="paired")
    report = SchedulePlanner(domain, prm=PRMConfig(num_samples=0)).plan(InspectionProducer(domain))
    assert report.result.makespan == 4.5
    with (
        ManipulationViewer(report.problem, report.result, port=0) as viewer,
        sync_playwright() as p,
    ):
        browser = p.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(viewer.url)
            expect(page.get_by_text("Manipulation playback", exact=False)).to_be_visible()
            frame = viewer.seek(2)
            assert frame.geometry.valid and len(frame.geometry.robots) == 8
            expect(page.get_by_text("Reference time: 2.000 s", exact=False)).to_be_visible()
            expect(page.locator("canvas[data-engine]")).to_be_visible()
            page.wait_for_timeout(1800)
            page.screenshot(path=tmp_path / "eight-tools.png", animations="disabled")
        finally:
            browser.close()
