import shutil

import pytest

from examples.eight_ur5 import WeldingViewer
from kinoweave.assets import load_visuals
from kinoweave.io import load

pytestmark = pytest.mark.viz


def test_browser_eight_real_mesh_arms_progress_and_controls(viz, tmp_path):
    import json
    from pathlib import Path

    from playwright.sync_api import expect, sync_playwright

    root = Path(__file__).parents[1] / "examples"
    p = load(root / "data/eight_ur5.problem.json")
    r = load(root / "data/eight_ur5.result.json")
    metadata = json.loads((root / "data/eight_ur5.demo.json").read_text())
    visuals = load_visuals(root / "assets/eight_ur5/assets.manifest.json", p.scene)
    with (
        WeldingViewer(p, r, metadata, visuals=visuals, port=0) as viewer,
        sync_playwright() as playwright,
    ):
        assert len(viewer._meshes) == 64
        assert not viewer.show_collision.value and not viewer.show_skeleton.value
        browser = playwright.chromium.launch(
            executable_path=shutil.which("google-chrome"),
            headless=True,
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(viewer.url)
            expect(page.get_by_text("Eight UR5 arms", exact=True)).to_be_visible()
            assert viewer.waiting_count == 4
            expect(page.get_by_text("4 waiting", exact=False)).to_be_visible()
            viewer.seek(1.8)
            page.wait_for_timeout(4000)
            page.screenshot(path=tmp_path / "eight-ur5.png", animations="disabled")
            page.get_by_role("button", name="Show baseline collision").click()
            expect(page.get_by_text("UNSAFE comparison", exact=False)).to_be_visible()
            assert viewer.colliding_pairs >= 4 and not viewer.frame.geometry.valid
            assert not viewer._verification_note.visible
            assert viewer.result.status == "success"  # Stored result is still the verified plan.
            page.screenshot(path=tmp_path / "uncoordinated.png", animations="disabled")
            viewer.set_comparison(False)
            assert viewer.frame.geometry.valid and viewer.colliding_pairs == 0
            assert viewer._verification_note.visible
            viewer.seek(r.makespan)
            expect(page.get_by_text("40 / 40 sites complete", exact=False)).to_be_visible()
            assert all(marker.color == (88, 225, 145) for marker in viewer._targets.values())
            viewer.show_meshes.value = False
            viewer.seek(0)
            assert all(not mesh.visible for mesh in viewer._meshes)
            assert "0 / 40 sites complete" in viewer.progress.content
            viewer.show_meshes.value = True
            viewer.loop.value = False
            viewer.playing.value = True
            viewer.tick(1)
            assert viewer.frame.time == 1
            assert not errors
        finally:
            browser.close()
