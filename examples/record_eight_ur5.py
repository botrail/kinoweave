"""Render the example's verified clock to MP4; optional Playwright and FFmpeg."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def record(
    viewer, output, *, fps=20, speed=1, width=1280, height=800, renderer="auto", compare=True
):
    from PIL import Image, ImageDraw, ImageFont
    from playwright.sync_api import expect, sync_playwright

    if not 1 <= fps <= 60 or not math.isfinite(speed) or not 0.1 <= speed <= 4:
        raise ValueError("video fps must be 1..60 and speed .1..4")
    if renderer not in ("auto", "software"):
        raise ValueError("video renderer must be auto or software")
    ffmpeg = shutil.which("ffmpeg")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg to encode the demo video")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    viewer.playing.value = False
    witness_time = sum(viewer.metadata["targets"][0]["weld_source"]) / 2
    intro_speed = 2 if witness_time > 3 else 1
    intro = (
        [
            (True, min(witness_time, i / fps * intro_speed))
            for i in range(math.ceil((witness_time / intro_speed + 1) * fps))
        ]
        if compare
        else []
    )
    sequence = intro + [
        (False, min(viewer.result.makespan, i / fps * speed))
        for i in range(math.ceil(viewer.result.makespan / speed * fps) + fps)
    ]
    frames = len(sequence)
    first_target = viewer.metadata["targets"][0]
    first_execution = next(s for s in viewer.result.plan.skills if s.skill == first_target["skill"])
    process_begin, process_end = (first_execution.start + s for s in first_target["weld_source"])
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    title = ImageFont.truetype(font_path, 32)
    with sync_playwright() as p:
        software_args = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
        native_args = (
            ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist"]
            if sys.platform.startswith("linux")
            else []
        )
        browser = p.chromium.launch(
            executable_path=chrome,
            headless=True,
            args=["--no-sandbox", *(software_args if renderer == "software" else native_args)],
        )
        effective_renderer = renderer
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            webgl = page.evaluate("() => !!document.createElement('canvas').getContext('webgl2')")
            if not webgl and renderer == "auto":
                browser.close()
                print("Native WebGL unavailable; using software rendering", flush=True)
                browser = p.chromium.launch(
                    executable_path=chrome, headless=True, args=["--no-sandbox", *software_args]
                )
                page = browser.new_page(viewport={"width": width, "height": height})
                effective_renderer = "software"
            page.goto(viewer.url)
            expect(page.get_by_text("Eight UR5 arms", exact=False)).to_be_visible()
            page.wait_for_timeout(5000)
            client = next(iter(viewer.server.get_clients().values()))
            encoder = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                previous, rendered = None, None
                for index, (unsafe, t) in enumerate(sequence):
                    if viewer._unsafe_comparison != unsafe:
                        viewer.set_comparison(unsafe)
                    detail = not unsafe and process_begin - 2 <= t <= process_end + 2
                    key = unsafe, t, detail
                    if key != previous:
                        old_detail = previous[2] if previous else False
                        if detail != old_detail:
                            client.camera.position = (
                                (0.72, -0.20, 0.84) if detail else (2.8, -3.4, 2.25)
                            )
                            client.camera.look_at = (
                                tuple(first_target["position"]) if detail else (0, 0, 0.72)
                            )
                        viewer.seek(t)
                        viewer.server.flush()
                        expect(
                            page.get_by_text(f"Shared time: {t:.3f} s", exact=True)
                        ).to_be_visible(timeout=60000)
                        page.evaluate(
                            "() => new Promise(resolve => "
                            "requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                        )
                        rgba = Image.fromarray(
                            client.get_render(height, width, transport_format="png", timeout=60)
                        )
                        rendered = Image.new("RGB", (width, height), (20, 28, 40))
                        rendered.paste(rgba, mask=rgba.getchannel("A"))
                        previous = key
                    image = rendered.copy()
                    draw = ImageDraw.Draw(image)
                    draw.rounded_rectangle((24, 24, 780, 178), radius=12, fill=(22, 34, 49))
                    draw.text(
                        (44, 39), "KinoWeave · Eight UR5 arms", font=title, fill=(237, 245, 255)
                    )
                    schedule = {s.skill: s for s in viewer.display_result.plan.skills}
                    definitions = {s.name: s for s in viewer.problem.skeleton.skills}
                    done = 0
                    for target in viewer.metadata["targets"]:
                        skill = schedule[target["skill"]]
                        end = skill.start + target["weld_source"][1] / definitions[
                            target["skill"]
                        ].options[0].trace.duration * (skill.end - skill.start)
                        done += t >= end
                    draw.text(
                        (44, 91),
                        f"{done:02d} / 40 sites complete   ·   "
                        f"{t:05.2f} / {viewer.display_result.makespan:.1f} s",
                        font=font,
                        fill=(104, 222, 185),
                    )
                    draw.text(
                        (44, 130),
                        f"UNCOORDINATED · {viewer.colliding_pairs} COLLIDING PAIRS"
                        if unsafe
                        else (
                            f"COORDINATED · {viewer.moving_count} moving · "
                            f"{viewer.waiting_count} waiting"
                        ),
                        font=font,
                        fill=(255, 114, 130) if unsafe else (255, 215, 115),
                    )
                    draw.text(
                        (28, height - 43),
                        (
                            f"Unsafe counterfactual · same workpiece and source motions · "
                            f"{intro_speed}x playback"
                        )
                        if unsafe
                        else (
                            f"CPU-verified kinematic plan · {speed:g}x playback · "
                            "Workpiece seam welding"
                        ),
                        font=font,
                        fill=(205, 218, 232),
                    )
                    encoder.stdin.write(image.tobytes())
                    if index == len(intro) + round(1.8 / speed * fps):
                        image.save(output.with_suffix(".png"))
                    if index == len(intro) + round((process_begin + process_end) / 2 / speed * fps):
                        image.save(output.with_suffix(".closeup.png"))
                    if index % fps == 0:
                        print(f"Rendering {index}/{frames} frames", flush=True)
                encoder.stdin.close()
                errors = encoder.stderr.read().decode()
                if encoder.wait():
                    raise RuntimeError(errors)
            except BaseException:
                encoder.kill()
                encoder.wait()
                raise
        finally:
            browser.close()
    details = {
        "frames": frames,
        "fps": fps,
        "playback_speed": speed,
        "execution_duration": viewer.result.makespan,
        "video_duration": frames / fps,
        "resolution": [width, height],
        "problem_digest": viewer.metadata["problem_digest"],
        "renderer_requested": renderer,
        "renderer_selected": effective_renderer,
        "counterfactual_intro_seconds": len(intro) / fps,
        "workpiece_detail_source_interval": [process_begin - 2, process_end + 2],
    }
    from kinoweave.manipulation.roadmap import fingerprint

    details["plan_digest"] = fingerprint(viewer.result.plan)
    output.with_suffix(".video.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"Video: {output}", flush=True)
    return details
