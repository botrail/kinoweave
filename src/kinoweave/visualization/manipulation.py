"""Optional browser playback of independently verified manipulation timelines."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote

from .._contracts import ContractError, value_as
from ..manipulation.geometry import ManipulationFrame, ModeGeometry
from ..manipulation.models import HeldObject, ManipulationMode
from ..manipulation.timeline import ManipulationTimeline
from ..manipulation.validation import verify_manipulation
from ..models import Box
from .viser import COLORS, _wxyz


@dataclass(frozen=True, slots=True)
class ManipulationPlaybackFrame:
    time: float
    mode: ManipulationMode
    geometry: ManipulationFrame


class ManipulationViewer:
    """Collision proxies, object modes and original local trace clocks; no physics.

    Input reports are not trusted: even an imported success is rechecked on CPU
    before a server is started. Robot meshes/simulator execution are not supplied.
    """

    def __init__(
        self,
        problem,
        result,
        *,
        policy=None,
        host="127.0.0.1",
        port=8090,
        fps=30,
        autoplay=False,
        max_validation_samples=100_000,
    ):
        self.fps = value_as(fps, float, "fps")
        port = value_as(port, int, "port")
        if not 1 <= self.fps <= 120 or not 0 <= port <= 65535:
            raise ContractError("viewer", "fps must be in [1,120], port in [0,65535]")
        self.result = verify_manipulation(
            problem, result, policy=policy, max_samples=max_validation_samples
        )
        if self.result.status != "success":
            raise ContractError(
                "viewer.plan", f"CPU verification failed: {self.result.validation.issues}"
            )
        try:
            import numpy as np
            import viser
        except ImportError as error:
            raise ContractError(
                "viewer.dependencies", "install kinoweave[viz] to use the viewer"
            ) from error
        self.problem, self._np, self._host = problem, np, host
        self.timeline_data = ManipulationTimeline(problem, self.result)
        self.geometry = ModeGeometry(problem, policy=policy)
        self._closed, self._time = False, 0.0
        self._lock = threading.RLock()
        self._last_wall = time.monotonic()
        self._links, self._spheres, self._skeletons, self._objects, self._labels = (
            {},
            {},
            {},
            {},
            {},
        )
        self._frame = self._evaluate(0)
        self.server = viser.ViserServer(host=host, port=port, label="KinoWeave", verbose=False)
        try:
            self.server.gui.configure_theme(
                control_layout="floating",
                control_width="medium",
                brand_color=(42, 136, 230),
                show_share_button=False,
            )
            self.server.scene.set_up_direction("+z")
            self._build_scene()
            self._build_gui(autoplay)
            self._render()
        except BaseException:
            self.close()
            raise

    @property
    def url(self):
        host = "127.0.0.1" if self._host in ("0.0.0.0", "::") else self._host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.server.get_port()}"

    @property
    def frame(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("viewer is closed")
            return self._frame

    def _evaluate(self, t):
        mode = self.timeline_data.mode(t)
        geometry = self.geometry.snapshot(
            mode, self.timeline_data.configurations(t), contacts=self.timeline_data.contacts(t)
        )
        if not geometry.valid:
            raise ContractError("viewer.frame", f"invalid frame at {t}: {geometry.issues}")
        return ManipulationPlaybackFrame(t, mode, geometry)

    def _build_scene(self):
        np, scene = self._np, self.server.scene
        bounds = []
        models = {m.name: m for m in self.problem.scene.models}
        for i, (name, fk) in enumerate(self._frame.geometry.robots):
            robot = next(r for r in self.problem.scene.robots if r.name == name)
            model = models[robot.model]
            color = COLORS[i % len(COLORS)]
            root = f"/robots/{quote(name, safe='')}"
            self._links[name] = {
                link: scene.add_frame(
                    f"{root}/links/{quote(link, safe='')}",
                    show_axes=False,
                    axes_length=0.05,
                    axes_radius=0.002,
                )
                for link in model.links
            }
            self._spheres[name] = [
                scene.add_icosphere(
                    f"{root}/collision/{j}",
                    radius=s.radius,
                    color=color,
                    subdivisions=2,
                    position=s.center,
                )
                for j, s in enumerate(fk.spheres)
            ]
            self._skeletons[name] = scene.add_line_segments(
                f"{root}/skeleton",
                points=np.zeros((len(model.joints), 2, 3)),
                colors=color,
                thickness=0.006,
            )
            tool = next(j.child for j in model.joints if j.name == model.joint_order[-1])
            points = []
            for t in np.linspace(0, self.result.makespan, 161):
                q = self.timeline_data.configurations(float(t))[name]
                frame = self.geometry.fk(name, q)
                point = frame.poses[frame.link_names.index(tool)].position
                points.append(point)
                bounds.append(point)
            scene.add_line_segments(
                f"{root}/path",
                points=np.asarray(list(zip(points[:-1], points[1:], strict=True))),
                colors=color,
                thickness=0.002,
            )
        for obj in self._frame.geometry.objects:
            root = f"/objects/{quote(obj.name, safe='')}"
            self._objects[obj.name] = [
                scene.add_icosphere(
                    f"{root}/collision/{i}",
                    radius=s.radius,
                    color=(95, 190, 130),
                    subdivisions=3,
                    position=s.center,
                )
                for i, s in enumerate(obj.spheres)
            ]
            self._labels[obj.name] = scene.add_label(
                f"{root}/label", obj.name, position=obj.pose.position
            )
        for obstacle in self.problem.scene.obstacles:
            root = f"/obstacles/{quote(obstacle.name, safe='')}"
            kwargs = dict(
                position=obstacle.pose.position, wxyz=_wxyz(obstacle.pose), color=(130, 144, 161)
            )
            if isinstance(obstacle.shape, Box):
                scene.add_box(root, dimensions=obstacle.shape.size, **kwargs)
                radius = math.hypot(*obstacle.shape.size) / 2
            else:
                scene.add_icosphere(root, radius=obstacle.shape.radius, **kwargs)
                radius = obstacle.shape.radius
            bounds.extend(
                (
                    np.asarray(obstacle.pose.position) - radius,
                    np.asarray(obstacle.pose.position) + radius,
                )
            )
            scene.add_label(
                f"{root}/name",
                obstacle.name,
                position=tuple(np.asarray(obstacle.pose.position) + (0, 0, 0.07)),
            )
        bounds.extend(s.center for o in self._frame.geometry.objects for s in o.spheres)
        low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
        center, extent = (low + high) / 2, max(1.0, float(np.max(high - low)))
        scene.add_grid(
            "/grid",
            width=extent * 2,
            height=extent * 2,
            cell_size=0.1,
            section_size=0.5,
            position=(0, 0, min(0, float(low[2]))),
        )
        self.server.initial_camera.position = tuple(center + np.array((0.8, -1.2, 0.9)) * extent)
        self.server.initial_camera.look_at = tuple(center)
        self.server.initial_camera.up = (0, 0, 1)
        self.server.initial_camera.fov = math.radians(45)

    def _build_gui(self, autoplay):
        gui = self.server.gui
        gui.add_markdown("## KinoWeave\nManipulation playback")
        self.timeline = gui.add_slider(
            "Time [s]",
            min=0.0,
            max=self.result.makespan,
            step=min(0.001, self.result.makespan / 1000),
            initial_value=0.0,
        )
        self.playing = gui.add_checkbox("Play", initial_value=autoplay)
        self.loop = gui.add_checkbox("Loop", initial_value=True)
        self.speed = gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        reset = gui.add_button("Reset")
        self.show_axes = gui.add_checkbox("Link axes", initial_value=False)
        self.show_skeleton = gui.add_checkbox("Link skeleton", initial_value=True)
        gui.add_markdown(
            "**CPU verified · sampled geometry**\n\n"
            "Collision proxies and kinematic playback. No dynamics or contact simulation."
        )
        self.inspector = gui.add_markdown("")

        @self.timeline.on_update
        def _(_event):
            with self._lock:
                if not self._closed and abs(self.timeline.value - self._time) > 1e-10:
                    self.seek(self.timeline.value)

        @self.playing.on_update
        def _(_event):
            with self._lock:
                self._last_wall = time.monotonic()

        @reset.on_click
        def _(_event):
            self.seek(0.0)

        for handle in (self.show_axes, self.show_skeleton):

            @handle.on_update
            def _(_event):
                with self._lock:
                    if not self._closed:
                        self._render()

    def _render(self):
        np, frame = self._np, self._frame
        models = {m.name: m for m in self.problem.scene.models}
        with self.server.atomic():
            for name, fk in frame.geometry.robots:
                poses = dict(zip(fk.link_names, fk.poses, strict=True))
                for link, pose in poses.items():
                    handle = self._links[name][link]
                    handle.position, handle.wxyz = pose.position, _wxyz(pose)
                    handle.show_axes = self.show_axes.value
                for handle, sphere in zip(self._spheres[name], fk.spheres, strict=True):
                    handle.position = sphere.center
                model = models[next(r.model for r in self.problem.scene.robots if r.name == name)]
                self._skeletons[name].points = np.asarray(
                    [[poses[j.parent].position, poses[j.child].position] for j in model.joints]
                )
                self._skeletons[name].visible = self.show_skeleton.value
            lines = [
                f"**Reference time: {frame.time:.3f} s**",
                "",
                "| Object | Mode | Position [m] |",
                "| --- | --- | --- |",
            ]
            for obj in frame.geometry.objects:
                state = next(s for s in frame.mode.states if s.object == obj.name)
                label = (
                    "held by " + ", ".join(g.robot for g in state.grasps)
                    if isinstance(state, HeldObject)
                    else "placed"
                )
                color = (241, 161, 55) if isinstance(state, HeldObject) else (95, 190, 130)
                for handle, sphere in zip(self._objects[obj.name], obj.spheres, strict=True):
                    handle.position, handle.color = sphere.center, color
                self._labels[obj.name].text = f"{obj.name}: {label}"
                self._labels[obj.name].position = tuple(
                    np.asarray(obj.pose.position) + (0, 0, 0.065)
                )
                position = ", ".join(f"{v:.3f}" for v in obj.pose.position)
                lines.append(f"| {obj.name} | {label} | {position} |")
            self.inspector.content = "\n".join(lines)

    def _seek(self, time_seconds, *, reset_clock):
        with self._lock:
            if self._closed:
                raise RuntimeError("viewer is closed")
            t = value_as(time_seconds, float, "time")
            frame = self._evaluate(t)
            self._frame, self._time = frame, t
            if reset_clock:
                self._last_wall = time.monotonic()
            self.timeline.value = t
            self._render()
            return frame

    def seek(self, time_seconds):
        return self._seek(time_seconds, reset_clock=True)

    def tick(self, elapsed):
        elapsed = value_as(elapsed, float, "elapsed")
        if elapsed < 0:
            raise ContractError("elapsed", "must be non-negative")
        with self._lock:
            if not self._closed and self.playing.value:
                target = self._time + elapsed * self.speed.value
                if target >= self.result.makespan and self.loop.value:
                    target %= self.result.makespan
                if target >= self.result.makespan:
                    target, self.playing.value = self.result.makespan, False
                self._seek(target, reset_clock=False)

    def run(self):
        try:
            while not self._closed:
                with self._lock:
                    now = time.monotonic()
                    elapsed = max(0.0, now - self._last_wall)
                    self._last_wall = now
                    self.tick(elapsed)
                time.sleep(1 / self.fps)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.server.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
