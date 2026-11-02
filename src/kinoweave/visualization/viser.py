"""Browser playback driven exclusively by the native memory player's frames."""

from __future__ import annotations

import math
import threading
import time
from itertools import product
from urllib.parse import quote

from .._contracts import ContractError, value_as
from ..adapters import MemoryPlayback
from ..geometry.reference import _rotate, forward
from ..models import Box, PlanningProblem
from ..results import PlanResult

COLORS = ((42, 136, 230), (237, 134, 45), (85, 180, 125), (188, 107, 220))


def _wxyz(pose):
    x, y, z, w = pose.quaternion
    return w, x, y, z


class ViserViewer:
    """One shared timeline for all connected clients, without physics or retiming.

    ``visuals`` maps (model name, link name) to static GLB bytes expressed in
    that link's local frame. The native world link pose is applied exactly once.
    """

    def __init__(
        self,
        problem: PlanningProblem,
        plan: PlanResult,
        *,
        visuals: dict[tuple[str, str], bytes] | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        fps: float = 30,
        autoplay: bool = False,
        max_validation_samples: int = 100_000,
    ):
        self.fps = value_as(fps, float, "fps")
        port = value_as(port, int, "port")
        if not 1 <= self.fps <= 120 or not 0 <= port <= 65535:
            raise ContractError("viewer", "fps must be in [1,120], port in [0,65535]")
        try:
            import numpy as np
            import viser
        except ImportError as error:
            raise ContractError(
                "viewer.dependencies", "install kinoweave[viz] to use the viewer"
            ) from error
        self._np = np
        self.problem = problem
        self.plan = plan
        self.player = MemoryPlayback(problem, max_validation_samples=max_validation_samples)
        self.player.load(problem, plan)
        self._lock = threading.RLock()
        self._closed = False
        self._time = 0.0
        self._last_wall = time.monotonic()
        self._models = {m.name: m for m in problem.models}
        self._links, self._spheres, self._skeletons, self._ghosts, self._meshes = {}, {}, {}, [], []
        self._visuals = visuals or {}
        self._host = host
        valid_links = {(m.name, link) for m in problem.models for link in m.links}
        if not set(self._visuals) <= valid_links:
            self.player.close()
            raise ContractError("visuals", "unknown model/link binding")
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
            self._render(self.player.frame)
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
            return self.player.frame

    def _build_scene(self):
        np, scene = self._np, self.server.scene
        bounds = []
        for i, robot in enumerate(self.problem.robots):
            color = COLORS[i % len(COLORS)]
            model = self._models[robot.model]
            root = f"/robots/{quote(robot.name, safe='')}"
            scene.add_label(
                f"{root}/name",
                robot.name,
                position=tuple(
                    a + b for a, b in zip(robot.base_pose.position, (0, 0, 0.12), strict=True)
                ),
            )
            self._links[robot.name] = {
                link: scene.add_frame(
                    f"{root}/links/{quote(link, safe='')}",
                    show_axes=False,
                    axes_length=0.06,
                    axes_radius=0.002,
                )
                for link in model.links
            }
            for link in model.links:
                glb = self._visuals.get((model.name, link))
                if glb is not None:
                    self._meshes.append(
                        scene.add_glb(f"{root}/links/{quote(link, safe='')}/visual", glb)
                    )
            current = next(state for state in self.player.frame.robots if state.robot == robot.name)
            self._spheres[robot.name] = [
                scene.add_icosphere(
                    f"{root}/collision/{j}",
                    radius=s.radius,
                    color=color,
                    subdivisions=1,
                    opacity=0.65,
                    position=s.center,
                )
                for j, s in enumerate(current.spheres)
            ]
            self._skeletons[robot.name] = scene.add_line_segments(
                f"{root}/skeleton",
                points=np.zeros((len(model.joints), 2, 3)),
                colors=color,
                thickness=0.008,
            )
            for label, positions in (("start", robot.start), ("goal", robot.goal)):
                fk = forward(model, robot, positions)
                for j, sphere in enumerate(fk.spheres):
                    bounds.extend(
                        (
                            np.asarray(sphere.center) - sphere.radius,
                            np.asarray(sphere.center) + sphere.radius,
                        )
                    )
                    if label == "goal":
                        self._ghosts.append(
                            scene.add_icosphere(
                                f"{root}/goal/{j}",
                                radius=sphere.radius,
                                color=color,
                                subdivisions=1,
                                opacity=0.15,
                                wireframe=True,
                                position=sphere.center,
                            )
                        )
                # The tool link follows the last independent joint's child.
                tool = next(j.child for j in model.joints if j.name == model.joint_order[-1])
                if label == "start":
                    times = np.linspace(
                        0, self.plan.makespan, min(121, max(2, math.ceil(self.plan.makespan * 20)))
                    )
                    points = []
                    for t in times:
                        state = self.plan.trajectory.evaluate(float(t))[robot.name]
                        poses = forward(model, robot, state.positions)
                        points.append(poses.poses[poses.link_names.index(tool)].position)
                    scene.add_line_segments(
                        f"{root}/trajectory",
                        points=np.array(list(zip(points[:-1], points[1:], strict=True))),
                        colors=color,
                        thickness=0.003,
                    )
        for obstacle in self.problem.obstacles:
            name = f"/obstacles/{quote(obstacle.name, safe='')}"
            kwargs = dict(
                position=obstacle.pose.position, wxyz=_wxyz(obstacle.pose), color=(125, 137, 152)
            )
            if isinstance(obstacle.shape, Box):
                scene.add_box(name, dimensions=obstacle.shape.size, **kwargs)
                bounds.extend(
                    self._np.asarray(obstacle.pose.position)
                    + self._np.asarray(_rotate(obstacle.pose.quaternion, corner))
                    for corner in product(*(tuple((-v / 2, v / 2)) for v in obstacle.shape.size))
                )
            else:
                scene.add_icosphere(name, radius=obstacle.shape.radius, **kwargs)
                radius = obstacle.shape.radius
                bounds.extend(
                    (
                        np.asarray(obstacle.pose.position) - radius,
                        np.asarray(obstacle.pose.position) + radius,
                    )
                )
        low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
        center = (low + high) / 2
        extent = max(1.0, float(np.max(high - low)))
        scene.add_grid(
            "/grid",
            width=extent * 2,
            height=extent * 2,
            position=(float(center[0]), float(center[1]), float(min(0, low[2]))),
            cell_size=0.25,
            section_size=1,
        )
        self.server.initial_camera.position = tuple(center + np.array((1.0, -1.2, 0.9)) * extent)
        self.server.initial_camera.look_at = tuple(center)
        self.server.initial_camera.up = (0, 0, 1)
        self.server.initial_camera.fov = math.radians(45)

    def _build_gui(self, autoplay):
        gui = self.server.gui
        gui.add_markdown("## KinoWeave\nShared-time robot playback")
        self.timeline = gui.add_slider(
            "Time [s]",
            min=0.0,
            max=self.plan.makespan,
            step=min(0.001, self.plan.makespan / 1000),
            initial_value=0.0,
        )
        self.playing = gui.add_checkbox("Play", initial_value=autoplay)
        self.loop = gui.add_checkbox("Loop", initial_value=True)
        self.speed = gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        reset = gui.add_button("Reset")
        self.show_meshes = gui.add_checkbox(
            "Robot meshes", initial_value=True, disabled=not self._meshes
        )
        missing_meshes = any(
            not any((r.model, link) in self._visuals for link in self._models[r.model].links)
            for r in self.problem.robots
        )
        self.show_collision = gui.add_checkbox("Collision spheres", initial_value=missing_meshes)
        self.show_skeleton = gui.add_checkbox("Link skeleton", initial_value=missing_meshes)
        self.show_axes = gui.add_checkbox("Link axes", initial_value=False)
        self.show_goals = gui.add_checkbox("Goal outlines", initial_value=not self._meshes)
        gui.add_markdown(
            f"Collision check: **{self.plan.validation.level}** · Kinematic playback\n\n"
            "No dynamics or contact simulation."
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

        for handle in (
            self.show_meshes,
            self.show_collision,
            self.show_skeleton,
            self.show_axes,
            self.show_goals,
        ):

            @handle.on_update
            def _(_event):
                with self._lock:
                    if not self._closed:
                        self._render(self.player.frame)

    def _render(self, frame):
        with self.server.atomic():
            for robot in frame.robots:
                poses = dict(zip(robot.link_names, robot.link_poses, strict=True))
                for link, pose in poses.items():
                    handle = self._links[robot.robot][link]
                    handle.position, handle.wxyz = pose.position, _wxyz(pose)
                    handle.show_axes = self.show_axes.value
                for handle, sphere in zip(self._spheres[robot.robot], robot.spheres, strict=True):
                    handle.position, handle.visible = sphere.center, self.show_collision.value
                model = self._models[
                    next(r.model for r in self.problem.robots if r.name == robot.robot)
                ]
                self._skeletons[robot.robot].points = self._np.asarray(
                    [[poses[j.parent].position, poses[j.child].position] for j in model.joints]
                )
                self._skeletons[robot.robot].visible = self.show_skeleton.value
            for handle in self._meshes:
                handle.visible = self.show_meshes.value
            for handle in self._ghosts:
                handle.visible = self.show_goals.value
            lines = [f"**Reference time: {frame.observation.reference_time:.3f} s**"]
            for state in frame.observation.robots:
                lines.extend(
                    (f"\n**{state.robot}**", "\n| Joint | q | dq |", "| --- | ---: | ---: |")
                )
                lines.extend(
                    f"| {name} | {q:.3f} | {dq:.3f} |"
                    for name, q, dq in zip(
                        state.joint_names, state.positions, state.velocities, strict=True
                    )
                )
            self.inspector.content = "\n".join(lines)

    def _seek(self, time_seconds, reset_clock):
        with self._lock:
            if self._closed:
                raise RuntimeError("viewer is closed")
            observation = self.player.set_time(time_seconds)
            self._time = observation.reference_time
            if reset_clock:
                self._last_wall = time.monotonic()
            self.timeline.value = self._time
            self._render(self.player.frame)
            return observation

    def seek(self, time_seconds):
        return self._seek(time_seconds, reset_clock=True)

    def tick(self, elapsed):
        """Advance the display clock; the source plan/time law stays unchanged."""
        elapsed = value_as(elapsed, float, "elapsed")
        if elapsed < 0:
            raise ContractError("elapsed", "must be non-negative")
        with self._lock:
            if not self._closed and self.playing.value:
                target = self._time + elapsed * self.speed.value
                if target >= self.plan.makespan and self.loop.value:
                    target %= self.plan.makespan
                if target >= self.plan.makespan:
                    target = self.plan.makespan
                    self.playing.value = False
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
            self.player.close()
        self.server.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
