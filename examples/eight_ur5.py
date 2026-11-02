"""Eight UR5 arms weld authored workpiece seams with geometric coordination."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path

from kinoweave import (
    Box,
    Joint,
    LinkSphere,
    Obstacle,
    PlannerConfig,
    PlanningProblem,
    Pose,
    RobotInstance,
    RobotTrajectory,
)
from kinoweave.assets import load_visuals
from kinoweave.assets.manifest import copy_assets, write_manifest
from kinoweave.assets.urdf import load_robot_asset
from kinoweave.geometry.reference import _rotate
from kinoweave.io import document_digest, dump, load
from kinoweave.manipulation import (
    ActionOrder,
    JointTrace,
    ManipulationMode,
    ManipulationProblem,
    ManipulationResult,
    ManipulationSearchConfig,
    MultiArmPlanner,
    ScheduleSkeleton,
    Skill,
    SkillOption,
)
from kinoweave.search.reference import JointGraph
from kinoweave.visualization import ManipulationViewer

if __package__:
    from .eight_ur5_coordination import coordination_report, independent_result
    from .eight_ur5_welding import (
        WELD_SEGMENTS,
        WELD_TIMES,
        TorchIK,
        bead_fraction,
        check_surface_point,
        read_workpiece,
        weld_targets,
        workpiece_geometry,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eight_ur5_coordination import coordination_report, independent_result
    from eight_ur5_welding import (
        WELD_SEGMENTS,
        WELD_TIMES,
        TorchIK,
        bead_fraction,
        check_surface_point,
        read_workpiece,
        weld_targets,
        workpiece_geometry,
    )

PALETTE = (
    (40, 180, 240),
    (255, 164, 63),
    (105, 216, 143),
    (213, 135, 242),
    (242, 109, 133),
    (75, 215, 208),
    (230, 211, 105),
    (155, 171, 255),
)


def configuration(lift, elbow, pan=0, roll=0):
    return (pan, lift, elbow, -math.pi - lift - elbow, -math.pi / 2, roll)


def build_demo(*, backend="reference", time_limit=1200):
    """Solve workpiece-authored surface targets; never place a target using FK."""
    import trimesh

    asset = load_robot_asset(Path(__file__).parent / "assets/ur5/robot.yaml")
    workpiece = read_workpiece()
    model = replace(
        asset.model,
        revision=asset.model.revision + "-welding-torch-1",
        links=(*asset.model.links, "weld_tcp"),
        joints=(
            *asset.model.joints,
            Joint(
                name="weld_torch_mount",
                kind="fixed",
                parent="tool0",
                child="weld_tcp",
                origin=Pose(position=(0, 0, workpiece["tool_length"])),
            ),
        ),
        collision_spheres=(
            *asset.model.collision_spheres,
            *tuple(
                LinkSphere(
                    link="weld_tcp", center=(0, 0, z), radius=0.025 if z == -0.035 else 0.023
                )
                for z in (-0.035, -0.065, -0.095, -0.125, -0.155, -0.18)
            ),
        ),
        allowed_collisions=(
            *asset.model.allowed_collisions,
            *tuple(
                ("weld_tcp", link)
                for link in ("wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link", "tool0")
            ),
        ),
    )
    torch_body = trimesh.creation.cylinder(
        radius=0.012, height=workpiece["tool_length"] - 0.035, sections=24
    )
    torch_body.apply_translation((0, 0, -(workpiece["tool_length"] + 0.035) / 2))
    torch_body.visual.face_colors = (110, 119, 132, 255)
    nozzle = trimesh.creation.cone(
        radius=0.012, height=0.035 - workpiece["nozzle_gap"], sections=24
    )
    nozzle.apply_translation((0, 0, -0.035))
    nozzle.visual.face_colors = (238, 175, 67, 255)
    visuals = {(model.name, link): data for link, data in asset.visuals.items()}
    visuals[model.name, "weld_tcp"] = trimesh.Scene([torch_body, nozzle]).export(file_type="glb")
    home = tuple(v + (1.0 if i == 3 else 0) for i, v in enumerate(configuration(-1.9, 2.1)))
    robots = []
    for level in range(2):
        for i in range(4):
            angle = i * math.pi / 2 + (math.pi / 4 if level == 0 else 0)
            yaw = angle + math.pi
            orientation = (
                (math.cos(yaw / 2), math.sin(yaw / 2), 0, 0)
                if level
                else (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            )
            robots.append(
                RobotInstance(
                    name=f"arm{len(robots) + 1:02d}",
                    model=model.name,
                    start=home,
                    goal=home,
                    base_pose=Pose(
                        position=(
                            0.94 * math.cos(angle),
                            0.94 * math.sin(angle),
                            1.45 if level else 0,
                        ),
                        quaternion=orientation,
                    ),
                )
            )
    obstacles = []

    def box(name, position, size):
        obstacles.append(Obstacle(name=name, pose=Pose(position=position), shape=Box(size=size)))

    box("floor", (0, 0, -0.15), (3.2, 3.2, 0.06))
    work_obstacles, _ = workpiece_geometry(workpiece)
    obstacles.extend(work_obstacles)
    # A visible open gantry, with clearance from the conservative base proxies.
    for side in (-1, 1):
        box(f"gantry_x_{side}", (0, side * 1.34, 1.68), (2.75, 0.065, 0.065))
        box(f"gantry_y_{side}", (side * 1.34, 0, 1.68), (0.065, 2.75, 0.065))
        box(f"gantry_post_{side}", (side * 1.34, 1.34, 0.775), (0.075, 0.075, 1.75))
    skills, edges, targets, graphs = [], [], [], []
    authored_targets = weld_targets(workpiece, robots)
    for index, robot in enumerate(robots):
        previous = None
        ik = TorchIK(model, robot)
        for target in authored_targets[index * 5 : (index + 1) * 5]:
            normal = target["normal"]
            visit = int(target["skill"][-2:]) - 1
            stage_q = tuple(
                v + (1.0 if i == 3 else 0)
                for i, v in enumerate(
                    configuration(-1.9, 2.1, (0.9 if visit % 2 == 0 else -0.9) if index < 4 else 0)
                )
            )
            orientation = target["quaternion_xyzw"]
            solved, seed = [], configuration(-1.5, 1.2)
            points = [
                tuple(
                    a + b * workpiece["approach_distance"]
                    for a, b in zip(target["start_position"], normal, strict=True)
                )
            ]
            points.extend(
                tuple(
                    a + (b - a) * step / WELD_SEGMENTS
                    for a, b in zip(target["start_position"], target["end_position"], strict=True)
                )
                for step in range(WELD_SEGMENTS + 1)
            )
            points.append(
                tuple(
                    a + b * workpiece["approach_distance"]
                    for a, b in zip(target["end_position"], normal, strict=True)
                )
            )
            for point in points:
                seed = ik.solve(Pose(position=point, quaternion=orientation), seed)
                solved.append(seed)
            times = (
                *WELD_TIMES[:3],
                *(WELD_TIMES[3] + i / WELD_SEGMENTS for i in range(WELD_SEGMENTS + 1)),
                *WELD_TIMES[5:],
            )
            positions = (home, stage_q, *solved, stage_q, home)
            name = target["skill"]
            skills.append(
                Skill(
                    name=name,
                    kind="weld",
                    robots=(robot.name,),
                    minimum_duration=times[-1],
                    options=(
                        SkillOption(
                            name="configured",
                            trace=JointTrace(
                                times=times,
                                interpolation="quintic_stop",
                                tracks=(
                                    RobotTrajectory(
                                        robot=robot.name,
                                        joint_names=model.joint_order,
                                        positions=positions,
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )
            if previous:
                edges.append(ActionOrder(before=previous, after=name))
            previous = name
            targets.append(target)
        graphs.append(
            JointGraph(
                robot=robot.name,
                joint_names=model.joint_order,
                vertices=(home,),
                edges=(),
                start=0,
                goal=0,
            )
        )
    scene = PlanningProblem(
        name="eight-ur5-shared-frame",
        models=(model,),
        robots=tuple(robots),
        obstacles=tuple(obstacles),
        config=PlannerConfig(
            backend=backend,
            device="cuda:0" if backend == "curobo" else "cpu",
            seed=7,
            time_step=7.5,
            validation_time_step=0.05,
            collision_margin=0.005,
            time_limit=time_limit,
        ),
    )
    problem = ManipulationProblem(
        scene=scene,
        objects=(),
        initial_mode=ManipulationMode(name="initial", states=()),
        goal_mode=ManipulationMode(name="goal", states=()),
        skeleton=ScheduleSkeleton(skills=tuple(skills), precedence=tuple(edges)),
    )
    metadata = {
        "format": "kinoweave.demo.eight-ur5",
        "schema_version": 2,
        "problem_digest": document_digest(problem),
        "tcp_link": "weld_tcp",
        "workpiece": workpiece,
        "targets": targets,
        "description": "Eight UR5 arms weld forty short beads on authored plate seams.",
        "scenario": "workpiece-welding-v1",
    }
    return problem, tuple(graphs), visuals, asset, metadata


def check_targets(problem, result, metadata):
    """Verify authored plate faces and the whole on-surface welding interval."""
    from kinoweave.manipulation import ManipulationTimeline, ModeGeometry, pose_errors

    if metadata.get("format") != "kinoweave.demo.eight-ur5" or metadata.get("schema_version") != 2:
        raise ValueError("unknown welding demo metadata format/version")
    if metadata["problem_digest"] != document_digest(problem):
        raise ValueError("welding metadata does not match the input")
    spec = metadata["workpiece"]
    authored_obstacles, _ = workpiece_geometry(spec)
    actual_obstacles = {
        o.name: o for o in problem.scene.obstacles if o.name.startswith("workpiece_")
    }
    if actual_obstacles != {o.name: o for o in authored_obstacles}:
        raise ValueError("workpiece definition does not match the checked collision geometry")
    expected_targets = json.loads(json.dumps(weld_targets(spec, problem.scene.robots)))
    if json.loads(json.dumps(metadata["targets"])) != expected_targets:
        raise ValueError("weld targets must match the authored workpiece seams")
    if len(problem.scene.robots) != 8 or len(expected_targets) != 40:
        raise ValueError("each of eight arms must have five welds")
    scheduled = {s.skill: s for s in result.plan.skills}
    definitions = {s.name: s for s in problem.skeleton.skills}
    if {t["skill"] for t in expected_targets} != definitions.keys():
        raise ValueError("targets must cover every welding skill exactly once")
    timeline, geometry = ManipulationTimeline(problem, result), ModeGeometry(problem)
    max_position = max_angle = max_surface = max_seam = 0.0
    min_gap, max_gap = math.inf, -math.inf
    for target in expected_targets:
        execution, definition = scheduled[target["skill"]], definitions[target["skill"]]
        if definition.kind != "weld" or definition.robots != (target["robot"],):
            raise ValueError("invalid welding skill binding")
        for point in (target["position"], target["start_position"], target["end_position"]):
            check_surface_point(point, target["normal"], target["parts"], actual_obstacles)
        begin, end = target["weld_source"]
        source = definition.options[0].trace.duration
        if not 0 <= begin < end <= source:
            raise ValueError("invalid welding interval")
        count = math.ceil((end - begin) / 0.025)
        for i in range(count + 1):
            fraction = i / count
            u = bead_fraction(fraction)
            position = tuple(
                a + (b - a) * u
                for a, b in zip(target["start_position"], target["end_position"], strict=True)
            )
            time = begin + fraction * (end - begin)
            absolute = execution.start + time / source * (execution.end - execution.start)
            fk = geometry.fk(target["robot"], timeline.configurations(absolute)[target["robot"]])
            tcp = fk.poses[fk.link_names.index(metadata["tcp_link"])]
            error, angle = pose_errors(
                tcp, Pose(position=position, quaternion=target["quaternion_xyzw"])
            )
            max_position, max_angle = max(max_position, error), max(max_angle, angle)
            normal = target["normal"]
            surface = sum(
                (a - b) * n
                for a, b, n in zip(tcp.position, target["position"], normal, strict=True)
            )
            tangent = (-normal[1], normal[0], 0)
            seam = sum(
                (a - b) * t
                for a, b, t in zip(tcp.position, target["position"], tangent, strict=True)
            )
            max_surface, max_seam = max(max_surface, abs(surface)), max(max_seam, abs(seam))
            check_surface_point(
                tcp.position, normal, target["parts"], actual_obstacles, tolerance=5e-5
            )
            nozzle = _rotate(tcp.quaternion, (0, 0, -spec["nozzle_gap"]))
            gap = surface + sum(v * n for v, n in zip(nozzle, normal, strict=True))
            min_gap, max_gap = min(min_gap, gap), max(max_gap, gap)
    if max_position > 5e-5 or max_angle > 1e-4:
        raise ValueError("welding TCP does not follow the authored seam pose")
    if not spec["nozzle_gap"] - 5e-5 <= min_gap <= max_gap <= spec["nozzle_gap"] + 5e-5:
        raise ValueError("welding nozzle does not maintain its declared surface gap")
    return {
        "targets": 40,
        "robots": 8,
        "seams": 8,
        "max_position_error_m": max_position,
        "max_orientation_error_rad": max_angle,
        "max_surface_error_m": max_surface,
        "max_seam_error_m": max_seam,
        "min_nozzle_gap_m": min_gap,
        "max_nozzle_gap_m": max_gap,
        "checked": "finite adjoining plate faces and weld interval at 0.025 s",
    }


def plan_demo(output, *, backend="reference", time_limit=1200, replan=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if replan or backend != "reference" or time_limit != 1200:
        problem, graphs, visuals, asset, metadata = build_demo(
            backend=backend, time_limit=time_limit
        )
        write_manifest(output, problem.scene, visuals, asset.sources, asset.notices)
        print("Planning eight UR5 arms and forty workpiece welds...", file=sys.stderr, flush=True)
        planner = MultiArmPlanner(
            problem,
            search_config=ManipulationSearchConfig(
                max_states=4000,
                max_constraints=40000,
                max_validation_samples=100000,
            ),
        )
        result = planner.plan(graphs)
        statistics = planner.statistics
        source = "new_plan"
    else:
        from kinoweave.manipulation import verify_manipulation

        print("Checking the bundled eight-arm plan on CPU...", file=sys.stderr, flush=True)
        data = Path(__file__).parent / "data"
        problem = load(data / "eight_ur5.problem.json", expected=ManipulationProblem)
        metadata = json.loads((data / "eight_ur5.demo.json").read_text())
        copy_assets(
            Path(__file__).parent / "assets/eight_ur5/assets.manifest.json",
            problem.scene,
            output,
            problem.scene,
        )
        result = load(
            Path(__file__).parent / "data/eight_ur5.result.json", expected=ManipulationResult
        )
        result.check_problem(problem)
        result = verify_manipulation(problem, result)
        statistics = {}
        source = "bundled_plan_reverified_on_cpu"
    dump(problem, output / "problem.json")
    (output / "demo.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dump(result, output / "result.json")
    report = {
        "status": result.status,
        "termination_reason": result.termination_reason,
        "makespan": result.makespan,
        "statistics": statistics,
        "source": source,
    }
    if result.status == "success":
        report["welding"] = check_targets(problem, result, metadata)
        report["coordination"] = coordination_report(problem, result, metadata)
        dump(independent_result(problem, result), output / "uncoordinated.candidate.json")
        (output / "coordination.json").write_text(
            json.dumps(report["coordination"], indent=2) + "\n"
        )
    (output / "planning.json").write_text(json.dumps(report, indent=2) + "\n")
    return problem, result, metadata


class WeldingViewer(ManipulationViewer):
    """Shared-frame presentation; progress follows the verified skill clock."""

    def __init__(self, problem, result, metadata, **options):
        from kinoweave.manipulation import ManipulationTimeline

        self.metadata = metadata
        self._targets = {}
        self._demo_ready = False
        self._unsafe_comparison = False
        self.baseline = independent_result(problem, result)
        self._baseline_timeline = ManipulationTimeline(problem, self.baseline)
        self.coordination = coordination_report(problem, result, metadata)
        self.weld_proof = check_targets(problem, result, metadata)
        super().__init__(problem, result, **options)
        self._grid.visible = False
        self.server.gui.configure_theme(
            dark_mode=True, brand_color=(43, 177, 216), show_share_button=False
        )
        scene = self.server.scene
        self._heading.visible = False
        scene.configure_default_lights(enabled=False)
        scene.configure_environment_map("studio", background=False, environment_intensity=0.7)
        scene.add_light_ambient("/demo/ambient", color=(255, 255, 255), intensity=0.25)
        scene.add_light_directional(
            "/demo/key", color=(235, 245, 255), intensity=0.9, position=(2, -2, 4)
        )
        for name, node in self._obstacle_nodes.items():
            node.color = (
                (35, 47, 62)
                if name == "floor"
                else (68, 85, 106)
                if name.startswith("gantry")
                else (118, 145, 163)
            )
        self._robot_badges = {}
        for robot in problem.scene.robots:
            self._robot_badges[robot.name] = scene.add_label(
                f"/demo/robots/{robot.name}",
                robot.name.upper(),
                position=tuple(
                    a + b for a, b in zip(robot.base_pose.position, (0, 0, 0.12), strict=True)
                ),
            )
        _, seams = workpiece_geometry(metadata["workpiece"])
        for seam in seams:
            scene.add_line_segments(
                f"/demo/seams/{seam['name']}",
                points=self._np.asarray(
                    [
                        [
                            tuple(
                                v + n * 0.0005 for v, n in zip(point, seam["normal"], strict=True)
                            )
                            for point in (seam["start"], seam["end"])
                        ]
                    ]
                ),
                colors=(38, 59, 72),
                thickness=0.002,
            )
        self._beads = {}
        self._arcs = {}
        for robot in problem.scene.robots:
            arc = scene.add_line_segments(
                f"/demo/arcs/{robot.name}",
                points=self._np.zeros((1, 2, 3)),
                colors=(255, 240, 170),
                thickness=0.003,
                visible=False,
            )
            sparks = scene.add_line_segments(
                f"/demo/sparks/{robot.name}",
                points=self._np.zeros((5, 2, 3)),
                colors=(255, 192, 72),
                thickness=0.001,
                visible=False,
            )
            self._arcs[robot.name] = arc, sparks
        for i, target in enumerate(metadata["targets"]):
            self._targets[target["id"]] = scene.add_icosphere(
                f"/demo/sites/{target['id']}",
                radius=0.006,
                subdivisions=2,
                position=tuple(target["position"]),
                color=PALETTE[i // 5],
            )
            self._beads[target["id"]] = scene.add_line_segments(
                f"/demo/beads/{target['id']}",
                points=self._np.asarray([[target["start_position"], target["start_position"]]]),
                colors=(99, 230, 159),
                thickness=0.004,
                visible=False,
            )
        self.server.initial_camera.position = (2.8, -3.4, 2.25)
        self.server.initial_camera.look_at = (0, 0, 0.72)
        self.server.initial_camera.fov = math.radians(42)
        self.progress = self.server.gui.add_markdown("", order=-1)
        self.comparison = self.server.gui.add_dropdown(
            "Execution",
            options=("Coordinated", "Uncoordinated (unsafe)"),
            initial_value="Coordinated",
            order=-0.8,
        )
        collision_button = self.server.gui.add_button("Show baseline collision", order=-0.7)
        self.camera = self.server.gui.add_dropdown(
            "Camera",
            options=("Overview", "Front", "Overhead", "Workpiece"),
            initial_value="Overview",
            order=-0.5,
        )

        @self.camera.on_update
        def _(_event):
            positions = {
                "Overview": (2.8, -3.4, 2.25),
                "Front": (0, -4.5, 1.2),
                "Overhead": (0, -0.01, 4.8),
                "Workpiece": (1.15, -1.45, 1.1),
            }
            for client in self.server.get_clients().values():
                client.camera.position = positions[self.camera.value]
                client.camera.look_at = (0, 0, 0.72)
                client.camera.up = (0, 0, 1) if self.camera.value != "Overhead" else (0, 1, 0)

        @self.comparison.on_update
        def _(_event):
            unsafe = self.comparison.value == "Uncoordinated (unsafe)"
            if unsafe != self._unsafe_comparison:
                self.set_comparison(unsafe)

        @collision_button.on_click
        def _(_event):
            self.set_comparison(True)
            self.seek(self.coordination["witnesses"][0]["time"])

        from itertools import combinations

        self._contact_markers = {}
        for first, second in combinations((r.name for r in problem.scene.robots), 2):
            root = f"/demo/conflicts/{first}/{second}"
            marker = scene.add_icosphere(
                root, radius=0.035, color=(255, 66, 85), opacity=0.8, visible=False
            )
            label = scene.add_label(root + "/label", "", visible=False)
            self._contact_markers[first, second] = marker, label
        self._demo_ready = True
        self._render()

    def set_comparison(self, unsafe):
        """Show the explicit invalid counterfactual, without changing the verified result."""
        with self._lock:
            self._unsafe_comparison = bool(unsafe)
            self.comparison.value = "Uncoordinated (unsafe)" if unsafe else "Coordinated"
            self.playing.value = False
            self.seek(0)

    def _evaluate(self, t):
        if not self._unsafe_comparison:
            return super()._evaluate(t)
        from kinoweave.visualization.manipulation import ManipulationPlaybackFrame

        source = min(t, self.baseline.makespan)
        mode = self._baseline_timeline.mode(source)
        geometry = self.geometry.snapshot(mode, self._baseline_timeline.configurations(source))
        return ManipulationPlaybackFrame(t, mode, geometry)

    @property
    def display_result(self):
        return self.baseline if self._unsafe_comparison else self.result

    def _render(self):
        super()._render()
        if not self._demo_ready:
            return
        schedule = {s.skill: s for s in self.display_result.plan.skills}
        definitions = {s.name: s for s in self.problem.skeleton.skills}
        done = active = 0
        rows, waiting, moving = [], 0, 0
        for arc, sparks in self._arcs.values():
            arc.visible = sparks.visible = False
        frames = dict(self.frame.geometry.robots)
        for i, target in enumerate(self.metadata["targets"]):
            execution = schedule[target["skill"]]
            source = definitions[target["skill"]].options[0].trace.duration
            begin, end = (
                execution.start + t / source * (execution.end - execution.start)
                for t in target["weld_source"]
            )
            complete = self.frame.time >= end
            working = begin <= self.frame.time < end
            done += complete
            active += working
            marker = self._targets[target["id"]]
            marker.color = (
                (88, 225, 145) if complete else (255, 222, 110) if working else PALETTE[i // 5]
            )
            marker.opacity = 1 if complete or working else 0.5
            bead = self._beads[target["id"]]
            bead.visible = complete or working
            if bead.visible:
                fraction = (
                    1 if complete else bead_fraction((self.frame.time - begin) / (end - begin))
                )
                finish = tuple(
                    a + (b - a) * fraction
                    for a, b in zip(target["start_position"], target["end_position"], strict=True)
                )
                offset = self._np.asarray(target["normal"]) * 0.0005
                bead.points = self._np.asarray([[target["start_position"], finish]]) + offset
            if working:
                fk = frames[target["robot"]]
                tcp = fk.poses[fk.link_names.index(self.metadata["tcp_link"])]
                point = self._np.asarray(tcp.position)
                nozzle = point + self._np.asarray(
                    _rotate(tcp.quaternion, (0, 0, -self.metadata["workpiece"]["nozzle_gap"]))
                )
                arc, sparks = self._arcs[target["robot"]]
                arc.points = self._np.asarray([[point, nozzle]])
                normal = self._np.asarray(target["normal"])
                tangent = self._np.asarray((-normal[1], normal[0], 0))
                rays = []
                for k in range(5):
                    phase = k * 2 * math.pi / 5 + self.frame.time * 18
                    finish = (
                        point
                        + normal * 0.004
                        + tangent * (0.018 * math.cos(phase))
                        + self._np.asarray((0, 0, 0.018 * math.sin(phase)))
                    )
                    rays.append((point + normal * 0.001, finish))
                sparks.points = self._np.asarray(rays)
                arc.visible = sparks.visible = True
        for robot in self.problem.scene.robots:
            count = sum(
                self.frame.time
                >= schedule[t["skill"]].start
                + t["weld_source"][1]
                / definitions[t["skill"]].options[0].trace.duration
                * (schedule[t["skill"]].end - schedule[t["skill"]].start)
                for t in self.metadata["targets"]
                if t["robot"] == robot.name
            )
            executions = [
                schedule[s.name] for s in self.problem.skeleton.skills if s.robots == (robot.name,)
            ]
            executing = any(s.start <= self.frame.time < s.end for s in executions)
            pending = any(self.frame.time < s.start for s in executions)
            welding = any(
                schedule[t["skill"]].start + t["weld_source"][0]
                <= self.frame.time
                < schedule[t["skill"]].start + t["weld_source"][1]
                for t in self.metadata["targets"]
                if t["robot"] == robot.name
            )
            state = (
                "Welding"
                if welding
                else "Moving"
                if executing
                else "Waiting"
                if pending
                else "Done"
            )
            waiting += pending and not executing
            moving += executing and not welding
            rows.append(f"{robot.name.upper()} {count}/5 · {state}")
            self._robot_badges[robot.name].text = f"{robot.name.upper()} · {state}"
        self.waiting_count, self.moving_count = waiting, moving
        colliding = 0
        for (first, second), (marker, label) in self._contact_markers.items():
            marker.visible = label.visible = False
            if not self._unsafe_comparison:
                continue
            a, b = frames[first].spheres, frames[second].spheres
            delta = (
                self._np.asarray([s.center for s in a])[:, None, :]
                - self._np.asarray([s.center for s in b])[None, :, :]
            )
            gaps = (
                self._np.linalg.norm(delta, axis=-1)
                - self._np.asarray([s.radius for s in a])[:, None]
                - self._np.asarray([s.radius for s in b])[None, :]
            )
            index = self._np.unravel_index(gaps.argmin(), gaps.shape)
            clearance = float(gaps[index])
            if clearance > self.geometry.threshold:
                continue
            colliding += 1
            marker.position = tuple(
                (x + y) / 2 for x, y in zip(a[index[0]].center, b[index[1]].center, strict=True)
            )
            label.position = (0, 0, 0.06)
            label.text = f"{first.upper()} × {second.upper()}: {clearance * 1000:.0f} mm"
            marker.visible = label.visible = True
        self.colliding_pairs = colliding
        self._verification_note.visible = not self._unsafe_comparison
        status = (
            f"**UNSAFE comparison · {colliding} colliding robot pairs**"
            if self._unsafe_comparison
            else "**Coordinated · CPU verified**"
        )
        self.progress.content = (
            f"## Eight UR5 arms\n{status}\n\n**{done} / 40 sites complete** · "
            f"{moving} moving · {active} welding · **{waiting} waiting**\n\n"
            + "  \n".join(rows)
            + f"\n\n{len(self.coordination['conflicting_pairs'])} baseline conflict pairs "
            + "connect all 8 arms."
            + "\n\n8 workpiece seams · 12 mm nozzle gap · "
            + f"surface error ≤ {self.weld_proof['max_surface_error_m'] * 1e6:.1f} µm"
            + "\n\nPending: arm color · Welding: yellow · Complete: green"
            + f"\n\nShared time: {self.frame.time:.3f} s"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--backend", choices=("reference", "curobo"), default="reference")
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument(
        "--plan", action="store_true", help="Compute a new plan instead of using the bundled demo"
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--video", type=Path, help="Render an MP4 with Chrome and FFmpeg")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-speed", type=float, default=1)
    parser.add_argument("--video-renderer", choices=("auto", "software"), default="auto")
    parser.add_argument("--video-planned-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.replay:
        directory = args.replay
        problem = load(directory / "problem.json", expected=ManipulationProblem)
        result = load(directory / "result.json", expected=ManipulationResult)
        metadata = json.loads((directory / "demo.json").read_text())
    else:
        directory = args.output
        with redirect_stdout(sys.stderr):
            problem, result, metadata = plan_demo(
                directory, backend=args.backend, time_limit=args.seconds, replan=args.plan
            )
    if result.status != "success":
        print(json.dumps(asdict(result), indent=2), flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "makespan": result.makespan,
                "welding": check_targets(problem, result, metadata),
                "output": str(directory),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.view or args.video:
        with WeldingViewer(
            problem,
            result,
            metadata,
            visuals=load_visuals(directory / "assets.manifest.json", problem.scene),
            host=args.host,
            port=args.port,
            autoplay=args.view and not args.video,
        ) as viewer:
            print(f"Open {viewer.url}", flush=True)
            if args.video:
                if __package__:
                    from .record_eight_ur5 import record
                else:
                    from record_eight_ur5 import record
                record(
                    viewer,
                    args.video,
                    fps=args.video_fps,
                    speed=args.video_speed,
                    renderer=args.video_renderer,
                    compare=not args.video_planned_only,
                )
            if args.view:
                viewer.seek(0)
                viewer.playing.value = True
                viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
