"""Independent CPU geometry for placed and link-attached object spheres."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations

from .._contracts import Contract, ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..geometry.reference import ReferenceFK, _compose, _rotate, forward, obstacle_clearance
from ..models import BackendInfo, Box, Diagnostic, Pose, Vector3
from .models import ContactBody, HeldObject, ManipulationMode, ManipulationProblem, PlacedObject


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryPolicy(Contract):
    position_tolerance: float = 1e-6
    orientation_tolerance: float = 1e-6
    contact_tolerance: float = 1e-6
    support_tolerance: float = 1e-6
    max_cached_fk: int = 2048

    def _validate(self):
        for name in (
            "position_tolerance",
            "orientation_tolerance",
            "contact_tolerance",
            "support_tolerance",
        ):
            self._require(getattr(self, name) >= 0, name, "must be non-negative")
        self._require(self.max_cached_fk >= 0, "max_cached_fk", "must be non-negative")


@dataclass(frozen=True, slots=True)
class GeometrySphere:
    body: ContactBody
    index: int
    center: Vector3
    radius: float

    @property
    def label(self):
        return f"{self.body.kind}/{self.body.name}/{self.body.link or ''}/sphere[{self.index}]"


@dataclass(frozen=True, slots=True)
class ObjectFrame:
    name: str
    pose: Pose
    spheres: tuple[GeometrySphere, ...]


@dataclass(frozen=True, slots=True)
class ManipulationFrame:
    robots: tuple[tuple[str, ReferenceFK], ...]
    objects: tuple[ObjectFrame, ...]
    issues: tuple[Diagnostic, ...]

    @property
    def valid(self):
        return not self.issues


def pose_errors(first: Pose, second: Pose):
    a, b = first.quaternion, second.quaternion
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True))) / (math.hypot(*a) * math.hypot(*b))
    return math.dist(first.position, second.position), 2 * math.acos(min(1.0, dot))


def pose_matches(first, second, policy):
    position, angle = pose_errors(first, second)
    return position <= policy.position_tolerance and angle <= policy.orientation_tolerance


class ModeGeometry:
    """Sampled geometric queries; local masks deliberately omit other robots.

    Local queries retain placed objects and objects held only by this robot.
    A dual-held object needs a joint configuration and is not a local PRM body.
    FK cache entries depend on the immutable scene and q, never on object mode.
    """

    def __init__(self, problem: ManipulationProblem, *, policy: GeometryPolicy | None = None):
        self._problem = value_as(problem, ManipulationProblem, "problem")
        self.policy = value_as(policy or GeometryPolicy(), GeometryPolicy, "policy")
        self.models = {m.name: m for m in problem.scene.models}
        self._allowed_links = {
            m.name: frozenset(frozenset(pair) for pair in m.allowed_collisions)
            for m in problem.scene.models
        }
        self.robots = {r.name: r for r in problem.scene.robots}
        self.objects = {o.name: o for o in problem.objects}
        self.obstacles = {o.name: o for o in problem.scene.obstacles}
        self._robot_bodies = {
            (r.name, link): ContactBody(kind="robot", name=r.name, link=link)
            for r in problem.scene.robots
            for link in self.models[r.model].links
        }
        self._obstacle_bodies = {
            name: ContactBody(kind="obstacle", name=name) for name in self.obstacles
        }
        self._axis_boxes = {
            o.name: (o.pose.position, tuple(v / 2 for v in o.shape.size))
            for o in problem.scene.obstacles
            if isinstance(o.shape, Box) and o.pose.quaternion[:3] == (0.0, 0.0, 0.0)
        }
        self.threshold = value_as(
            problem.scene.config.collision_margin + problem.scene.config.collision_tolerance,
            float,
            "threshold",
        )
        self.info = BackendInfo(name="reference", version="manipulation-geometry-1", device="cpu")
        self._fk_cache = OrderedDict()
        self.fk_hits = self.fk_misses = 0

    @property
    def problem(self):
        return self._problem

    def check_mode(self, mode):
        value_as(mode, ManipulationMode, "mode")
        if {s.object for s in mode.states} != self.objects.keys():
            raise ContractError("mode", "must cover every object exactly once")
        for state in mode.states:
            self.problem.check_state(state)

    def fk(self, robot, positions):
        if robot not in self.robots:
            raise ContractError("robot", "unknown robot")
        q = value_as(positions, tuple[float, ...], "positions")
        instance = self.robots[robot]
        model = self.models[instance.model]
        model.joint_values(q)
        key = (robot, q)
        if key in self._fk_cache:
            self.fk_hits += 1
            self._fk_cache.move_to_end(key)
            return self._fk_cache[key]
        self.fk_misses += 1
        result = forward(model, instance, q)
        if self.policy.max_cached_fk:
            if len(self._fk_cache) >= self.policy.max_cached_fk:
                self._fk_cache.popitem(last=False)
            self._fk_cache[key] = result
        return result

    def position_issues(self, robot, positions):
        model = self.models[self.robots[robot].model]
        values = model.joint_values(positions)
        return tuple(
            Diagnostic(
                code="position_limit",
                message=f"{robot}/{j.name}: outside position bounds",
                path=robot,
            )
            for j in model.joints
            if j.limits is not None
            and j.limits.lower is not None
            and not j.limits.lower <= values[j.name] <= j.limits.upper
        )

    def object_frame(self, state, poses):
        issues = []
        if isinstance(state, PlacedObject):
            pose = state.pose
        else:
            candidates = tuple(_compose(poses[g.robot][g.link], g.pose) for g in state.grasps)
            pose = candidates[0]
            if any(not pose_matches(pose, other, self.policy) for other in candidates[1:]):
                issues.append(
                    Diagnostic(
                        code="grasp_closure",
                        message=f"{state.object}: holder frames disagree",
                        path=state.object,
                    )
                )
        spheres = tuple(
            GeometrySphere(
                ContactBody(kind="object", name=state.object),
                i,
                _compose(pose, Pose(position=s.center)).position,
                s.radius,
            )
            for i, s in enumerate(self.objects[state.object].spheres)
        )
        return ObjectFrame(state.object, pose, spheres), issues

    def support_issues(self, state, frame):
        if not isinstance(state, PlacedObject) or state.support is None:
            return ()
        support = self.obstacles[state.support]
        if (
            not isinstance(support.shape, Box)
            or math.dist(_rotate(support.pose.quaternion, (0, 0, 1)), (0, 0, 1))
            > self.policy.orientation_tolerance
        ):
            return (
                Diagnostic(
                    code="unsupported_support",
                    message="support must be a horizontal box top face",
                    path=state.object,
                ),
            )
        x, y, z, w = support.pose.quaternion
        local = tuple(
            (
                _rotate(
                    (-x, -y, -z, w),
                    tuple(a - b for a, b in zip(s.center, support.pose.position, strict=True)),
                ),
                s.radius,
            )
            for s in frame.spheres
        )
        hx, hy, hz = (v / 2 for v in support.shape.size)
        gaps = tuple(p[2] - r - hz for p, r in local)
        tolerance = self.policy.support_tolerance
        supported = min(gaps) >= -tolerance and any(
            abs(p[2] - r - hz) <= tolerance
            and abs(p[0]) + r <= hx + tolerance
            and abs(p[1]) + r <= hy + tolerance
            for p, r in local
        )
        return (
            ()
            if supported
            else (
                Diagnostic(
                    code="object_support",
                    message=f"{state.object}: no valid top-face support",
                    path=state.object,
                ),
            )
        )

    def _contact_allowed(self, first, second, states, contacts):
        if frozenset((first, second)) in contacts:
            return True
        for obj, other in ((first, second), (second, first)):
            if obj.kind != "object":
                continue
            state = states[obj.name]
            if isinstance(state, HeldObject) and other.kind == "robot":
                if any(g.robot == other.name and g.link == other.link for g in state.grasps):
                    return True
            if (
                isinstance(state, PlacedObject)
                and other.kind == "obstacle"
                and state.support == other.name
            ):
                return True
        return False

    def _collision(self, first, second, clearance, states, contacts):
        clearance = value_as(clearance, float, "clearance")
        allowed = self._contact_allowed(first, second, states, contacts)
        return (
            clearance < -self.policy.contact_tolerance if allowed else clearance <= self.threshold
        )

    def _inspect(self, robots, object_frames, states, contacts):
        contacts = value_as(contacts, tuple[tuple[ContactBody, ContactBody], ...], "contacts")
        for pair in contacts:
            for body in pair:
                if body.kind == "robot":
                    if (
                        body.name not in self.robots
                        or body.link not in self.models[self.robots[body.name].model].links
                    ):
                        raise ContractError("contacts", "unknown robot/link")
                elif body.name not in (self.objects if body.kind == "object" else self.obstacles):
                    raise ContractError("contacts", "unknown body")
        contacts = {frozenset(pair) for pair in contacts}
        spheres = [
            GeometrySphere(self._robot_bodies[name, s.link], s.index, s.center, s.radius)
            for name, fk in robots
            for s in fk.spheres
        ]
        spheres += [s for frame in object_frames for s in frame.spheres]
        issues = []
        for first, second in combinations(spheres, 2):
            a, b = first.body, second.body
            if a == b:
                continue
            if a.kind == b.kind == "robot" and a.name == b.name:
                model = self.models[self.robots[a.name].model]
                if frozenset((a.link, b.link)) in self._allowed_links[model.name]:
                    continue
            clearance = math.dist(first.center, second.center) - first.radius - second.radius
            # Contact exceptions only affect nonpositive clearances. Far finite
            # pairs need no contact lookup; nonfinite arithmetic still fails closed.
            if math.isfinite(clearance) and clearance > self.threshold:
                continue
            if self._collision(a, b, clearance, states, contacts):
                issues.append(
                    Diagnostic(
                        code="collision",
                        message=f"{first.label} / {second.label}: {clearance:.9g} m",
                        path=first.label,
                    )
                )
        for sphere in spheres:
            for obstacle in self.problem.scene.obstacles:
                bounds = self._axis_boxes.get(obstacle.name)
                if bounds is not None:
                    center, half = bounds
                    separation = max(
                        abs(a - b) - h for a, b, h in zip(sphere.center, center, half, strict=True)
                    )
                    reach = sphere.radius + self.threshold
                    guard = 32 * math.ulp(
                        max(1.0, reach, *map(abs, sphere.center), *map(abs, center), *half)
                    )
                    # Conservative one-axis lower bound, with rounding slack.
                    # Near pairs and all rotated boxes retain the exact test.
                    if math.isfinite(separation) and separation > reach + guard:
                        continue
                body = self._obstacle_bodies[obstacle.name]
                clearance = obstacle_clearance(sphere, obstacle)
                if math.isfinite(clearance) and clearance > self.threshold:
                    continue
                if self._collision(sphere.body, body, clearance, states, contacts):
                    issues.append(
                        Diagnostic(
                            code="collision",
                            message=f"{sphere.label} / {obstacle.name}: {clearance:.9g} m",
                            path=sphere.label,
                        )
                    )
        return issues

    def static_frame(self, robot, positions):
        fk = self.fk(robot, positions)
        robots = ((robot, fk),)
        issues = list(self.position_issues(robot, positions))
        issues += self._inspect(robots, (), {}, ())
        return ManipulationFrame(robots, (), tuple(issues))

    def snapshot(self, mode, configurations, *, contacts=(), local_robot=None):
        self.check_mode(mode)
        names = {local_robot} if local_robot is not None else self.robots.keys()
        if set(configurations) != names or not names <= self.robots.keys():
            raise ContractError("configurations", "must cover the requested robots exactly once")
        robots = tuple(
            (name, self.fk(name, configurations[name])) for name in self.robots if name in names
        )
        poses = {name: dict(zip(fk.link_names, fk.poses, strict=True)) for name, fk in robots}
        issues = [
            issue for name in names for issue in self.position_issues(name, configurations[name])
        ]
        frames, states = [], {}
        for state in mode.states:
            if local_robot is not None and isinstance(state, HeldObject):
                holders = {g.robot for g in state.grasps}
                if local_robot not in holders:
                    continue
                if len(holders) != 1:
                    raise UnsupportedModelError(
                        "mode", "dual-held objects require joint skill configurations"
                    )
            frame, extra = self.object_frame(state, poses)
            states[state.object] = state
            frames.append(frame)
            issues += extra
            issues += self.support_issues(state, frame)
        issues += self._inspect(robots, frames, states, contacts)
        return ManipulationFrame(robots, tuple(frames), tuple(issues))
