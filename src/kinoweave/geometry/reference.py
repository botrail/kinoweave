"""Small CPU reference, independent of cuRobo, Torch, and simulator SDKs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from .._contracts import Contract, ContractError, value_as
from ..models import Box, PlanningProblem, Pose, RobotInstance, RobotModel, Sphere


def _rotate(quaternion, vector):
    x, y, z, w = quaternion
    norm = math.hypot(x, y, z, w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    a, b, c = vector
    tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
    return (
        a + w * tx + y * tz - z * ty,
        b + w * ty + z * tx - x * tz,
        c + w * tz + x * ty - y * tx,
    )


def _compose(parent: Pose, local: Pose) -> Pose:
    a, b, c, d = parent.quaternion
    x, y, z, w = local.quaternion
    quaternion = (
        d * x + a * w + b * z - c * y,
        d * y - a * z + b * w + c * x,
        d * z + a * y - b * x + c * w,
        d * w - a * x - b * y - c * z,
    )
    norm = math.hypot(*quaternion)
    rotated = _rotate(parent.quaternion, local.position)
    return Pose(
        position=tuple(a + b for a, b in zip(parent.position, rotated, strict=True)),
        quaternion=tuple(v / norm for v in quaternion),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WorldSphere(Contract):
    robot: str
    link: str
    index: int
    center: tuple[float, float, float]
    radius: float

    def _validate(self) -> None:
        self._require(self.radius > 0 and self.index >= 0, "radius", "invalid sphere")

    @property
    def label(self) -> str:
        return f"{self.robot}/{self.link}/sphere[{self.index}]"


@dataclass(frozen=True, slots=True)
class ReferenceFK:
    link_names: tuple[str, ...]
    poses: tuple[Pose, ...]
    spheres: tuple[WorldSphere, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Collision(Contract):
    first: str
    second: str
    clearance: float


def forward(model: RobotModel, instance: RobotInstance, positions) -> ReferenceFK:
    """CPU FK for all core joint types, preserving explicit joint/sphere order."""
    if instance.model != model.name:
        raise ContractError("model", "instance and model do not match")
    values = model.joint_values(positions)
    root = next(link for link in model.links if link not in {j.child for j in model.joints})
    poses = {root: instance.base_pose}
    pending = list(model.joints)
    while pending:
        for joint in pending[:]:
            if joint.parent not in poses:
                continue
            value = values[joint.name]
            axis_norm = math.hypot(*joint.axis)
            axis = tuple(v / axis_norm for v in joint.axis) if joint.kind != "fixed" else joint.axis
            if joint.kind == "prismatic":
                motion = Pose(position=tuple(v * value for v in axis))
            elif joint.kind in ("revolute", "continuous"):
                motion = Pose(
                    quaternion=(*tuple(v * math.sin(value / 2) for v in axis), math.cos(value / 2))
                )
            else:
                motion = Pose()
            poses[joint.child] = _compose(_compose(poses[joint.parent], joint.origin), motion)
            pending.remove(joint)
    spheres = []
    for index, sphere in enumerate(model.collision_spheres):
        center = _compose(poses[sphere.link], Pose(position=sphere.center)).position
        spheres.append(
            WorldSphere(
                robot=instance.name,
                link=sphere.link,
                index=index,
                center=center,
                radius=sphere.radius,
            )
        )
    return ReferenceFK(model.links, tuple(poses[link] for link in model.links), tuple(spheres))


def sphere_clearance(first: WorldSphere, second: WorldSphere) -> float:
    return value_as(
        math.dist(first.center, second.center) - first.radius - second.radius,
        float,
        "sphere_clearance",
    )


def obstacle_clearance(sphere: WorldSphere, obstacle) -> float:
    delta = tuple(a - b for a, b in zip(sphere.center, obstacle.pose.position, strict=True))
    if isinstance(obstacle.shape, Sphere):
        distance = math.hypot(*delta) - obstacle.shape.radius
    elif isinstance(obstacle.shape, Box):
        x, y, z, w = obstacle.pose.quaternion
        local = _rotate((-x, -y, -z, w), delta)
        excess = tuple(
            abs(v) - side / 2 for v, side in zip(local, obstacle.shape.size, strict=True)
        )
        distance = math.hypot(*(max(v, 0) for v in excess)) + min(max(excess), 0)
    else:
        raise ContractError("obstacle.shape", "unsupported shape")
    return value_as(distance - sphere.radius, float, "obstacle_clearance")


class ReferenceCollisionChecker:
    """Self, static, and inter-robot sphere checks at one shared configuration.

    All robots are required, including robots that are waiting or at a goal.
    This checks supplied geometry, not joint limits or continuous motion.
    """

    def __init__(self, problem: PlanningProblem):
        self.problem = problem
        self.models = {m.name: m for m in problem.models}
        self.robots = {r.name: r for r in problem.robots}
        self.threshold = value_as(
            problem.config.collision_margin + problem.config.collision_tolerance,
            float,
            "collision_threshold",
        )
        self.ignored = {
            m.name: {frozenset(pair) for pair in m.allowed_collisions} for m in problem.models
        }

    def forward(self, robot: str, positions) -> ReferenceFK:
        if robot not in self.robots:
            raise ContractError("robot", f"unknown robot {robot!r}")
        instance = self.robots[robot]
        return forward(self.models[instance.model], instance, positions)

    def collisions(self, configurations: Mapping[str, tuple[float, ...]]) -> tuple[Collision, ...]:
        if set(configurations) != self.robots.keys():
            raise ContractError("configurations", "must cover every robot exactly once")
        groups = []
        collisions = []
        for robot in self.problem.robots:
            spheres = self.forward(robot.name, configurations[robot.name]).spheres
            groups.append(spheres)
            for first, second in combinations(spheres, 2):
                if (
                    first.link == second.link
                    or frozenset((first.link, second.link)) in self.ignored[robot.model]
                ):
                    continue
                clearance = sphere_clearance(first, second)
                if clearance <= self.threshold:
                    collisions.append(
                        Collision(first=first.label, second=second.label, clearance=clearance)
                    )
            for sphere in spheres:
                for obstacle in self.problem.obstacles:
                    clearance = obstacle_clearance(sphere, obstacle)
                    if clearance <= self.threshold:
                        collisions.append(
                            Collision(first=sphere.label, second=obstacle.name, clearance=clearance)
                        )
        for first_group, second_group in combinations(groups, 2):
            for first in first_group:
                for second in second_group:
                    clearance = sphere_clearance(first, second)
                    if clearance <= self.threshold:
                        collisions.append(
                            Collision(first=first.label, second=second.label, clearance=clearance)
                        )
        return tuple(collisions)
