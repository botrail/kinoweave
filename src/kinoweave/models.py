"""Platform-neutral input, provenance, and validation contracts.

All quantities use metres, radians, and seconds in right-handed frames.
Pose quaternions are xyzw, and poses map local coordinates into the parent frame.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from ._contracts import Contract

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
JointKind = Literal["fixed", "revolute", "continuous", "prismatic"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Conventions(Contract):
    """Explicit wire-format conventions; unsupported conventions are rejected."""

    length: Literal["m"] = "m"
    angle: Literal["rad"] = "rad"
    time: Literal["s"] = "s"
    handedness: Literal["right"] = "right"
    quaternion_order: Literal["xyzw"] = "xyzw"
    pose_direction: Literal["parent_from_local"] = "parent_from_local"


@dataclass(frozen=True, slots=True, kw_only=True)
class Pose(Contract):
    position: Vector3 = (0.0, 0.0, 0.0)
    quaternion: Quaternion = (0.0, 0.0, 0.0, 1.0)

    def _validate(self) -> None:
        self._require(
            math.isclose(math.hypot(*self.quaternion), 1.0, rel_tol=0, abs_tol=1e-6),
            "quaternion",
            "expected a unit xyzw quaternion (not automatically normalized)",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Sphere(Contract):
    radius: float
    kind: Literal["sphere"] = "sphere"

    def _validate(self) -> None:
        self._require(self.radius > 0, "radius", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class Box(Contract):
    """Full side lengths, centred at the obstacle pose."""

    size: Vector3
    kind: Literal["box"] = "box"

    def _validate(self) -> None:
        self._require(all(side > 0 for side in self.size), "size", "sides must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class LinkSphere(Contract):
    link: str
    center: Vector3
    radius: float

    def _validate(self) -> None:
        self._require(self.radius > 0, "radius", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class Obstacle(Contract):
    name: str
    shape: Sphere | Box
    pose: Pose = field(default_factory=Pose)


@dataclass(frozen=True, slots=True, kw_only=True)
class JointLimits(Contract):
    velocity: float
    acceleration: float
    lower: float | None = None
    upper: float | None = None

    def _validate(self) -> None:
        self._require(self.velocity > 0, "velocity", "must be positive")
        self._require(self.acceleration > 0, "acceleration", "must be positive")
        self._require(
            (self.lower is None) == (self.upper is None),
            "lower",
            "lower and upper must either both be set or both be null",
        )
        if self.lower is not None and self.upper is not None:
            self._require(self.lower <= self.upper, "upper", "must be at least lower")


@dataclass(frozen=True, slots=True, kw_only=True)
class Mimic(Contract):
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Joint(Contract):
    name: str
    kind: JointKind
    parent: str
    child: str
    origin: Pose = field(default_factory=Pose)
    axis: Vector3 = (0.0, 0.0, 1.0)
    limits: JointLimits | None = None
    mimic: Mimic | None = None

    def _validate(self) -> None:
        self._require(self.parent != self.child, "child", f"joint {self.name!r} connects itself")
        if self.kind == "fixed":
            self._require(self.limits is None, "limits", "fixed joints have no motion limits")
            self._require(self.mimic is None, "mimic", "fixed joints cannot mimic a joint")
            return
        self._require(self.limits is not None, "limits", f"joint {self.name!r} needs limits")
        self._require(
            math.isclose(math.hypot(*self.axis), 1.0, rel_tol=0, abs_tol=1e-6),
            "axis",
            f"joint {self.name!r} needs a unit axis",
        )
        assert self.limits is not None
        if self.kind == "continuous":
            self._require(self.limits.lower is None, "limits", "continuous joints are unbounded")
        else:
            self._require(self.limits.lower is not None, "limits", "position bounds are required")


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotModel(Contract):
    name: str
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    joint_order: tuple[str, ...]
    collision_spheres: tuple[LinkSphere, ...]
    revision: str = "1"
    allowed_collisions: tuple[tuple[str, str], ...] = ()

    def _validate(self) -> None:
        self._require(bool(self.links), "links", f"model {self.name!r} needs a root link")
        self._unique(self.links, "links")
        self._unique(tuple(joint.name for joint in self.joints), "joints")
        self._unique(self.joint_order, "joint_order")
        by_name = {joint.name: joint for joint in self.joints}
        parents = {}
        for joint in self.joints:
            self._require(
                joint.parent in self.links and joint.child in self.links,
                "joints",
                f"joint {joint.name!r} refers to an unknown link",
            )
            self._require(
                joint.child not in parents,
                "joints",
                f"link {joint.child!r} has multiple parents",
            )
            parents[joint.child] = joint.parent
        roots = set(self.links) - parents.keys()
        self._require(len(roots) == 1, "links", f"model {self.name!r} must have one root")
        for link in self.links:
            visited = set()
            current = link
            while current in parents:
                self._require(current not in visited, "joints", f"cycle at link {current!r}")
                visited.add(current)
                current = parents[current]
        independent = {
            joint.name for joint in self.joints if joint.kind != "fixed" and joint.mimic is None
        }
        self._require(
            set(self.joint_order) == independent,
            "joint_order",
            f"must list every independent joint exactly once: {sorted(independent)!r}",
        )
        for joint in self.joints:
            chain = set()
            current = joint
            while current.mimic is not None:
                source = by_name.get(current.mimic.joint)
                self._require(
                    source is not None, "joints", f"unknown mimic source for {current.name!r}"
                )
                assert source is not None
                self._require(
                    source.kind != "fixed"
                    and (source.kind == "prismatic") == (current.kind == "prismatic"),
                    "joints",
                    f"incompatible mimic source for {current.name!r}",
                )
                self._require(
                    current.name not in chain, "joints", f"mimic cycle at {current.name!r}"
                )
                chain.add(current.name)
                current = source
        self._require(
            bool(self.collision_spheres),
            "collision_spheres",
            "explicit collision geometry is required",
        )
        for sphere in self.collision_spheres:
            self._require(
                sphere.link in self.links,
                "collision_spheres",
                f"unknown link {sphere.link!r}",
            )
        pairs = set()
        for first, second in self.allowed_collisions:
            self._require(
                first in self.links and second in self.links and first != second,
                "allowed_collisions",
                f"invalid link pair {(first, second)!r}",
            )
            pair = frozenset((first, second))
            self._require(
                pair not in pairs, "allowed_collisions", f"duplicate pair {(first, second)!r}"
            )
            pairs.add(pair)

    def joint_values(self, positions: tuple[float, ...]) -> dict[str, float]:
        """Resolve fixed and mimic joints without doing forward kinematics."""
        from ._contracts import value_as

        positions = value_as(positions, tuple[float, ...], f"RobotModel.{self.name}.positions")
        self._require(len(positions) == len(self.joint_order), "positions", "joint count mismatch")
        values = dict(zip(self.joint_order, positions, strict=True))
        pending = [joint for joint in self.joints if joint.name not in values]
        while pending:
            for joint in pending[:]:
                if joint.kind == "fixed":
                    values[joint.name] = 0.0
                elif joint.mimic is not None and joint.mimic.joint in values:
                    values[joint.name] = (
                        joint.mimic.multiplier * values[joint.mimic.joint] + joint.mimic.offset
                    )
                else:
                    continue
                self._require(
                    math.isfinite(values[joint.name]),
                    "positions",
                    f"mimic joint {joint.name!r} overflows",
                )
                pending.remove(joint)
        return values


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotInstance(Contract):
    name: str
    model: str
    start: tuple[float, ...]
    goal: tuple[float, ...]
    base_pose: Pose = field(default_factory=Pose)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannerConfig(Contract):
    backend: str = "curobo"
    device: str = "cuda:0"
    seed: int = 0
    time_limit: float = 10.0
    time_step: float = 0.1
    validation_time_step: float = 0.01
    collision_margin: float = 0.01
    collision_tolerance: float = 1e-6
    joint_tolerance: float = 1e-6

    def _validate(self) -> None:
        self._require(self.seed >= 0, "seed", "must be non-negative")
        for name in ("time_limit", "time_step", "validation_time_step"):
            self._require(getattr(self, name) > 0, name, "must be positive")
        for name in ("collision_margin", "collision_tolerance", "joint_tolerance"):
            self._require(getattr(self, name) >= 0, name, "must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanningProblem(Contract):
    name: str
    models: tuple[RobotModel, ...]
    robots: tuple[RobotInstance, ...]
    obstacles: tuple[Obstacle, ...] = ()
    config: PlannerConfig = field(default_factory=PlannerConfig)
    scene_revision: str = "1"
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.problem"] = "kinoweave.problem"
    schema_version: Literal[1] = 1

    def _validate(self) -> None:
        self._require(bool(self.models), "models", "at least one model is required")
        self._require(bool(self.robots), "robots", "at least one robot is required")
        self._unique(tuple(model.name for model in self.models), "models")
        self._unique(
            tuple(robot.name for robot in self.robots) + tuple(obj.name for obj in self.obstacles),
            "robots",  # Robots and obstacles share the scene entity namespace.
        )
        models = {model.name: model for model in self.models}
        for robot in self.robots:
            self._require(
                robot.model in models,
                f"robots.{robot.name}.model",
                f"unknown model {robot.model!r}",
            )
            model = models[robot.model]
            for state_name in ("start", "goal"):
                positions = getattr(robot, state_name)
                path = f"robots.{robot.name}.{state_name}"
                self._require(
                    len(positions) == len(model.joint_order),
                    path,
                    f"expected joint order {model.joint_order!r}",
                )
                values = model.joint_values(positions)
                for joint in model.joints:
                    if joint.limits is not None and joint.limits.lower is not None:
                        assert joint.limits.upper is not None
                        self._require(
                            joint.limits.lower <= values[joint.name] <= joint.limits.upper,
                            path,
                            f"joint {joint.name!r} is outside its position limits",
                        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Diagnostic(Contract):
    code: str
    message: str
    stage: str | None = None
    path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendInfo(Contract):
    name: str
    version: str
    device: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanProvenance(Contract):
    problem_digest: str
    config: PlannerConfig
    backend: BackendInfo | None = None
    package_version: str = "0.1.0.dev0"

    def _validate(self) -> None:
        self._require(
            re.fullmatch(r"[0-9a-f]{64}", self.problem_digest) is not None,
            "problem_digest",
            "expected a lowercase SHA-256 digest",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport(Contract):
    """Validation evidence supplied by a verifier, not computed by these types."""

    status: Literal["not_run", "passed", "failed"] = "not_run"
    level: Literal["sampled", "conservative"] | None = None
    checks: tuple[str, ...] = ()
    sample_time_step: float | None = None
    collision_margin: float | None = None
    collision_tolerance: float | None = None
    issues: tuple[Diagnostic, ...] = ()

    def _validate(self) -> None:
        self._unique(self.checks, "checks")
        if self.status == "not_run":
            self._require(
                self.level is None
                and not self.checks
                and not self.issues
                and self.sample_time_step is None
                and self.collision_margin is None
                and self.collision_tolerance is None,
                "status",
                "not_run cannot contain validation evidence",
            )
            return
        self._require(
            self.level is not None and bool(self.checks), "checks", "evidence is required"
        )
        if self.level == "sampled":
            self._require(
                self.sample_time_step is not None and self.sample_time_step > 0,
                "sample_time_step",
                "sampled validation needs a positive time step",
            )
        elif self.sample_time_step is not None:
            self._require(self.sample_time_step > 0, "sample_time_step", "must be positive")
        for name in ("collision_margin", "collision_tolerance"):
            value = getattr(self, name)
            self._require(
                value is not None and value >= 0, name, "must be supplied and non-negative"
            )
        self._require(
            bool(self.issues) == (self.status == "failed"),
            "issues",
            "failed validation needs issues; passed validation cannot contain issues",
        )
