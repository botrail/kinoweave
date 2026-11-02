"""Dependency-free manipulation inputs, with explicit per-object state changes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .._contracts import Contract
from ..models import Conventions, PlanningProblem, Pose, Vector3
from ..trajectory import RobotTrajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectSphere(Contract):
    center: Vector3
    radius: float

    def _validate(self):
        self._require(self.radius > 0, "radius", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class RigidObject(Contract):
    name: str
    spheres: tuple[ObjectSphere, ...]
    revision: str = "1"

    def _validate(self):
        self._require(bool(self.spheres), "spheres", "explicit object-local geometry is required")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraspAttachment(Contract):
    robot: str
    link: str
    pose: Pose = field(default_factory=Pose)  # T_link_object


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacedObject(Contract):
    object: str
    pose: Pose  # T_world_object
    support: str | None = None  # Static obstacle name; not proof of physical support.
    kind: Literal["placed"] = "placed"


@dataclass(frozen=True, slots=True, kw_only=True)
class HeldObject(Contract):
    object: str
    grasps: tuple[GraspAttachment, ...]
    kind: Literal["held"] = "held"

    def _validate(self):
        self._require(1 <= len(self.grasps) <= 2, "grasps", "one or two holders are required")
        self._unique(tuple(g.robot for g in self.grasps), "grasps")


ObjectState = PlacedObject | HeldObject


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationMode(Contract):
    name: str
    states: tuple[ObjectState, ...]

    def _validate(self):
        self._unique(tuple(s.object for s in self.states), "states")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeTransition(Contract):
    time: float  # Seconds relative to the option's source trace.
    before: ObjectState
    after: ObjectState

    def _validate(self):
        self._require(self.time >= 0, "time", "must be non-negative")
        self._require(self.before.object == self.after.object, "after", "object must not change")
        self._require(self.before != self.after, "after", "transition must change object state")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContactBody(Contract):
    kind: Literal["robot", "object", "obstacle"]
    name: str
    link: str | None = None

    def _validate(self):
        self._require(
            (self.kind == "robot") == (self.link is not None),
            "link",
            "robot contacts require a link; other bodies have no link",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContactWindow(Contract):
    first: ContactBody
    second: ContactBody
    start: float
    end: float

    def _validate(self):
        self._require(self.first != self.second, "second", "contact bodies must differ")
        self._require(0 <= self.start <= self.end, "end", "expected 0 <= start <= end")


@dataclass(frozen=True, slots=True, kw_only=True)
class JointTrace(Contract):
    """Explicit relative timestamps; this record does not execute or certify motion."""

    times: tuple[float, ...]
    tracks: tuple[RobotTrajectory, ...]
    interpolation: Literal["linear", "quintic_stop"] = "linear"

    def _validate(self):
        self._require(len(self.times) >= 2, "times", "at least two knots are required")
        self._require(self.times[0] == 0, "times", "trace must start at zero")
        self._require(
            all(a < b for a, b in zip(self.times, self.times[1:], strict=False)),
            "times",
            "times must strictly increase",
        )
        self._require(bool(self.tracks), "tracks", "at least one robot is required")
        self._unique(tuple(t.robot for t in self.tracks), "tracks")
        for track in self.tracks:
            self._require(
                len(track.positions) == len(self.times), "tracks", "tracks must cover every knot"
            )

    @property
    def duration(self):
        return self.times[-1]


@dataclass(frozen=True, slots=True, kw_only=True)
class SkillOption(Contract):
    name: str
    trace: JointTrace
    preconditions: tuple[ObjectState, ...] = ()
    transitions: tuple[ModeTransition, ...] = ()
    contacts: tuple[ContactWindow, ...] = ()

    def _validate(self):
        self._unique(tuple(s.object for s in self.preconditions), "preconditions")
        known = {s.object: s for s in self.preconditions}
        previous = -1.0
        simultaneous = set()
        for transition in self.transitions:
            self._require(
                previous <= transition.time <= self.trace.duration,
                "transitions",
                "transitions must be time-ordered within the trace",
            )
            if transition.time != previous:
                simultaneous.clear()
            obj = transition.before.object
            self._require(obj not in simultaneous, "transitions", "ambiguous simultaneous effects")
            self._require(
                known.get(obj) == transition.before,
                "transitions",
                "before state must match the declared precondition or preceding effect",
            )
            simultaneous.add(obj)
            known[obj] = transition.after
            previous = transition.time
        for contact in self.contacts:
            self._require(contact.end <= self.trace.duration, "contacts", "window exceeds trace")
            for body in (contact.first, contact.second):
                self._require(
                    body.kind != "object" or body.name in known,
                    "contacts",
                    "contact objects need an explicit precondition",
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class Skill(Contract):
    """One configured action instance, not a physics controller or skill sampler."""

    name: str
    robots: tuple[str, ...]
    options: tuple[SkillOption, ...]
    minimum_duration: float
    timing: Literal["fixed", "stretch"] = "fixed"
    kind: str = "custom"  # Descriptive label; preconditions/effects define its semantics.

    def _validate(self):
        self._require(1 <= len(self.robots) <= 2, "robots", "one or two participants are supported")
        self._unique(self.robots, "robots")
        self._require(bool(self.options), "options", "at least one option is required")
        self._unique(tuple(o.name for o in self.options), "options")
        self._require(self.minimum_duration > 0, "minimum_duration", "must be positive")
        for option in self.options:
            self._require(
                {t.robot for t in option.trace.tracks} == set(self.robots),
                "options",
                "option tracks must cover all participants exactly once",
            )
            self._require(
                self.timing != "fixed" or option.trace.duration >= self.minimum_duration,
                "minimum_duration",
                "fixed trace is shorter than the minimum duration",
            )
            for transition in option.transitions:
                for state in (transition.before, transition.after):
                    if isinstance(state, HeldObject):
                        self._require(
                            {g.robot for g in state.grasps} <= set(self.robots),
                            "options.transitions",
                            "all affected holders must participate in the skill",
                        )
            for contact in option.contacts:
                self._require(
                    all(
                        b.kind != "robot" or b.name in self.robots
                        for b in (contact.first, contact.second)
                    ),
                    "options.contacts",
                    "contact permissions cannot name non-participating robots",
                )

    @property
    def objects(self):
        return frozenset(s.object for o in self.options for s in o.preconditions)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionOrder(Contract):
    before: str
    after: str

    def _validate(self):
        self._require(self.before != self.after, "after", "an action cannot precede itself")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleSkeleton(Contract):
    skills: tuple[Skill, ...]
    precedence: tuple[ActionOrder, ...] = ()

    def _validate(self):
        self._require(bool(self.skills), "skills", "at least one configured skill is required")
        names = {s.name for s in self.skills}
        self._unique(tuple(s.name for s in self.skills), "skills")
        self._unique(tuple((p.before, p.after) for p in self.precedence), "precedence")
        for edge in self.precedence:
            self._require(
                {edge.before, edge.after} <= names, "precedence", "unknown action reference"
            )
        ancestors = self.ancestors()
        for i, first in enumerate(self.skills):
            for second in self.skills[i + 1 :]:
                shared = bool(
                    set(first.robots) & set(second.robots) or first.objects & second.objects
                )
                self._require(
                    not shared
                    or first.name in ancestors[second.name]
                    or second.name in ancestors[first.name],
                    "precedence",
                    "actions sharing a robot or manipulated object require an explicit order",
                )

    def ancestors(self):
        pending = {s.name: set() for s in self.skills}
        for edge in self.precedence:
            pending[edge.after].add(edge.before)
        result = {}
        while pending:
            ready = sorted(name for name, parents in pending.items() if parents <= result.keys())
            self._require(bool(ready), "precedence", "cycle in action dependencies")
            for name in ready:
                parents = pending.pop(name)
                result[name] = frozenset(parents.union(*(result[p] for p in parents)))
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationCost(Contract):
    makespan: float = 1.0
    completion: float = 0.0
    distance: float = 0.0
    distance_metric: Literal["joint_l1", "joint_l2", "tool_path", "sphere_volume_path"] = "joint_l1"

    def _validate(self):
        values = (self.makespan, self.completion, self.distance)
        self._require(all(v >= 0 for v in values), "weights", "weights must be non-negative")
        self._require(any(v > 0 for v in values), "weights", "at least one weight must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class ManipulationProblem(Contract):
    scene: PlanningProblem
    objects: tuple[RigidObject, ...]
    initial_mode: ManipulationMode
    goal_mode: ManipulationMode
    skeleton: ScheduleSkeleton
    cost: ManipulationCost = field(default_factory=ManipulationCost)
    conventions: Conventions = field(default_factory=Conventions)
    format: Literal["kinoweave.manipulation.problem"] = "kinoweave.manipulation.problem"
    schema_version: Literal[1] = 1

    def _validate(self):
        robots = {r.name: r for r in self.scene.robots}
        obstacles = {o.name for o in self.scene.obstacles}
        objects = {o.name for o in self.objects}
        self._unique(
            tuple(robots) + tuple(obstacles) + tuple(o.name for o in self.objects), "objects"
        )
        for mode in (self.initial_mode, self.goal_mode):
            self._require(
                {s.object for s in mode.states} == objects,
                mode.name,
                "mode must cover every object exactly once",
            )
            for state in mode.states:
                self.check_state(state)
        for skill in self.skeleton.skills:
            self._require(set(skill.robots) <= robots.keys(), "skeleton", "unknown robot")
            for option in skill.options:
                self.check_trace(option.trace)
                for state in option.preconditions:
                    self.check_state(state)
                for transition in option.transitions:
                    self.check_state(transition.before)
                    self.check_state(transition.after)
                for window in option.contacts:
                    for body in (window.first, window.second):
                        if body.kind == "robot":
                            self.check_grasp(GraspAttachment(robot=body.name, link=body.link))
                        else:
                            self._require(
                                body.name in (objects if body.kind == "object" else obstacles),
                                "contacts",
                                "unknown contact body",
                            )

    def check_grasp(self, grasp: GraspAttachment):
        robots = {r.name: r for r in self.scene.robots}
        self._require(grasp.robot in robots, "grasps", "unknown robot")
        model = next(m for m in self.scene.models if m.name == robots[grasp.robot].model)
        self._require(grasp.link in model.links, "grasps", "unknown robot link")

    def check_state(self, state: ObjectState):
        self._require(state.object in {o.name for o in self.objects}, "states", "unknown object")
        if isinstance(state, HeldObject):
            for grasp in state.grasps:
                self.check_grasp(grasp)
        else:
            self._require(
                state.support is None or state.support in {o.name for o in self.scene.obstacles},
                "states.support",
                "support must name a static obstacle",
            )

    def check_trace(self, trace: JointTrace):
        robots = {r.name: r for r in self.scene.robots}
        models = {m.name: m for m in self.scene.models}
        for track in trace.tracks:
            self._require(track.robot in robots, "trace", "unknown robot")
            model = models[robots[track.robot].model]
            self._require(
                track.joint_names == model.joint_order, "trace.joint_names", "joint order mismatch"
            )
            for row in track.positions:
                values = model.joint_values(row)
                for joint in model.joints:
                    limits = joint.limits
                    if limits is not None and limits.lower is not None:
                        self._require(
                            limits.lower <= values[joint.name] <= limits.upper,
                            "trace.positions",
                            f"joint {joint.name!r} is outside position limits",
                        )


def same_duration(first, second):
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=0)
