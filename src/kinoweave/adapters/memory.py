"""An in-memory kinematic scene/player for adapter-contract verification."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

from .._contracts import Contract, ContractError, value_as
from ..geometry.reference import WorldSphere, forward
from ..io import document_digest
from ..models import BackendInfo, PlanningProblem, Pose, ValidationReport
from ..results import ExecutionReport, JointState, Observation, PlanResult
from ..trajectory import MotionState
from .conformance import UnsupportedCapabilityError, validate_plan_for_adapter
from .protocols import AdapterCapabilities

MEMORY_CAPABILITIES = AdapterCapabilities(
    scene_import=True,
    playback=True,
    execution=False,
    shapes=("sphere", "box"),
    joint_types=("fixed", "revolute", "continuous", "prismatic"),
    mimic=True,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotBinding(Contract):
    """Planning model joints in runtime slot order, with optional runtime aliases.

    This memory runtime explicitly stores all model joints, including fixed
    and mimic joints. An SDK adapter can use its own actuator/binding layout.
    """

    robot: str
    runtime_robot: str
    joint_names: tuple[str, ...]
    runtime_joint_names: tuple[str, ...] = ()

    def _validate(self):
        self._require(bool(self.joint_names), "joint_names", "at least one model joint required")
        self._unique(self.joint_names, "joint_names")
        if not self.runtime_joint_names:
            object.__setattr__(self, "runtime_joint_names", self.joint_names)
        self._require(
            len(self.joint_names) == len(self.runtime_joint_names),
            "runtime_joint_names",
            "slot count mismatch",
        )
        self._unique(self.runtime_joint_names, "runtime_joint_names")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeRobotState(Contract):
    robot: str
    runtime_robot: str
    model_joint_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]
    link_names: tuple[str, ...]
    link_poses: tuple[Pose, ...]
    spheres: tuple[WorldSphere, ...]

    def _validate(self):
        self._require(
            len(self.model_joint_names)
            == len(self.joint_names)
            == len(self.positions)
            == len(self.velocities)
            == len(self.accelerations),
            "positions",
            "joint slot counts must match",
        )
        self._unique(self.model_joint_names, "model_joint_names")
        self._unique(self.joint_names, "joint_names")
        self._require(
            len(self.link_names) == len(self.link_poses), "link_poses", "link counts must match"
        )
        self._unique(self.link_names, "link_names")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaybackFrame(Contract):
    observation: Observation
    robots: tuple[RuntimeRobotState, ...]

    def _validate(self):
        self._unique(tuple(r.robot for r in self.robots), "robots")
        self._unique(tuple(r.runtime_robot for r in self.robots), "robots")
        self._require(
            {r.robot for r in self.robots} == {r.robot for r in self.observation.robots},
            "robots",
            "observation coverage must match runtime states",
        )


def _resolved(model, state):
    positions = model.joint_values(state.positions)
    velocity = dict(zip(model.joint_order, state.velocities, strict=True))
    acceleration = dict(zip(model.joint_order, state.accelerations, strict=True))
    pending = [j for j in model.joints if j.name not in velocity]
    while pending:
        for joint in pending[:]:
            if joint.kind == "fixed":
                velocity[joint.name] = acceleration[joint.name] = 0.0
            elif joint.mimic is not None and joint.mimic.joint in velocity:
                # Offsets affect positions only, never time derivatives.
                velocity[joint.name] = joint.mimic.multiplier * velocity[joint.mimic.joint]
                acceleration[joint.name] = joint.mimic.multiplier * acceleration[joint.mimic.joint]
            else:
                continue
            pending.remove(joint)
    return positions, velocity, acceleration


def _record_times(trajectory, time_step, max_samples):
    time_step = value_as(time_step, float, "time_step")
    max_samples = value_as(max_samples, int, "max_samples")
    if time_step <= 0 or max_samples < 2:
        raise ContractError("sampling", "positive time_step and at least two samples required")
    ratio = trajectory.time_step / time_step
    if not math.isfinite(ratio) or ratio > max_samples - 1:
        raise ContractError("max_samples", "playback sample budget exceeded")
    subdivisions = max(1, math.ceil(ratio))
    count = (len(trajectory.times) - 1) * subdivisions + 1
    if count > max_samples:
        raise ContractError("max_samples", "playback sample budget exceeded")
    times = [0.0]
    for start, end in pairwise(trajectory.times):
        times.extend(
            start + (end - start) * (i / subdivisions) if i < subdivisions else end
            for i in range(1, subdivisions + 1)
        )
    if any(b <= a for a, b in pairwise(times)):
        raise ContractError("time_step", "sample spacing is below floating-point resolution")
    return tuple(times), trajectory.time_step / subdivisions


class MemoryPlayback:
    """SDK-free SceneAdapter and PlaybackAdapter, with no physical execution.

    The configured problem and bindings are immutable. Loading a verified
    plan resets all robots atomically; seeking commits one complete frame.
    Observations read back independent joints from the runtime slot layout.
    """

    capabilities = MEMORY_CAPABILITIES
    info = BackendInfo(name="memory_playback", version="1", device="cpu")

    def __init__(
        self,
        problem: PlanningProblem,
        *,
        bindings: tuple[RobotBinding, ...] | None = None,
        max_validation_samples: int = 100_000,
    ):
        self._problem = value_as(problem, PlanningProblem, "problem")
        self._digest = document_digest(problem)
        self._max_validation_samples = value_as(
            max_validation_samples, int, "max_validation_samples"
        )
        if self._max_validation_samples < 2:
            raise ContractError("max_validation_samples", "at least two samples required")
        self._models = {m.name: m for m in problem.models}
        for robot in problem.robots:
            if not self._models[robot.model].joint_order:
                raise UnsupportedCapabilityError(
                    f"models.{robot.model}.joint_order",
                    "schema v1 observations require at least one independent joint per robot",
                )
        if bindings is None:
            bindings = tuple(
                RobotBinding(
                    robot=r.name,
                    runtime_robot=r.name,
                    joint_names=tuple(j.name for j in self._models[r.model].joints),
                )
                for r in problem.robots
            )
        bindings = value_as(bindings, tuple[RobotBinding, ...], "bindings")
        if len({b.robot for b in bindings}) != len(bindings) or {b.robot for b in bindings} != {
            r.name for r in problem.robots
        }:
            raise ContractError("bindings", "must cover every robot exactly once")
        if len({b.runtime_robot for b in bindings}) != len(bindings):
            raise ContractError("bindings.runtime_robot", "runtime robot names must be unique")
        self._bindings = {b.robot: b for b in bindings}
        for robot in problem.robots:
            binding = self._bindings[robot.name]
            if set(binding.joint_names) != {j.name for j in self._models[robot.model].joints}:
                raise ContractError(
                    f"bindings.{robot.name}", "must name every model joint exactly once"
                )
        self._closed = False
        self._plan = None
        self._plan_digest = None
        self._verification = ValidationReport()
        states = {
            r.name: MotionState(
                positions=r.start, velocities=(0,) * len(r.start), accelerations=(0,) * len(r.start)
            )
            for r in problem.robots
        }
        self._frame = self._make_frame(0.0, states)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("playback is closed")

    def _require_plan(self):
        self._check_open()
        if self._plan is None:
            raise RuntimeError("load a successful plan first")
        return self._plan

    @property
    def frame(self) -> PlaybackFrame:
        self._check_open()
        return self._frame

    @property
    def verification(self) -> ValidationReport:
        self._check_open()
        return self._verification

    @property
    def plan_digest(self) -> str | None:
        self._check_open()
        return self._plan_digest

    def _make_frame(self, time, states):
        robots = []
        observations = []
        for robot in self._problem.robots:
            model = self._models[robot.model]
            binding = self._bindings[robot.name]
            state = states[robot.name]
            q, dq, ddq = _resolved(model, state)
            fk = forward(model, robot, state.positions)
            runtime = RuntimeRobotState(
                robot=robot.name,
                runtime_robot=binding.runtime_robot,
                model_joint_names=binding.joint_names,
                joint_names=binding.runtime_joint_names,
                positions=tuple(q[n] for n in binding.joint_names),
                velocities=tuple(dq[n] for n in binding.joint_names),
                accelerations=tuple(ddq[n] for n in binding.joint_names),
                link_names=fk.link_names,
                link_poses=fk.poses,
                spheres=fk.spheres,
            )
            slots = {name: i for i, name in enumerate(runtime.model_joint_names)}
            observations.append(
                JointState(
                    robot=robot.name,
                    joint_names=model.joint_order,
                    positions=tuple(runtime.positions[slots[n]] for n in model.joint_order),
                    velocities=tuple(runtime.velocities[slots[n]] for n in model.joint_order),
                )
            )
            robots.append(runtime)
        return PlaybackFrame(
            observation=Observation(time=time, reference_time=time, robots=tuple(observations)),
            robots=tuple(robots),
        )

    def snapshot(self) -> PlanningProblem:
        """Export the current in-memory scene as a new planning start state."""
        self._check_open()
        states = {r.robot: r.positions for r in self._frame.observation.robots}
        return replace(
            self._problem,
            robots=tuple(replace(r, start=states[r.name]) for r in self._problem.robots),
        )

    def load(self, problem: PlanningProblem, plan: PlanResult) -> None:
        self._check_open()
        problem = value_as(problem, PlanningProblem, "problem")
        if document_digest(problem) != self._digest:
            raise ContractError("problem", "does not match the configured runtime scene")
        verification = validate_plan_for_adapter(
            problem, plan, self.capabilities, max_samples=self._max_validation_samples
        )
        assert plan.trajectory is not None
        frame = self._make_frame(0.0, plan.trajectory.evaluate(0.0))
        digest = document_digest(plan)
        self._plan, self._plan_digest, self._verification, self._frame = (
            plan,
            digest,
            verification,
            frame,
        )

    def set_time(self, time: float) -> Observation:
        plan = self._require_plan()
        assert plan.trajectory is not None
        time = value_as(time, float, "time")
        frame = self._make_frame(time, plan.trajectory.evaluate(time))
        self._frame = frame
        return frame.observation

    def reset(self) -> Observation:
        return self.set_time(0.0)

    def record(self, time_step: float, *, max_samples: int = 100_000) -> ExecutionReport:
        """Seek a forward nominal grid including knots, and record actual readback.

        This starts at reference time zero. One frame commits at a time; an
        evaluation failure leaves the last committed frame and emits no report.
        """
        plan = self._require_plan()
        assert plan.trajectory is not None and self._plan_digest is not None
        times, actual_step = _record_times(plan.trajectory, time_step, max_samples)
        observations = tuple(self.set_time(time) for time in times)
        report = ExecutionReport(
            plan_digest=self._plan_digest,
            runtime=self.info,
            mode="kinematic_playback",
            observations=observations,
            time_step=actual_step,
        )
        report.check_plan(self._problem, plan)
        return report

    def close(self):
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()
