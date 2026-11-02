"""Small structural interfaces for optional authoring and execution runtimes.

These are contracts only. Implementations declare supported features and
reject unsupported input rather than dropping joints or geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .._contracts import Contract
from ..models import JointKind, PlanningProblem
from ..results import ExecutionReport, Observation, PlanResult


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterCapabilities(Contract):
    scene_import: bool = False
    playback: bool = False
    execution: bool = False
    shapes: tuple[Literal["sphere", "box"], ...] = ()
    joint_types: tuple[JointKind, ...] = ()
    mimic: bool = False

    def _validate(self) -> None:
        self._unique(self.shapes, "shapes")
        self._unique(self.joint_types, "joint_types")


@runtime_checkable
class SceneAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def snapshot(self) -> PlanningProblem:
        """Export the scene and goals, using current positions as the new start."""
        ...


@runtime_checkable
class PlaybackAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def load(self, problem: PlanningProblem, plan: PlanResult) -> None:
        """Require a successful, matching plan; preserve its time law and joint order."""
        ...

    def set_time(self, time: float) -> Observation:
        """Seek all robots to one reference time, without physics or retiming.

        Observations use canonical problem robot/joint names, even when the
        runtime has a different slot order. Out-of-range times are rejected.
        """
        ...


@runtime_checkable
class ExecutionAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def run(self, problem: PlanningProblem, plan: PlanResult) -> ExecutionReport:
        """Track a successful matching plan and report actual clocked observations.

        The implementation owns reset, actuator mapping, stepping, and shutdown.
        Physical tracking is separate from exact kinematic playback.
        """
        ...
