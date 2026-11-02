"""Optional runtime boundaries. No simulator SDK is imported here."""

from .conformance import (
    UnsupportedCapabilityError,
    check_capabilities,
    check_playback_report,
    take_snapshot,
    validate_plan_for_adapter,
)
from .memory import MemoryPlayback, PlaybackFrame, RobotBinding, RuntimeRobotState
from .protocols import AdapterCapabilities, ExecutionAdapter, PlaybackAdapter, SceneAdapter

__all__ = [
    "AdapterCapabilities",
    "ExecutionAdapter",
    "PlaybackAdapter",
    "SceneAdapter",
    "UnsupportedCapabilityError",
    "check_capabilities",
    "take_snapshot",
    "validate_plan_for_adapter",
    "check_playback_report",
    "MemoryPlayback",
    "PlaybackFrame",
    "RobotBinding",
    "RuntimeRobotState",
]
