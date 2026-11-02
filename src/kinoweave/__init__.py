"""Platform-independent contracts and quintic motion for multi-robot planning.

CPU reference geometry, validation, search, and optional GPU geometry live in
their own modules. Importing the core requires no third-party SDKs.
"""

from ._contracts import ContractError
from .models import (
    BackendInfo,
    Box,
    Conventions,
    Diagnostic,
    Joint,
    JointLimits,
    LinkSphere,
    Mimic,
    Obstacle,
    PlannerConfig,
    PlanningProblem,
    PlanProvenance,
    Pose,
    RobotInstance,
    RobotModel,
    Sphere,
    ValidationReport,
)
from .results import (
    Contact,
    ExecutionReport,
    JointState,
    Observation,
    PlanResult,
    TrackingError,
)
from .trajectory import MotionState, RobotTrajectory, TrajectoryBundle

__version__ = "0.1.0.dev0"

from .planner import Planner
from .search.space import SearchConfig

__all__ = [
    "BackendInfo",
    "Box",
    "Contact",
    "ContractError",
    "Conventions",
    "Diagnostic",
    "ExecutionReport",
    "Joint",
    "JointLimits",
    "JointState",
    "LinkSphere",
    "Mimic",
    "MotionState",
    "Observation",
    "Obstacle",
    "PlanProvenance",
    "PlanResult",
    "PlannerConfig",
    "Planner",
    "PlanningProblem",
    "Pose",
    "RobotInstance",
    "RobotModel",
    "RobotTrajectory",
    "Sphere",
    "SearchConfig",
    "TrackingError",
    "TrajectoryBundle",
    "ValidationReport",
]
