"""Manipulation contracts, mode geometry, multi-arm planning and refinement."""

from importlib import import_module

from .models import (
    ActionOrder,
    ContactBody,
    ContactWindow,
    GraspAttachment,
    HeldObject,
    JointTrace,
    ManipulationCost,
    ManipulationMode,
    ManipulationProblem,
    ModeTransition,
    ObjectSphere,
    PlacedObject,
    RigidObject,
    ScheduleSkeleton,
    Skill,
    SkillOption,
)
from .results import (
    MANIPULATION_CHECKS,
    ManipulationPlan,
    ManipulationResult,
    ScheduledMotion,
    ScheduledSkill,
)
from .tasks import SkeletonCandidate, Task, TaskAlternative, TaskChoice, TaskDomain

_LAZY = {
    "geometry": ("GeometryPolicy", "ModeGeometry", "pose_errors"),
    "planner": ("SingleArmPlanner",),
    "multi": ("MultiArmPlanner",),
    "scheduling": (
        "ScheduleConfig",
        "SchedulePlanner",
        "ScheduleAttempt",
        "ScheduleUpdate",
        "ScheduleReport",
    ),
    "skeletons": (
        "SkeletonSearchConfig",
        "SkeletonProducer",
        "FiniteSkeletonProducer",
        "compile_skeleton",
        "check_candidate",
        "scoped_skill",
    ),
    "objective": ("MetricConfig", "ObjectiveValue", "evaluate_objective"),
    "refinement": (
        "RefinementConfig",
        "RefinementSample",
        "RefinementUpdate",
        "RefinementReport",
        "ManipulationRefiner",
        "refine_manipulation",
    ),
    "task": ("ManipulationSearchConfig",),
    "roadmap": ("ModeEvaluator", "ModeValidityMask"),
    "timeline": ("ManipulationTimeline", "evaluate_trace"),
    "validation": ("SkillEdge", "evaluate_skill", "validate_manipulation", "verify_manipulation"),
}


def __getattr__(name):
    for module, names in _LAZY.items():
        if name in names:
            value = getattr(import_module(f"{__name__}.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)


__all__ = [
    "MANIPULATION_CHECKS",
    "ScheduleConfig",
    "SchedulePlanner",
    "ScheduleAttempt",
    "ScheduleUpdate",
    "ScheduleReport",
    "Task",
    "TaskAlternative",
    "TaskChoice",
    "TaskDomain",
    "SkeletonCandidate",
    "SkeletonSearchConfig",
    "SkeletonProducer",
    "FiniteSkeletonProducer",
    "compile_skeleton",
    "check_candidate",
    "scoped_skill",
    "ActionOrder",
    "ContactBody",
    "ContactWindow",
    "GraspAttachment",
    "GeometryPolicy",
    "HeldObject",
    "JointTrace",
    "ManipulationCost",
    "ManipulationMode",
    "ManipulationTimeline",
    "ManipulationPlan",
    "ManipulationProblem",
    "ManipulationResult",
    "ModeTransition",
    "ModeGeometry",
    "ModeEvaluator",
    "ModeValidityMask",
    "ObjectSphere",
    "PlacedObject",
    "RigidObject",
    "ScheduleSkeleton",
    "ScheduledMotion",
    "ScheduledSkill",
    "Skill",
    "SkillOption",
    "SkillEdge",
    "SingleArmPlanner",
    "MultiArmPlanner",
    "ManipulationSearchConfig",
    "MetricConfig",
    "ObjectiveValue",
    "evaluate_objective",
    "RefinementConfig",
    "RefinementSample",
    "RefinementUpdate",
    "RefinementReport",
    "ManipulationRefiner",
    "refine_manipulation",
    "evaluate_skill",
    "evaluate_trace",
    "pose_errors",
    "validate_manipulation",
    "verify_manipulation",
]
