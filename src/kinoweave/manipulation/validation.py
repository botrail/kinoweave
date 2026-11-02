"""Independent CPU checks for complete plans and single-arm skill edges."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

from .._contracts import ContractError, value_as
from ..io import document_digest
from ..models import Diagnostic, PlanProvenance, ValidationReport
from ..trajectory import sample_times, segment_limit_issues
from .geometry import GeometryPolicy, ModeGeometry, pose_matches
from .models import ManipulationMode
from .results import MANIPULATION_CHECKS, ManipulationResult
from .roadmap import fingerprint
from .timeline import ManipulationTimeline, skill_timeline


def _rates(timeline):
    issues = []
    models = {m.name: m for m in timeline.problem.scene.models}
    for robot in timeline.problem.scene.robots:
        model = models[robot.model]
        previous = {j.name: 0.0 for j in model.joints}
        for activity in timeline.by_robot[robot.name]:
            track = next(t for t in activity.trace.tracks if t.robot == robot.name)
            for i, (first, last) in enumerate(pairwise(track.positions)):
                duration = (
                    (activity.trace.times[i + 1] - activity.trace.times[i])
                    / activity.trace.duration
                    * (activity.end - activity.start)
                )
                if activity.trace.interpolation == "quintic_stop":
                    for issue in segment_limit_issues(
                        model,
                        first,
                        last,
                        duration,
                        joint_tolerance=timeline.problem.scene.config.joint_tolerance,
                    ):
                        issues.append(
                            replace(issue, path=f"{activity.name}/{robot.name}/{i}/{issue.path}")
                        )
                    velocity = {j.name: 0.0 for j in model.joints}
                else:
                    a, b = model.joint_values(first), model.joint_values(last)
                    velocity = {
                        name: value_as((b[name] - a[name]) / duration, float, "velocity")
                        for name in a
                    }
                    for joint in model.joints:
                        if (
                            joint.limits is not None
                            and abs(velocity[joint.name]) > joint.limits.velocity
                        ):
                            issues.append(
                                Diagnostic(
                                    code="velocity_limit",
                                    message=f"{robot.name}/{joint.name}: excessive linear speed",
                                    path=activity.name,
                                )
                            )
                if any(
                    not math.isclose(previous[name], v, rel_tol=1e-12, abs_tol=0)
                    for name, v in velocity.items()
                ):
                    issues.append(
                        Diagnostic(
                            code="linear_velocity_jump",
                            message=f"{robot.name}: velocity jump needs unbounded acceleration",
                            path=activity.name,
                        )
                    )
                previous = velocity
        if any(v != 0 for v in previous.values()):
            issues.append(
                Diagnostic(
                    code="terminal_velocity",
                    message=f"{robot.name}: terminal state is not at rest",
                    path=robot.name,
                )
            )
    return issues


def _report(problem, checks, issues):
    config = problem.scene.config
    return ValidationReport(
        status="failed" if issues else "passed",
        level="sampled",
        checks=tuple(sorted(checks)),
        sample_time_step=config.validation_time_step,
        collision_margin=config.collision_margin,
        collision_tolerance=config.collision_tolerance,
        issues=tuple(issues),
    )


def _samples(timeline, max_samples):
    step = timeline.problem.scene.config.validation_time_step
    count = 1
    yield 0.0
    for first, last in pairwise(timeline.boundaries):
        offsets = sample_times(last - first, step, max_samples=max_samples - count + 1)
        count += len(offsets) - 1
        for i, offset in enumerate(offsets[1:], 1):
            time = last if i == len(offsets) - 1 else first + offset
            if not first < time <= last:
                raise ContractError("sampling", "sample timestamp is not representable")
            yield time


def _validate_timeline(timeline, geometry, max_samples, *, complete):
    problem = timeline.problem
    checks = {"structure", "joint_limits", "skill_traces", "rest_to_rest"}
    if complete:
        checks.update(("endpoints", "precedence", "object_goals"))
    try:
        max_samples = value_as(max_samples, int, "max_samples")
        if max_samples < 2:
            raise ContractError("max_samples", "at least two samples required")
        issues = _rates(timeline)
        if issues:
            return _report(problem, checks, issues)
        checks.update(
            (
                "collisions",
                "contacts",
                "mode_transitions",
                "object_support",
                "grasp_closure",
                "transition_continuity",
            )
        )
        boundaries = set(timeline.boundaries)
        for time in _samples(timeline, max_samples):
            for side in ("before", "after") if time in boundaries else ("after",):
                frame = geometry.snapshot(
                    timeline.mode(time, side=side),
                    timeline.configurations(time, side=side),
                    contacts=timeline.contacts(time, side=side),
                )
                if frame.issues:
                    return _report(
                        problem,
                        checks,
                        tuple(
                            replace(issue, message=f"t={time:.12g} {side}: {issue.message}")
                            for issue in frame.issues[:32]
                        ),
                    )
        # Check every individual effect, including intermediate states when
        # several effects share a timestamp. Permissions cannot hide teleportation.
        for time in sorted({e[0] for e in timeline.events}):
            mode = timeline.mode(time, side="before")
            configurations = timeline.configurations(time, side="after")
            permissions = tuple(
                dict.fromkeys(
                    (
                        *timeline.contacts(time, side="before"),
                        *timeline.contacts(time, side="after"),
                    )
                )
            )
            for at, _rank, _index, change in timeline.events:
                if at != time:
                    continue
                before = geometry.snapshot(mode, configurations, contacts=permissions)
                states = {s.object: s for s in mode.states}
                states[change.after.object] = change.after
                mode = replace(mode, states=tuple(states.values()))
                after = geometry.snapshot(mode, configurations, contacts=permissions)
                if before.issues or after.issues:
                    return _report(problem, checks, (*before.issues, *after.issues)[:32])
                a = next(o for o in before.objects if o.name == change.before.object)
                b = next(o for o in after.objects if o.name == change.after.object)
                if not pose_matches(a.pose, b.pose, geometry.policy):
                    return _report(
                        problem,
                        checks,
                        (
                            Diagnostic(
                                code="object_teleport",
                                message=(
                                    f"{change.after.object}: effect at {time:.12g} "
                                    "changes world pose discontinuously"
                                ),
                                path=change.after.object,
                            ),
                        ),
                    )
    except (ContractError, OverflowError) as error:
        return _report(
            problem, checks, (Diagnostic(code="manipulation_evaluation", message=str(error)),)
        )
    if complete:
        checks.update(MANIPULATION_CHECKS)
    return _report(problem, checks, ())


def validate_manipulation(problem, result, *, policy=None, max_samples=100_000):
    """Reconstruct and check a candidate independently, ignoring supplied evidence."""
    try:
        timeline = ManipulationTimeline(problem, result)
        geometry = ModeGeometry(problem, policy=policy)
    except ContractError as error:
        return _report(
            problem,
            {"structure"},
            (Diagnostic(code="manipulation_structure", message=str(error), path=error.path),),
        )
    return _validate_timeline(timeline, geometry, max_samples, complete=True)


def verify_manipulation(problem, result, *, policy=None, max_samples=100_000):
    """Return a fresh CPU-verified result; this function is not a GPU planner."""
    policy = policy or GeometryPolicy()
    report = validate_manipulation(problem, result, policy=policy, max_samples=max_samples)
    geometry = ModeGeometry(problem, policy=policy)
    diagnostic = Diagnostic(
        code="manipulation_geometry_policy",
        message=json.dumps(asdict(policy), sort_keys=True),
        stage="cpu_verification",
    )
    provenance = PlanProvenance(
        problem_digest=document_digest(problem), config=problem.scene.config, backend=geometry.info
    )
    return ManipulationResult(
        status="success" if report.status == "passed" else "validation_failed",
        plan=result.plan if report.status == "passed" else None,
        provenance=provenance,
        validation=report,
        diagnostics=(*result.diagnostics, diagnostic),
        termination_reason="cpu_verified"
        if report.status == "passed"
        else "independent_verification_failed",
    )


@dataclass(frozen=True, slots=True)
class SkillEdge:
    problem_digest: str
    skill: str
    option: str
    source_mode_digest: str
    target_mode: ManipulationMode
    duration: float
    validation: ValidationReport


def evaluate_skill(
    problem, mode, skill, option, *, duration=None, policy=None, max_samples=100_000
):
    """Create a checked single-arm skill edge; not a joint task-progress search."""
    if skill not in problem.skeleton.skills or option not in skill.options:
        raise ContractError("skill", "skill and option must belong to the problem")
    if duration is None:
        duration = (
            max(option.trace.duration, skill.minimum_duration)
            if skill.timing == "stretch"
            else option.trace.duration
        )
    else:
        duration = value_as(duration, float, "duration")
    geometry = ModeGeometry(problem, policy=policy)
    geometry.check_mode(mode)
    timeline = skill_timeline(problem, mode, skill, option, duration)
    if skill.timing == "fixed" and not math.isclose(duration, option.trace.duration, rel_tol=1e-9):
        raise ContractError("skill.duration", "fixed timing cannot change")
    if duration < option.trace.duration:
        raise ContractError("skill.duration", "cannot speed up a source trace")
    report = _validate_timeline(timeline, geometry, max_samples, complete=False)
    return SkillEdge(
        document_digest(problem),
        skill.name,
        option.name,
        fingerprint(mode),
        timeline.mode(duration),
        duration,
        report,
    )
