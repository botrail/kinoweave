"""Bounded single-arm construction through mode-filtered PRMs and checked skills."""

from __future__ import annotations

import random
import time
from dataclasses import replace

from .._contracts import ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..backends.curobo import BackendUnavailableError
from ..io import document_digest
from ..models import Diagnostic, PlanProvenance
from ..roadmap import PRMConfig, build_prm
from ..trajectory import RobotTrajectory
from .geometry import GeometryPolicy
from .models import JointTrace, ManipulationProblem
from .results import ManipulationPlan, ManipulationResult, ScheduledMotion, ScheduledSkill
from .roadmap import ModeEvaluator, fingerprint
from .validation import evaluate_skill, verify_manipulation


class _BudgetEnd(Exception):
    pass


class SingleArmPlanner:
    """A fixed-skeleton single-arm constructor, with independent CPU verification.

    Skills supply joint traces; this class plans their free-motion connectors.
    It tries finite options in input order and returns the first verified plan.
    It neither optimizes the declared cost nor searches a multi-arm task space.
    """

    def __init__(self, problem, *, prm=None, policy=None, max_states=10_000, **evaluator_options):
        self.problem = value_as(problem, ManipulationProblem, "problem")
        self.prm = value_as(prm or PRMConfig(), PRMConfig, "prm")
        self.policy = value_as(policy or GeometryPolicy(), GeometryPolicy, "policy")
        self.max_states = value_as(max_states, int, "max_states")
        if self.max_states < 1:
            raise ContractError("max_states", "must be positive")
        self.evaluator_options = evaluator_options
        self.roadmap = None
        self.statistics = {}

    def plan(self, *, samples=()):
        """Build one static PRM, filter it per mode, then verify the full schedule.

        ``samples`` adds explicit anchors to seeded random samples and every
        skill entry/exit. The time limit is cooperative between bounded batches;
        backend initialization and an in-flight native call cannot be preempted.
        """
        problem = self.problem
        began = time.monotonic()
        info, evaluator = None, None
        expanded = 0
        self.roadmap = None
        self.statistics = {}

        def budget():
            if time.monotonic() - began >= problem.scene.config.time_limit:
                raise _BudgetEnd("time_limit")

        def provenance():
            return PlanProvenance(
                problem_digest=document_digest(problem), config=problem.scene.config, backend=info
            )

        def failure(status, reason):
            return ManipulationResult(
                status=status,
                provenance=provenance(),
                termination_reason=reason,
                diagnostics=(Diagnostic(code=status, message=reason),),
            )

        if len(problem.scene.robots) != 1:
            return failure("unsupported", "single-arm construction requires exactly one robot")
        robot = problem.scene.robots[0]
        if any(s.robots != (robot.name,) for s in problem.skeleton.skills):
            return failure("unsupported", "single-arm construction requires single-arm skills")
        model = next(m for m in problem.scene.models if m.name == robot.model)
        skills = {s.name: s for s in problem.skeleton.skills}
        ordered = [skills[name] for name in problem.skeleton.ancestors()]
        try:
            budget()
            evaluator = ModeEvaluator(
                problem, policy=self.policy, check_budget=budget, **self.evaluator_options
            )
            info = evaluator.info
            budget()
            anchors = list(value_as(samples, tuple[tuple[float, ...], ...], "samples"))
            for skill in ordered:
                for option in skill.options:
                    anchors.extend(
                        (option.trace.tracks[0].positions[0], option.trace.tracks[0].positions[-1])
                    )
            rng = random.Random(problem.scene.config.seed)
            for _ in range(self.prm.num_samples):
                row = []
                for name in model.joint_order:
                    joint = next(j for j in model.joints if j.name == name)
                    if joint.limits is None or joint.limits.lower is None:
                        raise UnsupportedModelError("joints", "PRM needs bounded joint positions")
                    row.append(rng.uniform(joint.limits.lower, joint.limits.upper))
                anchors.append(tuple(row))
            self.roadmap = build_prm(
                evaluator,
                robot.name,
                config=self.prm,
                samples=tuple(dict.fromkeys(anchors)),
                check_budget=budget,
            )
            graph = self.roadmap.graph
            vertices = {q: i for i, q in enumerate(graph.vertices)}
            failed, checked_skills = set(), {}
            last_validation = None

            def connector(mode, source, target, start):
                if source == target:
                    return (), start
                mask = evaluator.mask(graph, mode)
                path = mask.path(graph, source, target)
                if path is None:
                    return None
                dt = problem.scene.config.time_step
                end = start + dt * (len(path) - 1)
                if not start < end:
                    raise ContractError("schedule", "connector clock is not representable")
                trace = JointTrace(
                    times=tuple(i * dt for i in range(len(path))),
                    interpolation="quintic_stop",
                    tracks=(
                        RobotTrajectory(
                            robot=robot.name,
                            joint_names=model.joint_order,
                            positions=tuple(graph.vertices[i] for i in path),
                        ),
                    ),
                )
                return (ScheduledMotion(start=start, end=end, trace=trace),), end

            # An explicit stack bounds Python call depth as well as explored states.
            stack = [(0, problem.initial_mode, graph.start, 0.0, (), ())]
            while stack:
                budget()
                if expanded >= self.max_states:
                    raise _BudgetEnd("state_limit")
                index, mode, source, at, motions, scheduled = stack.pop()
                key = (index, fingerprint(mode), source)
                if key in failed:
                    continue
                failed.add(key)
                expanded += 1
                if index == len(ordered):
                    if {s.object: s for s in mode.states} != {
                        s.object: s for s in problem.goal_mode.states
                    }:
                        continue
                    final = connector(mode, source, graph.goal, at)
                    if final is None:
                        continue
                    extra, end = final
                    candidate = ManipulationResult(
                        status="candidate",
                        provenance=provenance(),
                        plan=ManipulationPlan(
                            makespan=end, motions=motions + extra, skills=scheduled, final_mode=mode
                        ),
                    )
                    budget()
                    verified = verify_manipulation(problem, candidate, policy=self.policy)
                    budget()
                    if verified.status == "success":
                        self.statistics.update(expanded_states=expanded)
                        return replace(
                            verified,
                            provenance=provenance(),
                            diagnostics=(
                                *verified.diagnostics,
                                Diagnostic(
                                    code="single_arm_construction",
                                    message="PRM connectors; independent CPU verification",
                                    stage="planning",
                                ),
                            ),
                        )
                    last_validation = verified
                    continue
                skill = ordered[index]
                branches = []
                for option in skill.options:
                    budget()
                    if any(s not in mode.states for s in option.preconditions):
                        continue
                    positions = option.trace.tracks[0].positions
                    entry, exit = positions[0], positions[-1]
                    if entry not in vertices or exit not in vertices:
                        continue
                    connection = connector(mode, source, vertices[entry], at)
                    if connection is None:
                        continue
                    extra, start = connection
                    cache_key = (skill.name, option.name, fingerprint(mode))
                    if cache_key not in checked_skills:
                        checked_skills[cache_key] = evaluate_skill(
                            problem, mode, skill, option, policy=self.policy
                        )
                    edge = checked_skills[cache_key]
                    if edge.validation.status != "passed":
                        continue
                    end = start + edge.duration
                    branches.append(
                        (
                            index + 1,
                            edge.target_mode,
                            vertices[exit],
                            end,
                            motions + extra,
                            scheduled
                            + (
                                ScheduledSkill(
                                    skill=skill.name, option=option.name, start=start, end=end
                                ),
                            ),
                        )
                    )
                stack.extend(reversed(branches))
            if last_validation is not None:
                return replace(last_validation, provenance=provenance())
            return failure(
                "no_solution_in_discretization", "no checked path through supplied PRM/skills"
            )
        except _BudgetEnd as error:
            return failure("timeout", str(error))
        except (UnsupportedModelError, BackendUnavailableError) as error:
            return failure("unsupported", str(error))
        except ContractError as error:
            return failure("invalid_problem", str(error))
        finally:
            self.statistics.update(expanded_states=expanded, elapsed=time.monotonic() - began)
            if evaluator is not None:
                self.statistics.update(evaluator.cache_stats)
                evaluator.close()
