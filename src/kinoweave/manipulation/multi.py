"""Fixed-skeleton multi-arm manipulation through eager conflicts and lazy search."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, replace

from .._contracts import ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..backends.curobo import BackendUnavailableError
from ..io import document_digest
from ..models import Diagnostic, PlanProvenance
from ..roadmap import PRMConfig, build_prm
from ..search.space import Budget, SearchConfig, SearchLimit
from ..trajectory import RobotTrajectory
from .geometry import GeometryPolicy
from .models import JointTrace, ManipulationProblem, same_duration
from .results import ManipulationPlan, ManipulationResult, ScheduledMotion, ScheduledSkill
from .roadmap import ModeEvaluator
from .search import search
from .space import TaskSpace
from .task import ManipulationSearchConfig
from .validation import verify_manipulation


class _PlanningBudget(Budget):
    def __init__(self, seconds, config, check_budget):
        super().__init__(seconds, config)
        self.external = check_budget

    def check(self, stage):
        super().check(stage)
        if self.external is not None:
            self.external()


class MultiArmPlanner:
    """First verified fixed-skeleton solution; no cost or skeleton optimization.

    Motion edges take one shared tick. Skills retain their complete source
    clocks across ticks and reserve every participant until completion. Fixed
    nonintegral durations use an explicit post-skill hold to the tick boundary;
    stretch skills may uniformly slow to that boundary. No running skill can
    be interrupted. Every result is independently checked by the CPU verifier.
    """

    def __init__(self, problem, *, prm=None, search_config=None, policy=None, **evaluator_options):
        self.problem = value_as(problem, ManipulationProblem, "problem")
        self.prm = value_as(prm or PRMConfig(), PRMConfig, "prm")
        self.config = value_as(
            search_config or ManipulationSearchConfig(), ManipulationSearchConfig, "search_config"
        )
        self.policy = value_as(policy or GeometryPolicy(), GeometryPolicy, "policy")
        self.evaluator_options = evaluator_options
        self.roadmaps = ()
        self.statistics = {}

    def _graphs(self, evaluator, budget, samples):
        if not isinstance(samples, dict) or not set(samples) <= {
            r.name for r in self.problem.scene.robots
        }:
            raise ContractError(
                "samples", "expected a mapping of known robot names to extra anchors"
            )
        maps = []
        for i, robot in enumerate(self.problem.scene.robots):
            budget.check("roadmap_construction")
            model = evaluator.cpu.models[robot.model]
            anchors = list(
                value_as(samples.get(robot.name, ()), tuple[tuple[float, ...], ...], "samples")
            )
            for skill in self.problem.skeleton.skills:
                for option in skill.options:
                    for track in option.trace.tracks:
                        if track.robot == robot.name:
                            anchors.extend((track.positions[0], track.positions[-1]))
            rng = random.Random(self.problem.scene.config.seed + i)
            for _ in range(self.prm.num_samples):
                row = []
                for name in model.joint_order:
                    joint = next(j for j in model.joints if j.name == name)
                    if joint.limits is None or joint.limits.lower is None:
                        raise UnsupportedModelError("joints", "PRM needs bounded positions")
                    row.append(rng.uniform(joint.limits.lower, joint.limits.upper))
                anchors.append(tuple(row))
            maps.append(
                build_prm(
                    evaluator,
                    robot.name,
                    config=self.prm,
                    samples=tuple(dict.fromkeys(anchors)),
                    check_budget=lambda: budget.check("roadmap_construction"),
                )
            )
        self.roadmaps = tuple(maps)
        return tuple(m.graph for m in maps)

    def _plan(self, space, path):
        motions, skills = [], []

        def motion(robot, start, end, first, last):
            if not start < end:
                raise ContractError("schedule", "unrepresentable motion interval")
            name = space.names[robot]
            trace = JointTrace(
                times=(0, end - start),
                interpolation="quintic_stop",
                tracks=(
                    RobotTrajectory(
                        robot=name,
                        joint_names=space.models[name].joint_order,
                        positions=(first, last),
                    ),
                ),
            )
            motions.append(
                ScheduledMotion(
                    start=start, end=end, trace=trace, kind="wait" if first == last else "move"
                )
            )

        for step, (state, choices) in enumerate(path):
            start, end = step * space.dt, (step + 1) * space.dt
            if not math.isfinite(end) or not start < end:
                raise ContractError("schedule", "unrepresentable tick")
            for action in dict.fromkeys(choices):
                if action.kind != "skill":
                    for r, target in zip(action.robots, action.targets, strict=True):
                        motion(
                            r,
                            start,
                            end,
                            space.graphs[r].vertices[state.vertices[r]],
                            space.graphs[r].vertices[target],
                        )
                elif action.phase == 0:
                    skill = space.skills[action.skill]
                    option = skill.options[action.option]
                    duration, ticks = space.clocks[action.skill, action.option]
                    boundary = (step + ticks) * space.dt
                    actual = start + duration
                    if same_duration(boundary - start, duration):
                        actual = boundary
                    if not start < actual <= boundary:
                        raise ContractError("schedule", "unrepresentable skill clock")
                    skills.append(
                        ScheduledSkill(
                            skill=skill.name, option=option.name, start=start, end=actual
                        )
                    )
                    if actual < boundary:
                        for track in option.trace.tracks:
                            motion(
                                space.names.index(track.robot),
                                actual,
                                boundary,
                                track.positions[-1],
                                track.positions[-1],
                            )
        final = space.start
        for state, choices in path[-1:]:
            prepared = space.prepare(state)
            indices = tuple(prepared.table.actions.index(a) for a in choices)
            final = space.advance(state, prepared, indices)
        return ManipulationPlan(
            makespan=len(path) * space.dt,
            motions=tuple(motions),
            skills=tuple(skills),
            final_mode=final.mode,
        )

    def plan(self, graphs=None, *, samples=None, evaluator=None, check_budget=None):
        """Build/reuse supplied graphs, search once, and independently verify.

        Limits are cooperative between batches; they do not preempt native
        calls or final verification. Exhaustion refers to finite candidates
        under this sampled, tick-aligned model, never continuous free space.
        """
        budget = _PlanningBudget(
            self.problem.scene.config.time_limit,
            SearchConfig(
                max_states=self.config.max_states, max_constraints=self.config.max_constraints
            ),
            check_budget,
        )
        owned = evaluator is None
        old_check = None
        info = None
        self.roadmaps = ()

        def provenance():
            return PlanProvenance(
                problem_digest=document_digest(self.problem),
                config=self.problem.scene.config,
                backend=info,
            )

        def failure(status, reason):
            return ManipulationResult(
                status=status,
                provenance=provenance(),
                termination_reason=reason,
                diagnostics=(Diagnostic(code=status, message=reason, stage=budget.stage),),
            )

        try:
            budget.check("initialization")
            if owned:
                evaluator = ModeEvaluator(
                    self.problem,
                    policy=self.policy,
                    check_budget=lambda: budget.check(budget.stage),
                    **self.evaluator_options,
                )
            else:
                if evaluator.manipulation != self.problem or evaluator.cpu.policy != self.policy:
                    raise ContractError("evaluator", "borrowed context or geometry policy mismatch")
                old_check = evaluator.check_budget
                evaluator.check_budget = lambda: budget.check(budget.stage)
            info = evaluator.info
            if graphs is None:
                graphs = self._graphs(evaluator, budget, {} if samples is None else samples)
            elif samples is not None and (not isinstance(samples, dict) or samples):
                raise ContractError(
                    "samples", "explicit graphs and sample construction are exclusive"
                )
            space = TaskSpace(evaluator, graphs, self.config, budget)
            path = search(space)
            if path is None:
                return failure("no_solution_in_discretization", "exhausted_task_graph")
            budget.check("plan_reconstruction")
            candidate = ManipulationResult(
                status="candidate", provenance=provenance(), plan=self._plan(space, path)
            )
            budget.check("cpu_verification")
            checked = verify_manipulation(
                self.problem,
                candidate,
                policy=self.policy,
                max_samples=self.config.max_validation_samples,
            )
            # Verification is atomic. An outer anytime caller can retain a
            # freshly verified result even when that last check overruns.
            if check_budget is None or checked.status != "success":
                budget.check("cpu_verification")
            return replace(
                checked,
                provenance=provenance(),
                diagnostics=(
                    *checked.diagnostics,
                    Diagnostic(
                        code="multi_arm_construction",
                        stage="planning",
                        message=json.dumps(
                            {
                                "algorithm": self.config.algorithm,
                                "conflicts": "eager_state_tables",
                                "verification": "independent_cpu",
                                "search_config": asdict(self.config),
                            },
                            sort_keys=True,
                        ),
                    ),
                ),
            )
        except SearchLimit as error:
            return failure("timeout", error.reason)
        except (UnsupportedModelError, BackendUnavailableError) as error:
            return failure("unsupported", str(error))
        except ContractError as error:
            return failure("invalid_problem", str(error))
        finally:
            self.statistics = {**budget.stats, "elapsed": budget.elapsed()}
            if evaluator is not None:
                self.statistics.update(
                    {f"geometry_{k}": v for k, v in evaluator.cache_stats.items()}
                )
                if owned:
                    evaluator.close()
                elif old_check is not None:
                    evaluator.check_budget = old_check
