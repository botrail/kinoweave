"""Bounded task-skeleton retry with shared geometry and a verified incumbent."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Literal

from .._contracts import Contract, ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..backends.curobo import BackendUnavailableError
from ..io import document_digest
from ..models import BackendInfo
from ..roadmap import PRMConfig
from ..search.space import Budget, SearchConfig, SearchLimit
from .geometry import GeometryPolicy
from .models import ManipulationProblem
from .multi import MultiArmPlanner
from .objective import MetricConfig, ObjectiveValue, _measure
from .refinement import ManipulationRefiner, RefinementConfig
from .results import ManipulationResult
from .roadmap import ModeEvaluator
from .skeletons import FiniteSkeletonProducer, SkeletonSearchConfig, check_candidate
from .task import ManipulationSearchConfig
from .tasks import TaskChoice, TaskDomain
from .validation import verify_manipulation


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleConfig(Contract):
    time_limit: float = 60.0
    candidate_time_limit: float = 10.0
    max_attempts: int = 128
    stop_after_first: bool = False
    relative_improvement: float = 1e-9

    def _validate(self):
        self._require(self.time_limit >= 0, "time_limit", "must be non-negative")
        self._require(self.candidate_time_limit > 0, "candidate_time_limit", "must be positive")
        self._require(self.max_attempts > 0, "max_attempts", "must be positive")
        self._require(
            self.relative_improvement >= 0, "relative_improvement", "must be non-negative"
        )


@dataclass(frozen=True, slots=True)
class ScheduleAttempt:
    index: int
    candidate_id: str | None
    choices: tuple[TaskChoice, ...]
    order: tuple[str, ...]
    source_problem_digest: str | None
    effective_problem_digest: str | None
    status: str
    termination_reason: str
    elapsed: float
    objective: ObjectiveValue | None
    refinement_status: str | None
    statistics: dict


@dataclass(frozen=True, slots=True)
class ScheduleUpdate:
    attempt: int
    elapsed: float
    candidate_id: str
    problem: ManipulationProblem
    result: ManipulationResult
    objective: ObjectiveValue


@dataclass(frozen=True, slots=True)
class ScheduleReport:
    status: Literal["completed", "budget_exhausted", "cancelled", "unsupported", "invalid_domain"]
    termination_reason: str
    domain_digest: str
    problem: ManipulationProblem | None
    result: ManipulationResult | None
    objective: ObjectiveValue | None
    attempts: tuple[ScheduleAttempt, ...]
    history: tuple[ScheduleUpdate, ...]
    backend: BackendInfo | None
    config: ScheduleConfig
    skeleton_config: SkeletonSearchConfig
    search_config: ManipulationSearchConfig
    metric_config: MetricConfig
    refinement_config: RefinementConfig | None
    prm: PRMConfig
    policy: GeometryPolicy
    statistics: dict


class SchedulePlanner:
    """Choose declared assignments, parameters and orders, then plan each DAG.

    Every alternative is a concrete skill fragment, not an implicit grasp/IK
    solver. A producer must yield catalog-bound SkeletonCandidates and receive
    feedback after each attempt. Limits are cooperative between batches;
    native calls, independent CPU verification and objective measurement are
    atomic. The returned problem/result pair identifies the chosen skeleton.
    """

    def __init__(
        self,
        domain,
        *,
        config=None,
        skeleton_config=None,
        search_config=None,
        metric_config=None,
        refinement_config=None,
        prm=None,
        policy=None,
        **evaluator_options,
    ):
        self.domain = value_as(domain, TaskDomain, "domain")
        self.config = value_as(config or ScheduleConfig(), ScheduleConfig, "schedule")
        self.skeleton_config = value_as(
            skeleton_config or SkeletonSearchConfig(), SkeletonSearchConfig, "skeleton_search"
        )
        self.search_config = value_as(
            search_config or ManipulationSearchConfig(), ManipulationSearchConfig, "search"
        )
        self.metric_config = value_as(metric_config or MetricConfig(), MetricConfig, "metric")
        self.refinement_config = value_as(refinement_config, RefinementConfig | None, "refinement")
        self.prm = value_as(prm or PRMConfig(), PRMConfig, "prm")
        self.policy = value_as(policy or GeometryPolicy(), GeometryPolicy, "policy")
        self.evaluator_options = evaluator_options
        self.best_problem = self.best_result = self.best_objective = None
        self.roadmaps = ()

    def plan(self, producer=None, *, graphs=None, on_improvement=None):
        """Exhaust the bounded stream or stop, preserving a verified incumbent.

        Exhaustion is restricted to this catalog and sampled graph model. It
        is neither continuous infeasibility nor a global optimality certificate.
        Custom producers are synchronous and must cooperate with cancellation;
        their calls cannot be preempted. Unexpected producer/callback exceptions
        propagate after resource cleanup; live best_* attributes remain usable.
        """
        budget = Budget(self.config.time_limit, SearchConfig())
        self.best_problem = self.best_result = self.best_objective = None
        self.roadmaps = ()
        producer = (
            producer
            if producer is not None
            else FiniteSkeletonProducer(self.domain, config=self.skeleton_config)
        )
        if not callable(getattr(producer, "feedback", None)):
            raise ContractError("producer", "must implement feedback(attempt)")
        evaluator, backend = None, None
        attempts, history, seen = [], [], set()
        statistics = {"roadmap_builds": 0, "planned": 0, "duplicates": 0, "rejected": 0}
        statistics["timing_seconds"] = {}
        status, reason = "completed", "producer_exhausted"
        cancelled = False

        def check():
            budget.check("skeleton_scheduling")

        def timed(stage, function, *args, **kwargs):
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timers = statistics["timing_seconds"]
                timers[stage] = timers.get(stage, 0.0) + time.perf_counter() - began

        def adopt(index, candidate, problem, result, score):
            nonlocal cancelled
            old = self.best_objective
            if old is not None and score.total >= old.total - (
                self.config.relative_improvement * abs(old.total)
            ):
                return
            self.best_problem, self.best_result, self.best_objective = problem, result, score
            update = ScheduleUpdate(
                index, budget.elapsed(), candidate.identity, problem, result, score
            )
            history.append(update)
            if on_improvement is not None and timed("callbacks", on_improvement, update) is False:
                cancelled = True

        try:
            check()
            iterator = (
                producer.candidates(check_budget=check)
                if isinstance(producer, FiniteSkeletonProducer)
                else iter(producer)
            )
            while True:
                check()
                if len(attempts) >= self.config.max_attempts:
                    raise SearchLimit("attempt_limit", "skeleton_scheduling")
                try:
                    supplied = timed("generation", next, iterator)
                except StopIteration:
                    break
                check()
                index = len(attempts) + 1
                began = budget.elapsed()
                candidate = problem = score = None
                planner_stats = {}
                ref_status = None
                attempt_status, attempt_reason = "invalid_candidate", "catalog_binding_failed"
                try:
                    candidate = check_candidate(self.domain, supplied)
                    if candidate.identity in seen:
                        statistics["duplicates"] += 1
                        attempt_status, attempt_reason = "duplicate", "already_attempted"
                    else:
                        seen.add(candidate.identity)
                        seconds = min(
                            self.config.candidate_time_limit,
                            budget.deadline - budget.started - budget.elapsed(),
                        )
                        check()
                        problem = replace(
                            candidate.problem,
                            scene=replace(
                                candidate.problem.scene,
                                config=replace(candidate.problem.scene.config, time_limit=seconds),
                            ),
                        )
                        planner = MultiArmPlanner(
                            problem,
                            prm=self.prm,
                            search_config=self.search_config,
                            policy=self.policy,
                        )
                        if evaluator is None:
                            evaluator = timed(
                                "backend_initialization",
                                ModeEvaluator,
                                problem,
                                policy=self.policy,
                                check_budget=check,
                                **self.evaluator_options,
                            )
                            backend = evaluator.info
                            if graphs is None:
                                graphs = timed(
                                    "roadmaps",
                                    planner._graphs,
                                    evaluator,
                                    budget,
                                    self.domain.anchors(),
                                )
                                self.roadmaps = planner.roadmaps
                                statistics["roadmap_builds"] += 1
                        else:
                            evaluator.rebind(problem)
                        result = timed(
                            "fixed_skeleton",
                            planner.plan,
                            graphs,
                            evaluator=evaluator,
                            check_budget=check,
                        )
                        statistics["planned"] += 1
                        planner_stats = planner.statistics
                        attempt_status, attempt_reason = result.status, result.termination_reason
                        if result.status == "success":
                            checked = timed(
                                "acceptance_cpu_verification",
                                verify_manipulation,
                                problem,
                                result,
                                policy=self.policy,
                                max_samples=self.search_config.max_validation_samples,
                            )
                            attempt_status, attempt_reason = (
                                checked.status,
                                checked.termination_reason,
                            )
                            if checked.status == "success":
                                statistics.setdefault("first_verified_seconds", budget.elapsed())
                                result = replace(
                                    checked,
                                    provenance=result.provenance,
                                    diagnostics=(*result.diagnostics, *checked.diagnostics),
                                )
                                # Retain a first feasible plan even if the bounded
                                # metric evaluation subsequently exhausts samples.
                                if self.best_result is None:
                                    self.best_problem, self.best_result = problem, result
                                score = timed(
                                    "objective", _measure, problem, result, self.metric_config
                                )
                                adopt(index, candidate, problem, result, score)
                                if self.refinement_config is not None and not cancelled:
                                    check()
                                    refiner = ManipulationRefiner(
                                        problem,
                                        config=self.refinement_config,
                                        metric_config=self.metric_config,
                                        policy=self.policy,
                                        **self.evaluator_options,
                                    )

                                    def refined(
                                        update, index=index, candidate=candidate, problem=problem
                                    ):
                                        adopt(
                                            index,
                                            candidate,
                                            problem,
                                            update.result,
                                            update.sample.objective,
                                        )
                                        return not cancelled

                                    refinement = timed(
                                        "refinement",
                                        refiner.refine,
                                        result,
                                        on_improvement=refined,
                                        check_budget=check,
                                    )
                                    ref_status = refinement.status
                                    score = refinement.after or score
                        if result.status == "unsupported":
                            status, reason = "unsupported", result.termination_reason
                except (UnsupportedModelError, BackendUnavailableError) as error:
                    attempt_status, attempt_reason = "unsupported", str(error)
                    status, reason = "unsupported", str(error)
                except ContractError as error:
                    attempt_status, attempt_reason = (
                        "invalid_candidate" if candidate is None else "invalid_problem",
                        str(error),
                    )
                    statistics["rejected"] += 1
                    if candidate is not None:
                        status, reason = "invalid_domain", str(error)
                except SearchLimit as error:
                    attempt_status, attempt_reason = "budget_exhausted", error.reason
                    status, reason = "budget_exhausted", error.reason
                except KeyboardInterrupt:
                    attempt_status, attempt_reason = "cancelled", "interrupted"
                    raise
                finally:
                    attempt = ScheduleAttempt(
                        index,
                        candidate.identity if candidate else None,
                        candidate.choices if candidate else (),
                        candidate.order if candidate else (),
                        document_digest(candidate.problem) if candidate else None,
                        document_digest(problem) if problem else None,
                        attempt_status,
                        attempt_reason,
                        budget.elapsed() - began,
                        score,
                        ref_status,
                        planner_stats,
                    )
                    attempts.append(attempt)
                    timed("feedback", producer.feedback, attempt)
                if cancelled:
                    status, reason = "cancelled", "callback_cancelled"
                    break
                if status != "completed":
                    break
                check()
                if self.config.stop_after_first and self.best_result is not None:
                    reason = "first_verified_solution"
                    break
        except SearchLimit as error:
            status, reason = "budget_exhausted", error.reason
        except (UnsupportedModelError, BackendUnavailableError) as error:
            status, reason = "unsupported", str(error)
        except ContractError as error:
            status, reason = "invalid_domain", str(error)
        except KeyboardInterrupt:
            status, reason = "cancelled", "interrupted"
        finally:
            statistics["elapsed"] = budget.elapsed()
            statistics["producer"] = dict(getattr(producer, "statistics", {}))
            if evaluator is not None:
                statistics["geometry"] = dict(evaluator.cache_stats)
                evaluator.close()
        return ScheduleReport(
            status,
            reason,
            document_digest(self.domain),
            self.best_problem,
            self.best_result,
            self.best_objective,
            tuple(attempts),
            tuple(history),
            backend,
            self.config,
            self.skeleton_config,
            self.search_config,
            self.metric_config,
            self.refinement_config,
            self.prm,
            self.policy,
            statistics,
        )
