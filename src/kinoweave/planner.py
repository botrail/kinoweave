"""Reusable PRM preparation, GPU-backed coordination, and independent verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ._contracts import ContractError, value_as
from .backends._description import UnsupportedModelError
from .backends.curobo import BackendUnavailableError
from .collision import (
    CoordinationEvaluator,
    GeometryCache,
    GpuCollisionEvaluator,
    GpuTiling,
    ReferenceMotionEvaluator,
)
from .io import document_digest
from .models import BackendInfo, Diagnostic, PlanningProblem, PlanProvenance, ValidationReport
from .results import PlanResult
from .roadmap import PRMConfig, Roadmap, build_prm
from .search.joint_astar import joint_astar_search
from .search.lacam import lacam_search
from .search.prioritized import prioritized_search
from .search.reference import JointGraph
from .search.space import Budget, CoordinationSpace, SearchConfig, SearchLimit, SearchOutcome
from .trajectory import RobotTrajectory, TrajectoryBundle
from .validation import validate_trajectory


class Planner:
    """Plan all robot tracks on one clock, using an explicitly selected backend.

    Generated roadmaps and the evaluator can be reused across calls. Supplied
    graphs are bound to the problem and checked during search. The planner is
    not thread-safe; close it with a context manager. External evaluators and
    geometry caches remain owned by their caller.
    """

    def __init__(
        self,
        problem: PlanningProblem,
        *,
        search: SearchConfig | None = None,
        prm: PRMConfig | None = None,
        tiling: GpuTiling | None = None,
        cache: GeometryCache | None = None,
        evaluator: CoordinationEvaluator | None = None,
    ):
        self._problem = value_as(problem, PlanningProblem, "problem")
        self._search = value_as(
            SearchConfig() if search is None else search, SearchConfig, "search"
        )
        self._prm = value_as(PRMConfig() if prm is None else prm, PRMConfig, "prm")
        self._tiling, self._cache = tiling, cache
        self._evaluator = evaluator
        self._owns_evaluator = evaluator is None
        self._roadmaps = ()
        self._last_graphs = ()
        self._closed = False

    @property
    def problem(self):
        return self._problem

    @property
    def roadmaps(self) -> tuple[Roadmap, ...]:
        return self._roadmaps

    @property
    def last_graphs(self) -> tuple[JointGraph, ...]:
        return self._last_graphs

    def _get_evaluator(self):
        if self._evaluator is None:
            config = self.problem.config
            if config.backend == "curobo":
                self._evaluator = GpuCollisionEvaluator(
                    self.problem, tiling=self._tiling, cache=self._cache
                )
            elif config.backend == "reference" and config.device == "cpu":
                self._evaluator = ReferenceMotionEvaluator(self.problem)
            else:
                raise BackendUnavailableError(
                    f"Unsupported backend/device {config.backend!r}/{config.device!r}"
                )
        evaluator = self._evaluator
        if evaluator.problem != self.problem:
            raise ContractError("evaluator.problem", "does not match the requested problem")
        info = value_as(evaluator.info, BackendInfo, "evaluator.info")
        if info.name != self.problem.config.backend or info.device != self.problem.config.device:
            raise ContractError("evaluator.info", "does not match the requested backend/device")
        if not all(
            callable(getattr(evaluator, name, None)) for name in ("motion_validity", "interference")
        ):
            raise ContractError("evaluator", "motion and pair evaluation are required")
        return evaluator

    def plan(
        self,
        graphs: tuple[JointGraph | Roadmap, ...] | None = None,
        *,
        search: SearchConfig | None = None,
    ) -> PlanResult:
        if self._closed:
            raise RuntimeError("planner is closed")
        settings = value_as(self._search if search is None else search, SearchConfig, "search")
        self._last_graphs = ()
        budget = Budget(self.problem.config.time_limit, settings)
        diagnostics = [
            Diagnostic(
                code="search_settings",
                message=json.dumps(asdict(settings), sort_keys=True),
                stage="configuration",
            )
        ]
        attempts = []
        info = None
        timings = {}
        phase, phase_started = "configuration", 0.0

        def mark(name):
            nonlocal phase, phase_started
            now = budget.elapsed()
            timings[phase] = timings.get(phase, 0.0) + max(0.0, now - phase_started)
            phase, phase_started = name, now

        def finish(status, reason, *, trajectory=None, validation=None, error=None):
            mark("result_assembly")
            details = [
                *diagnostics,
                Diagnostic(
                    code="phase_timings",
                    message=json.dumps(
                        {"seconds": timings, "elapsed_seconds": phase_started}, sort_keys=True
                    ),
                    stage=budget.stage,
                ),
                Diagnostic(
                    code="search_attempts",
                    message=json.dumps(attempts, sort_keys=True),
                    stage="search",
                ),
                Diagnostic(
                    code="search_stats",
                    message=json.dumps(
                        {**budget.stats, "elapsed_seconds": budget.elapsed()}, sort_keys=True
                    ),
                    stage=budget.stage,
                ),
            ]
            if error is not None:
                details.append(
                    Diagnostic(
                        code=status,
                        message=str(error),
                        stage=budget.stage,
                        path=getattr(error, "path", None),
                    )
                )
            details.append(Diagnostic(code="termination", message=reason, stage=budget.stage))
            result = PlanResult(
                status=status,
                trajectory=trajectory,
                validation=validation or ValidationReport(),
                provenance=PlanProvenance(
                    problem_digest=document_digest(self.problem),
                    config=self.problem.config,
                    backend=info,
                ),
                diagnostics=tuple(details),
                termination_reason=reason,
            )
            if status == "success":
                result.check_problem(self.problem)
            return result

        try:
            mark("backend_preparation")
            budget.check("backend_preparation")
            evaluator = self._get_evaluator()
            info = evaluator.info
            mark("roadmap_preparation")
            budget.check("roadmap_preparation")
            if graphs is None:
                if not self._roadmaps:
                    built = []
                    for robot in self.problem.robots:
                        built.append(
                            build_prm(
                                evaluator,
                                robot.name,
                                config=self._prm,
                                check_budget=lambda: budget.check("roadmap_preparation"),
                            )
                        )
                    self._roadmaps = tuple(built)
                graphs = self._roadmaps
                diagnostics.append(
                    Diagnostic(
                        code="prm_settings",
                        message=json.dumps(asdict(self._prm), sort_keys=True),
                        stage="roadmap_preparation",
                    )
                )
            else:
                graphs = tuple(graphs)
            mark("graph_binding")
            selected = []
            for graph in graphs:
                if type(graph) is Roadmap:
                    if (
                        graph.problem_digest != document_digest(self.problem)
                        or graph.time_step != self.problem.config.time_step
                    ):
                        raise ContractError(
                            "roadmap", "does not match the requested problem/timing"
                        )
                    graph = graph.graph
                if type(graph) is not JointGraph:
                    raise ContractError("graphs", "expected JointGraph or Roadmap records")
                selected.append(graph)
            budget.check("graph_binding")
            space = CoordinationSpace(evaluator, tuple(selected), settings, budget)
            budget.check("graph_binding")
            self._last_graphs = space.graphs
            if settings.priority_order and set(settings.priority_order) != {
                g.robot for g in space.graphs
            }:
                raise ContractError("priority_order", "must be a permutation of every robot")
            graph_data = [
                {
                    "robot": g.robot,
                    "vertices": len(g.vertices),
                    "edges": len(g.edges),
                    "digest": hashlib.sha256(
                        json.dumps(asdict(g), sort_keys=True).encode()
                    ).hexdigest(),
                }
                for g in space.graphs
            ]
            diagnostics.append(
                Diagnostic(
                    code="graphs",
                    message=json.dumps(graph_data, sort_keys=True),
                    stage="graph_binding",
                )
            )
            mark("endpoint_validation")
            budget.check("start_validation")
            if not space.transition(space.start, space.start):
                return finish("invalid_problem", "invalid_start")
            budget.check("goal_validation")
            if not space.transition(space.goal, space.goal):
                return finish("invalid_problem", "invalid_goal")
            if space.distance(space.start) == float("inf"):
                return finish("no_solution_in_discretization", "disconnected_individual_graph")

            def run(name, fraction=None):
                budget.reserve_phase(fraction)
                before = budget.elapsed()
                record = {"algorithm": name, "started_seconds": before}
                attempts.append(record)
                try:
                    outcome = {
                        "prioritized": prioritized_search,
                        "lacam": lacam_search,
                        "joint_astar": joint_astar_search,
                    }[name](space)
                    record["reason"] = outcome.reason
                    return outcome
                except SearchLimit as error:
                    record["reason"] = error.reason
                    if error.reason != "phase_time_limit":
                        raise
                    return SearchOutcome(None, error.reason)
                finally:
                    record["elapsed_seconds"] = budget.elapsed() - before
                    budget.reserve_phase(None)

            mark("search")
            schedule = []
            algorithm = settings.algorithm
            complement = (
                settings.exact_fallback
                and len(space.graphs) == 2
                and algorithm in ("lacam", "auto")
            )
            if algorithm == "auto":
                schedule.append(("prioritized", 0.25))
            if algorithm in ("lacam", "auto"):
                schedule.append(("lacam", 0.8 if complement else None))
                if complement:
                    schedule.append(("joint_astar", None))
            elif algorithm in ("prioritized", "joint_astar"):
                schedule.append((algorithm, None))
            outcome = SearchOutcome(None, "search_incomplete")
            for name, fraction in schedule:
                budget.check(f"{name}_search")
                outcome = run(name, fraction)
                if outcome.path is not None or outcome.exhausted:
                    break
            if outcome.path is None:
                return finish(
                    "no_solution_in_discretization" if outcome.exhausted else "timeout",
                    outcome.reason,
                )
            budget.check("final_validation")
            mark("trajectory_reconstruction")
            path = outcome.path if len(outcome.path) > 1 else (*outcome.path, outcome.path[0])
            trajectory = TrajectoryBundle(
                times=tuple(t * self.problem.config.time_step for t in range(len(path))),
                tracks=tuple(
                    RobotTrajectory(
                        robot=g.robot,
                        joint_names=g.joint_names,
                        positions=tuple(g.vertices[state[i]] for state in path),
                    )
                    for i, g in enumerate(space.graphs)
                ),
            )
            mark("cpu_validation")
            report = validate_trajectory(
                self.problem, trajectory, max_samples=settings.max_validation_samples
            )
            if report.status != "passed":
                return finish("validation_failed", "final_validation", validation=report)
            reason = (
                "time_limit_after_validation"
                if budget.started + budget.elapsed() >= budget.deadline
                else "goal_reached"
            )
            return finish("success", reason, trajectory=trajectory, validation=report)
        except SearchLimit as error:
            return finish("timeout", error.reason, error=error)
        except (BackendUnavailableError, UnsupportedModelError) as error:
            return finish("unsupported", "backend_unsupported", error=error)
        except (ContractError, OverflowError) as error:
            return finish("invalid_problem", "evaluation_failed", error=error)
        except RuntimeError as error:
            return finish("invalid_problem", "backend_execution_failed", error=error)

    def close(self):
        if self._owns_evaluator and self._evaluator is not None:
            close = getattr(self._evaluator, "close", None)
            if close is not None:
                close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("planner is closed")
        return self

    def __exit__(self, *args):
        self.close()
