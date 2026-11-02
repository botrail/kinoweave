"""Anytime shortcut refinement with monotone, independently verified incumbents."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .._contracts import Contract, ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..backends.curobo import BackendUnavailableError
from ..models import BackendInfo, Diagnostic
from ..search.space import SearchLimit
from .geometry import GeometryPolicy, ModeGeometry
from .models import ManipulationProblem
from .objective import MetricConfig, ObjectiveValue, _measure
from .refinement_geometry import RefinementGeometry
from .results import ManipulationResult
from .roadmap import fingerprint
from .shortcuts import Proposal, apply_proposal, compatible, proposals, trim_tail
from .timeline import ManipulationTimeline
from .validation import verify_manipulation


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinementConfig(Contract):
    time_limit: float = 10.0
    max_iterations: int = 32
    max_candidates: int = 64
    windows_per_robot: int = 16
    batch_size: int = 4
    max_composite_arms: int = 2
    seed: int = 0
    minimum_duration: float = 0.01
    duration_safety: float = 1.000001
    relative_improvement: float = 1e-9
    pair_batch_size: int = 16
    sphere_tile: int = 16
    max_pair_bytes: int = 4 * 1024 * 1024
    max_geometry_bytes: int = 64 * 1024 * 1024
    max_validation_samples: int = 100_000

    def _validate(self):
        self._require(self.time_limit >= 0, "time_limit", "must be non-negative")
        self._require(self.max_iterations >= 0, "max_iterations", "must be non-negative")
        self._require(self.seed >= 0, "seed", "must be non-negative")
        for name in (
            "max_candidates",
            "windows_per_robot",
            "batch_size",
            "pair_batch_size",
            "sphere_tile",
            "max_geometry_bytes",
        ):
            self._require(getattr(self, name) > 0, name, "must be positive")
        self._require(self.max_composite_arms >= 2, "max_composite_arms", "need at least two")
        self._require(self.minimum_duration > 0, "minimum_duration", "must be positive")
        self._require(self.duration_safety >= 1, "duration_safety", "must be at least one")
        self._require(
            self.relative_improvement >= 0, "relative_improvement", "must be non-negative"
        )
        self._require(self.max_pair_bytes >= 128, "max_pair_bytes", "must fit one pair")
        self._require(
            self.max_validation_samples >= 2, "max_validation_samples", "need two samples"
        )


@dataclass(frozen=True, slots=True)
class RefinementSample:
    iteration: int
    elapsed: float
    kind: str
    objective: ObjectiveValue


@dataclass(frozen=True, slots=True)
class RefinementUpdate:
    result: ManipulationResult
    sample: RefinementSample


@dataclass(frozen=True, slots=True)
class RefinementReport:
    status: Literal["completed", "budget_exhausted", "cancelled", "unsupported", "invalid_seed"]
    termination_reason: str
    result: ManipulationResult
    before: ObjectiveValue | None
    after: ObjectiveValue | None
    history: tuple[RefinementSample, ...]
    backend: BackendInfo | None
    config: RefinementConfig
    metric_config: MetricConfig
    statistics: dict


class ManipulationRefiner:
    def __init__(
        self, problem, *, config=None, metric_config=None, policy=None, **evaluator_options
    ):
        self.problem = value_as(problem, ManipulationProblem, "problem")
        self.config = value_as(config or RefinementConfig(), RefinementConfig, "refinement")
        self.metric_config = value_as(
            metric_config or MetricConfig(), MetricConfig, "metric_config"
        )
        self.policy = value_as(policy or GeometryPolicy(), GeometryPolicy, "policy")
        self.evaluator_options = evaluator_options
        self.best_result = None

    def refine(self, result, *, on_improvement=None, check_budget=None):
        """Reverify a seed, then retain only strict objective improvements.

        The refinement clock starts after seed verification/measurement. Callbacks
        see each verified incumbent; returning False stops while retaining it.
        No improvement, limits, cancellation or unavailable CUDA discard a seed.
        Geometry checks are sampled; acceptance always uses independent CPU proof.
        """
        began = time.monotonic()
        cfg = self.config
        self.best_result = verify_manipulation(
            self.problem, result, policy=self.policy, max_samples=cfg.max_validation_samples
        )
        before = current = None
        history = []
        backend = None
        geometry = None
        started = None
        stats = {
            "proposals": 0,
            "scored": 0,
            "geometry_rejected": 0,
            "cpu_verified": 0,
            "cpu_rejected": 0,
            "accepted": 0,
            "iterations": 0,
        }
        reason, status = "no_improvement", "completed"

        def clock():
            return 0.0 if started is None else time.monotonic() - started

        def check():
            if check_budget is not None:
                check_budget()
            if started is not None and clock() >= cfg.time_limit:
                raise SearchLimit("time_limit", "refinement")

        def finish():
            stats["initialization_seconds"] = (
                time.monotonic() if started is None else started
            ) - began
            stats["refinement_seconds"] = clock()
            if geometry is not None:
                stats.update(geometry.stats)
                stats.update(
                    {f"geometry_{k}": v for k, v in geometry.evaluator.cache_stats.items()}
                )
            return RefinementReport(
                status,
                reason,
                self.best_result,
                before,
                current,
                tuple(history),
                backend,
                cfg,
                self.metric_config,
                dict(stats),
            )

        def improving(value, old):
            return value.total < old.total - cfg.relative_improvement * abs(old.total)

        metrics = ModeGeometry(self.problem)
        rng = random.Random(cfg.seed)
        if self.best_result.status != "success":
            status, reason = "invalid_seed", "independent_seed_verification_failed"
            return finish()
        try:
            before = current = _measure(
                self.problem,
                self.best_result,
                self.metric_config,
                geometry=metrics,
                check=check_budget or (lambda: None),
            )
            history.append(RefinementSample(0, 0.0, "seed", current))
            started = time.monotonic()
            if (
                on_improvement is not None
                and on_improvement(RefinementUpdate(self.best_result, history[-1])) is False
            ):
                status, reason = "cancelled", "callback_cancelled"
                return finish()
            if cfg.max_iterations == 0:
                reason = "iteration_limit"
                status = "budget_exhausted"
                return finish()
            if current.total == 0:
                reason = "zero_objective"
                return finish()
            check()
            geometry = RefinementGeometry(
                self.problem, cfg, self.policy, check, **self.evaluator_options
            )
            backend = geometry.info
            for iteration in range(1, cfg.max_iterations + 1):
                check()
                stats["iterations"] = iteration
                base = self.best_result
                timeline = ManipulationTimeline(self.problem, base)
                raw = proposals(self.problem, base, timeline, cfg, rng)
                ranked = []
                seen = {fingerprint(base.plan)}
                stats["proposals"] += len(raw)

                def candidate(proposal, *, base=base, seen=seen, old=current):
                    check()
                    plan = apply_proposal(self.problem, base.plan, proposal)
                    key = fingerprint(plan)
                    if key in seen:
                        return None
                    seen.add(key)
                    value = ManipulationResult(
                        status="candidate", plan=plan, provenance=base.provenance
                    )
                    score = _measure(
                        self.problem, value, self.metric_config, geometry=metrics, check=check
                    )
                    stats["scored"] += 1
                    if not improving(score, old):
                        return None
                    if score.makespan < plan.makespan:
                        value = replace(value, plan=trim_tail(plan, score.makespan))
                        score = _measure(
                            self.problem, value, self.metric_config, geometry=metrics, check=check
                        )
                    return score, proposal, value

                for proposal in raw:
                    try:
                        item = candidate(proposal)
                        if item is not None:
                            ranked.append(item)
                    except ContractError:
                        continue
                ranked.sort(key=lambda item: (item[0].total, item[0].makespan, item[0].completion))
                accepted = None
                for begin in range(0, len(ranked), cfg.batch_size):
                    check()
                    batch = ranked[begin : begin + cfg.batch_size]
                    try:
                        flags = geometry.check_many(tuple(item[2] for item in batch))
                    except SearchLimit as error:
                        if error.reason != "geometry_limit" or len(batch) == 1:
                            raise
                        # A complete batch need not fit when individual candidates do.
                        flags = tuple(geometry.check_many((item[2],))[0] for item in batch)
                    valid = [item for item, flag in zip(batch, flags, strict=True) if flag]
                    stats["geometry_rejected"] += len(batch) - len(valid)
                    # Greedily fuse disjoint promising windows, then recheck the
                    # entire composition. Individual validity is never sufficient.
                    edits = []
                    for _, proposal, _ in valid:
                        trial = tuple(dict.fromkeys((*edits, *proposal.edits)))
                        if (
                            proposal.cut is None
                            and compatible(trial)
                            and len({e.robot for e in trial}) <= cfg.max_composite_arms
                        ):
                            edits = list(trial)
                    if len(edits) > 1:
                        try:
                            combined = candidate(Proposal(tuple(edits), "greedy"))
                            if combined is not None and geometry.check_many((combined[2],))[0]:
                                valid.append(combined)
                        except ContractError:
                            pass
                    valid.sort(
                        key=lambda item: (item[0].total, item[0].makespan, item[0].completion)
                    )
                    for score, proposal, value in valid:
                        check()
                        checked = verify_manipulation(
                            self.problem,
                            value,
                            policy=self.policy,
                            max_samples=cfg.max_validation_samples,
                        )
                        stats["cpu_verified"] += 1
                        if checked.status != "success":
                            stats["cpu_rejected"] += 1
                            continue
                        evidence = Diagnostic(
                            code="anytime_refinement",
                            stage="refinement",
                            message=json.dumps(
                                {
                                    "kind": proposal.kind,
                                    "iteration": iteration,
                                    "before": current.total,
                                    "after": score.total,
                                    "metric_config": asdict(self.metric_config),
                                    "verification": "independent_cpu",
                                },
                                sort_keys=True,
                            ),
                        )
                        self.best_result = replace(
                            checked,
                            provenance=replace(checked.provenance, backend=backend),
                            diagnostics=(
                                *base.diagnostics,
                                evidence,
                                *checked.diagnostics[len(value.diagnostics) :],
                            ),
                        )
                        current = score
                        stats["accepted"] += 1
                        sample = RefinementSample(iteration, clock(), proposal.kind, current)
                        history.append(sample)
                        accepted = True
                        if (
                            on_improvement is not None
                            and on_improvement(RefinementUpdate(self.best_result, sample)) is False
                        ):
                            status, reason = "cancelled", "callback_cancelled"
                            return finish()
                        break
                    if accepted:
                        break
                if not accepted:
                    reason = "no_improvement_in_candidates"
                    return finish()
            status, reason = "budget_exhausted", "iteration_limit"
        except SearchLimit as error:
            status, reason = "budget_exhausted", error.reason
        except (UnsupportedModelError, BackendUnavailableError) as error:
            status, reason = "unsupported", str(error)
        except KeyboardInterrupt:
            status, reason = "cancelled", "interrupted"
        finally:
            # Snapshot statistics before clearing owned CUDA caches.
            report = finish()
            if geometry is not None:
                geometry.close()
        return report


def refine_manipulation(problem, result, **options):
    """Convenience entry point; use the class for callbacks and a live incumbent."""
    return ManipulationRefiner(problem, **options).refine(result)
