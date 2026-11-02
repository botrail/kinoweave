"""Whole-plan batch geometry for shortcut proposals, separate from CPU proof."""

from __future__ import annotations

from types import SimpleNamespace

from .._contracts import ContractError
from ..search.space import SearchLimit
from .conflicts import ConflictBuilder
from .roadmap import ModeEvaluator
from .task import ManipulationSearchConfig
from .timeline import ManipulationTimeline
from .validation import _rates, _samples


class _Clock:
    def __init__(self, timeline):
        self.timeline = timeline
        self.events = timeline.events

    def mode(self, time, side):
        return self.timeline.mode(time, side=side)

    def configurations(self, time, side):
        return self.timeline.configurations(time, side=side)

    def contacts(self, time, side):
        return self.timeline.contacts(time, side=side)


class RefinementGeometry:
    """Bounded batches, using actual requested FK backend and owned cache rows."""

    def __init__(self, problem, config, policy, check, **evaluator_options):
        self.problem, self.config, self.check = problem, config, check
        self.evaluator = ModeEvaluator(
            problem, policy=policy, check_budget=check, **evaluator_options
        )
        self.info = self.evaluator.info
        self.stats = {
            "cpu_pair_batches": 0,
            "gpu_pair_batches": 0,
            "max_pair_tile_bytes": 0,
            "peak_batch_geometry_bytes": 0,
            "geometry_candidates": 0,
        }
        space = SimpleNamespace(
            problem=problem,
            evaluator=self.evaluator,
            config=ManipulationSearchConfig(
                pair_batch_size=config.pair_batch_size,
                sphere_tile=config.sphere_tile,
                max_query_pairs=config.pair_batch_size,
                max_pair_bytes=config.max_pair_bytes,
                max_validation_samples=config.max_validation_samples,
            ),
            stats=self.stats,
            check=lambda stage: check(),
        )
        self.builder = ConflictBuilder(space)

    def check_many(self, results):
        self.evaluator._check_open()
        timelines, points, indices = [], [], []
        flags = [False] * len(results)
        count = 0
        spheres = sum(
            len(m.collision_spheres)
            for r in self.problem.scene.robots
            for m in self.problem.scene.models
            if m.name == r.model
        ) + sum(len(o.spheres) for o in self.problem.objects)
        links = sum(
            len(m.links)
            for r in self.problem.scene.robots
            for m in self.problem.scene.models
            if m.name == r.model
        )
        for i, result in enumerate(results):
            self.check()
            try:
                timeline = ManipulationTimeline(self.problem, result)
                if _rates(timeline):
                    continue
                boundaries = set(timeline.boundaries)
                times = tuple(_samples(timeline, self.config.max_validation_samples))
                samples = tuple(
                    (t, side)
                    for t in times
                    for side in (("before", "after") if t in boundaries else ("after",))
                )
            except ContractError:
                continue
            estimated = len(samples) * (16 * spheres + 28 * (links + len(self.problem.objects)))
            count += estimated
            if count > self.config.max_geometry_bytes:
                raise SearchLimit("geometry_limit", "refinement")
            timelines.append(timeline)
            points.append(samples)
            indices.append(i)
        self.stats["peak_batch_geometry_bytes"] = max(
            self.stats["peak_batch_geometry_bytes"], count
        )
        owned = tuple(o.name for o in self.problem.objects)
        state = SimpleNamespace(mode=self.problem.initial_mode)
        build = (
            self.builder._cpu_payload
            if self.evaluator._torch is None
            else self.builder._gpu_payload
        )
        payloads = []
        static = (
            self.builder._cpu_static if self.evaluator._torch is None else self.builder._gpu_static
        )
        for timeline, samples in zip(timelines, points, strict=True):
            self.check()
            payload = build(state, _Clock(timeline), owned, samples)
            payload.valid = payload.valid and static(payload)
            payloads.append(payload)
        pair_flags = self.builder._pairs([(p, p) for p in payloads])
        for i, p, flag in zip(indices, payloads, pair_flags, strict=True):
            flags[i] = p.valid and flag
        self.stats["geometry_candidates"] += len(payloads)
        return tuple(flags)

    def close(self):
        self.evaluator.close()
