"""Bounded CUDA motion geometry and pair interference, with an independent CPU oracle."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from ._contracts import Contract, ContractError, value_as
from .io import document_digest
from .models import BackendInfo, PlanningProblem
from .trajectory import evaluate_segment, sample_times, segment_limit_issues
from .validation import validate_interval

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class Motion(Contract):
    """One quintic-stop interval; duration is the problem's common time_step."""

    start: tuple[float, ...]
    end: tuple[float, ...]

    def _validate(self) -> None:
        self._require(
            bool(self.start) and len(self.start) == len(self.end),
            "end",
            "nonempty matching joint counts required",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuTiling(Contract):
    motion_batch_size: int = 32
    time_batch_size: int = 16
    pair_batch_size: int = 128
    sphere_batch_size: int = 32
    max_pair_bytes: int = 8 * 1024 * 1024
    max_geometry_bytes: int = 32 * 1024 * 1024
    max_table_pairs: int = 1_000_000

    def _validate(self) -> None:
        for name in self.__dataclass_fields__:
            self._require(getattr(self, name) > 0, name, "must be positive")
        self._require(self.max_pair_bytes >= 64, "max_pair_bytes", "at least 64 bytes required")


@dataclass(frozen=True, slots=True, kw_only=True)
class InterferenceTable(Contract):
    problem_digest: str
    first_robot: str
    second_robot: str
    first_motions: tuple[Motion, ...]
    second_motions: tuple[Motion, ...]
    first_valid: tuple[bool, ...]
    second_valid: tuple[bool, ...]
    compatible: tuple[tuple[bool, ...], ...]
    duration: float
    sample_time_step: float
    backend: BackendInfo

    def _validate(self) -> None:
        self._require(
            self.first_robot != self.second_robot, "second_robot", "distinct robots required"
        )
        self._require(
            len(self.first_valid) == len(self.first_motions) == len(self.compatible),
            "compatible",
            "row count mismatch",
        )
        self._require(
            len(self.second_valid) == len(self.second_motions),
            "second_valid",
            "column count mismatch",
        )
        self._require(
            all(len(row) == len(self.second_motions) for row in self.compatible),
            "compatible",
            "column count mismatch",
        )
        self._require(
            self.duration > 0 and self.sample_time_step > 0, "duration", "positive times required"
        )
        for i, row in enumerate(self.compatible):
            self._require(
                all(
                    not value or (self.first_valid[i] and self.second_valid[j])
                    for j, value in enumerate(row)
                ),
                "compatible",
                "invalid motions cannot be compatible",
            )


class MotionEvaluator(Protocol):
    problem: PlanningProblem
    info: BackendInfo

    def motion_validity(self, robot: str, motions: tuple[Motion, ...]) -> tuple[bool, ...]: ...


class CoordinationEvaluator(MotionEvaluator, Protocol):
    def interference(
        self,
        first_robot: str,
        first_motions: tuple[Motion, ...],
        second_robot: str,
        second_motions: tuple[Motion, ...],
    ) -> InterferenceTable: ...


def _robot(problem, robot):
    matches = [r for r in problem.robots if r.name == robot]
    if not matches:
        raise ContractError("robot", f"unknown robot {robot!r}")
    instance = matches[0]
    return instance, next(m for m in problem.models if m.name == instance.model)


def _motions(problem, robot, motions):
    _, model = _robot(problem, robot)
    motions = value_as(motions, tuple[Motion, ...], "motions")
    if any(len(m.start) != len(model.joint_order) for m in motions):
        raise ContractError("motions", "joint count must match model order")
    return motions


def _reference_validity(problem, robot, motions):
    instance, _ = _robot(problem, robot)
    single = replace(problem, robots=(instance,))
    results = []
    for motion in _motions(problem, robot, motions):
        report = validate_interval(
            single, {robot: motion.start}, {robot: motion.end}, problem.config.time_step
        )
        if any(i.code == "evaluation_failed" for i in report.issues):
            raise ContractError("motion", str(report.issues))
        results.append(report.status == "passed")
    return tuple(results)


class ReferenceMotionEvaluator:
    """Explicit CPU evaluation for PRM correctness checks, never an automatic fallback."""

    def __init__(self, problem: PlanningProblem):
        if problem.config.backend != "reference" or problem.config.device != "cpu":
            raise ContractError("config", "explicitly select backend='reference', device='cpu'")
        self.problem = problem
        self.info = BackendInfo(name="reference", version="0.1.0.dev0", device="cpu")

    def motion_validity(self, robot: str, motions: tuple[Motion, ...]) -> tuple[bool, ...]:
        return _reference_validity(self.problem, robot, motions)

    def interference(self, first_robot, first_motions, second_robot, second_motions):
        return reference_interference(
            self.problem, first_robot, first_motions, second_robot, second_motions
        )


def reference_interference(
    problem: PlanningProblem, first_robot: str, first_motions, second_robot: str, second_motions
) -> InterferenceTable:
    """Independent scalar CPU oracle for two robots, including individual validity."""
    if first_robot == second_robot:
        raise ContractError("robots", "distinct robots required")
    first = _motions(problem, first_robot, first_motions)
    second = _motions(problem, second_robot, second_motions)
    selected = replace(
        problem, robots=(_robot(problem, first_robot)[0], _robot(problem, second_robot)[0])
    )
    rows = []
    for a in first:
        row = []
        for b in second:
            report = validate_interval(
                selected,
                {first_robot: a.start, second_robot: b.start},
                {first_robot: a.end, second_robot: b.end},
                problem.config.time_step,
            )
            if any(i.code == "evaluation_failed" for i in report.issues):
                raise ContractError("motion", str(report.issues))
            row.append(report.status == "passed")
        rows.append(tuple(row))
    return InterferenceTable(
        problem_digest=document_digest(problem),
        first_robot=first_robot,
        second_robot=second_robot,
        first_motions=first,
        second_motions=second,
        first_valid=_reference_validity(problem, first_robot, first),
        second_valid=_reference_validity(problem, second_robot, second),
        compatible=tuple(rows),
        duration=problem.config.time_step,
        sample_time_step=problem.config.validation_time_step,
        backend=BackendInfo(name="reference", version="0.1.0.dev0", device="cpu"),
    )


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    bytes: int
    entries: int


@dataclass(frozen=True, slots=True)
class _Geometry:
    spheres: torch.Tensor  # [time,spheres,xyzr], owned
    valid: torch.Tensor  # [time], self/static only

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.spheres, self.valid))


class GeometryCache:
    """LRU with an exact tensor-payload byte cap; zero disables storage.

    Share across evaluators to reuse unchanged scenes. Keys include the entire
    problem digest, SDK versions, device, dtype, motion, and sample slice.
    It owns CUDA tensors and is not thread-safe.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self._max_bytes = value_as(max_bytes, int, "max_bytes")
        if self.max_bytes < 0:
            raise ContractError("max_bytes", "must be non-negative")
        self._entries = OrderedDict()
        self._bytes = self._hits = self._misses = self._evictions = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            self._hits, self._misses, self._evictions, self._bytes, len(self._entries)
        )

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def _get(self, key):
        if key not in self._entries:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def _put(self, key, value):
        size = value.nbytes
        if size > self.max_bytes:
            return
        if key in self._entries:
            self._bytes -= self._entries.pop(key).nbytes
        while self._bytes + size > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self._bytes -= old.nbytes
            self._evictions += 1
        self._entries[key] = value
        self._bytes += size


class GpuCollisionEvaluator:
    """CUDA FK and motion/pair checking with bounded tiles and reusable geometry.

    Only reduced boolean validity and compatibility tables return to the host.
    Byte budgets bound tensor payload estimates for this evaluator, not native
    cuRobo scratch, CUDA context, or PyTorch allocator reservations.
    """

    def __init__(
        self,
        problem: PlanningProblem,
        *,
        tiling: GpuTiling | None = None,
        cache: GeometryCache | None = None,
    ):
        from .backends.curobo import CUROBO_COMMIT, CuroboBackend

        self._problem = problem
        self.tiling = tiling or GpuTiling()
        self._owns_cache = cache is None
        self.cache = cache if cache is not None else GeometryCache()
        self._backends = {}
        self._closed = False
        self.max_pair_tile_bytes = 0
        self.max_geometry_tile_bytes = 0
        try:
            for robot in problem.robots:
                self._backends[robot.name] = CuroboBackend(problem, robot.name)
            import torch

            backend = next(iter(self._backends.values()))
            self._torch, self.device, self.info = torch, backend.device, backend.info
            self._times = sample_times(
                problem.config.time_step, problem.config.validation_time_step
            )
            self._blend = tuple(
                evaluate_segment((0,), (1,), problem.config.time_step, t).positions[0]
                for t in self._times
            )
            self.digest = document_digest(problem)
            signature = {
                "problem": self.digest,
                "commit": CUROBO_COMMIT,
                "versions": backend.dependency_versions,
                "device": str(self.device),
                "dtype": "float32",
                "geometry_api": 1,
                "interpolation": "quintic_stop",
                "sample_times": self._times,
            }
            self._key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        except Exception:
            self.close()
            raise

    @property
    def problem(self):
        return self._problem

    def _check_open(self):
        if self._closed:
            raise RuntimeError("evaluator is closed")

    def _geometry(self, robot, motions, begin, end):
        self._check_open()
        backend = self._backends[robot]
        torch = self._torch
        results = [None] * len(motions)
        pending = []
        precision = torch.get_float32_matmul_precision()
        for i, motion in enumerate(motions):
            key = (self._key, precision, robot, motion, begin, end)
            results[i] = self.cache._get(key)
            if results[i] is None:
                pending.append((i, key, motion))
        # Account conservatively for owned FK outputs, interpolation and clones.
        per_motion = (end - begin) * (
            64 * len(backend.sphere_links)
            + 64 * len(backend.model.links)
            + 48 * len(backend.joint_names)
            + 64
        )
        batch = min(self.tiling.motion_batch_size, self.tiling.max_geometry_bytes // per_motion)
        if batch < 1:
            raise ContractError(
                "max_geometry_bytes",
                "one geometry tile does not fit; reduce time_batch_size or increase budget",
            )
        for offset in range(0, len(pending), batch):
            group = pending[offset : offset + batch]
            size = len(group) * per_motion
            self.max_geometry_tile_bytes = max(self.max_geometry_tile_bytes, size)
            start = torch.tensor(
                [item[2].start for item in group], device=self.device, dtype=torch.float64
            )
            finish = torch.tensor(
                [item[2].end for item in group], device=self.device, dtype=torch.float64
            )
            blend = torch.tensor(self._blend[begin:end], device=self.device, dtype=torch.float64)
            positions = start[:, None, :] + (finish - start)[:, None, :] * blend[None, :, None]
            positions = torch.where(
                blend[None, :, None] == 1, finish[:, None, :], positions
            ).float()
            fk, masks = backend.evaluate(positions)
            valid = ~masks.self_collision & ~masks.world_collision
            for local, (index, key, _) in enumerate(group):
                value = _Geometry(fk.spheres[local].clone(), valid[local].clone())
                self.cache._put(key, value)
                results[index] = value
        return results

    def motion_validity(self, robot: str, motions: tuple[Motion, ...]) -> tuple[bool, ...]:
        self._check_open()
        motions = _motions(self.problem, robot, motions)
        _, model = _robot(self.problem, robot)
        torch = self._torch
        output = []
        backend = self._backends[robot]
        per_motion = min(len(self._times), self.tiling.time_batch_size) * (
            64 * len(backend.sphere_links)
            + 64 * len(model.links)
            + 48 * len(model.joint_order)
            + 64
        )
        batch = min(self.tiling.motion_batch_size, self.tiling.max_geometry_bytes // per_motion)
        if batch < 1:
            raise ContractError(
                "max_geometry_bytes",
                "one geometry tile does not fit; reduce time_batch_size or increase budget",
            )
        for offset in range(0, len(motions), batch):
            group = motions[offset : offset + batch]
            analytic = [
                not segment_limit_issues(
                    model,
                    m.start,
                    m.end,
                    self.problem.config.time_step,
                    joint_tolerance=self.problem.config.joint_tolerance,
                )
                for m in group
            ]
            valid = torch.tensor(analytic, device=self.device, dtype=torch.bool)
            for begin in range(0, len(self._times), self.tiling.time_batch_size):
                end = min(begin + self.tiling.time_batch_size, len(self._times))
                geometry = self._geometry(robot, group, begin, end)
                valid &= torch.stack([g.valid for g in geometry]).all(-1)
            output.extend(valid.tolist())
        return tuple(output)

    def _pair_shape(self, count, first_spheres, second_spheres):
        shape = [
            min(count, self.tiling.pair_batch_size),
            min(len(self._times), self.tiling.time_batch_size),
            min(first_spheres, self.tiling.sphere_batch_size),
            min(second_spheres, self.tiling.sphere_batch_size),
        ]
        while 64 * shape[0] * shape[1] * shape[2] * shape[3] > self.tiling.max_pair_bytes:
            index = max(range(4), key=lambda i: shape[i])
            shape[index] = max(1, shape[index] // 2)
        while (
            64 * shape[0] * shape[1] * (first_spheres + second_spheres)
            > self.tiling.max_geometry_bytes
        ):
            index = max((0, 1), key=lambda i: shape[i])
            if shape[index] == 1:
                raise ContractError("max_geometry_bytes", "one pair geometry tile does not fit")
            shape[index] = max(1, shape[index] // 2)
        return shape

    def interference(
        self, first_robot: str, first_motions, second_robot: str, second_motions
    ) -> InterferenceTable:
        self._check_open()
        if first_robot == second_robot:
            raise ContractError("robots", "distinct robots required")
        first = _motions(self.problem, first_robot, first_motions)
        second = _motions(self.problem, second_robot, second_motions)
        total = len(first) * len(second)
        if total > self.tiling.max_table_pairs:
            raise ContractError("max_table_pairs", "requested host table exceeds pair budget")
        first_valid = self.motion_validity(first_robot, first)
        second_valid = self.motion_validity(second_robot, second)
        rows = [[False] * len(second) for _ in first]
        if total:
            p, t, sa, sb = self._pair_shape(
                total,
                len(self._backends[first_robot].sphere_links),
                len(self._backends[second_robot].sphere_links),
            )
            torch = self._torch
            threshold = (
                self.problem.config.collision_margin + self.problem.config.collision_tolerance
            )
            for offset in range(0, total, p):
                indices = [
                    (k // len(second), k % len(second))
                    for k in range(offset, min(offset + p, total))
                    if first_valid[k // len(second)] and second_valid[k % len(second)]
                ]
                if not indices:
                    continue
                conflict = torch.zeros(len(indices), device=self.device, dtype=torch.bool)
                # Only retain the current candidate/time/sphere tiles.
                for begin in range(0, len(self._times), t):
                    end = min(begin + t, len(self._times))
                    a = self._geometry(first_robot, tuple(first[i] for i, _ in indices), begin, end)
                    b = self._geometry(
                        second_robot, tuple(second[j] for _, j in indices), begin, end
                    )
                    for ai in range(0, len(self._backends[first_robot].sphere_links), sa):
                        x = torch.stack([g.spheres[:, ai : ai + sa] for g in a])
                        for bi in range(0, len(self._backends[second_robot].sphere_links), sb):
                            y = torch.stack([g.spheres[:, bi : bi + sb] for g in b])
                            size = 64 * len(indices) * (end - begin) * x.shape[2] * y.shape[2]
                            self.max_pair_tile_bytes = max(self.max_pair_tile_bytes, size)
                            distance = torch.linalg.vector_norm(
                                x[..., :3].unsqueeze(3) - y[..., :3].unsqueeze(2), dim=-1
                            )
                            clearance = distance - x[..., 3].unsqueeze(3) - y[..., 3].unsqueeze(2)
                            if not torch.isfinite(clearance).all().item():
                                raise ContractError("clearance", "non-finite GPU pair distance")
                            conflict |= (clearance <= threshold).any(dim=(1, 2, 3))
                for (i, j), hit in zip(indices, conflict.tolist(), strict=True):
                    rows[i][j] = not hit
        return InterferenceTable(
            problem_digest=self.digest,
            first_robot=first_robot,
            second_robot=second_robot,
            first_motions=first,
            second_motions=second,
            first_valid=first_valid,
            second_valid=second_valid,
            compatible=tuple(tuple(row) for row in rows),
            duration=self.problem.config.time_step,
            sample_time_step=self.problem.config.validation_time_step,
            backend=self.info,
        )

    def close(self):
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        if self._owns_cache:
            self.cache.clear()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()
