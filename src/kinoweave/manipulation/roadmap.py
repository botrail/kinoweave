"""Mode-specific validity over reusable static PRMs and cached robot geometry."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, replace

from .._contracts import Contract, ContractError, value_as
from ..backends._description import UnsupportedModelError, check_float32
from ..backends.curobo import BackendUnavailableError
from ..collision import Motion
from ..geometry.reference import _compose, _rotate
from ..io import document_digest
from ..models import BackendInfo, Box, Pose
from ..search.reference import JointGraph
from ..trajectory import evaluate_segment, sample_times, segment_limit_issues
from .geometry import GeometryPolicy, ModeGeometry
from .models import HeldObject, PlacedObject


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(asdict(value), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeValidityMask(Contract):
    problem_digest: str
    mode_digest: str
    graph_digest: str
    robot: str
    node_flags: tuple[bool, ...]
    edge_flags: tuple[bool, ...]
    time_step: float
    sample_time_step: float
    policy: GeometryPolicy
    backend: BackendInfo

    def _validate(self):
        self._require(self.time_step > 0, "time_step", "must be positive")
        self._require(self.sample_time_step > 0, "sample_time_step", "must be positive")

    def path(self, graph: JointGraph, start: int, goal: int):
        value_as(graph, JointGraph, "graph")
        if fingerprint(graph) != self.graph_digest or graph.robot != self.robot:
            raise ContractError("mask.graph", "mask belongs to another graph")
        start, goal = value_as(start, int, "start"), value_as(goal, int, "goal")
        if not 0 <= start < len(graph.vertices) or not 0 <= goal < len(graph.vertices):
            raise ContractError("mask.path", "vertex index outside graph")
        if len(self.node_flags) != len(graph.vertices) or len(self.edge_flags) != len(graph.edges):
            raise ContractError("mask", "flag counts do not match graph")
        if not self.node_flags[start] or not self.node_flags[goal]:
            return None
        neighbors = [[] for _ in graph.vertices]
        for (a, b), valid in zip(graph.edges, self.edge_flags, strict=True):
            if valid and self.node_flags[a] and self.node_flags[b]:
                neighbors[a].append(b)
                neighbors[b].append(a)
        parents, queue = {start: None}, deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                result = []
                while current is not None:
                    result.append(current)
                    current = parents[current]
                return tuple(reversed(result))
            for other in sorted(neighbors[current]):
                if other not in parents:
                    parents[other] = current
                    queue.append(other)
        return None


class ModeEvaluator:
    """CPU or explicit CUDA masks; CPU final verification is a separate operation.

    Static FK/self/environment payloads are reused across object modes. Only
    placed objects and objects held exclusively by the selected robot enter
    its mask. Other arms/held objects require later joint conflict checking.
    """

    def __init__(
        self,
        problem,
        *,
        policy=None,
        batch_size=64,
        sphere_tile=32,
        max_pair_bytes=1024 * 1024,
        max_cache_bytes=32 * 1024 * 1024,
        max_masks=8,
        check_budget=None,
    ):
        self.cpu = ModeGeometry(problem, policy=policy)
        self.manipulation = self.cpu.problem
        self.problem = self.manipulation.scene  # Native MotionEvaluator interface for build_prm.
        for name, value in (
            ("batch_size", batch_size),
            ("sphere_tile", sphere_tile),
            ("max_pair_bytes", max_pair_bytes),
            ("max_masks", max_masks),
        ):
            value_as(value, int, name)
            if value <= 0:
                raise ContractError(name, "must be positive")
        max_cache_bytes = value_as(max_cache_bytes, int, "max_cache_bytes")
        if max_pair_bytes < 64 or max_cache_bytes < 0:
            raise ContractError("budgets", "pair budget must be >=64 and cache >=0")
        self.batch_size, self.sphere_tile = batch_size, sphere_tile
        self.max_pair_bytes, self.max_cache_bytes, self.max_masks = (
            max_pair_bytes,
            max_cache_bytes,
            max_masks,
        )
        self._backends, self._cache, self._masks = {}, OrderedDict(), OrderedDict()
        self._closed = False
        self.check_budget = check_budget or (lambda: None)
        self._link_names = {}
        self.cache_hits = self.cache_misses = self.cache_bytes = self.native_calls = 0
        self.max_pair_tile_bytes = 0
        self._torch = None
        self._precision = None
        backend = self.problem.config.backend
        if backend == "reference" and self.problem.config.device == "cpu":
            self.info = self.cpu.info
        elif backend == "curobo" and self.problem.config.device.startswith("cuda"):
            from ..backends.curobo import CuroboBackend

            try:
                for robot in self.problem.robots:
                    self._backends[robot.name] = CuroboBackend(self.problem, robot.name)
                import torch

                self._torch = torch
                first = next(iter(self._backends.values()))
                self.info, self.device = first.info, first.device
            except BaseException:
                self.close()
                raise
        else:
            raise BackendUnavailableError("explicit reference/cpu or curobo/CUDA backend required")

    def _check_open(self):
        if self._closed:
            raise RuntimeError("mode evaluator is closed")
        self.check_budget()
        if self._torch is not None:
            precision = (
                self._torch.get_float32_matmul_precision(),
                self._torch.backends.cuda.matmul.allow_tf32,
            )
            if precision != self._precision:
                self._cache.clear()
                self._masks.clear()
                self.cache_bytes = 0
                self._precision = precision

    def rebind(self, problem):
        """Reuse geometry across skeletons with identical physical context.

        Only the native wall deadline may differ. Mode-mask provenance is
        invalidated; robot FK and static-world geometry remain reusable.
        The session owner must serialize rebinding and queries.
        """
        from .models import ManipulationProblem

        self._check_open()
        problem = value_as(problem, ManipulationProblem, "problem")

        def physical(scene):
            return replace(scene, config=replace(scene.config, time_limit=1))

        if physical(problem.scene) != physical(self.problem) or (
            problem.objects != self.manipulation.objects
            or problem.conventions != self.manipulation.conventions
        ):
            raise ContractError("problem", "cannot rebind changed physical geometry or settings")
        self.manipulation, self.problem = problem, problem.scene
        self.cpu._problem = problem
        self._masks.clear()

    def _rows(self, robot, positions):
        if robot not in self.cpu.robots:
            raise ContractError("robot", "unknown robot")
        rows = value_as(positions, tuple[tuple[float, ...], ...], "positions")
        model = self.cpu.models[self.cpu.robots[robot].model]
        for q in rows:
            model.joint_values(q)
        return rows

    def _device_geometry(self, robot, rows):
        torch = self._torch
        missing = []
        current = {}
        for q in dict.fromkeys(rows):
            key = (robot, q)
            if key in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                current[q] = self._cache[key]
            else:
                self.cache_misses += 1
                missing.append(q)
        if missing:
            backend = self._backends[robot]
            output, collision = backend.evaluate(missing)
            self._link_names[robot] = output.link_names
            self.native_calls += 1
            for i, q in enumerate(missing):
                values = tuple(
                    t[i, 0].clone()
                    for t in (output.positions, output.quaternions, output.spheres, collision.valid)
                )
                current[q] = values
                size = sum(v.numel() * v.element_size() for v in values)
                if size <= self.max_cache_bytes:
                    while self.cache_bytes + size > self.max_cache_bytes:
                        _, old = self._cache.popitem(last=False)
                        self.cache_bytes -= sum(v.numel() * v.element_size() for v in old)
                    self._cache[(robot, q)] = values
                    self.cache_bytes += size
        return tuple(torch.stack([current[q][i] for q in rows]) for i in range(4))

    def _bodies(self, mode, robot):
        self.cpu.check_mode(mode)
        result = []
        for state in mode.states:
            if isinstance(state, HeldObject):
                holders = {g.robot for g in state.grasps}
                if robot not in holders:
                    continue
                if len(holders) != 1:
                    raise UnsupportedModelError(
                        "mode", "dual-held objects require joint skill configurations"
                    )
            result.append(state)
        return result

    def _pair_bad(self, first, second, allowed_rows=None):
        torch = self._torch
        batch, na, nb = len(first), first.shape[1], second.shape[1]
        if not na or not nb:
            return torch.zeros(batch, dtype=torch.bool, device=self.device)
        sa, sb = min(na, self.sphere_tile), min(nb, self.sphere_tile)
        while 64 * batch * sa * sb > self.max_pair_bytes:
            if sa >= sb:
                sa = max(1, sa // 2)
            else:
                sb = max(1, sb // 2)
            if sa == sb == 1 and 64 * batch > self.max_pair_bytes:
                raise ContractError("max_pair_bytes", "reduce batch_size to fit one sphere pair")
        bad = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for a in range(0, na, sa):
            x = first[:, a : a + sa]
            for b in range(0, nb, sb):
                y = second[:, b : b + sb]
                self.max_pair_tile_bytes = max(
                    self.max_pair_tile_bytes, 64 * batch * x.shape[1] * y.shape[1]
                )
                clearance = (
                    torch.linalg.vector_norm(x[:, :, None, :3] - y[:, None, :, :3], dim=-1)
                    - x[:, :, None, 3]
                    - y[:, None, :, 3]
                )
                if not torch.isfinite(clearance).all().item():
                    raise ContractError("clearance", "non-finite CUDA object distance")
                hit = clearance <= self.cpu.threshold
                if allowed_rows is not None:
                    allowed = torch.tensor(
                        allowed_rows[a : a + sa], device=self.device, dtype=torch.bool
                    )[None, :, None]
                    hit = torch.where(allowed, clearance < -self.cpu.policy.contact_tolerance, hit)
                bad |= hit.any(dim=(1, 2))
        return bad

    def _object_spheres(self, state, robot, positions, quaternions):
        torch = self._torch
        obj = self.cpu.objects[state.object]
        if isinstance(state, PlacedObject):
            centers = [_compose(state.pose, Pose(position=s.center)).position for s in obj.spheres]
            check_float32(tuple(v for p in centers for v in p), "object_world_centers")
        else:
            grasp = state.grasps[0]
            local = [_compose(grasp.pose, Pose(position=s.center)).position for s in obj.spheres]
            check_float32(tuple(v for p in local for v in p), "object_link_centers")
            index = self._link_names[robot].index(grasp.link)
            q = quaternions[:, index, None, :]
            q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
            v = torch.tensor(local, device=self.device, dtype=torch.float32)[None].expand(
                len(positions), -1, -1
            )
            xyz = q[..., :3].expand_as(v)
            cross = 2 * torch.linalg.cross(xyz, v, dim=-1)
            centers = (
                positions[:, index, None, :]
                + v
                + q[..., 3:] * cross
                + torch.linalg.cross(xyz, cross, dim=-1)
            )
        if not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, device=self.device, dtype=torch.float32)[None].expand(
                len(positions), -1, -1
            )
        radii = torch.tensor(
            [s.radius for s in obj.spheres], device=self.device, dtype=torch.float32
        )[None, :, None].expand(len(positions), -1, -1)
        return torch.cat((centers, radii), dim=-1)

    def _gpu_flags(self, mode, robot, rows):
        torch = self._torch
        states = self._bodies(mode, robot)
        for state in states:
            obj = self.cpu.objects[state.object]
            check_float32(tuple(v for s in obj.spheres for v in (*s.center, s.radius)), "objects")
            poses = (
                (state.pose,)
                if isinstance(state, PlacedObject)
                else tuple(g.pose for g in state.grasps)
            )
            for pose in poses:
                check_float32((*pose.position, *pose.quaternion), "object_pose")
        # Placed-object support and static/object pairs are configuration-independent.
        placed = [s for s in states if isinstance(s, PlacedObject)]
        frames = [self.cpu.object_frame(s, {})[0] for s in placed]
        if any(
            self.cpu.support_issues(s, f) for s, f in zip(placed, frames, strict=True)
        ) or self.cpu._inspect((), frames, {s.object: s for s in placed}, ()):
            return (False,) * len(rows)
        p, q, robot_spheres, valid = self._device_geometry(robot, rows)
        valid = valid.clone()
        objects = [self._object_spheres(s, robot, p, q) for s in states]
        links = self._backends[robot].sphere_links
        for state, spheres in zip(states, objects, strict=True):
            allowed = [
                isinstance(state, HeldObject) and state.grasps[0].link == link for link in links
            ]
            valid &= ~self._pair_bad(robot_spheres, spheres, allowed)
            if isinstance(state, HeldObject):
                for obstacle in self.problem.obstacles:
                    center = torch.tensor(
                        obstacle.pose.position, device=self.device, dtype=torch.float32
                    )
                    if isinstance(obstacle.shape, Box):
                        axes = [
                            _rotate(obstacle.pose.quaternion, axis)
                            for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                        ]
                        matrix = torch.tensor(
                            tuple(zip(*axes, strict=True)), device=self.device, dtype=torch.float32
                        )
                        half = (
                            torch.tensor(
                                obstacle.shape.size, device=self.device, dtype=torch.float32
                            )
                            / 2
                        )
                        excess = torch.abs((spheres[..., :3] - center) @ matrix) - half
                        clearance = (
                            torch.linalg.vector_norm(excess.clamp_min(0), dim=-1)
                            + excess.amax(-1).clamp_max(0)
                            - spheres[..., 3]
                        )
                    else:
                        clearance = (
                            torch.linalg.vector_norm(spheres[..., :3] - center, dim=-1)
                            - spheres[..., 3]
                            - obstacle.shape.radius
                        )
                    if not torch.isfinite(clearance).all().item():
                        raise ContractError("clearance", "non-finite object/static distance")
                    valid &= ~(clearance <= self.cpu.threshold).any(-1)
        for i, first in enumerate(objects):
            for j in range(i + 1, len(objects)):
                valid &= ~self._pair_bad(first, objects[j])
        return tuple(valid.tolist())

    def node_flags(self, mode, robot, positions):
        self._check_open()
        rows = self._rows(robot, positions)
        self._bodies(mode, robot)
        result = []
        for start in range(0, len(rows), self.batch_size):
            self.check_budget()
            group = rows[start : start + self.batch_size]
            if self._torch is None:
                result.extend(
                    self.cpu.snapshot(mode, {robot: q}, local_robot=robot).valid for q in group
                )
            else:
                result.extend(self._gpu_flags(mode, robot, group))
        return tuple(result)

    def _motion_flags(self, robot, motions, mode=None):
        self._check_open()
        self._rows(robot, ())
        motions = value_as(motions, tuple[Motion, ...], "motions")
        model = self.cpu.models[self.cpu.robots[robot].model]
        times = sample_times(
            self.problem.config.time_step, self.problem.config.validation_time_step
        )
        result = []
        for start in range(0, len(motions), 32):
            self.check_budget()
            group = motions[start : start + 32]
            rows = tuple(
                evaluate_segment(m.start, m.end, self.problem.config.time_step, t).positions
                for m in group
                for t in times
            )
            if mode is not None:
                flags = self.node_flags(mode, robot, rows)
            elif self._torch is None:
                flags = tuple(self.cpu.static_frame(robot, q).valid for q in rows)
            else:
                flags = tuple(
                    flag
                    for begin in range(0, len(rows), self.batch_size)
                    for flag in self._device_geometry(robot, rows[begin : begin + self.batch_size])[
                        3
                    ].tolist()
                )
            for i, motion in enumerate(group):
                result.append(
                    not segment_limit_issues(
                        model,
                        motion.start,
                        motion.end,
                        self.problem.config.time_step,
                        joint_tolerance=self.problem.config.joint_tolerance,
                    )
                    and all(flags[i * len(times) : (i + 1) * len(times)])
                )
        return tuple(result)

    def motion_validity(self, robot, motions):
        return self._motion_flags(robot, motions)

    def mask(self, graph, mode):
        self._check_open()
        value_as(graph, JointGraph, "graph")
        self._rows(graph.robot, graph.vertices)
        model = self.cpu.models[self.cpu.robots[graph.robot].model]
        if graph.joint_names != model.joint_order:
            raise ContractError("graph", "joint order mismatch")
        key = (fingerprint(graph), fingerprint(mode))
        if key in self._masks:
            self._masks.move_to_end(key)
            return self._masks[key]
        nodes = self.node_flags(mode, graph.robot, graph.vertices)
        edges = []
        for begin in range(0, len(graph.edges), 32):
            group = graph.edges[begin : begin + 32]
            motions = tuple(
                Motion(start=graph.vertices[a], end=graph.vertices[b]) for a, b in group
            )
            forward = self._motion_flags(graph.robot, motions, mode)
            reverse = self._motion_flags(
                graph.robot, tuple(Motion(start=m.end, end=m.start) for m in motions), mode
            )
            edges.extend(
                nodes[a] and nodes[b] and f and r
                for (a, b), f, r in zip(group, forward, reverse, strict=True)
            )
        result = ModeValidityMask(
            problem_digest=document_digest(self.manipulation),
            mode_digest=key[1],
            graph_digest=key[0],
            robot=graph.robot,
            node_flags=nodes,
            edge_flags=tuple(edges),
            time_step=self.problem.config.time_step,
            sample_time_step=self.problem.config.validation_time_step,
            policy=self.cpu.policy,
            backend=self.info,
        )
        if len(self._masks) >= self.max_masks:
            self._masks.popitem(last=False)
        self._masks[key] = result
        return result

    @property
    def cache_stats(self):
        return {
            "hits": self.cache_hits if self._torch is not None else self.cpu.fk_hits,
            "misses": self.cache_misses if self._torch is not None else self.cpu.fk_misses,
            "bytes": self.cache_bytes,
            "native_calls": self.native_calls,
            "max_pair_tile_bytes": self.max_pair_tile_bytes,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        self._cache.clear()
        self._masks.clear()
        self.cache_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
