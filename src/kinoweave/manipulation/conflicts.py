"""Eager, bounded CPU/CUDA conflict tables for a complete state's choices."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from .._contracts import ContractError
from ..backends._description import check_float32
from ..geometry.reference import obstacle_clearance
from ..models import Box
from ..search.space import SearchLimit
from ..trajectory import sample_times
from .geometry import GeometrySphere, pose_matches
from .models import ContactBody, HeldObject, PlacedObject
from .task import ActionClock


@dataclass(frozen=True, slots=True)
class StateConflictTable:
    actions: tuple
    points: tuple[tuple[float, str], ...]
    valid: tuple[bool, ...]
    pairs: tuple[tuple[bool, ...], ...]

    def compatible(self, first, second):
        return self.valid[first] and self.valid[second] and self.pairs[first][second]


@dataclass(slots=True)
class _Payload:
    bodies: tuple[ContactBody, ...]
    spheres: object  # CPU tuples or CUDA [time, spheres, xyzr].
    permissions: tuple[frozenset, ...]
    valid: bool


def _body_permissions(mode, owned, contacts):
    result = {frozenset(pair) for pair in contacts}
    for state in mode.states:
        if state.object not in owned:
            continue
        obj = ContactBody(kind="object", name=state.object)
        if isinstance(state, HeldObject):
            result.update(
                frozenset((obj, ContactBody(kind="robot", name=g.robot, link=g.link)))
                for g in state.grasps
            )
        elif state.support is not None:
            result.add(frozenset((obj, ContactBody(kind="obstacle", name=state.support))))
    return frozenset(result)


class ConflictBuilder:
    def __init__(self, space):
        self.space, self.evaluator = space, space.evaluator
        self.cpu = self.evaluator.cpu
        self.config = space.config
        self.torch = self.evaluator._torch
        self._support = {}

    def _supported(self, state):
        if state not in self._support:
            frame, _ = self.cpu.object_frame(state, {})
            self._support[state] = not self.cpu.support_issues(state, frame)
        return self._support[state]

    def _samples(self, clocks):
        boundaries = sorted({0.0, self.space.dt}.union(*(c.boundaries for c in clocks)))
        points = [(0.0, "before"), (0.0, "after")]
        count = 1
        for first, last in pairwise(boundaries):
            offsets = sample_times(
                last - first,
                self.space.problem.scene.config.validation_time_step,
                max_samples=self.config.max_validation_samples - count + 1,
            )
            count += len(offsets) - 1
            for i, offset in enumerate(offsets[1:], 1):
                at = last if i == len(offsets) - 1 else first + offset
                if not first < at <= last:
                    raise ContractError("conflicts.time", "unrepresentable sample")
                points.extend(((at, "before"), (at, "after")) if at == last else ((at, "after"),))
        return tuple(points)

    def _description(self, state, clock, owned, points):
        modes = tuple(clock.mode(t, side) if clock else state.mode for t, side in points)
        configs = tuple(clock.configurations(t, side) if clock else {} for t, side in points)
        permissions = tuple(
            _body_permissions(
                mode, owned, clock.contacts(t, "after" if t == 0 else side) if clock else ()
            )
            for (t, side), mode in zip(points, modes, strict=True)
        )
        return modes, configs, permissions

    def _cpu_payload(self, state, clock, owned, points):
        modes, configs, permissions = self._description(state, clock, owned, points)
        rows, objects, valid = [], [], True
        for mode, qs in zip(modes, configs, strict=True):
            self.space.check("conflict_geometry")
            robots = {r: self.cpu.fk(r, q) for r, q in qs.items()}
            poses = {r: dict(zip(f.link_names, f.poses, strict=True)) for r, f in robots.items()}
            spheres = [
                GeometrySphere(
                    ContactBody(kind="robot", name=r, link=s.link), s.index, s.center, s.radius
                )
                for r, fk in robots.items()
                for s in fk.spheres
            ]
            valid &= not any(self.cpu.position_issues(r, q) for r, q in qs.items())
            object_poses = {}
            for obj in mode.states:
                if obj.object not in owned:
                    continue
                frame, issues = self.cpu.object_frame(obj, poses)
                valid &= not issues
                if isinstance(obj, PlacedObject):
                    valid &= self._supported(obj)
                spheres.extend(frame.spheres)
                object_poses[obj.object] = frame.pose
            rows.append(tuple(spheres))
            objects.append(object_poses)
        if clock:
            for at, _rank, _index, change in clock.events:
                before = points.index((at, "before"))
                after = points.index((at, "after"))
                valid &= pose_matches(
                    objects[before][change.before.object],
                    objects[after][change.after.object],
                    self.cpu.policy,
                )
        bodies = tuple(s.body for s in rows[0])
        return _Payload(bodies, tuple(rows), permissions, bool(valid))

    def _rotate(self, q, v):
        torch = self.torch
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        xyz = q[..., :3].expand_as(v)
        cross = 2 * torch.linalg.cross(xyz, v, dim=-1)
        return v + q[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1)

    def _multiply(self, a, b):
        torch = self.torch
        a, b = torch.broadcast_tensors(a, b)
        return torch.cat(
            (
                a[..., 3:] * b[..., :3]
                + b[..., 3:] * a[..., :3]
                + torch.linalg.cross(a[..., :3], b[..., :3], dim=-1),
                a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True),
            ),
            -1,
        )

    def _pose_bad(self, a, b):
        torch, policy = self.torch, self.cpu.policy
        p, q = a
        other_p, other_q = b
        conjugate = torch.cat((-q[..., :3], q[..., 3:]), -1)
        relative = self._multiply(conjugate, other_q)
        angle = 2 * torch.atan2(
            torch.linalg.vector_norm(relative[..., :3], dim=-1), relative[..., 3].abs()
        )
        return (
            (torch.linalg.vector_norm(p - other_p, dim=-1) > policy.position_tolerance)
            | (angle > policy.orientation_tolerance)
            | ~torch.isfinite(p).all(-1)
            | ~torch.isfinite(other_p).all(-1)
            | ~torch.isfinite(angle)
        )

    def _gpu_payload(self, state, clock, owned, points):
        torch, device = self.torch, self.evaluator.device
        modes, configs, permissions = self._description(state, clock, owned, points)
        total = len(points)
        arrays, bodies, links = [], [], {}
        valid = True
        for name in configs[0]:
            qs = tuple(c[name] for c in configs)
            valid &= not any(self.cpu.position_issues(name, q) for q in qs)
            groups = []
            for start in range(0, total, self.evaluator.batch_size):
                self.space.check("conflict_geometry")
                groups.append(
                    self.evaluator._device_geometry(
                        name, qs[start : start + self.evaluator.batch_size]
                    )
                )
            p, q, spheres, _ = (torch.cat([group[i] for group in groups]) for i in range(4))
            links[name] = (p, q)
            arrays.append(spheres)
            bodies.extend(
                ContactBody(kind="robot", name=name, link=link)
                for link in self.evaluator._backends[name].sphere_links
            )

        def pose(obj):
            if isinstance(obj, PlacedObject):
                p = torch.tensor(obj.pose.position, device=device, dtype=torch.float32).expand(
                    total, -1
                )
                q = torch.tensor(obj.pose.quaternion, device=device, dtype=torch.float32).expand(
                    total, -1
                )
                return p, q
            values = []
            for grasp in obj.grasps:
                p, q = links[grasp.robot]
                index = self.evaluator._link_names[grasp.robot].index(grasp.link)
                p, q = p[:, index], q[:, index]
                local_p = torch.tensor(
                    grasp.pose.position, device=device, dtype=torch.float32
                ).expand(total, -1)
                local_q = torch.tensor(
                    grasp.pose.quaternion, device=device, dtype=torch.float32
                ).expand(total, -1)
                values.append((p + self._rotate(q, local_p), self._multiply(q, local_q)))
            return values

        object_poses = {}
        invalid = torch.zeros(total, dtype=torch.bool, device=device)
        for name in owned:
            obj = self.cpu.objects[name]
            check_float32(tuple(v for s in obj.spheres for v in (*s.center, s.radius)), "objects")
            states = tuple(next(s for s in m.states if s.object == name) for m in modes)
            world_p = torch.empty((total, 3), device=device, dtype=torch.float32)
            world_q = torch.empty((total, 4), device=device, dtype=torch.float32)
            for selected in dict.fromkeys(states):
                poses = (
                    (selected.pose,)
                    if isinstance(selected, PlacedObject)
                    else tuple(g.pose for g in selected.grasps)
                )
                for transform in poses:
                    check_float32((*transform.position, *transform.quaternion), "object_pose")
                indices = torch.tensor(
                    [i for i, s in enumerate(states) if s == selected], device=device
                )
                if isinstance(selected, PlacedObject):
                    valid &= self._supported(selected)
                    p, q = pose(selected)
                else:
                    candidates = pose(selected)
                    p, q = candidates[0]
                    for other in candidates[1:]:
                        invalid[indices] |= self._pose_bad((p, q), other)[indices]
                world_p[indices], world_q[indices] = p[indices], q[indices]
            object_poses[name] = (world_p, world_q)
            local = torch.tensor(
                [s.center for s in obj.spheres], device=device, dtype=torch.float32
            )
            centers = world_p[:, None] + self._rotate(
                world_q[:, None], local[None].expand(total, -1, -1)
            )
            # radii are [time, spheres, 1], without reusing mutable native buffers.
            radii = torch.tensor(
                [s.radius for s in obj.spheres], device=device, dtype=torch.float32
            )[None, :, None].expand(total, -1, -1)
            arrays.append(torch.cat((centers, radii), -1))
            bodies.extend((ContactBody(kind="object", name=name),) * len(obj.spheres))
        if clock:
            for at, _rank, _index, change in clock.events:
                before, after = points.index((at, "before")), points.index((at, "after"))
                p, q = object_poses[change.before.object]
                invalid[after] |= self._pose_bad((p[before], q[before]), (p[after], q[after]))
        spheres = (
            torch.cat(arrays, 1)
            if arrays
            else torch.empty((total, 0, 4), device=device, dtype=torch.float32)
        )
        if not torch.isfinite(spheres).all().item():
            raise ContractError("conflict_geometry", "non-finite CUDA geometry")
        valid &= not invalid.any().item()
        return _Payload(tuple(bodies), spheres, permissions, bool(valid))

    def _ignored(self, a, b):
        if a == b:
            return True
        if a.kind == b.kind == "robot" and a.name == b.name:
            model = self.cpu.models[self.cpu.robots[a.name].model]
            return frozenset((a.link, b.link)) in {frozenset(p) for p in model.allowed_collisions}
        return False

    def _cpu_static(self, payload):
        for row, permissions in zip(payload.spheres, payload.permissions, strict=True):
            for sphere in row:
                for obstacle in self.space.problem.scene.obstacles:
                    body = ContactBody(kind="obstacle", name=obstacle.name)
                    clearance = obstacle_clearance(sphere, obstacle)
                    allowed = frozenset((sphere.body, body)) in permissions
                    if not math.isfinite(clearance):
                        raise ContractError("clearance", "non-finite CPU distance")
                    if (
                        clearance < -self.cpu.policy.contact_tolerance
                        if allowed
                        else clearance <= self.cpu.threshold
                    ):
                        return False
        return True

    def _gpu_static(self, payload):
        torch, device = self.torch, self.evaluator.device
        bad = torch.zeros(len(payload.permissions), dtype=torch.bool, device=device)
        for obstacle in self.space.problem.scene.obstacles:
            body = ContactBody(kind="obstacle", name=obstacle.name)
            center = torch.tensor(obstacle.pose.position, device=device, dtype=torch.float32)
            if isinstance(obstacle.shape, Box):
                from ..geometry.reference import _rotate

                axes = [
                    _rotate(obstacle.pose.quaternion, axis)
                    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                ]
                matrix = torch.tensor(
                    tuple(zip(*axes, strict=True)), device=device, dtype=torch.float32
                )
                half = torch.tensor(obstacle.shape.size, device=device, dtype=torch.float32) / 2
                excess = ((payload.spheres[..., :3] - center) @ matrix).abs() - half
                clearance = (
                    torch.linalg.vector_norm(excess.clamp_min(0), dim=-1)
                    + excess.amax(-1).clamp_max(0)
                    - payload.spheres[..., 3]
                )
            else:
                clearance = (
                    torch.linalg.vector_norm(payload.spheres[..., :3] - center, dim=-1)
                    - obstacle.shape.radius
                    - payload.spheres[..., 3]
                )
            if not torch.isfinite(clearance).all().item():
                raise ContractError("clearance", "non-finite CUDA static distance")
            allowed = torch.tensor(
                [
                    [frozenset((b, body)) in permissions for b in payload.bodies]
                    for permissions in payload.permissions
                ],
                device=device,
                dtype=torch.bool,
            ).reshape(clearance.shape)
            hit = torch.where(
                allowed,
                clearance < -self.cpu.policy.contact_tolerance,
                clearance <= self.cpu.threshold,
            )
            bad |= hit.any(-1)
        return not bad.any().item()

    def _cpu_pairs(self, queries):
        flags = []
        for first, second in queries:
            self.space.check("conflict_prefetch")
            valid = True
            for k, (x, y) in enumerate(zip(first.spheres, second.spheres, strict=True)):
                for a in x:
                    for b in y:
                        if self._ignored(a.body, b.body):
                            continue
                        clearance = math.dist(a.center, b.center) - a.radius - b.radius
                        if not math.isfinite(clearance):
                            raise ContractError("clearance", "non-finite CPU pair distance")
                        pair = frozenset((a.body, b.body))
                        allowed = pair in first.permissions[k] or pair in second.permissions[k]
                        if (
                            clearance < -self.cpu.policy.contact_tolerance
                            if allowed
                            else clearance <= self.cpu.threshold
                        ):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            flags.append(valid)
        self.space.stats["cpu_pair_batches"] += 1
        return flags

    def _gpu_pairs(self, queries):
        torch, cfg = self.torch, self.config
        if not queries:
            return []
        total, na, nb = (
            len(queries[0][0].permissions),
            len(queries[0][0].bodies),
            len(queries[0][1].bodies),
        )
        if not na or not nb:
            return [True] * len(queries)
        k = len(queries)
        sa, sb = min(na, cfg.sphere_tile), min(nb, cfg.sphere_tile)
        while 128 * k * sa * sb > cfg.max_pair_bytes:
            if sa >= sb and sa > 1:
                sa = max(1, sa // 2)
            elif sb > 1:
                sb = max(1, sb // 2)
            else:
                raise SearchLimit("pair_tile_limit", "conflict_prefetch")
        time_tile = max(1, min(total, cfg.max_pair_bytes // (128 * k * sa * sb)))
        bad = torch.zeros(k, dtype=torch.bool, device=self.evaluator.device)
        for t in range(0, total, time_tile):
            self.space.check("conflict_prefetch")
            end = min(total, t + time_tile)
            for a in range(0, na, sa):
                for b in range(0, nb, sb):
                    x = torch.stack([p.spheres[t:end, a : a + sa] for p, _ in queries])
                    y = torch.stack([p.spheres[t:end, b : b + sb] for _, p in queries])
                    size = 128 * k * (end - t) * x.shape[2] * y.shape[2]
                    self.space.stats["max_pair_tile_bytes"] = max(
                        self.space.stats["max_pair_tile_bytes"], size
                    )
                    clearance = (
                        torch.linalg.vector_norm(
                            x[:, :, :, None, :3] - y[:, :, None, :, :3], dim=-1
                        )
                        - x[:, :, :, None, 3]
                        - y[:, :, None, :, 3]
                    )
                    if not torch.isfinite(clearance).all().item():
                        raise ContractError("clearance", "non-finite CUDA pair distance")
                    allowed, ignored = [], []
                    for first, second in queries:
                        allowed.append(
                            [
                                [
                                    [
                                        frozenset((u, v)) in first.permissions[s]
                                        or frozenset((u, v)) in second.permissions[s]
                                        for v in second.bodies[b : b + sb]
                                    ]
                                    for u in first.bodies[a : a + sa]
                                ]
                                for s in range(t, end)
                            ]
                        )
                        ignored.append(
                            [
                                [
                                    [self._ignored(u, v) for v in second.bodies[b : b + sb]]
                                    for u in first.bodies[a : a + sa]
                                ]
                            ]
                            * (end - t)
                        )
                    allowed = torch.tensor(allowed, device=self.evaluator.device, dtype=torch.bool)
                    ignored = torch.tensor(ignored, device=self.evaluator.device, dtype=torch.bool)
                    hit = (
                        torch.where(
                            allowed,
                            clearance < -self.cpu.policy.contact_tolerance,
                            clearance <= self.cpu.threshold,
                        )
                        & ~ignored
                    )
                    bad |= hit.any(dim=(1, 2, 3))
        self.space.stats["gpu_pair_batches"] += 1
        return [not value for value in bad.tolist()]

    def _pairs(self, queries):
        if self.torch is None:
            return self._cpu_pairs(queries)
        grouped = defaultdict(list)
        for i, (a, b) in enumerate(queries):
            grouped[len(a.permissions), len(a.bodies), len(b.bodies)].append(i)
        result = [False] * len(queries)
        width = min(
            self.config.pair_batch_size,
            self.config.max_query_pairs,
            max(1, self.config.max_pair_bytes // 128),
        )
        for indices in grouped.values():
            for begin in range(0, len(indices), width):
                group = indices[begin : begin + width]
                flags = self._gpu_pairs([queries[i] for i in group])
                for i, flag in zip(group, flags, strict=True):
                    result[i] = flag
        return result

    def build(self, state, actions, owners):
        self.evaluator._check_open()
        self.space.check("conflict_prefetch")
        cfg, count = self.config, len(actions)
        if count * count > cfg.max_state_pairs:
            raise SearchLimit("state_pair_limit", "conflict_prefetch")
        clocks = [ActionClock(self.space, state, a) for a in actions]
        points = self._samples(clocks)
        owned = [
            tuple(name for name, robot in owners.items() if robot in a.robots) for a in actions
        ]
        background_objects = tuple(name for name, robot in owners.items() if robot is None)
        estimated = 0
        for a, objects in zip(actions, owned, strict=True):
            spheres = sum(
                len(self.cpu.models[self.cpu.robots[self.space.names[r]].model].collision_spheres)
                for r in a.robots
            ) + sum(len(self.cpu.objects[o].spheres) for o in objects)
            links = sum(
                len(self.cpu.models[self.cpu.robots[self.space.names[r]].model].links)
                for r in a.robots
            )
            estimated += len(points) * (16 * spheres + 28 * (links + len(objects)))
        estimated += len(points) * sum(
            16 * len(self.cpu.objects[o].spheres) + 28 for o in background_objects
        )
        if estimated > cfg.max_state_bytes:
            raise SearchLimit("state_geometry_limit", "conflict_geometry")
        self.space.stats["peak_state_geometry_bytes"] = max(
            self.space.stats["peak_state_geometry_bytes"], estimated
        )
        payload = self._cpu_payload if self.torch is None else self._gpu_payload
        bodies = [
            payload(state, clock, objects, points)
            for clock, objects in zip(clocks, owned, strict=True)
        ]
        background = payload(state, None, background_objects, points)
        static = self._cpu_static if self.torch is None else self._gpu_static
        background_valid = (
            background.valid and static(background) and self._pairs([(background, background)])[0]
        )
        basic = [p.valid and static(p) and background_valid for p in bodies]
        unary_queries = [(p, p) for p in bodies] + [(p, background) for p in bodies]
        unary = self._pairs(unary_queries)
        valid = tuple(b and unary[i] and unary[i + count] for i, b in enumerate(basic))
        pairs = [[False] * count for _ in actions]
        indices, queries = [], []
        for i, first in enumerate(actions):
            pairs[i][i] = True
            for j in range(i + 1, count):
                if not set(first.robots).intersection(actions[j].robots):
                    indices.append((i, j))
                    queries.append((bodies[i], bodies[j]))
        # All candidate pairs are materialized before recursive selection.
        flags = self._pairs(queries)
        for (i, j), flag in zip(indices, flags, strict=True):
            pairs[i][j] = pairs[j][i] = flag
        self.space.stats["prefetch_calls"] += 1
        self.space.stats["prefetched_pairs"] += len(queries)
        return StateConflictTable(tuple(actions), points, valid, tuple(tuple(row) for row in pairs))
