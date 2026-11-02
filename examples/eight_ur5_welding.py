"""Workpiece-first weld targets and a small, example-only numerical IK solver.

The target geometry and checks use only the core. NumPy is imported lazily
when regenerating the source joint paths in the optional URDF environment.
"""

import json
import math
from pathlib import Path

from kinoweave import Box, Obstacle, Pose
from kinoweave.geometry.reference import _rotate

NORMALS = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))
WELD_TIMES = (0, 2, 6, 7, 8, 9, 13, 15)
WELD_SEGMENTS = 4


def bead_fraction(fraction):
    """Authored Cartesian law for four stopped-quintic short weld segments."""
    index = min(WELD_SEGMENTS - 1, int(fraction * WELD_SEGMENTS))
    local = fraction * WELD_SEGMENTS - index
    smooth = local**3 * (10 + local * (-15 + 6 * local))
    return (index + smooth) / WELD_SEGMENTS


def read_workpiece():
    return json.loads((Path(__file__).parent / "assets/eight_ur5/workpiece.json").read_text())


def _add(*vectors):
    return tuple(sum(values) for values in zip(*vectors, strict=True))


def _scale(vector, factor):
    return tuple(value * factor for value in vector)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def quaternion(matrix):
    """Rotation matrix to xyzw; also used for IK's rotation-vector residual."""
    m = matrix
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4)
    else:
        i = max(range(3), key=lambda i: m[i][i])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(max(0, 1 + m[i][i] - m[j][j] - m[k][k]))
        values = [0.0] * 4
        values[i], values[j], values[k], values[3] = (
            s / 4,
            (m[j][i] + m[i][j]) / s,
            (m[k][i] + m[i][k]) / s,
            (m[k][j] - m[j][k]) / s,
        )
        q = tuple(values)
    norm = math.hypot(*q)
    return tuple(v / norm * (1 if q[3] >= 0 else -1) for v in q)


def workpiece_geometry(spec):
    """Twelve finite steel plates form eight authored vertical butt seams."""
    if spec.get("format") != "kinoweave.demo.weld-workpiece" or spec.get("schema_version") != 1:
        raise ValueError("unknown workpiece definition")
    if spec["tool_length"] != 0.18 or spec["nozzle_gap"] != 0.012:
        raise ValueError("changed torch dimensions require matching visual and collision models")
    center = tuple(spec["center"])
    surface, half, thick, height = (
        spec[k] for k in ("surface_offset", "panel_half_width", "plate_thickness", "panel_height")
    )
    offsets = spec["seam_offsets"]
    if not -half < offsets[0] < offsets[1] < half or min(thick, height) <= 0:
        raise ValueError("invalid plate or seam dimensions")
    obstacles, seams = [], []

    def box(name, position, size):
        obstacles.append(Obstacle(name=name, pose=Pose(position=position), shape=Box(size=size)))

    for face, normal in enumerate(NORMALS):
        tangent = _cross((0, 0, 1), normal)
        edges = (-half, *offsets, half)
        names = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            name = f"workpiece_plate_{face}_{i}"
            names.append(name)
            box(
                name,
                _add(center, _scale(normal, surface - thick / 2), _scale(tangent, (lo + hi) / 2)),
                _add(
                    _scale(tuple(map(abs, normal)), thick),
                    _scale(tuple(map(abs, tangent)), hi - lo),
                    (0, 0, height),
                ),
            )
        for i, offset in enumerate(offsets):
            origin = _add(center, _scale(normal, surface), _scale(tangent, offset))
            seams.append(
                {
                    "name": f"joint{len(seams) + 1:02d}",
                    "parts": names[i : i + 2],
                    "origin": origin,
                    "normal": normal,
                    "start": _add(origin, (0, 0, -height / 2)),
                    "end": _add(origin, (0, 0, height / 2)),
                }
            )
        box(
            f"workpiece_backing_{face}",
            _add(center, _scale(normal, surface - thick - 0.015)),
            _add(
                _scale(tuple(map(abs, normal)), 0.035),
                _scale(tuple(map(abs, tangent)), 0.63),
                (0, 0, 0.035),
            ),
        )
    radius, tube = spec["frame_half_width"], spec["frame_tube_width"]
    bottom, top = spec["frame_bottom"], spec["frame_top"]
    cx, cy, _ = center
    for x in (-radius, radius):
        for y in (-radius, radius):
            box(
                f"workpiece_post_{x}_{y}",
                (cx + x, cy + y, (bottom + top) / 2),
                (tube, tube, top - bottom),
            )
    for z in (bottom, top):
        for side in (-1, 1):
            box(
                f"workpiece_rail_x_{z}_{side}",
                (cx, cy + side * radius, z),
                (2 * radius + tube, tube, tube),
            )
            box(
                f"workpiece_rail_y_{z}_{side}",
                (cx + side * radius, cy, z),
                (tube, 2 * radius + tube, tube),
            )
    return tuple(obstacles), tuple(seams)


def weld_targets(spec, robots):
    """Define positions/orientations on plate seams, without robot FK or IK."""
    _, seams = workpiece_geometry(spec)
    targets = []
    for index, robot in enumerate(robots):
        upper = index >= 4
        for j, offset in enumerate(spec["bead_center_offsets"]):
            station = (2 * (index % 4) + 1 + (-1 if upper else 1) * (j % 2)) % 8
            seam = seams[station]
            normal = seam["normal"]
            center = _add(seam["origin"], (0, 0, offset + spec["level_offsets"][int(upper)]))
            y, z = (0, 0, 1 if upper else -1), _scale(normal, -1)
            x = _cross(y, z)
            rotation = tuple(zip(x, y, z, strict=True))
            targets.append(
                {
                    "id": f"{robot.name}.weld{j + 1:02d}",
                    "skill": f"{robot.name}.weld{j + 1:02d}",
                    "robot": robot.name,
                    "seam": seam["name"],
                    "parts": seam["parts"],
                    "position": center,
                    "start_position": _add(center, (0, 0, -spec["bead_length"] / 2)),
                    "end_position": _add(center, (0, 0, spec["bead_length"] / 2)),
                    "normal": normal,
                    "quaternion_xyzw": quaternion(rotation),
                    "weld_source": (WELD_TIMES[3], WELD_TIMES[4]),
                }
            )
    return tuple(targets)


def check_surface_point(point, normal, parts, obstacles, tolerance=1e-9):
    """Require a point to lie on the outward face of both finite adjoining plates."""
    for name in parts:
        obstacle = obstacles[name]
        if not isinstance(obstacle.shape, Box):
            raise ValueError("weld seam requires explicit box plates")
        qx, qy, qz, qw = obstacle.pose.quaternion
        relative = tuple(a - b for a, b in zip(point, obstacle.pose.position, strict=True))
        local = _rotate((-qx, -qy, -qz, qw), relative)
        local_normal = _rotate((-qx, -qy, -qz, qw), normal)
        half = tuple(v / 2 for v in obstacle.shape.size)
        if any(abs(v) > h + tolerance for v, h in zip(local, half, strict=True)):
            raise ValueError("weld target is outside a finite workpiece face")
        plane = sum(v * n for v, n in zip(local, local_normal, strict=True))
        extent = sum(h * abs(n) for h, n in zip(half, local_normal, strict=True))
        if abs(plane - extent) > tolerance:
            raise ValueError("weld target is not on the workpiece surface")


class TorchIK:
    """Deterministic damped least squares on the URDF chain; source generation only."""

    def __init__(self, model, robot, link="weld_tcp"):
        import numpy as np

        self.np = np
        by_child = {j.child: j for j in model.joints}
        chain = []
        while link in by_child:
            joint = by_child[link]
            if joint.kind not in ("fixed", "revolute", "continuous") or joint.mimic is not None:
                raise ValueError("example IK supports a non-mimic revolute chain")
            chain.append(joint)
            link = joint.parent
        self.chain = []
        for joint in reversed(chain):
            index = model.joint_order.index(joint.name) if joint.name in model.joint_order else None
            axis = np.asarray(joint.axis, dtype=float)
            if index is not None:
                axis /= np.linalg.norm(axis)
            self.chain.append((self.matrix(joint.origin), axis, index))
        self.base = self.matrix(robot.base_pose)

    def matrix(self, pose):
        matrix = self.np.eye(4)
        matrix[:3, 3] = pose.position
        matrix[:3, :3] = self.np.asarray(
            [_rotate(pose.quaternion, axis) for axis in self.np.eye(3)]
        ).T
        return matrix

    def forward(self, positions):
        np = self.np
        transform = self.base.copy()
        axes = []
        for origin, axis, index in self.chain:
            transform = transform @ origin
            if index is None:
                continue
            axes.append((index, transform[:3, :3] @ axis, transform[:3, 3].copy()))
            x, y, z = axis
            skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
            motion = np.eye(4)
            angle = positions[index]
            motion[:3, :3] = (
                np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
            )
            transform = transform @ motion
        jacobian = np.zeros((6, len(positions)))
        for index, axis, center in axes:
            jacobian[:, index] = np.r_[np.cross(axis, transform[:3, 3] - center), axis]
        return transform, jacobian

    def solve(self, target, seed):
        np = self.np
        goal, q = self.matrix(target), np.asarray(seed, dtype=float).copy()
        lower, upper = (-3, -3, 0.1, -6, -3.13, -3), (3, 0.2, 3, 3, 3.13, 3)
        for _ in range(150):
            current, jacobian = self.forward(q)
            quaternion_error = np.asarray(quaternion(goal[:3, :3] @ current[:3, :3].T))
            norm = np.linalg.norm(quaternion_error[:3])
            angular = (
                2 * math.atan2(norm, quaternion_error[3]) * quaternion_error[:3] / norm
                if norm > 1e-10
                else 2 * quaternion_error[:3]
            )
            error = np.r_[goal[:3, 3] - current[:3, 3], angular]
            if np.linalg.norm(error[:3]) < 1e-9 and np.linalg.norm(error[3:]) < 1e-8:
                return tuple(round(float(value), 12) for value in q)
            error[3:] *= 0.35
            jacobian[3:] *= 0.35
            step = np.linalg.solve(jacobian.T @ jacobian + np.eye(6) * 1e-5, jacobian.T @ error)
            step *= min(1, 0.3 / max(max(abs(step)), 1e-12))
            q = np.clip(q + step, lower, upper)
        raise ValueError(f"example IK did not reach workpiece pose {target}")
