"""Preserve supported native kinematics in cuRobo's URDF input."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

from .. import ContractError, Pose, RobotModel


class UnsupportedModelError(ContractError):
    """A valid core model that this backend cannot preserve faithfully."""


def rotation(pose: Pose) -> tuple[tuple[float, ...], ...]:
    """Parent-from-local rotation for an already validated xyzw pose."""
    x, y, z, w = pose.quaternion
    scale = 2 / sum(component * component for component in pose.quaternion)
    return (
        (1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)),
        (scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)),
        (scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)),
    )


def rpy(pose: Pose) -> tuple[float, float, float]:
    matrix = rotation(pose)
    pitch = math.atan2(-matrix[2][0], math.hypot(matrix[0][0], matrix[1][0]))
    if abs(math.cos(pitch)) < 1e-10:
        return math.atan2(-matrix[1][2], matrix[1][1]), pitch, 0.0
    return math.atan2(matrix[2][1], matrix[2][2]), pitch, math.atan2(matrix[1][0], matrix[0][0])


def _floats(values: tuple[float, ...]) -> str:
    return " ".join(format(value, ".17g") for value in values)


def check_float32(values, path: str) -> None:
    for value in values:
        try:
            converted = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as error:
            raise UnsupportedModelError(path, "value is outside the float32 range") from error
        if not math.isfinite(converted) or (value != 0 and converted == 0):
            raise UnsupportedModelError(path, "value is outside the float32 range")


def has_self_pairs(model: RobotModel) -> bool:
    links = tuple(dict.fromkeys(s.link for s in model.collision_spheres))
    ignored = {frozenset(pair) for pair in model.allowed_collisions}
    return any(
        frozenset((a, b)) not in ignored for i, a in enumerate(links) for b in links[i + 1 :]
    )


def validate_model(model: RobotModel) -> None:
    """Reject upstream conversions that would clamp or alter input semantics."""
    if not model.joint_order:
        raise UnsupportedModelError(model.name, "cuRobo needs at least one independent joint")
    for joint in model.joints:
        path = f"models.{model.name}.joints.{joint.name}"
        if joint.kind == "continuous" or joint.mimic is not None:
            raise UnsupportedModelError(path, "continuous and mimic joints are unsupported on CUDA")
        if joint.kind != "fixed":
            if sorted(abs(value) for value in joint.axis) != [0.0, 0.0, 1.0]:
                raise UnsupportedModelError(
                    path + ".axis", "cuRobo requires a signed coordinate axis"
                )
            assert joint.limits is not None
            if joint.limits.lower == joint.limits.upper:
                raise UnsupportedModelError(
                    path + ".limits", "zero-width bounds need a fixed joint"
                )


def description(
    model: RobotModel, urdf_path: Path, threshold: float
) -> tuple[dict, tuple[int, ...]]:
    """Write kinematic-only URDF and preserve explicit spheres and ignore pairs.

    Effort is unbounded because the core model has no torque data. Dynamics
    loading is disabled. The generated file is not a dynamics robot model.
    """
    validate_model(model)
    root = ET.Element("robot", name=model.name)
    for name in model.links:
        ET.SubElement(root, "link", name=name)
    for joint in model.joints:
        node = ET.SubElement(root, "joint", name=joint.name, type=joint.kind)
        ET.SubElement(node, "parent", link=joint.parent)
        ET.SubElement(node, "child", link=joint.child)
        ET.SubElement(
            node, "origin", xyz=_floats(joint.origin.position), rpy=_floats(rpy(joint.origin))
        )
        if joint.kind != "fixed":
            ET.SubElement(node, "axis", xyz=_floats(joint.axis))
            assert joint.limits is not None
            ET.SubElement(
                node,
                "limit",
                lower=str(joint.limits.lower),
                upper=str(joint.limits.upper),
                velocity=str(joint.limits.velocity),
                effort="inf",
            )
    spheres = {}
    source_indices = {}
    for index, sphere in enumerate(model.collision_spheres):
        spheres.setdefault(sphere.link, []).append(
            {"center": list(sphere.center), "radius": sphere.radius}
        )
        source_indices.setdefault(sphere.link, []).append(index)
    # The pinned loader divides by zero when there are no self-collision pairs.
    # For one collision link, a fixed helper carries an identical sphere. The
    # occupied union stays exact. These artificial pairs are never queried and
    # IK self-checking is disabled when the original model has no active pairs.
    if len(spheres) == 1:
        only = next(iter(spheres))
        helper = "_kinoweave_sphere_helper"
        while helper in model.links or helper in {j.name for j in model.joints}:
            helper += "_"
        ET.SubElement(root, "link", name=helper)
        node = ET.SubElement(root, "joint", name=helper, type="fixed")
        ET.SubElement(node, "parent", link=only)
        ET.SubElement(node, "child", link=helper)
        spheres[helper] = [spheres[only][0].copy()]
        source_indices[helper] = [None]
    ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)
    native_order = []
    for link in spheres:
        native_order.extend(source_indices[link])
    restore = tuple(native_order.index(index) for index in range(len(model.collision_spheres)))
    ignores = {}
    for first, second in model.allowed_collisions:
        ignores.setdefault(first, []).append(second)
    if not has_self_pairs(model):
        ignores = {}
    by_name = {joint.name: joint for joint in model.joints}
    defaults = []
    accelerations = []
    for name in model.joint_order:
        limits = by_name[name].limits
        assert limits is not None and limits.lower is not None and limits.upper is not None
        defaults.append(limits.lower / 2 + limits.upper / 2)
        accelerations.append(limits.acceleration)
    numbers = [*defaults, *accelerations, threshold]
    for joint in model.joints:
        numbers.extend(joint.origin.position)
        if joint.limits is not None:
            numbers.extend((joint.limits.lower, joint.limits.upper, joint.limits.velocity))
    for sphere in model.collision_spheres:
        numbers.extend((*sphere.center, sphere.radius))
    check_float32(numbers, model.name)
    config = {
        "load_dynamics": False,
        "kinematics": {
            "urdf_path": str(urdf_path),
            "base_link": next(
                link for link in model.links if link not in {j.child for j in model.joints}
            ),
            "tool_frames": list(model.links),
            "collision_link_names": list(spheres),
            "collision_spheres": spheres,
            "collision_sphere_buffer": 0.0,
            "self_collision_ignore": ignores,
            "self_collision_buffer": {link: threshold / 2 for link in spheres},
            "cspace": {
                "joint_names": list(model.joint_order),
                "default_joint_position": defaults,
                "max_acceleration": accelerations,
                "position_limit_clip": 0.0,
                "cspace_distance_weight": [1.0] * len(model.joint_order),
                "null_space_weight": [1.0] * len(model.joint_order),
            },
        },
    }
    return config, restore
