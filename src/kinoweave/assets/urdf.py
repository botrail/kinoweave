"""Explicit URDF 1.0 kinematics, declared proxies, and static link-local visuals."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import ParseError

from .._contracts import ContractError, value_as
from ..models import Joint, JointLimits, LinkSphere, Mimic, Pose, RobotModel
from .manifest import SourceFile, sha256
from .scene import array, mapping, named_numbers, read_yaml


@dataclass(frozen=True)
class ImportedRobot:
    model: RobotModel
    visuals: dict[str, bytes]
    states: dict
    sources: tuple[SourceFile, ...]
    notices: tuple[tuple[str, bytes], ...]


def _xml(data, path):
    try:
        from defusedxml.common import DefusedXmlException
        from defusedxml.ElementTree import fromstring
    except ImportError as error:
        raise ContractError("assets.dependencies", "install kinoweave[urdf]") from error
    try:
        root = fromstring(data, forbid_dtd=True)
    except (DefusedXmlException, ValueError, ParseError) as error:
        raise ContractError(str(path), "invalid or unsafe XML") from error

    def numbers(text, count, label):
        try:
            result = tuple(float(item) for item in text.split())
        except (ValueError, AttributeError) as error:
            raise ContractError(label, "expected numeric attributes") from error
        if len(result) != count or not all(math.isfinite(item) for item in result):
            raise ContractError(label, f"expected {count} finite values")
        return result

    if root.tag != "robot" or root.get("version", "1.0") != "1.0":
        raise ContractError(
            str(path), "only URDF 1.0 is supported; extensions must not be silently dropped"
        )
    if any(
        e.tag not in ("link", "joint", "material", "transmission", "gazebo", "ros2_control")
        for e in root
    ):
        raise ContractError(str(path), "unsupported robot element")
    for link in root.findall("link"):
        if any(e.tag not in ("visual", "collision", "inertial") for e in link):
            raise ContractError(str(path), "unsupported link element")
        for element in (*link.findall("visual"), *link.findall("collision")):
            if len(element.findall("geometry")) != 1 or len(element.findall("origin")) > 1:
                raise ContractError(
                    str(path), "missing/duplicate visual or collision geometry/origin"
                )
    for origin in root.iter("origin"):
        if set(origin.attrib) - {"xyz", "rpy"}:
            raise ContractError(str(path), "unsupported origin attributes")
        for value in origin.attrib.values():
            numbers(value, 3, "urdf.origin")
    for axis in root.iter("axis"):
        mapping(axis.attrib, "urdf.axis", ("xyz",), ("xyz",))
        if math.hypot(*numbers(axis.get("xyz"), 3, "urdf.axis")) == 0:
            raise ContractError("urdf.axis", "zero axis")
    for limit in root.iter("limit"):
        if set(limit.attrib) - {"lower", "upper", "velocity", "effort"}:
            raise ContractError(str(path), "unsupported joint limit attributes")
        for value in limit.attrib.values():
            numbers(value, 1, "urdf.limit")
    for geometry in root.iter("geometry"):
        children = list(geometry)
        if len(children) != 1 or children[0].tag not in ("box", "sphere", "cylinder", "mesh"):
            raise ContractError(str(path), "unsupported or ambiguous geometry")
        shape = children[0]
        attributes = {
            "box": ("size",),
            "sphere": ("radius",),
            "cylinder": ("radius", "length"),
            "mesh": ("filename", "scale"),
        }[shape.tag]
        required = ("filename",) if shape.tag == "mesh" else attributes
        mapping(shape.attrib, "urdf.geometry", attributes, required)
        for key, value in shape.attrib.items():
            if key != "filename":
                if min(numbers(value, 3 if key in ("size", "scale") else 1, "urdf.geometry")) <= 0:
                    raise ContractError("urdf.geometry", "positive dimensions/scale required")
    for joint in root.findall("joint"):
        if joint.get("type") not in ("fixed", "revolute", "prismatic", "continuous"):
            raise ContractError(str(path), f"unsupported joint {joint.get('name')!r}")
        for tag in ("parent", "child", "origin", "axis", "limit", "mimic"):
            if len(joint.findall(tag)) > 1:
                raise ContractError(str(path), f"duplicate joint element {tag}")
    if any(isinstance(e.tag, str) and e.tag.startswith("{") for e in root.iter()):
        raise ContractError(str(path), "expand xacro/namespaced elements before loading")
    return root


def load_robot_asset(path: str | Path) -> ImportedRobot:
    try:
        import numpy as np
        import trimesh
        import yourdfpy
    except ImportError as error:
        raise ContractError("assets.dependencies", "install kinoweave[urdf]") from error
    path = Path(path).resolve()
    spec_bytes = path.read_bytes()
    spec = mapping(
        read_yaml(path, data=spec_bytes),
        "robot_asset",
        (
            "format",
            "schema_version",
            "name",
            "urdf",
            "packages",
            "joint_order",
            "joint_limits",
            "locked_joints",
            "states",
            "collision_spheres",
            "allowed_collisions",
            "notices",
        ),
        (
            "format",
            "schema_version",
            "name",
            "urdf",
            "joint_order",
            "joint_limits",
            "collision_spheres",
        ),
    )
    value_as(spec["format"], Literal["kinoweave.robot"], "robot_asset.format")
    value_as(spec["schema_version"], Literal[1], "robot_asset.schema_version")
    name = value_as(spec["name"], str, "robot_asset.name")
    sources = {f"{name}:configuration": sha256(spec_bytes)}

    def read(file, label):
        data = file.read_bytes()
        digest = sha256(data)
        if label in sources and sources[label] != digest:
            raise ContractError(label, "source changed while reading")
        sources[label] = digest
        return data

    urdf_path = (path.parent / value_as(spec["urdf"], str, "robot_asset.urdf")).resolve()
    urdf_bytes = read(urdf_path, f"{name}:urdf")
    root = _xml(urdf_bytes, urdf_path)
    packages = {
        key: (path.parent / value_as(value, str, f"packages.{key}")).resolve()
        for key, value in mapping(spec.get("packages", {}), "packages").items()
    }

    def resolve(filename):
        filename = value_as(filename, str, "mesh.filename")
        if filename.startswith("package://"):
            package, separator, relative = filename.removeprefix("package://").partition("/")
            if not separator or package not in packages:
                raise ContractError("mesh.filename", f"unmapped package URI {filename!r}")
            result = (packages[package] / relative).resolve()
            if not result.is_relative_to(packages[package]):
                raise ContractError("mesh.filename", "package path escapes its root")
        elif "://" in filename:
            raise ContractError(
                "mesh.filename", "only local files and explicit package mappings are supported"
            )
        else:
            result = (urdf_path.parent / filename).resolve()
        if not result.is_file():
            raise ContractError("mesh.filename", f"missing asset {filename!r}")
        read(result, f"{name}:mesh:{filename}")
        return result

    # Resolve both visual and collision resources, even though native planning
    # uses explicit spheres. A missing source collision mesh is not discarded.
    for mesh in root.iter("mesh"):
        mapping(mesh.attrib, "mesh", ("filename", "scale"), ("filename",))
        resolve(mesh.get("filename"))
    try:
        parsed = yourdfpy.URDF.load(
            io.BytesIO(urdf_bytes),
            load_meshes=False,
            build_scene_graph=False,
            build_collision_scene_graph=False,
        )
    except Exception as error:
        raise ContractError("urdf", f"parse failed: {error}") from error
    by_name = {j.name: j for j in parsed.robot.joints}
    if len(by_name) != len(parsed.robot.joints):
        raise ContractError("urdf.joints", "duplicate names")
    for joint in parsed.robot.joints:
        chain, current = set(), joint
        while current.mimic is not None:
            if current.name in chain or current.mimic.joint not in by_name:
                raise ContractError("urdf.mimic", "cyclic or unknown source")
            chain.add(current.name)
            source = by_name[current.mimic.joint]
            if source.type == "fixed" or (source.type == "prismatic") != (
                current.type == "prismatic"
            ):
                raise ContractError("urdf.mimic", "incompatible source")
            current = source
    overrides = mapping(spec["joint_limits"], "joint_limits")
    locked = named_numbers(spec.get("locked_joints", {}), "locked_joints")
    if set(overrides) - by_name.keys() or set(locked) - by_name.keys():
        raise ContractError("joint_settings", "unknown joint name")
    for joint_name, override in overrides.items():
        if by_name[joint_name].type == "fixed":
            raise ContractError("joint_limits", "fixed joints have no motion limits")
        values = named_numbers(
            mapping(
                override,
                f"joint_limits.{joint_name}",
                ("velocity", "acceleration", "lower", "upper"),
            ),
            f"joint_limits.{joint_name}",
        )
        limit = by_name[joint_name].limit
        if "velocity" in values and (
            values["velocity"] <= 0
            or limit is None
            or limit.velocity is None
            or values["velocity"] > limit.velocity
        ):
            raise ContractError(
                "joint_limits.velocity", "positive overrides cannot loosen source limits"
            )
        if "acceleration" in values and values["acceleration"] <= 0:
            raise ContractError("joint_limits.acceleration", "positive acceleration required")
        if by_name[joint_name].type == "continuous" and {"lower", "upper"} & values.keys():
            raise ContractError(
                "joint_limits", "continuous joints cannot acquire bounds implicitly"
            )
        for key in ("lower", "upper"):
            if key in values and (
                limit is None
                or getattr(limit, key) is None
                or (values[key] < limit.lower if key == "lower" else values[key] > limit.upper)
            ):
                raise ContractError("joint_limits", "override cannot loosen source bounds")
        if joint_name in locked and any(
            (
                key in values
                and (
                    locked[joint_name] < values[key]
                    if key == "lower"
                    else locked[joint_name] > values[key]
                )
            )
            for key in ("lower", "upper")
        ):
            raise ContractError("locked_joints", "locked value violates overridden bounds")
    for joint_name in locked:
        if by_name[joint_name].type == "fixed":
            raise ContractError("locked_joints", "fixed joints have no coordinate")
        if by_name[joint_name].mimic is not None:
            raise ContractError("locked_joints", "lock a mimic's independent source instead")
    for _ in parsed.robot.joints:
        for joint in parsed.robot.joints:
            if joint.mimic is not None and joint.mimic.joint in locked:
                multiplier = 1.0 if joint.mimic.multiplier is None else joint.mimic.multiplier
                offset = 0.0 if joint.mimic.offset is None else joint.mimic.offset
                locked[joint.name] = value_as(
                    multiplier * locked[joint.mimic.joint] + offset, float, "locked_mimic"
                )

    def pose(matrix):
        w, x, y, z = trimesh.transformations.quaternion_from_matrix(matrix)
        return Pose(
            position=tuple(matrix[:3, 3]), quaternion=(float(x), float(y), float(z), float(w))
        )

    joints = []
    for source in parsed.robot.joints:
        origin = np.eye(4) if source.origin is None else np.array(source.origin, dtype=float)
        axis = value_as(
            tuple(source.axis) if source.axis is not None else (1, 0, 0),
            tuple[float, float, float],
            f"joints.{source.name}.axis",
        )
        if not math.isclose(math.hypot(*axis), 1, abs_tol=1e-6, rel_tol=0):
            raise ContractError(f"joints.{source.name}.axis", "expected a unit axis")
        if source.name in locked:
            q = locked[source.name]
            if source.type != "continuous" and (
                source.limit is None
                or source.limit.lower is None
                or source.limit.upper is None
                or not source.limit.lower <= q <= source.limit.upper
            ):
                raise ContractError(f"locked_joints.{source.name}", "outside source bounds")
            motion = np.eye(4)
            if source.type == "prismatic":
                motion[:3, 3] = np.asarray(axis) * q
            else:
                motion = trimesh.transformations.rotation_matrix(q, axis)
            origin = origin @ motion
            kind, limits, mimic = "fixed", None, None
        elif source.type == "fixed":
            kind, limits, mimic = "fixed", None, None
        else:
            raw = mapping(
                overrides.get(source.name, {}),
                f"joint_limits.{source.name}",
                ("velocity", "acceleration", "lower", "upper"),
                ("acceleration",),
            )
            values = named_numbers(raw, f"joint_limits.{source.name}")
            if source.limit is None or source.limit.velocity is None:
                raise ContractError(
                    f"joints.{source.name}.velocity", "explicit source velocity required"
                )
            velocity = values.get("velocity", source.limit.velocity)
            if velocity > source.limit.velocity:
                raise ContractError(
                    f"joints.{source.name}.velocity", "override cannot loosen source limit"
                )
            lower, upper = source.limit.lower, source.limit.upper
            if source.type == "continuous":
                if "lower" in values or "upper" in values:
                    raise ContractError(
                        "joint_limits", "continuous joints cannot acquire bounds implicitly"
                    )
                lower = upper = None
            else:
                if lower is None or upper is None:
                    raise ContractError(f"joints.{source.name}", "source position bounds required")
                if values.get("lower", lower) < lower or values.get("upper", upper) > upper:
                    raise ContractError("joint_limits", "override cannot loosen source bounds")
                lower, upper = values.get("lower", lower), values.get("upper", upper)
            limits = JointLimits(
                velocity=velocity, acceleration=values["acceleration"], lower=lower, upper=upper
            )
            kind = source.type
            mimic = (
                None
                if source.mimic is None
                else Mimic(
                    joint=source.mimic.joint,
                    multiplier=1.0 if source.mimic.multiplier is None else source.mimic.multiplier,
                    offset=0.0 if source.mimic.offset is None else source.mimic.offset,
                )
            )
        joints.append(
            Joint(
                name=source.name,
                kind=kind,
                parent=source.parent,
                child=source.child,
                origin=pose(origin),
                axis=axis,
                limits=limits,
                mimic=mimic,
            )
        )

    proxy_path = (
        path.parent / value_as(spec["collision_spheres"], str, "collision_spheres")
    ).resolve()
    proxy_bytes = read(proxy_path, f"{name}:collision_spheres")
    proxy = mapping(
        read_yaml(proxy_path, data=proxy_bytes), "collision_spheres", ("spheres",), ("spheres",)
    )
    spheres = tuple(
        value_as(s, LinkSphere, "collision_spheres[]", decode=True)
        for s in array(proxy["spheres"], "spheres")
    )
    covered = {s.link for s in spheres}
    missing = {link.name for link in parsed.robot.links if link.collisions} - covered
    if missing:
        raise ContractError(
            "collision_spheres", f"source collision links have no proxy: {sorted(missing)}"
        )

    visuals = {}
    material_map = {m.name: m for m in parsed.robot.materials}
    for link in parsed.robot.links:
        scene = trimesh.Scene()
        for i, visual in enumerate(link.visuals):
            if visual.origin is None:
                visual.origin = np.eye(4)
            geometry = visual.geometry
            if geometry.mesh is not None:
                file = resolve(geometry.mesh.filename)
                loaded = trimesh.load_scene(file, allow_remote=False)
                scale = np.asarray(
                    (1, 1, 1) if geometry.mesh.scale is None else geometry.mesh.scale, dtype=float
                )
                if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                    raise ContractError("visual.mesh.scale", "positive finite xyz scale required")
                transform = np.asarray(visual.origin) @ np.diag((*scale, 1.0))
            elif geometry.box is not None:
                dimensions = value_as(
                    tuple(geometry.box.size), tuple[float, float, float], "visual.box"
                )
                if min(dimensions) <= 0:
                    raise ContractError("visual.box", "positive dimensions required")
                loaded, transform = (
                    trimesh.Scene(trimesh.creation.box(extents=dimensions)),
                    visual.origin,
                )
            elif geometry.sphere is not None:
                radius = value_as(geometry.sphere.radius, float, "visual.sphere.radius")
                if radius <= 0:
                    raise ContractError("visual.sphere", "positive radius required")
                loaded, transform = (
                    trimesh.Scene(trimesh.creation.icosphere(subdivisions=2, radius=radius)),
                    visual.origin,
                )
            else:
                radius = value_as(geometry.cylinder.radius, float, "visual.cylinder.radius")
                length = value_as(geometry.cylinder.length, float, "visual.cylinder.length")
                if min(radius, length) <= 0:
                    raise ContractError("visual.cylinder", "positive dimensions required")
                loaded, transform = (
                    trimesh.Scene(trimesh.creation.cylinder(radius=radius, height=length)),
                    visual.origin,
                )
            if not loaded.graph.nodes_geometry:
                raise ContractError("visual.mesh", "visual has no triangle geometry")
            material = visual.material
            if material is not None and material.name in material_map:
                material = material_map[material.name]
            if material is not None and material.texture is not None:
                raise ContractError(
                    "visual.material",
                    "external URDF texture overrides are unsupported; use a textured mesh",
                )
            for n, node in enumerate(loaded.graph.nodes_geometry):
                local, key = loaded.graph[node]
                mesh = loaded.geometry[key].copy()
                if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
                    raise ContractError("visual.mesh", "finite triangle geometry required")
                mesh.metadata = {}
                if material is not None and material.color is not None:
                    rgba = np.asarray(material.color.rgba)
                    if (
                        rgba.shape != (4,)
                        or not np.isfinite(rgba).all()
                        or (rgba < 0).any()
                        or (rgba > 1).any()
                    ):
                        raise ContractError("visual.material", "expected rgba in [0,1]")
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh, face_colors=np.round(rgba * 255).astype(np.uint8)
                    )
                scene.add_geometry(
                    mesh,
                    node_name=f"visual_{i}_{n}",
                    geom_name=f"geometry_{i}_{n}",
                    transform=np.asarray(transform) @ local,
                )
        if scene.geometry:
            visuals[link.name] = scene.export(file_type="glb")
            sources[f"{name}:compiled_visual:{link.name}"] = sha256(visuals[link.name])
    notices = []
    for filename in array(spec.get("notices", []), "notices"):
        file = (path.parent / value_as(filename, str, "notices[]")).resolve()
        notices.append((file.name, read(file, f"{name}:notice:{filename}")))
    revision = sha256(json.dumps(sorted(sources.items())).encode())
    model = RobotModel(
        name=name,
        revision=revision,
        links=tuple(link.name for link in parsed.robot.links),
        joints=tuple(joints),
        joint_order=tuple(array(spec["joint_order"], "joint_order")),
        collision_spheres=spheres,
        allowed_collisions=tuple(
            tuple(pair) for pair in array(spec.get("allowed_collisions", []), "allowed_collisions")
        ),
    )
    if not model.joint_order:
        raise ContractError(
            "joint_order", "use obstacles for static bodies; a robot needs an independent joint"
        )
    states = mapping(spec.get("states", {}), "states")
    for label, state in states.items():
        values = named_numbers(state, f"states.{label}")
        if set(values) != set(model.joint_order):
            raise ContractError(f"states.{label}", "must cover native independent joints")
    return ImportedRobot(
        model,
        visuals,
        states,
        tuple(SourceFile(name=key, sha256=value) for key, value in sorted(sources.items())),
        tuple(notices),
    )
