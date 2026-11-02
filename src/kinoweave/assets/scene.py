"""Strict YAML authoring mapped into existing native contracts and link visuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .._contracts import ContractError, value_as
from ..io import dump
from ..models import Obstacle, PlannerConfig, PlanningProblem, Pose, RobotInstance
from .manifest import SourceFile, sha256, write_manifest


def mapping(value, path, allowed=None, required=()):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(path, "expected a mapping with string keys")
    if allowed is not None and value.keys() - set(allowed):
        raise ContractError(path, f"unknown fields: {sorted(value.keys() - set(allowed))}")
    if set(required) - value.keys():
        raise ContractError(path, f"missing fields: {sorted(set(required) - value.keys())}")
    return value


def array(value, path):
    if not isinstance(value, list):
        raise ContractError(path, "expected an array")
    return value


def named_numbers(value, path):
    return {
        name: value_as(number, float, f"{path}.{name}")
        for name, number in mapping(value, path).items()
    }


def read_yaml(path, *, data=None):
    try:
        import yaml
    except ImportError as error:
        raise ContractError(
            "assets.dependencies", "install kinoweave[urdf] to read YAML/URDF"
        ) from error

    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key, value in node.value:
                name = self.construct_object(key, deep=deep)
                if not isinstance(name, str) or name in result:
                    raise ContractError(str(path), "duplicate or non-string YAML key")
                result[name] = self.construct_object(value, deep=deep)
            return result

    text = (Path(path).read_bytes() if data is None else data).decode("utf-8")
    try:
        if any(isinstance(token, (yaml.AliasToken, yaml.AnchorToken)) for token in yaml.scan(text)):
            raise ContractError(str(path), "YAML aliases and anchors are unsupported")
        return yaml.load(text, Loader=Loader)
    except yaml.YAMLError as error:
        raise ContractError(str(path), "invalid YAML") from error


def pose(value, path):
    value = mapping(value, path, ("position", "quaternion_xyzw"))
    return Pose(
        position=value.get("position", (0, 0, 0)),
        quaternion=value.get("quaternion_xyzw", (0, 0, 0, 1)),
    )


@dataclass(frozen=True)
class LoadedScene:
    problem: PlanningProblem
    visuals: dict[tuple[str, str], bytes]
    sources: tuple[SourceFile, ...]
    notices: tuple[tuple[str, bytes], ...] = ()


def load_scene(path: str | Path) -> LoadedScene:
    from .urdf import load_robot_asset

    path = Path(path).resolve()
    scene_bytes = path.read_bytes()
    document = mapping(
        read_yaml(path, data=scene_bytes),
        "scene",
        ("format", "schema_version", "name", "scene_revision", "robots", "obstacles", "config"),
        ("format", "schema_version", "name", "robots"),
    )
    value_as(document["format"], Literal["kinoweave.scene"], "scene.format")
    value_as(document["schema_version"], Literal[1], "scene.schema_version")
    loaded, models, robots, visuals, notices = {}, {}, [], {}, []
    sources = [SourceFile(name="scene", sha256=sha256(scene_bytes))]
    for entry in array(document["robots"], "scene.robots"):
        entry = mapping(
            entry,
            "scene.robots[]",
            ("name", "asset", "base_pose", "start", "goal"),
            ("name", "asset", "start", "goal"),
        )
        asset_path = (path.parent / value_as(entry["asset"], str, "robot.asset")).resolve()
        if asset_path not in loaded:
            asset = load_robot_asset(asset_path)
            loaded[asset_path] = asset
            if asset.model.name in models and models[asset.model.name] != asset.model:
                raise ContractError("scene.models", "different assets use the same model name")
            models[asset.model.name] = asset.model
            visuals.update({(asset.model.name, link): data for link, data in asset.visuals.items()})
            sources.extend(asset.sources)
            notices.extend(asset.notices)
        asset = loaded[asset_path]

        def state(value, label, asset=asset):
            if isinstance(value, str):
                if value not in asset.states:
                    raise ContractError(f"robot.{label}", f"unknown named state {value!r}")
                value = asset.states[value]
            values = named_numbers(value, f"robot.{label}")
            if set(values) != set(asset.model.joint_order):
                raise ContractError(
                    f"robot.{label}", "must name every independent joint exactly once"
                )
            return tuple(values[name] for name in asset.model.joint_order)

        robots.append(
            RobotInstance(
                name=entry["name"],
                model=asset.model.name,
                base_pose=pose(entry.get("base_pose", {}), "robot.base_pose"),
                start=state(entry["start"], "start"),
                goal=state(entry["goal"], "goal"),
            )
        )
    obstacles = []
    for entry in array(document.get("obstacles", []), "scene.obstacles"):
        entry = mapping(entry, "obstacle", ("name", "shape", "pose"), ("name", "shape"))
        obstacles.append(
            value_as(
                {**entry, "pose": asdict(pose(entry.get("pose", {}), "obstacle.pose"))},
                Obstacle,
                "obstacle",
                decode=True,
            )
        )
    config = value_as(document.get("config", {}), PlannerConfig, "scene.config", decode=True)
    problem = PlanningProblem(
        name=document["name"],
        scene_revision=document.get("scene_revision", "1"),
        models=tuple(models.values()),
        robots=tuple(robots),
        obstacles=tuple(obstacles),
        config=config,
    )
    return LoadedScene(problem, visuals, tuple(sources), tuple(notices))


def write_scene(scene: LoadedScene, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(directory, scene.problem, scene.visuals, scene.sources, scene.notices)
    dump(scene.problem, directory / "problem.json")
    return manifest
