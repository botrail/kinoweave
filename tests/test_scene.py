import io
import json
import shutil
from dataclasses import replace

import pytest

from kinoweave import ContractError
from kinoweave.assets import load_scene, load_visuals, read_manifest, write_scene
from kinoweave.assets.manifest import copy_assets
from kinoweave.assets.urdf import load_robot_asset
from kinoweave.cli import main
from kinoweave.geometry.reference import forward
from kinoweave.io import load

pytestmark = pytest.mark.assets


@pytest.fixture
def authoring(tmp_path, assets):
    import yaml

    urdf = """<robot name="slider">
      <link name="base"><visual><geometry><box size="0.2 0.1 0.1"/></geometry></visual></link>
      <link name="carriage"><visual><origin xyz="0.05 0 0"/>
        <geometry><sphere radius="0.08"/></geometry></visual>
        <collision><origin xyz="0.05 0 0"/>
        <geometry><sphere radius="0.08"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="carriage"/>
        <origin xyz="0.1 0.2 0.3" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/>
        <limit lower="-0.5" upper="0.5" velocity="0.5" effort="10"/></joint></robot>"""
    spec = {
        "format": "kinoweave.robot",
        "schema_version": 1,
        "name": "slider",
        "urdf": "robot.urdf",
        "joint_order": ["slide"],
        "joint_limits": {"slide": {"velocity": 0.4, "acceleration": 1}},
        "collision_spheres": "collision.yaml",
        "states": {"home": {"slide": 0}, "work": {"slide": 0.2}},
    }
    scene = {
        "format": "kinoweave.scene",
        "schema_version": 1,
        "name": "two-sliders",
        "robots": [
            {
                "name": name,
                "asset": "robot.yaml",
                "start": "home",
                "goal": "work",
                "base_pose": {"position": [offset, 0, 0]},
            }
            for name, offset in (("left", -1), ("right", 1))
        ],
        "config": {
            "backend": "reference",
            "device": "cpu",
            "time_step": 2,
            "validation_time_step": 0.1,
        },
    }
    (tmp_path / "robot.urdf").write_text(urdf)
    (tmp_path / "robot.yaml").write_text(yaml.safe_dump(spec))
    (tmp_path / "collision.yaml").write_text(
        yaml.safe_dump({"spheres": [{"link": "carriage", "center": [0.05, 0, 0], "radius": 0.1}]})
    )
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(scene))
    return tmp_path, spec, scene


def test_authoring_compiles_native_problem_and_portable_link_visuals(authoring):
    path, _, _ = authoring
    scene = load_scene(path / "scene.yaml")
    assert len(scene.problem.models) == 1 and len(scene.problem.robots) == 2
    model = scene.problem.models[0]
    assert model.joint_order == ("slide",) and model.joints[0].limits.velocity == 0.4
    assert model.joints[0].limits.acceleration == 1
    assert scene.problem.robots[0].start == (0,) and scene.problem.robots[0].goal == (0.2,)
    assert scene.problem.robots[1].base_pose.position == (1, 0, 0)
    assert set(scene.visuals) == {("slider", "base"), ("slider", "carriage")}
    write_scene(scene, path / "compiled")
    manifest = read_manifest(path / "compiled/assets.manifest.json", scene.problem)
    assert load_visuals(path / "compiled/assets.manifest.json", scene.problem) == scene.visuals
    assert len(manifest.sources) >= 4
    shutil.copytree(path / "compiled", path / "relocated")
    assert (
        load_visuals(path / "relocated/assets.manifest.json", load(path / "relocated/problem.json"))
        == scene.visuals
    )
    assert load_scene(path / "scene.yaml").problem == scene.problem


def test_link_origin_and_mesh_scale_are_preserved(authoring):
    import numpy as np
    import trimesh
    import yaml

    path, spec, _ = authoring
    trimesh.creation.box(extents=(1, 2, 3)).export(path / "box.stl")
    xml = (
        (path / "robot.urdf")
        .read_text()
        .replace(
            '<sphere radius="0.08"/>',
            '<mesh filename="package://test/box.stl" scale="0.1 0.2 0.3"/>',
            1,
        )
        .replace(
            '<origin xyz="0.05 0 0"/>',
            '<origin xyz="0.4 0.5 0.6" rpy="0 0 1.5707963267948966"/>',
            1,
        )
    )
    (path / "robot.urdf").write_text(xml)
    spec["packages"] = {"test": "."}
    (path / "robot.yaml").write_text(yaml.safe_dump(spec))
    loaded = load_robot_asset(path / "robot.yaml")
    visual = trimesh.load_scene(io.BytesIO(loaded.visuals["carriage"]), file_type="glb")
    np.testing.assert_allclose(visual.bounds, [[0.2, 0.45, 0.15], [0.6, 0.55, 1.05]], atol=1e-6)
    from kinoweave import RobotInstance

    poses = forward(
        loaded.model, RobotInstance(name="robot", model="slider", start=(0,), goal=(0,)), (0.2,)
    )
    assert poses.poses[poses.link_names.index("carriage")].position == pytest.approx(
        (0.1, 0.4, 0.3)
    )


def test_locked_mimic_chain_keeps_link_frames(authoring):
    import numpy as np
    import trimesh
    import yaml
    import yourdfpy

    path, spec, _ = authoring
    xml = (
        (path / "robot.urdf")
        .read_text()
        .replace(
            "</robot>",
            """
       <link name="finger"/><joint name="follower" type="prismatic">
         <parent link="carriage"/><child link="finger"/>
         <axis xyz="0 1 0"/><limit lower="-0.5" upper="0.5" velocity="1" effort="10"/>
         <mimic joint="slide" multiplier="-1" offset="0.1"/></joint>
       <link name="tool"/><joint name="hinge" type="revolute">
         <parent link="finger"/><child link="tool"/>
         <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="10"/>
       </joint></robot>""",
        )
    )
    (path / "robot.urdf").write_text(xml)
    spec.update(
        locked_joints={"slide": 0.2},
        joint_order=["hinge"],
        joint_limits={"hinge": {"acceleration": 2}},
        states={"home": {"hinge": 0}},
    )
    (path / "robot.yaml").write_text(yaml.safe_dump(spec))
    robot = load_robot_asset(path / "robot.yaml")
    assert robot.model.joints[0].kind == robot.model.joints[1].kind == "fixed"
    assert len(robot.model.links) == 4
    source = yourdfpy.URDF.load(path / "robot.urdf", load_meshes=False)
    source.update_cfg({"slide": 0.2, "hinge": 0.4})
    from kinoweave import Pose, RobotInstance

    instance = RobotInstance(name="robot", model="slider", start=(0,), goal=(0,), base_pose=Pose())
    native = forward(robot.model, instance, (0.4,))
    for link, pose in zip(native.link_names, native.poses, strict=True):
        matrix = source.get_transform(link, "base")
        np.testing.assert_allclose(pose.position, matrix[:3, 3], atol=1e-12)
        x, y, z, w = pose.quaternion
        np.testing.assert_allclose(
            trimesh.transformations.quaternion_matrix((w, x, y, z))[:3, :3],
            matrix[:3, :3],
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda text: text.replace(
                '<robot name="slider">', '<robot name="slider" version="1.2">'
            ),
            "1.0",
        ),
        (lambda text: text.replace('velocity="0.5"', 'velocity="0.5" acceleration="1"'), "limit"),
        (lambda text: text.replace('velocity="0.5"', 'velocity="nan"'), "finite"),
        (
            lambda text: text.replace('<origin xyz="0.05 0 0"/>', "<origin/><origin/>", 1),
            "duplicate",
        ),
        (lambda text: text.replace('type="prismatic"', 'type="floating"'), "joint"),
        (
            lambda text: text.replace(
                '<sphere radius="0.08"/>', '<capsule radius="0.08" length="0.1"/>'
            ),
            "geometry",
        ),
        (lambda text: '<!DOCTYPE robot [<!ENTITY x "value">]>' + text, "XML"),
    ],
)
def test_unsupported_xml_features_fail_closed(authoring, change, match):
    path, _, _ = authoring
    file = path / "robot.urdf"
    file.write_text(change(file.read_text()))
    with pytest.raises(ContractError, match=match):
        load_scene(path / "scene.yaml")


def test_yaml_unknown_fields_duplicate_names_and_missing_proxies_fail(authoring):
    import yaml

    path, spec, _ = authoring
    (path / "scene.yaml").write_text("name: first\nname: second\n")
    with pytest.raises(ContractError, match="duplicate"):
        load_scene(path / "scene.yaml")
    spec["joint_limts"] = {}
    (path / "robot.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(ContractError, match="unknown"):
        load_robot_asset(path / "robot.yaml")
    spec.pop("joint_limts")
    spec["joint_limits"]["slide"]["velocity"] = 10
    (path / "robot.yaml").write_text(yaml.safe_dump(spec))
    with pytest.raises(ContractError, match="loosen"):
        load_robot_asset(path / "robot.yaml")
    spec["joint_limits"]["slide"]["velocity"] = 0.4
    (path / "robot.yaml").write_text(yaml.safe_dump(spec))
    (path / "collision.yaml").write_text("spheres: []")
    with pytest.raises(ContractError, match="no proxy"):
        load_robot_asset(path / "robot.yaml")


def test_manifest_cannot_attach_to_another_problem_or_changed_blob(authoring):
    path, _, _ = authoring
    scene = load_scene(path / "scene.yaml")
    write_scene(scene, path / "compiled")
    manifest = path / "compiled/assets.manifest.json"
    with pytest.raises(ContractError, match="different problem"):
        load_visuals(manifest, replace(scene.problem, scene_revision="other"))
    changed = replace(
        scene.problem, config=replace(scene.problem.config, backend="curobo", device="cuda:0")
    )
    copy_assets(manifest, scene.problem, path / "copy", changed)
    assert load_visuals(path / "copy/assets.manifest.json", changed) == scene.visuals
    blob = next((path / "compiled/assets").glob("*.glb"))
    blob.write_bytes(b"glTFcorrupted")
    with pytest.raises(ContractError, match="hash"):
        load_visuals(manifest, scene.problem)


def test_cli_import_and_plan_preserve_portable_assets(authoring, capsys):
    path, _, _ = authoring
    assert main(["import-scene", str(path / "scene.yaml"), "--output", str(path / "compiled")]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "kinoweave.problem"
    assert (
        main(
            [
                "plan",
                str(path / "compiled/problem.json"),
                "--samples",
                "0",
                "--output",
                str(path / "run"),
            ]
        )
        == 0
    )
    plan = load(path / "run/result.json")
    problem = load(path / "run/problem.json")
    assert plan.status == "success"
    plan.check_problem(problem)
    assert set(load_visuals(path / "run/assets.manifest.json", problem)) == {
        ("slider", "base"),
        ("slider", "carriage"),
    }
