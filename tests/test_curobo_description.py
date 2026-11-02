from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from kinoweave import JointLimits, Mimic
from kinoweave.backends._description import UnsupportedModelError, description
from kinoweave.backends.curobo import BackendUnavailableError, CuroboBackend


def test_kinematic_export_preserves_explicit_geometry_and_limits(problem, tmp_path):
    model = problem.models[0]
    path = tmp_path / "robot.urdf"
    data, restore = description(model, path, 0.01)
    joint = ET.parse(path).find("joint")
    assert joint.attrib == {"name": "slide", "type": "prismatic"}
    assert joint.find("axis").get("xyz") == "1 0 0"
    assert float(joint.find("limit").get("velocity")) == 0.5
    assert data["load_dynamics"] is False
    assert data["kinematics"]["cspace"]["max_acceleration"] == [1.0]
    assert data["kinematics"]["collision_sphere_buffer"] == 0.0
    assert restore == (0,)
    # The fixed helper duplicates occupied volume without adding an exported sphere.
    spheres = data["kinematics"]["collision_spheres"]["carriage"]
    helper = data["kinematics"]["collision_spheres"]["_kinoweave_sphere_helper"]
    assert helper == spheres
    fixed = ET.parse(path).findall("joint")[1]
    assert fixed.get("type") == "fixed" and fixed.find("parent").get("link") == "carriage"
    assert len(model.collision_spheres) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "continuous", "limits": JointLimits(velocity=1, acceleration=1)},
        {"axis": (2**-0.5, 2**-0.5, 0)},
        {"limits": JointLimits(lower=0, upper=0, velocity=1, acceleration=1)},
    ],
)
def test_upstream_model_changes_are_rejected_before_sdk_loading(problem, tmp_path, change):
    from kinoweave.backends._description import validate_model

    joint = replace(problem.models[0].joints[0], **change)
    model = replace(problem.models[0], joints=(joint,))
    with pytest.raises(UnsupportedModelError, match="slide"):
        validate_model(model)


def test_valid_mimic_model_is_rejected(problem, tmp_path):
    model = problem.models[0]
    follower = replace(
        model.joints[0],
        name="follower",
        parent="carriage",
        child="follower_link",
        mimic=Mimic(joint="slide"),
    )
    model = replace(model, links=(*model.links, "follower_link"), joints=(*model.joints, follower))
    with pytest.raises(UnsupportedModelError, match="mimic"):
        description(model, tmp_path / "robot.urdf", 0.01)


@pytest.mark.parametrize("value", [1e40, 1e-50])
def test_nonrepresentable_geometry_is_rejected(problem, tmp_path, value):
    model = problem.models[0]
    model = replace(model, collision_spheres=(replace(model.collision_spheres[0], radius=value),))
    with pytest.raises(UnsupportedModelError, match="float32"):
        description(model, tmp_path / "robot.urdf", 0.01)


def test_backend_selection_and_device_are_explicit(problem):
    with pytest.raises(ValueError, match="explicitly select"):
        CuroboBackend(problem, "left")
    problem = replace(problem, config=replace(problem.config, backend="curobo", device="cpu"))
    with pytest.raises(BackendUnavailableError, match="CPU fallback is disabled"):
        CuroboBackend(problem, "left")
    with pytest.raises(ValueError, match="unknown robot"):
        CuroboBackend(problem, "unknown")
