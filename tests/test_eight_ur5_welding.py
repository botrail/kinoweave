from dataclasses import replace
from pathlib import Path

import pytest

from examples.eight_ur5_welding import (
    TorchIK,
    check_surface_point,
    read_workpiece,
    weld_targets,
    workpiece_geometry,
)
from kinoweave.io import load


@pytest.fixture
def weld_spec():
    return read_workpiece()


def test_authored_joint_coordinates_do_not_move_with_the_robots(weld_spec):
    problem = load(Path(__file__).parents[1] / "examples/data/eight_ur5.problem.json")
    targets = weld_targets(weld_spec, problem.scene.robots)
    first = targets[0]
    assert first["start_position"] == pytest.approx((0.34, 0.11, 0.679))
    assert first["end_position"] == pytest.approx((0.34, 0.11, 0.687))
    assert first["normal"] == (1, 0, 0)
    assert first["parts"] == ["workpiece_plate_0_1", "workpiece_plate_0_2"]
    moved = tuple(
        replace(r, base_pose=replace(r.base_pose, position=(10, 20, 30)))
        for r in problem.scene.robots
    )
    assert weld_targets(weld_spec, moved) == targets


@pytest.mark.parametrize("index", range(8))
def test_each_authored_seam_joins_two_finite_plate_surfaces(weld_spec, index):
    obstacles, seams = workpiece_geometry(weld_spec)
    seam = seams[index]
    bodies = {o.name: o for o in obstacles}
    assert len(seam["parts"]) == 2
    for point in (seam["start"], seam["origin"], seam["end"]):
        check_surface_point(point, seam["normal"], seam["parts"], bodies)
    floating = tuple(v + 0.005 * n for v, n in zip(seam["origin"], seam["normal"], strict=True))
    with pytest.raises(ValueError, match="workpiece"):
        check_surface_point(floating, seam["normal"], seam["parts"], bodies)
    outside = (*seam["origin"][:2], seam["end"][2] + 0.01)
    with pytest.raises(ValueError, match="finite"):
        check_surface_point(outside, seam["normal"], seam["parts"], bodies)


def test_tool_dimensions_require_matching_geometry(weld_spec):
    with pytest.raises(ValueError, match="matching visual and collision"):
        workpiece_geometry({**weld_spec, "tool_length": 0.25})


@pytest.mark.assets
def test_example_ik_forward_and_jacobian_match_independent_native_fk(assets):
    import numpy as np

    from kinoweave.geometry.reference import _rotate, forward

    problem = load(Path(__file__).parents[1] / "examples/data/eight_ur5.problem.json")
    model, robot = problem.scene.models[0], problem.scene.robots[4]
    solver = TorchIK(model, robot)
    q = np.asarray(robot.start)
    actual, jacobian = solver.forward(q)
    native = forward(model, robot, tuple(q))
    tcp = native.poses[native.link_names.index("weld_tcp")]
    assert np.max(abs(actual - solver.matrix(tcp))) < 1e-12
    epsilon = 1e-6
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = epsilon
        poses = []
        for row in (q + delta, q - delta):
            frame = forward(model, robot, tuple(row))
            poses.append(frame.poses[frame.link_names.index("weld_tcp")])
        position = (np.asarray(poses[0].position) - poses[1].position) / (2 * epsilon)
        assert np.max(abs(position - jacobian[:3, i])) < 1e-8
        rotations = [np.asarray([_rotate(p.quaternion, a) for a in np.eye(3)]).T for p in poses]
        difference = rotations[0] @ rotations[1].T
        angular = np.asarray(
            (
                difference[2, 1] - difference[1, 2],
                difference[0, 2] - difference[2, 0],
                difference[1, 0] - difference[0, 1],
            )
        ) / (4 * epsilon)
        assert np.max(abs(angular - jacobian[3:, i])) < 1e-8
