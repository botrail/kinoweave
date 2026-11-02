from dataclasses import replace

import pytest

from examples.single_arm_pick_place import build_problem
from kinoweave import Box, Obstacle, Pose, Sphere
from kinoweave.io import dump
from kinoweave.manipulation import (
    ManipulationMode,
    ModeEvaluator,
    ObjectSphere,
    PlacedObject,
    RigidObject,
    SingleArmPlanner,
    validate_manipulation,
)
from kinoweave.roadmap import PRMConfig
from kinoweave.search.reference import JointGraph

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("cache_bytes", [0, 4096])
def test_mode_masks_match_cpu_and_bound_cuda_cache_and_tiles(cuda, cache_bytes):
    problem = build_problem()
    graph = JointGraph(
        robot="gantry",
        joint_names=("x", "y", "z"),
        vertices=((0, 0, 0.45), (0, 0, 0.46), (0.05, 0, 0.45), (0.05, 0, 0.46)),
        edges=((0, 1), (2, 3)),
        start=0,
        goal=1,
    )
    held = ManipulationMode(
        name="held", states=(problem.skeleton.skills[0].options[0].transitions[0].after,)
    )
    gpu_problem = replace(
        problem,
        scene=replace(
            problem.scene, config=replace(problem.scene.config, backend="curobo", device="cuda:0")
        ),
    )
    with (
        ModeEvaluator(problem) as cpu,
        ModeEvaluator(
            gpu_problem,
            batch_size=2,
            sphere_tile=1,
            max_pair_bytes=128,
            max_cache_bytes=cache_bytes,
        ) as gpu,
    ):
        for mode in (problem.initial_mode, held):
            expected, actual = cpu.mask(graph, mode), gpu.mask(graph, mode)
            assert actual.node_flags == expected.node_flags
            assert actual.edge_flags == expected.edge_flags
        assert gpu.info.name == "curobo" and gpu.info.device.startswith("cuda")
        assert gpu.cache_stats["native_calls"] > 0
        assert gpu.cache_stats["bytes"] <= cache_bytes
        assert 0 < gpu.cache_stats["max_pair_tile_bytes"] <= 128
        rows = ((0, 0, 0.55),)
        gpu.node_flags(problem.initial_mode, "gantry", rows)
        calls = gpu.native_calls
        gpu.node_flags(held, "gantry", rows)
        if cache_bytes:
            assert gpu.native_calls == calls
            assert gpu.cache_hits > 0
        else:
            assert gpu.native_calls > calls


def test_cuda_object_transforms_match_rotated_world_frame_and_precision_reset(cuda):
    import math

    problem = build_problem(backend="curobo", device="cuda:0")
    h = math.sqrt(0.5)
    robot = replace(
        problem.scene.robots[0], base_pose=Pose(position=(2, 3, 1), quaternion=(0, 0, h, h))
    )
    problem = replace(problem, scene=replace(problem.scene, robots=(robot,)))
    state = problem.skeleton.skills[0].options[0].transitions[0].after
    grasp = replace(state.grasps[0], pose=Pose(position=(0.1, 0, 0), quaternion=(0, 0, h, h)))
    state = replace(state, grasps=(grasp,))
    obj = replace(
        problem.objects[0], spheres=(replace(problem.objects[0].spheres[0], center=(0.2, 0, 0)),)
    )
    problem = replace(problem, objects=(obj,))
    old_precision = cuda.get_float32_matmul_precision()
    try:
        with ModeEvaluator(problem) as gpu:
            gpu._check_open()
            p, q, _, _ = gpu._device_geometry("gantry", ((0.3, 0.4, 0.5),))
            spheres = gpu._object_spheres(state, "gantry", p, q).tolist()
            assert spheres[0][0][:3] == pytest.approx((1.4, 3.4, 1.5), abs=2e-6)
            calls = gpu.native_calls
            cuda.set_float32_matmul_precision("high" if old_precision != "high" else "highest")
            gpu.node_flags(
                ManipulationMode(name="held", states=(state,)), "gantry", ((0.3, 0.4, 0.5),)
            )
            assert gpu.native_calls > calls
    finally:
        cuda.set_float32_matmul_precision(old_precision)


def test_cuda_generated_pick_place_is_independently_cpu_verified(
    cuda, tmp_path, record_testsuite_property
):
    import json

    problem = build_problem(backend="curobo", device="cuda:0")
    planner = SingleArmPlanner(problem, prm=PRMConfig(num_samples=0, neighbors=8))
    result = planner.plan()
    assert result.status == "success", result
    assert result.provenance.backend.name == "curobo"
    assert validate_manipulation(problem, result).status == "passed"
    dump(problem, tmp_path / "problem.json")
    dump(result, tmp_path / "result.json")
    record_testsuite_property("manipulation_cuda", json.dumps(planner.statistics))


def test_carried_object_checks_other_objects_spheres_and_rotated_boxes(cuda):
    import math

    problem = build_problem()
    marker = RigidObject(name="marker", spheres=(ObjectSphere(center=(0, 0, 0), radius=0.02),))
    placed = PlacedObject(object="marker", pose=Pose(position=(-0.1, 0.3, 0.45)))
    obstacles = (
        *problem.scene.obstacles,
        Obstacle(name="ball", shape=Sphere(radius=0.05), pose=Pose(position=(-0.3, -0.3, 0.4))),
        Obstacle(
            name="rotated",
            shape=Box(size=(0.3, 0.1, 0.3)),
            pose=Pose(
                position=(0.3, 0.3, 0.25),
                quaternion=(0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)),
            ),
        ),
    )
    problem = replace(
        problem,
        scene=replace(problem.scene, obstacles=obstacles),
        objects=(*problem.objects, marker),
        initial_mode=replace(problem.initial_mode, states=(*problem.initial_mode.states, placed)),
        goal_mode=replace(problem.goal_mode, states=(*problem.goal_mode.states, placed)),
    )
    held = ManipulationMode(
        name="held", states=(problem.skeleton.skills[0].options[0].transitions[0].after, placed)
    )
    gpu_problem = replace(
        problem,
        scene=replace(
            problem.scene, config=replace(problem.scene.config, backend="curobo", device="cuda:0")
        ),
    )
    rows = ((-0.1, 0.3, 0.53), (-0.3, -0.3, 0.54), (0.3, 0.3, 0.5), (-0.3, -0.3, 0.65))
    with ModeEvaluator(problem) as cpu, ModeEvaluator(gpu_problem) as gpu:
        assert cpu.node_flags(problem.initial_mode, "gantry", rows) == (True,) * 4
        assert gpu.node_flags(problem.initial_mode, "gantry", rows) == (True,) * 4
        assert cpu.node_flags(held, "gantry", rows) == (False, False, False, True)
        assert gpu.node_flags(held, "gantry", rows) == (False, False, False, True)
