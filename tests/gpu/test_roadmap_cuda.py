from dataclasses import replace

import pytest

from kinoweave import Box, ContractError, LinkSphere, Obstacle, Pose, Sphere
from kinoweave.collision import (
    GeometryCache,
    GpuCollisionEvaluator,
    GpuTiling,
    Motion,
    ReferenceMotionEvaluator,
    reference_interference,
)
from kinoweave.roadmap import PRMConfig, build_prm
from kinoweave.validation import validate_trajectory

pytestmark = pytest.mark.cuda


def test_cache_separates_torch_matmul_precision_modes(scene, cuda):
    original = cuda.get_float32_matmul_precision()
    motion = (Motion(start=(-1, 0), end=(-1, 0)),)
    try:
        cuda.set_float32_matmul_precision("highest")
        with GpuCollisionEvaluator(scene) as evaluator:
            first = evaluator.motion_validity("alpha", motion)
            before = evaluator.cache.stats
            cuda.set_float32_matmul_precision("high")
            assert evaluator.motion_validity("alpha", motion) == first
            assert evaluator.cache.stats.misses > before.misses
            hits = evaluator.cache.stats.hits
            cuda.set_float32_matmul_precision("highest")
            assert evaluator.motion_validity("alpha", motion) == first
            assert evaluator.cache.stats.hits > hits
    finally:
        cuda.set_float32_matmul_precision(original)


@pytest.mark.parametrize("margin", [0.0, 0.125])
def test_inter_robot_contact_and_near_boundary_are_collisions(retreat_problem, cuda, margin):
    model = replace(
        retreat_problem.models[0],
        collision_spheres=(LinkSphere(link="tool", center=(0, 0, 0), radius=0.125),),
    )
    problem = replace(
        retreat_problem,
        models=(model,),
        config=replace(
            retreat_problem.config,
            backend="curobo",
            device="cuda:0",
            collision_margin=margin,
            collision_tolerance=0,
            validation_time_step=1,
        ),
    )
    first = (Motion(start=(0, 0), end=(0, 0)),)
    second = tuple(
        Motion(start=(x, 0), end=(x, 0))
        for x in (0.25 + margin - 0.0001, 0.25 + margin, 0.25 + margin + 0.0001)
    )
    expected = reference_interference(problem, "alpha", first, "beta", second)
    assert expected.compatible == ((False, False, True),)
    with GpuCollisionEvaluator(
        problem, tiling=GpuTiling(pair_batch_size=1, sphere_batch_size=1)
    ) as evaluator:
        assert (
            evaluator.interference("alpha", first, "beta", second).compatible == expected.compatible
        )


def test_gpu_roadmap_example_outputs_verified_geometry(cuda, capsys, record_testsuite_property):
    import json
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).parents[2] / "examples/gpu_roadmap.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["stage"] == "individual_roadmaps_and_interference"
    assert all(r["validation"]["status"] == "passed" for r in result["roadmaps"].values())
    assert result["table"]["compatible"] and result["cache"]["hits"] > 0
    record_testsuite_property("m3_gpu_example", json.dumps(result))


def test_gpu_prm_subdivides_into_checked_stops(problem, cuda):
    gpu_problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_step=0.5)
    )
    with GpuCollisionEvaluator(gpu_problem) as evaluator:
        roadmap = build_prm(evaluator, "left", config=PRMConfig(num_samples=0))
        assert len(roadmap.shortest_path()) == 8
        single = replace(gpu_problem, robots=gpu_problem.robots[:1])
        assert validate_trajectory(single, roadmap.trajectory()).status == "passed"


def test_two_seven_dof_models_with_different_sphere_counts(retreat_problem, cuda):
    import random

    from kinoweave import Joint, JointLimits, RobotInstance, RobotModel

    joints = tuple(
        Joint(
            name=f"j{i}",
            kind="revolute",
            parent=f"l{i}",
            child=f"l{i + 1}",
            origin=Pose(position=(0.16, 0, 0.05)),
            axis=((0, 0, 1), (0, 1, 0), (1, 0, 0))[i % 3],
            limits=JointLimits(lower=-1, upper=1, velocity=3, acceleration=8),
        )
        for i in range(7)
    )
    first_model = RobotModel(
        name="arm_a",
        links=tuple(f"l{i}" for i in range(8)),
        joints=joints,
        joint_order=tuple(j.name for j in reversed(joints)),
        collision_spheres=tuple(
            LinkSphere(link=f"l{i}", center=(0, 0, 0), radius=0.07) for i in (2, 4, 7)
        ),
    )
    second_model = replace(
        first_model,
        name="arm_b",
        collision_spheres=(
            *first_model.collision_spheres,
            LinkSphere(link="l4", center=(0.03, 0, 0.04), radius=0.06),
        ),
    )
    robots = (
        RobotInstance(
            name="a",
            model="arm_a",
            start=(0,) * 7,
            goal=(0.1,) * 7,
            base_pose=Pose(quaternion=(0, 0, 0.2588190451, 0.9659258263)),
        ),
        RobotInstance(
            name="b",
            model="arm_b",
            start=(0,) * 7,
            goal=(0.1,) * 7,
            base_pose=Pose(position=(1, 0.15, 0), quaternion=(0, 0, 1, 0)),
        ),
    )
    problem = replace(
        retreat_problem,
        models=(first_model, second_model),
        robots=robots,
        config=replace(
            retreat_problem.config, backend="curobo", device="cuda:0", validation_time_step=0.125
        ),
    )
    rng = random.Random(17)

    def motions():
        return tuple(
            Motion(
                start=tuple(rng.uniform(-0.6, 0.6) for _ in range(7)),
                end=tuple(rng.uniform(-0.6, 0.6) for _ in range(7)),
            )
            for _ in range(4)
        )

    first, second = motions(), motions()
    expected = reference_interference(problem, "a", first, "b", second)
    assert any(value for row in expected.compatible for value in row)
    assert any(not value for row in expected.compatible for value in row)
    for tiling in (
        GpuTiling(),
        GpuTiling(
            motion_batch_size=1,
            time_batch_size=2,
            pair_batch_size=3,
            sphere_batch_size=1,
            max_pair_bytes=1024,
        ),
    ):
        with GpuCollisionEvaluator(problem, tiling=tiling) as evaluator:
            actual = evaluator.interference("a", first, "b", second)
            assert actual.compatible == expected.compatible
            assert actual.first_valid == expected.first_valid
            assert actual.second_valid == expected.second_valid


@pytest.fixture
def scene(retreat_problem, cuda):
    model = retreat_problem.models[0]
    model = replace(
        model,
        joints=tuple(replace(j, limits=replace(j.limits, acceleration=20)) for j in model.joints),
        collision_spheres=(
            LinkSphere(link="tool", center=(0, 0, 0), radius=0.18),
            LinkSphere(link="tool", center=(0, 0, 0.25), radius=0.08),
            LinkSphere(link="tool", center=(0.05, 0, -0.25), radius=0.06),
        ),
    )
    return replace(
        retreat_problem,
        models=(model,),
        obstacles=(
            Obstacle(
                name="block",
                shape=Box(size=(0.2, 0.2, 0.2)),
                pose=Pose(position=(0, 0.7, 0), quaternion=(0, 0, 0.2588190451, 0.9659258263)),
            ),
        ),
        config=replace(
            retreat_problem.config, backend="curobo", device="cuda:0", validation_time_step=0.125
        ),
    )


def candidates():
    first = (
        Motion(start=(-1, 0), end=(-1, 0)),
        Motion(start=(-1, 0), end=(1, 0)),
        Motion(start=(-1, 0), end=(0, 0)),
        Motion(start=(0, 0), end=(0, 1)),
        Motion(start=(0, 0), end=(2.1, 0)),
    )
    second = (
        Motion(start=(1, 0), end=(1, 0)),
        Motion(start=(1, 0), end=(-1, 0)),
        Motion(start=(1, 0), end=(0, 0)),
        Motion(start=(0, 1), end=(0, 0)),
    )
    return first, second


@pytest.mark.parametrize(
    "tiling",
    [
        GpuTiling(motion_batch_size=1, time_batch_size=1, pair_batch_size=1, sphere_batch_size=1),
        GpuTiling(motion_batch_size=3, time_batch_size=4, pair_batch_size=5, sphere_batch_size=2),
        GpuTiling(max_pair_bytes=4096, max_geometry_bytes=65536),
    ],
)
@pytest.mark.parametrize("cache_bytes", [0, 1_000_000])
def test_tiled_interference_matches_cpu_and_cache_is_semantically_transparent(
    scene, cuda, tiling, cache_bytes
):
    first, second = candidates()
    expected = reference_interference(scene, "alpha", first, "beta", second)
    cache = GeometryCache(cache_bytes)
    with GpuCollisionEvaluator(scene, tiling=tiling, cache=cache) as evaluator:
        actual = evaluator.interference("alpha", first, "beta", second)
        assert actual.compatible == expected.compatible
        assert actual.first_valid == expected.first_valid
        assert actual.second_valid == expected.second_valid
        assert actual.first_valid[-1] is False  # explicit position/rate limits
        assert actual.first_valid[3] is False  # static obstacle along the sweep
        assert actual.compatible[0][0] is True and actual.compatible[1][1] is False
        again = evaluator.interference("alpha", first, "beta", second)
        assert again == actual
        reverse = evaluator.interference("beta", second, "alpha", first)
        assert reverse.compatible == tuple(zip(*actual.compatible, strict=True))
        assert evaluator.max_pair_tile_bytes <= tiling.max_pair_bytes
        assert evaluator.max_geometry_tile_bytes <= tiling.max_geometry_bytes
        assert cache.stats.bytes <= cache_bytes
        if cache_bytes:
            assert cache.stats.hits > 0
        else:
            assert cache.stats.entries == cache.stats.bytes == 0


@pytest.mark.parametrize(
    "change",
    [
        "base",
        "geometry",
        "environment",
        "scene_revision",
        "margin",
        "time_step",
        "sample_step",
        "model_revision",
        "limits",
    ],
)
def test_shared_cache_is_invalidated_by_problem_changes(scene, cuda, change):
    cache = GeometryCache(1_000_000)
    motion = (Motion(start=(-1, 0), end=(-1, 0)),)
    with GpuCollisionEvaluator(scene, cache=cache) as evaluator:
        assert evaluator.motion_validity("alpha", motion) == (True,)
        evaluator.motion_validity("alpha", motion)
    before = cache.stats
    if change == "base":
        changed = replace(
            scene,
            robots=(
                replace(scene.robots[0], base_pose=Pose(position=(0.2, 0.1, 0))),
                scene.robots[1],
            ),
        )
    elif change == "geometry":
        model = replace(
            scene.models[0],
            collision_spheres=tuple(
                replace(s, radius=s.radius * 1.1) for s in scene.models[0].collision_spheres
            ),
        )
        changed = replace(scene, models=(model,))
    elif change == "environment":
        changed = replace(
            scene, obstacles=(replace(scene.obstacles[0], pose=Pose(position=(-1, 0, 0))),)
        )
    elif change == "scene_revision":
        changed = replace(scene, scene_revision="2")
    elif change == "model_revision":
        changed = replace(scene, models=(replace(scene.models[0], revision="2"),))
    elif change == "limits":
        model = replace(
            scene.models[0],
            joints=tuple(
                replace(j, limits=replace(j.limits, velocity=6)) for j in scene.models[0].joints
            ),
        )
        changed = replace(scene, models=(model,))
    else:
        field, value = {
            "margin": ("collision_margin", 0.1),
            "time_step": ("time_step", 0.5),
            "sample_step": ("validation_time_step", 0.25),
        }[change]
        changed = replace(scene, config=replace(scene.config, **{field: value}))
    with GpuCollisionEvaluator(changed, cache=cache) as evaluator:
        actual = evaluator.motion_validity("alpha", motion)
        oracle = reference_interference(changed, "alpha", motion, "beta", ())
        assert actual == oracle.first_valid
        if change == "environment":
            assert actual == (False,)
    assert cache.stats.misses > before.misses
    assert cache.stats.hits == before.hits


def test_lru_eviction_and_resource_failures_do_not_mark_unknown_as_clear(scene, cuda):
    motions = tuple(Motion(start=(x, 1.5), end=(x, 1.5)) for x in (-1, -0.5, 0, 0.5, 1))
    cache = GeometryCache(500)
    with GpuCollisionEvaluator(scene, cache=cache) as evaluator:
        first = evaluator.motion_validity("alpha", motions)
        assert evaluator.motion_validity("alpha", motions) == first
        assert cache.stats.evictions > 0 and cache.stats.bytes <= 500
        cache.clear()
        assert cache.stats.bytes == 0
    with GpuCollisionEvaluator(scene, tiling=GpuTiling(max_table_pairs=1)) as evaluator:
        with pytest.raises(ContractError, match="pair budget"):
            evaluator.interference("alpha", motions, "beta", motions)
        assert evaluator.interference("alpha", (), "beta", ()).compatible == ()
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.motion_validity("alpha", motions)
    with GpuCollisionEvaluator(scene, tiling=GpuTiling(max_geometry_bytes=1)) as evaluator:
        with pytest.raises(ContractError, match="geometry tile"):
            evaluator.motion_validity("alpha", motions)


def test_gpu_prm_matches_cpu_and_its_single_robot_path_revalidates(scene, cuda):
    problem = replace(
        scene,
        robots=scene.robots[:1],
        obstacles=(Obstacle(name="block", shape=Sphere(radius=0.2), pose=Pose()),),
    )
    config = PRMConfig(num_samples=0, neighbors=2)
    samples = ((-1, 1), (1, 1))
    cpu_problem = replace(
        problem, config=replace(problem.config, backend="reference", device="cpu")
    )
    expected = build_prm(
        ReferenceMotionEvaluator(cpu_problem), "alpha", config=config, samples=samples
    )
    cache = GeometryCache(1_000_000)
    graphs = []
    for tiling in (GpuTiling(motion_batch_size=1, time_batch_size=2), GpuTiling()):
        with GpuCollisionEvaluator(problem, tiling=tiling, cache=cache) as evaluator:
            roadmap = build_prm(evaluator, "alpha", config=config, samples=samples)
            graphs.append(roadmap.graph)
            assert roadmap.backend.device.startswith("cuda")
            assert roadmap.graph == expected.graph
            assert roadmap.shortest_path() is not None
            assert validate_trajectory(problem, roadmap.trajectory()).status == "passed"
    assert graphs[0] == graphs[1]


def test_seeded_gpu_prm_reuses_cache_without_changing_topology(scene, cuda):
    problem = replace(scene, robots=scene.robots[:1], obstacles=())
    config = PRMConfig(num_samples=12, neighbors=3)
    with GpuCollisionEvaluator(problem) as evaluator:
        first = build_prm(evaluator, "alpha", config=config)
        before = evaluator.cache.stats
        second = build_prm(evaluator, "alpha", config=config)
        assert first == second and first.shortest_path() is not None
        assert evaluator.cache.stats.hits > before.hits
        cpu = replace(problem, config=replace(problem.config, backend="reference", device="cpu"))
        assert first.graph == build_prm(ReferenceMotionEvaluator(cpu), "alpha", config=config).graph
