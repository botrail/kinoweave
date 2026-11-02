import copy
import json
import math
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest

from examples.eight_ur5 import check_targets
from examples.eight_ur5_coordination import coordination_report, independent_result
from kinoweave import ValidationReport
from kinoweave.assets import load_visuals
from kinoweave.geometry.reference import obstacle_clearance
from kinoweave.io import document_digest, load
from kinoweave.manipulation import ContactBody, ModeGeometry
from kinoweave.manipulation.geometry import GeometrySphere

DATA = Path(__file__).parents[1] / "examples/data"


@pytest.fixture(scope="module")
def eight_demo():
    return (
        load(DATA / "eight_ur5.problem.json"),
        load(DATA / "eight_ur5.result.json"),
        json.loads((DATA / "eight_ur5.demo.json").read_text()),
    )


def test_bundled_plan_covers_eight_arms_and_forty_surface_welds(eight_demo):
    problem, result, metadata = eight_demo
    result.check_problem(problem)
    report = check_targets(problem, result, metadata)
    assert report["targets"] == 40 and report["robots"] == 8
    assert report["max_position_error_m"] < 5e-5
    assert report["max_surface_error_m"] < 5e-5
    assert report["max_seam_error_m"] < 5e-5
    assert 0.0119 < report["min_nozzle_gap_m"] <= report["max_nozzle_gap_m"] < 0.0121
    assert report["max_orientation_error_rad"] < 1e-4
    assert len(problem.scene.models[0].joint_order) == 6
    assert len(problem.scene.models[0].collision_spheres) > 30
    assert len(result.plan.skills) == 40 and result.makespan >= 75


def test_same_source_paths_collide_without_coordination_and_force_waits(eight_demo):
    problem, result, metadata = eight_demo
    baseline = independent_result(problem, result)
    assert baseline.status == "candidate" and baseline.validation.status == "not_run"
    assert baseline.makespan == 75
    definitions = {s.name: s for s in problem.skeleton.skills}
    assert all(
        definitions[e.before].robots == definitions[e.after].robots
        for e in problem.skeleton.precedence
    )  # No supplied cross-arm serialization can manufacture the solution.
    report = coordination_report(problem, result, metadata)
    assert report["baseline_status"] == "failed"
    assert report["conflict_graph_connected"] and len(report["conflicting_pairs"]) == 8
    assert {r for pair in report["conflicting_pairs"] for r in pair} == {
        r.name for r in problem.scene.robots
    }
    assert all(w["clearance_m"] < -0.05 for w in report["witnesses"])
    assert report["delayed_starts"] and max(report["pending_wait_s_by_robot"].values()) >= 2.5
    assert any(m.kind == "wait" for m in result.plan.motions)
    assert result.validation.status == "passed" and result.makespan < 40 * 15


def test_separated_workcells_cannot_pass_as_the_contention_demo(eight_demo):
    problem, result, metadata = eight_demo
    robots = tuple(
        replace(r, base_pose=replace(r.base_pose, position=(20 * i, 0, r.base_pose.position[2])))
        for i, r in enumerate(problem.scene.robots)
    )
    separated = replace(problem, scene=replace(problem.scene, robots=robots))
    candidate = replace(
        result,
        status="candidate",
        provenance=replace(result.provenance, problem_digest=document_digest(separated)),
        validation=ValidationReport(),
    )
    with pytest.raises(ValueError, match="connect all eight arms"):
        coordination_report(separated, candidate, metadata)


def test_target_metadata_cannot_skip_a_required_skill(eight_demo):
    problem, result, metadata = eight_demo
    forged = copy.deepcopy(metadata)
    forged["targets"][1]["skill"] = forged["targets"][0]["skill"]
    with pytest.raises(ValueError, match="authored workpiece seams"):
        check_targets(problem, result, forged)
    moved = copy.deepcopy(metadata)
    moved["targets"][0]["position"][0] += 0.05
    with pytest.raises(ValueError, match="authored workpiece seams"):
        check_targets(problem, result, moved)


def test_workpiece_definition_must_match_collision_bodies(eight_demo):
    problem, result, metadata = eight_demo
    bodies = tuple(
        replace(o, pose=replace(o.pose, position=(0.50, 0, 0.725)))
        if o.name == "workpiece_plate_0_1"
        else o
        for o in problem.scene.obstacles
    )
    changed = replace(problem, scene=replace(problem.scene, obstacles=bodies))
    meta = {**metadata, "problem_digest": document_digest(changed)}
    with pytest.raises(ValueError, match="checked collision geometry"):
        check_targets(changed, result, meta)


def test_a_quarter_bead_deviation_is_rejected_even_with_valid_endpoints(eight_demo):
    problem, result, metadata = eight_demo
    skill = problem.skeleton.skills[0]
    option = skill.options[0]
    track = option.trace.tracks[0]
    begin, end = metadata["targets"][0]["weld_source"]
    index = option.trace.times.index(begin + (end - begin) / 4)
    rows = list(track.positions)
    changed = list(rows[index])
    changed[0] += 0.01
    rows[index] = tuple(changed)
    trace = replace(option.trace, tracks=(replace(track, positions=tuple(rows)),))
    altered = replace(skill, options=(replace(option, trace=trace),))
    mutated = replace(
        problem, skeleton=replace(problem.skeleton, skills=(altered, *problem.skeleton.skills[1:]))
    )
    meta = {**metadata, "problem_digest": document_digest(mutated)}
    with pytest.raises(ValueError, match="workpiece|authored seam"):
        candidate = replace(
            result,
            status="candidate",
            validation=ValidationReport(),
            provenance=replace(result.provenance, problem_digest=document_digest(mutated)),
        )
        check_targets(mutated, candidate, meta)


def test_portable_demo_meshes_are_hash_bound_without_robot_imports(eight_demo):
    problem, _, _ = eight_demo
    visuals = load_visuals(DATA.parent / "assets/eight_ur5/assets.manifest.json", problem.scene)
    assert len(visuals) == 8 and ("ur5", "weld_tcp") in visuals
    assert all(blob.startswith(b"glTF") for blob in visuals.values())


def brute_collisions(engine, frame):
    """Unpruned scalar oracle, including every source pair and static shape."""
    spheres = [
        GeometrySphere(
            ContactBody(kind="robot", name=name, link=s.link), s.index, s.center, s.radius
        )
        for name, fk in frame.robots
        for s in fk.spheres
    ]
    output = []
    for first, second in combinations(spheres, 2):
        a, b = first.body, second.body
        if a == b:
            continue
        model = engine.models[engine.robots[a.name].model]
        if a.name == b.name and frozenset((a.link, b.link)) in {
            frozenset(p) for p in model.allowed_collisions
        }:
            continue
        clearance = math.dist(first.center, second.center) - first.radius - second.radius
        if engine._collision(a, b, clearance, {}, set()):
            output.append(f"{first.label} / {second.label}: {clearance:.9g} m")
    for sphere in spheres:
        for obstacle in engine.problem.scene.obstacles:
            body = ContactBody(kind="obstacle", name=obstacle.name)
            clearance = obstacle_clearance(sphere, obstacle)
            if engine._collision(sphere.body, body, clearance, {}, set()):
                output.append(f"{sphere.label} / {obstacle.name}: {clearance:.9g} m")
    return output


def test_cached_and_pruned_geometry_matches_unpruned_scalar_oracle(eight_demo):
    problem, _, _ = eight_demo
    first, second, *rest = problem.scene.robots
    overlap = replace(
        problem,
        scene=replace(
            problem.scene, robots=(first, replace(second, base_pose=first.base_pose), *rest)
        ),
    )
    for p in (problem, overlap):
        engine = ModeGeometry(p)
        frame = engine.snapshot(p.initial_mode, {r.name: r.start for r in p.scene.robots})
        expected = brute_collisions(engine, frame)
        assert [i.message for i in frame.issues if i.code == "collision"] == expected
        assert bool(expected) == (p is overlap)


@pytest.mark.assets
def test_source_robot_and_torch_match_the_saved_demo(assets, eight_demo):
    from examples.eight_ur5 import build_demo
    from kinoweave.geometry.reference import _rotate, forward

    problem, _, _ = eight_demo
    rebuilt, _, _, _, _ = build_demo()
    assert rebuilt == problem
    model = problem.scene.models[0]
    robot = problem.scene.robots[0]
    fk = forward(model, robot, robot.start)
    tip = fk.poses[fk.link_names.index("weld_tcp")]
    x, y, z, w = tip.quaternion
    intervals = []
    for sphere in fk.spheres:
        if sphere.link not in ("weld_tcp", "wrist_3_link"):
            continue
        c = _rotate(
            (-x, -y, -z, w), tuple(a - b for a, b in zip(sphere.center, tip.position, strict=True))
        )
        radius_needed = math.hypot(*c[:2]) + 0.012
        if sphere.radius >= radius_needed:
            half = math.sqrt(sphere.radius**2 - radius_needed**2)
            intervals.append((c[2] - half, c[2] + half))
    # Every circular cross-section of the visible cylindrical shank is enclosed.
    covered = -0.18
    for lo, hi in sorted(intervals):
        if lo <= covered + 1e-10:
            covered = max(covered, hi)
    assert covered >= -0.035
    tip_sphere = next(
        s for s in model.collision_spheres if s.link == "weld_tcp" and s.center[2] == -0.035
    )
    assert math.hypot(0.012, 0) <= tip_sphere.radius
    assert math.hypot(0.035 - 0.012, 0) <= tip_sphere.radius
    assert (
        max(s.center[2] + s.radius for s in model.collision_spheres if s.link == "weld_tcp")
        < -0.005
    )
