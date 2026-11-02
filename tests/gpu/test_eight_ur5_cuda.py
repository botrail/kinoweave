import json
from dataclasses import replace
from pathlib import Path

import pytest

from examples.eight_ur5 import check_targets, coordination_report
from kinoweave.io import document_digest, load
from kinoweave.manipulation import MultiArmPlanner
from kinoweave.search.reference import JointGraph

pytestmark = pytest.mark.cuda


def test_eight_real_arm_cuda_plan_matches_bundled_cpu_plan(cuda):
    from kinoweave.manipulation.roadmap import fingerprint

    root = Path(__file__).parents[2] / "examples/data"
    original = load(root / "eight_ur5.problem.json")
    baseline = load(root / "eight_ur5.result.json")
    problem = replace(
        original,
        scene=replace(
            original.scene,
            config=replace(
                original.scene.config,
                backend="curobo",
                device="cuda:0",
            ),
        ),
    )
    graphs = []
    for robot in problem.scene.robots:
        vertices = [robot.start]
        for skill in problem.skeleton.skills:
            for track in skill.options[0].trace.tracks:
                if track.robot == robot.name:
                    vertices.extend((track.positions[0], track.positions[-1]))
        graphs.append(
            JointGraph(
                robot=robot.name,
                joint_names=problem.scene.models[0].joint_order,
                vertices=tuple(dict.fromkeys(vertices)),
                edges=(),
                start=0,
                goal=0,
            )
        )
    planner = MultiArmPlanner(problem)
    result = planner.plan(tuple(graphs))
    assert result.status == "success"
    assert result.provenance.backend.name == "curobo"
    assert fingerprint(result.plan) == fingerprint(baseline.plan)
    assert planner.statistics["gpu_pair_batches"] > 0
    metadata = json.loads((root / "eight_ur5.demo.json").read_text())
    metadata["problem_digest"] = document_digest(problem)
    assert check_targets(problem, result, metadata)["targets"] == 40
    report = coordination_report(problem, result, metadata)
    assert report["baseline_status"] == "failed" and report["conflict_graph_connected"]
    assert len(report["conflicting_pairs"]) == 8 and report["delayed_starts"]
