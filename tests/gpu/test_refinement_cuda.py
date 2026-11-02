import json
from dataclasses import asdict, replace

import pytest

from examples.two_arm_handover import build_problem
from kinoweave.io import dump
from kinoweave.manipulation import (
    MultiArmPlanner,
    RefinementConfig,
    refine_manipulation,
    validate_manipulation,
)
from kinoweave.manipulation.refinement_geometry import RefinementGeometry
from kinoweave.manipulation.shortcuts import Proposal, Shortcut, apply_proposal
from kinoweave.roadmap import PRMConfig

pytestmark = pytest.mark.cuda


def test_cuda_shortcut_batches_match_cpu_with_different_grids(cuda, refinement_cases):
    problem, seed = refinement_cases(cooperative=True, weights=(0, 0, 1), backend="curobo")
    edits = tuple(
        Shortcut(
            m.trace.tracks[0].robot,
            m.start,
            m.end,
            m.trace.tracks[0].positions[0],
            m.trace.tracks[0].positions[-1],
            m.end - m.start,
        )
        for m in seed.plan.motions
    )
    choices = (
        Proposal((edits[0],), "single"),
        Proposal((edits[1],), "single"),
        Proposal(edits, "composite"),
    )
    candidates = tuple(replace(seed, plan=apply_proposal(problem, seed.plan, p)) for p in choices)
    expected = tuple(validate_manipulation(problem, c).status == "passed" for c in candidates)
    assert expected == (False, False, True)
    engine = RefinementGeometry(
        problem,
        RefinementConfig(pair_batch_size=2, sphere_tile=1, max_pair_bytes=4096),
        None,
        lambda: None,
    )
    try:
        assert engine.check_many(candidates) == expected
        assert engine.stats["gpu_pair_batches"] > 0
        assert 0 < engine.stats["max_pair_tile_bytes"] <= 4096
        assert engine.evaluator.native_calls > 0
    finally:
        engine.close()


def test_cuda_handover_refinement_is_monotone_and_independently_verified(cuda, tmp_path):
    problem = build_problem(backend="curobo", device="cuda:0")
    seed = MultiArmPlanner(problem, prm=PRMConfig(num_samples=0)).plan()
    assert seed.status == "success"
    report = refine_manipulation(
        problem, seed, config=RefinementConfig(time_limit=60, max_iterations=2, max_candidates=16)
    )
    assert report.result.status == "success" and report.after.total < report.before.total
    assert report.result.provenance.backend.name == "curobo" and report.backend.name == "curobo"
    assert (
        report.statistics["gpu_pair_batches"] > 0 and report.statistics["geometry_native_calls"] > 0
    )
    assert validate_manipulation(problem, report.result).status == "passed"
    assert all(
        b.objective.total < a.objective.total
        for a, b in zip(report.history, report.history[1:], strict=False)
    )
    dump(problem, tmp_path / "problem.json")
    dump(seed, tmp_path / "initial-result.json")
    dump(report.result, tmp_path / "result.json")
    data = asdict(report)
    data.pop("result")
    (tmp_path / "refinement.json").write_text(json.dumps(data, indent=2) + "\n")
