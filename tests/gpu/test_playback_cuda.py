from dataclasses import replace

import pytest

from kinoweave import Planner, SearchConfig
from kinoweave.adapters import MemoryPlayback, check_playback_report, take_snapshot
from kinoweave.io import dumps

pytestmark = pytest.mark.cuda


def test_gpu_plan_handoff_to_sdk_independent_playback(
    coordination_cases, cuda, record_testsuite_property
):
    problem, graphs = coordination_cases["retreat"]
    problem = replace(
        problem, config=replace(problem.config, backend="curobo", device="cuda:0", time_limit=30)
    )
    with Planner(problem, search=SearchConfig(algorithm="auto", priority_horizon=6)) as planner:
        plan = planner.plan(graphs)
    assert plan.status == "success" and plan.provenance.backend.device == "cuda:0"
    with MemoryPlayback(problem) as player:
        player.load(problem, plan)
        sample = player.set_time(0.5)
        assert sample.time == sample.reference_time == 0.5
        assert sample.robots[0].positions == (-1, 0)
        assert sample.robots[1].positions == pytest.approx((0.5, 0))
        assert sample.robots[1].velocities == pytest.approx((-1.875, 0))
        frame = player.frame
        report = player.record(0.3)
        assert frame.observation == sample
        assert len(report.observations) == 17 and report.time_step == 0.25
        assert report.runtime.device == "cpu" and report.mode == "kinematic_playback"
        assert tuple(r.start for r in take_snapshot(player).robots) == tuple(
            r.goal for r in problem.robots
        )
        check_playback_report(problem, plan, report, position_tolerance=0, velocity_tolerance=0)
    record_testsuite_property("m5_problem", dumps(problem))
    record_testsuite_property("m5_plan", dumps(plan))
    record_testsuite_property("m5_playback", dumps(report))
