import pytest

from examples.many_arm_inspection import InspectionProducer, build_domain
from kinoweave.manipulation import (
    ManipulationSearchConfig,
    ScheduleConfig,
    SchedulePlanner,
    check_candidate,
    validate_manipulation,
)
from kinoweave.roadmap import PRMConfig


@pytest.mark.parametrize("robots", [2, 4, 8])
def test_paired_scale_plan_keeps_all_tasks_and_all_robot_goals(robots):
    domain = build_domain(robots, robots, topology="paired")
    candidates = tuple(InspectionProducer(domain))
    assert len(candidates) == 2
    assert all(check_candidate(domain, c) == c for c in candidates)
    report = SchedulePlanner(
        domain,
        prm=PRMConfig(num_samples=0),
        config=ScheduleConfig(time_limit=60),
        search_config=ManipulationSearchConfig(max_states=512),
    ).plan(InspectionProducer(domain))
    assert report.result.status == "success"
    assert [h.objective.total for h in report.history] == [9, 4.5]
    assert len(report.result.plan.skills) == robots
    assert report.statistics["roadmap_builds"] == 1
    assert 0 <= report.statistics["first_verified_seconds"] <= report.history[0].elapsed
    timers = report.statistics["timing_seconds"]
    assert all(v >= 0 for v in timers.values())
    assert timers["fixed_skeleton"] > 0 and timers["acceptance_cpu_verification"] > 0
    assert validate_manipulation(report.problem, report.result).status == "passed"


def test_shared_and_paired_inputs_remain_distinct():
    shared = build_domain(8, 10)
    paired = build_domain(8, 10, topology="paired")
    assert all(len(t.alternatives) == 8 for t in shared.tasks)
    assert all(len(t.alternatives) == 2 for t in paired.tasks)
    assert shared.scene.robots != paired.scene.robots
    assert build_domain(4, 10, seed=1) == build_domain(4, 10, seed=1)
    assert build_domain(4, 10, seed=0) != build_domain(4, 10, seed=1)
