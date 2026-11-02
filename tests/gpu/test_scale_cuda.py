import pytest

from examples.many_arm_inspection import InspectionProducer, build_domain
from kinoweave.manipulation import ScheduleConfig, SchedulePlanner, validate_manipulation
from kinoweave.roadmap import PRMConfig

pytestmark = pytest.mark.cuda


def test_four_arm_shifted_workcells_are_verified_on_cuda(cuda):
    domain = build_domain(4, 4, topology="paired", backend="curobo")
    report = SchedulePlanner(
        domain,
        prm=PRMConfig(num_samples=0),
        config=ScheduleConfig(time_limit=60, candidate_time_limit=20),
    ).plan(InspectionProducer(domain))
    assert report.result.status == "success" and report.backend.name == "curobo"
    assert [h.objective.total for h in report.history] == [9, 4.5]
    assert report.statistics["geometry"]["native_calls"] > 0
    assert validate_manipulation(report.problem, report.result).status == "passed"
