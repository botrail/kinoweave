import pytest

from examples.task_scheduling import build_domain
from kinoweave.io import dump
from kinoweave.manipulation import (
    ManipulationSearchConfig,
    ScheduleConfig,
    SchedulePlanner,
    validate_manipulation,
)
from kinoweave.roadmap import PRMConfig

pytestmark = pytest.mark.cuda


def test_cuda_catalog_retry_cache_reuse_and_cpu_verified_best(cuda, tmp_path):
    domain = build_domain(backend="curobo", device="cuda:0")
    planner = SchedulePlanner(
        domain,
        config=ScheduleConfig(time_limit=120, candidate_time_limit=30),
        prm=PRMConfig(num_samples=0),
        search_config=ManipulationSearchConfig(
            max_states=512, pair_batch_size=2, sphere_tile=1, max_pair_bytes=4096
        ),
        batch_size=8,
        sphere_tile=1,
        max_pair_bytes=4096,
    )
    report = planner.plan()
    assert report.status == "completed", report.termination_reason
    assert len(report.attempts) == 9
    assert [a.status for a in report.attempts[:3]] == ["no_solution_in_discretization"] * 3
    assert [h.objective.total for h in report.history] == [12, 6]
    assert report.backend.name == report.result.provenance.backend.name == "curobo"
    assert report.statistics["roadmap_builds"] == 1
    assert report.statistics["geometry"]["native_calls"] > 0
    assert report.statistics["geometry"]["hits"] > 0
    assert report.statistics["geometry"]["max_pair_tile_bytes"] <= 4096
    assert any(a.statistics["gpu_pair_batches"] > 0 for a in report.attempts)
    for update in report.history:
        assert validate_manipulation(update.problem, update.result).status == "passed"
    dump(domain, tmp_path / "domain.json")
    dump(report.problem, tmp_path / "problem.json")
    dump(report.result, tmp_path / "result.json")
