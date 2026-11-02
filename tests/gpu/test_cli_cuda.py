import json

import pytest

from kinoweave import ExecutionReport, PlanningProblem, PlanResult
from kinoweave.adapters import check_playback_report
from kinoweave.cli import main
from kinoweave.io import load, loads

pytestmark = pytest.mark.cuda


def test_cli_plans_and_replays_the_packaged_arms_with_real_gpu(cuda, tmp_path, capsys):
    input_path = tmp_path / "input.json"
    assert main(["example", "--backend", "curobo", "--output", str(input_path)]) == 0
    capsys.readouterr()
    folder = tmp_path / "run"
    assert main(["plan", str(input_path), "--samples", "0", "--output", str(folder)]) == 0
    plan = loads(capsys.readouterr().out, expected=PlanResult)
    assert plan.provenance.backend.device == "cuda:0"
    assert main(["validate", str(folder / "problem.json"), str(folder / "result.json")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert main(["playback", str(folder / "problem.json"), str(folder / "result.json")]) == 0
    report = loads(capsys.readouterr().out, expected=ExecutionReport)
    problem = load(folder / "problem.json", expected=PlanningProblem)
    check_playback_report(problem, plan, report, position_tolerance=0, velocity_tolerance=0)
