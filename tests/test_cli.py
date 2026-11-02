import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kinoweave import ExecutionReport, PlanningProblem, PlanResult
from kinoweave.adapters import check_playback_report
from kinoweave.cli import main
from kinoweave.io import dump, load, loads


def test_installed_module_can_create_plan_validate_and_replay(tmp_path):
    def cli(*args):
        process = subprocess.run(
            [sys.executable, "-m", "kinoweave", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        return json.loads(process.stdout)

    cli("example", "--output", tmp_path / "input.json")
    problem = load(tmp_path / "input.json", expected=PlanningProblem)
    assert tuple(len(m.joint_order) for m in problem.models) == (7, 6)
    result = cli("plan", tmp_path / "input.json", "--output", tmp_path / "run", "--samples", 0)
    assert result["status"] == "success"
    assert (
        cli("validate", tmp_path / "run/problem.json", tmp_path / "run/result.json")["status"]
        == "passed"
    )
    observed = cli(
        "playback",
        tmp_path / "run/problem.json",
        tmp_path / "run/result.json",
        "--output",
        tmp_path / "playback.json",
    )
    report = load(tmp_path / "playback.json", expected=ExecutionReport)
    assert observed["mode"] == "kinematic_playback"
    plan = load(tmp_path / "run/result.json", expected=PlanResult)
    check_playback_report(problem, plan, report, position_tolerance=0, velocity_tolerance=0)


def test_backend_override_saves_the_actual_bound_input(tmp_path, capsys, problem):
    path = tmp_path / "input.json"
    original = replace(
        problem,
        robots=tuple(replace(robot, goal=robot.start) for robot in problem.robots),
        config=replace(problem.config, backend="other", device="other"),
    )
    dump(original, path)
    assert (
        main(
            [
                "plan",
                str(path),
                "--backend",
                "reference",
                "--samples",
                "0",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    result = loads(capsys.readouterr().out, expected=PlanResult)
    effective = load(tmp_path / "run/problem.json", expected=PlanningProblem)
    assert effective.config.backend == "reference" and effective.config.device == "cpu"
    assert effective != original
    result.check_problem(effective)


def test_failure_is_saved_and_has_nonzero_exit_without_fallback(tmp_path, capsys, problem):
    problem = replace(problem, config=replace(problem.config, backend="unknown"))
    dump(problem, tmp_path / "input.json")
    assert main(["plan", str(tmp_path / "input.json"), "--output", str(tmp_path / "run")]) == 1
    plan = loads(capsys.readouterr().out, expected=PlanResult)
    assert plan.status == "unsupported" and plan.trajectory is None
    assert load(tmp_path / "run/problem.json") == problem
    assert load(tmp_path / "run/result.json") == plan


def test_invalid_document_is_reported_without_stdout_or_partial_output(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text('{"format":"other"}')
    assert main(["plan", str(path), "--output", str(tmp_path / "run")]) == 2
    captured = capsys.readouterr()
    assert not captured.out and "kinoweave:" in captured.err
    assert not (tmp_path / "run").exists()


def test_invalid_plan_cannot_pass_cli_validation(problem, result, tmp_path, capsys):
    dump(problem, tmp_path / "input.json")
    dump(result, tmp_path / "result.json")
    assert main(["validate", str(tmp_path / "input.json"), str(tmp_path / "result.json")]) == 2
    assert "CPU revalidation" in capsys.readouterr().err


def test_module_reports_version():
    from kinoweave import __version__

    process = subprocess.run(
        [sys.executable, "-m", "kinoweave", "--version"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path("/tmp"),
    )
    assert process.stdout.strip() == f"KinoWeave {__version__}"


def test_phase_timings_cover_completed_and_failed_planning(problem):
    from kinoweave import Planner
    from kinoweave.roadmap import PRMConfig

    problem = replace(problem, robots=tuple(replace(r, goal=r.start) for r in problem.robots))
    for backend in ("reference", "unknown"):
        configured = replace(problem, config=replace(problem.config, backend=backend))
        with Planner(configured, prm=PRMConfig(num_samples=0)) as planner:
            plan = planner.plan()
        assert plan.status == ("success" if backend == "reference" else "unsupported")
        timing = json.loads(next(d.message for d in plan.diagnostics if d.code == "phase_timings"))
        assert all(seconds >= 0 for seconds in timing["seconds"].values())
        assert sum(timing["seconds"].values()) == pytest.approx(timing["elapsed_seconds"])
        assert "backend_preparation" in timing["seconds"]
        if plan.status == "success":
            assert "cpu_validation" in timing["seconds"]
