"""Exchange data without running a planner or loading a simulator SDK."""

from pathlib import Path
from tempfile import TemporaryDirectory

from kinoweave import PlanningProblem, PlanProvenance, PlanResult
from kinoweave.io import document_digest, dump, load


def main() -> None:
    source = Path(__file__).parent / "data" / "two_sliders.problem.json"
    problem = load(source, expected=PlanningProblem)
    result = PlanResult(
        provenance=PlanProvenance(problem_digest=document_digest(problem), config=problem.config),
    )
    with TemporaryDirectory(prefix="kinoweave-contract-") as directory:
        problem_path = Path(directory) / "problem.json"
        result_path = Path(directory) / "result.json"
        dump(problem, problem_path)
        dump(result, result_path)
        assert load(problem_path, expected=PlanningProblem) == problem
        restored = load(result_path, expected=PlanResult)
        restored.check_problem(problem)
    print(f"{problem.name}: {len(problem.robots)} robots, JSON round trip verified")
    print(f"result status: {result.status} (no planning or collision validation performed)")


if __name__ == "__main__":
    main()
