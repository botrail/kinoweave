import json
import subprocess
import sys
from dataclasses import replace

import pytest

from kinoweave import ContractError, PlanResult
from kinoweave.io import document_digest, dump, dumps, load, loads


def test_problem_and_result_survive_file_and_json_roundtrip(problem, result, tmp_path):
    for document in (problem, result, PlanResult()):
        path = tmp_path / f"{document.format}.json"
        dump(document, path)
        restored = load(path, expected=type(document))
        assert restored == document
        assert dumps(restored) == dumps(document)
        assert document_digest(restored) == document_digest(document)
    result.check_problem(problem)


def test_digest_ignores_json_formatting_but_changes_with_geometry(problem):
    data = json.loads(dumps(problem))
    compact = json.dumps(data, separators=(",", ":"))
    assert document_digest(loads(compact)) == document_digest(problem)
    sphere = replace(problem.models[0].collision_spheres[0], radius=0.06)
    changed = replace(problem, models=(replace(problem.models[0], collision_spheres=(sphere,)),))
    assert document_digest(changed) != document_digest(problem)


@pytest.mark.parametrize("field", ["format", "schema_version", "conventions"])
def test_envelope_fields_cannot_be_omitted(problem, field):
    data = json.loads(dumps(problem))
    del data[field]
    with pytest.raises(ContractError, match=field):
        loads(json.dumps(data))


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1"])
def test_unrecognized_or_coerced_schema_versions_are_rejected(problem, version):
    data = json.loads(dumps(problem))
    data["schema_version"] = version
    with pytest.raises(ContractError, match="schema_version"):
        loads(json.dumps(data))


def test_unknown_nested_fields_and_shape_kinds_are_not_silently_dropped(problem):
    data = json.loads(dumps(problem))
    data["models"][0]["joints"][0]["limtis"] = {}
    with pytest.raises(ContractError, match=r"models\[0\].joints\[0\].limtis"):
        loads(json.dumps(data))
    del data["models"][0]["joints"][0]["limtis"]
    data["obstacles"][0]["shape"]["kind"] = "mesh"
    with pytest.raises(ContractError, match=r"obstacles\[0\].shape.kind"):
        loads(json.dumps(data))


def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(problem):
    with pytest.raises(ContractError, match="duplicate object key"):
        loads('{"format": "kinoweave.problem", "format": "other"}')
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ContractError, match="non-finite"):
            loads('{"value": ' + token + "}")
    data = json.loads(dumps(problem))
    data["robots"][0]["goal"][0] = float("inf")
    text = json.dumps(data).replace("Infinity", "1e999")
    with pytest.raises(ContractError, match="finite"):
        loads(text)


def test_document_kind_and_units_are_checked(problem):
    with pytest.raises(ContractError, match="expected PlanResult"):
        loads(dumps(problem), expected=PlanResult)
    data = json.loads(dumps(problem))
    data["conventions"]["length"] = "mm"
    with pytest.raises(ContractError, match="conventions.length"):
        loads(json.dumps(data))


def test_import_and_data_loading_need_no_optional_sdk():
    # Fail even on machines where these SDKs happen to be installed.
    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {"numpy", "torch", "warp", "curobo", "botrail", "mujoco", "isaacsim",
                   "viser", "yourdfpy", "trimesh", "yaml", "defusedxml"}
        if fullname.split(".")[0] in blocked:
            raise AssertionError("unexpected optional dependency: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from kinoweave import PlanResult, PlanningProblem
from kinoweave.io import dumps, loads, load
from kinoweave.adapters import SceneAdapter, PlaybackAdapter, ExecutionAdapter
from kinoweave.backends.curobo import CuroboBackend
from kinoweave.geometry.reference import ReferenceCollisionChecker
from kinoweave.search.reference import JointGraph, reference_plan
from kinoweave.validation import validate_interval, validate_trajectory
from kinoweave.collision import GpuCollisionEvaluator, GeometryCache, ReferenceMotionEvaluator
from kinoweave.roadmap import PRMConfig, build_prm
from kinoweave.adapters import MemoryPlayback, check_playback_report
from kinoweave import Planner, SearchConfig
from kinoweave.assets import load_scene, load_visuals, read_manifest
from kinoweave.visualization import ViserViewer
assert loads(dumps(PlanResult())) == PlanResult()
problem = load("examples/data/retreat.problem.json", expected=PlanningProblem)
states = {r.name: r.start for r in problem.robots}
assert not ReferenceCollisionChecker(problem).collisions(states)
assert validate_interval(problem, states, states, 1).status == "passed"
vertices = ((-1,0),(0,0),(1,0),(0,1))
edges = ((0,1),(1,2),(1,3))
graphs = (
    JointGraph(robot="alpha",joint_names=("x","y"),vertices=vertices,
               edges=edges,start=0,goal=2),
    JointGraph(robot="beta",joint_names=("x","y"),vertices=vertices,
               edges=edges,start=2,goal=0),
)
with Planner(problem,search=SearchConfig(exact_fallback=False)) as planner:
    plan = planner.plan(graphs)
assert plan.status == "success"
with MemoryPlayback(problem) as player:
    player.load(problem,plan)
    report = player.record(0.25)
check_playback_report(problem,plan,report)
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
