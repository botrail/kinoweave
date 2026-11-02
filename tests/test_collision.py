from dataclasses import replace

import pytest

from kinoweave import ContractError
from kinoweave.collision import (
    GeometryCache,
    GpuTiling,
    Motion,
    ReferenceMotionEvaluator,
    reference_interference,
)


def test_motion_contract_and_resource_contracts_are_strict():
    for kwargs in (
        {"start": (), "end": ()},
        {"start": (0,), "end": (True,)},
        {"start": (0,), "end": (float("nan"),)},
    ):
        with pytest.raises(ContractError):
            Motion(**kwargs)
    for kwargs in ({"time_batch_size": 0}, {"max_pair_bytes": 63}, {"pair_batch_size": True}):
        with pytest.raises(ContractError):
            GpuTiling(**kwargs)
    assert GeometryCache(0).stats.bytes == 0
    with pytest.raises(ContractError):
        GeometryCache(-1)


def test_reference_table_includes_sweeps_and_waits(retreat_problem):
    model = retreat_problem.models[0]
    model = replace(
        model,
        joints=tuple(replace(j, limits=replace(j.limits, acceleration=20)) for j in model.joints),
    )
    problem = replace(retreat_problem, models=(model,))
    first = (Motion(start=(-1, 0), end=(-1, 0)), Motion(start=(-1, 0), end=(1, 0)))
    second = (Motion(start=(1, 0), end=(1, 0)), Motion(start=(1, 0), end=(-1, 0)))
    table = reference_interference(problem, "alpha", first, "beta", second)
    assert table.first_valid == table.second_valid == (True, True)
    assert table.compatible == ((True, False), (False, False))
    assert table.backend.device == "cpu"


def test_table_marks_derivative_violations_invalid(retreat_problem):
    first = (Motion(start=(-1, 0), end=(1, 0)),)
    second = (Motion(start=(1, 0), end=(1, 0)),)
    table = reference_interference(retreat_problem, "alpha", first, "beta", second)
    assert table.first_valid == (False,)
    assert table.compatible == ((False,),)
    with pytest.raises(ContractError, match="distinct"):
        reference_interference(retreat_problem, "alpha", first, "alpha", first)
    with pytest.raises(ContractError, match="joint count"):
        reference_interference(
            retreat_problem, "alpha", (Motion(start=(0,), end=(1,)),), "beta", second
        )


def test_explicit_cpu_selection_and_empty_candidate_sets(retreat_problem):
    assert ReferenceMotionEvaluator(retreat_problem).motion_validity("alpha", ()) == ()
    assert reference_interference(retreat_problem, "alpha", (), "beta", ()).compatible == ()
    with pytest.raises(ContractError, match="explicitly select"):
        ReferenceMotionEvaluator(
            replace(
                retreat_problem,
                config=replace(retreat_problem.config, backend="curobo", device="cuda:0"),
            )
        )
