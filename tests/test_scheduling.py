import json
from dataclasses import replace

import pytest

from examples.task_scheduling import build_domain
from kinoweave import ContractError, PlanningProblem
from kinoweave.io import document_digest, dumps, loads
from kinoweave.manipulation import (
    ActionOrder,
    FiniteSkeletonProducer,
    ManipulationRefiner,
    ManipulationSearchConfig,
    MetricConfig,
    ModeEvaluator,
    MultiArmPlanner,
    RefinementConfig,
    ScheduleConfig,
    SchedulePlanner,
    SkeletonSearchConfig,
    Task,
    TaskAlternative,
    TaskDomain,
    check_candidate,
    compile_skeleton,
    scoped_skill,
    validate_manipulation,
)
from kinoweave.roadmap import PRMConfig
from kinoweave.search.space import SearchLimit


def scheduler(domain, **kwargs):
    return SchedulePlanner(
        domain,
        prm=PRMConfig(num_samples=0),
        search_config=ManipulationSearchConfig(max_states=512),
        **kwargs,
    )


@pytest.fixture(scope="module")
def scheduled():
    domain = build_domain()
    planner = scheduler(domain)
    report = planner.plan()
    return domain, planner, report


def test_catalog_retry_parallel_assignment_and_shared_geometry(scheduled):
    domain, planner, report = scheduled
    assert report.status == "completed" and report.termination_reason == "producer_exhausted"
    assert len(report.attempts) == 9
    assert [a.status for a in report.attempts[:3]] == ["no_solution_in_discretization"] * 3
    assert [h.objective.total for h in report.history] == [12, 6]
    assert report.objective.total == report.result.makespan == 6
    best = next(a for a in report.attempts if a.candidate_id == report.history[-1].candidate_id)
    assert {c.alternative for c in best.choices} == {"alpha", "beta"}
    assert report.statistics["roadmap_builds"] == 1 and len(planner.roadmaps) == 2
    assert report.statistics["geometry"]["hits"] > 0
    assert report.statistics["producer"]["feedback_received"] == 9
    assert planner.best_problem == report.problem and planner.best_result == report.result
    assert report.domain_digest == document_digest(domain)
    assert report.result.provenance.problem_digest == document_digest(report.problem)
    assert validate_manipulation(report.problem, report.result).status == "passed"
    assert len({document_digest(h.problem) for h in report.history}) == 2


def test_task_json_roundtrip_and_old_root_rejection():
    domain = build_domain()
    assert loads(dumps(domain), expected=TaskDomain) == domain
    with pytest.raises(ContractError, match="expected PlanningProblem"):
        loads(dumps(domain), expected=PlanningProblem)
    data = json.loads(dumps(domain))
    data["tasks"][0]["alternatives"][0]["unexpected"] = True
    with pytest.raises(ContractError, match="unknown field"):
        loads(json.dumps(data))
    data = json.loads(dumps(domain))
    data["schema_version"] = True
    with pytest.raises(ContractError, match="schema version"):
        loads(json.dumps(data))


def test_enumeration_has_no_unnecessary_serialization_and_is_reproducible():
    domain = build_domain()
    candidates = tuple(FiniteSkeletonProducer(domain))
    assert len(candidates) == len({c.identity for c in candidates}) == 9
    parallel = [c for c in candidates if not c.problem.skeleton.precedence]
    assert len(parallel) == 3
    assert all(len(c.problem.skeleton.skills) == 2 for c in candidates)
    config = SkeletonSearchConfig(strategy="distance", seed=123)
    assert tuple(FiniteSkeletonProducer(domain, config=config)) == tuple(
        FiniteSkeletonProducer(domain, config=config)
    )
    assert {c.identity for c in FiniteSkeletonProducer(domain, config=config)} == {
        c.identity for c in candidates
    }


def test_task_dependencies_and_namespaces():
    domain = replace(build_domain(), precedence=(ActionOrder(before="A", after="B"),))
    for candidate in FiniteSkeletonProducer(domain):
        assert candidate.order == ("A/inspect", "B/inspect")
        assert (
            ActionOrder(before="A/inspect", after="B/inspect")
            in candidate.problem.skeleton.precedence
        )
    with pytest.raises(ContractError, match="cycle"):
        replace(domain, precedence=(*domain.precedence, ActionOrder(before="B", after="A")))
    assert scoped_skill("a/b", "c") != scoped_skill("a", "b/c")
    assert scoped_skill("a%2Fb", "c") != scoped_skill("a/b", "c")


@pytest.mark.parametrize(
    "config,reason",
    [
        (SkeletonSearchConfig(max_candidates=1), "skeleton_candidate_limit"),
        (SkeletonSearchConfig(max_expansions=1), "skeleton_expansion_limit"),
        (SkeletonSearchConfig(max_skills=1), "skeleton_skill_limit"),
        (SkeletonSearchConfig(max_frontier=1), "skeleton_frontier_limit"),
    ],
)
def test_producer_limits_are_explicit(config, reason):
    with pytest.raises(SearchLimit, match=reason):
        tuple(FiniteSkeletonProducer(build_domain(), config=config))


def test_compile_rejects_missing_task_options_bad_order_and_substituted_context():
    domain = build_domain()
    candidate = next(iter(FiniteSkeletonProducer(domain)))
    with pytest.raises(ContractError, match="every task"):
        compile_skeleton(domain, candidate.choices[:1], candidate.order)
    with pytest.raises(ContractError, match="every selected skill"):
        compile_skeleton(domain, candidate.choices, candidate.order[:1])
    choices = (replace(candidate.choices[0], skill_options=()), candidate.choices[1])
    with pytest.raises(ContractError, match="every local skill"):
        compile_skeleton(domain, choices, candidate.order)
    with pytest.raises(ContractError, match="another task domain"):
        check_candidate(domain, replace(candidate, domain_digest="other"))
    changed = replace(candidate.problem, scene=replace(candidate.problem.scene, obstacles=()))
    with pytest.raises(ContractError, match="catalog selection"):
        check_candidate(domain, replace(candidate, problem=changed))


class ReactiveProducer:
    def __init__(self, values):
        self.values = values
        self.received = []

    def __iter__(self):
        for index, value in enumerate(self.values):
            assert len(self.received) == index, "feedback must precede the next request"
            yield value

    def feedback(self, attempt):
        self.received.append(attempt)


def test_external_producer_feedback_duplicates_rejections_and_retry(scheduled):
    domain, _, report = scheduled
    candidates = tuple(FiniteSkeletonProducer(domain))
    good = next(c for c in candidates if c.identity == report.history[-1].candidate_id)
    forged = replace(
        good, problem=replace(good.problem, cost=replace(good.problem.cost, makespan=2))
    )
    producer = ReactiveProducer((forged, candidates[0], candidates[0], good))
    result = scheduler(domain).plan(producer)
    assert [a.status for a in result.attempts] == [
        "invalid_candidate",
        "no_solution_in_discretization",
        "duplicate",
        "success",
    ]
    assert result.objective.total == report.objective.total
    assert tuple(producer.received) == result.attempts


@pytest.mark.parametrize("where", ["cancel", "attempt_limit", "time_limit"])
def test_verified_incumbent_survives_stopping(scheduled, monkeypatch, where):
    domain, _, _ = scheduled
    candidates = tuple(FiniteSkeletonProducer(domain))[3:]
    config = ScheduleConfig(max_attempts=1 if where == "attempt_limit" else 128)
    if where == "time_limit":
        from kinoweave.manipulation import scheduling

        original = scheduling.Budget.check
        expired = [False]

        def check(self, stage):
            if expired[0]:
                raise SearchLimit("time_limit", stage)
            original(self, stage)

        monkeypatch.setattr(scheduling.Budget, "check", check)

    def update(value):
        if where == "time_limit":
            expired[0] = True
        return False if where == "cancel" else None

    report = scheduler(domain, config=config).plan(
        ReactiveProducer(candidates), on_improvement=update
    )
    assert report.status == ("cancelled" if where == "cancel" else "budget_exhausted")
    assert report.result.status == "success" and report.objective.total == 12
    assert validate_manipulation(report.problem, report.result).status == "passed"


def test_zero_total_budget_does_not_initialize_geometry():
    report = scheduler(build_domain(), config=ScheduleConfig(time_limit=0)).plan()
    assert report.status == "budget_exhausted" and report.result is None and report.backend is None
    assert report.attempts == ()


def test_atomic_verification_overrun_keeps_solution_and_reports_budget(scheduled, monkeypatch):
    from kinoweave.manipulation import multi, scheduling

    domain, planner, report = scheduled
    candidate = next(
        c for c in FiniteSkeletonProducer(domain) if c.identity == report.history[-1].candidate_id
    )
    expired = [False]
    original_check = scheduling.Budget.check
    original_verify = multi.verify_manipulation

    def check(self, stage):
        if expired[0]:
            raise SearchLimit("time_limit", stage)
        original_check(self, stage)

    def verify(*args, **kwargs):
        checked = original_verify(*args, **kwargs)
        expired[0] = True
        return checked

    monkeypatch.setattr(scheduling.Budget, "check", check)
    monkeypatch.setattr(multi, "verify_manipulation", verify)
    result = scheduler(domain, config=ScheduleConfig(stop_after_first=True)).plan(
        ReactiveProducer((candidate,)),
        graphs=tuple(m.graph for m in planner.roadmaps),
    )
    assert result.status == "budget_exhausted" and result.result.status == "success"
    assert result.attempts[0].status == "success" and result.objective.total == 6
    assert validate_manipulation(result.problem, result.result).status == "passed"


def test_producer_exception_closes_session_and_keeps_live_incumbent(scheduled, monkeypatch):
    domain, _, report = scheduled
    candidate = next(
        c for c in FiniteSkeletonProducer(domain) if c.identity == report.history[-1].candidate_id
    )
    closed = []
    original_close = ModeEvaluator.close

    def close(self):
        closed.append(self)
        original_close(self)

    class FailingProducer(ReactiveProducer):
        def __iter__(self):
            yield candidate
            assert len(self.received) == 1
            raise RuntimeError("external producer failed")

    monkeypatch.setattr(ModeEvaluator, "close", close)
    planner = scheduler(domain)
    with pytest.raises(RuntimeError, match="external producer failed"):
        planner.plan(FailingProducer(()))
    assert planner.best_result.status == "success" and planner.best_objective.total == 6
    assert len(closed) == 1 and closed[0]._closed


def test_fixed_candidate_limits_retry_next_assignment():
    domain = build_domain()
    report = SchedulePlanner(
        domain, prm=PRMConfig(num_samples=0), search_config=ManipulationSearchConfig(max_states=1)
    ).plan()
    assert len(report.attempts) == 9 and report.result is None
    assert all(
        a.status == "timeout" and a.termination_reason == "state_limit" for a in report.attempts
    )


def test_rebind_invalidates_mode_provenance_but_retains_fk(scheduled):
    domain, planner, _ = scheduled
    a, b = tuple(FiniteSkeletonProducer(domain))[3:5]
    graph = planner.roadmaps[0].graph
    with ModeEvaluator(a.problem) as evaluator:
        mask = evaluator.mask(graph, a.problem.initial_mode)
        misses = evaluator.cpu.fk_misses
        evaluator.rebind(b.problem)
        new = evaluator.mask(graph, b.problem.initial_mode)
        assert new.problem_digest == document_digest(b.problem) != mask.problem_digest
        assert evaluator.cpu.fk_misses == misses and evaluator.cpu.fk_hits > 0
        assert evaluator.manipulation == evaluator.cpu.problem == b.problem
        changed = replace(b.problem, scene=replace(b.problem.scene, obstacles=()))
        with pytest.raises(ContractError, match="physical"):
            evaluator.rebind(changed)


def test_borrowed_evaluator_is_not_closed_and_callback_restored(scheduled):
    domain, planner, _ = scheduled
    problem = tuple(FiniteSkeletonProducer(domain))[4].problem
    graphs = tuple(m.graph for m in planner.roadmaps)

    def check():
        pass

    with ModeEvaluator(problem, check_budget=check) as evaluator:
        result = MultiArmPlanner(problem).plan(graphs, evaluator=evaluator, check_budget=check)
        assert result.status == "success" and evaluator.check_budget is check
        assert not evaluator._closed
        mismatch = MultiArmPlanner(replace(problem, cost=replace(problem.cost, makespan=2))).plan(
            graphs, evaluator=evaluator
        )
        assert mismatch.status == "invalid_problem" and evaluator.check_budget is check


def handover_domain():
    from examples.two_arm_handover import build_problem

    p = build_problem()
    return TaskDomain(
        name="transfer",
        scene=p.scene,
        objects=p.objects,
        initial_mode=p.initial_mode,
        goal_mode=p.goal_mode,
        tasks=(
            Task(
                name="transfer",
                alternatives=(
                    TaskAlternative(
                        name="dual_handover",
                        skills=p.skeleton.skills,
                        precedence=p.skeleton.precedence,
                    ),
                ),
            ),
        ),
    )


def test_symbolic_object_chain_and_bimanual_refinement():
    domain = handover_domain()
    report = scheduler(
        domain,
        refinement_config=RefinementConfig(
            time_limit=20,
            max_iterations=1,
            max_candidates=16,
        ),
    ).plan()
    assert report.result.status == "success" and report.objective.total < 14.5
    assert report.attempts[0].refinement_status is not None
    assert len(report.history) >= 2
    assert all(s.skill.startswith("transfer/") for s in report.result.plan.skills)
    assert validate_manipulation(report.problem, report.result).status == "passed"
    p = domain.tasks[0].alternatives[0]
    impossible = replace(
        domain,
        tasks=(
            Task(name="transfer", alternatives=(replace(p, skills=p.skills[:1], precedence=()),)),
        ),
    )
    producer = FiniteSkeletonProducer(impossible)
    assert tuple(producer) == () and producer.statistics["symbolic_rejected"] > 0


def test_symbolic_order_can_interleave_alternative_fragments():
    domain = handover_domain()
    fragment = domain.tasks[0].alternatives[0]
    skills = {s.name: s for s in fragment.skills}
    domain = replace(
        domain,
        tasks=(
            Task(
                name="carry",
                alternatives=(
                    TaskAlternative(
                        name="pick_place",
                        skills=(skills["pick_alpha"], skills["place_beta"]),
                        precedence=(ActionOrder(before="pick_alpha", after="place_beta"),),
                    ),
                ),
            ),
            Task(
                name="exchange",
                alternatives=(
                    TaskAlternative(
                        name="handover",
                        skills=(skills["handover"],),
                    ),
                ),
            ),
        ),
    )
    producer = FiniteSkeletonProducer(domain)
    candidates = tuple(producer)
    assert len(candidates) == 1
    assert candidates[0].order == ("carry/pick_alpha", "exchange/handover", "carry/place_beta")
    assert producer.statistics["symbolic_rejected"] > 0


def test_refinement_obeys_external_budget_and_preserves_seed(scheduled):
    _, _, report = scheduled

    def expired():
        raise SearchLimit("outer_time_limit", "scheduling")

    refined = ManipulationRefiner(report.problem).refine(report.result, check_budget=expired)
    assert refined.status == "budget_exhausted" and refined.termination_reason == "outer_time_limit"
    assert refined.result.status == "success"


def test_unavailable_cuda_is_explicit_without_cpu_fallback(monkeypatch):
    from kinoweave.backends.curobo import BackendUnavailableError

    def unavailable(*args, **kwargs):
        raise BackendUnavailableError("deliberately unavailable")

    monkeypatch.setattr("kinoweave.backends.curobo.CuroboBackend", unavailable)
    report = scheduler(build_domain(backend="curobo", device="cuda:0")).plan()
    assert report.status == "unsupported" and report.result is None
    assert len(report.attempts) == 1 and report.attempts[0].status == "unsupported"


def test_metric_limit_retains_first_verified_plan():
    domain = build_domain()
    domain = replace(domain, cost=replace(domain.cost, distance_metric="tool_path", distance=1))
    candidate = tuple(FiniteSkeletonProducer(domain))[4]
    report = scheduler(domain, metric_config=MetricConfig(max_samples=2)).plan(
        ReactiveProducer((candidate,))
    )
    assert (
        report.status == "budget_exhausted" and report.termination_reason == "metric_sample_limit"
    )
    assert report.result.status == "success" and report.objective is None
    assert validate_manipulation(report.problem, report.result).status == "passed"
