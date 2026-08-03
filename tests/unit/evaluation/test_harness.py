import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from log_agent.adapters.fakes import DeterministicReasoningPort
from log_agent.application.executor import CommandExecutor
from log_agent.application.ports import PortError, SearchRequest
from log_agent.application.query_security import (
    LogSource,
    QueryOperation,
    SafeQueryPipeline,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.application.runner import InvestigationRunner
from log_agent.domain.models import (
    ConclusionOutcome,
    Investigation,
    InvestigationRequest,
    Phase,
    QueryBudget,
    QueryIntent,
    QueryKind,
    TerminationReason,
    TimeRange,
)
from log_agent.domain.state_machine import CancelRequested, StartRequested, transition
from log_agent.evaluation.fakes import (
    FakeEvalResponse,
    FakeEvalRow,
    FakeEvalRuntimeFactory,
    FakeEvalScenario,
)
from log_agent.evaluation.harness import (
    DeterministicEvalHarness,
    EvalCaseInput,
    EvalCaseRuntime,
    EvalHarnessError,
)
from log_agent.evaluation.models import (
    EvalFailureCategory,
    EvalViolationCode,
    ExpectedIncidentResult,
    IncidentCase,
    IncidentDataset,
)

NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)
REPORT_HMAC_KEY = b"0123456789abcdef" * 2


def policies(
    *,
    index: str = "checkout",
    max_time_span: timedelta = timedelta(hours=1),
) -> ScopePolicyRegistry:
    return ScopePolicyRegistry(
        (
            ScopePolicy(
                ref="checkout-prod",
                sources=(LogSource(index=index, sourcetype="checkout:json"),),
                allowed_template_ids=frozenset(
                    {"triage.error_summary.v1", "verify.event_sample.v1"}
                ),
                allowed_operations=frozenset(
                    {QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}
                ),
                max_time_span=max_time_span,
                max_result_limit=100,
            ),
        )
    )


def request(*, question: str = "Why did checkout fail?") -> InvestigationRequest:
    return InvestigationRequest(
        question=question,
        scope_ref="checkout-prod",
        time_range=TimeRange(start=NOW - timedelta(minutes=30), end=NOW),
    )


def completed_case(
    *,
    question: str = "Why did checkout fail?",
    evidence_label: str = "payment-timeout-event",
) -> IncidentCase:
    return IncidentCase(
        id="completed-case",
        request=request(question=question),
        replay_fixture_id="completed-fixture",
        expected=ExpectedIncidentResult(
            phase=Phase.COMPLETED,
            termination_reason=TerminationReason.ROOT_CAUSE_IDENTIFIED,
            conclusion_outcome=ConclusionOutcome.CONCLUSIVE,
            root_cause_key="payment_dependency_timeout",
            root_cause_summary="Payment dependency timed out.",
            required_evidence_labels=(evidence_label,),
            failure_code=None,
        ),
        tags=("conclusive",),
    )


def dataset(case: IncidentCase) -> IncidentDataset:
    return IncidentDataset(
        schema_version=1,
        dataset_id="unit-eval",
        revision="2026-08-03.1",
        cases=(case,),
    )


def completed_scenario(
    *,
    dynamic_text: str = "Payment timeout",
    evidence_label: str = "payment-timeout-event",
    evidence_occurred_at: datetime | None = None,
) -> FakeEvalScenario:
    return FakeEvalScenario(
        fixture_id="completed-fixture",
        revision="2026-08-03.1",
        triage_goals=(f"triage {dynamic_text}",),
        max_total_queries=4,
        max_verify_queries=2,
        responses=(
            FakeEvalResponse(
                template_id="triage.error_summary.v1",
                summary=f"triage summary {dynamic_text}",
                rows=(
                    FakeEvalRow(
                        evidence_label="triage-event",
                        fact_statement=f"triage fact {dynamic_text}",
                    ),
                ),
            ),
            FakeEvalResponse(
                template_id="verify.event_sample.v1",
                summary=f"verify summary {dynamic_text}",
                rows=(
                    FakeEvalRow(
                        evidence_label=evidence_label,
                        fact_statement=f"verification fact {dynamic_text}",
                        occurred_at=evidence_occurred_at,
                    ),
                ),
            ),
        ),
        root_cause_key="payment_dependency_timeout",
        hypothesis_statement=f"hypothesis {dynamic_text}",
        verification_goal=f"verify {dynamic_text}",
        conclusion_summary=f"conclusion {dynamic_text}",
        recommendations=(f"recommendation {dynamic_text}",),
    )


def run_completed_with_result_mutation(
    mutation: Callable[[Investigation], Investigation],
):
    inner = FakeEvalRuntimeFactory(policies(), (completed_scenario(),))

    class MutatingFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            runtime = inner.prepare(case)

            class MutatingDriver:
                async def run(self, initial: Investigation) -> Investigation:
                    result = await runtime.driver.run(initial)
                    return mutation(result)

            return replace(runtime, driver=MutatingDriver())

    return asyncio.run(
        DeterministicEvalHarness(
            MutatingFactory(),
            report_hmac_key=REPORT_HMAC_KEY,
        ).run(dataset(completed_case()))
    )


def test_report_is_content_free_and_byte_deterministic() -> None:
    canary = "SENSITIVE-CANARY-7291"
    evidence_label = "sensitive-evidence-label"
    case = completed_case(
        question=f"Question contains {canary}",
        evidence_label=evidence_label,
    )
    factory = FakeEvalRuntimeFactory(
        policies(index="sensitive-canary-index"),
        (completed_scenario(dynamic_text=canary, evidence_label=evidence_label),),
    )
    prepared = factory.prepare(
        EvalCaseInput(
            request=case.request,
            replay_fixture_id=case.replay_fixture_id,
        )
    )
    result = asyncio.run(prepared.driver.run(prepared.initial))

    assert case.id not in result.id
    assert case.replay_fixture_id not in repr(result.memory)
    assert evidence_label not in repr(result.memory)
    assert all(
        evidence_label not in binding.record_ref for binding in prepared.evidence_label_bindings
    )

    harness = DeterministicEvalHarness(
        factory,
        report_hmac_key=REPORT_HMAC_KEY,
    )

    first = asyncio.run(harness.run(dataset(case)))
    second = asyncio.run(harness.run(dataset(case)))
    report_json = first.to_json()

    assert first.to_json() == second.to_json()
    assert first.passed is True
    for forbidden in (
        canary,
        evidence_label,
        "sensitive-canary-index",
        "fake-eval://",
        "question",
        "scope_ref",
        "record_ref",
        "evidence_ids",
        "fact_statement",
        "conclusion_summary",
        "recommendations",
        case.id,
        case.replay_fixture_id,
        dataset(case).dataset_id,
        dataset(case).revision,
        dataset(case).content_hash,
        prepared.fixture_hash,
        prepared.query_policy_hash,
        "content_hash",
    ):
        assert forbidden not in report_json


def test_report_key_is_required_and_domain_separates_fingerprints() -> None:
    factory = FakeEvalRuntimeFactory(policies(), (completed_scenario(),))
    with pytest.raises(ValueError, match="at least 32 bytes"):
        DeterministicEvalHarness(factory, report_hmac_key=b"too-short")

    incident_dataset = dataset(completed_case())
    first = asyncio.run(
        DeterministicEvalHarness(
            factory,
            report_hmac_key=REPORT_HMAC_KEY,
        ).run(incident_dataset)
    )
    second = asyncio.run(
        DeterministicEvalHarness(
            factory,
            report_hmac_key=b"fedcba9876543210" * 2,
        ).run(incident_dataset)
    )

    assert first.dataset_fingerprint != second.dataset_fingerprint
    assert first.query_policy_fingerprint != second.query_policy_fingerprint
    assert first.cases[0].case_token != second.cases[0].case_token
    assert first.dataset_fingerprint != first.query_policy_fingerprint


def test_prepare_and_driver_exceptions_are_replaced_without_canary_context() -> None:
    canary = "SENSITIVE-EXCEPTION-CANARY-7291"

    class LeakyPrepareFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            del case
            raise RuntimeError(canary)

    with pytest.raises(EvalHarnessError) as prepare_error:
        asyncio.run(
            DeterministicEvalHarness(
                LeakyPrepareFactory(),
                report_hmac_key=REPORT_HMAC_KEY,
            ).run(dataset(completed_case()))
        )

    assert prepare_error.value.code == "eval.runtime_prepare_failed"
    assert canary not in str(prepare_error.value)
    assert prepare_error.value.__context__ is None

    class EmptySearchProbe:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

    class EmptyReasoningProbe:
        def __init__(self) -> None:
            self.calls: list[str] = []

    class LeakyDriver:
        async def run(self, initial: Investigation) -> Investigation:
            del initial
            raise RuntimeError(canary)

    class LeakyDriverFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            initial = Investigation(
                id="eval:opaque-run",
                request=case.request,
                triage_plan=(
                    QueryIntent(
                        kind=QueryKind.TRIAGE,
                        goal="summarize checkout errors",
                        time_range=case.request.time_range,
                    ),
                ),
                budget=QueryBudget(max_total_queries=2, max_verify_queries=1),
            )
            return EvalCaseRuntime(
                fixture_id=case.replay_fixture_id,
                fixture_revision="2026-08-03.1",
                fixture_hash="b" * 64,
                query_policy_hash="c" * 64,
                initial=initial,
                driver=LeakyDriver(),
                search_probe=EmptySearchProbe(),
                reasoning_probe=EmptyReasoningProbe(),
            )

    driver_case = IncidentCase(
        id="driver-failure",
        request=request(),
        replay_fixture_id="driver-failure-v1",
        expected=ExpectedIncidentResult(
            phase=Phase.INCONCLUSIVE,
            termination_reason=TerminationReason.NO_DATA,
            conclusion_outcome=ConclusionOutcome.INCONCLUSIVE,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=None,
        ),
    )
    with pytest.raises(EvalHarnessError) as driver_error:
        asyncio.run(
            DeterministicEvalHarness(
                LeakyDriverFactory(),
                report_hmac_key=REPORT_HMAC_KEY,
            ).run(dataset(driver_case))
        )

    assert driver_error.value.code == "eval.driver_failed"
    assert canary not in str(driver_error.value)
    assert driver_error.value.__context__ is None


def test_evidence_outside_its_query_window_fails_the_process_gate() -> None:
    case = completed_case()
    harness = DeterministicEvalHarness(
        FakeEvalRuntimeFactory(
            policies(),
            (completed_scenario(evidence_occurred_at=NOW + timedelta(days=10)),),
        ),
        report_hmac_key=REPORT_HMAC_KEY,
    )

    report = asyncio.run(harness.run(dataset(case)))
    case_report = report.cases[0]

    assert case_report.outcome_passed is True
    assert case_report.process_passed is False
    assert case_report.violations == (EvalViolationCode.EVIDENCE_TIME_RANGE_MISMATCH,)


def test_query_ledger_must_correspond_to_the_recorded_search_request() -> None:
    def mutate(result: Investigation) -> Investigation:
        first_query = result.memory.queries[0]
        outside_request = TimeRange(
            start=NOW - timedelta(days=2),
            end=NOW - timedelta(days=1),
        )
        mutated_query = replace(
            first_query,
            intent=replace(first_query.intent, time_range=outside_request),
        )
        return replace(
            result,
            memory=replace(
                result.memory,
                queries=(mutated_query, *result.memory.queries[1:]),
            ),
        )

    case_report = run_completed_with_result_mutation(mutate).cases[0]

    assert case_report.outcome_passed is True
    assert case_report.process_passed is False
    assert case_report.violations == (EvalViolationCode.QUERY_LEDGER_REQUEST_MISMATCH,)


def test_verify_query_must_reference_a_known_hypothesis() -> None:
    def mutate(result: Investigation) -> Investigation:
        verify_query = result.memory.queries[1]
        mutated_query = replace(
            verify_query,
            intent=replace(verify_query.intent, hypothesis_id="missing-hypothesis"),
        )
        return replace(
            result,
            memory=replace(
                result.memory,
                queries=(result.memory.queries[0], mutated_query),
            ),
        )

    case_report = run_completed_with_result_mutation(mutate).cases[0]

    assert case_report.outcome_passed is True
    assert case_report.process_passed is False
    assert case_report.violations == (EvalViolationCode.QUERY_LEDGER_REQUEST_MISMATCH,)


def test_successful_run_requires_exact_query_accounting() -> None:
    def mutate(result: Investigation) -> Investigation:
        return replace(
            result,
            budget=replace(result.budget, issued_total=result.budget.issued_total + 1),
        )

    case_report = run_completed_with_result_mutation(mutate).cases[0]

    assert case_report.outcome_passed is True
    assert case_report.process_passed is False
    assert case_report.violations == (EvalViolationCode.QUERY_ACCOUNTING_MISMATCH,)


def test_terminal_result_must_preserve_initial_state_lineage() -> None:
    def mutate(result: Investigation) -> Investigation:
        return replace(
            result,
            id="different-investigation",
            request=replace(
                result.request,
                question="A different question after execution.",
            ),
        )

    case_report = run_completed_with_result_mutation(mutate).cases[0]

    assert case_report.outcome_passed is True
    assert case_report.process_passed is False
    assert case_report.violations == (EvalViolationCode.STATE_LINEAGE_MISMATCH,)


def test_query_policy_hash_preserves_one_microsecond_changes_for_large_spans() -> None:
    scenario = completed_scenario()
    first = FakeEvalRuntimeFactory(
        policies(max_time_span=timedelta(days=200_000)),
        (scenario,),
    ).prepare(
        EvalCaseInput(
            request=request(),
            replay_fixture_id=scenario.fixture_id,
        )
    )
    second = FakeEvalRuntimeFactory(
        policies(max_time_span=timedelta(days=200_000, microseconds=1)),
        (scenario,),
    ).prepare(
        EvalCaseInput(
            request=request(),
            replay_fixture_id=scenario.fixture_id,
        )
    )

    assert first.query_policy_hash != second.query_policy_hash


def test_fixture_semantic_hash_normalizes_equivalent_timezone_offsets() -> None:
    utc_scenario = completed_scenario(evidence_occurred_at=NOW)
    offset_scenario = completed_scenario(
        evidence_occurred_at=NOW.astimezone(timezone(timedelta(hours=8)))
    )

    assert utc_scenario.content_hash == offset_scenario.content_hash


def test_missing_fixture_fails_before_any_case_runs() -> None:
    inner = FakeEvalRuntimeFactory(policies(), (completed_scenario(),))
    case = IncidentCase(
        id="missing-fixture-case",
        request=request(),
        replay_fixture_id="missing-fixture",
        expected=ExpectedIncidentResult(
            phase=Phase.INCONCLUSIVE,
            termination_reason=TerminationReason.NO_DATA,
            conclusion_outcome=ConclusionOutcome.INCONCLUSIVE,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=None,
        ),
    )

    with pytest.raises(EvalHarnessError) as caught:
        asyncio.run(
            DeterministicEvalHarness(
                inner,
                report_hmac_key=REPORT_HMAC_KEY,
            ).run(dataset(case))
        )

    assert caught.value.code == "eval.fixture_missing"


def test_missing_evidence_annotation_fails_preflight_without_io() -> None:
    class RecordingFactory:
        def __init__(self) -> None:
            self.inner = FakeEvalRuntimeFactory(policies(), (completed_scenario(),))
            self.runtime: EvalCaseRuntime | None = None

        def prepare(self, case_input: EvalCaseInput) -> EvalCaseRuntime:
            self.runtime = self.inner.prepare(case_input)
            return self.runtime

    factory = RecordingFactory()
    case = completed_case(evidence_label="missing-annotation")

    with pytest.raises(EvalHarnessError) as caught:
        asyncio.run(
            DeterministicEvalHarness(
                factory,
                report_hmac_key=REPORT_HMAC_KEY,
            ).run(dataset(case))
        )

    assert caught.value.code == "eval.evidence_annotation_missing"
    assert factory.runtime is not None
    assert factory.runtime.search_probe.requests == []
    assert factory.runtime.reasoning_probe.calls == []


def test_wrong_expected_outcome_fails_only_the_outcome_gate() -> None:
    actual_completed = completed_case()
    wrong_expected = IncidentCase(
        id=actual_completed.id,
        request=actual_completed.request,
        replay_fixture_id=actual_completed.replay_fixture_id,
        expected=ExpectedIncidentResult(
            phase=Phase.INCONCLUSIVE,
            termination_reason=TerminationReason.NO_DATA,
            conclusion_outcome=ConclusionOutcome.INCONCLUSIVE,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=None,
        ),
        tags=actual_completed.tags,
    )
    harness = DeterministicEvalHarness(
        FakeEvalRuntimeFactory(policies(), (completed_scenario(),)),
        report_hmac_key=REPORT_HMAC_KEY,
    )

    report = asyncio.run(harness.run(dataset(wrong_expected)))
    case_report = report.cases[0]

    assert case_report.actual_phase is Phase.COMPLETED
    assert case_report.outcome_passed is False
    assert case_report.process_passed is True
    assert case_report.violations == (
        EvalViolationCode.PHASE_MISMATCH,
        EvalViolationCode.TERMINATION_REASON_MISMATCH,
        EvalViolationCode.CONCLUSION_OUTCOME_MISMATCH,
    )


def test_arbitrary_port_error_code_and_message_do_not_enter_report() -> None:
    canary_code = "secret.SENSITIVE7291"
    canary_message = "sensitive failure SENSITIVE-CANARY-7291"

    class FailingSearch:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

        async def search(self, search_request: SearchRequest):
            self.requests.append(search_request)
            raise PortError(canary_code, canary_message)

    class RuntimeFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            search = FailingSearch()
            reasoning = DeterministicReasoningPort(
                hypothesis_id="h-unused",
                hypothesis_statement="Unused hypothesis.",
                verification_goal="unused verification",
                conclusion_summary="Unused conclusion.",
            )
            initial = Investigation(
                id="eval:port-failure",
                request=case.request,
                triage_plan=(
                    QueryIntent(
                        kind=QueryKind.TRIAGE,
                        goal="summarize checkout errors",
                        time_range=case.request.time_range,
                    ),
                ),
                budget=QueryBudget(max_total_queries=2, max_verify_queries=1),
            )
            driver = InvestigationRunner(
                CommandExecutor(search, reasoning, SafeQueryPipeline(policies()))
            )
            return EvalCaseRuntime(
                fixture_id=case.replay_fixture_id,
                fixture_revision="2026-08-03.1",
                fixture_hash="b" * 64,
                query_policy_hash="c" * 64,
                initial=initial,
                driver=driver,
                search_probe=search,
                reasoning_probe=reasoning,
            )

    case = IncidentCase(
        id="port-failure",
        request=request(),
        replay_fixture_id="port-failure-v1",
        expected=ExpectedIncidentResult(
            phase=Phase.FAILED,
            termination_reason=TerminationReason.OPERATION_FAILED,
            conclusion_outcome=None,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=canary_code,
        ),
    )

    report = asyncio.run(
        DeterministicEvalHarness(
            RuntimeFactory(),
            report_hmac_key=REPORT_HMAC_KEY,
        ).run(dataset(case))
    )
    report_json = report.to_json()

    assert report.passed is True
    assert report.cases[0].failure_category is EvalFailureCategory.EXTERNAL_ERROR
    assert report.cases[0].failure_code_matched is True
    assert canary_code not in report_json
    assert canary_message not in report_json


def test_verify_policy_rejection_does_not_treat_earlier_log_call_as_escape() -> None:
    case = IncidentCase(
        id="verify-policy-denied",
        request=request(),
        replay_fixture_id="verify-policy-denied-v1",
        expected=ExpectedIncidentResult(
            phase=Phase.FAILED,
            termination_reason=TerminationReason.OPERATION_FAILED,
            conclusion_outcome=None,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code="query_policy.denied",
        ),
    )
    scenario = FakeEvalScenario(
        fixture_id="verify-policy-denied-v1",
        revision="2026-08-03.1",
        triage_goals=("summarize checkout errors",),
        max_total_queries=4,
        max_verify_queries=2,
        responses=(
            FakeEvalResponse(
                template_id="triage.error_summary.v1",
                summary="Checkout errors were found.",
                rows=(
                    FakeEvalRow(
                        evidence_label="triage-error",
                        fact_statement="Checkout returned an error.",
                    ),
                ),
            ),
        ),
        root_cause_key="payment_dependency_timeout",
        hypothesis_statement="Payment dependency timeouts caused checkout failures.",
        verification_goal="timeout OR error",
        conclusion_summary="Unused conclusion.",
    )
    harness = DeterministicEvalHarness(
        FakeEvalRuntimeFactory(policies(), (scenario,)),
        report_hmac_key=REPORT_HMAC_KEY,
    )

    report = asyncio.run(harness.run(dataset(case)))
    case_report = report.cases[0]

    assert report.passed is True
    assert case_report.failure_category is EvalFailureCategory.QUERY_POLICY
    assert case_report.failure_code_matched is True
    assert case_report.issued_query_count == 2
    assert case_report.log_port_call_count == 1
    assert case_report.successful_query_count == 1
    assert EvalViolationCode.POLICY_FAILURE_REACHED_PORT not in case_report.violations


def test_cancelled_is_scored_only_when_an_explicit_domain_driver_returns_it() -> None:
    class EmptySearchProbe:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

    class EmptyReasoningProbe:
        def __init__(self) -> None:
            self.calls: list[str] = []

    class CancelDriver:
        async def run(self, initial: Investigation) -> Investigation:
            return transition(initial, CancelRequested()).state

    class CancelFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            initial = Investigation(
                id="eval:cancelled-contract",
                request=case.request,
                triage_plan=(
                    QueryIntent(
                        kind=QueryKind.TRIAGE,
                        goal="summarize checkout errors",
                        time_range=case.request.time_range,
                    ),
                ),
                budget=QueryBudget(max_total_queries=2, max_verify_queries=1),
            )
            return EvalCaseRuntime(
                fixture_id=case.replay_fixture_id,
                fixture_revision="2026-08-03.1",
                fixture_hash="d" * 64,
                query_policy_hash="e" * 64,
                initial=initial,
                driver=CancelDriver(),
                search_probe=EmptySearchProbe(),
                reasoning_probe=EmptyReasoningProbe(),
            )

    case = IncidentCase(
        id="cancelled-contract",
        request=request(),
        replay_fixture_id="cancelled-contract-v1",
        expected=ExpectedIncidentResult(
            phase=Phase.CANCELLED,
            termination_reason=TerminationReason.USER_CANCELLED,
            conclusion_outcome=None,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=None,
        ),
    )

    report = asyncio.run(
        DeterministicEvalHarness(
            CancelFactory(),
            report_hmac_key=REPORT_HMAC_KEY,
        ).run(dataset(case))
    )

    assert report.passed is True
    assert report.cases[0].actual_phase is Phase.CANCELLED
    assert report.cases[0].issued_query_count == 0
    assert report.cases[0].log_port_call_count == 0


def test_cancelled_after_query_issue_allows_one_uncompleted_operation() -> None:
    class EmptySearchProbe:
        def __init__(self) -> None:
            self.requests: list[SearchRequest] = []

    class EmptyReasoningProbe:
        def __init__(self) -> None:
            self.calls: list[str] = []

    class CancelAfterIssueDriver:
        async def run(self, initial: Investigation) -> Investigation:
            issued = transition(initial, StartRequested()).state
            return transition(issued, CancelRequested()).state

    class CancelAfterIssueFactory:
        def prepare(self, case: EvalCaseInput) -> EvalCaseRuntime:
            initial = Investigation(
                id="eval:cancel-after-issue",
                request=case.request,
                triage_plan=(
                    QueryIntent(
                        kind=QueryKind.TRIAGE,
                        goal="summarize checkout errors",
                        time_range=case.request.time_range,
                    ),
                ),
                budget=QueryBudget(max_total_queries=2, max_verify_queries=1),
            )
            return EvalCaseRuntime(
                fixture_id=case.replay_fixture_id,
                fixture_revision="2026-08-03.1",
                fixture_hash="f" * 64,
                query_policy_hash="1" * 64,
                initial=initial,
                driver=CancelAfterIssueDriver(),
                search_probe=EmptySearchProbe(),
                reasoning_probe=EmptyReasoningProbe(),
            )

    case = IncidentCase(
        id="cancel-after-issue",
        request=request(),
        replay_fixture_id="cancel-after-issue-v1",
        expected=ExpectedIncidentResult(
            phase=Phase.CANCELLED,
            termination_reason=TerminationReason.USER_CANCELLED,
            conclusion_outcome=None,
            root_cause_key=None,
            root_cause_summary=None,
            required_evidence_labels=(),
            failure_code=None,
        ),
    )

    report = asyncio.run(
        DeterministicEvalHarness(
            CancelAfterIssueFactory(),
            report_hmac_key=REPORT_HMAC_KEY,
        ).run(dataset(case))
    )

    assert report.passed is True
    assert report.cases[0].issued_query_count == 1
    assert report.cases[0].log_port_call_count == 0
    assert report.cases[0].successful_query_count == 0
