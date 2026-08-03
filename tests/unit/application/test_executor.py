import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

import log_agent.adapters as adapters
from log_agent.adapters.fakes import (
    DeterministicReasoningPort,
    FakeLogSearchPort,
    FakeSearchResponse,
)
from log_agent.application.executor import CommandExecutor, InvalidCommand
from log_agent.application.ports import (
    PortUnavailable,
    SearchRequest,
    SearchResult,
    VerificationAssessment,
)
from log_agent.application.query_security import (
    LogSource,
    QueryOperation,
    SafeQueryPipeline,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.domain.models import (
    EvidenceRef,
    Hypothesis,
    HypothesisStatus,
    Investigation,
    InvestigationRequest,
    Phase,
    QueryBudget,
    QueryIntent,
    QueryKind,
    QueryRecord,
    TimeRange,
    VerificationDecision,
)
from log_agent.domain.state_machine import (
    AssessVerification,
    ExecuteQuery,
    GenerateHypotheses,
    HypothesesGenerated,
    OperationFailed,
    QuerySucceeded,
    StartRequested,
    transition,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def make_investigation() -> Investigation:
    time_range = TimeRange(start=NOW - timedelta(hours=1), end=NOW)
    return Investigation(
        id="run-unit",
        request=InvestigationRequest(
            question="Why did checkout fail?",
            scope_ref="checkout-prod",
            time_range=time_range,
        ),
        triage_plan=(
            QueryIntent(
                kind=QueryKind.TRIAGE,
                goal="summarize checkout errors",
                time_range=time_range,
            ),
        ),
        budget=QueryBudget(max_total_queries=4, max_verify_queries=2),
    )


def make_reasoning() -> DeterministicReasoningPort:
    return DeterministicReasoningPort(
        hypothesis_id="h-1",
        hypothesis_statement="Payment timed out",
        verification_goal="find payment timeout events",
        conclusion_summary="Payment timeouts caused checkout failures.",
    )


def make_query_pipeline(*, scope_ref: str = "checkout-prod") -> SafeQueryPipeline:
    policy = ScopePolicy(
        ref=scope_ref,
        sources=(LogSource(index="checkout", sourcetype="checkout:json"),),
        allowed_template_ids=frozenset({"triage.error_summary.v1", "verify.event_sample.v1"}),
        allowed_operations=frozenset({QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}),
        max_time_span=timedelta(hours=1),
        max_result_limit=100,
    )
    return SafeQueryPipeline(ScopePolicyRegistry((policy,)))


def start_query() -> tuple[Investigation, ExecuteQuery]:
    started = transition(make_investigation(), StartRequested())
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    return started.state, command


def hypothesize_command() -> tuple[Investigation, GenerateHypotheses]:
    state, command = start_query()
    record = QueryRecord(
        id="triage-query",
        intent=command.intent,
        result_count=1,
        summary="One error found",
    )
    next_step = transition(
        state,
        QuerySucceeded(command_id=command.command_id, record=record),
    )
    next_command = next_step.commands[0]
    assert isinstance(next_command, GenerateHypotheses)
    return next_step.state, next_command


def assessment_command() -> tuple[Investigation, AssessVerification]:
    state, command = hypothesize_command()
    hypothesis = Hypothesis(
        id="h-1",
        statement="Payment timed out",
        verification_goal="find payment timeout events",
    )
    verify_step = transition(
        state,
        HypothesesGenerated(command_id=command.command_id, hypotheses=(hypothesis,)),
    )
    verify_command = verify_step.commands[0]
    assert isinstance(verify_command, ExecuteQuery)
    query_id = "verify-query"
    evidence = EvidenceRef(id="ev-verify", query_id=query_id, record_ref="row:verify")
    record = QueryRecord(
        id=query_id,
        intent=verify_command.intent,
        result_count=1,
        summary="Payment timeout found",
        evidence_ids=(evidence.id,),
    )
    assess_step = transition(
        verify_step.state,
        QuerySucceeded(
            command_id=verify_command.command_id,
            record=record,
            evidence=(evidence,),
        ),
    )
    assess_command = assess_step.commands[0]
    assert isinstance(assess_command, AssessVerification)
    return assess_step.state, assess_command


def test_execute_query_builds_application_owned_request_and_record() -> None:
    state, command = start_query()
    search = FakeLogSearchPort({"triage.error_summary.v1": FakeSearchResponse(summary="No rows")})
    executor = CommandExecutor(search, make_reasoning(), make_query_pipeline())

    first = asyncio.run(executor.execute(state, command))
    second = asyncio.run(executor.execute(state, command))

    assert isinstance(first, QuerySucceeded)
    assert isinstance(second, QuerySucceeded)
    assert first.command_id == command.command_id
    assert first.record.id == f"{command.command_id}:query"
    assert second.record.id == first.record.id
    assert first.record.intent is command.intent
    assert search.requests[0].operation_id == command.command_id
    assert search.requests[0].authorized_query.scope_ref == state.request.scope_ref
    assert search.requests[0].authorized_query.plan.kind is command.intent.kind
    assert search.requests[0].authorized_query.plan.time_range == command.intent.time_range


class ForeignEvidenceSearch:
    async def search(self, request: SearchRequest) -> SearchResult:
        return SearchResult(
            result_count=1,
            summary="Foreign evidence",
            evidence=(
                EvidenceRef(
                    id="ev-foreign",
                    query_id="an-old-query",
                    record_ref="row:1",
                ),
            ),
        )


def test_foreign_query_evidence_becomes_protocol_failure() -> None:
    state, command = start_query()
    executor = CommandExecutor(
        ForeignEvidenceSearch(),
        make_reasoning(),
        make_query_pipeline(),
    )

    event = asyncio.run(executor.execute(state, command))

    assert isinstance(event, OperationFailed)
    assert event.command_id == command.command_id
    assert event.code == "log_search.protocol_error"
    assert event.message == "Log search returned an invalid response."
    failed = transition(state, event).state
    assert failed.phase is Phase.FAILED


class UnavailableSearch:
    async def search(self, request: SearchRequest) -> SearchResult:
        del request
        raise PortUnavailable("splunk.unavailable", "Log search is unavailable.") from None


def test_port_error_maps_only_safe_fields_to_operation_failed() -> None:
    state, command = start_query()
    executor = CommandExecutor(UnavailableSearch(), make_reasoning(), make_query_pipeline())

    event = asyncio.run(executor.execute(state, command))

    assert isinstance(event, OperationFailed)
    assert event.code == "splunk.unavailable"
    assert event.message == "Log search is unavailable."


class OversizedSearch:
    async def search(self, request: SearchRequest) -> SearchResult:
        del request
        return SearchResult(result_count=101, summary="Too many normalized rows")


def test_result_above_authorized_limit_becomes_protocol_failure() -> None:
    state, command = start_query()
    executor = CommandExecutor(OversizedSearch(), make_reasoning(), make_query_pipeline())

    event = asyncio.run(executor.execute(state, command))

    assert isinstance(event, OperationFailed)
    assert event.code == "log_search.protocol_error"
    assert event.message == "Log search returned an invalid response."


def test_stale_command_is_rejected_before_calling_port() -> None:
    state, command = start_query()
    search = FakeLogSearchPort({"triage.error_summary.v1": FakeSearchResponse(summary="No rows")})
    executor = CommandExecutor(search, make_reasoning(), make_query_pipeline())
    stale = replace(command, command_id="run-unit:old")

    with pytest.raises(InvalidCommand, match="pending operation"):
        asyncio.run(executor.execute(state, stale))

    assert search.requests == []


def test_out_of_range_pending_query_is_rejected_before_calling_port() -> None:
    state, command = start_query()
    outside_intent = replace(
        command.intent,
        time_range=TimeRange(
            start=state.request.time_range.start - timedelta(minutes=1),
            end=state.request.time_range.end,
        ),
    )
    assert state.pending_operation is not None
    forged_state = replace(
        state,
        pending_operation=replace(state.pending_operation, query_intent=outside_intent),
    )
    forged_command = replace(command, intent=outside_intent)
    search = FakeLogSearchPort({"triage.error_summary.v1": FakeSearchResponse(summary="No rows")})
    executor = CommandExecutor(search, make_reasoning(), make_query_pipeline())

    with pytest.raises(InvalidCommand, match="payload"):
        asyncio.run(executor.execute(forged_state, forged_command))

    assert search.requests == []


class BlockingSearch:
    def __init__(self) -> None:
        self.cancelled = False

    async def search(self, request: SearchRequest) -> SearchResult:
        del request
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_search_timeout_cancels_port_and_returns_safe_failure() -> None:
    state, command = start_query()
    search = BlockingSearch()
    executor = CommandExecutor(
        search,
        make_reasoning(),
        make_query_pipeline(),
        search_timeout_seconds=0.01,
    )

    event = asyncio.run(executor.execute(state, command))

    assert isinstance(event, OperationFailed)
    assert event.code == "log_search.timeout"
    assert event.message == "Log search timed out."
    assert search.cancelled is True


class ExplodingSearch:
    async def search(self, request: SearchRequest) -> SearchResult:
        del request
        raise RuntimeError("secret raw adapter detail")


def test_unexpected_programming_error_is_not_hidden_as_domain_failure() -> None:
    state, command = start_query()
    executor = CommandExecutor(ExplodingSearch(), make_reasoning(), make_query_pipeline())

    with pytest.raises(RuntimeError, match="secret raw adapter detail"):
        asyncio.run(executor.execute(state, command))


class BlockingReasoning:
    def __init__(self, entered: asyncio.Event) -> None:
        self.entered = entered
        self.cancelled = False

    async def generate_hypotheses(self, request, memory):
        del request, memory
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def assess_verification(self, request, memory, hypothesis, query):
        raise AssertionError("unexpected assessment call")

    async def generate_conclusion(
        self,
        request,
        memory,
        outcome,
        termination_reason,
        root_cause_hypothesis_id,
    ):
        raise AssertionError("unexpected conclusion call")


def test_task_cancellation_propagates_through_reasoning_port() -> None:
    async def scenario() -> None:
        state, command = hypothesize_command()
        entered = asyncio.Event()
        reasoning = BlockingReasoning(entered)
        executor = CommandExecutor(
            FakeLogSearchPort({}),
            reasoning,
            make_query_pipeline(),
        )
        task = asyncio.create_task(executor.execute(state, command))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reasoning.cancelled is True

    asyncio.run(scenario())


def test_adapters_package_exports_fake_implementations() -> None:
    assert adapters.FakeLogSearchPort is FakeLogSearchPort
    assert adapters.DeterministicReasoningPort is DeterministicReasoningPort


def test_reasoning_timeout_cancels_port_and_returns_safe_failure() -> None:
    async def scenario() -> None:
        state, command = hypothesize_command()
        entered = asyncio.Event()
        reasoning = BlockingReasoning(entered)
        executor = CommandExecutor(
            FakeLogSearchPort({}),
            reasoning,
            make_query_pipeline(),
            reasoning_timeout_seconds=0.01,
        )

        event = await executor.execute(state, command)

        assert isinstance(event, OperationFailed)
        assert event.command_id == command.command_id
        assert event.code == "reasoning.timeout"
        assert event.message == "Reasoning operation timed out."
        assert reasoning.cancelled is True

    asyncio.run(scenario())


class ContradictoryReasoning:
    async def generate_hypotheses(self, request, memory):
        raise AssertionError("unexpected hypothesis generation")

    async def assess_verification(self, request, memory, hypothesis, query):
        del request, memory
        supported = replace(
            hypothesis,
            status=HypothesisStatus.SUPPORTED,
            supporting_evidence_ids=query.evidence_ids,
        )
        return VerificationAssessment(
            hypothesis=supported,
            decision=VerificationDecision.REHYPOTHESIZE,
        )

    async def generate_conclusion(
        self,
        request,
        memory,
        outcome,
        termination_reason,
        root_cause_hypothesis_id,
    ):
        raise AssertionError("unexpected conclusion generation")


def test_contradictory_assessment_becomes_protocol_failure() -> None:
    state, command = assessment_command()
    executor = CommandExecutor(
        FakeLogSearchPort({}),
        ContradictoryReasoning(),
        make_query_pipeline(),
    )

    event = asyncio.run(executor.execute(state, command))

    assert isinstance(event, OperationFailed)
    assert event.command_id == command.command_id
    assert event.code == "reasoning.protocol_error"
    assert event.message == "Reasoning capability returned an invalid response."


def test_task_cancellation_propagates_through_search_port() -> None:
    async def scenario() -> None:
        state, command = start_query()
        search = BlockingSearch()
        executor = CommandExecutor(search, make_reasoning(), make_query_pipeline())
        task = asyncio.create_task(executor.execute(state, command))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert search.cancelled is True

    asyncio.run(scenario())
