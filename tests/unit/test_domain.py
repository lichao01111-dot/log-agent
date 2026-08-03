from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from log_agent.domain.models import (
    Conclusion,
    ConclusionOutcome,
    EvidenceRef,
    Fact,
    Hypothesis,
    HypothesisStatus,
    Investigation,
    InvestigationRequest,
    Phase,
    QueryBudget,
    QueryIntent,
    QueryKind,
    QueryRecord,
    TerminationReason,
    TimeRange,
    VerificationDecision,
    WorkingMemory,
)
from log_agent.domain.state_machine import (
    CancelRequested,
    ConclusionGenerated,
    ExecuteQuery,
    GenerateConclusion,
    HypothesesGenerated,
    InvalidTransition,
    InvariantViolation,
    OperationFailed,
    QuerySucceeded,
    StartRequested,
    VerificationAssessed,
    transition,
)

NOW = datetime(2026, 8, 2, 0, 0, tzinfo=UTC)


def make_investigation(
    *,
    max_total_queries: int = 4,
    max_verify_queries: int = 2,
    triage_goals: tuple[str, ...] = ("summarize errors",),
) -> Investigation:
    time_range = TimeRange(start=NOW - timedelta(hours=1), end=NOW)
    request = InvestigationRequest(
        question="Why did checkout fail?",
        scope_ref="checkout-prod",
        time_range=time_range,
    )
    triage_plan = tuple(
        QueryIntent(kind=QueryKind.TRIAGE, goal=goal, time_range=time_range)
        for goal in triage_goals
    )
    return Investigation(
        id="run-1",
        request=request,
        triage_plan=triage_plan,
        budget=QueryBudget(
            max_total_queries=max_total_queries,
            max_verify_queries=max_verify_queries,
        ),
    )


def succeed_query(
    state: Investigation,
    command: ExecuteQuery,
    *,
    query_id: str,
    result_count: int,
    evidence_id: str | None = None,
) -> Investigation:
    evidence = ()
    evidence_ids = ()
    facts = ()
    if evidence_id is not None:
        evidence = (
            EvidenceRef(
                id=evidence_id,
                query_id=query_id,
                record_ref=f"row:{query_id}:1",
                occurred_at=NOW,
            ),
        )
        evidence_ids = (evidence_id,)
        facts = (
            Fact(
                id=f"fact-{query_id}",
                statement=f"Observed result for {query_id}",
                evidence_ids=evidence_ids,
            ),
        )
    record = QueryRecord(
        id=query_id,
        intent=command.intent,
        result_count=result_count,
        summary="no matching events" if result_count == 0 else "matching events found",
        evidence_ids=evidence_ids,
    )
    return transition(
        state,
        QuerySucceeded(
            command_id=command.command_id,
            record=record,
            facts=facts,
            evidence=evidence,
        ),
    ).state


def advance_to_assessment_request(
    hypotheses: tuple[Hypothesis, ...] | None = None,
    *,
    verification_result_count: int = 1,
    verification_evidence_id: str | None = "ev-verify",
) -> tuple[Investigation, tuple[Hypothesis, ...]]:
    started = transition(make_investigation(), StartRequested())
    query_command = started.commands[0]
    assert isinstance(query_command, ExecuteQuery)
    triaged = succeed_query(
        started.state,
        query_command,
        query_id="q-triage",
        result_count=2,
        evidence_id="ev-triage",
    )

    hypothesis_command = triaged.pending_operation
    assert hypothesis_command is not None
    hypotheses = hypotheses or (
        Hypothesis(
            id="h-1",
            statement="The payment dependency timed out",
            verification_goal="find payment timeout events",
        ),
    )
    verifying = transition(
        triaged,
        HypothesesGenerated(
            command_id=hypothesis_command.command_id,
            hypotheses=hypotheses,
        ),
    )
    verify_command = verifying.commands[0]
    assert isinstance(verify_command, ExecuteQuery)
    verified_query = succeed_query(
        verifying.state,
        verify_command,
        query_id="q-verify",
        result_count=verification_result_count,
        evidence_id=verification_evidence_id,
    )
    return verified_query, hypotheses


def advance_to_conclusion_request() -> tuple[Investigation, GenerateConclusion, Hypothesis]:
    verified_query, hypotheses = advance_to_assessment_request()
    assess_command = verified_query.pending_operation
    assert assess_command is not None
    hypothesis = hypotheses[0]
    supported = replace(
        hypothesis,
        status=HypothesisStatus.SUPPORTED,
        supporting_evidence_ids=("ev-verify",),
    )
    concluding = transition(
        verified_query,
        VerificationAssessed(
            command_id=assess_command.command_id,
            hypothesis=supported,
            decision=VerificationDecision.CONCLUDE,
        ),
    )
    conclusion_command = concluding.commands[0]
    assert isinstance(conclusion_command, GenerateConclusion)
    return concluding.state, conclusion_command, supported


def test_happy_path_reaches_grounded_completed_result() -> None:
    state, command, supported = advance_to_conclusion_request()
    result = Conclusion(
        outcome=ConclusionOutcome.CONCLUSIVE,
        summary="Payment dependency timeouts caused checkout failures.",
        termination_reason=TerminationReason.ROOT_CAUSE_IDENTIFIED,
        root_cause_hypothesis_id=supported.id,
        evidence_ids=("ev-verify",),
        recommendations=("Check the payment dependency.",),
    )

    finished = transition(
        state,
        ConclusionGenerated(command_id=command.command_id, conclusion=result),
    )

    assert finished.state.phase is Phase.COMPLETED
    assert finished.state.conclusion == result
    assert finished.state.pending_operation is None
    assert finished.commands == ()


def test_no_triage_data_finishes_as_inconclusive() -> None:
    started = transition(make_investigation(), StartRequested())
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    concluding = succeed_query(
        started.state,
        command,
        query_id="q-empty",
        result_count=0,
    )
    pending = concluding.pending_operation
    assert pending is not None
    assert pending.termination_reason is TerminationReason.NO_DATA

    result = Conclusion(
        outcome=ConclusionOutcome.INCONCLUSIVE,
        summary="No matching log events were found.",
        termination_reason=TerminationReason.NO_DATA,
    )
    finished = transition(
        concluding,
        ConclusionGenerated(command_id=pending.command_id, conclusion=result),
    )

    assert finished.state.phase is Phase.INCONCLUSIVE
    assert finished.state.termination_reason is TerminationReason.NO_DATA


def test_budget_is_charged_when_query_is_issued_and_stops_verification() -> None:
    started = transition(
        make_investigation(max_total_queries=1, max_verify_queries=0),
        StartRequested(),
    )
    assert started.state.budget.issued_total == 1
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    triaged = succeed_query(
        started.state,
        command,
        query_id="q-triage",
        result_count=1,
        evidence_id="ev-1",
    )
    pending = triaged.pending_operation
    assert pending is not None
    hypothesis = Hypothesis(
        id="h-budget",
        statement="A downstream timeout occurred",
        verification_goal="find downstream timeout events",
    )

    concluding = transition(
        triaged,
        HypothesesGenerated(
            command_id=pending.command_id,
            hypotheses=(hypothesis,),
        ),
    )

    assert concluding.state.phase is Phase.CONCLUDE
    assert concluding.state.budget.issued_total == 1
    stored = next(item for item in concluding.state.memory.hypotheses if item.id == hypothesis.id)
    assert stored.status is HypothesisStatus.PROPOSED
    generated = concluding.commands[0]
    assert isinstance(generated, GenerateConclusion)
    assert generated.termination_reason is TerminationReason.QUERY_BUDGET_EXHAUSTED


def test_conclusion_cannot_reference_unknown_evidence() -> None:
    state, command, supported = advance_to_conclusion_request()
    forged = Conclusion(
        outcome=ConclusionOutcome.CONCLUSIVE,
        summary="Forged conclusion",
        termination_reason=TerminationReason.ROOT_CAUSE_IDENTIFIED,
        root_cause_hypothesis_id=supported.id,
        evidence_ids=("ev-forged",),
    )

    with pytest.raises(InvariantViolation, match="unknown evidence"):
        transition(
            state,
            ConclusionGenerated(command_id=command.command_id, conclusion=forged),
        )


def test_stale_command_result_is_rejected_without_mutating_state() -> None:
    started = transition(make_investigation(), StartRequested())
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    record = QueryRecord(
        id="q-stale",
        intent=command.intent,
        result_count=0,
        summary="stale result",
    )

    with pytest.raises(InvalidTransition, match="command_id"):
        transition(
            started.state,
            QuerySucceeded(command_id="old-command", record=record),
        )

    assert started.state.phase is Phase.TRIAGE
    assert started.state.memory.queries == ()
    assert started.state.budget.issued_total == 1


def test_more_than_three_hypotheses_is_rejected() -> None:
    started = transition(make_investigation(), StartRequested())
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    triaged = succeed_query(
        started.state,
        command,
        query_id="q-1",
        result_count=1,
        evidence_id="ev-1",
    )
    pending = triaged.pending_operation
    assert pending is not None
    hypotheses = tuple(
        Hypothesis(
            id=f"h-{number}",
            statement=f"Hypothesis {number}",
            verification_goal=f"Verify {number}",
        )
        for number in range(4)
    )

    with pytest.raises(InvariantViolation, match="at most three"):
        transition(
            triaged,
            HypothesesGenerated(
                command_id=pending.command_id,
                hypotheses=hypotheses,
            ),
        )


def test_every_running_phase_can_be_cancelled() -> None:
    new_state = make_investigation()
    triage_transition = transition(make_investigation(), StartRequested())
    triage_command = triage_transition.commands[0]
    assert isinstance(triage_command, ExecuteQuery)
    hypothesize_state = succeed_query(
        triage_transition.state,
        triage_command,
        query_id="q-cancel-triage",
        result_count=1,
        evidence_id="ev-cancel-triage",
    )
    hypothesis_pending = hypothesize_state.pending_operation
    assert hypothesis_pending is not None
    verify_transition = transition(
        hypothesize_state,
        HypothesesGenerated(
            command_id=hypothesis_pending.command_id,
            hypotheses=(
                Hypothesis(
                    id="h-cancel",
                    statement="Cancellation test hypothesis",
                    verification_goal="verify cancellation test hypothesis",
                ),
            ),
        ),
    )
    conclude_state, _, _ = advance_to_conclusion_request()
    running_states = (
        new_state,
        triage_transition.state,
        hypothesize_state,
        verify_transition.state,
        conclude_state,
    )

    assert {state.phase for state in running_states} == {
        Phase.NEW,
        Phase.TRIAGE,
        Phase.HYPOTHESIZE,
        Phase.VERIFY,
        Phase.CONCLUDE,
    }
    for state in running_states:
        cancelled = transition(state, CancelRequested())
        assert cancelled.state.phase is Phase.CANCELLED
        assert cancelled.state.termination_reason is TerminationReason.USER_CANCELLED
        assert cancelled.state.pending_operation is None


def test_operation_failure_does_not_masquerade_as_a_diagnosis() -> None:
    started = transition(make_investigation(), StartRequested())
    pending = started.state.pending_operation
    assert pending is not None

    failed = transition(
        started.state,
        OperationFailed(
            command_id=pending.command_id,
            code="mcp_timeout",
            message="Splunk MCP timed out",
        ),
    )

    assert failed.state.phase is Phase.FAILED
    assert failed.state.failure is not None
    assert failed.state.conclusion is None


def test_fact_without_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one evidence"):
        Fact(id="fact-1", statement="Unsupported fact", evidence_ids=())


def test_refuted_hypotheses_are_derived_from_memory() -> None:
    refuted = Hypothesis(
        id="h-refuted",
        statement="The database was unavailable",
        verification_goal="find database connection failures",
        status=HypothesisStatus.REFUTED,
        contradicting_evidence_ids=("ev-healthy-db",),
    )
    open_hypothesis = Hypothesis(
        id="h-open",
        statement="The cache was unavailable",
        verification_goal="find cache connection failures",
    )

    memory = WorkingMemory(hypotheses=(refuted, open_hypothesis))

    assert memory.refuted_hypotheses == (refuted,)


def test_terminal_state_cannot_restart() -> None:
    terminal = transition(make_investigation(), CancelRequested()).state

    with pytest.raises(InvalidTransition, match="terminal phase"):
        transition(terminal, StartRequested())


def test_verify_query_requires_a_hypothesis() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=5), end=NOW)

    with pytest.raises(ValueError, match="must reference a hypothesis"):
        QueryIntent(
            kind=QueryKind.VERIFY,
            goal="verify something",
            time_range=time_range,
        )


def test_naive_time_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        TimeRange(
            start=datetime(2026, 8, 2, 0, 0),
            end=datetime(2026, 8, 2, 1, 0),
        )


def test_assessment_cannot_rewrite_hypothesis_text() -> None:
    state, hypotheses = advance_to_assessment_request()
    pending = state.pending_operation
    assert pending is not None
    rewritten = Hypothesis(
        id=hypotheses[0].id,
        statement="The database was unavailable",
        verification_goal=hypotheses[0].verification_goal,
        status=HypothesisStatus.SUPPORTED,
        supporting_evidence_ids=("ev-verify",),
    )

    with pytest.raises(InvariantViolation, match="cannot rewrite"):
        transition(
            state,
            VerificationAssessed(
                command_id=pending.command_id,
                hypothesis=rewritten,
                decision=VerificationDecision.CONCLUDE,
            ),
        )


def test_assessment_cannot_borrow_unrelated_triage_evidence() -> None:
    state, hypotheses = advance_to_assessment_request()
    pending = state.pending_operation
    assert pending is not None
    forged = replace(
        hypotheses[0],
        status=HypothesisStatus.SUPPORTED,
        supporting_evidence_ids=("ev-triage",),
    )

    with pytest.raises(InvariantViolation, match="verification query"):
        transition(
            state,
            VerificationAssessed(
                command_id=pending.command_id,
                hypothesis=forged,
                decision=VerificationDecision.CONCLUDE,
            ),
        )


def test_proposed_hypothesis_cannot_preload_triage_evidence() -> None:
    with pytest.raises(ValueError, match="preloaded evidence"):
        Hypothesis(
            id="h-preloaded",
            statement="The database timed out",
            verification_goal="find database timeouts",
            supporting_evidence_ids=("ev-triage",),
        )


def test_continue_query_cannot_expand_investigation_time_range() -> None:
    state, hypotheses = advance_to_assessment_request()
    pending = state.pending_operation
    assert pending is not None
    expanded_range = TimeRange(
        start=state.request.time_range.start - timedelta(days=1),
        end=state.request.time_range.end,
    )
    next_query = QueryIntent(
        kind=QueryKind.VERIFY,
        goal="search a much wider period",
        time_range=expanded_range,
        hypothesis_id=hypotheses[0].id,
    )
    testing = replace(hypotheses[0], status=HypothesisStatus.TESTING)

    with pytest.raises(InvariantViolation, match="time range"):
        transition(
            state,
            VerificationAssessed(
                command_id=pending.command_id,
                hypothesis=testing,
                decision=VerificationDecision.CONTINUE,
                next_query=next_query,
            ),
        )


def test_refuted_first_hypothesis_schedules_second_hypothesis() -> None:
    first = Hypothesis(
        id="h-first",
        statement="The database timed out",
        verification_goal="find database timeouts",
    )
    second = Hypothesis(
        id="h-second",
        statement="The payment service timed out",
        verification_goal="find payment timeouts",
    )
    state, _ = advance_to_assessment_request(
        hypotheses=(first, second),
        verification_evidence_id="ev-first",
    )
    pending = state.pending_operation
    assert pending is not None
    refuted = replace(
        first,
        status=HypothesisStatus.REFUTED,
        contradicting_evidence_ids=("ev-first",),
    )

    next_step = transition(
        state,
        VerificationAssessed(
            command_id=pending.command_id,
            hypothesis=refuted,
            decision=VerificationDecision.REHYPOTHESIZE,
        ),
    )

    command = next_step.commands[0]
    assert isinstance(command, ExecuteQuery)
    assert command.intent.hypothesis_id == second.id
    stored_second = next(item for item in next_step.state.memory.hypotheses if item.id == second.id)
    assert stored_second.status is HypothesisStatus.TESTING


def test_query_event_cannot_side_load_unregistered_evidence() -> None:
    started = transition(make_investigation(), StartRequested())
    command = started.commands[0]
    assert isinstance(command, ExecuteQuery)
    record = QueryRecord(
        id="q-side-load",
        intent=command.intent,
        result_count=1,
        summary="one result",
    )
    extra = EvidenceRef(
        id="ev-side-load",
        query_id=record.id,
        record_ref="row:q-side-load:1",
        occurred_at=NOW,
    )

    with pytest.raises(InvariantViolation, match="ledgers do not match"):
        transition(
            started.state,
            QuerySucceeded(
                command_id=command.command_id,
                record=record,
                evidence=(extra,),
            ),
        )


def test_zero_result_query_cannot_claim_evidence() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=5), end=NOW)
    intent = QueryIntent(kind=QueryKind.TRIAGE, goal="find errors", time_range=time_range)

    with pytest.raises(ValueError, match="more evidence"):
        QueryRecord(
            id="q-zero",
            intent=intent,
            result_count=0,
            summary="no results",
            evidence_ids=("ev-impossible",),
        )


def test_investigation_rejects_impossible_lifecycle_state() -> None:
    with pytest.raises(ValueError, match="termination reason"):
        replace(make_investigation(), phase=Phase.COMPLETED)
