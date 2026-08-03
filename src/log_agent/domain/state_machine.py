from __future__ import annotations

from dataclasses import dataclass, replace

from log_agent.domain.models import (
    Conclusion,
    ConclusionOutcome,
    EvidenceRef,
    Fact,
    Failure,
    Hypothesis,
    HypothesisStatus,
    Investigation,
    OperationKind,
    PendingOperation,
    Phase,
    QueryBudget,
    QueryIntent,
    QueryKind,
    QueryRecord,
    TerminationReason,
    VerificationDecision,
    WorkingMemory,
)


class InvalidTransition(ValueError):
    """Raised when an event is invalid for the current state."""


class InvariantViolation(ValueError):
    """Raised when an event would break a domain invariant."""


@dataclass(frozen=True, slots=True)
class StartRequested:
    pass


@dataclass(frozen=True, slots=True)
class QuerySucceeded:
    command_id: str
    record: QueryRecord
    facts: tuple[Fact, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class HypothesesGenerated:
    command_id: str
    hypotheses: tuple[Hypothesis, ...]


@dataclass(frozen=True, slots=True)
class VerificationAssessed:
    command_id: str
    hypothesis: Hypothesis
    decision: VerificationDecision
    next_query: QueryIntent | None = None

    def __post_init__(self) -> None:
        if self.decision is VerificationDecision.CONTINUE and self.next_query is None:
            raise ValueError("CONTINUE requires a next_query")
        if self.decision is not VerificationDecision.CONTINUE and self.next_query is not None:
            raise ValueError("next_query is only valid for CONTINUE")


@dataclass(frozen=True, slots=True)
class ConclusionGenerated:
    command_id: str
    conclusion: Conclusion


@dataclass(frozen=True, slots=True)
class OperationFailed:
    command_id: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CancelRequested:
    pass


Event = (
    StartRequested
    | QuerySucceeded
    | HypothesesGenerated
    | VerificationAssessed
    | ConclusionGenerated
    | OperationFailed
    | CancelRequested
)


@dataclass(frozen=True, slots=True)
class ExecuteQuery:
    command_id: str
    intent: QueryIntent


@dataclass(frozen=True, slots=True)
class GenerateHypotheses:
    command_id: str


@dataclass(frozen=True, slots=True)
class AssessVerification:
    command_id: str
    hypothesis_id: str
    query_id: str


@dataclass(frozen=True, slots=True)
class GenerateConclusion:
    command_id: str
    outcome: ConclusionOutcome
    termination_reason: TerminationReason
    root_cause_hypothesis_id: str | None = None


Command = ExecuteQuery | GenerateHypotheses | AssessVerification | GenerateConclusion


@dataclass(frozen=True, slots=True)
class Transition:
    state: Investigation
    commands: tuple[Command, ...] = ()


def transition(state: Investigation, event: Event) -> Transition:
    """Apply one event to the investigation and return its next command."""
    if state.phase.is_terminal:
        raise InvalidTransition(f"terminal phase {state.phase.value} cannot accept events")

    if isinstance(event, CancelRequested):
        return _terminal(
            state,
            phase=Phase.CANCELLED,
            reason=TerminationReason.USER_CANCELLED,
        )

    if isinstance(event, OperationFailed):
        _require_pending(state, event.command_id)
        return _terminal(
            state,
            phase=Phase.FAILED,
            reason=TerminationReason.OPERATION_FAILED,
            failure=Failure(code=event.code, message=event.message),
        )

    match state.phase, event:
        case Phase.NEW, StartRequested():
            return _issue_query_or_conclude(state, state.triage_plan[0], Phase.TRIAGE)
        case Phase.TRIAGE, QuerySucceeded():
            return _on_triage_query_succeeded(state, event)
        case Phase.HYPOTHESIZE, HypothesesGenerated():
            return _on_hypotheses_generated(state, event)
        case Phase.VERIFY, QuerySucceeded():
            return _on_verify_query_succeeded(state, event)
        case Phase.VERIFY, VerificationAssessed():
            return _on_verification_assessed(state, event)
        case Phase.CONCLUDE, ConclusionGenerated():
            return _on_conclusion_generated(state, event)
        case _:
            raise InvalidTransition(
                f"{type(event).__name__} is invalid while phase is {state.phase.value}"
            )


def _on_triage_query_succeeded(state: Investigation, event: QuerySucceeded) -> Transition:
    pending = _require_pending(state, event.command_id, OperationKind.EXECUTE_QUERY)
    if (
        pending.query_intent != event.record.intent
        or event.record.intent.kind is not QueryKind.TRIAGE
    ):
        raise InvariantViolation("query result does not match the pending triage intent")

    memory = _merge_query_result(state.memory, event)
    completed_triage = sum(query.intent.kind is QueryKind.TRIAGE for query in memory.queries)
    if completed_triage < len(state.triage_plan):
        next_intent = state.triage_plan[completed_triage]
        return _issue_query_or_conclude(
            state,
            next_intent,
            Phase.TRIAGE,
            memory=memory,
        )

    total_results = sum(
        query.result_count for query in memory.queries if query.intent.kind is QueryKind.TRIAGE
    )
    if total_results == 0:
        return _issue_conclusion(
            state,
            outcome=ConclusionOutcome.INCONCLUSIVE,
            reason=TerminationReason.NO_DATA,
            memory=memory,
        )
    return _issue_generate_hypotheses(state, memory)


def _on_hypotheses_generated(state: Investigation, event: HypothesesGenerated) -> Transition:
    _require_pending(state, event.command_id, OperationKind.GENERATE_HYPOTHESES)
    if len(event.hypotheses) > 3:
        raise InvariantViolation("a hypothesis cycle may produce at most three hypotheses")
    if any(h.status is not HypothesisStatus.PROPOSED for h in event.hypotheses):
        raise InvariantViolation("new hypotheses must start in PROPOSED status")

    memory = _merge_hypotheses(state.memory, event.hypotheses)
    if not event.hypotheses:
        return _issue_conclusion(
            state,
            outcome=ConclusionOutcome.INCONCLUSIVE,
            reason=TerminationReason.INSUFFICIENT_EVIDENCE,
            memory=memory,
        )

    return _issue_verification_query(state, event.hypotheses[0], memory)


def _on_verify_query_succeeded(state: Investigation, event: QuerySucceeded) -> Transition:
    pending = _require_pending(state, event.command_id, OperationKind.EXECUTE_QUERY)
    if (
        pending.query_intent != event.record.intent
        or event.record.intent.kind is not QueryKind.VERIFY
    ):
        raise InvariantViolation("query result does not match the pending verification intent")
    if pending.query_intent.hypothesis_id is None:
        raise InvariantViolation("verification query is missing its hypothesis")

    memory = _merge_query_result(state.memory, event)
    command_id = _next_command_id(state)
    hypothesis_id = pending.query_intent.hypothesis_id
    command = AssessVerification(
        command_id=command_id,
        hypothesis_id=hypothesis_id,
        query_id=event.record.id,
    )
    return _advance(
        state,
        phase=Phase.VERIFY,
        memory=memory,
        pending=PendingOperation(
            command_id=command_id,
            kind=OperationKind.ASSESS_VERIFICATION,
            hypothesis_id=hypothesis_id,
            query_id=event.record.id,
        ),
        commands=(command,),
    )


def _on_verification_assessed(state: Investigation, event: VerificationAssessed) -> Transition:
    pending = _require_pending(state, event.command_id, OperationKind.ASSESS_VERIFICATION)
    if pending.hypothesis_id != event.hypothesis.id:
        raise InvariantViolation("assessment does not match the pending hypothesis")

    current = _find_hypothesis(state.memory, event.hypothesis.id)
    if current.status is not HypothesisStatus.TESTING:
        raise InvariantViolation("only a hypothesis under test can be assessed")
    if (
        current.statement != event.hypothesis.statement
        or current.verification_goal != event.hypothesis.verification_goal
    ):
        raise InvariantViolation("assessment cannot rewrite the hypothesis")
    if event.hypothesis.status is HypothesisStatus.PROPOSED:
        raise InvariantViolation("assessment cannot move a hypothesis back to PROPOSED")

    query = _find_query(state.memory, pending.query_id)
    existing_references = set(current.supporting_evidence_ids)
    existing_references.update(current.contradicting_evidence_ids)
    referenced = set(event.hypothesis.supporting_evidence_ids)
    referenced.update(event.hypothesis.contradicting_evidence_ids)
    if not existing_references.issubset(referenced):
        raise InvariantViolation("assessment cannot discard previously cited evidence")
    allowed_references = existing_references | set(query.evidence_ids)
    if not referenced.issubset(allowed_references):
        raise InvariantViolation("assessment may only cite evidence from its verification query")
    memory = _replace_hypothesis(state.memory, event.hypothesis)

    if event.decision is VerificationDecision.CONTINUE:
        assert event.next_query is not None
        if event.hypothesis.status is not HypothesisStatus.TESTING:
            raise InvariantViolation("CONTINUE requires a hypothesis in TESTING status")
        if event.next_query.kind is not QueryKind.VERIFY:
            raise InvariantViolation("verification may only continue with a verify query")
        if event.next_query.hypothesis_id != event.hypothesis.id:
            raise InvariantViolation("next query must target the assessed hypothesis")
        if not _time_range_is_within(event.next_query, state):
            raise InvariantViolation("next query exceeds the investigation time range")
        return _issue_query_or_conclude(
            state,
            event.next_query,
            Phase.VERIFY,
            memory=memory,
        )

    if event.hypothesis.status is HypothesisStatus.SUPPORTED:
        if event.decision is not VerificationDecision.CONCLUDE:
            raise InvariantViolation("a supported hypothesis must conclude verification")
        return _issue_conclusion(
            state,
            outcome=ConclusionOutcome.CONCLUSIVE,
            reason=TerminationReason.ROOT_CAUSE_IDENTIFIED,
            memory=memory,
            root_cause_hypothesis_id=event.hypothesis.id,
        )

    next_hypothesis = next(
        (
            hypothesis
            for hypothesis in memory.hypotheses
            if hypothesis.status is HypothesisStatus.PROPOSED
        ),
        None,
    )
    if next_hypothesis is not None:
        return _issue_verification_query(state, next_hypothesis, memory)

    if event.decision is VerificationDecision.REHYPOTHESIZE:
        return _issue_generate_hypotheses(state, memory)

    return _issue_conclusion(
        state,
        outcome=ConclusionOutcome.INCONCLUSIVE,
        reason=TerminationReason.INSUFFICIENT_EVIDENCE,
        memory=memory,
    )


def _on_conclusion_generated(state: Investigation, event: ConclusionGenerated) -> Transition:
    pending = _require_pending(state, event.command_id, OperationKind.GENERATE_CONCLUSION)
    conclusion = event.conclusion
    if pending.conclusion_outcome is not conclusion.outcome:
        raise InvariantViolation("conclusion outcome does not match the requested outcome")
    if pending.termination_reason is not conclusion.termination_reason:
        raise InvariantViolation("conclusion reason does not match the requested reason")
    evidence_ids = {evidence.id for evidence in state.memory.evidence}
    if not set(conclusion.evidence_ids).issubset(evidence_ids):
        raise InvariantViolation("conclusion references unknown evidence")

    if conclusion.outcome is ConclusionOutcome.CONCLUSIVE:
        if pending.hypothesis_id != conclusion.root_cause_hypothesis_id:
            raise InvariantViolation("conclusion changed the requested root-cause hypothesis")
        hypothesis = _find_hypothesis(state.memory, conclusion.root_cause_hypothesis_id)
        if hypothesis.status is not HypothesisStatus.SUPPORTED:
            raise InvariantViolation("root-cause hypothesis is not supported")
        if not set(conclusion.evidence_ids).issubset(hypothesis.supporting_evidence_ids):
            raise InvariantViolation("conclusion evidence must support the root-cause hypothesis")
        phase = Phase.COMPLETED
    else:
        phase = Phase.INCONCLUSIVE

    return _terminal(
        state,
        phase=phase,
        reason=conclusion.termination_reason,
        conclusion=conclusion,
    )


def _issue_query_or_conclude(
    state: Investigation,
    intent: QueryIntent,
    phase: Phase,
    *,
    memory: WorkingMemory | None = None,
) -> Transition:
    memory = state.memory if memory is None else memory
    if not state.budget.can_issue(intent.kind):
        return _issue_conclusion(
            state,
            outcome=ConclusionOutcome.INCONCLUSIVE,
            reason=TerminationReason.QUERY_BUDGET_EXHAUSTED,
            memory=memory,
        )

    command_id = _next_command_id(state)
    command = ExecuteQuery(command_id=command_id, intent=intent)
    return _advance(
        state,
        phase=phase,
        memory=memory,
        budget=state.budget.issue(intent.kind),
        pending=PendingOperation(
            command_id=command_id,
            kind=OperationKind.EXECUTE_QUERY,
            query_intent=intent,
            hypothesis_id=intent.hypothesis_id,
        ),
        commands=(command,),
    )


def _issue_generate_hypotheses(state: Investigation, memory: WorkingMemory) -> Transition:
    command_id = _next_command_id(state)
    command = GenerateHypotheses(command_id=command_id)
    return _advance(
        state,
        phase=Phase.HYPOTHESIZE,
        memory=memory,
        pending=PendingOperation(
            command_id=command_id,
            kind=OperationKind.GENERATE_HYPOTHESES,
        ),
        commands=(command,),
    )


def _issue_verification_query(
    state: Investigation,
    hypothesis: Hypothesis,
    memory: WorkingMemory,
) -> Transition:
    intent = QueryIntent(
        kind=QueryKind.VERIFY,
        goal=hypothesis.verification_goal,
        time_range=state.request.time_range,
        hypothesis_id=hypothesis.id,
    )
    if not state.budget.can_issue(QueryKind.VERIFY):
        return _issue_conclusion(
            state,
            outcome=ConclusionOutcome.INCONCLUSIVE,
            reason=TerminationReason.QUERY_BUDGET_EXHAUSTED,
            memory=memory,
        )
    testing = replace(hypothesis, status=HypothesisStatus.TESTING)
    memory = _replace_hypothesis(memory, testing)
    return _issue_query_or_conclude(state, intent, Phase.VERIFY, memory=memory)


def _issue_conclusion(
    state: Investigation,
    *,
    outcome: ConclusionOutcome,
    reason: TerminationReason,
    memory: WorkingMemory,
    root_cause_hypothesis_id: str | None = None,
) -> Transition:
    command_id = _next_command_id(state)
    command = GenerateConclusion(
        command_id=command_id,
        outcome=outcome,
        termination_reason=reason,
        root_cause_hypothesis_id=root_cause_hypothesis_id,
    )
    return _advance(
        state,
        phase=Phase.CONCLUDE,
        memory=memory,
        pending=PendingOperation(
            command_id=command_id,
            kind=OperationKind.GENERATE_CONCLUSION,
            hypothesis_id=root_cause_hypothesis_id,
            conclusion_outcome=outcome,
            termination_reason=reason,
        ),
        commands=(command,),
    )


def _merge_query_result(memory: WorkingMemory, event: QuerySucceeded) -> WorkingMemory:
    if any(query.id == event.record.id for query in memory.queries):
        raise InvariantViolation(f"duplicate query id: {event.record.id}")

    new_evidence_ids = {evidence.id for evidence in event.evidence}
    if len(new_evidence_ids) != len(event.evidence):
        raise InvariantViolation("query event contains duplicate evidence ids")
    known_evidence_ids = {evidence.id for evidence in memory.evidence}
    if new_evidence_ids & known_evidence_ids:
        raise InvariantViolation("query event reuses an existing evidence id")
    if any(evidence.query_id != event.record.id for evidence in event.evidence):
        raise InvariantViolation("evidence query_id does not match its query record")
    if set(event.record.evidence_ids) != new_evidence_ids:
        raise InvariantViolation("query record and event evidence ledgers do not match")

    known_fact_ids = {fact.id for fact in memory.facts}
    new_fact_ids = tuple(fact.id for fact in event.facts)
    if len(new_fact_ids) != len(set(new_fact_ids)):
        raise InvariantViolation("query event contains duplicate fact ids")
    if any(fact.id in known_fact_ids for fact in event.facts):
        raise InvariantViolation("query event reuses an existing fact id")
    if any(not set(fact.evidence_ids).issubset(new_evidence_ids) for fact in event.facts):
        raise InvariantViolation("query fact must cite evidence from the same query")

    return replace(
        memory,
        facts=memory.facts + event.facts,
        evidence=memory.evidence + event.evidence,
        queries=memory.queries + (event.record,),
    )


def _merge_hypotheses(memory: WorkingMemory, hypotheses: tuple[Hypothesis, ...]) -> WorkingMemory:
    new_ids = tuple(hypothesis.id for hypothesis in hypotheses)
    if len(new_ids) != len(set(new_ids)):
        raise InvariantViolation("hypothesis event contains duplicate ids")
    known_ids = {hypothesis.id for hypothesis in memory.hypotheses}
    if known_ids & set(new_ids):
        raise InvariantViolation("new hypothesis reuses an existing id")
    return replace(memory, hypotheses=memory.hypotheses + hypotheses)


def _replace_hypothesis(memory: WorkingMemory, replacement: Hypothesis) -> WorkingMemory:
    if not any(hypothesis.id == replacement.id for hypothesis in memory.hypotheses):
        raise InvariantViolation(f"unknown hypothesis: {replacement.id}")
    return replace(
        memory,
        hypotheses=tuple(
            replacement if hypothesis.id == replacement.id else hypothesis
            for hypothesis in memory.hypotheses
        ),
    )


def _find_hypothesis(memory: WorkingMemory, hypothesis_id: str | None) -> Hypothesis:
    for hypothesis in memory.hypotheses:
        if hypothesis.id == hypothesis_id:
            return hypothesis
    raise InvariantViolation(f"unknown hypothesis: {hypothesis_id}")


def _find_query(memory: WorkingMemory, query_id: str | None) -> QueryRecord:
    for query in memory.queries:
        if query.id == query_id:
            return query
    raise InvariantViolation(f"unknown query: {query_id}")


def _time_range_is_within(intent: QueryIntent, state: Investigation) -> bool:
    requested = state.request.time_range
    return intent.time_range.start >= requested.start and intent.time_range.end <= requested.end


def _require_pending(
    state: Investigation,
    command_id: str,
    kind: OperationKind | None = None,
) -> PendingOperation:
    pending = state.pending_operation
    if pending is None:
        raise InvalidTransition("no external operation is pending")
    if pending.command_id != command_id:
        raise InvalidTransition("event command_id does not match the pending operation")
    if kind is not None and pending.kind is not kind:
        raise InvalidTransition(f"expected pending {kind.value}, found {pending.kind.value}")
    return pending


def _next_command_id(state: Investigation) -> str:
    return f"{state.id}:{state.revision + 1}"


def _advance(
    state: Investigation,
    *,
    phase: Phase,
    memory: WorkingMemory | None = None,
    budget: QueryBudget | None = None,
    pending: PendingOperation | None,
    commands: tuple[Command, ...],
) -> Transition:
    next_state = replace(
        state,
        phase=phase,
        memory=state.memory if memory is None else memory,
        budget=state.budget if budget is None else budget,
        pending_operation=pending,
        revision=state.revision + 1,
    )
    return Transition(state=next_state, commands=commands)


def _terminal(
    state: Investigation,
    *,
    phase: Phase,
    reason: TerminationReason,
    conclusion: Conclusion | None = None,
    failure: Failure | None = None,
) -> Transition:
    return Transition(
        state=replace(
            state,
            phase=phase,
            pending_operation=None,
            conclusion=conclusion,
            failure=failure,
            termination_reason=reason,
            revision=state.revision + 1,
        )
    )
