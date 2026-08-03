from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum


class Phase(StrEnum):
    NEW = "new"
    TRIAGE = "triage"
    HYPOTHESIZE = "hypothesize"
    VERIFY = "verify"
    CONCLUDE = "conclude"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            Phase.COMPLETED,
            Phase.INCONCLUSIVE,
            Phase.FAILED,
            Phase.CANCELLED,
        }


class QueryKind(StrEnum):
    TRIAGE = "triage"
    VERIFY = "verify"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"


class ConclusionOutcome(StrEnum):
    CONCLUSIVE = "conclusive"
    INCONCLUSIVE = "inconclusive"


class TerminationReason(StrEnum):
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    NO_DATA = "no_data"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    QUERY_BUDGET_EXHAUSTED = "query_budget_exhausted"
    OPERATION_FAILED = "operation_failed"
    USER_CANCELLED = "user_cancelled"


class OperationKind(StrEnum):
    EXECUTE_QUERY = "execute_query"
    GENERATE_HYPOTHESES = "generate_hypotheses"
    ASSESS_VERIFICATION = "assess_verification"
    GENERATE_CONCLUSION = "generate_conclusion"


class VerificationDecision(StrEnum):
    CONTINUE = "continue"
    REHYPOTHESIZE = "rehypothesize"
    CONCLUDE = "conclude"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not _is_timezone_aware(self.start) or not _is_timezone_aware(self.end):
            raise ValueError("time range must use timezone-aware datetimes")
        if self.start >= self.end:
            raise ValueError("time range start must be before end")


@dataclass(frozen=True, slots=True)
class InvestigationRequest:
    question: str
    scope_ref: str
    time_range: TimeRange

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.scope_ref, "scope_ref")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    id: str
    query_id: str
    record_ref: str
    occurred_at: datetime | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.id, "evidence id")
        _require_text(self.query_id, "evidence query_id")
        _require_text(self.record_ref, "evidence record_ref")
        if self.occurred_at is not None and not _is_timezone_aware(self.occurred_at):
            raise ValueError("evidence occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    statement: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.id, "fact id")
        _require_text(self.statement, "fact statement")
        if not self.evidence_ids:
            raise ValueError("a fact must reference at least one evidence item")
        _require_unique(self.evidence_ids, "fact evidence_ids")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    statement: str
    verification_goal: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.id, "hypothesis id")
        _require_text(self.statement, "hypothesis statement")
        _require_text(self.verification_goal, "hypothesis verification_goal")
        _require_unique(self.supporting_evidence_ids, "supporting_evidence_ids")
        _require_unique(self.contradicting_evidence_ids, "contradicting_evidence_ids")
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ValueError("the same evidence cannot both support and contradict a hypothesis")
        if self.status is HypothesisStatus.PROPOSED and (
            self.supporting_evidence_ids or self.contradicting_evidence_ids
        ):
            raise ValueError("a proposed hypothesis cannot carry preloaded evidence")
        if self.status is HypothesisStatus.SUPPORTED and not self.supporting_evidence_ids:
            raise ValueError("a supported hypothesis must reference supporting evidence")
        if self.status is HypothesisStatus.REFUTED and not self.contradicting_evidence_ids:
            raise ValueError("a refuted hypothesis must reference contradicting evidence")


@dataclass(frozen=True, slots=True)
class QueryIntent:
    kind: QueryKind
    goal: str
    time_range: TimeRange
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.goal, "query goal")
        if self.kind is QueryKind.VERIFY and not self.hypothesis_id:
            raise ValueError("a verify query must reference a hypothesis")
        if self.kind is QueryKind.TRIAGE and self.hypothesis_id is not None:
            raise ValueError("a triage query cannot reference a hypothesis")


@dataclass(frozen=True, slots=True)
class QueryRecord:
    id: str
    intent: QueryIntent
    result_count: int
    summary: str
    truncated: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.id, "query id")
        _require_text(self.summary, "query summary")
        if type(self.result_count) is not int:
            raise ValueError("query result_count must be an integer")
        if self.result_count < 0:
            raise ValueError("query result_count must not be negative")
        if type(self.truncated) is not bool:
            raise ValueError("query truncated must be a boolean")
        _require_unique(self.evidence_ids, "query evidence_ids")
        if len(self.evidence_ids) > self.result_count:
            raise ValueError("query cannot expose more evidence items than result rows")


@dataclass(frozen=True, slots=True)
class QueryBudget:
    max_total_queries: int
    max_verify_queries: int
    issued_total: int = 0
    issued_verify: int = 0

    def __post_init__(self) -> None:
        if self.max_total_queries < 1:
            raise ValueError("max_total_queries must be positive")
        if not 0 <= self.max_verify_queries <= self.max_total_queries:
            raise ValueError("max_verify_queries must be between zero and total budget")
        if not 0 <= self.issued_total <= self.max_total_queries:
            raise ValueError("issued_total is outside the configured budget")
        if not 0 <= self.issued_verify <= self.max_verify_queries:
            raise ValueError("issued_verify is outside the configured budget")
        if self.issued_verify > self.issued_total:
            raise ValueError("verify query count cannot exceed total query count")

    def can_issue(self, kind: QueryKind) -> bool:
        if self.issued_total >= self.max_total_queries:
            return False
        return kind is not QueryKind.VERIFY or self.issued_verify < self.max_verify_queries

    def issue(self, kind: QueryKind) -> QueryBudget:
        if not self.can_issue(kind):
            raise ValueError(f"query budget exhausted for {kind.value}")
        return replace(
            self,
            issued_total=self.issued_total + 1,
            issued_verify=self.issued_verify + (kind is QueryKind.VERIFY),
        )


@dataclass(frozen=True, slots=True)
class WorkingMemory:
    facts: tuple[Fact, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    queries: tuple[QueryRecord, ...] = ()

    @property
    def refuted_hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.status is HypothesisStatus.REFUTED)


@dataclass(frozen=True, slots=True)
class PendingOperation:
    command_id: str
    kind: OperationKind
    query_intent: QueryIntent | None = None
    query_id: str | None = None
    hypothesis_id: str | None = None
    conclusion_outcome: ConclusionOutcome | None = None
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        _require_text(self.command_id, "pending command_id")
        if self.kind is OperationKind.EXECUTE_QUERY and self.query_intent is None:
            raise ValueError("an execute-query operation requires a query intent")
        if self.kind is OperationKind.ASSESS_VERIFICATION and (
            not self.hypothesis_id or not self.query_id
        ):
            raise ValueError("an assessment operation requires a hypothesis and query")
        if self.kind is OperationKind.GENERATE_CONCLUSION and (
            self.conclusion_outcome is None or self.termination_reason is None
        ):
            raise ValueError("a conclusion operation requires an outcome and reason")


@dataclass(frozen=True, slots=True)
class Conclusion:
    outcome: ConclusionOutcome
    summary: str
    termination_reason: TerminationReason
    root_cause_hypothesis_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.summary, "conclusion summary")
        _require_unique(self.evidence_ids, "conclusion evidence_ids")
        if self.outcome is ConclusionOutcome.CONCLUSIVE:
            if not self.root_cause_hypothesis_id:
                raise ValueError("a conclusive result must reference a root-cause hypothesis")
            if not self.evidence_ids:
                raise ValueError("a conclusive result must reference evidence")
            if self.termination_reason is not TerminationReason.ROOT_CAUSE_IDENTIFIED:
                raise ValueError("a conclusive result requires ROOT_CAUSE_IDENTIFIED")
        else:
            if self.root_cause_hypothesis_id is not None:
                raise ValueError("an inconclusive result cannot claim a root-cause hypothesis")
            if self.termination_reason is TerminationReason.ROOT_CAUSE_IDENTIFIED:
                raise ValueError("an inconclusive result cannot claim ROOT_CAUSE_IDENTIFIED")
            if self.termination_reason in {
                TerminationReason.OPERATION_FAILED,
                TerminationReason.USER_CANCELLED,
            }:
                raise ValueError("failures and cancellations are not diagnostic conclusions")


@dataclass(frozen=True, slots=True)
class Failure:
    code: str
    message: str

    def __post_init__(self) -> None:
        _require_text(self.code, "failure code")
        _require_text(self.message, "failure message")


@dataclass(frozen=True, slots=True)
class Investigation:
    id: str
    request: InvestigationRequest
    triage_plan: tuple[QueryIntent, ...]
    budget: QueryBudget
    phase: Phase = Phase.NEW
    memory: WorkingMemory = field(default_factory=WorkingMemory)
    pending_operation: PendingOperation | None = None
    conclusion: Conclusion | None = None
    failure: Failure | None = None
    termination_reason: TerminationReason | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        _require_text(self.id, "investigation id")
        if not self.triage_plan:
            raise ValueError("triage_plan must contain at least one query intent")
        if any(intent.kind is not QueryKind.TRIAGE for intent in self.triage_plan):
            raise ValueError("triage_plan may only contain triage query intents")
        if any(intent.time_range != self.request.time_range for intent in self.triage_plan):
            raise ValueError("triage query time ranges must match the investigation request")
        goals = tuple(intent.goal for intent in self.triage_plan)
        _require_unique(goals, "triage query goals")
        if self.revision < 0:
            raise ValueError("revision must not be negative")
        self._validate_memory_references()
        self._validate_lifecycle()

    def _validate_memory_references(self) -> None:
        evidence_ids = tuple(item.id for item in self.memory.evidence)
        query_ids = tuple(item.id for item in self.memory.queries)
        fact_ids = tuple(item.id for item in self.memory.facts)
        hypothesis_ids = tuple(item.id for item in self.memory.hypotheses)
        _require_unique(evidence_ids, "working-memory evidence ids")
        _require_unique(query_ids, "working-memory query ids")
        _require_unique(fact_ids, "working-memory fact ids")
        _require_unique(hypothesis_ids, "working-memory hypothesis ids")

        known_evidence = set(evidence_ids)
        known_queries = set(query_ids)
        if any(item.query_id not in known_queries for item in self.memory.evidence):
            raise ValueError("every evidence item must reference a known query")
        if any(not set(item.evidence_ids).issubset(known_evidence) for item in self.memory.facts):
            raise ValueError("every fact must reference known evidence")
        if any(
            not (set(item.supporting_evidence_ids) | set(item.contradicting_evidence_ids)).issubset(
                known_evidence
            )
            for item in self.memory.hypotheses
        ):
            raise ValueError("every hypothesis must reference known evidence")
        for query in self.memory.queries:
            saved_for_query = {
                item.id for item in self.memory.evidence if item.query_id == query.id
            }
            if set(query.evidence_ids) != saved_for_query:
                raise ValueError("query evidence ledger does not match saved evidence")

    def _validate_lifecycle(self) -> None:
        active_operation_kinds = {
            Phase.TRIAGE: {OperationKind.EXECUTE_QUERY},
            Phase.HYPOTHESIZE: {OperationKind.GENERATE_HYPOTHESES},
            Phase.VERIFY: {
                OperationKind.EXECUTE_QUERY,
                OperationKind.ASSESS_VERIFICATION,
            },
            Phase.CONCLUDE: {OperationKind.GENERATE_CONCLUSION},
        }
        if self.phase is Phase.NEW:
            if self.pending_operation is not None:
                raise ValueError("a new investigation cannot have a pending operation")
        elif self.phase in active_operation_kinds:
            if self.pending_operation is None:
                raise ValueError(f"phase {self.phase.value} requires a pending operation")
            if self.pending_operation.kind not in active_operation_kinds[self.phase]:
                raise ValueError("pending operation does not match the investigation phase")
        elif self.pending_operation is not None:
            raise ValueError("a terminal investigation cannot have a pending operation")

        if not self.phase.is_terminal:
            if self.conclusion is not None or self.failure is not None:
                raise ValueError("a running investigation cannot contain a terminal result")
            if self.termination_reason is not None:
                raise ValueError("a running investigation cannot have a termination reason")
            return

        if self.termination_reason is None:
            raise ValueError("a terminal investigation requires a termination reason")
        if self.phase is Phase.COMPLETED:
            if (
                self.conclusion is None
                or self.conclusion.outcome is not ConclusionOutcome.CONCLUSIVE
            ):
                raise ValueError("a completed investigation requires a conclusive result")
            if self.failure is not None:
                raise ValueError("a completed investigation cannot contain a failure")
        elif self.phase is Phase.INCONCLUSIVE:
            if (
                self.conclusion is None
                or self.conclusion.outcome is not ConclusionOutcome.INCONCLUSIVE
            ):
                raise ValueError("an inconclusive investigation requires an inconclusive result")
            if self.failure is not None:
                raise ValueError("an inconclusive investigation cannot contain a failure")
        elif self.phase is Phase.FAILED:
            if self.failure is None or self.conclusion is not None:
                raise ValueError("a failed investigation requires only a failure")
            if self.termination_reason is not TerminationReason.OPERATION_FAILED:
                raise ValueError("a failed investigation requires OPERATION_FAILED")
        elif self.phase is Phase.CANCELLED:
            if self.conclusion is not None or self.failure is not None:
                raise ValueError("a cancelled investigation cannot contain a result")
            if self.termination_reason is not TerminationReason.USER_CANCELLED:
                raise ValueError("a cancelled investigation requires USER_CANCELLED")

        if self.conclusion is not None:
            if self.conclusion.termination_reason is not self.termination_reason:
                raise ValueError("conclusion and investigation termination reasons must match")
            known_evidence = {item.id for item in self.memory.evidence}
            if not set(self.conclusion.evidence_ids).issubset(known_evidence):
                raise ValueError("conclusion must reference known evidence")
            if self.conclusion.root_cause_hypothesis_id is not None:
                root = next(
                    (
                        item
                        for item in self.memory.hypotheses
                        if item.id == self.conclusion.root_cause_hypothesis_id
                    ),
                    None,
                )
                if root is None or root.status is not HypothesisStatus.SUPPORTED:
                    raise ValueError("root-cause hypothesis must exist and be supported")
                if not set(self.conclusion.evidence_ids).issubset(root.supporting_evidence_ids):
                    raise ValueError("conclusion evidence must support the root cause")
