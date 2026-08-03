from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from log_agent.application.query_security import AuthorizedQueryPlan
from log_agent.domain.models import (
    Conclusion,
    ConclusionOutcome,
    EvidenceRef,
    Fact,
    Hypothesis,
    InvestigationRequest,
    QueryIntent,
    QueryRecord,
    TerminationReason,
    VerificationDecision,
    WorkingMemory,
)

_MAX_SEARCH_SUMMARY_LENGTH = 2_000
_MAX_FACT_STATEMENT_LENGTH = 2_000
_MAX_RECORD_REF_LENGTH = 2_000


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Application-owned input contract carrying only an authorized query plan."""

    operation_id: str
    query_id: str
    authorized_query: AuthorizedQueryPlan

    def __post_init__(self) -> None:
        _require_text(self.operation_id, "search operation_id")
        _require_text(self.query_id, "search query_id")
        if not isinstance(self.authorized_query, AuthorizedQueryPlan):
            raise ValueError("search authorized_query is invalid")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Normalized search output, independent of any Splunk transport.

    result_count is the number of normalized rows returned to the application,
    not the backend's total number of matches. A future total_matches field may
    carry that separate value once the real MCP contract is known.

    The command handler must additionally verify that every evidence query_id
    matches the SearchRequest.query_id that produced this result.
    """

    result_count: int
    summary: str
    truncated: bool = False
    evidence: tuple[EvidenceRef, ...] = ()
    facts: tuple[Fact, ...] = ()

    def __post_init__(self) -> None:
        if type(self.result_count) is not int:
            raise ValueError("search result_count must be an integer")
        if self.result_count < 0:
            raise ValueError("search result_count must not be negative")
        if type(self.truncated) is not bool:
            raise ValueError("search truncated must be a boolean")
        _require_text(self.summary, "search summary")
        if len(self.summary) > _MAX_SEARCH_SUMMARY_LENGTH:
            raise ValueError("search summary exceeds the application limit")

        evidence_ids = tuple(item.id for item in self.evidence)
        fact_ids = tuple(item.id for item in self.facts)
        _require_unique(evidence_ids, "search evidence ids")
        _require_unique(fact_ids, "search fact ids")

        if len(self.evidence) > self.result_count:
            raise ValueError("search cannot expose more evidence items than result rows")
        if len(self.facts) > self.result_count:
            raise ValueError("search cannot expose more facts than result rows")
        if any(len(item.record_ref) > _MAX_RECORD_REF_LENGTH for item in self.evidence):
            raise ValueError("search evidence record_ref exceeds the application limit")
        if any(len(item.statement) > _MAX_FACT_STATEMENT_LENGTH for item in self.facts):
            raise ValueError("search fact statement exceeds the application limit")

        known_evidence = set(evidence_ids)
        if any(not set(fact.evidence_ids).issubset(known_evidence) for fact in self.facts):
            raise ValueError("every search fact must reference evidence from this result")


@dataclass(frozen=True, slots=True)
class VerificationAssessment:
    """Normalized reasoning output for one hypothesis-verification step."""

    hypothesis: Hypothesis
    decision: VerificationDecision
    next_query: QueryIntent | None = None

    def __post_init__(self) -> None:
        if self.decision is VerificationDecision.CONTINUE and self.next_query is None:
            raise ValueError("CONTINUE requires a next_query")
        if self.decision is not VerificationDecision.CONTINUE and self.next_query is not None:
            raise ValueError("next_query is only valid for CONTINUE")


class PortError(RuntimeError):
    """An external failure whose adapter-provided message must already be sanitized."""

    def __init__(self, code: str, safe_message: str) -> None:
        _require_text(code, "port error code")
        _require_text(safe_message, "port error safe_message")
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


class PortUnavailable(PortError):
    """The external capability is temporarily unavailable."""


class PortProtocolError(PortError):
    """The external capability returned data that violates the port contract."""


class LogSearchPort(Protocol):
    """Capability required by the application to execute an authorized query plan.

    Adapters must render only authorized_query. The original QueryIntent and raw
    SPL are absent; any dynamic terms have already been approved as literal data.
    """

    async def search(self, request: SearchRequest) -> SearchResult:
        """Run one search; coroutine cancellation must propagate unchanged."""
        ...


class ReasoningPort(Protocol):
    """Stage-specific reasoning capabilities required by the application."""

    async def generate_hypotheses(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
    ) -> tuple[Hypothesis, ...]: ...

    async def assess_verification(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
        hypothesis: Hypothesis,
        query: QueryRecord,
    ) -> VerificationAssessment: ...

    async def generate_conclusion(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
        outcome: ConclusionOutcome,
        termination_reason: TerminationReason,
        root_cause_hypothesis_id: str | None,
    ) -> Conclusion: ...
