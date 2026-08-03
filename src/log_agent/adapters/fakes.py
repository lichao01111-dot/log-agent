from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime

from log_agent.application.model_ports import (
    StructuredModelRequest,
    StructuredModelResponse,
)
from log_agent.application.ports import (
    PortProtocolError,
    SearchRequest,
    SearchResult,
    VerificationAssessment,
)
from log_agent.domain.models import (
    Conclusion,
    ConclusionOutcome,
    EvidenceRef,
    Fact,
    Hypothesis,
    HypothesisStatus,
    InvestigationRequest,
    QueryRecord,
    TerminationReason,
    VerificationDecision,
    WorkingMemory,
)


@dataclass(frozen=True, slots=True)
class FakeLogRow:
    record_ref: str
    fact_statement: str
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.record_ref.strip():
            raise ValueError("fake row record_ref must not be blank")
        if not self.fact_statement.strip():
            raise ValueError("fake row fact_statement must not be blank")
        if self.occurred_at is not None and (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("fake row occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FakeSearchResponse:
    summary: str
    rows: tuple[FakeLogRow, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("fake response summary must not be blank")


class FakeLogSearchPort:
    """Deterministic in-memory log adapter keyed by authorized template ID."""

    def __init__(self, responses_by_template: Mapping[str, FakeSearchResponse]) -> None:
        self._responses_by_template = dict(responses_by_template)
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        template_id = request.authorized_query.plan.template_id
        response = self._responses_by_template.get(template_id)
        if response is None:
            raise PortProtocolError(
                "fake_log_search.unexpected_query",
                "Fake log search received an unexpected query.",
            ) from None

        evidence = tuple(
            EvidenceRef(
                id=f"{request.query_id}:evidence:{index}",
                query_id=request.query_id,
                record_ref=row.record_ref,
                occurred_at=row.occurred_at,
            )
            for index, row in enumerate(response.rows, start=1)
        )
        facts = tuple(
            Fact(
                id=f"{request.query_id}:fact:{index}",
                statement=row.fact_statement,
                evidence_ids=(evidence[index - 1].id,),
            )
            for index, row in enumerate(response.rows, start=1)
        )
        return SearchResult(
            result_count=len(response.rows),
            summary=response.summary,
            truncated=response.truncated,
            evidence=evidence,
            facts=facts,
        )


class FakeStructuredModelClient:
    """Scripted provider-neutral model client for adapter and integration tests."""

    def __init__(self, responses: tuple[StructuredModelResponse, ...]) -> None:
        if not isinstance(responses, tuple) or not responses:
            raise ValueError("fake structured model requires at least one response")
        if any(not isinstance(item, StructuredModelResponse) for item in responses):
            raise ValueError("fake structured model responses contain an invalid item")
        self._responses = responses
        self._position = 0
        self.requests: list[StructuredModelRequest] = []

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResponse:
        if not isinstance(request, StructuredModelRequest):
            raise PortProtocolError(
                "fake_model.invalid_request",
                "Fake structured model received an invalid request.",
            ) from None
        self.requests.append(request)
        if self._position >= len(self._responses):
            raise PortProtocolError(
                "fake_model.unexpected_request",
                "Fake structured model received an unexpected request.",
            ) from None
        response = self._responses[self._position]
        self._position += 1
        return response


class DeterministicReasoningPort:
    """Small rule-based reasoner used to test orchestration without an LLM."""

    def __init__(
        self,
        *,
        hypothesis_id: str,
        hypothesis_statement: str,
        verification_goal: str,
        conclusion_summary: str,
        recommendations: tuple[str, ...] = (),
    ) -> None:
        for value, field_name in (
            (hypothesis_id, "hypothesis_id"),
            (hypothesis_statement, "hypothesis_statement"),
            (verification_goal, "verification_goal"),
            (conclusion_summary, "conclusion_summary"),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        self._hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=hypothesis_statement,
            verification_goal=verification_goal,
        )
        self._conclusion_summary = conclusion_summary
        self._recommendations = recommendations
        self.calls: list[str] = []

    async def generate_hypotheses(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
    ) -> tuple[Hypothesis, ...]:
        del request, memory
        self.calls.append("generate_hypotheses")
        return (self._hypothesis,)

    async def assess_verification(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
        hypothesis: Hypothesis,
        query: QueryRecord,
    ) -> VerificationAssessment:
        del request, memory
        self.calls.append("assess_verification")
        if hypothesis.id != self._hypothesis.id:
            raise PortProtocolError(
                "fake_reasoning.unexpected_hypothesis",
                "Fake reasoning received an unexpected hypothesis.",
            ) from None

        if not query.evidence_ids:
            return VerificationAssessment(
                hypothesis=hypothesis,
                decision=VerificationDecision.CONCLUDE,
            )
        supported = replace(
            hypothesis,
            status=HypothesisStatus.SUPPORTED,
            supporting_evidence_ids=tuple(
                dict.fromkeys((*hypothesis.supporting_evidence_ids, *query.evidence_ids))
            ),
        )
        return VerificationAssessment(
            hypothesis=supported,
            decision=VerificationDecision.CONCLUDE,
        )

    async def generate_conclusion(
        self,
        request: InvestigationRequest,
        memory: WorkingMemory,
        outcome: ConclusionOutcome,
        termination_reason: TerminationReason,
        root_cause_hypothesis_id: str | None,
    ) -> Conclusion:
        del request
        self.calls.append("generate_conclusion")
        if outcome is ConclusionOutcome.CONCLUSIVE:
            root = next(
                (item for item in memory.hypotheses if item.id == root_cause_hypothesis_id),
                None,
            )
            if root is None or root.status is not HypothesisStatus.SUPPORTED:
                raise PortProtocolError(
                    "fake_reasoning.missing_root_cause",
                    "Fake reasoning could not find the supported root cause.",
                ) from None
            return Conclusion(
                outcome=outcome,
                summary=self._conclusion_summary,
                termination_reason=termination_reason,
                root_cause_hypothesis_id=root.id,
                evidence_ids=root.supporting_evidence_ids,
                recommendations=self._recommendations,
            )
        return Conclusion(
            outcome=outcome,
            summary="The available evidence was insufficient to identify a root cause.",
            termination_reason=termination_reason,
        )
