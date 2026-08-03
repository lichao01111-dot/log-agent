from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

import log_agent.application as application
from log_agent.application.ports import (
    PortError,
    SearchRequest,
    SearchResult,
    VerificationAssessment,
)
from log_agent.application.query_security import (
    AuthorizedQueryPlan,
    LogSource,
    QueryOperation,
    SafeQueryPipeline,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.domain.models import (
    EvidenceRef,
    Fact,
    Hypothesis,
    QueryIntent,
    QueryKind,
    TimeRange,
    VerificationDecision,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def make_intent(kind: QueryKind = QueryKind.TRIAGE) -> QueryIntent:
    hypothesis_id = "h-1" if kind is QueryKind.VERIFY else None
    return QueryIntent(
        kind=kind,
        goal="find payment timeout events",
        time_range=TimeRange(
            start=NOW - timedelta(hours=1),
            end=NOW,
        ),
        hypothesis_id=hypothesis_id,
    )


def make_authorized_query(intent: QueryIntent) -> AuthorizedQueryPlan:
    policy = ScopePolicy(
        ref="checkout-prod",
        sources=(LogSource(index="checkout", sourcetype="checkout:json"),),
        allowed_template_ids=frozenset({"triage.error_summary.v1", "verify.event_sample.v1"}),
        allowed_operations=frozenset({QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}),
        max_time_span=timedelta(hours=1),
        max_result_limit=100,
    )
    pipeline = SafeQueryPipeline(ScopePolicyRegistry((policy,)))
    return pipeline.prepare(
        scope_ref=policy.ref,
        investigation_range=intent.time_range,
        intent=intent,
    )


@pytest.mark.parametrize("field", ["operation_id", "query_id"])
def test_search_request_rejects_blank_identifiers(field: str) -> None:
    intent = make_intent()
    values = {
        "operation_id": "cmd-1",
        "query_id": "query-1",
        "authorized_query": make_authorized_query(intent),
    }
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        SearchRequest(**values)


def test_search_request_exposes_no_raw_scope_intent_or_spl_field() -> None:
    assert tuple(field.name for field in fields(SearchRequest)) == (
        "operation_id",
        "query_id",
        "authorized_query",
    )


def test_search_result_accepts_grounded_facts() -> None:
    evidence = EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:1")
    fact = Fact(id="fact-1", statement="Payment timed out", evidence_ids=("ev-1",))

    result = SearchResult(
        result_count=2,
        summary="Two timeout events found",
        evidence=(evidence,),
        facts=(fact,),
    )

    assert result.evidence == (evidence,)
    assert result.facts == (fact,)


def test_search_result_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        SearchResult(result_count=-1, summary="invalid")


@pytest.mark.parametrize("count", [True, 1.5, "1"])
def test_search_result_rejects_non_integer_count(count: object) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        SearchResult(result_count=count, summary="invalid")  # type: ignore[arg-type]


def test_search_result_rejects_more_evidence_than_rows() -> None:
    evidence = EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:1")

    with pytest.raises(ValueError, match="more evidence items"):
        SearchResult(result_count=0, summary="no rows", evidence=(evidence,))


def test_search_result_rejects_duplicate_evidence_ids() -> None:
    evidence = (
        EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:1"),
        EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:2"),
    )

    with pytest.raises(ValueError, match="evidence ids must not contain duplicates"):
        SearchResult(result_count=2, summary="duplicate evidence", evidence=evidence)


def test_search_result_rejects_duplicate_fact_ids() -> None:
    evidence = EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:1")
    facts = (
        Fact(id="fact-1", statement="First claim", evidence_ids=("ev-1",)),
        Fact(id="fact-1", statement="Second claim", evidence_ids=("ev-1",)),
    )

    with pytest.raises(ValueError, match="fact ids must not contain duplicates"):
        SearchResult(
            result_count=1,
            summary="duplicate facts",
            evidence=(evidence,),
            facts=facts,
        )


def test_search_result_rejects_fact_with_foreign_evidence() -> None:
    fact = Fact(id="fact-1", statement="Unproven claim", evidence_ids=("ev-other",))

    with pytest.raises(ValueError, match="evidence from this result"):
        SearchResult(result_count=1, summary="invalid fact", facts=(fact,))


def test_search_result_rejects_more_facts_than_rows() -> None:
    evidence = EvidenceRef(id="ev-1", query_id="q-1", record_ref="row:1")
    facts = (
        Fact(id="fact-1", statement="First", evidence_ids=("ev-1",)),
        Fact(id="fact-2", statement="Second", evidence_ids=("ev-1",)),
    )

    with pytest.raises(ValueError, match="more facts than result rows"):
        SearchResult(
            result_count=1,
            summary="too many facts",
            evidence=(evidence,),
            facts=facts,
        )


def test_search_result_rejects_oversized_text() -> None:
    with pytest.raises(ValueError, match="summary exceeds"):
        SearchResult(result_count=0, summary="x" * 2_001)


def test_continue_assessment_requires_next_query() -> None:
    hypothesis = Hypothesis(
        id="h-1",
        statement="Payment timed out",
        verification_goal="find payment timeout events",
    )

    with pytest.raises(ValueError, match="CONTINUE requires"):
        VerificationAssessment(
            hypothesis=hypothesis,
            decision=VerificationDecision.CONTINUE,
        )


def test_terminal_assessment_rejects_next_query() -> None:
    hypothesis = Hypothesis(
        id="h-1",
        statement="Payment timed out",
        verification_goal="find payment timeout events",
    )

    with pytest.raises(ValueError, match="only valid for CONTINUE"):
        VerificationAssessment(
            hypothesis=hypothesis,
            decision=VerificationDecision.CONCLUDE,
            next_query=make_intent(QueryKind.VERIFY),
        )


def test_port_error_uses_adapter_sanitized_message_as_exception_text() -> None:
    error = PortError(code="search_unavailable", safe_message="Log search is unavailable")

    assert error.code == "search_unavailable"
    assert error.safe_message == "Log search is unavailable"
    assert str(error) == "Log search is unavailable"


def test_application_package_exports_the_port_api() -> None:
    assert application.CommandExecutor.__name__ == "CommandExecutor"
    assert application.InvestigationRunner.__name__ == "InvestigationRunner"
    assert application.SafeQueryPipeline.__name__ == "SafeQueryPipeline"
    assert application.ScopePolicy.__name__ == "ScopePolicy"
    assert application.QueryPlan.__name__ == "QueryPlan"
    assert application.SearchRequest is SearchRequest
    assert application.SearchResult is SearchResult
    assert application.VerificationAssessment is VerificationAssessment
