import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from log_agent.adapters.fakes import (
    DeterministicReasoningPort,
    FakeLogRow,
    FakeLogSearchPort,
    FakeSearchResponse,
)
from log_agent.adapters.knowledge_json import load_knowledge_json
from log_agent.application.configuration import AgentConfiguration
from log_agent.application.executor import CommandExecutor
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

NOW = datetime(2026, 8, 2, tzinfo=UTC)
KNOWLEDGE_PATH = Path(__file__).parents[2] / "config" / "knowledge" / "checkout-domain.json"


def make_query_pipeline() -> SafeQueryPipeline:
    policy = ScopePolicy(
        ref="checkout-prod",
        sources=(
            LogSource(index="checkout", sourcetype="checkout:json"),
            LogSource(index="payment", sourcetype="payment:json"),
        ),
        allowed_template_ids=frozenset({"triage.error_summary.v1", "verify.event_sample.v1"}),
        allowed_operations=frozenset({QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}),
        max_time_span=timedelta(hours=1),
        max_result_limit=100,
    )
    configuration = AgentConfiguration(
        knowledge=load_knowledge_json(KNOWLEDGE_PATH),
        scope_policies=ScopePolicyRegistry((policy,)),
    )
    return configuration.build_query_pipeline()


def test_runner_happy_path_completes_with_structural_evidence_links() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)
    initial = Investigation(
        id="run-e2e",
        request=InvestigationRequest(
            question="Why did checkout requests fail?",
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
    search = FakeLogSearchPort(
        {
            "triage.error_summary.v1": FakeSearchResponse(
                summary="Checkout returned dependency errors",
                rows=(
                    FakeLogRow(
                        record_ref="fake://checkout/1",
                        fact_statement="Checkout requests returned dependency errors.",
                        occurred_at=NOW - timedelta(minutes=5),
                    ),
                ),
            ),
            "verify.event_sample.v1": FakeSearchResponse(
                summary="Payment timeout matched the checkout failures",
                rows=(
                    FakeLogRow(
                        record_ref="fake://payment/1",
                        fact_statement="Payment calls timed out before checkout failed.",
                        occurred_at=NOW - timedelta(minutes=6),
                    ),
                ),
            ),
        }
    )
    reasoning = DeterministicReasoningPort(
        hypothesis_id="h-payment-timeout",
        hypothesis_statement="Payment dependency timeouts caused checkout failures.",
        verification_goal="find payment timeout events",
        conclusion_summary="Payment dependency timeouts caused the checkout failures.",
        recommendations=("Inspect payment dependency latency and timeout settings.",),
    )
    runner = InvestigationRunner(CommandExecutor(search, reasoning, make_query_pipeline()))

    result = asyncio.run(runner.run(initial))

    assert result.phase is Phase.COMPLETED
    assert result.termination_reason is TerminationReason.ROOT_CAUSE_IDENTIFIED
    assert result.pending_operation is None
    assert result.conclusion is not None
    assert result.conclusion.outcome is ConclusionOutcome.CONCLUSIVE
    assert result.conclusion.root_cause_hypothesis_id == "h-payment-timeout"
    assert result.conclusion.evidence_ids == ("run-e2e:3:query:evidence:1",)
    assert result.budget.issued_total == 2
    assert result.budget.issued_verify == 1
    assert tuple(query.intent.kind for query in result.memory.queries) == (
        QueryKind.TRIAGE,
        QueryKind.VERIFY,
    )
    assert tuple(request.query_id for request in search.requests) == (
        "run-e2e:1:query",
        "run-e2e:3:query",
    )
    assert all(
        request.authorized_query.scope_ref == initial.request.scope_ref
        for request in search.requests
    )
    assert all(
        request.authorized_query.plan.time_range == time_range for request in search.requests
    )
    assert tuple(request.authorized_query.plan.template_id for request in search.requests) == (
        "triage.error_summary.v1",
        "verify.event_sample.v1",
    )
    expected_sources = (
        LogSource(index="checkout", sourcetype="checkout:json"),
        LogSource(index="payment", sourcetype="payment:json"),
    )
    assert all(request.authorized_query.sources == expected_sources for request in search.requests)
    assert all(request.authorized_query.plan.result_limit <= 100 for request in search.requests)
    assert all(
        evidence.query_id == query.id
        for query in result.memory.queries
        for evidence in result.memory.evidence
        if evidence.id in query.evidence_ids
    )
    assert reasoning.calls == [
        "generate_hypotheses",
        "assess_verification",
        "generate_conclusion",
    ]


def test_runner_without_verification_evidence_is_inconclusive() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)
    initial = Investigation(
        id="run-no-proof",
        request=InvestigationRequest(
            question="Why did checkout requests fail?",
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
    search = FakeLogSearchPort(
        {
            "triage.error_summary.v1": FakeSearchResponse(
                summary="Checkout errors found",
                rows=(
                    FakeLogRow(
                        record_ref="fake://checkout/1",
                        fact_statement="Checkout requests returned errors.",
                    ),
                ),
            ),
            "verify.event_sample.v1": FakeSearchResponse(
                summary="No payment timeout evidence",
            ),
        }
    )
    reasoning = DeterministicReasoningPort(
        hypothesis_id="h-payment-timeout",
        hypothesis_statement="Payment dependency timeouts caused checkout failures.",
        verification_goal="find payment timeout events",
        conclusion_summary="Payment dependency timeouts caused the checkout failures.",
    )
    runner = InvestigationRunner(CommandExecutor(search, reasoning, make_query_pipeline()))

    result = asyncio.run(runner.run(initial))

    assert result.phase is Phase.INCONCLUSIVE
    assert result.termination_reason is TerminationReason.INSUFFICIENT_EVIDENCE
    assert result.conclusion is not None
    assert result.conclusion.outcome is ConclusionOutcome.INCONCLUSIVE
    assert result.conclusion.root_cause_hypothesis_id is None
    assert result.conclusion.evidence_ids == ()


def test_unknown_scope_fails_before_log_port_is_called() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)
    initial = Investigation(
        id="run-unknown-scope",
        request=InvestigationRequest(
            question="Why did checkout requests fail?",
            scope_ref="unknown-prod",
            time_range=time_range,
        ),
        triage_plan=(
            QueryIntent(
                kind=QueryKind.TRIAGE,
                goal="summarize checkout errors",
                time_range=time_range,
            ),
        ),
        budget=QueryBudget(max_total_queries=2, max_verify_queries=1),
    )
    search = FakeLogSearchPort(
        {"triage.error_summary.v1": FakeSearchResponse(summary="Must not be called")}
    )
    reasoning = DeterministicReasoningPort(
        hypothesis_id="h-unused",
        hypothesis_statement="This hypothesis must not be generated.",
        verification_goal="unused verification",
        conclusion_summary="Unused conclusion.",
    )
    runner = InvestigationRunner(CommandExecutor(search, reasoning, make_query_pipeline()))

    result = asyncio.run(runner.run(initial))

    assert result.phase is Phase.FAILED
    assert result.termination_reason is TerminationReason.OPERATION_FAILED
    assert result.failure is not None
    assert result.failure.code == "query_policy.unknown_scope"
    assert result.failure.message == "The requested log scope is not configured."
    assert search.requests == []
    assert reasoning.calls == []
