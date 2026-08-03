import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from log_agent.application.query_security import (
    LogSource,
    QueryOperation,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.domain.models import Phase
from log_agent.evaluation.fakes import (
    FakeEvalResponse,
    FakeEvalRow,
    FakeEvalRuntimeFactory,
    FakeEvalScenario,
)
from log_agent.evaluation.harness import DeterministicEvalHarness
from log_agent.evaluation.incident_json import load_incident_dataset_json
from log_agent.evaluation.models import EvalFailureCategory

DATASET_PATH = Path(__file__).parents[2] / "config" / "evaluation" / "checkout-incidents.json"
NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)
REPORT_HMAC_KEY = b"0123456789abcdef" * 2


def scope_policies() -> ScopePolicyRegistry:
    return ScopePolicyRegistry(
        (
            ScopePolicy(
                ref="checkout-prod",
                sources=(
                    LogSource(index="checkout", sourcetype="checkout:json"),
                    LogSource(index="payment", sourcetype="payment:json"),
                ),
                allowed_template_ids=frozenset(
                    {"triage.error_summary.v1", "verify.event_sample.v1"}
                ),
                allowed_operations=frozenset(
                    {QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}
                ),
                max_time_span=timedelta(hours=1),
                max_result_limit=100,
            ),
        )
    )


def scenario(
    *,
    fixture_id: str,
    triage_rows: tuple[FakeEvalRow, ...],
    verify_rows: tuple[FakeEvalRow, ...] | None,
    max_total_queries: int = 4,
    max_verify_queries: int = 2,
) -> FakeEvalScenario:
    responses = [
        FakeEvalResponse(
            template_id="triage.error_summary.v1",
            summary="Deterministic triage summary.",
            rows=triage_rows,
        )
    ]
    if verify_rows is not None:
        responses.append(
            FakeEvalResponse(
                template_id="verify.event_sample.v1",
                summary="Deterministic verification summary.",
                rows=verify_rows,
            )
        )
    return FakeEvalScenario(
        fixture_id=fixture_id,
        revision="2026-08-03.1",
        triage_goals=("summarize checkout errors",),
        max_total_queries=max_total_queries,
        max_verify_queries=max_verify_queries,
        responses=tuple(responses),
        root_cause_key="payment_dependency_timeout",
        hypothesis_statement="Payment dependency timeouts caused checkout failures.",
        verification_goal="find payment timeout events",
        conclusion_summary="Payment dependency timeouts caused checkout failures.",
        recommendations=("Review payment dependency latency.",),
    )


def scenarios() -> tuple[FakeEvalScenario, ...]:
    triage_row = FakeEvalRow(
        evidence_label="checkout-dependency-error",
        fact_statement="Checkout returned a dependency error.",
        occurred_at=NOW - timedelta(minutes=5),
    )
    return (
        scenario(
            fixture_id="payment-timeout-v1",
            triage_rows=(triage_row,),
            verify_rows=(
                FakeEvalRow(
                    evidence_label="payment-timeout-before-checkout-failure",
                    fact_statement="A payment timeout preceded the checkout failure.",
                    occurred_at=NOW - timedelta(minutes=6),
                ),
            ),
        ),
        scenario(
            fixture_id="no-data-v1",
            triage_rows=(),
            verify_rows=None,
        ),
        scenario(
            fixture_id="insufficient-v1",
            triage_rows=(triage_row,),
            verify_rows=(),
        ),
        scenario(
            fixture_id="budget-exhausted-v1",
            triage_rows=(triage_row,),
            verify_rows=None,
            max_total_queries=1,
            max_verify_queries=0,
        ),
        scenario(
            fixture_id="unknown-scope-v1",
            triage_rows=(),
            verify_rows=None,
            max_total_queries=2,
            max_verify_queries=1,
        ),
    )


def test_versioned_incident_dataset_runs_five_deterministic_fake_scenarios() -> None:
    dataset = load_incident_dataset_json(DATASET_PATH)
    harness = DeterministicEvalHarness(
        FakeEvalRuntimeFactory(scope_policies(), scenarios()),
        report_hmac_key=REPORT_HMAC_KEY,
    )

    report = asyncio.run(harness.run(dataset))

    assert report.passed is True
    assert report.passed_case_count == 5
    assert len(report.cases) == 5
    assert len(report.dataset_fingerprint) == 64
    assert len(report.query_policy_fingerprint) == 64

    by_id = {
        incident.id: case_report
        for incident, case_report in zip(dataset.cases, report.cases, strict=True)
    }
    completed = by_id["payment-timeout-completed"]
    assert completed.actual_phase is Phase.COMPLETED
    assert completed.root_cause_matched is True
    assert completed.matched_required_evidence_count == 1
    assert completed.issued_query_count == 2
    assert completed.log_port_call_count == 2

    no_data = by_id["empty-window-inconclusive"]
    assert no_data.issued_query_count == 1
    assert no_data.reasoning_call_count == 1

    budget = by_id["query-budget-inconclusive"]
    assert budget.issued_query_count == 1
    assert budget.issued_verify_query_count == 0
    assert budget.log_port_call_count == 1

    failed = by_id["unknown-scope-failed"]
    assert failed.failure_category is EvalFailureCategory.QUERY_POLICY
    assert failed.failure_code_matched is True
    assert failed.issued_query_count == 1
    assert failed.log_port_call_count == 0
    assert failed.process_passed is True
