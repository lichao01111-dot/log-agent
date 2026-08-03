import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from log_agent.adapters.fakes import (
    FakeLogRow,
    FakeLogSearchPort,
    FakeSearchResponse,
    FakeStructuredModelClient,
)
from log_agent.adapters.knowledge_json import load_knowledge_json
from log_agent.adapters.structured_reasoning import StructuredReasoningAdapter
from log_agent.application.configuration import AgentConfiguration
from log_agent.application.executor import CommandExecutor
from log_agent.application.knowledge_projection import (
    ProjectionBudget,
    ProjectionVisibilityPolicy,
    ReasoningStage,
)
from log_agent.application.model_ports import (
    ModelFinishReason,
    StructuredModelResponse,
)
from log_agent.application.query_security import (
    LogSource,
    QueryOperation,
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


class TestOnlySanitizer:
    __test__ = False

    def sanitize(self, text, *, category):
        del category
        return text


def make_configuration():
    snapshot = load_knowledge_json(KNOWLEDGE_PATH)
    scope_policy = ScopePolicy(
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
        knowledge=snapshot,
        scope_policies=ScopePolicyRegistry((scope_policy,)),
    )
    visibility = ProjectionVisibilityPolicy(
        id="checkout-model-visibility",
        revision="2026-08-02.1",
        scope_ref="checkout-prod",
        knowledge_bundle_id=snapshot.bundle_id,
        knowledge_content_hash=snapshot.content_hash,
        budget=ProjectionBudget(
            max_components=10,
            max_fields=10,
            max_error_codes=10,
            max_dependencies=10,
            max_known_failures=10,
            max_total_entities=30,
            max_serialized_characters=10_000,
        ),
        allowed_component_ids=frozenset(item.id for item in snapshot.components),
        allowed_field_ids=frozenset(item.id for item in snapshot.fields),
        allowed_error_code_ids=frozenset(item.id for item in snapshot.error_codes),
        allowed_dependency_ids=frozenset(item.id for item in snapshot.dependencies),
        allowed_known_failure_ids=frozenset(item.id for item in snapshot.known_failures),
    )
    return configuration, visibility


def test_full_runner_uses_structured_drafts_without_giving_model_domain_control() -> None:
    time_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)
    initial = Investigation(
        id="run-structured",
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
                summary="Checkout returned payment timeout errors",
                rows=(
                    FakeLogRow(
                        record_ref="private://checkout/triage-row",
                        fact_statement="Checkout emitted PAYMENT_TIMEOUT.",
                    ),
                ),
            ),
            "verify.event_sample.v1": FakeSearchResponse(
                summary="Payment timed out before checkout failed",
                rows=(
                    FakeLogRow(
                        record_ref="private://payment/verification-row",
                        fact_statement="A payment timeout preceded the checkout failure.",
                    ),
                ),
            ),
        }
    )
    model = FakeStructuredModelClient(
        (
            StructuredModelResponse(
                '{"schema_version":1,"hypotheses":[{"statement":'
                '"Payment dependency timeouts caused checkout failures.",'
                '"verification_goal":"find payment timeout events"}]}',
                ModelFinishReason.STOP,
            ),
            StructuredModelResponse(
                '{"schema_version":1,"verdict":"supported",'
                '"new_supporting_evidence_ids":["e1"],'
                '"new_contradicting_evidence_ids":[],"next_verification_goal":null}',
                ModelFinishReason.STOP,
            ),
            StructuredModelResponse(
                '{"schema_version":1,"summary":"Payment timeouts caused checkout failures.",'
                '"evidence_ids":["e1"],"recommendations":'
                '["Review payment dependency latency."]}',
                ModelFinishReason.STOP,
            ),
        )
    )
    configuration, visibility = make_configuration()
    reasoning = StructuredReasoningAdapter(
        model,
        configuration.build_knowledge_projector((visibility,)),
        TestOnlySanitizer(),
    )
    runner = InvestigationRunner(
        CommandExecutor(search, reasoning, configuration.build_query_pipeline())
    )

    result = asyncio.run(runner.run(initial))

    assert result.phase is Phase.COMPLETED
    assert result.termination_reason is TerminationReason.ROOT_CAUSE_IDENTIFIED
    assert result.conclusion is not None
    assert result.conclusion.outcome is ConclusionOutcome.CONCLUSIVE
    assert result.conclusion.root_cause_hypothesis_id is not None
    assert result.conclusion.root_cause_hypothesis_id.startswith("h-")
    assert result.conclusion.evidence_ids == ("run-structured:3:query:evidence:1",)
    assert [request.task for request in model.requests] == [
        ReasoningStage.GENERATE_HYPOTHESES,
        ReasoningStage.ASSESS_VERIFICATION,
        ReasoningStage.GENERATE_CONCLUSION,
    ]
    model_inputs = "".join(request.input_json for request in model.requests)
    assert "private://" not in model_inputs
    assert "run-structured:3:query:evidence:1" not in model_inputs
    assert all(request.repair_attempt == 0 for request in model.requests)
