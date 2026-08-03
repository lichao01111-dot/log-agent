from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import log_agent.application as application
from log_agent.adapters.knowledge_json import load_knowledge_json
from log_agent.application.configuration import (
    AgentConfiguration,
    ConfigurationCompatibilityError,
)
from log_agent.application.knowledge_projection import (
    KnowledgeProjector,
    ProjectionBudget,
    ProjectionVisibilityPolicy,
)
from log_agent.application.query_security import (
    LogSource,
    QueryOperation,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.domain.models import QueryIntent, QueryKind, TimeRange

EXAMPLE_PATH = Path(__file__).parents[3] / "config" / "knowledge" / "checkout-domain.json"
NOW = datetime(2026, 8, 2, tzinfo=UTC)


def make_registry(*, ref: str = "checkout-prod") -> ScopePolicyRegistry:
    return ScopePolicyRegistry(
        (
            ScopePolicy(
                ref=ref,
                sources=(LogSource(index="checkout", sourcetype="checkout:json"),),
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


def make_projection_policy(snapshot) -> ProjectionVisibilityPolicy:
    return ProjectionVisibilityPolicy(
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
    )


def test_configuration_binds_exact_knowledge_and_policy_scopes() -> None:
    configuration = AgentConfiguration(
        knowledge=load_knowledge_json(EXAMPLE_PATH),
        scope_policies=make_registry(),
    )

    assert configuration.knowledge.scope_refs == ("checkout-prod",)
    assert configuration.scope_policies.refs == ("checkout-prod",)


def test_scope_mismatch_fails_closed_at_startup() -> None:
    with pytest.raises(ConfigurationCompatibilityError, match="exactly match"):
        AgentConfiguration(
            knowledge=load_knowledge_json(EXAMPLE_PATH),
            scope_policies=make_registry(ref="other-prod"),
        )


def test_knowledge_does_not_supply_query_permissions() -> None:
    configuration = AgentConfiguration(
        knowledge=load_knowledge_json(EXAMPLE_PATH),
        scope_policies=make_registry(),
    )
    time_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)

    authorized = configuration.build_query_pipeline().prepare(
        scope_ref="checkout-prod",
        investigation_range=time_range,
        intent=QueryIntent(
            kind=QueryKind.TRIAGE,
            goal="summarize checkout errors",
            time_range=time_range,
        ),
    )

    assert authorized.sources == (LogSource(index="checkout", sourcetype="checkout:json"),)
    assert authorized.plan.result_limit == 100
    assert not hasattr(configuration.knowledge, "sources")
    assert not hasattr(configuration.knowledge, "allowed_operations")


def test_configuration_builds_projector_for_the_complete_scope_set() -> None:
    snapshot = load_knowledge_json(EXAMPLE_PATH)
    configuration = AgentConfiguration(
        knowledge=snapshot,
        scope_policies=make_registry(),
    )

    projector = configuration.build_knowledge_projector((make_projection_policy(snapshot),))

    assert isinstance(projector, KnowledgeProjector)


def test_missing_projection_scope_fails_closed_at_startup() -> None:
    configuration = AgentConfiguration(
        knowledge=load_knowledge_json(EXAMPLE_PATH),
        scope_policies=make_registry(),
    )

    with pytest.raises(ConfigurationCompatibilityError, match="exactly match"):
        configuration.build_knowledge_projector(())


def test_application_package_exports_configuration_api() -> None:
    assert application.AgentConfiguration is AgentConfiguration
    assert application.ConfigurationCompatibilityError is ConfigurationCompatibilityError
