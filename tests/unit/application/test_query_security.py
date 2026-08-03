from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from log_agent.application.query_security import (
    AuthorizedQueryPlan,
    LogSource,
    QueryLiteral,
    QueryOperation,
    QueryPlan,
    QueryPlanCompiler,
    QueryPolicyError,
    QueryPolicyGate,
    SafeQueryPipeline,
    ScopePolicy,
    ScopePolicyRegistry,
)
from log_agent.domain.models import QueryIntent, QueryKind, TimeRange

NOW = datetime(2026, 8, 2, tzinfo=UTC)
RANGE = TimeRange(start=NOW - timedelta(hours=1), end=NOW)


def make_policy(
    *,
    ref: str = "checkout-prod",
    sources: tuple[LogSource, ...] = (
        LogSource(index="checkout", sourcetype="checkout:json"),
        LogSource(index="payment", sourcetype="payment:json"),
    ),
    max_time_span: timedelta = timedelta(hours=1),
    max_result_limit: int = 100,
) -> ScopePolicy:
    return ScopePolicy(
        ref=ref,
        sources=sources,
        allowed_template_ids=frozenset({"triage.error_summary.v1", "verify.event_sample.v1"}),
        allowed_operations=frozenset({QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}),
        max_time_span=max_time_span,
        max_result_limit=max_result_limit,
    )


def make_pipeline(
    *,
    policy: ScopePolicy | None = None,
    compiler: QueryPlanCompiler | None = None,
) -> SafeQueryPipeline:
    registry = ScopePolicyRegistry((make_policy() if policy is None else policy,))
    return SafeQueryPipeline(registry, compiler=compiler)


def make_intent(
    kind: QueryKind,
    *,
    goal: str = "find payment timeout events",
    time_range: TimeRange = RANGE,
) -> QueryIntent:
    return QueryIntent(
        kind=kind,
        goal=goal,
        time_range=time_range,
        hypothesis_id="h-1" if kind is QueryKind.VERIFY else None,
    )


def test_pipeline_builds_spl_free_authorized_plans() -> None:
    pipeline = make_pipeline()

    triage = pipeline.prepare(
        scope_ref="checkout-prod",
        investigation_range=RANGE,
        intent=make_intent(QueryKind.TRIAGE),
    )
    verify = pipeline.prepare(
        scope_ref="checkout-prod",
        investigation_range=RANGE,
        intent=make_intent(QueryKind.VERIFY),
    )

    assert triage.scope_ref == "checkout-prod"
    assert triage.sources == (
        LogSource(index="checkout", sourcetype="checkout:json"),
        LogSource(index="payment", sourcetype="payment:json"),
    )
    assert triage.plan.template_id == "triage.error_summary.v1"
    assert triage.plan.operation is QueryOperation.ERROR_SUMMARY
    assert triage.plan.literal_terms == ()
    assert verify.plan.template_id == "verify.event_sample.v1"
    assert verify.plan.operation is QueryOperation.EVENT_SAMPLE
    assert verify.plan.literal_terms == (QueryLiteral("find payment timeout events"),)
    assert not hasattr(verify.plan, "spl")
    assert not hasattr(verify.plan, "raw_spl")
    assert not hasattr(verify.plan, "query_text")


@pytest.mark.parametrize(
    "scope_ref",
    [
        "missing",
        "CHECKOUT-PROD",
        " checkout-prod",
        "checkout-prod ",
        "checkout-prod | index=secret",
        "index=*",
    ],
)
def test_scope_registry_is_exact_and_has_no_fallback(scope_ref: str) -> None:
    pipeline = make_pipeline()

    with pytest.raises(QueryPolicyError) as caught:
        pipeline.prepare(
            scope_ref=scope_ref,
            investigation_range=RANGE,
            intent=make_intent(QueryKind.TRIAGE),
        )

    assert caught.value.code == "query_policy.unknown_scope"
    assert caught.value.safe_message == "The requested log scope is not configured."
    assert scope_ref not in caught.value.safe_message


@pytest.mark.parametrize(
    "index",
    [
        "*",
        "main*",
        "main | delete",
        "main OR index=secret",
        "main\n| collect",
        'main" OR "1"="1',
        "[search secret]",
    ],
)
def test_scope_policy_rejects_untrusted_index_tokens_at_startup(index: str) -> None:
    with pytest.raises(ValueError, match="index"):
        LogSource(index=index, sourcetype="checkout:json")


@pytest.mark.parametrize(
    "sourcetype",
    ["*", "app*", "app | collect", "app\n| delete", 'app" OR "1"="1'],
)
def test_scope_policy_rejects_untrusted_sourcetype_tokens_at_startup(
    sourcetype: str,
) -> None:
    with pytest.raises(ValueError, match="sourcetype"):
        LogSource(index="checkout", sourcetype=sourcetype)


@pytest.mark.parametrize(
    "goal",
    [
        "index=* | delete",
        "errors\n| collect",
        "errors | OUTPUTLOOKUP leaked.csv",
        "[search index=secret]",
        "`dangerous_macro`",
        'error " OR index=secret',
        "error\\ escaped",
        "OR index=secret",
        "* OR index=secret",
        "search index=secret",
        "foo) OR index=secret",
        "error AND sourcetype=secret",
        "field=value",
    ],
)
def test_structural_spl_text_is_rejected_as_a_literal(goal: str) -> None:
    pipeline = make_pipeline()

    with pytest.raises(QueryPolicyError) as caught:
        pipeline.prepare(
            scope_ref="checkout-prod",
            investigation_range=RANGE,
            intent=make_intent(QueryKind.VERIFY, goal=goal),
        )

    assert caught.value.code == "query_policy.denied"
    assert caught.value.safe_message == "The requested log query is not allowed."
    assert goal not in caught.value.safe_message


def test_dangerous_word_inside_plain_log_text_is_not_a_naive_blacklist_match() -> None:
    authorized = make_pipeline().prepare(
        scope_ref="checkout-prod",
        investigation_range=RANGE,
        intent=make_intent(QueryKind.VERIFY, goal="delete failed for temporary record"),
    )

    assert authorized.plan.literal_terms == (QueryLiteral("delete failed for temporary record"),)


def test_scope_max_time_span_is_enforced_without_silent_clamping() -> None:
    policy = make_policy(max_time_span=timedelta(minutes=30))
    pipeline = make_pipeline(policy=policy)

    with pytest.raises(QueryPolicyError, match="not allowed"):
        pipeline.prepare(
            scope_ref=policy.ref,
            investigation_range=RANGE,
            intent=make_intent(QueryKind.TRIAGE),
        )


def test_verify_query_at_exact_time_boundary_is_allowed() -> None:
    verify_range = TimeRange(start=NOW - timedelta(minutes=30), end=NOW)
    policy = make_policy(max_time_span=timedelta(minutes=30))

    authorized = make_pipeline(policy=policy).prepare(
        scope_ref=policy.ref,
        investigation_range=RANGE,
        intent=make_intent(QueryKind.VERIFY, time_range=verify_range),
    )

    assert authorized.plan.time_range == verify_range


def test_query_outside_investigation_range_is_rejected() -> None:
    expanded = TimeRange(start=RANGE.start - timedelta(microseconds=1), end=RANGE.end)

    with pytest.raises(QueryPolicyError, match="not allowed"):
        make_pipeline().prepare(
            scope_ref="checkout-prod",
            investigation_range=RANGE,
            intent=make_intent(QueryKind.VERIFY, time_range=expanded),
        )


def test_query_ending_after_investigation_range_is_rejected() -> None:
    expanded = TimeRange(start=RANGE.start, end=RANGE.end + timedelta(microseconds=1))

    with pytest.raises(QueryPolicyError, match="not allowed"):
        make_pipeline().prepare(
            scope_ref="checkout-prod",
            investigation_range=RANGE,
            intent=make_intent(QueryKind.VERIFY, time_range=expanded),
        )


def test_result_limit_above_scope_maximum_is_rejected() -> None:
    compiler = QueryPlanCompiler(verify_result_limit=101)

    with pytest.raises(QueryPolicyError, match="not allowed"):
        make_pipeline(compiler=compiler).prepare(
            scope_ref="checkout-prod",
            investigation_range=RANGE,
            intent=make_intent(QueryKind.VERIFY),
        )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_result_limit_types_and_values_are_rejected(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        QueryPlanCompiler(verify_result_limit=limit)  # type: ignore[arg-type]


def test_gate_rejects_disallowed_template_and_operation() -> None:
    policy = make_policy()
    intent = make_intent(QueryKind.VERIFY)
    plan = QueryPlanCompiler().compile(intent)
    gate = QueryPolicyGate()

    with pytest.raises(QueryPolicyError, match="not allowed"):
        gate.authorize(
            policy=replace(
                policy,
                allowed_template_ids=frozenset({"triage.error_summary.v1"}),
            ),
            investigation_range=RANGE,
            intent=intent,
            plan=plan,
        )
    with pytest.raises(QueryPolicyError, match="not allowed"):
        gate.authorize(
            policy=replace(
                policy,
                allowed_operations=frozenset({QueryOperation.ERROR_SUMMARY}),
            ),
            investigation_range=RANGE,
            intent=intent,
            plan=plan,
        )


def test_authorized_plan_cannot_be_constructed_outside_policy_gate() -> None:
    intent = make_intent(QueryKind.VERIFY)
    mismatched = QueryPlan(
        template_id="triage.error_summary.v1",
        kind=QueryKind.VERIFY,
        operation=QueryOperation.EVENT_SAMPLE,
        time_range=intent.time_range,
        result_limit=50,
        literal_terms=(QueryLiteral(intent.goal),),
    )

    with pytest.raises(TypeError, match="created by QueryPolicyGate"):
        AuthorizedQueryPlan(
            scope_ref="checkout-prod",
            sources=(LogSource(index="checkout", sourcetype="checkout:json"),),
            plan=mismatched,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"scope_ref": "secret-prod"},
        {"sources": (LogSource(index="secret", sourcetype="secret:json"),)},
    ],
)
def test_authorized_plan_cannot_be_replaced_with_unchecked_scope_data(
    changes: dict[str, object],
) -> None:
    authorized = make_pipeline().prepare(
        scope_ref="checkout-prod",
        investigation_range=RANGE,
        intent=make_intent(QueryKind.VERIFY),
    )

    with pytest.raises(TypeError, match="created by QueryPolicyGate"):
        replace(authorized, **changes)


def test_authorized_plan_cannot_be_replaced_with_unchecked_limit() -> None:
    authorized = make_pipeline().prepare(
        scope_ref="checkout-prod",
        investigation_range=RANGE,
        intent=make_intent(QueryKind.VERIFY),
    )
    unchecked_plan = replace(authorized.plan, result_limit=1_000_000)

    with pytest.raises(TypeError, match="created by QueryPolicyGate"):
        replace(authorized, plan=unchecked_plan)


def test_scope_policy_copies_mutable_allowlists_into_frozen_sets() -> None:
    templates = {"triage.error_summary.v1", "verify.event_sample.v1"}
    operations = {QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}
    policy = ScopePolicy(
        ref="checkout-prod",
        sources=(LogSource(index="checkout", sourcetype="checkout:json"),),
        allowed_template_ids=templates,  # type: ignore[arg-type]
        allowed_operations=operations,  # type: ignore[arg-type]
        max_time_span=timedelta(hours=1),
        max_result_limit=100,
    )

    templates.clear()
    operations.clear()

    assert policy.allowed_template_ids == frozenset(
        {"triage.error_summary.v1", "verify.event_sample.v1"}
    )
    assert policy.allowed_operations == frozenset(
        {QueryOperation.ERROR_SUMMARY, QueryOperation.EVENT_SAMPLE}
    )


def test_query_vocabulary_contains_no_side_effect_operations() -> None:
    operations = {item.value for item in QueryOperation}

    assert operations == {"error_summary", "event_sample"}
    assert operations.isdisjoint({"delete", "collect", "outputlookup", "sendemail", "script"})


def test_compilation_is_deterministic_and_does_not_mutate_policy() -> None:
    policy = make_policy(
        sources=(
            LogSource(index="payment", sourcetype="payment:json"),
            LogSource(index="checkout", sourcetype="checkout:json"),
        )
    )
    pipeline = make_pipeline(policy=policy)
    intent = make_intent(QueryKind.VERIFY)

    first = pipeline.prepare(
        scope_ref=policy.ref,
        investigation_range=RANGE,
        intent=intent,
    )
    second = pipeline.prepare(
        scope_ref=policy.ref,
        investigation_range=RANGE,
        intent=intent,
    )

    assert first == second
    expected = (
        LogSource(index="checkout", sourcetype="checkout:json"),
        LogSource(index="payment", sourcetype="payment:json"),
    )
    assert first.sources == expected
    assert policy.sources == expected
