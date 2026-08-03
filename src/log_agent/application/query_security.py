from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from log_agent.domain.models import QueryIntent, QueryKind, TimeRange

_SCOPE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_INDEX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SOURCETYPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_TEMPLATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_LITERAL_PATTERN = re.compile(r"^[\w .,:/@-]+$")
_BOOLEAN_OPERATOR_PATTERN = re.compile(
    r"(?:^|\s)(?:AND|OR|NOT)(?:\s|$)",
    re.IGNORECASE,
)
_MAX_LITERAL_LENGTH = 200


class QueryOperation(StrEnum):
    ERROR_SUMMARY = "error_summary"
    EVENT_SAMPLE = "event_sample"


class QueryPolicyError(RuntimeError):
    """A safe, expected rejection by the application query policy."""

    def __init__(self, code: str, safe_message: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("query policy error code must not be blank")
        if not isinstance(safe_message, str) or not safe_message.strip():
            raise ValueError("query policy safe_message must not be blank")
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


@dataclass(frozen=True, slots=True)
class LogSource:
    """One trusted Splunk index/sourcetype pairing."""

    index: str
    sourcetype: str

    def __post_init__(self) -> None:
        if not isinstance(self.index, str) or not _INDEX_PATTERN.fullmatch(self.index):
            raise ValueError("log source index is invalid")
        if not isinstance(self.sourcetype, str) or not _SOURCETYPE_PATTERN.fullmatch(
            self.sourcetype
        ):
            raise ValueError("log source sourcetype is invalid")


@dataclass(frozen=True, slots=True)
class QueryLiteral:
    """Untrusted model text restricted to a narrow literal-search vocabulary."""

    value: str

    def __post_init__(self) -> None:
        if not _is_safe_literal_text(self.value):
            raise ValueError("query literal is invalid")


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Provider-neutral query draft; it contains no SPL or external tool arguments."""

    template_id: str
    kind: QueryKind
    operation: QueryOperation
    time_range: TimeRange
    result_limit: int
    literal_terms: tuple[QueryLiteral, ...] = ()

    def __post_init__(self) -> None:
        if not _TEMPLATE_PATTERN.fullmatch(self.template_id):
            raise ValueError("query template_id is invalid")
        if not isinstance(self.kind, QueryKind):
            raise ValueError("query plan kind is invalid")
        if not isinstance(self.operation, QueryOperation):
            raise ValueError("query plan operation is invalid")
        if type(self.result_limit) is not int or self.result_limit < 1:
            raise ValueError("query result_limit must be a positive integer")
        if not isinstance(self.literal_terms, tuple) or any(
            not isinstance(term, QueryLiteral) for term in self.literal_terms
        ):
            raise ValueError("query literal terms must contain QueryLiteral values")


@dataclass(frozen=True, slots=True)
class ScopePolicy:
    """Trusted configuration for one exact investigation scope."""

    ref: str
    sources: tuple[LogSource, ...]
    allowed_template_ids: frozenset[str]
    allowed_operations: frozenset[QueryOperation]
    max_time_span: timedelta
    max_result_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not _SCOPE_REF_PATTERN.fullmatch(self.ref):
            raise ValueError("scope policy ref is invalid")
        if (
            not isinstance(self.sources, tuple)
            or not self.sources
            or any(not isinstance(source, LogSource) for source in self.sources)
            or any(not _is_valid_source(source) for source in self.sources)
            or len(self.sources) != len(set(self.sources))
        ):
            raise ValueError("scope policy sources must be non-empty and unique")

        template_ids = frozenset(self.allowed_template_ids)
        operations = frozenset(self.allowed_operations)
        if not template_ids or any(
            not isinstance(item, str) or not _TEMPLATE_PATTERN.fullmatch(item)
            for item in template_ids
        ):
            raise ValueError("allowed_template_ids must contain valid template identifiers")
        if not operations or any(not isinstance(item, QueryOperation) for item in operations):
            raise ValueError("allowed_operations must contain query operations")
        if not isinstance(self.max_time_span, timedelta) or self.max_time_span <= timedelta(0):
            raise ValueError("max_time_span must be positive")
        if type(self.max_result_limit) is not int or self.max_result_limit < 1:
            raise ValueError("max_result_limit must be a positive integer")

        object.__setattr__(
            self,
            "sources",
            tuple(sorted(self.sources, key=lambda item: (item.index, item.sourcetype))),
        )
        object.__setattr__(self, "allowed_template_ids", template_ids)
        object.__setattr__(self, "allowed_operations", operations)


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedQueryPlan:
    """A sealed local plan created only after QueryPolicyGate accepts a draft.

    This is an application-layer construction guard, not a security token against
    hostile Python code. Disabling public construction also prevents
    ``dataclasses.replace`` from copying an earlier authorization onto new data.
    """

    scope_ref: str
    sources: tuple[LogSource, ...]
    plan: QueryPlan

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("authorized query plan must be created by QueryPolicyGate")

    @classmethod
    def _create(
        cls,
        *,
        scope_ref: str,
        sources: tuple[LogSource, ...],
        plan: QueryPlan,
    ) -> AuthorizedQueryPlan:
        instance = object.__new__(cls)
        object.__setattr__(instance, "scope_ref", scope_ref)
        object.__setattr__(instance, "sources", sources)
        object.__setattr__(instance, "plan", plan)
        instance._validate()
        return instance

    def _validate(self) -> None:
        if not isinstance(self.scope_ref, str) or not _SCOPE_REF_PATTERN.fullmatch(self.scope_ref):
            raise ValueError("authorized scope_ref is invalid")
        if (
            not isinstance(self.sources, tuple)
            or not self.sources
            or any(not isinstance(source, LogSource) for source in self.sources)
            or any(not _is_valid_source(source) for source in self.sources)
            or len(self.sources) != len(set(self.sources))
        ):
            raise ValueError("authorized query sources are invalid")
        if not isinstance(self.plan, QueryPlan):
            raise ValueError("authorized query plan is invalid")
        if not _plan_shape_is_valid(self.plan):
            raise ValueError("authorized query plan shape is invalid")
        object.__setattr__(
            self,
            "sources",
            tuple(sorted(self.sources, key=lambda item: (item.index, item.sourcetype))),
        )


class ScopePolicyRegistry:
    """Resolve scope references by exact key with no wildcard or fallback."""

    def __init__(self, policies: tuple[ScopePolicy, ...]) -> None:
        if not isinstance(policies, tuple) or not policies:
            raise ValueError("at least one scope policy is required")
        if any(not isinstance(policy, ScopePolicy) for policy in policies):
            raise ValueError("scope policy registry contains an invalid policy")
        refs = tuple(policy.ref for policy in policies)
        if len(refs) != len(set(refs)):
            raise ValueError("scope policy refs must be unique")
        self._policies = {policy.ref: policy for policy in policies}

    def resolve(self, scope_ref: str) -> ScopePolicy:
        policy = self._policies.get(scope_ref)
        if policy is None:
            raise QueryPolicyError(
                "query_policy.unknown_scope",
                "The requested log scope is not configured.",
            ) from None
        return policy

    @property
    def refs(self) -> tuple[str, ...]:
        """Expose only stable semantic scope identifiers, never physical sources."""

        return tuple(sorted(self._policies))


class QueryPlanCompiler:
    """Compile a semantic intent into a small, provider-neutral query vocabulary."""

    def __init__(
        self,
        *,
        triage_result_limit: int = 100,
        verify_result_limit: int = 50,
    ) -> None:
        for value, field_name in (
            (triage_result_limit, "triage_result_limit"),
            (verify_result_limit, "verify_result_limit"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        self._triage_result_limit = triage_result_limit
        self._verify_result_limit = verify_result_limit

    def compile(self, intent: QueryIntent) -> QueryPlan:
        if intent.kind is QueryKind.TRIAGE:
            return QueryPlan(
                template_id="triage.error_summary.v1",
                kind=intent.kind,
                operation=QueryOperation.ERROR_SUMMARY,
                time_range=intent.time_range,
                result_limit=self._triage_result_limit,
            )
        if intent.kind is QueryKind.VERIFY:
            try:
                literal = QueryLiteral(intent.goal)
            except ValueError:
                raise QueryPolicyError(
                    "query_policy.denied",
                    "The requested log query is not allowed.",
                ) from None
            return QueryPlan(
                template_id="verify.event_sample.v1",
                kind=intent.kind,
                operation=QueryOperation.EVENT_SAMPLE,
                time_range=intent.time_range,
                result_limit=self._verify_result_limit,
                literal_terms=(literal,),
            )
        raise QueryPolicyError(
            "query_policy.denied",
            "The requested log query is not allowed.",
        )


class QueryPolicyGate:
    """Authorize a query draft using positive allowlists and hard limits."""

    def authorize(
        self,
        *,
        policy: ScopePolicy,
        investigation_range: TimeRange,
        intent: QueryIntent,
        plan: QueryPlan,
    ) -> AuthorizedQueryPlan:
        if not self._is_allowed(policy, investigation_range, intent, plan):
            raise QueryPolicyError(
                "query_policy.denied",
                "The requested log query is not allowed.",
            ) from None
        return AuthorizedQueryPlan._create(
            scope_ref=policy.ref,
            sources=policy.sources,
            plan=plan,
        )

    def _is_allowed(
        self,
        policy: ScopePolicy,
        investigation_range: TimeRange,
        intent: QueryIntent,
        plan: QueryPlan,
    ) -> bool:
        if plan.kind is not intent.kind or plan.time_range != intent.time_range:
            return False
        if plan.template_id not in policy.allowed_template_ids:
            return False
        if plan.operation not in policy.allowed_operations:
            return False
        if plan.result_limit > policy.max_result_limit:
            return False

        requested = investigation_range
        actual = plan.time_range
        if actual.start < requested.start or actual.end > requested.end:
            return False
        if actual.end - actual.start > policy.max_time_span:
            return False
        if plan.kind is QueryKind.TRIAGE and actual != requested:
            return False

        return _plan_shape_is_valid(plan)


class SafeQueryPipeline:
    """Resolve, compile and authorize every query before an adapter sees it."""

    def __init__(
        self,
        registry: ScopePolicyRegistry,
        *,
        compiler: QueryPlanCompiler | None = None,
        gate: QueryPolicyGate | None = None,
    ) -> None:
        self._registry = registry
        self._compiler = QueryPlanCompiler() if compiler is None else compiler
        self._gate = QueryPolicyGate() if gate is None else gate

    def prepare(
        self,
        *,
        scope_ref: str,
        investigation_range: TimeRange,
        intent: QueryIntent,
    ) -> AuthorizedQueryPlan:
        policy = self._registry.resolve(scope_ref)
        plan = self._compiler.compile(intent)
        return self._gate.authorize(
            policy=policy,
            investigation_range=investigation_range,
            intent=intent,
            plan=plan,
        )


def _is_safe_literal_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= _MAX_LITERAL_LENGTH
        and _LITERAL_PATTERN.fullmatch(value) is not None
        and _BOOLEAN_OPERATOR_PATTERN.search(value) is None
    )


def _is_valid_source(source: LogSource) -> bool:
    return (
        isinstance(source.index, str)
        and _INDEX_PATTERN.fullmatch(source.index) is not None
        and isinstance(source.sourcetype, str)
        and _SOURCETYPE_PATTERN.fullmatch(source.sourcetype) is not None
    )


def _plan_shape_is_valid(plan: QueryPlan) -> bool:
    if plan.kind is QueryKind.TRIAGE:
        return (
            plan.operation is QueryOperation.ERROR_SUMMARY
            and plan.template_id == "triage.error_summary.v1"
            and not plan.literal_terms
        )
    if plan.kind is QueryKind.VERIFY:
        return (
            plan.operation is QueryOperation.EVENT_SAMPLE
            and plan.template_id == "verify.event_sample.v1"
            and len(plan.literal_terms) == 1
            and isinstance(plan.literal_terms[0], QueryLiteral)
        )
    return False
