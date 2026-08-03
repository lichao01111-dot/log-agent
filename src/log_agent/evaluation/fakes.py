from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from log_agent.adapters.fakes import (
    DeterministicReasoningPort,
    FakeLogRow,
    FakeLogSearchPort,
    FakeSearchResponse,
)
from log_agent.application.executor import CommandExecutor
from log_agent.application.query_security import SafeQueryPipeline, ScopePolicyRegistry
from log_agent.application.runner import InvestigationRunner
from log_agent.domain.models import (
    Investigation,
    QueryBudget,
    QueryIntent,
    QueryKind,
)
from log_agent.evaluation.harness import (
    EvalCaseInput,
    EvalCaseRuntime,
    EvalHarnessError,
    EvidenceLabelBinding,
    RootCauseBinding,
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TEMPLATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_OPAQUE_HYPOTHESIS_ID = "eval-hypothesis-1"


def _require_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} is invalid")


def _require_text(value: object, field_name: str, *, max_length: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"{field_name} is invalid")


@dataclass(frozen=True, slots=True)
class FakeEvalRow:
    evidence_label: str
    fact_statement: str
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_id(self.evidence_label, "fake eval evidence_label")
        _require_text(self.fact_statement, "fake eval fact_statement", max_length=2_000)
        if self.occurred_at is not None and (
            self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("fake eval occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class FakeEvalResponse:
    template_id: str
    summary: str
    rows: tuple[FakeEvalRow, ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not _TEMPLATE_PATTERN.fullmatch(
            self.template_id
        ):
            raise ValueError("fake eval template_id is invalid")
        _require_text(self.summary, "fake eval summary", max_length=2_000)
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) > 100
            or any(not isinstance(row, FakeEvalRow) for row in self.rows)
        ):
            raise ValueError("fake eval rows are invalid")
        if type(self.truncated) is not bool:
            raise ValueError("fake eval truncated must be a boolean")


@dataclass(frozen=True, slots=True)
class FakeEvalScenario:
    fixture_id: str
    revision: str
    triage_goals: tuple[str, ...]
    max_total_queries: int
    max_verify_queries: int
    responses: tuple[FakeEvalResponse, ...]
    root_cause_key: str
    hypothesis_statement: str
    verification_goal: str
    conclusion_summary: str
    recommendations: tuple[str, ...] = ()
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_id(self.fixture_id, "fake eval fixture_id")
        if not isinstance(self.revision, str) or not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError("fake eval revision is invalid")
        if (
            not isinstance(self.triage_goals, tuple)
            or not self.triage_goals
            or len(self.triage_goals) > 8
        ):
            raise ValueError("fake eval triage_goals are invalid")
        for goal in self.triage_goals:
            _require_text(goal, "fake eval triage goal", max_length=200)
        if len(self.triage_goals) != len(set(self.triage_goals)):
            raise ValueError("fake eval triage_goals must not contain duplicates")
        if (
            type(self.max_total_queries) is not int
            or not 1 <= self.max_total_queries <= 32
            or type(self.max_verify_queries) is not int
        ):
            raise ValueError("fake eval query budget is invalid")
        QueryBudget(
            max_total_queries=self.max_total_queries,
            max_verify_queries=self.max_verify_queries,
        )
        if (
            not isinstance(self.responses, tuple)
            or not self.responses
            or len(self.responses) > 16
            or any(not isinstance(item, FakeEvalResponse) for item in self.responses)
        ):
            raise ValueError("fake eval responses are invalid")
        template_ids = tuple(response.template_id for response in self.responses)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("fake eval response template_ids must be unique")
        labels = tuple(row.evidence_label for response in self.responses for row in response.rows)
        if len(labels) != len(set(labels)):
            raise ValueError("fake eval evidence labels must be unique")
        _require_id(self.root_cause_key, "fake eval root_cause_key")
        for value, field_name, max_length in (
            (self.hypothesis_statement, "hypothesis_statement", 500),
            (self.verification_goal, "verification_goal", 200),
            (self.conclusion_summary, "conclusion_summary", 2_000),
        ):
            _require_text(value, f"fake eval {field_name}", max_length=max_length)
        if not isinstance(self.recommendations, tuple) or len(self.recommendations) > 5:
            raise ValueError("fake eval recommendations are invalid")
        for recommendation in self.recommendations:
            _require_text(recommendation, "fake eval recommendation", max_length=500)
        object.__setattr__(self, "content_hash", self._calculate_hash())

    def _calculate_hash(self) -> str:
        document = {
            "fixture_id": self.fixture_id,
            "revision": self.revision,
            "triage_goals": list(self.triage_goals),
            "budget": {
                "max_total_queries": self.max_total_queries,
                "max_verify_queries": self.max_verify_queries,
            },
            "responses": [
                {
                    "template_id": response.template_id,
                    "summary": response.summary,
                    "truncated": response.truncated,
                    "rows": [
                        {
                            "evidence_label": row.evidence_label,
                            "fact_statement": row.fact_statement,
                            "occurred_at": (
                                None if row.occurred_at is None else _canonical_utc(row.occurred_at)
                            ),
                        }
                        for row in response.rows
                    ],
                }
                for response in sorted(self.responses, key=lambda item: item.template_id)
            ],
            "reasoning": {
                "root_cause_key": self.root_cause_key,
                "hypothesis_statement": self.hypothesis_statement,
                "verification_goal": self.verification_goal,
                "conclusion_summary": self.conclusion_summary,
                "recommendations": list(self.recommendations),
            },
        }
        canonical = json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class FakeEvalRuntimeFactory:
    """Build a fresh existing-Fake runner for each incident case."""

    def __init__(
        self,
        scope_policies: ScopePolicyRegistry,
        scenarios: tuple[FakeEvalScenario, ...],
    ) -> None:
        if not isinstance(scope_policies, ScopePolicyRegistry):
            raise ValueError("scope_policies must be a ScopePolicyRegistry")
        if (
            not isinstance(scenarios, tuple)
            or not scenarios
            or any(not isinstance(item, FakeEvalScenario) for item in scenarios)
        ):
            raise ValueError("fake eval scenarios are invalid")
        fixture_ids = tuple(item.fixture_id for item in scenarios)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("fake eval fixture ids must be unique")
        self._scope_policies = scope_policies
        self._pipeline = SafeQueryPipeline(scope_policies)
        self._scenarios = {item.fixture_id: item for item in scenarios}
        self._query_policy_hash = _query_policy_hash(scope_policies)

    def prepare(self, case_input: EvalCaseInput) -> EvalCaseRuntime:
        if not isinstance(case_input, EvalCaseInput):
            raise ValueError("case_input must be an EvalCaseInput")
        scenario = self._scenarios.get(case_input.replay_fixture_id)
        if scenario is None:
            raise EvalHarnessError(
                "eval.fixture_missing",
                "An incident replay fixture is not configured.",
            )

        responses: dict[str, FakeSearchResponse] = {}
        evidence_label_bindings: list[EvidenceLabelBinding] = []
        ordered_responses = sorted(scenario.responses, key=lambda item: item.template_id)
        for response_position, response in enumerate(ordered_responses, start=1):
            fake_rows: list[FakeLogRow] = []
            for row_position, row in enumerate(response.rows, start=1):
                record_ref = _record_ref(response_position, row_position)
                fake_rows.append(
                    FakeLogRow(
                        record_ref=record_ref,
                        fact_statement=row.fact_statement,
                        occurred_at=row.occurred_at,
                    )
                )
                evidence_label_bindings.append(
                    EvidenceLabelBinding(
                        record_ref=record_ref,
                        evidence_label=row.evidence_label,
                    )
                )
            responses[response.template_id] = FakeSearchResponse(
                summary=response.summary,
                rows=tuple(fake_rows),
                truncated=response.truncated,
            )
        search = FakeLogSearchPort(responses)
        reasoning = DeterministicReasoningPort(
            hypothesis_id=_OPAQUE_HYPOTHESIS_ID,
            hypothesis_statement=scenario.hypothesis_statement,
            verification_goal=scenario.verification_goal,
            conclusion_summary=scenario.conclusion_summary,
            recommendations=scenario.recommendations,
        )
        initial = Investigation(
            # Domain IDs reach the reasoning boundary through WorkingMemory. Keep
            # them independent of semantic case and fixture identifiers.
            id="eval:opaque-run",
            request=case_input.request,
            triage_plan=tuple(
                QueryIntent(
                    kind=QueryKind.TRIAGE,
                    goal=goal,
                    time_range=case_input.request.time_range,
                )
                for goal in scenario.triage_goals
            ),
            budget=QueryBudget(
                max_total_queries=scenario.max_total_queries,
                max_verify_queries=scenario.max_verify_queries,
            ),
        )
        runner = InvestigationRunner(CommandExecutor(search, reasoning, self._pipeline))
        return EvalCaseRuntime(
            fixture_id=scenario.fixture_id,
            fixture_revision=scenario.revision,
            fixture_hash=scenario.content_hash,
            query_policy_hash=self._query_policy_hash,
            initial=initial,
            driver=runner,
            search_probe=search,
            reasoning_probe=reasoning,
            root_cause_bindings=(
                RootCauseBinding(
                    hypothesis_id=_OPAQUE_HYPOTHESIS_ID,
                    root_cause_key=scenario.root_cause_key,
                ),
            ),
            evidence_label_bindings=tuple(evidence_label_bindings),
        )


def _record_ref(response_position: int, row_position: int) -> str:
    """Return an opaque ref that cannot reveal case, fixture, or expected labels."""

    return f"fake-eval://row/{response_position:02d}/{row_position:03d}"


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _query_policy_hash(registry: ScopePolicyRegistry) -> str:
    policies = []
    for ref in registry.refs:
        policy = registry.resolve(ref)
        policies.append(
            {
                "ref": policy.ref,
                "sources": [
                    {"index": source.index, "sourcetype": source.sourcetype}
                    for source in policy.sources
                ],
                "allowed_template_ids": sorted(policy.allowed_template_ids),
                "allowed_operations": sorted(item.value for item in policy.allowed_operations),
                "max_time_span_microseconds": _timedelta_microseconds(policy.max_time_span),
                "max_result_limit": policy.max_result_limit,
            }
        )
    canonical = json.dumps(
        policies,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _timedelta_microseconds(value: timedelta) -> int:
    """Convert without a float round-trip, preserving one-microsecond changes."""

    return ((value.days * 86_400) + value.seconds) * 1_000_000 + value.microseconds
