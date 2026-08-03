from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from log_agent.domain.models import (
    ConclusionOutcome,
    InvestigationRequest,
    Phase,
    TerminationReason,
    TimeRange,
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SCOPE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TOKEN_PATTERN = re.compile(r"^[a-z]+-[0-9a-f]{32}$")


def _require_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} is invalid")


def _require_revision(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} is invalid")


def _require_report_token(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _REPORT_TOKEN_PATTERN.fullmatch(value):
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


def _require_unique_ids(values: tuple[str, ...], field_name: str, *, max_items: int) -> None:
    if not isinstance(values, tuple) or len(values) > max_items:
        raise ValueError(f"{field_name} is invalid")
    for value in values:
        _require_id(value, field_name)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ExpectedIncidentResult:
    phase: Phase
    termination_reason: TerminationReason
    conclusion_outcome: ConclusionOutcome | None
    root_cause_key: str | None
    root_cause_summary: str | None
    required_evidence_labels: tuple[str, ...]
    failure_code: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, Phase) or not self.phase.is_terminal:
            raise ValueError("expected phase must be terminal")
        if not isinstance(self.termination_reason, TerminationReason):
            raise ValueError("expected termination_reason is invalid")
        if self.conclusion_outcome is not None and not isinstance(
            self.conclusion_outcome, ConclusionOutcome
        ):
            raise ValueError("expected conclusion_outcome is invalid")
        _require_unique_ids(
            self.required_evidence_labels,
            "expected required_evidence_labels",
            max_items=50,
        )
        object.__setattr__(
            self,
            "required_evidence_labels",
            tuple(sorted(self.required_evidence_labels)),
        )
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str)
            or not _ERROR_CODE_PATTERN.fullmatch(self.failure_code)
        ):
            raise ValueError("expected failure_code is invalid")

        if self.phase is Phase.COMPLETED:
            if (
                self.termination_reason is not TerminationReason.ROOT_CAUSE_IDENTIFIED
                or self.conclusion_outcome is not ConclusionOutcome.CONCLUSIVE
                or self.root_cause_key is None
                or self.root_cause_summary is None
                or not self.required_evidence_labels
                or self.failure_code is not None
            ):
                raise ValueError("completed expectation fields are inconsistent")
            _require_id(self.root_cause_key, "expected root_cause_key")
            _require_text(
                self.root_cause_summary,
                "expected root_cause_summary",
                max_length=2_000,
            )
            return

        if self.phase is Phase.INCONCLUSIVE:
            if (
                self.termination_reason
                not in {
                    TerminationReason.NO_DATA,
                    TerminationReason.INSUFFICIENT_EVIDENCE,
                    TerminationReason.QUERY_BUDGET_EXHAUSTED,
                }
                or self.conclusion_outcome is not ConclusionOutcome.INCONCLUSIVE
                or self.root_cause_key is not None
                or self.root_cause_summary is not None
                or self.required_evidence_labels
                or self.failure_code is not None
            ):
                raise ValueError("inconclusive expectation fields are inconsistent")
            return

        if self.phase is Phase.FAILED:
            if (
                self.termination_reason is not TerminationReason.OPERATION_FAILED
                or self.conclusion_outcome is not None
                or self.root_cause_key is not None
                or self.root_cause_summary is not None
                or self.required_evidence_labels
                or self.failure_code is None
            ):
                raise ValueError("failed expectation fields are inconsistent")
            return

        if (
            self.phase is not Phase.CANCELLED
            or self.termination_reason is not TerminationReason.USER_CANCELLED
            or self.conclusion_outcome is not None
            or self.root_cause_key is not None
            or self.root_cause_summary is not None
            or self.required_evidence_labels
            or self.failure_code is not None
        ):
            raise ValueError("cancelled expectation fields are inconsistent")


@dataclass(frozen=True, slots=True)
class IncidentCase:
    id: str
    request: InvestigationRequest
    replay_fixture_id: str
    expected: ExpectedIncidentResult
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.id, "incident case id")
        if not isinstance(self.request, InvestigationRequest):
            raise ValueError("incident request is invalid")
        _require_text(self.request.question, "incident question", max_length=2_000)
        if not _SCOPE_REF_PATTERN.fullmatch(self.request.scope_ref):
            raise ValueError("incident scope_ref is invalid")
        if not isinstance(self.request.time_range, TimeRange):
            raise ValueError("incident time_range is invalid")
        _require_id(self.replay_fixture_id, "incident replay_fixture_id")
        if not isinstance(self.expected, ExpectedIncidentResult):
            raise ValueError("incident expected result is invalid")
        _require_unique_ids(self.tags, "incident tags", max_items=20)
        object.__setattr__(self, "tags", tuple(sorted(self.tags)))


@dataclass(frozen=True, slots=True)
class IncidentDataset:
    schema_version: int
    dataset_id: str
    revision: str
    cases: tuple[IncidentCase, ...]
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("incident dataset schema_version must be 1")
        _require_id(self.dataset_id, "incident dataset_id")
        _require_revision(self.revision, "incident revision")
        if (
            not isinstance(self.cases, tuple)
            or not self.cases
            or len(self.cases) > 500
            or any(not isinstance(case, IncidentCase) for case in self.cases)
        ):
            raise ValueError("incident cases are invalid")
        case_ids = tuple(case.id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("incident case ids must not contain duplicates")
        object.__setattr__(self, "cases", tuple(sorted(self.cases, key=lambda case: case.id)))
        object.__setattr__(self, "content_hash", _incident_semantic_content_hash(self))


def _incident_semantic_content_hash(dataset: IncidentDataset) -> str:
    """Hash the normalized, validated v1 DTO rather than caller-supplied bytes."""

    document = {
        "semantic_schema": "incident-dataset-v1",
        "schema_version": dataset.schema_version,
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "cases": [
            {
                "id": case.id,
                "request": {
                    "question": case.request.question,
                    "scope_ref": case.request.scope_ref,
                    "start": _canonical_utc(case.request.time_range.start),
                    "end": _canonical_utc(case.request.time_range.end),
                },
                "replay_fixture_id": case.replay_fixture_id,
                "expected": {
                    "phase": case.expected.phase.value,
                    "termination_reason": case.expected.termination_reason.value,
                    "conclusion_outcome": (
                        None
                        if case.expected.conclusion_outcome is None
                        else case.expected.conclusion_outcome.value
                    ),
                    "root_cause_key": case.expected.root_cause_key,
                    "root_cause_summary": case.expected.root_cause_summary,
                    "required_evidence_labels": list(case.expected.required_evidence_labels),
                    "failure_code": case.expected.failure_code,
                },
                "tags": list(case.tags),
            }
            for case in dataset.cases
        ],
    }
    canonical = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class EvalViolationCode(StrEnum):
    PHASE_MISMATCH = "outcome.phase_mismatch"
    TERMINATION_REASON_MISMATCH = "outcome.termination_reason_mismatch"
    CONCLUSION_OUTCOME_MISMATCH = "outcome.conclusion_outcome_mismatch"
    ROOT_CAUSE_MISMATCH = "outcome.root_cause_mismatch"
    FAILURE_CODE_MISMATCH = "outcome.failure_code_mismatch"
    REQUIRED_EVIDENCE_MISSING = "outcome.required_evidence_missing"
    RESULT_NOT_TERMINAL = "process.result_not_terminal"
    PENDING_OPERATION_PRESENT = "process.pending_operation_present"
    STATE_LINEAGE_MISMATCH = "process.state_lineage_mismatch"
    BUDGET_CONFIGURATION_CHANGED = "process.budget_configuration_changed"
    QUERY_BUDGET_EXCEEDED = "process.query_budget_exceeded"
    VERIFY_BUDGET_EXCEEDED = "process.verify_budget_exceeded"
    QUERY_LEDGER_EXCEEDS_CALLS = "process.query_ledger_exceeds_calls"
    QUERY_LEDGER_REQUEST_MISMATCH = "process.query_ledger_request_mismatch"
    QUERY_ACCOUNTING_MISMATCH = "process.query_accounting_mismatch"
    LOG_CALLS_EXCEED_ISSUED = "process.log_calls_exceed_issued"
    VERIFY_CALLS_EXCEED_ISSUED = "process.verify_calls_exceed_issued"
    QUERY_SCOPE_MISMATCH = "process.query_scope_mismatch"
    QUERY_TIME_RANGE_MISMATCH = "process.query_time_range_mismatch"
    EVIDENCE_TIME_RANGE_MISMATCH = "process.evidence_time_range_mismatch"
    POLICY_FAILURE_REACHED_PORT = "process.policy_failure_reached_port"
    EVIDENCE_LABEL_UNMAPPED = "process.evidence_label_unmapped"

    @property
    def is_outcome(self) -> bool:
        return self.value.startswith("outcome.")


class EvalFailureCategory(StrEnum):
    NONE = "none"
    QUERY_POLICY = "query_policy"
    LOG_SEARCH = "log_search"
    REASONING = "reasoning"
    EXTERNAL_ERROR = "external_error"


@dataclass(frozen=True, slots=True)
class EvalCaseReport:
    case_token: str
    fixture_token: str
    fixture_fingerprint: str
    expected_phase: Phase
    actual_phase: Phase
    expected_termination_reason: TerminationReason
    actual_termination_reason: TerminationReason | None
    expected_conclusion_outcome: ConclusionOutcome | None
    actual_conclusion_outcome: ConclusionOutcome | None
    failure_category: EvalFailureCategory
    root_cause_matched: bool | None
    failure_code_matched: bool | None
    required_evidence_count: int
    matched_required_evidence_count: int
    issued_query_count: int
    issued_verify_query_count: int
    successful_query_count: int
    log_port_call_count: int
    reasoning_call_count: int
    truncated_query_count: int
    violations: tuple[EvalViolationCode, ...]

    def __post_init__(self) -> None:
        _require_report_token(self.case_token, "eval report case_token")
        _require_report_token(self.fixture_token, "eval report fixture_token")
        if not isinstance(self.fixture_fingerprint, str) or not _CONTENT_HASH_PATTERN.fullmatch(
            self.fixture_fingerprint
        ):
            raise ValueError("eval report fixture_fingerprint is invalid")
        if not isinstance(self.expected_phase, Phase) or not isinstance(self.actual_phase, Phase):
            raise ValueError("eval report phase is invalid")
        if not isinstance(self.expected_termination_reason, TerminationReason):
            raise ValueError("eval report expected termination reason is invalid")
        if self.actual_termination_reason is not None and not isinstance(
            self.actual_termination_reason, TerminationReason
        ):
            raise ValueError("eval report actual termination reason is invalid")
        for value in (
            self.expected_conclusion_outcome,
            self.actual_conclusion_outcome,
        ):
            if value is not None and not isinstance(value, ConclusionOutcome):
                raise ValueError("eval report conclusion outcome is invalid")
        if not isinstance(self.failure_category, EvalFailureCategory):
            raise ValueError("eval report failure category is invalid")
        for value in (self.root_cause_matched, self.failure_code_matched):
            if value is not None and type(value) is not bool:
                raise ValueError("eval report match flag is invalid")
        counts = (
            self.required_evidence_count,
            self.matched_required_evidence_count,
            self.issued_query_count,
            self.issued_verify_query_count,
            self.successful_query_count,
            self.log_port_call_count,
            self.reasoning_call_count,
            self.truncated_query_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("eval report count is invalid")
        if self.matched_required_evidence_count > self.required_evidence_count:
            raise ValueError("eval report matched evidence count is invalid")
        if not isinstance(self.violations, tuple) or any(
            not isinstance(item, EvalViolationCode) for item in self.violations
        ):
            raise ValueError("eval report violations are invalid")
        if len(self.violations) != len(set(self.violations)):
            raise ValueError("eval report violations must not contain duplicates")

    @property
    def outcome_passed(self) -> bool:
        return not any(item.is_outcome for item in self.violations)

    @property
    def process_passed(self) -> bool:
        return not any(not item.is_outcome for item in self.violations)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_json_object(self) -> dict[str, Any]:
        return {
            "case_token": self.case_token,
            "fixture": {
                "token": self.fixture_token,
                "semantic_fingerprint": self.fixture_fingerprint,
            },
            "passed": self.passed,
            "outcome_passed": self.outcome_passed,
            "process_passed": self.process_passed,
            "expected": {
                "phase": self.expected_phase.value,
                "termination_reason": self.expected_termination_reason.value,
                "conclusion_outcome": (
                    None
                    if self.expected_conclusion_outcome is None
                    else self.expected_conclusion_outcome.value
                ),
                "required_evidence_count": self.required_evidence_count,
            },
            "actual": {
                "phase": self.actual_phase.value,
                "termination_reason": (
                    None
                    if self.actual_termination_reason is None
                    else self.actual_termination_reason.value
                ),
                "conclusion_outcome": (
                    None
                    if self.actual_conclusion_outcome is None
                    else self.actual_conclusion_outcome.value
                ),
                "failure_category": self.failure_category.value,
                "root_cause_matched": self.root_cause_matched,
                "failure_code_matched": self.failure_code_matched,
                "matched_required_evidence_count": self.matched_required_evidence_count,
                "issued_query_count": self.issued_query_count,
                "issued_verify_query_count": self.issued_verify_query_count,
                "successful_query_count": self.successful_query_count,
                "log_port_call_count": self.log_port_call_count,
                "reasoning_call_count": self.reasoning_call_count,
                "truncated_query_count": self.truncated_query_count,
            },
            "violations": [item.value for item in self.violations],
        }


@dataclass(frozen=True, slots=True)
class EvalRunReport:
    schema_version: int
    harness_version: str
    fingerprint_key_id: str
    dataset_token: str
    dataset_fingerprint: str
    query_policy_fingerprint: str
    cases: tuple[EvalCaseReport, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("eval run report schema_version must be 1")
        _require_revision(self.harness_version, "eval run report harness_version")
        _require_report_token(self.fingerprint_key_id, "eval report fingerprint_key_id")
        _require_report_token(self.dataset_token, "eval report dataset_token")
        for value, field_name in (
            (self.dataset_fingerprint, "dataset_fingerprint"),
            (self.query_policy_fingerprint, "query_policy_fingerprint"),
        ):
            if not isinstance(value, str) or not _CONTENT_HASH_PATTERN.fullmatch(value):
                raise ValueError(f"eval run report {field_name} is invalid")
        if (
            not isinstance(self.cases, tuple)
            or not self.cases
            or any(not isinstance(case, EvalCaseReport) for case in self.cases)
        ):
            raise ValueError("eval run report cases are invalid")
        case_tokens = tuple(case.case_token for case in self.cases)
        if len(case_tokens) != len(set(case_tokens)):
            raise ValueError("eval run report case tokens must be unique")

    @property
    def passed_case_count(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def passed(self) -> bool:
        return self.passed_case_count == len(self.cases)

    def to_json_object(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "harness_version": self.harness_version,
            "quality_scope": "deterministic_fake_wiring_only",
            "historical_incident_quality_validated": False,
            "fingerprinting": {
                "algorithm": "hmac-sha256-v1",
                "key_id": self.fingerprint_key_id,
            },
            "dataset": {
                "token": self.dataset_token,
                "semantic_fingerprint": self.dataset_fingerprint,
            },
            "query_policy_fingerprint": self.query_policy_fingerprint,
            "passed": self.passed,
            "case_count": len(self.cases),
            "passed_case_count": self.passed_case_count,
            "cases": [case.to_json_object() for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_json_object(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
