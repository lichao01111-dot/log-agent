from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Protocol

from log_agent.application.ports import SearchRequest
from log_agent.domain.models import (
    Investigation,
    InvestigationRequest,
    Phase,
    QueryKind,
    QueryRecord,
)
from log_agent.evaluation.models import (
    EvalCaseReport,
    EvalFailureCategory,
    EvalRunReport,
    EvalViolationCode,
    IncidentCase,
    IncidentDataset,
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HARNESS_VERSION = "deterministic-eval-v1"


class EvalHarnessError(RuntimeError):
    """An evaluation-boundary error with an input-independent safe message."""

    def __init__(self, code: str, safe_message: str) -> None:
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


class InvestigationDriver(Protocol):
    async def run(self, initial: Investigation) -> Investigation: ...


class SearchProbe(Protocol):
    requests: list[SearchRequest]


class ReasoningProbe(Protocol):
    calls: list[str]


class EvalRuntimeFactory(Protocol):
    def prepare(self, case_input: EvalCaseInput) -> EvalCaseRuntime: ...


@dataclass(frozen=True, slots=True)
class EvalCaseInput:
    """The only incident fields allowed to influence runtime construction."""

    request: InvestigationRequest
    replay_fixture_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, InvestigationRequest):
            raise ValueError("eval case input request is invalid")
        if not isinstance(self.replay_fixture_id, str) or not _ID_PATTERN.fullmatch(
            self.replay_fixture_id
        ):
            raise ValueError("eval case input replay_fixture_id is invalid")


@dataclass(frozen=True, slots=True)
class RootCauseBinding:
    hypothesis_id: str
    root_cause_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id.strip():
            raise ValueError("root-cause hypothesis_id must not be blank")
        if not isinstance(self.root_cause_key, str) or not _ID_PATTERN.fullmatch(
            self.root_cause_key
        ):
            raise ValueError("root-cause key is invalid")


@dataclass(frozen=True, slots=True)
class EvidenceLabelBinding:
    record_ref: str
    evidence_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.record_ref, str) or not self.record_ref.strip():
            raise ValueError("evidence label record_ref must not be blank")
        if not isinstance(self.evidence_label, str) or not _ID_PATTERN.fullmatch(
            self.evidence_label
        ):
            raise ValueError("evidence label is invalid")


@dataclass(frozen=True, slots=True)
class EvalCaseRuntime:
    fixture_id: str
    fixture_revision: str
    fixture_hash: str
    query_policy_hash: str
    initial: Investigation
    driver: InvestigationDriver
    search_probe: SearchProbe
    reasoning_probe: ReasoningProbe
    root_cause_bindings: tuple[RootCauseBinding, ...] = ()
    evidence_label_bindings: tuple[EvidenceLabelBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not _ID_PATTERN.fullmatch(self.fixture_id):
            raise ValueError("eval runtime fixture_id is invalid")
        if not isinstance(self.fixture_revision, str) or not _REVISION_PATTERN.fullmatch(
            self.fixture_revision
        ):
            raise ValueError("eval runtime fixture_revision is invalid")
        for value, field_name in (
            (self.fixture_hash, "fixture_hash"),
            (self.query_policy_hash, "query_policy_hash"),
        ):
            if not isinstance(value, str) or not _CONTENT_HASH_PATTERN.fullmatch(value):
                raise ValueError(f"eval runtime {field_name} is invalid")
        if not isinstance(self.initial, Investigation):
            raise ValueError("eval runtime initial investigation is invalid")
        if not callable(getattr(self.driver, "run", None)):
            raise ValueError("eval runtime driver is invalid")
        if not isinstance(getattr(self.search_probe, "requests", None), list):
            raise ValueError("eval runtime search probe is invalid")
        if not isinstance(getattr(self.reasoning_probe, "calls", None), list):
            raise ValueError("eval runtime reasoning probe is invalid")
        if not isinstance(self.root_cause_bindings, tuple) or any(
            not isinstance(item, RootCauseBinding) for item in self.root_cause_bindings
        ):
            raise ValueError("eval runtime root-cause bindings are invalid")
        if not isinstance(self.evidence_label_bindings, tuple) or any(
            not isinstance(item, EvidenceLabelBinding) for item in self.evidence_label_bindings
        ):
            raise ValueError("eval runtime evidence-label bindings are invalid")
        hypothesis_ids = tuple(item.hypothesis_id for item in self.root_cause_bindings)
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("root-cause hypothesis bindings must be unique")
        record_refs = tuple(item.record_ref for item in self.evidence_label_bindings)
        labels = tuple(item.evidence_label for item in self.evidence_label_bindings)
        if len(record_refs) != len(set(record_refs)) or len(labels) != len(set(labels)):
            raise ValueError("evidence-label bindings must be one-to-one")


class DeterministicEvalHarness:
    """Run frozen Fake scenarios and emit a keyed, content-minimized report."""

    def __init__(
        self,
        runtime_factory: EvalRuntimeFactory,
        *,
        report_hmac_key: bytes,
    ) -> None:
        if not callable(getattr(runtime_factory, "prepare", None)):
            raise ValueError("runtime_factory must implement prepare")
        if type(report_hmac_key) is not bytes or len(report_hmac_key) < 32:
            raise ValueError("report_hmac_key must contain at least 32 bytes")
        self._runtime_factory = runtime_factory
        self._report_hmac_key = report_hmac_key

    async def run(self, dataset: IncidentDataset) -> EvalRunReport:
        if not isinstance(dataset, IncidentDataset):
            raise ValueError("dataset must be an IncidentDataset")

        prepared_items: list[tuple[IncidentCase, EvalCaseRuntime]] = []
        for case in dataset.cases:
            prepare_failure: str | None = None
            runtime: EvalCaseRuntime | None = None
            try:
                runtime = self._runtime_factory.prepare(
                    EvalCaseInput(
                        request=case.request,
                        replay_fixture_id=case.replay_fixture_id,
                    )
                )
            except EvalHarnessError as error:
                prepare_failure = (
                    "fixture_missing"
                    if error.code == "eval.fixture_missing"
                    else "runtime_prepare_failed"
                )
            except Exception:
                prepare_failure = "runtime_prepare_failed"
            if prepare_failure is not None:
                code, message = _PREPARE_FAILURES[prepare_failure]
                raise EvalHarnessError(code, message) from None
            if runtime is None:
                raise EvalHarnessError(
                    "eval.runtime_prepare_failed",
                    "An eval runtime could not be prepared.",
                )
            prepared_items.append((case, runtime))
        prepared = tuple(prepared_items)
        for case, runtime in prepared:
            self._preflight(case, runtime)

        policy_hashes = {runtime.query_policy_hash for _, runtime in prepared}
        if len(policy_hashes) != 1:
            raise EvalHarnessError(
                "eval.query_policy_mismatch",
                "Prepared eval cases do not share one query policy snapshot.",
            )

        reports = []
        for case, runtime in prepared:
            driver_failed = False
            result: Investigation | None = None
            try:
                result = await runtime.driver.run(runtime.initial)
            except Exception:
                driver_failed = True
            if driver_failed:
                raise EvalHarnessError(
                    "eval.driver_failed",
                    "An eval driver failed unexpectedly.",
                ) from None
            if not isinstance(result, Investigation):
                raise EvalHarnessError(
                    "eval.invalid_result",
                    "An eval driver returned an invalid investigation result.",
                )
            evaluation_failed = False
            case_report: EvalCaseReport | None = None
            try:
                case_report = self._evaluate(
                    case,
                    runtime,
                    result,
                    case_token=self._report_token(
                        "case",
                        f"{dataset.dataset_id}\0{case.id}",
                    ),
                    fixture_token=self._report_token("fixture", runtime.fixture_id),
                    fixture_fingerprint=self._report_fingerprint(
                        "fixture",
                        runtime.fixture_hash,
                    ),
                )
            except EvalHarnessError:
                raise
            except Exception:
                evaluation_failed = True
            if evaluation_failed or case_report is None:
                raise EvalHarnessError(
                    "eval.evaluation_failed",
                    "An eval result could not be checked safely.",
                ) from None
            reports.append(case_report)

        return EvalRunReport(
            schema_version=1,
            harness_version=_HARNESS_VERSION,
            fingerprint_key_id=self._report_token("key", "hmac-sha256-v1"),
            dataset_token=self._report_token("dataset", dataset.dataset_id),
            dataset_fingerprint=self._report_fingerprint(
                "dataset",
                dataset.content_hash,
            ),
            query_policy_fingerprint=self._report_fingerprint(
                "query-policy",
                next(iter(policy_hashes)),
            ),
            cases=tuple(reports),
        )

    def _report_token(self, token_type: str, value: str) -> str:
        digest = self._hmac(f"token:{token_type}", value)
        return f"{token_type}-{digest[:32]}"

    def _report_fingerprint(self, artifact_type: str, semantic_digest: str) -> str:
        return self._hmac(f"fingerprint:{artifact_type}", semantic_digest)

    def _hmac(self, domain: str, value: str) -> str:
        message = f"log-agent-eval\0{domain}\0v1\0{value}".encode()
        return hmac.new(self._report_hmac_key, message, hashlib.sha256).hexdigest()

    @staticmethod
    def _preflight(case: IncidentCase, runtime: EvalCaseRuntime) -> None:
        if not isinstance(runtime, EvalCaseRuntime):
            raise EvalHarnessError(
                "eval.invalid_runtime",
                "An eval fixture produced an invalid runtime.",
            )
        if runtime.fixture_id != case.replay_fixture_id:
            raise EvalHarnessError(
                "eval.fixture_mismatch",
                "An eval fixture does not match its incident case.",
            )
        initial = runtime.initial
        if (
            initial.phase is not Phase.NEW
            or initial.request != case.request
            or initial.revision != 0
            or initial.budget.issued_total != 0
            or initial.budget.issued_verify != 0
            or initial.memory.facts
            or initial.memory.hypotheses
            or initial.memory.evidence
            or initial.memory.queries
            or runtime.search_probe.requests
            or runtime.reasoning_probe.calls
        ):
            raise EvalHarnessError(
                "eval.runtime_not_fresh",
                "An eval runtime is not a fresh execution of its incident case.",
            )

        available_labels = {item.evidence_label for item in runtime.evidence_label_bindings}
        if not set(case.expected.required_evidence_labels).issubset(available_labels):
            raise EvalHarnessError(
                "eval.evidence_annotation_missing",
                "An expected evidence label is absent from its replay fixture.",
            )
        if case.expected.root_cause_key is not None:
            available_roots = {item.root_cause_key for item in runtime.root_cause_bindings}
            if case.expected.root_cause_key not in available_roots:
                raise EvalHarnessError(
                    "eval.root_cause_annotation_missing",
                    "An expected root-cause key is absent from its replay fixture.",
                )

    @staticmethod
    def _evaluate(
        case: IncidentCase,
        runtime: EvalCaseRuntime,
        result: Investigation,
        *,
        case_token: str,
        fixture_token: str,
        fixture_fingerprint: str,
    ) -> EvalCaseReport:
        violations: list[EvalViolationCode] = []

        def fail(code: EvalViolationCode) -> None:
            if code not in violations:
                violations.append(code)

        expected = case.expected
        actual_outcome = None if result.conclusion is None else result.conclusion.outcome
        if result.phase is not expected.phase:
            fail(EvalViolationCode.PHASE_MISMATCH)
        if result.termination_reason is not expected.termination_reason:
            fail(EvalViolationCode.TERMINATION_REASON_MISMATCH)
        if actual_outcome is not expected.conclusion_outcome:
            fail(EvalViolationCode.CONCLUSION_OUTCOME_MISMATCH)

        root_cause_matched: bool | None = None
        root_map = {item.hypothesis_id: item.root_cause_key for item in runtime.root_cause_bindings}
        if expected.root_cause_key is not None:
            actual_root_id = (
                None if result.conclusion is None else result.conclusion.root_cause_hypothesis_id
            )
            root_cause_matched = root_map.get(actual_root_id) == expected.root_cause_key
            if not root_cause_matched:
                fail(EvalViolationCode.ROOT_CAUSE_MISMATCH)

        actual_failure_code = None if result.failure is None else result.failure.code
        failure_code_matched: bool | None = None
        if expected.failure_code is not None:
            failure_code_matched = actual_failure_code == expected.failure_code
            if not failure_code_matched:
                fail(EvalViolationCode.FAILURE_CODE_MISMATCH)

        evidence_by_id = {item.id: item for item in result.memory.evidence}
        label_by_record_ref = {
            item.record_ref: item.evidence_label for item in runtime.evidence_label_bindings
        }
        conclusion_labels = {
            label_by_record_ref[evidence_by_id[evidence_id].record_ref]
            for evidence_id in (() if result.conclusion is None else result.conclusion.evidence_ids)
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].record_ref in label_by_record_ref
        }
        required_labels = set(expected.required_evidence_labels)
        matched_required = len(required_labels & conclusion_labels)
        if not required_labels.issubset(conclusion_labels):
            fail(EvalViolationCode.REQUIRED_EVIDENCE_MISSING)

        if not result.phase.is_terminal:
            fail(EvalViolationCode.RESULT_NOT_TERMINAL)
        if result.pending_operation is not None:
            fail(EvalViolationCode.PENDING_OPERATION_PRESENT)
        if (
            result.id != runtime.initial.id
            or result.request != runtime.initial.request
            or result.triage_plan != runtime.initial.triage_plan
        ):
            fail(EvalViolationCode.STATE_LINEAGE_MISMATCH)
        if (
            result.budget.max_total_queries != runtime.initial.budget.max_total_queries
            or result.budget.max_verify_queries != runtime.initial.budget.max_verify_queries
        ):
            fail(EvalViolationCode.BUDGET_CONFIGURATION_CHANGED)
        if result.budget.issued_total > result.budget.max_total_queries:
            fail(EvalViolationCode.QUERY_BUDGET_EXCEEDED)
        if result.budget.issued_verify > result.budget.max_verify_queries:
            fail(EvalViolationCode.VERIFY_BUDGET_EXCEEDED)

        requests = tuple(runtime.search_probe.requests)
        if any(not isinstance(request, SearchRequest) for request in requests):
            raise EvalHarnessError(
                "eval.invalid_search_probe",
                "An eval search probe recorded an invalid request.",
            )
        queries = tuple(result.memory.queries)
        if len(queries) > len(requests):
            fail(EvalViolationCode.QUERY_LEDGER_EXCEEDS_CALLS)
        if len(requests) > result.budget.issued_total:
            fail(EvalViolationCode.LOG_CALLS_EXCEED_ISSUED)
        verify_calls = sum(
            request.authorized_query.plan.kind is QueryKind.VERIFY for request in requests
        )
        successful_verify_queries = sum(query.intent.kind is QueryKind.VERIFY for query in queries)
        if verify_calls > result.budget.issued_verify:
            fail(EvalViolationCode.VERIFY_CALLS_EXCEED_ISSUED)
        if not _query_ledger_matches(
            runtime.initial,
            queries,
            requests,
            hypothesis_ids=frozenset(hypothesis.id for hypothesis in result.memory.hypotheses),
        ):
            fail(EvalViolationCode.QUERY_LEDGER_REQUEST_MISMATCH)
        if not _query_accounting_matches(
            actual_phase=result.phase,
            actual_failure_code=actual_failure_code,
            issued_total=result.budget.issued_total,
            issued_verify=result.budget.issued_verify,
            successful_total=len(queries),
            successful_verify=successful_verify_queries,
            call_total=len(requests),
            call_verify=verify_calls,
        ):
            fail(EvalViolationCode.QUERY_ACCOUNTING_MISMATCH)
        if any(
            request.authorized_query.scope_ref != case.request.scope_ref for request in requests
        ):
            fail(EvalViolationCode.QUERY_SCOPE_MISMATCH)
        requested_range = case.request.time_range
        if any(
            request.authorized_query.plan.time_range.start < requested_range.start
            or request.authorized_query.plan.time_range.end > requested_range.end
            for request in requests
        ):
            fail(EvalViolationCode.QUERY_TIME_RANGE_MISMATCH)
        request_by_query_id = {request.query_id: request for request in requests}
        if any(
            evidence.occurred_at is not None
            and evidence.query_id in request_by_query_id
            and (
                evidence.occurred_at
                < request_by_query_id[evidence.query_id].authorized_query.plan.time_range.start
                or evidence.occurred_at
                > request_by_query_id[evidence.query_id].authorized_query.plan.time_range.end
            )
            for evidence in result.memory.evidence
        ):
            fail(EvalViolationCode.EVIDENCE_TIME_RANGE_MISMATCH)
        if (
            actual_failure_code is not None
            and actual_failure_code.startswith("query_policy.")
            and len(requests) != len(queries)
        ):
            fail(EvalViolationCode.POLICY_FAILURE_REACHED_PORT)
        if any(
            evidence.record_ref not in label_by_record_ref for evidence in result.memory.evidence
        ):
            fail(EvalViolationCode.EVIDENCE_LABEL_UNMAPPED)

        reasoning_calls = tuple(runtime.reasoning_probe.calls)
        failure_category = _failure_category(actual_failure_code)
        return EvalCaseReport(
            case_token=case_token,
            fixture_token=fixture_token,
            fixture_fingerprint=fixture_fingerprint,
            expected_phase=expected.phase,
            actual_phase=result.phase,
            expected_termination_reason=expected.termination_reason,
            actual_termination_reason=result.termination_reason,
            expected_conclusion_outcome=expected.conclusion_outcome,
            actual_conclusion_outcome=actual_outcome,
            failure_category=failure_category,
            root_cause_matched=root_cause_matched,
            failure_code_matched=failure_code_matched,
            required_evidence_count=len(required_labels),
            matched_required_evidence_count=matched_required,
            issued_query_count=result.budget.issued_total,
            issued_verify_query_count=result.budget.issued_verify,
            successful_query_count=len(queries),
            log_port_call_count=len(requests),
            reasoning_call_count=len(reasoning_calls),
            truncated_query_count=sum(query.truncated for query in result.memory.queries),
            violations=tuple(violations),
        )


def _failure_category(code: str | None) -> EvalFailureCategory:
    if code is None:
        return EvalFailureCategory.NONE
    if code.startswith("query_policy."):
        return EvalFailureCategory.QUERY_POLICY
    if code.startswith(("log_search.", "fake_log_search.")):
        return EvalFailureCategory.LOG_SEARCH
    if code.startswith(("reasoning.", "fake_reasoning.", "fake_model.")):
        return EvalFailureCategory.REASONING
    return EvalFailureCategory.EXTERNAL_ERROR


def _query_ledger_matches(
    initial: Investigation,
    queries: tuple[QueryRecord, ...],
    requests: tuple[SearchRequest, ...],
    *,
    hypothesis_ids: frozenset[str],
) -> bool:
    request_ids = tuple(request.query_id for request in requests)
    query_ids = tuple(query.id for query in queries)
    if len(request_ids) != len(set(request_ids)):
        return False
    if query_ids != request_ids[: len(query_ids)]:
        return False

    successful_triage = tuple(
        query.intent for query in queries if query.intent.kind is QueryKind.TRIAGE
    )
    if successful_triage != initial.triage_plan[: len(successful_triage)]:
        return False

    for query, request in zip(queries, requests, strict=False):
        plan = request.authorized_query.plan
        if plan.kind is not query.intent.kind or plan.time_range != query.intent.time_range:
            return False
        if query.intent.kind is QueryKind.TRIAGE:
            if plan.literal_terms:
                return False
        else:
            if query.intent.hypothesis_id not in hypothesis_ids:
                return False
            if tuple(term.value for term in plan.literal_terms) != (query.intent.goal,):
                return False
    return True


def _query_accounting_matches(
    *,
    actual_phase: Phase,
    actual_failure_code: str | None,
    issued_total: int,
    issued_verify: int,
    successful_total: int,
    successful_verify: int,
    call_total: int,
    call_verify: int,
) -> bool:
    if actual_failure_code is None:
        if actual_phase is Phase.CANCELLED:
            uncompleted_total = issued_total - successful_total
            uncompleted_verify = issued_verify - successful_verify
            return (
                uncompleted_total in {0, 1}
                and successful_total <= call_total <= issued_total
                and uncompleted_verify in {0, 1}
                and successful_verify <= call_verify <= issued_verify
                and uncompleted_verify <= uncompleted_total
                and call_verify - successful_verify <= call_total - successful_total
            )
        return (
            issued_total == call_total == successful_total
            and issued_verify == call_verify == successful_verify
        )

    category = _failure_category(actual_failure_code)
    if category is EvalFailureCategory.QUERY_POLICY:
        return (
            issued_total == call_total + 1
            and call_total == successful_total
            and issued_verify - call_verify in {0, 1}
            and call_verify == successful_verify
        )
    if category is EvalFailureCategory.LOG_SEARCH:
        return (
            issued_total == call_total
            and call_total == successful_total + 1
            and issued_verify == call_verify
            and call_verify - successful_verify in {0, 1}
        )
    if category is EvalFailureCategory.REASONING:
        return (
            issued_total == call_total == successful_total
            and issued_verify == call_verify == successful_verify
        )

    failed_calls = call_total - successful_total
    failed_verify_calls = call_verify - successful_verify
    return (
        issued_total == call_total
        and issued_verify == call_verify
        and failed_calls in {0, 1}
        and failed_verify_calls in {0, 1}
        and failed_verify_calls <= failed_calls
    )


_PREPARE_FAILURES = {
    "fixture_missing": (
        "eval.fixture_missing",
        "An incident replay fixture is not configured.",
    ),
    "runtime_prepare_failed": (
        "eval.runtime_prepare_failed",
        "An eval runtime could not be prepared.",
    ),
}
