from __future__ import annotations

import json
import re
import stat
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from log_agent.domain.models import (
    ConclusionOutcome,
    InvestigationRequest,
    Phase,
    TerminationReason,
    TimeRange,
)
from log_agent.evaluation.models import (
    ExpectedIncidentResult,
    IncidentCase,
    IncidentDataset,
)

_DEFAULT_MAX_BYTES = 1_000_000
_MAX_DEPTH = 16
_MAX_CASES = 500
_MAX_TAGS = 20
_MAX_EVIDENCE_LABELS = 50
_UTC_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class IncidentDatasetError(ValueError):
    """A deterministic, location-aware incident dataset error."""

    def __init__(self, location: str, reason: str) -> None:
        self.location = location
        self.reason = reason
        super().__init__(f"{location}: {reason}")


def load_incident_dataset_json(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> IncidentDataset:
    """Load one immutable, all-or-nothing incident evaluation dataset."""

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    dataset_path = Path(path)
    try:
        mode = dataset_path.stat().st_mode
    except OSError:
        raise IncidentDatasetError("$", "dataset must be a readable regular file") from None
    if not stat.S_ISREG(mode):
        raise IncidentDatasetError("$", "dataset must be a regular file")
    try:
        with dataset_path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError:
        raise IncidentDatasetError("$", "dataset must be a readable regular file") from None
    if len(raw) > max_bytes:
        raise IncidentDatasetError("$", "dataset exceeds the byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise IncidentDatasetError("$", "dataset must be valid UTF-8") from None

    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
            parse_float=_reject_float,
        )
    except IncidentDatasetError:
        raise
    except (ValueError, RecursionError):
        raise IncidentDatasetError("$", "dataset is not valid JSON") from None

    _require_bounded_depth(document)
    return _IncidentDatasetDecoder().decode(document)


class _IncidentDatasetDecoder:
    def decode(self, value: object) -> IncidentDataset:
        root = _strict_object(
            value,
            "$",
            required={"schema_version", "dataset_id", "revision", "cases"},
        )
        schema_version = _integer(root["schema_version"], "$.schema_version")
        if schema_version != 1:
            raise IncidentDatasetError("$.schema_version", "unsupported schema version")
        cases = self._cases(root["cases"])
        try:
            return IncidentDataset(
                schema_version=schema_version,
                dataset_id=_string(root["dataset_id"], "$.dataset_id"),
                revision=_string(root["revision"], "$.revision"),
                cases=cases,
            )
        except ValueError as error:
            raise IncidentDatasetError("$", str(error)) from None

    def _cases(self, value: object) -> tuple[IncidentCase, ...]:
        items = _array(value, "$.cases", max_items=_MAX_CASES, allow_empty=False)
        cases = tuple(self._case(item, f"$.cases[{index}]") for index, item in enumerate(items))
        ids = tuple(case.id for case in cases)
        if len(ids) != len(set(ids)):
            raise IncidentDatasetError("$.cases", "case ids must not contain duplicates")
        return cases

    def _case(self, value: object, path: str) -> IncidentCase:
        item = _strict_object(
            value,
            path,
            required={"id", "request", "replay_fixture_id", "expected", "tags"},
        )
        request = self._request(item["request"], f"{path}.request")
        expected = self._expected(item["expected"], f"{path}.expected")
        try:
            return IncidentCase(
                id=_string(item["id"], f"{path}.id"),
                request=request,
                replay_fixture_id=_string(
                    item["replay_fixture_id"],
                    f"{path}.replay_fixture_id",
                ),
                expected=expected,
                tags=_string_tuple(
                    item["tags"],
                    f"{path}.tags",
                    max_items=_MAX_TAGS,
                ),
            )
        except ValueError as error:
            raise IncidentDatasetError(path, str(error)) from None

    def _request(self, value: object, path: str) -> InvestigationRequest:
        item = _strict_object(
            value,
            path,
            required={"question", "scope_ref", "start", "end"},
        )
        start = _utc_datetime(item["start"], f"{path}.start")
        end = _utc_datetime(item["end"], f"{path}.end")
        try:
            return InvestigationRequest(
                question=_bounded_text(
                    item["question"],
                    f"{path}.question",
                    max_length=2_000,
                ),
                scope_ref=_string(item["scope_ref"], f"{path}.scope_ref"),
                time_range=TimeRange(start=start, end=end),
            )
        except ValueError as error:
            raise IncidentDatasetError(path, str(error)) from None

    def _expected(self, value: object, path: str) -> ExpectedIncidentResult:
        item = _strict_object(
            value,
            path,
            required={
                "phase",
                "termination_reason",
                "conclusion_outcome",
                "root_cause_key",
                "root_cause_summary",
                "required_evidence_labels",
                "failure_code",
            },
        )
        phase = _enum(Phase, item["phase"], f"{path}.phase", "phase")
        reason = _enum(
            TerminationReason,
            item["termination_reason"],
            f"{path}.termination_reason",
            "termination reason",
        )
        outcome = _nullable_enum(
            ConclusionOutcome,
            item["conclusion_outcome"],
            f"{path}.conclusion_outcome",
            "conclusion outcome",
        )
        try:
            return ExpectedIncidentResult(
                phase=phase,
                termination_reason=reason,
                conclusion_outcome=outcome,
                root_cause_key=_nullable_string(
                    item["root_cause_key"],
                    f"{path}.root_cause_key",
                ),
                root_cause_summary=_nullable_bounded_text(
                    item["root_cause_summary"],
                    f"{path}.root_cause_summary",
                    max_length=2_000,
                ),
                required_evidence_labels=_string_tuple(
                    item["required_evidence_labels"],
                    f"{path}.required_evidence_labels",
                    max_items=_MAX_EVIDENCE_LABELS,
                ),
                failure_code=_nullable_string(
                    item["failure_code"],
                    f"{path}.failure_code",
                ),
            )
        except ValueError as error:
            raise IncidentDatasetError(path, str(error)) from None


def _strict_object(
    value: object,
    path: str,
    *,
    required: set[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise IncidentDatasetError(path, "expected an object")
    actual = set(value)
    missing = sorted(required - actual)
    unknown = actual - required
    if missing:
        raise IncidentDatasetError(path, f"missing fields: {', '.join(missing)}")
    if unknown:
        raise IncidentDatasetError(path, f"unknown fields are not allowed ({len(unknown)})")
    return value


def _array(
    value: object,
    path: str,
    *,
    max_items: int,
    allow_empty: bool = True,
) -> list[object]:
    if type(value) is not list:
        raise IncidentDatasetError(path, "expected an array")
    if (not allow_empty and not value) or len(value) > max_items:
        raise IncidentDatasetError(path, "array size is outside the allowed range")
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise IncidentDatasetError(path, "expected a string")
    return value


def _nullable_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _bounded_text(value: object, path: str, *, max_length: int) -> str:
    text = _string(value, path)
    if (
        not text
        or text != text.strip()
        or len(text) > max_length
        or any(unicodedata.category(character).startswith("C") for character in text)
    ):
        raise IncidentDatasetError(path, "text is outside the allowed bounds")
    return text


def _nullable_bounded_text(
    value: object,
    path: str,
    *,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, path, max_length=max_length)


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise IncidentDatasetError(path, "expected an integer")
    return value


def _string_tuple(value: object, path: str, *, max_items: int) -> tuple[str, ...]:
    items = _array(value, path, max_items=max_items)
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(items))


def _enum(enum_type, value: object, path: str, label: str):
    text = _string(value, path)
    try:
        return enum_type(text)
    except ValueError:
        raise IncidentDatasetError(path, f"unknown {label}") from None


def _nullable_enum(enum_type, value: object, path: str, label: str):
    if value is None:
        return None
    return _enum(enum_type, value, path, label)


def _utc_datetime(value: object, path: str) -> datetime:
    text = _string(value, path)
    if not _UTC_TIMESTAMP_PATTERN.fullmatch(text):
        raise IncidentDatasetError(path, "expected an absolute UTC timestamp")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise IncidentDatasetError(path, "timestamp is invalid") from None
    return parsed.replace(tzinfo=UTC)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IncidentDatasetError("$", "JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    del value
    raise IncidentDatasetError("$", "non-finite numbers are not allowed")


def _reject_float(value: str) -> None:
    del value
    raise IncidentDatasetError("$", "floating-point numbers are not allowed")


def _require_bounded_depth(value: object) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_DEPTH:
            raise IncidentDatasetError("$", "dataset nesting is too deep")
        if type(current) is dict:
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            stack.extend((item, depth + 1) for item in current)
