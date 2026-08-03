from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from log_agent.domain.knowledge import (
    ComponentKind,
    ComponentKnowledge,
    DependencyKnowledge,
    ErrorCodeKnowledge,
    FieldKnowledge,
    FieldSemanticType,
    KnowledgeSnapshot,
    KnownFailurePattern,
)

_DEFAULT_MAX_BYTES = 1_000_000
_MAX_DEPTH = 16
_MAX_SCOPES = 32
_MAX_COMPONENTS = 100
_MAX_FIELDS = 500
_MAX_ERROR_CODES = 1_000
_MAX_DEPENDENCIES = 200
_MAX_PATTERNS = 200
_MAX_PATTERN_ITEMS = 20


class KnowledgeConfigError(ValueError):
    """A deterministic, location-aware domain knowledge configuration error."""

    def __init__(self, location: str, reason: str) -> None:
        self.location = location
        self.reason = reason
        super().__init__(f"{location}: {reason}")


def load_knowledge_json(
    path: str | Path,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> KnowledgeSnapshot:
    """Load one all-or-nothing, immutable knowledge snapshot from strict JSON."""

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    config_path = Path(path)
    if not stat.S_ISREG(config_path.stat().st_mode):
        raise KnowledgeConfigError("$", "configuration must be a regular file")
    with config_path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise KnowledgeConfigError("$", "configuration exceeds the byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise KnowledgeConfigError("$", "configuration must be valid UTF-8") from None

    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
            parse_float=_reject_float,
        )
    except KnowledgeConfigError:
        raise
    except (ValueError, RecursionError):
        raise KnowledgeConfigError("$", "configuration is not valid JSON") from None

    _require_bounded_depth(document)
    canonical = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical).hexdigest()
    return _KnowledgeDecoder(content_hash).decode(document)


class _KnowledgeDecoder:
    def __init__(self, content_hash: str) -> None:
        self._content_hash = content_hash

    def decode(self, value: object) -> KnowledgeSnapshot:
        root = _strict_object(
            value,
            "$",
            required={
                "schema_version",
                "bundle_id",
                "revision",
                "system_id",
                "scope_refs",
                "components",
                "fields",
                "error_codes",
                "dependencies",
                "known_failures",
            },
        )
        schema_version = _integer(root["schema_version"], "$.schema_version")
        if schema_version != 1:
            raise KnowledgeConfigError("$.schema_version", "unsupported schema version")

        try:
            return KnowledgeSnapshot(
                schema_version=schema_version,
                bundle_id=_string(root["bundle_id"], "$.bundle_id"),
                revision=_string(root["revision"], "$.revision"),
                system_id=_string(root["system_id"], "$.system_id"),
                scope_refs=_string_tuple(
                    root["scope_refs"],
                    "$.scope_refs",
                    max_items=_MAX_SCOPES,
                    allow_empty=False,
                ),
                components=self._components(root["components"]),
                fields=self._fields(root["fields"]),
                error_codes=self._error_codes(root["error_codes"]),
                dependencies=self._dependencies(root["dependencies"]),
                known_failures=self._known_failures(root["known_failures"]),
                content_hash=self._content_hash,
            )
        except KnowledgeConfigError:
            raise
        except ValueError as error:
            raise KnowledgeConfigError("$", str(error)) from None

    def _components(self, value: object) -> tuple[ComponentKnowledge, ...]:
        items = _array(
            value,
            "$.components",
            max_items=_MAX_COMPONENTS,
            allow_empty=False,
        )
        return tuple(
            self._component(item, f"$.components[{index}]") for index, item in enumerate(items)
        )

    def _component(self, value: object, path: str) -> ComponentKnowledge:
        item = _strict_object(
            value,
            path,
            required={"id", "name", "kind", "description"},
        )
        try:
            kind = ComponentKind(_string(item["kind"], f"{path}.kind"))
        except ValueError:
            raise KnowledgeConfigError(f"{path}.kind", "unknown component kind") from None
        try:
            return ComponentKnowledge(
                id=_string(item["id"], f"{path}.id"),
                name=_string(item["name"], f"{path}.name"),
                kind=kind,
                description=_string(item["description"], f"{path}.description"),
            )
        except ValueError as error:
            raise KnowledgeConfigError(path, str(error)) from None

    def _fields(self, value: object) -> tuple[FieldKnowledge, ...]:
        items = _array(value, "$.fields", max_items=_MAX_FIELDS)
        return tuple(self._field(item, f"$.fields[{index}]") for index, item in enumerate(items))

    def _field(self, value: object, path: str) -> FieldKnowledge:
        item = _strict_object(
            value,
            path,
            required={"id", "component_id", "semantic_type", "description"},
        )
        try:
            semantic_type = FieldSemanticType(
                _string(item["semantic_type"], f"{path}.semantic_type")
            )
        except ValueError:
            raise KnowledgeConfigError(
                f"{path}.semantic_type",
                "unknown field semantic type",
            ) from None
        try:
            return FieldKnowledge(
                id=_string(item["id"], f"{path}.id"),
                component_id=_string(item["component_id"], f"{path}.component_id"),
                semantic_type=semantic_type,
                description=_string(item["description"], f"{path}.description"),
            )
        except ValueError as error:
            raise KnowledgeConfigError(path, str(error)) from None

    def _error_codes(self, value: object) -> tuple[ErrorCodeKnowledge, ...]:
        items = _array(value, "$.error_codes", max_items=_MAX_ERROR_CODES)
        return tuple(
            self._error_code(item, f"$.error_codes[{index}]") for index, item in enumerate(items)
        )

    def _error_code(self, value: object, path: str) -> ErrorCodeKnowledge:
        item = _strict_object(
            value,
            path,
            required={"id", "code", "component_id", "meaning"},
        )
        try:
            return ErrorCodeKnowledge(
                id=_string(item["id"], f"{path}.id"),
                code=_string(item["code"], f"{path}.code"),
                component_id=_string(item["component_id"], f"{path}.component_id"),
                meaning=_string(item["meaning"], f"{path}.meaning"),
            )
        except ValueError as error:
            raise KnowledgeConfigError(path, str(error)) from None

    def _dependencies(self, value: object) -> tuple[DependencyKnowledge, ...]:
        items = _array(value, "$.dependencies", max_items=_MAX_DEPENDENCIES)
        return tuple(
            self._dependency(item, f"$.dependencies[{index}]") for index, item in enumerate(items)
        )

    def _dependency(self, value: object, path: str) -> DependencyKnowledge:
        item = _strict_object(
            value,
            path,
            required={"id", "caller_component_id", "callee_component_id", "description"},
        )
        try:
            return DependencyKnowledge(
                id=_string(item["id"], f"{path}.id"),
                caller_component_id=_string(
                    item["caller_component_id"],
                    f"{path}.caller_component_id",
                ),
                callee_component_id=_string(
                    item["callee_component_id"],
                    f"{path}.callee_component_id",
                ),
                description=_string(item["description"], f"{path}.description"),
            )
        except ValueError as error:
            raise KnowledgeConfigError(path, str(error)) from None

    def _known_failures(self, value: object) -> tuple[KnownFailurePattern, ...]:
        items = _array(value, "$.known_failures", max_items=_MAX_PATTERNS)
        return tuple(
            self._known_failure(item, f"$.known_failures[{index}]")
            for index, item in enumerate(items)
        )

    def _known_failure(self, value: object, path: str) -> KnownFailurePattern:
        item = _strict_object(
            value,
            path,
            required={
                "id",
                "title",
                "candidate_causes",
                "required_evidence",
                "related_error_code_ids",
                "related_component_ids",
            },
        )
        try:
            return KnownFailurePattern(
                id=_string(item["id"], f"{path}.id"),
                title=_string(item["title"], f"{path}.title"),
                candidate_causes=_string_tuple(
                    item["candidate_causes"],
                    f"{path}.candidate_causes",
                    max_items=_MAX_PATTERN_ITEMS,
                    allow_empty=False,
                ),
                required_evidence=_string_tuple(
                    item["required_evidence"],
                    f"{path}.required_evidence",
                    max_items=_MAX_PATTERN_ITEMS,
                    allow_empty=False,
                ),
                related_error_code_ids=_string_tuple(
                    item["related_error_code_ids"],
                    f"{path}.related_error_code_ids",
                    max_items=_MAX_PATTERN_ITEMS,
                ),
                related_component_ids=_string_tuple(
                    item["related_component_ids"],
                    f"{path}.related_component_ids",
                    max_items=_MAX_PATTERN_ITEMS,
                    allow_empty=False,
                ),
            )
        except ValueError as error:
            raise KnowledgeConfigError(path, str(error)) from None


def _strict_object(
    value: object,
    path: str,
    *,
    required: set[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise KnowledgeConfigError(path, "expected an object")
    actual = set(value)
    missing = sorted(required - actual)
    unknown = actual - required
    if missing:
        raise KnowledgeConfigError(path, f"missing fields: {', '.join(missing)}")
    if unknown:
        raise KnowledgeConfigError(path, f"unknown fields: {_format_unknown_fields(unknown)}")
    return value


def _array(
    value: object,
    path: str,
    *,
    max_items: int,
    allow_empty: bool = True,
) -> list[object]:
    if type(value) is not list:
        raise KnowledgeConfigError(path, "expected an array")
    if (not allow_empty and not value) or len(value) > max_items:
        raise KnowledgeConfigError(path, "array size is outside the allowed range")
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise KnowledgeConfigError(path, "expected a string")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise KnowledgeConfigError(path, "expected an integer")
    return value


def _string_tuple(
    value: object,
    path: str,
    *,
    max_items: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    items = _array(value, path, max_items=max_items, allow_empty=allow_empty)
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(items))


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KnowledgeConfigError("$", "JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> None:
    del value
    raise KnowledgeConfigError("$", "non-finite numbers are not allowed")


def _reject_float(value: str) -> None:
    del value
    raise KnowledgeConfigError("$", "floating-point numbers are not allowed")


def _format_unknown_fields(names: set[str]) -> str:
    labels = sorted(
        name
        if len(name) <= 64
        and name.isascii()
        and all(character.isalnum() or character in "_.-" for character in name)
        else "<invalid-key>"
        for name in names
    )
    displayed = labels[:5]
    if len(labels) > len(displayed):
        displayed.append("...")
    return ", ".join(displayed)


def _require_bounded_depth(value: object) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_DEPTH:
            raise KnowledgeConfigError("$", "configuration nesting is too deep")
        if type(current) is dict:
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            stack.extend((item, depth + 1) for item in current)
