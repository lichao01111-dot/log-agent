from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ComponentKind(StrEnum):
    INTERNAL = "internal"
    DEPENDENCY = "dependency"


class FieldSemanticType(StrEnum):
    TIMESTAMP = "timestamp"
    SEVERITY = "severity"
    ERROR_CODE = "error_code"
    TRACE_ID = "trace_id"
    MESSAGE = "message"
    DURATION = "duration"


@dataclass(frozen=True, slots=True)
class ComponentKnowledge:
    id: str
    name: str
    kind: ComponentKind
    description: str

    def __post_init__(self) -> None:
        _require_id(self.id, "component id")
        _require_text(self.name, "component name", max_length=100)
        if not isinstance(self.kind, ComponentKind):
            raise ValueError("component kind is invalid")
        _require_text(self.description, "component description", max_length=500)


@dataclass(frozen=True, slots=True)
class FieldKnowledge:
    """Semantic field information; it contains no physical name or query permission."""

    id: str
    component_id: str
    semantic_type: FieldSemanticType
    description: str

    def __post_init__(self) -> None:
        _require_id(self.id, "field id")
        _require_id(self.component_id, "field component_id")
        if not isinstance(self.semantic_type, FieldSemanticType):
            raise ValueError("field semantic_type is invalid")
        _require_text(self.description, "field description", max_length=500)


@dataclass(frozen=True, slots=True)
class ErrorCodeKnowledge:
    id: str
    code: str
    component_id: str
    meaning: str

    def __post_init__(self) -> None:
        _require_id(self.id, "error code id")
        if not isinstance(self.code, str) or not _ERROR_CODE_PATTERN.fullmatch(self.code):
            raise ValueError("error code is invalid")
        _require_id(self.component_id, "error code component_id")
        _require_text(self.meaning, "error code meaning", max_length=500)


@dataclass(frozen=True, slots=True)
class DependencyKnowledge:
    id: str
    caller_component_id: str
    callee_component_id: str
    description: str

    def __post_init__(self) -> None:
        _require_id(self.id, "dependency id")
        _require_id(self.caller_component_id, "dependency caller_component_id")
        _require_id(self.callee_component_id, "dependency callee_component_id")
        if self.caller_component_id == self.callee_component_id:
            raise ValueError("dependency must not point a component to itself")
        _require_text(self.description, "dependency description", max_length=500)


@dataclass(frozen=True, slots=True)
class KnownFailurePattern:
    """A candidate reasoning aid, never an executable rule or confirmed fact."""

    id: str
    title: str
    candidate_causes: tuple[str, ...]
    required_evidence: tuple[str, ...]
    related_error_code_ids: tuple[str, ...]
    related_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.id, "known failure id")
        _require_text(self.title, "known failure title", max_length=200)
        _require_text_tuple(
            self.candidate_causes,
            "known failure candidate_causes",
            max_items=20,
            max_text_length=500,
        )
        _require_text_tuple(
            self.required_evidence,
            "known failure required_evidence",
            max_items=20,
            max_text_length=500,
        )
        _require_id_tuple(
            self.related_error_code_ids,
            "known failure related_error_code_ids",
            allow_empty=True,
        )
        _require_id_tuple(
            self.related_component_ids,
            "known failure related_component_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    """One immutable, versioned snapshot of non-executable domain knowledge."""

    schema_version: int
    bundle_id: str
    revision: str
    system_id: str
    scope_refs: tuple[str, ...]
    components: tuple[ComponentKnowledge, ...]
    fields: tuple[FieldKnowledge, ...]
    error_codes: tuple[ErrorCodeKnowledge, ...]
    dependencies: tuple[DependencyKnowledge, ...]
    known_failures: tuple[KnownFailurePattern, ...]
    content_hash: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("knowledge schema_version must be 1")
        _require_id(self.bundle_id, "knowledge bundle_id")
        if not isinstance(self.revision, str) or not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError("knowledge revision is invalid")
        _require_id(self.system_id, "knowledge system_id")
        _require_id_tuple(self.scope_refs, "knowledge scope_refs", allow_empty=False)
        _require_typed_unique_tuple(
            self.components,
            ComponentKnowledge,
            "knowledge components",
            key=lambda item: item.id,
            allow_empty=False,
        )
        _require_typed_unique_tuple(
            self.fields,
            FieldKnowledge,
            "knowledge fields",
            key=lambda item: item.id,
        )
        _require_typed_unique_tuple(
            self.error_codes,
            ErrorCodeKnowledge,
            "knowledge error_codes",
            key=lambda item: item.id,
        )
        _require_typed_unique_tuple(
            self.dependencies,
            DependencyKnowledge,
            "knowledge dependencies",
            key=lambda item: item.id,
        )
        _require_typed_unique_tuple(
            self.known_failures,
            KnownFailurePattern,
            "knowledge known_failures",
            key=lambda item: item.id,
        )
        if not isinstance(self.content_hash, str) or not _CONTENT_HASH_PATTERN.fullmatch(
            self.content_hash
        ):
            raise ValueError("knowledge content_hash is invalid")

        self._validate_references()
        object.__setattr__(self, "scope_refs", tuple(sorted(self.scope_refs)))
        object.__setattr__(self, "components", tuple(sorted(self.components, key=lambda x: x.id)))
        object.__setattr__(self, "fields", tuple(sorted(self.fields, key=lambda x: x.id)))
        object.__setattr__(
            self,
            "error_codes",
            tuple(sorted(self.error_codes, key=lambda x: x.id)),
        )
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                sorted(
                    self.dependencies,
                    key=lambda x: x.id,
                )
            ),
        )
        object.__setattr__(
            self,
            "known_failures",
            tuple(sorted(self.known_failures, key=lambda x: x.id)),
        )

    def _validate_references(self) -> None:
        component_ids = {item.id for item in self.components}
        error_code_ids = {item.id for item in self.error_codes}
        error_code_locations = tuple((item.component_id, item.code) for item in self.error_codes)
        if len(error_code_locations) != len(set(error_code_locations)):
            raise ValueError("error code component/code pairs must not contain duplicates")
        dependency_edges = tuple(
            (item.caller_component_id, item.callee_component_id) for item in self.dependencies
        )
        if len(dependency_edges) != len(set(dependency_edges)):
            raise ValueError("dependency caller/callee pairs must not contain duplicates")

        for field in self.fields:
            if field.component_id not in component_ids:
                raise ValueError("field references an unknown component")
        for error_code in self.error_codes:
            if error_code.component_id not in component_ids:
                raise ValueError("error code references an unknown component")
        for dependency in self.dependencies:
            if (
                dependency.caller_component_id not in component_ids
                or dependency.callee_component_id not in component_ids
            ):
                raise ValueError("dependency references an unknown component")
        for pattern in self.known_failures:
            if not set(pattern.related_component_ids).issubset(component_ids):
                raise ValueError("known failure references an unknown component")
            if not set(pattern.related_error_code_ids).issubset(error_code_ids):
                raise ValueError("known failure references an unknown error code")


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


def _require_text_tuple(
    values: object,
    field_name: str,
    *,
    max_items: int,
    max_text_length: int,
) -> None:
    if not isinstance(values, tuple) or not values or len(values) > max_items:
        raise ValueError(f"{field_name} is invalid")
    for value in values:
        _require_text(value, field_name, max_length=max_text_length)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _require_id_tuple(values: object, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise ValueError(f"{field_name} is invalid")
    for value in values:
        _require_id(value, field_name)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _require_typed_unique_tuple(
    values: object,
    item_type: type,
    field_name: str,
    *,
    key: Callable[[Any], Hashable],
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise ValueError(f"{field_name} is invalid")
    if any(not isinstance(item, item_type) for item in values):
        raise ValueError(f"{field_name} contains an invalid item")
    keys = tuple(key(item) for item in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must not contain duplicates")
