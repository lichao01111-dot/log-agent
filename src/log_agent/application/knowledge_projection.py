from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from log_agent.domain.knowledge import (
    ComponentKnowledge,
    DependencyKnowledge,
    ErrorCodeKnowledge,
    FieldKnowledge,
    KnowledgeSnapshot,
    KnownFailurePattern,
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_PATTERN = re.compile(r"[\w.:-]{2,64}", re.UNICODE)
_ALGORITHM_VERSION = "knowledge-lexical-v1"
_MAX_POLICY_ITEMS = 500
_MAX_SIGNAL_IDS = 32
_MAX_SIGNAL_TERMS = 64
_MAX_SIGNAL_TERM_LENGTH = 200
_MAX_SIGNAL_CHARACTERS = 4_000


class ReasoningStage(StrEnum):
    GENERATE_HYPOTHESES = "generate_hypotheses"
    ASSESS_VERIFICATION = "assess_verification"
    GENERATE_CONCLUSION = "generate_conclusion"


class KnowledgeProjectionError(RuntimeError):
    """A fail-closed, sanitized projection failure."""

    def __init__(self, code: str, safe_message: str) -> None:
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


@dataclass(frozen=True, slots=True)
class ProjectionBudget:
    max_components: int
    max_fields: int
    max_error_codes: int
    max_dependencies: int
    max_known_failures: int
    max_total_entities: int
    max_serialized_characters: int

    def __post_init__(self) -> None:
        category_limits = (
            self.max_components,
            self.max_fields,
            self.max_error_codes,
            self.max_dependencies,
            self.max_known_failures,
        )
        if any(type(value) is not int or not 0 <= value <= 100 for value in category_limits):
            raise ValueError("projection category limits must be integers between 0 and 100")
        if type(self.max_total_entities) is not int or not 1 <= self.max_total_entities <= 200:
            raise ValueError("max_total_entities must be an integer between 1 and 200")
        if (
            type(self.max_serialized_characters) is not int
            or not 256 <= self.max_serialized_characters <= 50_000
        ):
            raise ValueError("max_serialized_characters must be an integer between 256 and 50000")


@dataclass(frozen=True, slots=True)
class ProjectionVisibilityPolicy:
    """Trusted allowlists for knowledge that may leave the application boundary."""

    id: str
    revision: str
    scope_ref: str
    knowledge_bundle_id: str
    knowledge_content_hash: str
    budget: ProjectionBudget
    allowed_component_ids: frozenset[str] = frozenset()
    allowed_field_ids: frozenset[str] = frozenset()
    allowed_error_code_ids: frozenset[str] = frozenset()
    allowed_dependency_ids: frozenset[str] = frozenset()
    allowed_known_failure_ids: frozenset[str] = frozenset()
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.id, "projection policy id"),
            (self.scope_ref, "projection policy scope_ref"),
            (self.knowledge_bundle_id, "projection policy knowledge_bundle_id"),
        ):
            _require_id(value, field_name)
        if not isinstance(self.revision, str) or not _REVISION_PATTERN.fullmatch(self.revision):
            raise ValueError("projection policy revision is invalid")
        if not isinstance(self.knowledge_content_hash, str) or not _HASH_PATTERN.fullmatch(
            self.knowledge_content_hash
        ):
            raise ValueError("projection policy knowledge_content_hash is invalid")
        if not isinstance(self.budget, ProjectionBudget):
            raise ValueError("projection policy budget is invalid")

        allowlists = (
            "allowed_component_ids",
            "allowed_field_ids",
            "allowed_error_code_ids",
            "allowed_dependency_ids",
            "allowed_known_failure_ids",
        )
        for field_name in allowlists:
            configured = getattr(self, field_name)
            if not isinstance(configured, (set, frozenset)):
                raise ValueError(f"{field_name} must be a set of identifiers")
            values = frozenset(configured)
            if len(values) > _MAX_POLICY_ITEMS:
                raise ValueError(f"{field_name} exceeds the policy item limit")
            for value in values:
                _require_id(value, field_name)
            object.__setattr__(self, field_name, values)

        canonical = {
            "id": self.id,
            "revision": self.revision,
            "scope_ref": self.scope_ref,
            "knowledge_bundle_id": self.knowledge_bundle_id,
            "knowledge_content_hash": self.knowledge_content_hash,
            "budget": {
                "max_components": self.budget.max_components,
                "max_fields": self.budget.max_fields,
                "max_error_codes": self.budget.max_error_codes,
                "max_dependencies": self.budget.max_dependencies,
                "max_known_failures": self.budget.max_known_failures,
                "max_total_entities": self.budget.max_total_entities,
                "max_serialized_characters": self.budget.max_serialized_characters,
            },
            **{field_name: sorted(getattr(self, field_name)) for field_name in allowlists},
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "policy_hash", hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class TaskSignals:
    """Bounded local signals used only to rank already-visible knowledge."""

    stage: ReasoningStage
    component_ids: tuple[str, ...] = ()
    error_code_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ReasoningStage):
            raise ValueError("task signal stage is invalid")
        for values, field_name in (
            (self.component_ids, "task signal component_ids"),
            (self.error_code_ids, "task signal error_code_ids"),
        ):
            if not isinstance(values, tuple) or len(values) > _MAX_SIGNAL_IDS:
                raise ValueError(f"{field_name} is invalid")
            for value in values:
                _require_id(value, field_name)

        if not isinstance(self.terms, tuple) or len(self.terms) > _MAX_SIGNAL_TERMS:
            raise ValueError("task signal terms are invalid")
        normalized_terms: set[str] = set()
        for term in self.terms:
            _require_text(term, "task signal term", max_length=_MAX_SIGNAL_TERM_LENGTH)
            normalized_terms.add(term.casefold())
        if sum(len(term) for term in normalized_terms) > _MAX_SIGNAL_CHARACTERS:
            raise ValueError("task signal terms exceed the character limit")

        object.__setattr__(self, "component_ids", tuple(sorted(set(self.component_ids))))
        object.__setattr__(self, "error_code_ids", tuple(sorted(set(self.error_code_ids))))
        object.__setattr__(self, "terms", tuple(sorted(normalized_terms)))


@dataclass(frozen=True, slots=True)
class ProjectedComponent:
    id: str
    name: str
    kind: str
    description: str


@dataclass(frozen=True, slots=True)
class ProjectedField:
    id: str
    component_id: str
    semantic_type: str
    description: str


@dataclass(frozen=True, slots=True)
class ProjectedErrorCode:
    id: str
    code: str
    component_id: str
    meaning: str


@dataclass(frozen=True, slots=True)
class ProjectedDependency:
    id: str
    caller_component_id: str
    callee_component_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ProjectedKnownFailure:
    id: str
    title: str
    candidate_causes: tuple[str, ...]
    required_evidence: tuple[str, ...]
    related_error_code_ids: tuple[str, ...]
    related_component_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, init=False)
class KnowledgeProjection:
    """Sealed projection envelope; only model_json may be sent to a model."""

    scope_ref: str
    stage: ReasoningStage
    bundle_id: str
    knowledge_revision: str
    knowledge_content_hash: str
    visibility_policy_id: str
    visibility_policy_revision: str
    visibility_policy_hash: str
    algorithm_version: str
    components: tuple[ProjectedComponent, ...]
    fields: tuple[ProjectedField, ...]
    error_codes: tuple[ProjectedErrorCode, ...]
    dependencies: tuple[ProjectedDependency, ...]
    known_failures: tuple[ProjectedKnownFailure, ...]
    truncated: bool
    model_json: str
    projection_hash: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("knowledge projections must be created by KnowledgeProjector")

    @classmethod
    def _create(
        cls,
        *,
        scope_ref: str,
        stage: ReasoningStage,
        snapshot: KnowledgeSnapshot,
        policy: ProjectionVisibilityPolicy,
        components: tuple[ProjectedComponent, ...],
        fields: tuple[ProjectedField, ...],
        error_codes: tuple[ProjectedErrorCode, ...],
        dependencies: tuple[ProjectedDependency, ...],
        known_failures: tuple[ProjectedKnownFailure, ...],
        truncated: bool,
    ) -> KnowledgeProjection:
        model_json = _serialize_model_data(
            components=components,
            fields=fields,
            error_codes=error_codes,
            dependencies=dependencies,
            known_failures=known_failures,
            truncated=truncated,
        )
        if len(model_json) > policy.budget.max_serialized_characters:
            raise KnowledgeProjectionError(
                "knowledge_projection.budget_exceeded",
                "Knowledge projection exceeds its configured budget.",
            ) from None

        instance = object.__new__(cls)
        values = {
            "scope_ref": scope_ref,
            "stage": stage,
            "bundle_id": snapshot.bundle_id,
            "knowledge_revision": snapshot.revision,
            "knowledge_content_hash": snapshot.content_hash,
            "visibility_policy_id": policy.id,
            "visibility_policy_revision": policy.revision,
            "visibility_policy_hash": policy.policy_hash,
            "algorithm_version": _ALGORITHM_VERSION,
            "components": components,
            "fields": fields,
            "error_codes": error_codes,
            "dependencies": dependencies,
            "known_failures": known_failures,
            "truncated": truncated,
            "model_json": model_json,
            "projection_hash": hashlib.sha256(model_json.encode()).hexdigest(),
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        return instance

    @property
    def entity_count(self) -> int:
        return sum(
            len(items)
            for items in (
                self.components,
                self.fields,
                self.error_codes,
                self.dependencies,
                self.known_failures,
            )
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    id: str
    score: int
    closure: _Selection


@dataclass(frozen=True, slots=True)
class _Selection:
    component_ids: frozenset[str] = frozenset()
    field_ids: frozenset[str] = frozenset()
    error_code_ids: frozenset[str] = frozenset()
    dependency_ids: frozenset[str] = frozenset()
    known_failure_ids: frozenset[str] = frozenset()

    def merge(self, other: _Selection) -> _Selection:
        return _Selection(
            component_ids=self.component_ids | other.component_ids,
            field_ids=self.field_ids | other.field_ids,
            error_code_ids=self.error_code_ids | other.error_code_ids,
            dependency_ids=self.dependency_ids | other.dependency_ids,
            known_failure_ids=self.known_failure_ids | other.known_failure_ids,
        )

    @property
    def count(self) -> int:
        return sum(
            len(items)
            for items in (
                self.component_ids,
                self.field_ids,
                self.error_code_ids,
                self.dependency_ids,
                self.known_failure_ids,
            )
        )


class KnowledgeProjector:
    """Project only explicitly visible, relevant and budgeted knowledge."""

    def __init__(
        self,
        snapshot: KnowledgeSnapshot,
        policies: tuple[ProjectionVisibilityPolicy, ...],
    ) -> None:
        if not isinstance(snapshot, KnowledgeSnapshot):
            raise ValueError("snapshot must be a KnowledgeSnapshot")
        if not isinstance(policies, tuple) or not policies:
            raise ValueError("at least one projection visibility policy is required")
        if any(not isinstance(policy, ProjectionVisibilityPolicy) for policy in policies):
            raise ValueError("projection policies contain an invalid item")
        refs = tuple(policy.scope_ref for policy in policies)
        if len(refs) != len(set(refs)):
            raise ValueError("projection policy scope refs must be unique")

        self._snapshot = snapshot
        self._components = {item.id: item for item in snapshot.components}
        self._fields = {item.id: item for item in snapshot.fields}
        self._error_codes = {item.id: item for item in snapshot.error_codes}
        self._dependencies = {item.id: item for item in snapshot.dependencies}
        self._known_failures = {item.id: item for item in snapshot.known_failures}
        self._policies = {policy.scope_ref: policy for policy in policies}
        for policy in policies:
            self._validate_policy(policy)

    def project(self, *, scope_ref: str, signals: TaskSignals) -> KnowledgeProjection:
        if not isinstance(signals, TaskSignals):
            raise ValueError("signals must be TaskSignals")
        policy = self._policies.get(scope_ref)
        if policy is None:
            raise KnowledgeProjectionError(
                "knowledge_projection.unknown_scope",
                "No knowledge projection policy is configured for this scope.",
            ) from None

        candidates = self._candidates(policy, signals)
        selected = _Selection()
        for candidate in candidates:
            tentative = selected.merge(candidate.closure)
            if not self._within_entity_budgets(tentative, policy.budget):
                continue
            try:
                projection = self._materialize(
                    policy=policy,
                    stage=signals.stage,
                    selected=tentative,
                    truncated=False,
                )
            except KnowledgeProjectionError as error:
                if error.code == "knowledge_projection.budget_exceeded":
                    continue
                raise
            if len(projection.model_json) <= policy.budget.max_serialized_characters:
                selected = tentative

        selected_keys = self._selection_keys(selected)
        candidate_keys = {(item.kind, item.id) for item in candidates}
        truncated = bool(candidate_keys - selected_keys)
        return self._materialize(
            policy=policy,
            stage=signals.stage,
            selected=selected,
            truncated=truncated,
        )

    def _validate_policy(self, policy: ProjectionVisibilityPolicy) -> None:
        if policy.knowledge_bundle_id != self._snapshot.bundle_id:
            raise ValueError("projection policy references the wrong knowledge bundle")
        if policy.knowledge_content_hash != self._snapshot.content_hash:
            raise ValueError("projection policy references the wrong knowledge content hash")
        if policy.scope_ref not in self._snapshot.scope_refs:
            raise ValueError("projection policy scope is absent from the knowledge snapshot")

        checks = (
            (policy.allowed_component_ids, self._components, "component"),
            (policy.allowed_field_ids, self._fields, "field"),
            (policy.allowed_error_code_ids, self._error_codes, "error code"),
            (policy.allowed_dependency_ids, self._dependencies, "dependency"),
            (policy.allowed_known_failure_ids, self._known_failures, "known failure"),
        )
        for allowed, known, kind in checks:
            if not allowed.issubset(known):
                raise ValueError(f"projection policy references an unknown {kind}")

    def _candidates(
        self,
        policy: ProjectionVisibilityPolicy,
        signals: TaskSignals,
    ) -> tuple[_Candidate, ...]:
        tokens = _signal_tokens(signals)
        candidates: list[_Candidate] = []
        kinds = _allowed_kinds(signals.stage)

        if "component" in kinds:
            for item_id in policy.allowed_component_ids:
                item = self._components[item_id]
                score = _component_score(item, signals, tokens)
                if score > 0:
                    candidates.append(
                        _Candidate(
                            "component",
                            item.id,
                            score,
                            _Selection(component_ids=frozenset({item.id})),
                        )
                    )

        if "field" in kinds:
            for item_id in policy.allowed_field_ids:
                item = self._fields[item_id]
                closure = self._field_closure(item, policy)
                score = _field_score(item, signals, tokens)
                if closure is not None and score > 0:
                    candidates.append(_Candidate("field", item.id, score, closure))

        if "error_code" in kinds:
            for item_id in policy.allowed_error_code_ids:
                item = self._error_codes[item_id]
                closure = self._error_code_closure(item, policy)
                score = _error_code_score(item, signals, tokens)
                if closure is not None and score > 0:
                    candidates.append(_Candidate("error_code", item.id, score, closure))

        if "dependency" in kinds:
            for item_id in policy.allowed_dependency_ids:
                item = self._dependencies[item_id]
                closure = self._dependency_closure(item, policy)
                score = _dependency_score(item, signals, tokens)
                if closure is not None and score > 0:
                    candidates.append(_Candidate("dependency", item.id, score, closure))

        if "known_failure" in kinds:
            for item_id in policy.allowed_known_failure_ids:
                item = self._known_failures[item_id]
                closure = self._known_failure_closure(item, policy)
                score = _known_failure_score(item, signals, tokens)
                if closure is not None and score > 0:
                    candidates.append(_Candidate("known_failure", item.id, score, closure))

        priority = {
            "known_failure": 0,
            "error_code": 1,
            "dependency": 2,
            "component": 3,
            "field": 4,
        }
        return tuple(sorted(candidates, key=lambda x: (-x.score, priority[x.kind], x.id)))

    def _field_closure(
        self,
        item: FieldKnowledge,
        policy: ProjectionVisibilityPolicy,
    ) -> _Selection | None:
        if item.component_id not in policy.allowed_component_ids:
            return None
        return _Selection(
            component_ids=frozenset({item.component_id}),
            field_ids=frozenset({item.id}),
        )

    def _error_code_closure(
        self,
        item: ErrorCodeKnowledge,
        policy: ProjectionVisibilityPolicy,
    ) -> _Selection | None:
        if item.component_id not in policy.allowed_component_ids:
            return None
        return _Selection(
            component_ids=frozenset({item.component_id}),
            error_code_ids=frozenset({item.id}),
        )

    def _dependency_closure(
        self,
        item: DependencyKnowledge,
        policy: ProjectionVisibilityPolicy,
    ) -> _Selection | None:
        component_ids = frozenset({item.caller_component_id, item.callee_component_id})
        if not component_ids.issubset(policy.allowed_component_ids):
            return None
        return _Selection(
            component_ids=component_ids,
            dependency_ids=frozenset({item.id}),
        )

    def _known_failure_closure(
        self,
        item: KnownFailurePattern,
        policy: ProjectionVisibilityPolicy,
    ) -> _Selection | None:
        component_ids = set(item.related_component_ids)
        error_code_ids = set(item.related_error_code_ids)
        if not component_ids.issubset(policy.allowed_component_ids):
            return None
        if not error_code_ids.issubset(policy.allowed_error_code_ids):
            return None
        for error_code_id in error_code_ids:
            component_ids.add(self._error_codes[error_code_id].component_id)
        if not component_ids.issubset(policy.allowed_component_ids):
            return None
        return _Selection(
            component_ids=frozenset(component_ids),
            error_code_ids=frozenset(error_code_ids),
            known_failure_ids=frozenset({item.id}),
        )

    def _within_entity_budgets(
        self,
        selected: _Selection,
        budget: ProjectionBudget,
    ) -> bool:
        return (
            len(selected.component_ids) <= budget.max_components
            and len(selected.field_ids) <= budget.max_fields
            and len(selected.error_code_ids) <= budget.max_error_codes
            and len(selected.dependency_ids) <= budget.max_dependencies
            and len(selected.known_failure_ids) <= budget.max_known_failures
            and selected.count <= budget.max_total_entities
        )

    def _materialize(
        self,
        *,
        policy: ProjectionVisibilityPolicy,
        stage: ReasoningStage,
        selected: _Selection,
        truncated: bool,
    ) -> KnowledgeProjection:
        components = tuple(
            ProjectedComponent(item.id, item.name, item.kind.value, item.description)
            for item in sorted(
                (self._components[item_id] for item_id in selected.component_ids),
                key=lambda item: item.id,
            )
        )
        fields = tuple(
            ProjectedField(
                item.id,
                item.component_id,
                item.semantic_type.value,
                item.description,
            )
            for item in sorted(
                (self._fields[item_id] for item_id in selected.field_ids),
                key=lambda item: item.id,
            )
        )
        error_codes = tuple(
            ProjectedErrorCode(item.id, item.code, item.component_id, item.meaning)
            for item in sorted(
                (self._error_codes[item_id] for item_id in selected.error_code_ids),
                key=lambda item: item.id,
            )
        )
        dependencies = tuple(
            ProjectedDependency(
                item.id,
                item.caller_component_id,
                item.callee_component_id,
                item.description,
            )
            for item in sorted(
                (self._dependencies[item_id] for item_id in selected.dependency_ids),
                key=lambda item: item.id,
            )
        )
        known_failures = tuple(
            ProjectedKnownFailure(
                item.id,
                item.title,
                item.candidate_causes,
                item.required_evidence,
                item.related_error_code_ids,
                item.related_component_ids,
            )
            for item in sorted(
                (self._known_failures[item_id] for item_id in selected.known_failure_ids),
                key=lambda item: item.id,
            )
        )
        return KnowledgeProjection._create(
            scope_ref=policy.scope_ref,
            stage=stage,
            snapshot=self._snapshot,
            policy=policy,
            components=components,
            fields=fields,
            error_codes=error_codes,
            dependencies=dependencies,
            known_failures=known_failures,
            truncated=truncated,
        )

    @staticmethod
    def _selection_keys(selected: _Selection) -> set[tuple[str, str]]:
        return {
            *(("component", item_id) for item_id in selected.component_ids),
            *(("field", item_id) for item_id in selected.field_ids),
            *(("error_code", item_id) for item_id in selected.error_code_ids),
            *(("dependency", item_id) for item_id in selected.dependency_ids),
            *(("known_failure", item_id) for item_id in selected.known_failure_ids),
        }


def _allowed_kinds(stage: ReasoningStage) -> frozenset[str]:
    if stage is ReasoningStage.GENERATE_HYPOTHESES:
        return frozenset({"component", "error_code", "dependency", "known_failure"})
    if stage is ReasoningStage.ASSESS_VERIFICATION:
        return frozenset({"component", "field", "error_code", "dependency", "known_failure"})
    return frozenset({"component", "field", "error_code", "dependency"})


def _signal_tokens(signals: TaskSignals) -> frozenset[str]:
    tokens: set[str] = set()
    for term in signals.terms:
        tokens.update(match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(term))
    return frozenset(tokens)


def _text_score(tokens: frozenset[str], *values: str) -> int:
    searchable = " ".join(values).casefold()
    return sum(token in searchable for token in tokens)


def _component_score(
    item: ComponentKnowledge,
    signals: TaskSignals,
    tokens: frozenset[str],
) -> int:
    return (100 if item.id in signals.component_ids else 0) + _text_score(
        tokens,
        item.id,
        item.name,
        item.kind.value,
        item.description,
    )


def _field_score(
    item: FieldKnowledge,
    signals: TaskSignals,
    tokens: frozenset[str],
) -> int:
    return (20 if item.component_id in signals.component_ids else 0) + _text_score(
        tokens,
        item.id,
        item.component_id,
        item.semantic_type.value,
        item.description,
    )


def _error_code_score(
    item: ErrorCodeKnowledge,
    signals: TaskSignals,
    tokens: frozenset[str],
) -> int:
    return (
        (100 if item.id in signals.error_code_ids else 0)
        + (20 if item.component_id in signals.component_ids else 0)
        + _text_score(tokens, item.id, item.code, item.component_id, item.meaning)
    )


def _dependency_score(
    item: DependencyKnowledge,
    signals: TaskSignals,
    tokens: frozenset[str],
) -> int:
    related = {item.caller_component_id, item.callee_component_id}
    return (50 if related & set(signals.component_ids) else 0) + _text_score(
        tokens,
        item.id,
        item.caller_component_id,
        item.callee_component_id,
        item.description,
    )


def _known_failure_score(
    item: KnownFailurePattern,
    signals: TaskSignals,
    tokens: frozenset[str],
) -> int:
    relation_score = 0
    if set(item.related_component_ids) & set(signals.component_ids):
        relation_score += 60
    if set(item.related_error_code_ids) & set(signals.error_code_ids):
        relation_score += 80
    return relation_score + _text_score(
        tokens,
        item.id,
        item.title,
        *item.candidate_causes,
        *item.required_evidence,
        *item.related_component_ids,
        *item.related_error_code_ids,
    )


def _serialize_model_data(
    *,
    components: tuple[ProjectedComponent, ...],
    fields: tuple[ProjectedField, ...],
    error_codes: tuple[ProjectedErrorCode, ...],
    dependencies: tuple[ProjectedDependency, ...],
    known_failures: tuple[ProjectedKnownFailure, ...],
    truncated: bool,
) -> str:
    data = {
        "trust": "untrusted_reference_data",
        "truncated_by_budget": truncated,
        "components": [
            {
                "id": item.id,
                "name": item.name,
                "kind": item.kind,
                "description": item.description,
            }
            for item in components
        ],
        "fields": [
            {
                "id": item.id,
                "component_id": item.component_id,
                "semantic_type": item.semantic_type,
                "description": item.description,
            }
            for item in fields
        ],
        "error_codes": [
            {
                "id": item.id,
                "code": item.code,
                "component_id": item.component_id,
                "meaning": item.meaning,
            }
            for item in error_codes
        ],
        "dependencies": [
            {
                "id": item.id,
                "caller_component_id": item.caller_component_id,
                "callee_component_id": item.callee_component_id,
                "description": item.description,
            }
            for item in dependencies
        ],
        "known_failures": [
            {
                "id": item.id,
                "title": item.title,
                "candidate_status": "unverified_reference",
                "candidate_causes": list(item.candidate_causes),
                "required_evidence": list(item.required_evidence),
                "related_error_code_ids": list(item.related_error_code_ids),
                "related_component_ids": list(item.related_component_ids),
            }
            for item in known_failures
        ],
    }
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
