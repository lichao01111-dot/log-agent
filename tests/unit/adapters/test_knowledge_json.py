import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from log_agent.adapters.knowledge_json import KnowledgeConfigError, load_knowledge_json
from log_agent.domain.knowledge import ComponentKind, FieldKnowledge, FieldSemanticType

EXAMPLE_PATH = Path(__file__).parents[3] / "config" / "knowledge" / "checkout-domain.json"


def valid_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "bundle_id": "checkout-domain",
        "revision": "2026-08-02.1",
        "system_id": "checkout",
        "scope_refs": ["checkout-prod"],
        "components": [
            {
                "id": "checkout-api",
                "name": "Checkout API",
                "kind": "internal",
                "description": "Coordinates checkout requests.",
            },
            {
                "id": "payment-service",
                "name": "Payment Service",
                "kind": "dependency",
                "description": "Authorizes payments.",
            },
        ],
        "fields": [
            {
                "id": "checkout.error-code",
                "component_id": "checkout-api",
                "semantic_type": "error_code",
                "description": "Stable application error code.",
            }
        ],
        "error_codes": [
            {
                "id": "checkout.payment-timeout",
                "code": "PAYMENT_TIMEOUT",
                "component_id": "checkout-api",
                "meaning": "Payment did not respond before the timeout.",
            }
        ],
        "dependencies": [
            {
                "id": "checkout-to-payment",
                "caller_component_id": "checkout-api",
                "callee_component_id": "payment-service",
                "description": "Checkout calls payment.",
            }
        ],
        "known_failures": [
            {
                "id": "payment-timeout",
                "title": "Payment dependency timeout",
                "candidate_causes": ["Payment latency exceeded the timeout budget."],
                "required_evidence": ["Payment timeouts preceded checkout failures."],
                "related_error_code_ids": ["checkout.payment-timeout"],
                "related_component_ids": ["checkout-api", "payment-service"],
            }
        ],
    }


def write_document(tmp_path: Path, document: object, name: str = "knowledge.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_example_loads_as_an_immutable_versioned_snapshot() -> None:
    snapshot = load_knowledge_json(EXAMPLE_PATH)

    assert snapshot.schema_version == 1
    assert snapshot.bundle_id == "checkout-domain"
    assert snapshot.revision == "2026-08-02.1"
    assert snapshot.system_id == "checkout"
    assert snapshot.scope_refs == ("checkout-prod",)
    assert tuple(item.id for item in snapshot.components) == (
        "checkout-api",
        "payment-service",
    )
    assert snapshot.components[0].kind is ComponentKind.INTERNAL
    assert snapshot.fields[0].semantic_type is FieldSemanticType.ERROR_CODE
    assert snapshot.error_codes[0].id == "checkout.payment-timeout"
    assert snapshot.error_codes[0].code == "PAYMENT_TIMEOUT"
    assert snapshot.known_failures[0].related_error_code_ids == ("checkout.payment-timeout",)
    assert len(snapshot.content_hash) == 64

    with pytest.raises(FrozenInstanceError):
        snapshot.system_id = "changed"  # type: ignore[misc]


def test_semantic_field_model_has_no_physical_name_or_query_capability() -> None:
    assert tuple(item.name for item in fields(FieldKnowledge)) == (
        "id",
        "component_id",
        "semantic_type",
        "description",
    )


def test_content_hash_is_independent_of_json_object_key_order(tmp_path: Path) -> None:
    document = valid_document()
    reversed_root = dict(reversed(tuple(document.items())))

    first = load_knowledge_json(write_document(tmp_path, document, "first.json"))
    second = load_knowledge_json(write_document(tmp_path, reversed_root, "second.json"))

    assert first == second
    assert first.content_hash == second.content_hash


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(KnowledgeConfigError, match="duplicate key"):
        load_knowledge_json(path)


@pytest.mark.parametrize("version", [2, "1", True])
def test_unsupported_or_wrongly_typed_schema_version_is_rejected(
    tmp_path: Path,
    version: object,
) -> None:
    document = valid_document()
    document["schema_version"] = version

    with pytest.raises(KnowledgeConfigError, match="schema_version"):
        load_knowledge_json(write_document(tmp_path, document))


@pytest.mark.parametrize("value", ["1.0", "1e309"])
def test_floating_point_numbers_are_rejected_during_parsing(
    tmp_path: Path,
    value: str,
) -> None:
    path = tmp_path / "float.json"
    path.write_text(f'{{"schema_version": {value}}}', encoding="utf-8")

    with pytest.raises(KnowledgeConfigError, match="floating-point"):
        load_knowledge_json(path)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "index",
        "sourcetype",
        "raw_spl",
        "query_text",
        "allowed_operations",
        "allowed_template_ids",
        "max_result_limit",
        "renderer",
        "tool_name",
    ],
)
def test_execution_and_privilege_fields_are_rejected(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    document = valid_document()
    pattern = document["known_failures"][0]  # type: ignore[index]
    pattern[forbidden_field] = "attempted privilege expansion"  # type: ignore[index]

    with pytest.raises(KnowledgeConfigError, match=rf"unknown fields: {forbidden_field}"):
        load_knowledge_json(write_document(tmp_path, document))


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    document = valid_document()
    del document["system_id"]

    with pytest.raises(KnowledgeConfigError, match="missing fields: system_id"):
        load_knowledge_json(write_document(tmp_path, document))


def test_untrusted_unknown_key_is_not_echoed_into_error_text(tmp_path: Path) -> None:
    document = valid_document()
    malicious_key = "line-break\n" + ("x" * 1_000)
    document[malicious_key] = "value"

    with pytest.raises(KnowledgeConfigError) as caught:
        load_knowledge_json(write_document(tmp_path, document))

    assert malicious_key not in str(caught.value)
    assert "<invalid-key>" in str(caught.value)


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(KnowledgeConfigError, match="valid UTF-8"):
        load_knowledge_json(path)


def test_unicode_control_characters_are_rejected_as_domain_text(tmp_path: Path) -> None:
    document = valid_document()
    document["components"][0]["name"] = "Checkout\ud800"  # type: ignore[index]

    with pytest.raises(KnowledgeConfigError, match="component name is invalid"):
        load_knowledge_json(write_document(tmp_path, document))


def test_oversized_document_is_rejected_before_parsing(tmp_path: Path) -> None:
    path = write_document(tmp_path, valid_document())

    with pytest.raises(KnowledgeConfigError, match="byte limit"):
        load_knowledge_json(path, max_bytes=10)


def test_non_regular_configuration_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeConfigError, match="regular file"):
        load_knowledge_json(tmp_path)


def test_excessive_nesting_is_rejected(tmp_path: Path) -> None:
    document: object = "leaf"
    for _ in range(20):
        document = {"nested": document}

    with pytest.raises(KnowledgeConfigError, match="nesting is too deep"):
        load_knowledge_json(write_document(tmp_path, document))


def test_duplicate_component_ids_are_rejected(tmp_path: Path) -> None:
    document = valid_document()
    components = document["components"]  # type: ignore[assignment]
    components.append(dict(components[0]))  # type: ignore[union-attr, index]

    with pytest.raises(KnowledgeConfigError, match="components must not contain duplicates"):
        load_knowledge_json(write_document(tmp_path, document))


def test_dangling_component_reference_is_rejected(tmp_path: Path) -> None:
    document = valid_document()
    document["fields"][0]["component_id"] = "missing"  # type: ignore[index]

    with pytest.raises(KnowledgeConfigError, match="field references an unknown component"):
        load_knowledge_json(write_document(tmp_path, document))


def test_dangling_error_code_reference_is_rejected(tmp_path: Path) -> None:
    document = valid_document()
    document["known_failures"][0]["related_error_code_ids"] = [  # type: ignore[index]
        "unknown"
    ]

    with pytest.raises(KnowledgeConfigError, match="unknown error code"):
        load_knowledge_json(write_document(tmp_path, document))


def test_same_code_value_can_be_namespaced_by_component(tmp_path: Path) -> None:
    document = valid_document()
    document["error_codes"].append(  # type: ignore[union-attr]
        {
            "id": "payment.payment-timeout",
            "code": "PAYMENT_TIMEOUT",
            "component_id": "payment-service",
            "meaning": "Payment recorded an internal timeout.",
        }
    )

    snapshot = load_knowledge_json(write_document(tmp_path, document))

    assert tuple(item.id for item in snapshot.error_codes) == (
        "checkout.payment-timeout",
        "payment.payment-timeout",
    )


def test_duplicate_code_within_one_component_is_rejected(tmp_path: Path) -> None:
    document = valid_document()
    document["error_codes"].append(  # type: ignore[union-attr]
        {
            "id": "checkout.duplicate-timeout",
            "code": "PAYMENT_TIMEOUT",
            "component_id": "checkout-api",
            "meaning": "Duplicate definition.",
        }
    )

    with pytest.raises(KnowledgeConfigError, match="component/code pairs"):
        load_knowledge_json(write_document(tmp_path, document))


def test_self_dependency_is_rejected(tmp_path: Path) -> None:
    document = valid_document()
    dependency = document["dependencies"][0]  # type: ignore[index]
    dependency["callee_component_id"] = dependency["caller_component_id"]  # type: ignore[index]

    with pytest.raises(KnowledgeConfigError, match="must not point a component to itself"):
        load_knowledge_json(write_document(tmp_path, document))


def test_non_finite_numbers_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(KnowledgeConfigError, match="non-finite"):
        load_knowledge_json(path)
