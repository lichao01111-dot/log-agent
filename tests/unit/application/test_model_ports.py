from dataclasses import replace

import pytest

import log_agent.application as application
from log_agent.application.knowledge_projection import ReasoningStage
from log_agent.application.model_ports import (
    ModelFinishReason,
    ModelTraceContext,
    ReasoningTextSanitizationError,
    StructuredModelRequest,
    StructuredModelResponse,
)
from log_agent.application.ports import PortError


def make_trace() -> ModelTraceContext:
    return ModelTraceContext(
        knowledge_bundle_id="checkout-domain",
        knowledge_revision="2026-08-02.1",
        knowledge_content_hash="a" * 64,
        visibility_policy_id="checkout-model-visibility",
        visibility_policy_revision="2026-08-02.1",
        visibility_policy_hash="b" * 64,
        projection_algorithm_version="knowledge-lexical-v1",
        projection_hash="c" * 64,
    )


def make_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task=ReasoningStage.GENERATE_HYPOTHESES,
        prompt_version="structured-reasoning-v1",
        instructions="Return only the requested structured draft.",
        input_json='{"untrusted":true}',
        response_schema_json='{"type":"object"}',
        trace_context=make_trace(),
        max_output_tokens=2_000,
        max_output_bytes=20_000,
    )


def test_structured_request_keeps_local_trace_separate_from_model_data() -> None:
    request = make_request()

    assert request.trace_context.knowledge_bundle_id not in request.instructions
    assert request.trace_context.knowledge_bundle_id not in request.input_json
    assert request.trace_context.projection_hash not in request.response_schema_json


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_output_tokens", 63),
        ("max_output_tokens", True),
        ("max_output_bytes", 999),
        ("max_output_bytes", 50_001),
        ("repair_attempt", 2),
    ],
)
def test_structured_request_rejects_invalid_resource_or_retry_limits(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(make_request(), **{field_name: value})


def test_structured_response_requires_typed_finish_reason_and_refusal() -> None:
    with pytest.raises(ValueError, match="finish_reason"):
        StructuredModelResponse("{}", "stop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refused"):
        StructuredModelResponse("{}", ModelFinishReason.STOP, refused=1)  # type: ignore[arg-type]


def test_sanitization_rejection_is_a_safe_port_error_type() -> None:
    error = ReasoningTextSanitizationError(
        "reasoning.sensitive_context",
        "Reasoning context was rejected by policy.",
    )

    assert isinstance(error, PortError)
    assert error.safe_message == "Reasoning context was rejected by policy."


def test_application_package_exports_model_boundary_api() -> None:
    assert application.StructuredModelRequest is StructuredModelRequest
    assert application.StructuredModelResponse is StructuredModelResponse
    assert application.ModelTraceContext is ModelTraceContext
    assert application.ReasoningTextSanitizationError is ReasoningTextSanitizationError
