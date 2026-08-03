from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from log_agent.application.knowledge_projection import ReasoningStage
from log_agent.application.ports import PortError


class ModelFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    OTHER = "other"


class ReasoningTextCategory(StrEnum):
    USER_QUESTION = "user_question"
    FACT = "fact"
    QUERY_GOAL = "query_goal"
    QUERY_SUMMARY = "query_summary"
    HYPOTHESIS = "hypothesis"
    VERIFICATION_GOAL = "verification_goal"


@dataclass(frozen=True, slots=True)
class ModelTraceContext:
    knowledge_bundle_id: str
    knowledge_revision: str
    knowledge_content_hash: str
    visibility_policy_id: str
    visibility_policy_revision: str
    visibility_policy_hash: str
    projection_algorithm_version: str
    projection_hash: str

    def __post_init__(self) -> None:
        for value in (
            self.knowledge_bundle_id,
            self.knowledge_revision,
            self.knowledge_content_hash,
            self.visibility_policy_id,
            self.visibility_policy_revision,
            self.visibility_policy_hash,
            self.projection_algorithm_version,
            self.projection_hash,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("model trace context contains an invalid identifier")


@dataclass(frozen=True, slots=True)
class StructuredModelRequest:
    """Provider-neutral request with trusted instructions and data kept separate.

    trace_context is local observability metadata. A provider client must never encode
    it into model-visible instructions, input, metadata, or tool arguments.
    """

    task: ReasoningStage
    prompt_version: str
    instructions: str
    input_json: str
    response_schema_json: str
    trace_context: ModelTraceContext
    max_output_tokens: int
    max_output_bytes: int
    repair_attempt: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.task, ReasoningStage):
            raise ValueError("structured model task is invalid")
        for value, field_name in (
            (self.prompt_version, "prompt_version"),
            (self.instructions, "instructions"),
            (self.input_json, "input_json"),
            (self.response_schema_json, "response_schema_json"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"structured model {field_name} must not be blank")
        if type(self.repair_attempt) is not int or not 0 <= self.repair_attempt <= 1:
            raise ValueError("structured model repair_attempt must be zero or one")
        if type(self.max_output_tokens) is not int or not 64 <= self.max_output_tokens <= 10_000:
            raise ValueError("structured model max_output_tokens is invalid")
        if type(self.max_output_bytes) is not int or not 1_000 <= self.max_output_bytes <= 50_000:
            raise ValueError("structured model max_output_bytes is invalid")
        if not isinstance(self.trace_context, ModelTraceContext):
            raise ValueError("structured model trace_context is invalid")


@dataclass(frozen=True, slots=True)
class StructuredModelResponse:
    output_json: str | None
    finish_reason: ModelFinishReason
    refused: bool = False

    def __post_init__(self) -> None:
        if self.output_json is not None and not isinstance(self.output_json, str):
            raise ValueError("structured model output_json is invalid")
        if not isinstance(self.finish_reason, ModelFinishReason):
            raise ValueError("structured model finish_reason is invalid")
        if type(self.refused) is not bool:
            raise ValueError("structured model refused must be a boolean")


class StructuredModelClient(Protocol):
    """Low-level capability implemented later by one concrete model provider.

    Implementations must apply max_output_tokens at generation time and enforce
    max_output_bytes while receiving a streamed response, before buffering it fully.
    Cancellation must propagate unchanged. Expected provider failures must be mapped
    to a sanitized PortError; unexpected programming errors must propagate.
    """

    async def generate(self, request: StructuredModelRequest) -> StructuredModelResponse: ...


class ReasoningTextSanitizationError(PortError):
    """Expected, sanitized policy rejection from a reasoning text sanitizer."""


class ReasoningTextSanitizer(Protocol):
    """Trusted policy boundary required before dynamic text may leave the process.

    A policy rejection must use ReasoningTextSanitizationError with no sensitive text
    in its code or safe_message. Unexpected implementation defects must propagate.
    """

    def sanitize(self, text: str, *, category: ReasoningTextCategory) -> str: ...
