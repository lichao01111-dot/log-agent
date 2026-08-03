from __future__ import annotations

import asyncio
import math
from typing import Never

from log_agent.application.ports import (
    LogSearchPort,
    PortError,
    PortProtocolError,
    ReasoningPort,
    SearchRequest,
    SearchResult,
    VerificationAssessment,
)
from log_agent.application.query_security import QueryPolicyError, SafeQueryPipeline
from log_agent.domain.models import (
    Conclusion,
    ConclusionOutcome,
    Hypothesis,
    HypothesisStatus,
    Investigation,
    OperationKind,
    QueryKind,
    QueryRecord,
    VerificationDecision,
)
from log_agent.domain.state_machine import (
    AssessVerification,
    Command,
    ConclusionGenerated,
    Event,
    ExecuteQuery,
    GenerateConclusion,
    GenerateHypotheses,
    HypothesesGenerated,
    OperationFailed,
    QuerySucceeded,
    VerificationAssessed,
)


class InvalidCommand(ValueError):
    """Raised before I/O when a command does not belong to the supplied state."""


class CommandExecutor:
    """Translate one current domain command into one domain event."""

    def __init__(
        self,
        log_search: LogSearchPort,
        reasoning: ReasoningPort,
        query_pipeline: SafeQueryPipeline,
        *,
        search_timeout_seconds: float = 30.0,
        reasoning_timeout_seconds: float = 30.0,
    ) -> None:
        if not math.isfinite(search_timeout_seconds) or search_timeout_seconds <= 0:
            raise ValueError("search_timeout_seconds must be positive")
        if not math.isfinite(reasoning_timeout_seconds) or reasoning_timeout_seconds <= 0:
            raise ValueError("reasoning_timeout_seconds must be positive")
        self._log_search = log_search
        self._reasoning = reasoning
        self._query_pipeline = query_pipeline
        self._search_timeout_seconds = search_timeout_seconds
        self._reasoning_timeout_seconds = reasoning_timeout_seconds

    async def execute(self, state: Investigation, command: Command) -> Event:
        """Execute one command without changing the investigation state."""
        self._require_current_command(state, command)

        if isinstance(command, ExecuteQuery):
            timeout = self._search_timeout_seconds
            capability = "log_search"
        elif isinstance(command, (GenerateHypotheses, AssessVerification, GenerateConclusion)):
            timeout = self._reasoning_timeout_seconds
            capability = "reasoning"
        else:  # pragma: no cover - the Command union is exhaustive
            raise InvalidCommand(f"unsupported command: {type(command).__name__}")

        try:
            async with asyncio.timeout(timeout):
                return await self._dispatch(state, command)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._failure(
                command,
                code=f"{capability}.timeout",
                message=(
                    "Log search timed out."
                    if capability == "log_search"
                    else "Reasoning operation timed out."
                ),
            )
        except PortError as error:
            return self._failure(command, code=error.code, message=error.safe_message)
        except QueryPolicyError as error:
            return self._failure(command, code=error.code, message=error.safe_message)

    async def _dispatch(self, state: Investigation, command: Command) -> Event:
        if isinstance(command, ExecuteQuery):
            return await self._execute_query(state, command)
        if isinstance(command, GenerateHypotheses):
            return await self._generate_hypotheses(state, command)
        if isinstance(command, AssessVerification):
            return await self._assess_verification(state, command)
        if isinstance(command, GenerateConclusion):
            return await self._generate_conclusion(state, command)
        raise InvalidCommand(f"unsupported command: {type(command).__name__}")

    async def _execute_query(
        self,
        state: Investigation,
        command: ExecuteQuery,
    ) -> QuerySucceeded:
        query_id = self._query_id(command)
        authorized_query = self._query_pipeline.prepare(
            scope_ref=state.request.scope_ref,
            investigation_range=state.request.time_range,
            intent=command.intent,
        )
        request = SearchRequest(
            operation_id=command.command_id,
            query_id=query_id,
            authorized_query=authorized_query,
        )
        result = await self._log_search.search(request)
        self._validate_search_result(state, request, result)
        evidence_ids = tuple(item.id for item in result.evidence)
        record = QueryRecord(
            id=query_id,
            intent=command.intent,
            result_count=result.result_count,
            summary=result.summary,
            truncated=result.truncated,
            evidence_ids=evidence_ids,
        )
        return QuerySucceeded(
            command_id=command.command_id,
            record=record,
            facts=result.facts,
            evidence=result.evidence,
        )

    async def _generate_hypotheses(
        self,
        state: Investigation,
        command: GenerateHypotheses,
    ) -> HypothesesGenerated:
        hypotheses = await self._reasoning.generate_hypotheses(state.request, state.memory)
        if not isinstance(hypotheses, tuple) or not all(
            isinstance(item, Hypothesis) for item in hypotheses
        ):
            self._raise_reasoning_protocol_error()

        hypothesis_ids = tuple(item.id for item in hypotheses)
        known_ids = {item.id for item in state.memory.hypotheses}
        if (
            len(hypotheses) > 3
            or len(hypothesis_ids) != len(set(hypothesis_ids))
            or bool(known_ids & set(hypothesis_ids))
            or any(item.status is not HypothesisStatus.PROPOSED for item in hypotheses)
        ):
            self._raise_reasoning_protocol_error()
        return HypothesesGenerated(command_id=command.command_id, hypotheses=hypotheses)

    async def _assess_verification(
        self,
        state: Investigation,
        command: AssessVerification,
    ) -> VerificationAssessed:
        hypothesis = self._find_hypothesis(state, command.hypothesis_id)
        query = self._find_query(state, command.query_id)
        assessment = await self._reasoning.assess_verification(
            state.request,
            state.memory,
            hypothesis,
            query,
        )
        self._validate_assessment(state, hypothesis, query, assessment)
        return VerificationAssessed(
            command_id=command.command_id,
            hypothesis=assessment.hypothesis,
            decision=assessment.decision,
            next_query=assessment.next_query,
        )

    async def _generate_conclusion(
        self,
        state: Investigation,
        command: GenerateConclusion,
    ) -> ConclusionGenerated:
        conclusion = await self._reasoning.generate_conclusion(
            state.request,
            state.memory,
            command.outcome,
            command.termination_reason,
            command.root_cause_hypothesis_id,
        )
        self._validate_conclusion(state, command, conclusion)
        return ConclusionGenerated(command_id=command.command_id, conclusion=conclusion)

    def _require_current_command(self, state: Investigation, command: Command) -> None:
        if state.phase.is_terminal:
            raise InvalidCommand("cannot execute a command for a terminal investigation")
        pending = state.pending_operation
        if pending is None:
            raise InvalidCommand("the investigation has no pending operation")
        if pending.command_id != command.command_id:
            raise InvalidCommand("command does not match the pending operation")

        if isinstance(command, ExecuteQuery):
            valid = (
                pending.kind is OperationKind.EXECUTE_QUERY
                and pending.query_intent == command.intent
                and self._intent_is_within_request(state, command)
                and all(query.id != self._query_id(command) for query in state.memory.queries)
            )
        elif isinstance(command, GenerateHypotheses):
            valid = pending.kind is OperationKind.GENERATE_HYPOTHESES
        elif isinstance(command, AssessVerification):
            valid = (
                pending.kind is OperationKind.ASSESS_VERIFICATION
                and pending.hypothesis_id == command.hypothesis_id
                and pending.query_id == command.query_id
            )
        elif isinstance(command, GenerateConclusion):
            valid = (
                pending.kind is OperationKind.GENERATE_CONCLUSION
                and pending.conclusion_outcome is command.outcome
                and pending.termination_reason is command.termination_reason
                and pending.hypothesis_id == command.root_cause_hypothesis_id
            )
        else:
            valid = False

        if not valid:
            raise InvalidCommand("command payload does not match the pending operation")

    def _validate_search_result(
        self,
        state: Investigation,
        request: SearchRequest,
        result: SearchResult,
    ) -> None:
        if not isinstance(result, SearchResult):
            self._raise_search_protocol_error()
        if any(item.query_id != request.query_id for item in result.evidence):
            self._raise_search_protocol_error()
        if result.result_count > request.authorized_query.plan.result_limit:
            self._raise_search_protocol_error()

        known_evidence_ids = {item.id for item in state.memory.evidence}
        known_fact_ids = {item.id for item in state.memory.facts}
        if known_evidence_ids & {item.id for item in result.evidence}:
            self._raise_search_protocol_error()
        if known_fact_ids & {item.id for item in result.facts}:
            self._raise_search_protocol_error()

    def _validate_assessment(
        self,
        state: Investigation,
        original: Hypothesis,
        query: QueryRecord,
        assessment: VerificationAssessment,
    ) -> None:
        if not isinstance(assessment, VerificationAssessment):
            self._raise_reasoning_protocol_error()
        assessed = assessment.hypothesis
        if (
            not isinstance(assessment.decision, VerificationDecision)
            or not isinstance(assessed.status, HypothesisStatus)
            or assessed.id != original.id
            or assessed.statement != original.statement
            or assessed.verification_goal != original.verification_goal
            or assessed.status is HypothesisStatus.PROPOSED
        ):
            self._raise_reasoning_protocol_error()

        if (
            assessment.decision is VerificationDecision.CONTINUE
            and assessed.status is not HypothesisStatus.TESTING
        ):
            self._raise_reasoning_protocol_error()
        if (
            assessed.status is HypothesisStatus.SUPPORTED
            and assessment.decision is not VerificationDecision.CONCLUDE
        ):
            self._raise_reasoning_protocol_error()

        existing_references = set(original.supporting_evidence_ids)
        existing_references.update(original.contradicting_evidence_ids)
        returned_references = set(assessed.supporting_evidence_ids)
        returned_references.update(assessed.contradicting_evidence_ids)
        if not existing_references.issubset(returned_references):
            self._raise_reasoning_protocol_error()
        if not returned_references.issubset(existing_references | set(query.evidence_ids)):
            self._raise_reasoning_protocol_error()

        next_query = assessment.next_query
        if next_query is not None:
            requested = state.request.time_range
            if (
                next_query.kind is not QueryKind.VERIFY
                or next_query.hypothesis_id != original.id
                or next_query.time_range.start < requested.start
                or next_query.time_range.end > requested.end
            ):
                self._raise_reasoning_protocol_error()

    def _validate_conclusion(
        self,
        state: Investigation,
        command: GenerateConclusion,
        conclusion: Conclusion,
    ) -> None:
        if not isinstance(conclusion, Conclusion):
            self._raise_reasoning_protocol_error()
        if (
            conclusion.outcome is not command.outcome
            or conclusion.termination_reason is not command.termination_reason
            or conclusion.root_cause_hypothesis_id != command.root_cause_hypothesis_id
        ):
            self._raise_reasoning_protocol_error()

        known_evidence = {item.id for item in state.memory.evidence}
        if not set(conclusion.evidence_ids).issubset(known_evidence):
            self._raise_reasoning_protocol_error()
        if conclusion.outcome is ConclusionOutcome.CONCLUSIVE:
            root = self._find_hypothesis(state, conclusion.root_cause_hypothesis_id)
            if root.status is not HypothesisStatus.SUPPORTED or not set(
                conclusion.evidence_ids
            ).issubset(root.supporting_evidence_ids):
                self._raise_reasoning_protocol_error()

    @staticmethod
    def _query_id(command: ExecuteQuery) -> str:
        return f"{command.command_id}:query"

    @staticmethod
    def _intent_is_within_request(state: Investigation, command: ExecuteQuery) -> bool:
        requested = state.request.time_range
        actual = command.intent.time_range
        return actual.start >= requested.start and actual.end <= requested.end

    @staticmethod
    def _find_hypothesis(state: Investigation, hypothesis_id: str | None) -> Hypothesis:
        for hypothesis in state.memory.hypotheses:
            if hypothesis.id == hypothesis_id:
                return hypothesis
        raise InvalidCommand(f"unknown hypothesis: {hypothesis_id}")

    @staticmethod
    def _find_query(state: Investigation, query_id: str | None) -> QueryRecord:
        for query in state.memory.queries:
            if query.id == query_id:
                return query
        raise InvalidCommand(f"unknown query: {query_id}")

    @staticmethod
    def _failure(command: Command, *, code: str, message: str) -> OperationFailed:
        return OperationFailed(command_id=command.command_id, code=code, message=message)

    @staticmethod
    def _raise_search_protocol_error() -> Never:
        raise PortProtocolError(
            "log_search.protocol_error",
            "Log search returned an invalid response.",
        )

    @staticmethod
    def _raise_reasoning_protocol_error() -> Never:
        raise PortProtocolError(
            "reasoning.protocol_error",
            "Reasoning capability returned an invalid response.",
        )
