from __future__ import annotations

from dataclasses import dataclass

from log_agent.application.knowledge_projection import (
    KnowledgeProjector,
    ProjectionVisibilityPolicy,
)
from log_agent.application.query_security import SafeQueryPipeline, ScopePolicyRegistry
from log_agent.domain.knowledge import KnowledgeSnapshot


class ConfigurationCompatibilityError(ValueError):
    """Raised when independently trusted configuration snapshots do not line up."""


@dataclass(frozen=True, slots=True)
class AgentConfiguration:
    """Bind knowledge to security policy without allowing knowledge to grant access."""

    knowledge: KnowledgeSnapshot
    scope_policies: ScopePolicyRegistry

    def __post_init__(self) -> None:
        if not isinstance(self.knowledge, KnowledgeSnapshot):
            raise ValueError("knowledge must be a KnowledgeSnapshot")
        if not isinstance(self.scope_policies, ScopePolicyRegistry):
            raise ValueError("scope_policies must be a ScopePolicyRegistry")
        if self.knowledge.scope_refs != self.scope_policies.refs:
            raise ConfigurationCompatibilityError(
                "knowledge scopes must exactly match configured security policy scopes"
            )

    def build_query_pipeline(self) -> SafeQueryPipeline:
        return SafeQueryPipeline(self.scope_policies)

    def build_knowledge_projector(
        self,
        policies: tuple[ProjectionVisibilityPolicy, ...],
    ) -> KnowledgeProjector:
        if not isinstance(policies, tuple) or any(
            not isinstance(policy, ProjectionVisibilityPolicy) for policy in policies
        ):
            raise ValueError("projection policies must be a tuple of visibility policies")
        projection_refs = tuple(sorted(policy.scope_ref for policy in policies))
        if projection_refs != self.knowledge.scope_refs:
            raise ConfigurationCompatibilityError(
                "knowledge and projection policy scopes must exactly match"
            )
        return KnowledgeProjector(self.knowledge, policies)
