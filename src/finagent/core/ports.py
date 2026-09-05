"""Dependency inversion ports used by the application service."""

from __future__ import annotations

from typing import Protocol

from finagent.contracts.api import AnalysisRequest, Sector
from finagent.contracts.mcp import (
    DatasetCatalog,
    EventResult,
    ObservationResult,
    ResolutionResult,
)
from finagent.core.models import DraftAnalysis, EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy


class DataGateway(Protocol):
    """Read-only sector data port implemented through MCP."""

    async def get_catalog(self, sector: Sector) -> DatasetCatalog:
        """Return available entities and metrics for a sector."""
        ...

    async def resolve_companies(self, sector: Sector, query: str) -> ResolutionResult:
        """Resolve explicit company mentions in a question."""
        ...

    async def query_observations(
        self,
        sector: Sector,
        entity_ids: list[str],
        metric_keys: list[str],
        latest_only: bool,
    ) -> ObservationResult:
        """Retrieve structured observations."""
        ...

    async def query_events(
        self,
        sector: Sector,
        entity_ids: list[str],
        event_kinds: list[str],
        latest_only: bool,
    ) -> EventResult:
        """Retrieve dated operating signals."""
        ...


class LlmGateway(Protocol):
    """Structured model operations required by the orchestrator."""

    async def plan(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        catalog: DatasetCatalog,
        entity_ids: list[str],
    ) -> RetrievalPlan:
        """Build a retrieval plan constrained by available data."""
        ...

    async def synthesize(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
    ) -> DraftAnalysis:
        """Produce a source-linked analytical draft."""
        ...

    async def repair(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
        draft: DraftAnalysis,
        issues: list[str],
    ) -> DraftAnalysis:
        """Repair one invalid draft without retrieving new evidence."""
        ...
