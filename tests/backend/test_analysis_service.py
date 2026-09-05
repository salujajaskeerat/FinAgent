"""Tests for bounded application behavior and domain fallbacks."""

import asyncio

from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisStatus,
    EvidenceStatus,
    Persona,
    Sector,
)
from finagent.contracts.mcp import DatasetCatalog, EventResult, ObservationResult
from finagent.core.analysis_service import AnalysisService
from finagent.core.models import RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy, PersonaPolicyStore
from finagent.gateways.llm import FakeLlmGateway
from tests.backend.support import StubDataGateway


def _service(data: StubDataGateway) -> AnalysisService:
    return AnalysisService(data, FakeLlmGateway(), PersonaPolicyStore.load())


class _UntrustedPlanner(FakeLlmGateway):
    """Return valid and invented identifiers like a compromised model might."""

    async def plan(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        catalog: DatasetCatalog,
        entity_ids: list[str],
    ) -> RetrievalPlan:
        """Propose a plan containing values outside the MCP catalog."""
        del request, policy, catalog, entity_ids
        return RetrievalPlan(
            entity_ids=["cmp_example", "cmp_cross_sector"],
            metric_keys=["revenue", "model_generated_sql"],
            event_kinds=["headcount", "google_search"],
        )


class _RecordingDataGateway(StubDataGateway):
    """Capture the plan after the application allowlist is applied."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_query: tuple[list[str], list[str]] | None = None
        self.event_query: tuple[list[str], list[str]] | None = None

    async def query_observations(
        self,
        sector: Sector,
        entity_ids: list[str],
        metric_keys: list[str],
        latest_only: bool,
    ) -> ObservationResult:
        """Record and delegate an observation query."""
        self.observation_query = (entity_ids, metric_keys)
        return await super().query_observations(
            sector, entity_ids, metric_keys, latest_only
        )

    async def query_events(
        self,
        sector: Sector,
        entity_ids: list[str],
        event_kinds: list[str],
        latest_only: bool,
    ) -> EventResult:
        """Record and delegate an event query."""
        self.event_query = (entity_ids, event_kinds)
        return await super().query_events(sector, entity_ids, event_kinds, latest_only)


def test_answer_is_source_linked_and_persona_specific() -> None:
    request = AnalysisRequest(
        query="How do the fundamentals look?",
        persona=Persona.EQUITY,
        sector=Sector.TECH,
    )

    result = asyncio.run(_service(StubDataGateway()).analyze(request))

    assert result.status is AnalysisStatus.ANSWERED
    assert result.evidence_status is EvidenceStatus.SUFFICIENT
    assert "Equity Analyst view" in result.answer_markdown
    assert result.findings[0].source_ids == ["src_fixture"]
    assert result.companies[0].company_id == "cmp_example"


def test_unknown_company_is_a_domain_outcome_without_synthesis() -> None:
    request = AnalysisRequest(
        query="What do you think about Unknown Corp?",
        persona=Persona.PE,
        sector=Sector.TECH,
    )

    result = asyncio.run(
        _service(StubDataGateway(unresolved=["Unknown Corp"])).analyze(request)
    )

    assert result.status is AnalysisStatus.OUT_OF_SCOPE
    assert result.evidence_status is EvidenceStatus.NONE
    assert result.sources == []
    assert result.limitations


def test_empty_retrieval_returns_honest_insufficient_data() -> None:
    request = AnalysisRequest(
        query="Is this sector attractive?",
        persona=Persona.MUTUAL_FUND,
        sector=Sector.TECH,
    )

    result = asyncio.run(_service(StubDataGateway(empty=True)).analyze(request))

    assert result.status is AnalysisStatus.INSUFFICIENT_DATA
    assert result.evidence_status is EvidenceStatus.NONE


def test_catalog_exposes_all_personas_and_selected_sector_companies() -> None:
    result = asyncio.run(_service(StubDataGateway()).catalog(Sector.TECH))

    assert {item.value for item in result.personas} == set(Persona)
    assert result.companies[0].ticker == "EXM"
    assert result.dataset_version == "fixture-v1"


def test_model_plan_is_allowlisted_before_any_mcp_query() -> None:
    """Prevent invented model identifiers from crossing the MCP boundary."""
    data = _RecordingDataGateway()
    service = AnalysisService(data, _UntrustedPlanner(), PersonaPolicyStore.load())

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="Analyze revenue.",
                persona=Persona.EQUITY,
                sector=Sector.TECH,
            )
        )
    )

    assert result.status is AnalysisStatus.ANSWERED
    assert data.observation_query == (["cmp_example"], ["revenue"])
    assert data.event_query == (["cmp_example"], ["headcount"])
