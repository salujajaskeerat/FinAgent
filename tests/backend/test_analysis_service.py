"""Tests for bounded application behavior and domain fallbacks."""

import asyncio

import pytest

from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisStatus,
    EvidenceStatus,
    Persona,
    Sector,
)
from finagent.contracts.entity_resolution import (
    EntityMatch,
    EntityResolution,
    EntityResolutionReason,
    EntityResolutionStatus,
)
from finagent.contracts.mcp import (
    CatalogEntity,
    DatasetCatalog,
    EventResult,
    ObservationResult,
)
from finagent.core.analysis_service import AnalysisService
from finagent.core.models import DraftAnalysis, EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy, PersonaPolicyStore
from finagent.gateways.entity_resolver import FakeEntityResolver
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

    def __init__(self, unresolved: list[str] | None = None) -> None:
        super().__init__(unresolved=unresolved)
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


class _CountingLlm(FakeLlmGateway):
    """Count downstream model operations for scope-exit assertions."""

    def __init__(self) -> None:
        self.plan_calls = 0
        self.synthesis_calls = 0

    async def plan(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        catalog: DatasetCatalog,
        entity_ids: list[str],
    ) -> RetrievalPlan:
        """Count one planning request."""
        self.plan_calls += 1
        return await super().plan(request, policy, catalog, entity_ids)

    async def synthesize(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
    ) -> DraftAnalysis:
        """Count one synthesis request."""
        self.synthesis_calls += 1
        return await super().synthesize(request, policy, evidence)


class _FailingResolver:
    """Simulate an unavailable or malformed resolver boundary."""

    async def resolve(
        self,
        query: str,
        sector: Sector,
        candidates: list[CatalogEntity],
    ) -> EntityResolution:
        """Raise without returning a result."""
        del query, sector, candidates
        raise RuntimeError("test resolver failure")


def _resolution(
    status: EntityResolutionStatus,
    *,
    entity_id: str = "cmp_example",
    confidence: float = 0.9,
    mention: str = "enterprise software example",
) -> EntityResolution:
    """Build a valid resolver result for an application test."""
    matches = (
        [
            EntityMatch(
                entity_id=entity_id,
                mention=mention,
                confidence=confidence,
            )
        ]
        if status is EntityResolutionStatus.MATCHED
        else []
    )
    reasons = {
        EntityResolutionStatus.MATCHED: EntityResolutionReason.SEMANTIC_REFERENCE,
        EntityResolutionStatus.AMBIGUOUS: EntityResolutionReason.AMBIGUOUS,
        EntityResolutionStatus.NO_MATCH: EntityResolutionReason.NOT_IN_CATALOG,
        EntityResolutionStatus.BROAD_QUERY: EntityResolutionReason.BROAD_QUESTION,
    }
    return EntityResolution(status=status, matches=matches, reason_code=reasons[status])


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
    assert data.observation_query is not None
    assert data.event_query is not None
    entity_ids, metric_keys = data.observation_query
    assert entity_ids == ["cmp_example"]
    assert "model_generated_sql" not in metric_keys
    assert "google_search" not in data.event_query[1]
    # The model's proposal seeds the plan; the persona policy completes it.
    assert metric_keys[0] == "revenue"
    assert {"operating_income", "operating_margin", "enterprise_value"} <= set(
        metric_keys
    )
    assert data.event_query == (["cmp_example"], ["headcount", "guidance"])


def test_required_metrics_are_retrieved_even_if_model_omits_them() -> None:
    """PE retrieval must include leverage inputs regardless of the model's plan."""
    data = _RecordingDataGateway()
    service = AnalysisService(data, _UntrustedPlanner(), PersonaPolicyStore.load())

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="Analyze revenue.", persona=Persona.PE, sector=Sector.TECH
            )
        )
    )

    assert result.status is AnalysisStatus.ANSWERED
    assert data.observation_query is not None
    assert {"free_cash_flow", "total_debt", "cash_and_equivalents"} <= set(
        data.observation_query[1]
    )
    assert "restructuring" in data.event_query[1]


def test_mutual_fund_plan_always_includes_the_sector_benchmark() -> None:
    data = _RecordingDataGateway()
    service = AnalysisService(data, _UntrustedPlanner(), PersonaPolicyStore.load())

    asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="Analyze revenue.",
                persona=Persona.MUTUAL_FUND,
                sector=Sector.TECH,
            )
        )
    )

    assert data.observation_query is not None
    assert data.observation_query[0] == ["cmp_example", "benchmark:tech"]


def test_evidence_status_reflects_required_metric_coverage() -> None:
    """Caveats become limitations; only missing required metrics downgrade status."""

    class _RevenueOnlyGateway(StubDataGateway):
        async def query_observations(
            self,
            sector: Sector,
            entity_ids: list[str],
            metric_keys: list[str],
            latest_only: bool,
        ) -> ObservationResult:
            result = await super().query_observations(
                sector, entity_ids, ["revenue"], latest_only
            )
            result.warnings.append("fixture caveat: verify before investment use")
            return result

    request = AnalysisRequest(
        query="How do the fundamentals look?",
        persona=Persona.EQUITY,
        sector=Sector.TECH,
    )
    partial = asyncio.run(_service(_RevenueOnlyGateway()).analyze(request))
    complete = asyncio.run(_service(StubDataGateway()).analyze(request))

    assert partial.evidence_status is EvidenceStatus.PARTIAL
    assert partial.coverage is not None
    assert partial.coverage.missing_metrics == ["operating_income", "operating_margin"]
    assert partial.limitations[0].startswith("Required metrics not available")
    assert complete.evidence_status is EvidenceStatus.SUFFICIENT
    assert complete.coverage is not None
    assert complete.coverage.missing_metrics == []


def test_semantic_fallback_selects_one_catalog_company() -> None:
    """Proceed normally after one high-confidence allowlisted fallback match."""
    data = _RecordingDataGateway(unresolved=["enterprise software example"])
    llm = _CountingLlm()
    service = AnalysisService(
        data,
        llm,
        PersonaPolicyStore.load(),
        entity_resolver=FakeEntityResolver(_resolution(EntityResolutionStatus.MATCHED)),
    )

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="Tell me about the enterprise software example.",
                persona=Persona.EQUITY,
                sector=Sector.TECH,
            )
        )
    )

    assert result.status is AnalysisStatus.ANSWERED
    assert data.observation_query is not None
    assert data.observation_query[0] == ["cmp_example"]
    assert llm.plan_calls == 1
    assert llm.synthesis_calls == 1


@pytest.mark.parametrize(
    "resolver",
    [
        FakeEntityResolver(_resolution(EntityResolutionStatus.NO_MATCH)),
        FakeEntityResolver(_resolution(EntityResolutionStatus.AMBIGUOUS)),
        FakeEntityResolver(
            _resolution(EntityResolutionStatus.MATCHED, confidence=0.84)
        ),
        FakeEntityResolver(
            _resolution(EntityResolutionStatus.MATCHED, entity_id="cmp_invented")
        ),
        FakeEntityResolver(
            _resolution(
                EntityResolutionStatus.MATCHED,
                mention="text absent from the user query",
            )
        ),
        _FailingResolver(),
    ],
)
def test_untrusted_fallback_exits_before_planning_or_retrieval(
    resolver: object,
) -> None:
    """Never broaden an unresolved explicit company into a sector-wide query."""
    data = _RecordingDataGateway(unresolved=["unknown"])
    llm = _CountingLlm()
    service = AnalysisService(
        data,
        llm,
        PersonaPolicyStore.load(),
        entity_resolver=resolver,
    )

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="What do you think about unknown?",
                persona=Persona.PE,
                sector=Sector.TECH,
            )
        )
    )

    assert result.status is AnalysisStatus.OUT_OF_SCOPE
    assert result.evidence_status is EvidenceStatus.NONE
    assert result.findings == []
    assert result.sources == []
    assert data.observation_query is None
    assert data.event_query is None
    assert llm.plan_calls == 0
    assert llm.synthesis_calls == 0
