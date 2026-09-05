"""Tests for bounded application behavior and domain fallbacks."""

import asyncio

from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisStatus,
    EvidenceStatus,
    Persona,
    Sector,
)
from finagent.core.analysis_service import AnalysisService
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.gateways.llm import FakeLlmGateway
from tests.backend.support import StubDataGateway


def _service(data: StubDataGateway) -> AnalysisService:
    return AnalysisService(data, FakeLlmGateway(), PersonaPolicyStore.load())


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
