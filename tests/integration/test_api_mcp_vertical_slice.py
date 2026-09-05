"""Complete API-to-MCP-to-SQLite vertical-slice tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from finagent.api.app import create_app
from finagent.contracts.api import AnalysisRequest
from finagent.contracts.mcp import DatasetCatalog
from finagent.core.analysis_service import AnalysisService
from finagent.core.models import DraftAnalysis, EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy, PersonaPolicyStore
from finagent.gateways.llm import FakeLlmGateway
from finagent.gateways.mcp_client import McpDataGateway, StreamableHttpToolCaller


class CountingLlmGateway(FakeLlmGateway):
    """Record whether an out-of-scope request reaches model operations."""

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
        """Count and delegate a planning call."""
        self.plan_calls += 1
        return await super().plan(request, policy, catalog, entity_ids)

    async def synthesize(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
    ) -> DraftAnalysis:
        """Count and delegate a synthesis call."""
        self.synthesis_calls += 1
        return await super().synthesize(request, policy, evidence)


async def _post(app, payload: dict[str, str]) -> httpx.Response:
    """POST one analysis request to an in-process FastAPI adapter."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/analyses", json=payload)


@pytest.mark.integration
def test_api_uses_real_mcp_for_latest_headcount(real_mcp_url: str) -> None:
    """Return the latest source-linked signal through the complete runtime path."""
    llm = CountingLlmGateway()
    service = AnalysisService(
        McpDataGateway(StreamableHttpToolCaller(real_mcp_url)),
        llm,
        PersonaPolicyStore.load(),
    )
    response = asyncio.run(
        _post(
            create_app(service),
            {
                "query": "What is the most recent headcount for Example Systems, Inc.?",
                "persona": "equity_analyst",
                "sector": "tech",
            },
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    # The fixture's quality caveat is surfaced as a limitation; every metric the
    # equity persona requires was retrieved, so evidence coverage is complete.
    assert body["evidence_status"] == "sufficient"
    assert body["coverage"]["missing_metrics"] == []
    assert body["data_as_of"] == "2025-02-15"
    assert any("950 employees" in item["text"] for item in body["findings"])
    assert body["sources"][0]["published_at"] == "2025-02-15"
    assert body["limitations"]
    assert llm.plan_calls == 1
    assert llm.synthesis_calls == 1


@pytest.mark.integration
def test_api_out_of_scope_stops_before_llm(real_mcp_url: str) -> None:
    """Prove that an unknown company cannot trigger model synthesis."""
    llm = CountingLlmGateway()
    service = AnalysisService(
        McpDataGateway(StreamableHttpToolCaller(real_mcp_url)),
        llm,
        PersonaPolicyStore.load(),
    )
    response = asyncio.run(
        _post(
            create_app(service),
            {
                "query": "What do you think about SpaceX?",
                "persona": "pe_analyst",
                "sector": "tech",
            },
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "out_of_scope"
    assert body["evidence_status"] == "none"
    assert body["findings"] == []
    assert body["sources"] == []
    assert llm.plan_calls == 0
    assert llm.synthesis_calls == 0
