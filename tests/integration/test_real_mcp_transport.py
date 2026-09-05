"""Real SDK calls over MCP Streamable HTTP."""

import asyncio

import pytest

from finagent.contracts.api import Sector
from finagent.contracts.mcp import EntityKind
from finagent.gateways.mcp_client import (
    McpDataGateway,
    McpToolError,
    StreamableHttpToolCaller,
)


@pytest.mark.integration
def test_all_tools_cross_real_mcp_http_transport(real_mcp_url: str) -> None:
    """Exercise every public tool through the official MCP client."""

    async def exercise() -> None:
        gateway = McpDataGateway(StreamableHttpToolCaller(real_mcp_url))
        catalog = await gateway.get_catalog(Sector.TECH)
        company = next(
            item for item in catalog.entities if item.kind is EntityKind.COMPANY
        )

        resolution = await gateway.resolve_companies(
            Sector.TECH, f"What about {company.ticker}?"
        )
        observations = await gateway.query_observations(
            Sector.TECH,
            [company.entity_id],
            ["revenue", "operating_margin"],
            latest_only=True,
        )
        events = await gateway.query_events(
            Sector.TECH,
            [company.entity_id],
            ["headcount"],
            latest_only=True,
        )

        assert resolution.resolved[0].entity_id == company.entity_id
        assert {item.metric_key for item in observations.observations} == {
            "revenue",
            "operating_margin",
        }
        assert observations.sources
        assert len(events.events) == 1
        assert events.events[0].occurred_at.isoformat() == "2024-12-31"
        assert events.events[0].published_at.isoformat() == "2025-02-15"
        assert "950 employees" in events.events[0].summary
        assert events.sources[0].source_id == events.events[0].source_id

        with pytest.raises(McpToolError, match="rejected the request"):
            await gateway.query_events(
                Sector.TECH,
                ["sec:9999999999"],
                ["headcount"],
                latest_only=True,
            )
        with pytest.raises(McpToolError, match="rejected the request"):
            await gateway.query_events(
                Sector.RETAIL,
                [company.entity_id],
                ["headcount"],
                latest_only=True,
            )

    asyncio.run(exercise())
