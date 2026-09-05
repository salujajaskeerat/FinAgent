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


@pytest.mark.integration
def test_tools_advertise_typed_schemas_and_read_only_annotations(
    real_mcp_url: str,
) -> None:
    """What an MCP client (or Inspector) sees when it lists the server."""
    from mcp import Client

    async def inspect() -> None:
        async with Client(real_mcp_url) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            resources = (await client.list_resources()).resources
            schema = await client.read_resource("finagent://schema")

        assert set(tools) == {
            "get_catalog",
            "resolve_companies",
            "query_observations",
            "query_events",
        }
        for tool in tools.values():
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert tool.annotations.idempotent_hint is True
            assert tool.annotations.open_world_hint is False
            assert tool.output_schema is not None
            assert "properties" in tool.output_schema
        observations = tools["query_observations"].output_schema
        assert {"observations", "sources", "warnings"} <= set(
            observations["properties"]
        )
        assert tools["query_observations"].input_schema["required"] == [
            "sector",
            "entity_ids",
            "metric_keys",
        ]
        assert [str(item.uri) for item in resources] == ["finagent://schema"]
        text = "".join(getattr(item, "text", "") for item in schema.contents)
        assert "CREATE TABLE annual_financial_snapshots" in text
        assert "CREATE TABLE source_lineage" in text

    asyncio.run(inspect())
