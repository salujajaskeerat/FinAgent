"""MCP tool host for the read-only financial dataset."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer

from finagent.contracts.api import Sector
from finagent.mcp_server.repository import SectorRepository

mcp = MCPServer("finagent-sector-data")


def _repository() -> SectorRepository:
    """Construct the repository from process configuration."""
    return SectorRepository(Path(os.getenv("FINAGENT_DB_PATH", "data/finagent.db")))


@mcp.tool()
def get_catalog(sector: str) -> dict[str, object]:
    """List entities, metrics, events, and coverage for one sector."""
    result = _repository().get_catalog(Sector(sector))
    return result.model_dump(mode="json")


@mcp.tool()
def resolve_companies(sector: str, query: str) -> dict[str, object]:
    """Resolve company names or tickers mentioned in a natural-language query."""
    result = _repository().resolve_companies(Sector(sector), query)
    return result.model_dump(mode="json")


@mcp.tool()
def query_observations(
    sector: str,
    entity_ids: list[str],
    metric_keys: list[str],
    latest_only: bool = False,
    limit: int = 100,
) -> dict[str, object]:
    """Read source-linked financial, market, and benchmark observations."""
    result = _repository().query_observations(
        Sector(sector), entity_ids, metric_keys, latest_only, limit
    )
    return result.model_dump(mode="json")


@mcp.tool()
def query_events(
    sector: str,
    entity_ids: list[str],
    event_kinds: list[str],
    latest_only: bool = False,
    limit: int = 100,
) -> dict[str, object]:
    """Read source-linked operating signals such as headcount and guidance."""
    result = _repository().query_events(
        Sector(sector), entity_ids, event_kinds, latest_only, limit
    )
    return result.model_dump(mode="json")


def run() -> None:
    """Run the MCP server over Streamable HTTP."""
    mcp.run(
        transport="streamable-http",
        host=os.getenv("FINAGENT_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("FINAGENT_MCP_PORT", "8001")),
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    run()
