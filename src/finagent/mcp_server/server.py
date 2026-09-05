"""MCP tool host for the read-only financial dataset.

Design notes for reviewers:

* Four narrow, typed tools instead of a ``run_sql`` tool. The catalog tool
  advertises the only identifiers the other tools accept, so it doubles as the
  allowlist the application enforces on model-proposed plans.
* Every result carries its provenance in-band (``sources``) so a consumer can
  cite without a second round-trip.
* Tools return the Pydantic contracts directly, so the SDK publishes a real
  ``outputSchema`` and validates ``structuredContent`` against it.
* All tools are annotated read-only and idempotent; the SQLite connection is
  opened ``mode=ro`` with ``query_only`` on.
* The server is stateless over Streamable HTTP so any number of API workers
  can share it; ``FINAGENT_MCP_TRANSPORT=stdio`` serves the same tools to a
  local client such as MCP Inspector.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from finagent.contracts.api import Sector
from finagent.contracts.mcp import (
    DatasetCatalog,
    EventResult,
    ObservationResult,
    ResolutionResult,
)
from finagent.mcp_server.repository import SectorRepository

mcp = MCPServer(
    "finagent-sector-data",
    instructions=(
        "Read-only sector financial dataset built from SEC filings. Call "
        "get_catalog first: its entity IDs, metric keys, and event kinds are the "
        "only values the query tools accept. Every result includes its sources."
    ),
)

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def _repository() -> SectorRepository:
    """Construct the repository from process configuration."""
    return SectorRepository(Path(os.getenv("FINAGENT_DB_PATH", "data/finagent.db")))


@mcp.tool(annotations=READ_ONLY)
def get_catalog(sector: str) -> DatasetCatalog:
    """List entities, metrics, events, and coverage for one sector."""
    return _repository().get_catalog(Sector(sector))


@mcp.tool(annotations=READ_ONLY)
def resolve_companies(sector: str, query: str) -> ResolutionResult:
    """Resolve company names or tickers mentioned in a natural-language query."""
    return _repository().resolve_companies(Sector(sector), query)


@mcp.tool(annotations=READ_ONLY)
def query_observations(
    sector: str,
    entity_ids: list[str],
    metric_keys: list[str],
    latest_only: bool = False,
    limit: int = 100,
) -> ObservationResult:
    """Read source-linked financial, market, and benchmark observations."""
    return _repository().query_observations(
        Sector(sector), entity_ids, metric_keys, latest_only, limit
    )


@mcp.tool(annotations=READ_ONLY)
def query_events(
    sector: str,
    entity_ids: list[str],
    event_kinds: list[str],
    latest_only: bool = False,
    limit: int = 100,
) -> EventResult:
    """Read source-linked operating signals such as headcount and guidance."""
    return _repository().query_events(
        Sector(sector), entity_ids, event_kinds, latest_only, limit
    )


@mcp.resource(
    "finagent://schema",
    name="dataset_schema",
    description="SQLite DDL of the purpose-built dataset behind the tools.",
    mime_type="text/plain",
)
def dataset_schema() -> str:
    """Expose the table definitions so clients can understand the data model."""
    return _repository().describe_schema()


def run() -> None:
    """Run the MCP server over Streamable HTTP, or stdio when requested."""
    transport = os.getenv("FINAGENT_MCP_TRANSPORT", "streamable-http").strip().lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    mcp.run(
        transport="streamable-http",
        host=os.getenv("FINAGENT_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("FINAGENT_MCP_PORT", "8001")),
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    run()
