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

import functools
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from finagent.contracts.api import Sector
from finagent.contracts.mcp import (
    DatasetCatalog,
    EventResult,
    ObservationResult,
    ResolutionResult,
)
from finagent.mcp_server.repository import RepositoryError, SectorRepository

logger = logging.getLogger("finagent.mcp_server")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
P = ParamSpec("P")
R = TypeVar("R")

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


def database_path() -> Path:
    """Resolve the dataset path from ``FINAGENT_DB_PATH``.

    A relative path is tried against the current directory first and then
    against the project root, so the server works no matter where it was
    started from.
    """
    configured = Path(os.getenv("FINAGENT_DB_PATH", "data/finagent.db"))
    if configured.is_absolute() or configured.is_file():
        return configured
    fallback = PROJECT_ROOT / configured
    return fallback if fallback.is_file() else configured


def _repository() -> SectorRepository:
    """Construct the repository from process configuration."""
    return SectorRepository(database_path())


def _tool(func: Callable[P, R]) -> Callable[P, R]:
    """Surface dataset and input problems as readable MCP tool errors."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except (RepositoryError, ValueError) as exc:
            logger.warning("tool %s failed: %s", func.__name__, exc)
            raise ToolError(str(exc)) from exc

    return wrapper


@mcp.tool(annotations=READ_ONLY)
@_tool
def get_catalog(sector: str) -> DatasetCatalog:
    """List entities, metrics, events, and coverage for one sector."""
    return _repository().get_catalog(Sector(sector))


@mcp.tool(annotations=READ_ONLY)
@_tool
def resolve_companies(sector: str, query: str) -> ResolutionResult:
    """Resolve company names or tickers mentioned in a natural-language query."""
    return _repository().resolve_companies(Sector(sector), query)


@mcp.tool(annotations=READ_ONLY)
@_tool
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
@_tool
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
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    path = database_path()
    if not path.is_file():
        logger.error(
            "dataset not found at %s (FINAGENT_DB_PATH=%r, cwd=%s); build it with "
            "`uv run python -m finagent.ingestion build` or check the path",
            path,
            os.getenv("FINAGENT_DB_PATH"),
            Path.cwd(),
        )
        raise SystemExit(2)
    logger.info("serving dataset %s", path.resolve())
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
