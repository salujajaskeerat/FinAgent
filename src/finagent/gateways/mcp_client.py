"""MCP client adapter implementing the sector data port."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from finagent.contracts.api import Sector
from finagent.contracts.mcp import (
    DatasetCatalog,
    EventResult,
    ObservationResult,
    ResolutionResult,
)
from finagent.core.errors import DependencyUnavailableError


class ToolCaller(Protocol):
    """Small seam around the MCP SDK for focused testing."""

    async def call(self, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        """Call a named MCP tool and return its structured object."""
        ...


class McpToolError(ValueError):
    """A valid MCP request was rejected by the data tool."""


class StreamableHttpToolCaller:
    """MCP Streamable HTTP caller with a bounded transient retry."""

    def __init__(self, url: str, timeout_seconds: float = 3.0) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def call(self, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        """Execute a tool call over an initialized MCP session.

        Parameters
        ----------
        name
            Registered MCP tool name.
        arguments
            JSON-compatible tool arguments.

        Returns
        -------
        dict[str, Any]
            Structured MCP tool result.

        Raises
        ------
        DependencyUnavailableError
            If both bounded transport attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    return await self._call_once(name, arguments)
            except McpToolError:
                raise
            except (OSError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.05)
        raise DependencyUnavailableError(
            "MCP data service is unavailable"
        ) from last_error

    async def _call_once(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> dict[str, Any]:
        from mcp import Client

        async with Client(self._url) as client:
            result = await client.call_tool(name, dict(arguments))
        if result.is_error:
            details = "; ".join(
                text
                for item in result.content
                if isinstance((text := getattr(item, "text", None)), str)
            )
            suffix = f": {details}" if details else ""
            raise McpToolError(f"MCP tool {name!r} rejected the request{suffix}")
        structured = result.structured_content
        if isinstance(structured, dict):
            return structured
        raise RuntimeError(f"MCP tool {name!r} returned no structured object")


class McpDataGateway:
    """Typed application-facing facade over four allowlisted MCP tools."""

    def __init__(self, caller: ToolCaller) -> None:
        self._caller = caller

    async def get_catalog(self, sector: Sector) -> DatasetCatalog:
        """Return the sector catalog."""
        payload = await self._caller.call("get_catalog", {"sector": sector.value})
        return DatasetCatalog.model_validate(payload)

    async def resolve_companies(self, sector: Sector, query: str) -> ResolutionResult:
        """Resolve company mentions without exposing SQL."""
        payload = await self._caller.call(
            "resolve_companies",
            {"sector": sector.value, "query": query},
        )
        return ResolutionResult.model_validate(payload)

    async def query_observations(
        self,
        sector: Sector,
        entity_ids: list[str],
        metric_keys: list[str],
        latest_only: bool,
    ) -> ObservationResult:
        """Retrieve source-linked observations."""
        payload = await self._caller.call(
            "query_observations",
            {
                "sector": sector.value,
                "entity_ids": entity_ids,
                "metric_keys": metric_keys,
                "latest_only": latest_only,
                "limit": 100,
            },
        )
        return ObservationResult.model_validate(payload)

    async def query_events(
        self,
        sector: Sector,
        entity_ids: list[str],
        event_kinds: list[str],
        latest_only: bool,
    ) -> EventResult:
        """Retrieve source-linked operating signals."""
        payload = await self._caller.call(
            "query_events",
            {
                "sector": sector.value,
                "entity_ids": entity_ids,
                "event_kinds": event_kinds,
                "latest_only": latest_only,
                "limit": 100,
            },
        )
        return EventResult.model_validate(payload)
