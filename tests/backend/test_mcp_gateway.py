"""Tests for typed MCP result validation."""

import asyncio
from collections.abc import Mapping
from typing import Any

from finagent.contracts.api import Sector
from finagent.gateways.mcp_client import McpDataGateway


class FakeToolCaller:
    """Record calls and return one valid catalog payload."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    async def call(self, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {
            "dataset_version": "fixture-v1",
            "sector": "tech",
            "entities": [],
            "metric_keys": ["revenue"],
            "event_kinds": ["headcount"],
        }


def test_gateway_calls_allowlisted_catalog_tool_and_validates_result() -> None:
    caller = FakeToolCaller()
    gateway = McpDataGateway(caller)

    result = asyncio.run(gateway.get_catalog(Sector.TECH))

    assert result.dataset_version == "fixture-v1"
    assert caller.calls == [("get_catalog", {"sector": "tech"})]
