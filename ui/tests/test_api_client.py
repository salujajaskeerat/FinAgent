"""Focused tests for the UI's backend boundary."""

from __future__ import annotations

import io
import json
from email.message import Message
from typing import Self
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from ui.api_client import ApiClientError, FinAgentApiClient


class _Response:
    """Minimal context-managed URL response used by client tests."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        """Return the encoded response body."""
        return self._body


def test_catalog_is_fetched_over_http_with_sector_query() -> None:
    """The UI client obtains its selectors from the API catalog."""
    payload = {
        "dataset_version": "fixture-v1",
        "personas": [
            {
                "value": "equity_analyst",
                "label": "Equity Analyst",
                "description": "Earnings, valuation, catalysts, and risks.",
            }
        ],
        "sectors": [{"value": "logistics", "label": "Logistics"}],
        "companies": [{"company_id": "cmp_x", "name": "Example", "ticker": "EX"}],
        "metric_keys": ["revenue"],
        "coverage_start": "2024-01-01",
        "coverage_end": "2025-12-31",
    }
    with patch("ui.api_client.urlopen", return_value=_Response(payload)) as request:
        catalog = FinAgentApiClient("http://api.test").get_catalog("logistics")

    outbound = request.call_args.args[0]
    assert outbound.full_url == "http://api.test/v1/catalog?sector=logistics"
    assert outbound.method == "GET"
    assert catalog.companies[0].ticker == "EX"


def test_analysis_posts_structured_selection_and_parses_provenance() -> None:
    """The UI submits a single turn and retains source metadata."""
    payload = {
        "schema_version": "1.0",
        "request_id": "29926b51-2896-4eab-86f0-3bb9059a605c",
        "status": "answered",
        "persona": "equity_analyst",
        "sector": "logistics",
        "answer_markdown": "Example has improving margins.",
        "findings": [
            {
                "text": "Margins improved.",
                "company_ids": ["cmp_x"],
                "source_ids": ["src_1"],
            }
        ],
        "companies": [{"company_id": "cmp_x", "name": "Example", "ticker": "EX"}],
        "sources": [
            {
                "source_id": "src_1",
                "title": "Annual Report",
                "url": "https://example.test/report",
                "publisher": "Example",
                "published_at": "2026-02-01",
                "retrieved_at": "2026-03-01",
            }
        ],
        "evidence_status": "sufficient",
        "data_as_of": "2025-12-31",
        "limitations": [],
    }
    with patch("ui.api_client.urlopen", return_value=_Response(payload)) as request:
        result = FinAgentApiClient("http://api.test").analyze(
            query="Compare margins",
            persona="equity_analyst",
            sector="logistics",
        )

    outbound = request.call_args.args[0]
    assert outbound.full_url == "http://api.test/v1/analyses"
    assert outbound.method == "POST"
    assert json.loads(outbound.data) == {
        "query": "Compare margins",
        "persona": "equity_analyst",
        "sector": "logistics",
    }
    assert result.findings[0].source_ids == ("src_1",)
    assert result.sources[0].published_at == "2026-02-01"


def test_problem_detail_becomes_honest_retryable_error() -> None:
    """Dependency failures retain their message, code, and request ID."""
    body = io.BytesIO(
        json.dumps(
            {
                "detail": "MCP data service is unavailable.",
                "code": "dependency_unavailable",
                "request_id": "35c26381-1871-4177-89c8-c39b70e6f3be",
            }
        ).encode("utf-8")
    )
    headers = Message()
    headers["X-Request-ID"] = "header-request-id"
    error = HTTPError(
        "http://api.test/v1/catalog?sector=tech",
        503,
        "Service Unavailable",
        headers,
        body,
    )
    with (
        patch("ui.api_client.urlopen", side_effect=error),
        pytest.raises(ApiClientError) as caught,
    ):
        FinAgentApiClient("http://api.test").get_catalog("tech")

    assert str(caught.value) == "MCP data service is unavailable."
    assert caught.value.code == "dependency_unavailable"
    assert caught.value.request_id == "35c26381-1871-4177-89c8-c39b70e6f3be"
    assert caught.value.retryable is True
