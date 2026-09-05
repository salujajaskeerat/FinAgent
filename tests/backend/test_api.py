"""HTTP contract tests for the shared analysis endpoint."""

import asyncio
from typing import Any

import httpx

from finagent.api.app import create_app
from finagent.core.analysis_service import AnalysisService
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.gateways.llm import FakeLlmGateway
from tests.backend.support import StubDataGateway


def _app():
    service = AnalysisService(
        StubDataGateway(),
        FakeLlmGateway(),
        PersonaPolicyStore.load(),
    )
    return create_app(service)


async def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def test_analysis_response_is_structured_and_correlated() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/v1/analyses",
            headers={"X-Correlation-ID": "review-demo"},
            json={
                "query": "How do the fundamentals look?",
                "persona": "equity_analyst",
                "sector": "tech",
            },
        )
    )

    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["X-Correlation-ID"] == "review-demo"
    assert response.json()["sources"][0]["source_id"] == "src_fixture"


def test_invalid_persona_uses_problem_detail_contract() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/v1/analyses",
            json={"query": "A valid question", "persona": "trader", "sector": "tech"},
        )
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "INVALID_REQUEST"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_catalog_drives_frontend_selectors() -> None:
    response = asyncio.run(_request("GET", "/v1/catalog", params={"sector": "tech"}))

    assert response.status_code == 200
    assert len(response.json()["personas"]) == 3
    assert response.json()["companies"][0]["ticker"] == "EXM"
