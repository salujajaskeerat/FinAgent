"""Tests for the constrained model-assisted entity resolver."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from google.genai import types
from pydantic import ValidationError

from finagent.contracts.api import Sector
from finagent.contracts.entity_resolution import (
    EntityMatch,
    EntityResolution,
    EntityResolutionReason,
    EntityResolutionStatus,
)
from finagent.core.errors import DependencyUnavailableError
from finagent.gateways.entity_resolver import LlmEntityResolver
from finagent.gateways.llm import LlmSettings
from finagent.gateways.providers.gemini import GeminiProvider
from tests.backend.support import StubDataGateway


class RecordingGenerateContent:
    """Capture the model payload and return a prepared SDK-shaped response."""

    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> types.GenerateContentResponse:
        """Record and return one response."""
        self.calls.append(kwargs)
        return self.response


def _settings() -> LlmSettings:
    return LlmSettings(
        provider="gemini", api_key="test-only-key", timeout_seconds=2, max_attempts=2
    )


def _resolver(generated: RecordingGenerateContent) -> LlmEntityResolver:
    return LlmEntityResolver(GeminiProvider(_settings(), generated))


def _match(entity_id: str = "cmp_example", confidence: float = 0.9) -> EntityResolution:
    return EntityResolution(
        status=EntityResolutionStatus.MATCHED,
        matches=[
            EntityMatch(
                entity_id=entity_id,
                mention="the enterprise software example",
                confidence=confidence,
            )
        ],
        reason_code=EntityResolutionReason.SEMANTIC_REFERENCE,
    )


def test_resolver_receives_only_selected_sector_catalog_candidates() -> None:
    """Send no tools or unrestricted entity universe to the resolver."""
    generated = RecordingGenerateContent(types.GenerateContentResponse(parsed=_match()))
    resolver = _resolver(generated)
    candidates = [StubDataGateway().catalog_value.entities[0]]

    result = asyncio.run(
        resolver.resolve(
            "Tell me about the enterprise software example.",
            Sector.TECH,
            candidates,
        )
    )

    assert result == _match()
    assert len(generated.calls) == 1
    call = generated.calls[0]
    payload = json.loads(call["contents"])
    assert payload == {
        "query": "Tell me about the enterprise software example.",
        "sector": "tech",
        "candidates": [
            {
                "entity_id": "cmp_example",
                "name": "Example Systems, Inc.",
                "ticker": "EXM",
                "aliases": [],
            }
        ],
    }
    assert "tools" not in call
    assert call["config"].response_schema is EntityResolution
    assert call["config"].temperature == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "matched",
            "matches": [],
            "reason_code": "semantic_reference",
        },
        {
            "status": "no_match",
            "matches": [{"entity_id": "invented", "mention": "x", "confidence": 0.9}],
            "reason_code": "not_in_catalog",
        },
        {
            "status": "matched",
            "matches": [
                {"entity_id": "cmp_example", "mention": "x", "confidence": 1.1}
            ],
            "reason_code": "semantic_reference",
        },
        {
            "status": "no_match",
            "matches": [],
            "reason_code": "not_in_catalog",
            "explanation": "free form is forbidden",
        },
        {
            "status": "ambiguous",
            "matches": [],
            "reason_code": "semantic_reference",
        },
    ],
)
def test_resolution_contract_rejects_malformed_output(payload: dict[str, Any]) -> None:
    """Reject extra fields, bad confidence, and invalid match cardinality."""
    with pytest.raises(ValidationError):
        EntityResolution.model_validate(payload)


def test_gateway_maps_malformed_structured_output_to_safe_failure() -> None:
    """Convert malformed provider output into a dependency failure."""
    response = types.GenerateContentResponse(
        parsed={
            "status": "matched",
            "matches": [],
            "reason_code": "semantic_reference",
        }
    )
    resolver = _resolver(RecordingGenerateContent(response))

    with pytest.raises(DependencyUnavailableError, match="invalid entity-resolution"):
        asyncio.run(
            resolver.resolve(
                "Tell me about an explicit company.",
                Sector.TECH,
                [StubDataGateway().catalog_value.entities[0]],
            )
        )
