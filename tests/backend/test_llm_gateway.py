"""Offline tests for structured LLM gateway behavior."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from google.genai import errors, types

from finagent.contracts.api import AnalysisRequest, Persona, Sector
from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.core.models import DraftAnalysis, EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.gateways.llm import (
    FakeLlmGateway,
    LlmSettings,
    StructuredLlmGateway,
    build_llm_gateway,
)
from finagent.gateways.providers.gemini import GeminiProvider
from tests.backend.support import StubDataGateway


class RecordingGenerateContent:
    """Capture SDK-shaped calls and return a prepared response."""

    def __init__(
        self,
        response: types.GenerateContentResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> types.GenerateContentResponse:
        """Return or raise the configured SDK result."""
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def _request() -> AnalysisRequest:
    return AnalysisRequest(
        query="What is the latest headcount signal?",
        persona=Persona.EQUITY,
        sector=Sector.TECH,
    )


def _settings(api_key: str | None = "test-only-key") -> LlmSettings:
    return LlmSettings(
        provider="gemini", api_key=api_key, timeout_seconds=2, max_attempts=2
    )


def _gemini_gateway(generate: RecordingGenerateContent) -> StructuredLlmGateway:
    return StructuredLlmGateway(GeminiProvider(_settings(), generate))


def test_gemini_plan_uses_structured_output_without_receiving_tools() -> None:
    data = StubDataGateway()
    expected = RetrievalPlan(
        entity_ids=["cmp_example"],
        metric_keys=["revenue"],
        event_kinds=["headcount"],
        latest_only=True,
    )
    generate = RecordingGenerateContent(types.GenerateContentResponse(parsed=expected))
    gateway = _gemini_gateway(generate)

    result = asyncio.run(
        gateway.plan(
            _request(),
            PersonaPolicyStore.load().get(Persona.EQUITY),
            data.catalog_value,
            ["cmp_example"],
        )
    )

    assert result == expected
    assert len(generate.calls) == 1
    call = generate.calls[0]
    assert call["model"] == "gemini-2.5-flash-lite"
    assert "tools" not in call
    assert json.loads(call["contents"])["target_entity_ids"] == ["cmp_example"]
    config = call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.response_mime_type == "application/json"
    assert config.response_schema is RetrievalPlan
    assert config.thinking_config.thinking_budget == 0
    assert config.tools is None


def test_gemini_synthesis_sends_only_source_linked_evidence() -> None:
    data = StubDataGateway()
    expected = DraftAnalysis.model_validate(
        {
            "answer_markdown": "Revenue evidence is available.",
            "findings": [
                {
                    "text": "Reported revenue was 1,200 USD.",
                    "company_ids": ["cmp_example"],
                    "source_ids": ["src_fixture"],
                }
            ],
        }
    )
    generate = RecordingGenerateContent(types.GenerateContentResponse(parsed=expected))
    gateway = _gemini_gateway(generate)
    observation_result = asyncio.run(
        data.query_observations(
            Sector.TECH,
            ["cmp_example"],
            ["revenue"],
            latest_only=True,
        )
    )
    evidence = EvidenceBundle(
        company_ids=["cmp_example"],
        observations=observation_result.observations,
        source_ids={"src_fixture"},
    )

    result = asyncio.run(
        gateway.synthesize(
            _request(), PersonaPolicyStore.load().get(Persona.EQUITY), evidence
        )
    )

    assert result == expected
    payload = json.loads(generate.calls[0]["contents"])
    assert payload["evidence"]["source_ids"] == ["src_fixture"]
    assert "api_key" not in generate.calls[0]["contents"].lower()


def test_gemini_missing_key_fails_without_a_network_call() -> None:
    with pytest.raises(DependencyUnavailableError, match="GEMINI_API_KEY"):
        GeminiProvider(_settings(api_key=None))


def test_llm_settings_repr_never_contains_the_api_key() -> None:
    settings = _settings(api_key="do-not-log-this-secret")

    assert "do-not-log-this-secret" not in repr(settings)


def test_gemini_maps_rate_limit_without_leaking_upstream_detail() -> None:
    upstream = errors.ClientError(
        429,
        {
            "error": {
                "message": "sensitive upstream message",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    gateway = _gemini_gateway(RecordingGenerateContent(error=upstream))

    with pytest.raises(RateLimitError, match="bounded retries") as caught:
        asyncio.run(
            gateway.plan(
                _request(),
                PersonaPolicyStore.load().get(Persona.EQUITY),
                StubDataGateway().catalog_value,
                ["cmp_example"],
            )
        )

    assert "sensitive upstream message" not in str(caught.value)


def test_provider_selection_requires_explicit_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert isinstance(build_llm_gateway(), FakeLlmGateway)

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(DependencyUnavailableError, match="GEMINI_API_KEY"):
        build_llm_gateway()

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    gateway = build_llm_gateway()
    assert isinstance(gateway, StructuredLlmGateway)
    assert gateway.provider_name == "gemini"


def test_generic_llm_api_key_selects_any_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One LLM_API_KEY variable configures every provider; legacy names still work."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-only-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = LlmSettings.from_env()
    assert settings.provider == "anthropic"
    assert settings.api_key == "test-only-key"
    assert settings.model == "claude-opus-5"
    assert build_llm_gateway(settings).provider_name == "anthropic"

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("LLM_API_KEY")
    settings = LlmSettings.from_env()
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.api_key is None  # local servers need no key
    assert build_llm_gateway(settings).provider_name == "openai_compatible"

    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        LlmSettings.from_env()
