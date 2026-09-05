"""Contract tests every vendor adapter must satisfy, with stubbed SDK calls."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import anthropic
import openai
import pytest
from google.genai import types

from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.core.models import RetrievalPlan
from finagent.gateways.llm import validate_structured_output
from finagent.gateways.providers import LlmSettings, StructuredRequest
from finagent.gateways.providers.anthropic import AnthropicProvider
from finagent.gateways.providers.gemini import GeminiProvider
from finagent.gateways.providers.openai_compatible import OpenAiCompatibleProvider

PLAN_JSON = json.dumps(
    {
        "entity_ids": ["cmp_example"],
        "metric_keys": ["revenue"],
        "event_kinds": [],
        "latest_only": False,
    }
)


def _request() -> StructuredRequest:
    return StructuredRequest(
        system_instruction="Plan retrieval.",
        payload={"question": "revenue?"},
        schema=RetrievalPlan,
        max_output_tokens=256,
    )


class _Stub:
    """Record calls and return or raise a configured result."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def _settings(provider: str) -> LlmSettings:
    return LlmSettings(provider=provider, api_key="test-only-key", timeout_seconds=2)


def _gemini(stub: _Stub) -> GeminiProvider:
    return GeminiProvider(_settings("gemini"), stub)


def _openai(stub: _Stub) -> OpenAiCompatibleProvider:
    return OpenAiCompatibleProvider(_settings("openai_compatible"), stub)


def _anthropic(stub: _Stub) -> AnthropicProvider:
    return AnthropicProvider(_settings("anthropic"), stub)


def _openai_completion(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _anthropic_message(text: str, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


def _rate_limit(provider: str) -> Exception:
    response = SimpleNamespace(status_code=429, headers={}, request=None)
    if provider == "gemini":
        from google.genai import errors

        return errors.ClientError(429, {"error": {"message": "secret upstream"}})
    if provider == "openai_compatible":
        return openai.RateLimitError("secret upstream", response=response, body=None)
    return anthropic.RateLimitError("secret upstream", response=response, body=None)


CASES: list[tuple[str, Callable[[_Stub], Any], Any]] = [
    (
        "gemini",
        _gemini,
        types.GenerateContentResponse(
            parsed=RetrievalPlan.model_validate_json(PLAN_JSON)
        ),
    ),
    ("openai_compatible", _openai, _openai_completion(PLAN_JSON)),
    ("anthropic", _anthropic, _anthropic_message(PLAN_JSON)),
]


@pytest.mark.parametrize(("name", "factory", "response"), CASES)
def test_every_provider_returns_schema_valid_json(
    name: str, factory: Callable[[_Stub], Any], response: Any
) -> None:
    stub = _Stub(result=response)
    provider = factory(stub)

    raw = asyncio.run(provider.complete_structured(_request()))
    plan = validate_structured_output(RetrievalPlan, raw, what="plan")

    assert provider.name == name
    assert plan.entity_ids == ["cmp_example"]
    call = stub.calls[0]
    assert "tools" not in call
    assert "test-only-key" not in json.dumps(call, default=str)


@pytest.mark.parametrize(("name", "factory", "_response"), CASES)
def test_every_provider_maps_rate_limits_without_leaking_detail(
    name: str, factory: Callable[[_Stub], Any], _response: Any
) -> None:
    provider = factory(_Stub(error=_rate_limit(name)))

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.complete_structured(_request()))

    assert "secret upstream" not in str(caught.value)


@pytest.mark.parametrize(("name", "factory", "_response"), CASES)
def test_every_provider_wraps_unknown_failures(
    name: str, factory: Callable[[_Stub], Any], _response: Any
) -> None:
    provider = factory(_Stub(error=RuntimeError("socket closed: secret host")))

    with pytest.raises(DependencyUnavailableError) as caught:
        asyncio.run(provider.complete_structured(_request()))

    assert "secret host" not in str(caught.value)


def test_openai_compatible_falls_back_to_json_object_mode() -> None:
    """Servers that reject json_schema still work through plain JSON mode."""
    response = SimpleNamespace(status_code=400, headers={}, request=None)
    attempts: list[dict[str, Any]] = []

    async def chat_completion(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema":
            raise openai.BadRequestError("unsupported", response=response, body=None)
        return _openai_completion(PLAN_JSON)

    provider = OpenAiCompatibleProvider(_settings("openai_compatible"), chat_completion)
    raw = asyncio.run(provider.complete_structured(_request()))

    assert validate_structured_output(RetrievalPlan, raw, what="plan").metric_keys == [
        "revenue"
    ]
    assert [item["response_format"]["type"] for item in attempts] == [
        "json_schema",
        "json_object",
    ]
    assert "JSON schema" in attempts[1]["messages"][0]["content"]


def test_anthropic_refusal_is_a_dependency_failure() -> None:
    provider = _anthropic(_Stub(result=_anthropic_message("", stop_reason="refusal")))

    with pytest.raises(DependencyUnavailableError, match="declined"):
        asyncio.run(provider.complete_structured(_request()))


def test_anthropic_requests_native_json_schema_output() -> None:
    stub = _Stub(result=_anthropic_message(PLAN_JSON))
    asyncio.run(_anthropic(stub).complete_structured(_request()))

    call = stub.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"] == "Plan retrieval."


def test_validation_tolerates_code_fences_and_rejects_garbage() -> None:
    fenced = "```json\n" + PLAN_JSON + "\n```"
    assert (
        validate_structured_output(RetrievalPlan, fenced, what="plan").latest_only
        is False
    )
    with pytest.raises(DependencyUnavailableError, match="invalid plan"):
        validate_structured_output(RetrievalPlan, "not json", what="plan")
    with pytest.raises(DependencyUnavailableError, match="no plan"):
        validate_structured_output(RetrievalPlan, "", what="plan")
