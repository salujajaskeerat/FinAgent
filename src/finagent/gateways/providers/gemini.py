"""Gemini adapter using the official Google Gen AI SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.gateways.providers.base import LlmSettings, StructuredRequest

GenerateContent = Callable[..., Awaitable[types.GenerateContentResponse]]
logger = logging.getLogger("finagent.llm.gemini")


class GeminiProvider:
    """Structured JSON completions with native response-schema enforcement.

    Gemini receives no tools of any kind; it only returns JSON matching the
    requested schema.
    """

    name = "gemini"

    def __init__(
        self,
        settings: LlmSettings,
        generate_content: GenerateContent | None = None,
    ) -> None:
        """Configure the adapter.

        Parameters
        ----------
        settings
            Provider settings. The API key is held in memory and never logged.
        generate_content
            Optional SDK-compatible async function used by offline unit tests.

        Raises
        ------
        DependencyUnavailableError
            If no API key is configured.
        """
        if not settings.api_key:
            raise DependencyUnavailableError(
                f"LLM_PROVIDER=gemini requires {settings.key_requirement()}."
            )
        self._settings = settings
        self._generate_content = generate_content

    async def complete_structured(self, request: StructuredRequest) -> object:
        """Return the SDK's parsed object, or its text when parsing is absent."""
        config = types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            candidate_count=1,
            max_output_tokens=request.max_output_tokens,
            # Gemini 3.x rejects thinking_budget=0; omit the config to use the
            # model's default and only pass an explicit positive budget.
            thinking_config=(
                types.ThinkingConfig(thinking_budget=request.thinking_budget)
                if request.thinking_budget > 0
                else None
            ),
            response_mime_type="application/json",
            response_schema=gemini_schema(request.schema),
        )
        generate_content = self._generate_content or self._generate_with_sdk
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                response = await generate_content(
                    model=self._settings.model,
                    contents=json.dumps(request.payload, separators=(",", ":")),
                    config=config,
                )
        except errors.APIError as exc:
            # Server-side log only: status and message help operators, never the key.
            logger.warning("gemini API error %s: %s", exc.code, exc.message)
            if exc.code == 429:
                raise RateLimitError(
                    "Gemini rate limit exceeded after bounded retries."
                ) from exc
            raise DependencyUnavailableError(
                f"Gemini rejected the structured model request (HTTP {exc.code})."
            ) from exc
        except TimeoutError as exc:
            logger.warning(
                "gemini request timed out after %ss", self._settings.timeout_seconds
            )
            raise DependencyUnavailableError(
                "Gemini did not respond within the configured timeout."
            ) from exc
        except Exception as exc:
            logger.warning("gemini request failed: %s: %s", type(exc).__name__, exc)
            raise DependencyUnavailableError(
                "Gemini could not complete the structured model request."
            ) from exc
        if response.parsed is not None:
            return response.parsed
        return response.text

    async def _generate_with_sdk(
        self, **kwargs: object
    ) -> types.GenerateContentResponse:
        """Execute one SDK request and close its HTTP resources afterward."""
        client = genai.Client(
            api_key=self._settings.api_key,
            http_options=types.HttpOptions(
                timeout=int(self._settings.timeout_seconds * 1_000),
                retry_options=types.HttpRetryOptions(
                    attempts=self._settings.max_attempts,
                    initial_delay=0.5,
                    max_delay=2.0,
                    exp_base=2.0,
                    jitter=0.2,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                ),
            ),
        )
        async with client.aio as async_client:
            return await async_client.models.generate_content(**kwargs)


# Keys the Gemini response_schema format understands. Everything else that
# Pydantic emits (title, default, additionalProperties, minLength, $defs, ...)
# is rejected with HTTP 400 and must be stripped.
_GEMINI_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "nullable",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    }
)


def gemini_schema(schema: type[BaseModel]) -> dict[str, object]:
    """Convert a Pydantic model into a schema Gemini's API accepts.

    Parameters
    ----------
    schema
        Strict Pydantic model describing the expected structured output.

    Returns
    -------
    dict[str, object]
        JSON schema with ``$ref`` inlined, ``Optional`` expressed as
        ``nullable``, and unsupported keywords removed. Field validation still
        happens application-side against the original model.
    """
    raw = schema.model_json_schema()
    definitions = raw.get("$defs", {})

    def clean(node: dict[str, object]) -> dict[str, object]:
        if "$ref" in node:
            name = str(node["$ref"]).rsplit("/", 1)[-1]
            return clean(dict(definitions[name]))
        if "anyOf" in node:
            branches = [item for item in node["anyOf"] if item.get("type") != "null"]
            nullable = len(branches) != len(node["anyOf"])
            merged = clean(dict(branches[0])) if len(branches) == 1 else {}
            if nullable:
                merged["nullable"] = True
            if "description" in node:
                merged["description"] = node["description"]
            return merged
        result: dict[str, object] = {}
        for key, value in node.items():
            if key not in _GEMINI_SCHEMA_KEYS:
                continue
            if key == "properties":
                result[key] = {name: clean(dict(prop)) for name, prop in value.items()}
            elif key == "items":
                result[key] = clean(dict(value))
            else:
                result[key] = value
        return result

    return clean(raw)
