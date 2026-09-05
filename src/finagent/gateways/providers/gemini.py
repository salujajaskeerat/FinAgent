"""Gemini adapter using the official Google Gen AI SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from google import genai
from google.genai import errors, types

from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.gateways.providers.base import LlmSettings, StructuredRequest

GenerateContent = Callable[..., Awaitable[types.GenerateContentResponse]]


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
            thinking_config=types.ThinkingConfig(
                thinking_budget=request.thinking_budget
            ),
            response_mime_type="application/json",
            response_schema=request.schema,
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
            if exc.code == 429:
                raise RateLimitError(
                    "Gemini rate limit exceeded after bounded retries."
                ) from exc
            raise DependencyUnavailableError(
                "Gemini could not complete the structured model request."
            ) from exc
        except TimeoutError as exc:
            raise DependencyUnavailableError(
                "Gemini did not respond within the configured timeout."
            ) from exc
        except Exception as exc:
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
