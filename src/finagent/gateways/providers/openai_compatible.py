"""Adapter for any server that speaks the OpenAI chat-completions format.

One adapter covers OpenAI itself and the many compatible endpoints (Groq,
Mistral, Together, OpenRouter, DeepSeek, local Ollama / LM Studio / vLLM)
by pointing ``LLM_BASE_URL`` at the server. Native JSON-schema enforcement
is requested first; servers that reject it fall back to plain JSON mode with
the schema described in the system instruction.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.gateways.providers.base import (
    LlmSettings,
    StructuredRequest,
    schema_instruction,
)

ChatCompletion = Callable[..., Awaitable[Any]]


class OpenAiCompatibleProvider:
    """Structured JSON completions over the OpenAI chat-completions API."""

    name = "openai_compatible"

    def __init__(
        self,
        settings: LlmSettings,
        chat_completion: ChatCompletion | None = None,
    ) -> None:
        """Configure the adapter.

        Parameters
        ----------
        settings
            Provider settings. A key is optional because local servers such
            as Ollama do not require one; ``base_url`` selects the server.
        chat_completion
            Optional SDK-compatible async function used by offline unit tests.

        Raises
        ------
        DependencyUnavailableError
            If the ``openai`` package is not installed.
        """
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise DependencyUnavailableError(
                "LLM_PROVIDER=openai_compatible requires the 'openai' extra: "
                "uv sync --extra openai"
            ) from exc
        self._openai = openai
        self._settings = settings
        self._chat_completion = chat_completion

    async def complete_structured(self, request: StructuredRequest) -> object:
        """Return the assistant message content as a JSON string."""
        messages = [
            {
                "role": "system",
                "content": request.system_instruction
                + schema_instruction(request.schema),
            },
            {
                "role": "user",
                "content": json.dumps(request.payload, separators=(",", ":")),
            },
        ]
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema.__name__,
                "schema": request.schema.model_json_schema(),
            },
        }
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                try:
                    completion = await self._create(
                        messages, request, response_format=schema_format
                    )
                except self._openai.BadRequestError:
                    # Older compatible servers only support plain JSON mode.
                    completion = await self._create(
                        messages, request, response_format={"type": "json_object"}
                    )
        except self._openai.RateLimitError as exc:
            raise RateLimitError(
                "The OpenAI-compatible endpoint rate limited the request."
            ) from exc
        except TimeoutError as exc:
            raise DependencyUnavailableError(
                "The OpenAI-compatible endpoint did not respond within the timeout."
            ) from exc
        except Exception as exc:
            raise DependencyUnavailableError(
                "The OpenAI-compatible endpoint could not complete the request."
            ) from exc
        try:
            return completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise DependencyUnavailableError(
                "The OpenAI-compatible endpoint returned no message content."
            ) from exc

    async def _create(
        self,
        messages: list[dict[str, str]],
        request: StructuredRequest,
        *,
        response_format: dict[str, Any],
    ) -> Any:
        chat_completion = self._chat_completion or self._create_with_sdk
        return await chat_completion(
            model=self._settings.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            response_format=response_format,
        )

    async def _create_with_sdk(self, **kwargs: Any) -> Any:
        client = self._openai.AsyncOpenAI(
            api_key=self._settings.api_key or "not-needed",
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_seconds,
            max_retries=max(self._settings.max_attempts - 1, 0),
        )
        async with client:
            return await client.chat.completions.create(**kwargs)
