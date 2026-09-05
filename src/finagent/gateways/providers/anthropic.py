"""Anthropic adapter using the official ``anthropic`` SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from finagent.core.errors import DependencyUnavailableError, RateLimitError
from finagent.gateways.providers.base import LlmSettings, StructuredRequest

CreateMessage = Callable[..., Awaitable[Any]]


class AnthropicProvider:
    """Structured JSON completions through the Messages API output format."""

    name = "anthropic"

    def __init__(
        self,
        settings: LlmSettings,
        create_message: CreateMessage | None = None,
    ) -> None:
        """Configure the adapter.

        Parameters
        ----------
        settings
            Provider settings. The API key is held in memory and never logged.
        create_message
            Optional SDK-compatible async function used by offline unit tests.

        Raises
        ------
        DependencyUnavailableError
            If the SDK is missing or no API key is configured.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise DependencyUnavailableError(
                "LLM_PROVIDER=anthropic requires the 'anthropic' extra: "
                "uv sync --extra anthropic"
            ) from exc
        if not settings.api_key:
            raise DependencyUnavailableError(
                f"LLM_PROVIDER=anthropic requires {settings.key_requirement()}."
            )
        self._anthropic = anthropic
        self._settings = settings
        self._create_message = create_message

    async def complete_structured(self, request: StructuredRequest) -> object:
        """Return the concatenated text blocks as a JSON string."""
        create_message = self._create_message or self._create_with_sdk
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                message = await create_message(
                    model=self._settings.model,
                    max_tokens=request.max_output_tokens,
                    system=request.system_instruction,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(
                                request.payload, separators=(",", ":")
                            ),
                        }
                    ],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": request.schema.model_json_schema(),
                        }
                    },
                )
        except self._anthropic.RateLimitError as exc:
            raise RateLimitError("Anthropic rate limited the request.") from exc
        except TimeoutError as exc:
            raise DependencyUnavailableError(
                "Anthropic did not respond within the configured timeout."
            ) from exc
        except Exception as exc:
            raise DependencyUnavailableError(
                "Anthropic could not complete the structured model request."
            ) from exc
        if getattr(message, "stop_reason", None) == "refusal":
            raise DependencyUnavailableError("Anthropic declined the request.")
        text = "".join(
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        )
        if not text:
            raise DependencyUnavailableError("Anthropic returned no text content.")
        return text

    async def _create_with_sdk(self, **kwargs: Any) -> Any:
        client = self._anthropic.AsyncAnthropic(
            api_key=self._settings.api_key,
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_seconds,
            max_retries=max(self._settings.max_attempts - 1, 0),
        )
        async with client:
            return await client.messages.create(**kwargs)
