"""Provider-neutral contract for structured (JSON) model completions.

The application never talks to a vendor SDK directly. It builds a
:class:`StructuredRequest` and hands it to whichever
:class:`StructuredCompletionProvider` the environment selected. Providers
return raw JSON-compatible output; the caller validates it with Pydantic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

SUPPORTED_PROVIDERS = ("fake", "gemini", "openai_compatible", "anthropic")

_DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai_compatible": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
}
_LEGACY_KEY_VARIABLES = {
    "gemini": "GEMINI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """Environment-backed configuration shared by every provider adapter."""

    provider: str = "fake"
    model: str = ""
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_seconds: float = 20.0
    max_attempts: int = 2
    synthesis_thinking_budget: int = 1024

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", _DEFAULT_MODELS.get(self.provider, ""))

    @classmethod
    def from_env(cls) -> LlmSettings:
        """Load settings from ``LLM_*`` variables without exposing secrets.

        ``LLM_API_KEY`` is the generic key variable. The provider-specific
        names (``GEMINI_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``)
        remain accepted as fallbacks.

        Returns
        -------
        LlmSettings
            Validated runtime configuration.

        Raises
        ------
        ValueError
            If the provider is unknown or a numeric setting is out of range.
        """
        provider = os.getenv("LLM_PROVIDER", "fake").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                "LLM_PROVIDER must be one of: " + ", ".join(SUPPORTED_PROVIDERS)
            )
        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
        max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))
        if not 1 <= timeout_seconds <= 60:
            raise ValueError("LLM_TIMEOUT_SECONDS must be between 1 and 60")
        if not 1 <= max_attempts <= 5:
            raise ValueError("LLM_MAX_ATTEMPTS must be between 1 and 5")
        thinking_budget = int(os.getenv("LLM_SYNTHESIS_THINKING_BUDGET", "1024"))
        if not 0 <= thinking_budget <= 8192:
            raise ValueError("LLM_SYNTHESIS_THINKING_BUDGET must be between 0 and 8192")
        api_key = os.getenv("LLM_API_KEY", "").strip() or None
        legacy_variable = _LEGACY_KEY_VARIABLES.get(provider)
        if api_key is None and legacy_variable:
            api_key = os.getenv(legacy_variable, "").strip() or None
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "").strip(),
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            synthesis_thinking_budget=thinking_budget,
        )

    def key_requirement(self) -> str:
        """Describe which variables can supply this provider's API key."""
        legacy = _LEGACY_KEY_VARIABLES.get(self.provider)
        return f"LLM_API_KEY (or {legacy})" if legacy else "LLM_API_KEY"


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """One structured completion request, independent of any vendor SDK."""

    system_instruction: str
    payload: dict[str, object]
    schema: type[BaseModel]
    max_output_tokens: int
    temperature: float = 0.1
    thinking_budget: int = 0


class StructuredCompletionProvider(Protocol):
    """Vendor adapter that returns JSON matching ``request.schema``.

    Implementations must raise :class:`finagent.core.errors.RateLimitError`
    for upstream throttling and :class:`DependencyUnavailableError` for any
    other failure, and must never include upstream messages or secrets in
    those errors.
    """

    name: str

    async def complete_structured(self, request: StructuredRequest) -> object:
        """Return the model output as a JSON string, mapping, or model instance."""
        ...


def schema_instruction(schema: type[BaseModel]) -> str:
    """Render a schema reminder for providers without native schema enforcement."""
    import json

    return (
        " Respond with exactly one JSON object and nothing else. It must match this "
        "JSON schema: " + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )
