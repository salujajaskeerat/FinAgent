"""Vendor adapters behind one structured-completion contract."""

from __future__ import annotations

from finagent.gateways.providers.base import (
    SUPPORTED_PROVIDERS,
    LlmSettings,
    StructuredCompletionProvider,
    StructuredRequest,
)


def build_provider(settings: LlmSettings) -> StructuredCompletionProvider | None:
    """Build the configured provider, or ``None`` for the offline fake.

    Parameters
    ----------
    settings
        Runtime LLM configuration.

    Returns
    -------
    StructuredCompletionProvider or None
        A vendor adapter, or ``None`` when ``LLM_PROVIDER=fake``.

    Raises
    ------
    ValueError
        If the provider name is unsupported.
    """
    if settings.provider == "fake":
        return None
    if settings.provider == "gemini":
        from finagent.gateways.providers.gemini import GeminiProvider

        return GeminiProvider(settings)
    if settings.provider == "openai_compatible":
        from finagent.gateways.providers.openai_compatible import (
            OpenAiCompatibleProvider,
        )

        return OpenAiCompatibleProvider(settings)
    if settings.provider == "anthropic":
        from finagent.gateways.providers.anthropic import AnthropicProvider

        return AnthropicProvider(settings)
    raise ValueError("LLM_PROVIDER must be one of: " + ", ".join(SUPPORTED_PROVIDERS))


__all__ = [
    "SUPPORTED_PROVIDERS",
    "LlmSettings",
    "StructuredCompletionProvider",
    "StructuredRequest",
    "build_provider",
]
