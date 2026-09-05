"""Opt-in smoke test for the real Gemini provider."""

from __future__ import annotations

import asyncio
import os

import pytest

from finagent.contracts.api import AnalysisRequest, AnalysisStatus, Persona, Sector
from finagent.core.analysis_service import AnalysisService
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.gateways.llm import GeminiLlmGateway, LlmSettings
from tests.backend.support import StubDataGateway

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_GEMINI_TEST") != "1" or not os.getenv("GEMINI_API_KEY"),
        reason="set RUN_LIVE_GEMINI_TEST=1 and GEMINI_API_KEY to call Gemini",
    ),
]


def test_real_gemini_returns_a_grounded_analysis() -> None:
    """Exercise real planning and synthesis against public fixture evidence."""
    service = AnalysisService(
        StubDataGateway(),
        GeminiLlmGateway(LlmSettings.from_env()),
        PersonaPolicyStore.load(),
        deadline_seconds=60,
    )

    result = asyncio.run(
        service.analyze(
            AnalysisRequest(
                query="Analyze the reported revenue using only the supplied evidence.",
                persona=Persona.EQUITY,
                sector=Sector.TECH,
            )
        )
    )

    assert result.status is AnalysisStatus.ANSWERED
    assert result.findings
    assert result.sources
