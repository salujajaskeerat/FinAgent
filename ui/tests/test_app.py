"""Headless render test for the Streamlit page against a stubbed API client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ui.api_client import (
    Analysis,
    Catalog,
    Company,
    Coverage,
    DerivedMetric,
    Finding,
    PersonaOption,
    SectorOption,
    Source,
    Trace,
)

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

CATALOG = Catalog(
    dataset_version="v1",
    personas=(
        PersonaOption("mutual_fund_analyst", "Mutual Fund Analyst", "long-only"),
        PersonaOption("equity_analyst", "Equity Analyst", "fundamentals"),
        PersonaOption("pe_analyst", "PE Analyst", "deals"),
    ),
    sectors=(SectorOption("tech", "Tech"),),
    companies=(Company("c1", "Example", "EXM"),),
    metric_keys=("revenue",),
)


def _analysis(persona: str) -> Analysis:
    return Analysis(
        request_id="req",
        status="answered",
        persona=persona,
        sector="tech",
        answer_markdown=f"## {persona} view\n\n### Section\nText",
        evidence_status="sufficient",
        findings=(Finding("Revenue grew.", ("c1",), ("s1",)),),
        companies=(Company("c1", "Example", "EXM"),),
        sources=(Source("s1", "10-K", "https://example.com", "SEC"),),
        limitations=(),
        derived_metrics=(
            DerivedMetric(
                "revenue_growth_yoy", "c1", 0.1, "ratio", "2024-12-31", "r/r-1"
            ),
        ),
        coverage=Coverage(("revenue",), ("revenue",), ()),
        trace=Trace(("received", "completed"), 2, False, ("revenue",), ("revenue",)),
    )


class _StubClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.calls: list[str] = []

    def get_catalog(self, sector: str) -> Catalog:
        return CATALOG

    def analyze(self, *, query: str, persona: str, sector: str) -> Analysis:
        return _analysis(persona)


def test_page_renders_and_compare_mode_shows_three_columns() -> None:
    with patch("ui.api_client.FinAgentApiClient", _StubClient):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        assert not app.exception
        assert app.title[0].value == "FinAgent"

        app.sidebar.toggle[0].set_value(True)
        app.main.selectbox[0].select("What do you think about Tesla?")
        app.button[0].click().run()
        assert not app.exception
        assert len(app.columns) == 3
        markdown = " ".join(item.value for item in app.markdown)
        assert "mutual_fund_analyst view" in markdown
        assert "pe_analyst view" in markdown
        assert "Derived metrics" in " ".join(item.label for item in app.expander)
