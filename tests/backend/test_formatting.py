"""Tests for deterministic answer Markdown normalization."""

from __future__ import annotations

from finagent.core.formatting import normalize_answer_markdown

SECTIONS = [
    "Benchmark-relative view",
    "Growth durability",
    "Portfolio fit and downside",
]


def test_run_on_headings_are_split_onto_their_own_lines() -> None:
    text = (
        "Benchmark-relative view Microsoft (MSFT) leads. ### Growth durability NVIDIA "
        "(NVDA) grew 65.5% in 2026. ### Portfolio fit and downside MSFT is a core holding."
    )

    result = normalize_answer_markdown("### " + text, SECTIONS)

    assert result.split("\n\n")[:2] == [
        "### Benchmark-relative view",
        "Microsoft (MSFT) leads.",
    ]
    assert "\n### Growth durability\n" in result
    assert "\n### Portfolio fit and downside\n" in result
    assert all(
        line.startswith("### ") or "###" not in line for line in result.splitlines()
    )


def test_inline_bullet_runs_become_one_bullet_per_line() -> None:
    text = "### Earnings and margins\nRevenue rose 6%. - Margin expanded 0.5 pp. - FCF was 98.8B USD."

    result = normalize_answer_markdown(text, ["Earnings and margins"])

    assert result.splitlines()[2:] == [
        "- Revenue rose 6%.",
        "- Margin expanded 0.5 pp.",
        "- FCF was 98.8B USD.",
    ]


def test_well_formed_markdown_is_left_intact() -> None:
    text = "### Valuation\n\n- Only public float exists.\n- Multiples cannot be assessed.\n\n**Decision:** watch.\n"

    assert normalize_answer_markdown(text, ["Valuation"]) == text


def test_hyphenated_words_and_negative_numbers_are_not_bullets() -> None:
    text = "### Entry and leverage case\nNet debt is -2.1B USD year-over-year; free cash flow - the key input - is strong."

    result = normalize_answer_markdown(text, ["Entry and leverage case"])

    assert result.count("\n- ") == 0


def test_decision_line_after_bullets_gets_its_own_paragraph() -> None:
    text = "### Valuation\n- Only public float exists.\n**Decision:** watch."

    result = normalize_answer_markdown(text, ["Valuation"])

    assert result.endswith("- Only public float exists.\n\n**Decision:** watch.\n")
