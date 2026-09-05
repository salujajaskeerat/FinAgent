"""Repository integration against the real offline builder schema."""

import sqlite3
from pathlib import Path

import pytest

from finagent.contracts.api import Sector
from finagent.ingestion.builder import build_database
from finagent.mcp_server.repository import SectorRepository

FIXTURES = Path(__file__).parents[1] / "ingestion" / "fixtures"


def _repository(tmp_path: Path) -> SectorRepository:
    database = tmp_path / "finagent.db"
    build_database(
        FIXTURES / "source_manifest.yaml",
        FIXTURES / "raw" / "sec",
        database,
    )
    return SectorRepository(database)


def test_repository_maps_purpose_built_tables_to_mcp_contracts(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    catalog = repository.get_catalog(Sector.TECH)
    company_id = next(
        item.entity_id for item in catalog.entities if item.ticker == "EXM"
    )
    result = repository.query_observations(
        Sector.TECH,
        [company_id],
        ["revenue", "operating_margin"],
        latest_only=True,
        limit=100,
    )

    assert catalog.dataset_version
    assert {item.metric_key for item in result.observations} == {
        "revenue",
        "operating_margin",
    }
    assert len(result.sources) == 1
    assert result.sources[0].retrieved_at is not None


def test_repository_returns_latest_headcount_signal(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    catalog = repository.get_catalog(Sector.TECH)
    company_id = next(
        item.entity_id for item in catalog.entities if item.ticker == "EXM"
    )

    result = repository.query_events(
        Sector.TECH,
        [company_id],
        ["headcount"],
        latest_only=True,
        limit=100,
    )

    assert len(result.events) == 1
    assert result.events[0].occurred_at.isoformat() == "2024-12-31"
    assert result.events[0].published_at.isoformat() == "2025-02-15"
    assert "950" in result.events[0].summary


def test_repository_resolves_known_and_explicit_unknown_companies(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    known = repository.resolve_companies(Sector.TECH, "What about EXM?")
    unknown = repository.resolve_companies(Sector.TECH, "What about SpaceX?")

    assert known.resolved[0].mention == "EXM"
    assert unknown.unresolved_mentions == ["SpaceX"]


@pytest.mark.parametrize(
    "query",
    [
        "What about Example Systems, Inc.?",
        "What about example systems, inc.?",
        "What about Example Systems Inc?",
        "What about Example   Systems,   Inc.?",
        "What about EXM?",
        "What about exm?",
        "Tell me about Example Tech.",
    ],
)
def test_repository_resolution_normalizes_catalog_aliases(
    tmp_path: Path, query: str
) -> None:
    """Match names, tickers, and configured aliases without case sensitivity."""
    result = _repository(tmp_path).resolve_companies(Sector.TECH, query)

    assert [item.entity_id for item in result.resolved] == ["sec:0000000001"]
    assert result.unresolved_mentions == []


@pytest.mark.parametrize("query", ["What about EX?", "Tell me about App."])
def test_repository_resolution_respects_token_boundaries(
    tmp_path: Path, query: str
) -> None:
    """Do not resolve a token that is merely a prefix of a catalog alias."""
    result = _repository(tmp_path).resolve_companies(Sector.TECH, query)

    assert result.resolved == []


def test_repository_does_not_match_app_as_apple(tmp_path: Path) -> None:
    """Apply token boundaries to configured aliases, not only tickers."""
    database = tmp_path / "finagent.db"
    build_database(
        FIXTURES / "source_manifest.yaml",
        FIXTURES / "raw" / "sec",
        database,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO company_aliases(company_id, alias) VALUES (?, ?)",
            ("sec:0000000001", "Apple"),
        )
        connection.commit()
    repository = SectorRepository(database)

    assert repository.resolve_companies(Sector.TECH, "Tell me about Apple.").resolved
    assert (
        repository.resolve_companies(Sector.TECH, "Tell me about App.").resolved == []
    )


@pytest.mark.parametrize(
    ("query", "mention"),
    [
        ("What do you think about SpaceX?", "SpaceX"),
        ("what do you think about spacex?", "spacex"),
        ("Tell me about spacex.", "spacex"),
        ("What is the revenue for spacex!", "spacex"),
        ("How is spacex performing?", "spacex"),
    ],
)
def test_repository_detects_explicit_unknown_without_capitalization(
    tmp_path: Path, query: str, mention: str
) -> None:
    """Detect explicit absent companies using phrasing rather than capitalization."""
    result = _repository(tmp_path).resolve_companies(Sector.TECH, query)

    assert result.resolved == []
    assert result.unresolved_mentions == [mention]


@pytest.mark.parametrize(
    "query",
    [
        "Compare the companies' margins.",
        "Which company is strongest?",
        "Compare companies in the dataset.",
        "What are the risks in this sector?",
        "What is the latest headcount for a company in this dataset?",
        "What is the latest headcount or hiring signal for a company in this dataset?",
    ],
)
def test_repository_keeps_broad_queries_in_scope(tmp_path: Path, query: str) -> None:
    """Leave broad sector questions unresolved so all sector companies are eligible."""
    result = _repository(tmp_path).resolve_companies(Sector.TECH, query)

    assert result.resolved == []
    assert result.unresolved_mentions == []


def test_repository_expands_derived_source_lineage(tmp_path: Path) -> None:
    """Return constituent SEC sources with a derived benchmark source."""
    database = tmp_path / "finagent.db"
    build_database(
        FIXTURES / "source_manifest.yaml",
        FIXTURES / "raw" / "sec",
        database,
    )
    with sqlite3.connect(database) as connection:
        input_source_id = connection.execute(
            "SELECT source_id FROM annual_financial_snapshots LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO sources(
                id, publisher, title, url, retrieved_at, raw_sha256
            ) VALUES (
                'derived:test', 'FinAgent derived from U.S. SEC filings',
                'Test median', 'https://www.sec.gov/edgar/search/',
                '2025-03-01', 'test-digest'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO source_lineage(derived_source_id, input_source_id)
            VALUES ('derived:test', ?)
            """,
            (input_source_id,),
        )
        connection.execute(
            """
            INSERT INTO sector_benchmarks(
                id, sector_id, as_of, metric, value, unit,
                quality_caveat, source_id
            ) VALUES (
                'benchmark:test', 'tech', '2024-12-31', 'revenue', 1200,
                'USD', 'Test derived median.', 'derived:test'
            )
            """
        )
        connection.commit()

    result = SectorRepository(database).query_observations(
        Sector.TECH,
        ["benchmark:tech"],
        ["revenue"],
        latest_only=True,
        limit=100,
    )

    assert {source.source_id for source in result.sources} == {
        "derived:test",
        input_source_id,
    }
