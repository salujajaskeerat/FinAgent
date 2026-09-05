"""Repository integration against the real offline builder schema."""

import sqlite3
from pathlib import Path

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
