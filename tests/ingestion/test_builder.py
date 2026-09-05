"""Offline integration tests for the SQLite builder."""

import sqlite3
from pathlib import Path

from finagent.ingestion.builder import build_database

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_database_from_cached_sec_fixtures(tmp_path: Path) -> None:
    output = tmp_path / "finagent.db"

    stats = build_database(
        FIXTURES / "source_manifest.yaml",
        FIXTURES / "raw" / "sec",
        output,
    )

    assert stats.sectors == 1
    assert stats.companies == 1
    assert stats.annual_snapshots == 2
    assert stats.operating_signals == 2
    assert stats.skipped_companies == 0
    assert len(stats.dataset_version) == 16

    with sqlite3.connect(output) as connection:
        connection.row_factory = sqlite3.Row
        latest = connection.execute(
            """
            SELECT * FROM annual_financial_snapshots
            ORDER BY period_end DESC LIMIT 1
            """
        ).fetchone()
        assert latest["revenue"] == 1200
        assert latest["operating_margin"] == 0.15
        assert latest["free_cash_flow"] == 170
        assert latest["total_debt"] == 200

        headcount = connection.execute(
            """
            SELECT value_numeric, observed_at FROM operating_signals
            ORDER BY observed_at DESC LIMIT 1
            """
        ).fetchone()
        assert headcount["value_numeric"] == 950
        assert headcount["observed_at"] == "2024-12-31"

        source = connection.execute(
            """
            SELECT s.url
            FROM annual_financial_snapshots AS f
            JOIN sources AS s ON s.id = f.source_id
            WHERE f.fiscal_year = 2024
            """
        ).fetchone()
        assert "000000000125000001/example-20241231.htm" in source["url"]
        version = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key = 'dataset_version'"
        ).fetchone()[0]
        assert version == stats.dataset_version

    # Rebuilding replaces rather than appends, producing the same domain row counts.
    build_database(
        FIXTURES / "source_manifest.yaml",
        FIXTURES / "raw" / "sec",
        output,
    )
    with sqlite3.connect(output) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM annual_financial_snapshots"
            ).fetchone()[0]
            == 2
        )


def test_build_allow_missing_keeps_declared_universe(tmp_path: Path) -> None:
    output = tmp_path / "partial.db"

    stats = build_database(
        FIXTURES / "source_manifest.yaml",
        tmp_path / "empty-cache",
        output,
        strict=False,
    )

    assert stats.companies == 1
    assert stats.skipped_companies == 1
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM annual_financial_snapshots"
            ).fetchone()[0]
            == 0
        )
