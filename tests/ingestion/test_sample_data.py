"""Tests for the deterministic all-sector sample database builder."""

import sqlite3
from pathlib import Path

from finagent.ingestion.sample_data import build_sample_database

MANIFEST = Path(__file__).parents[2] / "data" / "source_manifest.yaml"


def test_sample_builder_populates_all_sectors_and_schema(tmp_path: Path) -> None:
    """Build all configured sectors with source-linked records."""
    output = tmp_path / "finagent.db"
    stats = build_sample_database(MANIFEST, output)

    assert stats.sectors == 3
    assert stats.companies == 12
    assert stats.annual_snapshots == 24
    assert stats.market_snapshots == 12
    assert stats.operating_signals == 36
    assert stats.benchmark_observations == 27
    assert len(stats.dataset_version) == 16

    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sectors").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 12
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sources WHERE publisher = 'FinAgent sample fixture'"
            ).fetchone()[0]
            == 75
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM operating_signals WHERE signal_type = 'headcount'"
            ).fetchone()[0]
            == 12
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sector_benchmarks WHERE metric = 'operating_margin'"
            ).fetchone()[0]
            == 3
        )
        caveat = connection.execute(
            "SELECT quality_caveat FROM annual_financial_snapshots LIMIT 1"
        ).fetchone()[0]
        assert "Illustrative fixture" in caveat


def test_sample_builder_is_deterministic_and_replaces_output(tmp_path: Path) -> None:
    """Repeated builds have the same version and do not append rows."""
    output = tmp_path / "finagent.db"
    first = build_sample_database(MANIFEST, output)
    second = build_sample_database(MANIFEST, output)

    assert first.dataset_version == second.dataset_version
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 12
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM annual_financial_snapshots"
            ).fetchone()[0]
            == 24
        )
