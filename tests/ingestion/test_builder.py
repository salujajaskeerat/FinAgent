"""Offline integration tests for the SQLite builder."""

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from finagent.ingestion.builder import _extract_public_float, build_database

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


def test_build_enriches_from_cached_annual_report(tmp_path: Path) -> None:
    """Populate missing market and headcount data without network access."""
    raw_dir = tmp_path / "raw" / "sec"
    company_dir = raw_dir / "0000000001"
    shutil.copytree(FIXTURES / "raw" / "sec" / "0000000001", company_dir)
    companyfacts_path = company_dir / "companyfacts.json"
    companyfacts = json.loads(companyfacts_path.read_text(encoding="utf-8"))
    companyfacts["facts"]["dei"].pop("EntityNumberOfEmployees")
    companyfacts_path.write_text(
        json.dumps(companyfacts, sort_keys=True), encoding="utf-8"
    )
    companyfacts_digest = hashlib.sha256(companyfacts_path.read_bytes()).hexdigest()
    metadata_path = company_dir / "companyfacts.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sha256"] = companyfacts_digest
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    report_dir = company_dir / "filings" / "000000000125000001"
    report_dir.mkdir(parents=True)
    report = report_dir / "example-20241231.htm"
    report.write_text(
        """
        <html><body>
        <p>The Company employed approximately 1.275 thousand associates.</p>
        <p>The aggregate market value of voting stock held by non-affiliates as of
        June 28, 2024 was $1.5 billion, based on the closing price of $42.75 per
        share.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    report_metadata = {
        "accession_number": "0000000001-25-000001",
        "filed_at": "2025-02-15",
        "form": "10-K",
        "local_filename": report.name,
        "report_date": "2024-12-31",
        "retrieved_at": "2025-03-01",
        "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "url": (
            "https://www.sec.gov/Archives/edgar/data/1/"
            "000000000125000001/example-20241231.htm"
        ),
    }
    report.with_suffix(".metadata.json").write_text(
        json.dumps(report_metadata, sort_keys=True), encoding="utf-8"
    )
    output = tmp_path / "enriched.db"

    stats = build_database(FIXTURES / "source_manifest.yaml", raw_dir, output)

    assert stats.market_snapshots == 1
    assert stats.operating_signals == 1
    assert stats.benchmark_observations == 0
    with sqlite3.connect(output) as connection:
        market = connection.execute(
            "SELECT as_of, share_price, public_float, market_cap FROM market_snapshots"
        ).fetchone()
        assert market == ("2024-06-28", 42.75, 1.5e9, None)
        headcount = connection.execute(
            "SELECT observed_at, value_numeric FROM operating_signals"
        ).fetchone()
        assert headcount == ("2024-12-31", 1275)
        assert (
            connection.execute("SELECT COUNT(*) FROM sector_benchmarks").fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sources WHERE id LIKE 'derived:%'"
            ).fetchone()[0]
            == 0
        )


def test_benchmarks_require_three_constituents_and_record_lineage(
    tmp_path: Path,
) -> None:
    """Emit derived medians only with adequate source-linked coverage."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
        version: 1
        sectors:
          - id: tech
            name: Technology
            benchmark: {name: Example Benchmark, ticker: EXB}
            companies:
              - {name: Example One, ticker: EX1, cik: "1"}
              - {name: Example Two, ticker: EX2, cik: "2"}
              - {name: Example Three, ticker: EX3, cik: "3"}
        """,
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    fixture_company = FIXTURES / "raw" / "sec" / "0000000001"
    for cik in ("0000000001", "0000000002", "0000000003"):
        shutil.copytree(fixture_company, raw_dir / cik)

    stats = build_database(manifest, raw_dir, tmp_path / "benchmarks.db")

    assert stats.benchmark_observations == 8
    with sqlite3.connect(stats.output_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM source_lineage").fetchone()[0] == 3
        )
        revenue = connection.execute(
            "SELECT value FROM sector_benchmarks WHERE metric = 'revenue'"
        ).fetchone()[0]
        assert revenue == 1200


def test_build_keeps_only_cited_sources(tmp_path: Path) -> None:
    """Every stored source must be cited by a fact, signal, benchmark, or lineage link."""
    output = tmp_path / "finagent.db"
    build_database(FIXTURES / "source_manifest.yaml", FIXTURES / "raw" / "sec", output)

    with sqlite3.connect(output) as connection:
        uncited = connection.execute(
            """
            SELECT COUNT(*) FROM sources
            WHERE id NOT IN (
                SELECT source_id FROM annual_financial_snapshots
                UNION SELECT source_id FROM market_snapshots
                UNION SELECT source_id FROM operating_signals
                UNION SELECT source_id FROM sector_benchmarks
                UNION SELECT derived_source_id FROM source_lineage
                UNION SELECT input_source_id FROM source_lineage
            )
            """
        ).fetchone()[0]
        cited = connection.execute(
            "SELECT COUNT(DISTINCT source_id) FROM annual_financial_snapshots"
        ).fetchone()[0]
    assert uncited == 0
    assert cited >= 1


_TARGET = (
    "aggregate market value of the voting stock held by non-affiliates of the "
    "registrant as of August 1, 2025, wa s $ 45,284,343,058 based on the closing "
    "price of $99.77 per share."
)
_HOME_DEPOT = (
    "aggregate market value of voting common stock held by non-affiliates of the "
    "registrant on August 1, 2025 was $ 371.8 billion. Shares outstanding as of "
    "March 4, 2026 was 996,011,466."
)
_MICROSOFT = (
    "As of December 31, 2025, the aggregate market value of the registrant's "
    "common stock held by non-affiliates was $ 3.6 trillion based on the closing "
    "sale price. As of July 23, 2026, there were 7,425,545,491 shares outstanding."
)
_ADOBE = (
    "aggregate market value of the registrant's common stock, $0.0001 par value "
    "per share, held by non-affiliates on May 30, 2025 was $ 144.94 billion."
)
_UPS = (
    "aggregate market value of the class B common stock held by non-affiliates "
    "was $ 74,239,665,022 as of June 30, 2025."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_TARGET, ("2025-08-01", 45_284_343_058.0)),
        (_HOME_DEPOT, ("2025-08-01", 371.8e9)),
        (_MICROSOFT, ("2025-12-31", 3.6e12)),
        (_ADOBE, ("2025-05-30", 144.94e9)),
        (_UPS, ("2025-06-30", 74_239_665_022.0)),
        ("The registrant is a smaller reporting company.", None),
    ],
)
def test_public_float_parser_handles_real_cover_page_phrasings(
    text: str, expected: tuple[str, float] | None
) -> None:
    assert _extract_public_float(text) == expected
