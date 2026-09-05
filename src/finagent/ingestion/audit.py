"""Audit the coverage and provenance of a built FinAgent database."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class DataAudit:
    """Coverage and integrity counts for one database."""

    sectors: int
    companies: int
    annual_snapshots: int
    market_snapshots: int
    operating_signals: int
    benchmark_observations: int
    sources: int
    lineage_links: int
    illustrative_sources: int
    orphaned_source_references: int

    @property
    def has_real_enrichment(self) -> bool:
        """Return whether all real-data enrichment gates pass."""
        return (
            self.sectors == 3
            and self.companies > 0
            and self.annual_snapshots > 0
            and self.market_snapshots > 0
            and self.operating_signals > 0
            and self.benchmark_observations > 0
            and self.illustrative_sources == 0
            and self.orphaned_source_references == 0
        )


def audit_database(
    database_path: Path | str,
    *,
    require_real_enrichment: bool = False,
) -> DataAudit:
    """Inspect dataset coverage without modifying the database.

    Parameters
    ----------
    database_path
        SQLite database produced by the offline builder.
    require_real_enrichment
        Raise when any required real-data table is empty or fixture data is found.

    Returns
    -------
    DataAudit
        Row coverage and provenance-integrity counts.

    Raises
    ------
    FileNotFoundError
        If the database does not exist.
    ValueError
        If strict real-data coverage requirements are not met.
    """
    database = Path(database_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")
    uri = f"file:{quote(str(database))}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        counts = {
            table: _table_count(connection, table)
            for table in (
                "sectors",
                "companies",
                "annual_financial_snapshots",
                "market_snapshots",
                "operating_signals",
                "sector_benchmarks",
                "sources",
                "source_lineage",
            )
        }
        illustrative_sources = connection.execute(
            """
            SELECT COUNT(*) FROM sources
            WHERE publisher = 'FinAgent sample fixture'
               OR lower(title) LIKE '%illustrative%'
               OR url LIKE 'https://example.com/%'
            """
        ).fetchone()[0]
        orphaned = sum(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM {table} AS row
                LEFT JOIN sources AS source ON source.id = row.source_id
                WHERE source.id IS NULL
                """
            ).fetchone()[0]
            for table in (
                "annual_financial_snapshots",
                "market_snapshots",
                "operating_signals",
                "sector_benchmarks",
            )
        )
    audit = DataAudit(
        sectors=counts["sectors"],
        companies=counts["companies"],
        annual_snapshots=counts["annual_financial_snapshots"],
        market_snapshots=counts["market_snapshots"],
        operating_signals=counts["operating_signals"],
        benchmark_observations=counts["sector_benchmarks"],
        sources=counts["sources"],
        lineage_links=counts["source_lineage"],
        illustrative_sources=illustrative_sources,
        orphaned_source_references=orphaned,
    )
    if require_real_enrichment and not audit.has_real_enrichment:
        raise ValueError(f"real-data enrichment audit failed: {audit}")
    return audit


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
