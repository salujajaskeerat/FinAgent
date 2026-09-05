"""Create a deterministic, clearly labelled local dataset for development."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .builder import SCHEMA_SQL
from .manifest import CompanySpec, SectorSpec, load_manifest

_CAVEAT = (
    "Illustrative fixture; replace with verified SEC/IR data before investment use."
)
_BENCHMARK_METRICS = (
    "revenue",
    "operating_income",
    "operating_margin",
    "free_cash_flow",
    "capital_expenditure",
    "cash_and_equivalents",
    "total_debt",
    "market_cap",
    "enterprise_value",
)


@dataclass(frozen=True, slots=True)
class SampleBuildStats:
    """Counts emitted by a deterministic sample database build."""

    sectors: int
    companies: int
    annual_snapshots: int
    market_snapshots: int
    operating_signals: int
    benchmark_observations: int
    dataset_version: str
    output_path: Path


def build_sample_database(
    manifest_path: Path | str,
    output_path: Path | str,
) -> SampleBuildStats:
    """Build a complete local fixture without network access.

    Parameters
    ----------
    manifest_path
        Curated sector and company manifest.
    output_path
        SQLite file to create or atomically replace.

    Returns
    -------
    SampleBuildStats
        Counts and deterministic dataset version for the generated file.
    """
    manifest = load_manifest(manifest_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    dataset_version = hashlib.sha256(
        Path(manifest_path).read_bytes() + b"\0finagent-sample-v1"
    ).hexdigest()[:16]
    annual_count = 0
    market_count = 0
    signal_count = 0
    benchmark_count = 0
    company_number = 0
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript(SCHEMA_SQL)
            for sector in manifest.sectors:
                connection.execute(
                    """
                    INSERT INTO sectors(id, name, benchmark_name, benchmark_ticker)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        sector.id,
                        sector.name,
                        sector.benchmark_name,
                        sector.benchmark_ticker,
                    ),
                )
                for company in sector.companies:
                    company_number += 1
                    company_id = _company_id(company)
                    connection.execute(
                        """
                        INSERT INTO companies(id, sector_id, name, ticker, cik)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            company_id,
                            sector.id,
                            company.name,
                            company.ticker,
                            company.padded_cik,
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO company_aliases(company_id, alias) VALUES (?, ?)",
                        ((company_id, alias) for alias in company.aliases),
                    )
                    annual_count += _insert_annual_snapshots(
                        connection, sector, company, company_id, company_number
                    )
                    market_count += _insert_market_snapshot(
                        connection, sector, company, company_id, company_number
                    )
                    signal_count += _insert_operating_signals(
                        connection, sector, company, company_id, company_number
                    )
                benchmark_count += _insert_benchmarks(
                    connection, sector, company_number
                )
            connection.executemany(
                "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
                (
                    ("dataset_version", dataset_version),
                    ("schema_version", "1"),
                    ("manifest_version", str(manifest.version)),
                    ("dataset_kind", "illustrative_sample"),
                ),
            )
            connection.commit()
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return SampleBuildStats(
        sectors=len(manifest.sectors),
        companies=len(manifest.iter_companies()),
        annual_snapshots=annual_count,
        market_snapshots=market_count,
        operating_signals=signal_count,
        benchmark_observations=benchmark_count,
        dataset_version=dataset_version,
        output_path=target,
    )


def _company_id(company: CompanySpec) -> str:
    return f"sec:{company.padded_cik}"


def _source(
    connection: sqlite3.Connection,
    source_id: str,
    title: str,
    url: str,
    published_at: str,
) -> None:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    connection.execute(
        """
        INSERT INTO sources(
            id, publisher, title, url, published_at, retrieved_at, raw_sha256
        ) VALUES (?, 'FinAgent sample fixture', ?, ?, ?, '2026-01-01', ?)
        """,
        (source_id, title, url, published_at, digest),
    )


def _insert_annual_snapshots(
    connection: sqlite3.Connection,
    sector: SectorSpec,
    company: CompanySpec,
    company_id: str,
    ordinal: int,
) -> int:
    for fiscal_year in (2023, 2024):
        source_id = f"sample:annual:{company.ticker}:{fiscal_year}"
        _source(
            connection,
            source_id,
            f"{company.name} illustrative annual snapshot {fiscal_year}",
            f"https://example.com/finagent/sample/{sector.id}/{company.ticker}/{fiscal_year}",
            f"{fiscal_year + 1}-02-15",
        )
        revenue = (
            1_000_000_000 + ordinal * 125_000_000 + (fiscal_year - 2023) * 90_000_000
        )
        operating_income = revenue * (0.14 + ordinal * 0.002)
        operating_cash_flow = revenue * (0.18 + ordinal * 0.003)
        capex = revenue * (0.045 + ordinal * 0.001)
        connection.execute(
            """
            INSERT INTO annual_financial_snapshots(
                id, company_id, fiscal_year, period_end, filed_at, currency,
                revenue, operating_income, operating_margin, operating_cash_flow,
                capital_expenditure, free_cash_flow, cash_and_equivalents, total_debt,
                quality_caveat, source_id
            ) VALUES (?, ?, ?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"sample:annual-row:{company.ticker}:{fiscal_year}",
                company_id,
                fiscal_year,
                f"{fiscal_year}-12-31",
                f"{fiscal_year + 1}-02-15",
                revenue,
                operating_income,
                operating_income / revenue,
                operating_cash_flow,
                capex,
                operating_cash_flow - capex,
                revenue * (0.22 + ordinal * 0.004),
                revenue * (0.16 - ordinal * 0.002),
                _CAVEAT,
                source_id,
            ),
        )
    return 2


def _insert_market_snapshot(
    connection: sqlite3.Connection,
    sector: SectorSpec,
    company: CompanySpec,
    company_id: str,
    ordinal: int,
) -> int:
    source_id = f"sample:market:{company.ticker}:2025-06-30"
    _source(
        connection,
        source_id,
        f"{company.name} illustrative market snapshot",
        f"https://example.com/finagent/sample/{sector.id}/{company.ticker}/market-2025-06-30",
        "2025-07-01",
    )
    market_cap = 20_000_000_000 + ordinal * 3_000_000_000
    connection.execute(
        """
        INSERT INTO market_snapshots(
            id, company_id, as_of, currency, share_price, market_cap,
            enterprise_value, quality_caveat, source_id
        ) VALUES (?, ?, '2025-06-30', 'USD', ?, ?, ?, ?, ?)
        """,
        (
            f"sample:market-row:{company.ticker}:2025-06-30",
            company_id,
            50 + ordinal * 7.5,
            market_cap,
            market_cap * 1.08,
            _CAVEAT,
            source_id,
        ),
    )
    return 1


def _insert_operating_signals(
    connection: sqlite3.Connection,
    sector: SectorSpec,
    company: CompanySpec,
    company_id: str,
    ordinal: int,
) -> int:
    signals = (
        (
            "headcount",
            10_000 + ordinal * 350,
            f"{10_000 + ordinal * 350} employees",
            "employees",
            "2025-06-30",
        ),
        (
            "guidance",
            None,
            "Illustrative demand and margin outlook",
            None,
            "2025-05-15",
        ),
        (
            "restructuring",
            None,
            "Illustrative efficiency initiative signal",
            None,
            "2025-04-30",
        ),
    )
    for signal_type, numeric, text, unit, observed_at in signals:
        source_id = f"sample:signal:{company.ticker}:{signal_type}:2025"
        _source(
            connection,
            source_id,
            f"{company.name} illustrative {signal_type} signal",
            f"https://example.com/finagent/sample/{sector.id}/{company.ticker}/{signal_type}-2025",
            "2025-07-01",
        )
        connection.execute(
            """
            INSERT INTO operating_signals(
                id, company_id, signal_type, observed_at, value_numeric,
                value_text, unit, quality_caveat, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"sample:signal-row:{company.ticker}:{signal_type}:2025",
                company_id,
                signal_type,
                observed_at,
                numeric,
                text,
                unit,
                _CAVEAT,
                source_id,
            ),
        )
    return len(signals)


def _insert_benchmarks(
    connection: sqlite3.Connection,
    sector: SectorSpec,
    ordinal: int,
) -> int:
    source_id = f"sample:benchmark:{sector.id}:2025-06-30"
    _source(
        connection,
        source_id,
        f"{sector.name} illustrative benchmark snapshot",
        f"https://example.com/finagent/sample/{sector.id}/benchmark-2025-06-30",
        "2025-07-01",
    )
    values = {
        "revenue": 1_000_000_000 + ordinal * 100_000_000,
        "operating_income": 150_000_000 + ordinal * 12_000_000,
        "operating_margin": 0.16,
        "free_cash_flow": 125_000_000 + ordinal * 10_000_000,
        "capital_expenditure": 45_000_000 + ordinal * 4_000_000,
        "cash_and_equivalents": 225_000_000 + ordinal * 15_000_000,
        "total_debt": 175_000_000 + ordinal * 12_000_000,
        "market_cap": 25_000_000_000 + ordinal * 2_500_000_000,
        "enterprise_value": 27_000_000_000 + ordinal * 2_700_000_000,
    }
    for metric in _BENCHMARK_METRICS:
        connection.execute(
            """
            INSERT INTO sector_benchmarks(
                id, sector_id, as_of, metric, value, unit, quality_caveat, source_id
            ) VALUES (?, ?, '2025-06-30', ?, ?, ?, ?, ?)
            """,
            (
                f"sample:benchmark-row:{sector.id}:{metric}:2025-06-30",
                sector.id,
                metric,
                values[metric],
                "ratio" if metric == "operating_margin" else "USD",
                _CAVEAT,
                source_id,
            ),
        )
    return len(_BENCHMARK_METRICS)
