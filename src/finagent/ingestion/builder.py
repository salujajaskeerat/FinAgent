"""Build the runtime SQLite database exclusively from cached source files."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Any

from .manifest import CompanySpec, SectorSpec, load_manifest

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE dataset_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE sectors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    benchmark_ticker TEXT NOT NULL
);

CREATE TABLE companies (
    id TEXT PRIMARY KEY,
    sector_id TEXT NOT NULL REFERENCES sectors(id),
    name TEXT NOT NULL,
    ticker TEXT NOT NULL UNIQUE,
    cik TEXT NOT NULL UNIQUE
);

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    publisher TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    accession_number TEXT,
    raw_sha256 TEXT NOT NULL
);

CREATE TABLE source_lineage (
    derived_source_id TEXT NOT NULL REFERENCES sources(id),
    input_source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY(derived_source_id, input_source_id)
);

CREATE TABLE annual_financial_snapshots (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    fiscal_year INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    filed_at TEXT,
    currency TEXT NOT NULL DEFAULT 'USD',
    revenue REAL,
    operating_income REAL,
    operating_margin REAL,
    operating_cash_flow REAL,
    capital_expenditure REAL,
    free_cash_flow REAL,
    cash_and_equivalents REAL,
    total_debt REAL,
    quality_caveat TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id),
    UNIQUE(company_id, fiscal_year)
);

CREATE TABLE market_snapshots (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    as_of TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    share_price REAL,
    market_cap REAL,
    enterprise_value REAL,
    quality_caveat TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id),
    UNIQUE(company_id, as_of)
);

CREATE TABLE operating_signals (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL REFERENCES companies(id),
    signal_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    value_numeric REAL,
    value_text TEXT,
    unit TEXT,
    quality_caveat TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id),
    UNIQUE(company_id, signal_type, observed_at)
);

CREATE TABLE sector_benchmarks (
    id TEXT PRIMARY KEY,
    sector_id TEXT NOT NULL REFERENCES sectors(id),
    as_of TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    quality_caveat TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id),
    UNIQUE(sector_id, as_of, metric)
);

CREATE INDEX idx_companies_sector ON companies(sector_id);
CREATE INDEX idx_financials_company_period
    ON annual_financial_snapshots(company_id, period_end DESC);
CREATE INDEX idx_signals_company_date
    ON operating_signals(company_id, observed_at DESC);
"""


FLOW_METRICS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ),
}

INSTANT_METRICS: dict[str, tuple[str, ...]] = {
    "cash_and_equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "total_debt": (
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ),
}

HEADCOUNT_TAGS = ("EntityNumberOfEmployees", "NumberOfEmployees")
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
MIN_BENCHMARK_CONSTITUENTS = 3


@dataclass(frozen=True, slots=True)
class BuildStats:
    """Counts emitted by one deterministic database build."""

    sectors: int
    companies: int
    annual_snapshots: int
    market_snapshots: int
    operating_signals: int
    benchmark_observations: int
    skipped_companies: int
    dataset_version: str
    output_path: Path


@dataclass(frozen=True, slots=True)
class FactValue:
    """One selected annual SEC Companyfacts observation."""

    value: float
    end: str
    filed: str | None
    accession: str | None


def build_database(
    manifest_path: Path | str,
    raw_dir: Path | str,
    output_path: Path | str,
    *,
    annual_periods: int = 3,
    strict: bool = True,
) -> BuildStats:
    """Build a new SQLite database from cached SEC JSON.

    The function never accesses the network. It builds into a temporary file and
    atomically replaces ``output_path`` only after the complete transaction succeeds.

    Parameters
    ----------
    manifest_path:
        Curated YAML source manifest.
    raw_dir:
        Directory containing ``<CIK>/submissions.json`` and
        ``<CIK>/companyfacts.json`` cache files.
    output_path:
        SQLite database to create or atomically replace.
    annual_periods:
        Maximum latest annual periods retained per company.
    strict:
        If true, fail on any missing company cache. If false, skip missing companies
        while retaining their universe metadata.

    Returns
    -------
    BuildStats
        Inserted row counts and output path.

    Raises
    ------
    FileNotFoundError
        If a required cache file is absent in strict mode.
    ValueError
        If ``annual_periods`` is not positive or cached JSON is malformed.
    """
    if annual_periods <= 0:
        raise ValueError("annual_periods must be greater than 0")

    manifest = load_manifest(manifest_path)
    dataset_digest = sha256(Path(manifest_path).read_bytes())
    cache_root = Path(raw_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    annual_count = 0
    market_count = 0
    signal_count = 0
    benchmark_count = 0
    skipped_count = 0
    try:
        with sqlite3.connect(temporary) as connection:
            connection.row_factory = sqlite3.Row
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
                    cache_dir = cache_root / company.padded_cik
                    submissions_path = cache_dir / "submissions.json"
                    facts_path = cache_dir / "companyfacts.json"
                    facts_metadata_path = cache_dir / "companyfacts.metadata.json"
                    missing = [
                        path
                        for path in (
                            submissions_path,
                            facts_path,
                            facts_metadata_path,
                        )
                        if not path.exists()
                    ]
                    if missing:
                        if strict:
                            raise FileNotFoundError(
                                "missing cached SEC file(s): "
                                + ", ".join(str(path) for path in missing)
                            )
                        skipped_count += 1
                        dataset_digest.update(
                            f"missing:{company.padded_cik}".encode("ascii")
                        )
                        continue

                    submissions = _load_mapping(submissions_path)
                    companyfacts = _load_mapping(facts_path)
                    facts_metadata = _load_mapping(facts_metadata_path)
                    retrieved_at = facts_metadata.get("retrieved_at")
                    if not isinstance(retrieved_at, str) or not retrieved_at:
                        raise ValueError(
                            f"missing retrieved_at in cache metadata: {facts_metadata_path}"
                        )
                    raw_digest = sha256(facts_path.read_bytes()).hexdigest()
                    if facts_metadata.get("sha256") != raw_digest:
                        raise ValueError(
                            f"cache digest does not match metadata: {facts_path}"
                        )
                    dataset_digest.update(submissions_path.read_bytes())
                    dataset_digest.update(facts_path.read_bytes())
                    filing_sources = _insert_sources(
                        connection,
                        company,
                        submissions,
                        raw_digest=raw_digest,
                        retrieved_at=retrieved_at,
                    )
                    annual_count += _insert_annual_snapshots(
                        connection,
                        company,
                        companyfacts,
                        filing_sources,
                        annual_periods=annual_periods,
                    )
                    company_signal_count = _insert_headcount_signals(
                        connection,
                        company,
                        companyfacts,
                        filing_sources,
                        annual_periods=annual_periods,
                    )
                    signal_count += company_signal_count
                    cached_report = _latest_cached_annual_report(cache_dir)
                    if cached_report is not None:
                        report_path, report_metadata_path, report_metadata = (
                            cached_report
                        )
                        dataset_digest.update(report_path.read_bytes())
                        dataset_digest.update(report_metadata_path.read_bytes())
                        source_id = _upsert_annual_report_source(
                            connection,
                            company,
                            report_path,
                            report_metadata,
                        )
                        filing_text = _html_text(
                            report_path.read_text(encoding="utf-8", errors="replace")
                        )
                        if company_signal_count == 0:
                            signal_count += _insert_report_headcount(
                                connection,
                                company,
                                filing_text,
                                report_metadata,
                                source_id,
                            )
                        market_count += _insert_report_market_snapshot(
                            connection,
                            company,
                            filing_text,
                            source_id,
                        )
            benchmark_count = _insert_sector_benchmarks(connection, manifest.sectors)
            dataset_version = dataset_digest.hexdigest()[:16]
            connection.executemany(
                "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
                (
                    ("dataset_version", dataset_version),
                    ("schema_version", "1"),
                    ("manifest_version", str(manifest.version)),
                ),
            )
            connection.commit()
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return BuildStats(
        sectors=len(manifest.sectors),
        companies=len(manifest.iter_companies()),
        annual_snapshots=annual_count,
        market_snapshots=market_count,
        operating_signals=signal_count,
        benchmark_observations=benchmark_count,
        skipped_companies=skipped_count,
        dataset_version=dataset_version,
        output_path=target,
    )


def _insert_sources(
    connection: sqlite3.Connection,
    company: CompanySpec,
    submissions: dict[str, Any],
    *,
    raw_digest: str,
    retrieved_at: str,
) -> dict[str, str]:
    companyfacts_source_id = f"sec-companyfacts:{company.padded_cik}"
    connection.execute(
        """
        INSERT INTO sources(
            id, publisher, title, url, retrieved_at, raw_sha256
        ) VALUES (?, 'U.S. Securities and Exchange Commission', ?, ?, ?, ?)
        """,
        (
            companyfacts_source_id,
            f"{company.name} SEC Companyfacts",
            (
                "https://data.sec.gov/api/xbrl/companyfacts/"
                f"CIK{company.padded_cik}.json"
            ),
            retrieved_at,
            raw_digest,
        ),
    )

    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return {"__fallback__": companyfacts_source_id}
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])
    source_by_accession = {"__fallback__": companyfacts_source_id}
    for index, accession in enumerate(accessions):
        if not isinstance(accession, str) or not accession:
            continue
        form = _list_value(forms, index) or "filing"
        filed_at = _list_value(filing_dates, index)
        document = _list_value(primary_documents, index)
        accession_compact = accession.replace("-", "")
        url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(company.padded_cik)}/{accession_compact}/{document}"
            if document
            else f"https://www.sec.gov/edgar/browse/?CIK={int(company.padded_cik)}"
        )
        source_id = f"sec-filing:{company.padded_cik}:{accession}"
        connection.execute(
            """
            INSERT INTO sources(
                id, publisher, title, url, published_at,
                retrieved_at, accession_number, raw_sha256
            ) VALUES (?, 'U.S. Securities and Exchange Commission', ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                f"{company.name} {form} filed {filed_at or 'unknown date'}",
                url,
                filed_at,
                retrieved_at,
                accession,
                raw_digest,
            ),
        )
        source_by_accession[accession] = source_id
    return source_by_accession


def _insert_annual_snapshots(
    connection: sqlite3.Connection,
    company: CompanySpec,
    companyfacts: dict[str, Any],
    filing_sources: dict[str, str],
    *,
    annual_periods: int,
) -> int:
    values = {
        name: _extract_metric(companyfacts, tags, flow=True)
        for name, tags in FLOW_METRICS.items()
    }
    values.update(
        {
            name: _extract_metric(companyfacts, tags, flow=False)
            for name, tags in INSTANT_METRICS.items()
            if name != "total_debt"
        }
    )
    values["total_debt"] = _extract_total_debt(companyfacts)
    period_ends = sorted(
        {end for metric in values.values() for end in metric}, reverse=True
    )[:annual_periods]

    inserted = 0
    for end in sorted(period_ends):
        row = {name: metric.get(end) for name, metric in values.items()}
        if not any(row.values()):
            continue
        revenue = _number(row["revenue"])
        operating_income = _number(row["operating_income"])
        operating_margin = (
            operating_income / revenue
            if revenue not in (None, 0) and operating_income is not None
            else None
        )
        cash_flow = _number(row["operating_cash_flow"])
        capex = _number(row["capital_expenditure"])
        free_cash_flow = (
            cash_flow - capex if cash_flow is not None and capex is not None else None
        )
        provenance = next(
            (
                value
                for value in (
                    row["revenue"],
                    row["operating_income"],
                    row["operating_cash_flow"],
                )
                if value is not None
            ),
            next(value for value in row.values() if value is not None),
        )
        source_id = filing_sources.get(
            provenance.accession or "", filing_sources["__fallback__"]
        )
        company_id = _company_id(company)
        fiscal_year = int(end[:4])
        connection.execute(
            """
            INSERT INTO annual_financial_snapshots(
                id, company_id, fiscal_year, period_end, filed_at, currency,
                revenue, operating_income, operating_margin,
                operating_cash_flow, capital_expenditure, free_cash_flow,
                cash_and_equivalents, total_debt, quality_caveat, source_id
            ) VALUES (?, ?, ?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"annual:{company.padded_cik}:{fiscal_year}",
                company_id,
                fiscal_year,
                end,
                provenance.filed,
                revenue,
                operating_income,
                operating_margin,
                cash_flow,
                capex,
                free_cash_flow,
                _number(row["cash_and_equivalents"]),
                _number(row["total_debt"]),
                "Normalized from SEC Companyfacts; issuer taxonomy may vary.",
                source_id,
            ),
        )
        inserted += 1
    return inserted


def _insert_headcount_signals(
    connection: sqlite3.Connection,
    company: CompanySpec,
    companyfacts: dict[str, Any],
    filing_sources: dict[str, str],
    *,
    annual_periods: int,
) -> int:
    headcount = _extract_metric(
        companyfacts,
        HEADCOUNT_TAGS,
        flow=False,
        namespaces=("dei", "us-gaap"),
        allowed_units=("employees", "Employee", "shares", "pure"),
    )
    inserted = 0
    for end in sorted(headcount, reverse=True)[:annual_periods]:
        fact = headcount[end]
        source_id = filing_sources.get(
            fact.accession or "", filing_sources["__fallback__"]
        )
        connection.execute(
            """
            INSERT INTO operating_signals(
                id, company_id, signal_type, observed_at,
                value_numeric, unit, quality_caveat, source_id
            ) VALUES (?, ?, 'headcount', ?, ?, 'employees', ?, ?)
            """,
            (
                f"signal:{company.padded_cik}:headcount:{end}",
                _company_id(company),
                end,
                fact.value,
                "Company-reported headcount definitions may differ across issuers.",
                source_id,
            ),
        )
        inserted += 1
    return inserted


def _upsert_annual_report_source(
    connection: sqlite3.Connection,
    company: CompanySpec,
    report_path: Path,
    metadata: dict[str, Any],
) -> str:
    accession = _required_metadata_text(metadata, "accession_number", report_path)
    retrieved_at = _required_metadata_text(metadata, "retrieved_at", report_path)
    url = _required_metadata_text(metadata, "url", report_path)
    raw_digest = sha256(report_path.read_bytes()).hexdigest()
    source_id = f"sec-filing:{company.padded_cik}:{accession}"
    connection.execute(
        """
        INSERT INTO sources(
            id, publisher, title, url, published_at, retrieved_at,
            accession_number, raw_sha256
        ) VALUES (?, 'U.S. Securities and Exchange Commission', ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            url = excluded.url,
            published_at = excluded.published_at,
            retrieved_at = excluded.retrieved_at,
            raw_sha256 = excluded.raw_sha256
        """,
        (
            source_id,
            f"{company.name} {metadata.get('form', 'annual report')}",
            url,
            metadata.get("filed_at") or None,
            retrieved_at,
            accession,
            raw_digest,
        ),
    )
    return source_id


def _insert_report_headcount(
    connection: sqlite3.Connection,
    company: CompanySpec,
    filing_text: str,
    metadata: dict[str, Any],
    source_id: str,
) -> int:
    headcount = _extract_headcount(filing_text)
    report_date = metadata.get("report_date")
    if headcount is None or not isinstance(report_date, str) or not report_date:
        return 0
    connection.execute(
        """
        INSERT INTO operating_signals(
            id, company_id, signal_type, observed_at,
            value_numeric, value_text, unit, quality_caveat, source_id
        ) VALUES (?, ?, 'headcount', ?, ?, ?, 'employees', ?, ?)
        """,
        (
            f"signal:{company.padded_cik}:headcount:{report_date}",
            _company_id(company),
            report_date,
            headcount,
            f"{headcount:g} employees",
            (
                "Regex-extracted from an SEC annual-report employee disclosure; "
                "issuer workforce definitions may differ and the passage should be "
                "verified for investment use."
            ),
            source_id,
        ),
    )
    return 1


def _insert_report_market_snapshot(
    connection: sqlite3.Connection,
    company: CompanySpec,
    filing_text: str,
    source_id: str,
) -> int:
    disclosure = _extract_cover_share_price(filing_text)
    if disclosure is None:
        return 0
    as_of, share_price = disclosure
    connection.execute(
        """
        INSERT INTO market_snapshots(
            id, company_id, as_of, currency, share_price,
            market_cap, enterprise_value, quality_caveat, source_id
        ) VALUES (?, ?, ?, 'USD', ?, NULL, NULL, ?, ?)
        """,
        (
            f"market:{company.padded_cik}:{as_of}",
            _company_id(company),
            as_of,
            share_price,
            (
                "Closing share price regex-extracted from the SEC annual-report "
                "public-float disclosure. Market capitalization and enterprise "
                "value are intentionally unavailable."
            ),
            source_id,
        ),
    )
    return 1


def _insert_sector_benchmarks(
    connection: sqlite3.Connection, sectors: tuple[SectorSpec, ...]
) -> int:
    annual_metrics = (
        "revenue",
        "operating_income",
        "operating_margin",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "cash_and_equivalents",
        "total_debt",
    )
    inserted = 0
    for sector in sectors:
        annual_rows = connection.execute(
            """
            SELECT af.* FROM annual_financial_snapshots AS af
            JOIN companies AS c ON c.id = af.company_id
            WHERE c.sector_id = ?
            ORDER BY af.company_id, af.period_end DESC
            """,
            (sector.id,),
        ).fetchall()
        latest_annual = _latest_company_rows(annual_rows)
        market_rows = connection.execute(
            """
            SELECT ms.* FROM market_snapshots AS ms
            JOIN companies AS c ON c.id = ms.company_id
            WHERE c.sector_id = ?
            ORDER BY ms.company_id, ms.as_of DESC
            """,
            (sector.id,),
        ).fetchall()
        latest_market = _latest_company_rows(market_rows)
        candidates = [
            (
                metric,
                latest_annual,
                "period_end",
                "ratio" if metric == "operating_margin" else "USD",
            )
            for metric in annual_metrics
        ]
        candidates.append(("share_price", latest_market, "as_of", "USD"))
        eligible: list[tuple[str, list[sqlite3.Row], str, str, list[float]]] = []
        for metric, rows, date_key, unit in candidates:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            if len(values) >= MIN_BENCHMARK_CONSTITUENTS:
                eligible.append((metric, rows, date_key, unit, values))
        if not eligible:
            continue
        source_id = f"derived:sector-median:{sector.id}"
        source_rows = [*latest_annual, *latest_market]
        lineage = "\n".join(sorted(str(row["source_id"]) for row in source_rows))
        connection.execute(
            """
            INSERT INTO sources(
                id, publisher, title, url, published_at,
                retrieved_at, raw_sha256
            ) VALUES (?, 'FinAgent derived from U.S. SEC filings', ?, ?, NULL, ?, ?)
            """,
            (
                source_id,
                f"{sector.name} source-universe median",
                "https://www.sec.gov/edgar/search/",
                _latest_source_retrieval(connection, source_rows),
                sha256(lineage.encode("utf-8")).hexdigest(),
            ),
        )
        connection.executemany(
            """
            INSERT INTO source_lineage(derived_source_id, input_source_id)
            VALUES (?, ?)
            """,
            (
                (source_id, input_source_id)
                for input_source_id in sorted(set(lineage.splitlines()))
            ),
        )
        for metric, rows, date_key, unit, values in eligible:
            as_of = max(str(row[date_key]) for row in rows if row[metric] is not None)
            connection.execute(
                """
                INSERT INTO sector_benchmarks(
                    id, sector_id, as_of, metric, value, unit,
                    quality_caveat, source_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"benchmark:{sector.id}:{metric}:{as_of}",
                    sector.id,
                    as_of,
                    metric,
                    median(values),
                    unit,
                    (
                        f"Median of {len(values)} available latest SEC-backed "
                        "constituent observations; periods and issuer definitions "
                        "may differ. This is not the named ETF's reported metric."
                    ),
                    source_id,
                ),
            )
            inserted += 1
    return inserted


def _latest_company_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    latest: list[sqlite3.Row] = []
    company_ids: set[str] = set()
    for row in rows:
        company_id = str(row["company_id"])
        if company_id not in company_ids:
            company_ids.add(company_id)
            latest.append(row)
    return latest


def _latest_source_retrieval(
    connection: sqlite3.Connection, rows: list[sqlite3.Row]
) -> str:
    source_ids = sorted({str(row["source_id"]) for row in rows})
    marks = ",".join("?" for _ in source_ids)
    values = connection.execute(
        f"SELECT retrieved_at FROM sources WHERE id IN ({marks})", source_ids
    ).fetchall()
    return max(str(row[0]) for row in values)


class _FilingTextParser(HTMLParser):
    """Collect visible filing text while ignoring script and style content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _html_text(document: str) -> str:
    parser = _FilingTextParser()
    parser.feed(document)
    return " ".join(" ".join(parser.parts).split())


def _extract_headcount(text: str) -> float | None:
    number = r"(?P<count>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|thousand)?"
    workforce_noun = r"(?:employees|people|associates|team members)"
    patterns = (
        (
            rf"\b(?:had|has|have|employed|employs)\s+(?:a total of\s+)?"
            rf"(?:approximately\s+|about\s+)?{number}[^.;]{{0,100}}?"
            rf"\b{workforce_noun}\b"
        ),
        (
            rf"\b(?:workforce|employee base)\s+(?:of|was|is|totaled)\s+"
            rf"(?:approximately\s+|about\s+)?{number}\s+{workforce_noun}\b"
        ),
        (
            rf"\b(?:combined with|delivered by)\s+(?:our\s+)?"
            rf"(?:approximately\s+|about\s+)?{number}\s+{workforce_noun}\b"
        ),
    )
    scale_multiplier = {None: 1, "thousand": 1_000, "million": 1_000_000}
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            scale = match.group("scale")
            values.append(
                float(match.group("count").replace(",", ""))
                * scale_multiplier[scale.lower() if scale else None]
            )
    return max(values) if values else None


def _extract_cover_share_price(text: str) -> tuple[str, float] | None:
    date_pattern = r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
    price_pattern = r"\$\s*([\d,]+(?:\.\d+)?)"
    patterns = (
        (
            rf"aggregate market value.{{0,600}}?(?:as of|on)\s+{date_pattern}"
            rf".{{0,600}}?(?:closing|last sale)\s+(?:sale\s+)?"
            rf"price(?:\s+of)?\s+{price_pattern}"
        ),
        (
            rf"aggregate market value.{{0,800}}?(?:closing|last sale)\s+"
            rf"(?:sale\s+)?price(?:\s+of)?\s+{price_pattern}.{{0,300}}?"
            rf"(?:as of|on)\s+{date_pattern}"
        ),
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        date_text, price_text = (
            (match.group(1), match.group(2))
            if index == 0
            else (match.group(2), match.group(1))
        )
        try:
            parsed_date = time.strptime(date_text, "%B %d, %Y")
            observed_at = date(
                parsed_date.tm_year,
                parsed_date.tm_mon,
                parsed_date.tm_mday,
            ).isoformat()
            return observed_at, float(price_text.replace(",", ""))
        except ValueError:
            continue
    return None


def _validate_cached_pair(document: Path, metadata_path: Path) -> None:
    if not document.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"annual-report cache requires document and metadata: {document}"
        )
    metadata = _load_mapping(metadata_path)
    if metadata.get("sha256") != sha256(document.read_bytes()).hexdigest():
        raise ValueError(f"cache digest does not match metadata: {document}")


def _latest_cached_annual_report(
    company_cache_dir: Path,
) -> tuple[Path, Path, dict[str, Any]] | None:
    candidates: list[tuple[str, str, Path, Path, dict[str, Any]]] = []
    for metadata_path in company_cache_dir.glob("filings/*/*.metadata.json"):
        metadata = _load_mapping(metadata_path)
        filename = metadata.get("local_filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"missing local_filename in cache metadata: {metadata_path}"
            )
        document = metadata_path.parent / filename
        _validate_cached_pair(document, metadata_path)
        candidates.append(
            (
                str(metadata.get("filed_at") or ""),
                str(metadata.get("accession_number") or ""),
                document,
                metadata_path,
                metadata,
            )
        )
    if not candidates:
        return None
    _, _, document, metadata_path, metadata = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    return document, metadata_path, metadata


def _required_metadata_text(
    metadata: dict[str, Any], key: str, source_path: Path
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {key} in cache metadata: {source_path}")
    return value


def _extract_metric(
    companyfacts: dict[str, Any],
    tags: tuple[str, ...],
    *,
    flow: bool,
    namespaces: tuple[str, ...] = ("us-gaap",),
    allowed_units: tuple[str, ...] = ("USD",),
) -> dict[str, FactValue]:
    selected: dict[str, FactValue] = {}
    facts = companyfacts.get("facts", {})
    if not isinstance(facts, dict):
        return selected
    for tag in tags:
        tag_values: dict[str, FactValue] = {}
        for namespace in namespaces:
            namespace_facts = facts.get(namespace, {})
            if not isinstance(namespace_facts, dict):
                continue
            units = namespace_facts.get(tag, {}).get("units", {})
            if not isinstance(units, dict):
                continue
            for unit in allowed_units:
                observations = units.get(unit, [])
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    candidate = _annual_candidate(observation, flow=flow)
                    if candidate is None:
                        continue
                    previous = tag_values.get(candidate.end)
                    if previous is None or (candidate.filed or "") > (
                        previous.filed or ""
                    ):
                        tag_values[candidate.end] = candidate
        for end, candidate in tag_values.items():
            selected.setdefault(end, candidate)
        # Earlier tags are preferred. Later tags only fill missing periods.
    return selected


def _extract_total_debt(companyfacts: dict[str, Any]) -> dict[str, FactValue]:
    total_tags = ("LongTermDebt", "LongTermDebtAndFinanceLeaseObligations")
    total = _extract_metric(companyfacts, total_tags, flow=False)
    current = _extract_metric(
        companyfacts,
        (
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtCurrent",
        ),
        flow=False,
    )
    noncurrent = _extract_metric(
        companyfacts,
        (
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtNoncurrent",
        ),
        flow=False,
    )
    for end in set(current) & set(noncurrent):
        if end in total:
            continue
        first = current[end]
        second = noncurrent[end]
        total[end] = FactValue(
            value=first.value + second.value,
            end=end,
            filed=max(first.filed or "", second.filed or "") or None,
            accession=first.accession
            if first.accession == second.accession
            else first.accession or second.accession,
        )
    return total


def _annual_candidate(observation: Any, *, flow: bool) -> FactValue | None:
    if not isinstance(observation, dict):
        return None
    if observation.get("form") not in ANNUAL_FORMS:
        return None
    if observation.get("fp") not in (None, "FY"):
        return None
    end = observation.get("end")
    value = observation.get("val")
    if not isinstance(end, str) or len(end) < 4 or not isinstance(value, (int, float)):
        return None
    if flow:
        start = observation.get("start")
        if not isinstance(start, str):
            return None
        duration = _date_duration_days(start, end)
        if duration is None or not 250 <= duration <= 430:
            return None
    return FactValue(
        value=float(value),
        end=end,
        filed=observation.get("filed"),
        accession=observation.get("accn"),
    )


def _date_duration_days(start: str, end: str) -> int | None:
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError:
        return None


def _number(value: FactValue | None) -> float | None:
    return value.value if value is not None else None


def _company_id(company: CompanySpec) -> str:
    return f"sec:{company.padded_cik}"


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON cache file: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"cached SEC document must be an object: {path}")
    return value


def _list_value(values: Any, index: int) -> str | None:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    return str(value) if value not in (None, "") else None
