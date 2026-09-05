"""Build the runtime SQLite database exclusively from cached source files."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .manifest import CompanySpec, load_manifest

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


@dataclass(frozen=True, slots=True)
class BuildStats:
    """Counts emitted by one deterministic database build."""

    sectors: int
    companies: int
    annual_snapshots: int
    operating_signals: int
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
    signal_count = 0
    skipped_count = 0
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
                    signal_count += _insert_headcount_signals(
                        connection,
                        company,
                        companyfacts,
                        filing_sources,
                        annual_periods=annual_periods,
                    )
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
        operating_signals=signal_count,
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
    from datetime import date

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
