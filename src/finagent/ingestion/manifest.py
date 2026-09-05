"""Source-manifest loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CompanySpec:
    """One company selected for the curated dataset."""

    name: str
    ticker: str
    cik: str

    @property
    def padded_cik(self) -> str:
        """Return the SEC CIK padded to ten digits."""
        return self.cik.zfill(10)


@dataclass(frozen=True, slots=True)
class SectorSpec:
    """A supported sector and its deliberately small company universe."""

    id: str
    name: str
    benchmark_name: str
    benchmark_ticker: str
    companies: tuple[CompanySpec, ...]


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Validated input manifest for downloads and database builds."""

    version: int
    sectors: tuple[SectorSpec, ...]

    def iter_companies(self) -> tuple[tuple[SectorSpec, CompanySpec], ...]:
        """Return all sector-company pairs in manifest order."""
        return tuple(
            (sector, company) for sector in self.sectors for company in sector.companies
        )


def load_manifest(path: Path | str) -> SourceManifest:
    """Load and validate a source manifest.

    Parameters
    ----------
    path:
        YAML manifest path.

    Returns
    -------
    SourceManifest
        Immutable validated manifest.

    Raises
    ------
    ValueError
        If required fields are missing, identifiers are duplicated, or a CIK is
        malformed.
    """
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("source manifest must be a YAML mapping")
    if raw.get("version") != 1:
        raise ValueError("source manifest version must be 1")

    raw_sectors = raw.get("sectors")
    if not isinstance(raw_sectors, list) or not raw_sectors:
        raise ValueError("source manifest must define at least one sector")

    sectors: list[SectorSpec] = []
    sector_ids: set[str] = set()
    tickers: set[str] = set()
    ciks: set[str] = set()
    for raw_sector in raw_sectors:
        _require_mapping(raw_sector, "sector")
        sector_id = _required_text(raw_sector, "id").lower()
        if sector_id in sector_ids:
            raise ValueError(f"duplicate sector id: {sector_id}")
        sector_ids.add(sector_id)

        benchmark = raw_sector.get("benchmark")
        _require_mapping(benchmark, f"benchmark for {sector_id}")
        raw_companies = raw_sector.get("companies")
        if not isinstance(raw_companies, list) or not raw_companies:
            raise ValueError(f"sector {sector_id} must define at least one company")

        companies: list[CompanySpec] = []
        for raw_company in raw_companies:
            _require_mapping(raw_company, f"company in {sector_id}")
            ticker = _required_text(raw_company, "ticker").upper()
            cik = _required_text(raw_company, "cik")
            if not cik.isdigit() or len(cik) > 10:
                raise ValueError(f"invalid SEC CIK for {ticker}: {cik}")
            padded_cik = cik.zfill(10)
            if ticker in tickers:
                raise ValueError(f"duplicate ticker: {ticker}")
            if padded_cik in ciks:
                raise ValueError(f"duplicate CIK: {padded_cik}")
            tickers.add(ticker)
            ciks.add(padded_cik)
            companies.append(
                CompanySpec(
                    name=_required_text(raw_company, "name"),
                    ticker=ticker,
                    cik=padded_cik,
                )
            )

        sectors.append(
            SectorSpec(
                id=sector_id,
                name=_required_text(raw_sector, "name"),
                benchmark_name=_required_text(benchmark, "name"),
                benchmark_ticker=_required_text(benchmark, "ticker").upper(),
                companies=tuple(companies),
            )
        )

    return SourceManifest(version=1, sectors=tuple(sectors))


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a YAML mapping")


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"missing or invalid manifest field: {key}")
    return str(value).strip()
