"""Small, explicit, rate-limited SEC EDGAR downloader."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import SecDownloadConfig
from .manifest import CompanySpec, SourceManifest

Transport = Callable[[Request, float], bytes]


@dataclass(frozen=True, slots=True)
class DownloadedCompanyFiles:
    """Local cached files for one SEC company."""

    cik: str
    submissions: Path
    companyfacts: Path


@dataclass(frozen=True, slots=True)
class DownloadedAnnualReport:
    """Local cached copy of one company's latest annual filing document."""

    cik: str
    accession_number: str
    document: Path


class SecEdgarClient:
    """Download and cache the two small SEC JSON resources used by FinAgent.

    Parameters
    ----------
    config:
        Validated SEC identity, cache, rate, and timeout settings.
    transport:
        Optional byte-returning HTTP transport used by tests.
    sleeper:
        Optional sleep function used by tests.
    clock:
        Optional monotonic clock used by tests.
    """

    def __init__(
        self,
        config: SecDownloadConfig,
        *,
        transport: Transport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._transport = transport or self._default_transport
        self._sleeper = sleeper
        self._clock = clock
        self._last_request_started: float | None = None

    def download_company(
        self, company: CompanySpec, *, overwrite: bool = False
    ) -> DownloadedCompanyFiles:
        """Download submissions and Companyfacts JSON for one company.

        Existing valid cache files are reused unless ``overwrite`` is true. Calling
        this method is the explicit network boundary; importing the package and
        building SQLite never performs a download.

        Parameters
        ----------
        company:
            Company metadata containing the SEC CIK.
        overwrite:
            Whether to refresh files already in the cache.

        Returns
        -------
        DownloadedCompanyFiles
            Paths to the two cached JSON documents.
        """
        cik = company.padded_cik
        company_dir = self._config.cache_dir / cik
        submissions = self._fetch_json(
            f"{self._config.base_url}/submissions/CIK{cik}.json",
            company_dir / "submissions.json",
            overwrite=overwrite,
        )
        companyfacts = self._fetch_json(
            f"{self._config.base_url}/api/xbrl/companyfacts/CIK{cik}.json",
            company_dir / "companyfacts.json",
            overwrite=overwrite,
        )
        return DownloadedCompanyFiles(
            cik=cik, submissions=submissions, companyfacts=companyfacts
        )

    def download_latest_annual_report(
        self, company: CompanySpec, *, overwrite: bool = False
    ) -> DownloadedAnnualReport:
        """Cache the latest annual filing HTML declared by SEC submissions.

        The submissions cache must already exist. The selected accession, report
        date, and filing date are recorded beside the document so offline builders
        need not infer network state.

        Parameters
        ----------
        company:
            Company whose latest annual filing is selected.
        overwrite:
            Whether to refresh an existing valid document.

        Returns
        -------
        DownloadedAnnualReport
            Selected accession and cached document path.

        Raises
        ------
        FileNotFoundError
            If submissions have not been cached first.
        ValueError
            If submissions contain no usable annual filing.
        """
        company_dir = self._config.cache_dir / company.padded_cik
        submissions_path = company_dir / "submissions.json"
        if not submissions_path.exists():
            raise FileNotFoundError(
                f"missing submissions cache for {company.ticker}: {submissions_path}"
            )
        submissions = _read_json(submissions_path)
        if not isinstance(submissions, dict):
            raise TypeError(f"SEC submissions must be an object: {submissions_path}")
        filing = _latest_annual_filing(submissions)
        accession = filing["accession_number"]
        document_name = filing["primary_document"]
        if Path(document_name).name != document_name:
            raise ValueError(f"unsafe SEC primary document path: {document_name}")
        accession_compact = accession.replace("-", "")
        url = (
            f"{self._config.archives_base_url}/Archives/edgar/data/"
            f"{int(company.padded_cik)}/{accession_compact}/{document_name}"
        )
        destination = company_dir / "filings" / accession_compact / document_name
        self._fetch_document(
            url,
            destination,
            overwrite=overwrite,
            metadata={
                "accession_number": accession,
                "filed_at": filing["filed_at"],
                "form": filing["form"],
                "local_filename": document_name,
                "report_date": filing["report_date"],
            },
        )
        return DownloadedAnnualReport(
            cik=company.padded_cik,
            accession_number=accession,
            document=destination,
        )

    def _fetch_json(self, url: str, destination: Path, *, overwrite: bool) -> Path:
        if destination.exists() and not overwrite:
            _read_json(destination)
            metadata_path = _metadata_path(destination)
            if not metadata_path.exists():
                raise ValueError(
                    f"cache metadata is missing for {destination}; rerun with --overwrite"
                )
            metadata = _read_json(metadata_path)
            expected_digest = (
                metadata.get("sha256") if isinstance(metadata, dict) else None
            )
            actual_digest = sha256(destination.read_bytes()).hexdigest()
            if expected_digest != actual_digest:
                raise ValueError(
                    f"cache digest does not match metadata for {destination}; "
                    "rerun with --overwrite"
                )
            return destination

        self._wait_for_rate_limit()
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._config.user_agent,
            },
        )
        body = self._transport(request, self._config.timeout_seconds)
        json.loads(body.decode("utf-8"))
        _write_atomic(destination, body)
        metadata = json.dumps(
            {
                "retrieved_at": datetime.now(UTC).date().isoformat(),
                "sha256": sha256(body).hexdigest(),
                "url": url,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _write_atomic(_metadata_path(destination), metadata)
        return destination

    def _fetch_document(
        self,
        url: str,
        destination: Path,
        *,
        overwrite: bool,
        metadata: dict[str, str],
    ) -> Path:
        metadata_path = _metadata_path(destination)
        if destination.exists() and not overwrite:
            _validate_cached_file(destination, metadata_path)
            return destination

        self._wait_for_rate_limit()
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": self._config.user_agent,
            },
        )
        body = self._transport(request, self._config.timeout_seconds)
        if not body.strip():
            raise ValueError(f"empty SEC filing document: {url}")
        _write_atomic(destination, body)
        metadata_body = json.dumps(
            {
                **metadata,
                "retrieved_at": datetime.now(UTC).date().isoformat(),
                "sha256": sha256(body).hexdigest(),
                "url": url,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _write_atomic(metadata_path, metadata_body)
        return destination

    def _wait_for_rate_limit(self) -> None:
        minimum_interval = 1.0 / self._config.requests_per_second
        now = self._clock()
        if self._last_request_started is not None:
            remaining = minimum_interval - (now - self._last_request_started)
            if remaining > 0:
                self._sleeper(remaining)
                now = self._clock()
        self._last_request_started = now

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read()


def download_manifest(
    manifest: SourceManifest,
    config: SecDownloadConfig,
    *,
    sector: str | None = None,
    ticker: str | None = None,
    overwrite: bool = False,
) -> tuple[DownloadedCompanyFiles, ...]:
    """Explicitly download selected companies from a source manifest.

    Parameters
    ----------
    manifest:
        Validated source manifest.
    config:
        SEC download settings and identity.
    sector:
        Optional sector id filter.
    ticker:
        Optional ticker filter.
    overwrite:
        Whether to replace existing cache files.

    Returns
    -------
    tuple[DownloadedCompanyFiles, ...]
        Cached file paths for each selected company.

    Raises
    ------
    ValueError
        If the filters select no companies.
    """
    selected = [
        company
        for sector_spec, company in manifest.iter_companies()
        if (sector is None or sector_spec.id == sector.lower())
        and (ticker is None or company.ticker == ticker.upper())
    ]
    if not selected:
        raise ValueError("download filters selected no companies")
    client = SecEdgarClient(config)
    return tuple(
        client.download_company(company, overwrite=overwrite) for company in selected
    )


def download_manifest_annual_reports(
    manifest: SourceManifest,
    config: SecDownloadConfig,
    *,
    sector: str | None = None,
    ticker: str | None = None,
    overwrite: bool = False,
) -> tuple[DownloadedAnnualReport, ...]:
    """Download the latest cached-submissions annual report for each company.

    Parameters
    ----------
    manifest:
        Validated source manifest.
    config:
        SEC identity, cache, and rate-limit settings.
    sector:
        Optional sector-id filter.
    ticker:
        Optional ticker filter.
    overwrite:
        Whether to refresh cached filing documents.

    Returns
    -------
    tuple[DownloadedAnnualReport, ...]
        Cached annual filing documents in manifest order.

    Raises
    ------
    ValueError
        If the filters select no companies.
    """
    selected = [
        company
        for sector_spec, company in manifest.iter_companies()
        if (sector is None or sector_spec.id == sector.lower())
        and (ticker is None or company.ticker == ticker.upper())
    ]
    if not selected:
        raise ValueError("annual-report filters selected no companies")
    client = SecEdgarClient(config)
    return tuple(
        client.download_latest_annual_report(company, overwrite=overwrite)
        for company in selected
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_path(destination: Path) -> Path:
    return destination.with_suffix(".metadata.json")


def _validate_cached_file(path: Path, metadata_path: Path) -> None:
    if not metadata_path.exists():
        raise ValueError(
            f"cache metadata is missing for {path}; rerun with --overwrite"
        )
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise TypeError(f"cache metadata must be an object: {metadata_path}")
    if metadata.get("sha256") != sha256(path.read_bytes()).hexdigest():
        raise ValueError(
            f"cache digest does not match metadata for {path}; rerun with --overwrite"
        )


def _latest_annual_filing(submissions: dict[str, Any]) -> dict[str, str]:
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        raise TypeError("SEC submissions recent filing index must be an object")
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    documents = recent.get("primaryDocument", [])
    annual_forms = {"10-K", "20-F", "40-F"}
    candidates: list[dict[str, str]] = []
    for index, form in enumerate(forms if isinstance(forms, list) else []):
        accession = _list_text(accessions, index)
        document = _list_text(documents, index)
        if form not in annual_forms or accession is None or document is None:
            continue
        candidates.append(
            {
                "accession_number": accession,
                "filed_at": _list_text(filing_dates, index) or "",
                "form": str(form),
                "primary_document": document,
                "report_date": _list_text(report_dates, index) or "",
            }
        )
    if not candidates:
        raise ValueError("SEC submissions contain no annual filing document")
    return max(
        candidates, key=lambda item: (item["filed_at"], item["accession_number"])
    )


def _list_text(values: Any, index: int) -> str | None:
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    return str(value) if value not in (None, "") else None


def _write_atomic(destination: Path, body: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
