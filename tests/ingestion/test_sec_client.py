"""Offline tests for explicit SEC downloading and cache reuse."""

from pathlib import Path
from urllib.request import Request

import pytest

from finagent.ingestion.config import SecDownloadConfig
from finagent.ingestion.manifest import CompanySpec
from finagent.ingestion.sec_client import SecEdgarClient


def test_download_requires_identifying_user_agent() -> None:
    with pytest.raises(ValueError, match="contact email"):
        SecDownloadConfig(user_agent="anonymous-client")


def test_from_env_requires_sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    with pytest.raises(ValueError, match="contact email"):
        SecDownloadConfig.from_env()


def test_download_is_rate_limited_and_reuses_cache(tmp_path: Path) -> None:
    requests: list[Request] = []
    sleeps: list[float] = []

    def transport(request: Request, timeout: float) -> bytes:
        requests.append(request)
        assert timeout == 5
        assert request.get_header("User-agent") == "FinAgent tests@example.com"
        return b'{"valid": true}'

    client = SecEdgarClient(
        SecDownloadConfig(
            user_agent="FinAgent tests@example.com",
            cache_dir=tmp_path,
            requests_per_second=2,
            timeout_seconds=5,
        ),
        transport=transport,
        sleeper=sleeps.append,
        clock=lambda: 0,
    )
    company = CompanySpec(name="Example", ticker="EXM", cik="1")

    first = client.download_company(company)
    second = client.download_company(company)

    assert first == second
    assert len(requests) == 2
    assert sleeps == [0.5]
    assert first.submissions.read_text(encoding="utf-8") == '{"valid": true}'
    assert first.companyfacts.with_suffix(".metadata.json").is_file()


def test_download_latest_annual_report_uses_cached_submissions(tmp_path: Path) -> None:
    """Select the latest annual filing and record immutable retrieval metadata."""
    requests: list[Request] = []

    def transport(request: Request, timeout: float) -> bytes:
        requests.append(request)
        assert timeout == 5
        return b"<html><body>The Company had 1,234 employees.</body></html>"

    company = CompanySpec(name="Example", ticker="EXM", cik="1")
    company_dir = tmp_path / company.padded_cik
    company_dir.mkdir()
    (company_dir / "submissions.json").write_text(
        """
        {
          "filings": {"recent": {
            "accessionNumber": ["0000000001-24-000001", "0000000001-25-000001"],
            "form": ["10-K", "10-K"],
            "filingDate": ["2024-02-16", "2025-02-15"],
            "reportDate": ["2023-12-31", "2024-12-31"],
            "primaryDocument": ["old.htm", "latest.htm"]
          }}
        }
        """,
        encoding="utf-8",
    )
    client = SecEdgarClient(
        SecDownloadConfig(
            user_agent="FinAgent tests@example.com",
            cache_dir=tmp_path,
            timeout_seconds=5,
            archives_base_url="https://sec.test",
        ),
        transport=transport,
    )

    first = client.download_latest_annual_report(company)
    second = client.download_latest_annual_report(company)

    assert first == second
    assert first.accession_number == "0000000001-25-000001"
    assert len(requests) == 1
    assert requests[0].full_url.endswith(
        "/Archives/edgar/data/1/000000000125000001/latest.htm"
    )
    metadata = first.document.with_suffix(".metadata.json").read_text(encoding="utf-8")
    assert '"report_date": "2024-12-31"' in metadata
    assert '"sha256":' in metadata
