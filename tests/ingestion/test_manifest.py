"""Tests for curated source-manifest validation."""

from pathlib import Path

from finagent.ingestion.manifest import load_manifest

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_manifest_normalizes_cik_and_ticker() -> None:
    manifest = load_manifest(FIXTURES / "source_manifest.yaml")

    sector = manifest.sectors[0]
    company = sector.companies[0]
    assert sector.id == "tech"
    assert company.ticker == "EXM"
    assert company.padded_cik == "0000000001"
