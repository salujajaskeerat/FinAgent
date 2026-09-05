"""Coverage and provenance audit tests."""

from pathlib import Path

import pytest

from finagent.ingestion.audit import audit_database
from finagent.ingestion.sample_data import build_sample_database

MANIFEST = Path(__file__).parents[2] / "data" / "source_manifest.yaml"


def test_audit_reports_complete_sample_coverage(tmp_path: Path) -> None:
    """Report every populated table while identifying fixture sources."""
    database = tmp_path / "sample.db"
    build_sample_database(MANIFEST, database)

    audit = audit_database(database)

    assert audit.sectors == 3
    assert audit.companies == 12
    assert audit.market_snapshots == 12
    assert audit.operating_signals == 36
    assert audit.benchmark_observations == 27
    assert audit.illustrative_sources == 75
    assert audit.orphaned_source_references == 0
    assert not audit.has_real_enrichment


def test_strict_audit_rejects_illustrative_sources(tmp_path: Path) -> None:
    """Prevent a fixture database from passing the real-data gate."""
    database = tmp_path / "sample.db"
    build_sample_database(MANIFEST, database)

    with pytest.raises(ValueError, match="real-data enrichment audit failed"):
        audit_database(database, require_real_enrichment=True)
