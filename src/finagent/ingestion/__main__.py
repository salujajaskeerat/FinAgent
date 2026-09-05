"""Command-line entry point for explicit downloads and offline database builds."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audit import DataAudit, audit_database
from .builder import BuildStats, build_database
from .config import SecDownloadConfig
from .manifest import load_manifest
from .sample_data import build_sample_database
from .sec_client import download_manifest, download_manifest_annual_reports


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion command-line interface.

    Parameters
    ----------
    argv:
        Optional argument list. Defaults to process arguments.

    Returns
    -------
    int
        Process exit status.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        manifest = load_manifest(args.manifest)
        config = SecDownloadConfig.from_env(
            cache_dir=args.raw_dir,
            requests_per_second=args.requests_per_second,
        )
        downloaded = download_manifest(
            manifest,
            config,
            sector=args.sector,
            ticker=args.ticker,
            overwrite=args.overwrite,
        )
        for item in downloaded:
            print(f"{item.cik}: {item.submissions} {item.companyfacts}")
        return 0

    if args.command == "sample":
        stats = build_sample_database(args.manifest, args.output)
        print(
            f"built illustrative sample {stats.output_path}: "
            f"{stats.companies} companies, {stats.annual_snapshots} annual snapshots, "
            f"{stats.market_snapshots} market snapshots, "
            f"{stats.operating_signals} operating signals, "
            f"{stats.benchmark_observations} benchmark observations"
        )
        return 0

    if args.command == "download-filings":
        manifest = load_manifest(args.manifest)
        config = SecDownloadConfig.from_env(
            cache_dir=args.raw_dir,
            requests_per_second=args.requests_per_second,
        )
        downloaded = download_manifest_annual_reports(
            manifest,
            config,
            sector=args.sector,
            ticker=args.ticker,
            overwrite=args.overwrite,
        )
        for item in downloaded:
            print(f"{item.cik}: {item.accession_number} {item.document}")
        return 0

    if args.command == "refresh-public":
        manifest = load_manifest(args.manifest)
        config = SecDownloadConfig.from_env(
            cache_dir=args.raw_dir,
            requests_per_second=args.requests_per_second,
        )
        download_manifest(manifest, config, overwrite=True)
        download_manifest_annual_reports(manifest, config, overwrite=True)
        stats = build_database(
            args.manifest,
            args.raw_dir,
            args.output,
            annual_periods=args.annual_periods,
        )
        _print_build_stats(stats)
        _print_audit(audit_database(args.output, require_real_enrichment=True))
        return 0

    if args.command == "audit":
        _print_audit(
            audit_database(
                args.database,
                require_real_enrichment=args.require_real_enrichment,
            )
        )
        return 0

    stats = build_database(
        args.manifest,
        args.raw_dir,
        args.output,
        annual_periods=args.annual_periods,
        strict=not args.allow_missing,
    )
    _print_build_stats(stats)
    return 0


def _print_build_stats(stats: BuildStats) -> None:
    print(
        f"built {stats.output_path}: {stats.companies} companies, "
        f"{stats.annual_snapshots} annual snapshots, "
        f"{stats.market_snapshots} market snapshots, "
        f"{stats.operating_signals} operating signals, "
        f"{stats.benchmark_observations} benchmark observations"
    )


def _print_audit(audit: DataAudit) -> None:
    print(
        f"audit: {audit.sectors} sectors, {audit.companies} companies, "
        f"{audit.sources} sources, {audit.lineage_links} lineage links, "
        f"{audit.orphaned_source_references} orphaned source references"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m finagent.ingestion",
        description="Explicitly download SEC data or build SQLite from the local cache.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="Download and cache selected SEC JSON (networked)."
    )
    download.add_argument(
        "--manifest", type=Path, default=Path("data/source_manifest.yaml")
    )
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw/sec"))
    download.add_argument("--sector")
    download.add_argument("--ticker")
    download.add_argument("--requests-per-second", type=float, default=None)
    download.add_argument("--overwrite", action="store_true")

    build = subparsers.add_parser(
        "build", help="Build SQLite from cached files only (offline)."
    )
    build.add_argument(
        "--manifest", type=Path, default=Path("data/source_manifest.yaml")
    )
    build.add_argument("--raw-dir", type=Path, default=Path("data/raw/sec"))
    build.add_argument("--output", type=Path, default=Path("data/finagent.db"))
    build.add_argument("--annual-periods", type=int, default=3)
    build.add_argument("--allow-missing", action="store_true")

    filings = subparsers.add_parser(
        "download-filings",
        help="Download latest annual filing HTML from cached submissions (networked).",
    )
    filings.add_argument(
        "--manifest", type=Path, default=Path("data/source_manifest.yaml")
    )
    filings.add_argument("--raw-dir", type=Path, default=Path("data/raw/sec"))
    filings.add_argument("--sector")
    filings.add_argument("--ticker")
    filings.add_argument("--requests-per-second", type=float, default=None)
    filings.add_argument("--overwrite", action="store_true")

    refresh = subparsers.add_parser(
        "refresh-public",
        help="Refresh public SEC caches, build SQLite, and audit enrichment.",
    )
    refresh.add_argument(
        "--manifest", type=Path, default=Path("data/source_manifest.yaml")
    )
    refresh.add_argument("--raw-dir", type=Path, default=Path("data/raw/sec"))
    refresh.add_argument("--output", type=Path, default=Path("data/finagent.db"))
    refresh.add_argument("--annual-periods", type=int, default=3)
    refresh.add_argument("--requests-per-second", type=float, default=None)

    audit = subparsers.add_parser(
        "audit", help="Report database coverage and provenance integrity."
    )
    audit.add_argument("--database", type=Path, default=Path("data/finagent.db"))
    audit.add_argument("--require-real-enrichment", action="store_true")

    sample = subparsers.add_parser(
        "sample", help="Build a complete illustrative database without network access."
    )
    sample.add_argument(
        "--manifest", type=Path, default=Path("data/source_manifest.yaml")
    )
    sample.add_argument("--output", type=Path, default=Path("data/finagent_sample.db"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
