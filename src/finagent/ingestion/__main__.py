"""Command-line entry point for explicit downloads and offline database builds."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_database
from .config import SecDownloadConfig
from .manifest import load_manifest
from .sec_client import download_manifest


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

    stats = build_database(
        args.manifest,
        args.raw_dir,
        args.output,
        annual_periods=args.annual_periods,
        strict=not args.allow_missing,
    )
    print(
        f"built {stats.output_path}: {stats.companies} companies, "
        f"{stats.annual_snapshots} annual snapshots, "
        f"{stats.operating_signals} operating signals"
    )
    return 0


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
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
