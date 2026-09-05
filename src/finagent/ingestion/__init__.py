"""Offline data ingestion for the FinAgent sample database.

Nothing in this package performs network I/O at import time. Downloads and database
builds must be invoked explicitly through the public functions or command-line entry
point.
"""

from .builder import BuildStats, build_database
from .config import SecDownloadConfig
from .manifest import SourceManifest, load_manifest
from .sec_client import DownloadedCompanyFiles, SecEdgarClient, download_manifest

__all__ = [
    "BuildStats",
    "DownloadedCompanyFiles",
    "SecDownloadConfig",
    "SecEdgarClient",
    "SourceManifest",
    "build_database",
    "download_manifest",
    "load_manifest",
]
