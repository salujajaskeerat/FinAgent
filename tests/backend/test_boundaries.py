"""Architecture fitness tests for the mandatory MCP boundary."""

from pathlib import Path


def test_api_core_and_gateways_do_not_access_sqlite_directly() -> None:
    root = Path(__file__).parents[2] / "src" / "finagent"
    forbidden: list[str] = []
    for package in ("api", "core", "gateways"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "import sqlite3" in text or "from sqlite3" in text:
                forbidden.append(str(path.relative_to(root)))

    assert forbidden == []


def test_only_repository_owns_runtime_sql() -> None:
    root = Path(__file__).parents[2] / "src" / "finagent"
    runtime_files = [
        path
        for path in root.rglob("*.py")
        if "ingestion" not in path.parts and path.name != "repository.py"
    ]

    offenders = [
        str(path.relative_to(root))
        for path in runtime_files
        if "SELECT " in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
