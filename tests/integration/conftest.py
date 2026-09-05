"""Fixtures for the real MCP Streamable HTTP integration boundary."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from finagent.ingestion.builder import build_database

PROJECT_ROOT = Path(__file__).parents[2]
INGESTION_FIXTURES = PROJECT_ROOT / "tests" / "ingestion" / "fixtures"


def _available_port() -> int:
    """Return an ephemeral localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_listening(port: int, process: subprocess.Popen[str]) -> None:
    """Wait briefly for the child server to accept TCP connections."""
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=1)
            raise RuntimeError(f"MCP server exited during startup: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("MCP server did not start within 8 seconds")


@pytest.fixture
def real_mcp_url(tmp_path: Path) -> Iterator[str]:
    """Run the actual MCP HTTP server against a temporary fixture database."""
    database = tmp_path / "finagent.db"
    build_database(
        INGESTION_FIXTURES / "source_manifest.yaml",
        INGESTION_FIXTURES / "raw" / "sec",
        database,
    )
    port = _available_port()
    environment = {
        **os.environ,
        "FINAGENT_DB_PATH": str(database),
        "FINAGENT_MCP_HOST": "127.0.0.1",
        "FINAGENT_MCP_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "finagent.mcp_server.server"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_until_listening(port, process)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=3)
