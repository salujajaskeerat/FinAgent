"""Configuration for explicit SEC EDGAR downloads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SecDownloadConfig:
    """Settings used by the polite SEC downloader.

    Parameters
    ----------
    user_agent:
        Identifying user agent containing an application or organization name and a
        monitored contact email, as requested by the SEC.
    cache_dir:
        Root directory for immutable-ish raw SEC responses.
    requests_per_second:
        Maximum request start rate. The SEC permits up to ten requests per second;
        this client defaults to the more conservative value of two.
    timeout_seconds:
        Per-request network timeout.
    base_url:
        SEC data API origin. Override only in tests or controlled mirrors.
    """

    user_agent: str
    cache_dir: Path = Path("data/raw/sec")
    requests_per_second: float = 2.0
    timeout_seconds: float = 30.0
    base_url: str = "https://data.sec.gov"

    def __post_init__(self) -> None:
        """Validate limits and the required SEC identity."""
        identity = self.user_agent.strip()
        if not identity or "@" not in identity:
            raise ValueError(
                "SEC user agent must identify the application and include a contact "
                "email, for example 'FinAgent contact@example.com'."
            )
        if not 0 < self.requests_per_second <= 10:
            raise ValueError(
                "requests_per_second must be greater than 0 and at most 10"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP(S) URL")

    @classmethod
    def from_env(
        cls,
        *,
        cache_dir: Path | None = None,
        requests_per_second: float | None = None,
        timeout_seconds: float | None = None,
    ) -> SecDownloadConfig:
        """Create configuration from environment variables.

        Parameters
        ----------
        cache_dir:
            Optional explicit cache location. Otherwise ``FINAGENT_RAW_SEC_DIR`` or
            ``data/raw/sec`` is used.
        requests_per_second:
            Optional explicit rate limit. Otherwise ``SEC_REQUESTS_PER_SECOND`` or
            ``2`` is used.
        timeout_seconds:
            Optional explicit timeout. Otherwise ``SEC_TIMEOUT_SECONDS`` or ``30``
            is used.

        Returns
        -------
        SecDownloadConfig
            Validated downloader configuration.

        Raises
        ------
        ValueError
            If ``SEC_USER_AGENT`` is absent or does not include a contact email.
        """
        user_agent = os.environ.get("SEC_USER_AGENT", "")
        configured_cache = cache_dir or Path(
            os.environ.get("FINAGENT_RAW_SEC_DIR", "data/raw/sec")
        )
        configured_rate = requests_per_second
        if configured_rate is None:
            configured_rate = float(os.environ.get("SEC_REQUESTS_PER_SECOND", "2"))
        configured_timeout = timeout_seconds
        if configured_timeout is None:
            configured_timeout = float(os.environ.get("SEC_TIMEOUT_SECONDS", "30"))
        return cls(
            user_agent=user_agent,
            cache_dir=configured_cache,
            requests_per_second=configured_rate,
            timeout_seconds=configured_timeout,
        )
