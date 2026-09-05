"""Read-only SQLite repository owned exclusively by the MCP service."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from finagent.contracts.api import Sector, SourceRef
from finagent.contracts.mcp import (
    CatalogEntity,
    DatasetCatalog,
    EntityKind,
    Event,
    EventResult,
    Observation,
    ObservationResult,
    ResolutionResult,
    ResolvedEntity,
)

_ANNUAL_METRICS = (
    "revenue",
    "operating_income",
    "operating_margin",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "cash_and_equivalents",
    "total_debt",
)
_MARKET_METRICS = ("share_price", "market_cap", "enterprise_value")


class RepositoryError(RuntimeError):
    """The read-only dataset could not satisfy a valid operation."""


class SectorRepository:
    """Parameterized read model over the purpose-built financial schema."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path.resolve()

    def get_catalog(self, sector: Sector) -> DatasetCatalog:
        """Return companies, benchmark, metric keys, and date coverage."""
        with closing(self._connect()) as connection:
            sector_row = connection.execute(
                """
                SELECT id, name, benchmark_name, benchmark_ticker
                FROM sectors WHERE id = ?
                """,
                (sector.value,),
            ).fetchone()
            if sector_row is None:
                raise ValueError(f"unknown sector: {sector.value}")
            company_rows = connection.execute(
                "SELECT id, name, ticker FROM companies WHERE sector_id = ? ORDER BY name",
                (sector.value,),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT DISTINCT os.signal_type
                FROM operating_signals AS os
                JOIN companies AS c ON c.id = os.company_id
                WHERE c.sector_id = ? ORDER BY os.signal_type
                """,
                (sector.value,),
            ).fetchall()
            benchmark_rows = connection.execute(
                """
                SELECT DISTINCT metric FROM sector_benchmarks
                WHERE sector_id = ? ORDER BY metric
                """,
                (sector.value,),
            ).fetchall()
            coverage = connection.execute(
                """
                SELECT MIN(day), MAX(day)
                FROM (
                    SELECT af.period_end AS day
                    FROM annual_financial_snapshots AS af
                    JOIN companies AS c ON c.id = af.company_id
                    WHERE c.sector_id = ?
                    UNION ALL
                    SELECT ms.as_of AS day
                    FROM market_snapshots AS ms
                    JOIN companies AS c ON c.id = ms.company_id
                    WHERE c.sector_id = ?
                    UNION ALL
                    SELECT os.observed_at AS day
                    FROM operating_signals AS os
                    JOIN companies AS c ON c.id = os.company_id
                    WHERE c.sector_id = ?
                    UNION ALL
                    SELECT sb.as_of AS day
                    FROM sector_benchmarks AS sb WHERE sb.sector_id = ?
                )
                """,
                (sector.value, sector.value, sector.value, sector.value),
            ).fetchone()
            version = self._dataset_version(connection)

        entities = [
            CatalogEntity(
                entity_id=row["id"],
                kind=EntityKind.COMPANY,
                name=row["name"],
                ticker=row["ticker"],
                aliases=[row["ticker"]],
            )
            for row in company_rows
        ]
        entities.append(
            CatalogEntity(
                entity_id=self._benchmark_id(sector),
                kind=EntityKind.BENCHMARK,
                name=sector_row["benchmark_name"],
                ticker=sector_row["benchmark_ticker"],
                aliases=[sector_row["benchmark_ticker"]],
            )
        )
        benchmark_metrics = [row["metric"] for row in benchmark_rows]
        return DatasetCatalog(
            dataset_version=version,
            sector=sector,
            entities=entities,
            metric_keys=list(
                dict.fromkeys([*_ANNUAL_METRICS, *_MARKET_METRICS, *benchmark_metrics])
            ),
            event_kinds=[row["signal_type"] for row in event_rows],
            coverage_start=coverage[0] if coverage else None,
            coverage_end=coverage[1] if coverage else None,
        )

    def resolve_companies(self, sector: Sector, query: str) -> ResolutionResult:
        """Resolve company names and tickers and flag explicit unknown names."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, name, ticker FROM companies WHERE sector_id = ?",
                (sector.value,),
            ).fetchall()
        resolved: list[ResolvedEntity] = []
        lowered = query.casefold()
        for row in rows:
            for alias in sorted((row["name"], row["ticker"]), key=len, reverse=True):
                if re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", lowered):
                    resolved.append(ResolvedEntity(mention=alias, entity_id=row["id"]))
                    break

        unresolved: list[str] = []
        if not resolved:
            explicit = re.search(
                r"\b(?:about|for|company)\s+([A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*){0,3})",
                query,
            )
            if explicit:
                mention = explicit.group(1).rstrip("?.!,")
                if mention not in {"This", "The", "A", "An"}:
                    unresolved.append(mention)
        return ResolutionResult(resolved=resolved, unresolved_mentions=unresolved)

    def query_observations(
        self,
        sector: Sector,
        entity_ids: list[str],
        metric_keys: list[str],
        latest_only: bool,
        limit: int,
    ) -> ObservationResult:
        """Map financial, market, and benchmark rows to observations."""
        bounded_limit = self._bounded_limit(limit)
        if not entity_ids or not metric_keys:
            return ObservationResult(dataset_version=self.version())
        requested = set(metric_keys)
        company_ids = self._validated_company_ids(sector, entity_ids)
        observations: list[Observation] = []
        warnings: list[str] = []
        with closing(self._connect()) as connection:
            if company_ids and requested.intersection(_ANNUAL_METRICS):
                for row in self._annual_rows(connection, sector, company_ids):
                    self._append_caveat(warnings, row["id"], row["quality_caveat"])
                    for metric in _ANNUAL_METRICS:
                        if metric not in requested or row[metric] is None:
                            continue
                        is_ratio = metric == "operating_margin"
                        observations.append(
                            Observation(
                                observation_id=f"{row['id']}:{metric}",
                                entity_id=row["company_id"],
                                metric_key=metric,
                                value=row[metric],
                                unit="ratio" if is_ratio else row["currency"],
                                currency=None if is_ratio else row["currency"],
                                period_end=row["period_end"],
                                observed_at=row["filed_at"] or row["period_end"],
                                source_id=row["source_id"],
                            )
                        )
            if company_ids and requested.intersection(_MARKET_METRICS):
                for row in self._market_rows(connection, sector, company_ids):
                    self._append_caveat(warnings, row["id"], row["quality_caveat"])
                    for metric in _MARKET_METRICS:
                        if metric not in requested or row[metric] is None:
                            continue
                        observations.append(
                            Observation(
                                observation_id=f"{row['id']}:{metric}",
                                entity_id=row["company_id"],
                                metric_key=metric,
                                value=row[metric],
                                unit=row["currency"],
                                currency=row["currency"],
                                period_end=row["as_of"],
                                observed_at=row["as_of"],
                                source_id=row["source_id"],
                            )
                        )
            if self._benchmark_id(sector) in entity_ids:
                marks = ",".join("?" for _ in metric_keys)
                rows = connection.execute(
                    f"""
                    SELECT id, as_of, metric, value, unit, source_id, quality_caveat
                    FROM sector_benchmarks
                    WHERE sector_id = ? AND metric IN ({marks})
                    ORDER BY as_of DESC
                    """,
                    [sector.value, *metric_keys],
                ).fetchall()
                for row in rows:
                    self._append_caveat(warnings, row["id"], row["quality_caveat"])
                    observations.append(
                        Observation(
                            observation_id=row["id"],
                            entity_id=self._benchmark_id(sector),
                            metric_key=row["metric"],
                            value=row["value"],
                            unit=row["unit"] or "number",
                            period_end=row["as_of"],
                            observed_at=row["as_of"],
                            source_id=row["source_id"],
                        )
                    )
            observations.sort(
                key=lambda item: (item.observed_at, item.period_end), reverse=True
            )
            if latest_only:
                observations = self._latest_observations(observations)
            observations = observations[:bounded_limit]
            sources = self._sources(
                connection, [item.source_id for item in observations]
            )
            version = self._dataset_version(connection)
        return ObservationResult(
            dataset_version=version,
            observations=observations,
            sources=sources,
            warnings=list(dict.fromkeys(warnings)),
        )

    def query_events(
        self,
        sector: Sector,
        entity_ids: list[str],
        event_kinds: list[str],
        latest_only: bool,
        limit: int,
    ) -> EventResult:
        """Map operating signals to source-linked event records."""
        bounded_limit = self._bounded_limit(limit)
        company_ids = self._validated_company_ids(sector, entity_ids)
        if not company_ids or not event_kinds:
            return EventResult(dataset_version=self.version())
        company_marks = ",".join("?" for _ in company_ids)
        event_marks = ",".join("?" for _ in event_kinds)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT os.id, os.company_id, os.signal_type, os.observed_at,
                       os.value_numeric, os.value_text, os.unit, os.source_id,
                       os.quality_caveat, s.published_at AS source_published_at
                FROM operating_signals AS os
                JOIN companies AS c ON c.id = os.company_id
                JOIN sources AS s ON s.id = os.source_id
                WHERE c.sector_id = ?
                  AND os.company_id IN ({company_marks})
                  AND os.signal_type IN ({event_marks})
                ORDER BY os.observed_at DESC, s.published_at DESC, os.id DESC
                """,
                [sector.value, *company_ids, *event_kinds],
            ).fetchall()
            if latest_only:
                rows = self._latest_rows(rows, ("company_id", "signal_type"))
            rows = rows[:bounded_limit]
            events = [self._event(row) for row in rows]
            sources = self._sources(connection, [event.source_id for event in events])
            version = self._dataset_version(connection)
            warnings = [
                f"{row['id']}: {row['quality_caveat']}"
                for row in rows
                if row["quality_caveat"]
            ]
        return EventResult(
            dataset_version=version,
            events=events,
            sources=sources,
            warnings=list(dict.fromkeys(warnings)),
        )

    def version(self) -> str:
        """Return a stable description of the current source set."""
        with closing(self._connect()) as connection:
            return self._dataset_version(connection)

    def _connect(self) -> sqlite3.Connection:
        if not self._database_path.is_file():
            raise RepositoryError(f"database does not exist: {self._database_path}")
        uri = f"file:{quote(str(self._database_path))}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=1.0)
        except sqlite3.Error as exc:
            raise RepositoryError("unable to open the dataset read-only") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _validated_company_ids(
        self, sector: Sector, entity_ids: list[str]
    ) -> list[str]:
        requested = [item for item in entity_ids if not item.startswith("benchmark:")]
        if not requested:
            return []
        marks = ",".join("?" for _ in requested)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT id FROM companies WHERE sector_id = ? AND id IN ({marks})",
                [sector.value, *requested],
            ).fetchall()
        valid = [row["id"] for row in rows]
        if set(valid) != set(requested):
            raise ValueError(
                "one or more entity IDs do not belong to the selected sector"
            )
        return valid

    @staticmethod
    def _annual_rows(
        connection: sqlite3.Connection,
        sector: Sector,
        company_ids: list[str],
    ) -> list[sqlite3.Row]:
        marks = ",".join("?" for _ in company_ids)
        return connection.execute(
            f"""
            SELECT af.* FROM annual_financial_snapshots AS af
            JOIN companies AS c ON c.id = af.company_id
            WHERE c.sector_id = ? AND af.company_id IN ({marks})
            ORDER BY af.period_end DESC
            """,
            [sector.value, *company_ids],
        ).fetchall()

    @staticmethod
    def _market_rows(
        connection: sqlite3.Connection,
        sector: Sector,
        company_ids: list[str],
    ) -> list[sqlite3.Row]:
        marks = ",".join("?" for _ in company_ids)
        return connection.execute(
            f"""
            SELECT ms.* FROM market_snapshots AS ms
            JOIN companies AS c ON c.id = ms.company_id
            WHERE c.sector_id = ? AND ms.company_id IN ({marks})
            ORDER BY ms.as_of DESC
            """,
            [sector.value, *company_ids],
        ).fetchall()

    @staticmethod
    def _event(row: sqlite3.Row) -> Event:
        value = row["value_text"]
        if value is None and row["value_numeric"] is not None:
            value = f"{row['value_numeric']:g} {row['unit'] or ''}".strip()
        return Event(
            event_id=row["id"],
            entity_id=row["company_id"],
            event_kind=row["signal_type"],
            title=f"{row['signal_type'].replace('_', ' ').title()} signal",
            summary=value or "No additional description supplied.",
            occurred_at=row["observed_at"],
            published_at=row["source_published_at"] or row["observed_at"],
            source_id=row["source_id"],
        )

    @staticmethod
    def _sources(
        connection: sqlite3.Connection,
        source_ids: Iterable[str],
    ) -> list[SourceRef]:
        unique_ids = list(dict.fromkeys(source_ids))
        if not unique_ids:
            return []
        marks = ",".join("?" for _ in unique_ids)
        rows = connection.execute(
            f"""
            SELECT id, title, url, publisher, published_at, retrieved_at
            FROM sources WHERE id IN ({marks})
            """,
            unique_ids,
        ).fetchall()
        return [
            SourceRef(
                source_id=row["id"],
                title=row["title"],
                url=row["url"],
                publisher=row["publisher"],
                published_at=row["published_at"],
                retrieved_at=row["retrieved_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _dataset_version(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM dataset_metadata WHERE key = 'dataset_version'"
        ).fetchone()
        if row is None:
            raise RepositoryError("dataset metadata has no dataset_version")
        return row["value"]

    @staticmethod
    def _latest_observations(items: list[Observation]) -> list[Observation]:
        seen: set[tuple[str, str]] = set()
        result: list[Observation] = []
        for item in items:
            key = (item.entity_id, item.metric_key)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _latest_rows(
        rows: list[sqlite3.Row],
        keys: tuple[str, ...],
    ) -> list[sqlite3.Row]:
        seen: set[tuple[object, ...]] = set()
        result: list[sqlite3.Row] = []
        for row in rows:
            key = tuple(row[item] for item in keys)
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    @staticmethod
    def _benchmark_id(sector: Sector) -> str:
        return f"benchmark:{sector.value}"

    @staticmethod
    def _append_caveat(warnings: list[str], row_id: str, caveat: str) -> None:
        if caveat:
            warnings.append(f"{row_id}: {caveat}")

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        return limit
