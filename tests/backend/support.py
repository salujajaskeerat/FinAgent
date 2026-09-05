"""Shared typed fakes for backend tests."""

from __future__ import annotations

from datetime import date

from finagent.contracts.api import Sector, SourceRef
from finagent.contracts.mcp import (
    CatalogEntity,
    DatasetCatalog,
    EntityKind,
    EventResult,
    Observation,
    ObservationResult,
    ResolutionResult,
)


class StubDataGateway:
    """Predictable data port fake with configurable scope resolution."""

    def __init__(
        self, unresolved: list[str] | None = None, empty: bool = False
    ) -> None:
        self._unresolved = unresolved or []
        self._empty = empty
        self.catalog_value = DatasetCatalog(
            dataset_version="fixture-v1",
            sector=Sector.TECH,
            entities=[
                CatalogEntity(
                    entity_id="cmp_example",
                    kind=EntityKind.COMPANY,
                    name="Example Systems, Inc.",
                    ticker="EXM",
                ),
                CatalogEntity(
                    entity_id="benchmark:tech",
                    kind=EntityKind.BENCHMARK,
                    name="Technology Benchmark",
                    ticker="TEST",
                ),
            ],
            metric_keys=[
                "revenue",
                "operating_income",
                "operating_margin",
                "free_cash_flow",
                "market_cap",
                "enterprise_value",
                "total_debt",
                "cash_and_equivalents",
                "capital_expenditure",
            ],
            event_kinds=["headcount", "guidance", "restructuring"],
            coverage_start=date(2023, 12, 31),
            coverage_end=date(2025, 2, 1),
        )
        self.source = SourceRef(
            source_id="src_fixture",
            title="Annual report",
            url="https://example.com/annual-report",
            publisher="Example Systems",
            published_at=date(2025, 2, 1),
            retrieved_at=date(2025, 2, 2),
        )

    async def get_catalog(self, sector: Sector) -> DatasetCatalog:
        """Return a fixed catalog."""
        assert sector is Sector.TECH
        return self.catalog_value

    async def resolve_companies(self, sector: Sector, query: str) -> ResolutionResult:
        """Return a fixed known or unknown resolution."""
        del sector, query
        if self._unresolved:
            return ResolutionResult(unresolved_mentions=self._unresolved)
        return ResolutionResult()

    async def query_observations(
        self,
        sector: Sector,
        entity_ids: list[str],
        metric_keys: list[str],
        latest_only: bool,
    ) -> ObservationResult:
        """Return one source-linked observation unless configured empty."""
        del sector, metric_keys, latest_only
        if self._empty:
            return ObservationResult(dataset_version="fixture-v1")
        return ObservationResult(
            dataset_version="fixture-v1",
            observations=[
                Observation(
                    observation_id="obs_fixture",
                    entity_id=entity_ids[0],
                    metric_key="revenue",
                    value=1_200.0,
                    unit="USD",
                    currency="USD",
                    period_end=date(2024, 12, 31),
                    observed_at=date(2025, 2, 1),
                    source_id=self.source.source_id,
                )
            ],
            sources=[self.source],
        )

    async def query_events(
        self,
        sector: Sector,
        entity_ids: list[str],
        event_kinds: list[str],
        latest_only: bool,
    ) -> EventResult:
        """Return no events; observations are sufficient for these tests."""
        del sector, entity_ids, event_kinds, latest_only
        return EventResult(dataset_version="fixture-v1")
