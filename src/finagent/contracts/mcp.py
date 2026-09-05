"""Typed contracts exchanged with MCP data tools."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field

from finagent.contracts.api import Sector, SourceRef, StrictModel


class EntityKind(StrEnum):
    """Queryable entity type."""

    COMPANY = "company"
    BENCHMARK = "benchmark"


class CatalogEntity(StrictModel):
    """Entity advertised by the dataset."""

    entity_id: str
    kind: EntityKind
    name: str
    ticker: str | None = None
    aliases: list[str] = Field(default_factory=list)


class DatasetCatalog(StrictModel):
    """Available data scope for one sector."""

    dataset_version: str
    sector: Sector
    entities: list[CatalogEntity]
    metric_keys: list[str]
    event_kinds: list[str]
    coverage_start: date | None = None
    coverage_end: date | None = None


class ResolvedEntity(StrictModel):
    """Canonical resolution of a query mention."""

    mention: str
    entity_id: str


class ResolutionResult(StrictModel):
    """Entity resolution result used for scope enforcement."""

    resolved: list[ResolvedEntity] = Field(default_factory=list)
    unresolved_mentions: list[str] = Field(default_factory=list)


class Observation(StrictModel):
    """Dated numeric or textual observation."""

    observation_id: str
    entity_id: str
    metric_key: str
    value: float
    unit: str
    currency: str | None = None
    period_start: date | None = None
    period_end: date
    observed_at: date
    source_id: str


class ObservationResult(StrictModel):
    """Observation tool output with in-band provenance."""

    dataset_version: str
    observations: list[Observation] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Event(StrictModel):
    """Dated operating event or signal."""

    event_id: str
    entity_id: str
    event_kind: str
    title: str
    summary: str
    occurred_at: date | None = None
    published_at: date
    source_id: str


class EventResult(StrictModel):
    """Operating-event tool output with in-band provenance."""

    dataset_version: str
    events: list[Event] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
