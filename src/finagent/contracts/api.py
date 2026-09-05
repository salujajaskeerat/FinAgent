"""Public HTTP API models."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    """Base model that rejects accidental contract expansion."""

    model_config = ConfigDict(extra="forbid")


class Persona(StrEnum):
    """Supported analytical personas."""

    MUTUAL_FUND = "mutual_fund_analyst"
    EQUITY = "equity_analyst"
    PE = "pe_analyst"


class Sector(StrEnum):
    """Supported sector datasets."""

    TECH = "tech"
    RETAIL = "retail"
    LOGISTICS = "logistics"


class AnalysisStatus(StrEnum):
    """Domain outcome for a valid analysis request."""

    ANSWERED = "answered"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_DATA = "insufficient_data"


class EvidenceStatus(StrEnum):
    """Deterministic evidence coverage label."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    NONE = "none"


class AnalysisRequest(StrictModel):
    """Request for one bounded analysis run."""

    query: str = Field(min_length=3, max_length=2_000)
    persona: Persona
    sector: Sector


class CompanyRef(StrictModel):
    """Canonical company reference returned to an API consumer."""

    company_id: str
    name: str
    ticker: str | None = None


class SourceRef(StrictModel):
    """Public provenance record."""

    source_id: str
    title: str
    url: HttpUrl
    publisher: str
    published_at: date | None = None
    retrieved_at: date | None = None


class Finding(StrictModel):
    """A material claim and the source IDs that support it."""

    text: str = Field(min_length=1)
    company_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(min_length=1)


class EvidenceCoverage(StrictModel):
    """Which persona-required inputs the retrieved evidence actually contained.

    ``evidence_status`` is derived from ``missing_metrics``: an answer is
    ``sufficient`` only when every required metric returned at least one
    observation. This is the machine-readable substitute for a confidence score.
    """

    required_metrics: list[str] = Field(default_factory=list)
    available_metrics: list[str] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    requested_event_kinds: list[str] = Field(default_factory=list)
    available_event_kinds: list[str] = Field(default_factory=list)


class AnalysisResponse(StrictModel):
    """Structured, source-aware analysis response."""

    schema_version: Literal["1.0"] = "1.0"
    request_id: UUID
    status: AnalysisStatus
    persona: Persona
    sector: Sector
    answer_markdown: str
    findings: list[Finding] = Field(default_factory=list)
    companies: list[CompanyRef] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    evidence_status: EvidenceStatus
    coverage: EvidenceCoverage | None = None
    data_as_of: date | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_links(self) -> Self:
        """Ensure findings never cite absent sources or companies.

        Returns
        -------
        AnalysisResponse
            The validated response.

        Raises
        ------
        ValueError
            If the response is internally inconsistent.
        """
        source_ids = {source.source_id for source in self.sources}
        company_ids = {company.company_id for company in self.companies}
        for finding in self.findings:
            if not set(finding.source_ids) <= source_ids:
                raise ValueError("a finding cites a source absent from sources")
            if not set(finding.company_ids) <= company_ids:
                raise ValueError("a finding cites a company absent from companies")
        if self.status is AnalysisStatus.ANSWERED and not self.findings:
            raise ValueError("answered responses require at least one finding")
        if (
            self.evidence_status is not EvidenceStatus.SUFFICIENT
            and not self.limitations
        ):
            raise ValueError("partial or absent evidence requires a limitation")
        return self


class PersonaOption(StrictModel):
    """UI-safe persona metadata."""

    value: Persona
    label: str
    description: str


class SectorOption(StrictModel):
    """UI-safe sector metadata."""

    value: Sector
    label: str


class CatalogResponse(StrictModel):
    """Configuration and data scope consumed by the UI."""

    dataset_version: str
    personas: list[PersonaOption]
    sectors: list[SectorOption]
    companies: list[CompanyRef]
    metric_keys: list[str]
    coverage_start: date | None = None
    coverage_end: date | None = None


class ProblemDetail(StrictModel):
    """RFC 9457-inspired error response."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: UUID
    errors: list[dict[str, object]] = Field(default_factory=list)
