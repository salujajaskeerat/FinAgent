"""Internal structured planning and drafting models."""

from __future__ import annotations

from pydantic import Field

from finagent.contracts.api import DerivedMetric, Finding, StrictModel
from finagent.contracts.mcp import Event, Observation


class RetrievalPlan(StrictModel):
    """Validated request for MCP evidence retrieval."""

    entity_ids: list[str]
    metric_keys: list[str]
    event_kinds: list[str] = Field(default_factory=list)
    latest_only: bool = False


class DraftAnalysis(StrictModel):
    """Model-produced answer before final grounding validation."""

    answer_markdown: str
    findings: list[Finding]
    limitations: list[str] = Field(default_factory=list)


class EvidenceBundle(StrictModel):
    """Evidence made available to synthesis."""

    company_ids: list[str] = Field(default_factory=list)
    # entity_id -> "Name (TICKER)", so prose can name companies while findings
    # keep canonical IDs for validation.
    entity_names: dict[str, str] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    derived: list[DerivedMetric] = Field(default_factory=list)
    source_ids: set[str] = Field(default_factory=set)
    warnings: list[str] = Field(default_factory=list)
