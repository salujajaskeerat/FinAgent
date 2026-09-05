"""Strict contracts for constrained model-assisted entity resolution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from finagent.contracts.api import StrictModel


class EntityResolutionStatus(StrEnum):
    """Allowed outcomes from the constrained resolver."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"
    BROAD_QUERY = "broad_query"


class EntityResolutionReason(StrEnum):
    """Machine-readable reason for an entity-resolution outcome."""

    EXACT_NAME = "exact_name"
    TICKER = "ticker"
    ALIAS = "alias"
    SEMANTIC_REFERENCE = "semantic_reference"
    AMBIGUOUS = "ambiguous"
    NOT_IN_CATALOG = "not_in_catalog"
    BROAD_QUESTION = "broad_question"


class EntityMatch(StrictModel):
    """One model-selected catalog entity."""

    entity_id: str = Field(min_length=1)
    mention: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class EntityResolution(StrictModel):
    """Validated output from the constrained resolver."""

    status: EntityResolutionStatus
    matches: list[EntityMatch] = Field(default_factory=list)
    reason_code: EntityResolutionReason

    @model_validator(mode="after")
    def validate_status_shape(self) -> EntityResolution:
        """Enforce status-specific match cardinality and reason codes."""
        if self.status is EntityResolutionStatus.MATCHED and len(self.matches) != 1:
            raise ValueError("matched resolution must contain exactly one match")
        if self.status is not EntityResolutionStatus.MATCHED and self.matches:
            raise ValueError("non-matched resolution must not contain matches")
        allowed_reasons = {
            EntityResolutionStatus.MATCHED: {
                EntityResolutionReason.EXACT_NAME,
                EntityResolutionReason.TICKER,
                EntityResolutionReason.ALIAS,
                EntityResolutionReason.SEMANTIC_REFERENCE,
            },
            EntityResolutionStatus.AMBIGUOUS: {EntityResolutionReason.AMBIGUOUS},
            EntityResolutionStatus.NO_MATCH: {EntityResolutionReason.NOT_IN_CATALOG},
            EntityResolutionStatus.BROAD_QUERY: {EntityResolutionReason.BROAD_QUESTION},
        }
        if self.reason_code not in allowed_reasons[self.status]:
            raise ValueError("reason_code is inconsistent with resolution status")
        return self
