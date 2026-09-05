"""Gemini rejects Pydantic's default JSON schema; check the conversion."""

from __future__ import annotations

import json

from finagent.contracts.entity_resolution import EntityResolution
from finagent.core.models import DraftAnalysis, RetrievalPlan
from finagent.gateways.providers.gemini import gemini_schema

_FORBIDDEN = {
    "additionalProperties",
    "title",
    "default",
    "$defs",
    "$ref",
    "anyOf",
    "minLength",
}


def _keys(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        found |= set(node)
        for value in node.values():
            found |= _keys(value)
    elif isinstance(node, list):
        for value in node:
            found |= _keys(value)
    return found


def test_converted_schemas_contain_only_gemini_keywords() -> None:
    for model in (RetrievalPlan, DraftAnalysis, EntityResolution):
        schema = gemini_schema(model)
        assert not (_keys(schema) & _FORBIDDEN), model.__name__
        json.dumps(schema)  # serializable


def test_nested_models_are_inlined_and_optionals_become_nullable() -> None:
    draft = gemini_schema(DraftAnalysis)
    finding = draft["properties"]["findings"]["items"]
    assert finding["type"] == "object"
    assert finding["required"] == ["text", "source_ids"]

    resolution = gemini_schema(EntityResolution)
    assert resolution["properties"]["status"]["enum"] == [
        "matched",
        "ambiguous",
        "no_match",
        "broad_query",
    ]
