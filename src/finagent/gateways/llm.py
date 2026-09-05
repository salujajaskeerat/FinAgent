"""LLM gateway implementations."""

from __future__ import annotations

from finagent.contracts.api import AnalysisRequest, Finding, Persona
from finagent.contracts.mcp import DatasetCatalog, EntityKind
from finagent.core.models import DraftAnalysis, EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy


class FakeLlmGateway:
    """Deterministic no-key model substitute for tests and local smoke runs.

    The fake preserves the same typed boundary as a production provider adapter. It
    intentionally does not pretend to provide investment analysis.
    """

    async def plan(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        catalog: DatasetCatalog,
        entity_ids: list[str],
    ) -> RetrievalPlan:
        """Build a deterministic policy-aware retrieval plan."""
        selected_ids = list(entity_ids)
        if request.persona is Persona.MUTUAL_FUND:
            selected_ids.extend(
                entity.entity_id
                for entity in catalog.entities
                if entity.kind is EntityKind.BENCHMARK
            )
        query_lower = request.query.lower()
        latest_only = any(token in query_lower for token in ("latest", "most recent"))
        event_kinds = list(policy.event_kinds)
        if "headcount" in query_lower or "hiring" in query_lower:
            event_kinds = ["headcount"]
        return RetrievalPlan(
            entity_ids=list(dict.fromkeys(selected_ids)),
            metric_keys=list(policy.required_metrics),
            event_kinds=event_kinds,
            latest_only=latest_only,
        )

    async def synthesize(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
    ) -> DraftAnalysis:
        """Return a deterministic source-linked draft."""
        source_ids = sorted(evidence.source_ids)
        if not source_ids:
            return DraftAnalysis(
                answer_markdown="No sourced evidence was supplied.", findings=[]
            )
        sections = "\n\n".join(
            f"### {section}\nEvidence retrieved for the configured {policy.label} lens."
            for section in policy.required_sections
        )
        findings = [
            Finding(
                text=(
                    f"The dataset returned {len(evidence.observations)} observations and "
                    f"{len(evidence.events)} operating signals for analysis."
                ),
                company_ids=evidence.company_ids,
                source_ids=source_ids,
            )
        ]
        if evidence.events:
            latest_event = max(
                evidence.events,
                key=lambda item: (
                    item.occurred_at or item.published_at,
                    item.published_at,
                ),
            )
            findings.append(
                Finding(
                    text=(
                        f"Latest {latest_event.event_kind} signal: "
                        f"{latest_event.summary} "
                        f"(observed {latest_event.occurred_at or latest_event.published_at})."
                    ),
                    company_ids=[latest_event.entity_id],
                    source_ids=[latest_event.source_id],
                )
            )
        return DraftAnalysis(
            answer_markdown=(
                f"## {policy.label} view\n\n{sections}\n\n"
                "This deterministic fake gateway validates plumbing only; configure a real "
                "LLM adapter for substantive investment analysis."
            ),
            findings=findings,
        )

    async def repair(
        self,
        request: AnalysisRequest,
        policy: PersonaPolicy,
        evidence: EvidenceBundle,
        draft: DraftAnalysis,
        issues: list[str],
    ) -> DraftAnalysis:
        """Replace invalid links with identifiers present in retrieved evidence."""
        del draft, issues
        return await self.synthesize(request, policy, evidence)
