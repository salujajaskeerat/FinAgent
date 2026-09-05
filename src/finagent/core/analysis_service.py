"""Bounded application workflow shared by every interface."""

from __future__ import annotations

import asyncio
from datetime import date
from uuid import UUID, uuid4

from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    CatalogResponse,
    CompanyRef,
    EvidenceStatus,
    PersonaOption,
    Sector,
    SectorOption,
)
from finagent.contracts.mcp import (
    DatasetCatalog,
    EntityKind,
    EventResult,
    ObservationResult,
)
from finagent.core.errors import AnalysisTimeoutError
from finagent.core.grounding import grounding_issues
from finagent.core.models import EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.core.ports import DataGateway, LlmGateway
from finagent.core.state import AnalysisState, StateTrace


class AnalysisService:
    """Coordinate scope resolution, retrieval, synthesis, and grounding."""

    def __init__(
        self,
        data_gateway: DataGateway,
        llm_gateway: LlmGateway,
        policies: PersonaPolicyStore,
        deadline_seconds: float = 45.0,
    ) -> None:
        self._data = data_gateway
        self._llm = llm_gateway
        self._policies = policies
        self._deadline_seconds = deadline_seconds

    async def catalog(self, sector: Sector) -> CatalogResponse:
        """Return UI configuration and data coverage for a sector."""
        catalog = await self._data.get_catalog(sector)
        personas = [
            PersonaOption(
                value=policy.persona,
                label=policy.label,
                description=policy.description,
            )
            for policy in self._policies.all()
        ]
        companies = [
            CompanyRef(
                company_id=entity.entity_id, name=entity.name, ticker=entity.ticker
            )
            for entity in catalog.entities
            if entity.kind is EntityKind.COMPANY
        ]
        return CatalogResponse(
            dataset_version=catalog.dataset_version,
            personas=personas,
            sectors=[
                SectorOption(value=item, label=item.value.title()) for item in Sector
            ],
            companies=companies,
            metric_keys=catalog.metric_keys,
            coverage_start=catalog.coverage_start,
            coverage_end=catalog.coverage_end,
        )

    async def analyze(
        self,
        request: AnalysisRequest,
        request_id: UUID | None = None,
    ) -> AnalysisResponse:
        """Execute one analysis within a global deadline.

        Parameters
        ----------
        request
            Validated public request.
        request_id
            Correlation identifier supplied by the HTTP adapter.

        Returns
        -------
        AnalysisResponse
            A grounded answer or an honest domain fallback.

        Raises
        ------
        AnalysisTimeoutError
            If the bounded workflow exceeds its deadline.
        """
        run_id = request_id or uuid4()
        try:
            async with asyncio.timeout(self._deadline_seconds):
                return await self._analyze(request, run_id)
        except TimeoutError as exc:
            raise AnalysisTimeoutError("analysis deadline exceeded") from exc

    async def _analyze(
        self, request: AnalysisRequest, request_id: UUID
    ) -> AnalysisResponse:
        trace = StateTrace()
        policy = self._policies.get(request.persona)

        trace.move(AnalysisState.RESOLVING_SCOPE)
        catalog, resolution = await asyncio.gather(
            self._data.get_catalog(request.sector),
            self._data.resolve_companies(request.sector, request.query),
        )
        if resolution.unresolved_mentions:
            trace.move(AnalysisState.COMPLETED)
            supported = ", ".join(
                entity.name
                for entity in catalog.entities
                if entity.kind is EntityKind.COMPANY
            )
            return AnalysisResponse(
                request_id=request_id,
                status=AnalysisStatus.OUT_OF_SCOPE,
                persona=request.persona,
                sector=request.sector,
                answer_markdown=(
                    "The requested company is outside this dataset, so I cannot provide a "
                    "data-grounded opinion."
                ),
                evidence_status=EvidenceStatus.NONE,
                limitations=[
                    f"Supported companies for {request.sector.value}: {supported}."
                ],
            )

        company_entities = [
            entity for entity in catalog.entities if entity.kind is EntityKind.COMPANY
        ]
        target_ids = [item.entity_id for item in resolution.resolved]
        if not target_ids:
            target_ids = [entity.entity_id for entity in company_entities]
        if not target_ids:
            trace.move(AnalysisState.COMPLETED)
            return self._insufficient(
                request, request_id, "No companies are loaded for this sector."
            )

        trace.move(AnalysisState.PLANNING)
        proposed_plan = await self._llm.plan(request, policy, catalog, target_ids)
        plan = self._constrain_plan(proposed_plan, catalog, target_ids)

        trace.move(AnalysisState.RETRIEVING)
        observations, events = await asyncio.gather(
            self._get_observations(request, plan),
            self._get_events(request, plan),
        )
        if not observations.observations and not events.events:
            trace.move(AnalysisState.COMPLETED)
            return self._insufficient(
                request,
                request_id,
                "The dataset contains no observations relevant to this question.",
            )

        source_map = {
            source.source_id: source
            for source in [*observations.sources, *events.sources]
        }
        evidence = EvidenceBundle(
            company_ids=target_ids,
            observations=observations.observations,
            events=events.events,
            source_ids=set(source_map),
            warnings=[*observations.warnings, *events.warnings],
        )

        trace.move(AnalysisState.SYNTHESIZING)
        draft = await self._llm.synthesize(request, policy, evidence)
        trace.move(AnalysisState.VALIDATING)
        allowed_companies = {entity.entity_id for entity in company_entities}
        issues = grounding_issues(draft, allowed_companies, evidence.source_ids)
        if issues:
            trace.move(AnalysisState.REPAIRING)
            draft = await self._llm.repair(request, policy, evidence, draft, issues)
            trace.move(AnalysisState.VALIDATING)
            issues = grounding_issues(draft, allowed_companies, evidence.source_ids)
        if issues:
            trace.move(AnalysisState.COMPLETED)
            return self._insufficient(
                request,
                request_id,
                "The generated answer could not be linked reliably to retrieved evidence.",
            )

        trace.move(AnalysisState.COMPLETED)
        referenced_ids = {
            company_id
            for finding in draft.findings
            for company_id in finding.company_ids
        }
        referenced_companies = [
            CompanyRef(
                company_id=entity.entity_id, name=entity.name, ticker=entity.ticker
            )
            for entity in company_entities
            if entity.entity_id in referenced_ids
        ]
        limitations = [*draft.limitations, *evidence.warnings]
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.ANSWERED,
            persona=request.persona,
            sector=request.sector,
            answer_markdown=draft.answer_markdown,
            findings=draft.findings,
            companies=referenced_companies,
            sources=list(source_map.values()),
            evidence_status=(
                EvidenceStatus.PARTIAL if limitations else EvidenceStatus.SUFFICIENT
            ),
            data_as_of=self._data_as_of(observations, events),
            limitations=limitations,
        )

    async def _get_observations(
        self,
        request: AnalysisRequest,
        plan: RetrievalPlan,
    ) -> ObservationResult:
        if not plan.metric_keys:
            return ObservationResult(dataset_version="unknown")
        return await self._data.query_observations(
            request.sector,
            plan.entity_ids,
            plan.metric_keys,
            plan.latest_only,
        )

    async def _get_events(
        self,
        request: AnalysisRequest,
        plan: RetrievalPlan,
    ) -> EventResult:
        if not plan.event_kinds:
            return EventResult(dataset_version="unknown")
        return await self._data.query_events(
            request.sector,
            plan.entity_ids,
            plan.event_kinds,
            plan.latest_only,
        )

    @staticmethod
    def _constrain_plan(
        plan: RetrievalPlan,
        catalog: DatasetCatalog,
        target_ids: list[str],
    ) -> RetrievalPlan:
        allowed_entities = {entity.entity_id for entity in catalog.entities}
        allowed_metrics = set(catalog.metric_keys)
        allowed_events = set(catalog.event_kinds)
        selected_entities = [
            item for item in plan.entity_ids if item in allowed_entities
        ]
        if not selected_entities:
            selected_entities = [
                item for item in target_ids if item in allowed_entities
            ]
        return RetrievalPlan(
            entity_ids=selected_entities,
            metric_keys=[item for item in plan.metric_keys if item in allowed_metrics],
            event_kinds=[item for item in plan.event_kinds if item in allowed_events],
            latest_only=plan.latest_only,
        )

    @staticmethod
    def _data_as_of(
        observations: ObservationResult, events: EventResult
    ) -> date | None:
        dates = [item.observed_at for item in observations.observations]
        dates.extend(item.published_at for item in events.events)
        return max(dates, default=None)

    @staticmethod
    def _insufficient(
        request: AnalysisRequest,
        request_id: UUID,
        limitation: str,
    ) -> AnalysisResponse:
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            persona=request.persona,
            sector=request.sector,
            answer_markdown="I do not have enough sourced data to answer this question reliably.",
            evidence_status=EvidenceStatus.NONE,
            limitations=[limitation],
        )
