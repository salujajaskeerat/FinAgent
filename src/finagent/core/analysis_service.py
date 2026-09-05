"""Bounded application workflow shared by every interface."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from uuid import UUID, uuid4

from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisTrace,
    CatalogResponse,
    CompanyRef,
    EvidenceCoverage,
    EvidenceStatus,
    PersonaOption,
    PlanRef,
    Sector,
    SectorOption,
)
from finagent.contracts.entity_resolution import EntityResolutionStatus
from finagent.contracts.mcp import (
    CatalogEntity,
    DatasetCatalog,
    EntityKind,
    EventResult,
    ObservationResult,
)
from finagent.core.derived import derive
from finagent.core.errors import AnalysisTimeoutError
from finagent.core.grounding import grounding_issues
from finagent.core.models import EvidenceBundle, RetrievalPlan
from finagent.core.persona_policy import PersonaPolicy, PersonaPolicyStore
from finagent.core.ports import DataGateway, EntityResolver, LlmGateway
from finagent.core.state import AnalysisState, StateTrace

ProgressCallback = Callable[[AnalysisState, str], Awaitable[None]]


class AnalysisService:
    """Coordinate scope resolution, retrieval, synthesis, and grounding."""

    def __init__(
        self,
        data_gateway: DataGateway,
        llm_gateway: LlmGateway,
        policies: PersonaPolicyStore,
        deadline_seconds: float = 75.0,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        self._data = data_gateway
        self._llm = llm_gateway
        self._policies = policies
        self._deadline_seconds = deadline_seconds
        self._entity_resolver = entity_resolver

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
        on_progress: ProgressCallback | None = None,
    ) -> AnalysisResponse:
        """Execute one analysis within a global deadline.

        Parameters
        ----------
        request
            Validated public request.
        request_id
            Correlation identifier supplied by the HTTP adapter.
        on_progress
            Optional coroutine called on every state transition with the state
            just entered and a one-line summary of what the previous step did.

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
                return await self._analyze(request, run_id, on_progress)
        except TimeoutError as exc:
            raise AnalysisTimeoutError("analysis deadline exceeded") from exc

    async def _analyze(
        self,
        request: AnalysisRequest,
        request_id: UUID,
        on_progress: ProgressCallback | None,
    ) -> AnalysisResponse:
        trace = StateTrace()
        policy = self._policies.get(request.persona)

        async def step(state: AnalysisState, message: str = "") -> None:
            trace.move(state)
            if on_progress is not None:
                await on_progress(state, message)

        await step(AnalysisState.RESOLVING_SCOPE)
        catalog, resolution = await asyncio.gather(
            self._data.get_catalog(request.sector),
            self._data.resolve_companies(request.sector, request.query),
        )
        company_entities = [
            entity for entity in catalog.entities if entity.kind is EntityKind.COMPANY
        ]
        target_ids = [item.entity_id for item in resolution.resolved]
        if resolution.unresolved_mentions:
            fallback_id = await self._fallback_entity_id(
                request.query,
                request.sector,
                company_entities,
            )
            if fallback_id is not None:
                target_ids = [fallback_id]
            else:
                await step(
                    AnalysisState.COMPLETED,
                    f"No catalog match for '{resolution.unresolved_mentions[0]}'",
                )
                return self._out_of_scope(
                    request, request_id, company_entities, self._trace(trace, catalog)
                )

        if not target_ids:
            target_ids = [entity.entity_id for entity in company_entities]
        if not target_ids:
            await step(AnalysisState.COMPLETED, "No companies loaded")
            return self._insufficient(
                request,
                request_id,
                "No companies are loaded for this sector.",
                self._trace(trace, catalog),
            )

        await step(
            AnalysisState.PLANNING,
            f"Resolved {len(target_ids)} of {len(company_entities)} companies",
        )
        proposed_plan = await self._llm.plan(request, policy, catalog, target_ids)
        plan = self._constrain_plan(proposed_plan, catalog, target_ids, policy)
        llm_calls = 1

        await step(
            AnalysisState.RETRIEVING,
            f"Model proposed {len(proposed_plan.metric_keys)} metrics; running "
            f"{len(plan.metric_keys)} metrics and {len(plan.event_kinds)} event kinds",
        )
        observations, events = await asyncio.gather(
            self._get_observations(request, plan),
            self._get_events(request, plan),
        )
        if not observations.observations and not events.events:
            await step(AnalysisState.COMPLETED, "No observations returned")
            return self._insufficient(
                request,
                request_id,
                "The dataset contains no observations relevant to this question.",
                self._trace(trace, catalog, proposed_plan, plan, llm_calls=llm_calls),
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

        # Deterministic arithmetic happens here, never inside the model.
        await step(
            AnalysisState.CALCULATING,
            f"Retrieved {len(observations.observations)} observations and "
            f"{len(events.events)} signals from {len(source_map)} sources",
        )
        evidence.derived = derive(evidence, policy)

        await step(
            AnalysisState.SYNTHESIZING,
            f"Computed {len(evidence.derived)} derived metrics",
        )
        draft = await self._llm.synthesize(request, policy, evidence)
        llm_calls += 1
        await step(AnalysisState.VALIDATING, f"Drafted {len(draft.findings)} findings")
        allowed_companies = {entity.entity_id for entity in company_entities}
        issues = grounding_issues(draft, allowed_companies, evidence.source_ids)
        repaired = False
        if issues:
            await step(AnalysisState.REPAIRING, f"{len(issues)} grounding issue(s)")
            draft = await self._llm.repair(request, policy, evidence, draft, issues)
            llm_calls += 1
            repaired = True
            await step(AnalysisState.VALIDATING, "Repaired draft")
            issues = grounding_issues(draft, allowed_companies, evidence.source_ids)
        if issues:
            await step(AnalysisState.COMPLETED, "Findings still ungrounded")
            return self._insufficient(
                request,
                request_id,
                "The generated answer could not be linked reliably to retrieved evidence.",
                self._trace(trace, catalog, proposed_plan, plan, repaired, llm_calls),
            )

        await step(
            AnalysisState.COMPLETED,
            f"All {len(draft.findings)} findings grounded in {len(source_map)} sources",
        )
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
        coverage = self._coverage(policy, plan, observations, events)
        limitations = [*draft.limitations, *self._collapse_warnings(evidence.warnings)]
        if coverage.missing_metrics:
            limitations.insert(
                0,
                "Required metrics not available in the dataset: "
                + ", ".join(coverage.missing_metrics)
                + ".",
            )
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.ANSWERED,
            persona=request.persona,
            sector=request.sector,
            answer_markdown=draft.answer_markdown,
            findings=draft.findings,
            derived_metrics=evidence.derived,
            companies=referenced_companies,
            sources=list(source_map.values()),
            evidence_status=(
                EvidenceStatus.PARTIAL
                if coverage.missing_metrics
                else EvidenceStatus.SUFFICIENT
            ),
            coverage=coverage,
            data_as_of=self._data_as_of(observations, events),
            limitations=limitations,
            trace=self._trace(trace, catalog, proposed_plan, plan, repaired, llm_calls),
        )

    @staticmethod
    def _trace(
        trace: StateTrace,
        catalog: DatasetCatalog,
        proposed: RetrievalPlan | None = None,
        constrained: RetrievalPlan | None = None,
        repaired: bool = False,
        llm_calls: int = 0,
    ) -> AnalysisTrace:
        """Summarize the run: states visited and model proposal versus query run."""

        def ref(plan: RetrievalPlan | None) -> PlanRef | None:
            return PlanRef(**plan.model_dump()) if plan else None

        return AnalysisTrace(
            states=[state.value for state in trace.history],
            dataset_version=catalog.dataset_version,
            proposed_plan=ref(proposed),
            constrained_plan=ref(constrained),
            repaired=repaired,
            llm_calls=llm_calls,
        )

    @staticmethod
    def _collapse_warnings(warnings: list[str]) -> list[str]:
        """Group per-row MCP caveats ("<row id>: <text>") by their text."""
        counts: dict[str, int] = {}
        for warning in warnings:
            _, _, text = warning.partition(": ")
            text = text or warning
            counts[text] = counts.get(text, 0) + 1
        return [
            f"{text} ({count} rows)" if count > 1 else text
            for text, count in counts.items()
        ]

    @staticmethod
    def _coverage(
        policy: PersonaPolicy,
        plan: RetrievalPlan,
        observations: ObservationResult,
        events: EventResult,
    ) -> EvidenceCoverage:
        """Compare persona-required inputs with what MCP actually returned."""
        seen_metrics = {item.metric_key for item in observations.observations}
        seen_events = {item.event_kind for item in events.events}
        return EvidenceCoverage(
            required_metrics=list(policy.required_metrics),
            available_metrics=[
                key for key in policy.required_metrics if key in seen_metrics
            ],
            missing_metrics=[
                key for key in policy.required_metrics if key not in seen_metrics
            ],
            requested_event_kinds=list(plan.event_kinds),
            available_event_kinds=[
                kind for kind in plan.event_kinds if kind in seen_events
            ],
        )

    async def _fallback_entity_id(
        self,
        query: str,
        sector: Sector,
        candidates: list[CatalogEntity],
    ) -> str | None:
        """Accept one high-confidence model match from the supplied catalog only."""
        if self._entity_resolver is None:
            return None
        try:
            result = await self._entity_resolver.resolve(query, sector, candidates)
        except Exception:  # noqa: BLE001 - optional resolution must fail closed.
            return None
        if result.status is not EntityResolutionStatus.MATCHED:
            return None
        match = result.matches[0]
        allowed_ids = {candidate.entity_id for candidate in candidates}
        mention_is_from_query = match.mention.casefold() in query.casefold()
        if (
            match.entity_id not in allowed_ids
            or match.confidence < 0.85
            or not mention_is_from_query
        ):
            return None
        return match.entity_id

    @staticmethod
    def _out_of_scope(
        request: AnalysisRequest,
        request_id: UUID,
        company_entities: list[CatalogEntity],
        trace: AnalysisTrace | None = None,
    ) -> AnalysisResponse:
        """Return the stable response for an unresolved explicit company."""
        supported = ", ".join(entity.name for entity in company_entities)
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
            trace=trace,
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
        policy: PersonaPolicy,
    ) -> RetrievalPlan:
        """Turn the model's proposal into the query the application will run.

        The model's plan is only a seed. Every identifier is intersected with
        the MCP catalog, and the persona policy's required and preferred inputs
        are always added so persona retrieval is deterministic regardless of
        what the model proposed.
        """
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
        if policy.include_benchmark:
            selected_entities.extend(
                entity.entity_id
                for entity in catalog.entities
                if entity.kind is EntityKind.BENCHMARK
            )
        metric_keys = [
            *plan.metric_keys,
            *policy.required_metrics,
            *policy.preferred_metrics,
        ]
        event_kinds = [*plan.event_kinds, *policy.event_kinds]
        return RetrievalPlan(
            entity_ids=list(dict.fromkeys(selected_entities)),
            metric_keys=[
                item for item in dict.fromkeys(metric_keys) if item in allowed_metrics
            ],
            event_kinds=[
                item for item in dict.fromkeys(event_kinds) if item in allowed_events
            ],
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
        trace: AnalysisTrace | None = None,
    ) -> AnalysisResponse:
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.INSUFFICIENT_DATA,
            persona=request.persona,
            sector=request.sector,
            answer_markdown="I do not have enough sourced data to answer this question reliably.",
            evidence_status=EvidenceStatus.NONE,
            limitations=[limitation],
            trace=trace,
        )
