"""Deterministic derived metrics computed from retrieved evidence.

The model never does arithmetic. Every derived value is a pure function of
observations that MCP returned, carries the identifiers of its inputs, and is
selected per persona through the ``derived_metrics`` policy key.
"""

from __future__ import annotations

from collections.abc import Callable

from finagent.contracts.api import DerivedMetric
from finagent.contracts.mcp import Observation
from finagent.core.models import EvidenceBundle
from finagent.core.persona_policy import PersonaPolicy

_Series = dict[str, list[Observation]]  # metric_key -> observations, newest first
_Derivation = Callable[[str, _Series, _Series | None], DerivedMetric | None]


def derive(evidence: EvidenceBundle, policy: PersonaPolicy) -> list[DerivedMetric]:
    """Compute the persona's derived metrics for every company in the evidence.

    Parameters
    ----------
    evidence
        Retrieved observations and events.
    policy
        Persona policy whose ``derived_metrics`` selects the derivations.

    Returns
    -------
    list[DerivedMetric]
        Values that could be computed; derivations lacking inputs are skipped.
    """
    by_entity = _group(evidence.observations)
    benchmark = next(
        (series for key, series in by_entity.items() if key.startswith("benchmark:")),
        None,
    )
    results: list[DerivedMetric] = []
    for entity_id, series in by_entity.items():
        if entity_id.startswith("benchmark:"):
            continue
        for key in policy.derived_metrics:
            derivation = DERIVATIONS.get(key)
            if derivation is None:
                continue
            result = derivation(entity_id, series, benchmark)
            if result is not None:
                results.append(result)
    return results


def _group(observations: list[Observation]) -> dict[str, _Series]:
    grouped: dict[str, _Series] = {}
    for item in observations:
        grouped.setdefault(item.entity_id, {}).setdefault(item.metric_key, []).append(
            item
        )
    for series in grouped.values():
        for items in series.values():
            items.sort(key=lambda item: item.period_end, reverse=True)
    return dict(sorted(grouped.items()))


def _latest(series: _Series, metric: str, index: int = 0) -> Observation | None:
    items = series.get(metric, [])
    return items[index] if len(items) > index else None


def _metric(
    key: str,
    entity_id: str,
    value: float,
    unit: str,
    formula: str,
    inputs: list[Observation],
    caveat: str | None = None,
) -> DerivedMetric:
    return DerivedMetric(
        key=key,
        entity_id=entity_id,
        value=value,
        unit=unit,
        period_end=max(item.period_end for item in inputs),
        formula=formula,
        input_observation_ids=[item.observation_id for item in inputs],
        input_source_ids=sorted({item.source_id for item in inputs}),
        caveat=caveat,
    )


def _growth(metric: str, key: str) -> _Derivation:
    def derivation(
        entity_id: str, series: _Series, _benchmark: _Series | None
    ) -> DerivedMetric | None:
        current, prior = _latest(series, metric), _latest(series, metric, 1)
        if current is None or prior is None or prior.value == 0:
            return None
        return _metric(
            key,
            entity_id,
            current.value / prior.value - 1,
            "ratio",
            f"{metric}[{current.period_end}] / {metric}[{prior.period_end}] - 1",
            [current, prior],
        )

    return derivation


def _delta(metric: str, key: str) -> _Derivation:
    def derivation(
        entity_id: str, series: _Series, _benchmark: _Series | None
    ) -> DerivedMetric | None:
        current, prior = _latest(series, metric), _latest(series, metric, 1)
        if current is None or prior is None:
            return None
        return _metric(
            key,
            entity_id,
            current.value - prior.value,
            "ratio_points",
            f"{metric}[{current.period_end}] - {metric}[{prior.period_end}]",
            [current, prior],
        )

    return derivation


def _ratio(
    numerator: str, denominator: str, key: str, *, absolute: bool = False
) -> _Derivation:
    def derivation(
        entity_id: str, series: _Series, _benchmark: _Series | None
    ) -> DerivedMetric | None:
        top, bottom = _latest(series, numerator), _latest(series, denominator)
        if top is None or bottom is None or bottom.value == 0:
            return None
        if top.period_end != bottom.period_end:
            return None
        value = abs(top.value) if absolute else top.value
        return _metric(
            key,
            entity_id,
            value / bottom.value,
            "ratio",
            f"{'|' + numerator + '|' if absolute else numerator} / {denominator}",
            [top, bottom],
        )

    return derivation


def _net_debt(
    entity_id: str, series: _Series, _benchmark: _Series | None
) -> DerivedMetric | None:
    debt, cash = _latest(series, "total_debt"), _latest(series, "cash_and_equivalents")
    if debt is None or cash is None or debt.period_end != cash.period_end:
        return None
    return _metric(
        "net_debt",
        entity_id,
        debt.value - cash.value,
        debt.unit,
        "total_debt - cash_and_equivalents",
        [debt, cash],
    )


def _net_debt_to_fcf(
    entity_id: str, series: _Series, benchmark: _Series | None
) -> DerivedMetric | None:
    net_debt = _net_debt(entity_id, series, benchmark)
    fcf = _latest(series, "free_cash_flow")
    if net_debt is None or fcf is None or fcf.value <= 0:
        return None
    if fcf.period_end != net_debt.period_end:
        return None
    debt = _latest(series, "total_debt")
    cash = _latest(series, "cash_and_equivalents")
    assert debt is not None and cash is not None
    return _metric(
        "net_debt_to_free_cash_flow",
        entity_id,
        net_debt.value / fcf.value,
        "years",
        "(total_debt - cash_and_equivalents) / free_cash_flow",
        [debt, cash, fcf],
        caveat="Years of latest free cash flow needed to repay net debt; negative means net cash.",
    )


def _vs_benchmark(metric: str, key: str) -> _Derivation:
    def derivation(
        entity_id: str, series: _Series, benchmark: _Series | None
    ) -> DerivedMetric | None:
        if benchmark is None:
            return None
        own, median = _latest(series, metric), _latest(benchmark, metric)
        if own is None or median is None or median.value == 0:
            return None
        if own.unit == "ratio":
            value, unit, formula = (
                own.value - median.value,
                "ratio_points",
                f"{metric} - sector_median({metric})",
            )
        else:
            value, unit, formula = (
                own.value / median.value - 1,
                "ratio",
                f"{metric} / sector_median({metric}) - 1",
            )
        return _metric(
            key,
            entity_id,
            value,
            unit,
            formula,
            [own, median],
            caveat="Sector median is derived from the companies in this dataset, not an index.",
        )

    return derivation


DERIVATIONS: dict[str, _Derivation] = {
    "revenue_growth_yoy": _growth("revenue", "revenue_growth_yoy"),
    "operating_income_growth_yoy": _growth(
        "operating_income", "operating_income_growth_yoy"
    ),
    "operating_margin_delta_yoy": _delta(
        "operating_margin", "operating_margin_delta_yoy"
    ),
    "fcf_margin": _ratio("free_cash_flow", "revenue", "fcf_margin"),
    "capex_intensity": _ratio(
        "capital_expenditure", "revenue", "capex_intensity", absolute=True
    ),
    "net_debt": _net_debt,
    "net_debt_to_free_cash_flow": _net_debt_to_fcf,
    "operating_margin_vs_sector_median": _vs_benchmark(
        "operating_margin", "operating_margin_vs_sector_median"
    ),
    "revenue_vs_sector_median": _vs_benchmark("revenue", "revenue_vs_sector_median"),
    "free_cash_flow_vs_sector_median": _vs_benchmark(
        "free_cash_flow", "free_cash_flow_vs_sector_median"
    ),
}
