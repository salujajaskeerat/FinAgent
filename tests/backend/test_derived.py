"""Pure-function tests for deterministic derived metrics."""

from __future__ import annotations

from datetime import date

from finagent.contracts.api import Persona
from finagent.contracts.mcp import Observation
from finagent.core.derived import DERIVATIONS, derive
from finagent.core.models import EvidenceBundle
from finagent.core.persona_policy import PersonaPolicyStore


def _obs(
    entity: str,
    metric: str,
    value: float,
    year: int,
    unit: str = "USD",
    source: str = "src_a",
) -> Observation:
    return Observation(
        observation_id=f"{entity}:{metric}:{year}",
        entity_id=entity,
        metric_key=metric,
        value=value,
        unit=unit,
        period_end=date(year, 12, 31),
        observed_at=date(year + 1, 2, 1),
        source_id=source,
    )


def _policy(*keys: str):
    base = PersonaPolicyStore.load().get(Persona.PE)
    return base.model_copy(update={"derived_metrics": list(keys)})


def test_growth_and_delta_use_the_two_latest_periods_and_carry_lineage() -> None:
    evidence = EvidenceBundle(
        observations=[
            _obs("c1", "revenue", 100, 2022, source="src_old"),
            _obs("c1", "revenue", 110, 2023, source="src_a"),
            _obs("c1", "revenue", 121, 2024, source="src_b"),
            _obs("c1", "operating_margin", 0.10, 2023, unit="ratio"),
            _obs("c1", "operating_margin", 0.15, 2024, unit="ratio"),
        ]
    )

    results = {
        item.key: item
        for item in derive(
            evidence, _policy("revenue_growth_yoy", "operating_margin_delta_yoy")
        )
    }

    growth = results["revenue_growth_yoy"]
    assert round(growth.value, 4) == 0.1
    assert growth.period_end == date(2024, 12, 31)
    assert growth.input_observation_ids == ["c1:revenue:2024", "c1:revenue:2023"]
    assert growth.input_source_ids == ["src_a", "src_b"]
    assert round(results["operating_margin_delta_yoy"].value, 4) == 0.05
    assert results["operating_margin_delta_yoy"].unit == "ratio_points"


def test_derivations_are_skipped_when_inputs_are_missing_or_zero() -> None:
    evidence = EvidenceBundle(
        observations=[
            _obs("c1", "revenue", 100, 2024),  # single period: no growth
            _obs("c1", "free_cash_flow", 20, 2024),
            _obs("c2", "revenue", 0, 2024),  # zero denominator
            _obs("c2", "free_cash_flow", 5, 2024),
            _obs("c3", "total_debt", 50, 2024),  # no cash row: no net debt
        ]
    )

    results = derive(evidence, _policy("revenue_growth_yoy", "fcf_margin", "net_debt"))

    assert [(item.entity_id, item.key) for item in results] == [("c1", "fcf_margin")]
    assert results[0].value == 0.2


def test_ratio_inputs_must_share_a_period() -> None:
    evidence = EvidenceBundle(
        observations=[
            _obs("c1", "capital_expenditure", -30, 2023),
            _obs("c1", "revenue", 100, 2024),
        ]
    )
    assert derive(evidence, _policy("capex_intensity")) == []

    evidence.observations.append(_obs("c1", "capital_expenditure", -30, 2024))
    [capex] = derive(evidence, _policy("capex_intensity"))
    assert capex.value == 0.3  # sign-normalised


def test_leverage_metrics_for_pe() -> None:
    evidence = EvidenceBundle(
        observations=[
            _obs("c1", "total_debt", 120, 2024),
            _obs("c1", "cash_and_equivalents", 20, 2024),
            _obs("c1", "free_cash_flow", 25, 2024),
            _obs("c2", "total_debt", 10, 2024),
            _obs("c2", "cash_and_equivalents", 40, 2024),
            _obs("c2", "free_cash_flow", -5, 2024),  # negative FCF: no coverage ratio
        ]
    )

    results = {
        (item.entity_id, item.key): item
        for item in derive(evidence, _policy("net_debt", "net_debt_to_free_cash_flow"))
    }

    assert results[("c1", "net_debt")].value == 100
    assert results[("c1", "net_debt_to_free_cash_flow")].value == 4
    assert results[("c1", "net_debt_to_free_cash_flow")].unit == "years"
    assert results[("c2", "net_debt")].value == -30
    assert ("c2", "net_debt_to_free_cash_flow") not in results


def test_benchmark_relative_metrics_need_a_benchmark_row() -> None:
    company_only = EvidenceBundle(
        observations=[_obs("c1", "operating_margin", 0.30, 2024, unit="ratio")]
    )
    assert derive(company_only, _policy("operating_margin_vs_sector_median")) == []

    with_benchmark = EvidenceBundle(
        observations=[
            *company_only.observations,
            _obs(
                "benchmark:tech",
                "operating_margin",
                0.25,
                2024,
                unit="ratio",
                source="src_med",
            ),
            _obs("c1", "revenue", 300, 2024),
            _obs("benchmark:tech", "revenue", 200, 2024, source="src_med"),
        ]
    )
    results = {
        item.key: item
        for item in derive(
            with_benchmark,
            _policy("operating_margin_vs_sector_median", "revenue_vs_sector_median"),
        )
    }
    assert round(results["operating_margin_vs_sector_median"].value, 4) == 0.05
    assert results["operating_margin_vs_sector_median"].unit == "ratio_points"
    assert round(results["revenue_vs_sector_median"].value, 4) == 0.5
    assert "src_med" in results["revenue_vs_sector_median"].input_source_ids
    assert all(not item.entity_id.startswith("benchmark") for item in results.values())


def test_every_persona_selects_only_known_derivations() -> None:
    for policy in PersonaPolicyStore.load().all():
        assert policy.derived_metrics, policy.persona
        assert set(policy.derived_metrics) <= set(DERIVATIONS), policy.persona
