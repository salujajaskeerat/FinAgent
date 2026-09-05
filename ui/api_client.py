"""Typed standard-library HTTP client for the FinAgent API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ApiClientError(RuntimeError):
    """Base error suitable for presentation to a UI user."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "api_error",
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class ApiUnavailableError(ApiClientError):
    """Raised when the API cannot be reached."""


@dataclass(frozen=True)
class PersonaOption:
    """Persona metadata returned by the catalog endpoint."""

    value: str
    label: str
    description: str


@dataclass(frozen=True)
class SectorOption:
    """Sector metadata returned by the catalog endpoint."""

    value: str
    label: str


@dataclass(frozen=True)
class Company:
    """Canonical company reference."""

    company_id: str
    name: str
    ticker: str | None = None


@dataclass(frozen=True)
class Source:
    """Public provenance record supporting an analysis."""

    source_id: str
    title: str
    url: str
    publisher: str
    published_at: str | None = None
    retrieved_at: str | None = None


@dataclass(frozen=True)
class Finding:
    """Material claim with canonical evidence references."""

    text: str
    company_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """UI-safe dataset and selector metadata."""

    dataset_version: str
    personas: tuple[PersonaOption, ...]
    sectors: tuple[SectorOption, ...]
    companies: tuple[Company, ...]
    metric_keys: tuple[str, ...]
    coverage_start: str | None = None
    coverage_end: str | None = None


@dataclass(frozen=True)
class DerivedMetric:
    """Value computed by the application from retrieved observations."""

    key: str
    entity_id: str
    value: float
    unit: str
    period_end: str
    formula: str
    caveat: str | None = None


@dataclass(frozen=True)
class Coverage:
    """Persona-required inputs the retrieved evidence did or did not contain."""

    required_metrics: tuple[str, ...]
    available_metrics: tuple[str, ...]
    missing_metrics: tuple[str, ...]


@dataclass(frozen=True)
class Trace:
    """How the answer was produced."""

    states: tuple[str, ...]
    llm_calls: int
    repaired: bool
    proposed_metric_keys: tuple[str, ...]
    constrained_metric_keys: tuple[str, ...]


@dataclass(frozen=True)
class Analysis:
    """Evidence-aware analysis response consumed by the UI."""

    request_id: str
    status: str
    persona: str
    sector: str
    answer_markdown: str
    evidence_status: str
    findings: tuple[Finding, ...]
    companies: tuple[Company, ...]
    sources: tuple[Source, ...]
    limitations: tuple[str, ...]
    data_as_of: str | None = None
    derived_metrics: tuple[DerivedMetric, ...] = ()
    coverage: Coverage | None = None
    trace: Trace | None = None


class FinAgentApiClient:
    """Call the FinAgent API without importing application internals.

    Parameters
    ----------
    base_url
        Root URL of the running FastAPI service.
    timeout_seconds
        Socket timeout applied to each request.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 50.0) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def get_catalog(self, sector: str) -> Catalog:
        """Fetch selector metadata and coverage for a sector.

        Parameters
        ----------
        sector
            API sector identifier.

        Returns
        -------
        Catalog
            Parsed catalog response.
        """
        query = urlencode({"sector": sector})
        payload = self._request("GET", f"/v1/catalog?{query}")
        return _parse_catalog(payload)

    def analyze(self, *, query: str, persona: str, sector: str) -> Analysis:
        """Submit one non-streaming analysis request.

        Parameters
        ----------
        query
            User's analytical question.
        persona
            Selected API persona identifier.
        sector
            Selected API sector identifier.

        Returns
        -------
        Analysis
            Parsed and grounded analysis response.
        """
        payload = self._request(
            "POST",
            "/v1/analyses",
            body={"query": query, "persona": persona, "sector": sector},
        )
        return _parse_analysis(payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, object] | None = None,
    ) -> Mapping[str, Any]:
        encoded = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._base_url}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return _decode_object(response.read())
        except HTTPError as error:
            self._raise_http_error(error)
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ApiUnavailableError(
                f"The analysis service is unavailable: {reason}",
                code="service_unavailable",
                retryable=True,
            ) from error
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_http_error(error: HTTPError) -> None:
        request_id = error.headers.get("X-Request-ID")
        try:
            payload = _decode_object(error.read())
        except ApiClientError:
            payload = {}
        detail = payload.get("detail")
        message = (
            detail if isinstance(detail, str) else f"API request failed ({error.code})."
        )
        code = payload.get("code")
        if not isinstance(code, str):
            code = f"http_{error.code}"
        body_request_id = payload.get("request_id")
        if isinstance(body_request_id, str):
            request_id = body_request_id
        raise ApiClientError(
            message,
            code=code,
            request_id=request_id,
            retryable=error.code in {429, 502, 503, 504},
        ) from error


def _decode_object(raw: bytes) -> Mapping[str, Any]:
    """Decode a JSON object or raise a presentation-safe error."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiClientError("The service returned an unreadable response.") from error
    if not isinstance(value, dict):
        raise ApiClientError("The service returned an unexpected response shape.")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ApiClientError(f"The service response is missing '{key}'.")
    return value


def _optional_date(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiClientError(f"The service response has an invalid '{key}'.")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ApiClientError(f"The service response has an invalid '{key}'.") from error
    return value


def _object_list(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ApiClientError(f"The service response has an invalid '{key}'.")
    return value


def _string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ApiClientError(f"The service response has an invalid '{key}'.")
    return tuple(value)


def _parse_company(payload: Mapping[str, Any]) -> Company:
    ticker = payload.get("ticker")
    if ticker is not None and not isinstance(ticker, str):
        raise ApiClientError("The service response has an invalid company ticker.")
    return Company(
        company_id=_required_string(payload, "company_id"),
        name=_required_string(payload, "name"),
        ticker=ticker,
    )


def _parse_source(payload: Mapping[str, Any]) -> Source:
    return Source(
        source_id=_required_string(payload, "source_id"),
        title=_required_string(payload, "title"),
        url=_required_string(payload, "url"),
        publisher=_required_string(payload, "publisher"),
        published_at=_optional_date(payload, "published_at"),
        retrieved_at=_optional_date(payload, "retrieved_at"),
    )


def _parse_catalog(payload: Mapping[str, Any]) -> Catalog:
    personas = tuple(
        PersonaOption(
            value=_required_string(item, "value"),
            label=_required_string(item, "label"),
            description=_required_string(item, "description"),
        )
        for item in _object_list(payload, "personas")
    )
    sectors = tuple(
        SectorOption(
            value=_required_string(item, "value"),
            label=_required_string(item, "label"),
        )
        for item in _object_list(payload, "sectors")
    )
    if not personas or not sectors:
        raise ApiClientError("The catalog contains no available personas or sectors.")
    return Catalog(
        dataset_version=_required_string(payload, "dataset_version"),
        personas=personas,
        sectors=sectors,
        companies=tuple(
            _parse_company(item) for item in _object_list(payload, "companies")
        ),
        metric_keys=_string_tuple(payload, "metric_keys"),
        coverage_start=_optional_date(payload, "coverage_start"),
        coverage_end=_optional_date(payload, "coverage_end"),
    )


def _parse_analysis(payload: Mapping[str, Any]) -> Analysis:
    return Analysis(
        request_id=_required_string(payload, "request_id"),
        status=_required_string(payload, "status"),
        persona=_required_string(payload, "persona"),
        sector=_required_string(payload, "sector"),
        answer_markdown=_required_string(payload, "answer_markdown"),
        evidence_status=_required_string(payload, "evidence_status"),
        findings=tuple(
            Finding(
                text=_required_string(item, "text"),
                company_ids=_string_tuple(item, "company_ids"),
                source_ids=_string_tuple(item, "source_ids"),
            )
            for item in _object_list(payload, "findings")
        ),
        companies=tuple(
            _parse_company(item) for item in _object_list(payload, "companies")
        ),
        sources=tuple(_parse_source(item) for item in _object_list(payload, "sources")),
        limitations=_string_tuple(payload, "limitations"),
        data_as_of=_optional_date(payload, "data_as_of"),
        derived_metrics=tuple(
            DerivedMetric(
                key=_required_string(item, "key"),
                entity_id=_required_string(item, "entity_id"),
                value=float(item.get("value", 0.0)),
                unit=_required_string(item, "unit"),
                period_end=_required_string(item, "period_end"),
                formula=_required_string(item, "formula"),
                caveat=item.get("caveat") or None,
            )
            for item in _object_list(payload, "derived_metrics")
        ),
        coverage=_parse_coverage(payload.get("coverage")),
        trace=_parse_trace(payload.get("trace")),
    )


def _parse_coverage(value: Any) -> Coverage | None:
    if not isinstance(value, Mapping):
        return None
    return Coverage(
        required_metrics=_string_tuple(value, "required_metrics"),
        available_metrics=_string_tuple(value, "available_metrics"),
        missing_metrics=_string_tuple(value, "missing_metrics"),
    )


def _parse_trace(value: Any) -> Trace | None:
    if not isinstance(value, Mapping):
        return None
    proposed = value.get("proposed_plan") or {}
    constrained = value.get("constrained_plan") or {}
    return Trace(
        states=_string_tuple(value, "states"),
        llm_calls=int(value.get("llm_calls", 0)),
        repaired=bool(value.get("repaired", False)),
        proposed_metric_keys=_string_tuple(proposed, "metric_keys")
        if isinstance(proposed, Mapping)
        else (),
        constrained_metric_keys=_string_tuple(constrained, "metric_keys")
        if isinstance(constrained, Mapping)
        else (),
    )
