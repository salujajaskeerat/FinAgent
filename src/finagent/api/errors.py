"""HTTP error mapping for expected application failures."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from finagent.contracts.api import ProblemDetail
from finagent.core.errors import (
    AnalysisTimeoutError,
    DependencyUnavailableError,
    RateLimitError,
)


def _request_id(request: Request) -> UUID:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, UUID) else uuid4()


def _problem(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    errors: list[dict[str, object]] | None = None,
) -> JSONResponse:
    problem = ProblemDetail(
        type=f"https://finagent.local/problems/{code.lower().replace('_', '-')}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=_request_id(request),
        errors=errors or [],
    )
    return JSONResponse(
        status_code=status,
        content=problem.model_dump(mode="json"),
        media_type="application/problem+json",
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Normalize FastAPI request validation errors."""
    errors = [
        {
            "location": list(item["loc"]),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in exc.errors()
    ]
    return _problem(
        request,
        status=422,
        code="INVALID_REQUEST",
        title="Invalid request",
        detail="The request did not match the published API contract.",
        errors=errors,
    )


async def dependency_exception_handler(
    request: Request,
    exc: DependencyUnavailableError,
) -> JSONResponse:
    """Map unavailable dependencies to HTTP 503."""
    return _problem(
        request,
        status=503,
        code=exc.code,
        title="Analysis dependency unavailable",
        detail=str(exc),
    )


async def timeout_exception_handler(
    request: Request,
    exc: AnalysisTimeoutError,
) -> JSONResponse:
    """Map the global analysis deadline to HTTP 504."""
    return _problem(
        request,
        status=504,
        code=exc.code,
        title="Analysis timed out",
        detail=str(exc),
    )


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitError,
) -> JSONResponse:
    """Map exhausted upstream throttling retries to HTTP 429."""
    return _problem(
        request,
        status=429,
        code=exc.code,
        title="Analysis rate limited",
        detail=str(exc),
    )
