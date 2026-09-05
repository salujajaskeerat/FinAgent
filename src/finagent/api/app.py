"""FastAPI application factory and process entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import UUID, uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from finagent.api.errors import (
    dependency_exception_handler,
    rate_limit_exception_handler,
    timeout_exception_handler,
    validation_exception_handler,
)
from finagent.contracts.api import (
    AnalysisRequest,
    AnalysisResponse,
    CatalogResponse,
    Sector,
)
from finagent.core.analysis_service import AnalysisService
from finagent.core.errors import (
    AnalysisTimeoutError,
    DependencyUnavailableError,
    FinagentError,
    RateLimitError,
)
from finagent.core.persona_policy import PersonaPolicyStore
from finagent.core.state import AnalysisState
from finagent.gateways.entity_resolver import build_entity_resolver
from finagent.gateways.llm import build_llm_gateway
from finagent.gateways.mcp_client import McpDataGateway, StreamableHttpToolCaller
from finagent.gateways.providers import LlmSettings, build_provider

logger = logging.getLogger("finagent.api")

# Local development reads an ignored .env file; deployed environment variables win.
load_dotenv(override=False)


def build_service() -> AnalysisService:
    """Build the default application service from environment configuration."""
    mcp_url = os.getenv("FINAGENT_MCP_URL", "http://127.0.0.1:8001/mcp")
    # Plan, synthesis, and one repair each get LLM_TIMEOUT_SECONDS (20s default),
    # plus a 5s entity-resolution fallback, so the global deadline must exceed 65s.
    timeout = float(os.getenv("FINAGENT_ANALYSIS_TIMEOUT_SECONDS", "75"))
    llm_settings = LlmSettings.from_env()
    # One vendor adapter is shared by the analysis gateway and entity resolver.
    provider = build_provider(llm_settings)
    return AnalysisService(
        data_gateway=McpDataGateway(StreamableHttpToolCaller(mcp_url)),
        llm_gateway=build_llm_gateway(llm_settings, provider),
        policies=PersonaPolicyStore.load(),
        deadline_seconds=timeout,
        entity_resolver=build_entity_resolver(llm_settings, provider),
    )


def create_app(service: AnalysisService | None = None) -> FastAPI:
    """Create an API application with injectable dependencies.

    Parameters
    ----------
    service
        Optional service override used by contract and integration tests.

    Returns
    -------
    FastAPI
        Configured HTTP application.
    """
    app = FastAPI(
        title="FinAgent API",
        version="0.1.0",
        description="One persona-configurable financial agent backed by MCP data tools.",
    )
    app.state.analysis_service = service or build_service()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(DependencyUnavailableError, dependency_exception_handler)
    app.add_exception_handler(AnalysisTimeoutError, timeout_exception_handler)
    app.add_exception_handler(RateLimitError, rate_limit_exception_handler)

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = uuid4()
        request.state.request_id = request_id
        correlation_id = request.headers.get("X-Correlation-ID", str(request_id))[:128]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(request_id)
        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": str(request_id),
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 2),
                }
            )
        )
        return response

    async def get_service(request: Request) -> AnalysisService:
        """Return the application-scoped service without a worker-thread hop."""
        return request.app.state.analysis_service

    @app.post("/v1/analyses", response_model=AnalysisResponse)
    async def create_analysis(
        payload: AnalysisRequest,
        request: Request,
        analysis_service: AnalysisService = Depends(get_service),
    ) -> AnalysisResponse:
        """Run a non-streaming, evidence-validated analysis."""
        request_id: UUID = request.state.request_id
        return await analysis_service.analyze(payload, request_id=request_id)

    @app.post("/v1/analyses/stream")
    async def stream_analysis(
        payload: AnalysisRequest,
        request: Request,
        analysis_service: AnalysisService = Depends(get_service),
    ) -> StreamingResponse:
        """Run the same analysis, streaming one `step` event per state transition.

        The stream ends with a `result` event carrying the identical
        AnalysisResponse that POST /v1/analyses returns, or an `error` event
        with a Problem Details body.
        """
        request_id: UUID = request.state.request_id
        queue: asyncio.Queue[tuple[str, dict[str, object]] | None] = asyncio.Queue()
        started = time.perf_counter()

        async def on_progress(state: AnalysisState, message: str) -> None:
            await queue.put(
                (
                    "step",
                    {
                        "state": state.value,
                        "message": message,
                        "elapsed_ms": round((time.perf_counter() - started) * 1_000),
                    },
                )
            )

        async def run() -> None:
            try:
                result = await analysis_service.analyze(
                    payload, request_id=request_id, on_progress=on_progress
                )
                await queue.put(("result", result.model_dump(mode="json")))
            except FinagentError as exc:
                await queue.put(("error", _stream_problem(exc, request_id)))
            except Exception:  # keep the stream well-formed on unexpected failures
                logger.exception("streaming analysis failed")
                await queue.put(
                    (
                        "error",
                        _stream_problem(FinagentError("internal error"), request_id),
                    )
                )
            finally:
                await queue.put(None)

        async def events() -> AsyncIterator[str]:
            task = asyncio.create_task(run())
            try:
                while (item := await queue.get()) is not None:
                    name, data = item
                    yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
            finally:
                task.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/catalog", response_model=CatalogResponse)
    async def get_catalog(
        sector: Sector,
        analysis_service: AnalysisService = Depends(get_service),
    ) -> CatalogResponse:
        """Return UI selectors and the selected sector's data coverage."""
        return await analysis_service.catalog(sector)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        """Report process liveness without touching dependencies."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(
        analysis_service: AnalysisService = Depends(get_service),
    ) -> Response:
        """Report configuration and MCP/DB readiness without a paid model call."""
        try:
            await analysis_service.catalog(Sector.TECH)
        except (DependencyUnavailableError, OSError, RuntimeError, ValueError):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    return app


_STREAM_STATUS = {
    "DEPENDENCY_UNAVAILABLE": 503,
    "ANALYSIS_TIMEOUT": 504,
    "RATE_LIMITED": 429,
}


def _stream_problem(exc: FinagentError, request_id: UUID) -> dict[str, object]:
    """Problem Details payload for an `error` stream event."""
    status = _STREAM_STATUS.get(exc.code, 500)
    return {
        "type": f"https://finagent.local/problems/{exc.code.lower().replace('_', '-')}",
        "title": exc.__class__.__doc__ or exc.code,
        "status": status,
        "detail": str(exc),
        "code": exc.code,
        "request_id": str(request_id),
    }


app = create_app()


def run() -> None:
    """Run the API development server."""
    uvicorn.run(
        "finagent.api.app:app",
        host=os.getenv("FINAGENT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("FINAGENT_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
