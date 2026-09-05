# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FinAgent: one persona-configurable financial-research agent. A single bounded workflow
serves 3 personas x 3 sectors (9 combinations); there are deliberately **no** persona- or
sector-specific code paths. `docs/DESIGN.md` holds the rationale, state machine, and
sequence diagrams; `docs/IMPLEMENTATION_PLAN.md` holds the milestone plan.

## Commands

```bash
uv sync --extra test                     # httpx + pytest come from the `test` extra

uv run pytest                            # full suite; integration is NOT excluded by default
uv run pytest -m integration             # real MCP server subprocess on an ephemeral port
uv run pytest tests/backend/test_analysis_service.py::test_name   # single test
uv run pytest -k headcount               # by keyword
RUN_LIVE_GEMINI_TEST=1 uv run --env-file .env pytest -q -m live   # opt-in real Gemini call

uvx ruff check src tests ui              # ruff is configured in pyproject but not a dependency
python -m compileall -q src tests ui
```

Three processes, each in its own terminal (build the DB first):

```bash
uv run finagent-mcp                      # http://127.0.0.1:8001/mcp
uv run finagent-api                      # http://127.0.0.1:8000
uv run streamlit run ui/app.py           # http://localhost:8501
```

Data build. `data/finagent.db` (real SEC build, ~160 KB) is **committed**, so a fresh
clone runs without network access. Rebuild only when the manifest or builder changes:

```bash
export SEC_USER_AGENT="finagent-assignment/0.1 you@example.com"        # SEC fair access
uv run python -m finagent.ingestion refresh-public                     # networked; writes data/finagent.db
uv run python -m finagent.ingestion build --raw-dir data/raw/sec       # offline from cache
uv run python -m finagent.ingestion audit --require-real-enrichment
uv run python -m finagent.ingestion sample                             # illustrative data/finagent_sample.db, tests only
```

`download` / `download-filings` are the only networked steps; `build`, `sample`, and
`audit` are offline and idempotent. `refresh-public` chains download + build + audit.

## Architecture

Ports-and-adapters. Dependencies point inward to `core`, which knows only the Protocols
in [ports.py](src/finagent/core/ports.py) (`DataGateway`, `LlmGateway`, `EntityResolver`).

```
ui/ (Streamlit, HTTP only) -> api/ (FastAPI) -> core/AnalysisService
                                                  |-> gateways/mcp_client -> [separate process] mcp_server -> SQLite (read-only)
                                                  |-> gateways/llm, gateways/entity_resolver -> Gemini
ingestion/ (offline CLI) --------------------------------------------------> SQLite (write)
```

- `contracts/` — public API + MCP Pydantic models. All extend `StrictModel` (extra fields forbidden).
- `core/` — `analysis_service.py` (the whole workflow), `state.py`, `grounding.py`, `persona_policy.py`.
- `mcp_server/repository.py` — the **only** runtime file allowed to touch SQLite.
- `config/personas.yaml` — declarative persona policy (required metrics, event kinds, sections).

### Enforced boundaries

[test_boundaries.py](tests/backend/test_boundaries.py) fails the build if:

1. `api/`, `core/`, or `gateways/` import `sqlite3`;
2. the literal `"SELECT "` appears in any runtime file other than `mcp_server/repository.py`
   (ingestion is exempt) — this catches SQL in comments and docstrings too.

So runtime data access goes through the four MCP tools only (`get_catalog`,
`resolve_companies`, `query_observations`, `query_events`). By the same design rule --
not enforced by a test -- `ui/` imports nothing from `finagent` and speaks HTTP through
`ui/api_client.py`.

### Request flow and its invariants

`AnalysisService.analyze` runs under one `asyncio.timeout` deadline and walks
`AnalysisState` transitions validated by `_ALLOWED` in [state.py](src/finagent/core/state.py) —
adding a workflow step means editing that table, or `StateTrace.move` raises.

1. **resolving_scope** — `get_catalog` + `resolve_companies` in parallel. An explicit but
   unresolvable company mention gets one constrained `EntityResolver` retry (must return a
   catalog id with confidence >= 0.85) and otherwise short-circuits to `out_of_scope`
   **before any LLM planning or synthesis call**. Tests assert the LLM is never invoked here.
2. **planning** — Gemini proposes a `RetrievalPlan`, then `_constrain_plan` intersects it
   with the catalog's entity ids / metric keys / event kinds. The model's output is a
   suggestion, never a query.
3. **retrieving** — deterministic `query_observations` / `query_events` MCP calls.
4. **synthesizing / validating / repairing** — `grounding_issues` checks every finding's
   `source_ids` and `company_ids` against what MCP actually returned. Exactly one repair
   attempt (no new evidence); still-invalid drafts become `insufficient_data`.

The LLM never receives MCP, Search, URL, or SQL tools. System instructions treat payload
strings as untrusted data.

### Status conventions

Analytical outcomes (`answered`, `out_of_scope`, `insufficient_data`) are **HTTP 200** with
a separate `evidence_status` (`sufficient` / `partial` / `none`). Failures use RFC-7807
Problem Details via [api/errors.py](src/finagent/api/errors.py): 422 invalid request,
429 upstream rate limit, 503 dependency unavailable, 504 deadline exceeded.

## Working in this codebase

- **Never fabricate data.** Missing market cap, enterprise value, guidance, or restructuring
  rows stay missing; the analysis degrades to partial/insufficient instead. `audit` and the
  ingestion tests exist to keep this honest.
- **New metric or event kind** requires three coordinated edits: emit it in
  `ingestion/builder.py`, add it to the allowlist tuples (`_ANNUAL_METRICS`,
  `_MARKET_METRICS`) and mapping in `mcp_server/repository.py`, and reference it from
  `config/personas.yaml`. The catalog is the allowlist the plan is constrained against.
- **New persona** = a `Persona` enum value plus a `personas.yaml` block. `PersonaPolicyStore.load`
  raises if any enum member lacks a policy. Do not branch on persona in `AnalysisService`.
- The LLM layer is provider-neutral: `gateways/llm.py` owns prompts + validation and never
  imports a vendor SDK; `gateways/providers/` holds one adapter per vendor behind the
  `StructuredCompletionProvider` seam (`gemini`, `openai_compatible`, `anthropic`). Add a
  provider by adding one adapter file and a branch in `providers/__init__.py`; never put
  vendor code in the gateway or resolver. Config is `LLM_PROVIDER` / `LLM_MODEL` /
  `LLM_API_KEY` / `LLM_BASE_URL`. Default `LLM_PROVIDER=fake` keeps the suite offline.
  `.env` is loaded with `override=False`, so exported env vars win.
- Docstrings are NumPy-style with Parameters/Returns/Raises sections; models are strict
  Pydantic; async throughout `core`, `gateways`, and `api`.
- `data/finagent.db` is the real SEC build and the only database that should be shipped.
  The `sample` command's illustrative output (`finagent_sample.db`) is for offline plumbing
  checks and tests; never point a demo at it.
- Educational software, not investment advice; the demo must only ever see public data.
