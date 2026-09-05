# Implementation Plan

## Objective

Build a small, auditable financial-analysis system that supports three analyst
personas across three sectors. The implementation must prove four architectural
claims:

1. Streamlit and external consumers use the same FastAPI contract.
2. Runtime data access occurs only through a real MCP client/server boundary.
3. Persona selection changes the analysis policy, not the underlying evidence.
4. Every material claim is traceable to a dated database record and source.

The priority is a reliable vertical slice and demonstrable system boundaries,
not broad market coverage. Target roughly 35–50 well-sourced records per sector.

## Current Status

All milestones through M5 are implemented on the real SEC dataset:

- M0–M2: typed contracts, reproducible ingestion, and the enforced MCP boundary
  (typed tool schemas, read-only annotations, schema resource, stdio option).
- M3: bounded workflow with deterministic derived metrics (`calculating`
  state), coverage-driven evidence status, persona reasoning frames in YAML, a
  provider-neutral LLM gateway, and a response `trace`.
- M4: FastAPI contract (schema 1.1) and a Streamlit UI with side-by-side
  persona comparison.
- M5: `scripts/eval_matrix.py` runs the assignment's sample queries and the
  full persona × sector sweep; README carries the schema, MCP, and
  what-next write-up.

Remaining ideas are listed under "What I would improve with more time" in the
README.

## Delivery Principles

- Keep orchestration explicit and bounded; do not introduce a graph framework
  unless the typed state machine becomes materially difficult to maintain.
- Keep the LLM out of entity resolution, database querying, calculations, and
  evidence validation.
- Define contracts before adapters and services that consume them.
- Make offline and CI execution possible with a deterministic fake LLM.
- Use NumPy-style docstrings for public Python modules, classes, and functions.
- Prefer small modules with clear ownership over speculative abstraction.
- Complete and test each milestone before expanding the dataset or UI.

## Dependency Order

```text
domain contracts and persona policies
    -> database schema and reproducible sample database
        -> read-only repository
            -> MCP server and client
                -> deterministic calculations and analysis state machine
                    -> LLM adapter and grounding validator
                        -> FastAPI
                            -> Streamlit
                                -> full evaluation matrix and documentation
```

## Milestones

### M0 — Project Foundation and Contracts

Establish the repository conventions and types shared by all later layers.

Deliverables:

- Python project configuration with locked or bounded dependencies.
- Package layout for domain, data build, MCP, analysis, API, and UI code.
- Request and response models, enums, error model, and schema version.
- Persona policy schema plus Mutual Fund, Equity, and PE configurations.
- Sector and canonical company identifiers for Technology, Retail, and
  Logistics.
- Configuration loading from environment variables with a safe `.env.example`.
- Test, lint, formatting, and type-check commands suitable for CI.

Acceptance criteria:

- All persona and sector combinations validate at the contract layer.
- Invalid persona, sector, or blank question fails deterministically.
- Persona policy files satisfy one common schema and declare mandatory analysis
  dimensions.
- Tests run without network access or an LLM API key.

Milestone boundary: no database, MCP, HTTP, or UI implementation begins until
the public request, response, evidence, company, and error models are stable.

### M1 — Reproducible Evidence Dataset

Build a deliberately small point-in-time dataset through an offline pipeline.

Deliverables:

- Explicit relational schema for sectors, companies, annual financial
  snapshots, market snapshots, operating signals, sector benchmarks, and
  sources.
- Curated manifest covering about four companies per sector.
- SEC ingestion adapter for filing metadata and selected Companyfacts metrics.
- Adapters or manifest entries for investor-relations operating signals and
  the documented dated market source.
- Normalization for dates, fiscal periods, units, currencies, aliases, and
  source identifiers.
- Idempotent database build command, cached raw fixtures, and data-quality
  report.
- A versioned sample SQLite database or deterministic rebuild path.

Acceptance criteria:

- Two consecutive builds from identical inputs produce the same logical rows.
- Every observation has a source ID, observation/reporting date, publication
  date, unit, and quality/comparability status where applicable.
- Each configured company belongs to exactly one configured sector.
- Each sector has enough financial, market, benchmark, headcount, and/or hiring
  evidence to exercise the supplied examples.
- Fixture-based ingestion tests do not require live network access.

Milestone boundary: ingestion remains a build-time concern; runtime services do
not fetch the web or call SEC directly.

### M2 — Enforced MCP Data Boundary

Expose the dataset through a separate MCP service with four typed tools:
`get_catalog`, `resolve_companies`, `query_observations`, and `query_events`.

Deliverables:

- Parameterized, read-only repository queries with canonical IDs and row caps.
- MCP server using Streamable HTTP transport.
- Typed MCP client owned by the analysis layer.
- Structured evidence responses containing dates, units, quality flags, and
  source metadata.
- Dependency guard that prevents the API, UI, and analysis packages from
  importing SQLite or repository modules.

Acceptance criteria:

- A real MCP client/server integration test succeeds over the configured
  transport.
- Cross-sector and unknown company identifiers cannot reach observation
  queries.
- Arbitrary SQL is not exposed by any tool.
- `query_events(..., latest_only=true)` deterministically returns the newest
  qualifying effective observation, using publication date and stable ID as
  tie-breakers.
- SQLite is opened read-only at runtime, and all returned evidence includes its
  source metadata.

Milestone boundary: the analysis service may consume only the MCP client; direct
repository access is treated as an architectural test failure.

### M3 — Bounded Analysis Workflow

Implement the single configurable agent as an explicit typed state machine.

Required states:

```text
validate request -> load catalog -> parse scope -> resolve entities
    -> build evidence plan -> query MCP -> evidence gate
    -> deterministic calculations -> apply persona
    -> validate output -> complete
```

Terminal alternatives are `invalid_request`, `out_of_scope`,
`insufficient_data`, `dependency_failure`, and `safe_partial`. Permit at most
one structured-output repair attempt.

Deliverables:

- Evidence planner based on the question and selected persona policy.
- Deterministic calculations for ratios, comparisons, rankings, freshness, and
  evidence coverage.
- LLM interface with production and deterministic fake adapters.
- Structured synthesis output containing company IDs and evidence IDs.
- Grounding validator for entity membership, evidence references, and required
  persona sections.
- Early exit for unsupported or cross-sector companies before LLM synthesis.

Acceptance criteria:

- The same question and sector use the same evidence fingerprint for all three
  personas while producing persona-specific analysis dimensions.
- Every returned finding references evidence present in the MCP response.
- An unknown company returns `out_of_scope` and the LLM adapter is not called.
- A supported company with no requested signal returns `insufficient_data` or
  `safe_partial`; general model knowledge is never substituted.
- PE responses always identify themselves as public-data screening rather than
  full transaction underwriting.
- Workflow retries and deadlines are bounded and covered by tests.

Milestone boundary: do not build UI behavior around unvalidated prose. Only a
validated typed analysis result may leave this layer.

### M4 — Canonical API and Thin Human Interface

Expose the validated workflow through FastAPI, then build Streamlit strictly as
an HTTP client.

Deliverables:

- `POST /v1/analyses`, `GET /v1/catalog`, `GET /health/live`, and
  `GET /health/ready`.
- `application/problem+json` errors with stable code, detail, request ID, and
  retryability.
- Request ID returned in both the response body and `X-Request-ID` header.
- Synchronous Streamlit screen with persona/sector selectors, example prompts,
  answer, evidence, sources, dates, limitations, and optional developer details.
- Explicit frontend states for ready, submitting, answered, partial,
  out-of-scope, invalid request, dependency failure, timeout, and offline.

Acceptance criteria:

- The API-specific Equity/Logistics test returns structured companies,
  findings, evidence, sources, dates, and limitations rather than a text blob.
- Domain outcomes (`answered`, `out_of_scope`, `insufficient_data`) return HTTP
  200; validation, dependency, and deadline failures return 422, 503, and 504
  respectively.
- Streamlit obtains catalog data and analyses only through FastAPI.
- API contract tests and one UI smoke test pass with the fake LLM.
- No answer-token streaming occurs before grounding validation.

Milestone boundary: UI polish must not begin until API behavior and failure
semantics are contract-tested.

### M5 — Evaluation and Submission Hardening

Prove the architectural claims and make the project reproducible for a reviewer.

Deliverables:

- Nine-combination persona/sector evaluation matrix.
- Golden scenarios for the supplied persona examples, latest headcount/hiring
  stress test, out-of-scope test, and structured API test.
- Integration test covering API -> analysis -> MCP -> sample database.
- Architecture dependency tests and data-quality checks in CI.
- Setup, sample-data rebuild, service-run, test, and evaluation instructions.
- Short ADRs for the major design decisions and documented data limitations.

Acceptance criteria:

- A clean checkout can build or load the sample database and run the complete
  test suite using documented commands.
- All nine persona/sector combinations complete with the fake LLM.
- The grounding and out-of-scope stress tests pass without network access.
- With a configured production LLM, the three persona demonstrations satisfy
  their mandatory policy dimensions.
- No secrets, raw transient downloads, caches, local databases other than the
  intentional sample artifact, or generated UI output are tracked by Git.

Milestone boundary: the initial assignment is complete when these checks pass;
additional features belong in a subsequent iteration.

## Test Strategy

Use a narrow test pyramid focused on architectural risk:

- **Unit tests:** model validation, persona policy loading, normalization,
  calculations, evidence coverage, ordering, and state transitions.
- **Contract tests:** MCP tool inputs/outputs and FastAPI response/error schemas.
- **Integration tests:** repository against a temporary SQLite fixture, real MCP
  transport, and the complete API-to-database path with a fake LLM.
- **Evaluation tests:** nine configuration combinations plus the required
  grounding, out-of-scope, and API scenarios.
- **Dependency tests:** fail CI if forbidden packages import repository/SQLite
  code or if Streamlit imports application internals instead of the API client.

Live-source smoke tests and production-LLM evaluations should be opt-in because
they are slower, nondeterministic, and may require credentials. They must not be
the only proof that the core system works.

## Suggested Commit Sequence

Keep commits independently reviewable:

1. Repository foundation, contracts, and persona policies.
2. Database schema, fixtures, and deterministic builder.
3. SEC/IR enrichment adapters and provenance checks.
4. Read-only repository and MCP server/client contract.
5. Analysis workflow, calculations, fake LLM, and grounding validator.
6. FastAPI endpoints and contract tests.
7. Streamlit client and smoke test.
8. Evaluation matrix, ADRs, and submission documentation.

## Explicit Non-Goals

- Comprehensive sector coverage, live market breadth, or real-time news search.
- More than the three required sectors or a separate agent per persona/sector.
- Arbitrary SQL, model-authored SQL, or direct runtime database access outside
  the MCP service.
- Autonomous open-ended planning loops, multi-agent runtime orchestration, or
  exposed chain-of-thought.
- Full LBO modeling, investment advice, portfolio execution, or a claim of PE
  underwriting from public data alone.
- Multi-turn memory, user accounts, permissions, billing, queues, distributed
  tracing, or production-scale deployment.
- Token streaming, elaborate dashboards, or UI work that does not demonstrate
  the architecture.
- A generic data platform, generic metric ontology, or premature plugin system.

These exclusions are intentional. They preserve time for source provenance,
scope awareness, MCP enforcement, deterministic behavior, and high-signal tests.
