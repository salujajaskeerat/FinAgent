# Financial Research Agent - Design

## Design objective

Build the smallest auditable vertical slice that proves five properties:

1. One agent workflow serves both human and API clients.
2. Runtime data access crosses a real MCP boundary.
3. Persona and sector are independent configuration dimensions.
4. Persona changes evidence requirements and analysis structure, not only tone.
5. Unsupported or weakly supported questions fail honestly.

The design intentionally favours provenance, deterministic control, and tests
over data breadth. Technology, Retail, and Logistics each contain a small,
curated company universe with roughly a few dozen sourced domain records.

## Architectural decisions

### One configurable workflow

Mutual Fund, Equity, and Private Equity are declarative policies consumed by
one `AnalysisService`. There are no persona-specific services and no
persona-sector configuration matrix.

### FastAPI is the canonical interface

Streamlit is a thin HTTP client. External clients and the UI therefore execute
the same validation, orchestration, retrieval, and grounding path.

### MCP is an enforced data boundary

The application service accesses data only through a typed MCP client. Only the
MCP repository and offline ingestion packages may import SQLite. Architecture
tests enforce this rule.

### Offline, reproducible ingestion

External sources are downloaded and normalized outside the request path. The
runtime reads a locally built SQLite database, so a demonstration never depends
on live scraping.

### Bounded orchestration

The workflow has explicit states, no model-generated SQL, no autonomous web
search, and at most one constrained output repair. Derived financial ratios are
calculated by deterministic code before LLM synthesis.

### Gemini is a constrained reasoning adapter

The production adapter uses Gemini for two typed operations: proposing a small
retrieval plan and synthesizing a source-linked draft. Gemini receives no MCP,
Google Search, URL, SQL, or function tools. `AnalysisService` intersects every
proposed entity, metric, and event with the MCP catalog before performing the
queries itself. After synthesis, deterministic validation rejects company or
source identifiers that were not present in retrieved evidence. The fake
adapter remains available only as an explicit offline/test mode.

### Entity resolution is deterministic first

The MCP repository first matches normalized legal names, tickers, and configured
aliases. If that fails, its narrow phrasing detector distinguishes an explicit
company mention from a broad sector question. Only an explicit unresolved mention
may invoke the typed LLM resolver, which receives the selected sector's company
catalog and no data tools. `AnalysisService` accepts exactly one catalog ID at
confidence `>= 0.85`; ambiguous, low-confidence, invented, malformed, timed-out,
or unavailable results exit as `out_of_scope` before planning and synthesis.

```mermaid
flowchart LR
    Query["User query"] --> Exact{"Deterministic name,<br/>ticker, or alias match?"}
    Exact -->|Yes| Canonical["Canonical catalog ID"]
    Exact -->|No| Explicit{"Explicit company<br/>reference?"}
    Explicit -->|No: broad question| Broad["Selected-sector companies"]
    Explicit -->|Yes| Resolver["One constrained LLM<br/>resolution attempt"]
    Catalog["Selected-sector catalog IDs"] --> Resolver
    Resolver --> Validate{"Exactly one catalog ID<br/>and confidence >= 0.85?"}
    Validate -->|Yes| Canonical
    Validate -->|No| Out["out_of_scope<br/>no planning or synthesis"]
```

Fuzzy-search dependencies, embeddings, and unrestricted model-driven entity
discovery are not part of this design.

### Evidence coverage over subjective confidence

Responses report evidence status, dates, sources, unsupported-claim counts, and
limitations. They do not expose an uncalibrated LLM confidence score or hidden
chain-of-thought.

## Overall architecture

```mermaid
flowchart TB
    subgraph BuildTime["Offline data build"]
        Sources["SEC filings<br/>IR releases<br/>dated market source"]
        Manifest["Curated source manifest"]
        Builder["Normalize, validate<br/>and build database"]
        DB[("Reproducible SQLite database")]

        Sources --> Manifest
        Manifest --> Builder
        Builder --> DB
    end

    subgraph Runtime["Runtime"]
        Human["Human user"] --> UI["Thin Streamlit UI"]
        Consumer["External API consumer"] --> API["FastAPI"]
        UI -->|"HTTP only"| API
        API --> Agent["AnalysisService<br/>single bounded workflow"]
        Agent --> Policy["Persona policy registry"]
        Agent --> Calculator["Deterministic calculations"]
        Agent --> LLM["Gemini structured-output adapter<br/>or explicit offline fake"]
        Agent --> MCPClient["MCP client"]
        MCPClient -->|"MCP Streamable HTTP"| MCPServer["MCP data server"]
        MCPServer --> Repository["Read-only repository"]
        Repository --> DB
    end
```

The standalone Mermaid source is in
[`diagrams/architecture.mmd`](diagrams/architecture.mmd).

## Agent state machine

```mermaid
stateDiagram-v2
    [*] --> ValidateRequest
    ValidateRequest --> InvalidRequest: Invalid persona or sector
    ValidateRequest --> LoadCatalog: Valid request
    LoadCatalog --> DependencyFailure: MCP unavailable
    LoadCatalog --> ParseQuestion: Catalog loaded
    ParseQuestion --> ResolveEntities
    ResolveEntities --> OutOfScope: Unsupported company
    ResolveEntities --> BuildEvidencePlan: Scope supported
    BuildEvidencePlan --> QueryMCP
    QueryMCP --> DependencyFailure: Query failed
    QueryMCP --> EvidenceGate: Evidence returned
    EvidenceGate --> InsufficientData: Mandatory facts absent
    EvidenceGate --> CalculateFeatures: Evidence usable
    CalculateFeatures --> ApplyPersona
    ApplyPersona --> ValidateOutput
    ValidateOutput --> Complete: References valid
    ValidateOutput --> RepairOnce: Invalid evidence reference
    RepairOnce --> Complete: Repaired
    RepairOnce --> SafePartial: Still invalid
    InvalidRequest --> [*]
    OutOfScope --> [*]
    InsufficientData --> [*]
    DependencyFailure --> [*]
    SafePartial --> [*]
    Complete --> [*]
```

The standalone Mermaid source is in
[`diagrams/agent-state-machine.mmd`](diagrams/agent-state-machine.mmd).

## API and MCP sequence

```mermaid
sequenceDiagram
    actor User
    participant Client as "Streamlit or API client"
    participant API as "FastAPI"
    participant Agent as "AnalysisService"
    participant MCP as "MCP data service"
    participant DB as "SQLite"
    participant LLM as "LLM adapter"
    participant Validator as "Grounding validator"

    User->>Client: Submit query, persona, sector
    Client->>API: POST /v1/analyses
    API->>Agent: Analyze typed request
    Agent->>MCP: Resolve scope and load catalog
    MCP-->>Agent: Canonical entities and allowlists
    Agent->>LLM: Question, persona, catalog allowlists
    LLM-->>Agent: Typed retrieval plan
    Agent->>Agent: Constrain plan to catalog values
    Agent->>MCP: Query allowlisted evidence
    MCP->>DB: Parameterized read
    DB-->>MCP: Facts, events, sources
    MCP-->>Agent: Typed evidence
    Agent->>Agent: Calculate features and coverage
    Agent->>LLM: Evidence and persona policy
    LLM-->>Agent: Typed findings
    Agent->>Validator: Validate entities and evidence IDs
    Agent-->>API: Validated response
    API-->>Client: JSON and X-Request-ID
```

The standalone Mermaid source is in
[`diagrams/api-sequence.mmd`](diagrams/api-sequence.mmd).

## Runtime contracts

The public API exposes:

- `POST /v1/analyses`
- `GET /v1/catalog`
- `GET /health/live`
- `GET /health/ready`

The MCP server exposes four typed tools:

- `get_catalog`
- `resolve_companies`
- `query_observations`
- `query_events`

Every observation or event carries its source metadata in the same tool result.
The service returns `answered`, `out_of_scope`, or `insufficient_data` as domain
outcomes and separately reports evidence as `sufficient`, `partial`, or `none`.

## Data model

The purpose-built schema contains:

- sectors;
- companies;
- annual financial snapshots;
- market snapshots;
- operating signals;
- sector benchmarks; and
- sources plus derived-source lineage.

Public-data refresh is an explicit ingestion operation. SEC submissions,
Companyfacts, and accession-specific annual filings are cached with hashes and
retrieval metadata. Database construction and auditing then run offline. Sector
benchmarks are medians of at least three current SEC-backed company records;
their derived source retains links to each constituent source.

Each factual row stores the reporting/effective date, publication date, source,
unit or currency where applicable, and a quality caveat. The sample database is
generated from a versioned source manifest and cached raw inputs.

## Explicit non-goals for the first milestone

- Authentication or user accounts.
- Persistent chat memory.
- Streaming model tokens.
- Runtime scraping or autonomous web browsing.
- Vector databases or document RAG.
- Arbitrary SQL tools.
- Real-time market feeds.
- Full LBO underwriting or calibrated investment advice.
- Separate persona agents or sector-specific workflows.

## Proof strategy

Tests must demonstrate the architecture rather than snapshot fluent prose:

- all nine persona-sector combinations are accepted;
- persona outputs apply different required analysis dimensions;
- unknown companies exit before LLM synthesis;
- the latest headcount signal is selected deterministically;
- every finding references evidence returned by MCP;
- only the MCP repository and ingestion code import SQLite;
- Streamlit calls the API rather than importing the agent; and
- the real MCP client/server transport works against a temporary database.
