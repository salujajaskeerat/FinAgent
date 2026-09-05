"""Single-page Streamlit client for grounded financial analyses.

The page is a thin HTTP client. It streams the backend's workflow states into a
step timeline so the agent's progress is visible, then renders the structured
response. Colour is reserved for evidence status; personas are identified by
icon only.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field

import streamlit as st

from ui.api_client import (
    Analysis,
    ApiClientError,
    Catalog,
    FinAgentApiClient,
    StepEvent,
)

DEFAULT_SECTOR = "tech"
PERSONA_ICONS = {
    "mutual_fund_analyst": ":material/account_balance:",
    "equity_analyst": ":material/query_stats:",
    "pe_analyst": ":material/handshake:",
}
# The assignment's own sample questions, so reviewers can run them in one click.
PRESETS = {
    ":material/trending_up: Sector outlook": (
        "Is this sector a good place to be putting money to work right now?"
    ),
    ":material/inventory: Core holding vs avoid": (
        "Which of these companies would fit a long-term core holding versus a "
        "name I should avoid?"
    ),
    ":material/percent: Margin profile": (
        "Walk me through the margin profile of the companies in your data - "
        "who's improving and who's under pressure?"
    ),
    ":material/lock: Take-private pick": (
        "If I had to pick one company here to take private, which would it be "
        "and what's the operational thesis?"
    ),
    ":material/groups: Headcount signal": (
        "What's the most recent headcount or hiring signal you have for Apple?"
    ),
    ":material/block: Out-of-scope company": "What do you think about Tesla?",
}
# Workflow states in order, with the labels shown while running and when done.
STEPS: dict[str, tuple[str, str]] = {
    "resolving_scope": ("Resolving scope", "Resolved scope"),
    "planning": ("Planning retrieval", "Planned retrieval"),
    "retrieving": ("Retrieving evidence over MCP", "Retrieved evidence"),
    "calculating": ("Computing derived metrics", "Computed derived metrics"),
    "synthesizing": ("Drafting the persona's answer", "Drafted answer"),
    "validating": ("Validating grounding", "Validated grounding"),
    "repairing": ("Repairing ungrounded findings", "Repaired findings"),
}
MAIN_STEPS = [state for state in STEPS if state != "repairing"]
EVIDENCE_BADGE = {
    "sufficient": ":green-badge[:material/verified: Evidence sufficient]",
    "partial": ":orange-badge[:material/warning: Evidence partial]",
    "none": ":red-badge[:material/help: No evidence]",
}
STATUS_BADGE = {
    "out_of_scope": ":red-badge[:material/block: Out of scope]",
    "insufficient_data": ":red-badge[:material/help: Insufficient data]",
}


def prose(text: str) -> str:
    """Escape dollar signs so Streamlit renders currency as text, not LaTeX.

    ``st.markdown`` treats ``$...$`` as inline math, which turned every pair of
    dollar figures in an answer into italic or code-styled fragments.
    """
    return text.replace("\\$", "$").replace("$", "\\$")


def _client() -> FinAgentApiClient:
    """Build the API client from UI environment configuration."""
    base_url = os.getenv("FINAGENT_API_URL", "http://127.0.0.1:8000")
    timeout = float(os.getenv("FINAGENT_UI_TIMEOUT_SECONDS", "90"))
    return FinAgentApiClient(base_url, timeout_seconds=timeout)


def _load_catalog(client: FinAgentApiClient, sector: str) -> Catalog | None:
    """Load a catalog and render an honest unavailable state on failure."""
    try:
        return client.get_catalog(sector)
    except ApiClientError as error:
        st.error("Can't reach the analysis service.", icon=":material/cloud_off:")
        st.caption(f"{error} — start `finagent-mcp` and `finagent-api`, then refresh.")
        return None


def _format_metric(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    if unit == "ratio_points":
        return f"{value * 100:+.1f} pp"
    if unit == "years":
        return f"{value:.2f} yrs"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= threshold:
            return f"{value / threshold:.2f}{suffix} {unit}"
    return f"{value:,.0f} {unit}"


def _render_analysis(analysis: Analysis, catalog: Catalog, *, compact: bool) -> None:
    """Render the answer and its provenance without hiding limitations."""
    tickers = {item.company_id: item.ticker or item.name for item in catalog.companies}
    badge = STATUS_BADGE.get(analysis.status) or EVIDENCE_BADGE.get(
        analysis.evidence_status, ""
    )
    st.markdown(f"{badge} :gray[Data as of {analysis.data_as_of or 'n/a'}]")
    coverage = analysis.coverage
    if coverage and coverage.missing_metrics:
        marks = [
            f":green[:material/check:] {key}"
            if key not in coverage.missing_metrics
            else f":red[:material/close:] {key}"
            for key in coverage.required_metrics
        ]
        st.caption("Required inputs: " + " · ".join(marks))

    if analysis.status in STATUS_BADGE:
        st.warning(prose(analysis.answer_markdown), icon=":material/block:")
        if analysis.trace and analysis.trace.llm_calls == 0:
            st.caption(
                "Stopped after scope resolution. No planning or synthesis call was "
                "made, so nothing could be fabricated."
            )
    else:
        st.markdown(prose(analysis.answer_markdown))

    if analysis.findings:
        with st.expander(
            f":material/fact_check: Findings ({len(analysis.findings)}, all source-backed)",
            expanded=not compact,
        ):
            for finding in analysis.findings:
                companies = " ".join(
                    f":blue[{tickers.get(cid, cid)}]" for cid in finding.company_ids
                )
                sources = " ".join(f":gray-badge[{sid}]" for sid in finding.source_ids)
                st.markdown(f"- {prose(finding.text)} {companies} {sources}")

    if analysis.derived_metrics:
        with st.expander(
            f":material/calculate: Derived metrics ({len(analysis.derived_metrics)}) "
            "— computed in code, not by the model"
        ):
            rows = [
                {
                    "Company": tickers.get(item.entity_id, item.entity_id),
                    "Metric": item.key,
                    "Value": _format_metric(item.value, item.unit),
                    "Period": item.period_end,
                    "Formula": item.formula,
                }
                for item in analysis.derived_metrics
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
            for caveat in sorted(
                {m.caveat for m in analysis.derived_metrics if m.caveat}
            ):
                st.caption(caveat)

    if analysis.sources:
        with st.expander(f":material/link: Sources ({len(analysis.sources)})"):
            for index, source in enumerate(analysis.sources, start=1):
                dates = " · ".join(
                    text
                    for text in (
                        f"published {source.published_at}"
                        if source.published_at
                        else "",
                        f"retrieved {source.retrieved_at}"
                        if source.retrieved_at
                        else "",
                    )
                    if text
                )
                st.markdown(
                    f"{index}. [{source.title}]({source.url})  \n"
                    f":gray[{source.publisher} · `{source.source_id}`"
                    f"{' · ' + dates if dates else ''}]"
                )

    if analysis.limitations:
        with st.expander(
            f":material/report: Limitations ({len(analysis.limitations)})",
            expanded=analysis.evidence_status != "sufficient" and not compact,
        ):
            for limitation in analysis.limitations:
                st.markdown(f"- {prose(limitation)}")
    st.caption(f"request id `{analysis.request_id}`")


@dataclass
class Panel:
    """One persona's result card: a live step timeline above the answer."""

    persona: str
    label: str
    catalog: Catalog
    compact: bool
    status: object = None
    progress: object = None
    body: object = None
    current: object = None
    current_note: object = None
    planning: object = None
    seen: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def mount(self) -> None:
        st.markdown(f"**{PERSONA_ICONS.get(self.persona, '')} {self.label}**")
        self.status = st.status(":shimmer[Analyzing]", type="compact", expanded=True)
        with self.status:
            self.progress = st.empty()
        self.body = st.empty()
        self.body.skeleton(height=180)

    def on_step(self, event: StepEvent) -> None:
        self.elapsed_ms = event.elapsed_ms
        with self.status:
            if self.current is not None:
                if event.message:
                    self.current_note.caption(event.message)
                self.current.update(label=STEPS[self.seen[-1]][1], state="complete")
            if event.state in STEPS:
                self.current = st.status(
                    f":shimmer[{STEPS[event.state][0]}]", type="step"
                )
                with self.current:
                    # A step needs content to stay on the timeline; the summary
                    # of this step is written here when the next one begins.
                    self.current_note = st.empty()
                    self.current_note.caption(":gray[working…]")
                if event.state == "planning":
                    self.planning = self.current
                self.seen.append(event.state)
                done = sum(state in self.seen for state in MAIN_STEPS)
                remaining = [
                    STEPS[state][0].split(" ")[0].lower()
                    for state in MAIN_STEPS
                    if state not in self.seen
                ]
                self.progress.progress(
                    done / len(MAIN_STEPS),
                    text=f"Step {done} of {len(MAIN_STEPS)} · {STEPS[event.state][0]}"
                    + (f" · next: {', '.join(remaining)}" if remaining else ""),
                )
            elif event.state == "completed":
                self.current = None
                self.progress.empty()

    def on_result(self, analysis: Analysis) -> None:
        elapsed = self.elapsed_ms / 1_000
        trace = analysis.trace
        if self.planning is not None and trace is not None:
            self.planning.markdown(
                f"**Model proposed:** {', '.join(trace.proposed_metric_keys) or '—'}  \n"
                f"**Application ran:** {', '.join(trace.constrained_metric_keys) or '—'}"
            )
        if trace is not None and trace.llm_calls == 0:
            label = f"Stopped after scope check · 0 LLM calls · {elapsed:.1f} s"
        else:
            calls = trace.llm_calls if trace else "?"
            repaired = "1 repair" if trace and trace.repaired else "no repair"
            label = (
                f"Completed in {elapsed:.1f} s · {len(self.seen)} steps · "
                f"{calls} LLM calls · {repaired}"
            )
        self.status.update(label=label, state="complete", expanded=False)
        with self.body.container():
            _render_analysis(analysis, self.catalog, compact=self.compact)

    def on_error(self, error: ApiClientError) -> None:
        with self.status:
            if self.current is not None:
                self.current.update(state="error")
        where = STEPS[self.seen[-1]][0].lower() if self.seen else "starting"
        self.status.update(label=f"Failed while {where}", state="error", expanded=True)
        with self.body.container():
            st.error("The analysis didn't complete.", icon=":material/error:")
            st.write(str(error))
            if error.retryable:
                st.caption("Usually temporary; run it again.")
            if error.request_id:
                st.caption(f"request id `{error.request_id}`")


def _stream_all(
    client: FinAgentApiClient,
    question: str,
    sector: str,
    panels: dict[str, Panel],
) -> dict[str, list[StepEvent | Analysis | ApiClientError]]:
    """Consume one SSE stream per persona on worker threads; render on this one."""
    queues: dict[str, queue.Queue] = {p: queue.Queue() for p in panels}

    def worker(persona: str) -> None:
        try:
            for item in client.stream_analysis(
                query=question, persona=persona, sector=sector
            ):
                queues[persona].put(item)
        except ApiClientError as error:
            queues[persona].put(error)
        finally:
            queues[persona].put(None)

    for persona in panels:
        threading.Thread(target=worker, args=(persona,), daemon=True).start()

    history: dict[str, list] = {p: [] for p in panels}
    active = set(panels)
    while active:
        for persona in list(active):
            try:
                item = queues[persona].get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                active.discard(persona)
                continue
            history[persona].append(item)
            _dispatch(panels[persona], item)
    return history


def _dispatch(panel: Panel, item: StepEvent | Analysis | ApiClientError) -> None:
    if isinstance(item, StepEvent):
        panel.on_step(item)
    elif isinstance(item, Analysis):
        panel.on_result(item)
    else:
        panel.on_error(item)


def _apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice:
        st.session_state["question"] = PRESETS[choice]


def main() -> None:
    """Render the complete single-turn application."""
    compare = bool(st.session_state.get("compare", False))
    st.set_page_config(
        page_title="FinAgent",
        page_icon=":material/query_stats:",
        layout="wide" if compare else "centered",
    )
    st.title("FinAgent")
    st.caption(
        "Grounded sector research from public SEC data. One agent, three analyst "
        "personas, every claim tied to a source."
    )

    client = _client()
    bootstrap_sector = st.session_state.get("sector", DEFAULT_SECTOR)
    catalog = _load_catalog(client, bootstrap_sector)
    if catalog is None:
        st.stop()
    sector_values = [item.value for item in catalog.sectors]
    if bootstrap_sector not in sector_values:
        bootstrap_sector = sector_values[0]

    with st.sidebar:
        sector = st.selectbox(
            "Sector",
            options=sector_values,
            index=sector_values.index(bootstrap_sector),
            format_func={item.value: item.label for item in catalog.sectors}.get,
            key="sector",
        )
        if sector != bootstrap_sector:
            catalog = _load_catalog(client, sector)
            if catalog is None:
                st.stop()
        persona_values = [item.value for item in catalog.personas]
        persona_labels = {item.value: item.label for item in catalog.personas}
        descriptions = {item.value: item.description for item in catalog.personas}
        st.toggle(
            "Compare all three personas",
            key="compare",
            help="Ask the same question as every persona and show the answers side by side.",
        )
        persona = st.segmented_control(
            "Persona",
            options=persona_values,
            format_func=lambda v: f"{PERSONA_ICONS.get(v, '')} {persona_labels[v]}",
            default=persona_values[0],
            disabled=compare,
        )
        if persona and not compare:
            st.caption(descriptions[persona])
        st.space(size="small")
        st.markdown("**Dataset**")
        coverage = ""
        if catalog.coverage_start or catalog.coverage_end:
            coverage = f" · coverage {catalog.coverage_start} – {catalog.coverage_end}"
        st.caption(
            f"{len(catalog.companies)} companies · v{catalog.dataset_version}{coverage}"
        )
        st.caption(", ".join(item.ticker or item.name for item in catalog.companies))

    st.pills(
        "Sample questions",
        options=list(PRESETS),
        key="preset",
        on_change=_apply_preset,
        label_visibility="collapsed",
    )
    question = st.text_area(
        "Question",
        key="question",
        height=100,
        placeholder=(
            "Ask about the companies in this sector, e.g. who is improving margins "
            "and who is under pressure."
        ),
    )
    submitted = st.button(
        "Analyze", type="primary", icon=":material/play_arrow:", width="stretch"
    )

    personas = persona_values if compare else [persona or persona_values[0]]
    if submitted:
        if len((question or "").strip()) < 3:
            st.warning("Ask a question of at least three characters.")
            return
        st.session_state["run"] = {
            "question": question.strip(),
            "sector": sector,
            "personas": personas,
            "history": None,
        }
    run = st.session_state.get("run")
    if run is None:
        st.caption(
            ":material/manage_search: Pick a preset or write a question. Answers use "
            "only the sector dataset reached through MCP tools; nothing is fabricated."
        )
        return

    panels: dict[str, Panel] = {}
    columns = st.columns(len(run["personas"]), gap="medium")
    for column, name in zip(columns, run["personas"], strict=True):
        with column, st.container(border=True):
            panel = Panel(name, persona_labels[name], catalog, compact=len(columns) > 1)
            panel.mount()
            panels[name] = panel

    if run["history"] is None:
        run["history"] = _stream_all(client, run["question"], run["sector"], panels)
    else:
        # Replay the cached stream so reruns (expander clicks) do not re-request.
        for name, items in run["history"].items():
            for item in items:
                _dispatch(panels[name], item)


if __name__ == "__main__":
    main()
