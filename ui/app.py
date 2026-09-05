"""Single-page Streamlit client for grounded financial analyses."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from ui.api_client import (
    Analysis,
    ApiClientError,
    Catalog,
    FinAgentApiClient,
)

DEFAULT_SECTOR = "tech"
# The assignment's own sample questions, so reviewers can run them in one click.
EXAMPLE_QUESTIONS = (
    "Is this sector a good place to be putting money to work right now?",
    "Which of these companies would fit a long-term core holding versus a name I should avoid?",
    "Walk me through the margin profile of the companies in your data - who's improving and who's under pressure?",
    "If I had to pick one company here to take private, which would it be and what's the operational thesis?",
    "Which companies in this sector look like attractive buyout targets based on the data you have?",
    "What's the most recent headcount or hiring signal you have for Apple?",
    "What do you think about Tesla?",
)


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
        st.error("The dataset catalog could not be loaded.")
        st.caption(str(error))
        if error.request_id:
            st.caption(f"Request ID: `{error.request_id}`")
        return None


def _render_catalog_summary(catalog: Catalog) -> None:
    """Render compact dataset coverage metadata."""
    coverage = "Unknown"
    if catalog.coverage_start or catalog.coverage_end:
        coverage = f"{catalog.coverage_start or '—'} to {catalog.coverage_end or '—'}"
    with st.container(horizontal=True):
        st.metric("Companies", len(catalog.companies))
        st.metric("Dataset version", catalog.dataset_version)
        st.metric("Coverage", coverage)


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
    persona_labels = {item.value: item.label for item in catalog.personas}
    sector_labels = {item.value: item.label for item in catalog.sectors}
    company_names = {
        item.company_id: item.ticker or item.name for item in catalog.companies
    }
    status_colour = {"sufficient": "green", "partial": "orange", "none": "red"}
    colour = status_colour.get(analysis.evidence_status, "gray")
    st.markdown(
        f"**{persona_labels.get(analysis.persona, analysis.persona)}** · "
        f"{sector_labels.get(analysis.sector, analysis.sector)} · "
        f":{colour}-badge[Evidence {analysis.evidence_status}] · "
        f"Data as of {analysis.data_as_of or 'n/a'}"
    )
    if analysis.coverage and analysis.coverage.missing_metrics:
        st.caption(
            "Missing required inputs: " + ", ".join(analysis.coverage.missing_metrics)
        )

    if analysis.status in {"out_of_scope", "insufficient_data"}:
        st.warning(analysis.answer_markdown)
    else:
        st.markdown(analysis.answer_markdown)

    if analysis.companies:
        labels = [
            f"{company.name} ({company.ticker})" if company.ticker else company.name
            for company in analysis.companies
        ]
        st.caption("Companies referenced: " + ", ".join(labels))

    if analysis.derived_metrics:
        with st.expander(
            "Derived metrics (computed by the application, not the model)",
            expanded=not compact,
        ):
            rows = [
                {
                    "Company": company_names.get(item.entity_id, item.entity_id),
                    "Metric": item.key,
                    "Value": _format_metric(item.value, item.unit),
                    "Period": item.period_end,
                    "Formula": item.formula,
                }
                for item in analysis.derived_metrics
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
            caveats = sorted(
                {item.caveat for item in analysis.derived_metrics if item.caveat}
            )
            for caveat in caveats:
                st.caption(caveat)

    if analysis.findings:
        with st.expander("Evidence-backed findings", expanded=not compact):
            for finding in analysis.findings:
                st.markdown(f"- {finding.text}")
                st.caption("Sources: " + ", ".join(finding.source_ids))

    with st.expander("Sources", expanded=False):
        if not analysis.sources:
            st.caption("No supporting sources were returned.")
        for source in analysis.sources:
            st.markdown(f"[{source.title}]({source.url})")
            dates = []
            if source.published_at:
                dates.append(f"published {source.published_at}")
            if source.retrieved_at:
                dates.append(f"retrieved {source.retrieved_at}")
            suffix = f" · {' · '.join(dates)}" if dates else ""
            st.caption(f"{source.publisher} · `{source.source_id}`{suffix}")

    if analysis.limitations:
        with st.expander(
            "Limitations",
            expanded=analysis.evidence_status != "sufficient" and not compact,
        ):
            for limitation in analysis.limitations:
                st.markdown(f"- {limitation}")

    with st.expander("How this answer was produced"):
        if analysis.trace:
            st.markdown(" → ".join(analysis.trace.states))
            st.caption(
                f"LLM calls: {analysis.trace.llm_calls} · "
                f"grounding repair: {'yes' if analysis.trace.repaired else 'no'}"
            )
            if (
                analysis.trace.proposed_metric_keys
                or analysis.trace.constrained_metric_keys
            ):
                st.markdown(
                    "**Model proposed:** "
                    + (", ".join(analysis.trace.proposed_metric_keys) or "—")
                    + "  \n**Application ran:** "
                    + (", ".join(analysis.trace.constrained_metric_keys) or "—")
                )
        st.code(analysis.request_id, language=None)


def _run_analyses(
    client: FinAgentApiClient, question: str, personas: list[str], sector: str
) -> dict[str, Analysis | ApiClientError]:
    """Run one request per persona concurrently; the API is the only dependency."""

    def one(persona: str) -> Analysis | ApiClientError:
        try:
            return client.analyze(query=question, persona=persona, sector=sector)
        except ApiClientError as error:
            return error

    with ThreadPoolExecutor(max_workers=len(personas)) as pool:
        return dict(zip(personas, pool.map(one, personas), strict=True))


def _render_error(error: ApiClientError) -> None:
    st.error("The analysis could not be completed.")
    st.write(str(error))
    if error.retryable:
        st.info("This failure may be temporary. Try again after the services recover.")
    if error.request_id:
        st.caption(f"Request ID: `{error.request_id}`")


def main() -> None:
    """Render the complete single-turn application."""
    st.set_page_config(page_title="FinAgent", page_icon="📊", layout="wide")
    st.title("FinAgent")
    st.caption(
        "One persona-configurable agent over an SEC-sourced sector dataset, "
        "reached through MCP tools."
    )

    client = _client()
    bootstrap_sector = st.session_state.get("sector", DEFAULT_SECTOR)
    catalog = _load_catalog(client, bootstrap_sector)
    if catalog is None:
        st.info("Start the FastAPI and MCP services, then refresh this page.")
        st.stop()

    sector_values = [item.value for item in catalog.sectors]
    if bootstrap_sector not in sector_values:
        bootstrap_sector = sector_values[0]

    with st.sidebar:
        selected_sector = st.selectbox(
            "Sector",
            options=sector_values,
            index=sector_values.index(bootstrap_sector),
            format_func={item.value: item.label for item in catalog.sectors}.get,
            key="sector",
        )
        if selected_sector != bootstrap_sector:
            catalog = _load_catalog(client, selected_sector)
            if catalog is None:
                st.stop()

        persona_values = [item.value for item in catalog.personas]
        persona_labels = {item.value: item.label for item in catalog.personas}
        persona_descriptions = {
            item.value: item.description for item in catalog.personas
        }
        compare = st.toggle(
            "Compare all three personas",
            value=False,
            help="Ask the same question as every persona and show the answers side by side.",
        )
        selected_persona = st.segmented_control(
            "Analyst persona",
            options=persona_values,
            format_func=persona_labels.get,
            default=persona_values[0],
            disabled=compare,
        )
        if selected_persona:
            st.caption(persona_descriptions[selected_persona])
        _render_catalog_summary(catalog)
        st.caption(
            "Companies: "
            + ", ".join(item.ticker or item.name for item in catalog.companies)
        )

    with st.form("analysis_form", clear_on_submit=False):
        example = st.selectbox(
            "Sample question", options=("Write my own question", *EXAMPLE_QUESTIONS)
        )
        initial_question = "" if example == "Write my own question" else example
        question = st.text_area(
            "Question",
            value=initial_question,
            height=100,
            placeholder="Ask a question grounded in the selected sector dataset.",
        )
        submitted = st.form_submit_button("Analyze", type="primary", width="stretch")

    if not submitted:
        st.info("Select a sector and persona, then submit one question.")
        return
    if len(question.strip()) < 3:
        st.warning("Enter a question of at least three characters.")
        return

    personas = persona_values if compare else [selected_persona or persona_values[0]]
    with st.spinner("Retrieving, calculating, and validating source-backed evidence…"):
        results = _run_analyses(client, question.strip(), personas, selected_sector)

    if compare:
        columns = st.columns(len(personas))
        for column, persona in zip(columns, personas, strict=True):
            with column, st.container(border=True):
                result = results[persona]
                if isinstance(result, ApiClientError):
                    _render_error(result)
                else:
                    _render_analysis(result, catalog, compact=True)
        return

    result = results[personas[0]]
    if isinstance(result, ApiClientError):
        _render_error(result)
    else:
        _render_analysis(result, catalog, compact=False)


if __name__ == "__main__":
    main()
