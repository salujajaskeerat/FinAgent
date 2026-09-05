"""Single-page Streamlit client for grounded financial analyses."""

from __future__ import annotations

import os

import streamlit as st

from ui.api_client import (
    Analysis,
    ApiClientError,
    Catalog,
    FinAgentApiClient,
)

DEFAULT_SECTOR = "tech"
EXAMPLE_QUESTIONS = (
    "What is the most recent headcount or hiring signal for a company in this dataset?",
    "Compare the companies' margin profiles and explain the main risks.",
    "Which company appears strongest through the selected analyst lens?",
)


def _client() -> FinAgentApiClient:
    """Build the API client from UI environment configuration."""
    base_url = os.getenv("FINAGENT_API_URL", "http://127.0.0.1:8000")
    timeout = float(os.getenv("FINAGENT_UI_TIMEOUT_SECONDS", "50"))
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
    left, middle, right = st.columns(3)
    left.metric("Companies", len(catalog.companies))
    middle.metric("Dataset version", catalog.dataset_version)
    right.metric("Coverage", coverage)


def _render_analysis(analysis: Analysis, catalog: Catalog) -> None:
    """Render the answer and its provenance without hiding limitations."""
    persona_labels = {item.value: item.label for item in catalog.personas}
    sector_labels = {item.value: item.label for item in catalog.sectors}
    st.subheader("Analysis")
    st.caption(
        f"{persona_labels.get(analysis.persona, analysis.persona)} · "
        f"{sector_labels.get(analysis.sector, analysis.sector)} · "
        f"Evidence: {analysis.evidence_status.replace('_', ' ').title()} · "
        f"Data as of: {analysis.data_as_of or 'not available'}"
    )

    if analysis.status == "out_of_scope" or analysis.status == "insufficient_data":
        st.warning(analysis.answer_markdown)
    else:
        st.markdown(analysis.answer_markdown)

    if analysis.companies:
        labels = [
            f"{company.name} ({company.ticker})" if company.ticker else company.name
            for company in analysis.companies
        ]
        st.caption("Companies referenced: " + ", ".join(labels))

    if analysis.findings:
        with st.expander("Evidence-backed findings", expanded=True):
            for finding in analysis.findings:
                st.markdown(f"- {finding.text}")
                st.caption("Sources: " + ", ".join(finding.source_ids))

    with st.expander("Sources and metadata", expanded=bool(analysis.sources)):
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
            "Limitations", expanded=analysis.evidence_status != "sufficient"
        ):
            for limitation in analysis.limitations:
                st.markdown(f"- {limitation}")

    with st.expander("Developer details"):
        st.code(analysis.request_id, language=None)


def main() -> None:
    """Render the complete single-turn application."""
    st.set_page_config(page_title="FinAgent", page_icon="📊", layout="centered")
    st.title("FinAgent")
    st.caption(
        "Persona-configurable financial analysis grounded in the sample dataset."
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
    persona_descriptions = {item.value: item.description for item in catalog.personas}
    selected_persona = st.radio(
        "Analyst persona",
        options=persona_values,
        format_func=persona_labels.get,
        horizontal=True,
    )
    st.caption(persona_descriptions[selected_persona])
    _render_catalog_summary(catalog)

    with st.form("analysis_form", clear_on_submit=False):
        example = st.selectbox(
            "Example question", options=("Write my own question", *EXAMPLE_QUESTIONS)
        )
        initial_question = "" if example == "Write my own question" else example
        question = st.text_area(
            "Question",
            value=initial_question,
            height=120,
            placeholder="Ask a question grounded in the selected sector dataset.",
        )
        submitted = st.form_submit_button(
            "Analyze", type="primary", use_container_width=True
        )

    if not submitted:
        st.info("Select a persona and sector, then submit one question.")
        return
    if len(question.strip()) < 3:
        st.warning("Enter a question of at least three characters.")
        return

    try:
        with st.spinner("Retrieving and validating source-backed evidence…"):
            analysis = client.analyze(
                query=question.strip(),
                persona=selected_persona,
                sector=selected_sector,
            )
    except ApiClientError as error:
        st.error("The analysis could not be completed.")
        st.write(str(error))
        if error.retryable:
            st.info(
                "This failure may be temporary. Try again after the services recover."
            )
        if error.request_id:
            st.caption(f"Request ID: `{error.request_id}`")
        return

    _render_analysis(analysis, catalog)


if __name__ == "__main__":
    main()
