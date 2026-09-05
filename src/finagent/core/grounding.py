"""Deterministic validation of model-produced evidence links."""

from finagent.core.models import DraftAnalysis


def grounding_issues(
    draft: DraftAnalysis,
    allowed_company_ids: set[str],
    allowed_source_ids: set[str],
) -> list[str]:
    """Return all grounding violations in a draft.

    Parameters
    ----------
    draft
        Structured model output.
    allowed_company_ids
        Companies present in the selected sector catalog.
    allowed_source_ids
        Sources returned by MCP for this request.

    Returns
    -------
    list[str]
        Human-readable validation failures.
    """
    issues: list[str] = []
    if not draft.findings:
        issues.append("the draft has no source-linked findings")
    for index, finding in enumerate(draft.findings):
        unknown_sources = set(finding.source_ids) - allowed_source_ids
        unknown_companies = set(finding.company_ids) - allowed_company_ids
        if unknown_sources:
            issues.append(
                f"finding {index} cites unknown sources: {sorted(unknown_sources)}"
            )
        if unknown_companies:
            issues.append(
                f"finding {index} cites unknown companies: {sorted(unknown_companies)}"
            )
    return issues
