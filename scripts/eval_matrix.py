"""Run the assignment's sample queries against a live API and write a report.

Usage::

    uv run python scripts/eval_matrix.py --api http://127.0.0.1:8000 --out docs/EVAL.md

The script uses only the public HTTP contract, so it exercises exactly the
path a reviewer would. It reports, per query, the domain status, evidence
status, missing required inputs, whether the persona's required sections
appeared, and how much the three persona answers to the same question overlap.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

PERSONAS = ("mutual_fund_analyst", "equity_analyst", "pe_analyst")
SECTORS = ("tech", "retail", "logistics")

# The assignment's own sample queries. "Manufacturing" in the brief maps to
# Logistics here because those are the three sectors this dataset covers.
CROSS_PERSONA_QUESTION = (
    "Is this sector a good place to be putting money to work right now?"
)
SCENARIOS: list[tuple[str, str, str, str]] = [
    (
        "MF core holding vs avoid",
        "mutual_fund_analyst",
        "retail",
        "Which of these companies would fit a long-term core holding versus a name I should avoid?",
    ),
    (
        "Equity margin profile",
        "equity_analyst",
        "logistics",
        "Walk me through the margin profile of the companies in your data - who's improving and who's under pressure?",
    ),
    (
        "PE take-private pick",
        "pe_analyst",
        "tech",
        "If I had to pick one company here to take private, which would it be and what's the operational thesis?",
    ),
    (
        "PE buyout targets",
        "pe_analyst",
        "logistics",
        "Which companies in this sector look like attractive buyout targets based on the data you have?",
    ),
    (
        "Headcount grounding",
        "equity_analyst",
        "tech",
        "What's the most recent headcount or hiring signal you have for Apple?",
    ),
    (
        "Out-of-scope company",
        "mutual_fund_analyst",
        "tech",
        "What do you think about Tesla?",
    ),
    (
        "API contract check",
        "equity_analyst",
        "logistics",
        "Summarize the operating performance of the logistics companies you cover.",
    ),
]

REQUIRED_SECTIONS = {
    "mutual_fund_analyst": (
        "Benchmark-relative view",
        "Growth durability",
        "Portfolio fit and downside",
    ),
    "equity_analyst": ("Earnings and margins", "Valuation", "Catalysts and risks"),
    "pe_analyst": (
        "Entry and leverage case",
        "Operational value creation",
        "Exit and diligence gaps",
    ),
}
PERSONA_VOCABULARY = {
    "mutual_fund_analyst": (
        "benchmark",
        "median",
        "durab",
        "portfolio",
        "core holding",
    ),
    "equity_analyst": (
        "margin",
        "operating income",
        "earnings",
        "valuation",
        "catalyst",
    ),
    "pe_analyst": ("leverage", "net debt", "free cash flow", "exit", "operational"),
}


@dataclass
class Run:
    label: str
    persona: str
    sector: str
    query: str
    http_status: int
    latency_s: float
    body: dict

    @property
    def answer(self) -> str:
        return str(self.body.get("answer_markdown", ""))

    def sections_present(self) -> tuple[int, int]:
        required = REQUIRED_SECTIONS[self.persona]
        if self.body.get("status") != "answered":
            return 0, len(required)
        return sum(f"### {name}" in self.answer for name in required), len(required)

    def vocabulary_hits(self) -> int:
        text = self.answer.lower()
        return sum(term in text for term in PERSONA_VOCABULARY[self.persona])


def _post(
    api: str, persona: str, sector: str, query: str, timeout: float
) -> tuple[int, dict, float]:
    payload = json.dumps(
        {"query": query, "persona": persona, "sector": sector}
    ).encode()
    request = urllib.request.Request(
        f"{api.rstrip('/')}/v1/analyses",
        data=payload,
        headers={"Content-Type": "application/json", "X-Correlation-ID": "eval-matrix"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        try:
            body = json.load(error)
        except ValueError:
            body = {"detail": error.reason}
        return error.code, body, time.perf_counter() - started


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{4,}", text.lower())}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def _run_all(api: str, timeout: float) -> tuple[list[Run], list[Run]]:
    cross: list[Run] = []
    for persona in PERSONAS:
        status, body, latency = _post(
            api, persona, "tech", CROSS_PERSONA_QUESTION, timeout
        )
        cross.append(
            Run(
                "Cross-persona (tech)",
                persona,
                "tech",
                CROSS_PERSONA_QUESTION,
                status,
                latency,
                body,
            )
        )
        print(
            f"  cross-persona {persona:22} -> {status} {body.get('status')} {latency:.1f}s",
            file=sys.stderr,
        )
    scenarios: list[Run] = []
    for label, persona, sector, query in SCENARIOS:
        status, body, latency = _post(api, persona, sector, query, timeout)
        scenarios.append(Run(label, persona, sector, query, status, latency, body))
        print(
            f"  {label:24} -> {status} {body.get('status')} {latency:.1f}s",
            file=sys.stderr,
        )
    return cross, scenarios


def _sweep(api: str, timeout: float) -> list[Run]:
    runs: list[Run] = []
    for sector in SECTORS:
        for persona in PERSONAS:
            status, body, latency = _post(
                api, persona, sector, CROSS_PERSONA_QUESTION, timeout
            )
            runs.append(
                Run(
                    "Sweep",
                    persona,
                    sector,
                    CROSS_PERSONA_QUESTION,
                    status,
                    latency,
                    body,
                )
            )
            print(
                f"  sweep {sector:9} {persona:22} -> {status} {body.get('status')} {latency:.1f}s",
                file=sys.stderr,
            )
    return runs


def _row(run: Run) -> str:
    present, total = run.sections_present()
    coverage = run.body.get("coverage") or {}
    missing = ", ".join(coverage.get("missing_metrics", [])) or "—"
    trace = run.body.get("trace") or {}
    return (
        f"| {run.label} | {run.persona} | {run.sector} | {run.http_status} | "
        f"{run.body.get('status', '—')} | {run.body.get('evidence_status', '—')} | {missing} | "
        f"{present}/{total} | {run.vocabulary_hits()}/5 | {len(run.body.get('findings', []))} | "
        f"{len(run.body.get('derived_metrics', []))} | {trace.get('llm_calls', '—')} | {run.latency_s:.1f}s |"
    )


def _report(
    api: str, provider: str, cross: list[Run], scenarios: list[Run], sweep: list[Run]
) -> str:
    header = (
        "| Scenario | Persona | Sector | HTTP | Status | Evidence | Missing required | "
        "Sections | Persona vocab | Findings | Derived | LLM calls | Latency |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [
        "# Evaluation matrix",
        "",
        (
            f"Generated by `scripts/eval_matrix.py` against `{api}` with "
            f"`LLM_PROVIDER={provider}` on "
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}."
        ),
        "",
        (
            "The queries are the assignment's own sample questions. The brief's "
            '"Manufacturing" example is run against Logistics, the closest sector '
            "in this dataset. *Sections* counts the persona's required H3 headings "
            "found in the answer; *Persona vocab* counts persona-specific terms "
            "present."
        ),
        "",
        "## Same question, three personas (Tech)",
        "",
        f"> {CROSS_PERSONA_QUESTION}",
        "",
        header,
        *[_row(run) for run in cross],
        "",
        (
            "Pairwise answer overlap (Jaccard over word stems; lower means more "
            "differentiated):"
        ),
        "",
    ]
    for left, right in combinations(cross, 2):
        lines.append(
            f"- {left.persona} vs {right.persona}: {_jaccard(left.answer, right.answer):.2f}"
        )
    lines += [
        "",
        "## Scenario queries",
        "",
        header,
        *[_row(run) for run in scenarios],
        "",
    ]
    grounding = next(
        (run for run in scenarios if run.label == "Headcount grounding"), None
    )
    if grounding is not None:
        findings = grounding.body.get("findings", [])
        cited = any("employee" in item.get("text", "").lower() for item in findings)
        lines += [
            "Headcount grounding check: "
            + (
                "a finding cites an employee count from a source."
                if cited
                else "no finding cites an employee count."
            ),
            "",
        ]
    out_of_scope = next(
        (run for run in scenarios if run.label == "Out-of-scope company"), None
    )
    if out_of_scope is not None:
        trace = out_of_scope.body.get("trace") or {}
        lines += [
            (
                f"Out-of-scope check: status `{out_of_scope.body.get('status')}`, "
                f"LLM calls `{trace.get('llm_calls')}`, states "
                f"`{' → '.join(trace.get('states', []))}`."
            ),
            "",
        ]
    lines += [
        "## Full persona × sector sweep",
        "",
        header,
        *[_row(run) for run in sweep],
        "",
    ]
    answered = sum(run.body.get("status") == "answered" for run in sweep)
    lines.append(f"{answered}/{len(sweep)} combinations answered.")
    lines += ["", "## Sample answers", ""]
    for run in cross:
        lines += [f"### {run.persona} ({run.sector})", "", run.answer.strip(), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("docs/EVAL.md"))
    parser.add_argument(
        "--provider", default="unknown", help="Label for the report only."
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    print("Running assignment scenarios…", file=sys.stderr)
    cross, scenarios = _run_all(args.api, args.timeout)
    sweep = [] if args.skip_sweep else _sweep(args.api, args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _report(args.api, args.provider, cross, scenarios, sweep), encoding="utf-8"
    )
    print(f"wrote {args.out}", file=sys.stderr)
    failures = [run for run in [*cross, *scenarios, *sweep] if run.http_status != 200]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
