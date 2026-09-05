"""Deterministic Markdown normalization for model-written answers.

Models sometimes emit ``### Heading body text ### Next heading`` on one line,
which Markdown renders as a single giant heading. The answer contract is
enforced here rather than trusted from the prompt.
"""

from __future__ import annotations

import re

_HEADING_MID_LINE = re.compile(r"(?<!\n)\s*(#{1,6}\s+)")
_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_BULLET_MID_LINE = re.compile(r"(?<=[.;:!?])\s+[-*•]\s+(?=[A-Z0-9])")
_LEADING_BULLET = re.compile(r"^\s*[-*•]\s+")


def normalize_answer_markdown(text: str, sections: list[str] | None = None) -> str:
    """Return Markdown where every heading and bullet starts on its own line.

    Parameters
    ----------
    text
        Model-written answer.
    sections
        Persona's required section titles; headings matching them are forced
        to level three so the answer has one consistent hierarchy.

    Returns
    -------
    str
        Normalized Markdown with blank lines around headings and one bullet per
        line. Content is never added or removed, only re-flowed.
    """
    if not text.strip():
        return text
    wanted = [item.strip() for item in sections or []]
    flowed = _HEADING_MID_LINE.sub(lambda m: "\n\n" + m.group(1), text)
    lines: list[str] = []
    for raw_line in flowed.splitlines():
        line = raw_line.rstrip()
        heading = _HEADING_LINE.match(line)
        candidate = heading.group(2).strip("# ").strip() if heading else line.strip()
        title, body = _split_known_section(candidate, wanted)
        if title is not None:
            lines.extend(["", f"### {title}", ""])
            if body:
                line = body
            else:
                continue
        elif heading:
            lines.extend(["", f"{heading.group(1)} {candidate}", ""])
            continue
        if not line.strip():
            lines.append("")
            continue
        # Split "sentence. - Next point" runs into one bullet per line.
        parts = _BULLET_MID_LINE.split(line)
        if len(parts) > 1:
            first = parts[0]
            lines.append(first if _LEADING_BULLET.match(first) else f"- {first}")
            lines.extend(f"- {part.strip()}" for part in parts[1:])
        else:
            # A paragraph directly after a bullet would be absorbed into it.
            if (
                lines
                and _LEADING_BULLET.match(lines[-1])
                and not _LEADING_BULLET.match(line)
            ):
                lines.append("")
            lines.append(line)
    # Collapse runs of blank lines and trim.
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return result + "\n"


def _split_known_section(line: str, sections: list[str]) -> tuple[str | None, str]:
    """Return (section title, remaining text) when a line starts with a section."""
    lowered = line.lower()
    for section in sections:
        if lowered.startswith(section.lower()):
            rest = line[len(section) :].lstrip(" :-–—").strip()
            return section, rest
    return None, line
