"""Deterministic composer ghost-suggestion extractor (slice 1).

Given the assistant's final message text, propose up to three likely user
replies for the empty composer. Pure text analysis: no I/O, no config, no
model calls. Served over the ``complete.suggest`` RPC and rendered as ghost
text by the web and Ink composers; the renderer consumes the first
candidate, the rest are ranked spares for a future cycling UX.

Heuristics, in rank order:

- ``path``  — file paths the message talks about (outside code fences),
  most recently mentioned first. Fenced code is stripped first: a path in a
  command the user was told to RUN is not reply material.
- ``option`` — a trailing numbered list under a question ("Which
  approach?\\n1. ...\\n2. ...") yields the first two option texts.
- ``confirm`` — an approval-shaped closing question ("Want me to ...?",
  "Does that look right?") yields "Yes, go ahead".

No suggestion is a normal outcome: anything that does not end in an ask
returns an empty list, and the composer simply stays empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CANDIDATES = 3

# How many trailing non-empty prose lines count as "the ask" region. A
# question buried above a long report should not resurrect stale suggestions.
_TAIL_LINES = 15

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
# Paths inside backticks/quotes, plus bare ~/... tokens in prose.
_PATH_RE = re.compile(
    r"[`\"']((?:~|/)[^\s`\"']+)[`\"']"  # quoted or backticked absolute / ~ path
    r"|(?<![\w`\"'])(~/[^\s`\"',;)]+)"  # bare ~/token
)
_NUM_ITEM_RE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$")
_CONFIRM_RE = re.compile(
    r"(?:confirm|want me to|should i|shall i|ok to|okay to|look right|looks right"
    r"|sound good|sounds good|good to go|go ahead|proceed|happy with|is that right"
    r"|make sense)[^?]*\?\s*$",
    re.IGNORECASE,
)

_OPEN_QUESTION_RE = re.compile(r"\s*(?:which|what|where|when|who|whom|whose|how)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    text: str
    kind: str  # "path" | "option" | "confirm"


def extract_suggestions(assistant_text: str) -> list[Candidate]:
    if not assistant_text or not assistant_text.strip():
        return []

    prose = _FENCE_RE.sub("", assistant_text)
    tail = [line for line in prose.splitlines() if line.strip()][-_TAIL_LINES:]
    if not any("?" in line for line in tail):
        return []

    out: list[Candidate] = []
    seen: set[str] = set()

    def add(text: str, kind: str) -> None:
        cleaned = text.strip().rstrip(".,;:")
        if cleaned and cleaned not in seen and len(out) < MAX_CANDIDATES:
            seen.add(cleaned)
            out.append(Candidate(cleaned, kind))

    # Paths: most recently mentioned first.
    paths = [m.group(1) or m.group(2) for m in _PATH_RE.finditer(prose)]
    for path in reversed(paths):
        add(path, "path")

    # Options: numbered items in the tail region, first two in reading order.
    options = [m.group(1) for line in tail if (m := _NUM_ITEM_RE.match(line))]
    for option in options[:2]:
        add(option, "option")

    # Confirm: approval-shaped closing question. The closing line is the last
    # non-list prose line, so "Which approach?" followed by options still
    # yields options rather than a bare yes.
    closing = next(
        (line for line in reversed(tail) if not _NUM_ITEM_RE.match(line)), ""
    )
    # "Which file should I keep?" asks for a choice, not a yes: judge the last sentence only.
    ask = re.split(r"(?<=[.!])\s+", closing.strip())[-1]
    if _CONFIRM_RE.search(ask) and not _OPEN_QUESTION_RE.match(ask):
        add("Yes, go ahead", "confirm")

    return out
