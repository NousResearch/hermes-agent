#!/usr/bin/env python3
"""
Memory classification gate — semantic companion to the security scan in
``tools/memory_tool.py``.

The security scan (threat patterns) blocks adversarial content. This module
addresses the *chronic* failure mode: semantic pollution of the durable
memory store — task-state entries, completed-work logs, imperatives, raw
clinical/genomic data, secrets. Unlike adversarial strings, these are
ordinary English, so the gate is two-tier:

  Tier 1 — hard reject: a small set of HIGH-PRECISION patterns where a match
  is almost certainly wrong for durable memory (commit SHAs, issue/PR
  references, tracked ticket IDs, ISO dates, raw genomic markers, credential
  shapes, embedded file paths). No override; the model must rephrase.

  Tier 2 — warn-and-require-confirmation: softer signals (imperative mood,
  completed-work language, transient status verbs). The write is refused
  UNLESS the caller re-submits with ``override=True`` AND a non-empty
  ``rationale``, which is recorded in the memory ledger. This keeps
  guarantees model-agnostic where precision allows, and converts the fuzzy
  tail into an auditable decision instead of silent pollution.

Design note (2026-07-18): false-positive resistance beats coverage here.
A gate that is wrong often enough gets routed around, and a routed-around
gate is worse than prose. Patterns were chosen conservatively; Tier 2
exists precisely because natural-language classification is lossy.
"""

import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tier 1 — hard reject (high precision)
# ---------------------------------------------------------------------------

_HARD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # VCS / tracker identifiers — task state, not durable fact.
    (re.compile(r"\b[0-9a-f]{40}\b|\b[0-9a-f]{11,12}\b", re.IGNORECASE),
     "commit-SHA-like hex — task state belongs in session search / kanban, not memory"),
    (re.compile(r"(?:^|[\s(])#\d{2,7}\b"),
     "issue/PR number reference — task state, will be stale within weeks"),
    (re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b"),
     "tracker ticket ID (e.g. JIRA-style) — task state, not memory"),
    # Concrete dates — status anchored to a date rots; use durable phrasing.
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
     "ISO date — durable facts shouldn't hinge on a specific date"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
     "calendar date — durable facts shouldn't hinge on a specific date"),
    # Raw clinical / genomic payloads — explicit exclusion in Rob's doctrine.
    (re.compile(r"\brs\d{4,}\b", re.IGNORECASE),
     "rsID / raw genomic marker — raw genomic data does not belong in native memory"),
    (re.compile(r"\b(?:chr(?:1[0-9]|2[0-2]|[1-9]|X|Y|M))[: ]\d{4,}\b", re.IGNORECASE),
     "genomic coordinate — raw genomic data does not belong in native memory"),
    (re.compile(r"\bMRN\b[:\s#]*\w{4,}", re.IGNORECASE),
     "medical record number — clinical PII does not belong in native memory"),
    # Credential-shaped strings (belt-and-suspenders on top of redaction).
    (re.compile(r"\b(?:sk|pk|api|key|token|secret|password)[-_]?[A-Za-z0-9]{16,}\b"),
     "credential-shaped string — secrets never belong in memory"),
    # Embedded file paths — locations go stale; cite the convention, not the path.
    (re.compile(r"[A-Za-z]:\\[^\s]+|/Users/[^\s]+|/home/[^\s]+"),
     "hard file path — cite durable conventions, not locations that will rot"),
]


def scan_hard(content: str) -> Optional[str]:
    """Return a rejection reason if content hits a Tier-1 pattern, else None."""
    for pattern, why in _HARD_PATTERNS:
        if pattern.search(content):
            return f"memory policy reject: {why}"
    return None


# ---------------------------------------------------------------------------
# Tier 2 — warn (requires override + rationale to persist)
# ---------------------------------------------------------------------------

_IMPERATIVE_START = re.compile(
    r"^\s*(?:always|never|make sure|ensure|don't|do not|remember to|"
    r"run|use|check|ask|tell|verify|avoid|write|create|delete|"
    r"update|install|deploy|restart|commit|push)\b",
    re.IGNORECASE,
)

_COMPLETED_WORK = re.compile(
    r"\b(?:fixed|resolved|shipped|merged|closed|completed|done)\b.*\b(?:bug|issue|pr|ticket|task|feature|build|deploy)\b|"
    r"\b(?:bug|issue|pr|ticket|task|feature|build|deploy)\b.*\b(?:fixed|resolved|shipped|merged|closed|completed|done)\b",
    re.IGNORECASE,
)

_PROGRESS_STATE = re.compile(
    r"\b(?:currently|right now|this week|today|as of now|in progress|"
    r"blocked on|working on|phase\s*\d+|step\s*\d+\s*(?:of|/)\s*\d+)\b",
    re.IGNORECASE,
)

_TIER2_CHECKS: List[Tuple[re.Pattern, str]] = [
    (_IMPERATIVE_START,
     "imperative phrasing — memory entries should be declarative facts "
     "('Project uses pytest with xdist'), not directives ('Run pytest -n 4')"),
    (_COMPLETED_WORK,
     "completed-work language — work logs belong in session search, not durable memory"),
    (_PROGRESS_STATE,
     "transient status/progress language — task state belongs in kanban/session search"),
]


def scan_soft(content: str) -> List[str]:
    """Return a list of Tier-2 warnings for content (empty if none)."""
    return [why for pattern, why in _TIER2_CHECKS if pattern.search(content)]


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate(content: str, override: bool = False,
             rationale: Optional[str] = None) -> Tuple[str, List[str]]:
    """Classify a candidate memory entry.

    Returns ``(verdict, messages)`` where verdict is one of:

      - ``"pass"``    — no findings; persist normally.
      - ``"reject"``  — Tier-1 hit; messages[0] explains why. No override.
      - ``"warn"``    — Tier-2 hit(s) and no valid override; the caller must
                        re-submit with ``override=True`` + ``rationale``.
      - ``"override"``— Tier-2 hit(s) but a valid override+rationale was
                        supplied; persist and ledger the override.
    """
    hard = scan_hard(content)
    if hard:
        return "reject", [hard]

    warnings = scan_soft(content)
    if not warnings:
        return "pass", []

    if override and rationale and rationale.strip():
        return "override", warnings
    return "warn", warnings
