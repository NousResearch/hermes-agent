#!/usr/bin/env python3
"""Write-time memory acceptance gate (issue #135039).

A 7-stage acceptance pipeline invoked from ``tools.memory_tool._gate_or_stage``
BEFORE ``wa.evaluate_gate`` (``write_approval``).  Stages 1-6 run before the
write and either return ``None`` (admit / proceed to the next stage) or a
:class:`GateRefusal` (refuse and record a Door-B change-log line).  Stage 7
(post-write read-back) runs in the writer path AFTER the store write via
:func:`verify_post_write`.

The gate is imported lazily by ``_gate_or_stage``; if the import fails the
caller proceeds (fail-open / degrade to the pre-gate behaviour), matching the
prior fail-open precedent for lazy gate imports.

Size, limit and stale-entry resolution are SCOPED PER TARGET through the
store's own doc-aware primitives (``_char_limit`` / ``_char_count`` /
``_entries_for`` / ``_doc``).  When the store lacks the doc-chain primitives
(the #116673 work not present), the doc-chain arms degrade to the plain-memory
path (fail-closed) — the gate still refuses over-budget writes, without the
per-target scoping.  Chain docs (``doc:<name>`` targets) are uncapped
by store design (``_char_limit`` returns ``None``); the gate inherits that and
never charges a chain-doc write against the anchor's budget (INTEG-1), and it
searches the target's OWN entries for a stale-entry anchor (INTEG-2).  The
gate and the doc-chain are the SAME store — never re-derive a two-way
memory/user branch that ignores the doc arm.

Stages
------
1  Char-cap soft check (85% of the target's limit; hard cap is the limit).
   Docs are uncapped: skipped when ``_char_limit(target)`` is ``None``.
2  Date-currency check — an entry bearing a date within the configured window
   is flagged for session-log routing (advisory, never blocks; folded into the
   single ADMIT line).
3  Provenance lens — caller harness-set; ``caller == "user"`` admits,
   ``caller != "user"`` + imperative phrasing REFUSEs to ``skills/``.
4  Procedure-shape semantic check — structural markers
   ("step-sequence" / "when X, do Y") REFUSEs to ``skills/`` regardless of
   caller (keys on content, NOT caller).
5  Grammar / threat check — imperative regex + ``_scan_memory_content``.
6  Stale-entry refusal — whole-entry EXACT match via ``_find_unique_match``;
   fail-closed on drift (``anchor_drift`` receipt).  Searched against the
   TARGET's entries (``_entries_for(target)``), not the anchor's.
7  Post-write read-back — after the store write, re-read the target and assert
   ``old_text`` is gone from it (writer path: :func:`verify_post_write`).

Every admitted write appends exactly ONE Door-B change-log line
(``action=ADMIT``, ``rationale="accepted"``); every refusal appends exactly one
line with the violating stage's rationale.  Fields: ``action``, ``rationale``,
``reroute_target``, ``failure_class``, ``timestamp`` (Door-B shape).

"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home, get_scratch_dir

# Re-use the store's installed refusal surfaces — do NOT re-derive.
from tools.memory_tool_store import (
    ENTRY_DELIMITER,
    FAILURE_CLASS,
    _find_unique_match,
    _scan_memory_content,
    _stale_entry_message,
)

import logging
_logger = logging.getLogger(__name__)

# --- Configurable thresholds -------------------------------------------------

#: Soft threshold for Stage 1 — a write that would push the target past this
#: fraction of its limit is refused with ``over_budget``.
CHAR_CAP_SOFT_THRESHOLD = 0.85

#: Default date-currency window for Stage 2 (hours).
DATE_CURRENCY_WINDOW_HOURS = 48

#: Structural markers that indicate a procedure-shaped entry (Stage 4).
_PROCEDURE_STEP_RE = re.compile(
    r"^\s*(step\s+\d+|[a-zA-Z]\.\s|\d+[\.\)]\s|•\s|→\s|[-*]\s)", re.IGNORECASE
)
# "when X do/then Y" — ASCII only; no decorative unicode (T3-1).
_WHEN_DO_RE = re.compile(
    r"\bwhen\b.*\b(?:do|then)\b", re.IGNORECASE
)

#: Imperative verb patterns for Stage 3 / Stage 5 (caller != user).
#: Deduplicated verb list (T3-2); the config surface OD-5 tunes.
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:do|run|execute|call|use|write|patch|remove|add|delete|"
    r"set|update|modify|change|replace|append|prepend|insert|create|install|"
    r"configure|start|stop|apply|merge|commit|push|fetch|clear|reset|reload|"
    r"restart|refresh|sync|copy|move|rename|edit|open|close|build|test|deploy|"
    r"export|import|format|render|scan|check|verify|validate|approve|stage|"
    r"consolidate|drain|prune|sweep|backup|archive|migrate|rotate|flush|wipe|"
    r"purge|clean|trim|compact|gc|recover|restore|rollback|revert|relabel|"
    r"renumber|reorder|resort|split|join|concat)\b",
    re.IGNORECASE,
)


@dataclass
class GateRefusal:
    """Door-B change-log line.

    ``action``   — ADMIT / REFUSE / REMOVE / CONSOLIDATE.
    ``rationale`` — which rule was violated or the displacement decision.
    ``reroute_target`` — where the content belongs (None when not applicable).
    """

    action: str  # ADMIT, REFUSE, REMOVE, CONSOLIDATE
    rationale: str
    reroute_target: Optional[str] = None
    failure_class: str = "write_gated"
    # Extra structured detail for the log line.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "timestamp": _now_iso(),
            "action": self.action,
            "rationale": self.rationale,
            "reroute_target": self.reroute_target,
            "failure_class": self.failure_class,
            **self.detail,
        }


def _now() -> datetime:
    from datetime import timezone
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


# --- JSONL audit log --------------------------------------------------------

def _audit_log_path() -> Path:
    """Path to the write-gate JSONL change-log.

    Lives under ``<hermes_home>/logs/`` so tests can redirect via ``HERMES_HOME``.
    """
    try:
        return get_hermes_home() / "logs" / "write_gate_audit.jsonl"
    except Exception:  # health: allow BLE001 -- audit-log write must never block the gate
        return get_scratch_dir() / "logs" / "write_gate_audit.jsonl"


def _append_change_log_line(entry: dict[str, Any]) -> None:
    """Append one JSONL line to the audit log (best-effort, never raises)."""
    try:
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# --- Store-scoped primitives (INTEG-1 / INTEG-2) ----------------------------

def _is_doc(store, target: str) -> bool:
    """True when *target* addresses a chain doc (uncapped, doc-resident entries)."""
    _doc = getattr(store, "_doc", None)
    if not callable(_doc):
        return False
    return _doc(target) is not None


def _store_char_count(store, target: str) -> int:
    """Current char count for *target*, via the store's own doc-aware arm.

    Routes through ``store._char_count`` so a ``doc:`` target reports the doc's
    size, not the anchor's (INTEG-1).  Falls back to the two-way branch only
    for a store that predates the doc arm.
    """
    cc = getattr(store, "_char_count", None)
    if callable(cc):
        return cc(target)
    entries = store.user_entries if target == "user" else store.memory_entries
    return len(ENTRY_DELIMITER.join(entries))


def _store_char_limit(store, target: str) -> Optional[int]:
    """Char limit for *target* via the store's own doc-aware arm.

    Returns ``None`` for chain docs (uncapped by store design) — the gate
    inherits that and skips the cap check for docs (INTEG-1).
    """
    cl = getattr(store, "_char_limit", None)
    if callable(cl):
        return cl(target)
    return store.user_char_limit if target == "user" else store.memory_char_limit


def _store_entries(store, target: str) -> list[str]:
    """The TARGET's own entries via the store's doc-aware arm (INTEG-2).

    A ``doc:`` target returns the doc's entries (``_entries_for``), never the
    anchor's — so a stale-entry search anchors against the surface the write
    actually touches.
    """
    ef = getattr(store, "_entries_for", None)
    if callable(ef):
        return list(ef(target))
    return list(store.user_entries if target == "user" else store.memory_entries)


def _chain_doc_destination(store) -> str:
    """Name the chain-doc destination for an over_budget reroute (T1-3).

    Prefers a doc the anchor already references (the concrete doc-world
    destination); falls back to the generic ``chain-doc`` label.  Displacement
    itself (age-ordered eviction) stays out of scope (OD-8) — this only NAMES
    where the content should live instead.
    """
    try:
        docs = store.chain_docs()
    except Exception:  # health: allow BLE001 -- chain-doc naming must never block the gate
        docs = []
    if docs:
        return f"doc:{docs[0]}"
    return "chain-doc"


# --- Stage implementations ---------------------------------------------------

def _stage1_char_cap(store, content: str, target: str, hard_cap: bool = False) -> Optional[GateRefusal]:
    """Stage 1 — char-cap soft check at 85% of the target's limit.

    If the new content would push the target past ``soft_threshold * limit``,
    refuse with ``over_budget``.  When ``hard_cap`` is True the threshold is
    the target's own hard limit (used by the never-exceeds invariant).

    Docs are uncapped (``_char_limit`` is ``None``): the check is skipped, so a
    chain-doc write is never charged against the anchor's budget (INTEG-1).
    """
    if not content:
        return None
    limit = _store_char_limit(store, target)
    if limit is None:
        return None  # uncapped target (chain doc): nothing to outgrow (INTEG-1)
    current = _store_char_count(store, target)
    threshold = limit if hard_cap else int(limit * CHAR_CAP_SOFT_THRESHOLD)
    projected = current + len(content)
    if projected > threshold:
        return GateRefusal(
            action="REFUSE",
            rationale=(
                f"char-cap exceeded: {projected:,}/{limit:,} chars"
                + (f" (hard limit {threshold:,})" if hard_cap
                   else f" (soft {int(CHAR_CAP_SOFT_THRESHOLD * 100)}% = {threshold:,})")
            ),
            reroute_target=None if hard_cap else _chain_doc_destination(store),
            failure_class="over_budget",
            detail={"current": current, "projected": projected, "limit": limit,
                    "threshold": threshold, "hard_cap": hard_cap},
        )
    return None


def _stage1_projected_hard_cap(store, target: str, action: str,
                               content: Optional[str], old_text: Optional[str]) -> Optional[GateRefusal]:
    """Stage 1 (hard-cap arm) — projected FINAL serialized size for a
    replace/patch that would GROW the store past the hard limit (T1-7).

    ``add_contents`` is add-only in the main loop, so a short->long rewrite used
    to bypass the cap entirely.  This arm computes the net delta
    (``+len(new) - len(old)``) and the projected serialized size (matching the
    store's own ``len(ENTRY_DELIMITER.join(entries))`` accounting, including the
    delimiter boundary) and refuses when the FINAL state would exceed the limit.
    """
    if action not in ("replace", "patch"):
        return None
    limit = _store_char_limit(store, target)
    if limit is None:
        return None  # uncapped doc target
    new_len = len(content or "")
    old_len = len(old_text or "")
    net_delta = new_len - old_len
    if net_delta <= 0:
        return None  # the edit shrinks or holds the store: nothing to outgrow
    current = _store_char_count(store, target)
    projected = current + net_delta
    if projected > limit:
        return GateRefusal(
            action="REFUSE",
            rationale=(
                f"char-cap exceeded: this {action} grows the store to "
                f"{projected:,}/{limit:,} chars (hard limit); free space first"
            ),
            reroute_target=_chain_doc_destination(store),
            failure_class="over_budget",
            detail={"current": current, "projected": projected, "limit": limit,
                    "net_delta": net_delta, "action": action},
        )
    return None


def _stage2_date_currency(content: str) -> Optional[dict[str, Any]]:
    """Stage 2 — date-currency ADVISORY annotation (never blocks).

    Returns a detail dict when *content* bears a date within the freshness
    window (so the single ADMIT line can carry the flag), else ``None``.
    Future dates are NOT fresh (``0 <= now - parsed``), so a 2099 date does not
    fire (T2-2b).
    """
    date_re = re.compile(
        r"(\d{4}-\d{2}-\d{2}"          # 2026-10-13
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"  # Oct 13, 2026
        r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})"   # 13 Oct 2026
        , re.IGNORECASE
    )
    now = _now()
    window = timedelta(hours=DATE_CURRENCY_WINDOW_HOURS)
    for m in date_re.finditer(content):
        parsed = _parse_date(m.group(0))
        if parsed is not None:
            age = now - parsed
            if timedelta(0) <= age <= window:  # fresh, not future (T2-2b)
                return {"advisory": "date_currency", "date": m.group(0),
                        "window_hours": DATE_CURRENCY_WINDOW_HOURS}
    return None


def _parse_date(text: str) -> Optional[datetime]:
    """Best-effort parse of a date string into a timezone-aware datetime."""
    tz = _now().tzinfo
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(text.strip(), fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue
    return None


def _stage3_provenance(caller: str, content: str) -> Optional[GateRefusal]:
    """Stage 3 — provenance lens.

    ``caller`` is harness-set.  ``caller == "user"`` → standing preference,
    admit.  ``caller != "user"`` + imperative phrasing → REFUSE to skills/.
    An ``unknown`` caller (provenance resolution failed) is treated as non-user
    (T2-1): agent-inferred imperative content is never silently promoted to
    user-trusted.
    """
    if caller == "user":
        return None
    if _IMPERATIVE_RE.match(content or ""):
        return GateRefusal(
            action="REFUSE",
            rationale="provenance: imperative content from non-user caller",
            reroute_target="skills/",
            failure_class="scan_blocked",
            detail={"caller": caller, "rule": "imperative_non_user"},
        )
    return None


def _stage4_procedure_shape(content: str) -> Optional[GateRefusal]:
    """Stage 4 — procedure-shape semantic check.

    Detects recurring-task / procedure instructions by structural markers
    (step-sequence or "when X, do Y" templates).  Keys on content, NOT
    caller — a procedure-shaped entry from any caller is refused to skills/.
    """
    if not content:
        return None
    if _PROCEDURE_STEP_RE.match(content) or _WHEN_DO_RE.search(content):
        return GateRefusal(
            action="REFUSE",
            rationale="procedure-shape: procedure-shaped entry does not belong in memory",
            reroute_target="skills/",
            failure_class="scan_blocked",
            detail={"rule": "procedure_shape", "markers": "step-sequence|when-do"},
        )
    return None


def _stage5_grammar_threat(content: str) -> Optional[GateRefusal]:
    """Stage 5 — grammar / threat check.

    Reuses the store's ``_scan_memory_content`` (the installed threat-pattern
    gate).  If the content matches an injection / exfil pattern, refuse.
    """
    if not content:
        return None
    if scan_error := _scan_memory_content(content):
        return GateRefusal(
            action="REFUSE",
            rationale=f"grammar/threat: {scan_error}",
            reroute_target=None,
            failure_class="scan_blocked",
            detail={"rule": "threat_scan"},
        )
    return None


def _stage6_stale_entry(entries: list[str], old_text: str) -> Optional[GateRefusal]:
    """Stage 6 — stale-entry refusal.

    Uses the store's installed ``_find_unique_match`` (whole-entry EXACT match
    takes priority) against the TARGET's entries.  If ``old_text`` does not
    uniquely select an entry (or selects nothing → stale), refuse with
    ``anchor_drift``.  Fail-closed: drift → REFUSE, never silently overwrite.
    """
    if not old_text:
        return None  # nothing to anchor against (e.g. an ``add`` action)
    idx, ambiguous = _find_unique_match(entries, old_text)
    if ambiguous:
        return GateRefusal(
            action="REFUSE",
            rationale="anchor_drift: old_text matched multiple distinct entries — be more specific",
            reroute_target=None,
            failure_class="stale_entry",
            detail={"rule": "ambiguous_match", "old_text": old_text[:200]},
        )
    if idx is None:
        return GateRefusal(
            action="REFUSE",
            rationale=f"anchor_drift: _stale_entry_message — {_stale_entry_message(old_text)}",
            reroute_target=None,
            failure_class="stale_entry",
            detail={"rule": "no_match", "old_text": old_text[:200]},
        )
    return None


# --- Public entry point ------------------------------------------------------

def _run_prewrite_stages(
    store,
    action: str,
    target: str,
    content: Optional[str],
    old_text: Optional[str],
    operations: Optional[list[dict[str, Any]]],
    caller: str,
    entries: list[str],
    is_doc: bool,
    scan_texts: list[str],
    add_contents: list[str],
) -> tuple[Optional[GateRefusal], Optional[dict[str, Any]]]:
    """Run stages 1-6 (pre-write). Returns (first_refusal, date_flag).

    Extracted from ``acceptance_pipeline`` so the pipeline's complexity stays
    under the code-health cap; the stage order and semantics are unchanged.
    """
    # --- Stage 1: char-cap soft check (85%), per-target scoped ------------
    for c in add_contents:
        if refusal := _stage1_char_cap(store, c, target):
            return refusal, None
    # Also enforce the hard cap (store limit) as the ceiling.
    for c in add_contents:
        if refusal := _stage1_char_cap(store, c, target, hard_cap=True):
            return refusal, None

    # --- Stage 1 (hard-cap arm): growing replace/patch projection (T1-7) --
    if not operations:
        if refusal := _stage1_projected_hard_cap(store, target, action, content, old_text):
            return refusal, None

    # --- Stage 2: date-currency ADVISORY (never blocks) ------------------
    # Skipped for doc targets: a new dated note written into a chain doc
    # belongs in the chain (the chain is the long tail of dated notes), so the
    # "route to session-log" advisory does not apply there (T2-2c).
    date_flag: Optional[dict[str, Any]] = None
    if not is_doc:
        for c in add_contents:
            if c and (flag := _stage2_date_currency(c)):
                date_flag = flag
                break  # one flag is enough

    # --- Stage 3: provenance lens -----------------------------------------
    for c in scan_texts:
        if c and (refusal := _stage3_provenance(caller, c)):
            refusal.detail["write_caller"] = caller
            return refusal, None

    # --- Stage 4: procedure-shape semantic check --------------------------
    for c in scan_texts:
        if c and (refusal := _stage4_procedure_shape(c)):
            return refusal, None

    # --- Stage 5: grammar / threat check ----------------------------------
    # Reuse the store's _scan_memory_content for new content only.
    # old_text is NOT scanned here — the store's own _scan_memory_content already
    # guards new writes; scanning old_text would double-gate and break existing
    # fixtures that patch entries containing previously-scanned (pre-threat) text.
    for c in scan_texts:
        if c and (refusal := _stage5_grammar_threat(c)):
            return refusal, None

    # --- Stage 6: stale-entry refusal (whole-entry exact match) -----------
    # Only for single operations (not batches — the store's _pin_matched_entries
    # handles per-op matching in the batch path).  Searched against the TARGET's
    # entries, so a doc-resident span anchors against the doc (INTEG-2).
    if not operations and old_text and entries:
        if refusal := _stage6_stale_entry(entries, old_text):
            return refusal, None

    return None, date_flag


def acceptance_pipeline(
    store,
    action: str,
    target: str,
    content: Optional[str],
    old_text: Optional[str],
    operations: Optional[list[dict[str, Any]]] = None,
    caller: str = "user",
    entries: Optional[list[str]] = None,
) -> Optional[GateRefusal]:
    """Run the 7-stage acceptance gate (stages 1-6 pre-write).

    Returns ``None`` to proceed (admit), or a :class:`GateRefusal` (refuse).
    On admit, exactly ONE Door-B ADMIT line is appended (T1-1).  On refusal,
    exactly one line naming the violating stage.

    Size / limit / stale-entry resolution is SCOPED PER TARGET through the
    store's doc-aware arms, so a chain-doc write is charged against the doc
    (uncapped) and anchored against the doc's entries (INTEG-1 / INTEG-2).
    """
    is_doc = _is_doc(store, target)
    # Resolve the TARGET's own entries when not provided (INTEG-2).
    if entries is None:
        entries = _store_entries(store, target)

    # Collect the new content strings to scan (all operations' content/new_text).
    if operations:
        scan_texts = [op.get("content") or op.get("new_text") or "" for op in operations]
        # For char-cap we check only new entries being ADDED (not replace/patch,
        # which may shrink the store).  Growing replaces are caught by the
        # projected hard-cap arm (T1-7).
        add_contents = [op.get("content") or op.get("new_text") or ""
                        for op in operations if op.get("action") == "add"]
        # For batches, Stage 6 (stale-entry) is handled by the store's own
        # _pin_matched_entries — skip it here to avoid false negatives from
        # concatenated old_text.
    else:
        scan_texts = [content or ""] if content else []
        # Char-cap only applies to ``add`` actions — replace/patch/remove may
        # shrink the store, so their content is not subject to the soft-cap
        # check (the projected hard-cap arm covers the growing case).
        add_contents = [content or ""] if (action == "add" and content) else []

    def _refuse(refusal: GateRefusal) -> GateRefusal:
        _append_change_log_line(refusal.to_log_dict() | {"target": target})
        FAILURE_CLASS.set(refusal.failure_class)
        return refusal

    refusal, date_flag = _run_prewrite_stages(
        store, action, target, content, old_text, operations, caller,
        entries, is_doc, scan_texts, add_contents,
    )
    if refusal is not None:
        return _refuse(refusal)

    # --- ADMIT: exactly one Door-B line per admitted write (T1-1) ---------
    admit = GateRefusal(
        action="ADMIT",
        rationale="accepted",
        reroute_target=None,
        failure_class="none",
        detail={"target": target, **(date_flag or {})},
    )
    _append_change_log_line(admit.to_log_dict())
    FAILURE_CLASS.set("none")  # mirror the refusal/readback paths; the drain reads this per call
    return None  # proceed to write


# --- Stage 7: post-write read-back (writer path) ----------------------------

def verify_post_write(store, target: str, action: str, old_text: Optional[str]) -> Optional[str]:
    """Stage 7 — re-read *target* after the write and assert *old_text* is gone.

    Runs in the WRITER path AFTER the store write (bounded cost: one read per
    admitted replace/patch/remove).  Returns ``None`` when the read-back
    confirms eradication, or a JSON error string (failure_class
    ``readback_failed``) when ``old_text`` is still present — the store's own
    guards are the real safety net; this is an integrity attestation that the
    promised edit actually landed (T1-2).
    """
    if action not in ("replace", "patch", "remove") or not old_text:
        return None
    try:
        entries = _store_entries(store, target)
    except Exception:  # health: allow BLE001 -- read-back failure must not mask a landed write
        _logger.warning("verify_post_write: read-back raised on %s; pass-through by design",
                         target, exc_info=True)
        return None
    if any(old_text in e for e in entries):
        FAILURE_CLASS.set("readback_failed")
        _append_change_log_line({
            "timestamp": _now_iso(),
            "action": "REFUSE",
            "rationale": "read-back failed: old_text still present after the write",
            "reroute_target": None,
            "failure_class": "readback_failed",
            "target": target,
            "detail": {"old_text": old_text[:200]},
        })
        from tools.registry import tool_error
        return tool_error(
            "Write read-back failed: the edited span is still present after the write. "
            "The store may be inconsistent; re-read the target before retrying.",
            success=False, failure_class="readback_failed", target=target,
        )
    return None
