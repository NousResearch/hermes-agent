#!/usr/bin/env python3
"""Stage 7 — post-write read-back assertion for ``tools.memory_tool``.

This module ships the Stage-7 read-back fix (issue #140848; Enough1122 review)
surgically on main's ``MemoryStore``.  ``verify_post_write`` depends only on
primitives present on ``origin/main``:

* ``store._entries_for`` — the store's entry accessor (exists on main; resolves
  ``memory``/``user`` targets in memory).  The write-gate feature's full
  7-stage pre-write pipeline — which also touches chain-doc surfaces such as
  ``store._doc`` and the ``doc:`` keyword — lives in the write-gate feature's
  own PR and is deliberately NOT imported here.
* ``FAILURE_CLASS`` from ``tools.memory_tool_store`` (a ``ContextVar``; setting
  ``"readback_failed"`` is not validated against a frozenset on main).
* ``tool_error`` from ``tools.registry`` (lazy-imported at the call site).

``tools.memory_tool._memory_tool`` calls ``verify_post_write`` directly after an
admitted replace/patch/remove (bounded: one read per write).  If the import of
this module fails there, the caller proceeds (fail-open), matching the existing
lazy-gate fail-open precedent.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_scratch_dir
from tools.memory_tool_store import FAILURE_CLASS


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


def _audit_log_path() -> Path:
    """Path to the write-gate JSONL change-log.

    Lives under the Hermes scratch dir (``$TMPDIR``); tests redirect by
    monkeypatching ``tools.write_gate._audit_log_path``.
    """
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


def _store_entries(store, target: str) -> list[str]:
    """The TARGET's own entries via the store's doc-aware arm.

    Routes through ``store._entries_for`` so a ``doc:`` target resolves to the
    doc's entries.  Falls back to the two-way branch for a store that predates
    the doc arm.
    """
    ef = getattr(store, "_entries_for", None)
    if callable(ef):
        return list(ef(target))
    return list(store.user_entries if target == "user" else store.memory_entries)


def verify_post_write(store, target: str, action: str, old_text: Optional[str],
                      content: Optional[str] = None) -> Optional[str]:
    """Stage 7 — re-read *target* after the write and assert *old_text* is gone.

    Runs in the WRITER path AFTER the store write (bounded cost: one read per
    admitted replace/patch/remove).  Returns ``None`` when the read-back
    confirms eradication, or a JSON error string (``failure_class``
    ``readback_failed``) when ``old_text`` is still present — the store's own
    guards are the real safety net; this is an integrity attestation that the
    promised edit actually landed (T1-2).

    ``content`` is the new text the write carried (``new_text`` for patch,
    ``new_content`` for replace, ``None`` for remove).  A landed patch/replace
    may legitimately leave ``old_text`` present when the new text contains the
    old span (a span extension, or a rewrite that keeps the locating phrase) —
    that is not a failed write.  The read-back flags only when ``old_text``
    survives AND the new text does not explain it (``old_text not in content``);
    for ``remove``, ``content`` is ``None`` and the check collapses to the
    original eradication assertion.
    """
    if action not in ("replace", "patch", "remove") or not old_text:
        return None
    try:
        entries = _store_entries(store, target)
    except Exception:  # health: allow BLE001 -- read-back fails open: a log/entry error must not abort the landed store write
        return None  # a read failure here must not mask a landed write
    if any(old_text in e and old_text not in (content or "") for e in entries):
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
