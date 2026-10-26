"""Archived-hit reporting for ``session_search`` (``include_inactive=true``).

Compaction and prompt-edit truncation soft-archive the turns they drop
(``active=0``), and a user rewind marks rows the same way. Those rows stay in the
FTS5 index but are hidden from discovery by default, so when a caller explicitly
asks for them the discovery payload has to say *which* hits were archived — the
per-lineage dedupe ranks by relevance and almost never puts an archived row in the
displayed anchor slot.

Kept out of ``session_search_tool.py`` deliberately: the tool's discovery entry is
already over its code-health complexity target, and this is a separable concern.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Optional


def _archived_row_ids(
    state_loader: Callable[[], Any],
    log: logging.Logger,
) -> dict[int, tuple[int, int]]:
    """``{message_id: (active, compacted)}`` for the probed ids, in ONE query.

    *state_loader* runs the SELECT (the caller owns the DB lock). A broken query
    degrades to "no archived rows" — discovery must still answer without the
    annotation — and logs the traceback so the failure is not silent.
    """
    try:
        rows = state_loader()
    except sqlite3.Error:
        log.exception("include_inactive: batch storage-state probe failed")
        return {}
    return {int(row["id"]): (int(row["active"]), int(row["compacted"]))
            for row in (rows or []) if row["id"] is not None}


def attach_archived_hits(
    db: Any,
    results: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    *,
    scan_limit: int = 300,
    id_limit: int = 10,
    log: Optional[logging.Logger] = None,
) -> None:
    """Annotate each discovery *results* entry with the ids of matching archived rows.

    An archived row is ``active=0`` under either archive flavor: compacted away
    (``compacted=1``) or truncated/rewound (``compacted=0``). Callers expand an id
    with the scroll shape (``session_id`` + ``around_message_id``), which does read
    archived rows.
    """
    from tools.session_search_tool import _quiet, _resolve_lineage  # lazy: avoids an import cycle

    logger = log or logging.getLogger(__name__)
    probes = [r for r in raw_results[:scan_limit] if r.get("id") is not None]
    if not probes or not results:
        return
    probe_ids = [int(r["id"]) for r in probes]

    def _load(_ids: list[int] = probe_ids):
        marks = ",".join("?" * len(_ids))
        with db._lock:
            return db._conn.execute(
                f"SELECT id, active, compacted FROM messages WHERE id IN ({marks})", _ids).fetchall()

    states = _archived_row_ids(lambda: _quiet(_load, [], "archived-hit probe failed"), logger)
    entry_for_session = {e["session_id"]: e for e in results if e.get("session_id")}
    lineage_cache: dict[str, str] = {}

    def _owner(raw_sid: str) -> Optional[dict[str, Any]]:
        entry = entry_for_session.get(raw_sid)
        if entry is not None:
            return entry
        root = lineage_cache.setdefault(raw_sid, _resolve_lineage(db, raw_sid) or raw_sid)
        return entry_for_session.get(root)

    for row in probes:
        entry = _owner(row.get("session_id") or "")
        if entry is None or states.get(int(row["id"]), (1, 0))[0]:
            continue
        entry.setdefault("archived_hits", []).append(int(row["id"]))

    for entry in results:
        hits = entry.get("archived_hits")
        if not hits:
            continue
        deduped: set[int] = set()
        entry["archived_hits"] = [h for h in hits if not (h in deduped or deduped.add(h))][:id_limit]
        entry["archived_hits_note"] = (
            "Rows behind these ids are archived OUT of live context "
            "(compacted away, or undone/regenerated) and were only searched "
            "because include_inactive=true. Expand with session_id + "
            "around_message_id; treat undone/regenerated turns as superseded, "
            "not current.")
