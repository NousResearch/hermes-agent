"""``session.search``: full-text session search for the TUI (dashboard ``/api/sessions/search`` parity).

Port of ``hermes_cli.web_routers.sessions.search_sessions``' ``_search`` closure: id matches first,
FTS5 message content second, a title supplement third — all deduped by compression lineage root and
resolved to the lineage tip, so a rotated (compressed) conversation surfaces once, at its live tip.
Results are returned newest-first (``started_at`` descending): the lane order above is how hits are
collected, never how they are displayed.
"""

import re

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _is_compression_edge(child: dict, parent: dict) -> bool:
    """Copy of the web router's helper (importing it would drag fastapi into the gateway): the
    parent link is a compression continuation only when the parent ended for compression before
    the child started — branch children share ``parent_session_id`` but are real alternate
    conversations and stay separately searchable."""
    parent_ended_at = parent.get("ended_at")
    started_at = child.get("started_at")
    return (
        parent.get("end_reason") == "compression"
        and parent_ended_at is not None
        and started_at is not None
        and started_at >= parent_ended_at)


def _search(db, query: str, safe_limit: int) -> list:
    def get_session(sid):
        try:
            return db.get_session(sid)
        except Exception:  # health: allow BLE001 -- per-node row read on the lineage walk; an unreadable node (old/odd store) skips, it must not kill the whole search
            return None

    # Walk parent_session_id to the compression root, memoized per chain; stops at
    # branch/delegate edges (those stay searchable).
    root_cache: dict = {}

    def compression_root(session_id: str) -> str:
        chain, cur, root = [], session_id, session_id
        while cur and cur not in chain:  # ``not in chain`` guards parent cycles
            if cur in root_cache:
                root = root_cache[cur]
                break
            chain.append(cur)
            s = get_session(cur)
            parent = s.get("parent_session_id") if isinstance(s, dict) else None
            parent_session = get_session(parent) if parent else None
            if not parent_session or not _is_compression_edge(s, parent_session):
                root = cur
                break
            cur = parent
        for node in chain:
            root_cache[node] = root
        return root

    tip_cache: dict = {}

    def lineage_tip(session_id: str) -> str:
        # Resolve the tip from the MATCHED id, never from the lineage root: the forward
        # chain walk is defensively bounded, so a lineage deeper than the bound truncates
        # to a stale mid id when started at the root (#125041).
        if session_id not in tip_cache:
            try:
                tip_cache[session_id] = db.get_compression_tip(session_id) or session_id
            except Exception:  # health: allow BLE001 -- best-effort tip resolve: an unreadable chain falls back to the matched id instead of failing the hit
                tip_cache[session_id] = session_id
        return tip_cache[session_id]

    # One keyspace for id-hits and content-hits, keyed by lineage root; first hit wins,
    # and ID matches run first.
    seen: dict = {}

    def add_result(raw_sid: str, snippet: str, role, hit: dict) -> None:
        if not raw_sid or len(seen) >= safe_limit:
            return
        root = compression_root(raw_sid)
        if root in seen:
            return
        sid = lineage_tip(raw_sid)
        try:
            row = db.get_session_rich_row(sid) or {}
        except Exception:  # health: allow BLE001 -- best-effort tip-row enrichment; the hit-carried fields answer when the rich-row read fails
            row = {}
        seen[root] = {
            "id": row.get("id") or hit.get("id") or sid,
            "title": row.get("title") or hit.get("title") or "",
            "preview": row.get("preview") or hit.get("preview") or "",
            "started_at": row.get("started_at") or hit.get("started_at") or hit.get("session_started") or 0,
            "source": row.get("source") or hit.get("source") or "",
            "snippet": snippet, "role": role, "_lineage_root_id": root}

    # Direct ID matches first (pasted ids never appear in message text).
    for row in db.search_sessions_by_id(query, limit=safe_limit, include_archived=True):
        sid = row.get("id")
        preview = (row.get("preview") or "").strip()
        add_result(sid, preview or f"Session ID: {sid}", None, row)

    # Prefix wildcards so partial words match ("nimb" -> "nimb*"); quoted phrases and existing
    # wildcards are kept as-is. Over-fetch so lineage dedup can still surface `limit` distinct
    # conversations when several hits collapse onto one root.
    prefix_query = " ".join(
        tok if tok.startswith('"') or tok.endswith("*") else tok + "*"
        for tok in re.findall(r'"[^"]*"|\S+', query))
    for m in db.search_messages(
            query=prefix_query, limit=max(safe_limit * 5, 50),
            fields=("session_id", "role", "snippet", "source", "model", "session_started")):
        if len(seen) >= safe_limit:
            break
        add_result(m["session_id"], m.get("snippet", ""), m.get("role"), m)

    # Title matches fill any remaining slots (#66242): the FTS index only covers message
    # content, so a term that lives solely in a manually-set sessions.title would otherwise
    # return nothing. Best-effort: an old/odd store that rejects the read just skips the lane.
    if len(seen) < safe_limit:
        try:
            title_rows = db.list_sessions_rich(
                search_query=query, include_archived=True, order_by_last_active=True, limit=safe_limit)
        except Exception:  # health: allow BLE001 -- best-effort supplement lane: an old/odd store that rejects the search_query read must not fail the id+content results already collected
            logger.debug("title-match supplement skipped for %r", query[: 200])
            title_rows = []
        for row in title_rows:
            if len(seen) >= safe_limit:
                break
            sid = row.get("id")
            if not sid:
                continue
            preview = (row.get("preview") or "").strip()
            add_result(sid, preview or f"Session title matched: {query}", None, row)
    # Newest first is the row contract: lanes are collection order, not display order — without
    # this an id match pins a months-old session above yesterday's conversation. Stable sort, so
    # equal timestamps keep lane order; a missing/None/0 ``started_at`` falls back to 0, last.
    return sorted(seen.values(), key=lambda r: r.get("started_at") or 0, reverse=True)


@method("session.search")
def _session_search(rid, params: dict) -> dict:
    """Search over the profile's store; db handling mirrors ``session.list``'s ``_with_db``
    shape (profile-scoped, never session-scoped — no live session is involved)."""
    if not (query := str(params.get("query") or "").strip()):
        return _ok(rid, {"results": []})
    safe_limit = max(1, min(_int_param(params, "limit", 20), 100))
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5006)
        try:
            return _ok(rid, {"results": _search(db, query, safe_limit)})
        except Exception as e:  # health: allow BLE001 -- RPC boundary: any store failure folds into the 5006 error envelope, session.list's shape
            return _err(rid, 5006, str(e))


def register(server):
    bind_module(globals(), server)
