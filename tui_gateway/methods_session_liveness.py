"""Cross-process liveness rows for ``session.active_list`` (#85302).

Bodies are rebound onto server.py's globals at install time (method_ctx.py),
like the other split handler modules; the helpers they lean on
(``_snapshot_sessions``, ``_listing_rows``, ``_get_db``, ...) live on server.py
or in methods_session.py and are published before this module installs.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("session.active_list")
def _(rid, params: dict) -> dict:
    """Live TUI sessions in this process (not a DB browser)."""
    snapshot, err = _snapshot_sessions(rid)
    if err:
        return err
    current = str(params.get("current_session_id") or "")
    # ``_finalized`` sessions linger until the reaper pops them (they inflated the footer). Do NOT filter on
    # the WS-detached sentinel: detached is attachable until grace-reap, and ``hermes --tui`` rides stdio.
    # Keep insertion order (focused must not jump).
    rows = [_session_live_item(sid, session, current) for sid, session in snapshot if not session.get("_finalized")]
    rows.extend(_foreign_live_rows(snapshot))
    return _ok(rid, {"sessions": rows})


def _foreign_live_rows(snapshot) -> list[dict]:
    """Recently-active rows that exist only in state.db — cron runs, CLI
    one-shots, messaging turns written by OTHER processes, subagent children
    with a run in flight — never enter this gateway's ``_sessions``, so their
    stream events never reach clients. The shared SQLite file is the one thing
    they all move (#58671); report them here so clients can paint them from
    the same poll. Same 300s recency window the web sessions router uses for
    its ``is_active`` flag. Best-effort: a failed DB probe must never break the
    in-memory answer — but it is logged, not silently swallowed."""
    rows: list[dict] = []
    in_memory = {sid for sid, session in snapshot if not session.get("_finalized")}
    try:
        db = _get_db()
        if db is not None:
            for s in _listing_rows(db, 50):
                sid = str(s.get("id") or "")
                if not sid or sid in in_memory:
                    continue
                row = _foreign_row(sid, s, last_active=None)
                if row is not None:
                    rows.append(row)
        # Subagent children with a delegation run in flight but no watch
        # window: their own session row is live in the DB, and the child
        # mirror only streams into an opened window.
        for (profile_home, child_key), ts in list(_active_child_runs.items()):
            if child_key and child_key not in in_memory:
                rows.append({
                    "current": False,
                    "description": "subagent running",
                    "foreign": True,
                    "id": child_key,
                    "last_active": float(ts),
                    "message_count": 0,
                    "model": "",
                    "preview": "",
                    "provider": "",
                    "session_key": child_key,
                    "started_at": 0.0,
                    "status": "working",
                    "title": "",
                })
    except Exception:
        logger.debug("active_list foreign-row probe failed (in-memory answer kept)", exc_info=True)
    return rows


def _foreign_row(sid: str, s: dict, last_active) -> dict | None:
    """A live DB row as an ``active_list`` entry, or None when it is not live:
    no id, already ended, or outside the 300s recency window."""
    if not sid or s.get("ended_at") is not None:
        return None
    last_active = float(s.get("last_active") or s.get("started_at") or 0) if last_active is None else last_active
    if (time.time() - last_active) >= 300:
        return None
    return {
        "current": False,
        "description": str(s.get("last_activity_description") or ""),
        "foreign": True,
        "id": sid,
        "last_active": last_active,
        "message_count": int(s.get("message_count") or 0),
        "model": str(s.get("model") or ""),
        "preview": "",
        "provider": "",
        "session_key": sid,
        "started_at": float(s.get("started_at") or 0),
        "status": "working",
        "title": str(s.get("title") or s.get("display_name") or ""),
    }


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
