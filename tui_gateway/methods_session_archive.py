"""``session.archive`` RPC handler: the soft-hide flag, gated for lineage cascades (#70185).

Split out of methods_session.py (god-file cap): the handler keeps the module's contract —
a live runtime id first (unpersisted drafts defer via ``pending_archived``), then a stored
id/key in the profile db — and adds the cascade confirmation gate: archiving one session
flips the whole compression lineage, so a multi-row cascade is refused with error 4033 +
the preview payload unless the caller passes ``confirm_cascade``. Unarchive and
single-session archives pass through unchanged.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
names = _registry.names


@method("session.archive")
def _(rid, params: dict) -> dict:
    """Set/clear ``archived`` (out of the default list, messages kept — the Desktop PATCH parity flag)
    on a session + lineage: LIVE runtime id first (unpersisted drafts via ``pending_archived``),
    then a stored id/key in the profile db, like ``session.set_hidden``."""
    archived = is_truthy_value(params.get("archived", True))
    confirm_cascade = is_truthy_value(params.get("confirm_cascade", False))
    target = str(params.get("session_id") or params.get("session_key") or "")
    if not target:
        return _err(rid, 4006, "session_id required")
    # Quiet live lookup, the set_hidden reasoning: a stored id that is not in memory is this method's
    # expected second tier, not a rejection (session.list rows archive without a live runtime here).
    session = _sessions.get(target)
    with (_profile_db(params, writer=True) if session is None else _session_db(session)) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if session is not None:
                key = session["session_key"]
                if not db.set_session_archived(key, archived, trigger="desktop_rpc"):
                    session["pending_archived"] = archived  # no row yet: _ensure_session_db_row applies it
            else:
                # ``resolve_session_id`` follows key/title aliases like the REST pin/archive path.
                if not (key := db.resolve_session_id(target) if hasattr(db, "resolve_session_id") else target):
                    return _err(rid, 4001, "session not found")
                # Archive cascades through the whole compression lineage (#70185): surface the
                # blast radius and demand an explicit confirm instead of hiding N rows silently.
                if archived and not confirm_cascade:
                    preview = db.preview_session_archive_lineage(key, archived=True)
                    if preview["cascade_count"] > 1:
                        return _err(
                            rid, 4033,
                            "archiving this session also archives its compression lineage "
                            f"({preview['cascade_count']} sessions); pass confirm_cascade=true",
                            data={"preview": preview},
                        )
                db.set_session_archived(key, archived, trigger="desktop_rpc")
            return _ok(rid, {"archived": archived, "session_key": key})
        except Exception as e:  # health: allow BLE001 -- RPC boundary: any db error is the 5007 envelope
            return _err(rid, 5007, str(e))


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_", "names"))
