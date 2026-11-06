"""``session.family``: the read-only family read behind ``/tree`` (``methods_session`` split).

The walk itself is ``hermes_state_family``; this module is only the wire. It is a split sibling
because ``methods_session.py`` is over the file-line ratchet and may not grow: the ratchet's
remedy is a topical sibling, not a bigger facade.
"""

from __future__ import annotations

import logging

from .method_ctx import HandlerRegistry, bind_module

logger = logging.getLogger(__name__)

_registry = HandlerRegistry()
method = _registry.method


@method("session.family")
def _(rid, params: dict) -> dict:
    """Read-only family around one session: the nodes the classic CLI's /tree draws.

    Nothing here writes: the walk reads rows through the store and the view flags only reduce
    what comes back. Another profile's sessions cannot appear — the store is opened for the
    request's own profile, and a session it does not hold yields an empty family.
    """
    session_id = _str_param(params, "session_id")
    if not session_id:
        return _err(rid, 4004, "session_id is required")
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5006)
        try:
            return _ok(rid, db.session_family(
                session_id,
                show_compressions=_flag(params, "show_compressions"),
                show_subagents=_flag(params, "show_subagents"),
                folded=[str(item) for item in (params.get("folded") or [])],
            ))
        except Exception:
            logger.exception("session.family failed for %s", session_id)
            return _err(rid, 5006, "could not read this session's family")


def register(server) -> None:
    """Publish this module's handler onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
