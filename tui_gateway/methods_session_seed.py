"""Seeded-session row persistence (``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet offset):
the bodies close over server.py globals through ``method_ctx.bind_module`` exactly
as before - publication runs from server.py's import-time module registration.
"""

from .method_ctx import bind_module


def _seed_branch_row(record: dict, key: str, parent_session_id: str, history: list, source: str, profile_home):
    """Persist a seeded desktop branch child NOW (the one session.create exception to lazy rows): the
    renderer's post-create resume re-fetches it via REST/defer_history, so an unpersisted child 404s and
    the fail-latch spins forever. Best-effort — on failure the lazy first-prompt path is the fallback."""
    try:
        with _session_db(record) as db:
            if db is None:
                return
            _persist_branch(db, key, parent_session_id, _branch_title(db, parent_session_id), history,
                            source=source, cwd=None if _is_remote_launch_cwd(record) else record["cwd"],
                            profile_name=profile_name_for_home(profile_home) or _current_profile_name(),
                            model=_session_default_route(record)[0], compensate=True, title_source="derived", user_id=_session_auth_user_id(record))
            record["pending_title"] = None
            # The first submit's _persist_branch_seed is the fallback for a failed seed, not a second copy.
            record["_branch_seed_persisted"] = True
            record.pop("_db_row_pending", None)  # row committed: transcript reads go to the DB
    except Exception:
        logger.warning("seeded-branch persistence failed for %s; falling back to lazy row creation", key,
                       exc_info=True)


def _seed_row(record: dict) -> None:
    """Persist a parentless seeded session NOW, for the reason ``_seed_branch_row`` gives: seeded content is
    intent, not an abandoned draft, and the renderer's post-create hydration reads the DB. The client's title
    lands with the row so a restart before the first prompt keeps it. Best-effort — the first-prompt path is
    the fallback, and it re-copies the WHOLE seed, so a partial copy is rolled back here (the compensation
    ``_persist_branch`` applies to branch children) rather than left to be duplicated."""
    key = record.get("session_key")
    try:
        if _ensure_session_db_row(record) is False:
            return
        _persist_branch_seed(record)
    except Exception:
        logger.warning("seeded-session persistence failed for %s; falling back to lazy row creation", key, exc_info=True)
    if not record.get("_branch_seed_persisted"):
        with contextlib.suppress(Exception), _session_db(record) as db:
            if db is not None:
                db.discard_unadmitted_session(key)
        record["_db_row_pending"] = True  # compensation delete: draft again until the first prompt
        return
    try:
        if title := record.get("pending_title"):
            with _session_db(record) as db:
                if db is not None and db.set_session_title(key, title):
                    record["pending_title"] = None
    except Exception:
        logger.debug("seeded-session title write failed for %s; pending_title stays queued", key, exc_info=True)


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
