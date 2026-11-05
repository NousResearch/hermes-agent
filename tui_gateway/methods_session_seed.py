"""Seeded session rows: the branch child a session.create seeds at once, and its compensation.

Bodies are rebound onto server.py's globals at install time (``method_ctx.bind_module``), so they use server
helpers bare; the parent's ``register()`` installs this module.
"""

import contextlib

from .method_ctx import bind_module


def _persist_branch(db, new_key: str, parent_key: str, title: str, history: list, *, source, cwd, profile_name,
                    model: str, copy_fields=(), compensate: bool = False, title_source: str = "user",
                    user_id: str | None = None) -> None:
    """Branch child row + parent transcript (bounded-chunk transactions) + title. ``_branched_from`` keeps the
    row visible in list_sessions_rich() (the live parent never matches the legacy end_reason='branched'
    heuristic); NULL ``profile_name`` rows drop out of profile-keyed sidebar matching / deep links. ``compensate``
    deletes a committed row whose transcript/title failed (a durable-but-empty row would defeat the INSERT OR
    IGNORE first-prompt seed) — except on disk-full, where the delete cannot land. ``user_id`` is the creating
    login: the child is a Desktop session too, and the row only records identity at insert."""
    from agent.message_metadata import message_identity
    from agent.transcript_repair import sync_flushed_message_markers
    from persistence.conversation.sessions import NewSession, TitleSource

    with _persistence_for(db) as persistence:
        # The child sends the parent's exact system prompt: a row without one makes the branch's first
        # turn rebuild (re-probing the workspace) and forfeits the warm cache the copied transcript buys.
        parent_prompt = None
        try:
            if (parent := persistence.sessions.get(parent_key)) is not None:
                parent_prompt = parent.system_prompt
        except Exception:
            logger.debug("branch: parent system prompt read failed for %s", parent_key, exc_info=True)
        persistence.sessions.create(NewSession(
            session_id=new_key, source=source, model=model, model_config={"_branched_from": parent_key},
            parent_session_id=parent_key, cwd=cwd, profile_name=profile_name, user_id=user_id,
            system_prompt=parent_prompt or None))
        try:
            # Compensation guard (#93959 review): if the transcript copy or title write fails AFTER the row
            # committed, the durable-but-empty row would defeat the lazy first-prompt fallback
            # (_ensure_session_db_row is INSERT OR IGNORE — the row exists, so the seed never lands and the
            # renderer fail-latches on a "transcript-less" session again). Roll back just this child so the seed
            # path can retry cleanly on first submit.
            # Copy the whole parent history in bounded-chunk transactions — a branch seed can be hundreds of
            # rows, and per-row transactions were the write-amplification pattern removed in #23254.
            rows = [{"role": msg.get("role", "user"), "content": msg.get("content"),
                     **{field: msg.get(field) for field in copy_fields}, **message_identity(msg)} for msg in history]
            batch = persistence.messages.append_batch(new_key, [_new_message(row) for row in rows], chunk_rows=500)
            for row, outcome in zip(rows, batch.outcomes):  # the row state the write gave each copied row
                row.update({k: v for k, v in outcome.as_row_annotation().items() if v is not None})
            if title_source == "user":
                persistence.sessions.set_title(new_key, title)
            else:
                persistence.sessions.set_auto_title(new_key, title, source=TitleSource(title_source))
            sync_flushed_message_markers(history, rows)
        except Exception as exc:
            from storage_errors import is_disk_full_error
            if compensate and not is_disk_full_error(exc):
                try:
                    _discard_unadmitted(persistence, new_key)
                except Exception:
                    logger.debug("branch seed compensation delete failed for %s", new_key, exc_info=True)
            raise


def _discard_unadmitted(persistence, key: str) -> None:
    """Remove a row this flow just created and nobody was admitted against, so it can be lazily recreated."""
    from gateway.session_ingress_media import native_inputs_root
    from session_retention import after_sessions_removed
    removed = persistence.sessions.discard_unadmitted(key, media_root=str(native_inputs_root()))
    # An admitted row took the fenced delete, which retired the id: its post-commit work follows.
    if removed and persistence.runtime.retired_session(key):
        after_sessions_removed(persistence, persistence.scope.profile_dir, removed, reason="delete", primary=key)


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
        with contextlib.suppress(Exception), _session_db(record) as db, _persistence_for(db) as persistence:
            if persistence is not None:
                _discard_unadmitted(persistence, key)
        return
    try:
        if title := record.get("pending_title"):
            with _session_db(record) as db, _persistence_for(db) as persistence:
                if persistence is not None and persistence.sessions.set_title(key, title):
                    record["pending_title"] = None
    except Exception:
        logger.debug("seeded-session title write failed for %s; pending_title stays queued", key, exc_info=True)


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
