"""Branch (session.branch / session.branch_whole) helpers (``methods_session`` split).

Moved verbatim from ``tui_gateway/methods_session.py`` (file-line ratchet, #66663 CI repair):
the bodies close over server.py globals through ``method_ctx.bind_module`` exactly as before —
publication still runs from the parent's ``register()``.
"""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()

# ── session.branch ───────────────────────────────────────────────────
def _branch_title(db, parent_key: str) -> str:
    """Next title in the parent's lineage (mirrors the TUI /branch naming)."""
    current = db.get_session_title(parent_key) or "branch"
    if hasattr(db, "get_next_title_in_lineage"):
        return db.get_next_title_in_lineage(current)
    return f"{current} (branch)"


def _visible_branch_history(messages) -> list:
    """user/assistant rows with visible text, as FULL copies (reasoning + timeline-marker tags survive)."""
    return [dict(message) for message in messages or []
            if isinstance(message, dict) and message.get("role") in {"user", "assistant"}
            and _coerce_message_text(message.get("content")).strip()]


def _build_branch_agent(session: dict, new_sid: str, new_key: str, history: list, source: str):
    """Build + register the branched agent in the parent's profile; the DEDICATED db handle is ours until
    ``_transfer_db_to_agent`` (released here on failure)."""
    parent_home = session.get("profile_home")
    parent_user_id = _session_auth_user_id(session)
    branch_db, branch_owns_db = _profile_session_db(parent_home) if parent_home else (None, False)
    try:
        with _profile_build_scope(parent_home):
            agent = _make_agent_in_context(new_sid, new_key, session_db=branch_db, platform_override=source,
                                           cwd_override=_session_cwd(session),
                                           context_cwd_is_launch_artifact=_context_cwd_is_launch_artifact(session),
                                           auth_user_id=parent_user_id)
            _init_session(new_sid, new_key, agent, list(history), cols=session.get("cols", 80),
                          cwd=_session_cwd(session), session_db=branch_db, source=source, profile_home=parent_home,
                          explicit_cwd=bool(session.get("explicit_cwd")))
            _transfer_db_to_agent(agent, branch_db)
            branch_owns_db = False
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = None  # claimed lazily on the first turn
            _sessions[new_sid]["auth_user_id"] = parent_user_id
            # The parent's STORED key: the idempotent-hit reply for a retried
            # session.branch answers the same ``parent`` as the fresh path, and
            # later readers (lineage, retry) get the linkage from the runtime.
            _sessions[new_sid]["parent_session_id"] = session.get("session_key")
        return agent
    finally:
        if branch_owns_db and branch_db is not None:
            _release_db(branch_db)


_BRANCH_COPY_FIELDS = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    # Timeline markers ride as role=user; untagged they become bare user turns after a restart, corrupting
    # the truncate ordinal address space.
    "display_kind", "display_metadata",
    # Branch copies are history, not new activity: keep the parent's timestamps.
    "timestamp")

# The live branch's UNFLUSHED tail rows are appended after the durable copy with full model fidelity —
# CLI /branch parity: tool calls/results ride along, and the api_content sidecar keeps provider
# prompt-cache continuity for rows the parent never wrote.
_BRANCH_MODEL_COPY_FIELDS = _BRANCH_COPY_FIELDS + ("tool_calls", "tool_call_id", "tool_name", "api_content")


def _branch_seed_parts(db, session: dict, old_key: str) -> tuple[list, list]:
    """(display_rows, live_tail) for a live branch: the persisted DISPLAY projection reconciled with live
    memory, and the UNFLUSHED live tail past it. The child's transcript is the DURABLE rows copied
    verbatim (tools intact — see ``copy_transcript_for_branch``); only the tail needs a row-by-row append.
    Live history is the MODEL projection (post-compaction summary + tail), so the display read is what
    reconciles it against what the parent actually persisted."""
    with session["history_lock"]:
        in_memory_history = [
            dict(msg) for msg in list(session.get("display_history_prefix") or []) + list(session.get("history", []))
            if isinstance(msg, dict)]
    display_rows, live_tail = [], []
    if callable(get_resume_conversations := getattr(db, "get_resume_conversations", None)):
        try:
            _, display_history = get_resume_conversations(old_key)
            # _reconcile_display_with_live returns the DB display rows + the in-memory rows past the
            # last shared anchor — the tail slice is exactly the rows past the DB display set.
            reconciled = _reconcile_display_with_live(display_history, in_memory_history)
            display_rows, live_tail = reconciled[:len(display_history)], reconciled[len(display_history):]
        except Exception:
            logger.debug("branch display projection read failed", exc_info=True)
    if not display_rows:
        display_rows = list(in_memory_history)
    return display_rows, live_tail


def _branch_boundary_row_id(display_rows: list, count) -> int | None:
    """Row id bounding a ``count`` branch: the Nth VISIBLE (user/assistant with text) DB row, so the copy
    keeps interleaved tool rows between here and the boundary. None (copy everything) when *count* is not
    a positive int or the durable rows hold fewer than *count* visible rows."""
    if not isinstance(count, int) or count <= 0:
        return None
    seen = 0
    for msg in display_rows:
        if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
            continue
        if not _coerce_message_text(msg.get("content")).strip():
            continue
        seen += 1
        if seen >= count:
            return msg.get("_row_id")
    return None


def _branch_live(rid, params: dict, session: dict, *, omit_messages: bool = False) -> dict:
    # Idempotency (#65410, same registry as session.create): a client retrying a
    # branch whose first response was lost gets the SAME child, not a duplicate.
    idem_key = _str_param(params, "idempotency_key") or None
    if idem_key is not None:
        with _sessions_lock:
            now_gc = time.time()
            existing_sid, ts = _idempotency_keys.get(idem_key, (None, 0.0))
            if existing_sid is not None and existing_sid in _sessions:
                if now_gc - ts <= _IDEMPOTENCY_KEY_TTL:
                    # Refresh the TTL so back-to-back retries don't age out mid-flight.
                    _idempotency_keys[idem_key] = (existing_sid, now_gc)
                    return _ok(rid, _branch_idempotent_hit(existing_sid, _sessions[existing_sid], omit_messages))
                _idempotency_keys.pop(idem_key, None)
            # Stale key (child closed) or first attempt: fall through to a fresh
            # branch, which re-registers the key below.
    # Write into the parent's profile-scoped state.db; the launch handle would orphan rows.
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        display_rows, live_tail = _branch_seed_parts(db, session, old_key)
        if not display_rows and not live_tail:
            return _err(rid, 4008, "nothing to branch — send a message first")
        # Wire payload stays the VISIBLE prose view (client contract); the durable copy below is what
        # the child's model context becomes.
        paint_rows = _visible_branch_history(display_rows)
        boundary = _branch_boundary_row_id(display_rows, params.get("count"))
        new_key, new_sid, source = _new_session_key(), uuid.uuid4().hex[:8], _session_source(session)
        # Durable-verbatim seed when the store supports it (the fix for narration-only branch children);
        # the prose rebuild remains the fallback for stores without the copy API.
        copy_api = hasattr(db, "copy_transcript_for_branch")
        seed_rows = live_tail if copy_api else paint_rows
        try:
            title = params.get("name", "") or _branch_title(db, old_key)
            home = session.get("profile_home")
            _persist_branch(db, new_key, old_key, title, seed_rows, source=source,
                            cwd=None if _is_remote_launch_cwd(session) else _session_cwd(session),
                            profile_name=profile_name_for_home(home) or _current_profile_name(),
                            model=_session_default_route(session)[0],
                            copy_fields=_BRANCH_MODEL_COPY_FIELDS if copy_api else _BRANCH_COPY_FIELDS,
                            title_source="user" if params.get("name") else "derived",
                            user_id=_session_auth_user_id(session),
                            transcript_source=old_key if copy_api else None, up_to_row_id=boundary)
            # The child's live model context is its OWN rows (fresh row ids, born-durable stamps) — never
            # the parent's projection dicts, whose _row_id values address PARENT rows.
            history = db.get_resume_conversations(new_key)[0] if copy_api else seed_rows
        except Exception as e:
            return _err(rid, 5008, f"branch failed: {e}")
    try:
        agent = _build_branch_agent(session, new_sid, new_key, history, source)
    except Exception as e:
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    if idem_key is not None:
        with _sessions_lock:
            _idempotency_keys[idem_key] = (new_sid, time.time())
            # The fresh reply's title, so an idempotent hit answers the SAME one
            # without a DB round-trip.
            if new_sid in _sessions:
                _sessions[new_sid]["branch_title"] = title
    response = {"session_id": new_sid, "stored_session_id": new_key, "title": title, "parent": old_key,
                "message_count": len(paint_rows), "info": _session_info(agent, _sessions.get(new_sid))}
    if omit_messages:
        response["messages_omitted"] = True
    else:
        response["messages"] = _history_to_messages(paint_rows, profile_home=session.get("profile_home"))
    return _ok(rid, response)


def _branch_idempotent_hit(existing_sid: str, session: dict, omit_messages: bool) -> dict:
    """The SAME result shape a fresh ``_branch_live`` returns for the existing child. ``history`` is the
    child's MODEL context (tool rows included since the durable-copy fix); the wire stays the VISIBLE
    prose view, so the retry paints exactly what the fresh reply painted."""
    history = session.get("history") or []
    key = session.get("session_key") or ""
    # Same wire shape as the fresh reply: the VISIBLE count/payload, not the model projection's row
    # count (tool rows ride in history but are not paint rows).
    paint_rows = _visible_branch_history(list(session.get("display_history_prefix") or []) + history)
    response = {"session_id": existing_sid, "stored_session_id": key,
                "title": session.get("branch_title") or _branch_title_for(session),
                "parent": session.get("parent_session_id"), "message_count": len(paint_rows),
                "info": _fallback_session_info(session)}
    if omit_messages:
        response["messages_omitted"] = True
    else:
        response["messages"] = _history_to_messages(paint_rows, profile_home=session.get("profile_home"))
    return response


def _branch_title_for(session: dict) -> str:
    """The child's persisted title from its stored row, best-effort."""
    with contextlib.suppress(Exception):
        with _session_db(session) as db:
            if db is not None:
                return db.get_session_title(session.get("session_key") or "") or ""
    return ""


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals)."""
    bind_module(globals(), server, skip=("_",))
