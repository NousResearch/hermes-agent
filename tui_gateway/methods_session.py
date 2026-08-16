"""Session / delegation / spawn-tree / billing / pet JSON-RPC handlers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("session.create")
def _(rid, params: dict) -> dict:
    sid = uuid.uuid4().hex[:8]
    key = _new_session_key()
    cols = int(params.get("cols", 80))
    history = _coerce_seed_history(params.get("messages"))
    title = str(params.get("title") or "").strip()
    # When set, this is a branch: the new chat copies an existing conversation's
    # history and links back to it so list_sessions_rich keeps it visible and the
    # sidebar can nest it under its parent. Mirrors the TUI /branch marker.
    parent_session_id = str(params.get("parent_session_id") or "").strip() or None
    # Did the client pick a workspace, or are we falling back to the gateway's
    # launch directory? Only an explicit choice is persisted as the session's
    # workspace (see _ensure_session_db_row); otherwise it lands in "No
    # workspace" instead of whatever folder the desktop launched in.
    raw_cwd = str(params.get("cwd") or "").strip()
    try:
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    except Exception:
        explicit_cwd = False
    resolved_cwd = _completion_cwd(params)
    source = _resolve_session_source(str(params.get("source") or "").strip() or None)
    _enable_gateway_prompts()

    # ``profile`` (app-global remote mode): a new chat started under a non-launch
    # profile must build its agent + persist against THAT profile's home/state.db,
    # not the dashboard's launch profile. Stored on the session so _start_agent_build
    # and each turn re-bind HERMES_HOME. None/own profile → launch (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)

    # The desktop composer owns its model/effort/fast as plain UI state and ships
    # it on every session.create. Honor each as a PER-SESSION override (built into
    # the agent below) — never a global config write, so picking a model/effort
    # for a new chat can't mutate the profile default. provider is optional
    # (resolved at build).
    create_model = str(params.get("model") or "").strip()
    session_model_override = (
        {"model": create_model, "provider": str(params.get("provider") or "").strip() or None}
        if create_model
        else None
    )
    create_reasoning_override = None
    if effort := str(params.get("reasoning_effort") or "").strip():
        try:
            from hermes_constants import parse_reasoning_effort

            create_reasoning_override = parse_reasoning_effort(effort)
        except Exception:
            create_reasoning_override = None
    # Presence is part of the contract: omitted means inherit the profile,
    # true pins priority, and false pins normal. Empty string is the internal
    # explicit-normal sentinel because _make_agent uses None for inheritance.
    create_service_tier_override = None
    if "fast" in params:
        create_service_tier_override = (
            "priority" if is_truthy_value(params.get("fast")) else ""
        )

    ready = threading.Event()
    now = time.time()
    lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)

    with _sessions_lock:
        _sessions[sid] = {
            "agent": None,
            "agent_error": None,
            "agent_ready": ready,
            "attached_images": [],
            "close_on_disconnect": is_truthy_value(params.get("close_on_disconnect", False)),
            "active_session_lease": lease,
            "cols": cols,
            "created_at": now,
            "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "image_counter": 0,
            "cwd": resolved_cwd,
            "inflight_turn": None,
            "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id,
            "pending_title": title or None,
            "pending_hidden": is_truthy_value(params.get("hidden", False)),
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False,
            "session_key": key,
            "show_reasoning": _load_show_reasoning(),
            "source": source,
            "slash_worker": None,
            "tool_progress_mode": _load_tool_progress_mode(),
            "tool_started_at": {},
            "transport": current_transport() or _stdio_transport,
        }
        _register_session_cwd(_sessions[sid])

    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop
    # launch (and every "New agent" / draft) opens a session here just to paint
    # the composer, so eagerly creating a row left an "Untitled" empty session
    # behind for every launch the user never typed into. The row is now created
    # lazily on the first prompt (see _ensure_session_db_row + prompt.submit),
    # and the AIAgent's own INSERT-OR-IGNORE persists it on the first turn too.

    # Return the lightweight session immediately so Ink can paint the composer
    # + skeleton panel, then build the real AIAgent just after this response is
    # flushed.  This keeps startup responsive while still hydrating tools/skills
    # without requiring the user to submit a first prompt.
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap

    return _ok(
        rid,
        {
            "session_id": sid,
            "stored_session_id": key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": {
                # Reflect the per-session model override (desktop composer pick)
                # in the immediate response so the client doesn't briefly clobber
                # its sticky pick with the global default before the deferred
                # build's session.info lands.
                "model": (
                    session_model_override.get("model")
                    if session_model_override
                    else _resolve_model()
                ),
                **(
                    {"provider": session_model_override["provider"]}
                    if session_model_override and session_model_override.get("provider")
                    else {}
                ),
                "tools": {},
                "skills": {},
                "cwd": _sessions[sid]["cwd"],
                "branch": _git_branch_for_cwd(_sessions[sid]["cwd"]),
                "project": _project_info_for_cwd(_sessions[sid]["cwd"]),
                "lazy": True,
                "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                "profile_name": _response_profile_name(profile),
            },
        },
    )


@method("session.list")
def _(rid, params: dict) -> dict:
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5006)
        try:
            # Resume picker should surface human conversation sessions from every
            # user-facing surface — CLI, TUI, all gateway platforms (including new
            # ones not enumerated here), ACP adapter clients, webhook sessions,
            # custom `HERMES_SESSION_SOURCE` values, and older installs with
            # different source labels. We deny-list only the noisy internal
            # sources (``tool`` sub-agent runs and ``kanban`` dispatcher
            # workers) rather than allow-listing a fixed set of platform names
            # that goes stale whenever a new platform is added or a user names
            # their own source.
            deny = frozenset({"kanban", "tool"})

            limit = int(params.get("limit", 200) or 200)
            # Over-fetch modestly so per-source filtering doesn't leave us
            # short; the compression-tip projection in ``list_sessions_rich``
            # can also merge rows.
            fetch_limit = max(limit * 2, 200)
            rows = [
                s
                for s in db.list_sessions_rich(
                    source=None,
                    limit=fetch_limit,
                    order_by_last_active=True,
                    compact_rows=True,
                )
                if (s.get("source") or "").strip().lower() not in deny
            ][:limit]
            return _ok(
                rid,
                {
                    "sessions": [
                        {
                            "id": s["id"],
                            "title": s.get("title") or "",
                            "preview": s.get("preview") or "",
                            "started_at": s.get("started_at") or 0,
                            "message_count": s.get("message_count") or 0,
                            "source": s.get("source") or "",
                        }
                        for s in rows
                    ]
                },
            )
        except Exception as e:
            return _err(rid, 5006, str(e))


@method("session.most_recent")
def _(rid, params: dict) -> dict:
    """Return the most recent human-facing session id, or ``None``.

    Mirrors ``session.list``'s deny-list behaviour (drops ``tool``
    sub-agent rows and ``kanban`` worker rows).  Used by TUI auto-resume when
    ``display.tui_auto_resume_recent`` is on; the field is also handy
    for any CLI tooling that wants "latest session" without paginating
    the full list.

    Contract: a ``{"session_id": null}`` result means "no eligible
    session found right now".  Errors are also folded into that
    null-result shape (and logged) so callers don't have to special-
    case JSON-RPC error envelopes for what is a normal "no answer".

    Honors ``params.profile`` so app-global remote mode lists from the
    focused profile's ``state.db`` (mirrors ``session.resume``).
    """
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"session_id": None})
        try:
            deny = frozenset({"kanban", "tool"})
            # Over-fetch by a generous bounded amount so heavy sub-agent
            # users (lots of recent ``tool`` rows) don't get a false
            # "no eligible session" answer.  ``session.list`` uses a
            # similar over-fetch strategy.
            rows = db.list_sessions_rich(
                source=None, limit=200, order_by_last_active=True, compact_rows=True
            )
            for row in rows:
                src = (row.get("source") or "").strip().lower()
                if src in deny:
                    continue
                return _ok(
                    rid,
                    {
                        "session_id": row.get("id"),
                        "title": row.get("title") or "",
                        "started_at": row.get("started_at") or 0,
                        "source": row.get("source") or "",
                    },
                )
            return _ok(rid, {"session_id": None})
        except Exception:
            logger.exception("session.most_recent failed")
            return _ok(rid, {"session_id": None})


@method("project.facts")
def _(rid, params: dict) -> dict:
    """Structured project facts for a cwd — manifests, package manager, the
    exact verify commands, and context files.

    The same detection the coding-context posture (#43316) bakes into the system
    prompt, exposed so UIs (the desktop verify surface) consume it instead of
    re-sniffing. ``{"facts": null}`` means the cwd isn't a code workspace.
    """
    try:
        from agent.coding_context import project_facts_for

        return _ok(rid, {"facts": project_facts_for(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return _ok(rid, {"facts": None})


@method("verification.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Best known coding verification evidence for a cwd/session.

    Read-only consumer of the core ledger. It never runs checks and never
    upgrades targeted evidence into a repository-wide guarantee.
    """
    try:
        from agent.verification_evidence import verification_status

        return _ok(
            rid,
            {
                "verification": verification_status(
                    session_id=params.get("session_id") or params.get("session_key"),
                    cwd=params.get("cwd"),
                )
            },
        )
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


@method("session.resume")
def _(rid, params: dict) -> dict:
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    try:
        cols = int(params.get("cols", 80))
    except (TypeError, ValueError):
        cols = 80
    # ``profile`` (app-global remote mode): resume a session that lives in another
    # local profile's state.db. None/own profile → the launch profile (unchanged).
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)
    defer_history = is_truthy_value(params.get("defer_history", False))
    # Desktop hydrates persisted transcripts through the authenticated REST
    # route in parallel. Suppress the duplicate WebSocket transcript only when
    # the caller explicitly requests it; other clients keep upstream behavior.
    omit_messages = is_truthy_value(params.get("omit_messages", False))

    # In a profile scope this opens a DEDICATED handle we own until the agent
    # takes it (see the ownership transfer at _init_session below); every path
    # that returns before that transfer must close it. Otherwise reuse the
    # shared launch db, which outlives the RPC and is never closed here.
    owns_db = False
    if profile_home is not None:
        from hermes_state import SessionDB

        db = SessionDB(db_path=profile_home / "state.db")
        owns_db = True
    else:
        db = _get_db()
    try:
        if db is None:
            return _db_unavailable_error(rid, code=5000)

        found = db.get_session(target)
        if not found:
            found = db.get_session_by_title(target)
            if found:
                target = found["id"]
            elif is_truthy_value(params.get("lazy", False)) and _child_run_active(target):
                # Race: a watch window opened on a freshly-spawned subagent. The
                # child relays `subagent.start` (which carries child_session_id and
                # triggers the window) BEFORE its first run_conversation() flushes
                # the DB row via _ensure_db_session, so db.get_session(target) is
                # momentarily empty. On slower hosts (notably WSL2, where SQLite +
                # process scheduling widen the gap) the window's resume consistently
                # lands inside this window and used to hard-fail "session not found"
                # — the frontend then 404'd on the REST messages fallback and the
                # window spun forever. The child is provably live (_child_run_active),
                # so proceed into the lazy branch with empty history; the live mirror
                # streams the whole turn anyway and the row exists by upgrade time.
                found = {}
            else:
                return _err(rid, 4007, "session not found")

        # Follow the compression-continuation chain to the live tip so a resume on
        # a rotated-out parent id binds to the descendant that actually holds the
        # post-compression turns. Auto-compression ends the session and forks a
        # continuation child; without this, resuming the original id (the desktop's
        # routed id when the chat was opened before it rotated) reloads the parent
        # transcript and the response generated after compression is missing — the
        # "I came back and the reply isn't there" bug on large sessions. Resolving
        # here also re-anchors the fast path below so a still-live rotated session
        # is reused (by its new key) instead of rebuilding a duplicate agent on the
        # stale parent. Skipped for lazy watch windows, which intentionally attach
        # to the exact child branch they were opened on.
        if found and not is_truthy_value(params.get("lazy", False)):
            try:
                tip = db.resolve_resume_session_id(target)
            except Exception:
                tip = target
            if tip and tip != target:
                target = tip
                found = db.get_session(target) or found

        # Every interactive resume path materializes the model history, even when
        # omit_messages suppresses the response copy. Count the complete lineage
        # before any reopen/history read so a runaway transcript cannot exhaust
        # the dashboard. The metadata fallback keeps lightweight test/adaptor DBs
        # that predate the shared SessionDB guard compatible. The limit resolves
        # from config (sessions.max_resume_messages, 0 disables).
        from hermes_state import (
            SessionResumeTooLargeError,
            resolved_max_resume_messages,
        )

        safety_check = getattr(db, "assert_resume_safe", None)
        try:
            if callable(safety_check):
                safety_check(target)
            else:
                resume_limit = resolved_max_resume_messages()
                stored_message_count = int(found.get("message_count") or 0)
                if resume_limit and stored_message_count > resume_limit:
                    raise SessionResumeTooLargeError(stored_message_count, resume_limit)
        except SessionResumeTooLargeError as exc:
            return _err(rid, 4130, str(exc))
        except Exception as exc:
            # Fail OPEN: a transient guard failure (locked DB, schema skew on
            # an adaptor store) must not turn the safety check into a new way
            # to lose access to a session. Only a genuine over-limit blocks.
            logger.warning(
                "resume safety check failed for %s (proceeding without guard): %s",
                target, exc,
            )

        profile_resume_cwd = str(found.get("cwd") or "").strip() or _profile_configured_cwd(
            profile_home
        )

        def _reuse_live_payload(sid: str, session: dict) -> dict:
            payload = _live_session_payload(
                sid,
                session,
                cols=cols,
                touch=True,
                transport=current_transport() or _stdio_transport,
                omit_messages=omit_messages,
            )
            payload["resumed"] = target
            if defer_history:
                payload["messages"] = []
                payload["message_count"] = int(
                    session.get("resume_message_count") or payload["message_count"]
                )
                payload["hydrating"] = bool(session.get("resume_hydrating"))
            # A lazy watch session never owns a run loop, so its payload's running
            # flag is always False — overlay the child-run registry so a reconnecting
            # watch window keeps its busy indicator while the child is still mid-run.
            if session.get("agent") is None and _child_run_active(target):
                payload["running"] = True
                payload["status"] = "streaming"
            return payload

        # Fast path: if the session is already live, reuse it under the lock.
        with _session_resume_lock:
            live = _find_live_session_by_key(target)
            if live is not None:
                return _ok(rid, _reuse_live_payload(*live))

        # Lazy/watch resume: register the live session WITHOUT building an agent.
        # Used by the desktop's subagent windows — the child runs inside the
        # parent's turn, so its window only needs the stored history plus a
        # transport for the child-mirror's live events. Skipping _make_agent here
        # is what keeps the window cheap while the backend is busy running the
        # delegation. A later prompt.submit upgrades it via _start_agent_build
        # (resume_session_id keeps the upgrade on the stored conversation).
        if is_truthy_value(params.get("lazy", False)):
            sid = uuid.uuid4().hex[:8]
            source = _resolve_session_source(str(params.get("source") or "").strip() or None)
            lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
            try:
                db.reopen_session(target)
                # The child's OWN conversation only — include_ancestors would prepend
                # the parent's transcript onto the subagent's branch.
                # repair_alternation: this resume feeds LIVE REPLAY (the loaded
                # history becomes the resumed session record's working conversation),
                # so heal a durable ``user;user`` violation once here instead of
                # re-firing the pre-request repair on every subsequent turn.
                history = db.get_messages_as_conversation(
                    target, repair_alternation=True, include_row_ids=True
                )
            except Exception as e:
                if lease is not None:
                    lease.release()
                return _err(rid, 5000, f"resume failed: {e}")
            cwd = profile_resume_cwd or _default_session_cwd()
            record = _deferred_session_record(
                target,
                cols=cols,
                cwd=cwd,
                history=history,
                lease=lease,
                source=source,
                close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
                profile_home=profile_home,
                lazy=True,
            )
            if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
                return _ok(rid, _reuse_live_payload(*live))
            # A delegated child mid-run emits no session events of its own — report
            # its liveness from the relay registry so the window shows a busy turn.
            child_running = _child_run_active(target)
            # User-visible messages use the VERBATIM display projection (child-only,
            # no ancestors — matching the repaired read above), so model-invisible
            # rows persisted by #65919 (verification candidates collapsed by
            # repair_message_sequence) survive in the watch window just as they do
            # on the eager resume + REST paths. The repaired ``history`` above still
            # feeds live replay. Fall back to it if the display read fails.
            try:
                display_history = db.get_messages_as_conversation(
                    target, repair_alternation=False, include_row_ids=True
                )
            except Exception:
                logger.debug("child-watch display projection read failed", exc_info=True)
                display_history = history
            messages = [] if omit_messages else _history_to_messages(display_history)
            return _ok(
                rid,
                {
                    "session_id": sid,
                    "resumed": target,
                    "message_count": len(display_history) if omit_messages else len(messages),
                    "messages": messages,
                    "messages_omitted": omit_messages,
                    "info": _lazy_resume_info(cwd, profile=profile),
                    "inflight": None,
                    "running": child_running,
                    "session_key": target,
                    "started_at": record["created_at"],
                    "status": "streaming" if child_running else "idle",
                },
            )

        # Desktop can ask for a bounded acknowledgement and hydrate the display
        # transcript through the paginated REST endpoint. Register the runtime now,
        # then load model history and initialize optional providers in background.
        # Repeated requests reuse the record through the live fast path above.
        #
        # Precedence vs omit_messages: defer_history SUPERSEDES omit_messages.
        # Desktop sends both flags on a cold resume; when defer_history is set the
        # response never carries a transcript (messages is always []) and the ONE
        # history read happens in the background hydration worker — the synchronous
        # omit_messages read below (cold resume default) is skipped entirely, so
        # the transcript is never loaded twice for one resume. omit_messages only
        # governs the response shape of the non-deferred paths.
        if defer_history and not is_truthy_value(params.get("eager_build", False)):
            sid = uuid.uuid4().hex[:8]
            source = _resolve_session_source(str(params.get("source") or "").strip() or None)
            lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
            _enable_gateway_prompts()
            overrides = _stored_session_runtime_overrides(found) or {}
            model_override = overrides.get("model_override") or {}
            cwd = profile_resume_cwd or _default_session_cwd()
            record = _deferred_session_record(
                target,
                cols=cols,
                cwd=cwd,
                history=[],
                lease=lease,
                source=source,
                close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
                profile_home=profile_home,
                model_override=overrides.get("model_override"),
                resume_runtime_overrides=overrides or None,
            )
            record["resume_history_ready"] = threading.Event()
            record["resume_hydrating"] = True
            record["resume_message_count"] = int(found.get("message_count") or 0)
            if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
                return _ok(rid, _reuse_live_payload(*live))

            _schedule_resume_hydration(sid, target, db, close_db=owns_db)
            # The hydration worker now owns a profile-scoped handle and closes it
            # after the transcript read. The shared launch DB is process-owned.
            if owns_db:
                owns_db = False
            _schedule_session_cap_enforcement()
            return _ok(
                rid,
                {
                    "session_id": sid,
                    "resumed": target,
                    "message_count": record["resume_message_count"],
                    "messages": [],
                    "hydrating": True,
                    "info": _lazy_resume_info(
                        cwd,
                        model=model_override.get("model") or "",
                        provider=overrides.get("provider_override") or "",
                        profile=profile,
                    ),
                    "inflight": None,
                    "running": False,
                    "session_key": target,
                    "started_at": record["created_at"],
                    "status": "resuming",
                },
            )

        # Cold resume default: register the live session and read its stored
        # transcript, but build the agent OFF the response path. _make_agent can
        # block for seconds (MCP discovery, prompt/skill build, AIAgent
        # construction), and every resume caller (desktop + Ink TUI) awaits this RPC
        # before it paints — so building eagerly is the bulk of the multi-second
        # "switching sessions is frozen" latency. Return the full display transcript
        # immediately and pre-warm the agent on a short timer (the same deferred-
        # build contract session.create uses); _sess() also builds on demand if the
        # first prompt beats the timer. A caller that needs the agent built
        # synchronously (e.g. tests of the build race) passes ``eager_build: true``
        # to fall through to the eager path below. Distinct from the lazy/watch
        # branch above: a normal resume restores the full ancestor history and the
        # session's persisted runtime identity, and is a real (upgradable) session.
        if not is_truthy_value(params.get("eager_build", False)):
            sid = uuid.uuid4().hex[:8]
            source = _resolve_session_source(str(params.get("source") or "").strip() or None)
            lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
            # Interactive resume routes approvals/clarify through gateway prompts;
            # the deferred build wires the remaining per-session callbacks.
            _enable_gateway_prompts()
            try:
                db.reopen_session(target)
                # One lineage SELECT feeds both projections (#67142-adjacent perf,
                # from the desktop audit): the model-fed copy is alternation-repaired
                # (raw_history → sanitize_replay_history → the resumed session's
                # working conversation) and the display copy stays verbatim —
                # inspection/export must show what is actually stored.
                if omit_messages:
                    raw_history = db.get_messages_as_conversation(
                        target, repair_alternation=True, include_row_ids=True
                    )
                    display_history = []
                else:
                    raw_history, display_history = db.get_resume_conversations(target)
            except Exception as e:
                if lease is not None:
                    lease.release()
                return _err(rid, 5000, f"resume failed: {e}")
            # Display keeps the full transcript; the model-fed history drops a
            # dangling/interrupted tool-call tail so a session killed mid-loop does
            # not replay the unanswered call forever (#29086).
            prefix = [] if omit_messages else db.get_ancestor_display_prefix(target)
            history = sanitize_replay_history(raw_history)
            # Restore the model/provider/reasoning/tier this chat last used so the
            # deferred build (and the info below) match the eager path — without them
            # the build drops the provider ("No LLM provider configured").
            overrides = _stored_session_runtime_overrides(found) or {}
            model_override = overrides.get("model_override") or {}
            cwd = profile_resume_cwd or _default_session_cwd()
            record = _deferred_session_record(
                target,
                cols=cols,
                cwd=cwd,
                history=history,
                lease=lease,
                source=source,
                close_on_disconnect=is_truthy_value(params.get("close_on_disconnect", False)),
                display_history_prefix=prefix,
                profile_home=profile_home,
                model_override=overrides.get("model_override"),
                resume_runtime_overrides=overrides or None,
            )
            if (live := _claim_or_reuse_live(sid, target, record, lease)) is not None:
                return _ok(rid, _reuse_live_payload(*live))

            _schedule_agent_build(sid)
            _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap
            auto_continue = _maybe_schedule_auto_continue(sid, record, target)

            messages = [] if omit_messages else _history_to_messages(display_history)
            payload = {
                "session_id": sid,
                "resumed": target,
                "message_count": len(raw_history) if omit_messages else len(messages),
                "messages": messages,
                "messages_omitted": omit_messages,
                "info": _lazy_resume_info(
                    cwd,
                    model=model_override.get("model") or "",
                    provider=overrides.get("provider_override") or "",
                    profile=profile,
                ),
                "inflight": None,
                "running": False,
                "session_key": target,
                "started_at": record["created_at"],
                "status": "idle",
            }
            if auto_continue is not None:
                payload["auto_continue"] = auto_continue
            return _ok(rid, payload)

        # Build the agent OUTSIDE the lock — _make_agent can block for seconds
        # (MCP discovery, prompt/skill build, AIAgent construction). Holding
        # _session_resume_lock across it would stall session.close on the main
        # dispatch thread (it's not a _LONG_HANDLER), blocking fast-path RPCs.
        sid = uuid.uuid4().hex[:8]
        source = _resolve_session_source(str(params.get("source") or "").strip() or None)
        lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
        _enable_gateway_prompts()
        home_token = (
            set_hermes_home_override(str(profile_home)) if profile_home is not None else None
        )
        secret_token = (
            set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
            if profile_home is not None
            else None
        )
        try:
            db.reopen_session(target)
            # One lineage SELECT feeds both projections (see the interactive resume
            # above): the model-fed copy is alternation-repaired for LIVE REPLAY, the
            # display copy stays verbatim.
            if omit_messages:
                raw_history = db.get_messages_as_conversation(
                    target, repair_alternation=True, include_row_ids=True
                )
                display_history = []
            else:
                raw_history, display_history = db.get_resume_conversations(target)
            # The display transcript keeps every row so the user still sees their
            # full history.  The model-fed history is sanitized: a session whose
            # last turn died mid-tool-loop persists a dangling assistant(tool_calls)
            # (or interrupted assistant→tool) tail; replaying it makes the model
            # re-issue the unanswered call forever — the permanent-"thinking" stuck
            # session in #29086.  The messaging gateway already strips this; this is
            # the WebUI/TUI resume path picking up the same cleanup.
            display_history_prefix = (
                [] if omit_messages else db.get_ancestor_display_prefix(target)
            )
            history = sanitize_replay_history(raw_history)
            messages = [] if omit_messages else _history_to_messages(display_history)
            tokens = _set_session_context(target)
            try:
                # Pass the profile's db so the agent persists turns to the right
                # state.db; home override is active here so config/skills/model
                # resolve to the profile too. Runtime identity is restored from the
                # stored session row so switching chats does not inherit whatever
                # global model another chat last selected.
                stored_runtime_overrides = _stored_session_runtime_overrides(found)
                agent = _make_agent(
                    sid,
                    target,
                    session_id=target,
                    session_db=db,
                    platform_override=source,
                    **stored_runtime_overrides,
                )
            finally:
                _clear_session_context(tokens)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5000, f"resume failed: {e}")
        finally:
            if home_token is not None:
                reset_hermes_home_override(home_token)
            if secret_token is not None:
                reset_secret_scope(secret_token)

        # Double-checked locking: another concurrent resume may have created the
        # live session while we were building. Re-check under the lock; if it won,
        # discard our just-built agent and reuse theirs (no worker/poller wired yet).
        with _session_resume_lock:
            live = _find_live_session_by_key(target)
            if live is not None:
                try:
                    if hasattr(agent, "close"):
                        agent.close()
                except Exception:
                    pass
                if lease is not None:
                    lease.release()
                other_sid, other_session = live
                payload = _live_session_payload(
                    other_sid,
                    other_session,
                    cols=cols,
                    touch=True,
                    transport=current_transport() or _stdio_transport,
                    omit_messages=omit_messages,
                )
                payload["resumed"] = target
                return _ok(rid, payload)
            try:
                init_home_token = (
                    set_hermes_home_override(str(profile_home))
                    if profile_home is not None
                    else None
                )
                init_secret_token = (
                    set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
                    if profile_home is not None
                    else None
                )
                try:
                    _init_session(
                        sid,
                        target,
                        agent,
                        history,
                        cols=cols,
                        cwd=profile_resume_cwd,
                        session_db=db,
                        source=source,
                    )
                    # Ownership TRANSFER — the registered session's agent now
                    # holds this handle for its whole life, and _init_session
                    # never closes a caller-supplied session_db (its
                    # _init_owns_db stays False). Closing it in the finally
                    # below would fault every later turn on this session with
                    # "Cannot operate on a closed database".
                    #
                    # Ownership moves ONTO the agent rather than just being
                    # dropped: AIAgent.close() (reached from _teardown_session
                    # on session.close and the orphaned-session reaper) closes
                    # a handle it owns, so the dedicated fds and the token
                    # writer are released at teardown instead of living as
                    # long as the gateway process.
                    #
                    # The drop is UNCONDITIONAL and the transfer is best-effort
                    # on top of it, deliberately. Past this line the session is
                    # registered and holding this handle, so the finally must
                    # not close it even if the transfer was refused — a refusal
                    # leaves the old leak, which is survivable; closing under a
                    # live session is the permanent "Cannot operate on a closed
                    # database" break this patch exists to avoid.
                    _transfer_db_to_agent(agent, db)
                    owns_db = False
                finally:
                    if init_home_token is not None:
                        reset_hermes_home_override(init_home_token)
                    if init_secret_token is not None:
                        reset_secret_scope(init_secret_token)
                if sid in _sessions:
                    if stored_runtime_overrides.get("model_override") is not None:
                        _sessions[sid]["model_override"] = stored_runtime_overrides[
                            "model_override"
                        ]
                    _sessions[sid]["display_history_prefix"] = display_history_prefix
                    # Remember the profile home so each turn re-binds HERMES_HOME (the
                    # agent persists to its own db, but mid-turn home reads — memory,
                    # skills — must resolve to the resumed profile too).
                    if profile_home is not None:
                        _sessions[sid]["profile_home"] = str(profile_home)
                    _sessions[sid]["active_session_lease"] = lease
            except Exception as e:
                # _init_session registers _sessions[sid] BEFORE its first read
                # through this handle. If it raised in between — "database is
                # locked" is the realistic trigger — the half-built session is
                # still registered while the finally below closes the handle it
                # holds, and the live-session fast path above would then serve
                # that dead session on every later resume of this id
                # ("'NoneType' object has no attribute 'execute'", permanently).
                # owns_db still True means ownership never transferred, so the
                # registration is ours to undo.
                if owns_db:
                    with _sessions_lock:
                        _sessions.pop(sid, None)
                if lease is not None:
                    lease.release()
                return _err(rid, 5000, f"resume failed: {e}")
            session = _sessions.get(sid) or {}
    finally:
        # Every return that does NOT reach the transfer above abandons this
        # handle — session-not-found, both "resume failed" paths, the live-session
        # fast path (the hot one: reconnects re-resume live chats through it), the
        # deferred cold/lazy returns, and the double-checked-locking discard.
        # Dropping it merely relied on refcounting to release the sqlite fds; that
        # stops being true the moment anything pins the instance — SessionDB pins
        # ITSELF once its background token writer starts, via
        # atexit.register(_drain_token_queue_at_exit) (hermes_state.py), which only
        # close() unregisters. A pinned handle keeps its db/-wal/-shm fds and its
        # writer thread for the life of the process.
        if owns_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()
    auto_continue = (
        _maybe_schedule_auto_continue(sid, session, target) if session else None
    )
    payload = {
        "session_id": sid,
        "resumed": target,
        "message_count": len(raw_history) if omit_messages else len(messages),
        "messages": messages,
        "messages_omitted": omit_messages,
        "info": _session_info(agent, session),
        "inflight": None,
        "running": False,
        "session_key": target,
        "started_at": float(session.get("created_at") or time.time()),
        "status": "idle",
    }
    if auto_continue is not None:
        payload["auto_continue"] = auto_continue
    return _ok(rid, payload)


@method("session.cwd.set")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    agent = session.get("agent")
    info = _session_info(agent, session) if agent is not None else {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "lazy": True,
    }
    _emit("session.info", params.get("session_id", ""), info)
    return _ok(rid, info)


@method("session.workspace.move")
def _(rid, params: dict) -> dict:
    """Re-home a STORED session's workspace into another folder/project.

    Unlike ``session.cwd.set`` (which acts on a live runtime session by its UI
    id), this targets a persisted row by ``session_key`` so the desktop can fix
    a session that was created in the wrong directory — no live agent required.
    The git branch/root columns are REPLACED (not merely enriched), because the
    whole point of the move is to change which project claims the session; a
    stale ``git_repo_root`` would keep it grouped under the project it left.

    A live agent bound to the row follows through the runtime path too, so its
    terminal/file tools re-anchor immediately. An explicit move wins even
    mid-turn: refusing a running session made the desktop's "Move to project"
    claim success in the UI while ``state.db`` kept the old cwd — two sources
    of truth disagreeing (#86626). In-flight tool calls keep the cwd they were
    launched with; the NEXT tool call uses the new workspace.
    """
    target = str(params.get("session_key") or "").strip()
    if not target:
        return _err(rid, 4007, "session_key required")
    raw = str(params.get("cwd", "") or "").strip()
    if not raw:
        return _err(rid, 4016, "cwd required")
    from hermes_constants import translate_cwd_for_wsl_backend

    resolved = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(raw)))
    if not os.path.isdir(resolved):
        return _err(rid, 4017, f"working directory does not exist: {raw}")

    # Snapshot under the lock — concurrent RPCs mutate _sessions (same pattern
    # as _cwd_for_session_key).
    live = None
    live_sid = ""
    with _sessions_lock:
        for sid, sess in list(_sessions.items()):
            if sess.get("session_key") == target:
                live, live_sid = sess, sid
                break

    branch = _git_branch_for_cwd(resolved)
    root = _git_common_repo_root_for_cwd(resolved)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        # A brand-new draft has no persisted row yet; the live re-home below
        # still applies and the row inherits the cwd when it is first written.
        row_exists = bool(db.get_session(target))
        if not row_exists and live is None:
            return _err(rid, 4007, "session not found")
        if row_exists:
            try:
                db.update_session_cwd(
                    target, resolved, branch, root, replace_git_meta=True
                )
            except Exception as e:
                return _err(rid, 5007, f"move failed: {e}")

    if live is not None:
        try:
            _set_session_cwd(live, resolved)
        except ValueError as e:
            return _err(rid, 4017, str(e))
        agent = live.get("agent")
        info = _session_info(agent, live) if agent is not None else {
            "cwd": resolved,
            "branch": branch,
            "project": _project_info_for_cwd(resolved),
            "lazy": True,
        }
        _emit("session.info", live_sid, info)

    return _ok(rid, {"cwd": resolved, "branch": branch, "git_repo_root": root})


@method("session.active_list")
def _(rid, params: dict) -> dict:
    """Return live TUI sessions in this gateway process.

    Unlike ``session.list`` this is not a historical DB browser: it reports only
    sessions with in-memory agents/workers that the current TUI can switch to
    without closing siblings.
    """
    current = str(params.get("current_session_id") or "")
    try:
        with _sessions_lock:
            snapshot = list(_sessions.items())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")

    # Liveness filter (#38950): a session whose teardown has begun (``_finalized``)
    # is dead — its agent/worker are being released and it is no longer
    # attachable — but it can briefly remain in ``_sessions`` until the reaper
    # pops it (the WS grace-reap and idle reaper both set ``_finalized`` inside
    # ``_teardown_session`` before the pop). Counting these inflated the footer's
    # "N sessions" count, which only ever went up until a gateway restart. Drop
    # them here so the count reflects genuinely attachable sessions. We do NOT
    # filter on ``transport is _detached_ws_transport`` (the WS-detached drop
    # sentinel): a detached session is still attachable via a quick reconnect /
    # session.resume until the grace-reap finalizes it, and a standalone
    # ``hermes --tui`` session legitimately rides the real stdio transport and
    # must stay visible.
    # Keep the natural creation/insertion order from ``_sessions``.  The
    # frontend marks the focused session with ``current``; it should not jump to
    # the top just because the user switched to it.
    rows = [
        _session_live_item(sid, session, current)
        for sid, session in snapshot
        if not session.get("_finalized")
    ]
    return _ok(rid, {"sessions": rows})


@method("session.activate")
def _(rid, params: dict) -> dict:
    """Attach the frontend to an already-live TUI session.

    This intentionally does not close the previously focused session; it merely
    returns enough state for Ink to redraw around another live session id.
    """
    sid = str(params.get("session_id") or "")
    session, err = _sess_nowait({"session_id": sid}, rid)
    if err:
        return err
    assert session is not None

    return _ok(
        rid,
        _live_session_payload(
            sid,
            session,
            touch=True,
            transport=current_transport() or _stdio_transport,
            omit_messages=is_truthy_value(params.get("omit_messages", False)),
        ),
    )


@method("session.delete")
def _(rid, params: dict) -> dict:
    """Delete a stored session and its on-disk transcript files.

    Used by the TUI resume picker (``d`` key) so users can prune old
    sessions without dropping to the CLI.  Refuses to delete a session
    that is currently active in this gateway process — those rows are
    still being written to and removing them out from under the live
    agent corrupts message ordering and trips FK constraints when the
    next message append flushes.

    Honors ``params.profile`` so app-global remote mode deletes from the
    focused profile's ``state.db`` + sessions dir (mirrors ``session.resume``).
    """
    target = params.get("session_id", "")
    if not target:
        return _err(rid, 4006, "session_id required")
    # Block deletion of any session currently bound to a live TUI session
    # in this process.  The picker hides the active session anyway, but a
    # racing caller could still target it.  Snapshot via ``list(...)``
    # because ``_sessions`` is mutated by concurrent RPCs on the thread
    # pool — iterating the dict directly can raise ``RuntimeError:
    # dictionary changed size during iteration``.  If even the snapshot
    # raises, fail closed (refuse the delete) rather than fail open.
    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception as e:
        return _err(rid, 5036, f"could not enumerate active sessions: {e}")
    active = {s.get("session_key") for s in snapshot if s.get("session_key")}
    if target in active:
        return _err(rid, 4023, "cannot delete an active session")
    profile = (params.get("profile") or "").strip() or None
    profile_home = _profile_home(profile)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5036)
        if profile_home is not None:
            sessions_dir = Path(profile_home) / "sessions"
        else:
            sessions_dir = get_hermes_home() / "sessions"
        try:
            deleted = db.delete_session(target, sessions_dir=sessions_dir)
        except Exception as e:
            return _err(rid, 5036, f"delete failed: {e}")
        if not deleted:
            return _err(rid, 4007, "session not found")
        return _ok(rid, {"deleted": target})


@method("session.title")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        if "title" not in params:
            fallback = session.get("pending_title") or ""
            try:
                resolved_title = db.get_session_title(key) or ""
                if fallback:
                    if db.set_session_title(key, fallback):
                        session["pending_title"] = None
                        resolved_title = fallback
                    else:
                        existing_row = db.get_session(key)
                        existing_title = ((existing_row or {}).get("title") or "").strip()
                        if existing_title == fallback:
                            session["pending_title"] = None
                            resolved_title = fallback
                        elif not resolved_title:
                            resolved_title = fallback
                elif resolved_title:
                    session["pending_title"] = None
            except Exception:
                resolved_title = fallback
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(
                rid,
                {
                    "title": resolved_title,
                    "session_key": key,
                },
            )
        title = (params.get("title", "") or "").strip()
        if not title:
            return _err(rid, 4021, "title required")
        try:
            if db.set_session_title(key, title):
                session["pending_title"] = None
                _emit_session_info_for_session(params.get("session_id", ""), session)
                return _ok(rid, {"pending": False, "title": title})
            # rowcount == 0 can mean "same value" as well as "missing row".
            existing_row = db.get_session(key)
            if existing_row:
                session["pending_title"] = None
                _emit_session_info_for_session(params.get("session_id", ""), session)
                return _ok(
                    rid,
                    {
                        "pending": False,
                        "title": (existing_row.get("title") or title),
                    },
                )
            # No row yet (the DB write is deferred to the first prompt so empty
            # drafts don't litter the sidebar). An explicit /title is clear user
            # intent, not an abandoned draft — so persist the row NOW and set the
            # title, mirroring the messaging gateway's _handle_title_command. The
            # old behavior only queued pending_title and relied on the post-turn
            # apply block; if that turn never landed under this session_key the
            # title was silently lost and the sidebar fell back to the message
            # preview. Creating the row up front removes that race entirely. The
            # min-messages sidebar filter keeps a titled 0-message row hidden, so
            # a /title'd-but-never-used draft still doesn't clutter the list.
            _ensure_session_db_row(session)
            with _session_db(session) as scoped_db:
                if scoped_db is not None and scoped_db.set_session_title(key, title):
                    session["pending_title"] = None
                    _emit_session_info_for_session(params.get("session_id", ""), session)
                    return _ok(rid, {"pending": False, "title": title})
            # Row creation didn't take (DB unavailable, or a concurrent writer) —
            # fall back to queuing so the post-turn apply block can still recover.
            session["pending_title"] = title
            _emit_session_info_for_session(params.get("session_id", ""), session)
            return _ok(rid, {"pending": True, "title": title})
        except ValueError as e:
            return _err(rid, 4022, str(e))
        except Exception as e:
            return _err(rid, 5007, str(e))


@method("session.set_hidden")
def _(rid, params: dict) -> dict:
    """Set/clear the generic ``hidden`` flag on a session (and its lineage).

    Mirrors the durable ``pinned``/``archived`` setters: a hidden session is
    dropped from the default global Sessions list (``list_sessions_rich``
    without ``include_hidden``) but stays fully resumable by the surface that
    owns it — for plugins that manage their own sessions and don't want them
    cluttering the shared recents list. Flips the whole compression chain as a
    unit in the DB layer.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    hidden = is_truthy_value(params.get("hidden", True))
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            changed = db.set_session_hidden(key, hidden)
            if not changed:
                # No row yet (write deferred to the first prompt): remember the
                # intent so _ensure_session_db_row is born hidden, mirroring the
                # pending_title deferral.
                session["pending_hidden"] = hidden
            return _ok(rid, {"hidden": hidden, "session_key": key})
        except Exception as e:
            return _err(rid, 5007, str(e))


@method("message.react")
def _(rid, params: dict) -> dict:
    """Set or clear one author's emoji reaction on a persisted message.

    iOS Tapback semantics, enforced in the DB layer: one reaction per author
    per message, re-sending the same emoji retracts it. ``emoji: null`` clears
    unconditionally. ``row_id`` is the durable ``messages.id`` forwarded by
    ``_history_to_messages`` — the renderer's own message ids are ephemeral.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    # A live message hasn't round-tripped through a resume, so the desktop has
    # no durable row id for it yet. It can instead name the ROLE whose newest
    # row it means — which is the message the user just reacted to.
    newest_role = str(params.get("newest_role") or "").strip()
    row_id = params.get("row_id")
    if row_id is None and newest_role not in {"user", "assistant"}:
        return _err(rid, 4023, "row_id or newest_role required")

    emoji = params.get("emoji")
    if emoji is not None:
        emoji = str(emoji).strip()
        if not emoji:
            return _err(rid, 4024, "emoji must be a non-empty string or null")

    author = str(params.get("author") or "user").strip()
    if author not in {"user", "agent"}:
        return _err(rid, 4025, "author must be 'user' or 'agent'")

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if row_id is None:
                row_id = db.latest_message_row_id(
                    session["session_key"], role=newest_role
                )
                if row_id is None:
                    return _err(rid, 4040, "no message to react to yet")
            reactions = db.set_message_reaction(
                session["session_key"], int(row_id), emoji, author=author
            )
        except Exception as e:
            return _err(rid, 5007, str(e))

    if reactions is None:
        return _err(rid, 4040, "message not found in this session")

    return _ok(rid, {"row_id": int(row_id), "reactions": reactions})


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Run a single stateless LLM request outside any conversation.

    Generic helper for small generative chores (e.g. a commit message from a
    diff). Accepts either a named ``template`` + ``variables`` or an explicit
    ``instructions`` / ``input`` pair. When ``session_id`` resolves to a live
    session the call inherits that agent's model; otherwise it uses the
    configured auxiliary ``task`` backend. Never mutates session history, so
    prompt caching is untouched.
    """
    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    task = (params.get("task") or "title_generation").strip() or "title_generation"

    try:
        max_tokens = int(params.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    temperature = params.get("temperature")
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = None

    if not template and not str(instructions).strip() and not str(user_input).strip():
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")

    # Optional: inherit the live session's model (no error if absent).
    session = _sessions.get(params.get("session_id") or "")
    main_runtime = _main_runtime_from_agent(session.get("agent")) if session else None

    try:
        from agent.oneshot import run_oneshot

        text = run_oneshot(
            instructions=instructions,
            user_input=user_input,
            template=template,
            variables=variables,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.3,
            main_runtime=main_runtime,
        )
    except KeyError as e:
        return _err(rid, 4031, str(e))
    except ValueError as e:
        return _err(rid, 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")

    return _ok(rid, {"text": text})


@method("handoff.request")
def _(rid, params: dict) -> dict:
    """Queue a handoff of this session to a messaging platform.

    Desktop parity with the CLI ``/handoff`` command: we only write
    ``handoff_state='pending'`` onto the persisted session row. The actual
    transfer is performed by the separate ``hermes gateway`` process, whose
    ``_handoff_watcher`` claims the row, re-binds the session to the platform's
    home channel, and forges a synthetic turn. The desktop then polls
    ``handoff.state`` for the terminal result.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — wait for the current turn to finish, then retry the handoff",
        )

    platform_name = (params.get("platform", "") or "").strip().lower()
    if not platform_name:
        return _err(rid, 4023, "platform required")

    # Validate against the live gateway config — an unconfigured platform or a
    # missing home channel would leave the handoff pending forever, so reject
    # up front with a clear, actionable message (mirrors cli.py).
    try:
        from gateway.config import Platform, load_gateway_config
    except Exception as e:  # pragma: no cover — gateway pkg always ships
        return _err(rid, 5021, f"could not load gateway config: {e}")
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    pcfg = gw_config.platforms.get(platform)
    if not pcfg or not pcfg.enabled:
        return _err(
            rid,
            4025,
            f"platform '{platform_name}' is not configured/enabled in the gateway",
        )
    home = gw_config.get_home_channel(platform)
    if not home or not home.chat_id:
        return _err(
            rid,
            4026,
            f"no home channel configured for {platform_name} — set one with "
            "/sethome on the destination chat first",
        )

    # The watcher transfers a persisted DB row, so make sure one exists even
    # for a brand-new empty chat (mirrors the CLI's set_session_title stub).
    _ensure_session_db_row(session)

    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            ok = db.request_handoff(key, platform_name)
        except Exception as e:
            return _err(rid, 5007, str(e))

    if not ok:
        return _err(
            rid,
            4027,
            "session is already in flight for handoff — wait for it to settle, then retry",
        )
    return _ok(
        rid,
        {
            "queued": True,
            "session_key": key,
            "platform": platform_name,
            "home_name": home.name,
        },
    )


@method("handoff.state")
def _(rid, params: dict) -> dict:
    """Poll the handoff state for a session.

    Returns ``{state, platform, error}`` where ``state`` is one of
    ``pending|running|completed|failed`` (or empty when no handoff record
    exists). Desktop polls this after ``handoff.request``.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        record = db.get_handoff_state(session["session_key"])

    record = record or {}
    return _ok(
        rid,
        {
            "state": record.get("state") or "",
            "platform": record.get("platform") or "",
            "error": record.get("error") or "",
        },
    )


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Mark an in-flight handoff as failed so the user can retry.

    Desktop calls this when its bounded poll times out. Only pending/running
    rows are changed so a late success from the gateway watcher is not clobbered.
    """
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        record = db.get_handoff_state(key) or {}
        state = record.get("state") or ""
        if state in {"pending", "running"}:
            db.fail_handoff(key, reason)
            return _ok(rid, {"failed": True, "state": "failed"})

    return _ok(rid, {"failed": False, "state": state})


@method("session.usage")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    usage: dict = _session_usage_snapshot(session)
    if agent is None and not usage:
        usage = {"calls": 0, "input": 0, "output": 0, "total": 0}
    # Nous credits block — agent-independent (a portal fetch), so it shows even
    # with zero API calls or on a resumed session. The TUI /usage panel renders
    # these lines regardless of `calls`. Fail-open: [] when not logged into Nous
    # or on any portal hiccup.
    try:
        from agent.account_usage import nous_credits_lines

        credits = nous_credits_lines()
        if credits:
            usage["credits_lines"] = credits
    except Exception:
        pass
    return _ok(rid, usage)


@method("session.context_breakdown")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None:
        usage = _session_usage_snapshot(session) or _get_usage(None)
        return _ok(
            rid,
            {
                "categories": [],
                "context_max": usage.get("context_max", 0) or 0,
                "context_percent": usage.get("context_percent", 0) or 0,
                "context_used": usage.get("context_used", 0) or 0,
                "estimated_total": usage.get("context_used", 0) or usage.get("total", 0) or 0,
                "model": _metadata_mirror(session).get("model", ""),
            },
        )
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        from agent.context_breakdown import compute_session_context_breakdown

        payload = compute_session_context_breakdown(agent, history)
    except Exception as exc:
        return _err(rid, 5000, f"Could not compute context breakdown: {exc}")
    return _ok(rid, payload)


@method("pet.info")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return the active petdex pet for surfaces that render sprites.

    Shared by the desktop (canvas) and the TUI (half-block). Carries the
    spritesheet bytes (base64) plus the engine's frame geometry + state-row
    taxonomy so the renderer is a thin, framework-native consumer. The
    activity→state decision is mirrored from ``agent.pet.state`` client-side.

    Agent-independent (reads config + disk), so it works on any session and
    before the agent finishes building. Fail-open: returns ``enabled=False``
    on any error rather than erroring the surface.
    """
    try:
        enabled, pet, scale = _pet_active_selection()

        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        payload = {"enabled": True, **_pet_sprite_payload(pet, scale=scale)}

        # Send-once semantics for the multi-MB spritesheet (#54730): a caller
        # that already holds the sheet passes the revision it has, and an
        # unchanged sheet comes back as metadata only (spritesheetUnchanged).
        known_revision = str(params.get("knownRevision", "") or "")
        if known_revision and known_revision == payload.get("spritesheetRevision"):
            payload.pop("spritesheetBase64", None)
            payload["spritesheetUnchanged"] = True

        return _ok(rid, payload)
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.info.meta")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    try:
        enabled, pet, scale = _pet_active_selection()
        if not enabled or pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})
        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "scale": scale,
                "spritesheetRevision": _pet_sheet_revision(pet.spritesheet),
            },
        )
    except Exception as exc:  # noqa: BLE001 - cosmetic, never break the surface
        logger.debug("pet.info.meta failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.cells")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return half-block cell frames for one pet state (TUI renderer).

    The TUI can't draw a canvas, so the engine downsamples the spritesheet to
    a grid of half-block cells and the Ink side paints them with native color
    props. Each cell is ``[tr,tg,tb,ta, br,bg,bb,ba]`` (top + bottom pixel).

    Params: ``state`` (idle/run/review/failed/wave/jump), ``cols`` (width).
    Fail-open: ``enabled=False`` on any problem.
    """
    try:
        from agent.pet import constants, render, store
        from agent.pet.render import PetRenderer

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        if not is_truthy_value(pet_cfg.get("enabled"), default=False):
            return _ok(rid, {"enabled": False})

        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
        if pet is None or not pet.exists:
            return _ok(rid, {"enabled": False})

        state = str(params.get("state") or constants.PetState.IDLE.value)
        scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
        cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))

        # Graphics path: when the TUI is attached to a real TTY (``graphics``)
        # and the terminal speaks the kitty protocol, return a Unicode-
        # placeholder payload for a crisp image instead of half-blocks. Env
        # detection (KITTY_WINDOW_ID / TERM / TERM_PROGRAM) is shared with the
        # Ink process since it spawns us; the dashboard PTY (xterm.js) has no
        # such env, so it falls through to half-blocks automatically. Only
        # kitty is grid-safe in Ink — iTerm/sixel stay on the fallback.
        if params.get("graphics"):
            configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
            gmode = render.detect_terminal_graphics() if configured in ("", "auto") else configured
            if gmode == "kitty":
                image_id = render.kitty_image_id(pet.slug)
                # kitty sizes from scaled pixels (_cell_box), so unicode_cols is moot here.
                payload = PetRenderer(
                    str(pet.spritesheet), mode="kitty", scale=scale
                ).kitty_payload(state, image_id=image_id)
                if payload:
                    kcount = len(payload["frames"]) or 1
                    return _ok(
                        rid,
                        {
                            "enabled": True,
                            "slug": pet.slug,
                            "displayName": pet.display_name,
                            "state": state,
                            "graphics": "kitty",
                            "imageId": image_id,
                            "color": render.kitty_color_hex(image_id),
                            "cols": payload["cols"],
                            "rows": payload["rows"],
                            "placeholder": payload["placeholder"],
                            "frames": payload["frames"],
                            "frameMs": constants.LOOP_MS / max(1, kcount),
                            "scale": scale,
                        },
                    )

        renderer = PetRenderer(
            str(pet.spritesheet),
            mode="unicode",
            scale=scale,
            unicode_cols=cols,
        )
        count = renderer.frame_count(state) or 1
        frames = []
        for i in range(count):
            grid = renderer.cells(state, i, cols=cols)
            frames.append(
                [[[*top, *bottom] for (top, bottom) in row] for row in grid]
            )

        return _ok(
            rid,
            {
                "enabled": True,
                "slug": pet.slug,
                "displayName": pet.display_name,
                "state": state,
                "cols": cols,
                "frameMs": constants.LOOP_MS / max(1, count),
                "frames": frames,
                "scale": scale,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.cells failed: %s", exc)
        return _ok(rid, {"enabled": False})


@method("pet.gallery")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """List adoptable pets for the desktop appearance picker.

    Returns the petdex gallery merged with local install state plus the
    current config (active slug + enabled). Agent-independent. Fail-open:
    returns whatever is installed locally if the gallery can't be reached, so
    the picker still works offline.

    Param ``localOnly`` (bool): skip the remote petdex manifest fetch and return
    only locally-installed pets. The desktop loads this first so the user's own
    pets render instantly instead of waiting on the (possibly slow) manifest.
    """
    local_only = bool(params.get("localOnly"))
    try:
        from agent.pet import store

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
            pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        except Exception:
            pet_cfg = {}

        installed = {p.slug: p for p in store.installed_pets()}

        gallery: list[dict] = []
        seen: set[str] = set()
        try:
            from agent.pet.manifest import fetch_manifest, prefetch

            # Local-only: skip the network entirely, but kick off a background
            # warm so the follow-up full request usually hits a cached manifest.
            if local_only:
                prefetch()

            for entry in [] if local_only else fetch_manifest():
                seen.add(entry.slug)
                gallery.append(
                    {
                        "slug": entry.slug,
                        "displayName": entry.display_name,
                        "installed": entry.slug in installed,
                        "spritesheetUrl": entry.spritesheet_url,
                        # petdex exposes no popularity metric; "curated" (its
                        # hand-picked/official set, identified by the asset path)
                        # is the closest signal, so the picker can surface it first.
                        "curated": "/curated/" in entry.spritesheet_url,
                        "generated": entry.slug in installed and installed[entry.slug].generated,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
            logger.debug("pet.gallery manifest fetch failed: %s", exc)

        # Always include locally-installed pets even if the gallery is unreachable.
        for slug, pet in installed.items():
            if slug not in seen:
                gallery.append(
                    {
                        "slug": slug,
                        "displayName": pet.display_name,
                        "installed": True,
                        "spritesheetUrl": "",
                        "generated": pet.generated,
                    }
                )

        return _ok(
            rid,
            {
                "enabled": is_truthy_value(pet_cfg.get("enabled"), default=False),
                "active": str(pet_cfg.get("slug", "") or ""),
                "pets": gallery,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.gallery failed: %s", exc)
        return _ok(rid, {"enabled": False, "active": "", "pets": []})


@method("pet.select")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Adopt a pet from the desktop picker: install (if needed) + activate.

    Params: ``slug`` (required). Writes ``display.pet.*`` to config and returns
    ``{ok, slug, displayName}``. The surface re-pulls ``pet.info`` to render it.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from agent.pet.manifest import ManifestError
        from hermes_cli.pets import _set_active

        try:
            pet = store.install_pet(slug)
        except (store.PetStoreError, ManifestError) as exc:
            return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
        _set_active(slug)
        return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.select failed: %s", exc)
        return _err(rid, 5031, f"pet.select failed: {exc}")


@method("pet.remove")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Uninstall a pet from the desktop picker (delete its on-disk directory).

    Params: ``slug`` (required). If the removed pet was the active one, the
    display is turned off so nothing tries to render a now-missing sprite.
    Returns ``{ok, slug}`` where ``ok`` reflects whether a directory was deleted.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        from agent.pet import store
        from hermes_cli.pets import _clear_active_if

        removed = store.remove_pet(slug)

        # If that was the active pet, stop surfaces pointing at a deleted sprite.
        try:
            _clear_active_if(slug)
        except Exception as exc:  # noqa: BLE001 - removal already succeeded
            logger.debug("pet.remove config update failed: %s", exc)

        return _ok(rid, {"ok": removed, "slug": slug})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.remove failed: %s", exc)
        return _err(rid, 5031, f"pet.remove failed: {exc}")


@method("pet.export")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Export an installed pet as a re-importable ``.zip`` (pet.json + sprite).

    Params: ``slug`` (required). Returns ``{ok, filename, zipBase64}`` — the
    client decodes the base64 and saves it. Heavy-ish (reads + zips files) but
    small; runs inline.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        filename, data = store.export_pet(slug)
        return _ok(
            rid,
            {"ok": True, "filename": filename, "zipBase64": base64.standard_b64encode(data).decode("ascii")},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.export failed: %s", exc)
        return _err(rid, 5031, f"pet.export failed: {exc}")


@method("pet.rename")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Rename an installed pet's display name + realign its slug/dir.

    Params: ``slug`` + ``name`` (both required). Lets the generate flow hatch
    with a provisional name and apply the user's chosen name at adopt time.
    Returns ``{ok, slug, displayName}`` with the (possibly new) slug.
    """
    slug = str(params.get("slug") or "").strip()
    name = str(params.get("name") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        from agent.pet import store

        new_slug = store.rename_pet(slug, name)
        if not new_slug:
            return _err(rid, 5031, "pet.rename failed")

        # The dir may have moved; if the renamed pet was active, follow the slug
        # in config so surfaces don't point at the old (now-missing) directory.
        if new_slug != slug:
            try:
                from hermes_cli.pets import _rename_active_if

                _rename_active_if(slug, new_slug)
            except Exception as exc:  # noqa: BLE001 - rename already succeeded
                logger.debug("pet.rename config update failed: %s", exc)

        return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.rename failed: %s", exc)
        return _err(rid, 5031, f"pet.rename failed: {exc}")


@method("pet.thumb")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Return a small idle-frame PNG (data URI) for one pet — the picker preview.

    Cropped + cached server-side so the renderer gets a same-origin data URL
    instead of a CDN ``<img>`` (which the desktop CSP / R2 hotlink rules break).
    Params: ``slug`` (required), ``url`` (optional petdex spritesheet URL used
    only for not-yet-installed pets). Fail-open: ``{ok: false}`` with no error.
    """
    slug = str(params.get("slug") or "").strip()
    if not slug:
        return _err(rid, 4004, "missing slug")
    try:
        import base64

        from agent.pet import store

        data = store.thumbnail_png(slug, source_url=str(params.get("url") or ""))
        if not data:
            return _ok(rid, {"ok": False, "slug": slug})

        return _ok(
            rid,
            {
                "ok": True,
                "slug": slug,
                "dataUri": "data:image/png;base64," + base64.standard_b64encode(data).decode("ascii"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.thumb failed: %s", exc)
        return _ok(rid, {"ok": False, "slug": slug})


@method("pet.disable")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Turn the pet off from the desktop picker (``display.pet.enabled=false``)."""
    try:
        from hermes_cli.pets import _set_enabled

        _set_enabled(False)
        return _ok(rid, {"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.disable failed: %s", exc)
        return _err(rid, 5031, f"pet.disable failed: {exc}")


@method("pet.scale")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` from the desktop slider. Params: ``scale``.

    Clamped to the engine bounds. The renderer updates its own ``$petInfo`` for
    instant feedback; this just makes the change durable + visible to the other
    terminal surfaces on their next read.
    """
    try:
        from hermes_cli.pets import set_pet_scale

        scale, err = set_pet_scale(params.get("scale"))
        if err:
            return _err(rid, 4004, err)
        return _ok(rid, {"ok": True, "scale": scale})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.scale failed: %s", exc)
        return _err(rid, 5031, f"pet.scale failed: {exc}")


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Signal an in-flight ``pet.generate``/``pet.hatch`` (by token) to stop.

    Best-effort + idempotent: cancelling an unknown/finished token is a no-op.
    Stays off the worker pool so it lands while a heavy generation is occupying
    it. Returns ``{ok: True}``.
    """
    token = str(params.get("token") or "").strip()
    if token:
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@method("pet.generate.status")
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible right now.

    True only when a reference-capable image backend (Nous Portal / OpenRouter /
    OpenAI gpt-image) is configured — the desktop checks this on open so it can
    offer setup instead of a dead prompt. Cheap (config + plugin discovery).
    """
    try:
        from agent.pet.generate.imagegen import (
            GenerationError,
            list_sprite_providers,
            resolve_provider,
        )

        try:
            resolve_provider(require_references=True)
            available = True
        except GenerationError:
            available = False
        try:
            providers = list_sprite_providers()
        except Exception as exc:  # noqa: BLE001 - picker is best-effort
            logger.debug("pet provider list failed: %s", exc)
            providers = []
        return _ok(rid, {"available": available, "providers": providers})
    except Exception as exc:  # noqa: BLE001 - never break the surface
        logger.debug("pet.generate.status failed: %s", exc)
        return _ok(rid, {"available": False, "providers": []})


@method("pet.generate")
def _(rid, params: dict) -> dict:
    """Generate candidate base looks for a new pet (the draft/variant step).

    Params: ``prompt`` (required unless ``referenceImage`` is given), ``count``
    (default 4), ``style`` (default ``auto``), ``referenceImage`` (optional data
    URL — a user photo/reference every draft is grounded on, e.g. to make *their*
    pet). Returns ``{ok, token, drafts:[{index, dataUri}]}`` — the token keys the
    staged base images for a later ``pet.hatch``. Heavy (network): worker pool.
    """
    prompt = str(params.get("prompt") or "").strip()
    ref_raw = str(params.get("referenceImage") or "").strip()
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    try:
        count = max(1, min(4, int(params.get("count") or 4)))
    except (TypeError, ValueError):
        count = 4
    style = str(params.get("style") or "auto").strip() or "auto"

    try:
        import shutil
        import uuid

        from agent.pet.generate import generate_base_drafts
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        root = _pet_gen_root()
        _pet_gen_sweep(root)

        # Token up front so each draft can be staged + streamed the moment it
        # lands, instead of the user staring at a blank grid until all N finish.
        token = uuid.uuid4().hex[:12]
        _pet_cancel_arm(token)
        stage = root / token
        stage.mkdir(parents=True, exist_ok=True)

        reference_images = None
        if ref_raw:
            try:
                reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
            except ValueError as exc:
                _pet_cancel_release(token)
                return _err(rid, 4004, str(exc))

        # Optional desktop picker override: resolve the chosen provider up front so
        # a bad/uncredentialed pick fails fast instead of mid-fan-out.
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=bool(reference_images), prefer=provider_name)
            except GenerationError as exc:
                _pet_cancel_release(token)
                return _err(rid, 5031, str(exc))

        concept = prompt or "a pet based on the reference image"
        out: list[dict] = []

        # Hand the token to the client up front (token-only init event) so a Stop
        # fired before the first draft lands can still target this run.
        try:
            _emit("pet.generate.progress", "", {"token": token, "count": count})
        except Exception as exc:  # noqa: BLE001 - streaming is best-effort
            logger.debug("pet.generate init emit failed: %s", exc)

        def _on_draft(index: int, src) -> None:
            dest = stage / f"draft-{index}.png"
            try:
                shutil.copyfile(src, dest)
                data_uri = _pet_png_data_uri(dest)
            except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
                logger.debug("pet.generate draft %d failed: %s", index, exc)
                return
            out.append({"index": index, "dataUri": data_uri})
            # Stream this draft to the client so the grid fills in live. Best-
            # effort: a transport hiccup must not abort the generation itself.
            try:
                _emit(
                    "pet.generate.progress",
                    "",
                    {"token": token, "index": index, "dataUri": data_uri, "count": count},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.generate progress emit failed: %s", exc)

        try:
            generate_base_drafts(
                concept,
                n=count,
                style=style,
                reference_images=reference_images,
                provider=sprite,
                on_draft=_on_draft,
                is_cancelled=lambda: _pet_is_cancelled(token),
            )
        except GenerationError as exc:
            _pet_cancel_release(token)
            return _err(rid, 5031, str(exc))

        cancelled = _pet_is_cancelled(token)
        _pet_cancel_release(token)
        if cancelled:
            return _err(rid, 5031, "generation cancelled")
        if not out:
            return _err(rid, 5031, "generation produced no usable drafts")
        out.sort(key=lambda d: d["index"])
        return _ok(rid, {"ok": True, "token": token, "drafts": out})
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.generate failed: %s", exc)
        return _err(rid, 5031, f"pet.generate failed: {exc}")


@method("pet.hatch")
def _(rid, params: dict) -> dict:
    """Turn a chosen base draft into a full pet — installed but NOT yet active.

    Generation is expensive and the result varies, so hatch produces a *preview*
    the surface plays (all frames) before the user commits: the pet is written to
    the store (so it can be rendered + later activated) but the active pet is left
    untouched. Adopt with ``pet.select`` or throw it away with ``pet.remove``.

    Params: ``token`` + ``index`` (from ``pet.generate``), ``name`` (required),
    ``description`` (optional), ``prompt`` (optional concept for row prompts),
    ``style`` (optional). Returns ``{ok, slug, displayName, warnings, pet}`` where
    ``pet`` is the renderer payload. Heavy (network + raster): worker pool.
    """
    token = str(params.get("token") or "").strip()
    # Hatch cancellation rides its own key, not the generation token: hatching a
    # draft mid-generation means pet.generate is still releasing `token`, which
    # would otherwise wipe the arm we set here. Falls back to `token` for clients
    # that don't send one.
    cancel_token = str(params.get("cancelToken") or "").strip() or token
    index = params.get("index", 0)
    name = str(params.get("name") or "").strip()
    if not token:
        return _err(rid, 4004, "missing token")
    if not name:
        return _err(rid, 4004, "missing name")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0

    try:
        from agent.pet import store
        from agent.pet.generate import hatch_pet
        from agent.pet.generate.imagegen import GenerationError, resolve_provider

        base = _pet_gen_root() / token / f"draft-{index}.png"
        if not base.is_file():
            return _err(rid, 4004, "draft expired — generate again")

        # Optional desktop picker override (rows always need reference grounding).
        provider_name = str(params.get("provider") or "").strip()
        sprite = None
        if provider_name:
            try:
                sprite = resolve_provider(require_references=True, prefer=provider_name)
            except GenerationError as exc:
                return _err(rid, 5031, str(exc))

        _pet_cancel_arm(cancel_token)
        slug = store.unique_slug(name)

        def _on_progress(event: str, detail: str) -> None:
            # Row progress is encoded as "<state>:<done>:<total>" so the egg
            # screen can show "Drawing <state>… (n/total)"; other phases
            # (compose, save) pass through as-is. Best-effort streaming.
            payload: dict = {"event": event, "detail": detail}
            if event == "row" and detail.count(":") == 2:
                state, done, total = detail.split(":")
                payload = {"event": "row", "state": state, "done": done, "total": total}
            try:
                _emit("pet.hatch.progress", "", payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pet.hatch progress emit failed: %s", exc)

        try:
            result = hatch_pet(
                base_image=base,
                slug=slug,
                display_name=name,
                description=str(params.get("description") or ""),
                concept=str(params.get("prompt") or name),
                style=str(params.get("style") or "auto").strip() or "auto",
                provider=sprite,
                on_progress=_on_progress,
                is_cancelled=lambda: _pet_is_cancelled(cancel_token),
            )
        except GenerationError as exc:
            return _err(rid, 5031, str(exc))
        finally:
            _pet_cancel_release(cancel_token)

        pet = store.load_pet(result.slug)
        payload = _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}
        return _ok(
            rid,
            {
                "ok": True,
                "slug": result.slug,
                "displayName": result.display_name,
                "warnings": result.validation.get("warnings", []),
                "pet": payload,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pet.hatch failed: %s", exc)
        return _err(rid, 5031, f"pet.hatch failed: {exc}")


@method("billing.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/state → serialized BillingState (Screen 1 + 5).

    Fail-open like the other billing RPCs: a logged-out / unreachable portal yields
    {ok:true, logged_in:false}. No scope required for this endpoint.
    """
    try:
        from agent.billing_view import build_billing_state

        state = build_billing_state()
        return _ok(rid, _serialize_billing_state(state))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load billing state"})


@method("usage.bars")
def _(rid, params: dict) -> dict:
    """Shared dollar usage model (two-bar view) for /usage + /subscription.

    Fail-open: logged-out / unreachable portal → {ok:true, available:false}.
    No scope required (read-only).
    """
    try:
        from agent.billing_usage import build_usage_model

        return _ok(rid, _serialize_usage_model(build_usage_model()))
    except Exception:
        return _ok(rid, {"ok": True, "available": False})


@method("subscription.state")
def _(rid, params: dict) -> dict:
    """GET /api/billing/subscription → serialized SubscriptionState.

    Fail-open like billing.state: logged-out / unreachable portal →
    {ok:true, logged_in:false}. No scope required (read-only).
    """
    try:
        from agent.subscription_view import build_subscription_state

        state = build_subscription_state()
        return _ok(rid, _serialize_subscription_state(state))
    except Exception:
        return _ok(rid, {"ok": True, "logged_in": False, "error": "could not load subscription state"})


@method("subscription.preview")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/preview → serialized quote or typed error.

    params: {subscription_type_id: str}. Chargeless effect quote. Requires
    billing:manage (live Stripe calls + amounts), so a 403 → insufficient_scope
    drives the device step-up exactly like the mutations.
    """
    from agent.subscription_view import subscription_change_preview_from_payload
    from hermes_cli.nous_billing import BillingError, post_subscription_preview

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "subscription_type_id is required"})
    try:
        preview = subscription_change_preview_from_payload(
            post_subscription_preview(subscription_type_id=tier_id)
        )
        return _ok(rid, _serialize_subscription_preview(preview))
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("subscription.change")
def _(rid, params: dict) -> dict:
    """PUT /api/billing/subscription/pending-change → {ok, message} or typed error.

    params: {subscription_type_id?: str, cancel?: bool}. Schedules a downgrade /
    same-price change OR a cancellation at period end (chargeless). Requires
    billing:manage.
    """
    from hermes_cli.nous_billing import BillingError, put_subscription_pending_change

    cancel = bool(params.get("cancel"))
    tier_id = params.get("subscription_type_id")
    if not cancel and not tier_id:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "subscription_type_id or cancel is required"})
    try:
        result = put_subscription_pending_change(subscription_type_id=tier_id, cancel=cancel)
        return _ok(rid, {"ok": True, "message": result.get("message"), "payload": result})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("subscription.resume")
def _(rid, params: dict) -> dict:
    """DELETE /api/billing/subscription/pending-change → {ok, message} or typed error.

    Clears a scheduled downgrade or cancellation (resume / undo). Chargeless, but it
    re-enables recurring spend → requires billing:manage and honors the kill-switch.
    """
    from hermes_cli.nous_billing import BillingError, delete_subscription_pending_change

    try:
        result = delete_subscription_pending_change()
        return _ok(rid, {"ok": True, "message": result.get("message"), "payload": result})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("subscription.upgrade")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/upgrade → {ok, status, ...} or typed error.

    params: {subscription_type_id: str, idempotency_key?: str}. The single money
    route: prorate + charge the card on the subscription + flip the plan. SCA /
    decline come back as status requires_action / payment_failed with a recovery_url
    to finish in the portal. The idempotency key is minted if absent and echoed so
    the TUI reuses it on retry of the SAME upgrade. Requires billing:manage.
    """
    from agent.billing_view import new_idempotency_key
    from hermes_cli.nous_billing import BillingError, post_subscription_upgrade

    tier_id = params.get("subscription_type_id")
    if not tier_id:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "subscription_type_id is required"})
    key = params.get("idempotency_key") or new_idempotency_key()
    try:
        result = post_subscription_upgrade(subscription_type_id=tier_id, idempotency_key=key)
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "target_tier_name": result.get("targetTierName"),
                "recovery_url": result.get("recoveryUrl"),
                "reason": result.get("reason"),
                "idempotency_key": key,
            },
        )
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key  # so the TUI can reuse on retry
        return _ok(rid, env)
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "idempotency_key": key})


@method("billing.charge")
def _(rid, params: dict) -> dict:
    """POST /api/billing/charge → {ok, chargeId} or a typed error envelope.

    params: {amount_usd: str|number, idempotency_key?: str}. If no key is
    supplied, the server-side core mints a fresh one and returns it so the TUI can
    reuse it on retry of the SAME purchase.
    """
    from hermes_cli.nous_billing import BillingError, post_charge
    from agent.billing_view import new_idempotency_key

    amount = params.get("amount_usd")
    if amount is None:
        return _ok(rid, {"ok": False, "error": "invalid_request", "message": "amount_usd is required"})
    key = params.get("idempotency_key") or new_idempotency_key()
    try:
        result = post_charge(amount_usd=amount, idempotency_key=key)
        return _ok(rid, {"ok": True, "charge_id": result.get("chargeId"), "idempotency_key": key})
    except BillingError as exc:
        env = _serialize_billing_error(exc)
        env["idempotency_key"] = key  # so the TUI can reuse on retry
        return _ok(rid, env)
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "idempotency_key": key})


@method("billing.charge_status")
def _(rid, params: dict) -> dict:
    """GET /api/billing/charge/{id} → {ok, status, ...} or typed error.

    The poll. Caller drives the 2s/5-min cadence; this is a single status read.
    """
    from hermes_cli.nous_billing import BillingError, get_charge_status

    charge_id = params.get("charge_id")
    if not charge_id:
        return _ok(rid, {"ok": False, "error": "invalid_charge_id", "message": "charge_id is required"})
    try:
        result = get_charge_status(charge_id)
        return _ok(
            rid,
            {
                "ok": True,
                "status": result.get("status"),
                "amount_usd": result.get("amountUsd"),
                "settled_at": result.get("settledAt"),
                "reason": result.get("reason"),
            },
        )
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.auto_reload")
def _(rid, params: dict) -> dict:
    """PATCH /api/billing/auto-top-up → {ok:true} or typed error (Screen 2).

    params: {enabled: bool, threshold: number, top_up_amount: number}.
    """
    from hermes_cli.nous_billing import BillingError, patch_auto_top_up

    try:
        enabled = bool(params.get("enabled"))
        threshold = params.get("threshold")
        top_up_amount = params.get("top_up_amount")
        if threshold is None or top_up_amount is None:
            return _ok(rid, {"ok": False, "error": "invalid_request", "message": "threshold and top_up_amount are required"})
        patch_auto_top_up(enabled=enabled, threshold=threshold, top_up_amount=top_up_amount)
        return _ok(rid, {"ok": True})
    except BillingError as exc:
        return _ok(rid, _serialize_billing_error(exc))
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc)})


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """Run the lazy billing:manage step-up device flow → {ok, granted}.

    Triggered by the TUI after a billing call returns error=insufficient_scope.
    Returns granted:false when the server silently downscopes (non-admin / unticked).

    Runs on the thread pool (in _LONG_HANDLERS): the device flow blocks for the
    whole device-code lifetime (minutes), so it must not stall the main stdin loop.
    The verification URL/code reach the TUI via an out-of-band ``billing.step_up.
    verification`` event (a plain print would be dropped by the JSON-RPC stdout
    pipe), and the browser is opened TUI-side via openExternalUrl — never with the
    gateway's headless webbrowser.open (hence open_browser=False).
    """
    sid = params.get("session_id") or ""
    try:
        from hermes_cli.auth import step_up_nous_billing_scope
        from hermes_cli.nous_billing import BillingError

        def _on_verification(url: str, code: str) -> None:
            _emit(
                "billing.step_up.verification",
                sid,
                {"verification_url": url, "user_code": code},
            )

        granted = step_up_nous_billing_scope(
            open_browser=False, on_verification=_on_verification
        )
        return _ok(rid, {"ok": True, "granted": bool(granted)})
    except BillingError as exc:
        # Route typed billing errors (e.g. session_revoked when the token expires
        # mid-device-flow) through the shared spine like the other write handlers,
        # so the TUI maps them to the right copy instead of a generic failure.
        env = _serialize_billing_error(exc)
        env["granted"] = False
        return _ok(rid, env)
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), "granted": False})


@method("session.status")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    from hermes_constants import display_hermes_home

    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = {}
    # Prefer the live session's bound profile db, else params.profile, else launch.
    status_params = dict(params or {})
    if not status_params.get("profile") and session.get("profile_home"):
        # profile_home is a path; still allow _session_db via a synthetic session
        pass
    with _session_db(session) as db:
        if db is None:
            # Fall back to ~params.profile naming for not-yet-mapped sessions.
            with _profile_db(params) as db2:
                db = db2
                if db and key:
                    try:
                        meta = db.get_session(key) or {}
                    except Exception:
                        meta = {}
                db = None  # prevent double-use
        if db is not None and key:
            try:
                meta = db.get_session(key) or {}
            except Exception:
                meta = {}

    def _dt(value, fallback: datetime | None = None) -> datetime:
        if value:
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                pass
        return fallback or datetime.now()

    created = _dt(meta.get("started_at"))
    updated = created
    for field in ("updated_at", "last_updated_at", "last_activity_at"):
        if meta.get(field):
            updated = _dt(meta.get(field), created)
            break

    mirror = _metadata_mirror(session)
    usage = _session_usage_snapshot(session)
    provider = getattr(agent, "provider", None) or mirror.get("provider") or "unknown"
    model = getattr(agent, "model", None) or mirror.get("model") or "(unknown)"
    project = _project_info_for_cwd(_display_session_cwd(session))
    lines = [
        "Hermes TUI Status",
        "",
        f"Session ID: {key}",
        f"Path: {display_hermes_home()}",
    ]
    if project:
        lines.append(f"Project: {project['name']}")
    title = (meta.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")
    lines.extend(
        [
            f"Model: {model} ({provider})",
            f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
            f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
            f"Tokens: {int(usage.get('total') or 0):,}",
            f"Agent Running: {'Yes' if session.get('running') else 'No'}",
        ]
    )
    return _ok(rid, {"output": "\n".join(lines)})


@method("session.history")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    history = list(session.get("history", []))
    if session.get("session_key"):
        with _session_db(session) as db:
            if db is not None:
                try:
                    # include_row_ids: the durable row id is how clients address
                    # a specific persisted turn (reactions, and the Desktop's
                    # content-based truncation-target resolution — #87059). The
                    # projection in _history_to_messages only forwards row_id
                    # when the row carries a stamp, so an unstamped read here
                    # silently strips the one durable address clients can use.
                    history = db.get_messages_as_conversation(
                        session["session_key"],
                        include_ancestors=True,
                        include_row_ids=True,
                    )
                except Exception:
                    pass
    return _ok(
        rid,
        {
            "count": len(history),
            "messages": _history_to_messages(history),
        },
    )


@method("session.undo")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Reject during an in-flight turn.  If we mutated history while
    # the agent thread is running, prompt.submit's post-run history
    # write would either clobber the undo (version matches) or
    # silently drop the agent's output (version mismatch, see below).
    # Neither is what the user wants — make them /interrupt first.
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /undo"
        )
    removed = 0
    with session["history_lock"]:
        history = session.get("history", [])
        # Truncate from the last *real* user turn. Popping only trailing
        # assistant/tool then one user left timeline markers
        # (async_delegation_complete, model_switch, …) or compaction
        # handoffs as the undo target — so session.undo removed
        # bookkeeping instead of the last exchange (#80622).
        # Match list_recent_user_messages / CLI turn counting.
        from agent.context_compressor import is_user_originated_turn

        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if is_user_originated_turn(msg):
                last_user_idx = i
                break
        if last_user_idx is not None:
            removed = len(history) - last_user_idx
            del history[last_user_idx:]
            session["history_version"] = int(session.get("history_version", 0)) + 1
    return _ok(rid, {"removed": removed})


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    assert session is not None
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        focus_topic = str(params.get("focus_topic", "") or "").strip()
        command = "/compress" + (f" {focus_topic}" if focus_topic else "")
        try:
            ack = _send_compute_host_control(
                sid,
                route_name="session.compress",
                command=command,
                wait=True,
                timeout=120.0,
            )
        except Exception as exc:
            return _err(rid, 5019, f"compute-host compress failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 4009, str(ack.get("message") or "compute-host compress failed"))
        _apply_compute_host_metadata_mirror(session, ack)
        host_result = ack.get("result")
        if isinstance(host_result, dict):
            # The host owns the isolated session's agent/history, so preserve
            # its structured compression result verbatim. In particular this
            # carries `status: aborted` and `summary.aborted`; flattening the
            # old text-only acknowledgement made Desktop show aborted work as a
            # success toast.
            return _ok(rid, {**host_result, "turn_isolation": True})
        host_info = ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
        host_messages = _history_to_messages(ack.get("messages")) if isinstance(ack.get("messages"), list) else []
        # `messages` is returned at top level for the desktop transcript
        # replacement. Keep the host acknowledgement metadata, but do not send
        # the same (potentially large) transcript a second time inside it.
        host_ack = {key: value for key, value in ack.items() if key != "messages"}
        return _ok(
            rid,
            {
                "status": "compressed",
                "turn_isolation": True,
                "host_ack": host_ack,
                "info": host_info,
                "messages": host_messages,
                "usage": host_info.get("usage") if isinstance(host_info.get("usage"), dict) else {},
            },
        )
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(
            rid, 4009, "session busy — /interrupt the current turn before /compress"
        )
    from agent.conversation_compression import (
        finalize_context_engine_compression_notification,
    )

    sid = params.get("session_id", "")
    focus_topic = str(params.get("focus_topic", "") or "").strip()
    try:
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough

        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
        before_count = len(before_messages)
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None
        before_tokens = (
            estimate_request_tokens_rough(
                before_messages, system_prompt=_sys_prompt, tools=_tools
            )
            if before_count
            else 0
        )

        if before_count >= 4:
            focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
            _status_update(
                sid,
                "compressing",
                f"⠋ compressing {before_count} messages "
                f"(~{before_tokens:,} tok){focus_suffix}…",
            )

        try:
            removed, usage = _compress_session_history(
                session,
                focus_topic,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                messages = list(session.get("history", []))
            after_count = len(messages)
            # Re-read system prompt + tools after compression — _compress_context
            # may have rebuilt the system prompt (_cached_system_prompt=None).
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            agent = session["agent"]
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            info = _session_info(agent, session)
            _emit("session.info", sid, info)
            finalize_context_engine_compression_notification(
                agent,
                committed=True,
            )
            return _ok(
                rid,
                {
                    "status": "aborted" if summary["aborted"] else "compressed",
                    "removed": removed,
                    "before_messages": before_count,
                    "after_messages": after_count,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "summary": summary,
                    "usage": usage,
                    "info": info,
                    # Keep this identical to session.resume / session.history:
                    # raw tool results can contain large or sensitive payloads
                    # that belong in persisted history, not the transcript
                    # replacement response.
                    "messages": _history_to_messages(messages),
                },
            )
        finally:
            # Always clear the pinned compressing status so the bar
            # reverts to neutral whether compaction succeeded, was a
            # no-op, or raised.
            _status_update(sid, "ready")
    except CompressionLockHeld as e:
        _status_update(sid, "ready")
        from agent.manual_compression_feedback import (
            describe_compression_lock_skip,
        )
        return _ok(rid, {
            "compressed": False,
            "lock_held": True,
            "message": describe_compression_lock_skip(e.holder),
        })
    except Exception as e:
        finalize_context_engine_compression_notification(
            session["agent"],
            committed=False,
        )
        return _err(rid, 5005, str(e))


@method("session.save")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        try:
            ack = _send_compute_host_control(
                sid,
                route_name="session.save",
                wait=True,
            )
        except Exception as exc:
            return _err(rid, 5011, f"compute-host session save failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 5011, str(ack.get("message") or "compute-host session save failed"))
        result = ack.get("result")
        if not isinstance(result, dict):
            return _err(rid, 5011, "compute-host session save returned an invalid response")
        return _ok(rid, result)

    agent = session["agent"]
    # Mirror the classic CLI /save: snapshot under the Hermes profile home
    # (~/.hermes/sessions/saved/) rather than the project/workspace CWD, and
    # include the system prompt so the export matches the dashboard save.
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = saved_dir / f"hermes_conversation_{timestamp}.json"

    with session["history_lock"]:
        messages = list(session.get("history", []))

    session_id = getattr(agent, "session_id", None) or session.get("session_key") or ""
    # Prefer the agent's session_start datetime (matches the classic CLI export);
    # fall back to the gateway session's created_at timestamp.
    agent_start = getattr(agent, "session_start", None)
    if isinstance(agent_start, datetime):
        session_start = agent_start.isoformat()
    else:
        created_at = session.get("created_at")
        session_start = (
            datetime.fromtimestamp(created_at).isoformat()
            if isinstance(created_at, (int, float))
            else ""
        )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": getattr(agent, "model", ""),
                    "session_id": session_id,
                    "session_start": session_start,
                    "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                    "messages": messages,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return _ok(rid, {"file": str(path)})
    except Exception as e:
        return _err(rid, 5011, str(e))


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = params.get("session_id", "")
    # Serialize only the ownership claim against session.resume / the orphan
    # reaper. Finalization may run arbitrary plugin/agent cleanup and must not
    # keep every unrelated session.resume waiting behind it.
    with _session_resume_lock:
        session = _pop_session_by_id(sid)
    closed = _teardown_popped_session(session, end_reason="tui_close")
    return _ok(rid, {"closed": closed})


@method("session.branch")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    # Branch must write into the parent's profile-scoped state.db (app-global
    # remote mode). Using the launch handle would orphan branch rows + history.
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        with session["history_lock"]:
            in_memory_history = [
                dict(msg)
                for msg in list(session.get("display_history_prefix") or []) + list(session.get("history", []))
                if isinstance(msg, dict)
            ]

        def _visible_branch_history(messages):
            visible = []
            for message in messages or []:
                if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                    continue
                if not _coerce_message_text(message.get("content")).strip():
                    continue
                # Keep the FULL row — the copy loop below preserves reasoning
                # fields and timeline-marker tags (display_kind/display_metadata,
                # #82756); a minimal role/content copy would silently drop them.
                visible.append(dict(message))
            return visible

        # The live session history is the model projection. After compaction it
        # may contain only a summary and the protected tail, while the persisted
        # display projection still contains the complete visible transcript. A
        # branch must snapshot the latter; otherwise the child permanently loses
        # every turn archived before the fork.
        history = None
        get_resume_conversations = getattr(db, "get_resume_conversations", None)
        if callable(get_resume_conversations):
            try:
                _, display_history = get_resume_conversations(old_key)
                display_history = _reconcile_display_with_live(display_history, in_memory_history)
                history = _visible_branch_history(display_history)
            except Exception:
                logger.debug("branch display projection read failed", exc_info=True)
        if not history:
            history = _visible_branch_history(in_memory_history)
        if not history:
            return _err(rid, 4008, "nothing to branch — send a message first")
        count = params.get("count")
        if isinstance(count, int) and count > 0:
            history = history[:count]
        new_key = _new_session_key()
        new_sid = uuid.uuid4().hex[:8]
        source = _session_source(session)
        lease = None  # claimed lazily on the first turn (_ensure_active_session_slot)
        branch_name = params.get("name", "")
        try:
            if branch_name:
                title = branch_name
            else:
                current = db.get_session_title(old_key) or "branch"
                title = (
                    db.get_next_title_in_lineage(current)
                    if hasattr(db, "get_next_title_in_lineage")
                    else f"{current} (branch)"
                )
            db.create_session(
                new_key,
                source=source,
                model=_resolve_model(),
                # Stable _branched_from marker so list_sessions_rich() keeps the
                # branch visible in /resume and /sessions. The TUI branch leaves
                # the parent live (no end_reason='branched'), so the legacy
                # end_reason heuristic never matches it — the marker is the only
                # thing that surfaces TUI branches. See issue #20856.
                model_config={"_branched_from": old_key},
                parent_session_id=old_key,
                cwd=_session_cwd(session),
                # The branch stays on its parent's profile. Explicit stamp (not
                # just the parent-backfill) so it holds even when the parent row
                # predates the profile_name column.
                profile_name=(
                    Path(session["profile_home"]).name
                    if session.get("profile_home")
                    else None
                ),
            )
            # Copy the whole parent history in bounded-chunk transactions —
            # a branch seed can be hundreds of rows, and per-row transactions
            # were the write-amplification pattern removed in #23254.
            db.append_messages_batch(
                new_key,
                [
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content"),
                        "reasoning": msg.get("reasoning"),
                        "reasoning_content": msg.get("reasoning_content"),
                        "reasoning_details": msg.get("reasoning_details"),
                        "codex_reasoning_items": msg.get("codex_reasoning_items"),
                        "codex_message_items": msg.get("codex_message_items"),
                        # Timeline markers (model_switch, personality_switch,
                        # auto_continue, …) ride as role=user; dropping the tag
                        # here re-planted them as bare user turns after a
                        # restart, corrupting the truncate ordinal address
                        # space the same way #82756 did.
                        "display_kind": msg.get("display_kind"),
                        "display_metadata": msg.get("display_metadata"),
                        # Preserve the parent's original message timestamps —
                        # branch copies are history, not new activity (9d73006ad).
                        "timestamp": msg.get("timestamp"),
                    }
                    for msg in history
                ],
                chunk_rows=500,
            )
            db.set_session_title(new_key, title)
        except Exception as e:
            if lease is not None:
                lease.release()
            return _err(rid, 5008, f"branch failed: {e}")
    # Bound before the try so the ownership finally below can never see them
    # unbound, whatever raises inside.
    branch_db = None
    branch_owns_db = False
    try:
        # Bind the branched AGENT to the parent's profile, mirroring
        # session.create/resume: home override so config/skills/memory resolve
        # to the profile during the build, and the profile's own state.db
        # handle so the live agent's message flushes — and any later
        # compression rotation — persist there. Writing only the row to the
        # parent's db while the agent stayed on the launch handle would
        # recreate the cross-profile split one turn later.
        parent_home = session.get("profile_home")
        if parent_home:
            from hermes_state import SessionDB

            # DEDICATED handle, same ownership rule as session.resume: ours
            # until the branched agent takes it below. _make_agent raising, or
            # _init_session raising, both leave here without that transfer.
            branch_db = SessionDB(db_path=Path(parent_home) / "state.db")
            branch_owns_db = True
        home_token = (
            set_hermes_home_override(parent_home) if parent_home else None
        )
        # The home override alone only moves config/skills/memory; credentials
        # resolve through get_secret(), which without a scope falls through to
        # process os.environ — the LAUNCH profile's .env. Install the parent's
        # secret scope for the build, exactly as session.create/resume do
        # (#67605), so the branched agent authenticates as its own profile.
        secret_token = (
            set_secret_scope(build_profile_secret_scope(Path(parent_home)))
            if parent_home
            else None
        )
        try:
            tokens = _set_session_context(new_key)
            try:
                agent = _make_agent(
                    new_sid,
                    new_key,
                    session_id=new_key,
                    session_db=branch_db,
                    platform_override=source,
                )
            finally:
                _clear_session_context(tokens)
            _init_session(
                new_sid,
                new_key,
                agent,
                list(history),
                cols=session.get("cols", 80),
                cwd=_session_cwd(session),
                session_db=branch_db,
                source=source,
                profile_home=parent_home,
            )
            # Ownership TRANSFER — the branched session's agent holds this
            # handle for its whole life and closes it on teardown. Drop is
            # unconditional for the same reason as session.resume: past
            # _init_session the branched session is registered against this
            # handle, so the finally must not close it.
            _transfer_db_to_agent(agent, branch_db)
            branch_owns_db = False
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
            if home_token is not None:
                reset_hermes_home_override(home_token)
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = lease
    except Exception as e:
        if lease is not None:
            lease.release()
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    finally:
        if branch_owns_db and branch_db is not None:
            with contextlib.suppress(Exception):
                branch_db.close()
    branched_session = _sessions.get(new_sid)
    return _ok(
        rid,
        {
            "session_id": new_sid,
            "stored_session_id": new_key,
            "title": title,
            "parent": old_key,
            "message_count": len(history),
            "messages": _history_to_messages(history),
            "info": _session_info(agent, branched_session),
        },
    )


@method("session.interrupt")
def _(rid, params: dict) -> dict:
    # Keypress barge-in: stopping the turn also silences its streaming TTS
    # (voice is process-global, so no per-session scoping is needed).
    _tts_stream_stop()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if _session_uses_compute_host(session):
        sid = str(params.get("session_id") or "")
        if session.get("running"):
            try:
                _get_compute_host_supervisor().interrupt(sid, request_id=f"interrupt-{rid}")
            except Exception as exc:
                return _err(rid, 5019, f"compute-host interrupt failed: {exc}")
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
            session["queued_prompt"] = None
            session.pop("queued_prompts", None)
            session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
        _clear_pending(sid)
        try:
            from tools.approval import resolve_gateway_approval

            resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
        except Exception:
            pass
        return _ok(rid, {"status": "interrupted", "turn_isolation": True})
    session, err = _sess(params, rid)
    if err:
        return err
    # Safety net: if the turn's run thread is already gone but `running` stayed
    # stuck (a crash/desync that skipped the run loop's `finally`), force-clear it
    # so the session can't be permanently bricked at 4009 "session busy" — every
    # send/restore/resume would otherwise reject until a full backend restart.
    # Always tell the agent to interrupt when the session claims a run is active:
    # stale flags are cleared below, and fresh turns clear the interrupt flag at
    # entry. This keeps a stale/missing thread handle from making Stop a no-op.
    run_thread = session.get("_run_thread")
    run_thread_alive = run_thread is not None and run_thread.is_alive()
    should_interrupt = bool(session.get("running"))
    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    if should_interrupt:
        from agent.interrupt_compat import request_hard_interrupt

        request_hard_interrupt(session["agent"])
    if not run_thread_alive:
        with session["history_lock"]:
            if session.get("running"):
                session["running"] = False
                _clear_inflight_turn(session)

    # Stop = stop the TURN (cooperative interrupt above also kills the in-flight
    # foreground subprocess). Background processes the agent started (dev servers,
    # watchers) are intentionally left running — kill those individually with the
    # "x" on the task row (process.kill). Don't reap them here.
    # Scope the pending-prompt release to THIS session.  A global
    # _clear_pending() would collaterally cancel clarify/sudo/secret
    # prompts on unrelated sessions sharing the same tui_gateway
    # process, silently resolving them to empty strings.
    _clear_pending(params.get("session_id", ""))
    try:
        from tools.approval import resolve_gateway_approval

        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return _ok(rid, {"status": "interrupted"})


@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import (
        is_spawn_paused,
        list_active_subagents,
        _get_max_concurrent_children,
        _get_max_spawn_depth,
    )

    return _ok(
        rid,
        {
            "active": list_active_subagents(),
            "paused": is_spawn_paused(),
            "max_spawn_depth": _get_max_spawn_depth(),
            "max_concurrent_children": _get_max_concurrent_children(),
        },
    )


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused

    paused = bool(params.get("paused", True))
    return _ok(rid, {"paused": set_spawn_paused(paused)})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    ok = interrupt_subagent(subagent_id)
    return _ok(rid, {"found": ok, "subagent_id": subagent_id})


@method("subagent.steer")
def _(rid, params: dict) -> dict:
    """Queue steering text into a live delegated child without stopping it.

    The redirection-side mirror of subagent.interrupt: resolves the child in
    the delegation registry and calls AIAgent.steer(), which appends the text
    to the child's last tool result at its next iteration boundary — the
    in-flight tool call is never cut. "queued" is not "delivered": a child
    already past its final tool batch has no boundary left to drain into,
    and that race surfaces as ``missed_steer`` on the parent's completion
    entry instead of being silently dropped.
    """
    from tools.delegate_tool import steer_subagent

    subagent_id = str(params.get("subagent_id") or "").strip()
    if not subagent_id:
        return _err(rid, 4000, "subagent_id required")
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    _invoking_session, err = _sess_nowait(params, rid)
    if err:
        return err
    invoking_session_id = str(params.get("session_id") or "").strip()
    invoking_transport, invoking_session = _current_session_steer_authority(
        invoking_session_id
    )
    queued = False
    if invoking_transport is not None and invoking_session is not None:
        queued = steer_subagent(
            subagent_id,
            text,
            owner_session_id=invoking_session_id,
            owner_transport=invoking_transport,
            owner_session_record=invoking_session,
        )
    return _ok(
        rid,
        {
            "status": "queued" if queued else "rejected",
            "subagent_id": subagent_id,
            "text": text,
        },
    )


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")

    from datetime import datetime

    started_at = params.get("started_at")
    finished_at = params.get("finished_at") or time.time()
    label = str(params.get("label") or "")
    ts = datetime.utcfromtimestamp(float(finished_at)).strftime("%Y%m%dT%H%M%S")
    fname = f"{ts}.json"
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / fname
    try:
        payload = {
            "session_id": session_id,
            "started_at": float(started_at) if started_at else None,
            "finished_at": float(finished_at),
            "label": label,
            "subagents": subagents,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")

    _append_spawn_tree_index(
        d,
        {
            "path": str(path),
            "session_id": session_id,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "label": label,
            "count": len(subagents),
        },
    )

    return _ok(rid, {"path": str(path), "session_id": session_id})


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = str(params.get("session_id") or "").strip()
    limit = int(params.get("limit") or 50)
    cross_session = bool(params.get("cross_session"))

    if cross_session:
        root = _spawn_trees_root()
        roots = [p for p in root.iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]

    entries: list[dict] = []
    for d in roots:
        indexed = _read_spawn_tree_index(d)
        if indexed:
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(
                e for e in indexed if (p := e.get("path")) and Path(p).exists()
            )
            continue

        # Fallback for legacy (pre-index) sessions: full scan.  O(N) reads
        # but only runs once per session until the next save writes the index.
        for p in d.glob("*.json"):
            if p.name == _SPAWN_TREE_INDEX:
                continue
            try:
                stat = p.stat()
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    raw = {}
                subagents = raw.get("subagents") or []
                entries.append(
                    {
                        "path": str(p),
                        "session_id": raw.get("session_id") or d.name,
                        "finished_at": raw.get("finished_at") or stat.st_mtime,
                        "started_at": raw.get("started_at"),
                        "label": raw.get("label") or "",
                        "count": len(subagents) if isinstance(subagents, list) else 0,
                    }
                )
            except OSError:
                continue

    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:limit]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    from pathlib import Path

    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return _err(rid, 4000, "path required")

    # Reject paths escaping the spawn-trees root.
    root = _spawn_trees_root().resolve()
    try:
        resolved = Path(raw_path).resolve()
        resolved.relative_to(root)
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")

    return _ok(rid, payload)


@method("session.steer")
def _(rid, params: dict) -> dict:
    """Inject a user message into the next tool result without interrupting.

    Mirrors AIAgent.steer(). Safe to call while a turn is running — the text
    lands on the last tool result of the next tool batch and the model sees
    it on its next iteration. No interrupt, no new user turn, no role
    alternation violation.
    """
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if agent is None or not hasattr(agent, "steer"):
        return _err(rid, 4010, "agent does not support steer")
    try:
        accepted = agent.steer(text)
    except Exception as exc:
        return _err(rid, 5000, f"steer failed: {exc}")
    if accepted:
        # Record the correction on the live turn exactly like session.redirect
        # does. Without this, a resume/reconnect while the turn is running
        # rebuilds the transcript from the inflight snapshot and the steered
        # text has no user bubble — the "my message vanished on reload" loss.
        with session["history_lock"]:
            _record_inflight_correction(session, text)
            # #84417: steer does not cancel the live original, but a server
            # queue self-copy of that original must still not re-fire after
            # settle (same class as redirect).
            _drop_queued_duplicates_of_inflight_user(session)
            session["last_active"] = time.time()
    return _ok(rid, {"status": "queued" if accepted else "rejected", "text": text})


@method("session.redirect")
def _(rid, params: dict) -> dict:
    """Redirect the active model turn while preserving valid work/context."""
    text = (params.get("text") or "").strip()
    if not text:
        return _err(rid, 4002, "text is required")
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    # Turn-build window: a fresh turn flips running=True and kicks off an async
    # agent build, so session["agent"] is briefly None. That is not an
    # unsupported runtime — queue the correction server-side so it reaches the
    # model as the next turn, instead of a misleading 4010 the client silently
    # swallows into a lost follow-up.
    if agent is None and session.get("running"):
        _enqueue_prompt(session, text, current_transport() or _stdio_transport)
        session["last_active"] = time.time()
        return _ok(rid, {"status": "queued", "text": text})
    if (
        agent is None
        or getattr(agent, "_supports_active_turn_redirect", False) is not True
        or not hasattr(agent, "redirect")
    ):
        return _err(rid, 4010, "agent does not support active-turn redirect")
    try:
        accepted = agent.redirect(text)
    except Exception as exc:
        return _err(rid, 5000, f"redirect failed: {exc}")
    if accepted:
        with session["history_lock"]:
            _record_inflight_correction(session, text)
            # #84417: purge server-queue self-duplicates of the live original
            # so post-turn drain cannot restart the pre-correction prompt.
            _drop_queued_duplicates_of_inflight_user(session)
            session["last_active"] = time.time()
    return _ok(
        rid,
        {"status": "redirected" if accepted else "rejected", "text": text},
    )


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    session["cols"] = int(params.get("cols", 80))
    return _ok(rid, {"cols": session["cols"]})


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
