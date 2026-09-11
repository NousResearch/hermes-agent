"""TUI session_registry seam; functions bind to the server namespace at registration."""

from __future__ import annotations

from .method_ctx import bind_module

def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    return _err(rid, 5032, err) if (err := session.get("agent_error")) else None


# The deferred prompt path waits in short slices so a cancel is honored promptly and a slow
# build is reported to the client exactly once.
_AGENT_BUILD_WAIT_SLICE = 5.0
_AGENT_BUILD_SLOW_NOTICE_AFTER = 30.0
_AGENT_BUILD_SLOW_NOTICE_KEY = "agent-build-slow"


def _agent_build_wait_cap() -> float:
    """Seconds a submitted prompt waits for the deferred build before failing; ``agent.build_wait_timeout``
    overrides the 600s default (raise it for many slow MCP servers / high-latency provider metadata)."""
    with contextlib.suppress(Exception):
        raw = (_load_cfg().get("agent") or {}).get("build_wait_timeout")
        if raw is not None and float(raw) > 0:
            return float(raw)
    return 600.0


def _wait_agent_for_prompt(session: dict, rid: str, sid: str) -> dict | None:
    """Patient ``_wait_agent`` for deferred prompt.submit: the client already got ``streaming`` and the
    first message IS the turn, while a cold build routinely outlives the flat 30s ceiling (timing out
    silently discarded it). Waits in short slices (cancel honored promptly), notifies once (keyed) past
    ``_AGENT_BUILD_SLOW_NOTICE_AFTER``, fails only on a dead build thread or the bounded cap.
    Returns None on success OR cancel mid-wait (the caller's cancel branch owns that messaging).

    The flat 30s ``_wait_agent`` ceiling was a message-eating cliff (#63078): ``prompt.submit`` has already
    returned ``{"status": "streaming"}``, the user's first message IS the turn in flight, and the deferred
    agent build (MCP discovery with per-server retry backoff, synchronous model-metadata HTTP, skills
    scanning) routinely outlives 30 seconds on cold starts. On timeout the old path emitted an error EVENT
    and returned without ever calling ``_run_prompt_submit`` — the first message was permanently discarded
    while the build finished successfully in the background, leaving the blank first session.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return None
    start, cap, notified_slow = time.monotonic(), _agent_build_wait_cap(), False
    while not ready.wait(timeout=_AGENT_BUILD_WAIT_SLICE):
        with session["history_lock"]:
            cancelled = session.get("_turn_cancel_requested") or not session.get("running")
        if cancelled:
            return None
        waited = time.monotonic() - start
        if waited >= cap:
            return _err(rid, 5032, f"agent initialization timed out after {int(waited)}s — "
                        "your message was not sent; retry once the session is ready")
        build_thread = session.get("_agent_build_thread")
        if build_thread is not None and not build_thread.is_alive() and not ready.is_set():
            # _build's finally guarantees ready.set(); dead thread + unset ready = died hard.
            return _err(rid, 5032, session.get("agent_error") or "agent initialization failed before completing")
        if not notified_slow and waited >= _AGENT_BUILD_SLOW_NOTICE_AFTER:
            notified_slow = True  # one keyed, replace-in-place notice (toast / status bar)
            _emit("notification.show", sid, {
                "text": "Still starting the agent (tool discovery / model setup) — your message will be sent as soon as it's ready.",
                "level": "info", "kind": "agent", "ttl_ms": None,
                "key": _AGENT_BUILD_SLOW_NOTICE_KEY, "id": _AGENT_BUILD_SLOW_NOTICE_KEY})
    if notified_slow:
        _emit("notification.clear", sid, {"key": _AGENT_BUILD_SLOW_NOTICE_KEY})
    return _err(rid, 5032, err) if (err := session.get("agent_error")) else None


def _bind_build_profile_scopes(profile_home: str) -> "_TurnScopes":
    """Bind a session profile's HERMES_HOME / secret / terminal scopes for an agent build. Fail-open per
    scope (the build must not die on a scope helper); the terminal installer itself fails closed (malformed
    policy → refusal scope) so _make_agent's terminal probing / cwd hints resolve the routed profile."""
    scopes = _TurnScopes()
    scopes.home = set_hermes_home_override(profile_home)
    with contextlib.suppress(Exception):
        scopes.secret = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    scopes.terminal = None
    with contextlib.suppress(Exception):
        from tools.terminal_scope import install_profile_terminal_scope
        scopes.terminal = install_profile_terminal_scope(Path(profile_home))
    return scopes


def _release_build_profile_scopes(scopes: "_TurnScopes") -> None:
    if scopes.home is not None:
        reset_hermes_home_override(scopes.home)
    if scopes.secret is not None:
        with contextlib.suppress(Exception):
            reset_secret_scope(scopes.secret)
    if scopes.terminal is not None:
        with contextlib.suppress(Exception):
            from tools.terminal_scope import reset_terminal_scope
            reset_terminal_scope(scopes.terminal)


def _deferred_build_agent_kwargs(current: dict, session_db) -> dict:
    """_make_agent kwargs for a deferred (first-prompt) build. A lazy-resumed (watch) session carries the
    stored conversation id so the upgrade continues it; a cold deferred resume restores the full persisted
    runtime identity (like the eager resume's overrides splat) so the build can't drop the provider. No
    stored runtime, or an unroutable provider → this session's picked model/effort/tier, else the default."""
    kw = {"session_db": session_db, "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(current),
          "platform_override": _session_source(current)}
    if resume_sid := current.get("resume_session_id"):
        kw["session_id"] = resume_sid
    resume_overrides = current.get("resume_runtime_overrides")
    if isinstance(resume_overrides, dict) and resume_overrides and _overrides_have_routable_provider(resume_overrides):
        kw.update(resume_overrides)
    else:
        if override := current.get("model_override"):
            kw["model_override"] = override
        kw.update({k: v for k, v in (("reasoning_config_override", current.get("create_reasoning_override")),
                                     ("service_tier_override", current.get("create_service_tier_override")))
                   if v is not None})
    return kw


def _wire_session_agent(sid: str, key: str, agent) -> bool:
    """Post-build wiring; returns whether the approval notify got registered. Approval prompts route to the
    client; the self-improvement "💾 …" summary is emitted as review.summary (no print surface), honoring
    display.memory_notifications."""
    notify_registered = False
    with contextlib.suppress(Exception):
        from tools.approval import load_permanent_allowlist, register_gateway_notify
        register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))
        notify_registered = True
        load_permanent_allowlist()
    _wire_callbacks(sid)
    with contextlib.suppress(Exception):  # bare agents without the attribute must not break startup
        agent.background_review_callback = lambda message, _sid=sid: _emit("review.summary", _sid, {"text": str(message)})
        agent.memory_notifications = _load_memory_notifications()
    return notify_registered


def _start_session_services(sid: str, key: str, current: dict) -> None:
    """Start the notification poller and fire the session-reset boundary hook."""
    with _sessions_lock:
        if (rec := _sessions.get(sid)) is not None:
            rec["_notif_stop"] = _start_notification_poller(sid, rec)
    _notify_session_boundary("on_session_reset", key, _session_source(current))


def _await_resume_history(sid: str, current: dict) -> bool:
    """Block on a cold resume's transcript hydration; False when this record was replaced meanwhile."""
    history_ready = current.get("resume_history_ready")
    if history_ready is None:
        return True
    if not history_ready.wait(timeout=300.0):
        raise TimeoutError("session history hydration timed out")
    if history_error := current.get("resume_history_error"):
        raise RuntimeError(str(history_error))
    with _sessions_lock:
        return _sessions.get(sid) is current


def _attach_built_agent(current: dict, agent) -> None:
    """Attach a freshly built agent to its live record (session DB row deferred to first run_conversation())."""
    # Bot Mode gate hint: the DB title lands post-first-turn but the system prompt builds at turn START.
    if _title_hint := str(current.get("pending_title") or "").strip():
        agent._session_title_hint = _title_hint
    current["agent"] = agent
    _session_todo_state(current)
    # Baseline for the per-turn config sync (profile home override still active).
    current["config_model_seen"] = _config_model_target()


def _announce_built_agent(sid: str, key: str, current: dict, agent) -> None:
    """Post-wiring tail of a build: credits seed, session services, session.info, late MCP catch-up."""
    # Credits notices at session OPEN (notice_callback already wired) so depletion warnings show at "ready".
    with contextlib.suppress(Exception):
        from agent.credits_tracker import seed_credits_at_session_start
        seed_credits_at_session_start(agent)
    _start_session_services(sid, key, current)
    info = _session_info(agent, current)
    if cfg_warn := _probe_config_health(_load_cfg()):
        info["config_warning"] = cfg_warn
        logger.warning(cfg_warn)
    _emit("session.info", sid, info)
    _schedule_mcp_late_refresh(sid, agent)  # servers slower than the bounded discovery wait land here


def _finish_agent_build(sid: str, key: str, current: dict, *, notify_registered: bool, scopes, session_db) -> None:
    """Release build scopes and settle ownership of the late notify registration + dedicated db handle."""
    if scopes is not None:
        _release_build_profile_scopes(scopes)
    # Reaped mid-build: _attach_worker closed the worker; only a late notify registration can still
    # leak (session.close unregistered before _build registered).
    with _sessions_lock:
        replaced = _sessions.get(sid) is not current
    if replaced and notify_registered:
        with contextlib.suppress(Exception):
            from tools.approval import unregister_gateway_notify
            unregister_gateway_notify(key)
    # Dedicated profile handle: hand it to the agent that will be torn down, else close it (build
    # failed, or `replaced`: this agent is discarded and _teardown_session never reaches it).
    if session_db is not None and not _transfer_db_to_agent(None if replaced else current.get("agent"), session_db):
        with contextlib.suppress(Exception):
            session_db.close()


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once. Deferred until the first prompt (or any
    command needing the agent) so the composer isn't blocked on tool discovery / model metadata;
    the ready/error event contract is unchanged."""
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the subagent live-mirror keeps
    # flowing (it bails once agent is set); incidental RPCs via _sess() would upgrade it mid-stream.
    if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
        return
    with session.setdefault("agent_build_lock", threading.Lock()):
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True
        session.pop("lazy", None)  # now genuinely mid-construction: restore the "still starting" eviction exemption
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return
        notify_registered, scopes, session_db = False, None, None
        profile_home = current.get("profile_home")
        try:
            if not _await_resume_history(sid, current):
                return
            tokens = _set_session_context(key)
            # Global-remote: bind the session profile's HERMES_HOME and hand the agent that profile's db —
            # DEDICATED and ours until _transfer_db_to_agent in the finally; FAIL CLOSED rather than
            # binding the launch DB and bleeding rows into the wrong state.db.
            if profile_home:
                scopes = _bind_build_profile_scopes(profile_home)
                session_db = _open_profile_session_db(profile_home)
            try:
                from tui_gateway.entry import ensure_mcp_discovery_started
                ensure_mcp_discovery_started()
            except Exception:
                logger.warning("MCP discovery startup failed", exc_info=True)
            try:
                agent = _make_agent(sid, key, **_deferred_build_agent_kwargs(current, session_db))
            finally:
                _clear_session_context(tokens)
            _attach_built_agent(current, agent)
            # No eager slash-worker pre-warm (slash.exec spawns on demand): each worker forks the full stdio
            # MCP fleet, and live-transport sessions are never reaped, so fleets would accumulate.
            notify_registered = _wire_session_agent(sid, key, agent)
            _announce_built_agent(sid, key, current, agent)
        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            _finish_agent_build(
                sid, key, current, notify_registered=notify_registered, scopes=scopes, session_db=session_db)
            ready.set()

    build_thread = threading.Thread(target=_build, daemon=True)
    # _wait_agent_for_prompt handle: dead thread + unset agent_ready = died hard; waiters must not sit out the cap.
    session["_agent_build_thread"] = build_thread
    build_thread.start()


def _sess_nowait(params, rid):
    sid = params.get("session_id") or ""
    s = _sessions.get(sid)
    if s:
        return (s, None)
    # Stale runtime id (reaped/evicted/TTL): the client should session.resume the STORED id. Logged so
    # "message vanished" reads as "arrived and was rejected".
    logger.warning("session-scoped RPC rejected: method=%s session_id=%r not in memory "
                   # A session-scoped RPC hit a runtime id the gateway no longer holds (detached on WS
                   # disconnect and orphan-reaped, LRU-evicted, or torn down after an idle TTL). The client
                   # is expected to recover via session.resume on the STORED session id, but a plain
                   # stale-id send leaves no trace anywhere when the resume never fires — every RPC in this
                   # class returned a silent 4001. Log it so a "message vanished" report is diagnosable as
                   # "request arrived and was rejected" instead of "request never arrived" (see #90428).
                   "(detached/reaped runtime; client should resume the stored session), rid=%r",
                   _current_rpc_method.get() or "?", sid, rid)
    return (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_building(params, rid)
    return (None, err) if err else (s, _wait_agent(s, rid))


def _sess_building(params, rid):
    """Resolve a session and warm its agent build WITHOUT waiting — for the attach RPCs (image/file/pdf,
    clipboard.paste, image.detach), which only touch creation-time fields and run inline on the socket
    reader thread, where waiting on a cold build stalled every RPC behind it ("text is instant, images hang")."""
    s, err = _sess_nowait(params, rid)
    if not err:
        _start_agent_build(params.get("session_id") or "", s)
    return (None, err) if err else (s, None)

def _init_session(
    sid: str, key: str, agent, history: list, cols: int = 80, cwd: str | None = None,
    session_db=None, source: str | None = None, profile_home: str | None = None,
    explicit_cwd: bool = False):
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": agent, "session_key": key, "history": history, "history_lock": threading.Lock(),
            "history_version": 0, "inflight_turn": None, "created_at": now, "last_active": now,
            "running": False, "attached_images": [], "image_counter": 0, "cwd": cwd or _completion_cwd(),
            "explicit_cwd": bool(explicit_cwd), "cols": cols, "slash_worker": None,
            "show_reasoning": _load_show_reasoning(), "source": _resolve_session_source(source),
            "tool_progress_mode": _load_tool_progress_mode(), "edit_snapshots": {}, "tool_started_at": {},
            # Profile-scoped HERMES_HOME (None = launch); SessionBranch copies the parent's (same state.db).
            "profile_home": profile_home,
            # In-session /model switch, honored on rebuild (/new, resume) — never leaks to siblings via env vars.
            "model_override": None,
            # Async events go to the transport that created the session (stdio for Ink, WS for the dashboard).
            "transport": current_transport() or _stdio_transport,
        }
        _session_todo_state(_sessions[sid])
    _hydrate_session_cwd(sid, key, session_db, profile_home)
    _register_session_cwd(_sessions[sid])
    _wire_session_agent(sid, key, agent)  # no eager slash-worker pre-warm (see _start_agent_build)
    _start_session_services(sid, key, _sessions.get(sid, {}))
    _emit("session.info", sid, _session_info(agent, _sessions.get(sid, {})))
    _schedule_mcp_late_refresh(sid, agent)


def _new_session_key() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _with_checkpoints(session, fn):
    return fn(session["agent"]._checkpoint_mgr, _session_cwd(session))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")


# ── Methods: session ─────────────────────────────────────────────────


def _lazy_resume_info(cwd: str, *, model: str = "", provider: str = "", profile: str | None = None) -> dict:
    """session.info for a not-yet-built session (session.create's shape); tools/skills land with the deferred build."""
    return {
        "cwd": cwd, "branch": git_probe.branch(cwd), "project": _project_info_for_cwd(cwd),
        "model": model or _resolve_model(), "tools": {}, "skills": {}, "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT, "profile_name": _response_profile_name(profile),
        **({"provider": provider} if provider else {}),
    }


def _deferred_session_record(
    session_key: str, *, cols: int, cwd: str, history: list, lease, source: str = "tui",
    close_on_disconnect: bool = False, display_history_prefix: list | None = None,
    profile_home: Path | None = None, lazy: bool = False, model_override=None,
    resume_runtime_overrides: dict | None = None, todo_state: dict | None = None,
    explicit_cwd: bool = False) -> dict:
    """A live-session record whose AIAgent is built later (lazy watch / cold resume) — _init_session's shape minus the agent."""
    now = time.time()
    return {
        "agent": None, "agent_error": None, "agent_ready": threading.Event(), "attached_images": [],
        "close_on_disconnect": close_on_disconnect, "active_session_lease": lease, "cols": cols,
        "created_at": now, "cwd": cwd, "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {}, "explicit_cwd": bool(explicit_cwd), "history": history,
        "history_lock": threading.Lock(), "history_version": 0, "image_counter": 0,
        "inflight_turn": None, "last_active": now, "lazy": lazy, "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides, "resume_session_id": session_key,
        "running": False, "session_key": session_key, "show_reasoning": _load_show_reasoning(),
        "slash_worker": None, "source": source, "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {}, "todo_state": todo_state,
        "transport": current_transport() or _stdio_transport,
    }


_ANY_PROFILE = object()  # default: match a live session regardless of profile


def _live_profile_matches(session: dict, profile_home) -> bool:
    """True when ``session`` belongs to ``profile_home`` (None = launch profile; a record with no
    ``profile_home`` is the launch profile's). ``_ANY_PROFILE`` disables the check."""
    if profile_home is _ANY_PROFILE:
        return True
    return (session.get("profile_home") or None) == (str(profile_home) if profile_home else None)


def _claim_or_reuse_live(sid: str, session_key: str, record: dict, lease) -> tuple[str, dict] | None:
    """Register ``record`` as the live session for ``session_key`` under the resume lock, or — if a
    concurrent resume already won — release ``lease`` and return the winner for the caller to reuse."""
    # A live runtime of the same stored id under ANOTHER profile is not a winner to reuse.
    # See #100029.
    profile_home = record.get("profile_home")
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key, profile_home)
        if live is not None:
            if lease is not None:
                lease.release()
            # The reap is cancelled by the guarded reuse (_reattach_refusal), not here: a rejected
            # reattach must leave an in-flight orphan interrupt polling.
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
        # A PRIOR runtime for this stored id may still be sentinel-parked with a reap Timer armed; cancel +
        # finalize it quietly so the reap doesn't broadcast session.reclaimed (storm).
        _cancel_ws_orphan_reap(sid)
        stale = _claim_parked_runtimes(session_key, keep_sid=sid, profile_home=profile_home)
    _finalize_superseded_runtimes(stale)  # slow finalization stays OUTSIDE _session_resume_lock
    return None


def _claim_parked_runtimes(session_key: str, *, keep_sid: str, profile_home=_ANY_PROFILE) -> list[tuple[str, dict]]:
    """Claim sentinel-parked stale runtimes of ``session_key`` for supersession: cancel their orphan-reap
    Timer and pop them here (under the caller's _session_resume_lock); the caller finalizes after release."""
    stale: list[tuple[str, dict]] = []
    with _sessions_lock:
        candidates = [
            (old_sid, old) for old_sid, old in list(_sessions.items())
            if old_sid != keep_sid and not old.get("_finalized")
            and _session_lookup_key(old, fallback=old_sid) == session_key
            and _live_profile_matches(old, profile_home) and old.get("transport") is _detached_ws_transport]
    for old_sid, _old in candidates:
        _cancel_ws_orphan_reap(old_sid)
        if (popped := _pop_session_by_id(old_sid)) is not None:
            stale.append((old_sid, popped))
    return stale


def _finalize_superseded_runtimes(stale: list[tuple[str, dict]]) -> None:
    """end_reason ``superseded_by_resume`` is deliberately NOT in _RECLAIM_END_REASONS (no ``session.reclaimed``
    broadcast → no reap->broadcast->resume loop) but IN _RECOVERABLE_END_REASONS (Bot Chat resurrection applies)."""
    for old_sid, popped in stale:
        try:
            _teardown_popped_session(popped, end_reason="superseded_by_resume")
        except Exception:
            logger.exception("superseded runtime teardown failed sid=%s", old_sid)


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path (session.create + cold resume; _sess() also builds on demand)."""

    def _run():
        if (session := _sessions.get(sid)) is not None:
            _start_agent_build(sid, session)
    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


def _load_resume_transcript(db, stored_id: str) -> tuple[list, list, list]:
    """(raw_history, display_history, ancestor_prefix) for a cold resume. The full lineage is materialized
    only while it fits sessions.max_resume_messages (the transcript is REST-paginated), else the tip alone."""
    from hermes_state import SessionResumeTooLargeError
    prefix_fits = True
    guard = getattr(db, "assert_resume_safe", None)
    if callable(guard):
        try:
            guard(stored_id)
        except SessionResumeTooLargeError as exc:
            prefix_fits = False
            logger.info("resume %s: compression lineage exceeds the resume limit (%s); hydrating the tip segment only",
                        stored_id, exc)
        except Exception:
            logger.debug("resume lineage guard failed; loading full lineage", exc_info=True)
    if prefix_fits:
        raw_history, display_history = db.get_resume_conversations(stored_id)
        return raw_history, display_history, db.get_ancestor_display_prefix(stored_id)
    raw_history = db.get_messages_as_conversation(stored_id, repair_alternation=True, include_row_ids=True)
    return raw_history, raw_history, []


def _schedule_resume_hydration(sid: str, stored_id: str, db, *, close_db: bool = False) -> None:
    """Load a cold resume's transcript off the JSON-RPC response path."""

    def _run() -> None:
        session = _sessions.get(sid)
        try:
            if session is None:
                return
            _emit("session.resume_progress", sid, {"phase": "history", "status": "loading"})
            db.reopen_session(stored_id)
            raw_history, display_history, prefix = _load_resume_transcript(db, stored_id)
            # Display keeps the full transcript; the model-fed history drops a dangling/interrupted
            # tool-call tail so a session killed mid-loop does not replay the unanswered call forever
            # (#29086).
            history = sanitize_replay_history(raw_history)
            if _sessions.get(sid) is not session:
                return
            with session["history_lock"]:
                session.update(history=history, display_history_prefix=prefix, resume_hydrating=False,
                               resume_message_count=len(display_history))
            # Deferred resumes answered before the transcript existed; cache the derived todo snapshot now.
            todo_state = _todo_state_from_history(history)
            if todo_state is not None and session.get("todo_state") is None:
                session["todo_state"] = todo_state
            session["resume_history_ready"].set()
            _emit("session.resume_progress", sid,
                  {"message_count": len(display_history), "phase": "history", "status": "complete"})
            _maybe_schedule_auto_continue(sid, session, stored_id)
            _start_agent_build(sid, session)
        except Exception as exc:
            if _sessions.get(sid) is not session:
                return
            message = f"resume failed: {exc}"
            session.update(resume_hydrating=False, resume_history_error=message, agent_error=message)
            session["resume_history_ready"].set()
            session["agent_ready"].set()
            _emit("session.resume_progress", sid, {"message": message, "phase": "history", "status": "failed"})
            _emit("error", sid, {"message": message})
            with _sessions_lock:
                discarded = _sessions.pop(sid, None) if _sessions.get(sid) is session else None
            if (lease := (discarded or {}).get("active_session_lease")) is not None:
                lease.release()
        finally:
            if close_db and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    logger.debug("failed to close resume db for %s", sid, exc_info=True)
    threading.Thread(target=_run, daemon=True).start()


def _session_pending_kind(sid: str) -> str:
    return next((str(_pending_prompt_payloads.get(rid, ("input.request", {}))[0]).removesuffix(".request")
                 for rid, (owner_sid, _ev) in list(_pending.items()) if owner_sid == sid), "")


def _session_live_status(sid: str, session: dict) -> str:
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    # Unset + build never started = a lazy watch session idling, not one stuck mid-construction.
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    return "working" if session.get("running") else "idle"


def _session_live_title(session: dict, key: str) -> str:
    title = str(session.get("pending_title") or "").strip()
    with contextlib.suppress(Exception), _session_db(session) as db:
        title = str(db.get_session_title(key) or title or "").strip() if db is not None else title
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    queued = _queued_prompt_snapshot(session)
    preview = next((" ".join(text.split())[:160] for msg in reversed(history)
                    if (text := _content_display_text(msg.get("content", msg.get("text", ""))).strip())), "")
    if queued:
        preview = " ".join(str(queued.get("user") or preview).split())[:160]
    elif inflight:
        preview = " ".join(str(inflight.get("assistant") or inflight.get("user") or preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid, "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()), "preview": preview,
        "session_key": key, "started_at": float(session.get("created_at") or now), "status": status,
        "title": _session_live_title(session, key),
    }


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    return str(getattr(session.get("agent"), "session_id", None) or session.get("session_key") or fallback or "")


def _find_live_session_by_key(session_key: str, profile_home=_ANY_PROFILE) -> tuple[str, dict] | None:
    # Timestamp-based stored ids can exist in several profiles' stores; a bare-id match would hand
    # profile B's resume profile A's runtime, so profile-aware callers match on (profile_home, key).
    # Profile-aware callers pass the home they resolved; the match must then be on (profile_home,
    # session_key). See #100029.
    for sid, session in list(_sessions.items()):
        if (not session.get("_finalized") and _session_lookup_key(session, fallback=sid) == session_key
                and _live_profile_matches(session, profile_home)):
            return sid, session
    return None


def _fallback_session_info(session: dict) -> dict:
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent)
    # The SESSION's own workspace, not the launch dir (wrong project in the desktop Files pane). `branch` is
    # always emitted ("" outside git) so a stale label clears; `desktop_contract` missing reads as "out of date".
    # Reporting `_default_session_cwd()` here told a lazily-resumed session's client that its workspace was
    # wherever the gateway process happened to start, so the desktop Files pane painted the wrong project
    # even after the renderer rebound correctly (#71254). `branch` is always emitted ("" outside a git repo)
    # so a client can clear a stale label instead of retaining it — the same contract `_lazy_session_info`
    # above already follows.
    cwd = _session_cwd(session)
    return {
        "cwd": cwd, "branch": git_probe.branch(cwd), "project": _project_info_for_cwd(cwd), "lazy": True,
        "model": _resolve_model(), "skills": {}, "tools": {}, "desktop_contract": DESKTOP_BACKEND_CONTRACT,
    }


def _reconcile_display_with_live(db_display: list[dict], in_memory: list[dict]) -> list[dict]:
    """Merge the persisted DISPLAY lineage with the in-memory live history: ``db_display`` is verbatim and
    candidate-inclusive (verification rows the model history collapses out) but can lag by a flush;
    ``in_memory`` is the recency authority but the collapsed *model* projection. Keep the DB display as base,
    append only the in-memory tail past the last DB row's ``(role, text)`` anchor — the verification answer
    survives a warm switch AND a not-yet-flushed live turn is kept."""
    if not db_display:
        return in_memory
    if not in_memory:
        return db_display

    def _key(msg: dict) -> tuple:
        return (msg.get("role"), _coerce_message_text(msg.get("content")))
    anchor = _key(db_display[-1])
    last_shared = max((idx for idx, msg in enumerate(in_memory) if isinstance(msg, dict) and _key(msg) == anchor), default=-1)
    if last_shared == -1:
        return db_display  # DB tail not in memory (DB ahead, or diverged) — trust it over duplicating
    return list(db_display) + list(in_memory[last_shared + 1 :])


def _live_visible_history(session: dict, db, in_memory_fallback: list[dict]) -> list[dict]:
    """User-visible DISPLAY projection for a live/warm session: the persisted display lineage (same read as
    resume/REST so the payloads agree) reconciled with the in-memory tail; in-memory when the DB is unavailable."""
    key = session.get("session_key")
    if db is not None and key:
        try:
            # include_compacted: a compacted session's archived turns are still the user's
            # conversation; without them a warm switch repainted the chat as summary + tail only.
            display = db.get_messages_as_conversation(
                key, include_ancestors=True, include_row_ids=True, include_compacted=True)
            # See #92080.
            return _reconcile_display_with_live(display, in_memory_fallback)
        except Exception:
            logger.debug("live display projection read failed", exc_info=True)
    return in_memory_fallback


def _live_session_payload(
    sid: str, session: dict, *, cols: int | None = None, touch: bool = False,
    transport: Transport | None = None, omit_messages: bool = False) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            _rebind_live_transport(sid, session, transport)
        if touch:
            # #84417: do not re-fire the live turn's original user text from a stale server-queue
            # self-duplicate after settle.
            session["last_active"] = time.time()
        in_memory_history = list(session.get("display_history_prefix") or []) + list(session.get("history") or [])
        inflight, queued = _inflight_snapshot(session), _queued_prompt_snapshot(session)
        running, turn_started_at = bool(session.get("running")), _turn_started_at(session)
    # Persisted display lineage via the session's profile-aware DB (not the launch ``_get_db()``), read
    # outside the history lock (the DB has its own). ``omit_messages`` skips the read (fast path).
    if omit_messages:
        history = in_memory_history
    else:
        with _session_db(session) as db:
            history = _live_visible_history(session, db, in_memory_history)
    messages = [] if omit_messages else _history_to_messages(history)
    payload = {
        "info": _fallback_session_info(session), "message_count": len(history) if omit_messages else len(messages),
        "messages": messages,
        "messages_omitted": omit_messages, "running": running, "turn_started_at": turn_started_at,
        "session_id": sid, "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    for key, value in (("inflight", inflight), ("queued", queued),
                       ("pending_approval", _pending_approval_request_payload(str(session.get("session_key") or ""))),
                       ("pending_clarify", _pending_clarify_request_payload(sid))):
        if value:
            payload[key] = value
    return _attach_todo_state(payload, session)


def _main_runtime_from_agent(agent) -> dict | None:
    """Aux-client main_runtime override from a live agent, so a one-shot inherits the session's runtime."""
    if agent is None:
        return None
    runtime: dict = {}
    for field in ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode"):
        value = getattr(agent, field, None)
        if isinstance(value, str) and value.strip():
            runtime[field] = value.strip()
        elif field == "api_key" and callable(value):
            runtime[field] = value
    return runtime or None


def register(server):
    bind_module(globals(), server)
