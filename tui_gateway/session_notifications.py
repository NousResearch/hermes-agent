"""Per-session notification poller: kanban/loop/delegation events routed to the owning session,
desktop UI wiring, HUD surface note. Bodies are rebound onto server.py's globals at install time
(method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module


def _notif_locked_sessions(fn, default):
    """Run ``fn(_sessions)`` under ``_sessions_lock``; ``default`` on failure (poller must never crash)."""
    try:
        with _sessions_lock:
            return fn(_sessions)
    except Exception:
        return default


def _notif_current_keys(sid: str, session: dict) -> set:
    return {str(session.get("session_key") or ""), _session_lookup_key(session, fallback=sid)}


def _notif_session_matches(s: dict, keys) -> bool:
    return str(s.get("session_key") or "") in keys or _session_lookup_key(s, fallback="") in keys


def _notif_live_session_matches(keys, exclude: dict | None = None) -> bool:
    """Any non-finalized live session (other than ``exclude``) matches ``keys``; False if the registry can't be read
    (fail open rather than drop the event)."""
    return _notif_locked_sessions(
        lambda ss: any(s is not exclude and not s.get("_finalized") and _notif_session_matches(s, keys)
                       for s in ss.values()),
        False)


def _notif_resolve_event_key(evt_key: str) -> str:
    """Resolve a compression-rotated session key to its continuation tip (or itself)."""
    try:
        db = _get_db()
        return (db.resolve_resume_session_id(evt_key) if db is not None else evt_key) or evt_key
    except Exception:
        return evt_key


def _notification_event_belongs_elsewhere(sid: str, session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session. Background completions carry the ``session_key`` of the
    session that started the work; async delegation completions also carry ``origin_ui_session_id`` (the live TUI tab)."""
    evt_ui_sid = str(evt.get("origin_ui_session_id") or "")
    if evt_ui_sid:
        if evt_ui_sid == str(sid or "") and not session.get("_finalized"):
            return False
        if _notif_locked_sessions(lambda ss: evt_ui_sid in ss and not ss[evt_ui_sid].get("_finalized"), False):
            return True
        # Exact UI tab gone: fall through to durable session_key routing so a resumed continuation with the same
        # key/lineage can still claim it.
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    current_keys = _notif_current_keys(sid, session)
    # Compression can rotate AIAgent.session_id while the detached child is still running: map the event's original
    # key to its continuation tip so it reaches the live session instead of becoming an orphan any poller may consume.
    # A live continuation wins over the compressed parent, else a stale parent tab could consume the event first.
    resolved_key = _notif_resolve_event_key(evt_key)
    if resolved_key != evt_key:
        if resolved_key in current_keys:
            return False
        if _notif_live_session_matches({resolved_key}):
            return True
    if evt_key in current_keys:
        return False
    return _notif_live_session_matches({evt_key, resolved_key}, exclude=session)


def _session_owns_notification_event(sid: str, session: dict, evt: dict) -> bool:
    """True iff *this* session PROVABLY owns ``evt`` (UI origin is this live session, or ``session_key`` raw/compression-
    resolved matches) — the fail-closed gate for addressed notifications, without the orphan-adoption fallback."""
    if session.get("_finalized"):
        return False
    if evt.get("type") == "async_delegation_work_closeout":
        ledger_home = str(evt.get("_ledger_profile_home") or "")
        expected_home = str(session.get("profile_home") or _hermes_home)
        if not ledger_home:
            return False
        try:
            if Path(ledger_home).resolve() != Path(expected_home).resolve():
                return False
        except (OSError, RuntimeError, ValueError):
            return False
    if str(evt.get("origin_ui_session_id") or "") == str(sid or ""):
        return True
    evt_key = str(evt.get("session_key") or "")
    current_keys = _notif_current_keys(sid, session)
    return bool(evt_key) and (evt_key in current_keys or _notif_resolve_event_key(evt_key) in current_keys)


def _notification_event_requires_owner(evt: dict) -> bool:
    """Whether ``evt`` must be positively claimed before TUI delivery."""
    return evt.get("type") == "async_delegation" or bool(evt.get("origin_ui_session_id") or evt.get("session_key"))


# Extra dedup fields per event type. Completions are terminal (one-shot per process session); watch events are not —
# one process can match patterns many times, so their content is part of the key.
_DEDUP_EXTRA_FIELDS = {
    "watch_match": ("command", "pattern", "output", "suppressed", "message_id"),
    "watch_disabled": ("command", "message", "suppressed"),
    "watch_overflow_": ("command", "message", "suppressed"),  # prefix match
}


def _notification_event_dedup_key(evt: dict) -> tuple:
    """UI-emission identity for a process notification event."""
    evt_type = evt.get("type", "completion")
    if evt_type == "async_delegation":
        # No process session_id: else every completion keys as ("", "async_delegation") and the second is suppressed forever.
        return (evt.get("delegation_id", ""), evt_type)
    if evt_type == "async_delegation_work_closeout":
        return (evt.get("delivery_id", ""), evt_type, evt.get("_ledger_profile_home", ""))
    extra = _DEDUP_EXTRA_FIELDS.get("watch_overflow_" if evt_type.startswith("watch_overflow_") else evt_type, ())
    return (evt.get("session_id", ""), evt_type, *(evt.get(f, 0 if f == "suppressed" else "") for f in extra))


# Mirror gateway/kanban_watchers.py TERMINAL_KINDS: claim silent kinds (archived/unblocked) too so the cursor advances
# past them and they can't wedge a later completed/blocked event behind an unclaimed row.
_KANBAN_NOTIFY_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked")
_KANBAN_POLL_SECONDS = _LOOP_POLL_SECONDS = 5.0


def _notif_release_turn(session: dict) -> None:
    with session["history_lock"]:
        session["running"] = False


def _notif_claim_turn(session: dict) -> bool:
    """Claim the idle session (running=True) under history_lock; False if a turn is live."""
    with session["history_lock"]:
        claimed = not session.get("running")
        session["running"] = True
        return claimed


def _notif_log_failure(what: str, exc: BaseException) -> None:
    print(f"[tui_gateway] {what}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _notif_submit(rid: str, sid: str, session: dict, text: str, what: str, **kwargs) -> None:
    """message.start + _run_prompt_submit for a claimed (running=True) turn; releases on failure."""
    try:
        _emit("message.start", sid)
        _run_prompt_submit(rid, sid, session, text, **kwargs)
    except Exception as exc:
        _notif_log_failure(what, exc)
        _notif_release_turn(session)
        raise


def _notif_loop_status(sid: str, text: str) -> None:
    _emit("status.update", sid, {"kind": "loop", "text": text})


def _notif_slash_loop_tick(rid: str, sid: str, session: dict, mgr, wakeup: str) -> None:
    """Slash-command /loop wakeup: route through the slash pipeline, not the model. No model reply to evaluate, so the
    tick completes immediately — unless the command resolves to a prompt (skill command etc.), which runs as a normal
    turn whose post-turn hook completes the tick."""
    _notif_release_turn(session)
    try:
        parts = wakeup.lstrip()[1:].split(None, 1)
        resp = _methods["command.dispatch"](
            rid, {"name": parts[0] if parts else "", "arg": parts[1] if len(parts) > 1 else "", "session_id": sid})
        payload = (resp or {}).get("result") or {}
        if out := str(payload.get("output") or "").strip():
            _notif_loop_status(sid, out)
        if payload.get("type") == "send" and payload.get("message"):
            if not _notif_claim_turn(session):
                mgr.abandon_tick()
                return
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, payload["message"])
            return
    except Exception:
        pass
    decision = mgr.complete_tick("")
    if decision.get("message"):
        _notif_loop_status(sid, decision["message"])


def _maybe_fire_tui_loop_tick(sid: str, session: dict) -> None:
    """Fire a due /loop wakeup for an idle TUI/Desktop/dashboard session (per-session poller, coarse cadence). Claims
    the session (running=True) before dispatching so a racing user prompt wins; the post-turn hook completes the tick."""
    try:
        from hermes_cli.loops import LoopManager, goal_blocks_loop_tick
    except Exception:
        return
    if not (sid_key := session.get("session_key") or ""):
        return
    mgr = LoopManager(session_id=sid_key)
    if not mgr.is_due() or goal_blocks_loop_tick(sid_key) or not _notif_claim_turn(session):
        return  # busy — stays due, next poll retries
    if not (wakeup := mgr.fire_tick()):
        _notif_release_turn(session)
        return
    rid = f"__loop__{int(time.time() * 1000)}"
    try:
        _notif_loop_status(sid, f"↻ /loop wakeup #{mgr.state.ticks_fired if mgr.state else '?'} firing…")
        if wakeup.lstrip().startswith("/"):
            _notif_slash_loop_tick(rid, sid, session, mgr, wakeup)
        else:
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, wakeup)
    except Exception as exc:
        _notif_log_failure("loop wakeup dispatch failed", exc)
        _notif_release_turn(session)
        with contextlib.suppress(Exception):
            mgr.abandon_tick()


def _kb_first_line(value: Any, limit: int) -> str:
    lines = str(value).strip().splitlines()
    return f"\n{lines[0][:limit]}" if lines else ""


def _kb_completed(task, payload: dict, title: str) -> str:
    handoff = (_kb_first_line(payload["summary"], 200) if payload.get("summary")
               else _kb_first_line(task.result, 160) if getattr(task, "result", None) else "")
    return f" done — {title}{handoff}"


def _kb_timed_out(task, payload: dict, title: str) -> str:
    with contextlib.suppress(TypeError, ValueError):
        return f" timed out (max_runtime={int(payload.get('limit_seconds') or 0)}s); will retry"
    return " timed out (max_runtime=0s); will retry"


# kind -> (glyph, suffix after "Kanban <id>"); silent kinds (archived/unblocked) are absent → None.
_KANBAN_EVENT_FORMATTERS = {
    "completed": ("✔", _kb_completed),
    "blocked": ("⏸", lambda t, p, title: " blocked" + (f": {str(p.get('reason'))[:160]}" if p.get("reason") else "")),
    "gave_up": ("✖", lambda t, p, title: " gave up after repeated spawn failures"
                + (f"\n{str(p.get('error'))[:200]}" if p.get("error") else "")),
    "crashed": ("✖", lambda t, p, title: " worker crashed (pid gone); dispatcher will retry"),
    "timed_out": ("⏱", _kb_timed_out),
    "status": ("🔄", lambda t, p, title: f" → {p.get('status') or ''}"),
}


def _format_kanban_event_text(sub: dict, task, ev, board_slug: str) -> Optional[str]:
    """Single-line notification text for one kanban event; wording mirrors gateway/kanban_watchers.py (reads the same
    as on Telegram). None for silent kinds."""
    if (entry := _KANBAN_EVENT_FORMATTERS.get(getattr(ev, "kind", ""))) is None:
        return None
    glyph, fmt = entry
    task_id = sub.get("task_id", "")
    title = (getattr(task, "title", None) or task_id)[:120]
    who = getattr(task, "assignee", None) or ""
    prefix = f"{glyph} " + (f"[{board_slug}] " if board_slug else "") + (f"@{who} " if who else "")
    return f"{prefix}Kanban {task_id}{fmt(task, getattr(ev, 'payload', None) or {}, title)}"


def _kb_board_key(_kb, board_meta) -> tuple[str, str]:
    """(slug, resolved DB identity) — multiple slugs can point at one DB when HERMES_KANBAN_DB pins it."""
    slug = (board_meta or {}).get("slug") or _kb.DEFAULT_BOARD
    db_path = (board_meta or {}).get("db_path")
    try:
        return slug, str(Path(db_path).expanduser().resolve() if db_path else _kb.kanban_db_path(slug).resolve())
    except Exception:
        return slug, f"slug:{slug}"


def _kb_poll_board(_kb, slug: str, session_key: str) -> list:
    """Claim + format this session's unseen events on one board. One poller per live session: the board is not opened
    writable unless it has a subscription owned by this exact session (a failed read-only probe — locked/corrupt DB —
    falls through so delivery is preserved)."""
    from hermes_cli import kanban_db_connect as _kbc
    from hermes_cli import kanban_db_notify as _kbn
    with contextlib.suppress(Exception):
        if _kbn.count_notify_subs(board=slug, platform="tui", chat_id=session_key) == 0:
            return []
    try:
        conn = _kbc.connect(board=slug)
    except Exception:
        return []
    texts: list = []
    with contextlib.closing(conn):
        try:
            subs = _kbn.list_notify_subs(conn)
        except Exception:
            return []
        for sub in subs:
            if (sub.get("platform") or "").lower() != "tui" or sub.get("chat_id") != session_key:
                continue
            sub_ident = dict(task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
                             thread_id=sub.get("thread_id") or "")
            _old, _new, events = _kbn.claim_unseen_events_for_sub(conn, kinds=_KANBAN_NOTIFY_KINDS, **sub_ident)
            if not events:
                continue
            task = _kb.get_task(conn, sub["task_id"])
            texts.extend(t for t in (_format_kanban_event_text(sub, task, ev, slug) for ev in events) if t)
            # Unsubscribe only on archive: ``done`` is reversible in review/controller flows, so keeping the sub lets a
            # later reopen notify the same session. The claimed cursor prevents replay.
            if task and getattr(task, "status", "") == "archived":
                with contextlib.suppress(Exception):
                    _kbn.remove_notify_sub(conn, **sub_ident)
    return texts


def _collect_kanban_notifications(session: dict) -> list:
    """Claim unseen terminal kanban events for this session's ``platform="tui"`` subscriptions (``kanban_create``
    auto-subscribes with ``chat_id=HERMES_SESSION_KEY``; no "tui" messaging adapter exists, so this poller is the
    delivery path). Same atomic cursor-claim as the gateway notifier: exactly-once even if a gateway polls the same DB.

    See #59890.
    """
    session_key = str(session.get("session_key") or "")
    if not session_key or session.get("_finalized"):
        return []
    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        return []
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        try:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        except Exception:
            return []
    # dict keyed by resolved DB identity: first slug per DB wins (a pinned HERMES_KANBAN_DB aliases slugs).
    unique = {}
    for slug, resolved in (_kb_board_key(_kb, board_meta) for board_meta in boards):
        unique.setdefault(resolved, slug)
    return [t for slug in unique.values() for t in _kb_poll_board(_kb, slug, session_key)]


def _notif_poll_kanban(sid: str, session: dict) -> None:
    """One kanban poll: emit new texts, buffer them, and run the buffered batch as a turn if idle. Events are
    cursor-claimed (never re-queued), so they wait in the buffer instead of dropping the agent turn."""
    try:
        texts = _collect_kanban_notifications(session)
    except Exception as exc:
        _notif_log_failure("kanban notification poll failed", exc)
        texts = []
    for text in texts:
        _emit("status.update", sid, {"kind": "process", "text": text})
    if texts:
        session.setdefault("_kanban_pending", []).extend(texts)
    if not session.get("_kanban_pending") or not _notif_claim_turn(session):
        return
    with session["history_lock"]:
        batch, session["_kanban_pending"] = list(session.get("_kanban_pending") or []), []
    with contextlib.suppress(Exception):
        _notif_submit(f"__notif__{int(time.time() * 1000)}", sid, session, "\n".join(batch), "kanban notification dispatch failed")


def _notif_dispatch_event(sid: str, session: dict, evt: dict, text: str) -> None:
    """Run the claimed (running=True) agent turn for one notification event."""
    from tools.async_delegation import claim_event_delivery, complete_event_delivery, release_event_delivery
    if (claim := claim_event_delivery(evt, "tui-poller")) is None:
        return
    kwargs = ({"display_kind": "async_delegation_complete", "display_metadata": _async_delegation_display_metadata(evt)}
              if evt.get("type") == "async_delegation" else {})
    try:
        _notif_submit(f"__notif__{int(time.time() * 1000)}", sid, session, text, "notification poller dispatch failed", **kwargs)
    except Exception:
        release_event_delivery(evt, claim)
        return
    complete_event_delivery(evt, claim)


def _closeout_continuation(evt: dict, text: str) -> _InternalContinuation:
    return _InternalContinuation(
        text=text,
        work_id=str(evt.get("origin_work_id") or ""),
        generation=int(evt.get("work_generation") or 0),
        delivery_id=str(evt.get("delivery_id") or ""),
        claim_id=str(evt.get("claim_id") or ""),
        profile_home=str(evt.get("_ledger_profile_home") or ""),
    )


def _start_next_internal_continuation(rid, sid: str, session: dict) -> bool:
    """Start one trusted continuation after the current/user turn boundary."""
    with session["history_lock"]:
        pending = session.setdefault("internal_continuations", deque())
        if session.get("running") or not pending:
            return False
        item = pending.popleft()
        session["running"] = True
    try:
        if item.profile_home:
            expected_home = session.get("profile_home") or _hermes_home
            try:
                affinity_matches = (
                    Path(item.profile_home).resolve() == Path(expected_home).resolve()
                )
            except (OSError, RuntimeError, ValueError):
                affinity_matches = False
            if not affinity_matches:
                raise RuntimeError(
                    "closeout profile affinity does not match target TUI session"
                )
        _emit("message.start", sid)
        if not _run_prompt_submit(
            rid, sid, session, item.text, internal_continuation=item
        ):
            raise RuntimeError("target TUI session cannot resume continuation")
        return True
    except Exception:
        from tools.async_delegation import (
            recover_and_enqueue_work_groups,
            release_enqueued_work_group_event,
        )
        from tools.process_registry import process_registry

        try:
            scoped_session = (
                {"profile_home": item.profile_home} if item.profile_home else session
            )
            with _session_profile_scope(scoped_session):
                released = release_enqueued_work_group_event(
                    {
                        "type": "async_delegation_work_closeout",
                        "origin_work_id": item.work_id,
                        "work_generation": item.generation,
                        "delivery_id": item.delivery_id,
                        "claim_id": item.claim_id,
                        "_ledger_profile_home": item.profile_home,
                    }
                )
                if released:
                    recover_and_enqueue_work_groups(
                        consumer="tui-closeout-retry",
                        target_queue=process_registry.completion_queue,
                    )
        finally:
            with session["history_lock"]:
                session["running"] = False
        raise


def _notif_handle_event(sid, session, evt, emitted, registry, fmt, deferred) -> bool:
    """Route one dequeued event: foreign (another live session owns it) → requeued, or onto ``deferred`` during the
    shutdown drain; unowned (addressed but unprovable — never adopt an orphan) → dropped, except delegation payloads
    deferred for a resume; ours (or ownerless legacy, kept process-global) → status.update once, then an agent turn if
    idle. False = the drain must stop (session busy)."""
    queue = registry.completion_queue
    evt_type = evt.get("type", "completion")
    is_closeout = evt_type == "async_delegation_work_closeout"
    is_delegation = evt_type in {"async_delegation", "async_delegation_work_closeout"}
    if _notification_event_belongs_elsewhere(sid, session, evt):
        if deferred is not None:
            deferred.append(evt)
        else:  # otherwise a process started in session A surfaces in whichever poller wakes first
            queue.put(evt)
            time.sleep(0.1)
        return True
    if _notification_event_requires_owner(evt) and not _session_owns_notification_event(sid, session, evt):
        origin, key = str(evt.get("origin_ui_session_id") or ""), str(evt.get("session_key") or "")
        if is_closeout:
            if deferred is not None:
                deferred.append(evt)
            else:
                queue.put(evt)
                time.sleep(0.5)
            return True
        if deferred is None:
            (logger.warning if is_delegation else logger.debug)(
                "Dropping unowned %s notification (origin=%r key=%r) instead of delivering to session %s",
                evt_type, origin, key, sid)
        elif is_delegation:
            deferred.append(evt)
        else:
            logger.debug("Dropping unowned %s notification during shutdown drain (origin=%r key=%r)", evt_type, origin, key)
        return True
    if evt_type == "completion" and registry.is_completion_consumed(evt.get("session_id", "")):
        return True
    text = fmt(evt)
    if not text:
        return True
    # Emit once per dedup key: a re-queued completion would otherwise re-emit every 0.5s while the session is busy,
    # while distinct watch_match events from one process must stay visible.
    dedup_key = _notification_event_dedup_key(evt)
    if dedup_key not in emitted:
        _emit("status.update", sid, {"kind": "process", "text": text})
        emitted.add(dedup_key)
    if is_closeout:
        with session["history_lock"]:
            session.setdefault("internal_continuations", deque()).append(
                _closeout_continuation(evt, text)
            )
            busy = bool(session.get("running"))
        if not busy:
            try:
                _start_next_internal_continuation(
                    f"__notif__{int(time.time() * 1000)}", sid, session
                )
            except Exception as exc:
                _notif_log_failure("closeout continuation dispatch failed", exc)
        return True
    if not _notif_claim_turn(session):
        queue.put(evt)
        if deferred is not None:
            return False
        time.sleep(0.25)  # back off: the re-queued event keeps the queue non-empty, else this loop spins at 100% CPU
        return True
    _notif_dispatch_event(sid, session, evt, text)
    return True


def _notification_poller_loop(stop_event: threading.Event, sid: str, session: dict) -> None:
    """Daemon thread (started by _init_session()) that drains the process-global completion_queue for this session
    (ownership routing: _notif_handle_event) and polls ``kanban_notify_subs`` every ``_KANBAN_POLL_SECONDS`` — the
    delivery path for platform="tui" rows.

    Also polls ``kanban_notify_subs`` every ``_KANBAN_POLL_SECONDS`` for this session's TUI kanban
    subscriptions and delivers terminal task events the same way (status.update + agent turn) — the delivery
    path tools/kanban_tools.py documents for platform="tui" rows (issue #59890).
    """
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import format_process_notification
    from tools.async_delegation import recover_and_enqueue_work_groups
    queue = process_registry.completion_queue
    emitted: set = set()  # dedup re-queued events so one completion isn't emitted 50 times while busy
    handle = lambda evt, deferred: _notif_handle_event(  # noqa: E731
        sid, session, evt, emitted, process_registry, format_process_notification, deferred)
    last_kanban_poll = last_loop_poll = last_closeout_recovery = 0.0
    while not stop_event.is_set() and not session.get("_finalized"):
        now = time.monotonic()
        if now - last_closeout_recovery >= _LOOP_POLL_SECONDS:
            last_closeout_recovery = now
            try:
                with _session_profile_scope(session):
                    recover_and_enqueue_work_groups(
                        consumer="tui-closeout-poller",
                        target_queue=queue,
                    )
            except Exception:
                logger.warning("TUI closeout recovery poll failed", exc_info=True)
        # /loop wakeup driver: fire a due tick for THIS session while idle (same claim-under-lock as kanban dispatch).
        # An active non-parked /goal owns the idle boundary and defers it.
        if now - last_loop_poll >= _LOOP_POLL_SECONDS:
            last_loop_poll = now
            try:
                _maybe_fire_tui_loop_tick(sid, session)
            except Exception as loop_exc:
                _notif_log_failure("loop wakeup poll failed", loop_exc)
        if now - last_kanban_poll >= _KANBAN_POLL_SECONDS:
            last_kanban_poll = now
            _notif_poll_kanban(sid, session)
        try:
            evt = queue.get(timeout=0.5)
        except Exception:
            continue
        handle(evt, None)
    # Drain remaining events after the stop signal so nothing is lost on shutdown; foreign and orphaned-delegation
    # events are handed back to the shared queue afterwards.
    deferred: list = []
    while not queue.empty():
        try:
            evt = queue.get_nowait()
        except Exception:
            break
        if not handle(evt, deferred):
            break
    for evt in deferred:
        queue.put(evt)


def _async_delegation_display_metadata(evt: dict) -> dict:
    """Build display-only metadata before the completion event is formatted."""
    raw_results = evt.get("results")
    results: list[dict] = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(1 for r in results if r.get("status") in {"completed", "success"})
    failed_count = sum(1 for r in results if r.get("status") in {"failed", "error"})
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    return {"delegation_id": str(evt.get("delegation_id") or ""), "task_count": task_count,
            "completed_count": completed_count or task_count - failed_count, "failed_count": failed_count,
            **({"duration_seconds": duration} if isinstance(duration, (int, float)) else {})}


_desktop_ui_wired = False


def _wire_desktop_sinks() -> None:
    """Idempotently wire process-registry and desktop-tool sinks to renderer events: `agent.terminal.output` and
    `terminal.close` (drops a tab without killing the process) route to the window owning the process; desktop-only
    tools pass the turn's ``HERMES_UI_SESSION_ID`` as ``sid``. `_emit` is thread-safe."""
    global _desktop_ui_wired
    from tools.process_registry import process_registry

    def _owner_sid(session) -> str:
        # session may be None (process already finished/pruned) — the tab can still linger and be closed.
        session_key = str(getattr(session, "session_key", "") or "") if session is not None else ""
        if not session_key:
            return ""
        with _sessions_lock:
            return next((sid for sid, s in _sessions.items() if str(s.get("session_key") or "") == session_key), "")
    if getattr(process_registry, "on_output", None) is None:
        process_registry.on_output = lambda session, chunk: _emit(
            "agent.terminal.output", _owner_sid(session), {"process_id": session.id, "chunk": chunk})
    if getattr(process_registry, "on_close", None) is None:
        process_registry.on_close = lambda session, pid: _emit("terminal.close", _owner_sid(session), {"process_id": pid})
    if not _desktop_ui_wired:
        with contextlib.suppress(Exception):
            from tools import desktop_ui
            desktop_ui.set_emitter(lambda sid, event, payload: _emit(event, sid, payload))
            _desktop_ui_wired = True


# (stop_event, thread) for every poller started in this process, pruned of dead threads on each spawn. Test teardowns
# reap leaked pollers through it: an unjoined poller steals events off the process-global queue in a LATER test.
_notification_pollers: list = []


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session (thread name is greppable)."""
    _wire_desktop_sinks()
    stop = threading.Event()
    t = threading.Thread(target=_notification_poller_loop, args=(stop, sid, session), daemon=True, name=f"tui-notif-poller-{sid}")
    _notification_pollers[:] = [(s, th) for (s, th) in _notification_pollers if th.is_alive()] + [(stop, t)]
    t.start()
    return stop


def _hud_surface_note(session: dict) -> str:
    """The HUD-mode note for this turn, or "" when it was not typed there."""
    if session.get("client_surface") != "hud":
        return ""
    from agent.prompt_builder import hud_surface_note
    return hud_surface_note(getattr(session.get("agent"), "valid_tool_names", None))


def _prepend_note(run_message: Any, note: str) -> Any:
    """Prefix a per-turn note onto the MODEL INPUT, leaving the prompt alone: everything the model must know that the
    user did not type (interrupted reply, reactions, surface) arrives this way, so no scaffolding reaches the
    transcript and no sent message is rewritten — the cached prefix survives."""
    if note and isinstance(run_message, str):
        return f"{note}\n\n{run_message}"
    if note and isinstance(run_message, list):
        return [{"type": "text", "text": note}, *run_message]
    return run_message


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
