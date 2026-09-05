"""Session / delegation / spawn-tree / billing / pet JSON-RPC handlers.

Bodies are rebound onto server.py's globals at install time (method_ctx.py), so they use server
helpers (``_sessions``, ``_ok``, ``_err``, ...) bare; module-level helpers are published onto
server.py the same way (tests monkeypatching ``server.X`` still intercept)."""

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── shared handler plumbing ──────────────────────────────────────────
def _session_arg(resolve):
    """Resolve ``params.session_id`` via ``resolve`` (a lambda — decoration precedes bind_module) → 3rd arg."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            session, err = resolve(params, rid)
            return err or fn(rid, params, session)
        return handler
    return deco


_with_session = _session_arg(lambda params, rid: _sess_nowait(params, rid))  # no agent-build wait
_with_live_session = _session_arg(lambda params, rid: _sess(params, rid))  # waits for the agent build


def _session_method(name: str, *, live: bool = False):
    """``@method(name)`` over ``_with_live_session`` (waits for the agent build) or ``_with_session``."""
    return lambda fn: method(name)((_with_live_session if live else _with_session)(fn))


def _with_db(code: int, *, session_scoped: bool):
    """Append a db arg — the session's db (after ``_with_session``) or ``_profile_db(params)``; ``code`` when None."""
    def deco(fn):
        def handler(rid, params: dict, *session) -> dict:
            with (_session_db(session[0]) if session_scoped else _profile_db(params)) as db:
                if db is None:
                    return _db_unavailable_error(rid, code=code)
                return fn(rid, params, *session, db)
        return _with_session(handler) if session_scoped else handler
    return deco


def _str_param(params: dict, key: str, default: str = "") -> str:
    """``str(params[key]).strip()`` with ``default`` for missing / falsy values."""
    return str(params.get(key) or "").strip() or default


def _flag(params: dict, name: str) -> bool:
    return is_truthy_value(params.get(name, False))


def _int_param(params: dict, key: str, default: int) -> int:
    """``int(params[key])`` with ``default`` for missing / unparsable values."""
    try:
        return int(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _new_runtime_ids(params: dict) -> tuple[str, str]:
    """Fresh runtime sid + resolved DB ``source`` for a session minted from ``params``."""
    return uuid.uuid4().hex[:8], _resolve_session_source(_str_param(params, "source") or None)


@contextlib.contextmanager
def _profile_build_scope(profile_home):
    """Bind HERMES_HOME + secret scope for an agent build (home alone leaves get_secret() on the LAUNCH .env)."""
    if not profile_home:
        yield
        return
    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(str(profile_home))))
    try:
        yield
    finally:
        reset_hermes_home_override(home_token)
        reset_secret_scope(secret_token)


def _make_agent_in_context(sid: str, key: str, **kwargs):
    """``_make_agent`` with the session context bound for the build and cleared after."""
    tokens = _set_session_context(key)
    try:
        return _make_agent(sid, key, session_id=key, **kwargs)
    finally:
        _clear_session_context(tokens)


def _profile_session_db(profile_home):
    """``(db, owns)``: a DEDICATED handle on ``profile_home``'s state.db, else the shared launch db."""
    if profile_home:
        from hermes_state_registry import acquire
        return acquire(Path(profile_home) / "state.db"), True
    return _get_db(), False


def _release_db(db) -> None:
    with contextlib.suppress(Exception):
        from hermes_state_registry import release_or_close
        release_or_close(db)


def _branch_title(db, parent_key: str) -> str:
    """Next title in the parent's lineage (mirrors the TUI /branch naming)."""
    current = db.get_session_title(parent_key) or "branch"
    if hasattr(db, "get_next_title_in_lineage"):
        return db.get_next_title_in_lineage(current)
    return f"{current} (branch)"


def _cwd_info(session: dict, cwd: str, branch=None) -> dict:
    """session.info after a cwd change: the full agent view, or the lazy shape."""
    if (agent := session.get("agent")) is not None:
        return _session_info(agent, session)
    return {"cwd": cwd, "branch": git_probe.branch(cwd) if branch is None else branch,
            "project": _project_info_for_cwd(cwd), "lazy": True}


def _session_row_summary(row: dict, *, tip_row: dict | None = None, resolved_id=None) -> dict:
    """Compact session.list row; ``tip_row``/``resolved_id`` come from the compression tip."""
    tip_row = tip_row or row
    return {"id": row["id"], **({} if resolved_id is None else {"resolved_id": resolved_id}),
            "title": row.get("title") or "", "preview": tip_row.get("preview") or "",
            "started_at": row.get("started_at") or 0, "message_count": tip_row.get("message_count") or 0,
            "source": row.get("source") or ""}


# Hidden from human listings (sub-agent runs, kanban workers); a deny-list so new platforms surface automatically.
_LISTING_DENY_SOURCES = frozenset({"kanban", "tool"})


def _denied_source(row: dict) -> bool:
    return (row.get("source") or "").strip().lower() in _LISTING_DENY_SOURCES


def _listing_rows(db, limit: int, **kwargs) -> list:
    """Human-facing ``list_sessions_rich`` rows (most recent first), deny-list applied."""
    rows = db.list_sessions_rich(source=None, limit=limit, order_by_last_active=True, compact_rows=True, **kwargs)
    return [row for row in rows if not _denied_source(row)]


def _snapshot_sessions(rid):
    """``(list(_sessions.items()), None)`` under the lock, or ``(None, 5036 error)`` — fail CLOSED."""
    try:
        with _sessions_lock:
            return list(_sessions.items()), None
    except Exception as e:
        return None, _err(rid, 5036, f"could not enumerate active sessions: {e}")


def _pet_display_cfg() -> dict:
    """``display.pet`` config block, ``{}`` when config is unreadable."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        return display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        return {}


def _pet_emit(event: str, payload: dict, what: str) -> None:
    """Best-effort progress emit: a transport hiccup must never abort generation."""
    try:
        _emit(event, "", payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s emit failed: %s", what, exc)


def _pet_gen_abort(rid, token: str, code: int, message: str) -> dict:
    """Release the cancel arm for ``token`` and return ``_err``."""
    _pet_cancel_release(token)
    return _err(rid, code, message)


def _pet_method(name: str, *, fail_open=None, slug: bool = False, scoped: bool = True):
    """``@method`` (+ ``@_profile_scoped`` unless ``scoped=False``) whose exceptions never break the surface: logged
    at debug, then ``fail_open`` (payload or ``params -> payload``) or ``_err(5031)``. ``slug``: 3rd arg (4004)."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                if slug and not (value := _str_param(params, "slug")):
                    return _err(rid, 4004, "missing slug")
                return fn(rid, params, value) if slug else fn(rid, params)
            except Exception as exc:  # noqa: BLE001 - cosmetic surface
                logger.debug("%s failed: %s", name, exc)
                if fail_open is not None:
                    return _ok(rid, fail_open(params) if callable(fail_open) else dict(fail_open))
                return _err(rid, 5031, f"{name} failed: {exc}")
        return method(name)(_profile_scoped(handler) if scoped else handler)
    return deco


def _active_pet():
    """``(pet, scale)`` when the pet display is enabled and the pet exists, else None."""
    enabled, pet, scale = _pet_active_selection()
    return None if not enabled or pet is None or not pet.exists else (pet, scale)


def _billing_call(rid, fn, extra: dict | None = None) -> dict:
    """Portal call → ok; BillingError → serialized envelope, else generic; ``extra`` rides both ERROR envelopes."""
    from hermes_cli.nous_billing import BillingError
    try:
        return _ok(rid, fn())
    except BillingError as exc:
        return _ok(rid, {**_serialize_billing_error(exc), **(extra or {})})
    except Exception as exc:
        return _ok(rid, {"ok": False, "error": "error", "message": str(exc), **(extra or {})})


def _billing_invalid(rid, message: str, error: str = "invalid_request") -> dict:
    return _ok(rid, {"ok": False, "error": error, "message": message})


def _billing_pick(result: dict, **fields) -> dict:
    """``{"ok": True, <snake>: result[<camel>], ...}`` in ``fields`` order."""
    return {"ok": True, **{key: result.get(src) for key, src in fields.items()}}


def _billing_pending_change(result: dict) -> dict:
    return {"ok": True, "message": result.get("message"), "payload": result}


# ── session.create / list / most_recent / facts ──────────────────────
def _persist_branch(db, new_key: str, parent_key: str, title: str, history: list, *, source, cwd, profile_name,
                    copy_fields=(), compensate: bool = False) -> None:
    """Branch child row + parent transcript (bounded-chunk transactions) + title. ``_branched_from`` keeps the
    row visible in list_sessions_rich() (the live parent never matches the legacy end_reason='branched'
    heuristic); NULL ``profile_name`` rows drop out of profile-keyed sidebar matching / deep links. ``compensate``
    deletes a committed row whose transcript/title failed (a durable-but-empty row would defeat the INSERT OR
    IGNORE first-prompt seed) — except on disk-full, where the delete cannot land."""
    db.create_session(new_key, source=source, model=_resolve_model(), model_config={"_branched_from": parent_key},
                      parent_session_id=parent_key, cwd=cwd, profile_name=profile_name)
    try:
        # Compensation guard (#93959 review): if the transcript copy or title write fails AFTER the row
        # committed, the durable-but-empty row would defeat the lazy first-prompt fallback
        # (_ensure_session_db_row is INSERT OR IGNORE — the row exists, so the seed never lands and the
        # renderer fail-latches on a "transcript-less" session again). Roll back just this child so the seed
        # path can retry cleanly on first submit.
        # Copy the whole parent history in bounded-chunk transactions — a branch seed can be hundreds of
        # rows, and per-row transactions were the write-amplification pattern removed in #23254.
        db.append_messages_batch(
            new_key, [{"role": msg.get("role", "user"), "content": msg.get("content"),
                       **{field: msg.get(field) for field in copy_fields}} for msg in history], chunk_rows=500)
        db.set_session_title(new_key, title)
    except Exception as exc:
        from hermes_state_errors import is_disk_full_error
        if compensate and not is_disk_full_error(exc):
            try:
                db.delete_session(new_key)
            except Exception:
                logger.debug("branch seed compensation delete failed for %s", new_key, exc_info=True)
        raise


def _seed_branch_row(record: dict, key: str, parent_session_id: str, history: list, source: str, profile_home):
    """Persist a seeded desktop branch child NOW (the one session.create exception to lazy rows): the
    renderer's post-create resume re-fetches it via REST/defer_history, so an unpersisted child 404s and
    the fail-latch spins forever. Best-effort — on failure the lazy first-prompt path is the fallback."""
    try:
        with _session_db(record) as db:
            if db is None:
                return
            _persist_branch(db, key, parent_session_id, _branch_title(db, parent_session_id), history,
                            source=source, cwd=record["cwd"],
                            profile_name=(Path(profile_home).name if profile_home else None), compensate=True)
            record["pending_title"] = None
    except Exception:
        logger.warning("seeded-branch persistence failed for %s; falling back to lazy row creation", key,
                       exc_info=True)


def _create_overrides(params: dict) -> tuple:
    """PER-SESSION (model, reasoning, service_tier) overrides from the composer — never a global config
    write. ``fast`` presence is the contract: omitted inherits, true pins priority, false pins normal ("")."""
    create_model = _str_param(params, "model")
    model_override = None
    if create_model:
        model_override = {"model": create_model, "provider": _str_param(params, "provider") or None}
    reasoning_override = None
    if effort := _str_param(params, "reasoning_effort"):
        with contextlib.suppress(Exception):
            from hermes_constants import parse_reasoning_effort
            reasoning_override = parse_reasoning_effort(effort)
    service_tier_override = None
    if "fast" in params:
        service_tier_override = "priority" if is_truthy_value(params.get("fast")) else ""
    return model_override, reasoning_override, service_tier_override


@method("session.create")
def _(rid, params: dict) -> dict:
    (sid, source), key = _new_runtime_ids(params), _new_session_key()
    history = _coerce_seed_history(params.get("messages"))
    # Branch: links back so list_sessions_rich keeps it visible and the sidebar nests it.
    parent_session_id = _str_param(params, "parent_session_id") or None
    # Only an explicitly chosen existing workspace persists as cwd; the launch-dir fallback is "No workspace".
    explicit_cwd = False
    raw_cwd = _str_param(params, "cwd")  # unguarded, as on BASE: only the path check is best-effort
    with contextlib.suppress(Exception):
        explicit_cwd = bool(raw_cwd) and os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd)))
    _enable_gateway_prompts()
    # ``profile`` (app-global remote mode): stored so the build and every turn re-bind HERMES_HOME.
    profile_home = _profile_home(profile := (params.get("profile") or "").strip() or None)
    session_model_override, create_reasoning_override, create_service_tier_override = _create_overrides(params)
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": None, "agent_error": None, "agent_ready": threading.Event(), "attached_images": [],
            "close_on_disconnect": _flag(params, "close_on_disconnect"),
            "active_session_lease": None,  # claimed lazily on the first turn (_ensure_active_session_slot)
            "cols": int(params.get("cols", 80)), "created_at": now, "edit_snapshots": {},
            "explicit_cwd": explicit_cwd,
            "history": history, "history_lock": threading.Lock(), "history_version": 0, "image_counter": 0,
            "cwd": _completion_cwd(params), "inflight_turn": None, "last_active": now,
            "model_override": session_model_override,
            "create_reasoning_override": create_reasoning_override,
            "create_service_tier_override": create_service_tier_override,
            "parent_session_id": parent_session_id, "pending_title": _str_param(params, "title") or None,
            "pending_hidden": _flag(params, "hidden"), "room_plumbing": _flag(params, "room_plumbing"),
            "follow_profile_config": _flag(params, "follow_profile_config"),
            "profile_home": str(profile_home) if profile_home is not None else None,
            "running": False, "session_key": key, "show_reasoning": _load_show_reasoning(), "source": source,
            "slash_worker": None, "tool_progress_mode": _load_tool_progress_mode(), "tool_started_at": {},
            "transport": current_transport() or _stdio_transport}
        _register_session_cwd(_sessions[sid])
    # No DB row here (drafts left "Untitled" litter): created on the first prompt — except seeded branch children.
    # NOTE: we intentionally do NOT persist a DB row here. Every TUI/desktop launch (and every "New agent" /
    # draft) opens a session here just to paint the composer, so eagerly creating a row left an "Untitled"
    # empty session behind for every launch the user never typed into. The row is now created lazily on the
    # first prompt (see _ensure_session_db_row + prompt.submit), and the AIAgent's own INSERT-OR-IGNORE
    # persists it on the first turn too. EXCEPTION — seeded branch children (#93959): a desktop branch
    # carries parent_session_id AND a seeded transcript, which is explicit user intent, not an abandoned
    # draft. The row MUST exist immediately: the renderer's post-create resume re-fetches the child through
    # REST + defer_history hydration, both of which read the DB — an unpersisted child 404s, the fail-latch
    # then refuses to bind a "transcript-less" session, and the user sees an infinite spinner whose
    # optimistic row vanishes on restart. Persisting up front also means a restart keeps the branch (both
    # reports lost it) and the title lands in the parent's lineage instead of falling back to a
    # message-preview name. Title mirrors the TUI /branch naming.
    if parent_session_id and history:
        _seed_branch_row(_sessions[sid], key, parent_session_id, history, source, profile_home)
    # Return immediately so Ink can paint; the AIAgent builds right after the flush.
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap
    cwd = _sessions[sid]["cwd"]
    override = session_model_override or {}
    return _ok(rid, {
        "session_id": sid, "stored_session_id": key, "message_count": len(history),
        "messages": _history_to_messages(history),
        # Reflect the override now so the client doesn't clobber its sticky pick.
        "info": {"model": override.get("model") if override else _resolve_model(),
                 **({"provider": override["provider"]} if override.get("provider") else {}),
                 "tools": {}, "skills": {}, "cwd": cwd, "branch": git_probe.branch(cwd),
                 "project": _project_info_for_cwd(cwd), "lazy": True, "desktop_contract": DESKTOP_BACKEND_CONTRACT,
                 "profile_name": _response_profile_name(profile)}})


def _session_list_by_title(rid, db, title_lookup: str) -> dict:
    """EXACT-title lookup (title as identity), window-free on purpose (a busy profile's windowed listing can
    push the row out). Hidden rows resolve (canonical chats are born hidden); archived / deny-listed do not;
    lineages resolve to the live tip (``resolved_id``)."""
    row = db.get_session_by_title(title_lookup)
    if row and row.get("archived"):
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        # A Bot Chat archived by the ws-orphan reaper / agent_close is an accident (the desktop would mint
        # replacements forever): resurrect recoverable reasons only. Re-fetch by ID — title is not UNIQUE.
        if title_lookup == BOT_CHAT_TITLE and db.unarchive_recoverable_session(row["id"]):
            # The canonical Bot Chat is identity-scoped: an archive stamped by the ws-orphan reaper or older
            # agent cleanup (ws_orphan_reap / agent_close) is an accident, not user intent, and hiding the
            # row here makes the desktop mint transient replacements forever (#92687). Resurrect it — same
            # recoverable-reason set as stale-route recovery. Deliberate archives (no/explicit end_reason)
            # still hide. Re-fetch by ID: title has no DB-level UNIQUE, so a title re-query could grab a
            # different (still-archived) duplicate row.
            row = db.get_session(row["id"])
    if not row or row.get("archived") or _denied_source(row):
        return _ok(rid, {"sessions": []})
    tip = row["id"]
    with contextlib.suppress(Exception):
        # Real compression continuation only: the resolver's unmarked-child fallback could redirect Bot Chat.
        tip = db.get_compression_tip(row["id"]) or row["id"]
    tip_row = (db.get_session(tip) or row) if tip != row["id"] else row
    return _ok(rid, {"sessions": [_session_row_summary(row, tip_row=tip_row, resolved_id=tip)]})


@method("session.list")
@_with_db(5006, session_scoped=False)
def _(rid, params: dict, db) -> dict:
    try:
        if title_lookup := _str_param(params, "title"):
            return _session_list_by_title(rid, db, title_lookup)
        limit = int(params.get("limit", 200) or 200)
        # Over-fetch: per-source filtering + tip merging must not leave us short. ``include_hidden`` is for
        # surfaces that OWN hidden sessions (Bots pane, pickers).
        rows = _listing_rows(db, max(limit * 2, 200), include_hidden=_flag(params, "include_hidden"))[:limit]
        return _ok(rid, {"sessions": [_session_row_summary(s) for s in rows]})
    except Exception as e:
        return _err(rid, 5006, str(e))


@method("session.most_recent")
def _(rid, params: dict) -> dict:
    """Most recent human-facing session (session.list deny-list); errors fold into ``session_id: null``."""
    with _profile_db(params) as db:
        try:
            # Generous over-fetch: many ``tool`` rows must not yield a false "none".
            for row in _listing_rows(db, 200)[:1] if db is not None else ():
                return _ok(rid, {"session_id": row.get("id"), "title": row.get("title") or "",
                                 "started_at": row.get("started_at") or 0, "source": row.get("source") or ""})
        except Exception:
            logger.exception("session.most_recent failed")
        return _ok(rid, {"session_id": None})


@method("project.facts")
def _(rid, params: dict) -> dict:
    """The system prompt's coding-context detection for a cwd (UIs don't re-sniff); null = not code."""
    try:
        from agent.coding_context import project_facts_for
        return _ok(rid, {"facts": project_facts_for(params.get("cwd"))})
    except Exception:
        logger.exception("project.facts failed")
        return _ok(rid, {"facts": None})


@method("verification.status")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Best known verification evidence for a cwd/session. Read-only: never runs checks,
    never upgrades targeted evidence into a repository-wide guarantee."""
    try:
        from agent.verification_evidence import verification_status
        return _ok(rid, {"verification": verification_status(
            session_id=params.get("session_id") or params.get("session_key"), cwd=params.get("cwd"))})
    except Exception:
        logger.exception("verification.status failed")
        return _ok(rid, {"verification": {"status": "unknown", "evidence": None}})


# ── session.resume ───────────────────────────────────────────────────
class _Resume:
    """Per-call ``session.resume`` state. ``owns_db``: the DEDICATED profile handle is ours
    to close (handler ``finally``) until handed to the hydration worker or the agent."""

    def __init__(self, rid, params: dict, target: str) -> None:
        self.rid, self.params, self.target = rid, params, target
        self.db, self.owns_db, self.found, self.profile_resume_cwd = None, False, None, ""
        self.cols = _int_param(params, "cols", 80)
        # ``profile`` (app-global remote mode): resume from another local profile's state.db.
        self.profile = (params.get("profile") or "").strip() or None
        self.profile_home = _profile_home(self.profile)
        self.lazy, self.defer_history = _flag(params, "lazy"), _flag(params, "defer_history")
        # Desktop hydrates over REST; suppress the duplicate WS copy only when asked.
        self.omit_messages, self.eager_build = _flag(params, "omit_messages"), _flag(params, "eager_build")

    def mint(self, prompts: bool = True) -> tuple:
        """``(runtime sid, source, cwd)`` for the live record this resume registers (+ gateway prompts on)."""
        ids = _new_runtime_ids(self.params)
        if prompts:
            _enable_gateway_prompts()
        return *ids, self.profile_resume_cwd or _default_session_cwd()

    def record(self, source: str, cwd: str, history: list, overrides: dict | None = None, **extra) -> dict:
        """``_deferred_session_record`` with this resume's common fields (lease claimed lazily on turn 1);
        ``overrides`` restores the stored model/provider/reasoning/tier so the deferred build matches eager."""
        if overrides is not None:
            extra.update(model_override=overrides.get("model_override"), resume_runtime_overrides=overrides or None)
        return _deferred_session_record(
            self.target, cols=self.cols, cwd=cwd, history=history, lease=None, source=source,
            close_on_disconnect=_flag(self.params, "close_on_disconnect"),
            profile_home=self.profile_home, explicit_cwd=bool(self.profile_resume_cwd), **extra)

    def claim(self, sid: str, record: dict) -> dict | None:
        """Register ``record`` live under the resume lock, or reuse a concurrent winner's session."""
        live = _claim_or_reuse_live(sid, self.target, record, None)
        return None if live is None else _resume_reuse_live(self, *live)

    def restore(self):
        """``(sanitized model history, display history, raw history)`` for a cold/eager resume."""
        raw, display = self.read_history()
        return sanitize_replay_history(raw), display, raw

    def info(self, cwd: str, overrides: dict) -> dict:
        return _lazy_resume_info(cwd, model=(overrides.get("model_override") or {}).get("model") or "",
                                 provider=overrides.get("provider_override") or "", profile=self.profile)

    def child_history(self, repair: bool) -> list:
        """The child's OWN conversation (no ancestors), row ids included."""
        return self.db.get_messages_as_conversation(self.target, repair_alternation=repair, include_row_ids=True)

    def messages(self, display: list) -> list:
        return [] if self.omit_messages else _history_to_messages(display)

    def read_history(self) -> tuple:
        """One lineage SELECT, two projections: model-fed copy alternation-repaired (healed once
        here instead of every turn's pre-request repair), display copy verbatim."""
        self.db.reopen_session(self.target)
        if self.omit_messages:
            return self.child_history(repair=True), []
        return self.db.get_resume_conversations(self.target)

    def display_prefix(self) -> list:
        """Ancestor display rows (model-fed history drops a dangling tool-call tail — display keeps it)."""
        return [] if self.omit_messages else self.db.get_ancestor_display_prefix(self.target)


def _find_live_unpersisted(needle: str, home) -> str:
    """Runtime sid of a live, not-yet-persisted session matched by stored key or pending title."""
    want_home = str(home) if home is not None else None
    return next((
        live_sid for live_sid, record in list(_sessions.items())
        if isinstance(record, dict) and (record.get("profile_home") or None) == want_home
        and (str(record.get("session_key") or "") == needle or (record.get("pending_title") or "") == needle)), "")


def _resume_live_unpersisted(ctx: _Resume, live_sid: str, live: dict) -> dict:
    """Reattach a LIVE lazy session with no state.db row yet (every fresh Bot Chat; a 404 here killed messaging
    for never-spoken bots). Rebind the transport and cancel the armed orphan-reap Timer (a WS drop may have
    sentinel-parked the record) or it fires against this client."""
    if ctx.owns_db:
        _release_db(ctx.db)
    live["last_active"] = time.time()
    if (transport := current_transport()) is not None:
        # This resume reattaches the live record. A lazy session (no state.db row yet — every fresh Bot
        # Chat) that was sentinel-parked by a WS drop MUST be rebound here, or it keeps the drop sentinel
        # and the armed orphan-reap Timer fires against a client that is attached right now — the
        # unpersisted sibling of the storm-killer paths (#91276).
        with live.setdefault("history_lock", threading.Lock()):
            live["transport"] = transport
            live.setdefault("viewers", {})[transport] = time.time()
    _cancel_ws_orphan_reap(live_sid)
    history = live.get("history") or []
    return _ok(ctx.rid, _attach_todo_state({
        "session_id": live_sid, "stored_session_id": str(live.get("session_key") or ""),
        "message_count": len(history), "messages": ctx.messages(history),
        "info": {"model": _resolve_model(), "lazy": True, "profile_name": ctx.profile or ""}}, live))


def _resume_adopt_stranded(ctx: _Resume) -> None:
    """Adopt a lineage stranded in the DEFAULT store (older builds ran a profile bot's turns on the focused
    tile's backend; unadopted it 4001s forever). Exact-id ONLY — bot titles collide; never a retired donor."""
    try:
        # Stranded-session adoption (#93296 follow-up): before session RPCs routed by their TARGET session,
        # a profile bot's turns executed on the focused tile's backend — usually default — so its canonical
        # session accumulated in the DEFAULT profile's state.db. Now that routing is correct, this
        # profile-scoped resume is the first place the fix and the stranded data collide: the id exists in
        # the default store but not here, and without adoption the same chat 4001s forever (the fix made it
        # unreachable instead of misrouted). Adopt the full lineage from the default store into this
        # profile's db, then retry the lookup. Only profile-scoped resumes reach here (owns_db); unknown ids
        # in the default store still 4007 exactly as before.
        default_db = _get_db()
        donor_row = default_db.get_session(ctx.target) if default_db is not None else None
        if not donor_row or donor_row.get("archived"):
            return
        adoption = ctx.db.adopt_session_lineage_from(default_db, donor_row["id"])
        if adoption.get("adopted"):
            logger.info("adopted stranded session %s (lineage of %s segment(s)) from default store into profile %s",
                        donor_row["id"],
                        len(adoption.get("imported_ids") or []) + len(adoption.get("skipped_ids") or []),
                        ctx.profile or "?")
            ctx.found = ctx.db.get_session(donor_row["id"])
            if ctx.found:
                ctx.target = ctx.found["id"]
    except Exception:
        logger.exception("stranded-session adoption failed for %s", ctx.target)


def _resume_locate(ctx: _Resume) -> dict | None:
    """Resolve ``ctx.target`` to a stored row (``ctx.found``); a dict is an early response."""
    ctx.found = ctx.db.get_session(ctx.target)
    if ctx.found:
        return None
    ctx.found = ctx.db.get_session_by_title(ctx.target)
    if ctx.found:
        ctx.target = ctx.found["id"]
        return None
    if ctx.lazy and _child_run_active(ctx.target):
        # Fresh subagent watch window: `subagent.start` relays BEFORE the child's first DB flush. Proceed lazily
        # with empty history — the live mirror streams the turn and the row exists by upgrade time.
        ctx.found = {}
        return None
    live_sid = _find_live_unpersisted(ctx.target, ctx.profile_home)
    if (live := _sessions.get(live_sid) if live_sid else None) is not None:
        return _resume_live_unpersisted(ctx, live_sid, live)
    if ctx.owns_db:
        _resume_adopt_stranded(ctx)
    return None if ctx.found else _err(ctx.rid, 4007, "session not found")


def _resume_follow_tip(ctx: _Resume) -> None:
    """Rebind a rotated-out parent id to its compression tip (resuming the original reloads the parent
    transcript and loses the post-compression reply). Skipped for lazy watch windows (exact child); Bot Chat
    follows proven compression edges only."""
    if not ctx.found or ctx.lazy:
        return
    tip = ctx.target
    with contextlib.suppress(Exception):
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        if (ctx.found.get("title") or "").strip() == BOT_CHAT_TITLE:
            tip = ctx.db.get_compression_tip(ctx.target) or ctx.target
        else:
            tip = ctx.db.resolve_resume_session_id(ctx.target)
    if tip and tip != ctx.target:
        ctx.target = tip
        ctx.found = ctx.db.get_session(tip) or ctx.found


def _resume_guard(ctx: _Resume) -> dict | None:
    """Refuse a runaway transcript before any history read (sessions.max_resume_messages). Deferred /
    omit_messages / lazy paths load the TIP segment only and are guarded tip-only (a lineage count rejected
    exactly the well-compressed chats). Metadata fallback for lightweight adaptor DBs; fails OPEN on errors."""
    from hermes_state import SessionResumeTooLargeError, resolved_max_resume_messages
    tip_only = ctx.lazy or ctx.omit_messages or (ctx.defer_history and not ctx.eager_build)
    try:
        if callable(safety_check := getattr(ctx.db, "assert_resume_safe", None)):
            safety_check(ctx.target, **({"tip_only": True} if tip_only else {}))
        elif (limit := resolved_max_resume_messages()) and (n := int(ctx.found.get("message_count") or 0)) > limit:
            raise SessionResumeTooLargeError(n, limit)
    except SessionResumeTooLargeError as exc:
        return _err(ctx.rid, 4130, str(exc))
    except Exception as exc:
        logger.warning("resume safety check failed for %s (proceeding without guard): %s", ctx.target, exc)
    return None


def _resume_reuse_live(ctx: _Resume, sid: str, session: dict) -> dict:
    """Reattach an already-live session under the resume lock (held across the client-gone check,
    transport rebind and reap cancel so grace expiry is atomic)."""
    with _session_resume_lock:
        if _sessions.get(sid) is not session:
            return _err(ctx.rid, 4007, "session no longer live; retry resume")
        if session.get("_client_gone_interrupt_requested"):
            return _err(ctx.rid, 4009, "session disconnect interrupt settling")
        _cancel_ws_orphan_reap(sid)  # unconditionally: the fast path must never race the reap Timer
        payload = _live_session_payload(sid, session, cols=ctx.cols, touch=True, omit_messages=ctx.omit_messages,
                                        transport=current_transport() or _stdio_transport)
        payload["resumed"] = ctx.target
        if ctx.defer_history:
            payload.update(messages=[], hydrating=bool(session.get("resume_hydrating")),
                           message_count=int(session.get("resume_message_count") or payload["message_count"]))
        # A lazy watch session never owns a run loop — overlay the child-run registry.
        if session.get("agent") is None and _child_run_active(ctx.target):
            payload.update(running=True, status="streaming")
        return _ok(ctx.rid, payload)


def _resume_response(
    ctx: _Resume, sid: str, record: dict, *, info: dict, display: list = (), count_source: list | None = None,
    messages: list | None = None, message_count: int | None = None, running: bool = False,
    status: str = "idle", hydrating: bool | None = None, started_at=None, auto_continue=None,
) -> dict:
    """Common resume payload; omit_messages counts ``count_source`` (client still learns the stored size)."""
    if messages is None:
        messages = ctx.messages(display)
    if message_count is None:
        message_count = len(count_source) if ctx.omit_messages else len(messages)
    payload = {"session_id": sid, "resumed": ctx.target, "message_count": message_count, "messages": messages,
               **({"messages_omitted": ctx.omit_messages} if hydrating is None else {"hydrating": hydrating}),
               "info": info, "inflight": None, "running": running, "session_key": ctx.target,
               "started_at": record["created_at"] if started_at is None else started_at, "status": status}
    if auto_continue is not None:
        payload["auto_continue"] = auto_continue
    return _ok(ctx.rid, _attach_todo_state(payload, record))


def _resume_lazy(ctx: _Resume) -> dict:
    """Lazy/watch resume (desktop subagent windows): a live session WITHOUT an agent — the child runs
    inside the parent's turn, so the window needs stored history + a transport; prompt.submit upgrades it."""
    sid, source, cwd = ctx.mint(prompts=False)
    try:
        ctx.db.reopen_session(ctx.target)
        # repair_alternation heals a durable ``user;user`` once here.
        history = ctx.child_history(repair=True)
    except Exception as e:
        return _err(ctx.rid, 5000, f"resume failed: {e}")
    record = ctx.record(source, cwd, history, lazy=True, todo_state=_todo_state_from_history(history))
    if (reused := ctx.claim(sid, record)) is not None:
        return reused
    # A child mid-run emits no session events — liveness comes from the relay registry.
    running = _child_run_active(ctx.target)
    # Display uses the VERBATIM child-only projection so model-invisible rows survive; repaired ``history``
    # still feeds live replay.
    display = history
    try:
        display = ctx.child_history(repair=False)
    except Exception:
        logger.debug("child-watch display projection read failed", exc_info=True)
    return _resume_response(ctx, sid, record, info=_lazy_resume_info(cwd, profile=ctx.profile), display=display,
                            count_source=display, running=running, status="streaming" if running else "idle")


def _resume_deferred(ctx: _Resume) -> dict:
    """Bounded ack; the transcript hydrates in the background (the ONE history read) and pages over REST."""
    sid, source, cwd = ctx.mint()
    overrides = _stored_session_runtime_overrides(ctx.found)
    record = ctx.record(source, cwd, [], overrides)
    record.update(resume_history_ready=threading.Event(), resume_hydrating=True,
                  resume_message_count=int(ctx.found.get("message_count") or 0))
    if (reused := ctx.claim(sid, record)) is not None:
        return reused
    _schedule_resume_hydration(sid, ctx.target, ctx.db, close_db=ctx.owns_db)
    ctx.owns_db = False  # the hydration worker now owns (and closes) the profile-scoped handle
    _schedule_session_cap_enforcement()
    return _resume_response(ctx, sid, record, info=ctx.info(cwd, overrides), messages=[],
                            message_count=record["resume_message_count"], status="resuming", hydrating=True)


def _resume_cold(ctx: _Resume) -> dict:
    """Default cold resume: transcript now, agent OFF the response path (_make_agent can block for seconds;
    callers await this RPC before painting) — pre-warmed on a timer, _sess() builds on demand if the first
    prompt beats it. Unlike lazy, restores full ancestor history + persisted runtime identity."""
    sid, source, cwd = ctx.mint()
    try:
        history, display_history, raw_history = ctx.restore()
    except Exception as e:
        return _err(ctx.rid, 5000, f"resume failed: {e}")
    overrides = _stored_session_runtime_overrides(ctx.found)
    record = ctx.record(source, cwd, history, overrides, display_history_prefix=ctx.display_prefix(),
                        todo_state=_todo_state_from_history(history))
    if (reused := ctx.claim(sid, record)) is not None:
        return reused
    _schedule_agent_build(sid)
    _schedule_session_cap_enforcement()  # trim detached idle sessions over the cap
    return _resume_response(ctx, sid, record, info=ctx.info(cwd, overrides), display=display_history,
                            count_source=raw_history,
                            auto_continue=_maybe_schedule_auto_continue(sid, record, ctx.target))


def _resume_eager(ctx: _Resume) -> dict:
    """Synchronous build OUTSIDE _session_resume_lock (it would stall session.close), then double-checked."""
    sid, source, _cwd = ctx.mint()
    with _profile_build_scope(ctx.profile_home):
        try:
            history, display_history, raw_history = ctx.restore()
            display_history_prefix = ctx.display_prefix()
            # Profile db so turns persist to the right state.db; stored runtime identity so switching chats does
            # not inherit another chat's global model.
            stored_runtime_overrides = _stored_session_runtime_overrides(ctx.found)
            agent = _make_agent_in_context(
                sid, ctx.target, session_db=ctx.db, platform_override=source,
                context_cwd_is_launch_artifact=(source in _LAUNCH_CWD_NOT_A_WORKSPACE and not ctx.profile_resume_cwd),
                **stored_runtime_overrides)
        except Exception as e:
            return _err(ctx.rid, 5000, f"resume failed: {e}")
    with _session_resume_lock:
        live = _find_live_session_by_key(ctx.target, ctx.profile_home)
        if live is not None:
            with contextlib.suppress(Exception):
                agent.close()
            return _resume_reuse_live(ctx, *live)
        try:
            with _profile_build_scope(ctx.profile_home):
                _init_session(sid, ctx.target, agent, history, cols=ctx.cols, cwd=ctx.profile_resume_cwd,
                              session_db=ctx.db, source=source, explicit_cwd=bool(ctx.profile_resume_cwd))
                # Ownership TRANSFER: the agent holds the handle for life (AIAgent.close() releases it). The
                # owns_db drop is UNCONDITIONAL — the session is registered against the handle, so the finally
                # must not close it even if the transfer was refused (a leak beats "closed database" every
                # turn). Gated on owns_db: the SHARED launch handle must never move onto one session.
                if ctx.owns_db:
                    _transfer_db_to_agent(agent, ctx.db)
                ctx.owns_db = False
            if (session := _sessions.get(sid)) is not None:
                if stored_runtime_overrides.get("model_override") is not None:
                    session["model_override"] = stored_runtime_overrides["model_override"]
                # Each turn re-binds HERMES_HOME (mid-turn memory/skills reads); lease claimed lazily on turn 1.
                if ctx.profile_home is not None:
                    session["profile_home"] = str(ctx.profile_home)
                session.update(display_history_prefix=display_history_prefix, active_session_lease=None)
        except Exception as e:
            # _init_session registers _sessions[sid] BEFORE its first db read; left in place the fast path
            # would serve that dead session forever.
            if ctx.owns_db:
                with _sessions_lock:
                    _sessions.pop(sid, None)
            return _err(ctx.rid, 5000, f"resume failed: {e}")
        session = _sessions.get(sid) or {}
    return _resume_response(
        ctx, sid, session, info=_session_info(agent, session), display=display_history, count_source=raw_history,
        started_at=float(session.get("created_at") or time.time()),
        auto_continue=_maybe_schedule_auto_continue(sid, session, ctx.target) if session else None)


@method("session.resume")
def _(rid, params: dict) -> dict:
    if not (target := params.get("session_id", "")):
        return _err(rid, 4006, "session_id required")
    ctx = _Resume(rid, params, target)
    # Profile scope: a DEDICATED handle we own until the agent takes it; else the shared launch db.
    ctx.db, ctx.owns_db = _profile_session_db(ctx.profile_home)
    try:
        if ctx.db is None:
            return _db_unavailable_error(rid, code=5000)
        if (resp := _resume_locate(ctx)) is not None:
            return resp
        _resume_follow_tip(ctx)
        if (resp := _resume_guard(ctx)) is not None:
            return resp
        ctx.profile_resume_cwd = _str_param(ctx.found, "cwd") or _profile_configured_cwd(ctx.profile_home)
        # Fast path: reuse a session live IN THIS PROFILE (never another profile's runtime).
        with _session_resume_lock:
            live = _find_live_session_by_key(ctx.target, ctx.profile_home)
        if live is not None:
            return _resume_reuse_live(ctx, *live)
        if ctx.lazy:
            return _resume_lazy(ctx)
        if ctx.eager_build:
            return _resume_eager(ctx)
        return _resume_deferred(ctx) if ctx.defer_history else _resume_cold(ctx)
    finally:
        # Refcounting alone does not release the sqlite fds: SessionDB pins ITSELF (atexit.register) once its
        # background token writer starts; only close() unregisters.
        if ctx.owns_db and ctx.db is not None:
            with contextlib.suppress(Exception):
                ctx.db.close()


# ── cwd / workspace / live-session bookkeeping ───────────────────────
@_session_method("session.cwd.set")
def _(rid, params: dict, session: dict) -> dict:
    if session.get("running"):
        return _err(rid, 4009, "session busy")
    if not (raw := _str_param(params, "cwd")):
        return _err(rid, 4016, "cwd required")
    try:
        cwd = _set_session_cwd(session, raw)
    except ValueError as e:
        return _err(rid, 4017, str(e))
    info = _cwd_info(session, cwd)
    _emit("session.info", params.get("session_id", ""), info)
    return _ok(rid, info)


@method("session.workspace.move")
def _(rid, params: dict) -> dict:
    """Re-home a STORED session's workspace (by ``session_key``; no live agent required). git branch/root are
    REPLACED (a stale ``git_repo_root`` kept the session under the project it left); a live agent follows even
    mid-turn (refusing made the UI claim success while state.db kept the old cwd)."""
    if not (target := _str_param(params, "session_key")):
        return _err(rid, 4007, "session_key required")
    if not (raw := _str_param(params, "cwd")):
        return _err(rid, 4016, "cwd required")
    from hermes_constants import translate_cwd_for_wsl_backend
    resolved = os.path.abspath(os.path.expanduser(translate_cwd_for_wsl_backend(raw)))
    if not os.path.isdir(resolved):
        return _err(rid, 4017, f"working directory does not exist: {raw}")
    # Snapshot under the lock — concurrent RPCs mutate _sessions.
    with _sessions_lock:
        live_sid, live = next(
            ((sid, sess) for sid, sess in list(_sessions.items()) if sess.get("session_key") == target), ("", None))
    branch, root = git_probe.branch(resolved), git_probe.common_repo_root(resolved)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        # A draft has no row yet; the live re-home still applies (row inherits cwd on write).
        if not db.get_session(target):
            if live is None:
                return _err(rid, 4007, "session not found")
        else:
            try:
                db.update_session_cwd(target, resolved, branch, root, replace_git_meta=True)
            except Exception as e:
                return _err(rid, 5007, f"move failed: {e}")
    if live is not None:
        try:
            _set_session_cwd(live, resolved)
        except ValueError as e:
            return _err(rid, 4017, str(e))
        _emit("session.info", live_sid, _cwd_info(live, resolved, branch=branch))
    return _ok(rid, {"cwd": resolved, "branch": branch, "git_repo_root": root})


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
    return _ok(rid, {"sessions": rows})


@_session_method("session.activate")
def _(rid, params: dict, session: dict) -> dict:
    """Attach the frontend to a live TUI session without closing the previously focused one."""
    return _ok(rid, _live_session_payload(
        str(params.get("session_id") or ""), session, touch=True, transport=current_transport() or _stdio_transport,
        omit_messages=is_truthy_value(params.get("omit_messages", False))))


@method("session.delete")
def _(rid, params: dict) -> dict:
    """Delete a stored session + transcripts; refused while live here (FK trips on the agent's next flush)."""
    if not (target := params.get("session_id", "")):
        return _err(rid, 4006, "session_id required")
    snapshot, err = _snapshot_sessions(rid)
    if err:
        return err
    if any(s.get("session_key") == target for _sid, s in snapshot):
        return _err(rid, 4023, "cannot delete an active session")
    profile_home = _profile_home((params.get("profile") or "").strip() or None)
    with _profile_db(params) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5036)
        try:
            home = Path(profile_home) if profile_home is not None else get_hermes_home()
            deleted = db.delete_session(target, sessions_dir=home / "sessions")
        except Exception as e:
            return _err(rid, 5036, f"delete failed: {e}")
    return _ok(rid, {"deleted": target}) if deleted else _err(rid, 4007, "session not found")


def _title_read(session: dict, db, key: str) -> str:
    """``session.title`` without ``title``: read it, applying a queued pending_title if possible."""
    fallback = session.get("pending_title") or ""
    try:
        resolved_title = db.get_session_title(key) or ""
        if not fallback:
            if resolved_title:
                session["pending_title"] = None
        elif (db.set_session_title(key, fallback)
              or ((db.get_session(key) or {}).get("title") or "").strip() == fallback):
            session["pending_title"] = None
            resolved_title = fallback
        elif not resolved_title:
            resolved_title = fallback
    except Exception:
        resolved_title = fallback
    return resolved_title


@method("session.title")
@_with_db(5007, session_scoped=True)
def _(rid, params: dict, session: dict, db) -> dict:
    key = session["session_key"]
    if "title" not in params:
        result = {"title": _title_read(session, db, key), "session_key": key}
    elif not (title := (params.get("title", "") or "").strip()):
        return _err(rid, 4021, "title required")
    else:
        try:
            if db.set_session_title(key, title):
                pending, value = False, title
            # rowcount == 0 can mean "same value" as well as "missing row".
            elif existing_row := db.get_session(key):
                pending, value = False, existing_row.get("title") or title
            else:
                # No row yet: an explicit /title is clear intent, so persist the row NOW (as the gateway's
                # _handle_title_command); the min-messages sidebar filter hides a titled 0-message row. If
                # row creation didn't take, queue so the post-turn apply block can recover.
                _ensure_session_db_row(session)
                with _session_db(session) as scoped_db:
                    pending, value = not (scoped_db is not None and scoped_db.set_session_title(key, title)), title
        except ValueError as e:
            return _err(rid, 4022, str(e))
        except Exception as e:
            return _err(rid, 5007, str(e))
        session["pending_title"] = value if pending else None
        result = {"pending": pending, "title": value}
    _emit_session_info_for_session(params.get("session_id", ""), session)
    return _ok(rid, result)


@method("session.set_hidden")
def _(rid, params: dict) -> dict:
    """Set/clear ``hidden`` (leaves the default list, stays resumable by its owner) on a session + lineage:
    LIVE runtime id first (unpersisted drafts via ``pending_hidden``), then a stored id/key in the profile db."""
    hidden = is_truthy_value(params.get("hidden", True))
    session, err = _sess_nowait(params, rid)
    with (_profile_db(params) if session is None else _session_db(session)) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if session is not None:
                key = session["session_key"]
                if not db.set_session_hidden(key, hidden):
                    session["pending_hidden"] = hidden  # no row yet: _ensure_session_db_row is born hidden
            else:
                # ``resolve_session_id`` follows key/title aliases like the REST pin/archive path.
                target = _str_param(params, "session_id")
                if not (key := db.resolve_session_id(target) if hasattr(db, "resolve_session_id") else target):
                    return err
                db.set_session_hidden(key, hidden)
            return _ok(rid, {"hidden": hidden, "session_key": key})
        except Exception as e:
            return _err(rid, 5007, str(e))


@_session_method("message.react")
def _(rid, params: dict, session: dict) -> dict:
    """Set/clear one author's emoji reaction (Tapback semantics: one per author, same emoji retracts, null
    clears). ``row_id`` is ``messages.id``; a not-yet-persisted live message names ``newest_role`` instead."""
    newest_role = _str_param(params, "newest_role")
    row_id = params.get("row_id")
    if row_id is None and newest_role not in {"user", "assistant"}:
        return _err(rid, 4023, "row_id or newest_role required")
    if (emoji := params.get("emoji")) is not None and not (emoji := str(emoji).strip()):
        return _err(rid, 4024, "emoji must be a non-empty string or null")
    if (author := str(params.get("author") or "user").strip()) not in {"user", "agent"}:
        return _err(rid, 4025, "author must be 'user' or 'agent'")
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if row_id is None:
                row_id = db.latest_message_row_id(session["session_key"], role=newest_role)
                if row_id is None:
                    return _err(rid, 4040, "no message to react to yet")
            reactions = db.set_message_reaction(session["session_key"], int(row_id), emoji, author=author)
        except Exception as e:
            return _err(rid, 5007, str(e))
    if reactions is None:
        return _err(rid, 4040, "message not found in this session")
    return _ok(rid, {"row_id": int(row_id), "reactions": reactions})


@method("llm.oneshot")
def _(rid, params: dict) -> dict:
    """Stateless one-shot LLM request; a live ``session_id`` lends its model, else the ``task`` backend."""
    template = (params.get("template") or "").strip() or None
    instructions = params.get("instructions") or ""
    user_input = params.get("input") or ""
    variables = params.get("variables") if isinstance(params.get("variables"), dict) else {}
    try:
        temperature = float(params["temperature"]) if params.get("temperature") is not None else 0.3
    except (TypeError, ValueError):
        temperature = 0.3
    if not template and not str(instructions).strip() and not str(user_input).strip():
        return _err(rid, 4030, "llm.oneshot requires a template or instructions/input")
    session = _sessions.get(params.get("session_id") or "")
    try:
        from agent.oneshot import run_oneshot
        return _ok(rid, {"text": run_oneshot(
            instructions=instructions, user_input=user_input, template=template, variables=variables,
            task=(params.get("task") or "title_generation").strip() or "title_generation",
            max_tokens=_int_param(params, "max_tokens", 1024) or 1024, temperature=temperature,
            main_runtime=_main_runtime_from_agent(session.get("agent")) if session else None)})
    except (KeyError, ValueError) as e:
        return _err(rid, 4031 if isinstance(e, KeyError) else 4032, str(e))
    except Exception as e:
        logger.warning("llm.oneshot failed: %s", e)
        return _err(rid, 5030, f"one-shot generation failed: {e}")


# ── handoff ──────────────────────────────────────────────────────────
@_session_method("handoff.request")
def _(rid, params: dict, session: dict) -> dict:
    """Queue a handoff (desktop /handoff): only writes ``pending``; the gateway watcher claims and re-binds."""
    if session.get("running"):
        return _err(rid, 4009, "session busy — wait for the current turn to finish, then retry the handoff")
    if not (platform_name := (params.get("platform", "") or "").strip().lower()):
        return _err(rid, 4023, "platform required")
    # Validate up front: an unconfigured platform / missing home channel pends forever.
    from gateway.config import Platform, load_gateway_config
    try:
        platform = Platform(platform_name)
    except (ValueError, KeyError):
        return _err(rid, 4024, f"unknown platform '{platform_name}'")
    try:
        with _session_profile_runtime_scope(session):
            gw_config = load_gateway_config()
    except Exception as e:
        return _err(rid, 5021, f"could not load gateway config: {e}")
    if not getattr(gw_config.platforms.get(platform), "enabled", False):
        return _err(rid, 4025, f"platform '{platform_name}' is not configured/enabled in the gateway")
    if not (home := gw_config.get_home_channel(platform)) or not home.chat_id:
        return _err(rid, 4026, f"no home channel configured for {platform_name} — set one with "
                    "/sethome on the destination chat first")
    # The watcher transfers a persisted row, so make sure one exists for an empty chat.
    _ensure_session_db_row(session)
    key = session["session_key"]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        try:
            if not db.get_session(key):
                db.set_session_title(key, f"handoff-{key[:8]}")
            if not db.request_handoff(key, platform_name):
                return _err(rid, 4027, "session is already in flight for handoff — wait for it to settle, then retry")
        except Exception as e:
            return _err(rid, 5007, str(e))
    return _ok(rid, {"queued": True, "session_key": key, "platform": platform_name, "home_name": home.name})


@method("handoff.state")
@_with_db(5007, session_scoped=True)
def _(rid, params: dict, session: dict, db) -> dict:
    """Poll ``{state, platform, error}``; ``state`` is pending|running|completed|failed or empty."""
    record = db.get_handoff_state(session["session_key"]) or {}
    return _ok(rid, {field: record.get(field) or "" for field in ("state", "platform", "error")})


@method("handoff.fail")
def _(rid, params: dict) -> dict:
    """Fail a not-yet-claimed handoff (poll timeout); a claimed ``running`` row is the watcher's (CAS)."""
    # Undecorated on purpose: tests rebind this handler's __code__ directly.
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    reason = str(params.get("error") or "handoff failed").strip()[:500]
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5007)
        key = session["session_key"]
        try:
            failed = db.fail_handoff(key, reason, only_states=("pending",))
        except TypeError:
            # Older SessionDB without only_states: fail only when still pending.
            if failed := ((db.get_handoff_state(key) or {}).get("state") or "") == "pending":
                db.fail_handoff(key, reason)
        state = "failed" if failed else (db.get_handoff_state(key) or {}).get("state") or ""
    return _ok(rid, {"failed": bool(failed), "state": state})


# ── usage ────────────────────────────────────────────────────────────
@_session_method("session.usage")
def _(rid, params: dict, session: dict) -> dict:
    usage: dict = _session_usage_snapshot(session)
    if session.get("agent") is None and not usage:
        usage = {"calls": 0, "input": 0, "output": 0, "total": 0}
    # Nous credits are agent-independent (portal fetch); fail-open when absent.
    with contextlib.suppress(Exception):
        from agent.account_usage import nous_credits_lines
        if credits := nous_credits_lines():
            usage["credits_lines"] = credits
    return _ok(rid, usage)


@_session_method("session.context_breakdown")
def _(rid, params: dict, session: dict) -> dict:
    if (agent := session.get("agent")) is None:
        usage = _session_usage_snapshot(session) or _get_usage(None)
        return _ok(rid, {
            "categories": [], "context_max": usage.get("context_max", 0) or 0,
            "context_percent": usage.get("context_percent", 0) or 0,
            "context_used": usage.get("context_used", 0) or 0,
            "estimated_total": usage.get("context_used", 0) or usage.get("total", 0) or 0,
            "model": _metadata_mirror(session).get("model", "")})
    with session["history_lock"]:
        history = list(session.get("history", []))
    try:
        from agent.context_breakdown import compute_session_context_breakdown
        return _ok(rid, compute_session_context_breakdown(agent, history))
    except Exception as exc:
        return _err(rid, 5000, f"Could not compute context breakdown: {exc}")


# ── pet ──────────────────────────────────────────────────────────────
_PET_OFF = {"enabled": False}


@_pet_method("pet.info", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Active pet for sprite renderers: spritesheet (base64) + frame geometry + state-row taxonomy."""
    if (active := _active_pet()) is None:
        return _ok(rid, {"enabled": False})
    pet, scale = active
    payload = {"enabled": True, **_pet_sprite_payload(pet, scale=scale)}
    # Send-once for the multi-MB sheet: same revision → metadata only.
    if (known := str(params.get("knownRevision", "") or "")) and known == payload.get("spritesheetRevision"):
        # Send-once semantics for the multi-MB spritesheet (#54730): a caller that already holds the sheet
        # passes the revision it has, and an unchanged sheet comes back as metadata only
        # (spritesheetUnchanged).
        payload.pop("spritesheetBase64", None)
        payload["spritesheetUnchanged"] = True
    return _ok(rid, payload)


@_pet_method("pet.info.meta", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Cheap active-pet metadata used to avoid full payload refreshes."""
    if (active := _active_pet()) is None:
        return _ok(rid, {"enabled": False})
    pet, scale = active
    return _ok(rid, {"enabled": True, "slug": pet.slug, "displayName": pet.display_name, "scale": scale,
                     "spritesheetRevision": _pet_sheet_revision(pet.spritesheet)})


def _pet_kitty_cells(pet, pet_cfg: dict, state: str, scale: float) -> dict | None:
    """kitty payload for a TTY that speaks it (dashboard PTY falls through); only kitty is grid-safe in Ink."""
    from agent.pet import constants, render
    from agent.pet.render import PetRenderer
    configured = str(pet_cfg.get("render_mode", "auto") or "auto").lower()
    if (render.detect_terminal_graphics() if configured in ("", "auto") else configured) != "kitty":
        return None
    image_id = render.kitty_image_id(pet.slug)
    # kitty sizes from scaled pixels, so unicode_cols is moot here.
    payload = PetRenderer(str(pet.spritesheet), mode="kitty", scale=scale).kitty_payload(state, image_id=image_id)
    if not payload:
        return None
    return {"graphics": "kitty", "imageId": image_id, "color": render.kitty_color_hex(image_id),
            "cols": payload["cols"], "rows": payload["rows"], "placeholder": payload["placeholder"],
            "frames": payload["frames"], "frameMs": constants.LOOP_MS / max(1, len(payload["frames"]) or 1),
            "scale": scale}


@_pet_method("pet.cells", fail_open=_PET_OFF)
def _(rid, params: dict) -> dict:
    """Half-block cell frames (``[tr,tg,tb,ta, br,bg,bb,ba]``) for one pet ``state``; ``cols``, ``graphics``."""
    from agent.pet import constants, store
    from agent.pet.render import PetRenderer
    pet_cfg = _pet_display_cfg()
    pet = None
    if is_truthy_value(pet_cfg.get("enabled"), default=False):
        pet = store.resolve_active_pet(str(pet_cfg.get("slug", "") or ""))
    if pet is None or not pet.exists:
        return _ok(rid, {"enabled": False})
    state = str(params.get("state") or constants.PetState.IDLE.value)
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    cols = int(params.get("cols") or 0) or constants.resolve_cols(scale, pet_cfg.get("unicode_cols", 0))
    base = {"enabled": True, "slug": pet.slug, "displayName": pet.display_name, "state": state}
    if params.get("graphics") and (kitty := _pet_kitty_cells(pet, pet_cfg, state, scale)):
        return _ok(rid, {**base, **kitty})
    renderer = PetRenderer(str(pet.spritesheet), mode="unicode", scale=scale, unicode_cols=cols)
    count = renderer.frame_count(state) or 1
    frames = [[[[*top, *bottom] for (top, bottom) in row] for row in renderer.cells(state, i, cols=cols)]
              for i in range(count)]
    return _ok(rid, {**base, "cols": cols, "frameMs": constants.LOOP_MS / max(1, count), "frames": frames,
                     "scale": scale})


@_pet_method("pet.gallery", fail_open={"enabled": False, "active": "", "pets": []})
def _(rid, params: dict) -> dict:
    """Petdex gallery + local install state (installed-only offline); ``localOnly`` skips the remote manifest."""
    local_only = bool(params.get("localOnly"))
    from agent.pet import store
    pet_cfg = _pet_display_cfg()
    installed = {p.slug: p for p in store.installed_pets()}
    gallery: list[dict] = []
    try:
        from agent.pet.manifest import fetch_manifest, prefetch
        # Local-only still warms the manifest cache in the background.
        if local_only:
            prefetch()
        for entry in [] if local_only else fetch_manifest():
            gallery.append({
                "slug": entry.slug, "displayName": entry.display_name, "installed": entry.slug in installed,
                "spritesheetUrl": entry.spritesheet_url,
                # No popularity metric; petdex's hand-picked set (by asset path) is closest.
                "curated": "/curated/" in entry.spritesheet_url,
                "generated": entry.slug in installed and installed[entry.slug].generated})
    except Exception as exc:  # noqa: BLE001 - offline: fall back to installed
        logger.debug("pet.gallery manifest fetch failed: %s", exc)
    seen = {item["slug"] for item in gallery}
    gallery.extend(
        {"slug": slug, "displayName": pet.display_name, "installed": True, "spritesheetUrl": "",
         "generated": pet.generated}
        for slug, pet in installed.items() if slug not in seen)
    return _ok(rid, {"enabled": is_truthy_value(pet_cfg.get("enabled"), default=False),
                     "active": str(pet_cfg.get("slug", "") or ""), "pets": gallery})


@_pet_method("pet.select", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Adopt a pet: install (if needed) + activate; writes ``display.pet.*`` to config."""
    from agent.pet import store
    from agent.pet.manifest import ManifestError
    from hermes_cli.pets import _set_active
    try:
        pet = store.install_pet(slug)
    except (store.PetStoreError, ManifestError) as exc:
        return _err(rid, 5031, f"could not adopt '{slug}': {exc}")
    _set_active(slug)
    return _ok(rid, {"ok": True, "slug": slug, "displayName": pet.display_name})


@_pet_method("pet.remove", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Uninstall a pet (delete its directory); if it was active, turn the display off."""
    from agent.pet import store
    from hermes_cli.pets import _clear_active_if
    removed = store.remove_pet(slug)
    _pet_config_followup("pet.remove", _clear_active_if, slug)
    return _ok(rid, {"ok": removed, "slug": slug})


def _pet_config_followup(what: str, fn, *args) -> None:
    """Best-effort ``hermes_cli.pets`` active-slug update after a store op that already succeeded."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s config update failed: %s", what, exc)


def _b64(data: bytes) -> str:
    import base64
    return base64.standard_b64encode(data).decode("ascii")


@_pet_method("pet.export", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Export an installed pet as a re-importable ``.zip`` → ``{ok, filename, zipBase64}``."""
    from agent.pet import store
    filename, data = store.export_pet(slug)
    return _ok(rid, {"ok": True, "filename": filename, "zipBase64": _b64(data)})


@_pet_method("pet.rename", slug=True)
def _(rid, params: dict, slug: str) -> dict:
    """Rename a pet's display name + realign its slug/dir; follows the active slug in config."""
    if not (name := _str_param(params, "name")):
        return _err(rid, 4004, "missing name")
    from agent.pet import store
    if not (new_slug := store.rename_pet(slug, name)):
        return _err(rid, 5031, "pet.rename failed")
    if new_slug != slug:
        from hermes_cli.pets import _rename_active_if
        _pet_config_followup("pet.rename", _rename_active_if, slug, new_slug)
    return _ok(rid, {"ok": True, "slug": new_slug, "displayName": name})


@_pet_method("pet.thumb", slug=True, fail_open=lambda params: {"ok": False, "slug": _str_param(params, "slug")})
def _(rid, params: dict, slug: str) -> dict:
    """Idle-frame PNG data URI for the picker (desktop CSP breaks CDN ``<img>``); ``url``: not-yet-installed."""
    from agent.pet import store
    if not (data := store.thumbnail_png(slug, source_url=str(params.get("url") or ""))):
        return _ok(rid, {"ok": False, "slug": slug})
    return _ok(rid, {"ok": True, "slug": slug, "dataUri": "data:image/png;base64," + _b64(data)})


@_pet_method("pet.disable")
def _(rid, params: dict) -> dict:
    """``display.pet.enabled=false`` from the desktop picker."""
    from hermes_cli.pets import _set_enabled
    _set_enabled(False)
    return _ok(rid, {"ok": True})


@_pet_method("pet.scale")
def _(rid, params: dict) -> dict:
    """Persist ``display.pet.scale`` (clamped to engine bounds) from the desktop slider."""
    from hermes_cli.pets import set_pet_scale
    scale, err = set_pet_scale(params.get("scale"))
    return _err(rid, 4004, err) if err else _ok(rid, {"ok": True, "scale": scale})


@method("pet.cancel")
def _(rid, params: dict) -> dict:
    """Stop an in-flight generate/hatch by token (idempotent; off the pool so it lands mid-generation)."""
    if token := _str_param(params, "token"):
        _pet_cancel_request(token)
    return _ok(rid, {"ok": True})


@_pet_method("pet.generate.status", scoped=False, fail_open={"available": False, "providers": []})
def _(rid, params: dict) -> dict:
    """Whether pet generation is possible: a reference-capable image backend is configured."""
    from agent.pet.generate.imagegen import GenerationError, list_sprite_providers, resolve_provider
    available, providers = True, []
    try:
        resolve_provider(require_references=True)
    except GenerationError:
        available = False
    try:
        providers = list_sprite_providers()
    except Exception as exc:  # noqa: BLE001 - picker is best-effort
        logger.debug("pet provider list failed: %s", exc)
    return _ok(rid, {"available": available, "providers": providers})


def _pet_pick_provider(params: dict, *, require_references: bool):
    """Picker-chosen ``params.provider`` resolved up front (a bad pick fails fast, not mid-fan-out)."""
    from agent.pet.generate.imagegen import resolve_provider
    name = _str_param(params, "provider")
    return resolve_provider(require_references=require_references, prefer=name) if name else None


@_pet_method("pet.generate", scoped=False)
def _(rid, params: dict) -> dict:
    """Candidate base looks for a new pet (draft step; worker pool): ``prompt`` (or a ``referenceImage``
    data URL), ``count`` (≤4), ``style``, ``provider`` → ``{ok, token, drafts:[{index, dataUri}]}``."""
    prompt = _str_param(params, "prompt")
    ref_raw = _str_param(params, "referenceImage")
    if not prompt and not ref_raw:
        return _err(rid, 4004, "missing prompt")
    count = max(1, min(4, _int_param(params, "count", 4) or 4))
    import shutil
    from agent.pet.generate import generate_base_drafts
    from agent.pet.generate.imagegen import GenerationError
    root = _pet_gen_root()
    _pet_gen_sweep(root)
    # Token up front so each draft is staged + streamed the moment it lands.
    token = uuid.uuid4().hex[:12]
    _pet_cancel_arm(token)
    stage = root / token
    stage.mkdir(parents=True, exist_ok=True)
    reference_images = None
    if ref_raw:
        try:
            reference_images = _pet_reference_images_from_data_url(ref_raw, stage)
        except ValueError as exc:
            return _pet_gen_abort(rid, token, 4004, str(exc))
    try:
        sprite = _pet_pick_provider(params, require_references=bool(reference_images))
    except GenerationError as exc:
        return _pet_gen_abort(rid, token, 5031, str(exc))
    out: list[dict] = []
    # Token-only init event so a Stop fired before the first draft can target this run.
    _pet_emit("pet.generate.progress", {"token": token, "count": count}, "pet.generate init")

    def _on_draft(index: int, src) -> None:
        dest = stage / f"draft-{index}.png"
        try:
            shutil.copyfile(src, dest)
            data_uri = _pet_png_data_uri(dest)
        except Exception as exc:  # noqa: BLE001 - skip a bad draft, keep the rest
            logger.debug("pet.generate draft %d failed: %s", index, exc)
            return
        out.append({"index": index, "dataUri": data_uri})
        _pet_emit("pet.generate.progress", {"token": token, "index": index, "dataUri": data_uri, "count": count},
                  "pet.generate progress")
    try:
        generate_base_drafts(prompt or "a pet based on the reference image", n=count,
                             style=_str_param(params, "style", "auto"), reference_images=reference_images,
                             provider=sprite, on_draft=_on_draft, is_cancelled=lambda: _pet_is_cancelled(token))
    except GenerationError as exc:
        return _pet_gen_abort(rid, token, 5031, str(exc))
    cancelled = _pet_is_cancelled(token)
    _pet_cancel_release(token)
    if cancelled or not out:
        return _err(rid, 5031, "generation cancelled" if cancelled else "generation produced no usable drafts")
    return _ok(rid, {"ok": True, "token": token, "drafts": sorted(out, key=lambda d: d["index"])})


@_pet_method("pet.hatch", scoped=False)
def _(rid, params: dict) -> dict:
    """Turn a base draft (``token`` + ``index``) into a full pet — installed but NOT active (``pet.select``
    adopts, ``pet.remove`` discards) → ``{ok, slug, displayName, warnings, pet}``."""
    token, name = _str_param(params, "token"), _str_param(params, "name")
    if not token or not name:
        return _err(rid, 4004, "missing token" if not token else "missing name")
    # Own cancel key: pet.generate may still be releasing `token`. Falls back for old clients.
    cancel_token = _str_param(params, "cancelToken") or token
    from agent.pet import store
    from agent.pet.generate import hatch_pet
    from agent.pet.generate.imagegen import GenerationError
    base = _pet_gen_root() / token / f"draft-{_int_param(params, 'index', 0)}.png"
    if not base.is_file():
        return _err(rid, 4004, "draft expired — generate again")
    try:
        sprite = _pet_pick_provider(params, require_references=True)  # rows always need reference grounding
    except GenerationError as exc:
        return _err(rid, 5031, str(exc))
    _pet_cancel_arm(cancel_token)
    slug = store.unique_slug(name)

    def _on_progress(event: str, detail: str) -> None:
        # Row progress "<state>:<done>:<total>" → "Drawing <state>… (n/total)".
        payload: dict = {"event": event, "detail": detail}
        if event == "row" and detail.count(":") == 2:
            state, done, total = detail.split(":")
            payload = {"event": "row", "state": state, "done": done, "total": total}
        _pet_emit("pet.hatch.progress", payload, "pet.hatch progress")
    try:
        result = hatch_pet(
            base_image=base, slug=slug, display_name=name, description=str(params.get("description") or ""),
            concept=str(params.get("prompt") or name), style=_str_param(params, "style", "auto"), provider=sprite,
            on_progress=_on_progress, is_cancelled=lambda: _pet_is_cancelled(cancel_token))
    except GenerationError as exc:
        return _err(rid, 5031, str(exc))
    finally:
        _pet_cancel_release(cancel_token)
    pet = store.load_pet(result.slug)
    return _ok(rid, {"ok": True, "slug": result.slug, "displayName": result.display_name,
                     "warnings": result.validation.get("warnings", []),
                     "pet": _pet_sprite_payload(pet, scale=_pet_config_scale()) if pet else {}})


# ── billing / subscription ───────────────────────────────────────────
# All fail-open: a logged-out / unreachable portal yields an ``ok`` envelope with a typed
# ``error`` (not a JSON-RPC error) so the TUI maps it to copy. ``billing:manage`` routes
# return error=insufficient_scope on 403, which drives the ``billing.step_up`` device flow.
def _billing_view(name: str, module: str, builder: str, serializer: str, fallback: dict) -> None:
    """Read-only view RPC (no scope required): ``serializer(module.builder())``, ``fallback`` on any error.
    The view module stays a lazy import (startup budget); the serializer is a server global."""
    @method(name)
    def _(rid, params: dict) -> dict:
        try:
            from importlib import import_module
            return _ok(rid, globals()[serializer](getattr(import_module(module), builder)()))
        except Exception:
            return _ok(rid, dict(fallback))


_billing_view("billing.state", "agent.billing_view", "build_billing_state", "_serialize_billing_state",
              {"ok": True, "logged_in": False, "error": "could not load billing state"})
_billing_view("usage.bars", "agent.billing_usage", "build_usage_model", "_serialize_usage_model",  # two-bar $ view
              {"ok": True, "available": False})
_billing_view("subscription.state", "agent.subscription_view", "build_subscription_state",
              "_serialize_subscription_state",
              {"ok": True, "logged_in": False, "error": "could not load subscription state"})


@method("subscription.preview")
def _(rid, params: dict) -> dict:
    """POST /api/billing/subscription/preview → chargeless effect quote. billing:manage."""
    from agent.subscription_view import subscription_change_preview_from_payload
    from hermes_cli.nous_billing import post_subscription_preview
    if not (tier_id := params.get("subscription_type_id")):
        return _billing_invalid(rid, "subscription_type_id is required")
    return _billing_call(rid, lambda: _serialize_subscription_preview(
        subscription_change_preview_from_payload(post_subscription_preview(subscription_type_id=tier_id))))


def _billing_route(name: str, call, *, invalid=None, message: str = "", error: str = "invalid_request",
                   idempotent: bool = False):
    """Portal write route on ``hermes_cli.nous_billing`` (lazy; tests patch its functions): ``invalid(params)``
    → ``_billing_invalid(message, error)``; ``call(nb, params, key)`` performs the request. ``idempotent``
    mints ``idempotency_key`` if absent and echoes it (also on error) so the TUI retries the SAME operation."""
    @method(name)
    def _(rid, params: dict) -> dict:
        import hermes_cli.nous_billing as nb
        if invalid is not None and invalid(params):
            return _billing_invalid(rid, message, error=error)
        key = extra = None
        if idempotent:
            from agent.billing_view import new_idempotency_key
            key = params.get("idempotency_key") or new_idempotency_key()
            extra = {"idempotency_key": key}
        return _billing_call(rid, lambda: call(nb, params, key) | (extra or {}), extra=extra)


# PUT pending-change: schedule a downgrade / same-price change OR a period-end cancellation.
_billing_route("subscription.change", lambda nb, p, _k: _billing_pending_change(nb.put_subscription_pending_change(
    subscription_type_id=p.get("subscription_type_id"), cancel=bool(p.get("cancel")))),
    invalid=lambda p: not p.get("cancel") and not p.get("subscription_type_id"),
    message="subscription_type_id or cancel is required")
# DELETE pending-change: clear a scheduled downgrade / cancellation (re-enables recurring spend).
_billing_route("subscription.resume",
               lambda nb, p, _k: _billing_pending_change(nb.delete_subscription_pending_change()))
# The money route (prorate + charge + flip plan). SCA / decline → status requires_action / payment_failed +
# recovery_url.
_billing_route("subscription.upgrade", lambda nb, p, key: _billing_pick(
    nb.post_subscription_upgrade(subscription_type_id=p.get("subscription_type_id"), idempotency_key=key),
    status="status", target_tier_name="targetTierName", recovery_url="recoveryUrl", reason="reason"),
    invalid=lambda p: not p.get("subscription_type_id"), message="subscription_type_id is required", idempotent=True)
# POST /api/billing/charge → {ok, charge_id, idempotency_key}.
_billing_route("billing.charge", lambda nb, p, key: _billing_pick(
    nb.post_charge(amount_usd=p.get("amount_usd"), idempotency_key=key), charge_id="chargeId"),
    invalid=lambda p: p.get("amount_usd") is None, message="amount_usd is required", idempotent=True)
# GET /api/billing/charge/{id} — a single status read; the caller drives the poll cadence.
_billing_route("billing.charge_status", lambda nb, p, _k: _billing_pick(
    nb.get_charge_status(p.get("charge_id")), status="status", amount_usd="amountUsd", settled_at="settledAt",
    reason="reason"), invalid=lambda p: not p.get("charge_id"), message="charge_id is required",
    error="invalid_charge_id")


def _auto_reload(nb, p: dict, _key) -> dict:
    """PATCH /api/billing/auto-top-up. params: {enabled, threshold, top_up_amount}."""
    nb.patch_auto_top_up(enabled=bool(p.get("enabled")), threshold=p.get("threshold"),
                         top_up_amount=p.get("top_up_amount"))
    return {"ok": True}


_billing_route("billing.auto_reload", _auto_reload, message="threshold and top_up_amount are required",
               invalid=lambda p: p.get("threshold") is None or p.get("top_up_amount") is None)


@method("billing.step_up")
def _(rid, params: dict) -> dict:
    """billing:manage step-up device flow → {ok, granted} (false when the server downscopes). Pooled (blocks
    for minutes); URL/code reach the TUI via ``billing.step_up.verification`` (stdout is the RPC pipe) and the
    browser opens TUI-side, never via the gateway's headless webbrowser.open."""
    sid = params.get("session_id") or ""

    def call():
        from hermes_cli.auth import step_up_nous_billing_scope
        granted = step_up_nous_billing_scope(
            open_browser=False,
            on_verification=lambda url, code: _emit(
                "billing.step_up.verification", sid, {"verification_url": url, "user_code": code}))
        return {"ok": True, "granted": bool(granted)}
    return _billing_call(rid, call, extra={"granted": False})


# ── session status / history / undo / compress / save / close ────────
def _status_row(session: dict, params: dict, key: str) -> dict:
    """Stored row for ``key``: the live session's bound profile db first, else params.profile / launch."""
    if not key:
        return {}
    with _session_db(session) as db:
        if db is not None:
            return _try_get_session(db, key)
        with _profile_db(params) as db2:
            return _try_get_session(db2, key) if db2 else {}


def _try_get_session(db, key: str) -> dict:
    with contextlib.suppress(Exception):
        return db.get_session(key) or {}
    return {}


def _status_dt(value, fallback=None):
    if value:
        with contextlib.suppress(Exception):
            return datetime.fromtimestamp(float(value))
    return fallback or datetime.now()


@_session_method("session.status")
def _(rid, params: dict, session: dict) -> dict:
    from hermes_constants import display_hermes_home
    key = session.get("session_key") or params.get("session_id") or ""
    agent = session.get("agent")
    meta = _status_row(session, params, key)
    created = _status_dt(meta.get("started_at"))
    updated = next((_status_dt(meta[f], created) for f in ("updated_at", "last_updated_at", "last_activity_at")
                    if meta.get(f)), created)
    mirror = _metadata_mirror(session)
    provider = getattr(agent, "provider", None) or mirror.get("provider") or "unknown"
    model = getattr(agent, "model", None) or mirror.get("model") or "(unknown)"
    project = _project_info_for_cwd(_display_session_cwd(session))
    title = (meta.get("title") or "").strip()
    lines = [
        "Hermes TUI Status", "", f"Session ID: {key}", f"Path: {display_hermes_home()}",
        *([f"Project: {project['name']}"] if project else []), *([f"Title: {title}"] if title else []),
        f"Model: {model} ({provider})", f"Created: {created.strftime('%Y-%m-%d %H:%M')}",
        f"Last Activity: {updated.strftime('%Y-%m-%d %H:%M')}",
        f"Tokens: {int(_session_usage_snapshot(session).get('total') or 0):,}",
        f"Agent Running: {'Yes' if session.get('running') else 'No'}"]
    return _ok(rid, {"output": "\n".join(lines)})


@_session_method("session.history")
def _(rid, params: dict, session: dict) -> dict:
    history = list(session.get("history", []))
    if session.get("session_key"):
        with _session_db(session) as db:
            if db is not None:
                # include_row_ids: the durable row id is how clients address a persisted turn (reactions,
                # truncation targets); _history_to_messages forwards it.
                with contextlib.suppress(Exception):
                    # The projection in _history_to_messages only forwards row_id when the row carries a
                    # stamp, so an unstamped read here silently strips the one durable address clients can
                    # use. See #87059.
                    history = db.get_messages_as_conversation(
                        session["session_key"], include_ancestors=True, include_row_ids=True)
    return _ok(rid, {"count": len(history), "messages": _history_to_messages(history)})


@_session_method("session.undo", live=True)
def _(rid, params: dict, session: dict) -> dict:
    # Under a running turn the post-run write would clobber the undo — /interrupt first.
    busy = _err(rid, 4009, "session busy — /interrupt the current turn before /undo")
    if session.get("running"):
        return busy
    removed = 0
    with session["history_lock"]:
        if session.get("running"):
            return busy
        history = _history_without_ephemeral_scaffolding(session.get("history", []))
        # Truncate from the last *real* user turn (not a timeline marker / compaction handoff).
        from agent.context_compressor import user_originated_turn_view
        if user_turns := sum(1 for message in history if user_originated_turn_view(message) is not None):
            try:
                removed = _rewind_active_session_history(session, user_turns - 1)[2]
            except Exception as exc:
                return _err(rid, 5008, f"undo: {exc}")
    return _ok(rid, {"removed": removed})


def _compute_host_ack_error(rid, ack: dict, code: int, default: str):
    """``_err`` for a ``control.error``/``error`` ack, else None."""
    if ack.get("type") in {"control.error", "error"}:
        return _err(rid, code, str(ack.get("message") or default))
    return None


def _save_via_compute_host(rid, params: dict) -> dict:
    """``session.save`` for a turn-isolated session: the host owns the transcript file."""
    try:
        ack = _send_compute_host_control(str(params.get("session_id") or ""), route_name="session.save", wait=True)
    except Exception as exc:
        return _err(rid, 5011, f"compute-host session save failed: {exc}")
    if (resp := _compute_host_ack_error(rid, ack, 5011, "compute-host session save failed")) is not None:
        return resp
    if not isinstance(result := ack.get("result"), dict):
        return _err(rid, 5011, "compute-host session save returned an invalid response")
    return _ok(rid, result)


def _compress_via_compute_host(rid, params: dict, session: dict) -> dict:
    """``session.compress`` for a turn-isolated session: forward ``/compress`` to the host."""
    sid = str(params.get("session_id") or "")
    focus_topic = _str_param(params, "focus_topic")

    def _on_late_ack(late: dict, _sid=sid) -> None:
        _adopt_late_compute_host_compress_ack(_sid, session, late, route_name="session.compress")
    try:
        ack = _send_compute_host_control(
            sid, route_name="session.compress", command="/compress" + (f" {focus_topic}" if focus_topic else ""),
            # compression.context_total_ceiling_seconds: the host legitimately runs that long.
            wait=True, timeout=_compute_host_compress_wait_seconds(), on_late_ack=_on_late_ack)
    except queue.Empty:
        # Waiter gave up, host still compressing; the late-ack handler adopts the rotated session when it
        # lands. Not an error (a 5019 here reported timeouts that later succeeded).
        return _ok(rid, {"status": "pending", "turn_isolation": True,
                         "message": ("compression still running in the background; "
                                     "the transcript will refresh when it finishes")})
    except Exception as exc:
        return _err(rid, 5019, f"compute-host compress failed: {exc}")
    if (resp := _compute_host_ack_error(rid, ack, 4009, "compute-host compress failed")) is not None:
        return resp
    _apply_compute_host_metadata_mirror(session, ack)
    if isinstance(host_result := ack.get("result"), dict):
        # Host-owned result verbatim (carries `status: aborted` / `summary.aborted`).
        return _ok(rid, {**host_result, "turn_isolation": True})
    host_info = ack.get("session_info") if isinstance(ack.get("session_info"), dict) else {}
    return _ok(rid, {
        "status": "compressed", "turn_isolation": True,
        # `messages` goes top-level for the transcript replacement; don't duplicate it in the ack.
        "host_ack": {key: value for key, value in ack.items() if key != "messages"}, "info": host_info,
        "messages": _history_to_messages(ack.get("messages")) if isinstance(ack.get("messages"), list) else [],
        "usage": host_info.get("usage") if isinstance(host_info.get("usage"), dict) else {}})


def _compress_live(rid, sid: str, session: dict, focus_topic: str) -> dict:
    """In-process ``session.compress``: status pinned "compressing", then the before/after summary + messages."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import summarize_manual_compression
    from agent.model_metadata import estimate_request_tokens_rough
    with session["history_lock"]:
        before_messages = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
    before_count = len(before_messages)
    _agent = session["agent"]
    _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
    _tools = getattr(_agent, "tools", None) or None

    def _tokens(msgs) -> int:
        # Re-reads prompt + tools each call: _compress_context may have rebuilt the system prompt.
        sys_prompt = getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
        tools = getattr(_agent, "tools", None) or _tools
        return estimate_request_tokens_rough(msgs, system_prompt=sys_prompt, tools=tools) if msgs else 0
    before_tokens = _tokens(before_messages)
    if before_count >= 4:
        focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ""
        _status_update(sid, "compressing",
                       f"⠋ compressing {before_count} messages (~{before_tokens:,} tok){focus_suffix}…")
    try:
        removed, usage = _compress_session_history(
            session, focus_topic, approx_tokens=before_tokens, before_messages=before_messages,
            history_version=history_version)
        with session["history_lock"]:
            messages = list(session.get("history", []))
        after_tokens = _tokens(messages)
        agent = session["agent"]
        _sync_session_key_after_compress(sid, session)
        summary = summarize_manual_compression(before_messages, messages, before_tokens, after_tokens,
                                               compression_state=getattr(agent, "context_compressor", None))
        info = _session_info(agent, session)
        _emit("session.info", sid, info)
        finalize_context_engine_compression_notification(agent, committed=True)
        return _ok(rid, {
            "status": "aborted" if summary["aborted"] else "compressed", "removed": removed,
            "before_messages": before_count, "after_messages": len(messages),
            "before_tokens": before_tokens, "after_tokens": after_tokens, "summary": summary,
            "usage": usage, "info": info, "messages": _history_to_messages(messages)})
    finally:
        # Always clear the pinned compressing status (success, no-op, or raise).
        _status_update(sid, "ready")


@method("session.compress")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if _session_uses_compute_host(session):
        return _compress_via_compute_host(rid, params, session)
    session, err = _sess(params, rid)
    if err:
        return err
    if session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before /compress")
    sid = params.get("session_id", "")
    try:
        return _compress_live(rid, sid, session, _str_param(params, "focus_topic"))
    except CompressionLockHeld as e:
        _status_update(sid, "ready")
        from agent.manual_compression_feedback import describe_compression_lock_skip
        return _ok(rid, {"compressed": False, "lock_held": True, "message": describe_compression_lock_skip(e.holder)})
    except Exception as e:
        from agent.conversation_compression import finalize_context_engine_compression_notification
        finalize_context_engine_compression_notification(session["agent"], committed=False)
        return _err(rid, 5005, str(e))


@_session_method("session.save", live=True)
def _(rid, params: dict, session: dict) -> dict:
    if _session_uses_compute_host(session):
        return _save_via_compute_host(rid, params)
    agent = session["agent"]
    # Classic CLI /save: under the profile home, with the system prompt (dashboard parity).
    saved_dir = get_hermes_home() / "sessions" / "saved"
    try:
        saved_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return _err(rid, 5011, f"failed to create save directory {saved_dir}: {e}")
    path = saved_dir / f"hermes_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with session["history_lock"]:
        messages = list(session.get("history", []))
    # Prefer the agent's session_start (classic CLI export); else the gateway created_at.
    started = getattr(agent, "session_start", None)
    if not isinstance(started, datetime):
        created_at = session.get("created_at")
        started = datetime.fromtimestamp(created_at) if isinstance(created_at, (int, float)) else None
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": getattr(agent, "model", ""),
                       "session_id": getattr(agent, "session_id", None) or session.get("session_key") or "",
                       "session_start": started.isoformat() if started else "",
                       "system_prompt": getattr(agent, "_cached_system_prompt", "") or "",
                       "messages": messages}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return _err(rid, 5011, str(e))
    return _ok(rid, {"file": str(path)})


@method("session.close")
def _(rid, params: dict) -> dict:
    with _session_resume_lock:  # lock only the ownership claim; finalization must not block resumes
        session = _pop_session_by_id(params.get("session_id", ""))
    return _ok(rid, {"closed": _teardown_popped_session(session, end_reason="tui_close")})


# ── session.branch ───────────────────────────────────────────────────
def _visible_branch_history(messages) -> list:
    """user/assistant rows with visible text, as FULL copies (reasoning + timeline-marker tags survive)."""
    return [dict(message) for message in messages or []
            if isinstance(message, dict) and message.get("role") in {"user", "assistant"}
            and _coerce_message_text(message.get("content")).strip()]


def _build_branch_agent(session: dict, new_sid: str, new_key: str, history: list, source: str):
    """Build + register the branched agent in the parent's profile; the DEDICATED db handle is ours until
    ``_transfer_db_to_agent`` (released here on failure)."""
    parent_home = session.get("profile_home")
    branch_db, branch_owns_db = _profile_session_db(parent_home) if parent_home else (None, False)
    try:
        with _profile_build_scope(parent_home):
            agent = _make_agent_in_context(new_sid, new_key, session_db=branch_db, platform_override=source,
                                           context_cwd_is_launch_artifact=_context_cwd_is_launch_artifact(session))
            _init_session(new_sid, new_key, agent, list(history), cols=session.get("cols", 80),
                          cwd=_session_cwd(session), session_db=branch_db, source=source, profile_home=parent_home,
                          explicit_cwd=bool(session.get("explicit_cwd")))
            _transfer_db_to_agent(agent, branch_db)
            branch_owns_db = False
        if new_sid in _sessions:
            _sessions[new_sid]["active_session_lease"] = None  # claimed lazily on the first turn
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


def _branch_source_history(db, session: dict, old_key: str) -> list:
    """Rows a branch copies: the persisted DISPLAY projection reconciled with live memory (live history is
    the MODEL projection — post-compaction summary + tail — the child would lose every archived turn)."""
    with session["history_lock"]:
        in_memory_history = [
            dict(msg) for msg in list(session.get("display_history_prefix") or []) + list(session.get("history", []))
            if isinstance(msg, dict)]
    history = None
    if callable(get_resume_conversations := getattr(db, "get_resume_conversations", None)):
        try:
            _, display_history = get_resume_conversations(old_key)
            history = _visible_branch_history(_reconcile_display_with_live(display_history, in_memory_history))
        except Exception:
            logger.debug("branch display projection read failed", exc_info=True)
    return history or _visible_branch_history(in_memory_history)


@_session_method("session.branch", live=True)
def _(rid, params: dict, session: dict) -> dict:
    # Write into the parent's profile-scoped state.db; the launch handle would orphan rows.
    with _session_db(session) as db:
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        old_key = session["session_key"]
        history = _branch_source_history(db, session, old_key)
        if not history:
            return _err(rid, 4008, "nothing to branch — send a message first")
        if isinstance(count := params.get("count"), int) and count > 0:
            history = history[:count]
        new_key, new_sid, source = _new_session_key(), uuid.uuid4().hex[:8], _session_source(session)
        try:
            title = params.get("name", "") or _branch_title(db, old_key)
            home = session.get("profile_home")
            _persist_branch(db, new_key, old_key, title, history, source=source, cwd=_session_cwd(session),
                            # The branch stays on its parent's profile. Derive the label from
                            # the bound store handle (db is the parent's profile-scoped
                            # state.db, default-root-pinned for the launch profile) so the
                            # label is the SAME fact as the store and cannot diverge from it
                            # on a reused compute-host executor under a residual foreign home
                            # (the fifth profile seam). Explicit stamp (not just the
                            # parent-backfill) so it holds even when the parent row predates
                            # the profile_name column; launch-profile branches are stamped
                            # 'default' too, since NULL rows drop out of profile-keyed sidebar
                            # matching and deep-link resolution (#99222).
                            profile_name=db._own_profile_name(),
                            copy_fields=_BRANCH_COPY_FIELDS)
        except Exception as e:
            return _err(rid, 5008, f"branch failed: {e}")
    try:
        agent = _build_branch_agent(session, new_sid, new_key, history, source)
    except Exception as e:
        return _err(rid, 5000, f"agent init failed on branch: {e}")
    return _ok(rid, {"session_id": new_sid, "stored_session_id": new_key, "title": title, "parent": old_key,
                     "message_count": len(history), "messages": _history_to_messages(history),
                     "info": _session_info(agent, _sessions.get(new_sid))})


# ── interrupt / steer / redirect ─────────────────────────────────────
@method("session.interrupt")
def _(rid, params: dict) -> dict:
    _tts_stream_stop()  # keypress barge-in also silences streaming TTS (voice is process-global)
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    if expected := _str_param(params, "expected_hosted_task_id"):
        with session["history_lock"]:
            task = session.get("_hosted_room_task")
            if not (session.get("running") and isinstance(task, dict) and task.get("task_id") == expected):
                return _ok(rid, {"status": "not_interrupted", "interrupted": False})
    sid = str(params.get("session_id") or "")
    if _session_uses_compute_host(session):
        try:
            _interrupt_session_turn(sid, session, request_id=f"interrupt-{rid}")
        except Exception as exc:
            return _err(rid, 5019, f"compute-host interrupt failed: {exc}")
        return _ok(rid, {"status": "interrupted", "turn_isolation": True})
    session, err = _sess(params, rid)
    if err:
        return err
    _interrupt_session_turn(sid, session)
    # Retire the crash-recovery marker NOW: until the run thread's finally, a backend exit looks like a crash
    # and session.resume auto-continues the turn the user just stopped (the extra key covers compression
    # rotating session_key mid-turn).
    with session["history_lock"]:
        active_marker_key = str(session.pop("_active_turn_marker_key", "") or "")
    _retire_turn_marker(session, active_marker_key)
    return _ok(rid, {"status": "interrupted"})


def _apply_correction(rid, session: dict, verb: str, text: str, accepted_status: str) -> dict:
    """``agent.<verb>(text)``; on acceptance record it on the live turn (mid-turn resume rebuilds the bubble)
    and purge queued self-copies so post-turn drain cannot re-fire the old prompt."""
    try:
        accepted = getattr(session["agent"], verb)(text)
    except Exception as exc:
        return _err(rid, 5000, f"{verb} failed: {exc}")
    if accepted:
        with session["history_lock"]:
            _record_inflight_correction(session, text)
            # #84417: steer does not cancel the live original, but a server queue self-copy of that original
            # must still not re-fire after settle (same class as redirect).
            # #84417: purge server-queue self-duplicates of the live original so post-turn drain cannot
            # restart the pre-correction prompt.
            _drop_queued_duplicates_of_inflight_user(session)
            session["last_active"] = time.time()
    return _ok(rid, {"status": accepted_status if accepted else "rejected", "text": text})


def _correction_method(name: str, verb: str, accepted_status: str, supported, unsupported: str):
    """steer/redirect RPC: ``params.text`` (4002, checked before the session) into a live session;
    ``supported(agent)`` gates 4010."""
    @method(name)
    def _(rid, params: dict) -> dict:
        if not (text := (params.get("text") or "").strip()):
            return _err(rid, 4002, "text is required")
        session, err = _sess_nowait(params, rid)
        if err:
            return err
        agent = session.get("agent")
        # Redirect during the turn-build window (running=True, agent None): queue for the next turn instead of
        # a misleading 4010 the client swallows into a lost follow-up.
        if verb == "redirect" and agent is None and session.get("running"):
            _enqueue_prompt(session, text, current_transport() or _stdio_transport)
            session["last_active"] = time.time()
            return _ok(rid, {"status": "queued", "text": text})
        if not supported(agent):
            return _err(rid, 4010, unsupported)
        return _apply_correction(rid, session, verb, text, accepted_status)


# Inject text into the next tool result without interrupting (AIAgent.steer(): no new user turn, no role
# alternation violation).
_correction_method("session.steer", "steer", "queued", lambda agent: hasattr(agent, "steer"),
                   "agent does not support steer")
# Redirect the active model turn while preserving valid work/context.
_correction_method("session.redirect", "redirect", "redirected",
                   lambda agent: getattr(agent, "_supports_active_turn_redirect", False) is True
                   and hasattr(agent, "redirect"), "agent does not support active-turn redirect")


# ── delegation / spawn trees ─────────────────────────────────────────
@method("delegation.status")
def _(rid, params: dict) -> dict:
    from tools import delegate_tool as dt
    return _ok(rid, {"active": dt.list_active_subagents(), "paused": dt.is_spawn_paused(),
                     "max_spawn_depth": dt._get_max_spawn_depth(),
                     "max_concurrent_children": dt._get_max_concurrent_children()})


@method("delegation.pause")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import set_spawn_paused
    return _ok(rid, {"paused": set_spawn_paused(bool(params.get("paused", True)))})


@method("subagent.interrupt")
def _(rid, params: dict) -> dict:
    from tools.delegate_tool import interrupt_subagent
    if not (subagent_id := _str_param(params, "subagent_id")):
        return _err(rid, 4000, "subagent_id required")
    return _ok(rid, {"found": interrupt_subagent(subagent_id), "subagent_id": subagent_id})


@method("subagent.steer")
def _(rid, params: dict) -> dict:
    """Queue steering text into a live delegated child (the in-flight tool call is never cut). "queued"
    is not "delivered": a child past its final tool batch surfaces ``missed_steer`` on the parent entry."""
    from tools.delegate_tool import steer_subagent
    if not (subagent_id := _str_param(params, "subagent_id")):
        return _err(rid, 4000, "subagent_id required")
    if not (text := (params.get("text") or "").strip()):
        return _err(rid, 4002, "text is required")
    if (err := _sess_nowait(params, rid)[1]) is not None:
        return err
    owner_id = _str_param(params, "session_id")
    transport, owner = _current_session_steer_authority(owner_id)
    queued = transport is not None and owner is not None and steer_subagent(
        subagent_id, text, owner_session_id=owner_id, owner_transport=transport, owner_session_record=owner)
    return _ok(rid, {"status": "queued" if queued else "rejected", "subagent_id": subagent_id, "text": text})


@method("spawn_tree.save")
def _(rid, params: dict) -> dict:
    session_id = _str_param(params, "session_id")
    subagents = params.get("subagents") or []
    if not isinstance(subagents, list) or not subagents:
        return _err(rid, 4000, "subagents list required")
    started_at, label = params.get("started_at"), str(params.get("label") or "")
    finished_at = float(params.get("finished_at") or time.time())
    d = _spawn_tree_session_dir(session_id or "default")
    path = d / f"{datetime.utcfromtimestamp(finished_at).strftime('%Y%m%dT%H%M%S')}.json"
    meta = {"session_id": session_id, "started_at": float(started_at) if started_at else None,
            "finished_at": finished_at, "label": label}
    try:
        path.write_text(json.dumps({**meta, "subagents": subagents}, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return _err(rid, 5000, f"spawn_tree.save failed: {exc}")
    _append_spawn_tree_index(d, {"path": str(path), **meta, "count": len(subagents)})
    return _ok(rid, {"path": str(path), "session_id": session_id})


def _legacy_spawn_tree_entry(p, session_dir_name: str) -> dict | None:
    """Index-shaped entry for a pre-index snapshot file (None when unreadable)."""
    try:
        stat = p.stat()
    except OSError:
        return None
    raw = {}
    with contextlib.suppress(Exception):
        raw = json.loads(p.read_text(encoding="utf-8"))
    subagents = raw.get("subagents") or []
    return {"path": str(p), "session_id": raw.get("session_id") or session_dir_name,
            "finished_at": raw.get("finished_at") or stat.st_mtime, "started_at": raw.get("started_at"),
            "label": raw.get("label") or "", "count": len(subagents) if isinstance(subagents, list) else 0}


@method("spawn_tree.list")
def _(rid, params: dict) -> dict:
    session_id = _str_param(params, "session_id")
    if bool(params.get("cross_session")):
        roots = [p for p in _spawn_trees_root().iterdir() if p.is_dir()]
    else:
        roots = [_spawn_tree_session_dir(session_id or "default")]
    entries: list[dict] = []
    for d in roots:
        if indexed := _read_spawn_tree_index(d):
            # Skip index entries whose snapshot file was manually deleted.
            entries.extend(e for e in indexed if (p := e.get("path")) and Path(p).exists())
        else:  # Legacy (pre-index) sessions: full scan, once per session until the next save.
            entries.extend(
                entry for p in d.glob("*.json")
                if p.name != _SPAWN_TREE_INDEX and (entry := _legacy_spawn_tree_entry(p, d.name)) is not None)
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return _ok(rid, {"entries": entries[:int(params.get("limit") or 50)]})


@method("spawn_tree.load")
def _(rid, params: dict) -> dict:
    if not (raw_path := _str_param(params, "path")):
        return _err(rid, 4000, "path required")
    try:
        (resolved := Path(raw_path).resolve()).relative_to(_spawn_trees_root().resolve())
    except (ValueError, OSError) as exc:
        return _err(rid, 4030, f"path outside spawn-trees root: {exc}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _err(rid, 5000, f"spawn_tree.load failed: {exc}")
    return _ok(rid, payload)


# ── terminal / event replay ──────────────────────────────────────────
@_session_method("terminal.resize")
def _(rid, params: dict, session: dict) -> dict:
    session["cols"] = cols = int(params.get("cols", 80))
    return _ok(rid, {"cols": cols})


@method("session.events.since")
def _(rid, params: dict) -> dict:
    """Replay events after ``last_seen`` (WS reconnect); ``truncated`` past the ring window → client refetches."""
    sid = str(params.get("session_id") or "")
    try:
        last_seen = int(params.get("last_seen", 0))
    except (TypeError, ValueError):
        return _err(rid, -32602, "invalid params: last_seen must be an integer")
    from tui_gateway import event_replay as er
    frames = er.events_since(sid, last_seen)
    # ``epoch``: in-process seq — clients reset watermarks when this differs from gateway.ready's.
    return _ok(rid, {"events": frames, "latest_seq": er.latest_seq(sid), "truncated": er.is_truncated(sid, last_seen),
                     "count": len(frames), "epoch": er.replay_epoch()})


@method("session.events.stats")
def _(rid, params: dict) -> dict:
    """Replay-buffer telemetry (ops/debug)."""
    from tui_gateway import event_replay
    return _ok(rid, event_replay.replay_stats())


def register(server) -> None:
    """Publish this module's helpers onto ``server`` (rebound to its globals) and install handlers."""
    bind_module(globals(), server, skip=("_",))
