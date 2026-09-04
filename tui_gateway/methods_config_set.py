"""``config.set`` — one JSON-RPC method, dispatched on ``key`` through ``_CONFIG_SETTERS``. Bodies
are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare. Each
``_set_*`` takes ``(rid, params, key, value, session)`` and returns the JSON-RPC envelope.
Keys match exactly except ``details_mode.<section>`` (prefix) and ``_DISPLAY_TOGGLE_KEYS``.
"""

import os

from hermes_constants import INDICATOR_STYLES

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── shared helpers

def _write_display_sections(*, sections=None, drop_sections=(), **display_fields) -> None:
    """Persist ``display.<field>`` + ``display.sections`` edits via the raw (uncached) write-back."""
    cfg = _load_cfg_raw()
    display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
    cur = display.get("sections") if isinstance(display.get("sections"), dict) else {}
    display.update(display_fields)
    cur.update(sections or {})
    for name in drop_sections:
        cur.pop(name, None)
    display["sections"] = cur
    cfg["display"] = display
    _save_cfg(cfg)


def _emit_session_info(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is not None:
        _emit("session.info", sid, _session_info(agent, session))


def _emit_all_session_info() -> None:
    for sid, sess in list(_sessions.items()):
        _emit_session_info(sid, sess)


def _word(value) -> str:
    return str(value or "").strip().lower()


def _raw_word(value) -> str:
    """Like ``_word`` but only None is blank: falsy non-strings (0, False, []) keep their text."""
    return ("" if value is None else str(value)).strip().lower()


def _kv(rid, key, value, **extra):
    return _ok(rid, {"key": key, "value": value, **extra})


def _cfgset_await_agent(session, rid):
    """Wait for an in-progress agent build; the error envelope if it failed, else None."""
    init_err = _wait_agent(session, rid)
    if init_err:
        return init_err
    return _err(rid, 5032, "agent initialization failed") if session.get("agent") is None else None


def _cfgset_model_ok(rid, key, value, warning="", confirm_message="", scope="session", **extra):
    """Model-switch envelope; ``confirm_required`` follows ``confirm_message`` (canonical; ``warning``
    is its legacy alias on the deferred path)."""
    return _kv(rid, key, value, warning=warning, confirm_required=bool(confirm_message),
               confirm_message=confirm_message, scope=scope, **extra)


def _stash_pending_model_switch(rid, key, value, session, confirmed, parsed):
    """No live swap while a turn streams (agent.switch_model() mutates fields the worker thread
    reads every iteration): stash the pick for the NEXT turn start. Selection guards run HERE (the
    only moment a confirm round-trip is possible; an unconfirmed stashed pick is dropped at turn
    start) — on a warning nothing is stashed."""
    try:
        pending_model = parsed.model_input
    except Exception:
        pending_model = str(value)
    pending_provider = (getattr(parsed, "explicit_provider", "") or "").strip()
    if not confirmed:
        pending_warning = _pending_switch_selection_warning(pending_model, pending_provider)
        if pending_warning is not None:
            return _cfgset_model_ok(rid, key, pending_model, pending_warning, pending_warning, deferred=False)
    # display_*: _session_info shows the user's pick while pending, not the live old model.
    session["pending_model_switch"] = {
        "raw": value, "confirm_expensive_model": confirmed,
        "display_model": pending_model, "display_provider": pending_provider}
    return _cfgset_model_ok(rid, key, pending_model, deferred=True)


def _cfgset_guarded(fn):
    """Setter whose uncaught exception becomes ``_err(rid, 5001, str(e))``."""
    def setter(rid, params, key, value, session):
        try:
            return fn(rid, params, key, value, session)
        except Exception as e:
            return _err(rid, 5001, str(e))
    return setter


# ── per-key handlers

@_cfgset_guarded
def _set_model(rid, params, key, value, session):
    """Live/deferred model switch; see _apply_model_switch and _apply_pending_model_switch."""
    if not value:
        return _err(rid, 4002, "model value required")
    confirmed = bool(params.get("confirm_expensive_model", False))
    if session:
        from hermes_cli.model_switch import parse_model_switch_args
        sid = params.get("session_id", "")
        parsed_flags = parse_model_switch_args(value)
        if session.get("running"):
            return _stash_pending_model_switch(rid, key, value, session, confirmed, parsed_flags)
        explicit_provider = parsed_flags.explicit_provider
        failed_agent_init = session.get("agent") is None and session.get("agent_error") is not None
        failed_ready = session.get("agent_ready") if failed_agent_init else None
        if failed_agent_init:
            if failed_ready is None:
                return _err(rid, 5032, session.get("agent_error") or "agent initialization failed")
            if not failed_ready.wait(timeout=30.0):
                return _err(rid, 5032, "agent initialization timed out")
        failed_agent_init = (
            failed_agent_init and session.get("agent") is None and session.get("agent_error") is not None
            and session.get("agent_ready") is failed_ready and failed_ready.is_set())
        if session.get("agent") is None and not explicit_provider.strip() and not failed_agent_init:
            _start_agent_build(sid, session)
            if init_err := _cfgset_await_agent(session, rid):
                return init_err
        with _session_profile_runtime_scope(session):
            result = _apply_model_switch(sid, session, value, confirm_expensive_model=confirmed,
                                         parsed_flags=parsed_flags)
        if failed_agent_init and not result.get("confirm_required"):
            _restart_completed_failed_agent_build(sid, session, failed_ready)
            if init_err := _cfgset_await_agent(session, rid):
                return init_err
            with _session_profile_runtime_scope(session):
                _persist_live_session_runtime(session)
    else:
        result = _apply_model_switch("", {"agent": None}, value, confirm_expensive_model=confirmed)
    return _kv(rid, key, result["value"], warning=result["warning"],
               confirm_required=result.get("confirm_required", False),
               confirm_message=result.get("confirm_message", ""), scope=result.get("scope", "session"))


_FAST_WORDS = {"fast": "fast", "on": "fast", "normal": "normal", "off": "normal",
               "auto": "auto", "cold": "cold"}


def _set_fast(rid, params, key, value, session):
    raw = _word(value)
    agent = session.get("agent") if session else None
    if agent is not None:
        current_tier = getattr(agent, "service_tier", None)
    elif session is not None and session.get("create_service_tier_override") is not None:
        current_tier = session["create_service_tier_override"] or None  # pre-build pin beats global
    else:
        current_tier = _load_service_tier()
    if raw == "status":
        return _kv(rid, key, {"priority": "fast", None: "normal"}.get(current_tier, current_tier))
    nv = _FAST_WORDS.get(raw, ("normal" if current_tier == "priority" else "fast") if raw in {"", "toggle"} else None)
    if nv is None:
        return _err(rid, 4002, f"unknown fast mode: {value}")
    overrides = None
    if nv == "fast":
        from hermes_cli.models import resolve_fast_mode_overrides
        if agent is not None:
            target_model = getattr(agent, "model", None)
        else:  # a pre-build session may carry a picked model (desktop draft): validate against THAT
            session_override = (session or {}).get("model_override") or {}
            target_model = (isinstance(session_override, dict) and session_override.get("model")) or _resolve_model()
        if not target_model:
            return _err(rid, 4002, "fast mode is not available without a selected model")
        overrides = resolve_fast_mode_overrides(target_model, provider=getattr(agent, "provider", None),
                                                base_url=getattr(agent, "base_url", None))
        if overrides is None:
            return _err(rid, 4002, "fast mode is not available for this model")
    if session is not None:
        # Session-scoped like `reasoning` (global = `--global` / Settings → Model): writing config.yaml
        # here flipped fast mode for every surface. The create override survives rebuilds; "" pins normal.
        session["create_service_tier_override"] = {"fast": "priority", "normal": ""}.get(nv, nv)
    else:
        _write_config_key("agent.service_tier", nv)
    if agent is not None:
        agent.service_tier = {"fast": "priority", "normal": None}.get(nv, nv)
        current_overrides = {k: v for k, v in (getattr(agent, "request_overrides", {}) or {}).items()
                             if k not in ("service_tier", "speed")}
        agent.request_overrides = {**current_overrides, **(overrides or {})}
        _persist_live_session_runtime(session)
        _emit_session_info(params.get("session_id", ""), session)
    return _kv(rid, key, nv)


def _set_busy(rid, params, key, value, session):
    if _word(value) in {"", "status"}:
        return _kv(rid, key, _load_busy_input_mode())
    return _set_word(rid, params, key, value, session)


def _set_verbose(rid, params, key, value, session):
    cycle = ["off", "new", "all", "verbose"]
    if value and value != "cycle":
        nv = str(value).strip().lower()
        if nv not in cycle:
            return _err(rid, 4002, f"unknown verbose mode: {value}")
    else:
        cur = session.get("tool_progress_mode", _load_tool_progress_mode()) if session else _load_tool_progress_mode()
        nv = cycle[((cycle.index(cur) if cur in cycle else 2) + 1) % len(cycle)]
    _write_config_key("display.tool_progress", nv)
    if session:
        session["tool_progress_mode"] = nv
        if session.get("agent") is not None:
            session["agent"].verbose_logging = nv == "verbose"
    return _kv(rid, key, nv)


def _set_focus(rid, params, key, value, session):
    # /focus: enabling stashes the configured tool_progress mode and pins it "off"; disabling restores.
    from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode, resolve_focus_arg
    d_f = _display_cfg()
    cur_focus = bool(d_f.get("focus_view", False))
    action, target = resolve_focus_arg(str(value or ""), cur_focus)
    if action == "usage":
        return _err(rid, 4002, f"unknown focus value: {value} (use on|off|status)")
    if action == "status" or target is None:
        return _kv(rid, key, "on" if cur_focus else "off", tool_progress=_load_tool_progress_mode())
    if target:
        saved = (cur_focus and d_f.get("focus_saved_tool_progress")) or _load_tool_progress_mode()
        _write_config_key("display.focus_saved_tool_progress", normalize_tool_progress_mode(saved))
        effective = FOCUS_TOOL_PROGRESS_MODE
    else:
        effective = normalize_tool_progress_mode(d_f.get("focus_saved_tool_progress") or "all")
    _write_config_key("display.tool_progress", effective)
    _write_config_key("display.focus_view", bool(target))
    if session:
        session["focus_view"] = bool(target)
        session["tool_progress_mode"] = effective
        if session.get("agent") is not None:
            with contextlib.suppress(Exception):
                session["agent"].tool_progress_mode = effective
    return _kv(rid, key, "on" if target else "off", tool_progress=effective)


def _set_approval_mode(rid, params, key, value, session):
    return _set_word(rid, params, "approvals.mode", value, session)  # legacy alias reports the real key


@_cfgset_guarded
def _set_yolo(rid, params, key, value, session):
    # scope="session" (default; Shift+Tab) toggles ONLY this session's flag; scope="global"
    # (Shift+click the zap) flips persistent approvals.mode between "off" and "manual".
    scope = _word(params.get("scope") or "session")
    from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled
    raw = _word(value)
    if scope == "global":
        from tools.approval_context import _normalize_approval_mode
        appr = _load_cfg().get("approvals")
        appr = appr if isinstance(appr, dict) else {}
        enable = _BOOL_WORDS.get(raw, _normalize_approval_mode(appr.get("mode", "manual")) != "off")
        _write_config_key("approvals.mode", "off" if enable else "manual")  # binary: no "smart" restore
        _emit_all_session_info()  # reflect the flip in every live indicator
    elif session:
        skey = session["session_key"]
        enable = _BOOL_WORDS.get(raw, not is_session_yolo_enabled(skey))
        (enable_session_yolo if enable else disable_session_yolo)(skey)
        _emit_session_info(params.get("session_id", ""), session)
    else:
        enable = _BOOL_WORDS.get(raw, not is_truthy_value(os.environ.get("HERMES_YOLO_MODE")))
        if enable:
            os.environ["HERMES_YOLO_MODE"] = "1"
        else:
            os.environ.pop("HERMES_YOLO_MODE", None)
    return _kv(rid, key, "1" if enable else "0", scope=scope if scope == "global" else "session")


# /reasoning display words: (accepted inputs, reported value, display field, sections.thinking,
# session show_reasoning or None). full/clamp mirror the CLI's reasoning_full toggle.
_REASONING_DISPLAY_WORDS = (
    ({"show", "on"}, "show", {"show_reasoning": True}, "expanded", True),
    ({"hide", "off"}, "hide", {"show_reasoning": False}, "hidden", False),
    ({"full", "all"}, "full", {"reasoning_full": True}, "expanded", None),
    ({"clamp", "collapse", "short"}, "clamp", {"reasoning_full": False}, "collapsed", None))


@_cfgset_guarded
def _set_reasoning(rid, params, key, value, session):
    from hermes_constants import parse_reasoning_effort
    arg = _word(value)
    scope = _word(params.get("scope"))
    for words, reported, fields, thinking, show in _REASONING_DISPLAY_WORDS:
        if arg in words:
            _write_display_sections(sections={"thinking": thinking}, **fields)
            if show is not None and session:
                session["show_reasoning"] = show
            return _kv(rid, key, reported)
    parsed = parse_reasoning_effort(arg)
    if parsed is None:
        return _err(rid, 4002, f"unknown reasoning value: {value}")
    if scope == "global" or session is None:
        _write_config_key("agent.reasoning_effort", arg)
        if session is not None:
            # /new is a full conversation boundary: session-scoped runtime overrides (/model, /reasoning,
            # /fast) do NOT carry forward — the fresh agent re-derives model/provider, reasoning, and
            # service tier from config.yaml (#48055, #23131). Session pins are cleared below so a rebuild
            # can't resurrect them. (Global process state is still never touched — see the
            # cross-session-contamination note in _apply_model_switch.)
            session.pop("create_reasoning_override", None)
    else:  # session-scoped like the gateway's `/reasoning <level>`; a menu pick must not rewrite the global
        session["create_reasoning_override"] = parsed
    if session and session.get("agent") is not None:
        session["agent"].reasoning_config = parsed
        _persist_live_session_runtime(session)
        _emit_session_info(params.get("session_id", ""), session)
    return _kv(rid, key, arg)


def _word_setters() -> dict:
    """key -> (normaliser, accepted words, error template, apply(word)); the reported value is the
    accepted word. Built per call: the specs reference server.py globals (rebound at install)."""
    return {
        "busy": (_word, {"queue", "steer", "interrupt"}, "unknown busy mode: {value}",
                 lambda w: _write_config_key("display.busy_input_mode", w)),
        "approvals.mode": (_word, _APPROVAL_MODES, "unknown approval mode: {value}; pick one of manual|smart|off",
                           lambda w: (_write_config_key("approvals.mode", w), _emit_all_session_info())),
        "details_mode": (_word, _DETAIL_MODES, "unknown details_mode: {value}", lambda w: _write_display_sections(
            sections={section: w for section in _DETAIL_SECTION_NAMES}, details_mode=w)),
        # thinking_mode also keeps details_mode aligned (compat bridge).
        "thinking_mode": (_word, {"collapsed", "truncated", "full"}, "unknown thinking_mode: {value}", lambda w: (
            _write_config_key("display.thinking_mode", w),
            _write_config_key("display.details_mode", "expanded" if w == "full" else "collapsed"))),
        # 'light'/'dark' pin beats background auto-detection (xterm.js hosts misreport OSC 11).
        "theme": (_word, {"auto", "light", "dark"}, "unknown theme value: {value} (use auto|light|dark)",
                  lambda w: _write_config_key("display.tui_theme", w)),
        # _raw_word: 0/False/[] keep their text so the error names what was sent.
        "indicator": (_raw_word, INDICATOR_STYLES, "unknown indicator: {raw!r}; pick one of " + "|".join(INDICATOR_STYLES),
                      lambda w: _write_config_key("display.tui_status_indicator", w))}


def _set_word(rid, params, key, value, session):
    norm, allowed, err, apply = _word_setters()[key]
    raw = norm(value)
    if raw not in allowed:
        return _err(rid, 4002, err.format(value=value, raw=raw))
    apply(raw)
    return _kv(rid, key, raw)


def _set_details_section(rid, params, key, value, session):
    # `details_mode.<section>` -> `display.sections.<section>`; empty clears the override (frontend
    # then applies built-in section defaults before the global details_mode).
    section = key.split(".", 1)[1]
    if section not in _DETAIL_SECTION_NAMES:
        return _err(rid, 4002, f"unknown section: {section}")
    nv = _word(value)
    if nv and nv not in _DETAIL_MODES:
        return _err(rid, 4002, f"unknown details_mode: {value}")
    _write_display_sections(sections={section: nv} if nv else None, drop_sections=() if nv else (section,))
    return _kv(rid, key, nv)


def _toggle_setters() -> dict:
    """key -> (normaliser, cfg key, alias word -> value, flipped(current), report). ``""``/``toggle``
    flips the current value; an alias word maps directly; anything else is 4002. Built per call:
    the specs reference server.py globals (rebound at install)."""
    def on_off(v):
        return "on" if v else "off"
    return {
        # density/battery are on/off/toggle booleans on display.<field>.
        "density": (_word, "display.tui_compact", {"on": True, "off": False},
                    lambda: not bool(_display_cfg().get("tui_compact", False)), on_off),
        "battery": (_word, "display.battery",
                    {"on": True, "true": True, "yes": True, "off": False, "false": False, "no": False},
                    lambda: not bool(_display_cfg().get("battery", False)), on_off),
        "statusbar": (_word, "display.tui_statusbar", {"on": "top", **{m: m for m in _STATUSBAR_MODES}},
                      lambda: "top" if _coerce_statusbar(_display_cfg().get("tui_statusbar", "top")) == "off" else "off",
                      lambda v: v),
        # _raw_word: falsy non-strings (0, False) reach the alias map as themselves (-> 'off'), not toggle.
        "mouse": (_raw_word, "display.mouse_tracking", _MOUSE_TRACKING_ALIASES,
                  lambda: "all" if _display_mouse_tracking(_display_cfg()) == "off" else "off", lambda v: v)}


def _set_toggle(rid, params, key, value, session):
    norm, cfg_key, aliases, flipped, report = _toggle_setters()[key]
    raw = norm(value)
    nv = flipped() if raw in {"", "toggle"} else aliases.get(raw)
    if nv is None:
        return _err(rid, 4002, f"unknown {key} value: {value}")
    _write_config_key(cfg_key, nv)
    return _kv(rid, key, report(nv))


def _set_cwd(rid, params, key, value, session):
    raw = str(value or "").strip()
    if not raw:
        return _err(rid, 4002, "cwd required")
    cwd = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(cwd):
        return _err(rid, 4002, f"working directory does not exist: {raw}")
    _write_config_key("terminal.cwd", cwd)
    os.environ["TERMINAL_CWD"] = cwd
    return _kv(rid, "terminal.cwd", cwd, cwd=cwd, branch=git_probe.branch(cwd))


@_cfgset_guarded
def _set_prompt(rid, params, key, value, session):
    cfg = _load_cfg_raw()  # write-back round-trip
    if value == "clear":
        cfg.pop("custom_prompt", None)
    else:
        cfg["custom_prompt"] = value
    _save_cfg(cfg)
    return _kv(rid, key, "" if value == "clear" else value)


@_cfgset_guarded
def _set_personality(rid, params, key, value, session):
    pname, new_prompt = _validate_personality(str(value or ""), _load_cfg_raw())
    # Persists via hermes_cli.personality (single owner), never the user-owned system prompt.
    from hermes_cli.personality import persist_personality
    persist_personality(pname)
    history_reset, info = _apply_personality_to_session(params.get("session_id", ""), session, new_prompt, pname)
    return _kv(rid, key, str(value or "none"), history_reset=history_reset,
               **({"info": info} if info is not None else {}))


@_cfgset_guarded
def _set_skin(rid, params, key, value, session):
    _write_config_key("display.skin", value)
    # Every surface repaints; sync the watcher baseline so the poll loop doesn't re-broadcast.
    _broadcast_global_event("skin.changed", resolve_skin())
    _note_skin_broadcast()
    return _kv(rid, key, value)


def _set_display_toggle(rid, params, key, value, session):
    on = _BOOL_WORDS.get(str(value).strip().lower())
    if on is None:
        return _err(rid, 4002, f"{key} takes true or false")
    _write_config_key(key, on)
    return _kv(rid, key, on)


# ── dispatch

_CONFIG_SETTERS = {
    "model": _set_model, "fast": _set_fast, "busy": _set_busy, "verbose": _set_verbose, "focus": _set_focus,
    "approval_mode": _set_approval_mode, "approvals.mode": _set_word, "yolo": _set_yolo,
    "reasoning": _set_reasoning, "details_mode": _set_word, "thinking_mode": _set_word,
    "density": _set_toggle, "battery": _set_toggle, "theme": _set_word,
    "statusbar": _set_toggle, "mouse": _set_toggle, "indicator": _set_word,
    "cwd": _set_cwd, "terminal.cwd": _set_cwd, "workdir": _set_cwd,
    "prompt": _set_prompt, "personality": _set_personality, "skin": _set_skin}


@method("config.set")
@_profile_scoped
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))
    handler = _CONFIG_SETTERS.get(key)
    if handler is None and key.startswith("details_mode."):
        handler = _set_details_section
    elif handler is None and key in _DISPLAY_TOGGLE_KEYS:
        handler = _set_display_toggle
    if handler is None:
        return _err(rid, 4002, f"unknown config key: {key}")
    return handler(rid, params, key, value, session)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
