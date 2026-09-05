"""Agent callback wiring: child-session live mirror, per-session agent callbacks, personality
overlay, background/preview agent kwargs, agent reset. Bodies are rebound onto server.py's
globals at install time (method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import threading

from .method_ctx import bind_module


# Child-session live mirror: a delegated child's activity reaches the gateway only as
# relayed ``subagent.*`` events on the PARENT sid; translate them into native stream
# events on the CHILD sid (write_json routes by sid) so its own window is not silent.
_child_mirrors: dict[str, dict] = {}
_child_mirrors_lock = threading.Lock()
# Child sids with a run in flight (refreshed per relayed event, popped on complete) so a
# lazy watch resume reports running=true during a silent long tool.
_active_child_runs: dict[str, float] = {}
# Anything quiet this long lost its completion event — don't pin "running".
_CHILD_RUN_STALE_S = 3600.0
_CHILD_DELTA_EVENTS = {"subagent.thinking": "reasoning.delta", "subagent.text": "message.delta",
                       "subagent.start": "message.delta"}


def _child_run_active(child_key: str) -> bool:
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


def _mirror_subagent_to_child(event_type: str, payload: dict) -> None:
    child_key = str(payload.get("child_session_id") or "")
    if not child_key:
        return
    # Liveness registry first: accurate with no window open (one opened mid-run knows busy).
    if event_type == "subagent.complete":
        _active_child_runs.pop(child_key, None)
    else:
        _active_child_runs[child_key] = time.time()
    # Mirror only into a live watch session NOT upgraded to a full agent (an upgraded one owns
    # a real native stream). Either way drop state so a reopened window starts fresh.
    live = _find_live_session_by_key(child_key)
    if live is None or live[1].get("agent") is not None:
        with _child_mirrors_lock:
            _child_mirrors.pop(child_key, None)
        return
    csid = live[0]
    text = str(payload.get("text") or "")
    with _child_mirrors_lock:
        st = _child_mirrors.setdefault(child_key, {"seq": 0, "open_tool": None, "started": False})
        if not st["started"]:
            st["started"] = True
            _emit("message.start", csid)
        # thinking/text/start (the child's goal, as a one-time header) are plain deltas.
        if event_type in _CHILD_DELTA_EVENTS:
            if text:
                _emit(_CHILD_DELTA_EVENTS[event_type], csid,
                      {"text": f"{text}\n" if event_type == "subagent.start" else text})
            return
        if event_type not in ("subagent.tool", "subagent.complete"):
            return
        if st["open_tool"]:
            _emit("tool.complete", csid, st["open_tool"])
        if event_type == "subagent.tool":
            st["seq"] += 1
            tool = {"name": str(payload.get("tool_name") or "tool"),
                    "tool_id": f"submirror:{child_key}:{st['seq']}", "args": {}}
            if preview := str(payload.get("tool_preview") or payload.get("text") or ""):
                tool["preview"] = preview
            st["open_tool"] = tool
            _emit("tool.start", csid, tool)
        else:
            summary = str(payload.get("summary") or payload.get("text") or "")
            _emit("message.complete", csid, {"text": summary})
            _child_mirrors.pop(child_key, None)


def _agent_cbs(sid: str) -> dict:
    def _read_block(event: str, timeout: int):
        # read_terminal / read_preview (desktop GUI): blocking bridge like clarify; the preview
        # read gets longer since a URL tab extracts text from a live page.
        return lambda start=None, count=None: _block(
            event, sid, {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=timeout)

    callbacks = {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(sid, tc_id, name, args),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(sid, tc_id, name, args, result),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs),
        "tool_gen_callback": lambda name: _tool_progress_enabled(sid) and _emit("tool.generating", sid, {"name": name}),
        "thinking_callback": lambda text: _emit("thinking.delta", sid, {"text": text}),
        # Affection reaction (ily / <3 / good bot) → hearts; core-detected so TUI/desktop share it.
        "reaction_callback": lambda kind: _emit("reaction", sid, {"kind": kind}),
        "reasoning_callback": lambda text: _emit(
            "reasoning.delta", sid, {"text": text, **({"verbose": True} if _session_verbose(sid) else {})}),
        "status_callback": lambda kind, text=None: _status_update(sid, str(kind), None if text is None else str(text)),
        # Credits/notice spine: AgentNotice → notification.show; recovery → notification.clear.
        "notice_callback": lambda n: _emit(
            "notification.show", sid,
            {"text": n.text, "level": n.level, "kind": n.kind, "ttl_ms": n.ttl_ms, "key": n.key, "id": n.id}),
        "notice_clear_callback": lambda key: _emit("notification.clear", sid, {"key": key}),
        "clarify_callback": lambda q, c, multi_select=False, questions=None: (
            _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)),
        "read_terminal_callback": _read_block("terminal.read.request", 30),
        "read_preview_callback": _read_block("preview.read.request", 45),
        # drive_preview / annotate_preview (desktop GUI): same budget as the preview read it ends with.
        "drive_preview_callback": lambda payload: _block("preview.act.request", sid, dict(payload), timeout=45),
        # read_window_below (desktop GUI): main process enumerates native windows.
        "read_window_below_callback": lambda: _block("window.read.request", sid, {}, timeout=30),
        # setup_mcp (desktop GUI): consent card with a switch per connector; add/enable/key/OAuth via REST, then
        # mcp.setup.respond. 30 min on purpose: a catalog entry with setup steps sends the user off to somebody
        # else's admin console first (a Google Cloud project + client JSON is a quarter hour for a first-timer), and
        # the card expiring mid-console is the worst moment to lose it. Not now / Esc answer instantly, so nobody is
        # trapped by the longer wait. Like clarify, a late answer is tolerated. `server` rides alongside `servers`
        # so an older renderer that only reads the scalar keeps working.
        "setup_mcp_callback": lambda servers, action, reason, steps=(): _block(
            "mcp.setup.request", sid,
            {"server": servers[0] if servers else "", "servers": list(servers), "action": action,
             "reason": reason, "steps": list(steps)},
            timeout=1800),
        # tour (desktop GUI): renderer drives driver.js and answers tour.respond.
        "tour_callback": lambda payload: _tour_request(sid, payload)}

    # Interim assistant commentary (text alongside tool calls), gated on display.interim_assistant_
    # messages; _run_prompt_submit overwrites it per turn and clears it so a stale closure can't fire.
    if _load_interim_assistant_messages():
        callbacks["interim_assistant_callback"] = lambda text, *, already_streamed=False: _emit(
            "message.interim", sid, {"text": str(text), "already_streamed": bool(already_streamed)})
    return callbacks


def _apply_project_workspace(task_id: str, path: str, _name: str = "") -> None:
    """Intentional workspace move from the project_* tools: re-anchor the live session's cwd
    and push session.info. The ONLY auto-cwd path — an explicit tool call, never a `cd`."""
    if not path:
        return
    # task_id is the durable session_key; _sessions (and desktop event routing) key by sid.
    key = str(task_id or "")
    with _sessions_lock:
        sid, session = (key, _sessions[key]) if key in _sessions else next(
            ((s, c) for s, c in _sessions.items()
             if c.get("session_key") == key or getattr(c.get("agent"), "session_id", None) == key),
            ("", None))
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if session is None or not os.path.isdir(resolved):
        return
    # explicit switch supersedes a settle-adopted cwd
    session.update(cwd=resolved, explicit_cwd=True, cwd_from_settle=False)
    _register_session_cwd(session)
    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    try:
        agent = session.get("agent")
        info = _session_info(agent, session) if agent is not None else {
            "cwd": resolved, "branch": git_probe.branch(resolved),
            "project": _project_info_for_cwd(resolved), "lazy": True}
        _emit("session.info", sid, info)
    except Exception:
        logger.debug("failed to emit session.info after project workspace move", exc_info=True)


def _wire_callbacks(sid: str):
    from tools.terminal_tool import set_sudo_password_callback
    from tools.skills_tool import set_secret_capture_callback
    from tools.project_tools import set_project_workspace_callback

    def secret_cb(env_var, prompt, metadata=None):
        pl = {"prompt": prompt, "env_var": env_var, **({"metadata": metadata} if metadata else {})}
        val = _block("secret.request", sid, pl)
        if not val:
            return {"success": True, "stored_as": env_var, "validated": False, "skipped": True, "message": "skipped"}
        from hermes_cli.config import save_env_value_secure
        return {**save_env_value_secure(env_var, val), "skipped": False, "message": "ok"}

    set_sudo_password_callback(lambda: _block("sudo.request", sid, {}, timeout=120))
    set_project_workspace_callback(_apply_project_workspace)
    set_secret_capture_callback(secret_cb)
    # An MCP server asking for a value mid-tool-call is the same interaction as the agent asking one, so it
    # reuses the clarify prompt rather than introducing a second kind of question. Surfaces that don't wire
    # this decline the request, which is what every surface did before.
    from tools.mcp_tool_sampling import set_elicitation_input_callback
    set_elicitation_input_callback(lambda question, choices: _clarify_block(sid, question, choices))


def _available_personalities(cfg: dict | None = None) -> dict:
    """Built-ins + user overrides, via hermes_cli.personality (single owner)."""
    from hermes_cli.personality import available_personalities
    return available_personalities(_load_cfg() if cfg is None else cfg)


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    """(name, prompt) for a requested personality or ValueError; like resolve_personality but
    via the module-level _available_personalities so tests keep a single patch point."""
    from hermes_cli.personality import normalize_personality_name, render_personality_prompt
    if not (name := normalize_personality_name(value)):
        return "", ""
    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = ", ".join(f"`{n}`" for n in sorted(personalities))
        raise ValueError(f"Unknown personality: `{str(value).strip()}`.\n\nAvailable: `none`, {names}")
    return name, render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML for AIAgent (hermes_cli.personality owns this)."""
    from hermes_cli.personality import prompt_text
    return prompt_text(value)


def _apply_personality_to_session(
    sid: str, session: dict, new_prompt: str, personality: str = "") -> tuple[bool, dict | None]:
    """Apply a personality change without resetting history: the ephemeral system prompt is
    updated in place (appended at API-call time, so prompt-cache hits survive) plus a pivot
    marker so the model stops pattern-matching its earlier tone. Returns (False, info)."""
    if not session:
        return False, None
    session["personality"] = personality
    if not (agent := session.get("agent")):
        return False, None
    agent.ephemeral_system_prompt = new_prompt or None
    marker = (
        "[System: The user has changed the assistant's personality. "
        "From this point forward, adopt the following persona and respond "
        f"accordingly: {new_prompt}]"
        if new_prompt else
        "[System: The user has cleared the personality overlay. "
        "From this point forward, respond in your normal default style.]")
    # Like the model-switch marker: role=user so strict providers accept it mid-conversation,
    # but `display_kind` keeps it out of the `truncate_before_user_ordinal` addressing space
    # (untagged, every rewind would land one turn early and hard-delete the difference).
    # Untagged, it counts as a real user turn on the gateway side while no client counts it, so every later
    # rewind resolves one turn too early and `replace_messages` hard-deletes the difference (#82756).
    with session["history_lock"]:
        session["history"].append({"role": "user", "content": marker, "display_kind": "personality_switch"})
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(agent)
    _emit("session.info", sid, info)
    return False, info


def _cfg_max_turns(cfg: dict, default: int) -> int:
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    # Env override wins; resolve_turn_limit makes "none"/"unlimited"/0 first-class spellings.
    if env_val := os.environ.get("HERMES_TUI_MAX_TURNS"):
        return _resolve_turn_limit(env_val, default=default)
    raw = (cfg.get("agent") or {}).get("max_turns")
    if raw is None:
        raw = cfg.get("max_turns")
    return default if raw is None else _resolve_turn_limit(raw, default=default)


def _parse_tui_skills_env() -> list[str]:
    raw = os.environ.get("HERMES_TUI_SKILLS", "")
    return list(dict.fromkeys(p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()))


def _load_fallback_model():
    """Configured fallback chain via the shared ``get_fallback_chain`` (parity with
    HermesCLI/gateway: ``fallback_providers`` first, legacy ``fallback_model`` merged after)."""
    from hermes_cli.fallback_config import get_fallback_chain
    return get_fallback_chain(_load_cfg())


def _background_agent_kwargs(agent, task_id: str) -> dict:
    cfg = _load_cfg()

    def g(name, default=None):
        return getattr(agent, name, default)

    # Don't rehydrate a deliberately empty fallback chain.
    if hasattr(agent, "_fallback_chain"):
        fallback = agent._fallback_chain or []
    else:
        fallback = (agent._fallback_model if hasattr(agent, "_fallback_model")
                    else _load_fallback_model())
    # Detached tasks declare platform="tui" (no UI sid for renderer-routed events), so resolve
    # toolsets against it — never GUI schema they can't use.
    return {
        **{k: g(k) or None for k in ("base_url", "api_key", "provider", "api_mode", "acp_command",
                                     "acp_args", "ephemeral_system_prompt")},
        **{k: g(k) for k in ("providers_allowed", "providers_ignored", "providers_order", "provider_sort",
                             "provider_data_collection", "openrouter_min_coding_score")},
        "model": g("model") or _resolve_model(), "max_iterations": _cfg_max_turns(cfg, 25),
        "enabled_toolsets": g("enabled_toolsets") or _load_enabled_toolsets("tui"),
        "quiet_mode": True, "verbose_logging": False,
        "provider_require_parameters": g("provider_require_parameters", False), "session_id": task_id,
        "reasoning_config": g("reasoning_config") or _load_reasoning_config(str(g("model", "") or "")),
        "service_tier": g("service_tier") or _load_service_tier(),
        "request_overrides": dict(g("request_overrides", {}) or {}),
        "platform": "tui", "session_db": _get_db(), "fallback_model": fallback}


def _ephemeral_preview_agent_kwargs(agent, task_id: str) -> dict:
    return {**_background_agent_kwargs(agent, task_id),
            "enabled_toolsets": ["terminal", "file"], "session_db": None, "skip_memory": True}


def _preview_restart_history(session: dict, max_messages: int = 24, max_tool_chars: int = 1200) -> list[dict]:
    """Distill recent parent history for the ephemeral preview-restart agent (else it guesses
    app/cwd/port from the bare URL): last ``max_messages`` back to the last user turn, tool
    results truncated to ``max_tool_chars``."""
    try:
        with session["history_lock"]:
            history = list(session.get("history") or [])
    except Exception:
        history = list(session.get("history") or [])
    if not history:
        return []
    last_user = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
    start = max(0, len(history) - max_messages)
    if last_user is not None:
        start = min(start, last_user)
    trimmed: list[dict] = []
    for msg in history[start:]:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant", "tool", "system"):
            continue
        copy = {k: v for k, v in msg.items() if k != "reasoning"}
        content = copy.get("content")
        if msg.get("role") == "tool" and isinstance(content, str) and len(content) > max_tool_chars:
            copy["content"] = content[:max_tool_chars] + f"\n... (truncated, original {len(content)} chars)"
        trimmed.append(copy)
    return trimmed


def _preview_tool_result_preview(name: str, result: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return ""
    if name == "terminal":
        if output := str(data.get("output") or "").strip():
            return output[-1200:]
        if data.get("session_id"):
            return f"Background process started: {data.get('session_id')}"
        if data.get("exit_code") is not None:
            return f"terminal exited with code {data.get('exit_code')}"
    return str(data.get("error") or "").strip()[:1200]


def _preview_restart_callbacks(parent: str, task_id: str) -> dict:
    started_at: dict[str, float] = {}

    def progress(message: str, level: str = "info") -> None:
        if text := str(message or "").strip():
            _emit("preview.restart.progress", parent, {"task_id": task_id, "level": level, "text": text})

    def tool_start(tool_call_id: str, name: str, args: dict) -> None:
        started_at[tool_call_id] = time.time()
        ctx = _tool_ctx(name, args)
        progress(f"Running {name}{f': {ctx}' if ctx else ''}")

    def tool_complete(tool_call_id: str, name: str, _args: dict, result: str) -> None:
        duration_s = time.time() - started_at.get(tool_call_id, time.time())
        summary = _tool_summary(name, result, duration_s) or f"Finished {name}{f' in {_fmt_tool_duration(duration_s)}' if duration_s else ''}"
        output = _preview_tool_result_preview(name, result)
        progress(summary + (f"\n{output}" if output else ""))

    def tool_progress(event_type: str, name: str | None = None, preview: str | None = None, **_kwargs) -> None:
        if preview or name:
            progress(str(preview) if preview else f"{event_type.replace('.', ' ')}: {name}")

    return {
        "tool_start_callback": tool_start, "tool_complete_callback": tool_complete,
        "tool_progress_callback": tool_progress,
        "tool_gen_callback": lambda name: progress(f"Preparing {name}"),
        "status_callback": lambda kind, text=None: progress(text if text is not None else kind)}


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        # /new is a full conversation boundary: session-scoped runtime overrides (/model,
        # /reasoning, /fast) do NOT carry forward and the pins are cleared so a rebuild can't
        # resurrect them. Global process state is never touched (see _apply_model_switch).
        for k in ("model_override", "create_reasoning_override", "create_service_tier_override", "one_turn_model_restore"):
            session.pop(k, None)
        new_agent = _make_agent(
            sid, session["session_key"], session_id=session["session_key"],
            platform_override=_session_source(session),
            context_cwd_is_launch_artifact=_context_cwd_is_launch_artifact(session))
    finally:
        _clear_session_context(tokens)
    session.update(
        agent=new_agent, config_model_seen=_config_model_target(), attached_images=[],
        queued_prompt=None,
        _queued_prompt_generation=int(session.get("_queued_prompt_generation", 0)) + 1,
        edit_snapshots={}, image_counter=0, running=False, show_reasoning=_load_show_reasoning(),
        tool_progress_mode=_load_tool_progress_mode(), tool_started_at={})
    session.pop("queued_prompts", None)
    with session["history_lock"]:
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(new_agent, session)
    _emit("session.info", sid, info)
    _restart_slash_worker(sid, session)
    return info


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
