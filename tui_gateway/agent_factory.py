"""TUI agent_factory seam; functions bind to the server namespace at registration."""

from __future__ import annotations

from .method_ctx import bind_module

def _env_model_seed() -> str:
    """The launch-scoped model seed (``hermes --tui -m``, hosted provisioning); "" when unset."""
    return (os.environ.get("HERMES_MODEL", "") or os.environ.get("HERMES_INFERENCE_MODEL", "")).strip()


def _resolve_model() -> str:
    if env := _env_model_seed():
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return str(m.get("default", "") or "").strip()
    if isinstance(m, str) and m:
        return m.strip()
    # No env seed / config preference: the cost-safe silent default (cache-only read), never an unpicked flagship.
    with contextlib.suppress(Exception):
        from hermes_cli.models import get_preferred_silent_default_model
        return get_preferred_silent_default_model()
    return "z-ai/glm-5.2"


def _resolve_session_platform() -> str:
    """``HERMES_DESKTOP=1`` without ``HERMES_DESKTOP_TERMINAL`` → "desktop" (chat panel; the agent then
    suggests TUI-only slash commands), else "tui" (embedded terminal pane or standalone ``hermes --tui``)."""
    desktop = is_truthy_value(os.environ.get("HERMES_DESKTOP"))
    return "desktop" if desktop and not is_truthy_value(os.environ.get("HERMES_DESKTOP_TERMINAL")) else "tui"


def _resolve_session_source(explicit: str | None) -> str:
    """Session DB ``source``: an explicit caller value (plugin session tagged ``"telegram"``) is never
    rewritten; only empty/None falls back to the env-resolved platform."""
    return explicit or _resolve_session_platform()


def _resolve_agent_platform(source: str | None) -> str:
    return _resolve_session_source(source)


def _config_model_target() -> tuple[str, str]:
    """(model, provider) selected by config.yaml — and ONLY config: the HERMES_MODEL launch seed fed into
    the per-turn sync would be replayed as a /model switch and persisted globally, or pin the session so
    dashboard/CLI model changes never reach an open chat. Empty model = "no preference" → no-op sync."""
    cfg_model = _load_cfg().get("model")
    if isinstance(cfg_model, dict):
        provider = str(cfg_model.get("provider") or "").strip()
        return str(cfg_model.get("default", "") or "").strip(), "" if provider.lower() == "auto" else provider
    return (cfg_model.strip() if isinstance(cfg_model, str) else ""), ""


def _resolve_startup_runtime() -> tuple[str, str | None]:
    model = _resolve_model()
    if explicit_provider := os.environ.get("HERMES_TUI_PROVIDER", "").strip():
        return model, explicit_provider
    if not (explicit_model := _env_model_seed()):
        return model, None
    with contextlib.suppress(Exception):
        from hermes_cli.models import detect_static_provider_for_model
        cfg = _load_cfg().get("model") or {}
        current_provider = ((str(cfg.get("provider") or "").strip().lower() if isinstance(cfg, dict) else "")
                            or os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip().lower() or "auto")
        if detected := detect_static_provider_for_model(explicit_model, current_provider):
            provider, detected_model = detected
            return detected_model, provider
    return model, None


# Bare billing buckets are not routable provider identities; restoring one as a session provider override
# breaks resume. ``openrouter`` is deliberately NOT in this set (fully routable; agent_init's gate is a different set).
# (agent_init's fail-fast gate is a DIFFERENT set that also skips "openrouter" — there it means "default
# route, don't fail fast", not "unroutable".) ``openrouter`` is deliberately excluded here — it is a fully
# routable provider with its own API key and base_url. Sessions that used OpenRouter store
# ``billing_provider="openrouter"``; dropping it forces resume to the current global model (e.g. a custom
# endpoint), which is the wrong provider for the stored model. See #57588.


def _is_routable_provider(provider: str) -> bool:
    with contextlib.suppress(Exception):
        from hermes_cli.runtime_provider import is_routable_provider
        return is_routable_provider(provider)
    return False


def _overrides_have_routable_provider(overrides: dict) -> bool:
    """Whether persisted runtime overrides still name a routable provider (renamed/removed → "Unknown
    provider" at agent init). Empty = NOT routable, so the caller falls back to the session's picked model."""
    provider = str(overrides.get("provider_override") or "").strip()
    if not provider:
        provider = str((overrides.get("model_override") or {}).get("provider") or "").strip()
    return bool(provider) and _is_routable_provider(provider)


def _parse_model_config(raw, *, quiet: bool = False) -> dict:
    """A row's ``model_config`` (dict or JSON text) as a dict; ``{}`` when absent/invalid."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            if not quiet:
                raise
            logger.debug("failed to parse stored session model_config", exc_info=True)
    return {}


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Runtime fields persisted with a stored session (model column, ``billing_provider``, JSON ``model_config``):
    resume restores the model/provider/reasoning THAT chat used, not the global pick. Plugin-owned Bot-Mode
    sessions are exempt and rebuild from the member profile's CURRENT config (a stale provider pin left
    room bots "out of Nous credits" after a profile switch); signals: ``room_plumbing`` /
    ``follow_profile_config`` markers, the legacy hidden + "Group:" title, the title exactly "Bot Chat"."""
    if not row:
        return {}
    model_config = _parse_model_config(row.get("model_config"), quiet=True)
    _row_title = str(row.get("title") or "").strip()
    if (model_config.get("room_plumbing") or (row.get("hidden") and _row_title.startswith("Group:"))
            or model_config.get("follow_profile_config") or _row_title == "Bot Chat"):
        return {}
    overrides: dict = {}
    field = lambda k: str(model_config.get(k) or "").strip()
    model = str(row.get("model") or model_config.get("model") or "").strip()
    # ``billing_provider`` is only the billing bucket — for a custom endpoint the bare class "custom", which
    # agent_init treats as non-routable. Only restore an explicit provider; else resume uses the configured default.
    provider = field("provider")
    billing_provider = str(model_config.get("billing_provider") or row.get("billing_provider") or "").strip()
    if not provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
        provider = billing_provider
    base_url, api_mode, service_tier = field("base_url"), field("api_mode"), field("service_tier")
    reasoning_config = model_config.get("reasoning_config")
    # Heal a stale provider persisted by an older build (renamed/removed custom provider → "Unknown provider"):
    # recover ``custom:<name>`` from the stored base_url, then from the entry serving the model; else drop it.
    if provider and not _is_routable_provider(provider):
        healed = None
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            healed = canonical_custom_identity(base_url=base_url or None, model=model or None)
        except Exception:
            logger.debug("custom provider identity recovery failed", exc_info=True)
        if healed:
            logger.info("healed stale session provider %r to %r", provider, healed)
            provider = healed
            base_url = ""  # the healed identity owns a registered endpoint; the snapshot URL must not override it
        else:
            provider = ""
    if model:
        # Same dict-shaped override live /model switches use, so a DB-restored session keeps custom endpoint
        # metadata across resume and rebuilds (/new). Raw api_key is never persisted/restored.
        overrides["model_override"] = {
            "model": model, "provider": provider or None, "base_url": base_url or None, "api_mode": api_mode or None}
    if provider:
        overrides["provider_override"] = provider
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier:  # None = "inherit the profile" at _make_agent; "" = real override "no priority tier"
        overrides["service_tier_override"] = "" if service_tier.lower() == "normal" else service_tier
    return overrides


def _runtime_model_config(agent, existing: dict | None = None) -> dict:
    """Merge the agent's CURRENT runtime identity onto the row's persisted ``model_config``. Falsy agent
    attributes DELETE the key rather than skip the write: resume reads provider/endpoint from this JSON
    (model column written separately), so a stale provider would route the resumed chat to the wrong endpoint."""
    config = dict(existing or {})
    attr = lambda k: str(getattr(agent, k, "") or "").strip()
    model, provider, base_url = attr("model"), attr("provider"), attr("base_url")
    if provider.lower() == "custom":
        # ``agent.provider`` resolves every named custom entry to the literal "custom", losing the entry
        # identity (api_key is never persisted): recover ``custom:<name>`` from the endpoint URL.
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            provider = canonical_custom_identity(base_url=base_url, model=model or None) or provider
        except Exception:
            logger.debug("custom provider identity lookup failed", exc_info=True)
    reasoning_config = getattr(agent, "reasoning_config", None)
    live = {
        "model": model, "provider": provider, "base_url": base_url, "api_mode": attr("api_mode"),
        # An empty dict is still a real (present) reasoning config.
        "reasoning_config": reasoning_config if isinstance(reasoning_config, dict) else None,
        "service_tier": getattr(agent, "service_tier", None),
    }
    for key, value in live.items():
        if value or isinstance(value, dict):
            config[key] = value
        else:
            config.pop(key, None)
    return config


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist active session runtime so future resumes restore the same footer."""
    live = _live_session_agent_db(session)
    if live is None:
        return
    agent, session_key, db = live
    try:
        row = db.get_session(session_key) or {}
        model_config = _runtime_model_config(agent, _parse_model_config(row.get("model_config")))
        if (tier_override := session.get("create_service_tier_override")) is not None:
            # agent.service_tier is None for explicit normal; without this the distinction is erased on every persist.
            model_config["service_tier"] = tier_override or "normal"
        model = str(getattr(agent, "model", "") or "").strip()
        if hasattr(db, "update_session_meta"):
            db.update_session_meta(session_key, json.dumps(model_config), model or None)
        elif model and hasattr(db, "update_session_model"):
            db.update_session_model(session_key, model)
    except Exception:
        logger.debug("failed to persist live session runtime", exc_info=True)

def _load_approval_mode() -> str:
    """Effective ``approvals.mode`` via the gate's own ``_get_approval_mode`` (a raw re-read missed the
    managed overlay and ``${VAR}`` expansion)."""
    from tools.approval_context import _get_approval_mode
    mode = _get_approval_mode()
    return mode if mode in _APPROVAL_MODES else "manual"


def _coerce_statusbar(raw) -> str:
    if raw is False:
        return "off"
    return s if isinstance(raw, str) and (s := raw.strip().lower()) in _STATUSBAR_MODES else "top"


_MOUSE_TRACKING_ALIASES = {
    "0": "off", "1": "all", "all": "all", "any": "all", "button": "buttons", "buttons": "buttons",
    "click": "buttons", "false": "off", "full": "all", "no": "off", "off": "off", "on": "all",
    "scroll": "wheel", "true": "all", "wheel": "wheel", "yes": "all",
}


def _display_mouse_tracking(display: dict) -> str:
    """display.mouse_tracking → ``off|wheel|buttons|all`` (bools: True → all, False → off); ``wheel`` (DEC
    1000+1006) is the tmux-friendly subset without hover events. Legacy ``tui_mouse`` only when ``mouse_tracking`` is absent."""
    if not isinstance(display, dict):
        return "all"
    raw = display.get("mouse_tracking") if "mouse_tracking" in display else display.get("tui_mouse", True)
    if isinstance(raw, str):
        return _MOUSE_TRACKING_ALIASES.get(raw.strip().lower(), "all")
    return "off" if raw is False or raw == 0 else "all"


def _load_reasoning_config(model: str = "") -> dict | None:
    """Via the shared chokepoint :func:`hermes_constants.resolve_reasoning_config` (per-model override >
    global ``agent.reasoning_effort``; YAML False = disabled).

    Closes #21256.
    """
    from hermes_constants import resolve_reasoning_config
    return resolve_reasoning_config(_load_cfg(), model)


_SERVICE_TIER_ALIASES = {"fast": "priority", "priority": "priority", "on": "priority", "auto": "auto", "cold": "cold"}


def _load_service_tier() -> str | None:
    raw = str((_load_cfg().get("agent") or {}).get("service_tier", "") or "").strip().lower()
    return _SERVICE_TIER_ALIASES.get(raw)


def _load_provider_routing() -> dict:
    """OpenRouter ``provider_routing`` prefs (gateway/CLI parity — without them OpenRouter picks an effectively random provider)."""
    with contextlib.suppress(Exception):
        return _load_cfg().get("provider_routing", {}) or {}
    return {}


def _load_show_reasoning() -> bool:
    # Fallback True — keep in sync with DEFAULT_CONFIG display.show_reasoning (no DEFAULT_CONFIG merge here).
    return bool(_display_cfg().get("show_reasoning", True))


def _load_memory_notifications() -> str:
    """``display.memory_notifications`` (``off`` / ``on`` default / ``verbose``; bool normalized) — gates the
    "💾 Self-improvement review" summary (gateway/CLI parity)."""
    raw = _display_cfg().get("memory_notifications")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return str(raw).lower() if raw else "on"


_TOOL_PROGRESS_MODES = frozenset({"off", "new", "all", "verbose"})


def _load_tool_progress_mode() -> str:
    env = os.environ.get("HERMES_TUI_TOOL_PROGRESS", "").strip().lower()
    if env in _TOOL_PROGRESS_MODES:
        return env
    raw = _display_cfg().get("tool_progress", "all")
    if isinstance(raw, bool):
        return "all" if raw else "off"
    mode = str(raw or "all").strip().lower()
    return mode if mode in _TOOL_PROGRESS_MODES else "all"


def _gui_surface_toolsets(platform: str) -> set[str]:
    """Toolsets that exist because of the CLIENT (both off ``_HERMES_CORE_TOOLS``; this is the one gate).
    ``platform`` is the SESSION's source, never a process env var: the desktop may drive a URL/cloud
    backend where ``HERMES_DESKTOP`` is unset (AGENTS.md surface rule)."""
    return {"project", "desktop_ui"} if platform == "desktop" else {"project"}


def _tui_notice(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _resolve_explicit_toolsets(explicit: list[str], validate_toolset) -> list[str] | None | bool:
    """Resolve a HERMES_TUI_TOOLSETS pin: list, None for "all", False when nothing was valid."""
    built_in = [name for name in explicit if validate_toolset(name)]
    unresolved = [name for name in explicit if name not in built_in]
    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []
        built_in.extend(plugin_valid)
        unresolved = [name for name in unresolved if name not in plugin_valid]
    if any(name in {"all", "*"} for name in built_in):
        if ignored := [name for name in explicit if name not in {"all", "*"}]:
            _tui_notice(f"[tui] HERMES_TUI_TOOLSETS=all enables every toolset; ignoring additional entries: {', '.join(ignored)}")
        return None
    if not unresolved:
        return built_in
    try:  # (enabled, disabled) MCP server names from raw config; both empty on any failure
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag
        raw_cfg = read_raw_config()
        mcp_servers = raw_cfg.get("mcp_servers") if isinstance(raw_cfg.get("mcp_servers"), dict) else {}
        mcp_names, mcp_disabled = set(), set()
        for name, server_cfg in mcp_servers.items():
            if isinstance(server_cfg, dict):
                on = _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
                (mcp_names if on else mcp_disabled).add(str(name))
    except Exception:
        mcp_names, mcp_disabled = set(), set()
    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    if unknown:
        _tui_notice(f"[tui] ignoring unknown HERMES_TUI_TOOLSETS entries: {', '.join(unknown)}")
    if disabled:
        _tui_notice("[tui] ignoring disabled MCP servers in HERMES_TUI_TOOLSETS "
                    f"(set enabled: true in config.yaml to use): {', '.join(disabled)}")
    return (built_in + mcp_valid) or False


def _load_enabled_toolsets(platform: str | None = None) -> list[str] | None:
    """The agent's toolsets for this session (None = all): an explicit HERMES_TUI_TOOLSETS pin; else the
    coding posture (coding_context collapses to coding toolset + enabled MCP servers in a code workspace);
    else the configured CLI toolsets. Client-surface toolsets fold in here — only this surface can answer them."""
    session_platform = platform or _resolve_session_platform()
    explicit = [item.strip() for item in os.environ.get("HERMES_TUI_TOOLSETS", "").split(",") if item.strip()]
    fallback_notice = None
    if not explicit:
        with contextlib.suppress(Exception):
            from agent.coding_context import coding_selection
            selection = coding_selection(platform=session_platform)
            if selection is not None:
                return sorted({*selection, *_gui_surface_toolsets(session_platform)})
    try:
        from toolsets import validate_toolset
    except Exception:
        validate_toolset = None
    if explicit and validate_toolset is not None:
        resolved = _resolve_explicit_toolsets(explicit, validate_toolset)
        if resolved is not False:
            return resolved
        fallback_notice = "[tui] no valid HERMES_TUI_TOOLSETS entries; using configured CLI toolsets"
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        cfg = load_config()
        # include_default_mcp_servers=True is the runtime variant (the agent must be able to call
        # default MCP servers); the config-editing variant would silently drop MCP tools from the TUI.
        # Passing ``False`` here is the config-editing variant — used when we need to persist a toolset list
        # without baking in implicit MCP defaults. Using the wrong variant at agent creation time makes MCP
        # tools silently missing from the TUI. See PR #3252 for the original design split.
        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=True)
        if fallback_notice is not None:
            _tui_notice(fallback_notice)
        return sorted(enabled | _gui_surface_toolsets(session_platform)) if enabled else None
    except Exception:
        if fallback_notice is not None:
            _tui_notice("[tui] no valid HERMES_TUI_TOOLSETS entries and configured CLI toolsets could not be loaded; enabling all toolsets")
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _session_verbose(sid: str) -> bool:
    return _session_tool_progress_mode(sid) == "verbose"


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _tool_lifecycle_required_for_ui(name: str) -> bool:
    """Interactive UI, not optional chrome: Desktop renders clarify / setup_mcp cards from the tool-call part."""
    return name in ("clarify", "setup_mcp")

def _resolve_runtime_with_fallback(resolve_kwargs: dict | None = None) -> _RuntimeFallbackResolution:
    """Resolve the primary runtime or one complete provider/model fallback. Provider-only fallback entries
    are skipped so the unavailable primary model can never leak into a different runtime."""
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        return _RuntimeFallbackResolution(resolve_runtime_provider(**(resolve_kwargs or {})), None, False)
    except AuthError as primary_exc:
        for entry in _load_fallback_model() or []:
            fb_provider = str(entry.get("provider") or "").strip() if isinstance(entry, dict) else ""
            fb_model = str(entry.get("model") or "").strip() if isinstance(entry, dict) else ""
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key
                fb_kwargs: dict = {"requested": fb_provider, "target_model": fb_model,
                                   **({"explicit_base_url": entry["base_url"]} if entry.get("base_url") else {})}
                if fb_api_key := resolve_entry_api_key(entry):
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s model %s", primary_exc, fb_provider, fb_model)
                return _RuntimeFallbackResolution(runtime, fb_model, True)
            except Exception:
                continue
        raise


def _resolve_agent_model_runtime(model_override, provider_override) -> tuple[str, dict]:
    """(model, runtime) for a new agent; a per-session override (/model switch or a resumed row's persisted
    runtime) wins over global config/env. Older rows stored the resolved provider "custom" (no named entry
    matches) — recover the identity from the persisted base_url or the rebuild fails "No LLM provider
    configured". Persisted base_url/api_key/api_mode are honored only for the original runtime, never a fallback."""
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override.get("model") or "")
        requested_provider = model_override.get("provider") or provider_override or None
        override_base_url = model_override.get("base_url")
        resolve_kwargs = {}
        if str(requested_provider or "").strip().lower() == "custom":
            from hermes_cli.runtime_provider import canonical_custom_identity
            if recovered := canonical_custom_identity(base_url=override_base_url or None, model=model or None):
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand base_url to the direct-alias branch so pool/env credentials resolve.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs.update(requested=requested_provider, target_model=model or None)
        overrides = {k: model_override.get(k) for k in ("base_url", "api_key", "api_mode")}
    else:
        model, requested_provider = _resolve_startup_runtime()
        if isinstance(model_override, str) and model_override:
            model = model_override
        if provider_override:
            requested_provider = provider_override
        resolve_kwargs = {"requested": requested_provider, "target_model": model or None}
        overrides = {}
    resolution = _resolve_runtime_with_fallback(resolve_kwargs)
    if resolution.used_fallback:
        if not resolution.selected_model:
            raise RuntimeError("Auth fallback resolved without a model")
        return resolution.selected_model, resolution.runtime
    resolution.runtime.update({k: v for k, v in overrides.items() if v})
    return model, resolution.runtime


def _startup_system_prompt(cfg: dict, task_id: str) -> str:
    """Config ephemeral system prompt + HERMES_TUI_SKILLS preload block. Hard-fails only when EVERY requested
    skill is missing (cli.py parity): a typo'd name must not auto-block the Kanban task."""
    from hermes_cli.config import resolve_ephemeral_system_prompt_from_config
    system_prompt = resolve_ephemeral_system_prompt_from_config(cfg)
    startup_skills = _parse_tui_skills_env()
    if not startup_skills:
        return system_prompt
    from agent.skill_commands import build_preloaded_skills_prompt
    skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(startup_skills, task_id=task_id)
    if missing_skills:
        missing_display = ", ".join(missing_skills)
        if not loaded_skills:
            raise ValueError(f"Unknown skill(s): {missing_display}")
        logger.warning("Unknown skill(s) requested, skipping: %s. Continuing with: %s. "
                       "List available skills with `hermes skills list`.", missing_display, ", ".join(loaded_skills))
    if skills_prompt:
        system_prompt = "\n\n".join(part for part in (system_prompt, skills_prompt) if part).strip()
    return system_prompt


def _make_agent(
    sid: str, key: str, session_id: str | None = None, session_db=None,
    model_override: dict | str | None = None, provider_override: str | None = None,
    reasoning_config_override: dict | None = None, service_tier_override: str | None = None,
    platform_override: str | None = None, context_cwd_is_launch_artifact: bool | None = None):
    # AC-4 test seam: dead unless armed by the isolated certify harness.
    from tui_gateway.synthetic_turn import maybe_build_synthetic_agent
    synthetic = maybe_build_synthetic_agent(session_id or key, model_override)
    if synthetic is not None:
        return synthetic
    from run_agent import AIAgent
    # MCP discovery runs in a daemon thread (a dead server can't freeze the shell); the agent snapshots its tool
    # list once, so briefly wait for in-flight discovery. Dashboard /api/ws uses mcp_startup; TUI stdio uses entry.
    for _mod in ("hermes_cli.mcp_startup", "tui_gateway.entry"):
        with contextlib.suppress(Exception):
            importlib.import_module(_mod).wait_for_mcp_discovery()
    cfg = _load_cfg()
    # Load hooks alongside the same profile config used to construct this agent.
    from agent.shell_hooks import register_from_config
    register_from_config(cfg)
    system_prompt = _startup_system_prompt(cfg, session_id or key)
    model, runtime = _resolve_agent_model_runtime(model_override, provider_override)
    _pr = _load_provider_routing()
    platform = _resolve_agent_platform(platform_override)
    ignore_rules = is_truthy_value(os.environ.get("HERMES_IGNORE_RULES"))
    agent = AIAgent(
        model=model, max_iterations=_cfg_max_turns(cfg, 500), provider=runtime.get("provider"),
        base_url=runtime.get("base_url"), api_key=runtime.get("api_key"), api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"), acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"), quiet_mode=True,
        verbose_logging=False,  # DEBUG agent logging; independent of tool_progress_mode
        reasoning_config=(
            reasoning_config_override if reasoning_config_override is not None else _load_reasoning_config(str(model or ""))),
        service_tier=service_tier_override if service_tier_override is not None else _load_service_tier(),
        enabled_toolsets=_load_enabled_toolsets(platform),
        # OpenRouter provider_routing prefs (gateway + CLI parity).
        providers_allowed=_pr.get("only"), providers_ignored=_pr.get("ignore"), providers_order=_pr.get("order"),
        provider_sort=_pr.get("sort"), provider_require_parameters=_pr.get("require_parameters", False),
        provider_data_collection=_pr.get("data_collection"), platform=platform, session_id=session_id or key,
        session_db=session_db if session_db is not None else _get_db(), ephemeral_system_prompt=system_prompt or None,
        checkpoints_enabled=is_truthy_value(os.environ.get("HERMES_TUI_CHECKPOINTS")),
        pass_session_id=is_truthy_value(os.environ.get("HERMES_TUI_PASS_SESSION_ID")),
        skip_context_files=ignore_rules, skip_memory=ignore_rules, fallback_model=_load_fallback_model(),
        **_agent_cbs(sid))
    if context_cwd_is_launch_artifact is None:
        with _sessions_lock:
            context_cwd_is_launch_artifact = _context_cwd_is_launch_artifact(_sessions.get(sid))
    agent._context_cwd_is_launch_artifact = bool(context_cwd_is_launch_artifact)
    return agent


def register(server):
    bind_module(globals(), server)
