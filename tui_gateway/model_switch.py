"""Model switching for a live session: persist, snapshot/restore runtime, /model apply with
guards, bot-capability + config sync. Bodies are rebound onto server.py's globals at install
time (method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _persist_model_switch(result) -> None:
    # Targeted key writes: a full `model:` block rewrite via save_config() would destroy
    # sibling keys the user set there (`model_slots`, `model_fallback`, ...).
    from cli import save_config_value
    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    # A provider without a base_url must clear the stale one (custom endpoint -> native)
    # or the new model routes at the old host; reads coalesce null to absent.
    save_config_value("model.base_url", result.base_url or None)


_RUNTIME_KEYS = ("model", "provider", "api_key", "base_url", "api_mode")


def _snapshot_agent_model_runtime(agent) -> dict:
    """Capture the current agent model runtime for a one-turn restore."""
    return {**{k: getattr(agent, k, "") for k in _RUNTIME_KEYS},
            "primary_runtime": copy.deepcopy(getattr(agent, "_primary_runtime", None))}


def _restore_agent_model_runtime(agent, snapshot: dict | None) -> None:
    """Restore an agent model runtime captured before a one-turn override."""
    if not snapshot or agent is None:
        return
    primary = snapshot.get("primary_runtime")
    if primary and hasattr(agent, "_restore_primary_runtime"):
        try:
            agent._primary_runtime = copy.deepcopy(primary)
            agent._fallback_activated = True
            agent._rate_limited_until = 0
            if agent._restore_primary_runtime():
                return
        except Exception:
            logger.debug("TUI one-turn model restore via primary runtime failed", exc_info=True)
    if hasattr(agent, "switch_model"):
        model, provider, api_key, base_url, api_mode = (snapshot.get(k, "") for k in _RUNTIME_KEYS)
        agent.switch_model(
            new_model=model, new_provider=provider, api_key=api_key, base_url=base_url,
            api_mode=api_mode, capabilities=snapshot.get("capabilities"))


@contextlib.contextmanager
def _session_profile_runtime_scope(session: dict):
    """Bind model resolution to the session's profile config and secrets."""
    profile_home = session.get("profile_home")
    if not profile_home:
        yield
        return
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    # Same terminal policy the gateway binds per turn: a docker-configured profile
    # must never resolve the launch process's pinned env. Failure → refusal scope.
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    terminal_token = install_profile_terminal_scope(Path(profile_home))
    try:
        yield
    finally:
        reset_terminal_scope(terminal_token)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _restart_completed_failed_agent_build(sid: str, session: dict, failed_ready: threading.Event | None) -> bool:
    """Replace one completed failed build generation and start its retry."""
    if failed_ready is None:
        return False
    with session.setdefault("agent_build_lock", threading.Lock()):
        if (session.get("agent") is not None or session.get("agent_error") is None
                or session.get("agent_ready") is not failed_ready or not failed_ready.is_set()):
            return False
        model_override = session.get("model_override")
        resume_overrides = session.get("resume_runtime_overrides")
        if isinstance(model_override, dict) and isinstance(resume_overrides, dict):
            resume_overrides = {**resume_overrides, "model_override": model_override}
            if provider := model_override.get("provider"):
                resume_overrides["provider_override"] = provider
            else:
                resume_overrides.pop("provider_override", None)
            session["resume_runtime_overrides"] = resume_overrides
        session["agent_error"] = None
        session["agent_ready"] = threading.Event()
        session.pop("agent_build_started", None)
        session.pop("_agent_build_thread", None)
    _start_agent_build(sid, session)
    return True


def _switch_request(raw_input: str, parsed_flags, persist_override) -> tuple[str, str, bool, bool]:
    """Normalize /model flags → (model_input, explicit_provider, one_turn, persist_global)."""
    from hermes_cli.model_switch import (
        MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL, MODEL_SWITCH_ERROR_TEXT, parse_model_switch_args,
        resolve_persist_behavior)

    f = parse_model_switch_args(raw_input) if parsed_flags is None else parsed_flags
    model_input, explicit_provider, is_global_flag, is_session, one_turn = (
        f.model_input, f.explicit_provider, f.is_global, f.is_session, f.is_once)
    # Conflict validation is the shared parser's; surface it with the canonical copy.
    if is_global_flag and one_turn:
        raise ValueError(MODEL_SWITCH_ERROR_TEXT[MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL])
    if persist_override is None:
        persist_override = resolve_persist_behavior(
            is_global_flag, is_session, is_once=one_turn, explicit_provider=explicit_provider)
    if not model_input:
        raise ValueError("model value required")
    return model_input, explicit_provider, one_turn, persist_override


def _current_model_runtime(agent, explicit_provider: str) -> tuple:
    """(provider, model, base_url, api_key) to switch from: live agent, else configured runtime."""
    if agent:
        return tuple(
            getattr(agent, k, "") or "" for k in ("provider", "model", "base_url", "api_key"))
    current_model = _resolve_model()
    if explicit_provider:
        return explicit_provider.strip(), current_model, "", ""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(requested=None)
    # Keep a callable api_key (Azure Entra bearer) unchanged: ``str()`` would
    # yield "<function ...>" and poison switch_model validation.
    key = runtime.get("api_key", "")
    if not (callable(key) and not isinstance(key, str)):
        key = str(key or "")
    provider = str(runtime.get("provider", "") or "")
    return provider, current_model, str(runtime.get("base_url", "") or ""), key


def _merge_preflight_warning(result, agent, session: dict, cfg, custom_provs) -> None:
    """Fold the context-compression preflight warning into ``result`` (best-effort)."""
    try:
        from hermes_cli.context_switch_guard import merge_preflight_compression_warning
        cfg_ctx = None
        mc = cfg.get("model", {}) if isinstance(cfg, dict) else None
        if isinstance(mc, dict) and mc.get("context_length") is not None:
            cfg_ctx = int(mc["context_length"])
        merge_preflight_compression_warning(
            result, agent=agent, messages=list(session.get("history", [])),
            custom_providers=custom_provs, config_context_length=cfg_ctx)
    except Exception as exc:
        logger.debug("preflight-compression switch warning failed: %s", exc)


def _expensive_model_confirm(result, current_base_url: str, current_api_key) -> dict | None:
    """Deferred-confirm response when the selection guards flag the target model, else None."""
    try:
        from hermes_cli.model_selection_guards import combined_selection_warning
        warning = combined_selection_warning(
            result.new_model, provider=result.target_provider, base_url=result.base_url or current_base_url,
            api_key=result.api_key or current_api_key, model_info=result.model_info)
    except Exception:
        warning = None
    if warning is None:
        return None
    msg = f"{warning.message}\n\n{result.warning_message}" if result.warning_message else warning.message
    # Same contract as _set_model's deferred branch: confirm_message is canonical, warning legacy.
    return {"value": result.new_model, "warning": msg, "confirm_required": True, "confirm_message": msg}


def _commit_agent_switch(sid: str, session: dict, agent, result, current_model: str, snapshot):
    """Swap the live agent in place, then restart/persist/mark/announce; a failed swap aborts."""
    try:
        agent.switch_model(
            new_model=result.new_model, new_provider=result.target_provider, api_key=result.api_key,
            base_url=result.base_url, api_mode=result.api_mode,
            capabilities=getattr(result, "runtime_capabilities", None))
    except Exception as exc:
        # The in-place swap rolled the agent back and re-raised. Abort the whole commit (worker
        # restart, persist, marker, override, config write) or the session pins a broken model.
        # Abort the commit: do NOT restart the slash worker, persist runtime, append the switch marker, set
        # a session model_override, or persist to config — all of which would otherwise leave the session
        # pinned to a broken model and kill the conversation on the next turn (#50163). A failed switch is a
        # no-op; surface a clean error to the client.
        logger.warning("In-place model switch failed for TUI agent: %s", exc)
        raise ValueError(f"Model switch to {result.new_model} failed ({exc}); "
                         f"staying on {getattr(agent, 'model', current_model)}.") from exc
    _restart_slash_worker(sid, session)
    _persist_live_session_runtime(session)
    _persist_live_session_system_prompt(session)
    _append_model_switch_marker(session, model=result.new_model, provider=result.target_provider)
    _emit_session_info(sid, session)
    if snapshot is not None:
        session["one_turn_model_restore"] = snapshot
    else:
        session.pop("one_turn_model_restore", None)


def _apply_model_switch(
    sid: str, session: dict, raw_input: str, *, confirm_expensive_model: bool = False,
    pin_session_override: bool = True, parsed_flags: Any | None = None,
    persist_override: bool | None = None) -> dict:
    from hermes_cli.model_switch import switch_model
    model_input, explicit_provider, one_turn, persist_global = _switch_request(
        raw_input, parsed_flags, persist_override)
    agent = session.get("agent")
    if one_turn and not agent:
        raise ValueError("/model --once requires a live session")
    current_provider, current_model, current_base_url, current_api_key = _current_model_runtime(
        agent, explicit_provider)
    # User-defined providers let switch_model resolve named custom endpoints
    # (e.g. "ollama-launch") and validate against saved model lists.
    user_provs = custom_provs = cfg = None
    with contextlib.suppress(Exception):
        from hermes_cli.config import get_compatible_custom_providers, load_config
        cfg = load_config()
        user_provs = cfg.get("providers")
        custom_provs = get_compatible_custom_providers(cfg)
    result = switch_model(
        raw_input=model_input, current_provider=current_provider, current_model=current_model,
        current_base_url=current_base_url, current_api_key=current_api_key, is_global=persist_global,
        explicit_provider=explicit_provider, user_providers=user_provs,
        custom_providers=custom_provs)
    if not result.success:
        raise ValueError(result.error_message or "model switch failed")
    restore_snapshot = _snapshot_agent_model_runtime(agent) if (one_turn and agent) else None
    if agent:
        _merge_preflight_warning(result, agent, session, cfg, custom_provs)
    if not confirm_expensive_model:
        confirm = _expensive_model_confirm(result, current_base_url, current_api_key)
        if confirm is not None:
            return confirm
    if agent:
        _commit_agent_switch(sid, session, agent, result, current_model, restore_snapshot)
    # PER-SESSION override so a rebuild of THIS session (/new, resume) re-derives the model.
    # Deliberately NOT written to process-global env (HERMES_MODEL & co.): the desktop hosts
    # every same-profile session in one process, so os.environ would leak the switch to all.
    if pin_session_override and isinstance(session, dict) and not one_turn:
        session["model_override"] = {
            "model": result.new_model, "provider": result.target_provider,
            "base_url": result.base_url, "api_key": result.api_key, "api_mode": result.api_mode}
    if persist_global:
        _persist_model_switch(result)
    return {
        "value": result.new_model, "warning": result.warning_message or "",
        "confirm_required": False,
        "scope": "once" if one_turn else ("global" if persist_global else "session")}


def _sync_bot_capabilities(sid: str, session: dict) -> None:
    """Rebuild a Bot Chat session's agent when its capability surface changed. Bot Chats are
    eternal sessions with toolsets/MCP baked in at construction, so a capability edit would
    otherwise wait for /new: fingerprint at turn start and on change swap in a fresh agent for
    the SAME session (history is DB-backed)."""
    agent = session.get("agent")
    if agent is None:
        return
    try:
        title = str(getattr(agent, "_session_title_hint", "") or "").strip()
        if not title:
            db, key = getattr(agent, "_session_db", None), session.get("session_key") or ""
            title = str((db.get_session_title(key) if (db and key) else None) or "").strip()
        if title != "Bot Chat":
            return
        from tools.bot_mode_probe import capability_fingerprint
        current = capability_fingerprint(session.get("profile_home") or None)
        if current == "unavailable":
            return
        seen = session.get("bot_caps_seen")
        session["bot_caps_seen"] = current
        if seen is None or seen == current:
            return
    except Exception:
        return
    try:
        tokens = _set_session_context(sid, cwd=_session_cwd(session))
        try:
            new_agent = _rebuild_session_agent(sid, session, session_id=session["session_key"],
                                               platform_override=_session_source(session))
        finally:
            _clear_session_context(tokens)
        new_agent._session_title_hint = "Bot Chat"
        _emit("notice", sid, {"message": "Capabilities updated — this bot's tools and prompt were refreshed."})
    except Exception as e:
        logger.warning("Bot capability sync failed for %s: %s", sid, e)


def _sync_agent_model_with_config(sid: str, session: dict) -> None:
    """Adopt a config.yaml model change at turn start (like gateways do per message). Sessions
    pinned with /model keep their choice; a failed switch keeps the current model."""
    agent = session.get("agent")
    if agent is None or session.get("model_override"):
        return
    target = _config_model_target()
    if not target[0]:
        return
    seen = session.get("config_model_seen")
    # Record first so a broken config gets one attempt per edit, not per turn.
    session["config_model_seen"] = target
    model, provider = target
    # Already on the configured model (resumed before first sync, or a config revert after
    # a failed switch): adopt without switching.
    if target == seen or (
            model == getattr(agent, "model", "") and (not provider or provider == getattr(agent, "provider", ""))):
        return
    raw = f"{model} --provider {provider}" if provider else model
    try:
        # This sync ADOPTS a config.yaml change; it must never write config back (that is
        # how `hermes --tui -m` once leaked into config.yaml).
        _apply_model_switch(
            sid, session, raw, confirm_expensive_model=True, pin_session_override=False,
            persist_override=False)
    except Exception as e:
        _emit("error", sid, {"message": f"Could not switch to configured model {model}: {e}"})


def _pending_switch_selection_warning(model: str, provider: str) -> str | None:
    """Selection-guard message for a model queued mid-turn, or ``None``. Runs BEFORE the pick is
    stashed (the client can still turn the response into a confirm prompt); only pre-resolution
    inputs exist so it can only under-fire — ``_apply_model_switch`` is the backstop."""
    if not model:
        return None
    try:
        from hermes_cli.model_selection_guards import combined_selection_warning
        warning = combined_selection_warning(model, provider=provider or None)
    except Exception:
        return None
    return warning.message if warning is not None else None


def _mirror_resolved_model_switch(sid: str, session: dict, slash_meta: dict) -> str:
    """Mirror the worker-resolved model switch onto the live TUI session.

    Consumes ONLY the structured snapshot the slash worker reported
    (``resolved_model`` / ``resolved_provider`` / ``base_url`` / ``api_mode`` /
    ``scope``). The worker ran inside the session's ``profile_home`` scope, so those
    values are THAT profile's resolution: re-parsing ``raw_args`` (or consulting this
    process's alias cache) would pin Profile B's session to Profile A's resolution
    whenever the two disagree.

    Fail-closed contract:
      * The profile-home scope AND the secret scope must BOTH be established before any
        provider resolution / config / agent mutation runs. Every token is registered
        with an ``ExitStack``, so a failure that happens AFTER a token was installed —
        including a failure to establish the NEXT scope — restores it in reverse order
        and no ContextVar leaks.
      * If either scope fails, provider resolution fails, the agent apply raises, or the
        config write raises, the helper returns a warning and leaves the live agent,
        session and config untouched.
      * No global ``os.environ`` mutation is ever performed.

    Returns a warning string (empty on success) for the ``slash.exec`` payload.
    Raises ``ValueError`` when the metadata is malformed.
    """
    if not isinstance(slash_meta, dict) or slash_meta.get("side_effect") != "model_switch":
        return ""
    resolved_model = str(slash_meta.get("resolved_model") or "").strip()
    resolved_provider = str(slash_meta.get("resolved_provider") or "").strip()
    scope_value = str(slash_meta.get("scope") or "session").strip()
    if not resolved_model or not resolved_provider:
        # Worker reported the side effect but its resolved target is incomplete:
        # refuse to mirror rather than silently drift onto the old model.
        raise ValueError(
            "slash worker reported model_switch without resolved_model/resolved_provider")

    base_url = slash_meta.get("base_url") or ""
    api_mode = slash_meta.get("api_mode") or ""
    agent = session.get("agent")
    profile_home = session.get("profile_home")

    with contextlib.ExitStack() as stack:
        if profile_home:
            try:
                from hermes_constants import set_hermes_home_override
                home_token = set_hermes_home_override(Path(profile_home))
                stack.callback(_safe_reset_home, home_token)
            except Exception as exc:
                return f"mirror aborted: profile home scope failed ({exc})"
            try:
                from agent.secret_scope import (
                    build_profile_secret_scope,
                    set_secret_scope,
                )
                secret_token = set_secret_scope(
                    build_profile_secret_scope(Path(profile_home)))
                stack.callback(_safe_reset_secret, secret_token)
            except Exception as exc:
                return f"mirror aborted: secret scope failed ({exc})"

            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                runtime = resolve_runtime_provider(
                    requested=resolved_provider, target_model=resolved_model,
                    explicit_base_url=str(base_url or "") or None) or {}
            except Exception as exc:
                # The scopes are live here; the stack unwinds them before returning.
                return f"mirror aborted: provider resolution failed ({exc})"
            api_key = str(runtime.get("api_key") or "")
            base_url = str(runtime.get("base_url") or base_url or "")
            api_mode = str(runtime.get("api_mode") or api_mode or "")
        else:
            # No profile home: the worker resolved against the launch profile's config and
            # the live agent keeps the credentials it already holds (empty is the
            # ``switch_model`` default — "no explicit key", not a cleared one).
            api_key = ""

        restore_snapshot = None
        if agent is not None:
            # Snapshot BEFORE the switch so a --once turn can restore the exact
            # pre-switch runtime (upstream one-turn semantics).
            if scope_value == "once":
                restore_snapshot = _snapshot_agent_model_runtime(agent)
            try:
                agent.switch_model(
                    new_model=resolved_model, new_provider=resolved_provider,
                    api_key=api_key, base_url=base_url, api_mode=api_mode)
            except Exception as exc:
                # The agent helper rolls its own runtime back on failure; no session
                # state has been written yet, so nothing else needs unwinding.
                return f"model mirror failed: {exc}"

        if scope_value == "global":
            # Persist to THIS session profile's config.yaml; save_config_value reads the
            # context-local home override installed above.
            try:
                from cli import save_config_value
                save_config_value("model.default", resolved_model)
                save_config_value("model.provider", resolved_provider)
                save_config_value("model.base_url", base_url or None)
                save_config_value("model.api_mode", api_mode or None)
            except Exception as exc:
                return f"model persistence failed: {exc}"

        if scope_value == "once":
            # A one-turn switch must NOT pin a persistent session override: a
            # ``model_override`` entry would skip config sync (_sync_agent_model_with_config
            # returns early) and re-apply the temporary model on rebuild/resume//new. Only
            # the restore snapshot is latched; a pre-existing override is left untouched
            # and the production turn finally restores the pre-switch runtime.
            if restore_snapshot is not None:
                session["one_turn_model_restore"] = restore_snapshot
        else:
            session["model_override"] = {
                "provider": resolved_provider, "model": resolved_model,
                "base_url": base_url, "api_key": api_key, "api_mode": api_mode,
                "scope": scope_value}
            session.pop("one_turn_model_restore", None)
        # Success: the ExitStack restores both tokens in reverse order on exit.
        return ""


def _safe_reset_home(token) -> None:
    """ExitStack callback: reset the home override token (never raises)."""
    if token is None:
        return
    try:
        from hermes_constants import reset_hermes_home_override
        reset_hermes_home_override(token)
    except Exception:
        pass


def _safe_reset_secret(token) -> None:
    """ExitStack callback: reset the secret scope token (never raises)."""
    if token is None:
        return
    try:
        from agent.secret_scope import reset_secret_scope
        reset_secret_scope(token)
    except Exception:
        pass


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
