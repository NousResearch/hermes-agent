"""Tools & system / slash / insights / rollback / plugins / cron / skills / MCP JSON-RPC handlers.

Rebound onto server.py's globals at install time (``method_ctx.bind_module``), so
bodies reference server globals bare (``_ok``, ``_err``, ``_sessions``, ...).
Helper names must not collide with server.py's own (``_cmd_`` / ``_toolset_`` / ``_mcp_`` prefixes).
"""

import sys

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ─── Shared helpers ──────────────────────────────────────────────────────────
def _profile_scoped_rpc(
    fail_code: int, *, required=(), catch_resolve: bool = True, prefix: str = "",
    scoped: bool = True, live_session: bool = False,
):
    """Wrap a handler body with the optional ``profile`` HERMES_HOME scope. Order: ``required``
    params (4063 ``<key> required``) → ``live_session`` resolution via ``_sess`` (waits for the
    agent build; body gets ``session`` as 3rd arg) → profile (4064 when its dir is missing) → body;
    body exceptions become ``fail_code`` (``prefix`` + message). ``catch_resolve`` also maps
    resolve-time exceptions to ``fail_code``; mcp.servers.* let them propagate to dispatch().
    ``scoped=False`` ignores ``profile``. The override is always reset afterwards."""

    def deco(body):
        def handler(rid, params: dict) -> dict:
            for key, present in required:
                if not present(params.get(key)):
                    return _err(rid, 4063, f"{key} required")
            args = (rid, params)
            if live_session:
                session, err = _sess(params, rid)
                if err:
                    return err
                args = (rid, params, session)
            token = None
            if profile := _str_arg(params, "profile") if scoped else "":
                try:
                    profile_dir = _tools_mod("hermes_cli.profiles").get_profile_dir(profile)
                    if not profile_dir or not profile_dir.is_dir():
                        return _err(rid, 4064, f"profile '{profile}' not found")
                    token = _tools_mod("hermes_constants").set_hermes_home_override(str(profile_dir))
                except Exception as e:
                    if not catch_resolve:
                        raise
                    return _err(rid, fail_code, str(e))
            try:
                return body(*args)
            except Exception as e:
                return _err(rid, fail_code, f"{prefix}{e}")
            finally:
                _mcp_reset_profile(token)
        handler.__doc__ = body.__doc__
        return handler
    return deco


def _guarded(fail_code: int, prefix: str = "", *, live_session: bool = False):
    """Body exceptions → ``_err(rid, fail_code, prefix + str(e))``; no profile scope. ``live_session``
    resolves the session first and calls ``body(rid, params, session)``."""
    return _profile_scoped_rpc(fail_code, prefix=prefix, scoped=False, live_session=live_session)


def _rpc(name: str, fail_code: int, prefix: str = "", *, live_session: bool = False):
    """``@method(name)`` + ``_guarded``."""
    return lambda body: method(name)(_guarded(fail_code, prefix, live_session=live_session)(body))


def _scoped_rpc(name: str, fail_code: int = 5024, **kw):
    """``@method(name)`` + ``_profile_scoped_rpc`` (optional ``profile`` HERMES_HOME scope)."""
    return lambda body: method(name)(_profile_scoped_rpc(fail_code, **kw)(body))


def _str_arg(params: dict, key: str) -> str:
    return str(params.get(key) or "").strip()


def _tools_mod(module: str):
    """Deferred module import for one-liner bodies (startup budget: never import at load)."""
    import importlib
    return importlib.import_module(module)


_stripped = lambda v: bool(str(v or "").strip())  # noqa: E731 — required-param predicates
_nonempty = lambda v: not (v is None or str(v) == "")  # noqa: E731
_NAME = (("name", _stripped),)
_NAME_SESSION = (("name", _stripped), ("session_id", _stripped))


def _mcp_rpc(name: str, required=_NAME):
    """mcp.servers.* contract: profile scope, ``required`` params (default ``name``), body errors → 5024,
    profile-resolve errors propagate to dispatch()."""
    return _scoped_rpc(f"mcp.servers.{name}", required=required, catch_resolve=False)


def _mcp_named_server(rid, params):
    """(name, servers, None) for a configured server, else (name, servers, 4064 error)."""
    name, servers = _str_arg(params, "name"), _tools_mod("hermes_cli.mcp_config")._get_mcp_servers()
    return name, servers, None if name in servers else _err(rid, 4064, f"server '{name}' not found")


def _busy_error(rid, session, cmd: str):
    if session.get("running"):
        return _err(rid, 4009, f"session busy — /interrupt the current turn before /{cmd}")
    return None


def _session_key_or_err(rid, session, module: str, label: str):
    """(session_key, module, None) for the /goal and /loop managers, else (None, None, error):
    4001 without a session/key, 5030 when ``module`` fails to import."""
    if not session:
        return None, None, _err(rid, 4001, "no active session")
    if not (sid_key := session.get("session_key") or ""):
        return None, None, _err(rid, 4001, "no session key")
    try:
        return sid_key, _tools_mod(module), None
    except Exception as exc:
        return None, None, _err(rid, 5030, f"{label} unavailable: {exc}")


def _user_turn_indices(session):
    """(history, indices of user-originated turns) minus ephemeral scaffolding. Call under history_lock."""
    is_user = _tools_mod("agent.context_compressor").user_originated_turn_view
    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    return history, [i for i, m in enumerate(history) if is_user(m) is not None]


def _rewind_prelude(rid, session, cmd: str, empty_msg: str):
    """Under history_lock: re-check busy, then (history, user_indices, None) or (None, None, error)."""
    if busy := _busy_error(rid, session, cmd):
        return None, None, busy
    history, user_indices = _user_turn_indices(session)
    if not user_indices:
        return None, None, _err(rid, 4018, empty_msg)
    return history, user_indices, None


def _rewind_or_err(rid, session, keep: int, value_err: tuple, fail_prefix: str, **kw):
    """``_rewind_active_session_history`` → (result, None); ValueError → ``value_err`` (code, prefix),
    other exceptions → 5008 ``fail_prefix`` + message."""
    try:
        return _rewind_active_session_history(session, keep, **kw), None
    except ValueError as exc:
        return None, _err(rid, value_err[0], f"{value_err[1]}{exc}")
    except Exception as exc:
        return None, _err(rid, 5008, f"{fail_prefix}{exc}")


def _clip(text: str, n: int = 120) -> str:
    return text[:n] + ("…" if len(text) > n else "")


def _exec_out(rid, output: str) -> dict:
    """command.dispatch display-only result."""
    return _ok(rid, {"type": "exec", "output": output})


def _capture_run_kwargs(timeout: int) -> dict:
    """Shared captured-text subprocess.run kwargs: UTF-8 + lossy decode (non-UTF-8 child output must
    not crash the gateway thread on Windows), no stdin, no console flash under the desktop parent."""
    return dict(
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, creationflags=_tools_mod("hermes_cli._subprocess_compat").windows_hide_flags())


def _captured_exec(rid, cmd, timeout: int, *, on_result, timeout_err: tuple, fail_code: int,
                   shell: bool = False, env: "dict | None" = None) -> dict:
    """Run ``cmd`` captured (see ``_capture_run_kwargs``) and hand the CompletedProcess to
    ``on_result``; TimeoutExpired → ``timeout_err`` (code, message), other errors → ``fail_code``."""
    try:
        return on_result(subprocess.run(cmd, cwd=os.getcwd(), shell=shell, env=env, **_capture_run_kwargs(timeout)))
    except subprocess.TimeoutExpired:
        return _err(rid, *timeout_err)
    except Exception as e:
        return _err(rid, fail_code, str(e))


def _joined_output(r) -> str:
    """stdout + stderr of a CompletedProcess, non-empty parts only, newline-joined and stripped."""
    return "\n".join(p for p in (r.stdout or "", r.stderr or "") if p).strip()


def _toolset_rows(params: dict, *, with_tools: bool) -> list[dict]:
    toolsets = _tools_mod("toolsets")
    session = _sessions.get(params.get("session_id", ""))
    enabled = set((getattr(session["agent"], "enabled_toolsets", []) if session else _load_enabled_toolsets()) or [])
    items = []
    for name in sorted(toolsets.get_all_toolsets().keys()):
        if info := toolsets.get_toolset_info(name):
            row = {
                "name": name, "description": info["description"], "tool_count": info["tool_count"],
                "enabled": name in enabled if enabled else True}
            if with_tools:
                row["tools"] = info["resolved_tools"]
            items.append(row)
    return items


# ─── System / process ────────────────────────────────────────────────────────
@method("system.battery")
def _(rid, params: dict) -> dict:
    """Host battery for the status bar. Always resolves; ``available: false`` = no battery or read failed."""
    try:
        battery = _tools_mod("agent.battery")
        batt = battery.read_battery()
        return _ok(rid, {
            "available": batt.available, "percent": batt.percent, "plugged": batt.plugged,
            "category": battery.battery_category(batt)})
    except Exception:
        return _ok(rid, {"available": False, "percent": None, "plugged": None, "category": "dim"})


# One-expression handlers: name → (fail_code, payload builder(params)).
_SIMPLE_RPCS = {
    # Session-scoped view of the background process registry (desktop status stack).
    "process.stop": (5010, lambda params: {"killed": _tools_mod("tools.process_registry").process_registry.kill_all()}),
    # Re-read ``~/.hermes/.env`` (CLI ``/reload`` parity); built agents keep their pool, ``/new`` resolves fresh.
    "reload.env": (5015, lambda params: {"updated": int(_tools_mod("hermes_cli.config").reload_env())}),
    "plugins.list": (5032, lambda params: {"plugins": [
        {"name": n, "version": getattr(i, "version", "?"), "enabled": getattr(i, "enabled", True)}
        for n, i in _tools_mod("hermes_cli.plugins").get_plugin_manager()._plugins.items()]}),
    "tools.list": (5031, lambda params: {"toolsets": _toolset_rows(params, with_tools=True)}),
    "toolsets.list": (5032, lambda params: {"toolsets": _toolset_rows(params, with_tools=False)}),
    "agents.list": (5033, lambda params: {"processes": [
        {"session_id": p["session_id"], "command": p["command"][:80], "status": p["status"], "uptime": p["uptime_seconds"]}
        for p in _tools_mod("tools.process_registry").process_registry.list_sessions()]}),
}
for _name, (_code, _build) in _SIMPLE_RPCS.items():
    # Look the builder up at call time: bind_module rebinds the table's lambdas onto server globals.
    _rpc(_name, _code)(lambda rid, params, _n=_name: _ok(rid, _SIMPLE_RPCS[_n][1](params)))
del _name, _code, _build
_rpc("process.list", 5010, live_session=True)(
    lambda rid, params, session: _ok(rid, {"processes": _session_processes(session)}))


@_rpc("process.kill", live_session=True, fail_code=5010)
def _(rid, params: dict, session) -> dict:
    """Kill ONE background process, scoped to the caller's session (unlike process.stop's kill_all)."""
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    registry = _tools_mod("tools.process_registry").process_registry
    proc = registry.get(proc_id)
    if proc is None or str(getattr(proc, "session_key", "") or "") != str(session.get("session_key") or ""):
        return _err(rid, 4044, f"no such process: {proc_id}")
    return _ok(rid, registry.kill_process(proc_id))


def _mcp_reload_confirm_required() -> bool:
    """``approvals.mcp_reload_confirm`` from disk config; True (safe) on any failure."""
    try:
        cfg = _tools_mod("hermes_cli.config").load_config()
        approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
        return bool(approvals.get("mcp_reload_confirm", True)) if isinstance(approvals, dict) else True
    except Exception:
        return True


@_rpc("reload.mcp", 5015)
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    # Prompt-cache invalidation gate: without confirm=true honour ``approvals.mcp_reload_confirm``
    # (Ink prints ``message`` and re-invokes with confirm=true, or flips the config).
    if not bool(params.get("confirm", False)) and _mcp_reload_confirm_required():
        message = (
            "⚠️  /reload-mcp invalidates the prompt cache (next message re-sends full input tokens). "
            "Reply `/reload-mcp now` to proceed, or `/reload-mcp always` to proceed and "
            "silence this prompt permanently.")
        return _ok(rid, {"status": "confirm_required", "message": message})
    if session and _session_uses_compute_host(session):
        try:
            ack = _get_compute_host_supervisor().reload_mcp(
                str(params.get("session_id") or ""), request_id=f"reload-mcp-{rid}")
        except Exception as exc:
            return _err(rid, 5019, f"compute-host reload_mcp failed: {exc}")
        return _ok(rid, {"status": "reloaded", "turn_isolation": True, "host_ack": ack})
    _mcp_agent, _mcp_lifecycle, _mcp_discovery = (
        _tools_mod("tools.mcp_tool_agent"), _tools_mod("tools.mcp_tool_lifecycle"), _tools_mod("tools.mcp_tool_discovery"))
    global _mcp_reload_gen, _mcp_reload_loaded_rev
    # Revision the CALLER wants loaded; empty on legacy clients / manual /reload-mcp
    # (generation-only coalescing).
    req_rev = str(params.get("rev") or "")

    def _refresh_session_agent() -> None:
        """Rebuild THIS session's cached tool snapshot + push session.info (the agent never
        re-reads the registry). Runs under _mcp_reload_lock so a concurrent reload can't
        tear the registry down mid-refresh."""
        if not session:
            return
        agent = session["agent"]
        try:  # enabled_override re-resolves toolsets so a server enabled in config this session is picked up
            _mcp_agent.refresh_agent_mcp_tools(agent, enabled_override=_load_enabled_toolsets(), quiet_mode=True)
        except Exception as _exc:
            logger.warning("Failed to refresh cached agent tools after /reload-mcp: %s", _exc)
        _emit("session.info", params.get("session_id", ""), _session_info(agent, session))

    def _do_full_reload() -> None:
        """shutdown+discover+refresh under the lock, then mark a completed generation. Config
        can change WHILE discover connects: re-hash and repeat until stable so the marked
        generation matches what loaded."""
        global _mcp_reload_gen, _mcp_reload_loaded_rev
        loaded = _compute_mcp_rev()
        for _ in range(_MCP_RELOAD_MAX_PASSES):
            _mcp_lifecycle.shutdown_mcp_servers()
            _mcp_agent.reprobe_tool_availability()
            _mcp_discovery.discover_mcp_tools()
            after = _compute_mcp_rev()
            if after == loaded:
                break
            loaded = after
        _refresh_session_agent()
        _mcp_reload_loaded_rev = loaded
        _mcp_reload_gen += 1

    # LEADER (won the non-blocking acquire) runs the full reload. FOLLOWER waits, then — still
    # holding the lock — coalesces only if a reload COMPLETED meanwhile (generation advanced
    # ⇒ leader didn't throw) AND it loaded the requested revision; otherwise it re-runs.
    if _mcp_reload_lock.acquire(blocking=False):
        try:
            _do_full_reload()
        finally:
            _mcp_reload_lock.release()
        return _finish_reload(rid, params, coalesced=False)
    gen_before = _mcp_reload_gen
    with _mcp_reload_lock:
        coalesced = _mcp_reload_gen > gen_before and (not req_rev or req_rev == _mcp_reload_loaded_rev)
        _refresh_session_agent() if coalesced else _do_full_reload()
    return _finish_reload(rid, params, coalesced=coalesced)


# ─── Command catalog / dispatch ──────────────────────────────────────────────
class _Catalog:
    """Accumulator for commands.catalog: ``pairs`` (every [key, desc]), ``canon`` (lowercase
    key/alias → canonical key), ``commands`` (key → desktop meta) and ordered categories."""

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []
        self.canon: dict[str, str] = {}
        self.commands: dict[str, dict[str, str | None]] = {}
        self.cat_map: dict[str, list[list[str]]] = {}  # insertion order = category order

    def add(self, key: str, desc: str, cat: str) -> None:
        self.canon[key.lower()] = key
        self.pairs.append([key, desc])
        self.cat_map.setdefault(cat, []).append([key, desc])


def _catalog_registry(cat: _Catalog) -> None:
    commands = _tools_mod("hermes_cli.commands")
    for cmd in commands.COMMAND_REGISTRY:
        meta = commands.command_desktop_meta(cmd)
        cat.commands.update({f"/{key}": dict(meta) for key in (cmd.name, *cmd.aliases)})
        if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
            continue
        cat.add(f"/{cmd.name}", commands._build_description(cmd), cmd.category)
        for a in cmd.aliases:
            cat.canon[f"/{a}".lower()] = f"/{cmd.name}"
    for name, desc, category in _TUI_EXTRA:
        # Registry command/alias wins over a colliding TUI extra (e.g. /compact, /sessions).
        if name.lower() not in cat.canon:
            cat.add(name, desc, category)


def _catalog_quick_commands(cat: _Catalog) -> None:
    qcmds = _load_cfg().get("quick_commands", {}) or {}
    if not (isinstance(qcmds, dict) and qcmds):
        return
    cat.cat_map.setdefault("User commands", [])  # category exists even when every entry is malformed
    for qname, qc in sorted(qcmds.items()):
        if not isinstance(qc, dict):
            continue
        qtype = qc.get("type", "")
        default_desc = {"exec": f"exec: {qc.get('command', '')}", "alias": f"alias → {qc.get('target', '')}"}
        desc = str(qc.get("description") or default_desc.get(qtype, qtype or "quick command"))
        cat.add(f"/{qname}", _clip(desc), "User commands")


def _catalog_plugin_commands(cat: _Catalog) -> None:
    plugin_cmds = _tools_mod("hermes_cli.plugins").get_plugin_commands() or {}
    if plugin_cmds:
        cat.cat_map.setdefault("Plugin commands", [])
    for pname, info in sorted(plugin_cmds.items()):
        key = f"/{pname}"
        if not isinstance(info, dict) or key.lower() in cat.canon:
            continue
        cat.add(key, _clip(str(info.get("description") or "Plugin command")), "Plugin commands")
        mode = info.get("argument_mode")
        if mode not in {"options", "text", "mixed"}:
            mode = "text" if str(info.get("args_hint") or "").strip() else None
        cat.commands[key] = {"argument_mode": mode, "desktop": None}


def _catalog_skills(cat: _Catalog, skills: dict[str, dict]) -> None:
    """Append skill pairs and fill ``skills`` = ``{key: {usage, origin}}`` (every consumer ranks by them)."""
    usage, origin_of = _skill_usage_lookup()
    for k, info in sorted(_tools_mod("agent.skill_commands").scan_skill_commands().items()):
        cat.pairs.append([k, _clip(str(info.get("description", "Skill")))])
        name = str(info.get("name") or k.lstrip("/"))
        skills[k] = {"usage": usage(name), "origin": origin_of(name)}


@_rpc("commands.catalog", 5020)
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata, categorized, no aliases. Discovery failures land in ``warning``
    (skills' message wins, then quick commands', then plugins')."""
    cat = _Catalog()
    _catalog_registry(cat)
    warning = ""
    try:
        _catalog_quick_commands(cat)
    except Exception as e:
        warning = f"quick_commands discovery unavailable: {e}"
    try:
        _catalog_plugin_commands(cat)
    except Exception as e:
        warning = warning or f"plugin command discovery unavailable: {e}"
    skills: dict[str, dict] = {}
    try:
        _catalog_skills(cat, skills)
    except Exception as e:
        warning = f"skill discovery unavailable: {e}"
    return _ok(rid, {
        "pairs": cat.pairs, "sub": {k: v[:] for k, v in _tools_mod("hermes_cli.commands").SUBCOMMANDS.items()},
        "canon": cat.canon,
        "commands": cat.commands,
        "categories": [{"name": c, "pairs": rows} for c, rows in cat.cat_map.items()],
        "skills": skills, "skill_count": len(skills), "warning": warning})


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})

    # Can drive the agent → needs provider credentials; tier-1 secrets still stripped.
    return _captured_exec(
        rid, [sys.executable, "-m", "hermes_cli.main", *argv], min(int(params.get("timeout", 240)), 600),
        on_result=lambda r: _ok(rid, {
            "blocked": False, "code": r.returncode, "output": (_joined_output(r) or "(no output)")[:48_000]}),
        timeout_err=(5016, "cli.exec: timeout"), fail_code=5017,
        env=hermes_subprocess_env(inherit_credentials=True))


@_rpc("command.resolve", 5012)
def _(rid, params: dict) -> dict:
    r = _tools_mod("hermes_cli.commands").resolve_command(params.get("name", ""))
    if r:
        return _ok(rid, {"canonical": r.name, "description": r.description, "category": r.category})
    return _err(rid, 4011, f"unknown command: {params.get('name')}")


# command.dispatch stages. Each takes (rid, params, session, name, arg) and
# returns a JSON-RPC envelope, or None to fall through to the next stage.


def _dispatch_quick(rid, params, session, name, arg):
    qc = _load_cfg().get("quick_commands", {}).get(name)
    if qc is None:
        return None
    if qc.get("type") == "exec":
        # Sanitized env: the TUI server process holds every API key in os.environ.
        env = _tools_mod("tools.environments.local").build_subprocess_env()
        exec_cmd = qc.get("command", "")
        # Forward user arguments (e.g. `/poly white sox`): split-then-quote per
        # token so boundaries survive (see hermes_cli._subprocess_compat).
        args = (arg or "").strip()
        if args:
            quote_args = _tools_mod("hermes_cli._subprocess_compat").quote_args
            exec_cmd = f"{exec_cmd} {quote_args(args)}"
        r = subprocess.run(exec_cmd, shell=True, env=env, **_capture_run_kwargs(30))
        output = _joined_output(r)[:4000]
        output = _tools_mod("agent.redact").redact_sensitive_text(output) if output else output
        if r.returncode != 0:
            return _err(rid, 4018, output or f"quick command failed with exit code {r.returncode}")
        return _exec_out(rid, output)
    return _ok(rid, {"type": "alias", "target": qc.get("target", "")}) if qc.get("type") == "alias" else None


def _plugin_command_handler(name: str):
    try:
        return _tools_mod("hermes_cli.plugins").get_plugin_command_handler(name)
    except Exception:
        return None


def _run_plugin_command(handler, arg: str) -> str:
    return str(_tools_mod("hermes_cli.plugins").resolve_plugin_command_result(handler(arg)) or "")


def _is_profile_skill_command(session: dict, base: str) -> bool:
    """True when ``/base`` is a skill command of the session's profile (HERMES_HOME bound to it so
    get_skill_commands() sees its skills.external_dirs; nothing upstream binds it). False on failure."""
    try:
        hc = _tools_mod("hermes_constants")
        profile_home = session.get("profile_home")
        token = hc.set_hermes_home_override(profile_home) if profile_home else None
        try:
            return f"/{base}" in _tools_mod("agent.skill_commands").get_skill_commands()
        finally:
            if token is not None:
                hc.reset_hermes_home_override(token)
    except Exception:
        return False


def _dispatch_plugin(rid, params, session, name, arg):
    if handler := _plugin_command_handler(name):
        with contextlib.suppress(Exception):
            return _ok(rid, {"type": "plugin", "output": _run_plugin_command(handler, arg)})
    return None


def _bundle_key_for(name: str):
    """Skill-bundle key for ``name`` when it is NOT a registry command; None otherwise / on failure."""
    try:
        if _tools_mod("hermes_cli.commands").resolve_command(name) is None:
            return _tools_mod("agent.skill_bundles").resolve_bundle_command_key(name)
        return None
    except Exception:
        return None


def _dispatch_bundle(rid, params, session, name, arg):
    bundle_key = _bundle_key_for(name)
    if bundle_key is None:
        return None
    bundles = _tools_mod("agent.skill_bundles")
    try:
        bundle_result = bundles.build_bundle_invocation_message(
            bundle_key, arg, task_id=session.get("session_key", "") if session else "",
            platform=_resolve_session_platform())
    except Exception as exc:
        return _err(rid, 4018, f"bundle dispatch failed: {exc}")
    if not bundle_result:
        return _err(rid, 4018, f"failed to load bundle: {bundle_key}")
    msg, loaded_names, missing = bundle_result
    bundle_name = bundles.get_skill_bundles().get(bundle_key, {}).get("name", bundle_key.lstrip("/"))
    notice = f"⚡ Loading bundle: {bundle_name} ({len(loaded_names)} skills)"
    notice += f"\nSkipped missing skills: {', '.join(missing)}" if missing else ""
    # UIs render `display`, never `message`: the expanded body is model-facing scaffolding.
    return _ok(rid, {"type": "send", "message": msg, "notice": notice, "display": _skill_scaffold_projection(msg)})


def _dispatch_skill(rid, params, session, name, arg):
    with contextlib.suppress(Exception):
        sc = _tools_mod("agent.skill_commands")
        cmds, key = sc.scan_skill_commands(), f"/{name}"
        if key in cmds:
            msg = sc.build_skill_invocation_message(key, arg, task_id=session.get("session_key", "") if session else "")
            if msg:  # UIs render `display`, never `message`.
                return _ok(rid, {
                    "type": "skill", "message": msg, "name": cmds[key].get("name", name),
                    "display": _skill_scaffold_projection(msg)})
    return None


# Built-ins that queue onto _pending_input in the CLI; the TUI slash worker has no
# reader for that queue, so they are handled here and return a structured payload.


def _cmd_queue(rid, params, session, name, arg):
    return _ok(rid, {"type": "send", "message": arg}) if arg else _err(rid, 4004, "usage: /queue <prompt>")


def _prompt_builtin(module: str, fn: str, kw: str = ""):
    """/learn, /plan, /init: submit ``module.fn(arg)`` as a normal turn (the live agent does the work)."""

    def cmd(rid, params, session, name, arg):
        build = getattr(_tools_mod(module), fn)
        return _ok(rid, {"type": "send", "message": build(**{kw: arg}) if kw else build(arg)})
    return cmd


_cmd_learn = _prompt_builtin("agent.learn_prompt", "build_learn_prompt")
_cmd_plan = _prompt_builtin("agent.plan_prompt", "build_plan_prompt")
_cmd_init = _prompt_builtin("hermes_cli.init_command", "build_init_prompt_for_cwd", kw="extra")


def _cmd_moa(rid, params, session, name, arg):
    # One prompt through the default MoA preset, then restore the prior model (whole-session
    # switching goes through the model picker).
    try:
        moa = _tools_mod("hermes_cli.moa_config")
        if not arg:
            return _err(rid, 4004, moa.moa_usage())
        if not session:
            return _err(rid, 4001, "no active session")
        preset = moa.normalize_moa_config(_load_cfg().get("moa") or {})["default_preset"]
        # Record the live identity for post-turn restore, then swap the agent's client in
        # place: session["model_override"] alone never switches an already-built agent.
        agent = session.get("agent")
        # See #53444.
        session["moa_one_shot_restore"] = {
            "override": session.get("model_override"), "model": getattr(agent, "model", None),
            "provider": getattr(agent, "provider", None)}
        if agent is not None:
            try:  # persist_override=False: turn-scoped, never persist the MoA provider to config.yaml
                _apply_model_switch(
                    params.get("session_id", ""), session, f"{preset} --provider moa",
                    confirm_expensive_model=False, pin_session_override=True, persist_override=False)
            except Exception:
                session.pop("moa_one_shot_restore", None)
                raise
        else:  # lazy/fresh session: the override is consumed by the first build
            session["model_override"] = {
                "provider": "moa", "model": preset, "base_url": "moa://local",
                "api_key": "moa-virtual-provider", "api_mode": "chat_completions"}
        notice = f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn."
        return _ok(rid, {"type": "send", "notice": notice, "message": arg})
    except Exception as exc:
        return _err(rid, 5030, f"moa unavailable: {exc}")


def _cmd_focus(rid, params, session, name, arg):
    # Display-only; routed through the config.set branch Ink uses so both surfaces share one state machine.
    fv = _tools_mod("hermes_cli.focus_view")
    display = _load_cfg().get("display")
    display = display if isinstance(display, dict) else {}
    action, target = fv.resolve_focus_arg(arg, cur := bool(display.get("focus_view", False)))
    if action == "usage":
        return _err(rid, 4004, "usage: /focus [on|off|status]")
    if action == "status":
        saved = display.get("focus_saved_tool_progress") or _load_tool_progress_mode()
        return _exec_out(rid, fv.format_focus_status(cur, saved))
    res = _methods["config.set"](
        rid, {"key": "focus", "value": "on" if target else "off", "session_id": params.get("session_id", "")})
    if "error" in res:
        return res
    tool_progress = (res.get("result") or {}).get("tool_progress") or "all"
    return _exec_out(rid, fv.format_focus_toggle_message(bool(target), tool_progress))


def _cmd_retry(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session to retry")
    if busy := _busy_error(rid, session, "retry"):
        return busy
    cc = _tools_mod("agent.context_compressor")
    with session["history_lock"]:
        if busy := _busy_error(rid, session, "retry"):
            return busy
        if session.get("attached_images"):
            return _err(rid, 4018, "retry cannot safely reconstruct or combine attached media")
        history, user_indices, err = _rewind_prelude(rid, session, "retry", "no previous user message to retry")
        if err:
            return err
        _prefix, live_view = cc.history_before_user_originated_turn(history, user_indices[-1])
        try:
            content = cc.retryable_user_text(live_view.get("content"))
        except ValueError as exc:
            return _err(rid, 4018, str(exc))
        rewound, err = _rewind_or_err(
            rid, session, len(user_indices) - 1, (4018, ""), "retry: failed to persist history: ", require_retryable=True)
        if err:
            return err
        content = cc.retryable_user_text(rewound[1].get("content"))
    return _ok(rid, {"type": "send", "message": content})


def _cmd_steer(rid, params, session, name, arg):
    if not arg:
        return _err(rid, 4004, "usage: /steer <prompt>")
    agent = session.get("agent") if session else None
    if agent and hasattr(agent, "steer"):
        with contextlib.suppress(Exception):
            if agent.steer(arg):
                shown = f"{arg[:80]}{'...' if len(arg) > 80 else ''}"
                return _exec_out(rid, f"⏩ Steer queued — arrives after the next tool call: {shown}")
    return _ok(rid, {"type": "send", "message": arg})  # no active run: next-turn message


def _cmd_goal(rid, params, session, name, arg):
    sid_key, goals, err = _session_key_or_err(rid, session, "hermes_cli.goals", "goals")
    if err:
        return err
    try:
        max_turns = int((_load_cfg().get("goals") or {}).get("max_turns", 20) or 20)
    except Exception:
        max_turns = 20
    mgr = goals.GoalManager(session_id=sid_key, default_max_turns=max_turns)
    lower = arg.strip().lower()
    if not lower or lower == "status":
        return _exec_out(rid, mgr.status_line())
    if lower == "pause":
        state = mgr.pause(reason="user-paused")
        return _exec_out(rid, "No goal set." if state is None else f"⏸ Goal paused: {state.goal}")
    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return _exec_out(rid, "No goal to resume.")
        # Resume must restart work: `exec` is display-only, so return a `send`; `display`
        # keeps model-facing scaffolding out of the transcript.
        if not (prompt := mgr.next_continuation_prompt()):
            return _exec_out(rid, f"▶ Goal resumed: {state.goal}")
        notice = f"▶ Goal resumed: {state.goal}\nContinuing now — taking the next step."
        return _ok(rid, {"type": "send", "notice": notice, "message": prompt, "display": "/goal resume"})
    if lower in {"clear", "stop", "done"}:
        had = mgr.has_goal()
        mgr.clear()
        return _exec_out(rid, "✓ Goal cleared." if had else "No active goal.")
    # Remaining text = new goal. Client renders `notice`, submits `message`; the post-turn judge takes over.
    try:
        state = mgr.set(arg)
    except ValueError as exc:
        return _err(rid, 4004, f"invalid goal: {exc}")
    notice = (
        f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
        "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
        "Controls: /goal status · /goal pause · /goal resume · /goal clear")
    return _ok(rid, {"type": "send", "notice": notice, "message": state.goal})


def _cmd_loop(rid, params, session, name, arg):
    sid_key, loops, err = _session_key_or_err(rid, session, "hermes_cli.loops", "loops")
    if err:
        return err
    result = loops.dispatch_loop_command(loops.LoopManager(session_id=sid_key), arg)
    output = result.get("output") or ""
    if result.get("created"):
        with contextlib.suppress(Exception):
            if loops.goal_blocks_loop_tick(sid_key):
                output += ("\nNote: an active /goal is driving this session — loop "
                           "wakeups defer until the goal finishes, pauses, or parks.")
    return _exec_out(rid, output)


def _cmd_undo(rid, params, session, name, arg):
    if not session:
        # /undo [N]: back up N user turns (default 1), soft-delete the truncated rows on disk, and prefill
        # the composer with the text of the user message we backed up to so it can be edited and
        # resubmitted. N=1 is the Claude-Code-style single-step undo; /undo 3 backs up three user turns at
        # once. See issue #21910.
        return _err(rid, 4001, "no active session to undo")
    if busy := _busy_error(rid, session, "undo"):
        return busy
    if not (session_key := session.get("session_key", "")):
        return _err(rid, 4001, "no session key for undo")
    arg_str = (arg or "").strip()
    try:
        n = max(int(arg_str.split()[0]), 1) if arg_str else 1
    except (ValueError, IndexError):
        return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
    with session["history_lock"]:
        _history, user_indices, err = _rewind_prelude(rid, session, "undo", "no user messages to undo")
        if err:
            return err
        turns_undone = min(n, len(user_indices))
        rewound, err = _rewind_or_err(rid, session, len(user_indices) - turns_undone, (4004, "undo: "), "undo: ")
        if err:
            return err
        active, live_view, rewound_count = rewound
        target_text = _tools_mod("agent.message_content").flatten_message_text(live_view.get("content"))
    # Notify memory providers (same hook /branch fires) with rewound=True so cached per-turn state invalidates.
    agent = session.get("agent")
    if agent is not None:
        # See #6672 + #21910.
        mm = getattr(agent, "_memory_manager", None)
        for step in (
            lambda: mm is not None and mm.on_session_switch(
                session_key, parent_session_id="", reset=False, rewound=True),
            lambda: hasattr(agent, "_invalidate_system_prompt") and agent._invalidate_system_prompt(),
            lambda: hasattr(agent, "_last_flushed_db_idx") and setattr(agent, "_last_flushed_db_idx", len(active)),
        ):
            with contextlib.suppress(Exception):
                step()
    turn_word = "turn" if turns_undone == 1 else "turns"
    notice = f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). Edit and resubmit, or send a new message."
    return _ok(rid, {"type": "prefill", "message": target_text, "notice": notice})


def _is_snapshot_restore(arg: str) -> bool:
    return (arg.split(maxsplit=1)[0].lower() if arg else "") in {"restore", "rewind"}


def _cmd_snapshot(rid, params, session, name, arg):
    if not _is_snapshot_restore(arg):
        return None
    return _exec_out(
        rid, "/snapshot restore is blocked in the TUI because it changes config/state on disk "
        "while the live agent has cached settings. Run it in the classic CLI, then restart the TUI.")


def _cmd_compress(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session to compress")
    if busy := _busy_error(rid, session, "compress"):
        return busy
    sid = params.get("session_id", "")
    if _session_uses_compute_host(session):
        status, text = _compute_host_slash(sid, session, "compress", f"/{name}" + (f" {arg}" if arg else ""))
        if status in {"failed", "rejected"}:
            return _err(rid, 5019 if status == "failed" else 4009, text)
        if status == "pending":
            return _ok(rid, {"type": "exec", "status": "pending", "output": text})
        return _exec_out(rid, text)
    try:
        output = _compress_live_with_feedback(sid, session, session["agent"], arg, snapshot_kwargs=True)
        return _exec_out(rid, output)
    except Exception as exc:
        _tools_mod("agent.conversation_compression").finalize_context_engine_compression_notification(
            session["agent"], committed=False)
        return _err(rid, 5009, f"compress failed: {exc}")


_SLASH_BUILTINS = {
    "queue": _cmd_queue, "q": _cmd_queue, "learn": _cmd_learn, "plan": _cmd_plan, "init": _cmd_init,
    "moa": _cmd_moa, "focus": _cmd_focus, "retry": _cmd_retry, "steer": _cmd_steer, "goal": _cmd_goal,
    "loop": _cmd_loop, "undo": _cmd_undo, "snapshot": _cmd_snapshot, "snap": _cmd_snapshot,
    "compress": _cmd_compress, "compact": _cmd_compress}


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = _resolve_name(params.get("name", "").lstrip("/")), params.get("arg", "")
    session = _sessions.get(params.get("session_id", ""))

    # Stage order is load-bearing: quick > plugin > bundle > skill > built-in.
    stages = (_dispatch_quick, _dispatch_plugin, _dispatch_bundle, _dispatch_skill, _SLASH_BUILTINS.get(name))
    for stage in filter(None, stages):
        res = stage(rid, params, session, name, arg)
        if res is not None:
            return res
    return _err(rid, 4018, f"not a quick/plugin/bundle/skill command: {name}")


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")
    # Skill/bundle and _PENDING_INPUT_COMMANDS must NOT reach the slash worker. Plugin
    # commands also bypass it but return normal slash.exec output (TUI keeps the pager path).
    parts = cmd.lstrip("/").split(maxsplit=1)
    base = (parts[0] if parts else "").lower()
    arg = parts[1] if len(parts) > 1 else ""
    sid = params.get("session_id", "")
    live_output = _live_slash_command_output(sid, session, base, arg)
    if live_output is not None:
        return _ok(rid, {"output": live_output or "(no output)"})
    if base in _WORKER_BLOCKED_COMMANDS and _is_snapshot_restore(arg):
        return _err(rid, 4018, "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore")
    # Pending-input built-ins route straight to command.dispatch (some clients fail the
    # error-then-retry fallback); bundles go the same way under their resolved key.
    target = base if base in _PENDING_INPUT_COMMANDS else _bundle_key_for(base)
    if target is not None:
        return _methods["command.dispatch"](rid, {"name": target.lstrip("/"), "arg": arg, "session_id": sid})
    if _is_profile_skill_command(session, base):
        return _err(rid, 4018, f"skill command: use command.dispatch for /{base}")
    if plugin_handler := _plugin_command_handler(base) if base else None:
        try:
            return _ok(rid, {"output": _run_plugin_command(plugin_handler, arg) or "(no output)"})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})
    worker = session.get("slash_worker")
    if not worker:
        # slash.exec runs on the RPC pool: two concurrent commands could both see slash_worker=None
        # and each fork a full MCP-fleet worker (the loser leaks). Serialize first-use spawn.
        with _sessions_lock:
            spawn_lock = session.setdefault("_slash_spawn_lock", threading.Lock())
        with spawn_lock:
            worker = session.get("slash_worker")
            if not worker:
                try:
                    worker = _SlashWorker(
                        session["session_key"], getattr(session.get("agent"), "model", _resolve_model()),
                        profile_home=session.get("profile_home"))
                    _attach_worker(sid, session, worker)
                except Exception as e:
                    return _err(rid, 5030, f"slash worker start failed: {e}")
    try:
        payload = {"output": worker.run(cmd) or "(no output)"}
        if warning := _mirror_slash_side_effects(sid, session, cmd):
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        with contextlib.suppress(Exception):
            worker.close()
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


# ─── Insights / rollback / browser / config ──────────────────────────────────
@_rpc("insights.get", 5017)
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    if (db := _get_db()) is None:
        return _db_unavailable_error(rid, code=5017)
    cutoff = time.time() - days * 86400
    rows = [s for s in db.list_sessions_rich(limit=500, compact_rows=True) if (s.get("started_at") or 0) >= cutoff]
    return _ok(rid, {"days": days, "sessions": len(rows), "messages": sum(s.get("message_count", 0) for s in rows)})


@_rpc("rollback.list", live_session=True, fail_code=5020)
def _(rid, params: dict, session) -> dict:
    def go(mgr, cwd):
        if not mgr.enabled:
            return _ok(rid, {"enabled": False, "checkpoints": []})
        keys = ("hash", "timestamp", "message")
        rows = [{k: c.get(k, "") for k in keys} for c in mgr.list_checkpoints(cwd)]
        return _ok(rid, {"enabled": True, "checkpoints": rows})
    return _with_checkpoints(session, go)


@_rpc("rollback.restore", live_session=True, fail_code=5021)
def _(rid, params: dict, session) -> dict:
    target, file_path = params.get("hash", ""), params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history → rejected mid-turn (prompt.submit
    # would drop the agent's output or clobber it). File-scoped only touches disk.
    if not file_path and session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before full rollback.restore")

    def go(mgr, cwd):
        result = mgr.restore(cwd, _resolve_checkpoint_hash(mgr, cwd, target), file_path=file_path or None)
        if result.get("success") and not file_path:
            removed = 0
            with session["history_lock"]:
                _history, user_indices = _user_turn_indices(session)
                if user_indices:
                    try:
                        removed = _rewind_active_session_history(session, len(user_indices) - 1)[2]
                    except Exception as exc:
                        raise RuntimeError(f"checkpoint restored, but session history rewind failed: {exc}") from exc
            result["history_removed"] = removed
        return result
    return _ok(rid, _with_checkpoints(session, go))


@_rpc("rollback.diff", live_session=True, fail_code=5022)
def _(rid, params: dict, session) -> dict:
    if not (target := params.get("hash", "")):
        return _err(rid, 4014, "hash required")
    r = _with_checkpoints(session, lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)))
    raw = r.get("diff", "")[:4000]
    payload = {"stat": r.get("stat", ""), "diff": raw}
    if rendered := render_diff(raw, session.get("cols", 80)):
        payload["rendered"] = rendered
    return _ok(rid, payload)


@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")
    if action == "status":
        url = _resolve_browser_cdp_url()
        return _ok(rid, {"connected": bool(url), "url": url})
    if action == "disconnect":
        return _browser_disconnect(rid)
    if action == "connect":
        return _browser_connect(rid, params)
    return _err(rid, 4015, f"unknown action: {action}")


@_rpc("config.show", 5030)
def _(rid, params: dict) -> dict:
    cfg = _load_cfg()
    api_key = _tools_mod("agent.secret_scope").get_secret("HERMES_API_KEY", "") or cfg.get("api_key", "")
    masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
    base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")
    sections = [
        {"title": "Model", "rows": [
            ["Model", _resolve_model()], ["Base URL", base_url or "(default)"], ["API Key", masked]]},
        {"title": "Agent", "rows": [
            ["Max Turns", str(_cfg_max_turns(cfg, 500))],
            ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
            ["Verbose", str(cfg.get("verbose", False))]]},
        {"title": "Environment", "rows": [["Working Dir", os.getcwd()], ["Config File", str(_hermes_home / "config.yaml")]]},
    ]
    return _ok(rid, {"sections": sections})


# ─── Tools / toolsets / agents ───────────────────────────────────────────────
@_rpc("tools.show", 5034)
def _(rid, params: dict) -> dict:
    mt = _tools_mod("model_tools")
    session = _sessions.get(params.get("session_id", ""))
    enabled = getattr(session["agent"], "enabled_toolsets", None) if session else _load_enabled_toolsets()
    # Pre-assembly list: /tools must also show tools deferred behind the tool_search bridge (as the CLI).
    tools = mt.get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)
    sections = {}
    for tool in sorted(tools, key=lambda t: t["function"]["name"]):
        name = tool["function"]["name"]
        desc = str(tool["function"].get("description", "") or "").split("\n")[0]
        if ". " in desc:
            desc = desc[: desc.index(". ") + 1]
        sections.setdefault(mt.get_toolset_for_tool(name) or "unknown", []).append({"name": name, "description": desc})
    sections_out = [{"name": n, "tools": rows} for n, rows in sorted(sections.items())]
    return _ok(rid, {"sections": sections_out, "total": len(tools)})


@_rpc("tools.configure", 5035)
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [str(name).strip() for name in params.get("names", []) or [] if str(name).strip()]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")
    hc, tc = _tools_mod("hermes_cli.config"), _tools_mod("hermes_cli.tools_config")
    cfg = hc.load_config()
    valid_toolsets = {ts_key for ts_key, _, _ in tc.CONFIGURABLE_TOOLSETS} | tc._get_plugin_toolset_keys()
    mcp_targets = [name for name in targets if ":" in name]
    unknown = [name for name in targets if ":" not in name and name not in valid_toolsets]
    toolset_targets = [name for name in targets if ":" not in name and name in valid_toolsets]
    if toolset_targets:
        tc._apply_toolset_change(cfg, "cli", toolset_targets, action)
    missing_servers = tc._apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
    hc.save_config(cfg)
    sid = params.get("session_id", "")
    session = _sessions.get(sid)
    info = _reset_session_agent(sid, session) if session else None
    enabled = sorted(tc._get_platform_tools(hc.load_config(), "cli", include_default_mcp_servers=False))
    changed = [
        name for name in targets
        if name not in unknown and (":" not in name or name.split(":", 1)[0] not in missing_servers)]
    return _ok(rid, {
        "changed": changed, "enabled_toolsets": enabled, "info": info,
        "missing_servers": sorted(missing_servers), "reset": bool(session), "unknown": unknown})


# ─── Cron / learning / skills ────────────────────────────────────────────────
@_scoped_rpc("cron.manage", 5023)
def _(rid, params: dict) -> dict:
    """cronjob() keys off HERMES_HOME, so ``profile`` reaches a per-profile cron store."""
    cronjob = _tools_mod("tools.cronjob_tools").cronjob
    action, jid = params.get("action", "list"), params.get("name", "")
    if action == "list":
        # Paused jobs are excluded by default (reads as deletion in a toggle UI) — forward the flag.
        include_disabled = is_truthy_value(params.get("include_disabled", False))
        result = json.loads(cronjob(action="list", include_disabled=include_disabled))
        # ``scoped`` proves the profile scope was honored; older gateways omit it and clients
        # keep the safe [bot:<name>] filter.
        if profile := _str_arg(params, "profile"):
            result["scoped"] = profile
        return _ok(rid, result)
    if action == "add":
        # Optional repeat / continuity / deliver ('bot-chat[:name]'): None keeps each cronjob() default.
        raw = cronjob(
            action="create", name=jid, schedule=params.get("schedule", ""), prompt=params.get("prompt", ""),
            repeat=int(params["repeat"]) if str(params.get("repeat", "")).strip().isdigit() else None,
            continuity=is_truthy_value(params.get("continuity")) if params.get("continuity") is not None else None,
            deliver=_str_arg(params, "deliver") or None)
        return _ok(rid, json.loads(raw))
    if action in {"remove", "pause", "resume"}:
        return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
    return _err(rid, 4016, f"unknown cron action: {action}")


@_rpc("learning.frames", 5000, "learning.frames failed: ")
def _(rid, params: dict) -> dict:
    """Pre-render the ``/journey`` timeline (frames + legend/summary metadata) so Ink walks it locally."""
    try:
        cols, rows, frames = (
            int(params.get(k, d) or d) for k, d in (("cols", 80), ("rows", 24), ("frames", 48)))
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    graph = _tools_mod("agent.learning_graph").build_learning_graph()
    render_frames = _tools_mod("agent.learning_graph_render").render_frames
    return _ok(rid, render_frames(graph, cols=max(20, cols), rows=max(10, rows), frames=frames))


def _learning_mutation(fn_name: str, arg_keys: tuple):
    """learning.* body: ``agent.learning_mutations.<fn_name>(*str(params[k]) for k in arg_keys)``."""

    def body(rid, params: dict) -> dict:
        fn = getattr(_tools_mod("agent.learning_mutations"), fn_name)
        return _ok(rid, fn(*(str(params.get(k, "")) for k in arg_keys)))
    return body


# detail → node content for an edit prefill; delete → skills archived (restorable), memories
# removed; edit → rewrite a node's content (SKILL.md or memory chunk).
for _name, _fn, _keys in (
    ("detail", "node_detail", ("id",)), ("delete", "delete_node", ("id",)), ("edit", "edit_node", ("id", "content")),
):
    _rpc(f"learning.{_name}", 5000, f"learning.{_name} failed: ")(_learning_mutation(_fn, _keys))
del _name, _fn, _keys


def _skills_search(rid, params, query):
    search, gh = _tools_mod("tools.skills_hub_search"), _tools_mod("tools.skills_hub_github")
    raw = search.unified_search(query, search.create_source_router(gh.GitHubAuth()), source_filter="all", limit=20) or []
    return _ok(rid, {"results": [{"name": r.name, "description": r.description} for r in raw]})


def _skills_install(rid, params, query):
    quiet = _tools_mod("types").SimpleNamespace(print=lambda *a, **k: None)
    _tools_mod("hermes_cli.skills_hub").do_install(query, skip_confirm=True, console=quiet)
    return _ok(rid, {"installed": True, "name": query})


def _skills_browse(rid, params, query):
    pg = int(params.get("page", 0) or 0) or (int(query) if query.isdigit() else 1)
    browse = _tools_mod("hermes_cli.skills_hub").browse_skills
    return _ok(rid, browse(page=pg, page_size=int(params.get("page_size", 20))))


_SKILLS_ACTIONS = {
    "list": lambda rid, params, query: _ok(rid, {"skills": _tools_mod("hermes_cli.banner").get_available_skills()}),
    "search": _skills_search, "install": _skills_install, "browse": _skills_browse,
    "inspect": lambda rid, params, query: _ok(
        rid, {"info": _tools_mod("hermes_cli.skills_hub").inspect_skill(query) or {}})}


def _run_action(rid, params: dict, table: dict, label: str, *extra) -> dict:
    """Dispatch ``params['action']`` (default ``list``) through ``table``; unknown → 4017."""
    action = params.get("action", "list")
    handler = table.get(action)
    if handler is None:
        return _err(rid, 4017, f"unknown {label} action: {action}")
    return handler(rid, params, *extra)


@_scoped_rpc("skills.manage")
def _(rid, params: dict) -> dict:
    """list/install use the scoped profile's skills dir; search/browse/inspect hit the shared hub."""
    return _run_action(rid, params, _SKILLS_ACTIONS, "skills", params.get("query", ""))


@_rpc("skills.reload", 5025)
def _(rid, params: dict) -> dict:
    result = _tools_mod("agent.skill_commands").reload_skills()
    added, removed = result.get("added") or [], result.get("removed") or []
    lines = ["Reloading skills..."] + ([] if added or removed else ["No new skills detected."])
    for label, items in (("Added skills:", added), ("Removed skills:", removed)):
        if items:
            lines.append(label)
            lines.extend(f"  - {item.get('name', '')}" for item in items)
    lines.append(f"{int(result.get('total') or 0)} skill(s) available")
    return _ok(rid, {"output": "\n".join(lines), "result": result})


# ─── MCP catalog + per-profile server lifecycle (mcp.servers.*) ─────────────
# Gateway mirrors of the dashboard REST surface (hermes_cli/web_routers/mcp.py) so a
# desktop plugin can manage MCP servers for ANY profile. Persistence: hermes_cli/mcp_config.py.
@_scoped_rpc("mcp.catalog")
def _(rid, params: dict) -> dict:
    """``{servers: [{name, description, installed, enabled, requires: [env keys], transport}]}`` per profile."""
    mcp_catalog = _tools_mod("hermes_cli.mcp_catalog")
    out = []
    for entry in mcp_catalog.list_catalog():
        try:
            requires = [str(k) for k in (getattr(entry, "env_keys", None) or [])]
        except Exception:
            requires = []
        transport = getattr(entry, "transport", None)  # TransportSpec → its kind string
        out.append({
            "name": entry.name, "description": getattr(entry, "description", "") or "",
            "installed": bool(mcp_catalog.is_installed(entry.name)),
            "enabled": bool(mcp_catalog.is_enabled(entry.name)), "requires": requires,
            "transport": str(getattr(transport, "kind", "") or transport or "stdio")})
    return _ok(rid, {"servers": out})


@_mcp_rpc("list", required=())
def _(rid, params: dict) -> dict:
    """``{servers: [{name, transport, url, command, args, env (key names), auth, oauth_tokens_present,
    enabled, tools}]}``"""
    servers = _tools_mod("hermes_cli.mcp_config")._get_mcp_servers()
    return _ok(rid, {"servers": [_mcp_summarize_server(name, cfg) for name, cfg in sorted(servers.items())]})


@_mcp_rpc("add")
def _(rid, params: dict) -> dict:
    """Add ``name`` from ``preset`` (catalog id) and/or ``config`` (url/command/args/env/headers/auth/
    tools); ``bearer_token`` goes to the profile's .env (only the header template persists). Dup → 4090."""
    mc = _tools_mod("hermes_cli.mcp_config")
    name, preset = _str_arg(params, "name"), _str_arg(params, "preset")
    if name in mc._get_mcp_servers():
        return _err(rid, 4090, f"server '{name}' already exists")
    raw_cfg = params.get("config")
    server_config: dict = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if preset:  # fills url/command/args when omitted; mutates server_config in place
        mc._apply_mcp_preset(
            name, preset_name=preset, url=server_config.get("url"), command=server_config.get("command"),
            cmd_args=list(server_config.get("args") or []), server_config=server_config)
    if not server_config.get("url") and not server_config.get("command"):
        return _err(rid, 4063, "config must specify a 'url' (http) or 'command' (stdio), or a valid 'preset'")
    if bearer_token := params.get("bearer_token"):
        server_config["headers"] = mc._save_bearer_auth_token(name, str(bearer_token))
    if not mc._save_mcp_server(name, server_config):
        return _err(rid, 4001, f"server '{name}' rejected: suspicious command/args configuration")
    saved = mc._get_mcp_servers().get(name, server_config)
    return _ok(rid, {"ok": True, "name": name, "server": _mcp_summarize_server(name, saved)})


@_mcp_rpc("set_api_key", (*_NAME, ("value", _nonempty)))
def _(rid, params: dict) -> dict:
    """Secret → profile .env under ``env_var`` (default ``MCP_<NAME>_API_KEY``); config.yaml gets only
    a ``${ENV}`` reference (Bearer header for http, ``env`` entry for stdio)."""
    hc, mc = _tools_mod("hermes_cli.config"), _tools_mod("hermes_cli.mcp_config")
    name, servers, err = _mcp_named_server(rid, params)
    if err:
        return err
    value = params.get("value")
    env_var = _str_arg(params, "env_var") or mc._env_key_for_server(name)
    entry = servers[name]
    if not isinstance(entry, dict):
        return _err(rid, 4001, "malformed server config")
    if entry.get("url"):
        normalized = mc._strip_bearer_prefix(str(value))
        if not normalized or normalized.lower() == "bearer":
            return _err(rid, 4063, "value is not a valid credential")
        hc.save_env_value(env_var, normalized)
        is_default = env_var == mc._env_key_for_server(name)
        entry["headers"] = (
            mc._bearer_auth_headers(name) if is_default else {"Authorization": f"Bearer ${{{env_var}}}"})
    else:
        hc.save_env_value(env_var, str(value))
        env_block = entry.get("env")
        entry["env"] = env_block if isinstance(env_block, dict) else {}
        entry["env"][env_var] = f"${{{env_var}}}"
    cfg = hc.load_config()
    cfg.setdefault("mcp_servers", {})[name] = entry
    hc.save_config(cfg)
    return _ok(rid, {"ok": True, "name": name, "env_var": env_var, "server": _mcp_summarize_server(name, entry)})


@_mcp_rpc("test")
def _(rid, params: dict) -> dict:
    """Connect, list tools, disconnect → ``{ok, tools, prompts, resources, oauth_needed,
    oauth_tokens_present}`` (``{ok: false, error, tools: []...}`` on failure). RPC pool: cold npx blocks."""
    mc = _tools_mod("hermes_cli.mcp_config")
    name, servers, err = _mcp_named_server(rid, params)
    if err:
        return err
    cfg = servers[name]
    # An `auth: oauth` server serving tools/list anonymously would probe OK with no
    # token — a false green. Require a token on disk for it.
    needs_oauth_token = cfg.get("auth") == "oauth"
    details: dict = {}

    def failure(error: str, oauth_needed: bool, tokens_present) -> dict:
        return _ok(rid, {"ok": False, "error": error, "tools": [], "oauth_needed": oauth_needed,
                         "oauth_tokens_present": tokens_present})
    try:
        tools = mc._probe_single_server(name, cfg, details=details)
        token_present = mc._oauth_tokens_present(name) if needs_oauth_token else True
    except Exception as exc:
        return failure(str(exc), needs_oauth_token, mc._oauth_tokens_present(name) if needs_oauth_token else None)
    if not token_present:
        return failure("OAuth authentication required — no token found.", True, False)
    return _ok(rid, {
        "ok": True, "tools": [{"name": t, "description": d} for t, d in tools],
        "prompts": details.get("prompts", 0), "resources": details.get("resources", 0),
        "oauth_needed": needs_oauth_token, "oauth_tokens_present": True if needs_oauth_token else None})


@_mcp_rpc("remove")
def _(rid, params: dict) -> dict:
    """Remove a server from the profile's config.yaml → ``{ok: true, removed: true}``."""
    name = _str_arg(params, "name")
    if not _tools_mod("hermes_cli.mcp_config")._remove_mcp_server(name):
        return _err(rid, 4064, f"server '{name}' not found")
    return _ok(rid, {"ok": True, "removed": True})


@_mcp_rpc("oauth.start")
def _(rid, params: dict) -> dict:
    """Begin a session-backed OAuth flow → ``{ok, session_id, auth_url, flow: "pkce"}``; the client
    opens ``auth_url`` and polls ``mcp.servers.oauth.poll``. With ``client_redirect_uri`` the CLIENT
    hosts the loopback and relays the code via ``mcp.servers.oauth.callback`` (desktop and gateway
    on different machines). Runs on the RPC pool (_LONG_HANDLERS)."""
    client_redirect_uri = _str_arg(params, "client_redirect_uri") or None
    try:
        name, servers, err = _mcp_named_server(rid, params)
        if err:
            return err
        cfg = dict(servers[name])
        if not cfg.get("url"):
            return _err(rid, 4001, "stdio servers authenticate via env keys, not OAuth")
        if cfg.get("headers") and cfg.get("auth") != "oauth":
            return _err(rid, 4001, "this server uses header/API-key auth, not OAuth")
        cfg["auth"] = "oauth"
        hermes_home = str(_tools_mod("hermes_constants").get_hermes_home().expanduser().resolve(strict=False))
        result = _tools_mod("tui_gateway.mcp_oauth_sessions").start_flow(
            hermes_home, name, cfg, client_redirect_uri=client_redirect_uri)
    except ValueError as e:
        return _err(rid, 4001, str(e))
    return _ok(rid, {"ok": True, **{k: result[k] for k in ("session_id", "auth_url", "flow")}})


@_mcp_rpc("oauth.poll", _NAME_SESSION)
def _(rid, params: dict) -> dict:
    """Poll a flow → ``{ok, status: pending|approved|error, ...}``; ``approved`` persists tokens per profile."""
    poll = _tools_mod("tui_gateway.mcp_oauth_sessions").poll_flow
    return _ok(rid, {"ok": True, **poll(_str_arg(params, "session_id"), _str_arg(params, "name"))})


@_mcp_rpc("oauth.callback", _NAME_SESSION)
def _(rid, params: dict) -> dict:
    """Relay a client-captured redirect (``code``/``state``/``error``) into a ``client_redirect_uri`` flow."""
    code, state, error = (str(params.get(k) or "") or None for k in ("code", "state", "error"))
    deliver = _tools_mod("tui_gateway.mcp_oauth_sessions").deliver_callback_flow
    return _ok(rid, deliver(
        _str_arg(params, "session_id"), _str_arg(params, "name"), code=code, state=state, error=error))


# ─── Plugins ─────────────────────────────────────────────────────────────────
def _plugin_rows() -> list[dict]:
    pc = _tools_mod("hermes_cli.plugins_cmd")
    enabled, disabled = pc._get_enabled_set(), pc._get_disabled_set()
    out = []
    for name, version, desc, source, _dir, key in sorted(pc._discover_all_plugins()):
        status = pc._plugin_status(name, enabled, disabled, key=key)
        # Bundled backends/platforms/providers run without an explicit enable: report the
        # truthful default instead of "not enabled" (reads as OFF).
        if status == "not enabled" and source == "bundled" and pc._bundled_default_on(_dir):
            status = "enabled"
        # key = canonical registry key (names collide across category dirs); portable = Agent Plugins v1.
        out.append({
            "name": name, "key": key, "version": str(version or ""), "description": desc or "",
            "source": source, "status": status, "portable": pc._is_portable_plugin_dir(_dir)})
    return out


def _plugins_list(rid, params):
    rows = _plugin_rows()
    user_count = sum(1 for r in rows if r["source"] != "bundled")
    return _ok(rid, {"plugins": rows, "user_count": user_count, "bundled_count": len(rows) - user_count})


def _plugins_toggle(rid, params):
    # Prefer the canonical key — bare names are ambiguous across categories.
    ident = (params.get("key") or params.get("name") or "").strip()
    if not ident:
        return _err(rid, 4019, "plugins.toggle requires a 'key' or 'name'")
    toggle = _tools_mod("hermes_cli.plugins_cmd").dashboard_set_agent_plugin_enabled
    result = toggle(ident, enabled=bool(params.get("enable")))
    if not result.get("ok"):
        return _err(rid, 5026, result.get("error") or "toggle failed")
    row = next((r for r in _plugin_rows() if ident in (r["key"], r["name"])), None)
    return _ok(rid, {"ok": True, "unchanged": bool(result.get("unchanged")), "name": ident, "plugin": row})


def _plugins_install(rid, params):
    ident = (params.get("identifier") or params.get("repo") or "").strip()
    if not ident:
        return _err(rid, 4019, "plugins.install requires 'identifier' or 'repo'")
    result = _tools_mod("hermes_cli.plugins_cmd").dashboard_install_plugin(
        ident, force=bool(params.get("force")), enable=params.get("enable", True))
    return _ok(rid, result) if result.get("ok") else _err(rid, 5026, result.get("error") or "install failed")


_PLUGINS_ACTIONS = {"list": _plugins_list, "toggle": _plugins_toggle, "install": _plugins_install}


@_scoped_rpc("plugins.manage", 5026, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """TUI Plugins Hub backend (shares primitives with ``hermes plugins`` / the dashboard):
    ``list`` → {plugins, user_count, bundled_count}; ``toggle`` flips ``key``/``name`` per ``enable``;
    ``install`` git-clones ``identifier``/``repo`` (``force``, ``enable`` default True)."""
    return _run_action(rid, params, _PLUGINS_ACTIONS, "plugins")


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        approval = _tools_mod("tools.approval_detection")
        is_hardline, hardline_desc = approval.detect_hardline_command(cmd)
        if is_hardline:
            return _err(rid, 4005, f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands.")
        is_dangerous, _, desc = approval.detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands.")
    except ImportError:
        return _err(rid, 5001, "shell.exec unavailable: approval safety module not importable")
    return _captured_exec(
        rid, cmd, 30, shell=True, fail_code=5003, timeout_err=(5002, "command timed out (30s)"),
        on_result=lambda r: _ok(rid, {"stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:], "code": r.returncode}))


def register(server) -> None:
    """Rebind this module's helpers + handlers onto ``server`` and register the handlers."""
    bind_module(globals(), server, skip=("_",))
