"""Tools & system / slash.exec / insights / rollback / browser-plugins-cron-skills JSON-RPC handlers (moved verbatim from server.py).

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("system.battery")
def _(rid, params: dict) -> dict:
    """Return the host battery status for the status-bar read-out.

    Always resolves with a payload; ``available: false`` means there is no
    battery (desktop/server/VM) or the read failed. The TUI only polls this
    while the battery indicator is enabled.
    """
    try:
        from agent.battery import battery_category, read_battery

        batt = read_battery()
        return _ok(
            rid,
            {
                "available": batt.available,
                "percent": batt.percent,
                "plugged": batt.plugged,
                "category": battery_category(batt),
            },
        )
    except Exception:
        return _ok(rid, {"available": False, "percent": None, "plugged": None, "category": "dim"})


@method("process.stop")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("process.list")
def _(rid, params: dict) -> dict:
    """Session-scoped view of the background process registry (desktop status stack)."""
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        return _ok(rid, {"processes": _session_processes(session)})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("process.kill")
def _(rid, params: dict) -> dict:
    """Kill ONE background process — scoped to the caller's session so one
    window can't reap another session's work (unlike process.stop's kill_all)."""
    session, err = _sess(params, rid)
    if err:
        return err
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    try:
        from tools.process_registry import process_registry

        proc = process_registry.get(proc_id)
        if proc is None or str(getattr(proc, "session_key", "") or "") != str(
            session.get("session_key") or ""
        ):
            return _err(rid, 4044, f"no such process: {proc_id}")
        return _ok(rid, process_registry.kill_process(proc_id))
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    try:
        # Gate: /reload-mcp invalidates the prompt cache for this session.
        # Respect the ``approvals.mcp_reload_confirm`` config toggle — if
        # set (default true) AND the caller did not pass ``confirm=true``
        # in params, surface a warning to the transcript instead of just
        # reloading silently.  Users pass confirm=true either by
        # re-invoking after reading the warning, or by setting the
        # config key to false permanently.
        user_confirm = bool(params.get("confirm", False))
        if not user_confirm:
            try:
                from hermes_cli.config import load_config as _load_config

                _cfg = _load_config()
                _approvals = _cfg.get("approvals") if isinstance(_cfg, dict) else None
                _confirm_required = True
                if isinstance(_approvals, dict):
                    _confirm_required = bool(_approvals.get("mcp_reload_confirm", True))
            except Exception:
                _confirm_required = True
            if _confirm_required:
                # Return a structured response the Ink client can surface
                # as a warning/confirmation without actually reloading yet.
                # Ink's ops.ts reads ``status`` and prints ``message`` to
                # the transcript; a follow-up invocation with confirm=true
                # (or an `always` choice that flips the config) proceeds.
                return _ok(
                    rid,
                    {
                        "status": "confirm_required",
                        "message": (
                            "⚠️  /reload-mcp invalidates the prompt cache (next "
                            "message re-sends full input tokens). Reply `/reload-mcp "
                            "now` to proceed, or `/reload-mcp always` to proceed and "
                            "silence this prompt permanently."
                        ),
                    },
                )

        if session and _session_uses_compute_host(session):
            try:
                ack = _get_compute_host_supervisor().reload_mcp(
                    str(params.get("session_id") or ""),
                    request_id=f"reload-mcp-{rid}",
                )
            except Exception as exc:
                return _err(rid, 5019, f"compute-host reload_mcp failed: {exc}")
            return _ok(rid, {"status": "reloaded", "turn_isolation": True, "host_ack": ack})

        from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools

        def _refresh_session_agent() -> None:
            """Rebuild THIS session's cached tool snapshot from the live
            registry and push session.info. The agent snapshots tools once at
            build and never re-reads the registry, so an explicit rebuild is
            required (mirrors gateway/run.py::_execute_mcp_reload). Runs under
            _mcp_reload_lock so the registry it reads can't be torn down by a
            concurrent reload mid-refresh."""
            if not session:
                return
            agent = session["agent"]
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                # Explicit reload: re-resolve enabled toolsets so a server the
                # user just enabled in config this session is picked up.
                refresh_agent_mcp_tools(
                    agent,
                    enabled_override=_load_enabled_toolsets(),
                    quiet_mode=True,
                )
            except Exception as _exc:
                logger.warning(
                    "Failed to refresh cached agent tools after /reload-mcp: %s",
                    _exc,
                )
            _emit("session.info", params.get("session_id", ""), _session_info(agent, session))

        global _mcp_reload_gen, _mcp_reload_loaded_rev

        # The revision the CALLER is asking to load (the mcp_rev its poll
        # observed). Empty on legacy clients and manual /reload-mcp — those
        # coalesce on generation alone, as before.
        req_rev = str(params.get("rev") or "")

        def _do_full_reload() -> None:
            """shutdown+discover+refresh under the lock, then mark a completed
            generation. The lock spans the refresh too: releasing after
            discover would let a second reload tear the registry down while
            this one is still reading it to rebuild the session snapshot.

            Config can change WHILE discover is connecting servers (a slow
            reload racing a config edit): re-hash after discovery and repeat
            until the hash is stable, so the generation we mark completed
            always reflects the config that was actually loaded."""
            global _mcp_reload_gen, _mcp_reload_loaded_rev

            loaded = _compute_mcp_rev()
            for _ in range(_MCP_RELOAD_MAX_PASSES):
                shutdown_mcp_servers()
                discover_mcp_tools()
                after = _compute_mcp_rev()
                if after == loaded:
                    break
                loaded = after

            _refresh_session_agent()
            _mcp_reload_loaded_rev = loaded
            _mcp_reload_gen += 1

        # Serialize reloads. The LEADER (won the non-blocking acquire) runs the
        # full reload. A FOLLOWER (lock busy) snapshots the generation, waits,
        # then — still holding the lock — checks whether a reload that
        # actually COMPLETED while it waited satisfies ITS request: the
        # generation must have advanced (leader didn't throw) AND the loaded
        # revision must match the one this follower was asked to apply. Both
        # true → just refresh its own agent against the fresh registry
        # (coalesced). Leader threw, or leader loaded an older revision than
        # this request observed → re-run the full reload, so a failed or
        # stale leader can never leave a follower acking a revision that was
        # never loaded.
        if _mcp_reload_lock.acquire(blocking=False):
            try:
                _do_full_reload()
            finally:
                _mcp_reload_lock.release()

            return _finish_reload(rid, params, coalesced=False)

        gen_before = _mcp_reload_gen

        with _mcp_reload_lock:
            leader_completed = _mcp_reload_gen > gen_before
            rev_satisfied = not req_rev or req_rev == _mcp_reload_loaded_rev

            if leader_completed and rev_satisfied:
                _refresh_session_agent()
                coalesced = True
            else:
                _do_full_reload()
                coalesced = False

        return _finish_reload(rid, params, coalesced=coalesced)
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("reload.env")
def _(rid, params: dict) -> dict:
    """Re-read ``~/.hermes/.env`` into the gateway process via
    ``hermes_cli.config.reload_env``, matching classic CLI's ``/reload``
    handler.  Newly added API keys take effect on the next agent call
    without restarting the TUI.

    The credential pool / provider routing for any *already-constructed*
    agent does not auto-rebuild — that's the same behaviour as classic
    CLI's ``/reload``.  Users who want a brand-new credential resolution
    should follow with ``/new``.
    """
    try:
        from hermes_cli.config import reload_env

        count = reload_env()
        return _ok(rid, {"updated": int(count)})
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    try:
        from hermes_cli.commands import (
            COMMAND_REGISTRY,
            SUBCOMMANDS,
            _build_description,
        )

        all_pairs: list[list[str]] = []
        canon: dict[str, str] = {}
        categories: list[dict] = []
        cat_map: dict[str, list[list[str]]] = {}
        cat_order: list[str] = []

        for cmd in COMMAND_REGISTRY:
            if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
                continue

            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            desc = _build_description(cmd)
            all_pairs.append([c, desc])

            cat = cmd.category
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([c, desc])

        for name, desc, cat in _TUI_EXTRA:
            # Dedup guard: skip TUI extras that collide with a registry
            # command or one of its aliases (e.g. the historical /compact
            # collision, #57133, or /sessions which the registry also
            # advertises). The registry entry is canonical.
            if name.lower() in canon:
                continue
            canon[name.lower()] = name
            all_pairs.append([name, desc])
            if cat not in cat_map:
                cat_map[cat] = []
                cat_order.append(cat)
            cat_map[cat].append([name, desc])

        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                bucket = "User commands"
                if bucket not in cat_map:
                    cat_map[bucket] = []
                    cat_order.append(bucket)
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = str(qc.get("description") or default_desc)
                    qdesc = qdesc[:120] + ("…" if len(qdesc) > 120 else "")
                    all_pairs.append([key, qdesc])
                    cat_map[bucket].append([key, qdesc])
        except Exception as e:
            if not warning:
                warning = f"quick_commands discovery unavailable: {e}"

        skill_count = 0
        skills: dict[str, dict] = {}
        try:
            from agent.skill_commands import scan_skill_commands

            # Usage + origin per skill command. Surfaces here rather than in a
            # second RPC because every consumer that renders the catalog also
            # wants to rank it, and both reads are cheap sidecar files already
            # loaded once per catalog build.
            usage, origin_of = _skill_usage_lookup()

            for k, info in sorted(scan_skill_commands().items()):
                d = str(info.get("description", "Skill"))
                all_pairs.append([k, d[:120] + ("…" if len(d) > 120 else "")])
                name = str(info.get("name") or k.lstrip("/"))
                skills[k] = {"usage": usage(name), "origin": origin_of(name)}
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        for cat in cat_order:
            categories.append({"name": cat, "pairs": cat_map[cat]})

        sub = {k: v[:] for k, v in SUBCOMMANDS.items()}
        return _ok(
            rid,
            {
                "pairs": all_pairs,
                "sub": sub,
                "canon": canon,
                "categories": categories,
                "skills": skills,
                "skill_count": skill_count,
                "warning": warning,
            },
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        # CREATE_NO_WINDOW on Windows — under the desktop GUI's windowless
        # parent, this spawn otherwise flashes a console (#56747).
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            # Force UTF-8 + lossy decode so non-UTF-8 child output can't crash
            # the gateway thread on locale-mismatched Windows. See #53137.
            encoding="utf-8",
            errors="replace",
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            # cli.exec runs `python -m hermes_cli.main` (can drive the agent) →
            # needs provider credentials. Tier-1 secrets still stripped (#29157).
            env=hermes_subprocess_env(inherit_credentials=True),
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(
            rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]}
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(
                rid,
                {
                    "canonical": r.name,
                    "description": r.description,
                    "category": r.category,
                },
            )
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    resolved = _resolve_name(name)
    if resolved != name:
        name = resolved
    session = _sessions.get(params.get("session_id", ""))

    qcmds = _load_cfg().get("quick_commands", {})
    if name in qcmds:
        qc = qcmds[name]
        if qc.get("type") == "exec":
            # Sanitize env to prevent credential leakage —
            # quick commands run in the TUI server process which
            # has all API keys in os.environ.
            from tools.environments.local import build_subprocess_env
            sanitized_env = build_subprocess_env()
            from hermes_cli._subprocess_compat import windows_hide_flags

            r = subprocess.run(
                qc.get("command", ""),
                shell=True,
                capture_output=True,
                text=True,
                # Force UTF-8 + lossy decode so non-UTF-8 child output can't
                # crash the gateway thread on locale-mismatched Windows (#53137).
                encoding="utf-8", errors="replace",
                timeout=30,
                stdin=subprocess.DEVNULL,
                env=sanitized_env,
                creationflags=windows_hide_flags(),
            )
            output = (
                (r.stdout or "")
                + ("\n" if r.stdout and r.stderr else "")
                + (r.stderr or "")
            ).strip()[:4000]
            if output:
                from agent.redact import redact_sensitive_text
                output = redact_sensitive_text(output)
            if r.returncode != 0:
                return _err(
                    rid,
                    4018,
                    output or f"quick command failed with exit code {r.returncode}",
                )
            return _ok(rid, {"type": "exec", "output": output})
        if qc.get("type") == "alias":
            return _ok(rid, {"type": "alias", "target": qc.get("target", "")})

    try:
        from hermes_cli.plugins import (
            get_plugin_command_handler,
            resolve_plugin_command_result,
        )

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except Exception:
        pass

    try:
        from agent.skill_bundles import (
            build_bundle_invocation_message,
            get_skill_bundles,
            resolve_bundle_command_key,
        )

        from hermes_cli.commands import resolve_command

        bundle_key = (
            resolve_bundle_command_key(name)
            if resolve_command(name) is None
            else None
        )
    except Exception:
        bundle_key = None

    if bundle_key is not None:
        try:
            bundle_result = build_bundle_invocation_message(
                bundle_key,
                arg,
                task_id=session.get("session_key", "") if session else "",
                platform=_resolve_session_platform(),
            )
        except Exception as exc:
            return _err(rid, 4018, f"bundle dispatch failed: {exc}")

        if not bundle_result:
            return _err(rid, 4018, f"failed to load bundle: {bundle_key}")

        msg, loaded_names, missing = bundle_result
        bundle_info = get_skill_bundles().get(bundle_key, {})
        bundle_name = bundle_info.get("name", bundle_key.lstrip("/"))
        notice = f"⚡ Loading bundle: {bundle_name} ({len(loaded_names)} skills)"
        if missing:
            notice += f"\nSkipped missing skills: {', '.join(missing)}"
        return _ok(
            rid,
            {
                "type": "send",
                "message": msg,
                "notice": notice,
                # UIs render this, never `message` — the expanded bundle body
                # is model-facing scaffolding (see _skill_scaffold_projection).
                "display": _skill_scaffold_projection(msg),
            },
        )

    try:
        from agent.skill_commands import (
            scan_skill_commands,
            build_skill_invocation_message,
        )

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(
                key, arg, task_id=session.get("session_key", "") if session else ""
            )
            if msg:
                return _ok(
                    rid,
                    {
                        "type": "skill",
                        "message": msg,
                        "name": cmds[key].get("name", name),
                        # UIs render this, never `message` — the expanded skill
                        # body is model-facing scaffolding.
                        "display": _skill_scaffold_projection(msg),
                    },
                )
    except Exception:
        pass

    # ── Commands that queue messages onto _pending_input in the CLI ───
    # In the TUI the slash worker subprocess has no reader for that queue,
    # so we handle them here and return a structured payload.

    if name in {"queue", "q"}:
        if not arg:
            return _err(rid, 4004, "usage: /queue <prompt>")
        return _ok(rid, {"type": "send", "message": arg})

    if name == "learn":
        # Open-ended: build the standards-guided prompt and submit it as a
        # normal agent turn. The live agent gathers whatever the user
        # described (dirs, URLs, this conversation, pasted text) with its own
        # tools and authors the skill via skill_manage. Works on any backend.
        from agent.learn_prompt import build_learn_prompt

        return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})
    if name == "init":
        # Generate-or-update AGENTS.md: build the guidance-laden prompt and
        # submit it as a normal agent turn (same pattern as /learn). The live
        # agent scans the project with its own read-only tools and writes or
        # merge-updates AGENTS.md via write_file. Works on any backend.
        from hermes_cli.init_command import build_init_prompt_for_cwd

        return _ok(rid, {"type": "send", "message": build_init_prompt_for_cwd(extra=arg)})
    if name == "moa":
        # /moa is one-shot sugar only: run a single prompt through the default
        # MoA preset, then restore the prior model. To *switch* to a MoA preset
        # for the rest of the session, pick it from the model picker (MoA
        # presets surface as a virtual "Mixture of Agents" provider).
        try:
            from hermes_cli.moa_config import moa_usage, normalize_moa_config

            if not arg:
                return _err(rid, 4004, moa_usage())
            if not session:
                return _err(rid, 4001, "no active session")
            sid = params.get("session_id", "")
            moa_cfg = normalize_moa_config(_load_cfg().get("moa") or {})
            preset = moa_cfg["default_preset"]
            # Record the live model identity so it can be restored after the
            # one-shot turn, then swap the agent's client in place (#53444:
            # setting session["model_override"] alone never switched the
            # already-built agent, so the turn silently ran on the old model).
            agent = session.get("agent")
            session["moa_one_shot_restore"] = {
                "override": session.get("model_override"),
                "model": getattr(agent, "model", None) if agent else None,
                "provider": getattr(agent, "provider", None) if agent else None,
            }
            if agent is not None:
                # Live agent: swap its client in place so THIS turn runs MoA.
                try:
                    _apply_model_switch(
                        sid,
                        session,
                        f"{preset} --provider moa",
                        confirm_expensive_model=False,
                        pin_session_override=True,
                        # One-shot turn-scoped swap — never persist the MoA
                        # virtual provider to config.yaml.
                        persist_override=False,
                    )
                except Exception as exc:
                    session.pop("moa_one_shot_restore", None)
                    return _err(rid, 5030, f"moa unavailable: {exc}")
            else:
                # No agent built yet (lazy/fresh session): the override is
                # consumed by the first build, so the turn runs MoA without an
                # in-place switch.
                session["model_override"] = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
            return _ok(
                rid,
                {
                    "type": "send",
                    "notice": f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn.",
                    "message": arg,
                },
            )
        except Exception as exc:
            return _err(rid, 5030, f"moa unavailable: {exc}")

    if name == "focus":
        # /focus is display-only. Route it through the same config.set branch the
        # Ink TUI slash command uses so both surfaces share one state machine and
        # one persistence path. Returns a plain notice line for the transcript.
        from hermes_cli.focus_view import (
            format_focus_status,
            format_focus_toggle_message,
            resolve_focus_arg,
        )

        _display_focus = _load_cfg().get("display")
        _d_focus: dict = _display_focus if isinstance(_display_focus, dict) else {}
        _cur_focus = bool(_d_focus.get("focus_view", False))
        _action, _target = resolve_focus_arg(arg, _cur_focus)
        if _action == "usage":
            return _err(rid, 4004, "usage: /focus [on|off|status]")
        if _action == "status":
            _saved = _d_focus.get("focus_saved_tool_progress") or _load_tool_progress_mode()
            return _ok(
                rid,
                {"type": "exec", "output": format_focus_status(_cur_focus, _saved)},
            )
        _res = _methods["config.set"](
            rid,
            {
                "key": "focus",
                "value": "on" if _target else "off",
                "session_id": params.get("session_id", ""),
            },
        )
        if "error" in _res:
            return _res
        _payload = _res.get("result") or {}
        return _ok(
            rid,
            {
                "type": "exec",
                "output": format_focus_toggle_message(
                    bool(_target), _payload.get("tool_progress") or "all"
                ),
            },
        )

    if name == "retry":
        if not session:
            return _err(rid, 4001, "no active session to retry")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /retry"
            )
        history = session.get("history", [])
        if not history:
            return _err(rid, 4018, "no previous user message to retry")
        # Walk backwards to the last *real* user turn. Timeline bookkeeping
        # rows (display_kind set) and compaction handoffs are durable
        # role=user but must not count as user-originated asks — same
        # predicate as CLI resume/count and the prompt.submit ordinal fix.
        # Without this, /retry re-sends opaque markers (model_switch /
        # async_delegation_complete / auto_continue / CONTEXT COMPACTION
        # handoffs) and truncates only the marker instead of the failed
        # exchange (#80622).
        from agent.context_compressor import is_user_originated_turn

        last_user_idx = None
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if is_user_originated_turn(msg):
                last_user_idx = i
                break
        if last_user_idx is None:
            return _err(rid, 4018, "no previous user message to retry")
        content = history[last_user_idx].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not content:
            return _err(rid, 4018, "last user message is empty")
        # Truncate history: remove everything from the last user message onward
        # (mirrors CLI retry_last() which strips the failed exchange)
        with session["history_lock"]:
            session["history"] = history[:last_user_idx]
            session["history_version"] = int(session.get("history_version", 0)) + 1
        return _ok(rid, {"type": "send", "message": content})

    if name == "steer":
        if not arg:
            return _err(rid, 4004, "usage: /steer <prompt>")
        agent = session.get("agent") if session else None
        if agent and hasattr(agent, "steer"):
            try:
                accepted = agent.steer(arg)
                if accepted:
                    return _ok(
                        rid,
                        {
                            "type": "exec",
                            "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}",
                        },
                    )
            except Exception:
                pass
        # Fallback: no active run, treat as next-turn message
        return _ok(rid, {"type": "send", "message": arg})

    if name == "goal":
        if not session:
            return _err(rid, 4001, "no active session")
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            return _err(rid, 5030, f"goals unavailable: {exc}")

        sid_key = session.get("session_key") or ""
        if not sid_key:
            return _err(rid, 4001, "no session key")

        try:
            goals_cfg = _load_cfg().get("goals") or {}
            max_turns = int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            max_turns = 20
        mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

        lower = arg.strip().lower()
        if not arg.strip() or lower == "status":
            return _ok(rid, {"type": "exec", "output": mgr.status_line()})
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
            return _ok(rid, {"type": "exec", "output": out})
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return _ok(rid, {"type": "exec", "output": "No goal to resume."})
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        f"▶ Goal resumed: {state.goal}\n"
                        "Send any message to continue, or wait — I'll take the next step on the next turn."
                    ),
                },
            )
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "✓ Goal cleared." if had else "No active goal.",
                },
            )

        # Otherwise — treat the remaining text as the new goal.
        try:
            state = mgr.set(arg)
        except ValueError as exc:
            return _err(rid, 4004, f"invalid goal: {exc}")

        notice = (
            f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
            "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
            "Controls: /goal status · /goal pause · /goal resume · /goal clear"
        )
        # Send the goal text as the kickoff prompt. The TUI client sees
        # {type: send, notice, message} → renders `notice` as a sys line,
        # then submits `message` as a user turn. The post-turn judge
        # wired in _run_prompt_submit takes over from there.
        return _ok(
            rid,
            {"type": "send", "notice": notice, "message": state.goal},
        )

    if name == "undo":
        # /undo [N]: back up N user turns (default 1), soft-delete the
        # truncated rows on disk, and prefill the composer with the text
        # of the user message we backed up to so it can be edited and
        # resubmitted. N=1 is the Claude-Code-style single-step undo;
        # /undo 3 backs up three user turns at once. See issue #21910.
        if not session:
            return _err(rid, 4001, "no active session to undo")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /undo"
            )
        db = _get_db()
        if db is None:
            return _db_unavailable_error(rid, code=5008)
        session_key = session.get("session_key", "")
        if not session_key:
            return _err(rid, 4001, "no session key for undo")
        # Parse the optional count argument (e.g. "/undo 3" → 3).
        n = 1
        arg_str = (arg or "").strip()
        if arg_str:
            try:
                n = int(arg_str.split()[0])
            except (ValueError, IndexError):
                return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
        if n < 1:
            n = 1
        try:
            recents = db.list_recent_user_messages(session_key, limit=max(n, 10))
        except Exception as e:
            return _err(rid, 5008, f"undo: failed to load history: {e}")
        if not recents:
            return _err(rid, 4018, "no user messages to undo")
        # recents[0] is the most-recent user turn; pick the Nth-from-last.
        # If N exceeds the number of user turns, back up to the oldest.
        target_idx = min(n - 1, len(recents) - 1)
        target_id = recents[target_idx]["id"]
        try:
            result = db.rewind_to_message(session_key, target_id)
        except ValueError as e:
            return _err(rid, 4004, f"undo: {e}")
        except Exception as e:
            return _err(rid, 5008, f"undo: {e}")
        # Reload the active-only transcript into the in-memory session
        # history so subsequent turns see the truncated view.
        # repair_alternation: this reload feeds LIVE REPLAY — session["history"]
        # is the working conversation for subsequent turns, and a rewind that
        # lands on a durable user;user pair would otherwise re-fire the
        # pre-request repair on every request from here on.
        try:
            active = db.get_messages_as_conversation(session_key, repair_alternation=True)
        except Exception:
            active = []
        with session["history_lock"]:
            session["history"] = list(active)
            session["history_version"] = int(session.get("history_version", 0)) + 1
        # Notify memory providers — same hook /branch fires, plus the
        # rewound flag so providers caching per-turn document state
        # know to invalidate. See #6672 + #21910.
        agent = session.get("agent")
        if agent is not None:
            mm = getattr(agent, "_memory_manager", None)
            if mm is not None:
                try:
                    mm.on_session_switch(
                        session_key,
                        parent_session_id="",
                        reset=False,
                        rewound=True,
                    )
                except Exception:
                    pass
            if hasattr(agent, "_invalidate_system_prompt"):
                try:
                    agent._invalidate_system_prompt()
                except Exception:
                    pass
            if hasattr(agent, "_last_flushed_db_idx"):
                try:
                    agent._last_flushed_db_idx = len(active)
                except Exception:
                    pass
        target_msg = result.get("target_message") or {}
        target_text = target_msg.get("content") or ""
        if isinstance(target_text, list):
            parts = [
                p.get("text", "") for p in target_text
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            target_text = "\n".join(t for t in parts if t)
        if not isinstance(target_text, str):
            target_text = ""
        rewound_count = result.get("rewound_count", 0)
        turns_undone = target_idx + 1
        turn_word = "turn" if turns_undone == 1 else "turns"
        notice = (
            f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). "
            "Edit and resubmit, or send a new message."
        )
        return _ok(
            rid,
            {"type": "prefill", "message": target_text, "notice": notice},
        )

    if name in {"snapshot", "snap"}:
        subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
        if subcommand in {"restore", "rewind"}:
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": (
                        "/snapshot restore is blocked in the TUI because it changes "
                        "config/state on disk while the live agent has cached settings. "
                        "Run it in the classic CLI, then restart the TUI."
                    ),
                },
            )

    if name in {"compress", "compact"}:
        if not session:
            return _err(rid, 4001, "no active session to compress")
        if session.get("running"):
            return _err(
                rid, 4009, "session busy — /interrupt the current turn before /compress"
            )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )

        sid = params.get("session_id", "")
        if _session_uses_compute_host(session):
            command = f"/{name}" + (f" {arg}" if arg else "")
            try:
                ack = _send_compute_host_control(
                    sid,
                    route_name="slash.compress",
                    command=command,
                    wait=True,
                )
            except Exception as exc:
                return _err(rid, 5019, f"compute-host slash.compress failed: {exc}")
            if ack.get("type") in {"control.error", "error"}:
                return _err(
                    rid,
                    4009,
                    str(ack.get("message") or "compute-host slash.compress failed"),
                )
            _apply_compute_host_metadata_mirror(session, ack)
            return _ok(
                rid,
                {"type": "exec", "output": str(ack.get("output") or "")},
            )
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
            removed, usage = _compress_session_history(
                session,
                arg.strip() or None,
                approx_tokens=before_tokens,
                before_messages=before_messages,
                history_version=history_version,
            )
            with session["history_lock"]:
                after_messages = list(session.get("history", []))
            after_count = len(after_messages)
            _sys_prompt_after = (
                getattr(_agent, "_cached_system_prompt", "") or _sys_prompt
            )
            _tools_after = getattr(_agent, "tools", None) or _tools
            after_tokens = (
                estimate_request_tokens_rough(
                    after_messages,
                    system_prompt=_sys_prompt_after,
                    tools=_tools_after,
                )
                if after_count
                else 0
            )
            _sync_session_key_after_compress(sid, session)
            summary = summarize_manual_compression(
                before_messages,
                after_messages,
                before_tokens,
                after_tokens,
                compression_state=getattr(_agent, "context_compressor", None),
            )
            _emit("session.info", sid, _session_info(session.get("agent"), session))
            finalize_context_engine_compression_notification(
                _agent,
                committed=True,
            )
            return _ok(
                rid,
                {
                    "type": "exec",
                    "output": "\n".join(
                        filter(None, [summary["headline"], summary["token_line"], summary.get("note")])
                    ),
                },
            )
        except CompressionLockHeld as e:
            # Lock-skip is a clean no-op, not a failure: report it as
            # normal command output (matching the slash-mirror and
            # session.compress RPC), never as a "compress failed" error.
            # _compress_session_history already discarded the deferred
            # context-engine notification before raising.
            from agent.manual_compression_feedback import (
                describe_compression_lock_skip,
            )
            return _ok(
                rid,
                {"type": "exec", "output": describe_compression_lock_skip(e.holder)},
            )
        except Exception as exc:
            finalize_context_engine_compression_notification(
                session["agent"],
                committed=False,
            )
            return _err(rid, 5009, f"compress failed: {exc}")

    return _err(rid, 4018, f"not a quick/plugin/bundle/skill command: {name}")


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    # Skill and bundle slash commands plus _pending_input commands must NOT go
    # through the slash worker — see _PENDING_INPUT_COMMANDS definition above.
    # Plugin commands must also avoid the worker, but unlike skills and
    # pending-input commands they still return normal slash.exec output so the
    # TUI keeps the pager path.
    _cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0] if _cmd_parts else "").lower()
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""

    live_output = _live_slash_command_output(
        params.get("session_id", ""), session, _cmd_base, _cmd_arg
    )
    if live_output is not None:
        return _ok(rid, {"output": live_output or "(no output)"})

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        # Route directly to command.dispatch instead of returning an error
        # that requires the frontend to retry.  Some TUI clients fail the
        # fallback, leaving the command empty and showing "empty command".
        return _methods["command.dispatch"](
            rid,
            {
                "name": _cmd_base,
                "arg": _cmd_arg,
                "session_id": params.get("session_id", ""),
            },
        )

    if _cmd_base in _WORKER_BLOCKED_COMMANDS:
        subcommand = _cmd_arg.split(maxsplit=1)[0].lower() if _cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(
                rid,
                4018,
                "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore",
            )

    try:
        from agent.skill_bundles import resolve_bundle_command_key
        from hermes_cli.commands import resolve_command

        _bundle_key = (
            resolve_bundle_command_key(_cmd_base)
            if resolve_command(_cmd_base) is None
            else None
        )
        if _bundle_key is not None:
            return _methods["command.dispatch"](
                rid,
                {
                    "name": _bundle_key.lstrip("/"),
                    "arg": _cmd_arg,
                    "session_id": params.get("session_id", ""),
                },
            )
    except Exception:
        pass

    try:
        from agent.skill_commands import get_skill_commands

        _cmd_key = f"/{_cmd_base}"
        if _cmd_key in get_skill_commands():
            return _err(
                rid, 4018, f"skill command: use command.dispatch for {_cmd_key}"
            )
    except Exception:
        pass

    plugin_handler = None
    resolve_plugin_command_result = None
    if _cmd_base:
        try:
            from hermes_cli.plugins import (
                get_plugin_command_handler,
                resolve_plugin_command_result,
            )

            plugin_handler = get_plugin_command_handler(_cmd_base)
        except Exception:
            plugin_handler = None
            resolve_plugin_command_result = None

    if plugin_handler and resolve_plugin_command_result:
        try:
            result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})

    worker = session.get("slash_worker")
    if not worker:
        # On-demand spawn is now the ONLY spawn path for a fresh session
        # (eager pre-warm removed), and slash.exec handlers run on the RPC
        # thread pool — two concurrent slash commands on the same session
        # could both observe slash_worker=None and each fork a full
        # MCP-fleet worker (the loser of the _attach_worker race would leak
        # unclosed). Serialize first-use spawn per session.
        with _sessions_lock:
            spawn_lock = session.setdefault("_slash_spawn_lock", threading.Lock())
        with spawn_lock:
            worker = session.get("slash_worker")
            if not worker:
                try:
                    worker = _SlashWorker(
                        session["session_key"],
                        getattr(session.get("agent"), "model", _resolve_model()),
                        profile_home=session.get("profile_home"),
                    )
                    _attach_worker(params.get("session_id", ""), session, worker)
                except Exception as e:
                    return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        warning = _mirror_slash_side_effects(params.get("session_id", ""), session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        try:
            worker.close()
        except Exception:
            pass
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


@method("insights.get")
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5017)
    try:
        cutoff = time.time() - days * 86400
        rows = [
            s
            for s in db.list_sessions_rich(limit=500, compact_rows=True)
            if (s.get("started_at") or 0) >= cutoff
        ]
        return _ok(
            rid,
            {
                "days": days,
                "sessions": len(rows),
                "messages": sum(s.get("message_count", 0) for s in rows),
            },
        )
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("rollback.list")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:

        def go(mgr, cwd):
            if not mgr.enabled:
                return _ok(rid, {"enabled": False, "checkpoints": []})
            return _ok(
                rid,
                {
                    "enabled": True,
                    "checkpoints": [
                        {
                            "hash": c.get("hash", ""),
                            "timestamp": c.get("timestamp", ""),
                            "message": c.get("message", ""),
                        }
                        for c in mgr.list_checkpoints(cwd)
                    ],
                },
            )

        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("rollback.restore")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history.  Rejecting during
    # an in-flight turn prevents prompt.submit from silently dropping
    # the agent's output (version mismatch path) or clobbering the
    # rollback (version-matches path).  A file-scoped rollback only
    # touches disk, so we allow it.
    if not file_path and session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before full rollback.restore",
        )
    try:

        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    history = session.get("history", [])
                    # Truncate from the last *real* user turn. Same predicate
                    # as list_recent_user_messages / /undo / /retry —
                    # is_user_originated_turn also excludes compaction
                    # handoffs (durable role=user, sometimes without
                    # display_kind on legacy sessions; #80622).
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
                    if removed:
                        session["history_version"] = (
                            int(session.get("history_version", 0)) + 1
                        )
                result["history_removed"] = removed
            return result

        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(
            session,
            lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)),
        )
        raw = r.get("diff", "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")

    if action == "status":
        url = _resolve_browser_cdp_url()
        return _ok(rid, {"connected": bool(url), "url": url})

    if action == "disconnect":
        return _browser_disconnect(rid)

    if action != "connect":
        return _err(rid, 4015, f"unknown action: {action}")

    return _browser_connect(rid, params)


@method("plugins.list")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager

        return _ok(
            rid,
            {
                "plugins": [
                    {
                        "name": n,
                        "version": getattr(i, "version", "?"),
                        "enabled": getattr(i, "enabled", True),
                    }
                    for n, i in get_plugin_manager()._plugins.items()
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("config.show")
def _(rid, params: dict) -> dict:
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        from agent.secret_scope import get_secret

        api_key = get_secret("HERMES_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")

        sections = [
            {
                "title": "Model",
                "rows": [
                    ["Model", model],
                    ["Base URL", base_url or "(default)"],
                    ["API Key", masked],
                ],
            },
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(_cfg_max_turns(cfg, 500))],
                    ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [
                    ["Working Dir", os.getcwd()],
                    ["Config File", str(_hermes_home / "config.yaml")],
                ],
            },
        ]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


@method("tools.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                    "tools": info["resolved_tools"],
                }
            )
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5031, str(e))


@method("tools.show")
def _(rid, params: dict) -> dict:
    try:
        from model_tools import get_toolset_for_tool, get_tool_definitions

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            getattr(session["agent"], "enabled_toolsets", None)
            if session
            else _load_enabled_toolsets()
        )
        # Pre-assembly list: /tools is a discovery surface and must show
        # tools deferred behind the tool_search bridge (same as the CLI).
        tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True,
                                     skip_tool_search_assembly=True)
        sections = {}

        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append(
                {
                    "name": name,
                    "description": desc,
                }
            )

        return _ok(
            rid,
            {
                "sections": [
                    {"name": name, "tools": rows}
                    for name, rows in sorted(sections.items())
                ],
                "total": len(tools),
            },
        )
    except Exception as e:
        return _err(rid, 5034, str(e))


@method("tools.configure")
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [
        str(name).strip() for name in params.get("names", []) or [] if str(name).strip()
    ]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {
            ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS
        } | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = (
            _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        )
        save_config(cfg)

        session = _sessions.get(params.get("session_id", ""))
        info = (
            _reset_session_agent(params.get("session_id", ""), session)
            if session
            else None
        )
        enabled = sorted(
            _get_platform_tools(load_config(), "cli", include_default_mcp_servers=False)
        )
        changed = [
            name
            for name in targets
            if name not in unknown
            and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return _ok(
            rid,
            {
                "changed": changed,
                "enabled_toolsets": enabled,
                "info": info,
                "missing_servers": sorted(missing_servers),
                "reset": bool(session),
                "unknown": unknown,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("toolsets.list")
def _(rid, params: dict) -> dict:
    try:
        from toolsets import get_all_toolsets, get_toolset_info

        session = _sessions.get(params.get("session_id", ""))
        enabled = (
            set(getattr(session["agent"], "enabled_toolsets", []) or [])
            if session
            else set(_load_enabled_toolsets() or [])
        )

        items = []
        for name in sorted(get_all_toolsets().keys()):
            info = get_toolset_info(name)
            if not info:
                continue
            items.append(
                {
                    "name": name,
                    "description": info["description"],
                    "tool_count": info["tool_count"],
                    "enabled": name in enabled if enabled else True,
                }
            )
        return _ok(rid, {"toolsets": items})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("agents.list")
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        procs = process_registry.list_sessions()
        return _ok(
            rid,
            {
                "processes": [
                    {
                        "session_id": p["session_id"],
                        "command": p["command"][:80],
                        "status": p["status"],
                        "uptime": p["uptime_seconds"],
                    }
                    for p in procs
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5033, str(e))


@method("cron.manage")
def _(rid, params: dict) -> dict:
    action, jid = params.get("action", "list"), params.get("name", "")
    try:
        from tools.cronjob_tools import cronjob

        if action == "list":
            # Paused jobs are excluded by default, which reads as deletion in
            # any UI with an enable/disable toggle — forward the flag.
            return _ok(
                rid,
                json.loads(
                    cronjob(
                        action="list",
                        include_disabled=is_truthy_value(params.get("include_disabled", False)),
                    )
                ),
            )
        if action == "add":
            return _ok(
                rid,
                json.loads(
                    cronjob(
                        action="create",
                        name=jid,
                        schedule=params.get("schedule", ""),
                        prompt=params.get("prompt", ""),
                        # Optional repeat cap ("run N times"); None keeps the
                        # schedule-kind default (once for one-shot, forever
                        # for recurring).
                        repeat=(
                            int(params["repeat"])
                            if str(params.get("repeat", "")).strip().isdigit()
                            else None
                        ),
                    )
                ),
            )
        if action in {"remove", "pause", "resume"}:
            return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
        return _err(rid, 4016, f"unknown cron action: {action}")
    except Exception as e:
        return _err(rid, 5023, str(e))


@method("learning.frames")
def _(rid, params: dict) -> dict:
    """Pre-render the learning timeline for the TUI ``/journey`` overlay.

    Returns ``frames`` (reveal 0→1) plus static legend/summary/bucket metadata,
    so Ink can render and walk the tree locally without round-tripping the
    gateway. Shares its renderer with the ``hermes journey`` CLI.
    """
    try:
        cols = int(params.get("cols", 80) or 80)
        rows = int(params.get("rows", 24) or 24)
        frames = int(params.get("frames", 48) or 48)
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    try:
        from agent.learning_graph import build_learning_graph
        from agent.learning_graph_render import render_frames

        payload = build_learning_graph()
        return _ok(rid, render_frames(payload, cols=max(20, cols), rows=max(10, rows), frames=frames))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.frames failed: {exc}")


@method("learning.detail")
def _(rid, params: dict) -> dict:
    """Current content of a journey node, for an edit prefill."""
    try:
        from agent.learning_mutations import node_detail

        return _ok(rid, node_detail(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.detail failed: {exc}")


@method("learning.delete")
def _(rid, params: dict) -> dict:
    """Delete a journey node — skills are archived (restorable), memories removed."""
    try:
        from agent.learning_mutations import delete_node

        return _ok(rid, delete_node(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.delete failed: {exc}")


@method("learning.edit")
def _(rid, params: dict) -> dict:
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    try:
        from agent.learning_mutations import edit_node

        return _ok(rid, edit_node(str(params.get("id", "")), str(params.get("content", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.edit failed: {exc}")


@method("skills.manage")
def _(rid, params: dict) -> dict:
    action, query = params.get("action", "list"), params.get("query", "")
    # Optional profile scoping: list/install operate on that profile's
    # skills dir (capabilities UIs manage a bot's skills from the main
    # window). Search/browse/inspect hit the shared hub catalog — the
    # override is harmless there and keeps the semantics uniform.
    profile = str(params.get("profile") or "").strip()
    token = None
    if profile:
        try:
            from hermes_cli.profiles import get_profile_dir
            from hermes_constants import set_hermes_home_override

            profile_dir = get_profile_dir(profile)
            if not profile_dir or not profile_dir.is_dir():
                return _err(rid, 4064, f"profile '{profile}' not found")
            token = set_hermes_home_override(str(profile_dir))
        except Exception as e:
            return _err(rid, 5024, str(e))
    try:
        if action == "list":
            from hermes_cli.banner import get_available_skills

            return _ok(rid, {"skills": get_available_skills()})
        if action == "search":
            from tools.skills_hub import (
                GitHubAuth,
                create_source_router,
                unified_search,
            )

            raw = (
                unified_search(
                    query,
                    create_source_router(GitHubAuth()),
                    source_filter="all",
                    limit=20,
                )
                or []
            )
            return _ok(
                rid,
                {
                    "results": [
                        {"name": r.name, "description": r.description} for r in raw
                    ]
                },
            )
        if action == "install":
            from hermes_cli.skills_hub import do_install

            class _Q:
                def print(self, *a, **k):
                    pass

            do_install(query, skip_confirm=True, console=_Q())
            return _ok(rid, {"installed": True, "name": query})
        if action == "browse":
            from hermes_cli.skills_hub import browse_skills

            pg = int(params.get("page", 0) or 0) or (
                int(query) if query.isdigit() else 1
            )
            return _ok(
                rid, browse_skills(page=pg, page_size=int(params.get("page_size", 20)))
            )
        if action == "inspect":
            from hermes_cli.skills_hub import inspect_skill

            return _ok(rid, {"info": inspect_skill(query) or {}})
        return _err(rid, 4017, f"unknown skills action: {action}")
    except Exception as e:
        return _err(rid, 5024, str(e))
    finally:
        if token is not None:
            try:
                from hermes_constants import reset_hermes_home_override

                reset_hermes_home_override(token)
            except Exception:
                pass


@method("mcp.catalog")
def _(rid, params: dict) -> dict:
    """Bundled MCP catalog with per-profile install/enable state.

    Params: optional ``profile`` (defaults to the launch profile). Result:
    ``{servers: [{name, description, installed, enabled, requires: [env
    keys], transport}]}`` — the same catalog `hermes mcp` offers, so
    capability UIs can present the full menu and know which entries need
    setup (missing requires) before they'll work.
    """
    profile = str(params.get("profile") or "").strip()
    token = None
    try:
        if profile:
            from hermes_cli.profiles import get_profile_dir
            from hermes_constants import set_hermes_home_override

            profile_dir = get_profile_dir(profile)
            if not profile_dir or not profile_dir.is_dir():
                return _err(rid, 4064, f"profile '{profile}' not found")
            token = set_hermes_home_override(str(profile_dir))

        from hermes_cli import mcp_catalog

        out = []
        for entry in mcp_catalog.list_catalog():
            try:
                requires = [str(k) for k in (getattr(entry, "env_keys", None) or [])]
            except Exception:
                requires = []
            out.append(
                {
                    "name": entry.name,
                    "description": getattr(entry, "description", "") or "",
                    "installed": bool(mcp_catalog.is_installed(entry.name)),
                    "enabled": bool(mcp_catalog.is_enabled(entry.name)),
                    "requires": requires,
                    # TransportSpec object — reduce to its kind string.
                    "transport": str(
                        getattr(getattr(entry, "transport", None), "kind", "")
                        or getattr(entry, "transport", "")
                        or "stdio"
                    ),
                }
            )
        return _ok(rid, {"servers": out})
    except Exception as e:
        return _err(rid, 5024, str(e))
    finally:
        if token is not None:
            try:
                from hermes_constants import reset_hermes_home_override

                reset_hermes_home_override(token)
            except Exception:
                pass


@method("skills.reload")
def _(rid, params: dict) -> dict:
    try:
        from agent.skill_commands import reload_skills

        result = reload_skills()
        added = result.get("added") or []
        removed = result.get("removed") or []
        total = int(result.get("total") or 0)

        lines = ["Reloading skills..."]
        if not added and not removed:
            lines.append("No new skills detected.")
        if added:
            lines.append("Added skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in added)
        if removed:
            lines.append("Removed skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in removed)
        lines.append(f"{total} skill(s) available")
        return _ok(rid, {"output": "\n".join(lines), "result": result})
    except Exception as e:
        return _err(rid, 5025, str(e))


@method("plugins.manage")
def _(rid, params: dict) -> dict:
    """List installed plugins with activation state, or toggle one on/off.

    Backs the TUI Plugins Hub. Uses the same disk-discovery + enable/disable
    primitives as ``hermes plugins`` / the dashboard, so the three surfaces
    agree on what's installed and what's enabled.

    Actions:
      - ``list``   → {"plugins": [{name, key, version, description, source,
                       status, portable}], "user_count": N, "bundled_count": M}
      - ``toggle`` → flip ``key`` (or ``name``) based on ``enable`` (bool).
                       Returns the refreshed row plus {"ok", "unchanged"}.
    """
    action = params.get("action", "list")
    try:
        from hermes_cli.plugins_cmd import (
            _bundled_default_on,
            _discover_all_plugins,
            _get_disabled_set,
            _get_enabled_set,
            _is_portable_plugin_dir,
            _plugin_status,
        )

        def _rows():
            enabled = _get_enabled_set()
            disabled = _get_disabled_set()
            out = []
            for name, version, desc, source, _dir, key in sorted(
                _discover_all_plugins()
            ):
                status = _plugin_status(name, enabled, disabled, key=key)
                # Bundled backends/platforms/providers are active without an
                # explicit enable (they "just work" — plugins.py). Reporting
                # them "not enabled" reads as OFF in clients when they are in
                # fact running; surface the truthful default instead.
                if (
                    status == "not enabled"
                    and source == "bundled"
                    and _bundled_default_on(_dir)
                ):
                    status = "enabled"
                out.append(
                    {
                        "name": name,
                        # Canonical registry key (e.g. ``image_gen/fal``). Names
                        # can collide across category dirs — both fal backends
                        # are named "fal" — so toggles must address the key.
                        "key": key,
                        "version": str(version or ""),
                        "description": desc or "",
                        "source": source,
                        "status": status,
                        # Agent Plugins v1 package (plugin.json — the portable
                        # skills/MCP format) vs a native Hermes plugin.
                        "portable": _is_portable_plugin_dir(_dir),
                    }
                )
            return out

        if action == "list":
            rows = _rows()
            user_count = sum(1 for r in rows if r["source"] != "bundled")
            return _ok(
                rid,
                {
                    "plugins": rows,
                    "user_count": user_count,
                    "bundled_count": len(rows) - user_count,
                },
            )

        if action == "toggle":
            from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

            # Prefer the canonical key — bare names are ambiguous when two
            # category plugins share one (image_gen/fal vs video_gen/fal).
            ident = (params.get("key") or params.get("name") or "").strip()
            if not ident:
                return _err(rid, 4019, "plugins.toggle requires a 'key' or 'name'")
            enable = bool(params.get("enable"))
            result = dashboard_set_agent_plugin_enabled(ident, enabled=enable)
            if not result.get("ok"):
                return _err(rid, 5026, result.get("error") or "toggle failed")
            row = next(
                (r for r in _rows() if ident in (r["key"], r["name"])), None
            )
            return _ok(
                rid,
                {
                    "ok": True,
                    "unchanged": bool(result.get("unchanged")),
                    "name": ident,
                    "plugin": row,
                },
            )

        return _err(rid, 4017, f"unknown plugins action: {action}")
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command, detect_hardline_command

        is_hardline, hardline_desc = detect_hardline_command(cmd)
        if is_hardline:
            return _err(
                rid, 4005, f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands."
            )
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(
                rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands."
            )
    except ImportError:
        return _err(rid, 5001, "shell.exec unavailable: approval safety module not importable")
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd(),
            # Force UTF-8 + lossy decode so non-UTF-8 child output can't crash
            # the gateway thread on locale-mismatched Windows (#53137).
            encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        return _ok(
            rid,
            {
                "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-2000:],
                "code": r.returncode,
            },
        )
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
