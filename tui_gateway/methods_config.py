"""Config / projects / setup JSON-RPC handlers (moved verbatim from server.py).

NOTE: ``config.set`` stays in server.py for now — the in-flight
opt/model-resolution-core PR touches it; move it in a follow-up once merged.

Handler bodies are byte-identical to their pre-split server.py form; they
are rebound onto server.py's globals at install time — see method_ctx.py.
"""

from .method_ctx import HandlerRegistry

from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("projects.discover_repos")
def _(rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"repos": []})
        from hermes_cli import projects_db as pdb

        policy = _repo_discovery_policy()
        policy_key = _repo_discovery_policy_key(policy)
        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(
                conn,
                policy_key,
                preserve_unversioned=_repo_discovery_policy_is_default(policy),
            )
            repos = _discover_repos_payload(
                db, conn=conn, include_cached=policy["enabled"]
            )
        return _ok(rid, {"repos": repos, "discovery_policy": policy})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.record_repos")
def _(rid, params: dict) -> dict:
    """Persist git repo roots found by the client's filesystem scan, then return
    the merged repo list. The native crawl runs on the desktop (local fs); this
    caches the result so later reads are instant instead of re-walking disk."""
    try:
        from hermes_cli import projects_db as pdb

        policy = _repo_discovery_policy()
        policy_key = _repo_discovery_policy_key(policy)
        incoming_raw = params.get("discovery_policy")
        incoming_policy = (
            _repo_discovery_policy(incoming_raw)
            if isinstance(incoming_raw, dict)
            else None
        )
        incoming_matches = (
            incoming_policy is not None
            and _repo_discovery_policy_key(incoming_policy) == policy_key
        )
        accept_legacy_default = (
            incoming_policy is None and _repo_discovery_policy_is_default(policy)
        )

        pairs: list[tuple[str, str | None]] = []
        for item in params.get("repos") or []:
            if isinstance(item, str):
                pairs.append((item, None))
            elif isinstance(item, dict) and item.get("root"):
                pairs.append((str(item["root"]), item.get("label")))

        with pdb.connect_closing() as conn:
            pdb.reconcile_discovered_repos_policy(
                conn,
                policy_key,
                preserve_unversioned=_repo_discovery_policy_is_default(policy),
            )
            accepted = bool(
                policy["enabled"] and (incoming_matches or accept_legacy_default)
            )
            if accepted:
                pdb.record_discovered_repos(
                    conn, pairs, replace=True, policy_key=policy_key
                )
            elif not policy["enabled"]:
                pdb.clear_discovered_repos(conn, policy_key=policy_key)

        db = _get_db()
        return _ok(
            rid,
            {
                "repos": _discover_repos_payload(
                    db, include_cached=policy["enabled"]
                )
                if db is not None
                else [],
                "accepted": accepted,
                "discovery_policy": policy,
            },
        )
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.tree")
def _(rid, params: dict) -> dict:
    """Authoritative project overview: project -> repo -> lane structure with
    counts + a few preview sessions per project, plus the flat set of session
    ids claimed by any project (so the desktop excludes them from flat Recents).
    Lanes carry no session rows here; drill-in uses ``projects.project_sessions``.
    """
    try:
        db = _get_db()
        if db is None:
            return _ok(rid, {"projects": [], "active_id": None, "scoped_session_ids": []})

        tree, active_id = _build_project_tree(
            db,
            preview_limit=int(params.get("preview_limit") or 3),
            hydrate=False,
            session_limit=int(params.get("session_limit") or 2000),
            include_discovered=True,
        )
        return _ok(
            rid,
            {"projects": tree["projects"], "active_id": active_id, "scoped_session_ids": tree["scoped_session_ids"]},
        )
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("projects.project_sessions")
def _(rid, params: dict) -> dict:
    """Fully hydrated lanes (repo -> lane -> session rows) for one project,
    built from the same authoritative grouping as ``projects.tree`` so ids and
    membership match exactly. Used when the user enters a project."""
    try:
        project_id = str(params.get("project_id") or "")
        if not project_id:
            return _err(rid, 5063, "project_id required")

        db = _get_db()
        if db is None:
            return _ok(rid, {"project": None})

        # Drill-in only needs the entered project (which has sessions), so skip
        # the zero-session discovery tier entirely.
        tree, _active = _build_project_tree(
            db, preview_limit=0, hydrate=True, session_limit=int(params.get("session_limit") or 5000),
            include_discovered=False,
        )
        proj = next((p for p in tree["projects"] if p["id"] == project_id), None)
        return _ok(rid, {"project": proj})
    except Exception as e:
        return _err(rid, 5061, str(e))


@method("config.get")
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    if key == "provider":
        try:
            from hermes_cli.models import list_available_providers, normalize_provider

            model = _resolve_model()
            parts = model.split("/", 1)
            return _ok(
                rid,
                {
                    "model": model,
                    "provider": (
                        normalize_provider(parts[0]) if len(parts) > 1 else "unknown"
                    ),
                    "providers": list_available_providers(),
                },
            )
        except Exception as e:
            return _err(rid, 5013, str(e))
    if key == "profile":
        from hermes_constants import display_hermes_home

        return _ok(rid, {"home": str(_hermes_home), "display": display_hermes_home()})
    if key == "project":
        cfg_terminal = _load_cfg().get("terminal") or {}
        raw = str(params.get("cwd", "") or cfg_terminal.get("cwd", "") or "").strip()
        cwd = _completion_cwd({"cwd": raw} if raw else {})
        return _ok(rid, {"cwd": cwd, "branch": _git_branch_for_cwd(cwd)})
    if key == "full":
        return _ok(rid, {"config": _load_cfg()})
    if key == "prompt":
        return _ok(rid, {"prompt": _load_cfg().get("custom_prompt", "")})
    if key == "skin":
        return _ok(
            rid, {"value": (_load_cfg().get("display") or {}).get("skin", "default")}
        )
    if key == "indicator":
        # Normalize so a hand-edited config.yaml with stray casing or
        # an unknown value reads back the SAME value the TUI actually
        # rendered (frontend's `normalizeIndicatorStyle` falls back to
        # `DEFAULT_INDICATOR_STYLE` for the same inputs).  Otherwise
        # `/indicator` would print one thing while the UI shows another.
        raw = (_load_cfg().get("display") or {}).get("tui_status_indicator", "")
        norm = str(raw).strip().lower()
        return _ok(
            rid,
            {"value": norm if norm in INDICATOR_STYLES else DEFAULT_INDICATOR_STYLE},
        )
    if key == "personality":
        return _ok(
            rid,
            {"value": (_load_cfg().get("display") or {}).get("personality") or "none"},
        )
    if key == "reasoning":
        cfg = _load_cfg()
        session = _sessions.get(params.get("session_id", ""))
        reasoning_config = None
        if session is not None:
            if isinstance(session.get("create_reasoning_override"), dict):
                reasoning_config = session.get("create_reasoning_override")
            else:
                agent = session.get("agent")
                agent_reasoning = getattr(agent, "reasoning_config", None)
                if isinstance(agent_reasoning, dict):
                    reasoning_config = agent_reasoning

        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                effort = "none"
            else:
                effort = str(reasoning_config.get("effort") or "medium")
        else:
            raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
            if raw_effort is False:
                # YAML `reasoning_effort: false`/`off`/`no` — thinking
                # disabled, not "unset, show the medium default".
                effort = "none"
            else:
                effort = str(raw_effort or "medium")
        display = (
            "show"
            if bool((cfg.get("display") or {}).get("show_reasoning", True))
            else "hide"
        )
        return _ok(rid, {"value": effort, "display": display})
    if key == "fast":
        # Prefer the session's live/pinned value — `config.set fast` is
        # session-scoped, so the global key may not reflect this chat. A
        # pre-build session keeps its pin in create_service_tier_override.
        session = _sessions.get(params.get("session_id", ""))
        tier = None
        if session is not None:
            agent = session.get("agent")
            if agent is not None:
                tier = getattr(agent, "service_tier", None)
            elif session.get("create_service_tier_override") is not None:
                tier = session["create_service_tier_override"]
        if tier is None:
            tier = _load_service_tier()
        return _ok(rid, {"value": "fast" if tier == "priority" else "normal"})
    if key == "busy":
        return _ok(rid, {"value": _load_busy_input_mode()})
    if key in {"approval_mode", "approvals.mode"}:
        try:
            return _ok(rid, {"value": _load_approval_mode()})
        except Exception as e:
            return _err(rid, 5001, str(e))
    if key == "details_mode":
        allowed_dm = frozenset({"hidden", "collapsed", "expanded"})
        raw = (
            str(
                (_load_cfg().get("display") or {}).get("details_mode", "collapsed")
                or "collapsed"
            )
            .strip()
            .lower()
        )
        nv = raw if raw in allowed_dm else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "thinking_mode":
        allowed_tm = frozenset({"collapsed", "truncated", "full"})
        cfg = _load_cfg()
        raw = (
            str((cfg.get("display") or {}).get("thinking_mode", "") or "")
            .strip()
            .lower()
        )
        if raw in allowed_tm:
            nv = raw
        else:
            dm = (
                str(
                    (cfg.get("display") or {}).get("details_mode", "collapsed")
                    or "collapsed"
                )
                .strip()
                .lower()
            )
            nv = "full" if dm == "expanded" else "collapsed"
        return _ok(rid, {"value": nv})
    if key == "density":
        on = bool((_load_cfg().get("display") or {}).get("tui_compact", False))
        return _ok(rid, {"value": "on" if on else "off"})
    if key == "theme":
        display = _load_cfg().get("display")
        raw = str(display.get("tui_theme", "auto") if isinstance(display, dict) else "auto").strip().lower()
        return _ok(rid, {"value": raw if raw in {"auto", "light", "dark"} else "auto"})
    if key == "statusbar":
        display = _load_cfg().get("display")
        raw = (
            display.get("tui_statusbar", "top") if isinstance(display, dict) else "top"
        )
        return _ok(rid, {"value": _coerce_statusbar(raw)})
    if key == "focus":
        display = _load_cfg().get("display")
        on = bool(display.get("focus_view", False)) if isinstance(display, dict) else False
        return _ok(
            rid,
            {"value": "on" if on else "off", "tool_progress": _load_tool_progress_mode()},
        )
    if key == "mouse":
        display = _load_cfg().get("display")
        return _ok(rid, {"value": _display_mouse_tracking(display)})
    if key == "mtime":
        cfg_path = _hermes_home / "config.yaml"
        try:
            mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
        except Exception:
            return _ok(rid, {"mtime": 0})
        # Revision hash of the MCP-relevant config sections. The TUI's
        # config-change poller uses it to reload MCP servers only when their
        # config actually changed — a /skin or /statusbar write bumps mtime
        # but must not cost a multi-second MCP reconnect.
        return _ok(rid, {"mtime": mtime, "mcp_rev": _compute_mcp_rev()})
    return _err(rid, 4002, f"unknown config key: {key}")


@method("setup.status")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.main import _has_any_provider_configured

        return _ok(rid, {"provider_configured": bool(_has_any_provider_configured())})
    except Exception as e:
        return _err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: dict) -> dict:
    """Strict provider check: does the configured/default model actually resolve to a usable runtime?

    Unlike setup.status (which returns True if ANY provider auth state is
    discoverable, including indirect fallbacks like ``gh auth token`` for
    Copilot), this runs the same resolve_runtime_provider() call the agent
    uses on session creation. It returns ok=False with the auth error message
    when the user's configured model cannot actually be served, so UIs can
    surface onboarding before the user submits a doomed prompt.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured

        requested = str(params.get("provider") or "").strip() or None
        runtime = resolve_runtime_provider(requested=requested)
        provider_configured = bool(_has_any_provider_configured())
        provider = runtime.get("provider") or "provider"
        source = str(runtime.get("source") or "")
        if not provider_configured and provider == "bedrock" and source in {
            "iam-role",
            "aws-sdk-default-chain",
        }:
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": source,
                    "error": "No Hermes provider is configured.",
                },
            )

        api_key = runtime.get("api_key")
        api_key_text = "" if callable(api_key) else str(api_key or "").strip()
        credential_ok = (
            callable(api_key)
            or api_key_text in {"aws-sdk", "no-key-required"}
            or has_usable_secret(api_key_text)
            or bool(runtime.get("command"))
        )

        if not credential_ok:
            return _ok(
                rid,
                {
                    "ok": False,
                    "provider": provider,
                    "model": runtime.get("model"),
                    "source": runtime.get("source"),
                    "error": f"No usable credentials found for {provider}.",
                },
            )

        return _ok(
            rid,
            {
                "ok": True,
                "provider": runtime.get("provider"),
                "model": runtime.get("model"),
                "source": runtime.get("source"),
            },
        )
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
