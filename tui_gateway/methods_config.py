"""Config / projects / setup JSON-RPC handlers. Reaches server.py state through ``srv`` (method_ctx.py). ``config.set`` lives in methods_config_set.
"""

from .contracts.base import Params
from .contracts.config_free_tier_control import (
    ConfigGetParams,
    ConfigGetResult,
    DiagnosticsShareNousParams,
    DiagnosticsShareNousResult,
    SetupRuntimeCheckParams,
    SetupRuntimeCheckResult,
    SetupStatusResult,
)
from .method_ctx import HandlerRegistry, bind_module
from .contracts.projects_pets import (
    ProjectsDiscoverReposResult,
    ProjectsProjectSessionsResult,
    ProjectsRecordReposResult,
    ProjectsTreeResult,
)

from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES
from hermes_constants import display_hermes_home as _display_hermes_home

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


def _projects_handler(name: str):
    """``@method(name)`` (profile-scoped) whose body's uncaught exception becomes ``_err(rid, 5061)``."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                return fn(rid, params)
            except Exception as e:
                return srv._err(rid, 5061, str(e))
        return method(name)(_profile_scoped(handler))
    return deco


def _reconcile_repo_discovery(pdb, conn, policy, policy_key):
    pdb.reconcile_discovered_repos_policy(conn, policy_key,
                                          preserve_unversioned=srv._repo_discovery_policy_is_default(policy))


@_projects_handler("projects.discover_repos")
def _(rid, params) -> "ProjectsDiscoverReposResult | dict":
    from tui_gateway.contracts.projects_pets import (
        DiscoveredRepo, ProjectsDiscoverReposParams, ProjectsDiscoverReposResult, RepoDiscoveryPolicy,
    )
    assert isinstance(params, ProjectsDiscoverReposParams)
    with srv._profile_db(params) as db:
        if db is None:
            return ProjectsDiscoverReposResult(repos=[], discovery_policy=None)
        from hermes_cli import projects_db as pdb
        policy = srv._repo_discovery_policy()
        with pdb.connect_closing() as conn:
            srv._reconcile_repo_discovery(pdb, conn, policy, srv._repo_discovery_policy_key(policy))
            if params.scan and policy["enabled"]:
                srv._scan_discovered_repos_remote(conn, policy)
            repos = srv._discover_repos_payload(db, conn=conn, include_cached=policy["enabled"])
        return ProjectsDiscoverReposResult(
            repos=[DiscoveredRepo.model_validate(repo) for repo in repos],
            discovery_policy=RepoDiscoveryPolicy.model_validate(policy),
        )


@_projects_handler("projects.record_repos")
def _(rid, params) -> "ProjectsRecordReposResult | dict":
    from tui_gateway.contracts.projects_pets import (
        DiscoveredRepo, ProjectsRecordReposParams, ProjectsRecordReposResult, RepoDiscoveryPolicy,
    )
    assert isinstance(params, ProjectsRecordReposParams)
    from hermes_cli import projects_db as pdb
    policy = srv._repo_discovery_policy()
    policy_key = srv._repo_discovery_policy_key(policy)
    incoming = params.discovery_policy
    accepted = (
        srv._repo_discovery_policy_key(srv._repo_discovery_policy(incoming.model_dump(mode="json"))) == policy_key
        if incoming is not None else srv._repo_discovery_policy_is_default(policy))
    accepted = bool(policy["enabled"] and accepted)
    pairs = [(repo, None) if isinstance(repo, str) else (repo.root, repo.label) for repo in params.repos or []]
    with pdb.connect_closing() as conn:
        srv._reconcile_repo_discovery(pdb, conn, policy, policy_key)
        if accepted:
            pdb.record_discovered_repos(conn, pairs, replace=True, policy_key=policy_key)
        elif not policy["enabled"]:
            pdb.clear_discovered_repos(conn, policy_key=policy_key)
    with srv._profile_db(params) as db:
        repos = [] if db is None else srv._discover_repos_payload(db, include_cached=policy["enabled"])
        return ProjectsRecordReposResult(
            repos=[DiscoveredRepo.model_validate(repo) for repo in repos], accepted=accepted,
            discovery_policy=RepoDiscoveryPolicy.model_validate(policy),
        )


def _stamped_project_tree(db, params, **kwargs):
    """``_build_project_tree`` + profile stamping shared by the two tree RPCs."""
    from tui_gateway.project_tree import stamp_profile
    tree, active_id = srv._build_project_tree(db, **kwargs)
    stamp_profile(tree["projects"], srv._response_profile_name(params.profile))
    return tree, active_id


@_projects_handler("projects.tree")
def _(rid, params) -> "ProjectsTreeResult | dict":
    from tui_gateway.contracts.projects_pets import ProjectTreeNode, ProjectsTreeParams, ProjectsTreeResult
    assert isinstance(params, ProjectsTreeParams)
    with srv._profile_db(params) as db:
        if db is None:
            return ProjectsTreeResult(projects=[], active_id=None, scoped_session_ids=[])
        tree, active_id = srv._stamped_project_tree(
            db, params, preview_limit=params.preview_limit or 3, hydrate=True,
            session_limit=params.session_limit or 2000, include_discovered=True)
        return ProjectsTreeResult(
            projects=[ProjectTreeNode.model_validate(project) for project in tree["projects"]],
            active_id=active_id, scoped_session_ids=tree["scoped_session_ids"],
        )


@_projects_handler("projects.project_sessions")
def _(rid, params) -> "ProjectsProjectSessionsResult | dict":
    from tui_gateway.contracts.projects_pets import (
        ProjectsProjectSessionsParams, ProjectsProjectSessionsResult, ProjectTreeNode,
    )
    assert isinstance(params, ProjectsProjectSessionsParams)
    with srv._profile_db(params) as db:
        if db is None:
            return ProjectsProjectSessionsResult(project=None)
        tree, _active = srv._stamped_project_tree(
            db, params, preview_limit=0, hydrate=True,
            session_limit=params.session_limit or 5000, include_discovered=False)
        project = next((item for item in tree["projects"] if item["id"] == params.project_id), None)
        return ProjectsProjectSessionsResult(
            project=ProjectTreeNode.model_validate(project) if project else None)


# ── config.get — one getter per key returning the result payload.

def _display_raw() -> dict:
    return srv._load_cfg().get("display") or {}


def _display_word(key: str, default: str, allowed) -> str:
    """Normalised ``display.<key>``; unknown/garbage values read back as ``default``."""
    raw = str(_display_raw().get(key, default) or "").strip().lower()
    return raw if raw in allowed else default


_THINKING_MODES = frozenset({"collapsed", "truncated", "full"})


def _cfg_get_provider(params):
    from hermes_cli.models import list_available_providers, normalize_provider
    model = srv._resolve_model()
    parts = model.split("/", 1)
    return {"model": model, "provider": normalize_provider(parts[0]) if len(parts) > 1 else "unknown",
            "providers": list_available_providers()}


def _cfg_get_project(params: ConfigGetParams):
    raw = str(params.cwd or (srv._load_cfg().get("terminal") or {}).get("cwd", "") or "").strip()
    cwd = srv._completion_cwd({"cwd": raw} if raw else {})
    return {"cwd": cwd, "branch": srv.git_probe.branch(cwd)}


def _cfg_get_personality(params):
    # EFFECTIVE personality via the single owner — a stale/unknown name must not show as active.
    from hermes_cli.personality import active_personality_name
    return {"value": active_personality_name(srv._load_cfg()) or "none"}


def _cfg_get_reasoning(params: ConfigGetParams):
    cfg = srv._load_cfg()
    session = srv._sessions.get(params.session_id) or {}
    reasoning_config = session.get("create_reasoning_override")
    if session and not isinstance(reasoning_config, dict):
        reasoning_config = getattr(session.get("agent"), "reasoning_config", None)
    if isinstance(reasoning_config, dict):
        enabled = reasoning_config.get("enabled") is not False
        effort = str(reasoning_config.get("effort") or "medium") if enabled else "none"
    else:
        raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
        # YAML `reasoning_effort: false` means thinking disabled, not "unset".
        effort = "none" if raw_effort is False else str(raw_effort or "medium")
    display = "show" if (cfg.get("display") or {}).get("show_reasoning", True) else "hide"
    return {"value": effort, "display": display}


def _cfg_get_fast(params: ConfigGetParams):
    # `config.set fast` is session-scoped: prefer the session's live/pinned value over the
    # global key (a pre-build session keeps its pin in create_service_tier_override).
    session = srv._sessions.get(params.session_id) or {}
    agent = session.get("agent")
    tier = (getattr(agent, "service_tier", None) if agent is not None
            else session.get("create_service_tier_override"))
    if tier is None:
        tier = srv._load_service_tier()
    return {"value": "fast" if tier == "priority" else "normal"}


def _cfg_get_thinking_mode(params):
    raw = _display_word("thinking_mode", "", srv._THINKING_MODES)
    if not raw:  # legacy details_mode fallback
        raw = "full" if _display_word("details_mode", "collapsed", srv._DETAIL_MODES) == "expanded" else "collapsed"
    return {"value": raw}


def _cfg_get_mtime(params):
    cfg_path = srv._hermes_home / "config.yaml"
    try:
        mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
    except Exception:
        return {"mtime": 0}
    # mcp_rev: hash of the MCP-relevant sections so the poller reloads MCP servers only when
    # their config changed — a /skin write bumps mtime but must not cost an MCP reconnect.
    return {"mtime": mtime, "mcp_rev": srv._compute_mcp_rev()}


# key -> getter(params); bind_module publishes the table onto server.py.
_CONFIG_GETTERS = {
    "provider": _cfg_get_provider,
    "profile": lambda params: {"home": str(srv._hermes_home), "display": _display_hermes_home()},
    "project": _cfg_get_project,
    "full": lambda params: {"config": srv._load_cfg()},
    "prompt": lambda params: {"prompt": srv._load_cfg().get("custom_prompt", "")},
    "skin": lambda params: {"value": _display_raw().get("skin", "default")},
    # Normalised like the TUI renders it (frontend falls back to the default for the same inputs).
    "indicator": lambda params: {
        "value": _display_word("tui_status_indicator", DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES)},
    "personality": _cfg_get_personality,
    "reasoning": _cfg_get_reasoning,
    "fast": _cfg_get_fast,
    "busy": lambda params: {"value": srv._load_busy_input_mode()},
    "approval_mode": lambda params: {"value": srv._load_approval_mode()},
    "approvals.mode": lambda params: {"value": srv._load_approval_mode()},
    "details_mode": lambda params: {"value": _display_word("details_mode", "collapsed", srv._DETAIL_MODES)},
    "thinking_mode": _cfg_get_thinking_mode,
    "density": lambda params: {"value": "on" if bool(_display_raw().get("tui_compact", False)) else "off"},
    "theme": lambda params: {"value": _display_word("tui_theme", "auto", {"auto", "light", "dark"})},
    "statusbar": lambda params: {"value": srv._coerce_statusbar(srv._display_cfg().get("tui_statusbar", "top"))},
    "focus": lambda params: {"value": "on" if bool(srv._display_cfg().get("focus_view", False)) else "off",
                             "tool_progress": srv._load_tool_progress_mode()},
    "mouse": lambda params: {"value": srv._display_mouse_tracking(srv._load_cfg().get("display"))},
    "mtime": _cfg_get_mtime}
# Getters whose failure is a JSON-RPC error of this code (others propagate to dispatch).
_CONFIG_GET_ERR = {"provider": 5013, "approval_mode": 5001, "approvals.mode": 5001}


@method("config.get")
@_profile_scoped
def _(rid, params: ConfigGetParams) -> ConfigGetResult | dict:
    key = params.key
    getter = srv._CONFIG_GETTERS.get(key)
    if getter is None:
        return srv._err(rid, 4002, f"unknown config key: {key}")
    try:
        return ConfigGetResult(**getter(params))
    except Exception as e:
        if key not in srv._CONFIG_GET_ERR:
            raise
        return srv._err(rid, srv._CONFIG_GET_ERR[key], str(e))


# ── setup readiness

def _readiness_check(rid, params, probe, result_type):
    """Run a readiness probe in the optional profile's runtime scope."""
    import contextlib

    profile = str(params.profile or "").strip()
    scope = contextlib.nullcontext()
    if profile:
        from hermes_cli import profiles as profiles_mod
        if not profiles_mod.profile_exists(profile):
            return result_type(ok=False, profile=params.profile,
                               error=f"Profile '{profile}' does not exist on this backend.")
        home = srv._profile_home(profile)
        if home is not None:
            scope = srv._session_profile_runtime_scope({"profile_home": str(home)})
    with scope:
        return result_type(**probe(profile, {"profile": profile} if profile else {}))


@method("setup.status")
def _(rid, params: Params) -> SetupStatusResult | dict:
    """Loose provider check; ``profile`` (optional) scopes it to that profile's home.

    For the launch profile the answer is the boot bootstrap's record (``free_tier_bootstrap``):
    the call blocks up to ``SETUP_READY_WAIT_SECONDS`` for it, so a client's first poll lands after
    the free-tier identity exists (or has been refused) rather than racing the mint. If the record
    is still missing after the wait, or a named profile is asked about, today's live probe answers.
    The record's fields ride along additively (``ready``, ``free_tier``, ``other_providers``)."""
    try:
        from hermes_cli.main import _has_any_provider_configured
        from hermes_cli.free_tier_bootstrap import wait_for_record

        def probe(profile, scoped):
            record = None if profile else wait_for_record()
            if record is None:
                return {"provider_configured": bool(_has_any_provider_configured(strict_profile_scope=bool(profile))),
                        **scoped}
            # ``failure_fields`` rides along only when the free-tier mint did not happen: the code,
            # the sentence, and whether / when a retry can succeed (``free_tier.provision``).
            return {"provider_configured": record.provider_configured, "ready": True,
                    "free_tier": record.free_tier, "other_providers": record.other_providers,
                    "inference_provider": record.inference_provider, **record.failure_fields(), **scoped}
        return srv._readiness_check(rid, params, probe, SetupStatusResult)
    except Exception as e:
        return srv._err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: SetupRuntimeCheckParams) -> SetupRuntimeCheckResult | dict:
    """Strict provider check via the same resolve_runtime_provider() the agent uses on session
    creation (setup.status is True if ANY provider auth state is discoverable): ok=False + the auth
    error when the model can't be served, so UIs surface onboarding before a doomed prompt.
    ``profile`` answers for THAT profile's pin and ``.env``; unknown -> ``ok=False``."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured
        requested = str(params.provider or "").strip() or None

        def probe(profile, scoped):
            runtime = resolve_runtime_provider(requested=requested)
            provider_configured = bool(_has_any_provider_configured(strict_profile_scope=bool(profile)))
            provider = runtime.get("provider") or "provider"
            source = str(runtime.get("source") or "")

            def fail(error, src):
                return {"ok": False, "provider": provider, "model": runtime.get("model"),
                        "source": src, "error": error, **scoped}
            if (not provider_configured and provider == "bedrock"
                    and source in {"iam-role", "aws-sdk-default-chain"}):
                return fail("No Hermes provider is configured.", source)
            api_key = runtime.get("api_key")
            api_key_text = "" if callable(api_key) else str(api_key or "").strip()
            if not (callable(api_key) or api_key_text in {"aws-sdk", "no-key-required"}
                    or has_usable_secret(api_key_text) or bool(runtime.get("command"))):
                return fail(f"No usable credentials found for {provider}.", runtime.get("source"))
            from hermes_cli.anon_auth import route_is_welcome_host
            # free_tier is keyed on the SELECTED route (the welcome host serves only nous/welcome), not
            # on profile state: a paid Nous key beside a free-tier identity must not read as free.
            return {"ok": True, "provider": runtime.get("provider"), "model": runtime.get("model"),
                    "source": runtime.get("source"),
                    "free_tier": provider == "nous" and route_is_welcome_host(runtime.get("base_url")),
                    **scoped}
        return srv._readiness_check(rid, params, probe, SetupRuntimeCheckResult)
    except Exception as e:
        return SetupRuntimeCheckResult(ok=False, error=str(e))


def _safe_client_label(label: str) -> str:
    """Alnum/._- () only, ≤64 chars, dot-runs and leading dots collapsed (no traversal shapes)."""
    safe = "".join(ch for ch in label if ch.isalnum() or ch in "._- ()").strip()[:64]
    while ".." in safe:
        safe = safe.replace("..", ".")
    return safe.lstrip(".").strip()


@method("diagnostics.share_nous")
def _(rid, params: DiagnosticsShareNousParams) -> DiagnosticsShareNousResult | dict:
    """Upload a redacted debug bundle to Nous-internal diagnostics storage — same collection +
    force-redaction pipeline as ``hermes debug share --nous``; redaction is NOT client-controllable
    and consent lives with the CALLER (privacy notice first). Structured ``ok``/``error`` envelope so
    upload failures render inline. Optional: ``error_context`` (-> ``error-context.txt``),
    ``extra_files`` ({label -> text}), ``log_lines`` (default 200); all force-redacted."""
    try:
        from hermes_cli.debug import _redact_log_text, build_nous_bundle, collect_share_bundle
        from hermes_cli.diagnostics_upload import share_to_nous
        log_lines = params.log_lines
        if log_lines is None or not (10 <= log_lines <= 2000):
            log_lines = 200
        bundle = collect_share_bundle(log_lines=log_lines, redact=True)
        # Client text goes through the SAME upload-safe redactor as backend logs (force secret
        # redaction + email masking), never the weaker bare secret pass.
        error_context = params.error_context
        if error_context and error_context.strip():
            bundle["error-context.txt"] = _redact_log_text(error_context.strip()[:8_000])
        # Bounded: at most 4 files, 512KB each, sanitized labels — not an arbitrary upload surface.
        for label, text in list((params.extra_files or {}).items())[:4]:
            safe_label = srv._safe_client_label(label) if isinstance(label, str) else ""
            if safe_label and isinstance(text, str) and text.strip():
                bundle[f"client/{safe_label}"] = _redact_log_text(text[:524_288])
        res = share_to_nous(build_nous_bundle(bundle, redact=True))
        view_url = res.get("viewUrl") or res.get("view_url")
        upload_id = res.get("id")
        if not view_url and not upload_id:  # an upload the user can't reference is useless to support
            return DiagnosticsShareNousResult(ok=False, error="upload succeeded but returned no view URL or id")
        return DiagnosticsShareNousResult(
            ok=True, view_url=view_url, upload_id=upload_id,
            expires_at=res.get("expiresAt") or res.get("expires_at"))
    except Exception as e:
        return DiagnosticsShareNousResult(ok=False, error=str(e))


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server`` and install its handlers."""
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
