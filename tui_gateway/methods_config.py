"""Config / projects / setup JSON-RPC handlers. Bodies are rebound onto server.py's globals
(method_ctx.bind_module) and reference them bare. ``config.set`` lives in methods_config_set.
"""

import atexit
import concurrent.futures
import threading

from .method_ctx import HandlerRegistry, bind_module
from ._env import env_int

from hermes_constants import DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES
from hermes_constants import display_hermes_home as _display_hermes_home

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── setup readiness single-flight (#65151) ─────────────────────────────────
#
# Readiness probes are Desktop-polled and execute on the shared RPC pool via
# ``_LONG_HANDLERS``. A slow probe (blocked keyring, OAuth refresh, GIL pressure)
# used to run one provider-resolution call per overlapping poll, each occupying
# another shared worker while they all resolved the same state. The probe now
# runs on this small dedicated executor, single-flighted per
# ``(kind, profile, requested provider)``:
#
# * the first caller submits the probe and waits a bounded budget;
# * an overlapping poll for a still-running probe waits a short join grace for
#   it — the Desktop fires setup.status + setup.runtime_check from independent
#   consumers at the same seam (boot, the post-assignment ``setup.ready``
#   broadcast), and answering the retryable error AT ONCE made both legs of
#   one consumer transiently unknown, which the onboarding gate read as
#   not-ready and the overlay never closed. Fast (config-read) probes settle
#   well inside the grace, so an overlapping poll reads the shared result; a
#   probe still running after it answers a retryable error instead —
#   a JSON-RPC error, never a fabricated ``ok`` (the result contract requires
#   the real shape, and the desktop already treats an errored runtime_check as
#   unknown, keeping setup.status authoritative);
# * a probe that outlives the budget answers the same retryable error while it
#   keeps running in the background; the in-flight entry is cleared when it
#   settles, so the next poll starts a fresh probe and never reads a stale one.
_readiness_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, min(4, env_int("HERMES_TUI_RPC_POOL_WORKERS", 8))),
    thread_name_prefix="tui-readiness")
atexit.register(lambda: _readiness_pool.shutdown(wait=False, cancel_futures=True))
_readiness_lock = threading.Lock()
_readiness_inflight: dict[tuple, concurrent.futures.Future] = {}
_READINESS_SHARE_WAIT_SECONDS = 4.0
# Join grace for an overlapping poll on a still-running probe: fast (config-read)
# probes settle in milliseconds, so an overlapping consumer at the same seam
# (statusbar + onboarding, both polling setup.status / setup.runtime_check) reads
# the shared result instead of an unknown-readiness error pair — which the
# onboarding gate read as not-ready and the blocking overlay never closed.
# Kept well under a second: a joiner occupies its shared RPC worker for at most
# the grace, so overlapping polls still cannot starve the pool (#65151).
_READINESS_JOIN_GRACE_SECONDS = 0.5
# setup.status's probe legitimately blocks on the boot bootstrap's record
# (free_tier_bootstrap.SETUP_READY_WAIT_SECONDS = 8s): its join budget must
# cover that wait or every boot poll would answer the retryable error.
_READINESS_STATUS_SHARE_WAIT_SECONDS = 12.0
# Retryable-transient code for "the probe is still running / outlived its
# budget"; the client treats an errored runtime_check as unknown readiness.
_READINESS_IN_PROGRESS_ERR = 5097


def _projects_handler(name: str):
    """``@method(name)`` (profile-scoped) whose body's uncaught exception becomes ``_err(rid, 5061)``."""
    def deco(fn):
        def handler(rid, params: dict) -> dict:
            try:
                return fn(rid, params)
            except Exception as e:
                return _err(rid, 5061, str(e))
        return method(name)(_profile_scoped(handler))
    return deco


def _reconcile_repo_discovery(pdb, conn, policy, policy_key):
    pdb.reconcile_discovered_repos_policy(conn, policy_key,
                                          preserve_unversioned=_repo_discovery_policy_is_default(policy))


@_projects_handler("projects.discover_repos")
def _(rid, params: dict) -> dict:
    """Repos for the desktop overview: scanned-from-disk (cached) ∪ session-derived."""
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"repos": []})
        from hermes_cli import projects_db as pdb
        policy = _repo_discovery_policy()
        with pdb.connect_closing() as conn:
            _reconcile_repo_discovery(pdb, conn, policy, _repo_discovery_policy_key(policy))
            # `scan=true` (remote-gateway desktop): its native scan only sees its own filesystem,
            # so the host scans the policy roots so zero-session repos surface.
            # See #81723.
            if params.get("scan") and policy["enabled"]:
                _scan_discovered_repos_remote(conn, policy)
            repos = _discover_repos_payload(db, conn=conn, include_cached=policy["enabled"])
        return _ok(rid, {"repos": repos, "discovery_policy": policy})


@_projects_handler("projects.record_repos")
def _(rid, params: dict) -> dict:
    """Persist repo roots found by the client's (desktop-side) scan; return the merged list."""
    from hermes_cli import projects_db as pdb
    policy = _repo_discovery_policy()
    policy_key = _repo_discovery_policy_key(policy)
    incoming = params.get("discovery_policy")
    if isinstance(incoming, dict):
        accepted = _repo_discovery_policy_key(_repo_discovery_policy(incoming)) == policy_key
    else:
        accepted = _repo_discovery_policy_is_default(policy)  # legacy client without a policy
    accepted = bool(policy["enabled"] and accepted)
    pairs = [(item, None) if isinstance(item, str) else (str(item["root"]), item.get("label"))
             for item in params.get("repos") or []
             if isinstance(item, str) or (isinstance(item, dict) and item.get("root"))]
    with pdb.connect_closing() as conn:
        _reconcile_repo_discovery(pdb, conn, policy, policy_key)
        if accepted:
            pdb.record_discovered_repos(conn, pairs, replace=True, policy_key=policy_key)
        elif not policy["enabled"]:
            pdb.clear_discovered_repos(conn, policy_key=policy_key)
    with _profile_db(params) as db:
        repos = [] if db is None else _discover_repos_payload(db, include_cached=policy["enabled"])
        return _ok(rid, {"repos": repos, "accepted": accepted, "discovery_policy": policy})


def _stamped_project_tree(db, params, **kwargs):
    """``_build_project_tree`` + profile stamping shared by the two tree RPCs."""
    from tui_gateway.project_tree import stamp_profile
    tree, active_id = _build_project_tree(db, **kwargs)
    stamp_profile(tree["projects"], _response_profile_name(params.get("profile")))
    return tree, active_id


@_projects_handler("projects.tree")
def _(rid, params: dict) -> dict:
    """Project -> repo -> lane overview with counts + a few preview sessions per project, plus the
    flat set of session ids claimed by any project (excluded from flat Recents). Lanes carry no
    session rows; drill-in uses ``projects.project_sessions``."""
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"projects": [], "active_id": None, "scoped_session_ids": []})
        tree, active_id = _stamped_project_tree(
            db, params, preview_limit=int(params.get("preview_limit") or 3), hydrate=False,
            session_limit=int(params.get("session_limit") or 2000), include_discovered=True)
        return _ok(rid, {"projects": tree["projects"], "active_id": active_id,
                         "scoped_session_ids": tree["scoped_session_ids"]})


@_projects_handler("projects.project_sessions")
def _(rid, params: dict) -> dict:
    """Fully hydrated lanes for one project, from the same grouping as ``projects.tree``."""
    project_id = str(params.get("project_id") or "")
    if not project_id:
        return _err(rid, 5063, "project_id required")
    with _profile_db(params) as db:
        if db is None:
            return _ok(rid, {"project": None})
        # Drill-in only needs the entered project: skip the zero-session discovery tier.
        tree, _active = _stamped_project_tree(
            db, params, preview_limit=0, hydrate=True,
            session_limit=int(params.get("session_limit") or 5000), include_discovered=False)
        return _ok(rid, {"project": next((p for p in tree["projects"] if p["id"] == project_id), None)})


# ── config.get — one getter per key returning the result payload.

def _display_raw() -> dict:
    return _load_cfg().get("display") or {}


def _display_word(key: str, default: str, allowed) -> str:
    """Normalised ``display.<key>``; unknown/garbage values read back as ``default``."""
    raw = str(_display_raw().get(key, default) or "").strip().lower()
    return raw if raw in allowed else default


_THINKING_MODES = frozenset({"collapsed", "truncated", "full"})


def _cfg_get_provider(params):
    from hermes_cli.models import list_available_providers, normalize_provider
    model = _resolve_model()
    parts = model.split("/", 1)
    return {"model": model, "provider": normalize_provider(parts[0]) if len(parts) > 1 else "unknown",
            "providers": list_available_providers()}


def _cfg_get_project(params):
    raw = str(params.get("cwd", "") or (_load_cfg().get("terminal") or {}).get("cwd", "") or "").strip()
    cwd = _completion_cwd({"cwd": raw} if raw else {})
    return {"cwd": cwd, "branch": git_probe.branch(cwd)}


def _cfg_get_personality(params):
    # EFFECTIVE personality via the single owner — a stale/unknown name must not show as active.
    from hermes_cli.personality import active_personality_name
    return {"value": active_personality_name(_load_cfg()) or "none"}


def _cfg_get_reasoning(params):
    cfg = _load_cfg()
    session = _sessions.get(params.get("session_id", "")) or {}
    reasoning_config = session.get("create_reasoning_override")
    if session and not isinstance(reasoning_config, dict):
        reasoning_config = getattr(session.get("agent"), "reasoning_config", None)
    if isinstance(reasoning_config, dict):
        enabled = reasoning_config.get("enabled") is not False
        effort = str(reasoning_config.get("effort") or "medium") if enabled else "none"
    else:
        raw_effort = (cfg.get("agent") or {}).get("reasoning_effort", "")
        if isinstance(raw_effort, dict):  # {enabled, effort} form: render the tier, never str(dict)
            from hermes_constants import parse_reasoning_effort
            parsed = parse_reasoning_effort(raw_effort) or {}
            raw_effort = False if parsed.get("enabled") is False else parsed.get("effort")
        # YAML `reasoning_effort: false` means thinking disabled, not "unset".
        effort = "none" if raw_effort is False else str(raw_effort or "medium")
    display = "show" if (cfg.get("display") or {}).get("show_reasoning", True) else "hide"
    return {"value": effort, "display": display}


def _cfg_get_fast(params):
    # `config.set fast` is session-scoped: prefer the session's live/pinned value over the
    # global key (a pre-build session keeps its pin in create_service_tier_override).
    session = _sessions.get(params.get("session_id", "")) or {}
    agent = session.get("agent")
    tier = (getattr(agent, "service_tier", None) if agent is not None
            else session.get("create_service_tier_override"))
    if tier is None:
        tier = _load_service_tier()
    return {"value": "fast" if tier == "priority" else "normal"}


def _cfg_get_thinking_mode(params):
    raw = _display_word("thinking_mode", "", _THINKING_MODES)
    if not raw:  # legacy details_mode fallback
        raw = "full" if _display_word("details_mode", "collapsed", _DETAIL_MODES) == "expanded" else "collapsed"
    return {"value": raw}


def _cfg_get_mtime(params):
    cfg_path = _hermes_home / "config.yaml"
    try:
        mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0
    except Exception:
        return {"mtime": 0}
    # mcp_rev: hash of the MCP-relevant sections so the poller reloads MCP servers only when
    # their config changed — a /skin write bumps mtime but must not cost an MCP reconnect.
    return {"mtime": mtime, "mcp_rev": _compute_mcp_rev()}


# key -> getter(params); bind_module rebinds the table's functions onto server.py's globals.
_CONFIG_GETTERS = {
    "provider": _cfg_get_provider,
    "profile": lambda params: {"home": str(_hermes_home), "display": _display_hermes_home()},
    "project": _cfg_get_project,
    "full": lambda params: {"config": _load_cfg()},
    "prompt": lambda params: {"prompt": _load_cfg().get("custom_prompt", "")},
    "skin": lambda params: {"value": _display_raw().get("skin", "default")},
    # Normalised like the TUI renders it (frontend falls back to the default for the same inputs).
    "indicator": lambda params: {
        "value": _display_word("tui_status_indicator", DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES)},
    "personality": _cfg_get_personality,
    "reasoning": _cfg_get_reasoning,
    "fast": _cfg_get_fast,
    "busy": lambda params: {"value": _load_busy_input_mode()},
    "approval_mode": lambda params: {"value": _load_approval_mode()},
    "approvals.mode": lambda params: {"value": _load_approval_mode()},
    "details_mode": lambda params: {"value": _display_word("details_mode", "collapsed", _DETAIL_MODES)},
    "thinking_mode": _cfg_get_thinking_mode,
    "density": lambda params: {"value": "on" if bool(_display_raw().get("tui_compact", False)) else "off"},
    "theme": lambda params: {"value": _display_word("tui_theme", "auto", {"auto", "light", "dark"})},
    "statusbar": lambda params: {"value": _coerce_statusbar(_display_cfg().get("tui_statusbar", "top"))},
    "focus": lambda params: {"value": "on" if bool(_display_cfg().get("focus_view", False)) else "off",
                             "tool_progress": _load_tool_progress_mode()},
    "mouse": lambda params: {"value": _display_mouse_tracking(_load_cfg().get("display"))},
    "mtime": _cfg_get_mtime}
# Getters whose failure is a JSON-RPC error of this code (others propagate to dispatch).
_CONFIG_GET_ERR = {"provider": 5013, "approval_mode": 5001, "approvals.mode": 5001}


@method("config.get")
@_profile_scoped
def _(rid, params: dict) -> dict:
    key = params.get("key", "")
    getter = _CONFIG_GETTERS.get(key)
    if getter is None:
        return _err(rid, 4002, f"unknown config key: {key}")
    try:
        return _ok(rid, getter(params))
    except Exception as e:
        if key not in _CONFIG_GET_ERR:
            raise
        return _err(rid, _CONFIG_GET_ERR[key], str(e))


# ── setup readiness

def _readiness_cleared(key):
    """Done-callback for a single-flighted probe: forget the entry (the next poll
    re-probes — completed answers are never cached), and surface a failure nobody
    waited for (every caller timed out) in the log instead of dropping it."""
    def _clear(future):
        with _readiness_lock:
            if _readiness_inflight.get(key) is future:
                _readiness_inflight.pop(key, None)
        if not future.cancelled() and future.exception() is not None:
            logger.debug("readiness probe %s failed after its callers returned: %s",
                         key, future.exception())
    return _clear


def _readiness_share(rid, key, run_probe, wait_seconds):
    """Run ``run_probe`` single-flighted under ``key`` on the dedicated readiness pool.

    The first caller submits the probe and waits up to ``wait_seconds``; a caller that finds a
    still-running probe waits the short join grace for it first — the Desktop fires
    setup.status + setup.runtime_check from independent consumers at the same seam (boot, the
    post-assignment ``setup.ready`` broadcast), and answering the retryable error at once made
    both legs of one consumer transiently unknown, which its onboarding gate read as not-ready.
    Fast (config-read) probes settle well inside the grace, so an overlapping poll reads the
    shared result; one that outlives the grace answers a retryable error, freeing its shared RPC
    worker (the probe keeps running for the first caller). A probe that outlives ``wait_seconds``
    answers the same retryable error while it continues in the background."""
    with _readiness_lock:
        future = _readiness_inflight.get(key)
        owner = future is None
        if owner:
            future = _readiness_pool.submit(run_probe)
            _readiness_inflight[key] = future
    if owner:
        # Registered OUTSIDE the lock: a probe that already settled runs the
        # callback inline on this thread, and the callback acquires the same
        # non-reentrant lock — under the lock that self-deadlocks the RPC
        # worker and every later readiness call blocks on it (the gateway
        # hangs answering setup.status / setup.runtime_check at all).
        future.add_done_callback(_readiness_cleared(key))
    try:
        return _ok(rid, future.result(timeout=wait_seconds if owner else
                                      min(wait_seconds, _READINESS_JOIN_GRACE_SECONDS)))
    except concurrent.futures.TimeoutError:
        logger.warning("readiness probe %s exceeded its budget (%.1fs); it continues in the background",
                       key, wait_seconds)
        return _err(rid, _READINESS_IN_PROGRESS_ERR,
                    "readiness check still in progress; retrying next tick")


def _readiness_check(rid, params, probe, *, probe_key, wait_seconds):
    """Shared shell of setup.status / setup.runtime_check. ``probe(profile, scoped)`` runs inside the
    optional ``profile`` param's HERMES_HOME + ``.env`` secret scope (ContextVars: concurrent checks
    stay isolated); ``scoped`` is the ``{"profile": ...}`` payload stamp (``{}`` for the launch
    profile). An unknown profile answers ``ok=False`` (never a JSON-RPC error, never a quiet answer
    for the launch profile instead). ``probe_key`` + the profile single-flight the probe, and
    ``wait_seconds`` bounds how long this shared-RPC worker waits for it (#65151)."""
    profile = str(params.get("profile") or "").strip() if isinstance(params, dict) else ""
    home = None
    if profile:
        from hermes_cli import profiles as profiles_mod
        if not profiles_mod.profile_exists(profile):
            return _ok(rid, {"ok": False, "profile": params.get("profile"),
                             "error": f"Profile '{profile}' does not exist on this backend."})
        home = _profile_home(profile)
    # ``profile_home=None`` is the launch profile: once this process multiplexes its probe must
    # run under its own frozen secret scope too (``_profile_runtime_scope_tokens`` binds nothing in
    # a single-profile process), or the first profile-scoped read inside the resolver
    # (``HERMES_CODEX_BASE_URL`` for openai-codex) fails closed and the UI shows onboarding.
    def run_probe():
        # Applied on the readiness pool thread: ContextVars do not cross threads, and
        # concurrent probes (different profiles) stay isolated exactly as they did when
        # each ran on its caller's handler thread.
        with _session_profile_runtime_scope({"profile_home": str(home) if home is not None else None}):
            return probe(profile, {"profile": profile} if profile else {})

    return _readiness_share(rid, (probe_key, profile), run_probe, wait_seconds)


@method("setup.status")
def _(rid, params: dict) -> dict:
    """Loose provider check; ``profile`` (optional) scopes it to that profile's home.

    For the launch profile the answer is the boot bootstrap's record (``free_tier_bootstrap``):
    the call blocks up to ``SETUP_READY_WAIT_SECONDS`` for it, so a client's first poll lands after
    the free-tier identity exists (or has been refused) rather than racing the mint. A record that
    says ``False`` is reconciled with the config files first (``reconcile_record``): a provider
    added after boot — the Models page, a picker key, ``hermes setup`` from a shell — flips it
    without a restart. If the record
    is still missing after the wait, or a named profile is asked about, today's live probe answers.
    The record's fields ride along additively (``ready``, ``free_tier``, ``other_providers``)."""
    try:
        from hermes_cli.main import _has_any_provider_configured
        from hermes_cli.free_tier_bootstrap import wait_for_record

        def probe(profile, scoped):
            record = None if profile else wait_for_record()
            if record is None:
                # ``ready`` = this process's boot bootstrap has settled (a named profile has no
                # record of its own; the launch record says whether the free tier is minted).
                # Since one host backend serves every profile (#118246), the desktop's
                # setup-profile probe lands here, and its kickoff requires ``ready``.
                launch = wait_for_record() if profile else None
                return {"provider_configured": bool(_has_any_provider_configured(strict_profile_scope=bool(profile))),
                        **({"ready": True, "free_tier": launch.free_tier} if launch is not None else {}),
                        **scoped}
            # ``failure_fields`` rides along only when the free-tier mint did not happen: the code,
            # the sentence, and whether / when a retry can succeed (``free_tier.provision``).
            return {"provider_configured": record.provider_configured, "ready": True,
                    "free_tier": record.free_tier, "other_providers": record.other_providers,
                    "inference_provider": record.inference_provider, **record.failure_fields(), **scoped}
        return _readiness_check(rid, params, probe, probe_key="status",
                                wait_seconds=_READINESS_STATUS_SHARE_WAIT_SECONDS)
    except Exception as e:
        return _err(rid, 5016, str(e))


@method("setup.runtime_check")
def _(rid, params: dict) -> dict:
    """Readiness probe for the session a client is about to open (setup.status is True if ANY
    provider auth state is discoverable): ok=False + the auth error when the model can't be served,
    so UIs surface onboarding before a doomed prompt. Without ``provider`` it runs the SAME
    resolver as session creation (``_resolve_agent_model_runtime``: startup model + provider pin,
    then the configured fallback chain) — a probe that ignores the chain shows onboarding for a
    backend whose sessions build fine. An explicit ``provider`` stays a strict single-provider
    check so onboarding can verify the provider just connected without another provider's
    fallback masking a failed connection. ``profile`` answers for THAT profile's pin and ``.env``;
    unknown -> ``ok=False``."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import has_usable_secret
        from hermes_cli.main import _has_any_provider_configured
        requested = str(params.get("provider") or "").strip() or None

        def probe(profile, scoped):
            if requested:
                model, _startup_provider = _resolve_startup_runtime()
                runtime = resolve_runtime_provider(requested=requested, target_model=model or None)
            else:
                model, runtime = _resolve_agent_model_runtime(None, None)
            provider_configured = bool(_has_any_provider_configured(strict_profile_scope=bool(profile)))
            provider = runtime.get("provider") or "provider"
            source = str(runtime.get("source") or "")

            def fail(error, src):
                return {"ok": False, "provider": provider, "model": model,
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
            return {"ok": True, "provider": runtime.get("provider"), "model": model,
                    "source": runtime.get("source"),
                    "free_tier": provider == "nous" and route_is_welcome_host(runtime.get("base_url")),
                    **scoped}
        return _readiness_check(rid, params, probe, probe_key=f"runtime:{requested or ''}",
                                wait_seconds=_READINESS_SHARE_WAIT_SECONDS)
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


def _safe_client_label(label: str) -> str:
    """Alnum/._- () only, ≤64 chars, dot-runs and leading dots collapsed (no traversal shapes)."""
    safe = "".join(ch for ch in label if ch.isalnum() or ch in "._- ()").strip()[:64]
    while ".." in safe:
        safe = safe.replace("..", ".")
    return safe.lstrip(".").strip()


@method("diagnostics.share_nous")
def _(rid, params: dict) -> dict:
    """Upload a redacted debug bundle to Nous-internal diagnostics storage — same collection +
    force-redaction pipeline as ``hermes debug share --nous``; redaction is NOT client-controllable
    and consent lives with the CALLER (privacy notice first). Structured ``ok``/``error`` envelope so
    upload failures render inline. Optional: ``error_context`` (-> ``error-context.txt``),
    ``extra_files`` ({label -> text}), ``log_lines`` (default 200); all force-redacted."""
    try:
        from hermes_cli.debug import _redact_log_text, build_nous_bundle, collect_share_bundle
        from hermes_cli.diagnostics_upload import share_to_nous
        log_lines = params.get("log_lines")
        if not isinstance(log_lines, int) or not (10 <= log_lines <= 2000):
            log_lines = 200
        bundle = collect_share_bundle(log_lines=log_lines, redact=True)
        # Client text goes through the SAME upload-safe redactor as backend logs (force secret
        # redaction + email masking), never the weaker bare secret pass.
        error_context = params.get("error_context")
        if isinstance(error_context, str) and error_context.strip():
            bundle["error-context.txt"] = _redact_log_text(error_context.strip()[:8_000])
        # Bounded: at most 4 files, 512KB each, sanitized labels — not an arbitrary upload surface.
        extra_files = params.get("extra_files")
        for label, text in list(extra_files.items())[:4] if isinstance(extra_files, dict) else ():
            safe_label = _safe_client_label(label) if isinstance(label, str) else ""
            if safe_label and isinstance(text, str) and text.strip():
                bundle[f"client/{safe_label}"] = _redact_log_text(text[:524_288])
        res = share_to_nous(build_nous_bundle(bundle, redact=True))
        view_url = res.get("viewUrl") or res.get("view_url")
        upload_id = res.get("id")
        if not view_url and not upload_id:  # an upload the user can't reference is useless to support
            return _ok(rid, {"ok": False, "error": "upload succeeded but returned no view URL or id"})
        return _ok(rid, {"ok": True, "view_url": view_url, "upload_id": upload_id,
                         "expires_at": res.get("expiresAt") or res.get("expires_at")})
    except Exception as e:
        return _ok(rid, {"ok": False, "error": str(e)})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
