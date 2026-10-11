"""Session working-directory + durable session row: cwd resolution/healing, session.db row ensure,
branch seed, history rewind, git meta persistence. Bodies are rebound onto server.py's globals at
install time (method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib
from typing import Any

from tui_gateway import git_probe

from .method_ctx import bind_module


def _normalize_completion_path(path_part: str) -> str:
    expanded = os.path.expanduser(path_part)
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return expanded


def _completion_cwd(params: dict | None = None) -> str:
    params = params or {}
    # Provenance for the client-sent ``cwd`` (#52589): the desktop seeds a new chat's cwd
    # from its app-global workspace (the launch profile's configured directory or the
    # project scope) when the user did NOT pick one. That inherited default must NOT
    # override a NAMED profile's own ``terminal.cwd`` — only a deliberate per-session
    # workspace pick (``cwd_explicit``) wins over the profile config. Path equality
    # cannot tell the two apart, so the desktop ships the flag alongside the path.
    client_cwd = params.get("cwd")
    profile = params.get("profile")
    session_cwd = _sessions.get(params.get("session_id") or "", {}).get("cwd")
    try:
        profile_home = _profile_home(profile) if profile else None
    except ProfileUnavailableError:
        # Main only resolved the profile when it consulted the profile's config; keep a deleted one fatal exactly there.
        if (client_cwd and not params.get("cwd_explicit")) or not (client_cwd or session_cwd):
            raise
        profile_home = None
    if not params.get("cwd_explicit") and client_cwd:
        if profile_cwd := _profile_workspace_cwd(profile_home):
            return profile_cwd
    # A session bound to another profile resolves its workspace from THAT profile's config before the launch profile's
    # env var; the dashboard's in-memory gateway does NOT inherit the PTY child's bridged TERMINAL_CWD, so a configured
    # terminal.cwd is read directly.
    named_ssh = profile_home is not None and _cwd_is_remote(profile_home)
    # A NAMED profile with no configured workspace (placeholder/unset terminal.cwd) never inherits
    # the LAUNCH profile's cwd (#87584): the Desktop stamps the app-global workspace into every
    # pooled backend's TERMINAL_CWD, and _launch_configured_cwd()/that env var hold the launch
    # profile's value — a session for another profile would land in the wrong workspace. Its own
    # home is the same default its standalone gateway would use (placeholder → $HOME).
    named_local_default = (
        str(profile_home) if profile_home is not None and not named_ssh and not client_cwd and not session_cwd else None
    )
    raw = str(client_cwd or session_cwd or _profile_workspace_cwd(profile_home)
              # A named ssh profile never inherits the LAUNCH profile's host cwd: its remote default is ~.
              or ("~" if named_ssh else "") or named_local_default or _launch_configured_cwd()
              or os.environ.get("TERMINAL_CWD") or _sandbox_workspace_cwd(None) or os.getcwd())
    # An ssh cwd lives on the remote host: host expansion/isdir cannot vouch for it, and ``~`` names the REMOTE
    # user's home, never this host's. The launch profile keeps main's host fast path for everything else.
    if named_ssh:
        return raw if _is_remote_cwd_shape(raw) else (_declared_remote_profile_cwd(profile_home) or "~")
    if profile_home is None and raw.startswith("~") and _cwd_is_remote(None):
        return raw
    with contextlib.suppress(Exception):
        resolved = os.path.abspath(os.path.expanduser(raw))
        if os.path.isdir(resolved):
            return resolved
    if profile_home is None and _is_remote_cwd_shape(raw) and _cwd_is_remote(None):
        return raw
    # A container backend's cwd (docker ``/workspace``) lives inside the sandbox, like ``_terminal_task_cwd``'s: the
    # host has no such dir. Falling back to the gateway's own cwd ($HOME for the desktop) made every host file look
    # "inside the workspace", so attachments were never staged into the mounted dir (#103147).
    if _is_container_path(raw) and _bound_terminal_backend(profile_home) != "local":
        return raw
    return os.getcwd()


def _workdir_terminal_cfg(key: str) -> str:
    """Stripped ``terminal.<key>`` from config, or "" when unset/unreadable."""
    with contextlib.suppress(Exception):
        terminal_cfg = _load_cfg().get("terminal", {})
        if isinstance(terminal_cfg, dict):
            return str(terminal_cfg.get(key) or "").strip()
    return ""


def _profile_terminal_policy(profile_home) -> dict:
    """A named profile's effective ``TERMINAL_*`` policy — the one its turns run under
    (``build_profile_terminal_scope``: defaults <- ``.env`` <- ``config.yaml``). {} for the launch profile or an
    unreadable profile."""
    if not profile_home:
        return {}
    from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope
    try:
        return build_profile_terminal_scope(Path(profile_home))
    except TerminalPolicyUnavailable:
        return {}


def _policy_backend(policy: dict) -> str:
    return str(policy.get("TERMINAL_ENV") or "").strip().lower() or "local"


def _bound_terminal_backend(profile_home) -> str:
    """Terminal backend of the profile a session/RPC is bound to. A named profile reads ITS policy: at
    ``session.create`` the multiplex gateway has not rebound HERMES_HOME yet, so ``_effective_terminal_backend()``
    would report the LAUNCH profile's backend, which must never leak into a named one."""
    return _policy_backend(_profile_terminal_policy(profile_home)) if profile_home else _effective_terminal_backend()


def _cwd_is_remote(profile_home) -> bool:
    """Whether the bound profile's working directory lives on another host (ssh), so a host ``isdir`` check cannot
    vouch for it. Docker and the other backends mount or copy HOST paths and keep the host checks."""
    return _bound_terminal_backend(profile_home) == "ssh"


def _declared_remote_profile_cwd(profile_home) -> str | None:
    """A named ssh profile's own ``terminal.cwd`` (``~``, ``~/…`` or absolute), unchecked against this host.

    ``_profile_configured_cwd`` requires ``os.path.isdir``; an ssh working directory lives on the remote, so that
    check drops it (or expands ``~`` to THIS host's home) and the launch profile's ``TERMINAL_CWD`` wins.
    """
    policy = _profile_terminal_policy(profile_home)
    raw = str(policy.get("TERMINAL_CWD") or "").strip()
    return raw if _policy_backend(policy) == "ssh" and _is_remote_cwd_shape(raw) else None


def _is_remote_cwd_shape(raw: str) -> bool:
    """An ssh working directory the remote shell can resolve: ``~``, ``~/…`` or absolute (a relative one would be
    stored and git-probed relative to the gateway's own cwd)."""
    from hermes_cli.config import _is_ssh_remote_tilde_cwd

    return _is_ssh_remote_tilde_cwd("ssh", raw) or os.path.isabs(raw)


def _is_container_path(raw: str) -> bool:
    """An absolute POSIX path, the only shape a container backend's working directory takes."""
    return raw.startswith("/") and not raw.startswith("//")


def _sandbox_workspace_cwd(profile_home) -> str | None:
    """A container backend's configured ``terminal.cwd`` (docker ``/workspace``), unchecked against this host.

    ``_profile_configured_cwd``/``_launch_configured_cwd`` require ``os.path.isdir``, which a path inside the
    sandbox fails. Only a path missing on the host qualifies: an existing host dir keeps the host path (it is the
    ``docker_mount_cwd_to_workspace`` source). ssh has its own remote-cwd rules above."""
    if profile_home:
        policy = _profile_terminal_policy(profile_home)
        backend, raw = _policy_backend(policy), str(policy.get("TERMINAL_CWD") or "").strip()
    else:
        backend = _effective_terminal_backend()
        raw = os.environ.get("TERMINAL_CWD", "").strip() or _workdir_terminal_cfg("cwd")
    if backend in {"local", "ssh"} or not _is_container_path(raw) or os.path.isdir(raw):
        return None
    return raw


def _profile_workspace_cwd(profile_home) -> str | None:
    """A named profile's configured workspace: an ssh profile's remote dir, a host dir, else a container dir."""
    return (_declared_remote_profile_cwd(profile_home) or _profile_configured_cwd(profile_home)
            or (_sandbox_workspace_cwd(profile_home) if profile_home else None))


def _workspace_cwd(profile_home, raw: str) -> str:
    """A picked workspace for a session bound to ``profile_home``: an ssh dir raw, else an existing host dir.
    Raises ValueError when a host dir does not exist."""
    if _cwd_is_remote(profile_home):
        if _is_remote_cwd_shape(raw):
            return raw
        raise ValueError(f"remote working directory must be absolute or ~-relative: {raw}")
    resolved = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {raw}")
    return resolved


def _terminal_task_cwd(session: dict | None) -> str:
    """The cwd terminal_tool should use for this TUI session (NOT host-validated: a non-local backend's cwd lives
    inside the target environment)."""
    return _terminal_task_cwd_with_source(session)[0]


def _terminal_task_cwd_with_source(session: dict | None) -> tuple[str, str]:
    """``(cwd, source)``: ``"session"`` for THIS session's workspace (``explicit_cwd``/tracked dir), ``"process"`` for
    the global ``TERMINAL_CWD``/``terminal.cwd`` fallback — under per-session docker isolation that is a PREVIOUS
    session's launch artifact, so terminal_tool refuses it as a bind-mount source."""
    profile_home = (session or {}).get("profile_home")
    # A named ssh profile's session is ssh whatever the launch process runs; other named backends keep main's
    # process-backend semantics (docker isolation's "process" vs "session" source depends on it).
    named_ssh = bool(profile_home) and _cwd_is_remote(profile_home)
    backend = "ssh" if named_ssh else _effective_terminal_backend()
    if backend != "local":
        # THIS session's explicit workspace beats the LAST session's env var.
        if session and session.get("explicit_cwd") and session.get("cwd"):
            return str(session["cwd"]), "session"
        # Process TERMINAL_CWD/terminal.cwd are the LAUNCH profile's: a named ssh profile uses its own cwd, else ~.
        if named_ssh:
            remote_cwd = _declared_remote_profile_cwd(profile_home)
            return (remote_cwd, "session") if remote_cwd else ("~", "process")
        raw = os.environ.get("TERMINAL_CWD", "").strip() or _workdir_terminal_cfg("cwd")
        if raw and raw not in {".", "auto", "cwd"}:
            return raw, "process"
        if backend == "ssh":
            return "~", "process"
    if session and session.get("cwd"):
        return str(session["cwd"]), "session"
    return _completion_cwd(), "process"


def _session_cwd(session: dict | None) -> str:
    return str(session["cwd"]) if session and session.get("cwd") else _completion_cwd()


# Sources whose launch directory is an artifact of how the app was started, not a workspace the user picked.
_LAUNCH_CWD_NOT_A_WORKSPACE = {"desktop"}


def _context_cwd_is_launch_artifact(session: dict | None) -> bool:
    """Whether the session cwd came from app launch rather than user intent."""
    return bool(session and not session.get("explicit_cwd") and _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE)


def _resolve_create_cwd(params: dict, source: str, profile_home) -> tuple[bool, str, bool]:
    """``(explicit_cwd, session_cwd, remote_cwd)`` for a freshly created session.

    Only a chosen workspace persists as cwd; the launch-dir fallback is "No workspace". A
    chosen workspace is one the gateway host can ``isdir``-probe, an ssh-shaped cwd on a remote
    profile, or — the #108205 desktop arm — a nonblank desktop-sourced cwd the host probe could
    not vouch for: the client names a workspace its gateway host cannot see (Docker/remote
    backend topology), and a failed host-side isdir is a topology artifact, not a verdict on
    the client's path. The CLIENT vouches instead, with #52589 provenance: a deliberate pick
    (``cwd_explicit``) adopts the raw path outright; an inherited app-global workspace only
    counts once the completion resolution actually adopted it, so a launch-dir fallback still
    persists nothing and a named profile's configured ``terminal.cwd`` keeps winning.
    """
    raw_cwd = str(params.get("cwd") or "").strip()
    remote_cwd = bool(raw_cwd) and _is_remote_cwd_shape(raw_cwd) and _cwd_is_remote(profile_home)
    explicit_cwd = False
    with contextlib.suppress(Exception):
        explicit_cwd = bool(raw_cwd) and (
            remote_cwd or os.path.isdir(os.path.abspath(os.path.expanduser(raw_cwd))))
    session_cwd = _completion_cwd(params)
    if raw_cwd and not explicit_cwd and source == "desktop":
        if params.get("cwd_explicit"):
            explicit_cwd = True
            session_cwd = raw_cwd
        elif session_cwd and session_cwd == os.path.abspath(os.path.expanduser(raw_cwd)):
            explicit_cwd = True
    return explicit_cwd, session_cwd, remote_cwd


def _persisted_session_cwd(session: dict) -> str | None:
    """The cwd to stamp on the session's DB row, or None to leave it unset (launch-dir rule: ``_ensure_session_db_row``)."""
    if session.get("explicit_cwd"):
        return _session_cwd(session)
    if _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE or _is_remote_launch_cwd(session):
        return None
    return str(session.get("cwd") or "") or None  # the session's OWN dir, never _session_cwd's gateway-wide fallback


def _is_remote_launch_cwd(session: dict | None) -> bool:
    """An ssh session's cwd that nobody picked: the gateway's launch directory, a path on THIS host. Host-side context
    discovery reads it from memory, but it is never persisted: a resume adopts a stored ssh cwd as the remote
    workspace."""
    return bool(session) and not session.get("explicit_cwd") and _cwd_is_remote(session.get("profile_home"))


def _is_hermes_owned_cwd(cwd: str, profile_home) -> bool:
    """Whether ``cwd`` is inside Hermes's own host tree: the Hermes root (``/opt/data`` and its ``/opt/data/home``
    subprocess home in the Docker image, which also holds every named profile) or the install tree
    (``/opt/hermes``). A ``~`` path is the remote's home, never this host's."""
    from agent.runtime_cwd import _is_install_tree
    from hermes_constants import get_default_hermes_root

    if not os.path.isabs(cwd):
        return False
    try:
        path = Path(cwd).resolve()
        home = Path(profile_home or get_hermes_home()).expanduser()
        roots = {home.resolve(), get_default_hermes_root(home=home).resolve()}
    except (OSError, RuntimeError):
        return False
    return any(path == root or root in path.parents for root in roots) or _is_install_tree(path)


def _resumable_stored_cwd(cwd, profile_home) -> str:
    """A session row's stored cwd as a resume may adopt it: empty when an ssh session's row holds a path in Hermes's
    own host tree (a host launch directory, never a remote workspace)."""
    cwd = str(cwd or "")
    if cwd and _cwd_is_remote(profile_home) and _is_hermes_owned_cwd(cwd, profile_home):
        return ""
    return cwd


def _heal_dead_cwd(cwd: str) -> str:
    """Resolve a session cwd inside a now-deleted directory (e.g. a removed linked worktree, which probes to no branch
    while the sidebar folds it to the main lane): walk up to the first existing ancestor and take its common git root.
    Local backends only — a remote/SSH cwd may legitimately not exist on the host, so callers skip healing there."""
    raw = (cwd or "").strip()
    if not raw or os.path.isdir(raw):
        return raw
    probe = raw
    for _ in range(64):
        parent = os.path.dirname(probe)
        if not parent or parent == probe:
            break
        probe = parent
        if os.path.isdir(probe):
            break
    if not os.path.isdir(probe):
        return raw
    with contextlib.suppress(Exception):
        return git_probe.common_repo_root(probe) or git_probe.repo_root(probe) or probe
    return probe


def _session_is_local_backend(session: dict | None) -> bool:
    """Whether THIS session's cwd can be stat'ed / git-probed here. A session bound to a named ssh profile never can,
    whatever the launch process runs (one multiplexed gateway serves many profiles), and a per-profile gateway
    (``hermes -p x``) may set ``terminal.backend: ssh`` in config without ``TERMINAL_ENV``: an env-only check would
    heal a live remote cwd to its nearest host ancestor (``/home``) and persist that."""
    if session and session.get("profile_home") and _bound_terminal_backend(session["profile_home"]) != "local":
        return False
    # Otherwise the launch backend, env or config: an in-process gateway (no TERMINAL_ENV bridge) under
    # ``terminal.backend: docker`` holds a container cwd (``/workspace``) that healing would walk up to ``/``.
    return _effective_terminal_backend() == "local"


def _effective_terminal_backend() -> str:
    """Active terminal backend name (``local``, ``docker``, ``ssh``, ...): ``TERMINAL_ENV`` when set (launchers bridge
    ``terminal.backend`` into env), else the ``terminal.backend`` config key (in-process gateways skip that bridge)."""
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    if not backend or backend == "local":
        backend = _workdir_terminal_cfg("backend").lower()
    return backend or "local"


def _display_session_cwd(session: dict | None) -> str:
    """Session cwd for display/probe surfaces, healed past deleted worktrees (healed value persisted back; local only)."""
    cwd = _session_cwd(session)
    if not _session_is_local_backend(session):
        return cwd
    healed = _heal_dead_cwd(cwd)
    if healed and healed != cwd and session is not None:
        session["cwd"] = healed
        _persist_session_cwd_and_schedule_git_meta(session, healed)
    return healed


def _reconcile_session_cwd_from_terminal(session: dict | None) -> bool:
    """Re-anchor a session that SETTLED in another worktree of the SAME repo. Returns moved. An agent told to work in
    a fresh worktree `git worktree add`s and `cd`s in while the session stays pinned (labelled with the primary
    checkout's branch). A plain `cd` is deliberately NOT a workspace move (see ``_apply_project_workspace``): a non-git
    workspace stepping into a repo or a visit to an unrelated repo is browsing, and a workspace the user deliberately
    moved the chat into is never overridden. Local backends only (a remote cwd cannot be stat'ed or git-probed
    here)."""
    # A workspace the USER put the chat in (the composer's folder picker -> session.cwd.set, the sidebar's
    # move-to-project -> session.workspace.move) only moves by another deliberate action; a cwd adopted HERE is
    # marked `cwd_from_settle` so successive settles keep following. NOT `explicit_cwd`: session.create sets that for
    # ANY session whose cwd exists on disk, so keying the pin on it made every desktop session — all of which are
    # created with a workspace — unfollowable, which is exactly the agent-made-a-worktree case this reconcile exists
    # for.
    if not session or (session.get("cwd_pinned") and not session.get("cwd_from_settle")):
        return False
    if not _session_is_local_backend(session):
        return False
    try:
        from tools.terminal_tool import get_session_cwd
        if not (recorded := get_session_cwd(session.get("session_key") or "")):
            return False
    except Exception:
        return False
    resolved = os.path.abspath(os.path.expanduser(str(recorded)))
    current = os.path.abspath(os.path.expanduser(_session_cwd(session)))
    if resolved == current or not os.path.isdir(resolved):
        return False
    # Worktree ROOTS (folding to the common root would hide the move), both in a git tree, different from each other,
    # sharing the SAME common .git dir.
    landed, current_root = git_probe.repo_root(resolved), git_probe.repo_root(current)
    if not landed or not current_root or landed == current_root:
        return False
    landed_common = git_probe.common_repo_root(resolved)
    if not landed_common or landed_common != git_probe.common_repo_root(current):
        return False
    # This is the session's workspace now (a desktop launch-artifact cwd earns a real row); the settle marker keeps it
    # overridable by the NEXT settle.
    session.update(cwd=resolved, explicit_cwd=True, cwd_from_settle=True)
    _register_session_cwd(session)
    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    return True


def _emit_settled_session_info(sid: str, session: dict, agent) -> None:
    """Emit end-of-turn ``session.info``, reconciling a settled cwd first (the agent has stopped moving; riding the
    turn-end event needs no new event type/round trip)."""
    try:
        _reconcile_session_cwd_from_terminal(session)
    except Exception:
        logger.debug("failed to reconcile settled session cwd", exc_info=True)
    _emit("session.info", sid, _session_info(agent, session))


def _session_source(session: dict | None) -> str:
    source = str(session.get("source") or "").strip() if session else ""
    return source or _resolve_session_platform()


def _register_session_cwd(session: dict | None) -> None:
    if not session:
        return
    # Workspace moves must reach lazy/restarted runtimes, not just terminal tools.
    # Do not reinitialize memory providers or invalidate the cached system prompt.
    if hasattr(agent := session.get("agent"), "session_cwd"):
        agent.session_cwd = session.get("cwd") or None
    # A session that adopted a real workspace out of a home-fallback cwd (#76902: the
    # packaged Desktop pins $HOME when no default project dir is configured) resumes
    # subdirectory-hint discovery anchored to that project. No prompt/system-prompt
    # state changes — the tracker only scopes future tool-result hints.
    hints = getattr(agent, "_subdirectory_hints", None) if session.get("cwd") else None
    if hints is not None and hasattr(hints, "rebind_working_dir"):
        hints.rebind_working_dir(str(session.get("cwd")))
    with contextlib.suppress(Exception):
        from tools.terminal_tool import register_task_env_overrides
        cwd, cwd_source = _terminal_task_cwd_with_source(session)
        # The cwd/override record is keyed by the ROUTED home (#123989). session.create is a plain
        # @method, so bind the session's own profile home here or the record lands under the raw key
        # and the scoped turn (`profile:<p>:<key>`) misses it until the first `cd`. Callers already
        # inside the session's scope (the turn) bind nothing: the routed home is theirs already.
        import hermes_constants as hc

        with contextlib.ExitStack() as stack:
            profile_home = session.get("profile_home")
            if profile_home and hc.hermes_home_key(hc.get_hermes_home()) != hc.hermes_home_key(profile_home):
                stack.callback(hc.reset_hermes_home_override, hc.set_hermes_home_override(str(profile_home)))
            register_task_env_overrides(session["session_key"], {"cwd": cwd, "cwd_source": cwd_source})


def _workdir_row_model_config(session: dict) -> tuple[str, dict]:
    """``(model, model_config)`` for a fresh session row. The session's own model/effort/fast pick (composer override
    or restored /model switch) must own the row: the agent isn't built yet at first prompt.submit, and writing the
    global default here wins the INSERT-OR-IGNORE race (a reconnect silently reverts to the profile default).
    model_config carries provider/reasoning/service_tier so resume restores effort + fast too."""
    override = raw if isinstance(raw := session.get("model_override"), dict) else {}
    row_model = str(override.get("model") or "").strip() or _session_default_route(session)[0]
    model_config: dict = {k: str(v) for k in ("model", "provider", "base_url", "api_mode") if (v := override.get(k))}
    if "opusplan" in override:
        model_config["opusplan"] = override["opusplan"] is True
    # A RESOLVED provider "custom" (named ``providers:``/``custom_providers:`` entry) persisted bare here is the origin
    # of "No LLM provider configured" rows (resume routes to OpenRouter with no key). Recover the durable
    # ``custom:<name>`` identity (matches _runtime_model_config).
    if str(model_config.get("provider") or "").strip().lower() == "custom":
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            healed = canonical_custom_identity(
                base_url=model_config.get("base_url") or None, model=model_config.get("model") or row_model or None)
            if healed:
                model_config["provider"] = healed
        except Exception:
            logger.debug("custom provider identity recovery failed (db row)", exc_info=True)
    if (reasoning := session.get("create_reasoning_override")) is not None:
        model_config["reasoning_config"] = reasoning
    if (service_tier := session.get("create_service_tier_override")) is not None:
        # "" is the in-memory sentinel for an explicit normal tier (bypasses _make_agent's profile fallback); persist a
        # durable marker so resume can tell it from an inherited tier.
        model_config["service_tier"] = service_tier or "normal"
    # Same ``_branched_from`` marker the TUI /branch uses (list_sessions_rich + sidebar nesting).
    if parent_session_id := session.get("parent_session_id"):
        model_config["_branched_from"] = parent_session_id
    # Room plumbing always follows the member profile. Canonical Bot Chats do too until the composer records an
    # explicit chat-scoped pick plus the profile model it diverged from (see _stored_session_runtime_overrides).
    for flag in ("room_plumbing", "follow_profile_config"):
        if session.get(flag):
            model_config[flag] = True
    if isinstance(composer_profile := session.get("composer_override_profile"), dict):
        model_config["composer_override_profile"] = composer_profile
    from tools.approval_yolo import with_session_yolo  # a /yolo toggled before the row existed
    return row_model, with_session_yolo(model_config, session.get("session_key") or "")


def _ensure_session_db_row(session: dict) -> bool:
    """Idempotently persist the session's DB row on first real activity (prompt.submit), so abandoned drafts never
    leave an empty "Untitled" session. INSERT OR IGNORE: re-calls and the AIAgent's lazy create are no-ops. Returns
    False only when the store is unavailable (no openable state.db) — prompt.submit fails the send loudly instead of
    streaming into a store that will never save it; no key / best-effort / success are all True.

    A cwd the user *chose* is always persisted. Otherwise the launch directory stands in only for terminal sessions
    (the user deliberately ``cd``'d there; dropping it left the sidebar with no cwd AND no git_repo_root); desktop
    launch dirs (``/``, home) stay null -> "No workspace".

    See #98924.
    """
    if not (key := session.get("session_key")):
        return
    # Persist into the session's own profile db (global remote mode), not the launch profile's — otherwise the unified
    # list mis-tags the row and resume 404s ("session not found").
    profile_home = session.get("profile_home")
    with _workdir_owner_db(session, "failed to open profile db for session row") as db:
        if db is _WORKDIR_DB_OPEN_FAILED:
            return False
        if db is None:
            # Fail loud ONLY when the store failed to open (_db_error records the SessionDB open exception); None with
            # no recorded error means "no store in this context" -> True.
            # A None db with no recorded error means "no store in this context" (degraded harness, store
            # deliberately absent) — that keeps the pinned best-effort contract and stays True. See #98924.
            return _db_error is None
        row_model, model_config = _workdir_row_model_config(session)
        try:
            db.create_session(
                key, source=_session_source(session), model=row_model, model_config=model_config or None,
                parent_session_id=session.get("parent_session_id") or None, cwd=_persisted_session_cwd(session),
                # The login this session was opened under, in the same ``<provider>:<id>`` form the agent is
                # built with — the row is the only place the identity reaches the store, and the upsert can't
                # add it later (user_id is set at insert). None (no password provider, legacy token, stdio)
                # leaves the column empty exactly as before.
                user_id=_session_auth_user_id(session),
                # Self-describing rows: aggregators merging several profile DBs can't rely on which file a row came
                # from; a NULL is only repaired by the one-shot backfill.
                # Stamp the launch profile explicitly instead of leaving NULL — NULL is exactly what the
                # #94724 legacy-owner backfill exists to repair, and rows minted AFTER that one-shot
                # backfill ran stayed NULL forever: profile-keyed matching then drops them from the sidebar
                # and deep links can't resolve them (#99222).
                profile_name=profile_name_for_home(profile_home) or _current_profile_name())
            # Born hidden (session.create hidden=true, or set_hidden before the row existed): apply the deferred intent.
            if session.get("pending_hidden"):
                try:
                    if db.set_session_hidden(key, True):
                        session.pop("pending_hidden", None)
                except Exception:
                    logger.debug("failed to apply pending hidden flag", exc_info=True)
            # Same deferral for session.archive before the row existed (mirrors pending_hidden).
            if session.get("pending_archived"):
                try:
                    if db.set_session_archived(key, True):
                        session.pop("pending_archived", None)
                except Exception:
                    logger.debug("failed to apply pending archived flag", exc_info=True)
            _schedule_row_git_meta(session, key, db)
        except Exception as exc:
            # Disk-full is not a soft failure: swallowed here, prompt.submit returns {"status":"streaming"} and the
            # message vanishes silently.
            _workdir_reraise_disk_full(exc, "failed to persist desktop session row")
    return True


def _schedule_row_git_meta(session: dict, key: str, db) -> None:
    """Git-enrich a lazily created row once per live session. The row lands here with its cwd on the first submit, so
    ``_hydrate_session_cwd`` (which ran when no row existed) never claimed a probe, and a desktop row kept NULL
    git_branch/git_repo_root for life: its lane fell back to a fake ``main`` label (#108784). Probes the row's own
    cwd (the upsert never overwrites it), and only when enrichment is missing."""
    if session.get("row_git_meta_checked"):
        return
    session["row_git_meta_checked"] = True
    row = (db.get_session(key) if hasattr(db, "get_session") else None) or {}
    cwd = str(row.get("cwd") or "").strip()
    if cwd and not (row.get("git_branch") and row.get("git_repo_root")):
        _persist_session_cwd_and_schedule_git_meta(session, cwd, db=db)


def _workdir_reraise_disk_full(exc: BaseException, log_msg: str) -> None:
    """Re-raise a disk-full write error (the caller must surface it); debug-log the rest."""
    from hermes_state_errors import is_disk_full_error
    if is_disk_full_error(exc):
        raise exc
    logger.debug(log_msg, exc_info=True)


# Seed row fields copied from the parent transcript. display_kind/metadata: timeline markers ride as role=user;
# dropping the tag re-plants them as bare user turns after a restart and corrupts the truncate ordinal address space.
_WORKDIR_SEED_FIELDS = (
    "content", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "display_kind", "display_metadata", "timestamp")


def _persist_branch_seed(session: dict) -> None:
    """Persist a seeded transcript once its row exists. Seeded messages (a branch's copied parent, a client's
    opening turns) live only in ``session["history"]`` (ridden into the agent as ``conversation_history``, which
    ``_flush_messages_to_session_db`` skips by identity), so the row would otherwise resume without them. Runs
    once: at create for a seeded session, else at the first submit after ``_ensure_session_db_row`` wrote the
    row. ``seeded`` is stamped by session.create; a resumed session carries its stored transcript in
    ``history`` and must never re-append it."""
    if not (key := session.get("session_key")) or not session.get("seeded") or session.get("_branch_seed_persisted"):
        return
    from agent.message_metadata import message_identity
    from agent.transcript_repair import sync_flushed_message_markers
    with session["history_lock"]:  # message_identity stamps the live dicts
        live = list(session.get("history") or [])
        seed = [{"role": msg.get("role", "user"), **{f: msg.get(f) for f in _WORKDIR_SEED_FIELDS},
                 **message_identity(msg)} for msg in live]
    if not seed:
        return
    with _session_db(session) as db:
        if db is None:
            return
        try:
            # Chunked so each BEGIN IMMEDIATE stays short (a seed can be hundreds of rows); a mid-copy failure leaves a
            # partial seed with _branch_seed_persisted unset.
            # Bounded-chunk transactions (see #23254): a branch seed can be hundreds of rows; chunking keeps
            # each BEGIN IMMEDIATE short so concurrent writers aren't starved.
            db.append_messages_batch(key, seed, chunk_rows=500)
            with session["history_lock"]:
                sync_flushed_message_markers(live, seed)
            session["_branch_seed_persisted"] = True
        except Exception as exc:
            _workdir_reraise_disk_full(exc, "branch seed persist failed")


def _submit_row_target_key(session: dict) -> str:
    """The session row an off-turn submit write must use: the live ``agent.session_id`` when it has
    rotated off ``session_key``, else ``session_key`` itself.

    The turn's own transcript is flushed under ``agent.session_id`` (``_db_flush_write``), while an
    off-turn write only has ``session_key`` to go on — and those diverge for the whole of a turn that
    begins after the agent's session rotated: a compression publish, an adopted continuation tip, or a
    lease-wait re-resolve all move ``agent.session_id`` while ``session_key`` is re-anchored only at
    TURN END (``_absorb_turn_result`` -> ``_sync_session_key_after_compress``). Writing the submit row
    to the stale key splits one turn's rows across two sessions: the user row in the parent, every tool
    row and the final text in the child (#123545, evidence A + B).

    The live id IS the authority — it is the value the turn will flush under, and the same
    ``getattr(agent, "session_id", None) or session_key`` the sibling system-prompt persist already uses
    against this handle (``_persist_live_session_system_prompt``). Do NOT re-resolve through the lineage
    here: ``resolve_resume_session_id`` returns the deepest node that has MESSAGES, so the freshly-minted
    child of a just-published rotation resolves back to the parent and the fix would no-op exactly when
    it is needed. The choice is made ONCE here and recorded on the staged dict under
    ``_SUBMIT_ROW_SESSION_KEY`` so every later addresser of that row — the @-expansion rewrite, the
    queue merge, the drain deactivation — reads the same key instead of re-deriving one that a rotation
    can invalidate mid-turn.
    """
    return str(getattr(session.get("agent"), "session_id", None) or "") or str(session.get("session_key") or "")


# Wire-sanitizer-safe key carrying the session the submit row was actually written under. Mirrors
# ``_DB_PERSISTED_MARKER``'s contract (leading underscore, stripped from provider payloads).
_SUBMIT_ROW_SESSION_KEY = "_submit_row_session_id"


def _submit_row_owner_key(staged: dict, session: dict) -> str:
    """The session id a staged submit row lives under: recorded at write time, else the current best.
    Every caller narrows to a dict (and checks ``_row_id``) immediately before, so the recorded value is
    the answer whenever the row exists; the re-derivation only covers a dict that predates the stamp."""
    recorded = str(staged.get(_SUBMIT_ROW_SESSION_KEY) or "")
    return recorded or _submit_row_target_key(session)


def _write_submit_user_row(session: dict, text: Any, display_kind: str | None,
                           accept_metadata: dict | None = None) -> dict | None:
    """Write the submitted user turn to the transcript and RETURN the durable dict (stamped
    ``_DB_PERSISTED_MARKER``/``_row_id``) WITHOUT slotting it on the session. The write half of
    :func:`_persist_submit_user_row`, shared by the busy-queue accept (which attaches the dict to
    the queue envelope, never the shared session slot a possibly-still-staged in-flight turn owns).
    ``accept_metadata`` merges into ``display_metadata`` (the busy-queue accept's never-drained
    marker, retired by ``reopen_session`` — #125577).
    Returns None when nothing was written (no key / non-text / store unavailable / failed write)."""
    # ``session_key`` is only an "is this a real session" probe — the row is written to ``target`` below,
    # which a rotation can already have moved off ``session_key`` (#123545). One guard, one value: the
    # writer must not read a different key than the one it checks.
    if not session.get("session_key") or not isinstance(text, str) or not text.strip():
        return None
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.message_metadata import stamp_message_timestamp, stamp_message_uid
    staged = stamp_message_timestamp({"role": "user", "content": text})
    if display_kind:
        staged["display_kind"] = display_kind
    if accept_metadata:
        staged["display_metadata"] = {**accept_metadata}
    with _session_db(session) as db:
        if db is None:
            return None
        target = _submit_row_target_key(session)
        try:
            staged["_row_id"] = db.append_message(
                target, "user", content=text, display_kind=display_kind, timestamp=staged["timestamp"],
                message_uid=stamp_message_uid(staged),  # the live dict the turn adopts carries the row's uid
                display_metadata=staged.get("display_metadata"))
        except Exception as exc:
            _workdir_reraise_disk_full(exc, "submit-time user row persist failed")
            return None
    staged[_DB_PERSISTED_MARKER] = True
    # Record the owning session so every later addresser of THIS row (the @-expansion rewrite, the
    # queue merge, the drain deactivation) finds it by the same key even if a rotation lands mid-turn.
    staged[_SUBMIT_ROW_SESSION_KEY] = target
    return staged


def _persist_submit_user_row(session: dict, text: Any, display_kind: str | None,
                             accept_metadata: dict | None = None) -> None:
    """Write the submitted user turn at send time, before the agent build and turn: the agent's own
    crash persist only runs once the build finished, so quitting a frozen app during a slow first build
    left a session row with no message (#111868). The dict is staged on the session already stamped
    durable (the shape ``quiet_single_query`` re-stages an unanswered DM in) so the turn adopts it via
    ``_stage_turn_user_message`` and the flush writes no second row. A failed write stages nothing:
    the turn's crash persist then writes the row as before. ``accept_metadata`` marks a row that
    belongs to a still-QUEUED envelope (#125577); a dispatching turn's row is never marked."""
    session.pop("_submit_user_row", None)  # a failed/unsupported write must not acknowledge an older send
    if (staged := _write_submit_user_row(session, text, display_kind, accept_metadata)) is not None:
        session["_submit_user_row"] = staged


def _adopt_submit_user_row(session: dict, agent, persist_user_message: Any, text: Any) -> None:
    """Hand the row written at submit to the turn as its user dict (``agent._pending_cli_user_message``,
    adopted by ``_stage_turn_user_message`` when the content matches). A prompt the prologue rewrote
    (@-expansion, image parts) first updates that row so the durable transcript replays what the model
    was sent and the ``api_content`` sidecar can address it; ``_row_id`` rides along for that stamp.
    ``text`` is THIS turn's raw submit: a staged row from an earlier send (its turn ended before the agent
    ran) is discarded untouched, so the DB row stays the user's message and never a synthesized turn's text."""
    staged = session.pop("_submit_user_row", None)
    if not isinstance(staged, dict) or agent is None or staged.get("content") != text:
        return
    if staged["content"] != persist_user_message:
        from agent.session_persistence import _durable_content
        with _session_db(session) as db:
            if db is None:
                return
            try:
                # Address the row where it was actually WRITTEN (``_write_submit_user_row`` recorded it);
                # a rotated-away ``session_key`` would miss that row's session_id and silently skip the
                # rewrite, leaving the transcript replaying the raw keystrokes.
                db.set_user_message_content(
                    _submit_row_owner_key(staged, session), staged["_row_id"],
                    _durable_content(persist_user_message))
            except Exception:
                logger.debug("submit-time user row update failed; the turn writes its own row", exc_info=True)
                return
        staged["content"] = persist_user_message
    from agent.session_persistence import _persist_lock
    with _persist_lock(agent):
        agent._pending_cli_user_message = staged


# Yielded by _workdir_owner_db when the profile db failed to OPEN (vs "no store in this context"); row creation fails loud.
_WORKDIR_DB_OPEN_FAILED = object()


@contextlib.contextmanager
def _workdir_owner_db(session: dict, fail_log: str):
    """Body of :func:`_session_db`; ``_ensure_session_db_row`` uses it directly so a patched ``_session_db`` can't alter rows."""
    db, close_db = None, False
    if profile_home := session.get("profile_home"):
        try:
            from hermes_state_registry import acquire
            db, close_db = acquire(Path(profile_home) / "state.db"), True
        except Exception:
            logger.debug(fail_log, exc_info=True)
            db = _WORKDIR_DB_OPEN_FAILED
    else:
        db = _get_db()
    try:
        yield db
    finally:
        if close_db and db is not None:
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close
                release_or_close(db)


@contextlib.contextmanager
def _session_db(session: dict):
    """Yield the SessionDB that owns this session's row (profile-aware): a remote/profile session persists into its own
    profile's ``state.db`` (fresh handle, closed on exit); else the shared ``_get_db()`` handle (left open). None if unavailable."""
    with _workdir_owner_db(session, "failed to open profile db for session") as db:
        yield None if db is _WORKDIR_DB_OPEN_FAILED else db


def _rewind_active_session_history(
    session: dict, user_ordinal: int, *, require_retryable: bool = False) -> tuple[list[dict], dict, int]:
    """Rewind one canonical user turn while retaining carrier scaffolding. Caller holds ``history_lock``. Persistent
    sessions go through ``SessionDB.rewind_user_turn`` (the durable transcript is the authority; memory is installed
    only after the commit); a session without a key rewinds the warm history alone."""
    from agent.context_compressor import history_before_user_originated_turn, retryable_user_text, user_originated_turn_view

    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    user_indices = [i for i, m in enumerate(history) if user_originated_turn_view(m) is not None]
    if user_ordinal < 0 or user_ordinal >= len(user_indices):
        raise ValueError("target user message is no longer in session history")
    session_key = str(session.get("session_key") or "").strip()
    if session_key:
        with _session_db(session) as db:
            if db is None:
                raise RuntimeError("session database is unavailable")
            outcome = db.rewind_user_turn(
                session_key, user_ordinal, warm_history=history, require_retryable=require_retryable,
                adopt_row_ids=True)
        installed, live_view, rewound_count = outcome.prefix, outcome.live_view, outcome.rewound_count
    else:
        target_index = user_indices[user_ordinal]
        installed, live_view = history_before_user_originated_turn(history, target_index)
        rewound_count = len(history) - target_index
        if require_retryable:
            retryable_user_text(live_view.get("content"))

    installed = [message.copy() for message in installed]
    session["history"] = installed
    session["history_version"] = int(session.get("history_version", 0)) + 1
    agent = session.get("agent")
    if agent is not None:
        agent._session_messages = installed
        if hasattr(agent, "_last_flushed_db_idx"):
            agent._last_flushed_db_idx = len(installed) if session_key else 0
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = installed[:] if session_key else None
    return installed, live_view, rewound_count


def _history_without_ephemeral_scaffolding(history: list[dict]) -> list[dict]:
    """Return the durable transcript shape without transient recovery rows."""
    from agent.session_persistence import _is_ephemeral_scaffolding
    return [message.copy() for message in history if not _is_ephemeral_scaffolding(message)]


def _workdir_valid_generation(generation) -> bool:
    """A claimed DB probe generation: a positive int (bool excluded)."""
    return not isinstance(generation, bool) and isinstance(generation, int) and generation >= 1


def _persist_session_git_meta(session: dict, cwd: str, generation: int) -> None:
    """Resolve + persist a session's git branch / repo root on a daemon thread: inline ``git`` probes on the
    session-init / cwd-set path would stall startup on a slow or unreachable ``cwd``. Persists via the same
    profile-aware db the caller wrote ``cwd`` to. Best-effort: a probe failure leaves the enrichment columns unset."""
    session_key = session.get("session_key", "")
    if not session_key or not cwd or not _workdir_valid_generation(generation):
        return
    # Snapshot routing fields; the live session dict may be gone when the thread runs.
    db_session = {"session_key": session_key, "profile_home": session.get("profile_home")}

    def _run() -> None:
        try:
            branch, root = git_probe.branch(cwd), git_probe.common_repo_root(cwd)
            if not (branch or root):
                return
            with _session_db(db_session) as db:
                if db is not None:
                    db.publish_session_git_metadata(session_key, cwd, generation, branch, root)
        except Exception:
            logger.debug("failed to persist session git metadata", exc_info=True)

    threading.Thread(target=_run, name="git-meta", daemon=True).start()


def _persist_session_cwd_and_schedule_git_meta(session: dict, cwd: str, *, db=None) -> int | None:
    """Claim a DB-backed probe generation, then start Git enrichment."""
    try:
        with (contextlib.nullcontext(db) if db is not None else _session_db(session)) as owner_db:
            if owner_db is None:
                return None
            generation = owner_db.update_session_cwd(session.get("session_key", ""), cwd)
    except Exception:
        logger.debug("failed to persist session cwd", exc_info=True)
        return None
    if not _workdir_valid_generation(generation):
        return None
    _persist_session_git_meta(session, cwd, generation)
    return generation


def _set_session_cwd(session: dict, cwd: str) -> str:
    from hermes_constants import translate_cwd_for_wsl_backend
    cwd = translate_cwd_for_wsl_backend(str(cwd))
    resolved = _workspace_cwd(session.get("profile_home"), cwd)
    # An explicit user choice: persisted as the workspace (not the launch-dir fallback), superseding a settle-adopted
    # cwd — and PINNED, so the settle reconcile cannot drag it to another worktree the agent merely visited.
    session.update(cwd=resolved, explicit_cwd=True, cwd_pinned=True, cwd_from_settle=False)
    _register_session_cwd(session)
    # The synchronous DB write claims ordering authority; git probes may publish only for that exact generation.
    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    with contextlib.suppress(Exception):
        from tools.terminal_tool_lifecycle import cleanup_vm
        cleanup_vm(session["session_key"])
    return resolved


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
