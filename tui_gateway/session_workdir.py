"""Session working-directory + durable session row: cwd resolution/healing, session.db row ensure,
branch seed, history rewind, git meta persistence. Bodies are rebound onto server.py's globals at
install time (method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib
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
    # A session bound to another profile resolves its workspace from THAT profile's config before the launch profile's
    # env var; the dashboard's in-memory gateway does NOT inherit the PTY child's bridged TERMINAL_CWD, so a configured
    # terminal.cwd is read directly.
    raw = (params.get("cwd") or _sessions.get(params.get("session_id") or "", {}).get("cwd")
           or _profile_configured_cwd(_profile_home(params.get("profile"))) or _launch_configured_cwd()
           or os.environ.get("TERMINAL_CWD") or os.getcwd())
    with contextlib.suppress(Exception):
        resolved = os.path.abspath(os.path.expanduser(str(raw)))
        if os.path.isdir(resolved):
            return resolved
    return os.getcwd()


def _workdir_terminal_cfg(key: str) -> str:
    """Stripped ``terminal.<key>`` from config, or "" when unset/unreadable."""
    with contextlib.suppress(Exception):
        terminal_cfg = _load_cfg().get("terminal", {})
        if isinstance(terminal_cfg, dict):
            return str(terminal_cfg.get(key) or "").strip()
    return ""


def _terminal_task_cwd(session: dict | None) -> str:
    """The cwd terminal_tool should use for this TUI session (NOT host-validated: a non-local backend's cwd lives
    inside the target environment)."""
    return _terminal_task_cwd_with_source(session)[0]


def _terminal_task_cwd_with_source(session: dict | None) -> tuple[str, str]:
    """``(cwd, source)``: ``"session"`` for THIS session's workspace (``explicit_cwd``/tracked dir), ``"process"`` for
    the global ``TERMINAL_CWD``/``terminal.cwd`` fallback — under per-session docker isolation that is a PREVIOUS
    session's launch artifact, so terminal_tool refuses it as a bind-mount source."""
    backend = _effective_terminal_backend()
    if backend != "local":
        # THIS session's explicit workspace beats the LAST session's env var.
        if session and session.get("explicit_cwd") and session.get("cwd"):
            return str(session["cwd"]), "session"
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


def _persisted_session_cwd(session: dict) -> str | None:
    """The cwd to stamp on the session's DB row, or None to leave it unset (launch-dir rule: ``_ensure_session_db_row``)."""
    if session.get("explicit_cwd"):
        return _session_cwd(session)
    if _session_source(session) in _LAUNCH_CWD_NOT_A_WORKSPACE:
        return None
    return str(session.get("cwd") or "") or None  # the session's OWN dir, never _session_cwd's gateway-wide fallback


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


def _is_local_terminal_backend() -> bool:
    backend = (os.environ.get("TERMINAL_ENV") or "").strip().lower()
    return not backend or backend == "local"


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
    if not _is_local_terminal_backend():
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
    workspace stepping into a repo or a visit to an unrelated repo is browsing, and an explicitly chosen workspace is
    never overridden. Local backends only (a remote cwd cannot be stat'ed or git-probed here)."""
    # An explicit choice only moves by another explicit action; a cwd adopted HERE is marked `cwd_from_settle` so
    # successive settles keep following.
    if not session or not _is_local_terminal_backend():
        return False
    if session.get("explicit_cwd") and not session.get("cwd_from_settle"):
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
    with contextlib.suppress(Exception):
        from tools.terminal_tool import register_task_env_overrides
        cwd, cwd_source = _terminal_task_cwd_with_source(session)
        register_task_env_overrides(session["session_key"], {"cwd": cwd, "cwd_source": cwd_source})


def _workdir_row_model_config(session: dict) -> tuple[str, dict]:
    """``(model, model_config)`` for a fresh session row. The session's own model/effort/fast pick (composer override
    or restored /model switch) must own the row: the agent isn't built yet at first prompt.submit, and writing the
    global default here wins the INSERT-OR-IGNORE race (a reconnect silently reverts to the profile default).
    model_config carries provider/reasoning/service_tier so resume restores effort + fast too."""
    override = raw if isinstance(raw := session.get("model_override"), dict) else {}
    row_model = str(override.get("model") or "").strip() or _resolve_model()
    model_config: dict = {k: str(v) for k in ("model", "provider", "base_url", "api_mode") if (v := override.get(k))}
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
    # Bot-Mode canonical chats / room plumbing are plugin-owned scratch conversations whose runtime must ALWAYS follow
    # the member profile's CURRENT config, never the provider pinned at first write (see _stored_session_runtime_overrides).
    for flag in ("room_plumbing", "follow_profile_config"):
        if session.get(flag):
            model_config[flag] = True
    return row_model, model_config


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
                # Self-describing rows: aggregators merging several profile DBs can't rely on which file a row came
                # from; a NULL is only repaired by the one-shot backfill.
                # Stamp the store's OWN profile (derived from its db_path) rather than _current_profile_name() (the
                # serving process's launch profile), so store and label are one fact and cannot diverge -- a
                # default-root store stamps 'default', a profile store stamps its own name. An out-of-tree store
                # (explicit db_path / degraded fallback / test FakeDB) has no derivable owner and returns None;
                # fall back to the serving profile so the row still carries a routing label. NULL is exactly what
                # the #94724 legacy-owner backfill exists to repair, and rows minted AFTER that one-shot backfill
                # ran stayed NULL forever: profile-keyed matching then drops them from the sidebar and deep links
                # can't resolve them (#99222).
                profile_name=(getattr(db, "_own_profile_name", lambda: None)()
                              or (Path(profile_home).name if profile_home else _current_profile_name())))
            # Born hidden (session.create hidden=true, or set_hidden before the row existed): apply the deferred intent.
            if session.get("pending_hidden"):
                try:
                    db.set_session_hidden(key, True)
                except Exception:
                    logger.debug("failed to apply pending hidden flag", exc_info=True)
        except Exception as exc:
            # Disk-full is not a soft failure: swallowed here, prompt.submit returns {"status":"streaming"} and the
            # message vanishes silently.
            _workdir_reraise_disk_full(exc, "failed to persist desktop session row")
    return True


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
    """First-turn persist of a branch's copied transcript. A branch is a draft until its first submit: the parent's
    messages live only in ``session["history"]`` (ridden into the agent as ``conversation_history``, which
    ``_flush_messages_to_session_db`` skips by identity), so the row would otherwise resume missing its pre-branch
    context. Runs once, after ``_ensure_session_db_row`` wrote the row + parent link."""
    if not (key := session.get("session_key")) or not session.get("parent_session_id") or session.get("_branch_seed_persisted"):
        return
    with session["history_lock"]:
        seed = [dict(msg) for msg in (session.get("history") or [])]
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
            db.append_messages_batch(
                key, [{"role": msg.get("role", "user"), **{f: msg.get(f) for f in _WORKDIR_SEED_FIELDS}} for msg in seed],
                chunk_rows=500)
            session["_branch_seed_persisted"] = True
        except Exception as exc:
            _workdir_reraise_disk_full(exc, "branch seed persist failed")


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
    sessions archive the target and tail, inserting a composite carrier's hidden handoff in the same transaction; memory
    is installed only after the durable commit, from the validated prefix + returned scaffold row id (no reload)."""
    from agent.context_compressor import (
        history_before_user_originated_turn, retryable_user_text, split_user_originated_turn,
        user_originated_turn_view)
    from agent.memory_manager import sanitize_context
    from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary

    def _comparison_content(message: dict) -> Any:
        content = message.get("content")
        if _is_multimodal_tool_result(content):
            content = _multimodal_text_summary(content)
        elif isinstance(content, list):
            text_parts = [
                str(part.get("text", "")) if part.get("type") == "text" else "[screenshot]"
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "image", "image_url", "input_image"}]
            content = "\n".join(text_parts) if text_parts else None
        if message.get("role") in {"user", "assistant"} and isinstance(content, str):
            return sanitize_context(content).strip()
        return content

    def _user_indices(messages: list[dict]) -> list[int]:
        return [i for i, m in enumerate(messages) if user_originated_turn_view(m) is not None]

    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    user_indices = _user_indices(history)
    if user_ordinal < 0 or user_ordinal >= len(user_indices):
        raise ValueError("target user message is no longer in session history")
    target_index = user_indices[user_ordinal]
    installed, live_view = history_before_user_originated_turn(history, target_index)
    rewound_count = len(history) - target_index

    session_key = str(session.get("session_key") or "").strip()
    persisted = False
    if session_key:
        with _session_db(session) as db:
            if db is None:
                raise RuntimeError("session database is unavailable")
            expected_active_ids = db.get_active_message_ids(session_key)
            durable = db.get_messages_as_conversation(session_key, include_row_ids=True)
            durable_user_indices = _user_indices(durable)
            if len(durable_user_indices) != len(user_indices):
                raise RuntimeError("session history changed before the rewind could be persisted")
            durable_target_index = durable_user_indices[user_ordinal]
            durable_target = durable[durable_target_index]
            durable_prefix, durable_live_view = history_before_user_originated_turn(durable, durable_target_index)
            if _comparison_content(durable_live_view) != _comparison_content(live_view):
                raise RuntimeError("session history changed before the rewind could be persisted")
            if not isinstance(target_row_id := durable_target.get("_row_id"), int):
                raise RuntimeError("rewind target has no durable row identity")
            if require_retryable:
                retryable_user_text(durable_live_view.get("content"))
            scaffold, _ = split_user_originated_turn(durable_target)
            result = db.rewind_to_message(
                session_key, target_row_id, preserve_compaction_handoff=scaffold is not None,
                expected_active_ids=expected_active_ids, expected_target_content=durable_live_view.get("content"))
            if scaffold is not None:
                if not isinstance(replacement_id := result.get("replacement_message_id"), int):
                    raise RuntimeError("rewind commit did not return the replacement scaffold id")
                durable_prefix[-1].update(_row_id=replacement_id, _db_persisted=True)
                installed[-1] = durable_prefix[-1]
            # Clients address follow-ups by durable row id: keep the richer warm content but copy row identities when
            # the shapes align.
            if len(installed) == len(durable_prefix) and all(
                warm.get("role") == durable_message.get("role")
                and bool(warm.get("display_kind")) == bool(durable_message.get("display_kind"))
                and _comparison_content(warm) == _comparison_content(durable_message)
                for warm, durable_message in zip(installed, durable_prefix)
            ):
                for warm, durable_message in zip(installed, durable_prefix):
                    if isinstance(row_id := durable_message.get("_row_id"), int):
                        warm["_row_id"] = row_id
            live_view = durable_live_view
            rewound_count = int(result.get("rewound_count", 0))
            persisted = True
    elif require_retryable:
        retryable_user_text(live_view.get("content"))

    installed = [message.copy() for message in installed]
    session["history"] = installed
    session["history_version"] = int(session.get("history_version", 0)) + 1
    agent = session.get("agent")
    if agent is not None:
        agent._session_messages = installed
        if hasattr(agent, "_last_flushed_db_idx"):
            agent._last_flushed_db_idx = len(installed) if persisted else 0
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = installed[:] if persisted else None
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
    resolved = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(resolved):
        raise ValueError(f"working directory does not exist: {cwd}")
    # An explicit user choice: persisted as the workspace (not the launch-dir fallback), superseding a settle-adopted cwd.
    session.update(cwd=resolved, explicit_cwd=True, cwd_from_settle=False)
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
