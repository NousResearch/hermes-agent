"""Write-side safety guards for write_file / patch.

Every guard returns ``None`` when the write may proceed, else an error string
the tool returns verbatim.
Guards, in the order the tools apply them: ``_check_sensitive_path`` (hard
deny), ``_check_binary_document_write``, ``_check_protected_instruction_write``
(ALWAYS ask), ``_check_approval_required_write`` (normal gate),
``_check_cross_profile_path`` (sandbox-mirror lost-work), ``_is_internal_file_tool_content``.
``_stale_overwrite_blocker`` (write_file only, under the per-path lock) refuses a
whole-file overwrite of content this task never saw or that changed since.
"""

import fnmatch
import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from agent.file_safety import get_nt_namespace_error
from tools import file_state
from tools.binary_extensions import (
    has_binary_extension,
    has_opaque_document_extension,
    is_pdf_path,
    is_sqlite_sidecar,
)
from tools.file_tools_paths import _expand_tilde, _resolve_path_for_task
from tools.file_tools_read_tracking import _has_full_write_baseline, _read_mtime_drifted

logger = logging.getLogger(__name__)

# Prefixes matched after realpath. macOS: /private/var mirrors /var — block the
# sensitive subtrees only; a blanket "/private/var/" refuses every temp-file
# write because $TMPDIR, /tmp and /var/folders all realpath there.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/",
    "/private/var/db/", "/private/var/root/")
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

# NOTE: these four are a TEST-OVERRIDE surface, not a process cache. Both getters
# resolve per call in production (below) because ``get_hermes_home()`` /
# ``get_config_path()`` are per-turn contextvar-scoped: a multiplexed gateway
# (``gateway.multiplex_profiles: true``) serves many profiles in one process, and
# a process-wide memo would freeze whichever profile's home/config ran first —
# making the protected-instruction gate and the ``config.yaml`` hard-block
# order-dependent, up to letting a later profile rewrite its own ``config.yaml``
# the block exists to protect (#107327). A test can still pin a value by setting
# the slot and its ``_loaded`` flag.
_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False
_real_hermes_home_cached: str | None = None
_real_hermes_home_loaded = False


def _config_path_resolved() -> str:
    from hermes_cli.config import get_config_path
    return str(get_config_path().resolve())


def _hermes_home_real() -> str:
    from hermes_constants import get_hermes_home
    return os.path.realpath(str(get_hermes_home()))


def _get_hermes_config_resolved() -> str | None:
    """Resolved absolute path of the Hermes config file for the ACTIVE profile.

    Resolved per call so it tracks the per-turn ``HERMES_HOME`` scope (#107327);
    a test may pin it via ``_hermes_config_resolved`` + ``_hermes_config_resolved_loaded``."""
    if _hermes_config_resolved_loaded:
        return _hermes_config_resolved
    try:
        return _config_path_resolved()
    except Exception:
        # Resolver failure must stay bound to the ACTIVE profile's home, not the
        # subprocess HOME. ``_expand_tilde("~/...")`` follows the subprocess-HOME
        # contract, which under host ``auto`` mode can be the real/default user
        # home rather than the active multiplex ``HERMES_HOME`` — comparing
        # beta's ``config.yaml`` against the default/root config would let the
        # hard-block fail open on the exception path. Re-derive from the same
        # ``get_hermes_home()`` key the happy path uses, and substitute no
        # unrelated home if even that is gone (#107327 follow-up; PR #107335).
        try:
            from hermes_constants import get_hermes_home
            return str((Path(str(get_hermes_home())) / "config.yaml").resolve())
        except Exception:
            return None


def _get_real_hermes_home() -> str | None:
    """Realpath of the authoritative Hermes home for the ACTIVE profile.

    Resolved per call so it tracks the per-turn ``HERMES_HOME`` scope (#107327);
    a test may pin it via ``_real_hermes_home_cached`` + ``_real_hermes_home_loaded``.
    Consumers exempting a whole TREE want ``_hermes_exempt_homes()``: under a named
    profile this home is ``<root>/profiles/<name>`` and the root is exempt too."""
    if _real_hermes_home_loaded:
        return _real_hermes_home_cached
    try:
        return _hermes_home_real()
    except Exception:
        # Same active-profile binding on the exception path (see
        # ``_get_hermes_config_resolved``): re-derive from ``get_hermes_home()``
        # rather than ``_expand_tilde("~/.hermes")`` so the protected-instruction
        # exemption resolves against the active profile — not the subprocess /
        # default home — and substitute no unrelated home if the active security
        # path cannot be established (PR #107335). A ``None`` here fails closed at
        # the consumer: the ``~/.hermes`` exemption is skipped, so the gate runs.
        try:
            from hermes_constants import get_hermes_home
            return os.path.realpath(str(get_hermes_home()))
        except Exception:
            return None


def _hermes_exempt_homes() -> tuple[str, ...]:
    """Realpaths of the Hermes home tree(s) the protected-instruction gate must stay out of:
    the ACTIVE profile's home, plus the Hermes ROOT when that home is a named profile
    (``<root>/profiles/<name>``). Exempting only the profile dir left the root's DIRECT files
    (LEDGER.md / MEMORY.md / SOUL.md / AGENTS.md ...) to the ``.hermes`` component rule, which
    gated them like a project-local ``<repo>/.hermes/config.yaml`` — fail-closed headless
    (#110630). They are the agent's own store, governed by their own guards, exactly like
    ``~/.hermes`` under the default profile. The root is added only when the shape really is a
    named profile (``named_profile_home``), so a coincidental ``profiles/`` dir elsewhere never
    exempts its parent; the home comes from the ACTIVE scope, never ``HERMES_HOME`` alone."""
    home = _get_real_hermes_home()
    if not home:
        return ()
    try:
        from hermes_constants import named_profile_home
        profile_home = named_profile_home(home)
    except Exception:
        profile_home = None
    if profile_home is None:
        return (home,)
    root = os.path.realpath(str(Path(str(profile_home)).parent.parent))
    return (home, root) if root and root != home else (home,)


def _resolved_or_raw(filepath: str, task_id: str) -> str:
    """Task-resolved path string, falling back to the raw input on resolution failure."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return filepath


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    # NT/device-namespace guard on the RAW string, BEFORE the task-base join:
    # on POSIX a leading "\??\" reads as a relative segment and gets anchored
    # under the base dir, hiding the prefix from the resolved-path checks,
    # while the same string relayed to a Windows host (remote backend, desktop
    # bridge) triggers the NTLM-leak vector. See agent/file_safety.py.
    nt_err = get_nt_namespace_error(filepath, verb="Write")
    if nt_err:
        return nt_err
    candidates = (_resolved_or_raw(filepath, task_id), os.path.normpath(_expand_tilde(filepath)))
    if any(c.startswith(_SENSITIVE_PATH_PREFIXES) or c in _SENSITIVE_EXACT_PATHS for c in candidates):
        return (
            f"Refusing to write to sensitive system path: {filepath}\n"
            "Use the terminal tool with sudo if you need to modify system files.")
    # approvals.mode and other security settings live in config.yaml; a
    # prompt-injected agent could silently disable exec approval by editing it.
    hermes_config = _get_hermes_config_resolved()
    if hermes_config and hermes_config in candidates:
        return (
            f"Refusing to write to Hermes config file: {filepath}\n"
            "Agent cannot modify security-sensitive configuration. "
            "Edit ~/.hermes/config.yaml directly or use 'hermes config' instead.")
    return None


# ── Protected agent-instruction files (always-ask approval gate) ─────────
# Files that steer FUTURE agent behavior are a prompt-injection persistence
# vector (AGENTS.md / CLAUDE.md / SOUL.md / .cursorrules / project .hermes tree).
# Writes ALWAYS require human approval — even under --yolo — and fail closed
# without a human channel. Basenames match in ANY directory, case-insensitively.
# Ported from: RooCodeInc/Roo-Code RooProtectedController (Apache-2.0). Companion: the terminal-tool vector
# is covered separately (#58631); this gate covers the write_file/patch vector. Symlink lesson from #41351:
# always realpath before matching. Scope decision (documented): basenames match in ANY directory, because
# project-context instruction files are loaded from cwd trees — an AGENTS.md anywhere the agent might later
# run from is a live target. Basenames match case-insensitively so case-variant spellings on
# case-insensitive filesystems (macOS/Windows) cannot slip past; on case-sensitive filesystems most loaders
# probe common case variants too, so the stricter behavior is kept uniform.
_PROTECTED_INSTRUCTION_BASENAMES = frozenset({
    "agents.md", "claude.md", "soul.md", ".cursorrules"})


def _protected_instruction_config() -> tuple[bool, list[str]]:
    """Return ``(enabled, extra_patterns)`` from ``security.protected_instruction_files`` /
    ``security.protected_instruction_extra_patterns`` (fnmatch on basename). Config read
    failures keep the gate ON — fail-safe for a security boundary."""
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        enabled = cfg_get(cfg, "security", "protected_instruction_files", default=True)
        extra = cfg_get(cfg, "security", "protected_instruction_extra_patterns", default=[])
    except Exception:
        return True, []
    if not isinstance(enabled, bool):
        enabled = True
    if not isinstance(extra, list):
        extra = []
    return enabled, [str(p) for p in extra if p]


def _protected_instruction_reason(filepath: str, task_id: str = "default",
                                  *, enabled: bool | None = None,
                                  extra_patterns: list[str] | None = None) -> str | None:
    """Return a short label when ``filepath`` targets a protected instruction file, else ``None``.
    Matches BOTH the normalized input and its realpath so no symlink direction escapes.

    Matching runs on BOTH the normalized input path and its realpath so neither a symlink pointing AT a
    protected file (#41351) nor a protected name that is itself a symlink escapes the gate. ``..`` traversal
    is neutralized by normpath/realpath before the basename compare.
    """
    if enabled is None or extra_patterns is None:
        enabled, extra_patterns = _protected_instruction_config()
    if not enabled:
        return None

    normalized = os.path.normpath(_expand_tilde(filepath))
    try:
        resolved = os.path.realpath(str(_resolve_path_for_task(filepath, task_id)))
    except (OSError, ValueError, RuntimeError):
        resolved = os.path.realpath(normalized)

    # ~/.hermes itself is governed by its own guards (config.yaml hard-block,
    # mirror guard, write_approval); this gate targets PROJECT-LOCAL files only.
    # Must run before the ``.hermes`` component rule, which would match the home.
    # ``_hermes_exempt_homes`` also covers the ROOT when the active home is a named
    # profile, so ~/.hermes/<file> cannot read as project-local ``.hermes`` config.
    for real_home in _hermes_exempt_homes():
        if resolved == real_home or resolved.startswith(real_home + os.sep):
            return None

    for candidate in (normalized, resolved):
        base = os.path.basename(candidate)
        base_lower = base.lower()
        if base_lower in _PROTECTED_INSTRUCTION_BASENAMES or any(
                fnmatch.fnmatch(base_lower, pattern.lower()) for pattern in extra_patterns):
            return base
        # Project-local .hermes config dirs (<repo>/.hermes/config.yaml) steer
        # behavior too. Only the IMMEDIATE parent counts — matching any ancestor
        # would gate every write inside a checkout living under ~/.hermes.
        parts = candidate.replace("\\", "/").rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2] == ".hermes":
            return candidate
    return None


_APPROVAL_UNAVAILABLE = "requires approval but the approval subsystem is unavailable."
_NO_HUMAN = "requires approval but no interactive user or gateway is present to approve it."

# A turn's serial batch of protected-instruction writes (e.g. three patch
# calls to AGENTS.md) used to re-prompt per call: one "Allow Once" covered
# only the first, and the rest failed closed as timed out for a batch the
# human had answered. An IDENTICAL follow-up in the same turn reuses the
# fresh grant — "identical" includes a digest of the exact operation payload,
# so a later write of different content/patch to the same path re-asks.
# Deny/timeout never record, nothing persists past the window, and no turn or
# payload identity means no reuse.
_PROTECTED_GRANT_REUSE_WINDOW_S = 120.0
_protected_grant_times: "dict[tuple[str, str, str, str, str], float]" = {}
_protected_grant_lock = threading.Lock()


def protected_write_digest(*parts: object) -> str:
    """Digest an operation's identity: tool tag, target path(s), payload text.

    Built by the write/patch entry points and again over the paths in
    ``_check_protected_instruction_write``. Length-prefixed so no two different
    payloads can concatenate to one string.
    """
    payload = "\x00".join(f"{len(text)}:{text}" for text in (str(part) for part in parts))
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def _prune_expired_protected_grants(now: float) -> None:
    """Drop grants past the reuse window (caller holds ``_protected_grant_lock``)."""
    for stale in [key for key, ts in _protected_grant_times.items()
                  if now - ts > _PROTECTED_GRANT_REUSE_WINDOW_S]:
        _protected_grant_times.pop(stale, None)


def _protected_grant_key(session_key: str, turn_id: str, display: str, description: str,
                         write_digest: str) -> tuple[str, str, str, str, str]:
    return (session_key or "", turn_id or "", display or "", description or "", write_digest or "")


def _protected_grant_is_fresh(key: tuple[str, str, str, str, str]) -> bool:
    if not key[1] or not key[4]:
        # Fail closed to prompting: without a turn identity AND a payload digest
        # a grant cannot be scoped to the exact write the human was shown.
        return False
    with _protected_grant_lock:
        _prune_expired_protected_grants(time.monotonic())
        return key in _protected_grant_times


def _record_protected_grant(key: tuple[str, str, str, str, str]) -> None:
    if not key[1] or not key[4]:
        return
    with _protected_grant_lock:
        now = time.monotonic()
        _prune_expired_protected_grants(now)
        _protected_grant_times[key] = now


def _request_protected_instruction_approval(reasons: list[str], task_id: str = "default",
                                            write_digest: str = "") -> str | None:
    """Ask the human to approve a write to protected instruction file(s); ``None`` when approved.

    Deliberately NOT routed through ``_run_approval_gate`` (honors --yolo and
    allowlists): this gate is one-operation approval, no persisted scope,
    fail-closed without a human channel. The ONE relaxation: on the gateway
    surface, an identical follow-up request within the SAME TURN reuses a
    freshly granted decision — a serial batch is one human answer, not N.
    """
    targets = ", ".join(dict.fromkeys(reasons))
    description = (
        f"Write to protected agent-instruction file(s): {targets}. "
        "These files steer future agent behavior; approval is always "
        "required (not bypassed by auto-approve).")
    display = f"<write to {targets}>"
    blocked = (
        f"BLOCKED: write to protected agent-instruction file(s) ({targets}) "
        "{why} The user has NOT consented to this write. Do NOT retry it or "
        "attempt the same edit via another path (terminal, execute_code, "
        "etc.).")
    timed_out = blocked.format(why="approval prompt timed out without a user response. Silence is not consent.")
    denied = blocked.format(why="was denied by the user.")

    try:
        import tools.approval as _approval
        from tools.approval_context import get_current_session_key, get_current_turn_id
        from tools.approval_gateway_wait import _await_gateway_decision
        from tools.approval_prompt import prompt_dangerous_approval
    except Exception:
        return blocked.format(why=_APPROVAL_UNAVAILABLE)

    # Gateway surface: block on the button round-trip when a notify callback
    # is registered for this session. One-operation only — no scope buttons.
    session_key = get_current_session_key()
    try:
        with _approval._lock:
            notify_cb = _approval._gateway_notify_cbs.get(session_key)
    except Exception:
        notify_cb = None

    if notify_cb is not None:
        approval_data = {
            "command": display,
            "pattern_key": "protected_instruction_file",
            "pattern_keys": ["protected_instruction_file"],
            "description": description,
            "allow_permanent": False,
            "allow_session": False}
        grant_key = _protected_grant_key(
            session_key, get_current_turn_id(), display, description, write_digest)
        if _protected_grant_is_fresh(grant_key):
            # A batch follow-up this turn: the human granted this exact request
            # moments ago — don't stack another card they answered.
            logger.info("Protected-instruction write covered by a fresh identical grant (session=%s, target=%s)",
                        session_key, targets)
            return None
        decision = _await_gateway_decision(session_key, notify_cb, approval_data, surface="gateway")
        if decision.get("notify_failed"):
            return blocked.format(why="requires approval but the approval request could not be delivered.")
        if decision.get("cancelled"):
            return blocked.format(why=f"approval was withdrawn before the user answered ({decision['cancelled']}).")
        choice, timed = decision.get("choice"), not decision.get("resolved")
        if not timed and choice in {"once", "session", "always"}:
            # Remember this grant for the turn's identical follow-ups; it
            # expires with the reuse window and never leaves this process.
            _record_protected_grant(grant_key)
    else:
        # CLI surface: per-thread approval callback (prompt_toolkit panel).
        try:
            from tools.terminal_tool import _get_approval_callback
            callback = _get_approval_callback()
        except Exception:
            callback = None
        if callback is None:
            # No human channel (script, cron, background thread): fail closed —
            # auto-approving here would recreate the persistence vector.
            return blocked.format(why=_NO_HUMAN)
        choice = prompt_dangerous_approval(
            display, description, allow_permanent=False, allow_session=False, approval_callback=callback)
        if choice == "cancelled":
            return blocked.format(why="approval prompt could not be delivered or was not answered "
                                      f"({getattr(choice, 'cause', 'no answer')}).")
        timed = choice == "timeout"
    # Any tapped scope is a one-operation grant; nothing is persisted.
    if not timed and choice in {"once", "session", "always"}:
        return None
    return timed_out if timed else denied


def _check_protected_instruction_write(paths: list[str], task_id: str = "default",
                                       write_digest: str = "") -> str | None:
    """Gate a write/patch touching protected instruction files. ONE protected file gates
    the ENTIRE multi-file patch (one prompt, all-or-nothing). ``write_digest`` identifies the
    exact operation payload a gateway approval may be reused for this turn."""
    enabled, extra = _protected_instruction_config()
    if not enabled:
        return None
    reasons = [r for r in (_protected_instruction_reason(p, task_id, enabled=enabled, extra_patterns=extra)
                           for p in paths) if r]
    if not reasons:
        return None
    # The grant identity includes the paths themselves: two same-basename
    # targets in different directories must not share one answered card.
    return _request_protected_instruction_approval(
        reasons, task_id, protected_write_digest(write_digest, *paths))


def _check_approval_required_write(paths: list[str], task_id: str = "default") -> str | None:
    """Gate a write/patch touching an approval-required path (``~/.ssh/config`` can steer
    execution via ``ProxyCommand``). Routine gate: once/session/always, honors --yolo,
    fail-closed without an interactive/gateway channel."""
    try:
        from agent.file_safety import is_write_approval_required
    except Exception:
        return None

    targets = [p for p in paths if is_write_approval_required(p)]
    if not targets:
        return None

    display_targets = ", ".join(dict.fromkeys(targets))
    description = (
        f"Write to SSH client config file(s): {display_targets}. "
        "The SSH config can carry ProxyCommand / Match exec directives that "
        "run commands, so writes require your approval.")
    blocked = (
        f"BLOCKED: write to SSH config file(s) ({display_targets}) "
        "{why} Do NOT retry it via another path (terminal, execute_code) "
        "without the user's explicit consent.")

    try:
        import tools.approval as _approval
    except Exception:
        return blocked.format(why=_APPROVAL_UNAVAILABLE)

    result = _approval._run_approval_gate(
        pattern_key="ssh_config_write",
        description=description,
        display_target=f"<write to {display_targets}>",
        cron_deny_message=blocked.format(why="requires approval but this cron session denies it."),
        single_query_deny_message=blocked.format(
            why="requires approval but single-query (-q) sessions run "
                "without a user present to approve it. To allow flagged "
                "actions in single-query mode, set approvals.single_query_mode: "
                "approve in config.yaml."),
        autoapprove_log_prefix="ssh_config_write",
        fail_closed_when_no_human=True,
        no_human_block_message=blocked.format(why=_NO_HUMAN))
    if result.get("approved"):
        return None
    return result.get("message") or blocked.format(why="was denied.")


def _get_container_mirror_prefix_for_task(task_id: str = "default") -> str | None:
    """Return the container-side Hermes mirror prefix for persistent Docker file tools."""
    try:
        from tools.terminal_tool import (
            _active_environments, _env_lock, _get_env_config, _resolve_container_task_id)
        container_key = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            persistent_docker = (env.__class__.__name__ == "DockerEnvironment"
                                 and bool(getattr(env, "_persistent", False)))
            return "/root/.hermes" if persistent_docker else None
        config = _get_env_config()
    except Exception:
        return None
    if config.get("env_type") == "docker" and config.get("container_persistent", True):
        return "/root/.hermes"
    return None


def _check_cross_profile_path(filepath: str, task_id: str = "default") -> str | None:
    """Soft-guard: warn when ``filepath`` lands on a host-side or Docker sandbox MIRROR of
    Hermes state (a write the host never reads). Not profile isolation — that guard was
    removed; ``cross_profile=True`` keeps bypassing this one for replay compat. Fails open."""
    try:
        from agent.file_safety import get_container_mirror_warning, get_sandbox_mirror_warning
    except Exception:
        return None
    resolved = _resolved_or_raw(filepath, task_id)
    warning = get_sandbox_mirror_warning(resolved)
    if warning is not None:
        return warning
    return get_container_mirror_warning(resolved, mirror_prefix=_get_container_mirror_prefix_for_task(task_id))


def _check_binary_document_write(filepath: str, task_id: str = "default") -> str | None:
    """Reject text-tool writes that would corrupt a binary document (read_file showed
    EXTRACTED text, so the model may write it back). Opaque document formats and
    SQLite sidecars (-wal/-shm/-journal) are always rejected; .pdf and every other
    BINARY_EXTENSIONS suffix only when OVERWRITING an existing file (raw PDF syntax
    is text-authorable and text fixtures named ``*.db`` exist).

    ``read_file`` auto-extracts .docx/.xlsx/.pptx (and PDF, via anydoc) to readable text, so the model
    plausibly believes it holds the file's contents and tries to write the edited text back with
    write_file/patch. A plain-text write can never produce a valid OOXML/OLE/ODF container, so that write
    silently destroys the document (port of nearai/ironclaw#7109).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if has_opaque_document_extension(filepath):
        return (
            f"Refusing to write plain text to binary document '{filepath}' ({ext}). "
            "A text write cannot produce a valid document container and would "
            "corrupt the file (read_file showed you EXTRACTED text, not the real "
            "bytes). Use the docx/xlsx/powerpoint skills or a library like "
            "python-docx/openpyxl/python-pptx via the terminal to create or edit "
            "this document.")
    # A -wal/-shm/-journal path is never a legitimate text target, even when
    # no sidecar exists yet: a checkpointed db has none on disk, and a garbage
    # WAL dropped next to a live database is picked up on the next open.
    if is_sqlite_sidecar(filepath):
        return (
            f"Refusing to write plain text to binary SQLite sidecar '{filepath}' ({ext}). "
            "A -wal/-shm/-journal file holds raw database pages that SQLite "
            "reads on the next open; text there corrupts the database. Use the "
            "sqlite3 CLI or a SQLite library via the terminal to modify the "
            "database instead.")
    # Overwriting an existing binary (PDF, image, archive, SQLite db, ...)
    # with text destroys it — the model only ever saw extracted or mojibake
    # text. Creating a NEW file with such an extension stays allowed: raw PDF
    # syntax is text-authorable and text fixtures named ``*.db`` exist.
    pdf = is_pdf_path(filepath)
    if pdf or has_binary_extension(filepath):
        try:
            resolved = Path(_resolve_path_for_task(filepath, task_id))
        except Exception:
            resolved = Path(_expand_tilde(filepath))
        try:
            if resolved.is_file():
                if pdf:
                    return (
                        f"Refusing to overwrite existing PDF '{filepath}' with plain text. "
                        "read_file showed you EXTRACTED text, not the real bytes — writing "
                        "text back would destroy the document. Use the pdf skill or a PDF "
                        "library via the terminal to modify it. (Creating a NEW .pdf file "
                        "is allowed.)")
                return (
                    f"Refusing to overwrite existing binary file '{filepath}' ({ext}) "
                    "with plain text — read_file showed you extracted or mojibake "
                    "text, not the real bytes, and writing text back would destroy "
                    "the file. Use a binary-aware tool via the terminal to modify it "
                    "(for SQLite databases, the sqlite3 CLI or a SQLite library). "
                    "(Creating a NEW file with this extension is allowed.)")
        except OSError:
            pass
    return None


# ── Internal display text must never be persisted as file content ────────
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current — refer to that instead of re-reading.")


def _stale_overwrite_blocker(filepath: str, resolved: str | None, task_id: str) -> str | None:
    """Reason write_file must NOT replace the existing file, else ``None``.

    Refuses BEFORE any disk mutation (the pre-#65604 warning arrived after the
    clobber): a sibling/external/partial-read staleness finding, or an existing
    file with no full-content baseline for this task (never read in full, read
    redacted, only patched). Net-new files, files this task fully read (in one
    page or by paging contiguously to the last line) or wrote, unresolvable
    paths and the file-state kill switch all let the write proceed.
    """
    if file_state.guard_disabled():
        return None
    stale = file_state.check_stale(task_id, resolved) if resolved else None
    if stale:
        return stale
    if _read_mtime_drifted(filepath, task_id):
        return (
            f"{filepath} was modified since you last read it (external edit or "
            "concurrent agent). Re-read the file before writing.")
    if not resolved or _has_full_write_baseline(resolved, task_id):
        return None
    try:
        exists = Path(resolved).exists()
    except OSError:
        return None
    if not exists:
        return None
    return (
        f"{resolved} exists but this task has not seen its full current content "
        "(never read, only patched, or only a redacted/partial view). Read the "
        "file — every page of it, if it needs offset/limit — or use patch for a "
        "targeted edit; a stale conversation copy must not overwrite the current "
        "disk content.")


def _stale_write_refusal(filepath: str, reason: str, resolved: str | None = None) -> dict:
    """Model-facing refusal payload for write_file; ``stale_write_blocked`` lets
    callers tell it apart from I/O errors."""
    result = {
        "error": (
            f"Refusing to overwrite {filepath}: {reason} "
            "The file was NOT modified. Reload the current contents with read_file "
            "(every page, for a file that needs offset/limit), merge the requested "
            "change, then call write_file again. For small edits, prefer patch so "
            "existing unrelated changes are preserved."),
        "stale_write_blocked": True,
        "path": filepath,
    }
    if resolved:
        result["resolved_path"] = resolved
    return result


def _is_internal_file_status_text(content: str) -> bool:
    """True when content is the read_file dedup status message, verbatim or lightly framed
    (contains the full message and is <=2x its length — a real file quoting it would be longer)."""
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    return bool(stripped) and _READ_DEDUP_STATUS_MESSAGE in stripped and (
        len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE))


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """True for content dominated by read_file's ``LINE_NUM|CONTENT`` display (>=60% of
    non-empty lines are consecutive numbered lines; a lone ``1|value`` is allowed)."""
    if not isinstance(content, str):
        return False
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered: list[int] = []
    for line in lines:
        prefix, sep, _rest = line.lstrip().partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))
    if len(numbered) < 2 or len(numbered) / len(lines) < 0.6:
        return False
    consecutive_pairs = sum(1 for prev, current in zip(numbered, numbered[1:]) if current == prev + 1)
    return consecutive_pairs >= len(numbered) - 1


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return _is_internal_file_status_text(content) or _looks_like_read_file_line_numbered_content(content)
