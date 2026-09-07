"""Write-side safety guards for write_file / patch.

Every guard returns ``None`` when the write may proceed, else an error string
the tool returns verbatim.
Guards, in the order the tools apply them: ``_check_sensitive_path`` (hard
deny), ``_check_binary_document_write``, ``_check_protected_instruction_write``
(ALWAYS ask), ``_check_approval_required_write`` (normal gate),
``_check_cross_profile_path`` (sandbox-mirror lost-work), ``_is_internal_file_tool_content``.
"""

import fnmatch
import os
from pathlib import Path

from tools.binary_extensions import has_opaque_document_extension, is_pdf_path
from tools.file_tools_paths import _expand_tilde, _resolve_path_for_task

# Prefixes matched after realpath. macOS: /private/var mirrors /var — block the
# sensitive subtrees only; a blanket "/private/var/" refuses every temp-file
# write because $TMPDIR, /tmp and /var/folders all realpath there.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/",
    "/private/var/db/", "/private/var/root/")
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False
_real_hermes_home_cached: str | None = None
_real_hermes_home_loaded = False


def _cached_lookup(slot: str, flag: str, primary, fallback) -> str | None:
    """Fill module global *slot* once (guarded by *flag*) from ``primary()``, else
    ``fallback()``, else None. Module globals so tests can monkeypatch the slots."""
    g = globals()
    if not g[flag]:
        g[flag] = True
        try:
            g[slot] = primary()
        except Exception:
            try:
                g[slot] = fallback()
            except Exception:
                g[slot] = None
    return g[slot]


def _config_path_resolved() -> str:
    from hermes_cli.config import get_config_path
    return str(get_config_path().resolve())


def _hermes_home_real() -> str:
    from hermes_constants import get_hermes_home
    return os.path.realpath(str(get_hermes_home()))


def _get_hermes_config_resolved() -> str | None:
    """Return the resolved absolute path of the Hermes config file (cached)."""
    return _cached_lookup("_hermes_config_resolved", "_hermes_config_resolved_loaded", _config_path_resolved,
                          lambda: str(Path(_expand_tilde("~/.hermes/config.yaml")).resolve()))


def _get_real_hermes_home() -> str | None:
    """Return the realpath of the authoritative Hermes home (cached)."""
    return _cached_lookup("_real_hermes_home_cached", "_real_hermes_home_loaded", _hermes_home_real,
                          lambda: os.path.realpath(_expand_tilde("~/.hermes")))


def _resolved_or_raw(filepath: str, task_id: str) -> str:
    """Task-resolved path string, falling back to the raw input on resolution failure."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return filepath


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
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
    real_home = _get_real_hermes_home()
    if real_home and (resolved == real_home or resolved.startswith(real_home + os.sep)):
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


def _request_protected_instruction_approval(reasons: list[str], task_id: str = "default") -> str | None:
    """Ask the human to approve a write to protected instruction file(s); ``None`` when approved.

    Deliberately NOT routed through ``_run_approval_gate`` (honors --yolo and
    allowlists): this gate is one-operation approval EVERY time, no persisted
    scope, fail-closed without a human channel.
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
        from tools.approval_context import get_current_session_key
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
        decision = _await_gateway_decision(session_key, notify_cb, approval_data, surface="gateway")
        if decision.get("notify_failed"):
            return blocked.format(why="requires approval but the approval request could not be delivered.")
        choice, timed = decision.get("choice"), not decision.get("resolved")
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
        timed = choice == "timeout"
    # Any tapped scope is a one-operation grant; nothing is persisted.
    if not timed and choice in {"once", "session", "always"}:
        return None
    return timed_out if timed else denied


def _check_protected_instruction_write(paths: list[str], task_id: str = "default") -> str | None:
    """Gate a write/patch touching protected instruction files. ONE protected file gates
    the ENTIRE multi-file patch (one prompt, all-or-nothing)."""
    enabled, extra = _protected_instruction_config()
    if not enabled:
        return None
    reasons = [r for r in (_protected_instruction_reason(p, task_id, enabled=enabled, extra_patterns=extra)
                           for p in paths) if r]
    if not reasons:
        return None
    return _request_protected_instruction_approval(reasons, task_id)


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
    EXTRACTED text, so the model may write it back). Opaque formats are always rejected;
    .pdf only when OVERWRITING an existing file (raw PDF syntax is text-authorable).

    ``read_file`` auto-extracts .docx/.xlsx/.pptx (and PDF, via anydoc) to readable text, so the model
    plausibly believes it holds the file's contents and tries to write the edited text back with
    write_file/patch. A plain-text write can never produce a valid OOXML/OLE/ODF container, so that write
    silently destroys the document (port of nearai/ironclaw#7109).
    """
    if has_opaque_document_extension(filepath):
        ext = filepath[filepath.rfind("."):].lower()
        return (
            f"Refusing to write plain text to binary document '{filepath}' ({ext}). "
            "A text write cannot produce a valid document container and would "
            "corrupt the file (read_file showed you EXTRACTED text, not the real "
            "bytes). Use the docx/xlsx/powerpoint skills or a library like "
            "python-docx/openpyxl/python-pptx via the terminal to create or edit "
            "this document.")
    if is_pdf_path(filepath):
        try:
            resolved = Path(_resolve_path_for_task(filepath, task_id))
        except Exception:
            resolved = Path(_expand_tilde(filepath))
        try:
            if resolved.is_file():
                return (
                    f"Refusing to overwrite existing PDF '{filepath}' with plain text. "
                    "read_file showed you EXTRACTED text, not the real bytes — writing "
                    "text back would destroy the document. Use the pdf skill or a PDF "
                    "library via the terminal to modify it. (Creating a NEW .pdf file "
                    "is allowed.)")
        except OSError:
            pass
    return None


# ── Internal display text must never be persisted as file content ────────
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current — refer to that instead of re-reading.")


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
