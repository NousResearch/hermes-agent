#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools.

Companions: ``file_tools_paths`` (task-aware resolution), ``file_tools_write_guards``
(write-side guards), ``file_tools_read_tracking`` (per-task dedup / loop-detection /
staleness state).
"""

import base64
import errno
import json
import logging
import os
import re
import stat
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    ShellFileOperations, normalize_read_pagination, normalize_search_pagination)
from tools import file_state
from agent.redact import redact_sensitive_text
from tools.file_tools_paths import (
    _expand_tilde, _path_resolution_warning, _resolve_base_dir, _resolve_path_for_task)
from tools.file_tools_write_guards import (
    _READ_DEDUP_STATUS_MESSAGE, _check_approval_required_write, _check_binary_document_write,
    _check_cross_profile_path, _check_protected_instruction_write, _check_sensitive_path,
    _is_internal_file_tool_content)
from tools.file_tools_read_tracking import (
    _bump_consecutive, _cap_read_tracker_data, _check_file_staleness, _check_not_found_cache,
    _mark_verification_stale, _patch_failure_lock, _patch_failure_tracker, _read_tracker,
    _read_tracker_lock, _record_not_found, _record_patch_failure, _reset_patch_failures,
    _task_data, _update_read_timestamp)

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}

# Read-size guard. Model-agnostic, so characters proxy tokens: 100K chars is
# ~25-35K tokens across typical tokenisers. Configurable: file_read_max_chars.
_DEFAULT_MAX_READ_CHARS = 100_000
_max_read_chars_cached: int | None = None


def _get_max_read_chars() -> int:
    """Return ``file_read_max_chars`` from config.yaml (cached per process; default on missing/invalid)."""
    global _max_read_chars_cached
    if _max_read_chars_cached is None:
        try:
            from hermes_cli.config import load_config
            val = load_config().get("file_read_max_chars")
        except Exception:
            val = None
        valid = isinstance(val, (int, float)) and val > 0
        _max_read_chars_cached = int(val) if valid else _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to the last COMPLETE line within *max_chars*.

    Returns ``(kept_text, lines_kept, truncated)`` so the caller can offer a
    ``next_offset`` instead of rejecting the read. If not even the first line
    fits it is clamped mid-line so the read is never empty and the cursor advances.

    Ported in spirit from nearai/ironclaw#5029 (dual line/byte cap on ``read_file``). Where hermes
    previously hard-rejected an oversized read (forcing the model to guess a smaller ``limit`` and burn a
    round-trip returning nothing), this trims the content to the last *complete line* that fits within
    ``max_chars`` and reports how many lines were kept so the caller can offer a ``next_offset``
    continuation.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        addition = len(line) + (1 if kept else 0)  # +1 for the rejoining "\n"
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition
    if not kept:
        kept.append(lines[0][:max_chars])
    return "\n".join(kept), len(kept), True


def _apply_char_budget(result_dict: dict, content: str, offset: int, total_lines, max_chars: int) -> str:
    """Trim *content* to the char budget, annotate *result_dict* with the
    continuation hint, and return the trimmed text."""
    trimmed, lines_kept, _ = _truncate_to_char_budget(content, max_chars)
    next_offset = offset + lines_kept
    result_dict["content"] = trimmed
    result_dict["truncated"] = True
    result_dict["truncated_by"] = "bytes"
    result_dict["next_offset"] = next_offset
    result_dict["hint"] = (
        f"Output truncated at the {max_chars:,}-char read budget after "
        f"{lines_kept} line(s) (showing lines {offset}-{next_offset - 1} of "
        f"{total_lines}). Use offset={next_offset} to continue.")
    if len(trimmed.split("\n", 1)[0]) >= max_chars:
        result_dict["hint"] += (
            " Note: the first line alone exceeded the budget and was "
            "clamped mid-line; its remainder is not retrievable via offset.")
    return trimmed


# Above this size, a wide read (limit > 200) gets a hint toward targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000

# Device/fd paths whose reads hang the process. Checked by path only — no I/O.
_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",     # never reach EOF
    "/dev/stdin", "/dev/tty", "/dev/console",                    # block on input
    "/dev/stdout", "/dev/stderr",                                # nonsensical to read
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",                       # fd aliases
})
# /proc/<pid>/... (and /proc/<pid>/task/<tid>/...) files that leak secrets,
# argv, memory layout (ASLR oracle: maps family, auxv, pagemap) or raw memory.
_BLOCKED_PROC_SUFFIXES = (
    "/fd/0", "/fd/1", "/fd/2",  # stdio aliases
    "/environ", "/cmdline", "/maps", "/smaps", "/smaps_rollup", "/numa_maps",
    "/mem", "/auxv", "/pagemap")


def _file_ops_uses_host_paths(file_ops) -> bool:
    """True when *file_ops* targets the host filesystem (only then may we stat paths
    or rewrite V4A headers to host-absolute paths; sandboxes have their own namespace)."""
    env = getattr(file_ops, "env", None)
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment
    except ImportError:
        return True
    return isinstance(env, LocalEnvironment)


# V4A file headers: group 1 = header prefix, 2 = op, 3 = path. ``\s*`` after
# ``***`` mirrors patch_parser's leniency (``***Update File:`` applies, so it
# must be checked).
_V4A_SINGLE_HEADER_RE = re.compile(r'^(\*\*\*\s*(Update|Add|Delete)\s+File:\s*)(.+)$', re.MULTILINE)
_V4A_MOVE_HEADER_RE = re.compile(r'^(\*\*\*\s*Move\s+File:\s*)(.+?)\s*->\s*(.+)$', re.MULTILINE)


def _rewrite_v4a_patch_paths_for_host(patch: str, path_to_resolved: dict, file_ops) -> str:
    """Rewrite V4A file headers to the resolved host paths (host backends only).

    The shell layer must patch the SAME files ``patch_tool`` resolved for
    locking/staleness, not re-resolve a relative header against its own cwd
    (which can differ — the git-worktree cwd bug).
    """
    if not _file_ops_uses_host_paths(file_ops):
        return patch

    def _res(raw: str) -> str:
        raw = raw.strip()
        return path_to_resolved.get(raw) or raw

    patch = _V4A_SINGLE_HEADER_RE.sub(lambda m: f"{m.group(1)}{_res(m.group(3))}", patch)
    return _V4A_MOVE_HEADER_RE.sub(lambda m: f"{m.group(1)}{_res(m.group(2))} -> {_res(m.group(3))}", patch)


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd/proc paths that can hang reads or leak process state."""
    normalized = os.path.normpath(_expand_tilde(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    return normalized.startswith("/proc/") and normalized.endswith(_BLOCKED_PROC_SUFFIXES)


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """True if the path (literal, any symlink hop, or final realpath) is a blocked device.

    Literal first so /dev/stdin is caught before resolving to a terminal path;
    every symlink hop is checked so an alias cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    return _is_blocked_device_path(resolved)


def _filter_read_blocked_search_results(result, task_id: str = "default") -> int:
    """Remove credential/cache/env paths from a SearchResult in-place; return the omitted count.

    Each path is resolved against the task cwd first (search backends may
    return cwd-relative paths; the process cwd can differ).
    """
    omitted = 0

    def _allowed(path: str) -> bool:
        nonlocal omitted
        try:
            target = str(_resolve_path_for_task(path, task_id))
        except (OSError, ValueError, RuntimeError):
            target = path
        if get_read_block_error(target):
            omitted += 1
            return False
        return True

    if getattr(result, "matches", None):
        result.matches = [m for m in result.matches if _allowed(m.path)]
    if getattr(result, "files", None):
        result.files = [f for f in result.files if _allowed(f)]
    if getattr(result, "counts", None):
        result.counts = {f: c for f, c in result.counts.items() if _allowed(f)}
    return omitted


def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials that should not hit error logs."""
    return isinstance(exc, PermissionError) or (
        isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS)


# ── ShellFileOperations per terminal environment ─────────────────────────
_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}


def _create_terminal_env_for_file_ops(raw_task_id: str, task_id: str):
    """Build the terminal environment for *task_id* via the shared ``_create_configured_env``,
    so a file tool that runs before any terminal command still gets the configured backend."""
    from tools.terminal_tool_config import _is_container_backend
    from tools.terminal_tool import (
        _create_configured_env, _get_env_config, _is_unusable_container_cwd,
        _resolve_task_host_cwd, _select_image, get_session_cwd, resolve_task_overrides)

    config = _get_env_config()
    env_type = config["env_type"]
    overrides = resolve_task_overrides(raw_task_id)
    try:
        recorded_cwd = get_session_cwd(raw_task_id)
    except Exception:
        recorded_cwd = None
    cwd = overrides.get("cwd") or recorded_cwd or config["cwd"]
    # Re-apply the container cwd guard: a gateway/TUI/ACP override is a raw HOST
    # path and ``docker run -w <host-path>`` makes search_files & co silently
    # return nothing. Valid in-container overrides (/workspace, /root) pass.
    # Re-apply the container cwd guard that _get_env_config() already ran on config["cwd"] (see #50636). A
    # per-task cwd override registered by the gateway/TUI/ACP for workspace tracking is a raw host path
    # (e.g. a Desktop session's /Users/<me>/workspace or C:\\Users\\<me>). On a container backend that
    # reaches ``docker run -w <host-path>`` and the container starts in a directory that doesn't exist
    # inside the sandbox, so search_files and friends silently return empty results (#54447). Sanitize it
    # back to the already-validated config["cwd"] so the override can't bypass the guard.
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        if cwd != config["cwd"]:
            logger.info(
                "Ignoring host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, config["cwd"])
        cwd = config["cwd"]
    logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])
    terminal_env = _create_configured_env(
        config, env_type, image=_select_image(env_type, overrides, config), cwd=cwd,
        timeout=config["timeout"], task_id=task_id,
        host_cwd=_resolve_task_host_cwd(config, raw_task_id),
        local_config={"persistent": config.get("local_persistent", False)} if env_type == "local" else None,
    )
    return env_type, terminal_env


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for the task's terminal environment.

    Uses terminal_tool's per-task creation locks (no duplicate sandboxes).
    Subagent task_ids collapse to "default" (``_resolve_container_task_id``) so
    delegate_task children share the parent's container; RL/benchmark task_ids
    with a registered env override keep their isolation.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _last_activity, _start_cleanup_thread,
        _creation_locks, _creation_locks_lock, _resolve_container_task_id,
        get_session_cwd, record_session_cwd)

    raw_task_id = task_id or "default"
    task_id = _resolve_container_task_id(raw_task_id)

    # Fast path: cached AND the environment is still alive (cleanup thread may have killed it).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                return cached
            # Env was cleaned up: rescue its cwd into the session record FILL-ONLY
            # (``cached.cwd`` is the SHARED env's cwd, not this session's own).
            # Environment was cleaned up -- preserve the old cwd in the session record before invalidating
            # the stale cache entry (fixes #26211: silent file-creation failures in long-running
            # conversations). Usually a no-op: every completed command already recorded its cwd. Fill-only:
            # ``cached.cwd`` is a snapshot of the SHARED env's cwd at cache-build time, so it is not
            # attributable to this session (same class as the interrupted-command bug, #85658). Rescue a
            # session that has no record, but never overwrite a record the session wrote for itself.
            old_cwd = getattr(cached, "cwd", None)
            if old_cwd:
                try:
                    if get_session_cwd(raw_task_id) is None:
                        record_session_cwd(raw_task_id, old_cwd)
                except Exception:
                    pass
            with _file_ops_lock:
                _file_ops_cache.pop(task_id, None)

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(task_id, threading.Lock())

    with task_lock:
        # Double-check: another thread may have created it while we waited.
        with _env_lock:
            terminal_env = _active_environments.get(task_id)
            if terminal_env is not None:
                _last_activity[task_id] = time.time()
        if terminal_env is None:
            env_type, terminal_env = _create_terminal_env_for_file_ops(raw_task_id, task_id)
            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()
            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops


def clear_file_ops_cache(task_id: str = None):
    """Clear file-operation state for a finished task, or all tasks."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()

    with _read_tracker_lock:
        if task_id:
            _read_tracker.pop(task_id, None)
        else:
            _read_tracker.clear()

    with _patch_failure_lock:
        if task_id:
            _patch_failure_tracker.pop(task_id, None)
        else:
            _patch_failure_tracker.clear()

    if task_id:
        file_state.get_registry().forget_task(task_id)
    else:
        file_state.get_registry().clear()


_SPECIAL_FILE_KINDS = (
    (stat.S_ISFIFO, "a FIFO (named pipe)"),
    (stat.S_ISSOCK, "a socket"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"))


def _special_file_kind(path) -> str | None:
    """Human name for a non-regular file type that would hang a read, else None.

    Stat-based sibling of ``_is_blocked_device``: a FIFO/socket in a workspace
    hangs like ``/dev/zero`` but has no recognizable name. Host filesystems
    only; unstattable paths return None and flow to the normal read path.
    """
    try:
        mode = os.stat(os.fspath(path)).st_mode  # follows symlinks, matching a real read
    except OSError:
        return None
    if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
        return None
    return next((label for predicate, label in _SPECIAL_FILE_KINDS if predicate(mode)),
                "a special (non-regular) file")


def _read_extracted_document(path: str, _resolved, offset: int, limit: int, task_id: str) -> str | None:
    """Render an extractable document (.docx/.xlsx/.pdf/...) as paginated text.

    Returns the JSON result, a tool_error for an actionable extraction failure
    (size cap, encrypted, malformed), or ``None`` to fall through to the normal
    read path. Runs BEFORE the binary-extension guard.
    """
    from tools.read_extract import (
        ANYDOC_EXTENSIONS, EXTRACTABLE_EXTENSIONS, MAX_DOCUMENT_BYTES, ExtractionError,
        extract_document_bytes, is_extractable_document)

    if not is_extractable_document(str(_resolved)):
        return None
    file_ops = _get_file_ops(task_id)
    try:
        binary = file_ops.read_file_bytes(str(_resolved), max_bytes=MAX_DOCUMENT_BYTES)
        if binary.error or binary.base64_content is None:
            raise ExtractionError(binary.error or "Document bytes unavailable")
        document_bytes = base64.b64decode(binary.base64_content, validate=True)
        extracted_text = extract_document_bytes(document_bytes, str(_resolved))
    except (ExtractionError, ValueError, base64.binascii.Error) as exc:
        logger.debug("document extraction failed for %s", path, exc_info=True)
        # Binary formats surface the specific failure (fallthrough would only
        # give a generic binary error); .ipynb and byte-transport errors fall through.
        _doc_ext = _resolved.suffix.lower()
        _binary_doc = _doc_ext in ANYDOC_EXTENSIONS or (
            _doc_ext in EXTRACTABLE_EXTENSIONS and _doc_ext != ".ipynb")
        if (_binary_doc and isinstance(exc, ExtractionError)
                and not str(exc).startswith("Unsupported document type")):
            return tool_error(
                f"Cannot read '{path}' ({_doc_ext}): document "
                f"extraction failed — {exc}. Use terminal utilities "
                "to inspect or convert the file.")
        return None

    lines = extracted_text.splitlines()
    total_lines = len(lines)
    end_line = offset + limit - 1
    page_text = "\n".join(lines[offset - 1:end_line])
    result_dict = {
        "content": file_ops._add_line_numbers(page_text, offset) if page_text else "",
        "total_lines": total_lines,
        "file_size": binary.file_size,
        "truncated": total_lines > end_line,
        "extracted_document": True}
    if result_dict["truncated"]:
        result_dict["hint"] = (
            f"Use offset={end_line + 1} to continue reading "
            f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)")
    max_chars = _get_max_read_chars()
    if len(result_dict["content"]) > max_chars:
        _apply_char_budget(result_dict, result_dict["content"], offset, total_lines, max_chars)
    if result_dict["content"]:
        result_dict["content"] = redact_sensitive_text(result_dict["content"], file_read=True)
    return json.dumps(result_dict, ensure_ascii=False)


def _dedup_stub_or_block(task_data: dict, dedup_key: tuple, path: str) -> str:
    """Return the "unchanged" stub for a repeated identical read, escalating to a
    hard BLOCK after 2 stubs so weak tool-followers don't loop forever."""
    with _read_tracker_lock:
        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
        task_data["dedup_hits"][dedup_key] = hits
        _cap_read_tracker_data(task_data)

    if hits >= 2:
        return tool_error(
            f"BLOCKED: You have called read_file on this "
            f"exact region {hits + 1} times and the file "
            "has NOT changed. STOP calling read_file for "
            "this path — the content from your earlier "
            "read_file result in this conversation is "
            "still current. Proceed with your task using "
            "the information you already have.",
            path=path,
            already_read=hits + 1)

    return json.dumps({
        "status": "unchanged",
        "message": _READ_DEDUP_STATUS_MESSAGE,
        "path": path,
        "dedup": True,
        "content_returned": False,
    }, ensure_ascii=False)


def _record_successful_read(task_data: dict, task_id: str, path: str, resolved_str: str,
                            offset: int, limit: int, dedup_key: tuple, *, partial: bool) -> int:
    """Bookkeeping after a real (non-stub) read; returns the consecutive-read count.

    Per-task tracker under the lock (stub counter, history, consecutive count,
    mtime for dedup + staleness). Then OUTSIDE our lock (no nested locking): the
    cross-agent registry, and the background-review read-mark (a FULL read of a
    skill file counts like skill_view so a follow-up skill_manage(patch) is accepted).
    """
    with _read_tracker_lock:
        task_data["dedup_hits"].pop(dedup_key, None)
        task_data["dedup_generation_reads"].add(dedup_key)
        task_data["read_history"].add((path, offset, limit))
        count = _bump_consecutive(task_data, ("read", path, offset, limit))
        try:
            _mtime_now = os.path.getmtime(resolved_str)
            task_data["dedup"][dedup_key] = _mtime_now
            task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
        except OSError:
            pass
        _cap_read_tracker_data(task_data)

    try:
        file_state.record_read(task_id, resolved_str, partial=partial)
    except Exception:
        logger.debug("file_state.record_read failed", exc_info=True)

    if not partial:
        try:
            # Background-review read-before-write guard integration (#61521): when the self-improvement
            # review fork reads a skill file with read_file (now whitelisted dispatch-side), register the
            # read the same way skill_view does, so a follow-up skill_manage(action='patch') on the loaded
            # file is accepted. A partial read doesn't count — the guard requires the CURRENT full content
            # to have been seen. No-op outside review forks (mark_background_review_skill_read gates on
            # is_background_review).
            from tools.skill_manager_guards import mark_background_review_skill_read
            mark_background_review_skill_read(Path(resolved_str))
        except Exception:
            logger.debug("background-review read-mark failed", exc_info=True)
    return count


def read_file_tool(path: str, offset: int = 1, limit: int = 2000, task_id: str = "default") -> str:
    """Read a file with pagination and line numbers.

    Guard order: device-path blocklist (no I/O) → stat-based special-file
    guard (host only) → document extraction → binary-extension guard → Hermes
    internal denylist → negative-result cache → dedup stub → real read.
    """
    try:
        offset, limit = normalize_read_pagination(offset, limit)

        device_base = None if Path(path).expanduser().is_absolute() else _resolve_base_dir(task_id)
        if _is_blocked_device(path, base_dir=device_base):
            return tool_error(
                f"Cannot read '{path}': this is a device file that would "
                "block or produce infinite output.")

        _resolved = _resolve_path_for_task(path, task_id)

        # A read on a FIFO/socket blocks until the exec timeout: a self-shipped DoS.
        if _file_ops_uses_host_paths(_get_file_ops(task_id)):
            kind = _special_file_kind(_resolved)
            if kind is not None:
                return json.dumps({
                    "success": False,
                    "note": (
                        f"'{path}' is {kind}, not a regular file — reading "
                        "it would block indefinitely, so no read was "
                        "attempted. Use terminal utilities if you need to "
                        "interact with it.")})

        extracted = _read_extracted_document(path, _resolved, offset, limit, task_id)
        if extracted is not None:
            return extracted

        # The extension is a claim, so this message names only the extension;
        # the content-sniffing path names the actual magic-byte type.
        if has_binary_extension(str(_resolved)):
            return tool_error(
                f"Cannot read binary file '{path}' ({_resolved.suffix.lower()}). "
                "Use vision_analyze for images, or terminal to inspect binary files.")

        # Hermes internal denylist (prompt injection via catalog metadata,
        # credential stores). Pass the RESOLVED path: the denylist's own
        # resolve() uses the process cwd and would miss a relative "auth.json".
        block_error = get_read_block_error(str(_resolved))
        if block_error:
            return tool_error(block_error)

        resolved_str = str(_resolved)
        cached_not_found = _check_not_found_cache("read", resolved_str, task_id)
        if cached_not_found is not None:
            return cached_not_found

        # Dedup: identical (path, offset, limit) on an unchanged file returns a
        # lightweight stub instead of re-sending the content.
        dedup_key = (resolved_str, offset, limit)
        with _read_tracker_lock:
            task_data = _task_data(task_id)
            cached_mtime = task_data["dedup"].get(dedup_key)
            # First unchanged read after a compaction boundary serves full content
            # (the summary may have dropped exact bytes); later ones get the stub.
            content_served_in_generation = dedup_key in task_data["dedup_generation_reads"]
        if cached_mtime is not None:
            try:
                if os.path.getmtime(resolved_str) == cached_mtime and content_served_in_generation:
                    return _dedup_stub_or_block(task_data, dedup_key, path)
            except OSError:
                pass  # stat failed — fall through to full read

        result = _get_file_ops(task_id).read_file(path, offset, limit)
        result_dict = result.to_dict()

        # Cache a not-found result for retries. Deliberately NO early return:
        # error results still flow through the tracking below unchanged.
        _err = result_dict.get("error") or ""
        if isinstance(_err, str) and _err.startswith("File not found:"):
            _record_not_found("read", resolved_str, task_id, json.dumps(result_dict, ensure_ascii=False))

        # Char budget on the FORMATTED content (what enters context), BEFORE
        # redaction (skip the regex pass on huge content); truncate gracefully
        # with a next_offset instead of rejecting.
        file_size = result_dict.get("file_size", 0)
        max_chars = _get_max_read_chars()
        if len(result.content or "") > max_chars:
            result.content = _apply_char_budget(
                result_dict, result.content or "", offset,
                result_dict.get("total_lines", "unknown"), max_chars)
        if result.content:
            result.content = redact_sensitive_text(result.content, file_read=True)
            result_dict["content"] = result.content

        if (file_size and file_size > _LARGE_FILE_HINT_BYTES
                and limit > 200 and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."))

        count = _record_successful_read(task_data, task_id, path, resolved_str, offset, limit,
                                        dedup_key, partial=(offset > 1) or bool(result_dict.get("truncated")))
        if count >= 4:
            return tool_error(
                f"BLOCKED: You have read this exact file region {count} times in a row. "
                "The content has NOT changed. You already have this information. "
                "STOP re-reading and proceed with your task.",
                path=path,
                already_read=count)
        if count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times consecutively. "
                "The content has not changed since your last read. Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with writing or responding.")
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


# ── Shared write/patch plumbing ──────────────────────────────────────────

def _resolve_or_none(filepath: str, task_id: str) -> str | None:
    """Task-resolved path string, or None when resolution fails for any reason."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except Exception:
        return None


def _write_precheck_error(paths: list[str], content_paths: list[str], task_id: str,
                          cross_profile: bool) -> str | None:
    """Run the shared write/patch guards in order; return the first error string.

    Order matters: hard denies (sensitive path, mirror) and the corruption
    guard run before anything that could prompt the user, and ONE approval
    prompt covers every path of a multi-file patch.
    """
    for p in paths:
        err = _check_sensitive_path(p, task_id) or (
            None if cross_profile else _check_cross_profile_path(p, task_id))
        if err:
            return err
    for p in content_paths:
        err = _check_binary_document_write(p, task_id)
        if err:
            return err
    return (_check_protected_instruction_write(paths, task_id)
            or _check_approval_required_write(paths, task_id))


def _edit_warnings(paths: list[str], path_to_resolved: dict, task_id: str) -> list[str]:
    """One pre-edit warning per path, in priority order: cross-agent registry
    (names the sibling subagent) > per-task staleness > workspace divergence
    (relative path resolving outside the terminal's cwd — the worktree-cwd bug)."""
    warnings: list[str] = []
    for p in paths:
        r = path_to_resolved.get(p)
        w = (file_state.check_stale(task_id, r) if r else None) or _check_file_staleness(p, task_id)
        if not w and r:
            w = _path_resolution_warning(p, Path(r), task_id)
        if w:
            warnings.append(w)
    return warnings


def _note_edited(task_id: str, paths: list[str], path_to_resolved: dict, session_id: str | None) -> None:
    """Post-success bookkeeping: verification-stale marker, then per path refresh
    the read stamp (no false staleness on the next edit) and record the write."""
    _mark_verification_stale(task_id, [path_to_resolved.get(p) or p for p in paths], session_id=session_id)
    for p in paths:
        _update_read_timestamp(p, task_id)
        if path_to_resolved.get(p):
            file_state.note_write(task_id, path_to_resolved[p])


def write_file_tool(path: str, content: str, task_id: str = "default",
                    cross_profile: bool = False,
                    session_id: str | None = None) -> str:
    """Write content to a file.

    ``cross_profile`` bypasses the sandbox-mirror lost-write guards only
    (unadvertised in the schema; the mirror rejection error teaches it — the
    cross-PROFILE guard it was named for no longer exists).
    """
    # write_file checks the binary-document guard before the mirror guard.
    err = (_check_sensitive_path(path, task_id)
           or _check_binary_document_write(path, task_id)
           or _check_protected_instruction_write([path], task_id)
           or _check_approval_required_write([path], task_id)
           or (None if cross_profile else _check_cross_profile_path(path, task_id)))
    if not err and _is_internal_file_tool_content(content):
        err = ("Refusing to write internal read_file display text as file content. "
               "Strip read_file line-number prefixes or reconstruct the intended "
               "file contents before writing.")
    if err:
        return tool_error(err)
    try:
        # Resolution failure falls back to the legacy unlocked path (the write
        # still proceeds; the per-task staleness check still runs).
        _resolved = _resolve_or_none(path, task_id)
        path_to_resolved = {path: _resolved}
        with ExitStack() as _lock:
            if _resolved:
                # Per-path lock serializes read→modify→write across concurrent
                # subagents; different paths stay fully parallel.
                _lock.enter_context(file_state.lock_path(_resolved))
            warnings = _edit_warnings([path], path_to_resolved, task_id)
            result_dict = _get_file_ops(task_id).write_file(_resolved or path, content).to_dict()
            if warnings:
                result_dict["_warning"] = warnings[0]
            if _resolved:
                # Always report the ABSOLUTE path written so a wrong-cwd mismatch
                # is visible in the response instead of silently landing elsewhere.
                result_dict["resolved_path"] = _resolved
            if result_dict.get("error"):
                _update_read_timestamp(path, task_id)
            else:
                if _resolved:
                    result_dict["files_modified"] = [_resolved]
                _note_edited(task_id, [path], path_to_resolved, session_id)
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))


def _collect_v4a_header_paths(patch: str) -> tuple[list[str], list[str]] | str:
    """Extract every path named in V4A headers, rejecting ``..`` traversal.

    Returns ``(all_paths, content_write_paths)`` or a tool_error string. Header
    paths come from patch CONTENT (more attacker-influenceable than ``path=``,
    which keeps its legitimate ``..`` use). Move headers check BOTH endpoints;
    only Update/Add write text and feed the binary-document guard.
    """
    from tools.path_security import has_traversal_component

    headers = [(m.group(3), m.group(2) in ("Update", "Add")) for m in _V4A_SINGLE_HEADER_RE.finditer(patch)]
    headers += [(g, False) for m in _V4A_MOVE_HEADER_RE.finditer(patch) for g in (m.group(2), m.group(3))]
    paths: list[str] = []
    content_paths: list[str] = []
    for raw, writes_text in headers:
        v4a_path = raw.strip()
        if has_traversal_component(v4a_path):
            return tool_error(
                f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                "Use the agent's cwd-relative path (no '..') or an absolute "
                "path in '*** Update File:' / '*** Add File:' / "
                "'*** Delete File:' / '*** Move File:' headers.")
        paths.append(v4a_path)
        if writes_text:
            content_paths.append(v4a_path)
    return paths, content_paths


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default", cross_profile: bool = False,
               session_id: str | None = None) -> str:
    """Patch a file using replace mode or V4A patch format.

    ``cross_profile``: same semantics as ``write_file``'s flag (mirror-guard
    bypass only; unadvertised).
    """
    _paths_to_check = [path] if path else []
    _content_write_paths = list(_paths_to_check)
    if mode == "patch" and patch:
        collected = _collect_v4a_header_paths(patch)
        if isinstance(collected, str):
            return collected
        _paths_to_check += collected[0]
        _content_write_paths += collected[1]
    precheck_err = _write_precheck_error(_paths_to_check, _content_write_paths, task_id, cross_profile)
    if precheck_err:
        return tool_error(precheck_err)
    try:
        # Lock paths in sorted, deduplicated order so concurrent callers with
        # overlapping multi-file patches can't deadlock (every caller locks in
        # the same order). An unresolvable path is simply not locked.
        _path_to_resolved: dict[str, str] = {_p: _resolve_or_none(_p, task_id) for _p in _paths_to_check}
        with ExitStack() as _locks:
            for _r in sorted({_r for _r in _path_to_resolved.values() if _r}):
                _locks.enter_context(file_state.lock_path(_r))
            stale_warnings = _edit_warnings(_paths_to_check, _path_to_resolved, task_id)
            file_ops = _get_file_ops(task_id)

            # Hand the shell layer the RESOLVED targets so both layers agree on
            # which file is edited even when the shell's cwd differs.
            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                _replace_target = _path_to_resolved.get(path) or path
                result = file_ops.patch_replace(_replace_target, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                result = file_ops.patch_v4a(_rewrite_v4a_patch_paths_for_host(patch, _path_to_resolved, file_ops))
            else:
                return tool_error(f"Unknown mode: {mode}")

            result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = " | ".join(stale_warnings)
            if not result_dict.get("error"):
                # Report the ABSOLUTE path(s) actually patched so a wrong-cwd
                # mismatch is visible instead of silently landing elsewhere.
                _resolved_modified = [_path_to_resolved.get(_p) or _p for _p in _paths_to_check]
                result_dict["files_modified"] = _resolved_modified
                if len(_resolved_modified) == 1:
                    result_dict["resolved_path"] = _resolved_modified[0]
                _note_edited(task_id, _paths_to_check, _path_to_resolved, session_id)
                # Clear failure counters so a future miss starts a fresh count.
                _reset_patch_failures(task_id, [_r for _r in _path_to_resolved.values() if _r])
        # old_string-not-found hint. Failure escalation is tracked for replace
        # mode only (V4A misses are rare); the generic hint is suppressed when
        # patch_replace already attached a richer "Did you mean?" snippet.
        if result_dict.get("error") and "Could not find" in str(result_dict["error"]):
            failure_count = 0
            if mode == "replace" and path:
                failure_count = _record_patch_failure(task_id, _path_to_resolved.get(path) or path)
            if failure_count >= 3:
                result_dict["_hint"] = (
                    f"This is failure #{failure_count} patching {path!r}. "
                    "Stop retrying with variations of the same old_string. "
                    "Either: (1) re-read the file fresh to verify current "
                    "content, (2) use a longer / more unique old_string with "
                    "surrounding context lines, or (3) use write_file to "
                    "replace the entire file if the targeted region is hard "
                    "to anchor.")
            elif "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text.")
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                order: str = "discovery",
                task_id: str = "default") -> str:
    """Search for content or files."""
    try:
        offset, limit = normalize_search_pagination(offset, limit)

        # Pagination args (and order) are part of the key so paging through truncated
        # results doesn't trip the repeated-search guard.
        search_key = ("search", pattern, target, str(path), file_glob or "", limit, offset, order)
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0, "read_history": set()})
            count = _bump_consecutive(task_data, search_key)

        if count >= 4:
            return tool_error(
                f"BLOCKED: You have run this exact search {count} times in a row. "
                "The results have NOT changed. You already have this information. "
                "STOP re-searching and proceed with your task.",
                pattern=pattern,
                already_searched=count)

        try:
            resolved_search_path = str(_resolve_path_for_task(path, task_id))
        except (OSError, ValueError, RuntimeError) as exc:
            resolved_search_path = path
            # A RuntimeError still surfaces as the tool error unless the raw
            # path is itself denylisted (that error wins).
            if isinstance(exc, RuntimeError) and not get_read_block_error(path):
                raise
        block_error = get_read_block_error(resolved_search_path)
        if block_error:
            return tool_error(block_error)

        # A missing search root costs two shells (search + parent listing for
        # "Similar paths"); cache the miss so a retry skips both.
        cached_search_nf = _check_not_found_cache("search", resolved_search_path, task_id)
        if cached_search_nf is not None:
            return cached_search_nf

        result = _get_file_ops(task_id).search(
            pattern=pattern, path=path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context, order=order)
        omitted = _filter_read_blocked_search_results(result, task_id)
        for m in getattr(result, "matches", None) or ():
            if getattr(m, "content", None):
                m.content = redact_sensitive_text(m.content, file_read=True)
        result_dict = result.to_dict(densify=True)

        if omitted:
            result_dict["_omitted"] = (
                f"{omitted} result(s) omitted because they target credential, "
                "token, cache, or secret-bearing environment files.")

        # No early return on a cached miss — same rationale as the read path.
        _search_err = result_dict.get("error") or ""
        if isinstance(_search_err, str) and _search_err.startswith("Path not found:"):
            _record_not_found("search", resolved_search_path, task_id, json.dumps(result_dict, ensure_ascii=False))

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "The results have not changed. Use the information you already have.")

        result_json = json.dumps(result_dict, ensure_ascii=False)
        if result_dict.get("truncated"):
            next_offset = offset + limit
            result_json += f"\n\n[Hint: Results truncated. Use offset={next_offset} to see more, or narrow with a more specific pattern or file_glob.]"
        return result_json
    except Exception as e:
        return tool_error(str(e))


# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    # Document formats are stated unconditionally: firecrawl-anydoc is a
    # core dependency (bundled), so its absence is a broken install, not a
    # configuration — the teaching error in read_extract handles that rare
    # case with the pip-install fix. The ONE dynamic word: "PDF (text
    # layer)" upgrades to "PDF (scanned or text)" when hosted OCR has a
    # route we trust (_read_file_schema_overrides). Scanned-page coverage
    # teaching lives in the response-time NEEDS-OCR warning
    # (read_extract.py); the schema doesn't pre-teach it.
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Documents auto-extract to readable text: .ipynb, Office (.docx/.xlsx/.pptx and legacy .doc/.ppt/.xls), PDF (text layer), OpenDocument, RTF, EPUB. Cannot read images/binary — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 2000, max: 2000). Reads are additionally capped at a ~100K-character budget with a next_offset continuation.", "default": 2000, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). The result's verified:true means the on-disk content hash was confirmed — do NOT re-read the file to check the write landed.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            # NOTE: the handler still accepts `cross_profile` (bool) — it now
            # bypasses only the #32049 sandbox-mirror lost-write guards, whose
            # rejection error teaches it. Unadvertised: the cross-PROFILE
            # guard it was named for was removed (profiles are not isolated,
            # maintainer decision), and mirror hits are rare + self-teaching.
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    # BASE = replace-only (what nearly every model family was trained on).
    # The V4A patch mode (mode + patch params, dual-mode description) is
    # LAYERED ON dynamically for OpenAI-family mains only — V4A is the
    # OpenAI apply_patch dialect their models emit natively; advertising
    # it to everyone cost every other session ~148 tok/call
    # (_patch_schema_overrides below). The handler accepts BOTH shapes
    # from any model regardless (replay compat + strong models that know
    # V4A anyway): mode defaults to 'replace' when omitted.
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing. "
        "Finds a unique string and replaces it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "Changed replacement text; it must differ from old_string. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            # NOTE: handler still accepts `cross_profile` — see write_file's
            # NOTE (mirror-guard bypass only; unadvertised by design).
            # NOTE: handler still accepts `mode` + `patch` (V4A) from ANY
            # model — the schema just doesn't advertise them off-family.
        },
        "required": ["path", "old_string", "new_string"],
    },
}


# V4A layer, rendered only for OpenAI-family main models (see PATCH_SCHEMA
# comment). Kept as data so the override composes it deterministically.
_PATCH_V4A_DESCRIPTION = (
    "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
    "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
    "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
    "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
    "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
    "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
    "REQUIRED PARAMETERS: mode, patch."
)

_PATCH_V4A_PARAMS = {
    "mode": {
        "type": "string",
        "enum": ["replace", "patch"],
        "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
        "default": "replace",
    },
    "patch": {
        "type": "string",
        "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
    },
}


def _is_openai_family_main() -> bool:
    """Whether the active main provider/model is the OpenAI/codex family —
    the population trained on the V4A apply_patch dialect.

    Provider-family-coarse on purpose (no per-model training-diet table to
    go stale): direct OpenAI providers always qualify; on aggregators
    (openrouter/nous/azure...) the MODEL slug decides (gpt-*/o-series/
    codex). Fail-closed to the universal replace-only schema.
    """
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider

        provider = (_read_main_provider() or "").strip().lower()
        model = (_read_main_model() or "").strip().lower()
    except Exception:  # noqa: BLE001
        return False
    if provider in {"openai", "openai-chat", "openai-codex", "azure-openai", "codex"}:
        return True
    # Aggregators: the model slug carries the family.
    slug = model.split("/", 1)[-1]
    if slug.startswith(("gpt-", "gpt.", "chatgpt", "codex", "o1", "o3", "o4", "o5")):
        return True
    return "openai/" in model


SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. On macOS, broad searches above the user home automatically skip TCC-protected folders (Desktop, Documents, Downloads, Library, Movies, Music, Pictures); target one directly when access is intentional.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls. Discovery order is the fast bounded default; exact global newest-first order is an explicit opt-in and may scan the full tree.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "order": {"type": "string", "enum": ["discovery", "modified"], "description": "File-search order: 'discovery' is fast bounded traversal order; 'modified' is exact global newest-first and may scan the full tree; ignored for content", "default": "discovery"},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(path=args.get("path", ""), offset=args.get("offset", 1), limit=args.get("limit", 500), task_id=tid)


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hermes_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    return write_file_tool(
        path=args["path"], content=args["content"], task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0),
        order=args.get("order", "discovery"), task_id=tid)


def _read_file_schema_overrides():
    """One-word capability upgrade: "PDF (text layer)" → "PDF (scanned or
    text)" when hosted OCR has a trusted route (see
    read_extract.hosted_ocr_available). Config/env probe only — no
    network at schema-build time. Compaction's tool refresh (#97073)
    picks up a key added mid-session.
    """
    try:
        from tools.read_extract import hosted_ocr_available

        if hosted_ocr_available():
            return {
                "description": READ_FILE_SCHEMA["description"].replace(
                    "PDF (text layer)", "PDF (scanned or text)"
                )
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000, dynamic_schema_overrides=_read_file_schema_overrides)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
def _patch_schema_overrides():
    """Layer the V4A patch mode onto the base replace-only schema for
    OpenAI-family mains (see PATCH_SCHEMA comment). Config/context probe
    only — no I/O at schema-build time; compaction's tool refresh
    (#97073) re-evaluates on model switches."""
    try:
        if not _is_openai_family_main():
            return {}
        params = {
            "type": "object",
            "properties": {
                "mode": _PATCH_V4A_PARAMS["mode"],
                **PATCH_SCHEMA["parameters"]["properties"],
                "patch": _PATCH_V4A_PARAMS["patch"],
            },
            "required": ["mode"],
        }
        return {"description": _PATCH_V4A_DESCRIPTION, "parameters": params}
    except Exception:  # noqa: BLE001
        return {}


registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000, dynamic_schema_overrides=_patch_schema_overrides)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import PurePosixPath  # noqa: F401,E402
import posixpath  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'has_opaque_document_extension': ('tools.binary_extensions', 'has_opaque_document_extension'),
    'is_pdf_path': ('tools.binary_extensions', 'is_pdf_path'),
    'notify_other_tool_call': ('tools.file_tools_read_tracking', 'notify_other_tool_call'),
    'reset_file_dedup': ('tools.file_tools_read_tracking', 'reset_file_dedup'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
