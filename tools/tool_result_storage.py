"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is persisted and the
   in-context content is replaced with a preview + file path reference.

   The canonical home is ALWAYS host-side:
   ``$HERMES_HOME/cache/spillover/{tool_use_id}.txt`` — alongside the other
   Hermes-owned caches (images, audio, documents, ...) instead of littering
   the OS temp dir. This needs no sandbox environment, so it also works for
   sessions that never ran a terminal command (MCP-only, cron, gateway) —
   previously those hit the inline-truncate fallback because
   ``get_active_env()`` returned None until the first terminal call created
   an environment.

   What the model sees depends on the backend:

   - **Local backend (or no active env):** the host path itself.
   - **Remote backends (docker/ssh/modal/daytona):** ``cache/spillover`` is
     in the auto-mounted/synced cache-dir list (tools/credential_files.py),
     so the reference is the translated in-sandbox path (probed for
     readability first). When the sandbox can't see it (e.g. a persistent
     container created before spillover joined the mount list), fall back
     to writing a copy into the sandbox temp dir via env.execute().

   The spillover dir is pruned two ways: the gateway housekeeping loop
   sweeps it hourly with the other media caches, and a once-per-process
   best-effort prune runs on the first spill so CLI-only installs (which
   never run gateway housekeeping) self-clean too.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import hashlib
import logging
import os
import re
import shlex
import threading
import time
import uuid

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
SPILLOVER_SUBDIR = "cache/spillover"
SPILLOVER_MAX_AGE_HOURS = 24
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120

_spillover_prune_lock = threading.Lock()
_spillover_pruned_once = False


def get_spillover_dir():
    """Return $HERMES_HOME/cache/spillover as a Path (not created)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / SPILLOVER_SUBDIR


def cleanup_spillover_cache(max_age_hours: int = SPILLOVER_MAX_AGE_HOURS) -> int:
    """Delete spillover files older than *max_age_hours*.

    Same contract as the ``cleanup_*_cache`` helpers in
    ``gateway.platforms.base`` — returns the number of files removed —
    so the gateway housekeeping loop can prune this dir on the same
    hourly cadence as the media caches.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    try:
        entries = list(get_spillover_dir().iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _prune_spillover_once() -> None:
    """Best-effort prune, at most once per process.

    The gateway housekeeping loop prunes hourly, but CLI-only installs
    never run it — without this, spillover files would accumulate
    forever on pure-CLI setups.
    """
    global _spillover_pruned_once
    with _spillover_prune_lock:
        if _spillover_pruned_once:
            return
        _spillover_pruned_once = True
    try:
        removed = cleanup_spillover_cache()
        if removed:
            logger.debug("Pruned %d expired spillover file(s)", removed)
    except Exception as exc:
        logger.debug("Spillover prune failed: %s", exc)


def _is_host_side_env(env) -> bool:
    """True when the spill file should be written by this process directly.

    Covers ``env=None`` (no sandbox environment active — e.g. a session
    that has not run a terminal command yet) and the local backend
    (where env.execute() runs on this same host anyway). Remote backends
    (docker/ssh/modal/daytona) return False: their read_file resolves
    inside the sandbox, so the spill must be written there.
    """
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment

        return isinstance(env, LocalEnvironment)
    except Exception:
        return False


def _write_to_spillover(content: str, filename: str):
    """Write content host-side to $HERMES_HOME/cache/spillover.

    Returns the absolute path string on success, None on failure.
    """
    try:
        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / filename
        path.write_text(content, encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Spillover write failed for %s: %s", filename, exc)
        return None
    _prune_spillover_once()
    return str(path)


def _sandbox_visible_spillover_path(host_path: str, env) -> str | None:
    """Return the path where a remote backend can read *host_path*, or None.

    ``cache/spillover`` is one of the auto-mounted/synced cache dirs
    (tools/credential_files.py), so on docker it is bind-mounted and on
    modal/ssh/daytona it is file-synced into the sandbox. Translate the
    host path with the same helper the image tools use, force a sync for
    synced backends, then PROBE readability — a persistent docker
    container created before spillover joined the mount list won't have
    the bind mount, and must fall back to the in-sandbox write.
    """
    try:
        from tools.credential_files import to_agent_visible_cache_path

        visible = to_agent_visible_cache_path(host_path)
    except Exception as exc:
        logger.debug("Spillover path translation failed: %s", exc)
        return None

    sync_manager = getattr(env, "_sync_manager", None)
    if sync_manager is not None:
        try:
            sync_manager.sync(force=True)
        except Exception as exc:
            logger.debug("Spillover sync failed: %s", exc)

    try:
        result = env.execute(f"test -r {shlex.quote(visible)}", timeout=15)
        if result.get("returncode", 1) == 0:
            return visible
    except Exception as exc:
        logger.debug("Spillover readability probe failed: %s", exc)
    return None


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n"
    msg += (
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
    )
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


_PERSISTED_PATH_RE = re.compile(r"^Full output saved to: (.+)$", re.MULTILINE)


def extract_persisted_path(content: str) -> str | None:
    """Return the file path from a <persisted-output> replacement block.

    Used by the result-reference stubbing guard (agent/tool_guardrails.py) so
    a stub referencing a persisted first occurrence can carry the spillover
    path instead of dangling. Returns None for non-persisted content.
    """
    if not isinstance(content, str) or PERSISTED_OUTPUT_TAG not in content:
        return None
    match = _PERSISTED_PATH_RE.search(content)
    return match.group(1).strip() if match else None


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    filename = _safe_result_filename(tool_use_id)
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    # Always persist host-side first: $HERMES_HOME/cache/spillover is the
    # single canonical home for spilled results (with the other Hermes-owned
    # caches, pruned by gateway housekeeping) regardless of backend.
    host_path = _write_to_spillover(content, filename)

    if _is_host_side_env(env):
        if host_path is not None:
            logger.info(
                "Persisted large tool result: %s (%s, %d chars -> %s)",
                tool_name, tool_use_id, len(content), host_path,
            )
            return _build_persisted_message(preview, has_more, len(content), host_path)
    elif env is not None:
        # Remote backend: the spillover dir is auto-mounted (docker) or
        # file-synced (modal/ssh/daytona) into the sandbox, so reference the
        # translated path when the sandbox can actually read it.
        if host_path is not None:
            visible = _sandbox_visible_spillover_path(host_path, env)
            if visible is not None:
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s [host: %s])",
                    tool_name, tool_use_id, len(content), visible, host_path,
                )
                return _build_persisted_message(preview, has_more, len(content), visible)
        # Fallback: write into the sandbox temp dir (pre-existing containers
        # without the spillover mount, translation/probe failures).
        storage_dir = _resolve_storage_dir(env)
        remote_path = f"{storage_dir}/{filename}"
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages
