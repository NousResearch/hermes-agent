"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

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
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120
DEFAULT_TOOL_RESULTS_DIR = "~/.hermes/tool-results"
_DEFAULT_HEAD_TAIL_LINES = 5
_DEFAULT_CLEANUP_AGE_HOURS = 24
_DEFAULT_CLEANUP_MAX_MB = 100


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


def generate_head_tail_preview(
    content: str,
    head_lines: int = _DEFAULT_HEAD_TAIL_LINES,
    tail_lines: int = _DEFAULT_HEAD_TAIL_LINES,
) -> tuple[str, str, str, bool]:
    """Generate a head + tail preview from content.

    Args:
        content: The full tool result string.
        head_lines: Number of leading lines to include.
        tail_lines: Number of trailing lines to include.

    Returns:
        (head, tail, separator_message, has_overlap):
        - head: first ``head_lines`` lines (or fewer if content is shorter).
        - tail: last ``tail_lines`` lines (or fewer).
        - separator_message: Human-readable indicator like
          ``"[... 1234 lines omitted ...]"``.
        - has_overlap: True when the head and tail regions overlap,
          meaning the full content fits within head+tail lines.
    """
    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines:
        # Content fits entirely in the preview — show it all.
        return content, "", "", False

    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])
    omitted = total_lines - head_lines - tail_lines
    sep = f"[... {omitted} lines omitted. Full result saved to disk ...]"
    return head, tail, sep, False


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
    head: str,
    tail: str,
    separator: str,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block with head/tail preview."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool to access the full output.\n\n"
    msg += "HEAD:\n"
    msg += head.rstrip("\n")
    if separator:
        msg += f"\n{separator}\n"
    if tail:
        msg += f"\nTAIL:\n{tail}"
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


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

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{_safe_result_filename(tool_use_id)}"
    head, tail, sep, _ = generate_head_tail_preview(content)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(head, tail, sep, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{head}\n"
        f"{sep}\n"
        f"\nTAIL:\n{tail}\n\n"
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


def cleanup_old_results(
    storage_dir: str | None = None,
    max_age_hours: int = _DEFAULT_CLEANUP_AGE_HOURS,
    max_total_mb: int = _DEFAULT_CLEANUP_MAX_MB,
) -> int:
    """Remove tool result files older than ``max_age_hours`` and, if the total
    disk usage still exceeds ``max_total_mb`` MB, remove the oldest results
    first until under budget.

    Args:
        storage_dir: Directory to clean. Defaults to ``DEFAULT_TOOL_RESULTS_DIR``.
        max_age_hours: Remove files older than this many hours.
        max_total_mb: Maximum total size in MB before oldest files are evicted.

    Returns:
        Number of files removed.
    """
    import glob
    import time

    if storage_dir is None:
        storage_dir = os.path.expanduser(DEFAULT_TOOL_RESULTS_DIR)

    if not os.path.isdir(storage_dir):
        return 0

    removed = 0
    now = time.time()
    cutoff = now - max_age_hours * 3600

    files: list[tuple[str, float, int]] = []
    pattern = os.path.join(storage_dir, "*.txt")
    for fpath in glob.glob(pattern):
        try:
            st = os.stat(fpath)
            age_seconds = now - st.st_mtime
            if age_seconds > cutoff:
                os.remove(fpath)
                removed += 1
                logger.debug("Cleaned old tool result: %s (age=%.1fh)", fpath, age_seconds / 3600)
                continue
            files.append((fpath, st.st_mtime, st.st_size))
        except OSError:
            continue

    if not files:
        return removed

    total_bytes = sum(sz for _, _, sz in files)
    max_bytes = max_total_mb * 1024 * 1024

    if total_bytes <= max_bytes:
        return removed

    # Sort oldest-first so we evict the stalest results.
    files.sort(key=lambda x: x[1])

    for fpath, _, sz in files:
        if total_bytes <= max_bytes:
            break
        try:
            os.remove(fpath)
            removed += 1
            total_bytes -= sz
            logger.debug("Evicted oversized tool result: %s (%.1f MB total)", fpath, total_bytes / (1024 * 1024))
        except OSError:
            continue

    if removed:
        logger.info("cleanup_old_results: removed %d files from %s", removed, storage_dir)
    return removed
