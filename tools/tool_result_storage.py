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
import json
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
_UNTRUSTED_CLOSE = "</untrusted_tool_result>"
_ANOMALY_RE = re.compile(
    r"(?:error|warning|exception|traceback|failed|failure|timeout|cancelled)",
    re.IGNORECASE,
)


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
    cmd = (
        f"umask 077 && mkdir -p {shlex.quote(storage_dir)} "
        f"&& cat > {shlex.quote(remote_path)}"
    )
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
    *,
    tool_name: str = "",
    tail: str = "",
    omitted_chars: int = 0,
    content_sha256: str = "",
    status_lines: tuple[str, ...] = (),
    anomaly_lines: tuple[str, ...] = (),
) -> str:
    """Build a bounded, retrieval-backed context-tray representation."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    if tool_name:
        msg += f"Tool: {tool_name}\n"
    msg += f"Original: {original_size:,} characters ({size_str})\n"
    if content_sha256:
        msg += f"sha256: {content_sha256}\n"
    for line in status_lines:
        msg += f"{line}\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += (
        "Use read_file with this exact path, offset, and limit to retrieve "
        "bounded pages (for example offset=1, limit=200).\n\n"
    )
    if anomaly_lines:
        msg += "STATUS / ANOMALY LINES (verbatim):\n"
        msg += "\n".join(anomaly_lines) + "\n\n"
    msg += f"HEAD ({len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += f"\n\n[omitted characters: {omitted_chars:,}]"
    if tail:
        msg += f"\n\nTAIL ({len(tail)} chars):\n{tail}"
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def _split_untrusted_wrapper(content: str) -> tuple[str, str, str] | None:
    leading_len = len(content) - len(content.lstrip())
    leading = content[:leading_len]
    rest = content[leading_len:]
    if not rest.startswith("<untrusted_tool_result"):
        return None
    close_idx = rest.rfind(_UNTRUSTED_CLOSE)
    header_end = rest.find("\n\n")
    if close_idx < 0 or header_end < 0 or header_end + 2 > close_idx:
        return None
    suffix_start = close_idx
    if suffix_start and rest[suffix_start - 1] == "\n":
        suffix_start -= 1
    return (
        leading + rest[: header_end + 2],
        rest[header_end + 2 : suffix_start],
        rest[suffix_start:],
    )


def _status_lines(content: str) -> tuple[str, ...]:
    wrapped = _split_untrusted_wrapper(content)
    body = wrapped[1] if wrapped else content
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return ()
    if not isinstance(parsed, dict):
        return ()
    lines = []
    for key in ("success", "status", "exit_code", "returncode", "error", "code"):
        if key in parsed:
            lines.append(f"{key}: {json.dumps(parsed[key], ensure_ascii=False)}")
    return tuple(lines)


def _anomaly_lines(content: str, limit: int = 12) -> tuple[str, ...]:
    wrapped = _split_untrusted_wrapper(content)
    body = wrapped[1] if wrapped else content
    scan_text = body
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        scan_text = "\n".join(
            value
            for key, value in parsed.items()
            if key in {"output", "content", "message", "error"}
            and isinstance(value, str)
        )
    found = []
    seen = set()
    try:
        from agent.redact import redact_sensitive_text
    except Exception:
        redact_sensitive_text = None
    for line in scan_text.splitlines():
        if not _ANOMALY_RE.search(line):
            continue
        clipped = line[:500]
        if redact_sensitive_text is not None:
            clipped = redact_sensitive_text(clipped, force=True)
        if clipped in seen:
            continue
        seen.add(clipped)
        found.append(clipped)
        if len(found) >= limit:
            break
    return tuple(found)


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
    wrapped = _split_untrusted_wrapper(content)
    preview_source = wrapped[1] if wrapped else content
    preview, has_more = generate_preview(
        preview_source, max_chars=config.preview_size
    )
    tail = preview_source[-config.preview_size:] if has_more else ""
    omitted_chars = max(0, len(content) - len(preview) - len(tail))
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                replacement = _build_persisted_message(
                    preview,
                    has_more,
                    len(content),
                    remote_path,
                    tool_name=tool_name,
                    tail=tail,
                    omitted_chars=omitted_chars,
                    content_sha256=content_sha256,
                    status_lines=_status_lines(content),
                    anomaly_lines=_anomaly_lines(content),
                )
                if wrapped:
                    return wrapped[0] + replacement + wrapped[2]
                return replacement
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.warning(
        "Context tray write unavailable for %s; keeping %d-char result inline",
        tool_name, len(content),
    )
    return content


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
