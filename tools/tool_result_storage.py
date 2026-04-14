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

import logging
import os
import shlex
import uuid

from tools.budget_config import (
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
INLINE_TRUNCATED_OUTPUT_TAG = "<inline-truncated-output>"
INLINE_TRUNCATED_OUTPUT_CLOSING_TAG = "</inline-truncated-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
DEFAULT_RESULT_THRESHOLD_CHARS = DEFAULT_BUDGET.default_result_size
DEFAULT_TURN_BUDGET_CHARS = DEFAULT_BUDGET.turn_budget
DEFAULT_PREVIEW_CHARS = DEFAULT_BUDGET.preview_size
TURN_BUDGET_PERSIST_THRESHOLD_CHARS = 0
SINGLE_RESULT_REASON = "per-result threshold exceeded"
TURN_BUDGET_REASON = "turn-level cumulative budget exceeded"
_LEGACY_INLINE_TRUNCATION_TEXT = "Full output could not be saved to sandbox.]"


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


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> tuple[str, bool]:
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
    """Write content into the sandbox via env.execute(). Returns True on success."""
    marker = _heredoc_marker(content)
    storage_dir = os.path.dirname(remote_path)
    cmd = (
        f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)} << '{marker}'\n"
        f"{content}\n"
        f"{marker}"
    )
    result = env.execute(cmd, timeout=30)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
    reason: str = SINGLE_RESULT_REASON,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"Reason: {reason}.\n"
    if reason == TURN_BUDGET_REASON:
        msg += (
            "This tool result was moved out of the conversation because the turn's "
            f"cumulative tool budget was exceeded ({original_size:,} characters, {size_str}).\n"
        )
    else:
        msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def _build_inline_truncation_message(
    preview: str,
    has_more: bool,
    original_size: int,
    reason: str,
) -> str:
    """Build the inline fallback block used when sandbox persistence is unavailable."""
    msg = f"{INLINE_TRUNCATED_OUTPUT_TAG}\n"
    msg += f"Reason: {reason}.\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += (
        f"\n\n[Truncated: tool response was {original_size:,} chars. "
        "Full output could not be saved to sandbox.]"
    )
    msg += f"\n{INLINE_TRUNCATED_OUTPUT_CLOSING_TAG}"
    return msg


def _is_inline_truncated_content(content: str) -> bool:
    return (
        INLINE_TRUNCATED_OUTPUT_TAG in content
        or _LEGACY_INLINE_TRUNCATION_TEXT in content
    )


def _is_already_processed(content: str) -> bool:
    return (
        PERSISTED_OUTPUT_TAG in content
        or _is_inline_truncated_content(content)
    )


def _process_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env,
    config: BudgetConfig,
    threshold: int | float,
    reason: str,
    remote_path: str | None = None,
) -> str:
    """Persist or inline-truncate a tool result for a specific budget reason."""
    if _is_already_processed(content):
        return content

    if threshold == float("inf"):
        return content

    if len(content) <= threshold:
        return content

    remote_path = remote_path or f"{_resolve_storage_dir(env)}/{tool_use_id}.txt"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted tool result: %s (%s, %d chars -> %s, reason=%s)",
                    tool_name, tool_use_id, len(content), remote_path, reason,
                )
                return _build_persisted_message(
                    preview=preview,
                    has_more=has_more,
                    original_size=len(content),
                    file_path=remote_path,
                    reason=reason,
                )
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating tool result: %s (%d chars, reason=%s, no sandbox write)",
        tool_name, len(content), reason,
    )
    return _build_inline_truncation_message(
        preview=preview,
        has_more=has_more,
        original_size=len(content),
        reason=reason,
    )


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
        Original content if small, or a persisted/inline-truncated replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)
    storage_dir = _resolve_storage_dir(env)
    return _process_tool_result(
        content=content,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        env=env,
        config=config,
        threshold=effective_threshold,
        reason=SINGLE_RESULT_REASON,
        remote_path=f"{storage_dir}/{tool_use_id}.txt",
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, handle the largest raw results first until
    under budget. Results already transformed by per-result persistence or
    inline truncation are skipped to avoid double-processing.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if not _is_already_processed(content):
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

        replacement = _process_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=TURN_BUDGET_PERSIST_THRESHOLD_CHARS,
            reason=TURN_BUDGET_REASON,
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
