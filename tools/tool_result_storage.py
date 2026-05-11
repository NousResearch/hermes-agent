"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written to disk and
   the in-context content is replaced with a preview + file path reference.
   When a sandbox env is available, the file is written INTO THE SANDBOX
   temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on standard
   Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux) via
   env.execute(). When no sandbox env is available (bare CLI, local custom
   providers), the file is written to the local filesystem at
   $HERMES_HOME/tool_results/{safe_id}.txt with owner-only permissions and
   an atomic rename. Inline truncation is the last-resort fallback when
   both write paths fail. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import logging
import os
import re
import shlex
import tempfile
import uuid

from hermes_constants import get_hermes_home
from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
LOCAL_FS_SPILL_SUBDIR = "tool_results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"


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


def _write_to_local_filesystem(content: str, tool_use_id: str) -> str | None:
    """Spill content to ``<HERMES_HOME>/tool_results/<safe_id>.txt``.

    Used when no sandbox env is available so ``read_file`` can still access
    the full output. Writes are atomic (tempfile + rename) and owner-only
    (mode 0o600), and refuse to follow a symlink at the spill root.
    Returns the absolute path on success, ``None`` on any filesystem
    failure (the caller falls back to inline truncation).
    """
    # TODO: consolidate with tools/hook_output_spill.py once #20468 merges --
    # both share head/tail-preview-plus-disk-spill semantics.
    storage_dir = get_hermes_home() / LOCAL_FS_SPILL_SUBDIR
    tmp_path: "Path | None" = None  # noqa: F821 -- forward ref for cleanup
    try:
        # Refuse to write into a symlinked spill root so a pre-planted symlink
        # cannot redirect attacker-influenced tool output onto arbitrary files.
        if storage_dir.is_symlink():
            logger.warning(
                "Local-fs persistence refused: spill root is a symlink (%s)", storage_dir,
            )
            return None
        storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_use_id).strip("._") or uuid.uuid4().hex
        target = storage_dir / f"{safe_name}.txt"
        # surrogateescape preserves arbitrary bytes from tool output round-trip
        # (errors='replace' would silently map malformed bytes to U+FFFD).
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="surrogateescape",
            dir=str(storage_dir),
            prefix=f".{safe_name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = type(target)(tmp.name)
            tmp.write(content)
        # NamedTemporaryFile creates with 0o600 on POSIX; os.replace preserves
        # the source's permissions, so the visible target also lands at 0o600.
        os.replace(tmp_path, target)
        tmp_path = None  # successfully renamed; nothing to clean up
        return str(target)
    except Exception as exc:
        logger.warning(
            "Local-fs persistence failed for %s -> %s: %s", tool_use_id, storage_dir, exc,
        )
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None


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
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
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
    """Layer 2: persist oversized result and return preview + path.

    Three-step fallback chain:
      1. ``env`` is given -> write into the sandbox via ``env.execute()`` so
         the file is reachable from any backend (local, Docker, SSH, Modal,
         Daytona) using the agent's own ``read_file``.
      2. ``env`` is ``None`` -> write to the local filesystem at
         ``<HERMES_HOME>/tool_results/<safe_id>.txt`` so ``read_file`` can
         still reach the full output on bare CLI / local-provider setups
         where the inline-truncation path drives a compression-grows-size
         loop on lower-context models (see #23767).
      3. Both writes failed -> fall back to inline truncation. The agent
         loses access to the full content; the warning log on step 1 or 2
         and the elevated WARNING-level fallthrough log are the operator's
         signal that the session is now in degraded mode.

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

    preview, has_more = generate_preview(content, max_chars=config.preview_size)
    local_fs_attempted = False

    if env is not None:
        storage_dir = _resolve_storage_dir(env)
        remote_path = f"{storage_dir}/{tool_use_id}.txt"
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                return _build_persisted_message(preview, has_more, len(content), remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
    else:
        local_fs_attempted = True
        local_path = _write_to_local_filesystem(content, tool_use_id)
        if local_path is not None:
            logger.info(
                "Persisted large tool result to local fs: %s (%s, %d chars -> %s)",
                tool_name, tool_use_id, len(content), local_path,
            )
            return _build_persisted_message(preview, has_more, len(content), local_path)

    # Degraded-mode fallthrough. Use WARNING when the local-fs path was tried
    # and failed -- monitors should see this; INFO when env != None and the
    # caller knows their sandbox is the only persistence option.
    fallthrough_log = logger.warning if local_fs_attempted else logger.info
    fallthrough_log(
        "Inline-truncating large tool result: %s (%d chars, no sandbox or local-fs write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved.]"
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
