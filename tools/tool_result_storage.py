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
from dataclasses import dataclass

from agent.redact import redact_sensitive_text
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
_SAFE_ARTIFACT_EXTENSION = re.compile(r"(?:\.[A-Za-z0-9_-]+)+")
_MAX_RESULT_FILENAME_STEM = 120


@dataclass(frozen=True)
class PersistedToolArtifact:
    kind: str
    path: str
    chars: int
    sha256: str
    redacted: bool
    created: bool


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


def _publish_immutable_to_sandbox(
    content: str,
    remote_path: str,
    env,
) -> bool | None:
    """Publish verified bytes without replacing an existing artifact.

    Returns ``True`` when this call created the path, ``False`` when an
    identical content-addressed artifact already existed, and ``None`` on any
    transport, integrity, or collision failure.
    """
    storage_dir = os.path.dirname(remote_path)
    artifact_root = os.path.dirname(storage_dir)
    temp_path = f"{remote_path}.tmp-{uuid.uuid4().hex}"
    quoted_artifact_root = shlex.quote(artifact_root)
    quoted_storage_dir = shlex.quote(storage_dir)
    quoted_temp_path = shlex.quote(temp_path)
    quoted_remote_path = shlex.quote(remote_path)
    payload = content.encode("utf-8")
    payload_size = len(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    created_marker = "HERMES_ARTIFACT_CREATED"
    reused_marker = "HERMES_ARTIFACT_REUSED"
    cmd = (
        "{ __hermes_sha256() { "
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum; "
        "elif command -v shasum >/dev/null 2>&1; then shasum -a 256; "
        "else return 127; fi; }; "
        "__hermes_matches() { "
        f"test -f {quoted_remote_path} && test ! -L {quoted_remote_path} && "
        f"__hermes_sha=$(__hermes_sha256 < {quoted_remote_path} 2>/dev/null) && "
        f"test \"${{__hermes_sha%%[[:space:]]*}}\" = {expected_sha256}; }}; "
        f"umask 077 && mkdir -p {quoted_artifact_root} && "
        f"test -d {quoted_artifact_root} && test ! -L {quoted_artifact_root} && "
        f"chmod 700 {quoted_artifact_root} && mkdir -p {quoted_storage_dir} && "
        f"test -d {quoted_storage_dir} && test ! -L {quoted_storage_dir} && "
        f"chmod 700 {quoted_storage_dir} && "
        f"head -c {payload_size} > {quoted_temp_path} && "
        f"test \"$(wc -c < {quoted_temp_path})\" -eq {payload_size} && "
        f"__hermes_temp_sha=$(__hermes_sha256 < {quoted_temp_path} 2>/dev/null) && "
        f"test \"${{__hermes_temp_sha%%[[:space:]]*}}\" = {expected_sha256} && "
        f"chmod 600 {quoted_temp_path} && "
        "__hermes_created=0 && "
        f"if test -e {quoted_remote_path} || test -L {quoted_remote_path}; then "
        f"__hermes_matches && rm -f -- {quoted_temp_path} && "
        f"printf '%s\\n' {reused_marker}; "
        f"elif ln {quoted_temp_path} {quoted_remote_path} 2>/dev/null; then "
        f"__hermes_created=1 && chmod 600 {quoted_remote_path} && __hermes_matches && "
        f"rm -f -- {quoted_temp_path} && printf '%s\\n' {created_marker}; "
        f"else __hermes_matches && rm -f -- {quoted_temp_path} && "
        f"printf '%s\\n' {reused_marker}; fi; "
        f"__hermes_ec=$?; if test \"$__hermes_ec\" -ne 0; then "
        f"rm -f -- {quoted_temp_path} 2>/dev/null || true; "
        f"if test \"$__hermes_created\" -eq 1; then "
        f"rm -f -- {quoted_remote_path} 2>/dev/null || true; fi; fi; "
        f"(exit \"$__hermes_ec\"); }}"
    )
    cleanup_cmd = f"rm -f -- {quoted_temp_path} 2>/dev/null || true"

    try:
        result = env.execute(cmd, timeout=30, stdin_data=content)
    except Exception:
        try:
            env.execute(cleanup_cmd, timeout=30)
        except Exception:
            pass
        return None
    if result.get("returncode", 1) != 0:
        try:
            env.execute(cleanup_cmd, timeout=30)
        except Exception:
            pass
        return None
    output = str(result.get("output", ""))
    if reused_marker in output:
        return False
    if created_marker in output:
        return True
    return None


def persist_tool_artifact(
    content: str,
    *,
    kind: str,
    tool_name: str,
    tool_use_id: str,
    task_scope: str,
    env,
    extension: str,
) -> PersistedToolArtifact | None:
    """Persist a redacted, immutable artifact in the active sandbox."""
    if not _SAFE_ARTIFACT_EXTENSION.fullmatch(extension):
        raise ValueError(f"Unsafe artifact extension: {extension!r}")
    if env is None or not isinstance(task_scope, str) or not task_scope.strip():
        return None

    remote_path = "<unresolved>"
    try:
        persisted_content = redact_sensitive_text(content, force=True)
        if not isinstance(persisted_content, str):
            return None
        payload = persisted_content.encode("utf-8")
        content_sha256 = hashlib.sha256(payload).hexdigest()
        scope_hash = hashlib.sha256(task_scope.encode("utf-8")).hexdigest()
        safe_stem = _safe_result_filename(tool_use_id).removesuffix(".txt")
        # Attempt-unique publication prevents a failed concurrent prune from
        # deleting bytes already referenced by another successful prune. The
        # scope and content digests still make provenance and integrity
        # explicit, while the nonce gives each rollback exclusive ownership.
        attempt_token = uuid.uuid4().hex
        remote_path = (
            f"{_resolve_storage_dir(env)}/{scope_hash}/"
            f"{content_sha256}_{safe_stem}_{attempt_token}{extension}"
        )
        created = _publish_immutable_to_sandbox(persisted_content, remote_path, env)
        if created is None:
            logger.warning(
                "Sandbox artifact write failed: kind=%s tool=%s path=%s",
                kind,
                tool_name,
                remote_path,
            )
            return None
    except Exception:
        logger.warning(
            "Sandbox artifact write failed: kind=%s tool=%s path=%s",
            kind,
            tool_name,
            remote_path,
        )
        return None

    return PersistedToolArtifact(
        kind=kind,
        path=remote_path,
        chars=len(persisted_content),
        sha256=content_sha256,
        redacted=persisted_content != content,
        created=created,
    )


def remove_created_tool_artifact(artifact: PersistedToolArtifact, env) -> bool:
    """Remove only a verified artifact created by the current prune attempt."""
    if not artifact.created:
        return True
    quoted_path = shlex.quote(artifact.path)
    expected_sha256 = artifact.sha256
    cmd = (
        "{ __hermes_sha256() { "
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum; "
        "elif command -v shasum >/dev/null 2>&1; then shasum -a 256; "
        "else return 127; fi; }; "
        f"test -f {quoted_path} && test ! -L {quoted_path} && "
        f"__hermes_sha=$(__hermes_sha256 < {quoted_path} 2>/dev/null) && "
        f"test \"${{__hermes_sha%%[[:space:]]*}}\" = {expected_sha256} && "
        f"rm -f -- {quoted_path} && test ! -e {quoted_path} && test ! -L {quoted_path}; }}"
    )
    try:
        result = env.execute(cmd, timeout=30)
    except Exception:
        return False
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
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
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
