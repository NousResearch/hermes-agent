"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written into a readable
   artifact file and the in-context content is replaced with a preview + file
   path reference. Remote/sandbox environments are preferred when available;
   local Hermes runs fall back to HERMES_HOME/tool-artifacts.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   DEFAULT_TURN_BUDGET_CHARS, the largest non-persisted results are spilled to
   disk until the aggregate is under budget. This catches cases where many
   medium-sized results combine to overflow context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
EXTERNALIZED_OUTPUT_MARKER = "[Large tool output externalized by Hermes]"
STORAGE_DIR = "/tmp/hermes-results"
LOCAL_STORAGE_ENV = "HERMES_TOOL_ARTIFACT_DIR"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_MAX_SLUG_LEN = 64


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


def _resolve_local_storage_dir() -> Path:
    """Return the host-local artifact directory, honoring HERMES_HOME profiles."""
    override = os.environ.get(LOCAL_STORAGE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "tool-artifacts"


def _safe_slug(value: str, *, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    if not slug:
        slug = default
    return slug[:_MAX_SLUG_LEN]


def _local_artifact_paths(content: str, tool_name: str, tool_use_id: str) -> tuple[Path, Path, str]:
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    day = time.strftime("%Y-%m-%d", time.gmtime())
    tool_slug = _safe_slug(tool_name, default="tool")
    call_slug = _safe_slug(tool_use_id, default="call")
    base = _resolve_local_storage_dir() / day / f"{tool_slug}_{call_slug}_{digest[:12]}"
    return base.with_suffix(".txt"), base.with_suffix(".json"), digest


def _write_atomic_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_to_local_artifact(content: str, tool_name: str, tool_use_id: str) -> tuple[str, str] | None:
    """Write exact raw output to a local artifact plus metadata sidecar."""
    try:
        artifact_path, meta_path, digest = _local_artifact_paths(content, tool_name, tool_use_id)
        _write_atomic_text(artifact_path, content)
        metadata = {
            "tool": tool_name,
            "tool_call_id": tool_use_id,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "original_chars": len(content),
            "sha256": digest,
            "artifact_path": str(artifact_path),
        }
        _write_atomic_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        logger.info(
            "Persisted large tool result locally: %s (%s, %d chars -> %s)",
            tool_name,
            tool_use_id,
            len(content),
            artifact_path,
        )
        return str(artifact_path), digest
    except Exception as exc:
        logger.warning("Local artifact write failed for %s: %s", tool_use_id, exc)
        return None


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
    *,
    tool_name: str | None = None,
    sha256: str | None = None,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"{EXTERNALIZED_OUTPUT_MARKER}\n"
    if tool_name:
        msg += f"Tool: {tool_name}\n"
    msg += f"Original: {original_size:,} characters ({size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    if sha256:
        msg += f"SHA256: {sha256}\n"
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
    """Layer 2: persist oversized result, return preview + path.

    Remote/sandbox writes are preferred when an env is available. When no env
    is available (normal local CLI/gateway path), full raw output is written to
    HERMES_HOME/tool-artifacts and can be recovered with read_file.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    safe_call_id = _safe_slug(tool_use_id, default="tool_result")
    remote_path = f"{storage_dir}/{safe_call_id}.txt"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        try:
            if _write_to_sandbox(content, remote_path, env):
                logger.info(
                    "Persisted large tool result: %s (%s, %d chars -> %s)",
                    tool_name, tool_use_id, len(content), remote_path,
                )
                digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                return _build_persisted_message(
                    preview,
                    has_more,
                    len(content),
                    remote_path,
                    tool_name=tool_name,
                    sha256=digest,
                )
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
        # Do not return host-local paths for remote env failures. File tools for
        # that task may read inside the remote env, making host paths broken.
    else:
        local_result = _write_to_local_artifact(content, tool_name, tool_use_id)
        if local_result is not None:
            local_path, digest = local_result
            return _build_persisted_message(
                preview,
                has_more,
                len(content),
                local_path,
                tool_name=tool_name,
                sha256=digest,
            )

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no readable artifact write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to a readable artifact.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn."""
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
