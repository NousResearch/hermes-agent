"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is persisted and the
   in-context content is replaced with a preview + file path reference.

   The canonical home is ALWAYS host-side under
   ``$HERMES_HOME/cache/spillover`` — alongside the other Hermes-owned caches
   (images, audio, documents, ...) instead of littering the OS temp dir. Real
   sessions receive an opaque ``hermes-spill://`` capability bound to their
   session scope; legacy/direct callers without session context retain the
   path-based behavior. This needs no sandbox environment, so it also works
   for sessions that never ran a terminal command (MCP-only, cron, gateway).

   What the model sees depends on the backend:

   - **Session-scoped calls:** an opaque URI resolved host-side by read_file.
   - **Legacy local calls (or no active env):** the host path itself.
   - **Legacy remote calls (docker/ssh/modal/daytona):** ``cache/spillover`` is
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
import hmac
import logging
import os
import re
import secrets
import shlex
import stat
import threading
import time
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
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
_CAPABILITY_URI_RE = re.compile(
    r"^hermes-spill://v1/"
    r"(?P<scope>[0-9a-f]{64})/"
    r"(?P<digest>[0-9a-f]{64})/"
    r"(?P<capability>[0-9a-f]{32})$"
)
_CAPABILITY_FILENAME = "spill_{scope}_{digest}_{locator}.bin"
_SAFE_CAPABILITY_ERROR = "invalid or inaccessible result-spill capability"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_spillover_prune_lock = threading.Lock()
_spillover_pruned_once = False


class SpillCapabilityError(Exception):
    """Fail-closed capability read whose message never reveals a host path."""


def _scope_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _capability_uri(scope: str, digest: str, capability: str) -> str:
    return f"hermes-spill://v1/{scope}/{digest}/{capability}"


def _capability_filename(scope: str, digest: str, capability: str) -> str:
    locator = hashlib.sha256(capability.encode("ascii")).hexdigest()
    return _CAPABILITY_FILENAME.format(
        scope=scope,
        digest=digest,
        locator=locator,
    )


def _capability_aad(scope: str, digest: str) -> bytes:
    return f"hermes-spill-v1:{scope}:{digest}".encode("ascii")


def _capability_key(capability: str) -> bytes:
    return hashlib.sha256(capability.encode("ascii")).digest()


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


def _write_capability_spillover(content: str, session_id: str) -> str | None:
    """Persist authenticated ciphertext and return an opaque same-session URI."""
    scope = str(session_id or "").strip()
    if not scope:
        return None
    data = content.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    spill_dir = get_spillover_dir()
    try:
        spill_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(spill_dir, 0o700)
        scope_hash = _scope_hash(scope)
        for _attempt in range(3):
            capability = secrets.token_hex(16)
            path = spill_dir / _capability_filename(
                scope_hash, digest, capability,
            )
            try:
                fd = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                    0o600,
                )
            except FileExistsError:
                continue
            nonce = secrets.token_bytes(12)
            encrypted = nonce + AESGCM(_capability_key(capability)).encrypt(
                nonce,
                data,
                _capability_aad(scope_hash, digest),
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encrypted)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    path.unlink()
                except OSError:
                    pass
                raise
            _prune_spillover_once()
            return _capability_uri(scope_hash, digest, capability)
    except OSError as exc:
        logger.warning("Capability spillover write failed: %s", exc)
    return None


def resolve_spill_capability(uri: str, session_id: str = "") -> str:
    """Resolve an opaque spill URI only for the session that minted it."""
    scope = str(session_id or "").strip()
    parsed = _CAPABILITY_URI_RE.fullmatch(str(uri or "").strip())
    if not scope or parsed is None:
        raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
    expected_scope = _scope_hash(scope)
    if not hmac.compare_digest(expected_scope, parsed.group("scope")):
        raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
    path = get_spillover_dir() / _capability_filename(
        expected_scope,
        parsed.group("digest"),
        parsed.group("capability"),
    )
    fd = -1
    try:
        link_metadata = os.lstat(path)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            stat.S_ISLNK(link_metadata.st_mode)
            or getattr(link_metadata, "st_file_attributes", 0) & reparse_flag
        ):
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        fd = os.open(
            path,
            os.O_RDONLY | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
        )
        metadata = os.fstat(fd)
        if (
            getattr(link_metadata, "st_dev", None),
            getattr(link_metadata, "st_ino", None),
        ) != (
            getattr(metadata, "st_dev", None),
            getattr(metadata, "st_ino", None),
        ):
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        if not stat.S_ISREG(metadata.st_mode):
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        encrypted = b"".join(chunks)
        if len(encrypted) < 13:
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        nonce, ciphertext = encrypted[:12], encrypted[12:]
        data = AESGCM(_capability_key(parsed.group("capability"))).decrypt(
            nonce,
            ciphertext,
            _capability_aad(expected_scope, parsed.group("digest")),
        )
        digest = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(digest, parsed.group("digest")):
            raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR)
        return data.decode("utf-8")
    except SpillCapabilityError:
        raise
    except (OSError, UnicodeDecodeError, InvalidTag):
        raise SpillCapabilityError(_SAFE_CAPABILITY_ERROR) from None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


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
    session_id: str = "",
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
        session_id: Session scope used to mint an opaque recovery capability.

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

    # Real sessions get host-resolved capabilities instead of predictable
    # filesystem paths. This also prevents reused tool-call IDs in different
    # sessions from overwriting each other's payloads.
    capability_uri = _write_capability_spillover(content, session_id)
    if capability_uri is not None:
        logger.info(
            "Persisted large tool result behind session capability: %s (%s, %d chars)",
            tool_name, tool_use_id, len(content),
        )
        return _build_persisted_message(
            preview, has_more, len(content), capability_uri,
        )
    if str(session_id or "").strip():
        # Never downgrade a scoped session to a predictable host/sandbox path.
        # If secure capability creation fails, preserve bounded context and fail
        # closed on recovery rather than leaking a path or crossing sessions.
        logger.warning(
            "Secure result-spill capability unavailable for %s; returning bounded preview",
            tool_use_id,
        )
        return (
            f"{preview}\n\n"
            f"[Truncated: tool response was {len(content):,} chars. "
            "Secure result storage was unavailable; no recovery path was emitted.]"
        )

    # Calls without session context retain the legacy path-based behavior:
    # write the single canonical host copy under the spillover cache.
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
    session_id: str = "",
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
            session_id=session_id,
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
