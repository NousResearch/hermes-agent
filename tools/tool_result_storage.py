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
   ``$HERMES_HOME/cache/spillover/tool-results/{tool_use_id}.txt`` — alongside
   the other Hermes-owned caches (images, audio, documents, ...) instead of
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
import posixpath
import re
import shlex
import stat
import threading
import time

from tools.budget_config import DEFAULT_PREVIEW_SIZE_CHARS, BudgetConfig, DEFAULT_BUDGET

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
SPILLOVER_SUBDIR = "cache/spillover"
PERSISTED_SPILLOVER_SUBDIR = "tool-results"
SPILLOVER_MAX_AGE_HOURS = 24
RESULT_TTL_DAYS = 7
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120

_spillover_prune_lock = threading.Lock()
_spillover_pruned_once = False


def get_spillover_root():
    """Return the shared ``$HERMES_HOME/cache/spillover`` root."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / SPILLOVER_SUBDIR


def get_spillover_dir():
    """Return the private tool-result spill directory (not created)."""
    return get_spillover_root() / PERSISTED_SPILLOVER_SUBDIR


def cleanup_spillover_cache(max_age_hours: int = SPILLOVER_MAX_AGE_HOURS) -> int:
    """Delete spillover files older than *max_age_hours*; returns count removed (same
    contract as the ``cleanup_*_cache`` helpers the gateway housekeeping loop runs hourly)."""
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    entries = []
    spillover_root = get_spillover_root()
    try:
        root_stat = os.lstat(spillover_root)
    except OSError:
        return 0
    if not stat.S_ISDIR(root_stat.st_mode):
        return 0
    for directory in (spillover_root, get_spillover_dir()):
        try:
            directory_stat = os.lstat(directory)
            if not stat.S_ISDIR(directory_stat.st_mode):
                continue
            entries.extend(directory.iterdir())
        except OSError:
            continue
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def _prune_spillover_once() -> None:
    """Best-effort prune, at most once per process (CLI-only installs never run housekeeping)."""
    global _spillover_pruned_once
    with _spillover_prune_lock:
        if _spillover_pruned_once:
            return
        _spillover_pruned_once = True
    try:
        if removed := cleanup_spillover_cache():
            logger.debug("Pruned %d expired spillover file(s)", removed)
    except Exception as exc:
        logger.debug("Spillover prune failed: %s", exc)


def _expire_host_spillover_on_access(path) -> bool | None:
    """Delete an expired canonical host spill before it is served.

    Returns ``True`` when an expired file was removed, ``False`` when a regular
    file is still current (or already absent), and ``None`` when the path cannot
    be verified safely. Symlinks and other non-regular files fail closed.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return None

    if not stat.S_ISREG(st.st_mode):
        return None
    if st.st_mtime >= time.time() - (SPILLOVER_MAX_AGE_HOURS * 3600):
        return False
    try:
        os.unlink(path)
    except FileNotFoundError:
        return True
    except OSError:
        return None
    return True


def _is_host_side_env(env) -> bool:
    """True when this process should write the spill file directly: ``env=None`` (no sandbox
    yet) or the local backend. Remote backends resolve ``read_file`` inside the sandbox."""
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment
        return isinstance(env, LocalEnvironment)
    except Exception:
        return False


def _write_to_spillover(content: str, filename: str):
    """Write content privately to ``$HERMES_HOME/cache/spillover``.

    Returns the absolute path string on success, None on failure. Existing
    targets are replaced via a symlink-refusing exclusive create.
    """
    if os.path.basename(filename) != filename:
        logger.warning("Spillover write refused unsafe filename: %s", filename)
        return None
    try:
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive

        spill_root = ensure_spill_dir(get_spillover_root(), private=False)
        spill_dir = ensure_spill_dir(
            spill_root / PERSISTED_SPILLOVER_SUBDIR,
            private=True,
        )
        path = spill_dir / filename
        write_text_exclusive(
            path,
            content,
            private=True,
            overwrite=True,
            errors="replace",
        )
    except Exception as exc:
        logger.warning("Spillover write failed for %s: %s", filename, exc)
        return None
    _prune_spillover_once()
    return str(path)


def _sandbox_visible_spillover_path(host_path: str, env) -> str | None:
    """Path where a remote backend can read *host_path*, or None. Translates via the image
    tools' helper, forces a sync for synced backends, then PROBES readability — a persistent
    container created before spillover joined the mount list lacks the bind mount and must
    fall back to the in-sandbox write."""
    try:
        from tools.credential_files import to_agent_visible_cache_path
        visible = to_agent_visible_cache_path(host_path)
    except Exception as exc:
        logger.debug("Spillover path translation failed: %s", exc)
        return None
    try:
        if (sync_manager := getattr(env, "_sync_manager", None)) is not None:
            sync_manager.sync(force=True)
    except Exception as exc:
        logger.debug("Spillover sync failed: %s", exc)
    try:
        if env.execute(f"test -r {shlex.quote(visible)}", timeout=15).get("returncode", 1) == 0:
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
                if isinstance(temp_dir, str) and temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id
    safe_stem = safe_stem or "tool_result"
    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"
    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    last_nl = content.rfind("\n", 0, max_chars)
    return content[:last_nl + 1 if last_nl > max_chars // 2 else max_chars], True


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(); True on success. Content goes through
    stdin, not the command string: Linux ``MAX_ARG_STRLEN`` caps one argv element at 128 KB,
    so a heredoc-in-command silently failed for exactly the oversized results this handles."""
    storage_dir = os.path.dirname(remote_path)
    quoted_dir = shlex.quote(storage_dir)
    quoted_path = shlex.quote(remote_path)
    # Persisted results can contain credentials or private session history.
    # Create them under a private directory with a restrictive umask, and
    # remove the target before a retry so an older permissive mode cannot
    # survive shell redirection. On systems with NFSv4-style ACLs (notably
    # macOS), chmod alone does not revoke inherited grants, so strip those
    # ACLs before applying and verifying the private mode. POSIX ACLs are
    # constrained by the mode bits, and setfacl removes any residual entries
    # where available. Reject a symlinked or foreign-owned leaf directory
    # before writing beneath a shared temp root. Cleanup is deliberately
    # best-effort: a backend without a compatible `find` must still persist.
    cmd = (
        "umask 077 && "
        f"[ ! -L {quoted_dir} ] && mkdir -p {quoted_dir} && "
        f"[ -O {quoted_dir} ] && "
        f"if [ \"$(uname -s 2>/dev/null)\" = Darwin ]; then "
        f"chmod -N {quoted_dir}; "
        "elif command -v setfacl >/dev/null 2>&1; then "
        f"setfacl -b -k {quoted_dir}; fi && "
        f"chmod 700 {quoted_dir} && "
        f"(find {quoted_dir} \\( -type f -o -type l \\) "
        f"-name '*.txt' -mtime +{RESULT_TTL_DAYS - 1} "
        "-exec rm -f {} + 2>/dev/null || true) && "
        f"rm -f {quoted_path} && cat > {quoted_path} && "
        f"if [ \"$(uname -s 2>/dev/null)\" = Darwin ]; then "
        f"chmod -N {quoted_path}; "
        "elif command -v setfacl >/dev/null 2>&1; then "
        f"setfacl -b {quoted_path}; fi && "
        f"chmod 600 {quoted_path} && "
        f"mode=$(stat -c '%a' {quoted_path} 2>/dev/null || "
        f"stat -f '%Lp' {quoted_path} 2>/dev/null) && [ \"$mode\" = 600 ]"
    )
    return env.execute(cmd, timeout=30, stdin_data=content).get("returncode", 1) == 0


def _expire_persisted_result_on_access(remote_path: str, env) -> bool | None:
    """Delete an expired persisted result before it is served.

    Returns ``True`` when an existing result was expired and removed, ``False``
    when the retention probe completed and the result is still current, and
    ``None`` when expiry or deletion could not be verified. Callers restrict
    this helper to the active environment's resolved ``hermes-results``
    directory and must fail closed on ``None``.
    """
    quoted_path = shlex.quote(remote_path)
    quoted_dir = shlex.quote(posixpath.dirname(remote_path))
    cmd = (
        f"[ ! -L {quoted_dir} ] && [ -d {quoted_dir} ] && "
        f"expired=$(find {quoted_path} -prune \\( -type f -o -type l \\) "
        f"-mtime +{RESULT_TTL_DAYS - 1} -print -quit 2>/dev/null) && "
        "if [ -n \\\"$expired\\\" ]; then "
        f"rm -f {quoted_path} && printf '%s' expired; "
        "fi"
    )
    result = env.execute(cmd, timeout=30)
    if result.get("returncode", 1) != 0:
        return None
    output = result.get("output", result.get("stdout", ""))
    return output.strip() == "expired"


def _expire_remote_spillover_on_access(remote_path: str, env) -> bool | None:
    """Delete an expired sandbox-visible canonical spill before serving it."""
    quoted_path = shlex.quote(remote_path)
    quoted_dir = shlex.quote(posixpath.dirname(remote_path))
    max_age_minutes = SPILLOVER_MAX_AGE_HOURS * 60
    cmd = (
        f"[ ! -L {quoted_dir} ] && [ -d {quoted_dir} ] && "
        f"expired=$(find {quoted_path} -prune \\( -type f -o -type l \\) "
        f"-mmin +{max_age_minutes - 1} -print -quit 2>/dev/null) && "
        "if [ -n \\\"$expired\\\" ]; then "
        f"rm -f {quoted_path} && printf '%s' expired; "
        "fi"
    )
    result = env.execute(cmd, timeout=30)
    if result.get("returncode", 1) != 0:
        return None
    output = result.get("output", result.get("stdout", ""))
    return output.strip() == "expired"


def _build_persisted_message(preview: str, has_more: bool, original_size: int,
                             file_path: str) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
        f"Full output saved to: {file_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        + preview + ("\n..." if has_more else "")
        + f"\n{PERSISTED_OUTPUT_CLOSING_TAG}")


_PERSISTED_PATH_RE = re.compile(r"^Full output saved to: (.+)$", re.MULTILINE)


def extract_persisted_path(content: str) -> str | None:
    """File path from a <persisted-output> block, or None (lets the result-reference stubbing
    guard in agent/tool_guardrails.py carry the spillover path instead of leaving it dangling)."""
    match = (_PERSISTED_PATH_RE.search(content)
             if isinstance(content, str) and PERSISTED_OUTPUT_TAG in content else None)
    return match.group(1).strip() if match else None


def maybe_persist_tool_result(content: str, tool_name: str, tool_use_id: str, env=None,
                              config: BudgetConfig = DEFAULT_BUDGET,
                              threshold: int | float | None = None) -> str:
    """Layer 2: persist an oversized result, return preview + path. ``threshold`` overrides
    ``config.resolve_threshold(tool_name)``; falls back to inline truncation when no write
    location succeeds."""
    if threshold is None:
        threshold = config.resolve_threshold(tool_name)
    if threshold == float("inf") or len(content) <= threshold:
        return content
    filename = _safe_result_filename(tool_use_id)
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    def _persisted(path: str, host_suffix: str = "") -> str:
        logger.info("Persisted large tool result: %s (%s, %d chars -> %s%s)",
                    tool_name, tool_use_id, len(content), path, host_suffix)
        return _build_persisted_message(preview, has_more, len(content), path)

    # Always persist host-side first: cache/spillover is the single canonical home.
    host_path = _write_to_spillover(content, filename)
    host_side = _is_host_side_env(env)
    if host_side and host_path is not None:
        return _persisted(host_path)
    if not host_side:
        # Remote backend: reference the mounted/synced path when the sandbox can actually read
        # it, else write into the sandbox temp dir (containers without the spillover mount).
        visible = _sandbox_visible_spillover_path(host_path, env) if host_path else None
        if visible is not None:
            return _persisted(visible, f" [host: {host_path}]")
        remote_path = f"{_resolve_storage_dir(env)}/{filename}"
        try:
            if _write_to_sandbox(content, remote_path, env):
                return _persisted(remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
    logger.info("Inline-truncating large tool result: %s (%d chars, no sandbox write)",
                tool_name, len(content))
    return (f"{preview}\n\n[Truncated: tool response was {len(content):,} chars. "
            "Full output could not be saved to sandbox.]")


def enforce_turn_budget(tool_messages: list[dict], env=None,
                        config: BudgetConfig = DEFAULT_BUDGET) -> list[dict]:
    """Layer 3: persist the largest non-persisted results first until the turn's aggregate is
    under budget. Mutates the list in-place and returns it."""
    sizes = [len(msg.get("content", "")) for msg in tool_messages]
    total_size = sum(sizes)
    candidates = [(i, size) for i, size in enumerate(sizes)
                  if PERSISTED_OUTPUT_TAG not in tool_messages[i].get("content", "")]
    if total_size <= config.turn_budget:
        return tool_messages
    for idx, size in sorted(candidates, key=lambda x: x[1], reverse=True):
        if total_size <= config.turn_budget:
            break
        content = tool_messages[idx]["content"]
        tool_use_id = tool_messages[idx].get("tool_call_id", f"budget_{idx}")
        replacement = maybe_persist_tool_result(
            content=content, tool_name=_BUDGET_TOOL_NAME, tool_use_id=tool_use_id,
            env=env, config=config, threshold=0)
        if replacement != content:
            total_size += len(replacement) - size
            tool_messages[idx]["content"] = replacement
            logger.info("Budget enforcement: persisted tool result %s (%d chars)",
                        tool_use_id, size)
    return tool_messages


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import uuid  # noqa: F401,E402

HEREDOC_MARKER = "HERMES_PERSIST_EOF"
# ---- END PLUGIN-COMPAT ----
