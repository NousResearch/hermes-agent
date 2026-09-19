#!/usr/bin/env python3
"""Send File Tool - Transfer files from local or sandboxed environments to users.

Extracts files generated inside sandboxed backends (Docker, SSH, Modal, Daytona,
Singularity, Vercel) or local execution environments to the host cache and delivers
them via the gateway's native MEDIA: attachment pipeline.
"""

import base64
import binascii
import logging
import os
import posixpath
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agent.file_safety import get_read_block_error
from tools.file_tools import _get_file_ops, _is_blocked_device, _special_file_kind
from tools.file_tools_paths import _expand_tilde, _resolve_path_for_task
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Default max file size for outbound transfer (50 MB matching platform upload limits).
_DEFAULT_MAX_SEND_BYTES = 50 * 1024 * 1024

# Remote sensitive patterns that must not be exfiltrated from remote/sandboxed environments.
_REMOTE_SENSITIVE_PATTERNS = (
    re.compile(r"(?:^|/)\.ssh(?:/|$)"),
    re.compile(r"(?:^|/)\.aws(?:/|$)"),
    re.compile(r"(?:^|/)\.env(?:\.[^/]+)?$"),
    re.compile(r"(?:^|/)\.(?:netrc|pgpass|npmrc|pypirc|dockercfg)$"),
    re.compile(r"(?:^|/)\.docker/config\.json$"),
    re.compile(r"^/(?:etc|private/etc)/(?:shadow|master\.passwd|sudoers|security)"),
    re.compile(r"^/(?:proc|dev|sys)/"),
    re.compile(r"(?:^|/)\.hermes(?:/|$)"),
)

SEND_FILE_SCHEMA = {
    "name": "send_file",
    "description": (
        "Send a file from the terminal environment to the user or messaging platform. "
        "Extracts generated files, reports, charts, documents, or data artifacts from local or "
        "sandboxed terminal environments (Docker, SSH, Modal, Singularity, Daytona, Vercel) "
        "and delivers them as native media attachments or download references."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file inside the terminal environment to send "
                    "(e.g. '/workspace/report.pdf', 'output/chart.png', or 'data.csv')"
                ),
            },
            "message": {
                "type": "string",
                "description": "Optional caption or description message to accompany the file delivery",
            },
        },
        "required": ["path"],
    },
}


def _format_size(num_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _is_device_or_proc_path(path: str) -> bool:
    """Check if a path is a /dev/, /proc/, or /sys/ device path across platforms."""
    if not path:
        return False
    clean = path.replace("\\", "/").strip()
    if clean.startswith("/dev/") or clean.startswith("/proc/") or clean.startswith("/sys/"):
        return True
    return _is_blocked_device(path)


def _check_remote_sensitive_path(path: str) -> Optional[str]:
    """Check if a remote/sandbox path (or its resolved realpath) targets sensitive files."""
    if not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    for pattern in _REMOTE_SENSITIVE_PATTERNS:
        if pattern.search(normalized) or pattern.search(path):
            return f"access denied for protected remote path '{path}'"
    return None


def _cache_extracted_bytes(data: bytes, filename: str) -> str:
    """Save raw file bytes to the host document/media cache and return the host path."""
    try:
        from gateway.platforms.base import cache_document_from_bytes

        return cache_document_from_bytes(data, filename)
    except Exception:
        # Fallback if gateway platform helper is not initialized (e.g. CLI standalone)
        from hermes_constants import get_hermes_dir
        import uuid

        cache_dir = get_hermes_dir("cache/documents", "document_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "document"
        target = cache_dir / f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
        target.write_bytes(data)
        return str(target)


def _extract_local_file(resolved_path: str, max_bytes: int) -> Tuple[Optional[bytes], Optional[str]]:
    """Read a local file from the host filesystem using object-bound file descriptor inspection."""
    # 1. Directory check
    if os.path.isdir(resolved_path):
        return None, f"'{resolved_path}' is a directory, not a regular file. Archive it into a .zip or .tar.gz first."

    # 2. Reject special files (FIFOs, sockets, char/block devices) before opening to prevent hangs
    special_kind = _special_file_kind(resolved_path)
    if special_kind:
        return None, f"'{resolved_path}' is {special_kind}, not a regular file."

    # 3. Name / path guard check
    if _is_device_or_proc_path(resolved_path):
        return None, f"access denied for device or proc path '{resolved_path}'."

    # 4. Object-bound open and fstat check
    fd = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK

        fd = os.open(resolved_path, flags)
        st = os.fstat(fd)

        # Check directory vs regular file on opened descriptor
        if stat.S_ISDIR(st.st_mode):
            return None, f"'{resolved_path}' is a directory, not a regular file. Archive it into a .zip or .tar.gz first."
        if not stat.S_ISREG(st.st_mode):
            return None, f"'{resolved_path}' is a special (non-regular) file, not a regular file."

        if st.st_size > max_bytes:
            return None, f"File size ({_format_size(st.st_size)}) exceeds the maximum allowed transfer size ({_format_size(max_bytes)})."

        # Read at most max_bytes + 1 to enforce size limit directly on descriptor
        data = os.read(fd, max_bytes + 1)
        if len(data) > max_bytes:
            return None, f"File size ({_format_size(len(data))}) exceeds the maximum allowed transfer size ({_format_size(max_bytes)})."

        return data, None
    except FileNotFoundError:
        return None, f"File not found: '{resolved_path}'"
    except PermissionError:
        return None, f"Permission denied reading file: '{resolved_path}'"
    except BlockingIOError:
        return None, f"Cannot read '{resolved_path}': resource would block (special file or FIFO)."
    except OSError as exc:
        return None, f"Failed to read file '{resolved_path}': {exc}"
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _extract_remote_file(file_ops, raw_remote_path: str, max_bytes: int) -> Tuple[Optional[bytes], Optional[str]]:
    """Extract a file from a sandboxed or remote terminal backend with atomic realpath resolution and size validation."""
    quoted_raw = shlex.quote(raw_remote_path)
    cwd = getattr(file_ops, "cwd", "/workspace")
    if not isinstance(cwd, str) or not cwd:
        cwd = "/workspace"
    quoted_cwd = shlex.quote(cwd)

    # Compound script to resolve path, expand ~, probe existence, inspect realpath, check size, and base64 dump
    script = (
        f'p={quoted_raw}\n'
        f'case "$p" in\n'
        f'  "~"*) p="$HOME${{p#\\~}}" ;;\n'
        f'  /*) ;;\n'
        f'  *) p="{quoted_cwd}/$p" ;;\n'
        f'esac\n'
        f'if [ ! -e "$p" ] && [ ! -L "$p" ]; then\n'
        f'  echo "__HERMES_NOT_FOUND__"\n'
        f'  exit 0\n'
        f'fi\n'
        f'if [ -d "$p" ]; then\n'
        f'  echo "__HERMES_DIR__"\n'
        f'  exit 0\n'
        f'fi\n'
        f'rp=""\n'
        f'if command -v realpath >/dev/null 2>&1; then\n'
        f'  rp=$(realpath "$p" 2>/dev/null || echo "")\n'
        f'elif command -v readlink >/dev/null 2>&1; then\n'
        f'  rp=$(readlink -f "$p" 2>/dev/null || echo "")\n'
        f'fi\n'
        f'[ -z "$rp" ] && rp="$p"\n'
        f'echo "__HERMES_REALPATH__:$rp"\n'
        f'if [ ! -f "$p" ]; then\n'
        f'  echo "__HERMES_SPECIAL__"\n'
        f'  exit 0\n'
        f'fi\n'
        f'sz=$(wc -c < "$p" 2>/dev/null || echo "0")\n'
        f'echo "__HERMES_SIZE__:$sz"\n'
        f'if [ "$sz" -gt {max_bytes} ]; then\n'
        f'  echo "__HERMES_TOO_LARGE__:$sz"\n'
        f'  exit 0\n'
        f'fi\n'
        f'echo "__HERMES_DATA__"\n'
        f'(base64 -w 0 "$p" 2>/dev/null || base64 "$p" 2>/dev/null)\n'
    )

    res = file_ops._exec(script)
    output = (res.output or "").strip()

    if "__HERMES_NOT_FOUND__" in output or (res.returncode != 0 and not output):
        return None, f"File not found in sandbox: '{raw_remote_path}'"
    if "__HERMES_DIR__" in output:
        return None, f"'{raw_remote_path}' is a directory in the sandbox, not a regular file. Archive it into a .zip or .tar.gz first."
    if "__HERMES_SPECIAL__" in output:
        return None, f"'{raw_remote_path}' is a special (non-regular) file or FIFO in the sandbox."
    if "__HERMES_TOO_LARGE__" in output:
        m = re.search(r"__HERMES_TOO_LARGE__:(\d+)", output)
        sz = int(m.group(1)) if m else max_bytes + 1
        return None, f"File size ({_format_size(sz)}) exceeds the maximum allowed transfer size ({_format_size(max_bytes)})."

    # 1. Verify remote realpath for symlink targets against sensitive targets
    rp_match = re.search(r"__HERMES_REALPATH__:(.+)", output)
    if rp_match:
        remote_realpath = rp_match.group(1).strip()
        sec_err = _check_remote_sensitive_path(remote_realpath)
        if sec_err:
            return None, f"Access denied for protected sandbox target '{remote_realpath}': {sec_err}"

    # 2. Extract base64 payload after marker
    if "__HERMES_DATA__" not in output:
        return None, f"Failed to extract file '{raw_remote_path}' from sandbox: {output or 'empty output'}"

    b64_part = output.split("__HERMES_DATA__", 1)[1].strip()
    clean_b64 = re.sub(r"\s+", "", b64_part)
    try:
        raw_bytes = base64.b64decode(clean_b64)
    except (binascii.Error, ValueError) as exc:
        return None, f"Failed to decode base64 file data for '{raw_remote_path}': {exc}"

    # 3. Postcondition: Enforce decoded payload cap strictly
    if len(raw_bytes) > max_bytes:
        return None, f"File size ({_format_size(len(raw_bytes))}) exceeds the maximum allowed transfer size ({_format_size(max_bytes)})."

    return raw_bytes, None


def send_file_tool(path: str, message: Optional[str] = None, task_id: str = "default") -> str:
    """Core implementation of the send_file tool."""
    if not path or not isinstance(path, str) or not path.strip():
        return tool_error("send_file: missing required field 'path'. Specify the file path to send.")

    raw_path = path.strip()

    # Name-based blocked device check upfront
    if _is_device_or_proc_path(raw_path):
        return tool_error(f"send_file: access denied for device or proc path '{raw_path}'.")

    # Obtain the file operations manager for the active task environment
    file_ops = _get_file_ops(task_id)
    env = getattr(file_ops, "env", None)
    is_local = getattr(env, "is_local", False) or env is None

    filename = Path(raw_path).name or "file"

    if is_local:
        # Resolve task path first
        try:
            resolved = str(_resolve_path_for_task(raw_path, task_id))
        except Exception:
            resolved = os.path.abspath(_expand_tilde(raw_path))

        # Security checks on host resolved path
        if _is_device_or_proc_path(resolved):
            return tool_error(f"send_file: access denied for device or proc path '{raw_path}'.")

        sec_err = get_read_block_error(resolved)
        if sec_err:
            return tool_error(f"send_file: access denied for protected path '{raw_path}': {sec_err}")

        data, err = _extract_local_file(resolved, _DEFAULT_MAX_SEND_BYTES)
        if err:
            return tool_error(f"send_file: {err}")

        cached_host_path = _cache_extracted_bytes(data, filename)
    else:
        # Remote / Sandbox environment
        # Pre-check raw path against remote sensitive patterns
        sec_err = _check_remote_sensitive_path(raw_path)
        if sec_err:
            return tool_error(f"send_file: access denied for protected path '{raw_path}': {sec_err}")

        data, err = _extract_remote_file(file_ops, raw_path, _DEFAULT_MAX_SEND_BYTES)
        if err:
            return tool_error(f"send_file: {err}")

        cached_host_path = _cache_extracted_bytes(data, filename)

    size_str = _format_size(len(data))
    caption = f"{message.strip()}\n\n" if message and message.strip() else ""
    return f"File ready for delivery: {filename} ({size_str})\n{caption}MEDIA:{cached_host_path}"


def _handle_send_file(args: Dict[str, Any], **kw) -> str:
    tid = kw.get("task_id") or "default"
    return send_file_tool(
        path=args.get("path", ""),
        message=args.get("message"),
        task_id=tid,
    )


def _check_send_file_reqs() -> Tuple[bool, str]:
    return True, ""


# Register tool with central Hermes registry
registry.register(
    name="send_file",
    toolset="file",
    schema=SEND_FILE_SCHEMA,
    handler=_handle_send_file,
    check_fn=_check_send_file_reqs,
    emoji="📤",
    max_result_size_chars=10_000,
)
