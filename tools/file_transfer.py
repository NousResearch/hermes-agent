"""File transfer layer for moving files between sandboxes and the host.

Supports extraction (sandbox → host) and injection (host → sandbox) for all
Hermes execution backends: local, Docker, SSH, Modal, Singularity, Daytona.

Files are staged through a host-side cache (~/.hermes/file_cache/) with
TTL-based cleanup.
"""

import base64
import gzip
import logging
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
FILE_CACHE_DIR = Path(os.path.expanduser("~/.hermes/file_cache"))

# ---------------------------------------------------------------------------
# Deferred injection queue (thread-safe)
# ---------------------------------------------------------------------------
# Files uploaded by the user are queued here *before* the sandbox exists.
# When terminal_tool creates the sandbox environment, it calls
# process_pending_injections() to push queued files into the new container.
_pending_injections: Dict[str, list] = {}
_pending_lock = threading.Lock()


def queue_injection(task_id: str, host_path: str, remote_path: str) -> None:
    """Queue a file for injection once the sandbox for *task_id* is ready."""
    with _pending_lock:
        _pending_injections.setdefault(task_id, []).append({
            "host_path": host_path,
            "remote_path": remote_path,
        })
    logger.info("Queued injection: %s -> %s (task %s)", host_path, remote_path, task_id)


def process_pending_injections(task_id: str) -> list:
    """Inject all queued files for *task_id*.  Called by terminal_tool after
    the sandbox environment is created.  Returns a list of result dicts."""
    with _pending_lock:
        pending = _pending_injections.pop(task_id, [])
    if not pending:
        return []
    results = []
    for item in pending:
        result = inject_file_to_sandbox(
            host_path=item["host_path"],
            remote_path=item["remote_path"],
            task_id=task_id,
        )
        if result.get("success"):
            logger.info("Deferred injection succeeded: %s", item["remote_path"])
        else:
            logger.warning("Deferred injection failed: %s — %s",
                           item["remote_path"], result.get("error"))
        results.append(result)
    return results


def is_safe_file_path(path_str: str) -> Optional[Path]:
    """Validate that *path_str* is inside the file cache directory.

    Returns the canonical ``Path`` if safe, ``None`` otherwise.
    This prevents the LLM from exfiltrating arbitrary host files
    (e.g. ``.env``, SSH keys) by crafting rogue ``FILE:<path>`` tags.
    """
    try:
        cache_root = FILE_CACHE_DIR.resolve(strict=False)
        candidate = Path(path_str).resolve(strict=True)
        candidate.relative_to(cache_root)  # ValueError if outside
        if not candidate.is_file():
            return None
        return candidate
    except (ValueError, OSError, RuntimeError):
        return None

# Extension → MIME type for common file types
_MIME_OVERRIDES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".py": "text/x-python",
    ".js": "application/javascript",
    ".ts": "application/typescript",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".log": "text/plain",
    ".sh": "application/x-sh",
}


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def get_file_cache_dir() -> Path:
    """Return the file cache directory, creating it if needed."""
    FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FILE_CACHE_DIR


def cleanup_file_cache(max_age_hours: int = 24) -> int:
    """Delete cached files older than *max_age_hours*. Returns count removed."""
    cache_dir = get_file_cache_dir()
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ---------------------------------------------------------------------------
# Validation & MIME
# ---------------------------------------------------------------------------

def detect_mime_type(path: str) -> str:
    """Detect MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def validate_file_path(path: str) -> Optional[str]:
    """Validate a file path for safety. Returns error string or None if valid.

    Accepts both Unix (/...) and Windows (C:\\...) absolute paths since
    the tool may run on a Windows host while validating sandbox paths.
    """
    if not path:
        return "Path is empty"
    if "\x00" in path:
        return "Path contains null bytes"
    # Accept Unix absolute paths (start with /) and Windows absolute paths
    is_abs = path.startswith("/") or os.path.isabs(path)
    if not is_abs:
        return "Path must be absolute"
    # Block path traversal: check the raw path for .. components before
    # normalization resolves them (normpath collapses .. on Windows)
    raw_parts = path.replace("\\", "/").split("/")
    if ".." in raw_parts:
        return "Path traversal detected"
    # Block shell injection *patterns* rather than individual characters,
    # so legitimate filenames like "report (final).csv" are accepted.
    # shlex.quote() in extraction/injection is the primary defense;
    # this is a defense-in-depth layer that catches obvious payloads
    # before they reach any shell pipeline.
    _DANGEROUS_PATTERNS = ["$(", "`", ";", "&&", "||", "|", ">>", "<<"]
    for pat in _DANGEROUS_PATTERNS:
        if pat in path:
            return f"Path contains dangerous shell pattern: {pat}"
    return None


# ---------------------------------------------------------------------------
# Extraction: sandbox → host
# ---------------------------------------------------------------------------

def extract_file_from_sandbox(path: str, task_id: str = "default") -> Dict[str, Any]:
    """Extract a file from the active sandbox to the host file cache.

    Args:
        path: Absolute path inside the sandbox.
        task_id: The agent task ID (used to find the active environment).

    Returns:
        Dict with keys: success, host_path, filename, mime_type, size, error.
    """
    error = validate_file_path(path)
    if error:
        return {"success": False, "error": error}

    filename = os.path.basename(path)
    cache_dir = get_file_cache_dir()
    host_path = str(cache_dir / f"{uuid.uuid4().hex[:12]}_{filename}")

    env_type = os.getenv("TERMINAL_ENV", "local")

    try:
        if env_type == "local":
            return _extract_local(path, host_path)
        elif env_type == "docker":
            return _extract_docker(path, task_id, host_path)
        elif env_type == "ssh":
            return _extract_ssh(path, task_id, host_path)
        else:
            # Modal, Singularity, Daytona: gzip+base64 fallback
            return _extract_via_base64(path, task_id, host_path)
    except Exception as e:
        logger.error("File extraction failed: %s", e)
        return {"success": False, "error": str(e)}


def _make_result(host_path: str) -> Dict[str, Any]:
    """Build a success result dict from a cached file."""
    return {
        "success": True,
        "host_path": host_path,
        "filename": os.path.basename(host_path).split("_", 1)[-1],
        "mime_type": detect_mime_type(host_path),
        "size": os.path.getsize(host_path),
    }


def _extract_local(path: str, host_path: str) -> Dict[str, Any]:
    """Local backend: simple file copy."""
    if not os.path.exists(path):
        return {"success": False, "error": f"File not found: {path}"}
    if os.path.isdir(path):
        return {"success": False, "error": "Path is a directory, not a file"}
    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        return {"success": False, "error": f"File too large ({size} bytes, max {MAX_FILE_SIZE})"}
    shutil.copy2(path, host_path)
    return _make_result(host_path)


def _extract_docker(path: str, task_id: str, host_path: str) -> Dict[str, Any]:
    """Docker backend: docker cp container:path host_path."""
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active Docker environment for task {task_id}"}

    container_id = getattr(env, "_container_id", None)
    if not container_id:
        return {"success": False, "error": "Could not determine container ID"}

    result = subprocess.run(
        ["docker", "cp", f"{container_id}:{path}", host_path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {"success": False, "error": f"docker cp failed: {result.stderr.strip()}"}

    if not os.path.exists(host_path):
        return {"success": False, "error": "docker cp completed but file not found on host"}

    size = os.path.getsize(host_path)
    if size > MAX_FILE_SIZE:
        os.unlink(host_path)
        return {"success": False, "error": f"File too large ({size} bytes, max {MAX_FILE_SIZE})"}

    return _make_result(host_path)


def _extract_ssh(path: str, task_id: str, host_path: str) -> Dict[str, Any]:
    """SSH backend: scp with ControlMaster socket."""
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active SSH environment for task {task_id}"}

    control_socket = getattr(env, "control_socket", None)
    host = getattr(env, "host", None)
    user = getattr(env, "user", None)
    port = getattr(env, "port", 22)

    if not all([control_socket, host, user]):
        return {"success": False, "error": "Incomplete SSH environment configuration"}

    # Quote remote path for scp (spaces/special chars in filenames)
    quoted_remote = shlex.quote(path)
    cmd = [
        "scp",
        "-o", f"ControlPath={control_socket}",
        "-P", str(port),
        f"{user}@{host}:{quoted_remote}",
        host_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"success": False, "error": f"scp failed: {result.stderr.strip()}"}

    if not os.path.exists(host_path):
        return {"success": False, "error": "scp completed but file not found on host"}

    size = os.path.getsize(host_path)
    if size > MAX_FILE_SIZE:
        os.unlink(host_path)
        return {"success": False, "error": f"File too large ({size} bytes, max {MAX_FILE_SIZE})"}

    return _make_result(host_path)


def _extract_via_base64(path: str, task_id: str, host_path: str) -> Dict[str, Any]:
    """Fallback extraction via gzip+base64 through the terminal.

    Works for Modal, Singularity, Daytona, or any backend where the only
    interface is command execution.
    """
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active environment for task {task_id}"}

    # First check file exists and size
    safe_path = shlex.quote(path)
    check_cmd = f'stat -c "%s" {safe_path} 2>/dev/null || stat -f "%z" {safe_path} 2>/dev/null'
    check_result = env.execute(check_cmd, timeout=10)
    output = check_result.get("output", "").strip()
    if check_result.get("returncode", 1) != 0 or not output:
        return {"success": False, "error": f"File not found or not accessible: {path}"}

    try:
        file_size = int(output.strip().split("\n")[-1])
    except ValueError:
        return {"success": False, "error": f"Could not determine file size: {output}"}

    if file_size > MAX_FILE_SIZE:
        return {"success": False, "error": f"File too large ({file_size} bytes, max {MAX_FILE_SIZE})"}

    # gzip + base64 encode the file
    encode_cmd = f'gzip -c {safe_path} | base64'
    encode_result = env.execute(encode_cmd, timeout=120)
    if encode_result.get("returncode", 1) != 0:
        return {"success": False, "error": f"Encoding failed: {encode_result.get('output', '')}"}

    b64_data = encode_result.get("output", "").strip()
    if not b64_data:
        return {"success": False, "error": "Encoding returned empty output"}

    # Decode on host
    try:
        compressed = base64.b64decode(b64_data)
        raw_data = gzip.decompress(compressed)
    except Exception as e:
        return {"success": False, "error": f"Decoding failed: {e}"}

    with open(host_path, "wb") as f:
        f.write(raw_data)

    return _make_result(host_path)


# ---------------------------------------------------------------------------
# Injection: host → sandbox
# ---------------------------------------------------------------------------

def inject_file_to_sandbox(
    host_path: str,
    remote_path: str,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Inject a file from the host into the active sandbox.

    Args:
        host_path: Absolute path on the host.
        remote_path: Destination path inside the sandbox.
        task_id: The agent task ID.

    Returns:
        Dict with keys: success, remote_path, error.
    """
    if not os.path.exists(host_path):
        return {"success": False, "error": f"Source file not found: {host_path}"}
    if os.path.getsize(host_path) > MAX_FILE_SIZE:
        return {"success": False, "error": f"File too large (max {MAX_FILE_SIZE} bytes)"}

    env_type = os.getenv("TERMINAL_ENV", "local")

    try:
        if env_type == "local":
            return _inject_local(host_path, remote_path)
        elif env_type == "docker":
            return _inject_docker(host_path, remote_path, task_id)
        elif env_type == "ssh":
            return _inject_ssh(host_path, remote_path, task_id)
        else:
            return _inject_via_base64(host_path, remote_path, task_id)
    except Exception as e:
        logger.error("File injection failed: %s", e)
        return {"success": False, "error": str(e)}


def _inject_local(host_path: str, remote_path: str) -> Dict[str, Any]:
    """Local backend: copy file to destination."""
    os.makedirs(os.path.dirname(remote_path), exist_ok=True)
    shutil.copy2(host_path, remote_path)
    return {"success": True, "remote_path": remote_path}


def _inject_docker(host_path: str, remote_path: str, task_id: str) -> Dict[str, Any]:
    """Docker backend: docker cp host_path container:remote_path."""
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active Docker environment for task {task_id}"}

    container_id = getattr(env, "_container_id", None)
    if not container_id:
        return {"success": False, "error": "Could not determine container ID"}

    # Ensure destination directory exists
    remote_dir = os.path.dirname(remote_path)
    if remote_dir:
        env.execute(f'mkdir -p {shlex.quote(remote_dir)}', timeout=10)

    result = subprocess.run(
        ["docker", "cp", host_path, f"{container_id}:{remote_path}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {"success": False, "error": f"docker cp failed: {result.stderr.strip()}"}

    return {"success": True, "remote_path": remote_path}


def _inject_ssh(host_path: str, remote_path: str, task_id: str) -> Dict[str, Any]:
    """SSH backend: scp reverse."""
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active SSH environment for task {task_id}"}

    control_socket = getattr(env, "control_socket", None)
    host = getattr(env, "host", None)
    user = getattr(env, "user", None)
    port = getattr(env, "port", 22)

    if not all([control_socket, host, user]):
        return {"success": False, "error": "Incomplete SSH environment configuration"}

    # Ensure destination directory exists
    remote_dir = os.path.dirname(remote_path)
    if remote_dir:
        env.execute(f'mkdir -p {shlex.quote(remote_dir)}', timeout=10)

    # Quote remote path for scp (spaces/special chars in filenames)
    quoted_remote = shlex.quote(remote_path)
    cmd = [
        "scp",
        "-o", f"ControlPath={control_socket}",
        "-P", str(port),
        host_path,
        f"{user}@{host}:{quoted_remote}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"success": False, "error": f"scp failed: {result.stderr.strip()}"}

    return {"success": True, "remote_path": remote_path}


def _inject_via_base64(host_path: str, remote_path: str, task_id: str) -> Dict[str, Any]:
    """Fallback injection via base64 through the terminal.

    Reads the file, gzip compresses, base64-encodes, and writes via echo
    piped through base64 decode + gunzip on the remote side.
    """
    from tools.terminal_tool import _active_environments

    env = _active_environments.get(task_id)
    if not env:
        return {"success": False, "error": f"No active environment for task {task_id}"}

    # Read and encode the file
    with open(host_path, "rb") as f:
        raw_data = f.read()

    compressed = gzip.compress(raw_data)
    b64_data = base64.b64encode(compressed).decode("ascii")

    # Ensure destination directory exists
    remote_dir = os.path.dirname(remote_path)
    if remote_dir:
        env.execute(f'mkdir -p {shlex.quote(remote_dir)}', timeout=10)

    # Write in chunks to avoid command-line length limits (60KB per chunk)
    chunk_size = 60000
    chunks = [b64_data[i:i + chunk_size] for i in range(0, len(b64_data), chunk_size)]

    # Write first chunk (overwrite) - b64 data is safe (alphanumeric+/+=)
    write_result = env.execute(
        f'echo -n "{chunks[0]}" > /tmp/_hermes_upload.b64',
        timeout=30,
    )
    if write_result.get("returncode", 1) != 0:
        return {"success": False, "error": "Failed to write base64 data chunk"}

    # Append remaining chunks
    for chunk in chunks[1:]:
        env.execute(f'echo -n "{chunk}" >> /tmp/_hermes_upload.b64', timeout=30)

    # Decode and decompress
    safe_remote = shlex.quote(remote_path)
    decode_cmd = f'base64 -d /tmp/_hermes_upload.b64 | gunzip > {safe_remote} && rm -f /tmp/_hermes_upload.b64'
    decode_result = env.execute(decode_cmd, timeout=60)
    if decode_result.get("returncode", 1) != 0:
        return {"success": False, "error": f"Decoding failed: {decode_result.get('output', '')}"}

    return {"success": True, "remote_path": remote_path}
