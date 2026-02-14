#!/usr/bin/env python3
"""
Sandbox Server - HTTP server that runs inside Nomad containers.

This server handles tool execution requests from the SlotPool/SandboxExecutor.
Each slot has an isolated workspace directory where tools execute.

Usage (inside container):
    python -m atropos_agent.sandbox_server --port 8080 --slots 10

API:
    POST /execute  - Execute a single tool in a slot's workspace
    POST /batch    - Execute multiple tools in parallel
    GET  /health   - Health check and status
    POST /reset    - Reset a slot's workspace (clear files)
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import socket
import shutil
import signal
import subprocess
import tempfile
import tarfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web


# Check if bubblewrap is available
def _check_bwrap_available() -> bool:
    """Check if bubblewrap (bwrap) is installed and usable for sandboxing."""
    try:
        version = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            timeout=5,
        )
        if version.returncode != 0:
            return False

        # Some environments (e.g. Docker without extra capabilities) have bwrap installed
        # but can't create the required namespaces/mounts. Probe a minimal sandbox so we
        # can fall back to unsandboxed execution instead of failing all bash tools.
        probe = subprocess.run(
            [
                "bwrap",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-uts",
                "--unshare-ipc",
                "--die-with-parent",
                "--ro-bind",
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "/bin/sh",
                "-c",
                "true",
            ],
            capture_output=True,
            timeout=5,
        )
        return probe.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


BWRAP_AVAILABLE = _check_bwrap_available()


class SlotState(Enum):
    """State of a sandbox slot."""
    AVAILABLE = "available"
    EXECUTING = "executing"


@dataclass
class SlotInfo:
    """Information about a slot in this container."""
    slot_id: str
    workspace_dir: Path
    state: SlotState = SlotState.AVAILABLE
    current_execution_id: Optional[str] = None


@dataclass
class ExecuteRequest:
    """Request to execute a tool in a slot."""
    slot_id: str
    tool: str  # Tool name: "bash", "bash_stateful", "read_file", "write_file", "tmux"
    args: Dict[str, Any]
    execution_id: Optional[str] = None  # For tracking
    timeout: float = 30.0


@dataclass 
class ExecuteResponse:
    """Response from tool execution."""
    success: bool
    output: str = ""
    error: str = ""
    execution_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_id": self.execution_id,
            "metadata": self.metadata,
        }


@dataclass
class _StatefulTmuxSession:
    """
    Per-slot stateful tmux session hosted inside a long-lived bubblewrap sandbox.

    This provides:
    - PID namespace isolation (so state doesn't leak across slots)
    - Persistent process state across tool calls (tmux session stays alive)
    - Kill/cleanup by terminating the bwrap process
    """

    bwrap_proc: Optional[asyncio.subprocess.Process] = None
    sock_relpath: str = "tmux.sock"
    session_name: str = "s"
    pane_width: int = 120
    pane_height: int = 40
    allow_network: bool = True
    prev_capture: str = ""
    record_relpath: str = ".asciinema.cast"
    record_offset: int = 0


class SandboxServer:
    """
    HTTP server for tool execution inside a Nomad container.
    
    Manages multiple slots, each with an isolated workspace directory.
    Tools execute within the slot's workspace (via chdir).
    """
    
    def __init__(
        self,
        data_dir: str = "/data",
        num_slots: int = 10,
        max_output_size: int = 50000,
        max_file_size: int = 100000,
        max_artifact_size: int = 5_000_000,
        max_artifact_entries: int = 5_000,
        stateful_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.num_slots = num_slots
        self.max_output_size = max_output_size
        self.max_file_size = max_file_size
        self.max_artifact_size = max_artifact_size
        self.max_artifact_entries = max_artifact_entries

        # Directory for stateful per-slot runtime artifacts (tmux socket, temp files).
        # IMPORTANT: this MUST be on a filesystem that supports unix domain sockets.
        #
        # On macOS + Docker Desktop + Nomad, `${NOMAD_ALLOC_DIR}/data` is often backed by a
        # host-shared filesystem that rejects unix sockets (e.g. Errno 95).
        #
        # TODO(prod): confirm best default on Linux clusters (likely `${NOMAD_ALLOC_DIR}/local/atropos_stateful`)
        # and ensure it is backed by node-local disk (not NFS/CSI that may not support unix sockets).
        self._stateful_dir = self._choose_stateful_dir(stateful_dir)

        # Initialize slots
        self.slots: Dict[str, SlotInfo] = {}
        self._bwrap_available = BWRAP_AVAILABLE
        self._init_slots()
        
        # Lock per slot to prevent concurrent execution in same slot
        self.slot_locks: Dict[str, asyncio.Lock] = {
            slot_id: asyncio.Lock() for slot_id in self.slots
        }

        # Per-slot stateful session state (created lazily).
        self._stateful_tmux: Dict[str, _StatefulTmuxSession] = {}

    def _choose_stateful_dir(self, stateful_dir: Optional[str]) -> Path:
        configured = stateful_dir or os.environ.get("ATROPOS_STATEFUL_DIR")
        if configured:
            path = Path(configured)
            if not self._dir_supports_unix_sockets(path):
                raise RuntimeError(
                    f"Configured stateful_dir={path} does not support unix domain sockets; "
                    "set ATROPOS_STATEFUL_DIR to a different path."
                )
            return path

        candidates: List[Path] = []
        nomad_alloc_dir = os.environ.get("NOMAD_ALLOC_DIR")
        if nomad_alloc_dir:
            candidates.append(Path(nomad_alloc_dir) / "local" / "atropos_stateful")
        candidates.append(Path("/tmp/atropos_stateful"))

        for path in candidates:
            if self._dir_supports_unix_sockets(path):
                return path

        # Absolute last resort: try /tmp even if the probe failed (should be rare).
        fallback = Path("/tmp/atropos_stateful")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _dir_supports_unix_sockets(self, path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False

        probe_path = path / f".atropos_socket_probe_{uuid.uuid4().hex}.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(str(probe_path))
        except OSError:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass
            try:
                probe_path.unlink(missing_ok=True)
            except Exception:
                pass

        return True

    def _stateful_runtime_dir(self, slot_id: str) -> Path:
        # slot_id is always server-generated (slot_0, slot_1, ...) and not user-controlled.
        return self._stateful_dir / slot_id

    def _stateful_tmux_sock_path(self, slot_id: str, sess: _StatefulTmuxSession) -> Path:
        return self._stateful_runtime_dir(slot_id) / sess.sock_relpath

    def _stateful_tmux_target(self, sess: _StatefulTmuxSession) -> str:
        # Be explicit about pane targeting. Some tmux operations behave differently
        # when only the session name is provided and there is no attached client.
        return f"{sess.session_name}:0.0"
    
    def _init_slots(self) -> None:
        """Initialize slot workspaces."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.num_slots):
            slot_id = f"slot_{i}"
            workspace_dir = self.data_dir / slot_id
            workspace_dir.mkdir(parents=True, exist_ok=True)
            
            self.slots[slot_id] = SlotInfo(
                slot_id=slot_id,
                workspace_dir=workspace_dir,
                state=SlotState.AVAILABLE,
            )
    
    def _validate_path(self, workspace_dir: Path, path: str) -> Optional[Path]:
        """
        Validate and resolve a path within the workspace.
        Returns None if path escapes workspace.
        """
        try:
            workspace_root = workspace_dir.resolve()
            full_path = (workspace_root / path).resolve()
            if not full_path.is_relative_to(workspace_root):
                return None
            return full_path
        except Exception:
            return None

    def _mime_type_from_suffix(self, path: Path) -> str:
        suffix = path.suffix.lower()
        mapping = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".txt": "text/plain",
            ".log": "text/plain",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".md": "text/markdown",
            ".tar.gz": "application/gzip",
            ".tgz": "application/gzip",
        }
        return mapping.get(suffix, "application/octet-stream")

    def _clamp_int(self, value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            n = int(value)
        except Exception:
            return default
        return max(minimum, min(maximum, n))

    def _safe_relpath(self, workspace_dir: Path, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(workspace_dir.resolve()))
        except Exception:
            return str(path.name)

    async def artifacts_read(
        self,
        workspace_dir: Path,
        path: str,
        *,
        encoding: str = "text",
        max_bytes: Optional[int] = None,
        include_sha256: bool = False,
    ) -> web.Response:
        resolved = self._validate_path(workspace_dir, path)
        if resolved is None:
            return web.json_response({"success": False, "error": "Access denied: path outside workspace"}, status=403)
        if not resolved.exists():
            return web.json_response({"success": False, "error": f"Not found: {path}"}, status=404)
        if not resolved.is_file():
            return web.json_response({"success": False, "error": f"Not a file: {path}"}, status=400)

        enc = (encoding or "text").strip().lower()
        if enc not in {"text", "base64"}:
            return web.json_response({"success": False, "error": f"Unsupported encoding: {encoding}"}, status=400)

        limit = self.max_artifact_size
        if max_bytes is not None:
            limit = self._clamp_int(max_bytes, default=limit, minimum=1, maximum=self.max_artifact_size)

        file_size = resolved.stat().st_size
        truncated = file_size > limit

        with resolved.open("rb") as f:
            data = f.read(limit)

        sha256_hex: Optional[str] = None
        if include_sha256:
            sha256_hex = hashlib.sha256(data).hexdigest()

        if enc == "text":
            content = data.decode("utf-8", errors="replace")
        else:
            content = base64.b64encode(data).decode("ascii")

        return web.json_response(
            {
                "success": True,
                "content": content,
                "encoding": enc,
                "truncated": truncated,
                "bytes": len(data),
                "file_size": file_size,
                "path": self._safe_relpath(workspace_dir, resolved),
                "mime": self._mime_type_from_suffix(resolved),
                "sha256": sha256_hex,
            }
        )

    async def artifacts_list(
        self,
        workspace_dir: Path,
        path: str,
        *,
        recursive: bool = False,
        max_entries: Optional[int] = None,
    ) -> web.Response:
        resolved = self._validate_path(workspace_dir, path)
        if resolved is None:
            return web.json_response({"success": False, "error": "Access denied: path outside workspace"}, status=403)
        if not resolved.exists():
            return web.json_response({"success": False, "error": f"Not found: {path}"}, status=404)

        limit = self.max_artifact_entries
        if max_entries is not None:
            limit = self._clamp_int(max_entries, default=limit, minimum=1, maximum=self.max_artifact_entries)

        workspace_root = workspace_dir.resolve()

        entries: List[Dict[str, Any]] = []

        def _maybe_add(p: Path) -> None:
            if len(entries) >= limit:
                return
            try:
                rp = p.resolve()
                if not rp.is_relative_to(workspace_root):
                    return
                st = rp.stat()
                entries.append(
                    {
                        "path": str(rp.relative_to(workspace_root)),
                        "is_dir": rp.is_dir(),
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    }
                )
            except Exception:
                return

        if resolved.is_file():
            _maybe_add(resolved)
            return web.json_response({"success": True, "entries": entries, "truncated": False})

        if not resolved.is_dir():
            return web.json_response({"success": False, "error": f"Unsupported path type: {path}"}, status=400)

        if recursive:
            for root, dirnames, filenames in os.walk(resolved, followlinks=False):
                # Ensure we don't traverse out via tricky paths.
                root_path = Path(root)
                for name in sorted(dirnames):
                    _maybe_add(root_path / name)
                for name in sorted(filenames):
                    _maybe_add(root_path / name)
                if len(entries) >= limit:
                    break
        else:
            for child in sorted(resolved.iterdir()):
                _maybe_add(child)
                if len(entries) >= limit:
                    break

        truncated = len(entries) >= limit
        return web.json_response({"success": True, "entries": entries, "truncated": truncated})

    async def artifacts_archive(
        self,
        workspace_dir: Path,
        path: str,
        *,
        archive_format: str = "tar.gz",
        max_bytes: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> web.Response:
        fmt = (archive_format or "tar.gz").strip().lower()
        if fmt not in {"tar.gz", "tgz"}:
            return web.json_response({"success": False, "error": f"Unsupported archive format: {archive_format}"}, status=400)

        resolved = self._validate_path(workspace_dir, path)
        if resolved is None:
            return web.json_response({"success": False, "error": "Access denied: path outside workspace"}, status=403)
        if not resolved.exists():
            return web.json_response({"success": False, "error": f"Not found: {path}"}, status=404)

        byte_limit = self.max_artifact_size
        if max_bytes is not None:
            byte_limit = self._clamp_int(max_bytes, default=byte_limit, minimum=1, maximum=self.max_artifact_size)

        entry_limit = self.max_artifact_entries
        if max_entries is not None:
            entry_limit = self._clamp_int(max_entries, default=entry_limit, minimum=1, maximum=self.max_artifact_entries)

        workspace_root = workspace_dir.resolve()

        # Create a temp archive, then enforce size limits.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="atropos-artifacts-", suffix=".tar.gz", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            count = 0
            with tarfile.open(tmp_path, mode="w:gz") as tf:
                if resolved.is_file():
                    arcname = str(resolved.resolve().relative_to(workspace_root))
                    tf.add(resolved, arcname=arcname, recursive=False)
                    count = 1
                else:
                    for root, _dirnames, filenames in os.walk(resolved, followlinks=False):
                        root_path = Path(root)
                        for name in sorted(filenames):
                            if count >= entry_limit:
                                break
                            fp = root_path / name
                            try:
                                rp = fp.resolve()
                                if not rp.is_relative_to(workspace_root):
                                    continue
                                if not rp.is_file():
                                    continue
                                arcname = str(rp.relative_to(workspace_root))
                                tf.add(rp, arcname=arcname, recursive=False)
                                count += 1
                            except Exception:
                                continue
                        if count >= entry_limit:
                            break

            archive_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            if archive_size > byte_limit:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Archive too large: {archive_size} bytes (max {byte_limit})",
                        "bytes": archive_size,
                    },
                    status=413,
                )

            data = tmp_path.read_bytes() if tmp_path.exists() else b""
            content = base64.b64encode(data).decode("ascii")
            return web.json_response(
                {
                    "success": True,
                    "content": content,
                    "encoding": "base64",
                    "format": "tar.gz",
                    "bytes": len(data),
                    "entry_count": count,
                }
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
    
    def _build_bwrap_command(
        self,
        workspace_dir: Path,
        command: str,
        allow_network: bool = True,
        isolate_pid: bool = True,
        tmp_bind_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Build bubblewrap command for isolated execution.
        
        Creates a sandboxed environment where:
        - Only the slot's workspace is visible and writable
        - System binaries are read-only
        - PID namespace is isolated (can't see other processes)
        - User namespace isolates privileges
        """
        bwrap_args = [
            "bwrap",
            # Namespace isolation
            "--unshare-user",         # User namespace (root in sandbox)
            "--unshare-uts",          # UTS namespace (hostname)
            "--unshare-ipc",          # IPC namespace
            "--die-with-parent",      # Kill sandbox when parent dies
            
            # Filesystem: read-only system, writable workspace only
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
        ]

        if isolate_pid:
            # PID namespace isolation prevents persistent background processes.
            # For "stateful" terminal sessions (tmux), we intentionally disable this.
            bwrap_args.insert(3, "--unshare-pid")
        
        # Handle /lib and /lib64 (may be symlinks or directories)
        if Path("/lib").exists():
            if Path("/lib").is_symlink():
                bwrap_args.extend(["--symlink", os.readlink("/lib"), "/lib"])
            else:
                bwrap_args.extend(["--ro-bind", "/lib", "/lib"])
        
        if Path("/lib64").exists():
            if Path("/lib64").is_symlink():
                bwrap_args.extend(["--symlink", os.readlink("/lib64"), "/lib64"])
            else:
                bwrap_args.extend(["--ro-bind", "/lib64", "/lib64"])
        
        # /etc is needed for things like /etc/passwd, timezone, etc.
        bwrap_args.extend(["--ro-bind", "/etc", "/etc"])
        
        # Required virtual filesystems
        bwrap_args.extend(["--proc", "/proc", "--dev", "/dev"])

        if tmp_bind_dir is None:
            bwrap_args.extend(["--tmpfs", "/tmp"])
        else:
            bwrap_args.extend(["--bind", str(tmp_bind_dir.resolve()), "/tmp"])
        
        # THE KEY: Only this slot's workspace is visible and writable!
        # Map to /workspace inside the sandbox
        bwrap_args.extend([
            "--bind", str(workspace_dir.resolve()), "/workspace",
            "--chdir", "/workspace",
        ])
        
        # Network isolation (optional)
        if not allow_network:
            bwrap_args.append("--unshare-net")
        
        # Execute the command via sh
        bwrap_args.extend(["/bin/sh", "-c", command])
        
        return bwrap_args

    # ---------------------------------------------------------------------
    # Stateful tmux session helpers
    # ---------------------------------------------------------------------

    def _get_stateful_tmux(self, slot_id: str) -> _StatefulTmuxSession:
        existing = self._stateful_tmux.get(slot_id)
        if existing is not None:
            return existing
        sess = _StatefulTmuxSession()
        self._stateful_tmux[slot_id] = sess
        return sess

    async def _run_host_cmd(
        self,
        argv: List[str],
        *,
        cwd: Path,
        timeout_s: float,
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", "timeout"

        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        return int(proc.returncode or 0), stdout, stderr

    def _stateful_init_script_path(self, workspace_dir: Path) -> Path:
        return workspace_dir / ".atropos_stateful_init.sh"

    async def _ensure_stateful_tmux_session(
        self,
        *,
        slot_id: str,
        workspace_dir: Path,
        allow_network: bool,
        require_stateful_sandbox: bool = False,
        pane_width: Optional[int] = None,
        pane_height: Optional[int] = None,
    ) -> _StatefulTmuxSession:
        """
        Ensure a per-slot tmux server is running inside a long-lived bwrap sandbox.

        If bubblewrap is unavailable, may fall back to starting tmux directly in the
        container's host namespace (less isolated; intended for local/dev only).
        """
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is not installed")
        if shutil.which("asciinema") is None:
            raise RuntimeError("asciinema is not installed")
        sess = self._get_stateful_tmux(slot_id)

        if pane_width is not None:
            sess.pane_width = self._clamp_int(pane_width, default=sess.pane_width, minimum=20, maximum=500)
        if pane_height is not None:
            sess.pane_height = self._clamp_int(pane_height, default=sess.pane_height, minimum=10, maximum=200)

        # Restart if network policy changed.
        if sess.bwrap_proc is not None and sess.allow_network != bool(allow_network):
            await self._stop_stateful_tmux(slot_id)

        sess.allow_network = bool(allow_network)

        runtime_dir = self._stateful_runtime_dir(slot_id)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        sock_path = self._stateful_tmux_sock_path(slot_id, sess)
        record_path = workspace_dir / sess.record_relpath

        # In bwrap mode, the presence of a live bwrap process is the source of truth.
        if self._bwrap_available:
            if sess.bwrap_proc is not None and sess.bwrap_proc.returncode is None:
                return sess
        else:
            # In host fallback mode, check whether the tmux session already exists.
            if require_stateful_sandbox:
                raise RuntimeError("Stateful sandboxing required but bubblewrap is unavailable in this environment")
            if not allow_network:
                raise RuntimeError("Network isolation requested but bubblewrap is unavailable in this environment")
            if sock_path.exists():
                rc, _out, _err = await self._run_host_cmd(
                    ["tmux", "-S", str(sock_path), "has-session", "-t", sess.session_name],
                    cwd=workspace_dir,
                    timeout_s=1.0,
                )
                if rc == 0:
                    return sess

        # Cleanup any stale socket/recording file.
        try:
            sock_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            record_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Best-effort: kill any tmux server that might still be attached to this socket.
        try:
            await self._run_host_cmd(["tmux", "-S", str(sock_path), "kill-server"], cwd=workspace_dir, timeout_s=2.0)
        except Exception:
            pass

        # Start sandboxed tmux server.
        if self._bwrap_available:
            init_script = self._stateful_init_script_path(workspace_dir)
            init_script.write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        "set -eu",
                        "export TERM=xterm-256color",
                        "export SHELL=/bin/bash",
                        f"sock=/tmp/{sess.sock_relpath}",
                        f"record=/workspace/{sess.record_relpath}",
                        f"session={sess.session_name!s}",
                        "rm -f \"$sock\"",
                        # Start a shell in a new tmux session.
                        # NOTE: Use `script` to allocate a PTY so tmux behaves consistently
                        # even when started from a non-interactive background process.
                        f"script -qc \"tmux -S \\\"$sock\\\" new-session -d -s \\\"$session\\\" -x {sess.pane_width} -y {sess.pane_height} asciinema rec --overwrite -c sh \\\"$record\\\"\" /dev/null",
                        # Large but bounded history for capture-pane -S -
                        "tmux -S \"$sock\" set-option -g history-limit 20000 >/dev/null 2>&1 || true",
                        # Keep PID namespace alive so background processes persist.
                        "exec tail -f /dev/null",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            init_script.chmod(0o700)

            bwrap_cmd = self._build_bwrap_command(
                workspace_dir,
                "/workspace/.atropos_stateful_init.sh",
                allow_network=bool(allow_network),
                isolate_pid=True,
                tmp_bind_dir=runtime_dir,
            )

            proc = await asyncio.create_subprocess_exec(
                *bwrap_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            sess.bwrap_proc = proc
        else:
            # Dev fallback: start tmux in host namespace.
            rc, _out, _err = await self._run_host_cmd(
                [
                    "tmux",
                    "-S",
                    str(sock_path),
                    "new-session",
                    "-d",
                    "-s",
                    sess.session_name,
                    "-x",
                    str(sess.pane_width),
                    "-y",
                    str(sess.pane_height),
                    "asciinema",
                    "rec",
                    "--overwrite",
                    "-c",
                    "sh",
                    str(record_path),
                ],
                cwd=workspace_dir,
                timeout_s=5.0,
            )
            if rc != 0:
                raise RuntimeError("Failed to start tmux (host fallback)")
            sess.bwrap_proc = None

        # Wait for the socket to appear.
        for _ in range(200):  # ~10s
            if sock_path.exists():
                break
            if sess.bwrap_proc is not None and sess.bwrap_proc.returncode is not None:
                raise RuntimeError("Stateful bwrap process exited during tmux startup")
            await asyncio.sleep(0.05)
        else:
            await self._stop_stateful_tmux(slot_id)
            raise RuntimeError("Timed out waiting for tmux socket to appear")

        # Wait for the recorded shell to be ready. We use the asciinema .cast file
        # as a simple readiness signal because `tmux new-session -d` can return before
        # the pane's command is actually accepting input, causing early `send-keys`
        # (especially blocking ones) to hang.
        for _ in range(200):  # ~10s
            try:
                if record_path.exists() and record_path.stat().st_size > 0:
                    # Header is a single JSON object line; once we see any event line
                    # (starts with '[') we consider the shell "ready enough".
                    head = record_path.read_text(encoding="utf-8", errors="ignore")[:4096]
                    if "\n[" in head:
                        break
            except Exception:
                pass
            await asyncio.sleep(0.05)

        sess.prev_capture = ""
        return sess

    async def _stop_stateful_tmux(self, slot_id: str) -> None:
        sess = self._stateful_tmux.get(slot_id)
        if sess is None:
            return

        # Best-effort: ask tmux to stop (works for both bwrap-backed and host fallback sessions).
        try:
            slot = self.slots.get(slot_id)
            if slot is not None:
                sock_path = self._stateful_tmux_sock_path(slot_id, sess)
                await self._run_host_cmd(
                    ["tmux", "-S", str(sock_path), "kill-server"],
                    cwd=slot.workspace_dir,
                    timeout_s=2.0,
                )
        except Exception:
            pass

        proc = sess.bwrap_proc
        sess.bwrap_proc = None
        sess.prev_capture = ""

        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except Exception:
                    pass

        # Best-effort cleanup of the per-slot runtime dir (socket/temp files).
        try:
            runtime_dir = self._stateful_runtime_dir(slot_id)
            # Safety: never delete outside our configured base dir.
            if runtime_dir.resolve().is_relative_to(self._stateful_dir.resolve()):
                shutil.rmtree(runtime_dir, ignore_errors=True)
        except Exception:
            pass
    
    async def execute_bash_sandboxed(
        self,
        workspace_dir: Path,
        command: str,
        timeout: float = 30.0,
        allow_network: bool = True,
        isolate_pid: bool = True,
    ) -> ExecuteResponse:
        """
        Execute a bash command in an isolated namespace via bubblewrap.
        
        Provides:
        - Filesystem isolation: Only /workspace (slot's dir) is writable
        - PID isolation: Can't see other processes
        - User isolation: Runs as root in sandbox but unprivileged outside
        """
        try:
            bwrap_cmd = self._build_bwrap_command(
                workspace_dir,
                command,
                allow_network,
                isolate_pid=isolate_pid,
            )
            
            process = await asyncio.create_subprocess_exec(
                *bwrap_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecuteResponse(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                    metadata={"exit_code": -1, "timeout": True, "sandboxed": True},
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Truncate if too long
            if len(stdout_str) > self.max_output_size:
                stdout_str = stdout_str[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr_str) > self.max_output_size:
                stderr_str = stderr_str[:self.max_output_size] + "\n... (output truncated)"
            
            exit_code = process.returncode
            success = exit_code == 0
            
            output = stdout_str
            if stderr_str:
                output = f"{stdout_str}\n[stderr]\n{stderr_str}" if stdout_str else stderr_str
            
            return ExecuteResponse(
                success=success,
                output=output.strip(),
                error="" if success else f"Exit code: {exit_code}",
                metadata={
                    "exit_code": exit_code,
                    "sandboxed": True,
                    "network_isolated": not allow_network,
                },
            )
            
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to execute sandboxed command: {str(e)}",
                metadata={"sandboxed": True},
            )
    
    async def execute_bash_unsandboxed(
        self, 
        workspace_dir: Path, 
        command: str, 
        timeout: float = 30.0
    ) -> ExecuteResponse:
        """Execute a bash command without sandboxing (development/fallback mode)."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(workspace_dir),
                start_new_session=True,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill the whole process group so shell children don't keep pipes open.
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        process.kill()
                else:
                    process.kill()
                await process.wait()
                return ExecuteResponse(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                    metadata={
                        "exit_code": -1,
                        "timeout": True,
                        "sandboxed": False,
                        "network_isolated": False,
                    },
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            # Truncate if too long
            if len(stdout_str) > self.max_output_size:
                stdout_str = stdout_str[:self.max_output_size] + "\n... (output truncated)"
            if len(stderr_str) > self.max_output_size:
                stderr_str = stderr_str[:self.max_output_size] + "\n... (output truncated)"
            
            exit_code = process.returncode
            success = exit_code == 0
            
            output = stdout_str
            if stderr_str:
                output = f"{stdout_str}\n[stderr]\n{stderr_str}" if stdout_str else stderr_str
            
            return ExecuteResponse(
                success=success,
                output=output.strip(),
                error="" if success else f"Exit code: {exit_code}",
                metadata={
                    "exit_code": exit_code,
                    "sandboxed": False,
                    "network_isolated": False,
                },
            )
            
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to execute command: {str(e)}",
                metadata={"sandboxed": False},
            )
    
    async def execute_bash(
        self, 
        workspace_dir: Path, 
        command: str, 
        timeout: float = 30.0,
        allow_network: bool = True,
        require_sandbox: bool = False,
    ) -> ExecuteResponse:
        """
        Execute a bash command in the workspace.
        
        Uses bubblewrap for sandboxed execution if available,
        otherwise falls back to unsandboxed execution.
        """
        if self._bwrap_available:
            result = await self.execute_bash_sandboxed(
                workspace_dir,
                command,
                timeout,
                allow_network=allow_network,
            )
            if result.success:
                return result

            # Bubblewrap can be installed but unusable in some container runtimes.
            # If we detect a namespace failure, disable bwrap for the remainder of
            # this server lifetime and fall back to unsandboxed execution.
            output = f"{result.output}\n{result.error}".lower()
            if "no permissions to create new namespace" in output:
                self._bwrap_available = False
                if require_sandbox:
                    return ExecuteResponse(
                        success=False,
                        error="Sandboxing required but bubblewrap is unavailable in this environment",
                        metadata={"sandboxed": False, "network_isolated": False, "requires_bwrap": True},
                    )
                if not allow_network:
                    return ExecuteResponse(
                        success=False,
                        error="Network isolation requested but bubblewrap is unavailable in this environment",
                        metadata={"sandboxed": False, "network_isolated": False, "requires_bwrap": True},
                    )

                fallback = await self.execute_bash_unsandboxed(workspace_dir, command, timeout)
                fallback.metadata = dict(fallback.metadata)
                fallback.metadata["sandbox_fallback"] = True
                return fallback

            return result

        if require_sandbox:
            return ExecuteResponse(
                success=False,
                error="Sandboxing required but bubblewrap is unavailable in this environment",
                metadata={"sandboxed": False, "network_isolated": False, "requires_bwrap": True},
            )

        if not allow_network:
            return ExecuteResponse(
                success=False,
                error="Network isolation requested but bubblewrap is unavailable in this environment",
                metadata={"sandboxed": False, "network_isolated": False, "requires_bwrap": True},
            )

        return await self.execute_bash_unsandboxed(workspace_dir, command, timeout)

    async def execute_bash_stateful(
        self,
        slot_id: str,
        workspace_dir: Path,
        command: str,
        timeout: float = 30.0,
        allow_network: bool = True,
        require_stateful_sandbox: bool = False,
    ) -> ExecuteResponse:
        """
        Execute a command in a per-slot stateful terminal session.

        Implementation:
        - Start (lazily) a long-lived bwrap sandbox per slot with a tmux server inside.
        - Send the command into a tmux-backed shell and wait for completion.
        - Capture the pane output and return a diff vs the previous capture.

        This provides both:
        - persistent process state across tool calls (good for TUIs / background tasks)
        - PID namespace isolation (no global process visibility across slots)
        """
        command = (command or "").strip()
        if not command:
            return ExecuteResponse(success=False, error="Missing command")

        # Ensure the per-slot stateful sandbox exists.
        try:
            sess = await self._ensure_stateful_tmux_session(
                slot_id=slot_id,
                workspace_dir=workspace_dir,
                allow_network=allow_network,
                require_stateful_sandbox=require_stateful_sandbox,
            )
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to start stateful session: {e}",
                metadata={"stateful": True, "requires_bwrap": bool(require_stateful_sandbox)},
            )

        sock_path = self._stateful_tmux_sock_path(slot_id, sess)

        marker = uuid.uuid4().hex
        wait_name = f"done_{marker}"
        exit_marker = f"__ATROPOS_EXIT_{marker}__"

        # Append a unique completion signal and exit code marker.
        # NOTE: This assumes we're at a shell prompt (not a full-screen TUI).
        cmd_to_send = f"{command}; echo {exit_marker}:$?; tmux wait-for -S {wait_name}"

        # Send command
        rc, _out, err = await self._run_host_cmd(
            [
                "tmux",
                "-S",
                str(sock_path),
                "send-keys",
                "-t",
                self._stateful_tmux_target(sess),
                cmd_to_send,
                "Enter",
            ],
            cwd=workspace_dir,
            timeout_s=5.0,
        )
        if rc != 0:
            # Try restarting the session once (e.g. user killed tmux server).
            try:
                await self._stop_stateful_tmux(slot_id)
                sess = await self._ensure_stateful_tmux_session(
                    slot_id=slot_id,
                    workspace_dir=workspace_dir,
                    allow_network=allow_network,
                    require_stateful_sandbox=require_stateful_sandbox,
                )
                sock_path = self._stateful_tmux_sock_path(slot_id, sess)
                rc, _out, err = await self._run_host_cmd(
                    [
                        "tmux",
                        "-S",
                        str(sock_path),
                        "send-keys",
                        "-t",
                        self._stateful_tmux_target(sess),
                        cmd_to_send,
                        "Enter",
                    ],
                    cwd=workspace_dir,
                    timeout_s=5.0,
                )
            except Exception:
                pass
        if rc != 0:
            return ExecuteResponse(success=False, error=f"tmux send-keys failed: {err.strip()}")

        # Wait for completion.
        rc, _out, err = await self._run_host_cmd(
            ["tmux", "-S", str(sock_path), "wait-for", wait_name],
            cwd=workspace_dir,
            timeout_s=float(timeout),
        )
        if rc != 0:
            # Timeout or failure: tear down the stateful sandbox to avoid leaking runaway processes.
            await self._stop_stateful_tmux(slot_id)
            return ExecuteResponse(
                success=False,
                error=f"Command timed out after {timeout}s",
                metadata={"exit_code": -1, "timeout": True, "sandboxed": True, "stateful": True},
            )

        # Capture output.
        rc, capture, err = await self._run_host_cmd(
            ["tmux", "-S", str(sock_path), "capture-pane", "-p", "-S", "-", "-t", self._stateful_tmux_target(sess)],
            cwd=workspace_dir,
            timeout_s=10.0,
        )
        if rc != 0:
            return ExecuteResponse(success=False, error=f"tmux capture-pane failed: {err.strip()}")

        # Diff against previous capture for this slot.
        prev = sess.prev_capture or ""
        if prev and capture.startswith(prev):
            delta = capture[len(prev) :]
        else:
            delta = capture
        sess.prev_capture = capture

        # Extract exit code from the capture.
        exit_code = 0
        marker_idx = capture.rfind(exit_marker)
        if marker_idx != -1:
            line = capture[marker_idx:].splitlines()[0]
            try:
                exit_code = int(line.split(":", 1)[1].strip())
            except Exception:
                exit_code = 0

        # Remove the marker line from delta.
        cleaned_lines = []
        for line in delta.splitlines():
            if exit_marker in line:
                continue
            cleaned_lines.append(line)
        output = "\n".join(cleaned_lines).strip()

        if len(output) > self.max_output_size:
            output = output[: self.max_output_size] + "\n... (output truncated)"

        success = exit_code == 0
        return ExecuteResponse(
            success=success,
            output=output,
            error="" if success else f"Exit code: {exit_code}",
            metadata={
                "exit_code": exit_code,
                "sandboxed": bool(self._bwrap_available),
                "network_isolated": not allow_network,
                "stateful": True,
            },
        )

    async def execute_tmux(
        self,
        slot_id: str,
        workspace_dir: Path,
        args: Dict[str, Any],
        timeout: float,
        allow_network: bool,
        require_stateful_sandbox: bool = False,
    ) -> ExecuteResponse:
        """
        Direct tmux session control for TUI-style terminal interactions.

        Supported actions:
        - start: ensure session exists (optionally sets pane size)
        - send_keys: send keys into the session (optionally block using tmux wait-for)
        - capture: capture the current pane contents (or entire history)
        - stop: stop the session and tear down the stateful sandbox
        """
        action = str(args.get("action") or "capture").strip().lower()

        # Always resolve session state (creates default entry).
        sess = self._get_stateful_tmux(slot_id)
        sock_path = self._stateful_tmux_sock_path(slot_id, sess)

        if action == "stop":
            try:
                await self._run_host_cmd(["tmux", "-S", str(sock_path), "kill-server"], cwd=workspace_dir, timeout_s=2.0)
            except Exception:
                pass
            await self._stop_stateful_tmux(slot_id)
            try:
                sock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._stateful_tmux.pop(slot_id, None)
            return ExecuteResponse(success=True, output="stopped", metadata={"stateful": True})

        # start/send/capture: ensure session exists
        try:
            await self._ensure_stateful_tmux_session(
                slot_id=slot_id,
                workspace_dir=workspace_dir,
                allow_network=allow_network,
                require_stateful_sandbox=require_stateful_sandbox,
                pane_width=args.get("pane_width"),
                pane_height=args.get("pane_height"),
            )
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to ensure tmux session: {e}",
                metadata={"stateful": True, "requires_bwrap": bool(require_stateful_sandbox)},
            )

        if action == "start":
            sess.record_offset = 0
            return ExecuteResponse(
                success=True,
                output="started",
                metadata={
                    "stateful": True,
                    "pane_width": sess.pane_width,
                    "pane_height": sess.pane_height,
                    "socket": sess.sock_relpath,
                    "session": sess.session_name,
                    "recording": sess.record_relpath,
                },
            )

        if action == "send_keys":
            keys = args.get("keys")
            if isinstance(keys, str):
                key_list = [keys]
            elif isinstance(keys, list) and all(isinstance(k, str) for k in keys):
                key_list = list(keys)
            else:
                return ExecuteResponse(success=False, error="tmux send_keys requires 'keys' as string or list[string]")

            block = bool(args.get("block", False))
            min_wait_s = float(args.get("min_wait_s", 0.0) or 0.0)
            max_wait_s = float(args.get("max_wait_s", timeout) or timeout)

            wait_name = None
            if block:
                # Blocking mode assumes we're at a shell prompt (not a full-screen TUI).
                marker = uuid.uuid4().hex
                wait_name = f"keys_{marker}"
                if key_list:
                    key_list[-1] = key_list[-1] + f"; tmux wait-for -S {wait_name}"
                else:
                    key_list = [f"tmux wait-for -S {wait_name}"]
                if not key_list or key_list[-1] not in {"Enter", "C-m", "KPEnter"}:
                    key_list.append("Enter")

            rc, _out, err = await self._run_host_cmd(
                ["tmux", "-S", str(sock_path), "send-keys", "-t", self._stateful_tmux_target(sess), *key_list],
                cwd=workspace_dir,
                timeout_s=5.0,
            )
            if rc != 0:
                return ExecuteResponse(success=False, error=f"tmux send-keys failed: {err.strip()}")

            if block and wait_name:
                rc, _out, err = await self._run_host_cmd(
                    ["tmux", "-S", str(sock_path), "wait-for", wait_name],
                    cwd=workspace_dir,
                    timeout_s=max_wait_s,
                )
                if rc != 0:
                    await self._stop_stateful_tmux(slot_id)
                    return ExecuteResponse(success=False, error=f"tmux blocked keys timed out after {max_wait_s}s")
            elif min_wait_s > 0:
                await asyncio.sleep(min_wait_s)

            return ExecuteResponse(success=True, output="ok", metadata={"stateful": True, "blocked": block})

        if action == "capture":
            capture_entire = bool(args.get("capture_entire", False))
            tmux_args: List[str] = ["tmux", "-S", str(sock_path), "capture-pane", "-p"]
            if capture_entire:
                tmux_args.extend(["-S", "-"])
            tmux_args.extend(["-t", self._stateful_tmux_target(sess)])

            rc, out, err = await self._run_host_cmd(
                tmux_args,
                cwd=workspace_dir,
                timeout_s=min(10.0, max(1.0, float(timeout or 10.0))),
            )
            if rc != 0:
                return ExecuteResponse(success=False, error=f"tmux capture-pane failed: {err.strip()}")

            if len(out) > self.max_output_size:
                out = out[: self.max_output_size] + "\n... (output truncated)"
            return ExecuteResponse(
                success=True,
                output=out,
                metadata={"stateful": True, "capture_entire": capture_entire},
            )

        if action == "stream":
            # Streaming: return asciinema .cast lines since last offset.
            record_path = workspace_dir / sess.record_relpath
            if not record_path.exists():
                return ExecuteResponse(
                    success=True,
                    output="",
                    metadata={"stateful": True, "stream": True, "offset": sess.record_offset, "recording": sess.record_relpath},
                )

            if bool(args.get("reset", False)):
                sess.record_offset = 0

            max_bytes = self._clamp_int(
                args.get("max_bytes", self.max_output_size),
                default=self.max_output_size,
                minimum=1,
                maximum=self.max_output_size * 4,
            )

            try:
                with record_path.open("rb") as f:
                    f.seek(sess.record_offset)
                    data = f.read(max_bytes)
            except Exception as e:
                return ExecuteResponse(success=False, error=f"Failed to read asciinema recording: {e}")

            if not data:
                return ExecuteResponse(
                    success=True,
                    output="",
                    metadata={"stateful": True, "stream": True, "offset": sess.record_offset, "recording": sess.record_relpath},
                )

            # Avoid returning partial lines.
            last_nl = data.rfind(b"\n")
            if last_nl == -1:
                return ExecuteResponse(
                    success=True,
                    output="",
                    metadata={"stateful": True, "stream": True, "offset": sess.record_offset, "recording": sess.record_relpath},
                )
            chunk = data[: last_nl + 1]
            sess.record_offset += len(chunk)

            out = chunk.decode("utf-8", errors="replace")
            return ExecuteResponse(
                success=True,
                output=out,
                metadata={"stateful": True, "stream": True, "offset": sess.record_offset, "recording": sess.record_relpath},
            )

        return ExecuteResponse(success=False, error=f"Unknown tmux action: {action}")
    
    async def execute_read_file(
        self, 
        workspace_dir: Path, 
        path: str
    ) -> ExecuteResponse:
        """Read a file from the workspace."""
        try:
            file_path = self._validate_path(workspace_dir, path)
            if file_path is None:
                return ExecuteResponse(
                    success=False,
                    error="Access denied: path outside workspace",
                )
            
            if not file_path.exists():
                return ExecuteResponse(
                    success=False,
                    error=f"File not found: {path}",
                )
            
            if not file_path.is_file():
                return ExecuteResponse(
                    success=False,
                    error=f"Not a file: {path}",
                )
            
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return ExecuteResponse(
                    success=False,
                    error=f"File too large: {size} bytes (max {self.max_file_size})",
                )
            
            content = file_path.read_text(encoding="utf-8", errors="replace")
            
            return ExecuteResponse(
                success=True,
                output=content,
                metadata={"path": str(file_path), "size": size},
            )
            
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to read file: {str(e)}",
            )
    
    async def execute_write_file(
        self, 
        workspace_dir: Path, 
        path: str, 
        content: str
    ) -> ExecuteResponse:
        """Write content to a file in the workspace."""
        try:
            if len(content) > self.max_file_size:
                return ExecuteResponse(
                    success=False,
                    error=f"Content too large: {len(content)} bytes (max {self.max_file_size})",
                )
            
            file_path = self._validate_path(workspace_dir, path)
            if file_path is None:
                return ExecuteResponse(
                    success=False,
                    error="Access denied: path outside workspace",
                )
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_text(content, encoding="utf-8")
            
            return ExecuteResponse(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {path}",
                metadata={"path": str(file_path), "size": len(content)},
            )
            
        except Exception as e:
            return ExecuteResponse(
                success=False,
                error=f"Failed to write file: {str(e)}",
            )
    
    async def execute_tool(self, request: ExecuteRequest) -> ExecuteResponse:
        """Execute a tool in a slot's workspace."""
        # Validate slot
        if request.slot_id not in self.slots:
            return ExecuteResponse(
                success=False,
                error=f"Unknown slot: {request.slot_id}",
                execution_id=request.execution_id,
            )
        
        slot = self.slots[request.slot_id]
        
        # Acquire slot lock
        async with self.slot_locks[request.slot_id]:
            slot.state = SlotState.EXECUTING
            slot.current_execution_id = request.execution_id
            
            try:
                # Route to appropriate tool
                if request.tool == "bash":
                    command = request.args.get("command", "")
                    allow_network = bool(request.args.get("allow_network", True))
                    require_sandbox = bool(request.args.get("require_sandbox", False))
                    result = await self.execute_bash(
                        slot.workspace_dir, 
                        command, 
                        request.timeout,
                        allow_network=allow_network,
                        require_sandbox=require_sandbox,
                    )
                elif request.tool == "bash_stateful":
                    command = request.args.get("command", "")
                    allow_network = bool(request.args.get("allow_network", True))
                    require_stateful_sandbox = bool(
                        request.args.get("require_stateful_sandbox", request.args.get("require_sandbox", False))
                    )
                    result = await self.execute_bash_stateful(
                        request.slot_id,
                        slot.workspace_dir,
                        command,
                        request.timeout,
                        allow_network=allow_network,
                        require_stateful_sandbox=require_stateful_sandbox,
                    )
                elif request.tool == "read_file":
                    path = request.args.get("path", "")
                    result = await self.execute_read_file(slot.workspace_dir, path)
                elif request.tool == "write_file":
                    path = request.args.get("path", "")
                    content = request.args.get("content", "")
                    result = await self.execute_write_file(slot.workspace_dir, path, content)
                elif request.tool == "tmux":
                    allow_network = bool(request.args.get("allow_network", True))
                    require_stateful_sandbox = bool(
                        request.args.get("require_stateful_sandbox", request.args.get("require_sandbox", False))
                    )
                    result = await self.execute_tmux(
                        request.slot_id,
                        slot.workspace_dir,
                        request.args,
                        request.timeout,
                        allow_network,
                        require_stateful_sandbox=require_stateful_sandbox,
                    )
                else:
                    result = ExecuteResponse(
                        success=False,
                        error=f"Unknown tool: {request.tool}",
                    )
                
                result.execution_id = request.execution_id
                return result
                
            finally:
                slot.state = SlotState.AVAILABLE
                slot.current_execution_id = None
    
    async def reset_slot(self, slot_id: str) -> ExecuteResponse:
        """Reset a slot's workspace (delete all files)."""
        if slot_id not in self.slots:
            return ExecuteResponse(
                success=False,
                error=f"Unknown slot: {slot_id}",
            )
        
        slot = self.slots[slot_id]
        
        async with self.slot_locks[slot_id]:
            try:
                # Stop any long-lived per-slot stateful sandbox process.
                try:
                    await self._stop_stateful_tmux(slot_id)
                except Exception:
                    pass
                self._stateful_tmux.pop(slot_id, None)

                # Best-effort cleanup of stateful runtime dir (socket/temp files),
                # even if the session was never registered (e.g. crash mid-start).
                try:
                    runtime_dir = self._stateful_runtime_dir(slot_id)
                    if runtime_dir.resolve().is_relative_to(self._stateful_dir.resolve()):
                        shutil.rmtree(runtime_dir, ignore_errors=True)
                except Exception:
                    pass

                # Remove all contents but keep the directory
                for item in slot.workspace_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                
                return ExecuteResponse(
                    success=True,
                    output=f"Reset workspace for {slot_id}",
                )
            except Exception as e:
                return ExecuteResponse(
                    success=False,
                    error=f"Failed to reset workspace: {str(e)}",
                )
    
    # HTTP Handlers
    
    async def handle_execute(self, request: web.Request) -> web.Response:
        """Handle POST /execute."""
        try:
            data = await request.json()
            
            exec_request = ExecuteRequest(
                slot_id=data.get("slot_id", ""),
                tool=data.get("tool", ""),
                args=data.get("args", {}),
                execution_id=data.get("execution_id"),
                timeout=data.get("timeout", 30.0),
            )
            
            result = await self.execute_tool(exec_request)
            return web.json_response(result.to_dict())
            
        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400,
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )
    
    async def handle_batch(self, request: web.Request) -> web.Response:
        """Handle POST /batch - execute multiple tools in parallel."""
        try:
            data = await request.json()
            
            if not isinstance(data, list):
                return web.json_response(
                    {"success": False, "error": "Expected array of requests"},
                    status=400,
                )
            
            # Create execution requests
            exec_requests = [
                ExecuteRequest(
                    slot_id=item.get("slot_id", ""),
                    tool=item.get("tool", ""),
                    args=item.get("args", {}),
                    execution_id=item.get("execution_id"),
                    timeout=item.get("timeout", 30.0),
                )
                for item in data
            ]
            
            # Execute in parallel
            results = await asyncio.gather(
                *[self.execute_tool(req) for req in exec_requests],
                return_exceptions=True,
            )
            
            # Convert results
            response_data = []
            for result in results:
                if isinstance(result, BaseException):
                    response_data.append({
                        "success": False,
                        "error": str(result),
                    })
                elif isinstance(result, ExecuteResponse):
                    response_data.append(result.to_dict())
                else:
                    response_data.append({
                        "success": False,
                        "error": f"Unexpected result type: {type(result)}",
                    })
            
            return web.json_response(response_data)
            
        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400,
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health."""
        available = sum(
            1 for slot in self.slots.values() 
            if slot.state == SlotState.AVAILABLE
        )
        executing = sum(
            1 for slot in self.slots.values() 
            if slot.state == SlotState.EXECUTING
        )
        
        return web.json_response({
            "status": "ok",
            "slots": self.num_slots,
            "available": available,
            "executing": executing,
            "data_dir": str(self.data_dir),
            "bwrap_available": bool(self._bwrap_available),
            "stateful_dir": str(self._stateful_dir),
        })
    
    async def handle_reset(self, request: web.Request) -> web.Response:
        """Handle POST /reset - reset a slot's workspace."""
        try:
            data = await request.json()
            slot_id = data.get("slot_id", "")
            
            result = await self.reset_slot(slot_id)
            return web.json_response(result.to_dict())
            
        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400,
            )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )
    
    async def handle_list_slots(self, request: web.Request) -> web.Response:
        """Handle GET /slots - list all slots and their status."""
        slots_info = []
        for slot_id, slot in self.slots.items():
            slots_info.append({
                "slot_id": slot.slot_id,
                "state": slot.state.value,
                "workspace_dir": str(slot.workspace_dir),
                "current_execution_id": slot.current_execution_id,
            })
        
        return web.json_response({"slots": slots_info})

    async def handle_artifacts_read(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        slot_id = data.get("slot_id", "")
        if slot_id not in self.slots:
            return web.json_response({"success": False, "error": f"Unknown slot: {slot_id}"}, status=404)

        path = data.get("path", "")
        encoding = data.get("encoding", "text")
        max_bytes = data.get("max_bytes")
        include_sha256 = bool(data.get("include_sha256", False))

        slot = self.slots[slot_id]
        async with self.slot_locks[slot_id]:
            return await self.artifacts_read(
                slot.workspace_dir,
                path,
                encoding=encoding,
                max_bytes=max_bytes,
                include_sha256=include_sha256,
            )

    async def handle_artifacts_list(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        slot_id = data.get("slot_id", "")
        if slot_id not in self.slots:
            return web.json_response({"success": False, "error": f"Unknown slot: {slot_id}"}, status=404)

        path = data.get("path", ".")
        recursive = bool(data.get("recursive", False))
        max_entries = data.get("max_entries")

        slot = self.slots[slot_id]
        async with self.slot_locks[slot_id]:
            return await self.artifacts_list(slot.workspace_dir, path, recursive=recursive, max_entries=max_entries)

    async def handle_artifacts_archive(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        slot_id = data.get("slot_id", "")
        if slot_id not in self.slots:
            return web.json_response({"success": False, "error": f"Unknown slot: {slot_id}"}, status=404)

        path = data.get("path", ".")
        archive_format = data.get("format", "tar.gz")
        max_bytes = data.get("max_bytes")
        max_entries = data.get("max_entries")

        slot = self.slots[slot_id]
        async with self.slot_locks[slot_id]:
            return await self.artifacts_archive(
                slot.workspace_dir,
                path,
                archive_format=archive_format,
                max_bytes=max_bytes,
                max_entries=max_entries,
            )
    
    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()
        
        app.router.add_post("/execute", self.handle_execute)
        app.router.add_post("/batch", self.handle_batch)
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/reset", self.handle_reset)
        app.router.add_get("/slots", self.handle_list_slots)
        app.router.add_post("/artifacts/read", self.handle_artifacts_read)
        app.router.add_post("/artifacts/list", self.handle_artifacts_list)
        app.router.add_post("/artifacts/archive", self.handle_artifacts_archive)
        
        return app


def main():
    """Run the sandbox server."""
    parser = argparse.ArgumentParser(description="Sandbox Server for Nomad containers")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--slots", type=int, default=10, help="Number of slots")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=os.environ.get("NOMAD_ALLOC_DIR", "/data"),
        help="Base directory for slot workspaces"
    )
    
    args = parser.parse_args()
    
    # If in Nomad, use alloc dir
    data_dir = args.data_dir
    if os.environ.get("NOMAD_ALLOC_DIR"):
        data_dir = os.path.join(os.environ["NOMAD_ALLOC_DIR"], "data")
    
    print(f"Starting Sandbox Server on {args.host}:{args.port}")
    print(f"Data directory: {data_dir}")
    print(f"Number of slots: {args.slots}")
    
    server = SandboxServer(
        data_dir=data_dir,
        num_slots=args.slots,
    )
    
    app = server.create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
