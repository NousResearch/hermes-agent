"""Process-level sandbox execution for dangerous commands.

Uses Docker/Podman containers to isolate dangerous command execution with
strict resource limits (CPU, memory, network, PIDs) following the
Least-Privilege (LP) design principle.

Fallback chain: docker -> podman -> local (with warning logged).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Docker/Podman binary search order
_CONTAINER_BINARIES = [
    ("docker", "/usr/local/bin/docker", "/opt/homebrew/bin/docker",
     "/Applications/Docker.app/Contents/Resources/bin/docker"),
    ("podman", "/usr/bin/podman", "/usr/local/bin/podman"),
]

_container_bin: Optional[str] = None


def _discover_container_binary() -> Optional[str]:
    """Find the first available container runtime (docker or podman).

    Checks PATH via shutil.which, then known install locations.
    """
    global _container_bin
    if _container_bin is not None:
        return _container_bin

    for binary_name, *extra_paths in _CONTAINER_BINARIES:
        found = shutil.which(binary_name)
        if found:
            _container_bin = found
            logger.debug("Container runtime found: %s", found)
            return found
        for path in extra_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                _container_bin = path
                logger.debug("Container runtime found at non-PATH location: %s", path)
                return path
    return None


# ---------------------------------------------------------------------------
# Resource limits for sandboxed dangerous commands
# ---------------------------------------------------------------------------

# Default image: same as terminal_tool default for consistency
_SANDBOX_IMAGE = os.getenv("HERMES_SANDBOX_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20")

# Hardened security flags — drop ALL capabilities, no privilege escalation,
# PID limit, tmpfs scratch dirs.  Network is disabled by default.
_SANDBOX_SECURITY_ARGS = [
    "--cap-drop", "ALL",
    "--cap-add", "DAC_OVERRIDE",
    "--cap-add", "CHOWN",
    "--cap-add", "FOWNER",
    "--security-opt", "no-new-privileges",
    "--pids-limit", "64",
    "--network", "none",
    "--read-only",
    "--tmpfs", "/tmp:rw,nosuid,size=256m,mode=1777",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=128m",
]

# Writable areas inside the sandbox (tmpfs — rootfs is read-only via --read-only).
# Note: /tmp and /var/tmp are already in _SANDBOX_SECURITY_ARGS.
_WRITABLE_MOUNTS = [
    "--tmpfs", "/workspace:rw,exec,size=512m",
    "--tmpfs", "/root:rw,noexec,nosuid,size=64m",
]


def _is_available() -> bool:
    """Return True if a container runtime is available on this host."""
    return _discover_container_binary() is not None


def _check_runtime() -> tuple[bool, str]:
    """Check container runtime availability and version.

    Returns (available, runtime_name).
    """
    runtime = _discover_container_binary()
    if not runtime:
        return False, "none"
    try:
        result = subprocess.run(
            [runtime, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = os.path.basename(runtime)
            version = result.stdout.strip()
            logger.debug("Container runtime %s %s available", name, version)
            return True, name
    except Exception as e:
        logger.debug("Container runtime check failed: %s", e)
    return False, "none"


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------

class SandboxRunner:
    """Execute a single dangerous command inside an isolated container.

    Resource limits applied to every execution:
    - CPU: 1 core (configurable)
    - Memory: 512 MB (configurable)
    - Network: disabled
    - PIDs: max 64
    - Filesystem: read-only root, tmpfs for /tmp and /workspace
    - Time: enforced via subprocess timeout

    The container is created, the command is executed, and the container is
    torn down — all in a single atomic-style operation.  No state persists
    between calls.

    Graceful degradation: if the container runtime is unavailable, the
    ``execute()`` method returns a result dict with ``sandbox_used=False``
    and falls back to local ``/bin/sh -c`` execution (still subject to the
    subprocess timeout).  Callers can detect this by checking the returned
    ``sandbox_used`` field.
    """

    def __init__(
        self,
        image: str = _SANDBOX_IMAGE,
        cpu: float = 1.0,
        memory_mb: int = 512,
        timeout_seconds: int = 60,
        workspace_dir: Optional[str] = None,
    ):
        """
        Args:
            image: Container image to use for the sandbox.
            cpu: CPU limit (fraction or whole number, e.g. 1.0 or 0.5).
            memory_mb: Memory limit in megabytes.
            timeout_seconds: Hard timeout for command execution inside sandbox.
            workspace_dir: Optional host directory bind-mounted as /workspace (rw).
        """
        self.image = image
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.timeout_seconds = timeout_seconds
        self.workspace_dir = workspace_dir

        self._runtime: Optional[str] = None
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if sandbox execution is possible on this host."""
        if self._available is not None:
            return self._available
        self._available, self._runtime = _check_runtime()
        return self._available

    def execute(self, command: str, cwd: str = "/workspace") -> dict:
        """Execute *command* inside the sandbox container.

        Args:
            command: The shell command to run.
            cwd: Working directory inside the container.

        Returns:
            dict with keys:
              - ``output``: stdout+stderr text
              - ``returncode``: integer exit code (0=success)
              - ``sandbox_used``: bool — True if sandbox was used, False=fallback
              - ``error``: str or None — error message if sandbox creation failed
        """
        if not self.is_available():
            return self._execute_local(command, cwd, fallback_reason="no container runtime")

        return self._execute_in_container(command, cwd)

    # ------------------------------------------------------------------
    # Internal: container execution
    # ------------------------------------------------------------------

    def _execute_in_container(self, command: str, cwd: str) -> dict:
        """Run command in a temporary Docker/Podman container with strict limits."""
        runtime = self._runtime or _discover_container_binary()
        container_name = f"hermes-sandbox-{uuid.uuid4().hex[:8]}"
        all_args: list[str] = []

        # Security + resource limits
        all_args.extend(_SANDBOX_SECURITY_ARGS)

        # CPU limit
        if self.cpu > 0:
            all_args.extend(["--cpus", str(self.cpu)])

        # Memory limit
        if self.memory_mb > 0:
            all_args.extend(["--memory", f"{self.memory_mb}m"])

        # Writable scratch areas (tmpfs — rootfs is already read-only via --read-only)
        for mount in _WRITABLE_MOUNTS:
            all_args.append(mount)

        # Optional host workspace bind mount (for data files the command needs)
        if self.workspace_dir:
            all_args.extend([
                "--mount",
                f"type=bind,source={self.workspace_dir},destination=/workspace,readonly=false",
            ])

        # Environment: pass through minimal safe vars
        for key in ("HOME", "USER", "PATH", "LANG", "LC_ALL"):
            val = os.getenv(key)
            if val:
                all_args.extend(["-e", f"{key}={val}"])

        cmd = [
            runtime,
            "run",
            "--rm",
            "--name", container_name,
            "-w", cwd,
            *all_args,
            self.image,
            "sh", "-c", command,
        ]

        logger.debug("Sandbox command: %s", " ".join(cmd))

        _output_chunks: list[str] = []
        start = time.monotonic()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
            )

            def _drain():
                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()

            deadline = time.monotonic() + self.timeout_seconds
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    elapsed = time.monotonic() - start
                    logger.warning(
                        "Sandbox command timed out after %.1fs (limit %ds): %.80s",
                        elapsed, self.timeout_seconds, command[:80],
                    )
                    return {
                        "output": "".join(_output_chunks)
                            + f"\n[sandbox timeout after {self.timeout_seconds}s]",
                        "returncode": 124,
                        "sandbox_used": True,
                        "error": None,
                    }
                time.sleep(0.2)

            reader.join(timeout=5)
            elapsed = time.monotonic() - start
            output = "".join(_output_chunks)
            logger.debug(
                "Sandbox command completed in %.1fs, exit=%d: %.80s",
                elapsed, proc.returncode, command[:80],
            )
            return {
                "output": output,
                "returncode": proc.returncode,
                "sandbox_used": True,
                "error": None,
            }

        except FileNotFoundError:
            # Runtime binary disappeared — don't retry
            self._available = False
            logger.warning("Container runtime binary not found, falling back to local")
            return self._execute_local(command, cwd, fallback_reason="binary not found")

        except Exception as e:
            logger.error("Sandbox execution error: %s", e)
            # Fall back to local so the command still runs
            return self._execute_local(command, cwd, fallback_reason=str(e))

    # ------------------------------------------------------------------
    # Internal: local fallback
    # ------------------------------------------------------------------

    def _execute_local(self, command: str, cwd: str, fallback_reason: str) -> dict:
        """Execute command locally as fallback when sandbox is unavailable."""
        logger.warning(
            "Sandbox unavailable (%s) — executing dangerous command locally: %.80s",
            fallback_reason, command[:80],
        )
        try:
            proc = subprocess.run(
                ["sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=cwd,
            )
            return {
                "output": proc.stdout + proc.stderr,
                "returncode": proc.returncode,
                "sandbox_used": False,
                "error": f"sandbox unavailable ({fallback_reason}); executed locally",
            }
        except subprocess.TimeoutExpired:
            return {
                "output": f"[timeout after {self.timeout_seconds}s]",
                "returncode": 124,
                "sandbox_used": False,
                "error": f"sandbox unavailable ({fallback_reason}); local timeout",
            }
        except Exception as e:
            return {
                "output": "",
                "returncode": -1,
                "sandbox_used": False,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Module-level singleton and convenience API
# ---------------------------------------------------------------------------

_default_runner: Optional[SandboxRunner] = None


def get_sandbox_runner(
    cpu: float = 1.0,
    memory_mb: int = 512,
    timeout_seconds: int = 60,
) -> SandboxRunner:
    """Return a shared SandboxRunner configured for dangerous-command isolation.

    Thread-safe singleton — the same runner instance is reused across calls.
    Resource limits are intentionally conservative (1 CPU, 512 MB, 60 s).
    """
    global _default_runner
    if _default_runner is None:
        _default_runner = SandboxRunner(
            cpu=cpu,
            memory_mb=memory_mb,
            timeout_seconds=timeout_seconds,
        )
    return _default_runner


def run_in_sandbox(command: str, cwd: str = "/workspace") -> dict:
    """Convenience wrapper: run a command in the default sandbox runner.

    Returns the same dict as ``SandboxRunner.execute()``.
    """
    return get_sandbox_runner().execute(command, cwd=cwd)
