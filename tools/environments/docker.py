"""Docker execution environment wrapping mini-swe-agent's DockerEnvironment.

Adds security hardening, configurable resource limits (CPU, memory, disk),
and optional filesystem persistence via `docker commit`/`docker create --image`.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)



# Security flags applied to every container
_SECURITY_ARGS = [
    "--read-only",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--pids-limit", "256",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
    "--tmpfs", "/run:rw,noexec,nosuid,size=64m",
]


_storage_opt_ok: Optional[bool] = None  # cached result across instances


class DockerEnvironment(BaseEnvironment):
    """Hardened Docker container execution with resource limits and persistence.

    Security: read-only root, all capabilities dropped, no privilege escalation,
    PID limits, tmpfs for writable scratch. Writable overlay for /home and cwd
    via tmpfs or bind mounts.

    Persistence: when enabled, `docker commit` saves the container state on
    cleanup, and the next creation restores from that image.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        network: bool = True,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)
        self._base_image = image
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._container_id: Optional[str] = None

        from minisweagent.environments.docker import DockerEnvironment as _Docker

        # Build resource limit args
        resource_args = []
        if cpu > 0:
            resource_args.extend(["--cpus", str(cpu)])
        if memory > 0:
            resource_args.extend(["--memory", f"{memory}m"])
        if disk > 0 and sys.platform != "darwin":
            if self._storage_opt_supported():
                resource_args.extend(["--storage-opt", f"size={disk}m"])
            else:
                logger.warning(
                    "Docker storage driver does not support per-container disk limits "
                    "(requires overlay2 on XFS with pquota). Container will run without disk quota."
                )
        if not network:
            resource_args.append("--network=none")

        # Persistent workspace via bind mounts from a configurable host directory
        # (TERMINAL_SANDBOX_DIR, default ~/.hermes/sandboxes/). Non-persistent
        # mode uses tmpfs (ephemeral, fast, gone on cleanup).
        from tools.environments.base import get_sandbox_dir

        self._workspace_dir: Optional[str] = None
        self._home_dir: Optional[str] = None
        if self._persistent:
            sandbox = get_sandbox_dir() / "docker" / task_id
            self._workspace_dir = str(sandbox / "workspace")
            self._home_dir = str(sandbox / "home")
            os.makedirs(self._workspace_dir, exist_ok=True)
            os.makedirs(self._home_dir, exist_ok=True)
            writable_args = [
                "-v", f"{self._workspace_dir}:/workspace",
                "-v", f"{self._home_dir}:/root",
            ]
        else:
            writable_args = [
                "--tmpfs", "/workspace:rw,exec,size=10g",
                "--tmpfs", "/home:rw,exec,size=1g",
                "--tmpfs", "/root:rw,exec,size=1g",
            ]

        # All containers get full security hardening (read-only root + writable
        # mounts for the workspace). Persistence uses Docker volumes, not
        # filesystem layer commits, so --read-only is always safe.
        all_run_args = list(_SECURITY_ARGS) + writable_args + resource_args

        self._inner = _Docker(
            image=image, cwd=cwd, timeout=timeout,
            run_args=all_run_args,
        )
        self._container_id = self._inner.container_id

    @staticmethod
    def _storage_opt_supported() -> bool:
        """Check if Docker's storage driver supports --storage-opt size=.
        
        Only overlay2 on XFS with pquota supports per-container disk quotas.
        Ubuntu (and most distros) default to ext4, where this flag errors out.
        """
        global _storage_opt_ok
        if _storage_opt_ok is not None:
            return _storage_opt_ok
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Driver}}"],
                capture_output=True, text=True, timeout=10,
            )
            driver = result.stdout.strip().lower()
            if driver != "overlay2":
                _storage_opt_ok = False
                return False
            # overlay2 only supports storage-opt on XFS with pquota.
            # Probe by attempting a dry-ish run — the fastest reliable check.
            probe = subprocess.run(
                ["docker", "create", "--storage-opt", "size=1m", "hello-world"],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                # Clean up the created container
                container_id = probe.stdout.strip()
                if container_id:
                    subprocess.run(["docker", "rm", container_id],
                                   capture_output=True, timeout=5)
                _storage_opt_ok = True
            else:
                _storage_opt_ok = False
        except Exception:
            _storage_opt_ok = False
        logger.debug("Docker --storage-opt support: %s", _storage_opt_ok)
        return _storage_opt_ok

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        exec_command = self._prepare_command(command)
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout

        # docker exec -w doesn't expand ~, so prepend a cd into the command
        if work_dir == "~" or work_dir.startswith("~/"):
            exec_command = f"cd {work_dir} && {exec_command}"
            work_dir = "/"

        assert self._inner.container_id, "Container not started"
        cmd = [self._inner.config.executable, "exec"]
        if stdin_data is not None:
            cmd.append("-i")
        cmd.extend(["-w", work_dir])
        for key in self._inner.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self._inner.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._inner.container_id, "bash", "-lc", exec_command])

        try:
            _output_chunks = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
                text=True,
            )
            if stdin_data:
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except Exception:
                    pass

            def _drain():
                try:
                    for line in proc.stdout:
                        _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + effective_timeout

            while proc.poll() is None:
                if is_interrupted():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}
        except Exception as e:
            return {"output": f"Docker execution error: {e}", "returncode": 1}

    def cleanup(self):
        """Stop and remove the container. Bind-mount dirs persist if persistent=True."""
        self._inner.cleanup()

        if not self._persistent:
            import shutil
            for d in (self._workspace_dir, self._home_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)
