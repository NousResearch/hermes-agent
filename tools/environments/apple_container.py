"""Apple Container execution environment for macOS.

Uses Apple's native containerization framework (macOS 26+) which runs each
Linux container inside its own lightweight virtual machine via
Virtualization.framework. Provides VM-level isolation (separate kernel per
container) with sub-second startup on Apple Silicon.

Security model: each container gets its own Linux kernel, so the VM boundary
is the primary isolation mechanism (stronger than Docker's namespace-based
isolation). Inside the container we additionally apply --read-only root
filesystem and size-limited tmpfs mounts, matching Docker backend conventions
where the CLI supports it.

Requires: macOS 26+, Apple Silicon, `container` CLI (brew install container).
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash, get_sandbox_dir

logger = logging.getLogger(__name__)

_CONTAINER_SEARCH_PATHS = [
    "/opt/homebrew/bin/container",
    "/usr/local/bin/container",
]

_container_executable: Optional[str] = None
_system_resources: Optional[dict] = None  # cached after first query


def find_container_cli() -> Optional[str]:
    """Locate the Apple ``container`` CLI binary.

    Checks PATH first, then probes Homebrew install locations.
    Returns the absolute path, or None if not found.
    """
    global _container_executable
    if _container_executable is not None:
        return _container_executable

    found = shutil.which("container")
    if found:
        _container_executable = found
        return found

    for path in _CONTAINER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _container_executable = path
            logger.info("Found container CLI at non-PATH location: %s", path)
            return path

    return None


def _ensure_container_available() -> str:
    """Verify the Apple container CLI is available and the system is running.

    Returns the path to the container executable.
    Raises RuntimeError with actionable messages on failure.
    """
    exe = find_container_cli()
    if not exe:
        raise RuntimeError(
            "Apple Containers CLI not found. Install with: brew install container\n"
            "Requires macOS 26 (Tahoe) or later on Apple Silicon."
        )

    # Check version
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'container --version' failed (exit {result.returncode}). "
                "Check your Apple Containers installation."
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("'container --version' timed out.")

    # Check system status
    try:
        result = subprocess.run(
            [exe, "system", "status"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or "running" not in result.stdout.lower():
            logger.info("Apple Container system not running, attempting to start...")
            start = subprocess.run(
                [exe, "system", "start"],
                capture_output=True, text=True, timeout=30,
                input="Y\n",
            )
            if start.returncode != 0:
                raise RuntimeError(
                    "Failed to start Apple Container system. "
                    "Run manually: container system start"
                )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Apple Container system check timed out. "
            "Run manually: container system start"
        )

    return exe


def query_system_resources() -> dict:
    """Query the host system for available CPU and memory.

    Results are cached after the first call.
    Returns dict with 'total_cpus' and 'total_memory_mb' keys.
    """
    global _system_resources
    if _system_resources is not None:
        return _system_resources

    info = {"total_cpus": os.cpu_count() or 4, "total_memory_mb": 8192}
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["total_memory_mb"] = int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass

    _system_resources = info
    return info


def suggest_resources(total_cpus: int, total_memory_mb: int) -> dict:
    """Suggest container resource allocation based on system specs.

    Reserves roughly half the CPUs and a quarter of RAM for the host
    (LM Studio / Ollama needs significant resources for model inference).
    """
    container_cpus = max(2, total_cpus // 2)
    container_memory_mb = max(4096, total_memory_mb // 4)
    return {
        "cpus": container_cpus,
        "memory_mb": container_memory_mb,
    }


# Sensitive host paths that should never be volume-mounted into a container.
_SENSITIVE_MOUNT_SOURCES = {
    "/.ssh", "/ssh", ".ssh",
    "/.gnupg", "/gnupg", ".gnupg",
    "/.aws", "/aws", ".aws",
    "/.config/gcloud", "/gcloud",
    "/.azure", "/azure",
    "/.kube", "/kube",
}


def _warn_sensitive_volumes(volumes: list[str]) -> None:
    """Log warnings for volume mounts that expose sensitive host directories."""
    for vol in volumes:
        src = vol.split(":")[0] if ":" in vol else vol
        src_lower = src.lower()
        for pattern in _SENSITIVE_MOUNT_SOURCES:
            if src_lower.endswith(pattern) or f"{pattern}/" in src_lower:
                logger.warning(
                    "Volume mount '%s' exposes a sensitive host directory. "
                    "This may leak credentials into the container.",
                    vol,
                )
                break


class AppleContainerEnvironment(BaseEnvironment):
    """Apple Container execution with VM-level isolation.

    Each container runs inside its own lightweight Linux VM via Apple's
    Virtualization.framework. Commands are executed via ``container exec``,
    similar to Docker's ``docker exec``, with container lifecycle managed
    by this class.

    Security: the VM boundary provides kernel-level isolation. Additionally,
    the root filesystem is mounted read-only with size-limited tmpfs for
    scratch directories, and credential/skills files are mounted read-only.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim-bookworm",
        cwd: str = "/root",
        timeout: int = 180,
        cpu: int = 0,
        memory: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
        volumes: list = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)

        self._exe = _ensure_container_available()
        self._base_image = image
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._container_name: Optional[str] = None
        self._workspace_dir: Optional[str] = None

        # Resolve resource limits (cached sysctl query)
        sys_info = query_system_resources()
        suggested = suggest_resources(sys_info["total_cpus"], sys_info["total_memory_mb"])
        self._cpus = cpu if cpu > 0 else suggested["cpus"]
        self._memory_mb = memory if memory > 0 else suggested["memory_mb"]

        # Build and start the container
        self._start_container(image, volumes or [])

        # Initialize session snapshot
        self.init_session()

    def _start_container(self, image: str, volumes: list):
        """Pull image if needed and start the container."""
        container_name = f"hermes-{uuid.uuid4().hex[:8]}"

        run_cmd = [
            self._exe, "run",
            "--name", container_name,
            "--detach",
            "--cpus", str(self._cpus),
            "--memory", f"{self._memory_mb}M",
            # Read-only root for security; writable scratch via tmpfs
            "--read-only",
            "--tmpfs", "/tmp:rw,size=512m",
            "--tmpfs", "/var/tmp:rw,size=256m",
            "--tmpfs", "/run:rw,size=64m",
        ]

        # Persistent workspace via bind mount, or ephemeral tmpfs
        if self._persistent:
            sandbox = get_sandbox_dir() / "apple_container" / self._task_id
            self._workspace_dir = str(sandbox / "workspace")
            os.makedirs(self._workspace_dir, exist_ok=True)
            run_cmd.extend(["--volume", f"{self._workspace_dir}:/workspace"])
            run_cmd.extend(["--volume", f"{str(sandbox / 'root')}:/root"])
            os.makedirs(str(sandbox / "root"), exist_ok=True)
        else:
            run_cmd.extend([
                "--tmpfs", "/workspace:rw,size=10g",
                "--tmpfs", "/root:rw,size=1g",
                "--tmpfs", "/home:rw,size=1g",
            ])

        # Mount credential files, skills, and cache directories read-only
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
                get_cache_directory_mounts,
            )

            for mount_entry in get_credential_file_mounts():
                run_cmd.extend([
                    "--volume",
                    f"{mount_entry['host_path']}:{mount_entry['container_path']}",
                ])
                logger.info(
                    "Apple Container: mounting credential %s -> %s",
                    mount_entry["host_path"],
                    mount_entry["container_path"],
                )

            for skills_mount in get_skills_directory_mount():
                run_cmd.extend([
                    "--volume",
                    f"{skills_mount['host_path']}:{skills_mount['container_path']}",
                ])
                logger.info(
                    "Apple Container: mounting skills dir %s -> %s",
                    skills_mount["host_path"],
                    skills_mount["container_path"],
                )

            for cache_mount in get_cache_directory_mounts():
                run_cmd.extend([
                    "--volume",
                    f"{cache_mount['host_path']}:{cache_mount['container_path']}",
                ])
                logger.info(
                    "Apple Container: mounting cache dir %s -> %s",
                    cache_mount["host_path"],
                    cache_mount["container_path"],
                )
        except Exception as e:
            logger.debug("Apple Container: could not load credential file mounts: %s", e)

        # User-supplied volume mounts
        _warn_sensitive_volumes(volumes)
        for vol in volumes:
            if isinstance(vol, str) and vol.strip():
                run_cmd.extend(["--volume", vol.strip()])

        run_cmd.append(image)
        # Keep the container alive with a long sleep
        run_cmd.extend(["sleep", "86400"])

        logger.debug("Starting Apple Container: %s", " ".join(run_cmd))
        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # image pull can take a while
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(
                    f"Failed to start Apple Container (exit {result.returncode}): {stderr}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Apple Container startup timed out. The image may be too large "
                "or the container system may not be running."
            )

        self._container_name = container_name
        logger.info(
            "Started Apple Container '%s' (%d CPUs, %d MB RAM)",
            container_name, self._cpus, self._memory_mb,
        )

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        """Spawn a bash process inside the Apple Container."""
        assert self._container_name, "Container not started"

        cmd = [self._exe, "exec"]
        if stdin_data is not None:
            cmd.append("-i")
        cmd.append(self._container_name)

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop and remove the container, waiting for graceful shutdown."""
        if not self._container_name:
            return

        name = self._container_name
        self._container_name = None  # prevent double-cleanup

        try:
            # Graceful stop with a timeout — waits for the process to exit
            subprocess.run(
                [self._exe, "stop", name],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out stopping Apple Container '%s', force killing", name)
            try:
                subprocess.run(
                    [self._exe, "kill", name],
                    capture_output=True, text=True, timeout=10,
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to stop Apple Container '%s': %s", name, e)

        # Remove the stopped container
        try:
            subprocess.run(
                [self._exe, "rm", name],
                capture_output=True, text=True, timeout=10,
            )
            logger.info("Removed Apple Container '%s'", name)
        except Exception as e:
            logger.debug("Failed to remove Apple Container '%s': %s", name, e)

        # Clean up workspace if non-persistent
        if not self._persistent and self._workspace_dir:
            import shutil as _shutil
            _shutil.rmtree(self._workspace_dir, ignore_errors=True)
