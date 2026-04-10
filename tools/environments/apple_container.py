"""Apple Container execution environment for macOS.

Uses Apple's native containerization framework (macOS 26+) which runs each
Linux container inside its own lightweight virtual machine via
Virtualization.framework. Provides VM-level isolation (separate kernel per
container) with sub-second startup on Apple Silicon.

Requires: macOS 26+, Apple Silicon, `container` CLI (brew install container).
"""

import logging
import os
import shutil
import subprocess
import sys
import uuid
from typing import Optional

from tools.environments.base import BaseEnvironment, _popen_bash

logger = logging.getLogger(__name__)

_CONTAINER_SEARCH_PATHS = [
    "/opt/homebrew/bin/container",
    "/usr/local/bin/container",
]

_container_executable: Optional[str] = None


def find_container_cli() -> Optional[str]:
    """Locate the Apple `container` CLI binary.

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

    Returns dict with 'total_cpus' and 'total_memory_mb' keys.
    """
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


class AppleContainerEnvironment(BaseEnvironment):
    """Apple Container execution with VM-level isolation.

    Each container runs inside its own lightweight Linux VM via Apple's
    Virtualization.framework. Provides stronger isolation than Docker
    (separate kernel per container) with sub-second startup on Apple Silicon.

    The container runs an SSH server for command execution, matching the
    pattern used by SSHEnvironment but with container lifecycle management.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim-bookworm",
        cwd: str = "/home/hermes/work",
        timeout: int = 180,
        cpu: int = 0,
        memory: int = 0,
        task_id: str = "default",
        volumes: list = None,
    ):
        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd, timeout=timeout)

        self._exe = _ensure_container_available()
        self._base_image = image
        self._task_id = task_id
        self._container_name: Optional[str] = None

        # Resolve resource limits
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
        ]

        # Add volume mounts
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
        cmd.append(self._container_name)

        if login:
            cmd.extend(["bash", "-l", "-c", cmd_string])
        else:
            cmd.extend(["bash", "-c", cmd_string])

        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop and remove the container."""
        if self._container_name:
            try:
                subprocess.Popen(
                    f"({self._exe} stop {self._container_name} && "
                    f"{self._exe} rm {self._container_name}) >/dev/null 2>&1 &",
                    shell=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to stop Apple Container '%s': %s",
                    self._container_name, e,
                )
            self._container_name = None
