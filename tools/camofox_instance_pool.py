"""Task-scoped Camofox server process pool.

Each Hermes conversation gets a dedicated Camofox API server.  Because each
server launches its own Camoufox process and Xvfb display, browser focus,
contexts, and VNC input cannot cross conversation boundaries.
"""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, Optional

import requests


@dataclass
class CamofoxInstance:
    task_id: str
    api_port: int
    vnc_port: int
    novnc_port: int
    process: subprocess.Popen
    viewer_process: Optional[subprocess.Popen] = None
    log_file: Optional[IO[bytes]] = None

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def viewer_url(self) -> str:
        return (
            f"http://127.0.0.1:{self.novnc_port}/vnc.html"
            "?autoconnect=1&reconnect=1&reconnect_delay=2000&resize=scale"
        )


class CamofoxInstancePool:
    """Own one independent headed Camofox server per Hermes task."""

    def __init__(
        self,
        server_dir: Path,
        *,
        port_start: int = 19400,
        port_end: int = 19999,
        startup_timeout: float = 60.0,
        launch_viewer: bool = False,
        viewer_executable: Path = Path("~/.cache/camoufox/camoufox"),
        viewer_profile_root: Path = Path("~/.camoufox-hermes-thread-viewers"),
        log_root: Path = Path("~/.hermes/logs/camofox-instances"),
    ) -> None:
        self.server_dir = Path(server_dir).expanduser().resolve()
        self.port_start = port_start
        self.port_end = port_end
        self.startup_timeout = startup_timeout
        self.launch_viewer = launch_viewer
        self.viewer_executable = Path(viewer_executable).expanduser().resolve()
        self.viewer_profile_root = Path(viewer_profile_root).expanduser().resolve()
        self.log_root = Path(log_root).expanduser().resolve()
        self._instances: Dict[str, CamofoxInstance] = {}
        self._lock = threading.RLock()
        self._scope_locks: Dict[str, threading.RLock] = {}

    def _scope_lock(self, task_id: Optional[str]) -> threading.RLock:
        key = task_id or "default"
        with self._lock:
            return self._scope_locks.setdefault(key, threading.RLock())

    @contextmanager
    def scope_lifecycle(self, task_id: Optional[str]):
        """Exclude start/stop/profile mutation for exactly one browser scope."""
        lock = self._scope_lock(task_id)
        with lock:
            yield

    def _ports_for_task(self, task_id: str) -> tuple[int, int, int]:
        slots = (self.port_end - self.port_start + 1) // 3
        if slots < 1:
            raise ValueError("Camofox instance port range must contain at least three ports")
        initial = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big") % slots
        for offset in range(slots):
            slot = (initial + offset) % slots
            ports = tuple(self.port_start + slot * 3 + index for index in range(3))
            if all(self._port_available(port) for port in ports):
                return ports  # type: ignore[return-value]
        raise RuntimeError("No free task-scoped Camofox port triple is available")

    @staticmethod
    def _port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    def get_or_start(self, task_id: Optional[str]) -> CamofoxInstance:
        key = task_id or "default"
        with self.scope_lifecycle(key):
            with self._lock:
                current = self._instances.get(key)
                if current and current.process.poll() is None:
                    return current
                if current:
                    self.stop(key)

                if not (self.server_dir / "server.js").is_file():
                    raise RuntimeError(f"Camofox server.js not found under {self.server_dir}")

                api_port, vnc_port, novnc_port = self._ports_for_task(key)
                env = os.environ.copy()
                env.update({
                    "CAMOFOX_PORT": str(api_port),
                    "VNC_PORT": str(vnc_port),
                    "NOVNC_PORT": str(novnc_port),
                    "VNC_BIND": "127.0.0.1",
                    "NODE_ENV": "production",
                })
                self.log_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(key.encode()).hexdigest()[:16]
                log_file = (self.log_root / f"{digest}.log").open("ab", buffering=0)
                process = subprocess.Popen(
                    ["node", "server.js"],
                    cwd=self.server_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                instance = CamofoxInstance(
                    key, api_port, vnc_port, novnc_port, process, log_file=log_file
                )
                self._instances[key] = instance
                try:
                    # Serialize same-process acquisition until the published
                    # instance is actually ready. A second caller must never
                    # receive a merely-spawned server that can still fail startup.
                    self._wait_until_ready(instance)
                except Exception:
                    self.stop(key)
                    raise
                return instance

    def ensure_viewer(self, instance: CamofoxInstance, *, force: bool = False) -> None:
        """Launch this instance's popup after noVNC is live.

        ``force`` replaces a still-running popup whose URL can point at an old
        Xvfb display after browser recovery.  Readiness is established before
        the old popup is stopped, and the pool lock serializes replacement so
        two callers cannot leave duplicate viewers behind.
        """
        if not self.launch_viewer:
            return
        with self._lock:
            viewer = instance.viewer_process
            if not force and viewer and viewer.poll() is None:
                return
            deadline = time.monotonic() + self.startup_timeout
            last_error = "not ready"
            while time.monotonic() < deadline:
                try:
                    response = requests.get(instance.viewer_url, timeout=1)
                    if response.status_code == 200:
                        if viewer and viewer.poll() is None:
                            self._terminate_viewer(viewer)
                        instance.viewer_process = None
                        self._launch_viewer(instance)
                        return
                    last_error = f"HTTP {response.status_code}"
                except requests.RequestException as exc:
                    last_error = str(exc)
                time.sleep(0.2)
        raise RuntimeError(f"Task-scoped Camofox noVNC did not become ready: {last_error}")

    @staticmethod
    def _terminate_viewer(viewer: subprocess.Popen) -> None:
        os.killpg(os.getpgid(viewer.pid), signal.SIGTERM)
        try:
            viewer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(viewer.pid), signal.SIGKILL)
            viewer.wait(timeout=5)

    def _launch_viewer(self, instance: CamofoxInstance) -> None:
        """Open a dedicated native popup showing this instance's VNC display."""
        if not self.viewer_executable.is_file():
            raise RuntimeError(f"Camoufox viewer executable not found: {self.viewer_executable}")
        digest = hashlib.sha256(instance.task_id.encode()).hexdigest()[:16]
        profile_dir = self.viewer_profile_root / digest
        profile_dir.mkdir(parents=True, exist_ok=True)
        instance.viewer_process = subprocess.Popen(
            [
                str(self.viewer_executable),
                "--new-instance",
                "--profile",
                str(profile_dir),
                instance.viewer_url,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _wait_until_ready(self, instance: CamofoxInstance) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "not ready"
        while time.monotonic() < deadline:
            if instance.process.poll() is not None:
                raise RuntimeError(
                    f"Task-scoped Camofox exited during startup ({instance.process.returncode})"
                )
            try:
                response = requests.get(f"{instance.api_url}/health", timeout=1)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"Task-scoped Camofox did not become ready: {last_error}")

    def stop(self, task_id: Optional[str]) -> None:
        key = task_id or "default"
        with self.scope_lifecycle(key):
            with self._lock:
                instance = self._instances.pop(key, None)
            if not instance:
                return
            if instance.viewer_process and instance.viewer_process.poll() is None:
                self._terminate_viewer(instance.viewer_process)
            if instance.process.poll() is None:
                os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)
                try:
                    instance.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                    instance.process.wait(timeout=5)
            if instance.log_file:
                instance.log_file.close()

    def stop_all(self) -> None:
        with self._lock:
            task_ids = list(self._instances)
        for task_id in task_ids:
            self.stop(task_id)
