"""Persistent, profile-scoped bridge to Stagehand's production facade."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Mapping, TextIO

from agent.secret_scope import get_secret
from hermes_constants import hermes_home_key
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

_PROTOCOL = "hermes-stagehand-facade-v1"
_INITIALIZATION_ALLOWANCE_S = 180


def _worker_environment(stagehand_root: Path) -> dict[str, str]:
    """Expose only the two Browserbase credentials the worker requires."""
    env = hermes_subprocess_env(inherit_credentials=False)
    for name in ("BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"):
        value = (get_secret(name, "") or "").strip()
        if value:
            env[name] = value
    env["STAGEHAND_FACADE_ROOT"] = str(stagehand_root)
    return env


class _FacadeWorkerClient:
    def __init__(self, *, node_executable: str, stagehand_root: str) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._request_id = 0
        self._node = str(Path(node_executable).expanduser().resolve())
        self._stagehand_root = str(Path(stagehand_root).expanduser().resolve())

    def call(self, *, code: str, timeout_s: int) -> dict[str, Any]:
        with self._lock:
            self._ensure_worker()
            self._request_id += 1
            request_id = self._request_id
            self._send(
                {
                    "protocol": _PROTOCOL,
                    "type": "call",
                    "request_id": request_id,
                    "code": code,
                }
            )
            response = self._receive(timeout_s + _INITIALIZATION_ALLOWANCE_S)
            if (
                response.get("type") != "response"
                or response.get("request_id") != request_id
            ):
                raise RuntimeError(
                    "Stagehand facade returned an out-of-order response"
                )
            return response

    def _ensure_worker(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return

        node = Path(self._node)
        root = Path(self._stagehand_root)
        if not node.is_file():
            raise RuntimeError(
                f"Stagehand facade Node executable is missing: {node}"
            )
        required = (
            root
            / "packages"
            / "integrations"
            / "core"
            / "dist"
            / "facade"
            / "index.mjs"
        )
        sdk = root / "packages" / "sdk-ts" / "dist" / "index.mjs"
        if not required.is_file() or not sdk.is_file():
            raise RuntimeError(
                "Built Stagehand V4 facade artifacts are missing under "
                f"{root}. Build @browserbasehq/stagehand and "
                "@browserbasehq/stagehand-integrations first."
            )

        worker = Path(__file__).with_name("stagehand_facade_worker.mjs")
        self._queue = queue.Queue()
        popen_extra: dict[str, Any] = {}
        if hasattr(subprocess, "STARTUPINFO"):
            from hermes_cli._subprocess_compat import windows_hide_flags

            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_extra = {
                "creationflags": windows_hide_flags(),
                "startupinfo": startup_info,
            }
        self._process = subprocess.Popen(
            [str(node), str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_worker_environment(root),
            **popen_extra,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            name="stagehand-facade-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            name="stagehand-facade-stderr",
            daemon=True,
        ).start()
        ready = self._receive(30)
        if ready.get("type") != "ready" or ready.get("protocol") != _PROTOCOL:
            self.close()
            raise RuntimeError("Stagehand facade did not complete its handshake")

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    self._queue.put(
                        RuntimeError(
                            f"Stagehand facade emitted invalid JSON: {error}"
                        )
                    )
                    continue
                self._queue.put(
                    value
                    if isinstance(value, dict)
                    else RuntimeError(
                        "Stagehand facade emitted a non-object response"
                    )
                )
            self._queue.put(RuntimeError("Stagehand facade stdout closed"))
        except BaseException as error:
            self._queue.put(error)

    @staticmethod
    def _read_stderr(stream: TextIO) -> None:
        for line in stream:
            if line.strip():
                logger.debug("Stagehand facade: %s", line.strip()[:1000])

    def _send(self, value: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Stagehand facade is not running")
        process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _receive(self, timeout_s: int) -> dict[str, Any]:
        try:
            value = self._queue.get(timeout=timeout_s)
        except queue.Empty as error:
            raise TimeoutError(
                f"Stagehand facade did not respond within {timeout_s}s"
            ) from error
        if isinstance(value, BaseException):
            raise RuntimeError(str(value)[:1000]) from value
        return value

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None:
                    self._request_id += 1
                    self._send(
                        {
                            "protocol": _PROTOCOL,
                            "type": "shutdown",
                            "request_id": self._request_id,
                        }
                    )
                    self._receive(35)
                    process.wait(timeout=5)
            except Exception:
                logger.debug("Stagehand facade cleanup failed", exc_info=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                self._process = None


_CLIENTS: dict[tuple[str, str, str, str], _FacadeWorkerClient] = {}
_CLIENTS_LOCK = threading.RLock()


def _close_all_workers() -> None:
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        client.close()


atexit.register(_close_all_workers)


def close_stagehand_facade_workers() -> None:
    """Close all profile/task-scoped browser workers (primarily for tests)."""
    _close_all_workers()


def call_stagehand_facade(
    *,
    code: str,
    timeout_s: int,
    node_executable: str,
    stagehand_root: str,
    task_key: str,
) -> dict[str, Any]:
    """Call the persistent worker isolated to the active profile and task."""
    node = str(Path(node_executable).expanduser().resolve())
    root = str(Path(stagehand_root).expanduser().resolve())
    key = (hermes_home_key(), task_key, node, root)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = _FacadeWorkerClient(
                node_executable=node,
                stagehand_root=root,
            )
            _CLIENTS[key] = client
    return client.call(code=code, timeout_s=timeout_s)
