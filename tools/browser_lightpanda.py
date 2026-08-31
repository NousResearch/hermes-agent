"""Lightpanda local engine for Browser Use mode.

With ``browser.engine: lightpanda``, Browser Use mode spawns one
``lightpanda serve`` per browser session and points ``browser_exec`` at its
CDP endpoint (``BU_CDP_URL``). The built-in ``browser_*`` tools keep driving
Lightpanda through ``agent-browser --engine lightpanda``; this module is the
launcher for the path where no agent-browser daemon is involved.

Lifecycle: ``tools.browser_tool`` owns the session cache, the inactivity
reaper and the atexit sweep; it calls :func:`launch_lightpanda` /
:func:`stop_lightpanda` and :func:`reap_orphaned_lightpanda` for processes
left behind by a crashed Hermes.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

LIGHTPANDA_INSTALL_URL = "https://lightpanda.io/docs/run-locally/installation/one-liner"
LIGHTPANDA_INSTALL_HINT = (
    f"Install Lightpanda from {LIGHTPANDA_INSTALL_URL} and make sure "
    "`lightpanda` is on PATH"
)

_READY_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 0.1
_STDERR_TAIL_LIMIT = 2000

_servers: Dict[str, "LightpandaServer"] = {}
_servers_lock = threading.Lock()


@dataclass
class LightpandaServer:
    session_name: str
    port: int
    proc: subprocess.Popen
    log_path: str
    start_time: Optional[int] = None

    @property
    def cdp_url(self) -> str:
        # The http discovery URL: the browser-use harness resolves
        # /json/version itself on every daemon start (BU_CDP_URL).
        return f"http://127.0.0.1:{self.port}"

    def is_alive(self) -> bool:
        return self.proc.poll() is None


def _home_candidates() -> list:
    home = Path.home()
    candidates = [
        home / ".lightpanda" / "lightpanda",
        home / ".local" / "bin" / "lightpanda",
    ]
    try:
        from hermes_constants import get_hermes_home

        candidates.append(Path(get_hermes_home()) / "bin" / "lightpanda")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("hermes home unavailable for lightpanda lookup: %s", e)
    return candidates


def find_lightpanda_binary() -> Optional[str]:
    """Return the lightpanda executable, or None.

    Order: PATH (with the same Homebrew/managed-node fallbacks agent-browser
    gets), then the locations the Lightpanda installer and agent-browser use
    (``~/.lightpanda/lightpanda``, ``~/.local/bin/lightpanda``), then
    ``$HERMES_HOME/bin/lightpanda``. Lightpanda has no Windows build.
    """
    if os.name == "nt":
        logger.debug("Lightpanda has no Windows build")
        return None
    path_env = os.environ.get("PATH", "")
    try:
        from tools.browser_tool import _merge_browser_path

        path_env = _merge_browser_path(path_env)
    except Exception as e:
        logger.debug("browser PATH merge unavailable: %s", e)
    found = shutil.which("lightpanda", path=path_env)
    if found:
        return found
    for candidate in _home_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _pick_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _state_dir() -> Path:
    from hermes_constants import get_hermes_home

    path = Path(get_hermes_home()) / "cache" / "browser-use" / "lightpanda"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_path(session_name: str) -> Path:
    return _state_dir() / f"{session_name}.json"


def _browser_env() -> dict:
    try:
        from tools.browser_tool import _build_browser_env

        return _build_browser_env()
    except Exception as e:
        logger.debug("credential-scrubbed browser env unavailable: %s", e)
        return os.environ.copy()


def _cdp_ready(url: str) -> bool:
    try:
        from hermes_cli.browser_connect import is_browser_debug_ready

        return is_browser_debug_ready(url, timeout=0.2)
    except Exception as e:
        logger.debug("CDP readiness probe failed for %s: %s", url, e)
        return False


def _read_log_tail(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return ""
    text = data[-_STDERR_TAIL_LIMIT:].decode("utf-8", errors="replace").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _terminate(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    except Exception as e:
        logger.debug("lightpanda terminate failed: %s", e)


def _safe_start_time(pid: int) -> Optional[int]:
    try:
        from tools.process_registry import ProcessRegistry

        return ProcessRegistry._safe_host_start_time(pid)
    except Exception:
        return None


def _write_record(server: LightpandaServer) -> None:
    record = {
        "pid": server.proc.pid,
        "port": server.port,
        "owner_pid": os.getpid(),
        "start_time": server.start_time,
        "started_at": time.time(),
    }
    try:
        _record_path(server.session_name).write_text(json.dumps(record), encoding="utf-8")
    except OSError as e:
        logger.debug("could not write lightpanda record for %s: %s", server.session_name, e)


def _unlink_record(session_name: str) -> None:
    try:
        _record_path(session_name).unlink(missing_ok=True)
    except OSError as e:
        logger.debug("could not remove lightpanda record for %s: %s", session_name, e)


def launch_lightpanda(
    session_name: str, *, block_private_networks: bool = False
) -> Tuple[Optional[LightpandaServer], Optional[str]]:
    """Start ``lightpanda serve`` on a free loopback port for ``session_name``.

    Returns ``(server, None)`` once ``/json/version`` answers, or
    ``(None, error)`` with an actionable message. The child's stderr goes to
    ``$HERMES_HOME/cache/browser-use/lightpanda/<session>.log`` so a chatty
    process can never block on a pipe; only the tail is read on failure.
    """
    binary = find_lightpanda_binary()
    if not binary:
        if os.name == "nt":
            return None, (
                "browser.engine is 'lightpanda' but Lightpanda has no Windows "
                "build. Set browser.engine to auto (or run Hermes under WSL2)."
            )
        return None, (
            "browser.engine is 'lightpanda' but no lightpanda binary was found "
            "on PATH, ~/.lightpanda or ~/.local/bin. "
            f"{LIGHTPANDA_INSTALL_HINT}, or set browser.engine to auto."
        )

    port = _pick_free_loopback_port()
    argv = [binary, "serve", "--host", "127.0.0.1", "--port", str(port)]
    if block_private_networks:
        argv.append("--block-private-networks")
    log_path = str(_state_dir() / f"{session_name}.log")

    if os.name == "nt":  # pragma: no cover - no Windows build today
        from hermes_cli._subprocess_compat import windows_hide_flags

        popen_kwargs: dict = {"creationflags": windows_hide_flags(), "close_fds": True}
    else:
        popen_kwargs = {"start_new_session": True}

    try:
        with open(log_path, "wb") as log_file:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                env=_browser_env(),
                **popen_kwargs,
            )
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"Failed to launch lightpanda serve ({binary}): {e}"

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        rc = proc.poll()
        if rc is not None:
            tail = _read_log_tail(log_path)
            detail = f": {tail}" if tail else ""
            return None, (
                f"lightpanda serve exited with code {rc} before {url}/json/version "
                f"answered{detail}"
            )
        if _cdp_ready(url):
            break
        if time.monotonic() >= deadline:
            _terminate(proc)
            tail = _read_log_tail(log_path)
            detail = f" (last stderr line: {tail})" if tail else ""
            return None, (
                f"lightpanda serve did not expose {url}/json/version within "
                f"{int(_READY_TIMEOUT_S)}s{detail}"
            )
        time.sleep(_POLL_INTERVAL_S)

    server = LightpandaServer(
        session_name=session_name,
        port=port,
        proc=proc,
        log_path=log_path,
        start_time=_safe_start_time(proc.pid),
    )
    _write_record(server)
    with _servers_lock:
        _servers[session_name] = server
    logger.info(
        "Started lightpanda serve (pid %s, port %s) for session %s",
        proc.pid, port, session_name,
    )
    return server, None


def get_server(session_name: str) -> Optional[LightpandaServer]:
    with _servers_lock:
        return _servers.get(session_name)


def stop_lightpanda(session_name: str) -> None:
    """Stop the server for ``session_name`` (tree-kill) and drop its record."""
    with _servers_lock:
        server = _servers.pop(session_name, None)
    if server is None:
        _unlink_record(session_name)
        return
    if server.is_alive():
        try:
            from tools.process_registry import ProcessRegistry

            ProcessRegistry._terminate_host_pid(
                server.proc.pid, expected_start=server.start_time
            )
        except Exception as e:
            logger.debug("lightpanda tree-kill failed for %s: %s", session_name, e)
            _terminate(server.proc)
        try:
            server.proc.wait(timeout=5)
        except Exception:
            _terminate(server.proc)
    _unlink_record(session_name)
    logger.debug("Stopped lightpanda serve for session %s", session_name)


def stop_all_lightpanda() -> None:
    """Stop every server this process started. Idempotent; safe from atexit."""
    with _servers_lock:
        names = list(_servers)
    for name in names:
        try:
            stop_lightpanda(name)
        except Exception as e:
            logger.debug("lightpanda stop failed for %s: %s", name, e)


def _is_lightpanda_process(pid: int, port, start_time) -> bool:
    """True only when ``pid`` is verifiably the ``lightpanda serve`` we recorded."""
    try:
        import psutil

        proc = psutil.Process(pid)
        if "lightpanda" not in proc.name().lower():
            return False
        cmdline = proc.cmdline()
        if "serve" not in cmdline or str(port) not in cmdline:
            return False
    except Exception:
        return False
    if start_time:
        try:
            from gateway.status import get_process_start_time

            return get_process_start_time(pid) == start_time
        except Exception:
            return False
    return True


def reap_orphaned_lightpanda() -> int:
    """Kill ``lightpanda serve`` processes whose owning Hermes is gone.

    Records are written by :func:`launch_lightpanda`; a live owner (another
    Hermes process, or this one still tracking the session) is never
    touched, and a PID is only signalled after psutil confirms it is still
    a ``lightpanda serve`` on the recorded port. Returns the reap count.
    """
    try:
        state_dir = _state_dir()
    except Exception as e:
        logger.debug("lightpanda state dir unavailable: %s", e)
        return 0
    try:
        from gateway.status import _pid_exists
    except Exception:  # pragma: no cover - defensive
        return 0

    reaped = 0
    for record_path in sorted(state_dir.glob("*.json")):
        session_name = record_path.stem
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record_path.unlink(missing_ok=True)
            continue
        owner_pid = record.get("owner_pid")
        if owner_pid == os.getpid():
            with _servers_lock:
                if session_name in _servers:
                    continue
        elif owner_pid and _pid_exists(int(owner_pid)):
            continue
        pid = record.get("pid")
        if not pid or not _is_lightpanda_process(int(pid), record.get("port"), record.get("start_time")):
            record_path.unlink(missing_ok=True)
            continue
        try:
            from tools.process_registry import ProcessRegistry

            ProcessRegistry._terminate_host_pid(int(pid), expected_start=record.get("start_time"))
            reaped += 1
            logger.info("Reaped orphaned lightpanda serve pid %s (session %s)", pid, session_name)
        except Exception as e:
            logger.debug("orphan lightpanda kill failed for pid %s: %s", pid, e)
        record_path.unlink(missing_ok=True)
    return reaped
