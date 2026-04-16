"""Lightpanda local headless browser provider.

Lightpanda is an open-source, self-hosted headless browser written in Zig.
It exposes a Chrome DevTools Protocol (CDP) server that the browser tool
connects to, making it a fast, lightweight alternative to cloud providers or
local Chromium.

Binary installation::

    npm install -g @lightpanda/browser   # npm (installs 'lightpanda' binary)
    brew install lightpanda-io/browser/lightpanda  # macOS Homebrew

Docker (set LIGHTPANDA_CDP_HOST to the container's IP)::

    docker run -p 9222:9222 lightpanda/browser

GitHub: https://github.com/lightpanda-io/browser
"""

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_DEFAULT = 10.0  # seconds to wait for CDP to become ready
_POLL_INTERVAL = 0.2             # seconds between readiness polls


class LightpandaProvider(CloudBrowserProvider):
    """Lightpanda local headless browser backend.

    Spawns a lightpanda process on a free port for each session and connects
    via the Chrome DevTools Protocol websocket URL it exposes.  The process is
    terminated when the session is closed.

    Environment variables
    ---------------------
    LIGHTPANDA_CDP_URL
        If set, the provider skips subprocess spawning entirely and uses this
        WebSocket URL as the CDP endpoint for all sessions.  Use this when
        Lightpanda is already running externally (e.g. ``docker run -d
        --network host lightpanda/browser``).  This mode is more reliable than
        per-session spawning when Docker container startup is slow.
        Example: ``LIGHTPANDA_CDP_URL=ws://127.0.0.1:9222/``
    LIGHTPANDA_PATH
        Explicit path to the lightpanda binary (overrides PATH search).
        Used when LIGHTPANDA_CDP_URL is not set.
    LIGHTPANDA_CDP_HOST
        Host the CDP server should bind to (default: ``127.0.0.1``).
    LIGHTPANDA_PORT
        Fixed port instead of a random free port.  Useful when running a
        single session or behind a firewall.
    LIGHTPANDA_STARTUP_TIMEOUT
        Seconds to wait for the CDP server to become ready (default: 10).
    """

    def __init__(self) -> None:
        self._processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CloudBrowserProvider interface
    # ------------------------------------------------------------------

    def provider_name(self) -> str:
        return "Lightpanda"

    def is_configured(self) -> bool:
        """Return True when Lightpanda can be used.

        Either an external CDP URL is configured, or a lightpanda binary
        is locatable on the system.
        """
        if os.environ.get("LIGHTPANDA_CDP_URL", "").strip():
            return True
        return self._find_binary() is not None

    def create_session(self, task_id: str) -> Dict[str, object]:
        """Spawn a lightpanda process and return its CDP websocket URL.

        Returns a dict compatible with the rest of browser_tool.py::

            {
                "session_name": str,   # unique human-readable label
                "bb_session_id": str,  # internal session key (used for close/cleanup)
                "cdp_url": str,        # ws://host:port CDP websocket
                "features": dict,
            }
        """
        # External-URL mode: reuse a pre-existing Lightpanda CDP endpoint.
        # Skips all subprocess / container lifecycle management — useful when
        # Lightpanda runs in Docker and per-session spawns are too slow.
        external_url = os.environ.get("LIGHTPANDA_CDP_URL", "").strip()
        if external_url:
            session_id = uuid.uuid4().hex[:12]
            session_name = f"lightpanda_{task_id}_{session_id[:8]}"
            logger.info(
                "Using external Lightpanda session %s at %s",
                session_name, external_url,
            )
            return {
                "session_name": session_name,
                "bb_session_id": session_id,
                "cdp_url": external_url,
                "features": {"lightpanda": True, "external": True},
            }

        binary = self._find_binary()
        if binary is None:
            raise RuntimeError(
                "lightpanda binary not found. Install it with:\n"
                "  npm install -g @lightpanda/browser\n"
                "Or set LIGHTPANDA_PATH to point to the binary.\n"
                "Or set LIGHTPANDA_CDP_URL to point to a running Lightpanda instance."
            )

        host = os.environ.get("LIGHTPANDA_CDP_HOST", "127.0.0.1")
        port = self._pick_port()
        session_id = uuid.uuid4().hex[:12]
        session_name = f"lightpanda_{task_id}_{session_id[:8]}"

        cmd = self._build_serve_cmd(binary, host, port)
        logger.info(
            "Starting Lightpanda session %s on %s:%s",
            session_name, host, port,
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        startup_timeout = float(
            os.environ.get("LIGHTPANDA_STARTUP_TIMEOUT", str(_STARTUP_TIMEOUT_DEFAULT))
        )
        try:
            self._wait_for_cdp(host, port, startup_timeout)
        except RuntimeError:
            # Process started but CDP never came up — kill it before re-raising.
            proc.kill()
            proc.wait()
            raise

        with self._lock:
            self._processes[session_id] = proc

        cdp_url = f"ws://{host}:{port}"
        logger.info("Lightpanda session %s ready at %s", session_name, cdp_url)

        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": cdp_url,
            "features": {"lightpanda": True},
        }

    def close_session(self, session_id: str) -> bool:
        """Terminate the lightpanda process for *session_id*.

        Returns True on success, False if the session ID is unknown.
        External-URL sessions (LIGHTPANDA_CDP_URL) return True immediately
        since there is no per-session process to terminate.
        """
        with self._lock:
            proc = self._processes.pop(session_id, None)
        if proc is None:
            # External-URL mode has no tracked process; treat close as a no-op success.
            if os.environ.get("LIGHTPANDA_CDP_URL", "").strip():
                logger.debug("Lightpanda close_session (external mode): %s", session_id)
                return True
            logger.debug("Lightpanda close_session: unknown session_id %s", session_id)
            return False
        return _terminate_process(proc, session_id)

    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort kill on process exit — must not raise."""
        with self._lock:
            proc = self._processes.pop(session_id, None)
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception as exc:
            logger.debug(
                "Lightpanda emergency_cleanup error for session %s: %s", session_id, exc
            )

    def is_session_alive(self, session_id: str) -> bool:
        """Return True if the Lightpanda process for *session_id* is running.

        Returns True for external-URL sessions (no managed process) and for
        unknown session IDs (benefit of the doubt — may be external mode).
        """
        with self._lock:
            proc = self._processes.get(session_id)
        if proc is None:
            # External-URL mode or unknown — assume alive.
            return True
        return proc.poll() is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_binary(self) -> Optional[str]:
        """Return the path to the lightpanda binary, or None if not found.

        The ``@lightpanda/browser`` npm package stores the actual native binary
        at ``~/.cache/lightpanda-node/lightpanda`` and exposes a Node.js CLI
        wrapper in PATH.  The wrapper requires Node ≥ 20, so on older runtimes
        it crashes when invoked as a subprocess.  We therefore prefer the
        native binary directly over whatever PATH resolves to.
        """
        # 1. Explicit env-var override
        explicit = os.environ.get("LIGHTPANDA_PATH", "").strip()
        if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit

        # 2. Native binary downloaded by `@lightpanda/browser` npm postinstall.
        #    This is the actual compiled binary and works on any Node version.
        npm_cache_bin = Path.home() / ".cache" / "lightpanda-node" / "lightpanda"
        if npm_cache_bin.exists() and os.access(str(npm_cache_bin), os.X_OK):
            return str(npm_cache_bin)

        # 3. Global PATH search (may return a Node.js wrapper on some installs)
        found = shutil.which("lightpanda")
        if found:
            return found

        # 4. Local node_modules/.bin/ (npm install in project root)
        repo_root = Path(__file__).resolve().parent.parent.parent
        local_bin = repo_root / "node_modules" / ".bin" / "lightpanda"
        if local_bin.exists() and os.access(str(local_bin), os.X_OK):
            return str(local_bin)

        return None

    def _pick_port(self) -> int:
        """Return LIGHTPANDA_PORT if set, otherwise a random free port."""
        fixed = os.environ.get("LIGHTPANDA_PORT", "").strip()
        if fixed:
            try:
                return int(fixed)
            except ValueError:
                logger.warning("Invalid LIGHTPANDA_PORT=%r, using a random port", fixed)
        return _find_free_port()

    @staticmethod
    def _build_serve_cmd(binary: str, host: str, port: int) -> List[str]:
        """Return the subprocess argv to start lightpanda's CDP server."""
        return [binary, "serve", "--host", host, "--port", str(port)]

    @staticmethod
    def _wait_for_cdp(host: str, port: int, timeout: float) -> None:
        """Poll the CDP /json/version endpoint until it responds or *timeout* expires."""
        import urllib.error
        import urllib.request

        url = f"http://{host}:{port}/json/version"
        deadline = time.monotonic() + timeout
        while True:
            try:
                urllib.request.urlopen(url, timeout=1)
                return
            except (urllib.error.URLError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Lightpanda CDP server did not become ready on {host}:{port} "
                        f"within {timeout:.0f}s"
                    )
                time.sleep(_POLL_INTERVAL)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _find_free_port() -> int:
    """Return an ephemeral TCP port that is currently unbound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _terminate_process(proc: subprocess.Popen, session_id: str) -> bool:
    """Gracefully terminate *proc*; fall back to SIGKILL if needed."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
        logger.debug("Lightpanda session %s terminated gracefully", session_id)
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "Lightpanda session %s did not exit after SIGTERM — sending SIGKILL",
            session_id,
        )
        try:
            proc.kill()
            proc.wait(timeout=2)
            return True
        except Exception as exc:
            logger.error(
                "Failed to kill Lightpanda session %s: %s", session_id, exc
            )
            return False
    except Exception as exc:
        logger.error("Error terminating Lightpanda session %s: %s", session_id, exc)
        return False
