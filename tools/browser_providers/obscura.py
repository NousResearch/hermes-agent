"""Obscura local headless browser provider.

Resolves https://github.com/NousResearch/hermes-agent/issues/15445.

Obscura (https://github.com/h4ckf0r0day/obscura) is a Rust headless browser
that exposes a Chrome DevTools Protocol WebSocket and acts as a drop-in
replacement for headless Chrome with Puppeteer / Playwright.

Unlike the existing cloud providers (Browserbase, Browser Use, Firecrawl),
Obscura runs locally as a subprocess. The provider:

  * spawns ``obscura serve --port <PORT>`` per session,
  * waits for the CDP endpoint to come up,
  * returns the local CDP WebSocket URL,
  * terminates the subprocess on close / emergency cleanup.

Configuration (all optional):

  ``OBSCURA_BINARY_PATH``  Path to the Obscura binary (default: ``obscura``).
  ``OBSCURA_PORT``         CDP port. Default: auto-allocate an ephemeral port.
  ``OBSCURA_STEALTH``      ``true`` to enable Obscura's stealth flag (default).
  ``OBSCURA_STARTUP_TIMEOUT``  Seconds to wait for CDP readiness (default 15).

Because Obscura is local, it has no per-session API cost and no network
latency to a cloud broker. Memory footprint is ~30 MB per session vs
~200 MB for headless Chrome, which matters on small hosts running an
LLM in parallel.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, Optional

import requests

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)


def _allocate_ephemeral_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then close and return it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


class ObscuraProvider(CloudBrowserProvider):
    """Obscura local headless browser backend.

    The "Cloud" in :class:`CloudBrowserProvider` is a slight stretch for a
    local subprocess, but the contract is identical from the caller's
    perspective: ``create_session`` returns a CDP URL the rest of
    ``browser_tool`` connects to.
    """

    def __init__(self) -> None:
        # session_id -> Popen handle
        self._processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def provider_name(self) -> str:
        return "Obscura"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _binary_path(self) -> str:
        return os.environ.get("OBSCURA_BINARY_PATH", "obscura")

    def _stealth_enabled(self) -> bool:
        return os.environ.get("OBSCURA_STEALTH", "true").lower() != "false"

    def _startup_timeout(self) -> float:
        try:
            return float(os.environ.get("OBSCURA_STARTUP_TIMEOUT", "15"))
        except ValueError:
            return 15.0

    def _configured_port(self) -> Optional[int]:
        raw = os.environ.get("OBSCURA_PORT")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid OBSCURA_PORT=%r, falling back to auto-allocate", raw)
            return None

    def is_configured(self) -> bool:
        # is_configured must be cheap (no subprocess launch). We just check
        # whether the binary is resolvable. shutil.which handles both PATH
        # lookup and absolute paths correctly.
        return shutil.which(self._binary_path()) is not None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _wait_for_cdp(self, port: int, deadline: float) -> Optional[str]:
        """Poll the CDP /json/version endpoint until it responds or the deadline passes.

        Returns the WebSocket debugger URL on success, None on timeout.
        """
        url = f"http://127.0.0.1:{port}/json/version"
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=1.5)
                if resp.ok:
                    data = resp.json()
                    ws_url = data.get("webSocketDebuggerUrl")
                    if ws_url:
                        return ws_url
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.25)
        return None

    def create_session(self, task_id: str) -> Dict[str, object]:
        binary = self._binary_path()
        if shutil.which(binary) is None:
            raise ValueError(
                f"Obscura binary not found at {binary!r}. "
                "Install Obscura (https://github.com/h4ckf0r0day/obscura) "
                "or set OBSCURA_BINARY_PATH."
            )

        port = self._configured_port() or _allocate_ephemeral_port()
        cmd = [binary, "serve", "--port", str(port)]
        if self._stealth_enabled():
            cmd.append("--stealth")

        logger.info("Launching Obscura: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise ValueError(f"Failed to launch Obscura ({binary!r}): {e}") from e

        deadline = time.monotonic() + self._startup_timeout()
        ws_url = self._wait_for_cdp(port, deadline)
        if ws_url is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            err_tail = ""
            if proc.stderr is not None:
                try:
                    err_tail = proc.stderr.read().decode("utf-8", errors="replace")[-500:]
                except Exception:
                    pass
            raise RuntimeError(
                f"Obscura did not become ready on port {port} within "
                f"{self._startup_timeout()}s. stderr tail: {err_tail!r}"
            )

        session_id = uuid.uuid4().hex
        with self._lock:
            self._processes[session_id] = proc

        session_name = f"hermes_{task_id}_{session_id[:8]}"
        logger.info(
            "Obscura session %s ready on port %s (stealth=%s)",
            session_name,
            port,
            self._stealth_enabled(),
        )

        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": ws_url,
            "features": {
                "obscura": True,
                "stealth": self._stealth_enabled(),
                "local": True,
                "port": port,
            },
        }

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _terminate(self, session_id: str, force: bool) -> bool:
        with self._lock:
            proc = self._processes.pop(session_id, None)
        if proc is None:
            return False
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            return True
        except Exception as e:
            logger.warning("Error tearing down Obscura session %s: %s", session_id, e)
            return False

    def close_session(self, session_id: str) -> bool:
        ok = self._terminate(session_id, force=False)
        if ok:
            logger.debug("Closed Obscura session %s", session_id)
        else:
            logger.warning("Could not close Obscura session %s (unknown id)", session_id)
        return ok

    def emergency_cleanup(self, session_id: str) -> None:
        try:
            self._terminate(session_id, force=True)
        except Exception as e:
            logger.debug("Emergency cleanup failed for Obscura session %s: %s", session_id, e)
