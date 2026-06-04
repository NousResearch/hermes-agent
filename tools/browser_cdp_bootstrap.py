"""
Auto-bootstrap a local Chrome/Chromium with CDP debugging enabled.

Mirrors the `/browser connect` logic from `cli.py` so that tool layers
(e.g. `browser_navigate`) can ensure the debug Chrome is reachable
before issuing CDP commands.

Behaviour:
  1. Parse host/port from a CDP URL (defaults to http://localhost:9223).
  2. If the port is already open, set BROWSER_CDP_URL and return.
  3. Otherwise discover a Chrome/Chromium binary and launch it with the
     same flags `/browser connect` uses (remote-allow-origins, persist
     cookies, dedicated user-data-dir, etc.).
  4. Wait up to ~10s for the port to come up.
  5. Optionally inject cookies via `inject_cookies.chromium_cookie`.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CDP_URL = "http://localhost:9223"
DEFAULT_PORT = 9223


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _chrome_candidates(system: str) -> List[str]:
    """Discover Chrome/Chromium executables (subset of cli.py logic)."""
    if system == "Darwin":
        paths = [
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    elif system == "Windows":
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    else:  # Linux
        paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ]
    return [p for p in paths if os.path.isfile(p)]


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
        return True
    except Exception:
        return False


def _launch_chrome_debug(port: int) -> bool:
    """Spawn Chrome with the same flags as `/browser connect`."""
    system = platform.system()
    candidates = _chrome_candidates(system)
    if not candidates:
        logger.warning("CDP bootstrap: no chrome binary found on %s", system)
        return False

    chrome = candidates[0]
    data_dir = str(_hermes_home() / "chrome-debug")
    os.makedirs(data_dir, exist_ok=True)
    logger.info("CDP bootstrap: launching %s with port=%s data_dir=%s",
                chrome, port, data_dir)
    try:
        subprocess.Popen(
            [
                chrome,
                f"--remote-debugging-port={port}",
                f"--user-data-dir={data_dir}",
                "--remote-allow-origins=*",
                "--disable-features=ClearSiteDataOnExit",
                "--disable-clear-browsing-data-on-exit",
                "--persist-cookies",
                "--no-first-run",
                "--no-default-browser-check",
                "--password-store=basic",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception as e:
        logger.warning("CDP bootstrap: chrome launch failed: %s", e)
        return False


def _wait_port(host: str, port: int, retries: int = 20, delay: float = 0.5) -> bool:
    for _ in range(retries):
        if _port_open(host, port):
            return True
        time.sleep(delay)
    return False


def _try_inject_cookies(host: str, port: int) -> None:
    try:
        from inject_cookies import chromium_cookie
        chromium_cookie(["--host", host, "--port", str(port)])
    except Exception as e:
        logger.warning("CDP bootstrap: cookie injection skipped/failed: %s", e)


def _parse_host_port(cdp_url: str) -> tuple[str, int]:
    host = "localhost"
    port = DEFAULT_PORT
    try:
        body = cdp_url.split("://", 1)[-1]
        netloc = body.split("/", 1)[0]
        if ":" in netloc:
            host_part, port_part = netloc.rsplit(":", 1)
            host = host_part or "localhost"
            port = int(port_part)
        else:
            host = netloc or "localhost"
    except Exception:
        pass
    return host, port


def ensure_cdp_browser_ready(
    cdp_url: Optional[str] = None,
    inject_cookies: bool = True,
) -> str:
    """
    Ensure a CDP-enabled Chrome is reachable; auto-launch if not.

    Args:
        cdp_url: Target CDP endpoint. Defaults to BROWSER_CDP_URL env, then
            ``http://localhost:9223``.
        inject_cookies: If True, run `inject_cookies.chromium_cookie` to push
            stored cookies into the freshly-launched browser.

    Returns:
        The resolved CDP URL (also written to BROWSER_CDP_URL).

    Raises:
        RuntimeError: if Chrome cannot be launched or the port never opens.
    """
    cdp_url = (cdp_url or os.environ.get("BROWSER_CDP_URL") or DEFAULT_CDP_URL).strip()
    host, port = _parse_host_port(cdp_url)

    if _port_open(host, port):
        logger.debug("CDP bootstrap: %s:%d already responsive — skip launch", host, port)
        os.environ["BROWSER_CDP_URL"] = cdp_url
        return cdp_url

    logger.info("CDP bootstrap: %s:%d not responding, launching Chrome", host, port)
    if not _launch_chrome_debug(port):
        raise RuntimeError(f"CDP bootstrap failed: no chrome binary or launch error (port={port})")
    if not _wait_port(host, port):
        raise RuntimeError(f"CDP bootstrap failed: chrome launched but port {port} never came up")

    os.environ["BROWSER_CDP_URL"] = cdp_url
    if inject_cookies:
        _try_inject_cookies(host, port)
    return cdp_url
