"""Provision a local headless Chrome for the Browser Use backend on headless Linux.

The browser-use harness's "local" mode assumes a GUI: it launches a windowed
Chrome and waits for the user to click "Allow remote debugging". On a headless
Linux VPS there is no ``DISPLAY``, so that path can never work and
``browser_exec`` fails with "chrome-not-running".

This module closes that gap. When Hermes runs on headless Linux with no
cloud/CDP backend configured, it provisions a detached headless Chrome on a
local CDP port (a non-default ``--user-data-dir``, since Chrome 136+ refuses
``--remote-debugging-port`` on the default profile) and points the harness at
it via ``BU_CDP_URL``. Concurrent calls reuse the already-running Chrome
instead of spawning duplicates.
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Fixed port so later calls detect and reuse an already-running Chrome. It
# matches the browser-harness daemon's own default local probe port.
_HEADLESS_PORT = 9222
_CDP_URL = f"http://127.0.0.1:{_HEADLESS_PORT}"

_SYSTEM_CHROME_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def is_headless_linux() -> bool:
    """True on Linux with no display server (a VPS/container over SSH)."""
    return (
        sys.platform == "linux"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )


def _profile_dir() -> str:
    from hermes_constants import get_hermes_home

    return os.path.join(get_hermes_home(), "cache", "browser-use", "headless-chrome")


def _find_chromium_binary() -> Optional[str]:
    """Locate a Chrome/Chromium binary: env override → PATH → Playwright cache."""
    for key in ("AGENT_BROWSER_EXECUTABLE_PATH", "BH_CHROME_PATH", "CHROME_PATH"):
        path = (os.environ.get(key) or "").strip()
        if path and os.path.isfile(path):
            return path

    for name in _SYSTEM_CHROME_NAMES:
        path = shutil.which(name)
        if path:
            return path

    from tools.browser_tool import _chromium_search_roots

    for root in _chromium_search_roots():
        for pattern in (
            os.path.join("chromium-*", "chrome-linux*", "chrome"),
            os.path.join("chromium_headless_shell-*", "chrome-linux*", "headless_shell"),
        ):
            try:
                hits = sorted(glob.glob(os.path.join(root, pattern)))
            except OSError:
                continue
            if hits:
                return hits[-1]
    return None


def _cdp_ready(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(_CDP_URL + "/json/version", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _launch(binary: str) -> bool:
    from tools.browser_tool import _needs_chromium_sandbox_bypass

    profile = _profile_dir()
    os.makedirs(profile, exist_ok=True)

    args = [binary]
    if _needs_chromium_sandbox_bypass():
        args.append("--no-sandbox")
    args += [
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-sync",
        "--metrics-recording-only",
        "--mute-audio",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={_HEADLESS_PORT}",
        f"--user-data-dir={profile}",
        "about:blank",
    ]

    try:
        # Detach: the Chrome must outlive the browser_exec subprocess (and even
        # the agent process) so subsequent calls reuse it.
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except OSError as e:
        logger.warning("browser: headless Chrome launch failed: %s", e)
        return False


def ensure_headless_chrome(env: dict) -> Optional[str]:
    """Ensure a headless Chrome is serving CDP and set ``BU_CDP_URL`` on ``env``.

    No-op when a display server is present (the harness's GUI path works) or
    when a CDP endpoint is already resolved (explicit override / cloud). On
    failure returns a human-readable error string; on success/no-op returns
    None.
    """
    if not is_headless_linux():
        return None
    # Already resolved: explicit CDP override, a cloud provider, or Browser Use
    # cloud autospawn (BU_AUTOSPAWN) — the harness will attach elsewhere.
    if env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or env.get("BU_AUTOSPAWN"):
        return None

    # Already running (launched by a previous call) — just point at it.
    if _cdp_ready():
        env["BU_CDP_URL"] = _CDP_URL
        return None

    binary = _find_chromium_binary()
    if not binary:
        from tools.browser_tool import _chromium_installed, _maybe_autoinstall_chromium

        if _chromium_installed():
            return (
                "browser-use: a Chromium browser was detected but its binary path "
                "could not be resolved; set browser.cdp_url (or BU_CDP_URL) to a "
                "running browser."
            )
        if not _maybe_autoinstall_chromium():
            return (
                "browser-use: no Chrome/Chromium browser is available on this "
                "headless server. Install one (e.g. `sudo apt-get install -y "
                "chromium`) or set browser.cdp_url / BU_CDP_URL to an existing "
                "browser."
            )
        binary = _find_chromium_binary()
        if not binary:
            return (
                "browser-use: Chromium auto-install completed but no binary could "
                "be resolved; set browser.cdp_url / BU_CDP_URL manually."
            )

    if not _launch(binary):
        return f"browser-use: failed to launch headless Chrome ({binary})."

    deadline = time.time() + 20.0
    while time.time() < deadline:
        if _cdp_ready():
            env["BU_CDP_URL"] = _CDP_URL
            return None
        time.sleep(0.2)

    return (
        "browser-use: launched headless Chrome but its CDP endpoint did not come "
        "up; set browser.cdp_url / BU_CDP_URL manually."
    )
