"""``browser.live_chrome``: let ``browser_exec`` drive the user's own running, signed-in Chrome.

Chrome 144+ lets a user opt a running default-profile browser into remote debugging at
``chrome://inspect/#remote-debugging``; every CDP connection then raises its own "Allow remote debugging?"
popup. In this mode Hermes exports NO ``BU_CDP_*`` so browser-harness finds that Chrome itself, and this
module keeps the harness inside four limits:

* Hermes never launches or closes the user's Chrome. The harness autostarts a browser that is not running
  and has no switch to stop it, so ``preflight`` refuses before the harness runs.
* Consent stays with the user: nothing here invokes the harness's ``mac-approve`` (it AXPresses "Allow").
* A macOS privacy block on the browser's data folder is reported, never worked around.
* The CDP supervisor opens no second connection (a second Allow popup): ``_attach_vault_supervisor`` and
  ``ensure_cdp_supervisor`` only attach to an exported/override endpoint, and live mode has neither.
"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Optional

from tools.browser_session_cloud import browser_cfg
from utils import is_truthy_value

# Internal route provenance on the env dict (popped before launching the CLI): this exec drives the live Chrome.
LIVE_CHROME_SENTINEL = "_HERMES_BU_LIVE_CHROME"

# The harness's own user-data-dir list (browser_harness.daemon.PROFILES), copied because importing that
# module reads .env files into os.environ and creates harness dirs as a side effect.
_PROFILES = {
    "Darwin": (
        "Library/Application Support/Google/Chrome", "Library/Application Support/Google/Chrome Canary",
        "Library/Application Support/Comet", "Library/Application Support/Arc/User Data",
        "Library/Application Support/Dia/User Data", "Library/Application Support/Microsoft Edge",
        "Library/Application Support/Microsoft Edge Beta", "Library/Application Support/Microsoft Edge Dev",
        "Library/Application Support/Microsoft Edge Canary",
        "Library/Application Support/BraveSoftware/Brave-Browser",
        "Library/Application Support/BraveSoftware/Brave-Origin",
    ),
    "Linux": (
        ".config/google-chrome", ".config/chromium", ".config/chromium-browser", ".config/microsoft-edge",
        ".config/microsoft-edge-beta", ".config/microsoft-edge-dev",
        ".var/app/org.chromium.Chromium/config/chromium", ".var/app/com.google.Chrome/config/google-chrome",
        ".var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser",
        ".var/app/com.microsoft.Edge/config/microsoft-edge",
    ),
    "Windows": (
        "Google/Chrome/User Data", "Google/Chrome SxS/User Data", "Google/Chrome Beta/User Data",
        "Google/Chrome Dev/User Data", "Chromium/User Data", "Microsoft/Edge/User Data",
        "Microsoft/Edge Beta/User Data", "Microsoft/Edge Dev/User Data", "Microsoft/Edge SxS/User Data",
        "BraveSoftware/Brave-Browser/User Data",
    ),
}
_WINDOWS_BROWSER_EXES = ("chrome.exe", "msedge.exe", "chromium.exe", "brave.exe", "helium.exe")

# Code naming the harness's approval helper is refused: the user clicks Allow, never the model.
_APPROVE_RE = re.compile(r"mac[-_]approve|browser_harness\.macos|from\s+browser_harness\s+import\s+macos", re.IGNORECASE)

DESCRIPTION = (
    "\n\nLIVE CHROME: this tool drives the user's OWN running Chrome, signed in as them. Each session works "
    "in a tab it opens itself and never touches their existing tabs. Chrome shows an \"Allow remote "
    "debugging?\" popup for each connection and the USER clicks Allow. Never try to approve it yourself: do "
    "not run `browser-harness mac-approve` or any script that clicks it, even if a harness message suggests "
    "it. If a call reports permission-blocked or remote-debugging-setup, relay the message to the user, wait "
    "for them to confirm, and only then retry. If Chrome is not running, ask the user to open it and enable "
    "chrome://inspect/#remote-debugging; Hermes never launches or closes their Chrome."
)


def live_chrome_enabled() -> bool:
    """``browser.live_chrome``. Read on EVERY call: it is a consent switch and each served profile decides."""
    return browser_cfg("live_chrome", False, lambda v: is_truthy_value(v, default=False))


def approval_attempt(code: str) -> Optional[str]:
    """In live mode, an error when ``code`` reaches for the harness's Allow-clicking helper, else None."""
    if live_chrome_enabled() and _APPROVE_RE.search(code or ""):
        return ("Refused: the user must click Allow on Chrome's \"Allow remote debugging?\" popup themselves. "
                "Hermes never approves it for them. Ask the user to click Allow, then retry.")
    return None


def live_chrome_timeout_hint(live: bool) -> str:
    """A long hang in live mode is usually Chrome's Allow popup waiting for the user (the harness has no deadline)."""
    return (" Chrome may be waiting for the user to click Allow on its \"Allow remote debugging?\" popup; "
            "ask them.") if live else ""


def _profile_dirs(system: str) -> list[Path]:
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    else:
        base = Path.home()
    return [base / rel for rel in _PROFILES.get(system, _PROFILES["Linux"])]


def _read_probe(path: Path) -> None:
    """Open ``path`` for reading; a missing file is fine, a macOS privacy block raises PermissionError."""
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except (FileNotFoundError, NotADirectoryError):
        pass


def _singleton_pid_alive(base: Path) -> bool:
    """A running browser holds ``base`` through the SingletonLock symlink ``<host>-<pid>`` (POSIX)."""
    try:
        pid = int(os.readlink(base / "SingletonLock").rsplit("-", 1)[-1])
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return False
    import psutil  # lazy: not installed on every platform (pyproject marks it non-android)
    return psutil.pid_exists(pid)


def _windows_browser_running() -> bool:
    import psutil
    return any((proc.info["name"] or "").lower() in _WINDOWS_BROWSER_EXES
               for proc in psutil.process_iter(["name"]))


def _access_error(folder: Path, system: str) -> str:
    where = (" The app that launched Hermes needs access to it: open System Settings → Privacy & Security "
             "(Files and Folders, or Full Disk Access), enable it for that app, restart it, and retry."
             ) if system == "Darwin" else " The user running Hermes needs read access to it."
    return (f"Live Chrome: the operating system denied reading {folder} (Operation not permitted).{where} "
            "Hermes does not work around this protection. Ask the user to grant the access.")


def preflight(system: Optional[str] = None) -> Optional[str]:
    """None when a supported browser is running and its profile folders are readable, else the error for
    the model. Read-only: never starts, stops or signals a browser."""
    system = system or platform.system()
    running = False
    for base in _profile_dirs(system):
        try:
            if not base.is_dir():
                continue
            for name in ("Local State", "DevToolsActivePort"):
                _read_probe(base / name)
            running = running or (system != "Windows" and _singleton_pid_alive(base))
        except PermissionError:
            return _access_error(base, system)
        except OSError:
            continue
    if system == "Windows":
        running = _windows_browser_running()
    if running:
        return None
    return ("Live Chrome: no running Chrome was found, and Hermes never launches or closes the user's "
            "Chrome. Ask the user to open Chrome and turn on remote debugging at "
            "chrome://inspect/#remote-debugging (tick \"Allow remote debugging for this browser instance\"), "
            "then retry.")
