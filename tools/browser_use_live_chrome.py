"""``browser.live_chrome``: let ``browser_exec`` drive the user's own running, signed-in Chrome.

Chrome 144+ lets a user opt a running default-profile browser into remote debugging at
``chrome://inspect/#remote-debugging``; every CDP connection then raises its own "Allow remote debugging?"
popup. In this mode Hermes exports NO ``BU_CDP_*`` so browser-harness finds that Chrome itself, and this
module keeps the harness inside four limits:

* Hermes never launches or closes the user's Chrome. The harness autostarts a browser that is not running
  and has no switch to stop it, so ``preflight`` refuses before the harness runs.
* Consent stays with the user: nothing here invokes the harness's ``mac-approve`` (it AXPresses "Allow").
* A macOS privacy block on the browser's data folder is reported, never worked around.
* With ``browser.live_chrome_cdp_url`` set, ``browser_exec`` attaches to that fixed loopback DevTools
  endpoint (a dedicated agents' Chrome) through ``BU_CDP_*`` instead, and never discovers the user's own
  Chrome; no Allow popup is involved there.
* The CDP supervisor opens no second connection (a second Allow popup): ``_attach_vault_supervisor`` and
  ``ensure_cdp_supervisor`` only attach to an exported/override endpoint, and discovery live mode has
  neither. With ``browser.live_chrome_cdp_url`` the endpoint is exported and there is no popup, so the
  supervisor attaches there normally.
"""

from __future__ import annotations

import json
import os
import platform
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
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


# The own-tab marker: one file per (user, BU_NAME, daemon pid) in the harness process's temp dir, holding the
# id of the tab the preamble opened. Shared by the preamble (writes it) and the stop-time close (reads it) so
# both name the same file. The pid file is JSON (``{"pid", "started"}``) or a bare pid: only the integer
# goes into the name, which is then a valid file name on every platform.
_MARKER_PATH_SRC = """\
def _hermes_owntab_marker():
    import json as _json, os as _os, tempfile as _tf
    _name = _os.environ.get("BU_NAME", "default")
    try:
        from browser_harness import _ipc as _bipc
        _raw = _bipc.pid_path(_name).read_text(encoding="utf-8-sig").strip()
        _dpid = str(int(_json.loads(_raw)["pid"] if _raw.startswith("{") else _raw.split()[0]))
    except Exception:
        _dpid = "0"
    _uid = _os.getuid() if hasattr(_os, "getuid") else 0
    return _os.path.join(_tf.gettempdir(), "hermes-bu-owntab-%s-%s-%s" % (_uid, _name, _dpid))
"""

# Live Chrome's own-tab step FAILS CLOSED: a call that can't prove it is in a tab this daemon opened stops
# before the model's code runs. (Named harness daemons already attach to their own dedicated background tab
# rather than an existing page, but that is harness behaviour Hermes does not verify per call.) The marker is
# written only after that proof, so a failure is retried on the next call instead of pinning the session to a
# tab it did not open. (The shared-browser preamble in browser_use_cli stays best-effort.) The tab is created
# with background=True so Chrome does not come to the front.
LIVE_OWN_TAB_PREAMBLE = _MARKER_PATH_SRC + """\
# hermes: live Chrome -- work only in a tab this daemon opened; stop if that can't be confirmed
def _hermes_live_own_tab():
    import os as _os
    _marker = _hermes_owntab_marker()
    if _os.path.exists(_marker):
        return
    try:
        _tid = (cdp("Target.createTarget", url="about:blank", background=True) or {}).get("targetId")
        if not _tid:
            raise RuntimeError("Chrome returned no id for the new tab")
        switch_tab(_tid)
        _now = (current_tab() or {}).get("targetId")
        if _now != _tid:
            raise RuntimeError("still attached to another tab after switching")
    except Exception as _exc:
        raise RuntimeError(
            "LIVE CHROME STOPPED: Hermes could not open and attach to its own tab (%s), so nothing ran, "
            "to keep the user's own tabs untouched. Tell the user; do not retry until they say so." % (_exc,)
        ) from None
    try:
        with open(_marker, "w") as _fh:
            _fh.write(_tid)
    except OSError:
        pass  # no marker: the next call opens and checks a fresh tab again
_hermes_live_own_tab()
del _hermes_live_own_tab, _hermes_owntab_marker
"""

# Run through the harness CLI when Hermes stops a live daemon: closes ONLY the tab recorded in this daemon's
# marker (the one its preamble opened), then drops the marker. No marker, or a malformed one, closes nothing.
LIVE_CLOSE_OWN_TAB_CODE = _MARKER_PATH_SRC + """\
import os as _os, re as _re
_marker = _hermes_owntab_marker()
try:
    _tid = open(_marker).read().strip()
except OSError:
    _tid = ""
if _re.fullmatch(r"[A-Za-z0-9]+", _tid):
    try:
        cdp("Target.closeTarget", targetId=_tid)
    except Exception:
        pass
try:
    _os.remove(_marker)
except OSError:
    pass
"""

ENDPOINT_DESCRIPTION = (
    "\n\nLIVE CHROME: this tool drives a dedicated, already-running Chrome at {endpoint}, kept apart from "
    "the user's own browser. Each session works in a tab it opens itself and never touches existing tabs. "
    "Hermes never launches or closes that Chrome; if it is not reachable, tell the user to start it. Never "
    "try to approve any remote-debugging prompt yourself: do not run `browser-harness mac-approve` or any "
    "script that clicks one."
)


def own_tab_preamble(live: bool, shared_preamble: str) -> str:
    """The preamble that keeps a call in its own tab: the fail-closed one on the user's live Chrome."""
    return LIVE_OWN_TAB_PREAMBLE if live else shared_preamble


def live_chrome_enabled() -> bool:
    """``browser.live_chrome``. Read on EVERY call: it is a consent switch and each served profile decides."""
    return browser_cfg("live_chrome", False, lambda v: is_truthy_value(v, default=False))


def live_chrome_endpoint() -> str:
    """``browser.live_chrome_cdp_url`` ('' = discover the user's own Chrome). Only used with live mode on."""
    return browser_cfg("live_chrome_cdp_url", "", lambda v: str(v or "").strip())


def endpoint_format_error(url: str) -> Optional[str]:
    """None when ``url`` is a loopback http(s) URL, else why not (it carries a signed-in browser's debugger)."""
    try:
        parsed = urlparse(url)
        host, scheme, _ = (parsed.hostname or "").lower(), parsed.scheme, parsed.port  # port: ValueError if bad
    except ValueError:
        host = scheme = ""
    if scheme not in ("http", "https") or host not in ("localhost", "127.0.0.1", "::1"):
        return (f"browser.live_chrome_cdp_url must be a loopback http(s) URL such as http://127.0.0.1:9223; "
                f"got {url!r}. Hermes only attaches live Chrome to a debugger on this machine.")
    return None


def endpoint_unreachable(url: str) -> Optional[str]:
    """None when the DevTools endpoint answers ``/json/version``, else the error for the model. Read-only."""
    try:
        # loopback: never via a proxy, and no redirect handler so the probe cannot leave the endpoint
        opener = urllib.request.OpenerDirector()
        opener.add_handler(urllib.request.HTTPHandler())
        opener.add_handler(urllib.request.HTTPSHandler())
        opener.add_handler(urllib.request.HTTPErrorProcessor())
        opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
        with opener.open(url.rstrip("/") + "/json/version", timeout=3):
            return None
    except (OSError, ValueError, urllib.error.URLError):
        return (f"Live Chrome: the configured live Chrome endpoint {url} is not reachable, and Hermes never "
                "launches or closes Chrome. Ask the user to start that Chrome so it listens there, then retry.")


def approval_attempt(code: str, live_selected: bool) -> Optional[str]:
    """When live mode is the selected route, an error when ``code`` reaches for the harness's Allow-clicking
    helper, else None. Best-effort: code can spell the helper in ways no pattern catches, so the consent
    boundary is the tool description, this refusal and ``scrub_approval_hint`` together, not this alone."""
    if live_selected and _APPROVE_RE.search(code or ""):
        return ("Refused: the user must click Allow on Chrome's \"Allow remote debugging?\" popup themselves. "
                "Hermes never approves it for them. Ask the user to click Allow, then retry.")
    return None


# The harness's own stderr hint, "run `BU_NAME=… browser-harness mac-approve` in another shell or click Allow".
_APPROVE_HINT_RE = re.compile(r"run `[^`]*mac-approve[^`]*`(?: in another shell)? or click Allow", re.IGNORECASE)
_APPROVE_CMD_RE = re.compile(r"`[^`]*mac-approve[^`]*`", re.IGNORECASE)


def scrub_approval_hint(text: str) -> str:
    """Live mode: the harness's stderr hint offers its mac-approve helper, which would have the model click
    Allow. Rewrite it (and any other mention of that command) to say the user must click Allow."""
    text = _APPROVE_HINT_RE.sub("the user must click Allow", text)
    return _APPROVE_CMD_RE.sub("the user's own Allow click", text)


def live_chrome_timeout_hint(live: bool, endpoint: str = "") -> str:
    """A long hang in live mode is usually Chrome's Allow popup waiting for the user (the harness has no
    deadline). A fixed endpoint raises no popup, so there is nothing to say."""
    return (" Chrome may be waiting for the user to click Allow on its \"Allow remote debugging?\" popup; "
            "ask them.") if live and not endpoint else ""


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
    import psutil
    return psutil.pid_exists(pid)


def _windows_browser_running() -> bool:
    import psutil
    return any((proc.info["name"] or "").lower() in _WINDOWS_BROWSER_EXES
               for proc in psutil.process_iter(["name"]))


def _remote_debugging_toggled_on(base: Path) -> bool:
    """``Local State`` records chrome://inspect's "Allow remote debugging" toggle as on (the harness reads
    the same key). A privacy block raises PermissionError; any other unreadable file reads as off."""
    try:
        with open(base / "Local State", "rb") as fh:
            state = json.load(fh)
        return ((state.get("devtools") or {}).get("remote_debugging") or {}).get("user-enabled") is True
    except PermissionError:
        raise
    except (OSError, ValueError, AttributeError):
        return False


def _access_error(folder: Path, system: str) -> str:
    where = (" The app that launched Hermes needs access to it: open System Settings → Privacy & Security "
             "(Files and Folders, or Full Disk Access), enable it for that app, restart it, and retry."
             ) if system == "Darwin" else " The user running Hermes needs read access to it."
    return (f"Live Chrome: the operating system denied reading {folder} (Operation not permitted).{where} "
            "Hermes does not work around this protection. Ask the user to grant the access.")


_NOT_RUNNING = ("Live Chrome: no running Chrome was found, and Hermes never launches or closes the user's "
                "Chrome. Ask the user to open Chrome and turn on remote debugging at "
                "chrome://inspect/#remote-debugging (tick \"Allow remote debugging for this browser "
                "instance\"), then retry.")
_TOGGLE_OFF = ("Live Chrome: a browser is running, but none of them has remote debugging turned on, and "
               "Hermes never launches, closes or reconfigures the user's Chrome. Ask the user to turn it on "
               "in Chrome at chrome://inspect/#remote-debugging (tick \"Allow remote debugging for this "
               "browser instance\"), then retry.")
_NO_PSUTIL = ("Live Chrome: the psutil package is not installed here, so Hermes cannot tell whether Chrome "
              "is running. Live Chrome is unavailable on this install; tell the user.")


def preflight(system: Optional[str] = None) -> Optional[str]:
    """None when a running browser has remote debugging on (``Local State`` toggle and a
    ``DevToolsActivePort``) and nothing needed to tell is blocked, else the error for the model. The
    harness attaches to such a browser by itself; for anything else its setup branch would open or launch
    Chrome, which Hermes never allows. Read-only: never starts, stops or signals a browser. A privacy
    block on a browser's folder is reported only when no readable running browser was found."""
    system = system or platform.system()
    try:
        import psutil  # (not installed on every platform; pyproject marks it non-android)
    except ImportError:
        return _NO_PSUTIL
    running = False
    denied: Optional[Path] = None
    windows_running = system == "Windows" and _windows_browser_running()
    for base in _profile_dirs(system):
        try:
            if not base.is_dir():
                continue
            for name in ("Local State", "DevToolsActivePort"):
                _read_probe(base / name)
            alive = windows_running if system == "Windows" else _singleton_pid_alive(base)
            running = running or alive
            if alive and (base / "DevToolsActivePort").is_file() and _remote_debugging_toggled_on(base):
                return None
        except PermissionError:
            denied = denied or base
        except OSError:
            continue
    if denied:
        return _access_error(denied, system)
    return _TOGGLE_OFF if running else _NOT_RUNNING
