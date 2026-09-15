"""Real-profile local browsing: snapshot the user's default Chromium profile into a
hermes-owned copy, launch the real browser binary on it, and attach agent-browser.

State (``_REAL_PROFILE_SESSION``, ``_real_profile_cdp_lock``, ``_real_profile_cdp_cache``,
``_real_profile_chrome_procs``) lives in ``tools.browser_tool``; it is read
through ``_bt`` (resolved per call — never import ``tools.browser_tool`` at import time).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_install as _install
from tools import browser_tool_lightpanda_fallback as _lp
from tools import browser_tool_session as _session

_RP = "browser.use_real_profile is on, but "


def _terminate_real_profile_chrome() -> None:
    """Terminate real-browser processes launched for real-profile sessions (idempotent, atexit-safe);
    agent-browser only ATTACHED to them, so its own session cleanup never kills them."""
    from tools.browser_lightpanda import _terminate
    _bt = _origin()
    while _bt._real_profile_chrome_procs:
        _terminate(_bt._real_profile_chrome_procs.pop(), what="real-profile chrome")


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    from tools.browser_lightpanda import _cdp_ready
    return _cdp_ready(http_cdp, timeout=1.0)


def _real_profile_daemon_env() -> dict:
    """Reaper-visible socket dir + ``owner_pid`` claim like every other lane (agent-browser's
    default dir is invisible to the reaper — #100855). The daemon-side idle timeout is dropped:
    Chrome is launched by Hermes, not the daemon, so a self-exiting daemon would leave Chrome
    holding the copy dir under the next snapshot overlay."""
    _bt = _origin()
    socket_dir = _session._prepare_session_socket_dir(_bt._REAL_PROFILE_SESSION)
    env = _session._agent_browser_command_env(socket_dir)
    env.pop("AGENT_BROWSER_IDLE_TIMEOUT_MS", None)
    return env


def _agent_browser_session_cmd(session_name: str, *cmd: str, log_label: str) -> Optional[subprocess.CompletedProcess]:
    """Run ``agent-browser --session <name> <cmd...>``; None when agent-browser is missing or the run fails."""
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        return subprocess.run([*_session._agent_browser_argv(browser_cmd), "--session", session_name, *cmd],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                              env=_real_profile_daemon_env(), stdin=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError) as e:
        _bt.logger.debug("real-profile %s failed: %s", log_label, e)
        return None


def _agent_browser_get_cdp(session_name: str) -> Optional[str]:
    """HTTP CDP discovery root of an agent-browser session (from its ``ws://`` cdp-url), or None."""
    proc = _agent_browser_session_cmd(session_name, "get", "cdp-url", log_label="get cdp-url")
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", (proc.stdout or "").strip()) if proc is not None else None
    return f"http://127.0.0.1:{m.group(1)}" if m else None


def _read_devtools_port(data_dir: str) -> Optional[str]:
    """First line of Chrome's ``DevToolsActivePort`` in ``data_dir`` (None when unreadable)."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            return fh.readline().strip()
    except OSError:
        return None


def _surviving_chrome_cdp(data_dir: str) -> Optional[str]:
    """HTTP CDP root of a Chrome still running on ``data_dir``, or None. ``DevToolsActivePort``
    outlives a crashed Chrome and its port can be recycled by another local CDP server, so the
    file's browser id (line 2) must match what ``/json/version`` reports before it is trusted."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port, browser_path = fh.readline().strip(), fh.readline().strip()
    except OSError:
        return None
    if not port.isdigit() or not browser_path.startswith("/devtools/browser/"):
        return None
    http_cdp = f"http://127.0.0.1:{port}"
    try:
        import requests
        ws_url = str(requests.get(f"{http_cdp}/json/version", timeout=2).json().get("webSocketDebuggerUrl") or "")
    except Exception:
        return None
    return http_cdp if ws_url.endswith(browser_path) else None


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser runs on ``data_dir`` (DevToolsActivePort match proves it
    is our profile copy, not a throwaway temp dir a raced/stale launch fell back to)."""
    m = re.search(r":(\d+)", http_cdp or "")
    return bool(m) and _read_devtools_port(data_dir) == m.group(1)


def _agent_browser_close_session(session_name: str) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    _agent_browser_session_cmd(session_name, "close", log_label="session close")


_REAL_PROFILE_CHROME_FLAGS = (
    "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
    "--disable-background-networking", "--disable-component-update", "--disable-default-apps",
    "--disable-hang-monitor", "--disable-popup-blocking", "--disable-prompt-on-repost",
    "--disable-sync", "--disable-features=Translate", "--no-startup-window",
)


def _real_profile_unsupported_reason(browser) -> Optional[str]:
    """Fail-closed message when the default browser can't be used, else None.

    A pre-release channel lives in a profile dir we don't resolve; normalizing to the stable
    family would drive a DIFFERENT profile/account (wrong-principal bug), so refuse rather than guess.
    """
    from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL
    if browser is None:
        return (_RP + "your default browser is not a supported Chromium browser (Chrome, Edge, Brave, "
                "Brave Origin, Chromium). Real-profile browsing requires a Chromium default; set one or turn the toggle off.")
    if browser == UNSUPPORTED_CHANNEL:
        return (_RP + "your default browser is a pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn the toggle off.")
    return None


def _real_profile_snapshot_error(err: str) -> str:
    """User-facing message for a failed profile snapshot; a locked profile adds the approved-close
    command, which the agent must ASK the user about first (it quits their browser)."""
    from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX
    if err and err.startswith(_PROFILE_LOCKED_PREFIX):
        return (err[len(_PROFILE_LOCKED_PREFIX):] + " To close it (only after the user approves — it "
                "quits their browser and loses unsaved tabs), run: `hermes browser close-profile`, then retry.")
    return f"{_RP}{err}"


def _read_real_profile_headed_mode(copy_dir: str) -> Optional[bool]:
    """Read the persisted effective mode for a managed profile runtime."""
    try:
        value = Path(copy_dir, ".hermes-browser-mode").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return True if value == "headed" else False if value == "headless" else None


def _mode_conflict(requested: Optional[bool], effective: bool, running: Optional[bool]) -> Optional[str]:
    if running is None:
        return ("The Hermes real-profile browser is already running, but its headed mode cannot be "
                "verified. Close it and retry to apply an effective headed value safely.") if requested is not None else None
    if effective != running:
        return (f"The Hermes real-profile browser is already running {'headed' if running else 'headless'}; "
                f"it cannot be reused as {'headed' if effective else 'headless'}. Close the existing browser "
                "session or use the same headed value, then retry.")
    return None


def _launch_real_profile_chrome(real_binary: str, copy_dir: str, effective_headed: bool,
                                explicit_headed: Optional[bool]) -> Tuple[Optional[int], Optional[str]]:
    """Launch the user's real browser binary on the profile copy."""
    _bt = _origin()
    try:
        os.unlink(os.path.join(copy_dir, "DevToolsActivePort"))
    except OSError:
        pass
    chrome_argv = [real_binary, f"--user-data-dir={copy_dir}", *_REAL_PROFILE_CHROME_FLAGS]
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    want_headed = effective_headed and (has_display or not sys.platform.startswith("linux"))
    if explicit_headed is True and not want_headed:
        return None, "headed=true requires a graphical display, but no DISPLAY or WAYLAND_DISPLAY is available on this Linux host."
    if not want_headed:
        chrome_argv.append("--headless=new")
    try:
        Path(copy_dir, ".hermes-browser-mode").write_text("headed" if want_headed else "headless", encoding="utf-8")
    except OSError as e:
        return None, f"{_RP}mode state could not be saved: {e}"
    try:
        chrome_proc = subprocess.Popen(chrome_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=_bt._build_browser_env())
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    _bt._real_profile_chrome_procs.append(chrome_proc)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        line = _read_devtools_port(copy_dir) or ""
        if line.isdigit():
            return int(line), None
        if chrome_proc.poll() is not None:
            _terminate_real_profile_chrome()
            return None, _RP + "Chrome exited during startup (another instance may hold the profile copy)."
        time.sleep(0.25)
    _terminate_real_profile_chrome()
    return None, _RP + "the real-profile browser did not expose a debug port in time. Retry, or turn the toggle off."


def _attach_agent_browser_to_real_profile(port: int, copy_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Make agent-browser ATTACH to the running Chrome (never launch its own); returns ``(http_cdp, error)``.

    The daemon may answer with the endpoint of a browser IT spawned (throwaway temp profile);
    the DevToolsActivePort OUR Chrome wrote is authoritative on disagreement.
    """
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError as e:
        return None, f"{_RP}the local browser engine (agent-browser) is not installed: {e}"
    argv = [*_session._agent_browser_argv(browser_cmd), "--session", _bt._REAL_PROFILE_SESSION,
            "--cdp", str(port), "open", "about:blank"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=_bt._get_open_command_timeout(first_open=True), env=_real_profile_daemon_env(),
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, _RP + "the real-profile browser took too long to start. Retry, or turn the toggle off."
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"{_RP}the real-profile browser failed to start: {tail[-1] if tail else f'exit {proc.returncode}'}"
    cdp = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
    our_port = _read_devtools_port(copy_dir)
    if our_port is not None and (m := re.search(r":(\d+)", cdp or "")) and m.group(1) != our_port:
        cdp = f"http://127.0.0.1:{our_port}"
    if not cdp:
        return None, _RP + "the real-profile browser started without exposing a devtools endpoint. Retry, or turn the toggle off."
    return cdp, None


def _real_profile_cdp(headed: Optional[bool] = None) -> tuple:
    """Resolve a managed real-profile CDP endpoint, honoring an optional per-runtime mode."""
    _bt = _origin()
    if not _cloud._use_real_profile():
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots
            cleanup_real_profile_snapshots()
        except Exception as e:
            _bt.logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _bt._real_profile_cdp_cache.clear()
        return None, None
    if _lp._using_lightpanda_engine():
        return None, (_RP + "browser.engine is set to 'lightpanda', which cannot load a real Chromium profile. "
                      "Set browser.engine to 'auto' or 'chrome' to use real-profile browsing, or turn the toggle off.")

    effective_headed = _cloud._is_headed_mode() if headed is None else headed
    from hermes_cli.browser_connect import (chromium_executable, detect_default_chromium,
                                            real_profile_copy_dir, snapshot_real_profile)
    with _bt._real_profile_cdp_lock:
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if cached and _cdp_http_ready(cached):
            conflict = _mode_conflict(headed, effective_headed, _bt._real_profile_cdp_cache.get("headed"))
            if conflict:
                return None, conflict
            _session._prepare_session_socket_dir(_bt._REAL_PROFILE_SESSION)
            return cached, None
        _bt._real_profile_cdp_cache.clear()

        browser = detect_default_chromium()
        unsupported = _real_profile_unsupported_reason(browser)
        if unsupported:
            return None, unsupported
        copy_dir = real_profile_copy_dir(browser)
        existing = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
        if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
            existing_headed = _read_real_profile_headed_mode(copy_dir)
            conflict = _mode_conflict(headed, effective_headed, existing_headed)
            if conflict:
                return None, conflict
            _bt._real_profile_cdp_cache["cdp"] = existing
            if existing_headed is not None:
                _bt._real_profile_cdp_cache["headed"] = existing_headed
            return existing, None
        if existing:
            _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        surviving = _surviving_chrome_cdp(copy_dir)
        if surviving:
            surviving_headed = _read_real_profile_headed_mode(copy_dir)
            conflict = _mode_conflict(headed, effective_headed, surviving_headed)
            if conflict:
                return None, conflict
            cdp, err = _attach_agent_browser_to_real_profile(int(surviving.rsplit(":", 1)[1]), copy_dir)
            if not cdp:
                return None, err
            _bt._real_profile_cdp_cache["cdp"] = cdp
            if surviving_headed is not None:
                _bt._real_profile_cdp_cache["headed"] = surviving_headed
            _bt.logger.info("real-profile: re-attached to surviving Chrome at %s (%s)", cdp, copy_dir)
            return cdp, None

        copy_dir, err = snapshot_real_profile(browser)
        if err or not copy_dir:
            return None, _real_profile_snapshot_error(err)
        real_binary = chromium_executable(browser)
        if real_binary is None:
            return None, f"{_RP}the real browser binary for '{browser}' could not be found. Reinstall it or turn the toggle off."
        port, err = _launch_real_profile_chrome(real_binary, copy_dir, effective_headed, headed)
        if port is None:
            return None, err
        cdp, err = _attach_agent_browser_to_real_profile(port, copy_dir)
        if not cdp:
            return None, err
        _bt._real_profile_cdp_cache.update(cdp=cdp, headed=effective_headed)
        _bt.logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None


def _preserve_browser_between_turns() -> bool:
    """Effective persistence mode for a live managed real-profile runtime."""
    _bt = _origin()
    cached = _bt._real_profile_cdp_cache.get("cdp")
    if cached and _cdp_http_ready(cached):
        runtime_headed = _bt._real_profile_cdp_cache.get("headed")
        if isinstance(runtime_headed, bool):
            return runtime_headed
    if _cloud._use_real_profile() and not _lp._using_lightpanda_engine():
        try:
            from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL, detect_default_chromium, real_profile_copy_dir
            browser = detect_default_chromium()
            if browser and browser != UNSUPPORTED_CHANNEL:
                copy_dir = real_profile_copy_dir(browser)
                existing = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
                if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
                    persisted = _read_real_profile_headed_mode(copy_dir)
                    if isinstance(persisted, bool):
                        _bt._real_profile_cdp_cache.update(cdp=existing, headed=persisted)
                        return persisted
                    return True
        except Exception as exc:
            _bt.logger.debug("real-profile mode recovery failed: %s", exc)
            return True
    return _cloud._is_headed_mode()
