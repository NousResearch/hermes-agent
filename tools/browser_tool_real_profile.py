"""Real-profile local browsing: snapshot the user's default Chromium profile into a
hermes-owned copy, launch the real browser binary on it, and attach agent-browser.

State (``_REAL_PROFILE_SESSION``, ``_real_profile_cdp_lock``, ``_real_profile_cdp_cache``,
``_real_profile_chrome_procs``) lives in ``tools.browser_tool``; it is read
through ``_bt`` (resolved per call — never import ``tools.browser_tool`` at import time).
"""

import json
import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_install as _install
from tools import browser_tool_lightpanda_fallback as _lp
from tools import browser_tool_session as _session

_RP = "browser.use_real_profile is on, but "
_OWNERS_DIRNAME = ".hermes-owners"
_SNAPSHOT_LOCK_NAMES = ("SingletonLock", "SingletonSocket", "SingletonCookie", "DevToolsActivePort")


def _owners_dir() -> str:
    """Sidecar dir for snapshot-browser owner records (outside each Chrome user-data-dir)."""
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "browser-profile" / _OWNERS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return str(path)


def _owner_record_path(copy_dir: str) -> str:
    name = os.path.basename(os.path.normpath(copy_dir)) or "unknown"
    return os.path.join(_owners_dir(), f"{name}.json")


def _write_owner_record(copy_dir: str, chrome_pid: int) -> None:
    """Persist enough identity for a later process to reap this snapshot browser after a crash."""
    from tools.process_registry import ProcessRegistry
    record = {
        "copy_dir": copy_dir,
        "chrome_pid": chrome_pid,
        "chrome_start_time": ProcessRegistry._safe_host_start_time(chrome_pid),
        "owner_pid": os.getpid(),
        "owner_start_time": ProcessRegistry._safe_host_start_time(os.getpid()),
    }
    path = _owner_record_path(copy_dir)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        os.chmod(path, 0o600)
    except OSError as e:
        _origin().logger.debug("real-profile owner record write failed: %s", e)


def _read_owner_records() -> list:
    """``(path, record)`` for every readable owner sidecar. Bad JSON is skipped."""
    try:
        names = os.listdir(_owners_dir())
    except OSError:
        return []
    out = []
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(_owners_dir(), name)
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("copy_dir") and rec.get("chrome_pid"):
            out.append((path, rec))
    return out


def _pid_is_ours(pid, expected_start) -> bool:
    """True when ``pid`` is alive and still the process we recorded (start-time match)."""
    from tools.process_registry import ProcessRegistry
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    return ProcessRegistry._host_pid_is_ours(pid, expected_start)


def _cmdline_bound_to_copy_dir(pid: int, copy_dir: str) -> bool:
    """True only when the live process is the snapshot browser (``--user-data-dir=<copy>``).

    Refuses the user's daily browser: that process never uses the hermes snapshot dir.
    """
    needle = f"--user-data-dir={os.path.normpath(copy_dir)}".replace("\\", "/")
    try:
        import psutil
        cmd = " ".join(psutil.Process(pid).cmdline() or [])
    except Exception:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            return False
    return needle in cmd.replace("\\", "/")


def _clear_snapshot_lock_files(copy_dir: str) -> None:
    """Remove Chrome's 'this dir is in use' markers after the snapshot browser is gone."""
    for name in _SNAPSHOT_LOCK_NAMES:
        try:
            os.unlink(os.path.join(copy_dir, name))
        except OSError:
            pass


def _kill_snapshot_chrome(pid, copy_dir: str, expected_start=None) -> bool:
    """Tree-kill ``pid`` if it is still the snapshot browser on ``copy_dir``. False if refused."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if not _cmdline_bound_to_copy_dir(pid, copy_dir):
        return False
    from tools.process_registry import ProcessRegistry
    ProcessRegistry._terminate_host_pid(pid, expected_start=expected_start)
    return True


def _forget_real_profile_attach() -> None:
    """Drop cached CDP and the agent-browser attach so the next launch is a fresh headed window."""
    _bt = _origin()
    _bt._real_profile_cdp_cache.pop("cdp", None)
    try:
        _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
    except Exception:
        pass


def _terminate_real_profile_chrome() -> None:
    """Terminate real-browser processes launched for real-profile sessions (idempotent).

    agent-browser only ATTACHED to them, so its own session cleanup never kills them.
    Also kills snapshot browsers this process recorded on disk (Popen handle lost)
    and clears the copy-dir lock files.
    """
    from tools.browser_lightpanda import _terminate
    _bt = _origin()
    while _bt._real_profile_chrome_procs:
        _terminate(_bt._real_profile_chrome_procs.pop(), what="real-profile chrome")
    for path, rec in _read_owner_records():
        if rec.get("owner_pid") != os.getpid():
            continue
        copy_dir = rec["copy_dir"]
        _kill_snapshot_chrome(rec.get("chrome_pid"), copy_dir, rec.get("chrome_start_time"))
        _clear_snapshot_lock_files(copy_dir)
        try:
            os.unlink(path)
        except OSError:
            pass
    _forget_real_profile_attach()


def _reap_orphaned_real_profile_browsers() -> int:
    """Kill snapshot browsers whose launching Hermes process is dead. Returns reaped count.

    Live owners are left alone (another Hermes still owns that copy). Identity-checked
    against ``--user-data-dir=<copy_dir>`` so a recycled PID cannot become a kill.
    """
    reaped = 0
    for path, rec in _read_owner_records():
        if _pid_is_ours(rec.get("owner_pid"), rec.get("owner_start_time")):
            continue
        copy_dir = rec["copy_dir"]
        if _kill_snapshot_chrome(rec.get("chrome_pid"), copy_dir, rec.get("chrome_start_time")):
            reaped += 1
        _clear_snapshot_lock_files(copy_dir)
        try:
            os.unlink(path)
        except OSError:
            pass
    if reaped:
        _origin().logger.info("Reaped %d orphaned real-profile browser(s)", reaped)
    return reaped


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    from tools.browser_lightpanda import _cdp_ready
    return _cdp_ready(http_cdp, timeout=1.0)


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
                              env=_bt._build_browser_env(), stdin=subprocess.DEVNULL)
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


def _launch_real_profile_chrome(real_binary: str, copy_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Launch the user's REAL browser binary on the profile COPY; return (debug_port, error).

    agent-browser's own launch force-adds --use-mock-keychain / --password-store=basic, which makes
    macOS Chrome drop every keychain-encrypted cookie (signed-out copy); launching the real binary
    ourselves keeps the OS keychain path intact and agent-browser attaches via ``--cdp <port>``.
    Headless by default (a focus-stealing window defeats a background capability); Chrome's NEW
    headless shares the profile's cookie store (legacy --headless does not). browser.headed /
    AGENT_BROWSER_HEADED opts into a window, except on a display-less Linux host (launch would die).
    """
    _bt = _origin()
    try:
        os.unlink(os.path.join(copy_dir, "DevToolsActivePort"))  # stale port confuses reuse probes
    except OSError:
        pass
    chrome_argv = [real_binary, f"--user-data-dir={copy_dir}", *_REAL_PROFILE_CHROME_FLAGS]
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not (_cloud._is_headed_mode() and (_has_display or not sys.platform.startswith("linux"))):
        chrome_argv.append("--headless=new")
    try:
        chrome_proc = subprocess.Popen(chrome_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=_bt._build_browser_env())
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    _bt._real_profile_chrome_procs.append(chrome_proc)
    if getattr(chrome_proc, "pid", None):
        _write_owner_record(copy_dir, chrome_proc.pid)

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
                              timeout=_bt._get_open_command_timeout(first_open=True), env=_bt._build_browser_env(),
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


def _real_profile_cdp() -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshot -> launch real binary on the copy -> return its HTTP CDP endpoint. The copy is a
    non-default dir, so it sidesteps the Chrome >=136 default-profile remote-debugging block and
    never contends with the user's running browser. One shared agent-browser session is reused
    across calls (cached, re-validated). ``(None, message)`` fail-closed; ``(None, None)`` when consent is off.
    """
    _bt = _origin()
    if not _cloud._use_real_profile():
        # Consent is off: delete any snapshot store (copies of cookies/logins) so
        # revoking consent actually removes the credential copies.
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots
            cleanup_real_profile_snapshots()
        except Exception as e:
            _bt.logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _bt._real_profile_cdp_cache.pop("cdp", None)
        return None, None

    # Lightpanda rejects ``--profile``; check BEFORE default-browser detection so a
    # host with no Chromium default still reports the actionable engine conflict.
    if _lp._using_lightpanda_engine():
        return None, (_RP + "browser.engine is set to 'lightpanda', which cannot load a real Chromium profile. "
                      "Set browser.engine to 'auto' or 'chrome' to use real-profile browsing, or turn the toggle off.")

    from hermes_cli.browser_connect import (chromium_executable, detect_default_chromium,
                                            real_profile_copy_dir, snapshot_real_profile)

    with _bt._real_profile_cdp_lock:
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if cached and _cdp_http_ready(cached):
            return cached, None
        _bt._real_profile_cdp_cache.pop("cdp", None)

        browser = detect_default_chromium()
        unsupported = _real_profile_unsupported_reason(browser)
        if unsupported:
            return None, unsupported

        # Reuse BEFORE writing anything. CRITICAL: the snapshot overlay (truncates/rewrites
        # Cookies / Login Data) must NOT run while a live copy-browser (maybe from a previous
        # hermes process) holds the user-data-dir open — that corrupts the databases.
        copy_dir = real_profile_copy_dir(browser)
        existing = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
        if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
            _bt._real_profile_cdp_cache["cdp"] = existing
            return existing, None
        if existing:  # stale/wrong-dir session: close it so nothing holds the dir open
            _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)

        copy_dir, err = snapshot_real_profile(browser)
        if err or not copy_dir:
            return None, _real_profile_snapshot_error(err)
        real_binary = chromium_executable(browser)
        if real_binary is None:
            return None, f"{_RP}the real browser binary for '{browser}' could not be found. Reinstall it or turn the toggle off."
        port, err = _launch_real_profile_chrome(real_binary, copy_dir)
        if port is None:
            return None, err
        cdp, err = _attach_agent_browser_to_real_profile(port, copy_dir)
        if not cdp:
            return None, err
        _bt._real_profile_cdp_cache["cdp"] = cdp
        _bt.logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None
