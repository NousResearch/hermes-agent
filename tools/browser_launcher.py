"""Launch, track and stop the Chromium processes ``browser_exec`` drives.

One browser per launcher key (the session layer's cache key, or ``profile:<dir>`` for a persistent
profile shared by every key). Chromium picks its own DevTools port (``--remote-debugging-port=0``)
and writes it to ``<user-data-dir>/DevToolsActivePort``; the CDP URL comes from that file, so there
is no port race. Every owned launch leaves a record in a private (0700) launcher state dir so a later
Hermes process can reap the Chromium of an owner that crashed, after proving the pid is still that browser:
a throwaway profile's in its home (``<HERMES_HOME>/cache/browser/chromium-records``), a persistent profile's
in that profile's own dir under :func:`_shared_profiles_root` (per OS user: every home pinning the dir shares
it). A shared-profile Chromium other Hermes processes attached to is stopped only by its last holder
(``tools/browser_launcher_holders.py``).

Stop is always parent-first (``ProcessRegistry._terminate_host_pid``): Chromium reaps its own
zygotes and renderers on SIGTERM, and killing a zygote first turns a clean exit into a crash dump.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# THE Chromium startup flags for a host where its sandbox cannot work. The Bot Desktop dock's Browser
# icon (same binary, same profile) uses the same list (``tools.bot_desktop.browser.dock_argv``), or the
# human's click dies while the agent's launch works.
CHROMIUM_SANDBOX_BYPASS_ARGS: tuple[str, ...] = ("--no-sandbox", "--disable-dev-shm-usage")

# Base flags, each kept for a reason (the minimum for an unattended, disposable automation browser):
#   --no-first-run / --no-default-browser-check  no welcome tab or default-browser bar on a fresh profile
#   --disable-background-networking               no field-trial / safe-browsing / update pings per launch
#   --disable-component-update                    no component downloads into a throwaway profile
#   --disable-default-apps / --disable-sync       nothing installed or signed in behind the agent's back
#   --disable-features=Translate                  no translate bubble covering the page
_BASE_FLAGS: tuple[str, ...] = (
    "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
    "--disable-component-update", "--disable-default-apps", "--disable-sync", "--disable-features=Translate",
)
# Only for the launcher-resolved packaged Chromium: a keyring unlock prompt (Linux desktops) or the macOS
# keychain dialog would block an unattended launch. Never for a real or shared profile, whose cookies are
# encrypted with the OS store (``--password-store=basic`` would sign the copy out).
_PACKAGED_ONLY_FLAGS: tuple[str, ...] = ("--password-store=basic",)
_MACOS_PACKAGED_ONLY_FLAGS: tuple[str, ...] = ("--use-mock-keychain",)

_DEVTOOLS_FILE = "DevToolsActivePort"
_LAUNCH_LOG = "hermes-chromium.log"
_MIN_FIRST_LAUNCH_TIMEOUT_S = 120
_UNRESPONSIVE_WAIT_S = 10  # how long a live browser whose endpoint went silent (paused, swapped out) is waited for
_POLL_INTERVAL_S = 0.1
_STDERR_TAIL_CHARS = 1500


@dataclass
class LaunchedBrowser:
    """A Chromium this process launched (``owned``) or attached to (a running instance on a shared
    profile, never killed by Hermes)."""

    key: str
    pid: Optional[int]
    start_time: Optional[int]
    cdp_url: str
    http_url: str
    user_data_dir: str
    owned: bool
    ephemeral: bool
    home_key: str
    # File name of the launch record its leases are keyed by, when that is not this process's own name for
    # it: an attached browser keeps its launcher's record, whichever PID namespace wrote it.
    record_name: Optional[str] = None
    # Held from an environment this process cannot verify (another PID namespace: a container's browser seen
    # from the host, or the reverse): ``pid`` is that namespace's number, names nobody here and is never
    # signalled; liveness is its DevTools endpoint.
    foreign: bool = False


_browsers: dict[str, LaunchedBrowser] = {}
_procs: dict[str, subprocess.Popen] = {}
_launching: dict[int, LaunchedBrowser] = {}  # pid -> a spawned Chromium still in its readiness wait
_key_locks: dict[str, threading.Lock] = {}
_lock = threading.RLock()
_autoinstall_attempted = False
# Profile homes (``hermes_home_key``) this process launched or attached a browser for: the reaper reads
# each one's records, since one process serves many profiles.
_served_homes: set[str] = set()


# ── Host facts ───────────────────────────────────────────────────────────────────────────────────────

def apparmor_restricts_unprivileged_userns() -> bool:
    """Ubuntu 23.10+ default: unprivileged user namespaces are denied, so a Chromium whose
    ``chrome_sandbox`` helper is not setuid (Playwright's bundle) dies with 'No usable sandbox'."""
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def needs_chromium_sandbox_bypass() -> bool:
    """True when Chromium needs --no-sandbox to start reliably (root, Docker, AppArmor userns)."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if running_in_docker():
        return True
    return apparmor_restricts_unprivileged_userns()


def _chromium_missing_hint() -> str:
    from hermes_cli.browser_runtime import chromium_missing_hint
    return chromium_missing_hint()


def _resolve_executable() -> Optional[str]:
    from hermes_cli.browser_runtime import chromium_executable
    exe = chromium_executable()
    if not exe:
        return None
    if os.path.isfile(exe):
        return exe
    from hermes_platform.resolver import locate_command
    found = locate_command(exe).command  # a bare command name pinned in browser.executable_path
    return found[0] if found else None


def _explicit_executable_configured(chromium_executable) -> bool:
    """An operator-pinned browser binary (even a missing one) is their choice: never download over it."""
    return chromium_executable(allow_override=True) != chromium_executable(allow_override=False)


def _maybe_autoinstall_chromium() -> bool:
    """One-shot ``pm.ensure("chromium")`` (a failed ~170 MB download must not retry per call). Docker
    supplies the binary, Termux has no PM Chromium, and an explicit executable is the user's choice."""
    global _autoinstall_attempted
    if _autoinstall_attempted:
        return False
    _autoinstall_attempted = True
    from hermes_cli.browser_runtime import chromium_executable
    from hermes_constants import is_termux
    if running_in_docker() or is_termux() or _explicit_executable_configured(chromium_executable):
        return False
    from pm import InstallError, ensure, lazy_installs_allowed
    if not lazy_installs_allowed():
        return False
    logger.info("browser: installing PM's pinned Chromium")
    try:
        ensure("chromium")
    except (InstallError, OSError) as exc:
        logger.warning("browser: Chromium auto-install failed: %s", exc)
        return False
    return True


def chromium_ready() -> tuple[bool, str]:
    """``(True, "")`` when a Chromium is on disk (after at most one PM auto-install per process), else
    ``(False, actionable hint)``."""
    if _resolve_executable():
        return True, ""
    if _maybe_autoinstall_chromium() and _resolve_executable():
        return True, ""
    return False, _chromium_missing_hint()


# ── Launch ───────────────────────────────────────────────────────────────────────────────────────────

def _default_user_data_dir(key: str, home_key: str) -> str:
    """Per process too: two Hermes processes on one profile use the same keys (``browser-exec-default``,
    named sessions), and each launch wipes its throwaway dir first. The name carries the owner's pid and
    start time: a process that dies before its launch record exists leaves the dir to the reaper
    (:func:`_reclaim_throwaway_dirs`)."""
    from tools.browser_launcher_holders import ns_tag
    digest = hashlib.sha256(f"{home_key}\0{key}".encode()).hexdigest()[:16]
    pid = os.getpid()
    started = _start_time(pid)
    owner = f"{pid}-{started}" if started is not None else str(pid)
    return str(_ephemeral_root(home_key) / f"{digest}-{owner}{ns_tag()}")


def _launch_timeout(timeout: Optional[float]) -> float:
    if timeout is not None:
        return float(timeout)
    from tools.browser_session_cloud import command_timeout
    return float(max(command_timeout(), _MIN_FIRST_LAUNCH_TIMEOUT_S))


def _has_display(env: dict) -> bool:
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def _resolve_headed(headed: Optional[bool], env: dict) -> bool:
    """Headless unless ``browser.headed``; a display-less Linux host stays headless (a headed launch
    would die there). Windows obeys the same rule: headless unless the user opted in (#64867)."""
    if headed is None:
        from tools.browser_session_cloud import is_headed_mode
        headed = is_headed_mode()
    if not headed:
        return False
    return _has_display(env) or not sys.platform.startswith("linux")


def _configured_chromium_args() -> list[str]:
    from tools.browser_session_cloud import chromium_args
    return chromium_args()


def build_chromium_argv(executable: str, user_data_dir: str, *, packaged: bool, headed: bool,
                        sandbox_bypass: bool, extra_args=()) -> list[str]:
    """The launch argv (pure in its inputs + host sandbox facts + ``browser.chromium_args``).

    ``sandbox_bypass`` lets the host policy (root, Docker, AppArmor userns) add
    ``CHROMIUM_SANDBOX_BYPASS_ARGS``; it is False for the user's own browser binary, which is never
    started without its sandbox."""
    argv = [executable, "--remote-debugging-port=0", f"--user-data-dir={user_data_dir}", *_BASE_FLAGS]
    if packaged:
        argv += _PACKAGED_ONLY_FLAGS
        if sys.platform == "darwin":
            argv += _MACOS_PACKAGED_ONLY_FLAGS
    if not headed:
        argv.append("--headless=new")
    configured = _configured_chromium_args()
    if sandbox_bypass and needs_chromium_sandbox_bypass():
        argv += [flag for flag in CHROMIUM_SANDBOX_BYPASS_ARGS if flag not in configured]
    argv += [str(a) for a in extra_args]
    argv += configured
    argv.append("about:blank")
    return argv


def _spawn_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        from hermes_cli._subprocess_compat import windows_hide_flags
        return {"creationflags": windows_hide_flags() | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    # detached-spawn: ok — a shared-profile Chromium outlives its launcher for its other holders (browser_launcher_holders); a crashed owner's throwaway one is reaped from its launch record by the next Hermes
    return {"start_new_session": True}


def _read_devtools_file(user_data_dir: str) -> Optional[tuple[int, str]]:
    """``(port, "/devtools/browser/<id>")`` once Chromium wrote both lines, else None."""
    try:
        with open(os.path.join(user_data_dir, _DEVTOOLS_FILE), encoding="utf-8-sig") as fh:
            port, path = fh.readline().strip(), fh.readline().strip()
    except OSError:
        return None
    if not port.isdigit() or not path.startswith("/devtools/browser/"):
        return None
    return int(port), path


def _log_tail(user_data_dir: str) -> str:
    try:
        raw = Path(user_data_dir, _LAUNCH_LOG).read_bytes()[-_STDERR_TAIL_CHARS:]
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace").strip()


def _wait_for_devtools(proc: subprocess.Popen, user_data_dir: str, deadline: float) -> tuple[Optional[tuple[int, str]], Optional[str]]:
    while True:
        found = _read_devtools_file(user_data_dir)
        if found:
            return found, None
        rc = proc.poll()
        if rc is not None:
            tail = _log_tail(user_data_dir)
            hint = (" Chromium's sandbox could not start; add --no-sandbox to browser.chromium_args."
                    if "sandbox" in tail.lower() else "")
            return None, f"Chromium exited with code {rc} during startup.{hint}" + (f"\n{tail}" if tail else "")
        if time.monotonic() >= deadline:
            return None, "Chromium did not expose a DevTools endpoint in time."
        time.sleep(_POLL_INTERVAL_S)


def _singleton_pid(user_data_dir: str) -> Optional[int]:
    """Pid of the Chromium holding ``user_data_dir``: the ``SingletonLock`` symlink (POSIX), else (Windows
    has no symlink, only a locked ``lockfile``) the browser process whose argv names the dir."""
    owner = _singleton_owner(user_data_dir)
    if owner is None:
        return _pid_naming_dir(user_data_dir) if os.name == "nt" else None
    return owner[1]


def _singleton_owner(user_data_dir: str) -> Optional[tuple[str, int]]:
    """``(hostname, pid)`` Chromium wrote into ``user_data_dir``'s ``SingletonLock`` symlink (POSIX), else None."""
    try:
        target = os.readlink(os.path.join(user_data_dir, "SingletonLock"))
    except OSError:
        return None
    host, _, pid_text = target.rpartition("-")
    return (host, int(pid_text)) if pid_text.isdigit() else None


def _foreign_holder(user_data_dir: str) -> Optional[int]:
    """The ``SingletonLock`` pid of a Chromium on ``user_data_dir`` that runs in an environment this process cannot
    verify, else None. Foreign: the lock is not a live process here whose argv names the dir, and it was written
    under another hostname (another host or container; Chromium too refuses such a lock) or names the browser
    of a launch record from another PID namespace. A lock of this hostname with no such record is stale (this
    environment's crashed browser): Chromium itself takes it over, and so does a launch."""
    owner = _singleton_owner(user_data_dir)
    if owner is None:
        return None
    host, pid = owner
    if _pid_alive(pid) and _cmdline_names_dir(pid, user_data_dir):
        return None
    if host != socket.gethostname() or _foreign_record_names(user_data_dir, pid):
        return pid
    return None


def _foreign_record_names(user_data_dir: str, pid: int) -> bool:
    """A launch record from another PID namespace names ``pid`` as shared profile ``user_data_dir``'s browser: the one
    serving the profile's DevTools endpoint now, or one not yet up there (its record has no endpoint). An earlier
    launch there that reached another endpoint is not that browser, whatever pid number it had."""
    from tools.browser_launcher_holders import same_namespace
    state = _shared_state_dir(user_data_dir)
    # A record from another PID namespace is why this profile is refused, so it must be found wherever the launcher
    # that wrote it kept it: this root, and the PRE-FIX roots a profile ``$HOME`` launcher used before the root was
    # unified. Read-only there (:func:`_read_only_json`): the guard/refusal path never edits old state.
    states = [state] if state is not None else []
    states += _legacy_state_dirs(_profile_digest(user_data_dir)[:32])
    found = _read_devtools_file(user_data_dir)
    for records in states:
        for name in _record_names(records, user_data_dir, pid):
            if records == state:
                record = _load_record(records / name)
            else:
                status, record = _read_only_json(records / name)
                if status == _MISSING:
                    continue
            if (record is not None and record.get("pid") == pid and not same_namespace(record)
                    and _same_profile(record.get("user_data_dir"), user_data_dir)
                    and record.get("devtools") in (None, found and found[1])):
                return True
    return False


def _record_names(records: Path, user_data_dir: str, pid) -> list[str]:
    """Names of the launch records in ``records`` for profile ``user_data_dir``'s browser ``pid``, whichever PID
    namespace wrote them (each carries its writer's ``ns_tag``), sorted."""
    stem = f"{_profile_digest(user_data_dir)[:16]}-{pid}"
    pattern = re.compile(rf"^{re.escape(stem)}(-n[0-9a-f]{{12}})?\.json$")
    return sorted(p.name for p in records.glob(f"{stem}*.json") if pattern.match(p.name))


def _pid_naming_dir(user_data_dir: str) -> Optional[int]:
    """The browser (not a ``--type=`` helper) process whose argv carries ``--user-data-dir=<dir>``."""
    import psutil
    for proc in psutil.process_iter(["pid"]):
        try:
            argv = proc.cmdline()
        except (psutil.Error, OSError):
            continue
        if not any(tok.startswith("--type=") for tok in argv) and _cmdline_names_dir(proc.pid, user_data_dir):
            return proc.pid
    return None


def _browser_running_on(user_data_dir: str) -> bool:
    """A live Chromium still uses ``user_data_dir`` (by argv: a ``SingletonLock`` outlives a crashed browser)."""
    return _pid_naming_dir(user_data_dir) is not None


def _devtools_endpoint_live(port: int, path: str) -> bool:
    """The port answers ``/json/version`` with the browser id the DevToolsActivePort file names (the file
    outlives a crashed Chromium and its port can be recycled by another CDP server)."""
    try:
        import requests
        from agent.proxy_bypass import loopback_request_kwargs
        url = f"http://127.0.0.1:{port}/json/version"
        ws = str(requests.get(url, timeout=2, **loopback_request_kwargs(url)).json().get("webSocketDebuggerUrl") or "")
    except Exception:  # health: allow BLE001 -- any probe failure means "not a live browser on this port"
        return False
    return ws.endswith(path)


def running_instance(user_data_dir: str) -> Optional[tuple[int, str, int]]:
    """``(port, browser_path, pid)`` of a Chromium already running on ``user_data_dir`` (a human-started
    Bot Desktop dock instance, a surviving real-profile browser), else None. A second launch on that dir
    would be forwarded into it by Chromium's singleton and exit without a DevTools endpoint."""
    pid = _singleton_pid(user_data_dir)
    found = _read_devtools_file(user_data_dir)
    if pid is None or found is None or not _pid_alive(pid):
        return None
    port, path = found
    return (port, path, pid) if _devtools_endpoint_live(port, path) else None


def _pid_alive(pid: int) -> bool:
    from gateway.status import _pid_exists
    return pid > 0 and _pid_exists(pid)


def _start_time(pid: int) -> Optional[int]:
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def _make_browser(key: str, port: int, path: str, *, pid, start_time, user_data_dir: str, owned: bool,
                  ephemeral: bool, home_key: str) -> LaunchedBrowser:
    return LaunchedBrowser(key=key, pid=pid, start_time=start_time, cdp_url=f"ws://127.0.0.1:{port}{path}",
                           http_url=f"http://127.0.0.1:{port}", user_data_dir=user_data_dir, owned=owned,
                           ephemeral=ephemeral, home_key=home_key)


def _prepare_dir(user_data_dir: str, ephemeral: bool) -> None:
    if ephemeral:
        Path(user_data_dir).parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # the private ephemeral root
        shutil.rmtree(user_data_dir, ignore_errors=True)  # fresh profile per launch
    os.makedirs(user_data_dir, mode=0o700, exist_ok=True)
    try:
        os.unlink(os.path.join(user_data_dir, _DEVTOOLS_FILE))  # a stale port confuses the readiness poll
    except OSError:
        pass


def _spawn(argv: list[str], env: dict, user_data_dir: str) -> subprocess.Popen:
    log_path = os.path.join(user_data_dir, _LAUNCH_LOG)
    with open(log_path, "wb") as log:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                                **_spawn_kwargs())


def _attach_running(key: str, user_data_dir: str, home_key: str) -> Optional[LaunchedBrowser]:
    """The live Chromium already on shared profile ``user_data_dir``, leased (caller holds its lock)."""
    running = running_instance(user_data_dir)
    if running is None:
        return None
    port, path, pid = running
    logger.info("browser: attaching to the Chromium already running on %s (pid %s)", user_data_dir, pid)
    browser = _make_browser(key, port, path, pid=pid, start_time=_start_time(pid), user_data_dir=user_data_dir,
                            owned=False, ephemeral=False, home_key=home_key)
    records = _state_dir(browser)
    if records is not None:
        from tools import browser_launcher_holders as holders
        _bind_launch_record(records, browser)
        _adopt_record(browser)
        holders.write_lease(_record_path(browser), browser)  # its launcher's exit must not stop it under us
    return browser


def _bind_launch_record(records: Path, browser: LaunchedBrowser) -> None:
    """Lease ``browser`` under the launch record that describes THIS launch of it (:func:`_record_is`), written in
    ANY PID namespace: its launcher (a container gateway a host CLI attached to) finds its holders only there.
    Records of earlier containers and boots are kept, so the pid number alone can name another browser's record.
    This namespace's record wins; with none (a browser a human started) the lease goes under this namespace's
    name. A browser of another environment takes its record's start time: its full identity from then on."""
    own = _record_name(browser)
    names = sorted(_record_names(records, browser.user_data_dir, browser.pid), key=lambda name: name != own)
    for name in names:
        record = _load_record(records / name)
        if record is not None and _record_is(record, browser):
            browser.record_name = name
            if browser.foreign:
                browser.start_time = record.get("start_time")
            return


def _attach_foreign(key: str, user_data_dir: str, home_key: str, pid: int) -> Optional[LaunchedBrowser]:
    """The Chromium another environment holds on ``user_data_dir``, leased, when its DevTools endpoint answers here
    (caller holds the profile's lock); None when it does not. Never adopted: its record is not ours to take."""
    found = _read_devtools_file(user_data_dir)
    if found is None or not _devtools_endpoint_live(*found):
        return None
    logger.info("browser: attaching to the Chromium another environment runs on %s (its pid %s there)",
                user_data_dir, pid)
    browser = _make_browser(key, found[0], found[1], pid=pid, start_time=None, user_data_dir=user_data_dir,
                            owned=False, ephemeral=False, home_key=home_key)
    browser.foreign = True
    records = _state_dir(browser)
    if records is not None:
        from tools import browser_launcher_holders as holders
        _bind_launch_record(records, browser)
        holders.write_lease(_record_path(browser), browser)  # its launcher's exit must not stop it under us
    return browser


def _live_local_holder(user_data_dir: str) -> Optional[int]:
    """Pid of the live Chromium of this environment holding ``user_data_dir`` (its ``SingletonLock`` names a live
    process whose argv carries the dir), else None: no browser, or a stale lock a launch may take over."""
    pid = _singleton_pid(user_data_dir)
    return pid if pid is not None and _pid_alive(pid) and _cmdline_names_dir(pid, user_data_dir) else None


def _attach_local(key: str, user_data_dir: str, home_key: str,
                  timeout: Optional[float]) -> tuple[Optional[LaunchedBrowser], Optional[str]]:
    """Attach to this environment's live Chromium on ``user_data_dir``; ``(None, None)`` when none holds it (a
    launch may start one). A live holder whose endpoint does not answer (paused, hung) is waited for, then
    refused: a launch over it would only wipe its ``DevToolsActivePort`` and lose it for every later attach."""
    attached = _attach_running(key, user_data_dir, home_key)
    pid = _live_local_holder(user_data_dir) if attached is None else None
    if pid is None:
        return attached, None
    deadline = time.monotonic() + min(_UNRESPONSIVE_WAIT_S, _launch_timeout(timeout))
    delay = 0.25
    while attached is None and _live_local_holder(user_data_dir) == pid:
        if time.monotonic() >= deadline:
            return None, (f"The browser profile {user_data_dir} is in use by Chromium pid {pid}, which is not "
                          "responding on its DevTools endpoint (paused, hung, or started without remote debugging). "
                          "Hermes will not start a second browser on it; try again once it responds, or stop it.")
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay *= 2
        attached = _attach_running(key, user_data_dir, home_key)
    return attached, None


def _foreign_unreachable(user_data_dir: str) -> str:
    return (f"The browser profile {user_data_dir} is in use by a Chromium in another environment (a container or "
            "the host) that this process can neither reach nor verify, so Hermes will not start a second browser "
            "on it. Stop that browser, or set browser.user_data_dir to another directory for this environment; if "
            f"no browser runs there any more, delete {os.path.join(user_data_dir, 'SingletonLock')}.")


def _start_shared(key: str, executable: str, udd: str, home_key: str, prepare,
                  **launch) -> tuple[Optional[LaunchedBrowser], Optional[str]]:
    """Attach to, or launch, shared profile ``udd``'s one Chromium as ONE step under the profile's cross-process
    lock: of two processes starting it at once the later waits, then attaches (a second launch would wipe the
    first's DevToolsActivePort and lose Chromium's singleton), and an owner's stop cannot fall between the live
    probe and the lease. A first launch holds the lock for its whole readiness wait; the reaper skips a busy
    profile instead of waiting (:func:`_reap_shared_profiles`). A browser another environment holds is attached
    through its endpoint, or refused; never launched over."""
    from tools import browser_launcher_holders as holders
    state = _shared_state_dir(udd)
    if state is None:
        # No private state dir means no profile lock: a launch here would race whoever is already driving the
        # profile. Refuse instead of falling through to an unsynchronised start (the pre-existing fail-open path).
        return None, (f"Hermes cannot serialise access to the shared browser profile {udd}: its launcher state "
                      "directory is not private to this OS user. Repair the ownership/permissions of the launcher "
                      "state root, or point HERMES_GATEWAY_LOCK_DIR at a directory private to this user.")
    with holders.records_lock(state):
        # EVERY path that would lease this browser passes the pre-fix guard first — including the attach a reachable
        # foreign endpoint allows (``_attach_foreign`` writes a lease just like a local attach does). A browser this
        # environment cannot verify is still refused as before; that verdict itself is unchanged, and it reads the
        # same evidence (pre-fix records included, see ``_foreign_record_names``).
        blocked = _legacy_guard(state.name)
        foreign = _foreign_holder(udd)
        if foreign is not None:
            if blocked is not None:
                logger.warning("browser: not attaching to %s — %s", udd, blocked)
                return None, _foreign_unreachable(udd)
            attached = _attach_foreign(key, udd, home_key, foreign)
            if attached is None:
                return None, _foreign_unreachable(udd)
        elif blocked is not None:
            # Pre-fix state still names a live launcher, holder or browser: starting here would put two launchers in
            # charge of one Chromium, so the profile stays out of this version's hands.
            return None, blocked
        else:
            attached, refused = _attach_local(key, udd, home_key, launch.get("timeout"))
            if refused:
                return None, refused
        if attached is None:
            failure = prepare() if prepare is not None else None
            if failure:
                return None, failure
            return _launch_owned(key, executable, udd, ephemeral=False, home_key=home_key, **launch)
        with _lock:  # registered under the lock too: another key's stop must see this key as its heir
            _browsers[key] = attached
    return attached, None


def launch_chromium(key: str, *, executable: Optional[str] = None, user_data_dir: Optional[str] = None,
                    ephemeral: bool = True, headed: Optional[bool] = None, extra_args=(),
                    env: Optional[dict] = None, timeout: Optional[float] = None, sandbox_bypass: bool = True,
                    prepare: Optional[Callable[[], Optional[str]]] = None,
                    ) -> tuple[Optional[LaunchedBrowser], Optional[str]]:
    """Start (or reuse) the Chromium for ``key``; ``(browser, None)`` or ``(None, error)``.

    ``executable=None`` resolves the packaged Chromium (``chromium_ready``); an explicit executable (the
    Bot Desktop's, the user's real browser) is used as given. ``user_data_dir=None`` is a fresh
    ephemeral profile under ``<HERMES_HOME>/cache/browser/chromium/``; a persistent dir that already has
    a live Chromium on it is ATTACHED (``owned=False``, never killed). ``sandbox_bypass=False`` (the
    user's real browser) never gets the host's sandbox-bypass flags. ``prepare`` (a persistent dir only) runs
    right before a fresh launch, inside the same cross-process step, never when attaching: a profile copy
    written there can neither race another process's copy nor land under a browser that just came up; a
    message it returns is the launch's error, as is."""
    from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal
    from hermes_constants import hermes_home_key
    if (refused := run_elsewhere_refusal()) is not None:  # that backend writes browser.executable_path
        return None, refused
    # Loaded now, not first at teardown: stopping the browser from an atexit hook must not be this module's
    # first import (Python refuses new atexit registrations during shutdown, orphaning the browser). An
    # attacher needs it too: it is the one that stops a shared browser when it is the last holder.
    import tools.process_registry
    with _key_lock(key):  # one launch per key; other keys launch concurrently
        existing = get_browser(key)
        if existing is not None:
            return existing, None
        home_key = hermes_home_key()
        with _lock:
            _served_homes.add(home_key)
        packaged = executable is None
        if packaged:
            ok, hint = chromium_ready()
            if not ok:
                return None, hint
            executable = _resolve_executable()
        udd = user_data_dir or _default_user_data_dir(key, home_key)
        launch = dict(packaged=packaged, headed=headed, extra_args=extra_args, env=env, timeout=timeout,
                      sandbox_bypass=sandbox_bypass)
        if ephemeral:  # a per-process dir: nobody else starts or attaches to it
            return _launch_owned(key, executable, udd, ephemeral=True, home_key=home_key, **launch)
        return _start_shared(key, executable, udd, home_key, prepare, **launch)


def _key_lock(key: str) -> threading.Lock:
    with _lock:
        return _key_locks.setdefault(key, threading.Lock())


def _launch_owned(key: str, executable: str, udd: str, *, ephemeral: bool, packaged: bool, headed,
                  extra_args, env, timeout, home_key: str, sandbox_bypass: bool,
                  ) -> tuple[Optional[LaunchedBrowser], Optional[str]]:
    if env is None:
        from tools.browser_session import build_browser_env
        env = build_browser_env()
    try:
        _prepare_dir(udd, ephemeral)
    except OSError as exc:
        return None, f"Could not create the browser profile directory {udd}: {exc}"
    argv = build_chromium_argv(executable, udd, packaged=packaged, headed=_resolve_headed(headed, env),
                               sandbox_bypass=sandbox_bypass, extra_args=extra_args)
    try:
        proc = _spawn(argv, env, udd)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"Failed to launch Chromium ({executable}): {exc}"
    start_time = _start_time(proc.pid)
    starting = _make_browser(key, 0, "", pid=proc.pid, start_time=start_time, user_data_dir=udd, owned=True,
                             ephemeral=ephemeral, home_key=home_key)
    # Recorded before the readiness wait (a first launch may take 120 s): if this process dies in it, the
    # record is what lets the next reaper stop this Chromium. Tracked as launching so this process's own
    # reaper leaves it alone meanwhile.
    with _lock:
        _launching[proc.pid] = starting
    _write_record(starting)
    try:
        found, err = _wait_for_devtools(proc, udd, time.monotonic() + _launch_timeout(timeout))
    except BaseException:  # interrupted (Ctrl-C, shutdown): nobody else knows this browser exists
        _abandon_launch(proc, starting)
        raise
    if found is None:
        _abandon_launch(proc, starting)
        return None, err
    browser = _make_browser(key, found[0], found[1], pid=proc.pid, start_time=start_time, user_data_dir=udd,
                            owned=True, ephemeral=ephemeral, home_key=home_key)
    _write_record(browser)  # now with its DevTools browser id: how another environment tells this launch apart
    with _lock:
        _launching.pop(proc.pid, None)
        _browsers[key] = browser
        _procs[key] = proc
    logger.info("browser: launched Chromium pid %s for %s (port %s)", proc.pid, key, found[0])
    return browser, None


def _abandon_launch(proc: subprocess.Popen, starting: LaunchedBrowser) -> None:
    with _lock:
        _launching.pop(proc.pid, None)
    if proc.poll() is None:  # one that already exited is reaped by poll(); its pid is no longer ours
        _terminate(proc.pid, starting.start_time, proc)
    _remove_record(starting)
    if starting.ephemeral:
        shutil.rmtree(starting.user_data_dir, ignore_errors=True)


# ── Lookup / stop ────────────────────────────────────────────────────────────────────────────────────

def _alive(browser: LaunchedBrowser) -> bool:
    proc = _procs.get(browser.key)
    if proc is not None and browser.owned:
        return proc.poll() is None
    if browser.pid is None:
        return False
    if not _pid_alive(browser.pid):
        return False
    return browser.start_time is None or _start_time(browser.pid) == browser.start_time


def get_browser(key: str) -> Optional[LaunchedBrowser]:
    """The live browser for ``key``, or None (a dead one is dropped and its leftovers removed)."""
    with _lock:
        browser = _browsers.get(key)
        if browser is None:
            return None
        if not browser.foreign:
            if _alive(browser):
                return browser
            logger.info("browser: Chromium for %s (pid %s) is gone", key, browser.pid)
            _forget(key, browser)
            return None
    if _foreign_alive(browser):  # an HTTP probe: never under the process-wide lock
        return browser
    with _lock:
        if _browsers.get(key) is browser:
            logger.info("browser: the Chromium another environment ran for %s is gone", key)
            _forget(key, browser)
    return None


def _foreign_alive(browser: LaunchedBrowser) -> bool:
    """A browser another environment holds is alive while its profile still names this endpoint and it answers."""
    found = _read_devtools_file(browser.user_data_dir)
    return (found is not None and browser.cdp_url == f"ws://127.0.0.1:{found[0]}{found[1]}"
            and _devtools_endpoint_live(*found))


def _heir_key(browser: LaunchedBrowser) -> Optional[str]:
    """Another key of this process on the same shared-profile Chromium (one dir reached by two paths, e.g. a
    symlink): it inherits this process's hold, so dropping ``browser``'s key must neither drop the lease nor
    stop the browser."""
    if browser.ephemeral:
        return None
    with _lock:
        return next((k for k, b in _browsers.items() if b is not browser and not b.ephemeral
                     and _same_browser(b, browser)), None)


def _same_browser(a: LaunchedBrowser, b: LaunchedBrowser) -> bool:
    """One launch: same profile (realpath), pid, start time and launch record (whose name carries the PID namespace
    and boot its pid belongs to); a browser of another environment also by the DevTools browser id it serves (two
    containers can give two browsers one pid number, neither with a start time readable here)."""
    return (a.foreign == b.foreign and a.pid == b.pid and a.start_time == b.start_time
            and _record_name(a) == _record_name(b) and _same_profile(a.user_data_dir, b.user_data_dir)
            and (not a.foreign or a.cdp_url == b.cdp_url))


def _forget(key: str, browser: LaunchedBrowser, *, drop_record: bool = True) -> None:
    proc = None
    heir = _heir_key(browser)
    if _browsers.get(key) is browser:  # a replacement registered since is not this browser's to drop
        _browsers.pop(key)
        proc = _procs.pop(key, None)
    if proc is not None and heir is not None:
        _procs.setdefault(heir, proc)  # the heir stops (and reaps) it when it is the last one out
        proc = None
    if proc is not None:
        try:
            proc.wait(timeout=0)  # reap the zombie
        except subprocess.TimeoutExpired:
            pass
    if not browser.ephemeral and heir is None:
        record = _record_path(browser)
        if record is not None:
            from tools.browser_launcher_holders import drop_lease
            drop_lease(record)
    if browser.owned and drop_record:
        _remove_record(browser)
        if browser.ephemeral:
            shutil.rmtree(browser.user_data_dir, ignore_errors=True)


def _terminate(pid: Optional[int], start_time: Optional[int], proc: Optional[subprocess.Popen]) -> None:
    """Parent-first teardown (tools/AGENTS.md): SIGTERM the browser alone, then its snapshot survivors."""
    if not pid:
        return
    from tools.process_registry import ProcessRegistry
    try:
        ProcessRegistry._terminate_host_pid(pid, start_time)
    except OSError as exc:
        logger.debug("browser: terminating Chromium pid %s failed: %s", pid, exc)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def stop_browser(key: str) -> None:
    """Let go of ``key``'s browser. A throwaway one this process launched is stopped. A shared-profile one is
    stopped only by its LAST holder across Hermes processes (:func:`_last_holder_out`); one a human started is
    never stopped. Holds ``key``'s launch lock: a relaunch of ``key`` waits for this stop instead of
    registering a replacement on the dir it is about to delete."""
    with _key_lock(key):
        _stop_locked(key)


def _stop_locked(key: str) -> None:
    with _lock:
        browser = _browsers.get(key)
        proc = _procs.get(key)
    if browser is None:
        return
    if browser.ephemeral:
        # A throwaway profile is this process's alone: no record, no lock, nobody else can be driving it.
        if browser.owned:
            _terminate(browser.pid, browser.start_time, proc)
        with _lock:
            _forget(key, browser)
        return
    record = _record_path(browser)
    if record is None:
        # A persistent profile whose launcher state dir is gone or unusable cannot be serialised, so this process
        # cannot prove it is the last holder: it must not stop the browser (the same fail-closed rule the launch
        # path applies when it refuses a profile it cannot lock).
        logger.warning("browser: not stopping Chromium pid %s on %s: its launcher state dir is unavailable",
                       browser.pid, browser.user_data_dir)
        with _lock:
            _forget(key, browser, drop_record=False)
        return
    from tools import browser_launcher_holders as holders
    with holders.records_lock(record.parent):  # the decision and the forget are one step: another key may be the heir
        blocked = _legacy_guard(record.parent.name)
        if blocked is not None:
            # A holder from before this version may still be driving it: do not stop this browser. The lease is
            # _forget()'s to drop, and only when no other key of this process still holds the same browser (what
            # _heir_key() decides) — so nothing here touches the lease itself.
            logger.warning("browser: not stopping Chromium pid %s on %s — %s", browser.pid, browser.user_data_dir,
                           blocked)
            with _lock:
                _forget(key, browser, drop_record=False)
            return
        if _heir_key(browser) is None and _last_holder_out(record, browser):
            _terminate(browser.pid, browser.start_time, proc)
            record.unlink(missing_ok=True)
        with _lock:
            _forget(key, browser, drop_record=False)


def _last_holder_out(record: Path, browser: LaunchedBrowser) -> bool:
    """Drop this process's hold on shared-profile ``browser`` (caller holds the records lock). True when it
    must stop it now: no other live Hermes process holds it, and Hermes launched it (a launch record names
    it). Otherwise a record this process holds, or whose owner is gone, goes to a live holder."""
    from tools import browser_launcher_holders as holders
    holders.drop_lease(record)
    if browser.foreign:  # its pid names nobody here: stopped (or reaped) only from its own environment
        return False
    others = holders.other_live_holders(record, browser.start_time)
    data = _load_record(record) if record.exists() else None
    if data is not None and not _record_is(data, browser):
        data = None  # an earlier browser on a recycled pid
    held_elsewhere = data is not None and not holders.held_here(data, "owner_pid") and _owner_alive(data)
    heir = holders.successor(others)
    if heir is not None and data is not None and not held_elsewhere:
        holders.hand_over(record, data, heir)
    if others or held_elsewhere:
        return False
    return browser.owned or data is not None


def stop_all_browsers() -> None:
    """Stop every browser this process launched. Idempotent."""
    with _lock:
        keys = list(_browsers)
    for key in keys:
        stop_browser(key)


# ── Per-profile records + orphan reap ────────────────────────────────────────────────────────────────

def _ephemeral_root(home: str) -> Path:
    """Where ``home``'s throwaway profiles live; the reaper deletes nothing outside it."""
    return Path(home) / "cache" / "browser" / "chromium"


def _records_dir(home: str) -> Path:
    """``home``'s throwaway-profile records (not the process's home): a crashed owner's Chromium is reaped by the
    next Hermes process serving that profile. Read only after ``_private_dir`` vouches for it."""
    return Path(home) / "cache" / "browser" / "chromium-records"


def _is_private_dir(path: Path) -> bool:
    """A real directory (not a symlink) owned by this user and closed to group/other writes. Windows has no
    uid and synthesizes the mode bits, so there only the directory check applies."""
    try:
        st = path.lstat()
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode):
        return False
    return not hasattr(os, "getuid") or (st.st_uid == os.getuid() and not st.st_mode & 0o022)


def _private_dir(path: Path) -> Optional[Path]:
    """``path``, created 0700 when missing; None (logged) when an existing one is not private to this user."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    if _is_private_dir(path):
        return path
    logger.debug("browser: not using %s for launcher state: not a private directory of this user", path)
    return None


def _shared_profiles_root() -> Path:
    """Launcher state of persistent (shared) profiles, per OS USER rather than per ``$HOME``: two processes pinning
    one ``browser.user_data_dir`` drive its one Chromium, so they must see one launch record, one lease set and one
    lock — whichever ``$HOME`` Hermes handed them. In a container that is the whole point: the gateway runs on
    ``HOME={HERMES_HOME}`` while the tool children it spawns run on ``HOME={HERMES_HOME}/home``
    (``hermes_constants.get_subprocess_home``), so a root resolved from ``$HOME`` split one profile's state in two
    and a holder's stop then read no peer. ``HERMES_GATEWAY_LOCK_DIR``, then an absolute ``XDG_STATE_HOME``, still
    pick the root explicitly. One root, one lock, one record/lease protocol: :func:`_legacy_guard` keeps a launcher
    from entering a second ownership domain implied by pre-fix state."""
    from gateway.status import _get_lock_dir
    return _get_lock_dir(os_user=True) / "browser-shared-profiles"


def _canonical_shared_root() -> Optional[Path]:
    """The one shared-profile root, or ``None`` when no stable per-OS-user root can be resolved.

    ``None`` costs a persistent profile its launcher state, so it is not a fallback: a launch/attach is refused and
    the reaper skips, because the alternative — resolving from ``$HOME`` again — is exactly the split this change
    removes."""
    try:
        return _shared_profiles_root()
    except RuntimeError as exc:
        logger.debug("browser: shared-profile launcher state unavailable: %s", exc)
        return None


def _legacy_state_roots(homes=()) -> list[Path]:
    """Shared-profile roots a PRE-FIX launcher resolved for the synthetic profile ``HOME`` Hermes itself creates.

    That HOME is ``{HERMES_HOME}/home``: for this process and for every served home path handed in (the janitor
    runs across profiles, so it passes its served homes). Nothing else is scanned — an arbitrary ``$HOME`` is not
    evidence of this installation's state, and reading one would reach into a user's real home.

    An explicit root means the caller has already said where state lives, so nothing is probed at all:
    ``HERMES_GATEWAY_LOCK_DIR`` and an absolute ``XDG_STATE_HOME`` keep their present meaning.
    """
    if os.getenv("HERMES_GATEWAY_LOCK_DIR"):
        return []
    if os.path.isabs(os.getenv("XDG_STATE_HOME") or ""):
        return []
    from hermes_constants import _profile_home_path
    # One discovery rule for launch/attach, stop, the foreign check and the reaper: the homes this process serves,
    # exactly what the janitor threads through (``_served_home_keys``). Each contributes the synthetic HOME Hermes
    # makes for it — ``{home}/home`` — and nothing else is looked at.
    candidates: list[Path] = []
    profile_home = _profile_home_path(None)
    if profile_home:
        candidates.append(Path(profile_home))
    for home in sorted(_served_home_keys(homes)):
        synthetic = Path(home) / "home"
        # ``is_dir()`` would report False for an unreadable directory; only a definite absence skips it here, so an
        # inaccessible pre-fix root is still examined (and then refused by the guard).
        if _probe_state_dir(synthetic) != _MISSING and synthetic not in candidates:
            candidates.append(synthetic)
    canonical = _canonical_shared_root()
    current = os.path.realpath(str(canonical)) if canonical is not None else None
    roots: list[Path] = []
    for candidate in candidates:
        root = candidate / ".local" / "state" / "hermes" / "gateway-locks" / "browser-shared-profiles"
        if (current is not None and os.path.realpath(str(root)) == current) or root in roots:
            continue
        roots.append(root)
    return roots


def _probe_state_dir(path: Path) -> str:
    """``_PRESENT`` / ``_MISSING`` / ``_UNCERTAIN`` for a directory this process may have to judge.

    ``os.path.isdir`` and ``os.path.lexists`` answer False for "there but not readable", which would let an
    inaccessible pre-fix directory pass as absent. Only a definite ``FileNotFoundError`` means absent here; every
    other failure, a symlink, and a non-directory are UNCERTAIN and make the caller refuse.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return _MISSING
    except OSError:
        return _UNCERTAIN
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return _UNCERTAIN
    return _PRESENT


def _legacy_state_dirs(state_name: str, homes=()) -> list[Path]:
    """This profile's state dir in each PRE-FIX root that has one *and* is safe to read.

    A directory that cannot be confirmed (unreadable, a symlink, not a directory) is reported by
    :func:`_legacy_guard` as a refusal; here it is simply not read."""
    dirs = []
    for root in _legacy_state_roots(homes):
        state = root / state_name
        if _probe_state_dir(state) == _PRESENT and _is_private_dir(state):
            dirs.append(state)
    return dirs


_PRESENT, _MISSING, _UNCERTAIN = "present", "missing", "uncertain"


def _read_only_json(path: Path) -> tuple[str, Optional[dict]]:
    """``(state, data)`` for a launch record or holder lease, read WITHOUT side effects.

    :func:`_load_record` deletes a record it cannot parse; a guard must not write anything, so it reads here and
    keeps "the file is gone" apart from "the file is there but cannot be confirmed". Only the first means there is
    nothing to judge: a permission error, an I/O error or unparseable bytes name an owner, a holder or a browser
    that this process cannot rule out, and the caller refuses instead of calling that safe.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return _MISSING, None
    except (OSError, UnicodeError):
        return _UNCERTAIN, None
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        return _UNCERTAIN, None
    return (_PRESENT, data) if isinstance(data, dict) else (_UNCERTAIN, None)


def _legacy_guard_message(state: Path, why: str) -> str:
    """The refusal a guard raises, naming the pre-fix state and what the operator can do about it."""
    return (f"This browser profile still has launcher state from before this Hermes version, in {state}, where "
            f"{why}. Hermes will not start, attach to or stop a browser on it: that state cannot be re-homed "
            "safely from here, and starting anyway would put two launchers in charge of one Chromium. Stop the "
            "older Hermes processes that used this profile, then remove that directory once no browser runs on it.")


def _legacy_guard(state_name: str, homes=()) -> Optional[str]:
    """Why a launch, attach or stop of the profile whose state dir is ``state_name`` must not proceed, or ``None``.

    Judged from PRE-FIX state only: a root this build no longer writes, left by a launcher that resolved its state
    from a profile ``HOME`` Hermes repointed. Such state can still name a live launcher, a live holder or a running
    browser, and this version does not re-home live ownership — it stops instead of entering a second ownership
    domain that would let one side stop a browser the other is still driving.

    READ-ONLY by construction (``_read_only_json``, no lock of the pre-fix root, no create/move/copy/delete), so a
    guard never repairs, migrates or removes what it inspects — it only refuses. State it cannot confirm is refused
    as well: a directory that is absent is nothing to judge, but one that exists and cannot be read, listed or
    parsed may still name a live holder, and calling that safe would be a guess. ``None`` is the normal answer, and
    always the answer on an install that never ran the older layout. This is a safety check, not a cross-version
    mutex: pre-upgrade launchers must be stopped.
    """
    from tools.browser_launcher_holders import process_alive, same_namespace
    for root in _legacy_state_roots(homes):
        state = root / state_name
        probe = _probe_state_dir(state)
        if probe == _MISSING:
            continue
        if probe == _UNCERTAIN:
            return _legacy_guard_message(state, "it cannot be read or confirmed")
        if not _is_private_dir(state):
            # The same rule every other launcher directory is held to: a state dir that is not this user's private
            # directory is not judged, it is refused, because nothing in it can be trusted either way.
            return _legacy_guard_message(state, "it is not a private directory of this user")
        try:
            entries = sorted(p for p in state.iterdir() if p.suffix in (".json", ".lease"))
        except OSError:
            return _legacy_guard_message(state, "it cannot be listed")
        for path in entries:
            status, data = _read_only_json(path)
            if status == _MISSING:
                continue
            if status == _UNCERTAIN or data is None:
                return _legacy_guard_message(state, f"{path.name} cannot be read or confirmed")
            recorded = data.get("user_data_dir")
            if isinstance(recorded, str) and _profile_digest(recorded)[:32] != state.name:
                continue
            if path.suffix == ".lease":
                if not (same_namespace(data) and not process_alive(data.get("holder_pid"),
                                                                  data.get("holder_start_time"))):
                    return _legacy_guard_message(state, "a Hermes from before this version still holds it")
                continue
            if _owner_alive(data):
                return _legacy_guard_message(state, "a Hermes from before this version may still own it")
            pid = data.get("pid")
            if isinstance(pid, int) and recorded and _pid_alive(pid) and _cmdline_names_dir(pid, recorded):
                return _legacy_guard_message(state, "a Chromium started before this version is still running on it")
    return None


def _profile_digest(user_data_dir: str) -> str:
    """One profile dir, however it is spelled (a symlink, a trailing slash)."""
    return hashlib.sha256(os.path.realpath(user_data_dir).encode()).hexdigest()


def _same_profile(recorded, user_data_dir: str) -> bool:
    return isinstance(recorded, str) and os.path.realpath(recorded) == os.path.realpath(user_data_dir)


def _devtools_id(browser: LaunchedBrowser) -> Optional[str]:
    """The ``/devtools/browser/<id>`` path of ``browser``'s endpoint (new per launch); None before it is up."""
    at = browser.cdp_url.find("/devtools/browser/")
    return browser.cdp_url[at:] if at >= 0 else None


def _record_is(record: dict, browser: LaunchedBrowser) -> bool:
    """``record`` describes this launch of ``browser``: same pid and profile, and its start time; for a browser of
    another environment (its pid is another namespace's number and its start time unreadable here), a record from
    another namespace naming the DevTools browser id its endpoint serves."""
    from tools.browser_launcher_holders import same_namespace
    if record.get("pid") != browser.pid or not _same_profile(record.get("user_data_dir"), browser.user_data_dir):
        return False
    if browser.foreign:
        return not same_namespace(record) and record.get("devtools") is not None and (
            record.get("devtools") == _devtools_id(browser))
    return record.get("start_time") == browser.start_time


def _shared_state_dir(user_data_dir: str) -> Optional[Path]:
    """Shared profile ``user_data_dir``'s own private dir (its launch record, leases and lock)."""
    canonical = _canonical_shared_root()
    root = _private_dir(canonical) if canonical is not None else None
    return _private_dir(root / _profile_digest(user_data_dir)[:32]) if root is not None else None


def _state_dir(browser: LaunchedBrowser) -> Optional[Path]:
    """The private dir holding ``browser``'s launch record, leases and lock: its home's records dir for a
    throwaway profile, else the shared profile's own dir under :func:`_shared_profiles_root`."""
    if browser.ephemeral:
        return _private_dir(_records_dir(browser.home_key))
    return _shared_state_dir(browser.user_data_dir)


def _record_path(browser: LaunchedBrowser) -> Optional[Path]:
    records = _state_dir(browser)
    if records is None:
        return None
    return records / _record_name(browser)


def _record_name(browser: LaunchedBrowser) -> str:
    """The launch record ``browser``'s leases are keyed by: the one it was bound to, else this namespace's name."""
    from tools.browser_launcher_holders import ns_tag
    return browser.record_name or f"{_profile_digest(browser.user_data_dir)[:16]}-{browser.pid}{ns_tag()}.json"


def _write_record(browser: LaunchedBrowser) -> None:
    from tools import browser_launcher_holders as holders
    from utils import atomic_json_write
    record = {"owner_pid": os.getpid(), "owner_start_time": _start_time(os.getpid()), "pid": browser.pid,
              "start_time": browser.start_time, "user_data_dir": browser.user_data_dir,
              "home_key": browser.home_key, "ephemeral": browser.ephemeral, "ns": holders.ns_identity(),
              "devtools": _devtools_id(browser)}
    try:
        path = _record_path(browser)
        if path is not None:
            atomic_json_write(path, record, indent=None)
    except OSError as exc:
        logger.debug("browser: could not write launch record for pid %s: %s", browser.pid, exc)


def _remove_record(browser: LaunchedBrowser) -> None:
    try:
        path = _record_path(browser)
        if path is not None:
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("browser: could not remove launch record for pid %s: %s", browser.pid, exc)


def _owner_alive(record: dict) -> bool:
    """The record's owner is alive, or cannot be proven gone (it was written in another PID namespace)."""
    from tools.browser_launcher_holders import same_namespace
    if not same_namespace(record):
        return True
    owner = record.get("owner_pid")
    if not isinstance(owner, int) or not _pid_alive(owner):
        return False
    expected = record.get("owner_start_time")
    return expected is None or _start_time(owner) == expected


def _cmdline_names_dir(pid: int, user_data_dir: str) -> bool:
    """True only when ``pid``'s argv carries ``--user-data-dir=`` naming ``user_data_dir`` (however either is
    spelled: a symlink, a trailing slash)."""
    try:
        import psutil
        argv = psutil.Process(pid).cmdline()
    except Exception:  # health: allow BLE001 -- unreadable identity fails closed (no kill)
        return False
    return any(tok.startswith("--user-data-dir=") and _same_profile(tok.split("=", 1)[1], user_data_dir)
               for tok in argv)


def _inside_ephemeral_root(user_data_dir: str, home: Optional[str]) -> bool:
    if home is None:  # a shared-profile state dir holds no throwaway profiles
        return False
    try:
        rel = Path(user_data_dir).resolve().relative_to(_ephemeral_root(home).resolve())
    except (OSError, ValueError):
        return False
    return bool(rel.parts)


def _is_recorded_browser(record: dict, home: Optional[str]) -> bool:
    pid, udd = record.get("pid"), record.get("user_data_dir")
    if not isinstance(pid, int) or not isinstance(udd, str) or not _pid_alive(pid):
        return False
    if record.get("ephemeral") and not _inside_ephemeral_root(udd, home):
        logger.debug("browser: ignoring launch record for pid %s: %s is not a launcher profile dir", pid, udd)
        return False
    expected = record.get("start_time")
    if expected is None or _start_time(pid) != expected:
        return False  # no recorded fingerprint, or a recycled pid: never signal on argv alone
    if not _cmdline_names_dir(pid, udd):
        logger.warning("browser: refusing to reap pid %s: its command line does not name %s", pid, udd)
        return False
    return True


def _tracked_browser(record: dict) -> Optional[LaunchedBrowser]:
    with _lock:  # a browser another environment holds has a pid of THAT namespace: it tracks no record here
        return next((b for b in (*_browsers.values(), *_launching.values())
                     if not b.foreign and _record_is(record, b)), None)


def _adopt_record(browser: LaunchedBrowser) -> None:
    """A crashed owner's record still names the browser this process now uses (attached after the crash):
    take the record over, or every reaper (this process's included) would kill it as an orphan. Only the owner
    changes: the launch as recorded (its ``--user-data-dir`` spelling, DevTools id) stays what verifies it."""
    from utils import atomic_json_write
    path = _record_path(browser)
    if path is None or not path.exists():
        return
    record = _load_record(path)
    if record is not None and _record_is(record, browser) and not _owner_alive(record):
        me = os.getpid()
        atomic_json_write(path, {**record, "owner_pid": me, "owner_start_time": _start_time(me)}, indent=None)


def _load_record(path: Path) -> Optional[dict]:
    """The record, or None (a corrupt one is removed)."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(record, dict):
            return record
    except (OSError, ValueError):
        pass
    path.unlink(missing_ok=True)
    return None


def _reap_record(path: Path, home: Optional[str]) -> bool:
    """Caller holds the records lock."""
    from tools import browser_launcher_holders as holders
    record = _load_record(path)
    if record is None or not holders.same_namespace(record):  # another namespace's pids name nobody here
        return False
    tracked = _tracked_browser(record)
    if tracked is not None:  # in use here, whoever launched it
        _adopt_record(tracked)
        return False
    if record.get("owner_pid") != os.getpid() and _owner_alive(record):
        return False
    others = holders.other_live_holders(path, record.get("start_time"))
    if others:  # another live Hermes process still drives it
        heir = holders.successor(others)
        if heir is not None:
            holders.hand_over(path, record, heir)
        return False
    reaped = _is_recorded_browser(record, home)
    if reaped:
        _terminate(record["pid"], record.get("start_time"), None)
        logger.info("browser: reaped orphaned Chromium pid %s (%s)", record["pid"], record["user_data_dir"])
    udd = record.get("user_data_dir")
    if (record.get("ephemeral") and isinstance(udd, str) and _inside_ephemeral_root(udd, home)
            and (reaped or not _browser_running_on(udd))):
        shutil.rmtree(udd, ignore_errors=True)  # its owner is gone: no launch reuses a per-process dir
    path.unlink(missing_ok=True)
    return reaped


def _reap_home(home: str) -> int:
    return _reap_state_dir(_records_dir(home), home)


def _reap_shared_profiles(homes=()) -> int:
    """Every shared profile's records, whichever home launched its browser.

    ``homes`` is the janitor's served home paths: they are the only pre-fix roots that are looked at, because a
    profile ``HOME`` is ``{HERMES_HOME}/home`` for each of them and nothing else is this installation's state."""
    canonical = _canonical_shared_root()
    if canonical is None or _probe_state_dir(canonical) != _PRESENT or not _is_private_dir(canonical):
        return 0
    try:
        states = sorted(canonical.iterdir())
    except OSError as exc:
        logger.debug("browser: shared-profile launcher state unavailable in %s: %s", canonical, exc)
        return 0
    return sum(_reap_state_dir(state, None, homes) for state in states)


def _reap_state_dir(records: Path, home: Optional[str], homes=()) -> int:
    """Reap one launcher state dir (``home``'s records, or a shared profile's dir when ``home`` is None). A
    shared profile whose lock is taken (a first launch holds it for up to its whole readiness wait) is left to the
    next pass, and so is one whose PRE-FIX state still names a holder — judged under this profile's lock, inside the
    same mutual-exclusion domain, never beside it."""
    if not os.path.lexists(records):
        return 0
    if not _is_private_dir(records):
        logger.debug("browser: not reaping from %s: not a private directory of this user", records)
        return 0
    from tools import browser_launcher_holders as holders
    with holders.records_lock(records, blocking=home is not None) as locked:
        if not locked:
            logger.debug("browser: %s is busy (a launch or stop holds it); reaping it next pass", records)
            return 0
        if home is None and _legacy_guard(records.name, homes) is not None:
            logger.debug("browser: not reaping %s: pre-fix launcher state still names a holder", records)
            return 0
        try:
            paths = sorted(records.glob("*.json"))
        except OSError as exc:
            logger.debug("browser: launch records unavailable in %s: %s", records, exc)
            return 0
        reaped = sum(1 for path in paths if _reap_record(path, home))
        holders.drop_dead_leases(records)
    return reaped


_THROWAWAY_DIR = re.compile(r"^[0-9a-f]{16}-(\d+)(?:-(\d+))?(-n[0-9a-f]{12})?$")  # _default_user_data_dir's names


def _reclaim_throwaway_dirs(home: str) -> None:
    """Remove ``home``'s throwaway profiles whose owning process is gone (pid and start time in the name): one
    that died before writing its launch record left nothing else that would. The dir of a live owner, or of an
    owner in another PID namespace or boot (its name's ``ns_tag`` differs or is missing), is never touched; a
    name without a start time is reclaimed only once its pid is gone."""
    root = _ephemeral_root(home)
    if not os.path.lexists(root) or not _is_private_dir(root):
        return
    from tools.browser_launcher_holders import ns_tag, process_alive
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.debug("browser: throwaway profiles unavailable in %s: %s", root, exc)
        return
    for entry in entries:
        match = _THROWAWAY_DIR.match(entry.name)
        if match is None or (match.group(3) or "") != ns_tag() or int(match.group(1)) == os.getpid():
            continue  # not a launcher dir, another PID namespace's (its owner cannot be checked here), or ours
        if process_alive(int(match.group(1)), int(match.group(2)) if match.group(2) else None):
            continue
        if _browser_running_on(str(entry)):  # its record was lost or could not prove the pid: never pull the dir
            logger.debug("browser: keeping the throwaway profile %s: a Chromium still runs on it", entry)
            continue
        shutil.rmtree(entry, ignore_errors=True)  # a symlink is refused, never followed
        logger.info("browser: removed the throwaway profile %s of a Hermes process that is gone", entry)


def _served_home_keys(homes) -> set[str]:
    """The launch profile, every profile this process launched a browser for, and ``homes``."""
    from hermes_constants import get_process_hermes_home, hermes_home_key
    with _lock:
        keys = set(_served_homes)
    keys.add(hermes_home_key(get_process_hermes_home()))
    keys.update(hermes_home_key(home) for home in homes)
    return keys


def reap_orphaned_browsers(homes=()) -> int:
    """Stop Chromium processes whose owning Hermes process is gone, in every profile this process serves
    (``_served_home_keys``; the janitor and atexit run outside any profile scope, so each home's records
    are read by path); returns the count. A pid is signalled only after its start time AND its
    ``--user-data-dir`` argv token match the record. Throwaway profiles of owners that are gone go too."""
    reaped = _reap_shared_profiles(homes)
    for home in sorted(_served_home_keys(homes)):
        reaped += _reap_home(home)
        _reclaim_throwaway_dirs(home)
    return reaped
