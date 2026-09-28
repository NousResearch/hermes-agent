"""Bot Desktop runtime: one headless Xfce desktop per Hermes profile, served over RFB on a private
Unix socket, viewed and driven from Hermes Desktop.

Layout under ``<HERMES_HOME>/bot-desktop/``: ``display`` (allocated X display number), ``rfb.sock``
(Xvnc RFB Unix socket, 0600), ``Xauthority``, ``env`` (DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS
published by the launcher once Xfce's bus exists), ``launcher.pid``, ``launcher.log``, ``xdg/``
(per-profile XDG_CONFIG_HOME so two profiles never share xfconf). Everything is profile-scoped via
``get_hermes_home()`` so N profiles in one gateway get N desktops: one screen per bot on the shared
machine.

The launcher is ``launcher.sh`` next to this module; :func:`desktop_env` is what cua-driver and headed
Chromium spawns merge in so the agent acts on this profile's screen and nowhere else.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LAUNCHER = Path(__file__).with_name("launcher.sh")

# Display numbers below 10 collide with real seats and default Xvfb recipes (:99 is popular too); scan a
# private band and record the choice so restarts reuse it.
_DISPLAY_MIN, _DISPLAY_MAX = 20, 89

# Binaries the launcher execs; the package hint is per distro family.
REQUIRED_BINARIES = ("Xvnc", "xfwm4", "xfce4-panel", "xfdesktop", "xfsettingsd", "dbus-run-session",
                     "xauth", "xdpyinfo", "setxkbmap", "xprop")

# Which package in each distro list ships each required binary. Fedora retired the xorg-x11-utils /
# xorg-x11-server-utils umbrellas (per-binary packages since F35) and dnf5 refuses the whole transaction on
# one unknown name, so every binary must map to a package that still resolves; the test suite checks that
# each mapped package is in PACKAGES for its manager.
BINARY_PACKAGES = {
    "apt": {"Xvnc": "tigervnc-standalone-server", "xfwm4": "xfwm4", "xfce4-panel": "xfce4-panel",
            "xfdesktop": "xfdesktop4", "xfsettingsd": "xfce4-settings", "dbus-run-session": "dbus-x11",
            "xauth": "xauth", "xdpyinfo": "x11-utils", "setxkbmap": "x11-xkb-utils", "xprop": "x11-utils"},
    # tigervnc-x11-server is the real package (tigervnc-server-minimal is only a Provides on it); dbus-run-session
    # is in dbus-daemon (dbus-x11 ships dbus-launch only).
    "dnf": {"Xvnc": "tigervnc-x11-server", "xfwm4": "xfwm4", "xfce4-panel": "xfce4-panel",
            "xfdesktop": "xfdesktop", "xfsettingsd": "xfce4-settings", "dbus-run-session": "dbus-daemon",
            "xauth": "xorg-x11-xauth", "xdpyinfo": "xdpyinfo", "setxkbmap": "setxkbmap", "xprop": "xprop"},
    "pacman": {"Xvnc": "tigervnc", "xfwm4": "xfwm4", "xfce4-panel": "xfce4-panel", "xfdesktop": "xfdesktop",
               "xfsettingsd": "xfce4-settings", "dbus-run-session": "dbus",
               "xauth": "xorg-xauth", "xdpyinfo": "xorg-xdpyinfo", "setxkbmap": "xorg-setxkbmap", "xprop": "xorg-xprop"},
}

PACKAGES = {
    "apt": ["tigervnc-standalone-server", "xfce4-panel", "xfwm4", "xfdesktop4", "xfce4-settings",
            "xfce4-terminal", "dbus-x11", "x11-xserver-utils", "x11-utils", "x11-xkb-utils", "xauth",
            "fonts-dejavu-core"],
    "dnf": ["tigervnc-x11-server", "xfce4-panel", "xfwm4", "xfdesktop", "xfce4-settings",
            "xfce4-terminal", "dbus-daemon", "xsetroot", "xset", "xdpyinfo", "xprop", "xorg-x11-xauth", "setxkbmap",
            "dejavu-sans-fonts"],
    "pacman": ["tigervnc", "xfce4-panel", "xfwm4", "xfdesktop", "xfce4-settings", "xfce4-terminal", "dbus",
               "xorg-xsetroot", "xorg-xset", "xorg-xdpyinfo", "xorg-xprop", "xorg-xauth", "xorg-setxkbmap",
               "ttf-dejavu"],
}


def state_dir() -> Path:
    return get_hermes_home() / "bot-desktop"


def is_supported_host() -> bool:
    return sys.platform.startswith("linux")


def missing_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if shutil.which(b) is None]


def package_manager() -> Optional[str]:
    for pm in ("apt-get", "dnf", "pacman"):
        if shutil.which(pm):
            return "apt" if pm == "apt-get" else pm
    return None


def install_command() -> Optional[str]:
    """The distro command that installs the Bot Desktop packages, as the human would type it on THIS host:
    prefixed with ``sudo`` unless Hermes already runs as root (the official Docker image is uid 0 with no
    sudo binary), so it is both what the pane shows and what :mod:`tools.bot_desktop.install` runs."""
    pm = package_manager()
    if pm is None:
        return None
    pkgs = " ".join(PACKAGES[pm])
    body = {
        "apt": f"apt-get install -y --no-install-recommends {pkgs}",
        "dnf": f"dnf install -y {pkgs}",
        "pacman": f"pacman -S --needed --noconfirm {pkgs}",
    }[pm]
    return body if is_root() else f"sudo {body}"


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


@dataclass
class DesktopStatus:
    profile: str
    supported: bool
    installed: bool
    missing: list[str]
    running: bool
    pid: Optional[int]
    display: Optional[str]
    socket: Optional[str]
    geometry: str
    install_command: Optional[str]
    browser: Optional[str]  # headed Chromium the dock's Browser icon and agent-browser share; None = no headed browser
    blocker: Optional[str] = None  # why start() would refuse right now (memory); None = may start
    memory_available_mb: Optional[int] = None
    memory_limit_mb: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    """A zombie is dead for our purposes: a SIGKILLed launcher stays a zombie in the gateway until the next
    Popen reaps it, and reporting it as running would hide its orphaned X server behind a live status."""
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _create_time(pid: int) -> Optional[float]:
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OverflowError, ValueError):
        return None


def _launcher_pid() -> Optional[int]:
    """The live launcher's pid, or None. ``launcher.pid`` holds ``"<pid> <create_time>"``: a recycled pid
    with a different start time is somebody else's process and must never be reported as ours nor
    killed by :func:`stop`. The pre-identity single-number format is treated as not running."""
    raw = _read(state_dir() / "launcher.pid")
    pid_s, _, born_s = (raw or "").partition(" ")
    if not pid_s.isdigit() or not born_s:
        return None
    try:
        pid, born = int(pid_s), float(born_s)
    except ValueError:
        return None
    actual = _create_time(pid)
    return pid if actual is not None and abs(actual - born) < 0.01 and _pid_alive(pid) else None


def _recorded_launcher_pid() -> Optional[int]:
    """The pid ``launcher.pid`` names, alive or not (the orphan sweep matches process groups against it)."""
    pid_s, _, born_s = (_read(state_dir() / "launcher.pid") or "").partition(" ")
    return int(pid_s) if pid_s.isdigit() and born_s else None


_X_LOCK_DIR = Path("/tmp")  # no-tmp: ok — X servers write .X<n>-lock here by protocol (tests point it at a scratch dir)
_X_UNIX_TABLE = Path("/proc/net/unix")  # the kernel's list of bound Unix sockets (tests point it at a fixture)


def _x_lock_pid(num: int) -> Optional[int]:
    try:
        return int((_X_LOCK_DIR / f".X{num}-lock").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _x_socket_bound(num: int) -> bool:
    """A running X server keeps ``@/tmp/.X11-unix/X<n>`` (abstract namespace) bound for its whole life; the
    kernel drops it only when the process exits. A /tmp reaper can remove the lock file under a live Xvnc,
    and only this binding then still says the number is taken (a new server on it dies 'already running')."""
    try:
        lines = _X_UNIX_TABLE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    # no-tmp: ok — detects the X server's display socket at the path the X11 protocol fixes
    return any(line.split()[-1].lstrip("@") == f"/tmp/.X11-unix/X{num}" for line in lines if line.strip())


def _display_in_use(num: int) -> bool:
    """A live X server owns ``:num``: its lock file names a running pid, or its X11 socket is bound. A lock
    left by a crashed server (dead pid, no socket) does not count, so the number can be reclaimed."""
    pid = _x_lock_pid(num)
    return (pid is not None and _pid_alive(pid)) or _x_socket_bound(num)


def _reap_orphaned_server(sd: Path) -> bool:
    """Caller holds ``start.lock`` and has established that no live launcher exists. The launcher runs Xvnc in
    its own session, so a SIGKILLed launcher leaves the X server alive, holding the display and ``rfb.sock``;
    ``status()`` keys on the launcher and says stopped, and a naive restart allocates a second server next
    to it and overwrites the socket path both now claim. The X lock of the recorded display names that
    server: it is ours when it sits in the dead launcher's process group or its command line binds OUR
    socket. When a /tmp reaper took the lock too (or the failed launch dropped ``display``), the socket path
    on the Xvnc command line is the remaining handle — without it one X server leaks per occurrence. Kill it
    (group first), drop the state it left, and report whether anything was signalled."""
    import psutil

    def _binds_our_socket(cmdline: list) -> bool:
        return "Xvnc" in Path(cmdline[0] if cmdline else "").name and str(sd / "rfb.sock") in cmdline

    recorded = _read(sd / "display")
    pid = _x_lock_pid(int(recorded)) if recorded and recorded.isdigit() else None
    if pid is None or not _pid_alive(pid):
        pid = next((p.pid for p in psutil.process_iter(["cmdline"]) if _binds_our_socket(p.info["cmdline"] or [])), None)
        if pid is None:
            return False
    launcher = _recorded_launcher_pid()
    try:
        pgid = os.getpgid(pid)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start/stop)
        cmdline = psutil.Process(pid).cmdline()
    except (ProcessLookupError, psutil.Error):
        return False
    if pgid != launcher and not _binds_our_socket(cmdline):
        return False  # somebody else's server took the number after we died; never touch it
    logger.warning("Bot Desktop launcher %s is gone but its X server (pid %s) survived on :%s; reaping",
                   launcher, pid, recorded or "?")
    _kill_group_then_wait(pgid if pgid == launcher else None, pid)
    if recorded:
        (_X_LOCK_DIR / f".X{recorded}-lock").unlink(missing_ok=True)
    for name in ("launcher.pid", "env", "rfb.sock"):
        (sd / name).unlink(missing_ok=True)
    return True


def _kill_group_then_wait(pgid: Optional[int], pid: int, grace: float = 2.0) -> None:
    """SIGTERM the group (or the lone pid), SIGKILL whatever is still there after ``grace``."""
    def _signal(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if pgid is not None:
                os.killpg(pgid, sig)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start/stop)
            else:
                os.kill(pid, sig)
    def _anything_left() -> bool:
        # The leader dying first is the common case (bash exits on TERM, Xvnc traps it); the group is
        # done only when killpg(0) finds nobody, else a TERM-ignoring descendant keeps the display.
        if pgid is None:
            return _pid_alive(pid)
        try:
            os.killpg(pgid, 0)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start/stop)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _reap_if_ours() -> None:
        # The launcher was Popen'd by whichever gateway started it; a later gateway that stops it holds
        # no Popen, so the dead leader would sit as a zombie in our table (and count as "left").
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, os.WNOHANG)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start/stop)

    _signal(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        _reap_if_ours()
        if not _anything_left():
            return
        time.sleep(0.05)
    _signal(signal.SIGKILL)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start/stop)
    time.sleep(0.1)
    _reap_if_ours()


# Host-wide (every profile allocates from one band), so it lives outside any profile home — but not in
# world-writable /tmp, where a predictable name lets another local user pre-create or squat the file.
_ALLOC_LOCK = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache") / "hermes-bot-desktop-alloc.lock"


@contextlib.contextmanager
def _flocked(path: Path):
    import fcntl  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start)
    with open(path, "a+", encoding="utf-8") as fh:  # windows-footgun: ok — Linux-only runtime
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _pick_display() -> int:
    """Caller holds ``_ALLOC_LOCK``. The recorded number is only reused when no OTHER server holds it now:
    after profile A stops, B may have taken A's number, and A's launcher must never unlink B's socket."""
    recorded = _read(state_dir() / "display")
    if recorded and recorded.isdigit() and not _display_in_use(int(recorded)):
        return int(recorded)
    for num in range(_DISPLAY_MIN, _DISPLAY_MAX + 1):
        if not _display_in_use(num):
            return num
    raise RuntimeError("no free X display number in the Bot Desktop band")


def _allocate_display() -> int:
    _ALLOC_LOCK.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _flocked(_ALLOC_LOCK):
        return _pick_display()


def desktop_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``base_env`` (default ``os.environ``) with this profile's DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS
    merged in when its desktop is running. Unchanged otherwise, so hosts with a real seat keep it.
    Pure: never starts anything (it is called from env builders, status probes and tests)."""
    env = dict(os.environ if base_env is None else base_env)
    published = published_env()
    if published:
        touch_activity()  # a browser / cua-driver spawn is the agent using its screen
        env.update(published)
        env.pop("WAYLAND_DISPLAY", None)  # X11 desktop; a leaked Wayland socket flips GTK/Chromium backends
        from tools.bot_desktop.browser import env_for_agent
        env_for_agent(env)  # same binary + user-data-dir as the dock's Browser icon
    return env


def ensure_started_for_tool() -> None:
    """Tool-boundary hook (``computer_use`` dispatch and the headed Chromium spawn sites of the browser tool): with
    ``bot_desktop.auto_start`` (opt-in, default off) a Linux host that has NO display and the packages installed gets
    its screen started on first use, so a headless gateway works the first time instead of answering "no DISPLAY is
    set". Failure is not an error here; the tool's own "no display" diagnosis is the right message then."""
    if published_env():
        touch_activity()
        return
    if not _should_auto_start(os.environ):
        return
    try:
        start()
    except Exception as exc:
        logger.info("Bot Desktop auto-start skipped: %s", exc)


def _should_auto_start(env: Dict[str, str]) -> bool:
    if not is_supported_host() or env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return False
    if missing_binaries():
        return False
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    return bool(cfg.get("auto_start", False))


# ---- idle auto-stop -------------------------------------------------------------------------------
# A desktop nobody is using still holds ~220 MiB (plus whatever browser was left open). Every use —
# a computer_use action, a browser spawn onto the screen, a viewer attached, a human takeover — stamps
# ``activity``; the gateway's display watcher stops a screen idle past ``bot_desktop.idle_stop_minutes``
# unless a human holds it. The next use starts it again (auto_start or the pane's Start).
DEFAULT_IDLE_STOP_MINUTES = 30


def touch_activity() -> None:
    path = state_dir() / "activity"
    try:
        path.touch()
        os.utime(path, None)
    except OSError:
        pass


def idle_seconds() -> Optional[float]:
    """Seconds since the last stamped use; None when the screen never recorded one (falls back to the
    env file's publish time so a screen started and then forgotten still ages)."""
    for name in ("activity", "env"):
        try:
            return max(0.0, time.time() - (state_dir() / name).stat().st_mtime)
        except OSError:
            continue
    return None


def idle_stop_seconds() -> float:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    try:
        minutes = float(cfg.get("idle_stop_minutes", DEFAULT_IDLE_STOP_MINUTES))
    except (TypeError, ValueError):
        minutes = DEFAULT_IDLE_STOP_MINUTES
    return max(0.0, minutes) * 60


def stop_if_idle() -> bool:
    """Stop this profile's screen when it has been idle past the limit and no human holds it. True when
    it was stopped."""
    limit = idle_stop_seconds()
    if limit <= 0 or _launcher_pid() is None:
        return False
    idle = idle_seconds()
    if idle is None or idle < limit:
        return False
    from tools.bot_desktop import lease as _bd_lease
    if _bd_lease.get().holder == _bd_lease.HUMAN:
        return False
    logger.info("Bot Desktop for profile %s idle for %.0f min; stopping", _profile_name(), idle / 60)
    return stop()


def published_env() -> Dict[str, str]:
    """Variables the launcher wrote once Xfce's private bus existed; empty when the desktop is down."""
    if _launcher_pid() is None:
        return {}
    raw = _read(state_dir() / "env")
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for line in raw.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def rfb_socket_path() -> Optional[Path]:
    sock = state_dir() / "rfb.sock"
    return sock if _launcher_pid() is not None and sock.exists() else None


def geometry() -> str:
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("bot_desktop") or {}
    return str(cfg.get("geometry") or "1440x900")


def status(profile: Optional[str] = None) -> DesktopStatus:
    from tools.bot_desktop import browser as _bd_browser
    from tools.bot_desktop import resources
    missing: list[str] = missing_binaries() if is_supported_host() else list(REQUIRED_BINARIES)
    pid = _launcher_pid()
    env = published_env()
    running = pid is not None and bool(env.get("DISPLAY"))
    mem = resources.memory_info() if is_supported_host() else resources.MemoryInfo(None, None)
    return DesktopStatus(
        profile=profile or _profile_name(),
        supported=is_supported_host(),
        installed=not missing,
        missing=missing,
        running=running,
        pid=pid,
        display=env.get("DISPLAY"),
        socket=str(rfb_socket_path()) if rfb_socket_path() else None,
        geometry=geometry(),
        install_command=install_command() if missing else None,
        browser=_bd_browser.executable() if is_supported_host() else None,
        # A running screen is never "blocked": the check guards the allocation, not the session.
        blocker=None if running or missing or not is_supported_host() else resources.memory_blocker(mem),
        memory_available_mb=mem.available_mb,
        memory_limit_mb=mem.limit_mb,
    )


def _profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def start(*, wait_seconds: float = 15.0) -> DesktopStatus:
    """Start this profile's desktop (idempotent). Blocks until the launcher publishes its env file or
    ``wait_seconds`` pass; raises ``RuntimeError`` naming the blocker.

    Two locks: the per-profile ``start.lock``, held from the running-check to the launcher's publish so two
    start() calls for one profile spawn one launcher (the loser sees it running), and the host-wide
    display-allocation lock, held only until this Xvnc has written ``/tmp/.X<n>-lock`` (a second profile
    picking the same number before that would fail and its stale-lock cleanup could remove our socket).
    Holding it for the whole Xfce bring-up serialized every profile's start behind one desktop launch."""
    if not is_supported_host():
        raise RuntimeError("Bot Desktop runs on Linux gateway hosts only")
    missing = missing_binaries()
    if missing:
        hint = install_command() or "install TigerVNC (Xvnc) and the Xfce core components"
        raise RuntimeError(f"Bot Desktop needs {', '.join(missing)} on the gateway host. Install: {hint}")
    sd = state_dir()
    sd.mkdir(parents=True, exist_ok=True)
    os.chmod(sd, 0o700)
    with _flocked(sd / "start.lock"):
        if _launcher_pid() is not None and published_env().get("DISPLAY"):
            return status()
        from tools.bot_desktop import resources
        if (blocker := resources.memory_blocker()) is not None:
            raise RuntimeError(blocker)
        if _launcher_pid() is None:
            _reap_orphaned_server(sd)
        _ALLOC_LOCK.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            return _spawn_and_wait(sd, wait_seconds)
        except RuntimeError:
            # _pick_display reuses the recorded number first: left in place after a failed launch (e.g. Xvnc
            # 'server already running' on it), every retry would pick the same number and the profile wedges.
            (sd / "display").unlink(missing_ok=True)
            raise


def _spawn_and_wait(sd: Path, wait_seconds: float) -> DesktopStatus:
    """Caller holds ``start.lock``. Takes ``_ALLOC_LOCK`` itself, from picking the number to Xvnc's claim."""
    with contextlib.ExitStack() as alloc:
        alloc.enter_context(_flocked(_ALLOC_LOCK))
        num = _pick_display()
        (sd / "display").write_text(str(num), encoding="utf-8")
        env_file = sd / "env"
        env_file.unlink(missing_ok=True)

        child_env = {k: v for k, v in os.environ.items() if k not in {
            "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "SESSION_MANAGER"}}
        child_env.update({
            "HERMES_BD_PROFILE": _profile_name(),
            "HERMES_BD_DISPLAY_NUM": str(num),
            "HERMES_BD_SOCKET": str(sd / "rfb.sock"),
            "HERMES_BD_XAUTH": str(sd / "Xauthority"),
            "HERMES_BD_ENV_FILE": str(env_file),
            "HERMES_BD_CONFIG_HOME": str(sd / "xdg"),
            "HERMES_BD_GEOMETRY": geometry(),
        })
        from tools.bot_desktop.browser import dock_exec_line, dock_launch
        if (browser := dock_launch()) is not None:
            # The bare executable (the launcher checks it exists) and the ready-made, spec-quoted Exec= line.
            child_env["HERMES_BD_BROWSER_EXEC"] = browser[0]
            child_env["HERMES_BD_BROWSER_EXEC_LINE"] = dock_exec_line(*browser)
        # Truncated per start: the log is a diagnostic for THIS launch, and nothing rotates it otherwise.
        log = open(sd / "launcher.log", "wb")  # noqa: SIM115 — handed to the child, closed by it
        proc = subprocess.Popen(  # windows-footgun: ok — Linux-only runtime (is_supported_host)
            ["bash", str(_LAUNCHER)], env=child_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, close_fds=True)
        log.close()
        born = _create_time(proc.pid)
        (sd / "launcher.pid").write_text(f"{proc.pid} {born if born is not None else 0}", encoding="utf-8")

        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if _x_lock_pid(num) is not None:
                alloc.close()  # the number is Xvnc's now; other profiles may allocate (idempotent)
            if proc.poll() is not None:
                tail = (sd / "launcher.log").read_bytes()[-2000:].decode("utf-8", "replace")
                raise RuntimeError(f"Bot Desktop launcher exited with {proc.returncode}:\n{tail}")
            if env_file.exists() and (sd / "rfb.sock").exists():
                logger.info("Bot Desktop for profile %s up on :%s", _profile_name(), num)
                touch_activity()
                return status()
            time.sleep(0.1)
        # Giving up must take the launch down: left alone, the launcher publishes DISPLAY and rfb.sock a moment
        # later and a screen whose start() reported failure stays up as "running". The launcher is its own
        # session leader, so its group is exactly this launch (Xvnc, dbus, Xfce) and nothing else.
        _kill_group_then_wait(proc.pid, proc.pid)
        proc.wait()
        for name in ("launcher.pid", "env", "rfb.sock"):
            (sd / name).unlink(missing_ok=True)
        raise RuntimeError(f"Bot Desktop did not publish its display within {wait_seconds:.0f}s (see {sd / 'launcher.log'})")


def stop() -> bool:
    """Stop this profile's desktop; True when a running launcher (or the X server a dead one left behind)
    was signalled."""
    if not is_supported_host():
        return False
    sd = state_dir()
    sd.mkdir(parents=True, exist_ok=True)
    with _flocked(sd / "start.lock"):
        return _stop_locked(sd)


def _stop_locked(sd: Path) -> bool:
    pid = _launcher_pid()
    if pid is None:
        reaped = _reap_orphaned_server(sd)
        (sd / "env").unlink(missing_ok=True)
        return reaped
    # The launcher runs in its own session; killing the group takes Xvnc, dbus and Xfce with it.
    _kill_group_then_wait(pid, pid, grace=5.0)
    for name in ("launcher.pid", "env", "activity"):
        (sd / name).unlink(missing_ok=True)
    return True
