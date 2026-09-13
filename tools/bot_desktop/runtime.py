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
import shlex
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

# Package installation is privileged. Never resolve sudo or a package manager through the gateway's
# inherited PATH, which commonly contains user-writable directories such as ~/.local/bin.
_SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
_PACKAGE_MANAGER_BINARIES = {"apt": "apt-get", "dnf": "dnf", "pacman": "pacman"}

# ``hermes_subprocess_env(inherit_credentials=False)`` intentionally preserves the standard AWS
# credential chain for ordinary terminal children. A graphical desktop is a less trusted boundary:
# Xfce applications and browser extensions must not inherit credentials or paths to credential files.
_DESKTOP_AWS_CREDENTIAL_ENV = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
})

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
    "dnf": {"Xvnc": "tigervnc-server-minimal", "xfwm4": "xfwm4", "xfce4-panel": "xfce4-panel",
            "xfdesktop": "xfdesktop", "xfsettingsd": "xfce4-settings", "dbus-run-session": "dbus-x11",
            "xauth": "xorg-x11-xauth", "xdpyinfo": "xdpyinfo", "setxkbmap": "setxkbmap", "xprop": "xprop"},
    "pacman": {"Xvnc": "tigervnc", "xfwm4": "xfwm4", "xfce4-panel": "xfce4-panel", "xfdesktop": "xfdesktop",
               "xfsettingsd": "xfce4-settings", "dbus-run-session": "dbus",
               "xauth": "xorg-xauth", "xdpyinfo": "xorg-xdpyinfo", "setxkbmap": "xorg-setxkbmap", "xprop": "xorg-xprop"},
}

PACKAGES = {
    "apt": ["tigervnc-standalone-server", "xfce4-panel", "xfwm4", "xfdesktop4", "xfce4-settings",
            "xfce4-terminal", "dbus-x11", "x11-xserver-utils", "x11-utils", "x11-xkb-utils", "xauth",
            "fonts-dejavu-core"],
    "dnf": ["tigervnc-server-minimal", "xfce4-panel", "xfwm4", "xfdesktop", "xfce4-settings",
            "xfce4-terminal", "dbus-x11", "xsetroot", "xset", "xdpyinfo", "xprop", "xorg-x11-xauth", "setxkbmap",
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


def _system_executable(name: str) -> Optional[str]:
    """Absolute executable resolved only from root-managed system directories."""
    executable = shutil.which(name, path=_SYSTEM_PATH)
    return str(Path(executable).resolve()) if executable else None


def package_manager() -> Optional[str]:
    for pm, executable in _PACKAGE_MANAGER_BINARIES.items():
        if _system_executable(executable):
            return pm
    return None


def install_command() -> Optional[str]:
    pm = package_manager()
    sudo = _system_executable("sudo")
    package_manager_executable = _system_executable(_PACKAGE_MANAGER_BINARIES[pm]) if pm else None
    if pm is None or sudo is None or package_manager_executable is None:
        return None
    pkgs = " ".join(PACKAGES[pm])
    return {
        "apt": f"{shlex.quote(sudo)} {shlex.quote(package_manager_executable)} "
               f"install -y --no-install-recommends {pkgs}",
        "dnf": f"{shlex.quote(sudo)} {shlex.quote(package_manager_executable)} install -y {pkgs}",
        "pacman": f"{shlex.quote(sudo)} {shlex.quote(package_manager_executable)} "
                  f"-S --needed --noconfirm {pkgs}",
    }[pm]


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

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    import psutil
    return psutil.pid_exists(pid)


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
    return pid if actual is not None and abs(actual - born) < 0.01 else None


_X_LOCK_DIR = Path("/tmp")  # where X servers write .X<n>-lock (tests point it at a scratch dir)


def _display_in_use(num: int) -> bool:
    """A live X server owns ``:num``: its lock file names a running pid. A lock left by a crashed
    server (dead pid) does not count, so the number can be reclaimed."""
    lock = _X_LOCK_DIR / f".X{num}-lock"
    try:
        pid = int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return _pid_alive(pid)


_ALLOC_LOCK = Path("/tmp/.hermes-bot-desktop-alloc.lock")  # host-wide: profiles allocate from one band


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
    with _flocked(_ALLOC_LOCK):
        return _pick_display()


def desktop_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``base_env`` (default ``os.environ``) with this profile's DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS
    merged in when its desktop is running. Unchanged otherwise, so hosts with a real seat keep it.
    Pure: never starts anything (it is called from env builders, status probes and tests)."""
    env = dict(os.environ if base_env is None else base_env)
    published = published_env()
    if published:
        env.update(published)
        env.pop("WAYLAND_DISPLAY", None)  # X11 desktop; a leaked Wayland socket flips GTK/Chromium backends
        from tools.bot_desktop.browser import env_for_agent
        env_for_agent(env)  # same binary + user-data-dir as the dock's Browser icon
    return env


def ensure_started_for_tool() -> None:
    """Tool-boundary hook (``computer_use`` dispatch): with ``bot_desktop.auto_start`` (opt-in, default off) a Linux
    host that has NO display and the packages installed gets its screen started on first use, so a headless
    gateway works the first time instead of answering "no DISPLAY is set". Failure is not an error here;
    the tool's own "no display" diagnosis is the right message then."""
    if published_env() or not _should_auto_start(os.environ):
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
    missing: list[str] = missing_binaries() if is_supported_host() else list(REQUIRED_BINARIES)
    pid = _launcher_pid()
    env = published_env()
    return DesktopStatus(
        profile=profile or _profile_name(),
        supported=is_supported_host(),
        installed=not missing,
        missing=missing,
        running=pid is not None and bool(env.get("DISPLAY")),
        pid=pid,
        display=env.get("DISPLAY"),
        socket=str(rfb_socket_path()) if rfb_socket_path() else None,
        geometry=geometry(),
        install_command=install_command() if missing else None,
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

    Two locks, both held from the running-check to the launcher's publish: the per-profile ``start.lock``
    so two start() calls for one profile spawn one launcher (the loser sees it running), and the host-wide
    display-allocation lock so a second profile cannot pick the same number before this Xvnc has written
    ``/tmp/.X<n>-lock`` (it would then fail and its launcher's stale-lock cleanup could remove our socket)."""
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
        with _flocked(_ALLOC_LOCK):
            return _spawn_and_wait(sd, _pick_display(), wait_seconds)


def _spawn_and_wait(sd: Path, num: int, wait_seconds: float) -> DesktopStatus:
    (sd / "display").write_text(str(num), encoding="utf-8")
    env_file = sd / "env"
    env_file.unlink(missing_ok=True)

    # Xfce, its terminal, and browser extensions are untrusted subprocess surfaces. Keep benign host
    # settings, but do not expose provider, plugin, gateway, or internal Hermes credentials to them.
    from tools.environments.local import hermes_subprocess_env
    child_env = hermes_subprocess_env(inherit_credentials=False)
    for key in _DESKTOP_AWS_CREDENTIAL_ENV | {
        "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "SESSION_MANAGER",
    }:
        child_env.pop(key, None)
    child_env.update({
        "HERMES_BD_PROFILE": _profile_name(),
        "HERMES_BD_DISPLAY_NUM": str(num),
        "HERMES_BD_SOCKET": str(sd / "rfb.sock"),
        "HERMES_BD_XAUTH": str(sd / "Xauthority"),
        "HERMES_BD_ENV_FILE": str(env_file),
        "HERMES_BD_CONFIG_HOME": str(sd / "xdg"),
        "HERMES_BD_GEOMETRY": geometry(),
    })
    from tools.bot_desktop.browser import dock_command, dock_launch
    if (browser := dock_launch()) is not None:
        child_env["HERMES_BD_BROWSER_EXEC"] = dock_command(*browser)
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
        if proc.poll() is not None:
            tail = (sd / "launcher.log").read_bytes()[-2000:].decode("utf-8", "replace")
            raise RuntimeError(f"Bot Desktop launcher exited with {proc.returncode}:\n{tail}")
        if env_file.exists() and (sd / "rfb.sock").exists():
            logger.info("Bot Desktop for profile %s up on :%s", _profile_name(), num)
            return status()
        time.sleep(0.1)
    raise RuntimeError(f"Bot Desktop did not publish its display within {wait_seconds:.0f}s (see {sd / 'launcher.log'})")


def stop() -> bool:
    """Stop this profile's desktop; True when a running launcher was signalled."""
    if not is_supported_host():
        return False
    sd = state_dir()
    sd.mkdir(parents=True, exist_ok=True)
    with _flocked(sd / "start.lock"):
        return _stop_locked(sd)


def _stop_locked(sd: Path) -> bool:
    pid = _launcher_pid()
    if pid is None:
        (sd / "env").unlink(missing_ok=True)
        return False
    # The launcher runs in its own session; killing the group takes Xvnc, dbus and Xfce with it.
    try:
        os.killpg(pid, signal.SIGTERM)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start)
    except ProcessLookupError:
        pass
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)  # windows-footgun: ok — Linux-only runtime (is_supported_host gates start)
        except ProcessLookupError:
            pass
    (sd / "launcher.pid").unlink(missing_ok=True)
    (sd / "env").unlink(missing_ok=True)
    return True
