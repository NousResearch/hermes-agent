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
                    "xauth", "xdpyinfo", "setxkbmap")

PACKAGES = {
    "apt": ["tigervnc-standalone-server", "xfce4-panel", "xfwm4", "xfdesktop4", "xfce4-settings",
            "xfce4-terminal", "dbus-x11", "x11-xserver-utils", "x11-utils", "xauth", "fonts-dejavu-core"],
    "dnf": ["tigervnc-server-minimal", "xfce4-panel", "xfwm4", "xfdesktop", "xfce4-settings",
            "xfce4-terminal", "dbus-x11", "xorg-x11-server-utils", "xorg-x11-utils", "xorg-x11-xauth",
            "dejavu-sans-fonts"],
    "pacman": ["tigervnc", "xfce4-panel", "xfwm4", "xfdesktop", "xfce4-settings", "xfce4-terminal",
               "xorg-xsetroot", "xorg-xset", "xorg-xdpyinfo", "xorg-xauth", "xorg-setxkbmap", "ttf-dejavu"],
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
    pm = package_manager()
    if pm is None:
        return None
    pkgs = " ".join(PACKAGES[pm])
    return {
        "apt": f"sudo apt-get install -y --no-install-recommends {pkgs}",
        "dnf": f"sudo dnf install -y {pkgs}",
        "pacman": f"sudo pacman -S --needed --noconfirm {pkgs}",
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


def _launcher_pid() -> Optional[int]:
    raw = _read(state_dir() / "launcher.pid")
    if not raw or not raw.isdigit():
        return None
    pid = int(raw)
    return pid if _pid_alive(pid) else None


def _display_in_use(num: int) -> bool:
    return Path(f"/tmp/.X{num}-lock").exists() or Path(f"/tmp/.X11-unix/X{num}").exists()


def _allocate_display() -> int:
    recorded = _read(state_dir() / "display")
    if recorded and recorded.isdigit():
        return int(recorded)
    for num in range(_DISPLAY_MIN, _DISPLAY_MAX + 1):
        if not _display_in_use(num):
            return num
    raise RuntimeError("no free X display number in the Bot Desktop band")


def desktop_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """``base_env`` (default ``os.environ``) with this profile's DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS
    merged in when its desktop is running. Unchanged otherwise, so hosts with a real seat keep it.
    Pure: never starts anything (it is called from env builders, status probes and tests)."""
    env = dict(os.environ if base_env is None else base_env)
    published = published_env()
    if published:
        env.update(published)
        env.pop("WAYLAND_DISPLAY", None)  # X11 desktop; a leaked Wayland socket flips GTK/Chromium backends
    return env


def ensure_started_for_tool() -> None:
    """Tool-boundary hook (``computer_use`` dispatch): with ``bot_desktop.auto_start`` (default on) a Linux
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
    return bool(cfg.get("auto_start", True))


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
    ``wait_seconds`` pass; raises ``RuntimeError`` naming the blocker."""
    if not is_supported_host():
        raise RuntimeError("Bot Desktop runs on Linux gateway hosts only")
    missing = missing_binaries()
    if missing:
        hint = install_command() or "install TigerVNC (Xvnc) and the Xfce core components"
        raise RuntimeError(f"Bot Desktop needs {', '.join(missing)} on the gateway host. Install: {hint}")
    if _launcher_pid() is not None and published_env().get("DISPLAY"):
        return status()

    sd = state_dir()
    sd.mkdir(parents=True, exist_ok=True)
    os.chmod(sd, 0o700)
    num = _allocate_display()
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
    log = open(sd / "launcher.log", "ab")  # noqa: SIM115 — handed to the child, closed by it
    proc = subprocess.Popen(  # windows-footgun: ok — Linux-only runtime (is_supported_host)
        ["bash", str(_LAUNCHER)], env=child_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True, close_fds=True)
    log.close()
    (sd / "launcher.pid").write_text(str(proc.pid), encoding="utf-8")

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
    pid = _launcher_pid()
    sd = state_dir()
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
