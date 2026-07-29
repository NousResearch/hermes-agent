"""Linux session and native-Wayland capability diagnosis for Computer Use.

This is intentionally Hermes-side policy, not a second input implementation.
``cua-driver`` owns compositor protocols, portals, AT-SPI and input dispatch;
Hermes owns the user-visible decision to opt in, the graphical-session checks,
and actionable Arch diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


WAYLAND_ENABLE_ENV = "CUA_DRIVER_RS_ENABLE_WAYLAND"
# Hermes deliberately requires a machine-readable feature claim before auto
# enabling upstream's experimental mode. A release number does *not* prove that
# optional portal/libei features were compiled into its Linux asset.
MIN_NATIVE_WAYLAND_DRIVER = (0, 12, 0)  # retained for diagnostic context only
ARCH_SHARED_PACKAGES = (
    "xdg-desktop-portal", "pipewire", "at-spi2-core", "libei", "libxkbcommon",
)
ARCH_PORTAL_PACKAGES = {
    "gnome": "xdg-desktop-portal-gnome",
    "kde": "xdg-desktop-portal-kde",
    "hyprland": "xdg-desktop-portal-hyprland",
    "sway": "xdg-desktop-portal-wlr",
    "wlroots": "xdg-desktop-portal-wlr",
}


@dataclass(frozen=True)
class LinuxSession:
    kind: str
    wayland_display: Optional[str]
    display: Optional[str]
    compositor: Optional[str]
    desktop: Optional[str]
    runtime_dir: Optional[str]
    dbus_session_bus: Optional[str]
    wayland_socket_exists: bool
    dbus_socket_exists: bool
    reasons: tuple[str, ...] = ()


@dataclass
class LinuxComputerUseCapabilities:
    session_type: str
    compositor: Optional[str]
    desktop: Optional[str]
    window_enumeration: bool = False
    app_scoped_capture: bool = False
    window_scoped_capture: bool = False
    desktop_capture: bool = False
    accessibility_tree: bool = False
    background_element_actions: bool = False
    background_pixel_actions: bool = False
    foreground_pointer_input: bool = False
    foreground_keyboard_input: bool = False
    target_activation: bool = False
    focus_restore: bool = False
    capture_path: Optional[str] = None
    input_path: Optional[str] = None
    activation_path: Optional[str] = None
    consent_expected: bool = False
    restore_token_present: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _socket_path(env: Mapping[str, str], name: str) -> Optional[Path]:
    runtime = env.get("XDG_RUNTIME_DIR")
    if not runtime or not name:
        return None
    path = Path(name)
    return path if path.is_absolute() else Path(runtime) / name


def _dbus_socket_path(address: Optional[str]) -> Optional[Path]:
    if not address:
        return None
    match = re.search(r"(?:^|[,:;])path=([^,;]+)", address)
    return Path(match.group(1)) if match else None


def _desktop_name(env: Mapping[str, str]) -> Optional[str]:
    values = " ".join(
        env.get(k, "") for k in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION")
    ).lower()
    for name, needles in {
        "gnome": ("gnome",), "kde": ("kde", "plasma"),
        "hyprland": ("hyprland",), "sway": ("sway",),
        "xfce": ("xfce",), "wlroots": ("wlroots", "labwc", "river", "niri"),
    }.items():
        if any(needle in values for needle in needles):
            return name
    return None


def detect_linux_session(env: Optional[Mapping[str, str]] = None) -> LinuxSession:
    """Classify Linux display state from one authoritative environment snapshot."""
    e = dict(os.environ if env is None else env)
    wayland = (e.get("WAYLAND_DISPLAY") or "").strip() or None
    display = (e.get("DISPLAY") or "").strip() or None
    declared = (e.get("XDG_SESSION_TYPE") or "").strip().lower()
    runtime = (e.get("XDG_RUNTIME_DIR") or "").strip() or None
    dbus = (e.get("DBUS_SESSION_BUS_ADDRESS") or "").strip() or None
    wl_path = _socket_path(e, wayland or "")
    dbus_path = _dbus_socket_path(dbus)
    wl_ok = bool(wl_path and wl_path.exists() and wl_path.is_socket())
    dbus_ok = bool(dbus_path and dbus_path.exists() and dbus_path.is_socket())
    reasons: list[str] = []

    if wayland:
        if not wl_ok:
            reasons.append("WAYLAND_DISPLAY is set but its runtime socket is missing or inaccessible")
        if display:
            kind = "wayland-xwayland"
        else:
            kind = "wayland" if wl_ok else "wayland-stale"
    elif declared == "wayland":
        kind = "wayland-stale"
        reasons.append("XDG_SESSION_TYPE=wayland but WAYLAND_DISPLAY is absent")
    elif display:
        kind = "x11"
    elif declared in {"tty", "unspecified", ""}:
        kind = "headless"
    else:
        kind = "headless"
        reasons.append(f"XDG_SESSION_TYPE={declared!r} has no usable display socket")
    if not dbus:
        reasons.append("DBUS_SESSION_BUS_ADDRESS is absent; AT-SPI and portals cannot be reached")
    elif not dbus_ok:
        reasons.append("DBUS_SESSION_BUS_ADDRESS does not point to an accessible session bus socket")
    return LinuxSession(
        kind=kind, wayland_display=wayland, display=display, compositor=_desktop_name(e),
        desktop=_desktop_name(e), runtime_dir=runtime, dbus_session_bus=dbus,
        wayland_socket_exists=wl_ok, dbus_socket_exists=dbus_ok, reasons=tuple(reasons),
    )


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value or "")
    return tuple(int(part) for part in match.groups()) if match else (0, 0, 0)


def driver_supports_native_wayland(driver_cmd: Optional[str], env: Optional[Mapping[str, str]] = None) -> bool:
    """Conservative, offline-safe native-Wayland driver gate.

    The manifest's explicit feature bit is the compatibility contract. Current
    release manifests do not expose compile features, so ``auto`` stays off
    rather than assuming every Linux release contains portal/libei support.
    Failure is deliberately a refusal: Hermes must not export the upstream
    opt-in flag to an unknown binary.
    """
    if not driver_cmd:
        return False
    child_env = dict(os.environ if env is None else env)
    try:
        manifest = subprocess.run(
            [driver_cmd, "manifest"], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=3, stdin=subprocess.DEVNULL, env=child_env,
        )
        if manifest.returncode == 0:
            data = json.loads(manifest.stdout)
            features = data.get("features") if isinstance(data, dict) else None
            if isinstance(features, dict) and isinstance(features.get("wayland_native"), bool):
                return bool(features["wayland_native"])
            if isinstance(features, list) and "wayland_native" in features:
                return True
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        pass
    return False


def configured_wayland_mode(config: Optional[Mapping[str, Any]] = None) -> str:
    cfg = dict(config or {})
    value = ((cfg.get("computer_use") or {}).get("linux") or {}).get("wayland", {}).get("enabled", "auto")
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    value = str(value).strip().lower()
    return value if value in {"auto", "enabled", "disabled"} else "auto"


def native_wayland_enabled(
    driver_cmd: Optional[str], config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    session = detect_linux_session(env)
    mode = configured_wayland_mode(config)
    if mode == "disabled" or session.kind not in {"wayland", "wayland-xwayland"}:
        return False
    if mode == "enabled":
        return True
    return driver_supports_native_wayland(driver_cmd, env)


def native_wayland_child_env(
    driver_cmd: Optional[str], config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    out = dict(os.environ if env is None else env)
    if sys.platform == "linux" and native_wayland_enabled(driver_cmd, config, out):
        out[WAYLAND_ENABLE_ENV] = "1"
    else:
        # Hermes owns this upstream experimental flag. In auto mode an inherited
        # shell export must not bypass the session + driver capability gate;
        # explicit `enabled` is the only operator override.
        if configured_wayland_mode(config) != "enabled":
            out[WAYLAND_ENABLE_ENV] = "0"
    return out


def _run(args: Sequence[str], env: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3, env=dict(env), stdin=subprocess.DEVNULL)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _package_installed(package: str, env: Mapping[str, str]) -> bool:
    pacman = shutil.which("pacman", path=env.get("PATH"))
    return bool(pacman and _run([pacman, "-Q", package], env))


def diagnose_arch_wayland(
    driver_cmd: Optional[str], config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Read-only Arch/Wayland diagnosis used by doctor and Desktop status."""
    e = dict(os.environ if env is None else env)
    session = detect_linux_session(e)
    mode = configured_wayland_mode(config)
    package_state = {package: _package_installed(package, e) for package in ARCH_SHARED_PACKAGES}
    portal_package = ARCH_PORTAL_PACKAGES.get(session.compositor or "")
    if portal_package:
        package_state[portal_package] = _package_installed(portal_package, e)
    portal_owner = "org.freedesktop.portal.Desktop" in _run(["busctl", "--user", "--no-pager", "--list"], e)
    a11y_owner = "org.a11y.Bus" in _run(["busctl", "--user", "--no-pager", "--list"], e)
    services = {
        name: _run(["systemctl", "--user", "is-active", name], e) == "active"
        for name in ("xdg-desktop-portal.service", "pipewire.service", "at-spi-dbus-bus.service")
    }
    restore_token = Path(e.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "cua-driver" / "libei-persistent.token"
    caps = LinuxComputerUseCapabilities(
        session_type=session.kind, compositor=session.compositor, desktop=session.desktop,
        accessibility_tree=a11y_owner,
        capture_path=("x11" if session.kind == "x11" else "native_wayland_candidate" if session.kind.startswith("wayland") else None),
        input_path=("x11" if session.kind == "x11" else "portal_or_wlroots_candidate" if session.kind.startswith("wayland") else None),
        consent_expected=session.kind.startswith("wayland") and portal_owner,
        restore_token_present=restore_token.is_file(),
    )
    if session.kind == "x11":
        caps.window_enumeration = caps.app_scoped_capture = caps.window_scoped_capture = caps.desktop_capture = True
        caps.background_element_actions = caps.background_pixel_actions = True
        caps.foreground_pointer_input = caps.foreground_keyboard_input = caps.target_activation = caps.focus_restore = True
        caps.activation_path = "x11_ewmh"
    elif session.kind in {"wayland", "wayland-xwayland"}:
        # Do not turn protocol availability into a delivery claim. The driver
        # health report, after native mode is launched, is the final authority.
        caps.window_enumeration = True
        if not native_wayland_enabled(driver_cmd, config, e):
            caps.hard_failures.append("Native Wayland is disabled or the installed driver is too old to pass Hermes' safe capability gate.")
        if not a11y_owner:
            caps.degraded_reasons.append("AT-SPI bus is unavailable; semantic inspection and safe element actions are unavailable.")
        if not portal_owner:
            caps.degraded_reasons.append("xdg-desktop-portal is unavailable; portal capture/input cannot request user consent.")
        if portal_package and not package_state.get(portal_package):
            caps.degraded_reasons.append(f"Expected {portal_package} for detected {session.compositor} desktop is missing.")
    else:
        caps.hard_failures.append("No graphical Linux session is reachable from this process.")
    caps.hard_failures.extend(session.reasons)
    missing = [name for name, installed in package_state.items() if not installed]
    return {
        "session": asdict(session), "wayland_mode": mode,
        "native_wayland_enabled": native_wayland_enabled(driver_cmd, config, e),
        "driver_native_wayland_capable": driver_supports_native_wayland(driver_cmd, e),
        "packages": package_state, "missing_packages": missing,
        "selected_portal_package": portal_package, "portal_service": services["xdg-desktop-portal.service"],
        "pipewire_service": services["pipewire.service"], "atspi_service": services["at-spi-dbus-bus.service"],
        "portal_dbus_available": portal_owner, "atspi_dbus_available": a11y_owner,
        "capabilities": caps.as_dict(),
    }


def arch_install_hint(report: Mapping[str, Any]) -> Optional[str]:
    missing = report.get("missing_packages") or []
    if not missing:
        return None
    return "Install only the missing packages for this desktop: sudo pacman -S " + " ".join(str(x) for x in missing)
