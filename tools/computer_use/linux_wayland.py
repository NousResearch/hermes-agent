"""Linux desktop and native-Wayland capability diagnosis for Computer Use.

``cua-driver`` owns compositor protocol support and portal dispatch. Hermes owns
safe backend selection, an operator-readable desktop diagnosis, and optional
Arch package remediation. Generic Linux diagnosis must never recommend a
package-manager command for a different distribution.
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
from typing import Any, Dict, Mapping, Optional, Sequence


WAYLAND_ENABLE_ENV = "CUA_DRIVER_RS_ENABLE_WAYLAND"
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
class LinuxDistribution:
    id: Optional[str] = None
    id_like: tuple[str, ...] = ()
    name: Optional[str] = None
    pretty_name: Optional[str] = None

    @property
    def is_arch_like(self) -> bool:
        return self.id == "arch" or "arch" in self.id_like

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "id_like": list(self.id_like),
            "name": self.name,
            "pretty_name": self.pretty_name,
            "arch_like": self.is_arch_like,
        }


@dataclass(frozen=True)
class CuaDriverFeatures:
    """Strictly parsed feature claims from ``cua-driver manifest``.

    A missing, malformed, or non-boolean claim remains false. ``manifest_supported``
    means the manifest command returned a JSON object, not that any capability is
    enabled.
    """

    wayland_native: bool = False
    portal_input: bool = False
    portal_capture: bool = False
    manifest_supported: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


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
        env.get(key, "")
        for key in ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION")
    ).lower()
    for name, needles in {
        "gnome": ("gnome",),
        "kde": ("kde", "plasma"),
        "hyprland": ("hyprland",),
        "sway": ("sway",),
        "xfce": ("xfce",),
        "wlroots": ("wlroots", "labwc", "river", "niri"),
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
        if wl_ok:
            kind = "wayland-xwayland" if display else "wayland"
        else:
            # DISPLAY can remain usable through XWayland after a stale Wayland
            # environment leaks into a child process. Keep that route visible,
            # but never mistake it for a live native Wayland session.
            kind = "wayland-stale-xwayland" if display else "wayland-stale"
    elif declared == "wayland":
        kind = "wayland-stale"
        reasons.append("XDG_SESSION_TYPE=wayland but WAYLAND_DISPLAY is absent")
    elif display:
        kind = "x11"
    else:
        kind = "headless"
        if declared not in {"tty", "unspecified", ""}:
            reasons.append(f"XDG_SESSION_TYPE={declared!r} has no usable display socket")
    if not dbus:
        reasons.append("DBUS_SESSION_BUS_ADDRESS is absent; AT-SPI and portals cannot be reached")
    elif not dbus_ok:
        reasons.append("DBUS_SESSION_BUS_ADDRESS does not point to an accessible session bus socket")
    desktop = _desktop_name(e)
    return LinuxSession(
        kind=kind,
        wayland_display=wayland,
        display=display,
        compositor=desktop,
        desktop=desktop,
        runtime_dir=runtime,
        dbus_session_bus=dbus,
        wayland_socket_exists=wl_ok,
        dbus_socket_exists=dbus_ok,
        reasons=tuple(reasons),
    )


def _unquote_os_release(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def parse_os_release(content: str) -> LinuxDistribution:
    """Parse the small os-release subset needed for package-policy decisions."""
    fields: Dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"ID", "ID_LIKE", "NAME", "PRETTY_NAME"}:
            fields[key] = _unquote_os_release(value)
    distro_id = fields.get("ID", "").strip().lower() or None
    id_like = tuple(part.lower() for part in fields.get("ID_LIKE", "").split() if part.strip())
    return LinuxDistribution(
        id=distro_id,
        id_like=id_like,
        name=fields.get("NAME") or None,
        pretty_name=fields.get("PRETTY_NAME") or None,
    )


def detect_linux_distribution(os_release_path: Path | str = "/etc/os-release") -> LinuxDistribution:
    try:
        return parse_os_release(Path(os_release_path).read_text(encoding="utf-8"))
    except OSError:
        return LinuxDistribution()


def probe_driver_features(
    driver_cmd: Optional[str], env: Optional[Mapping[str, str]] = None,
) -> CuaDriverFeatures:
    """Run ``manifest`` once and fail closed on every unsupported value."""
    if not driver_cmd:
        return CuaDriverFeatures()
    try:
        result = subprocess.run(
            [driver_cmd, "manifest"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            stdin=subprocess.DEVNULL,
            env=dict(os.environ if env is None else env),
        )
        if result.returncode != 0:
            return CuaDriverFeatures()
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return CuaDriverFeatures()
    if not isinstance(payload, dict):
        return CuaDriverFeatures()
    features = payload.get("features")
    if not isinstance(features, dict):
        return CuaDriverFeatures(manifest_supported=True)
    return CuaDriverFeatures(
        wayland_native=features.get("wayland_native") if isinstance(features.get("wayland_native"), bool) else False,
        portal_input=features.get("portal_input") if isinstance(features.get("portal_input"), bool) else False,
        portal_capture=features.get("portal_capture") if isinstance(features.get("portal_capture"), bool) else False,
        manifest_supported=True,
    )


def driver_supports_native_wayland(driver_cmd: Optional[str], env: Optional[Mapping[str, str]] = None) -> bool:
    """Compatibility wrapper for callers that only need the native bit."""
    return probe_driver_features(driver_cmd, env).wayland_native


def configured_wayland_mode(config: Optional[Mapping[str, Any]] = None) -> str:
    cfg = dict(config or {})
    # Accept either the full Hermes config (with a ``computer_use`` top-level
    # key) or the ``computer_use`` sub-dict already unwrapped (the shape
    # returned by ``_computer_use_cfg()`` in cua_backend.py).
    if "linux" in cfg:
        cu = cfg
    else:
        cu = cfg.get("computer_use") or {}
    if not isinstance(cu, dict):
        cu = {}
    value = (cu.get("linux") or {}).get("wayland", {}).get("enabled", "auto")
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    return str(value).strip().lower() if str(value).strip().lower() in {"auto", "enabled", "disabled"} else "auto"


def native_wayland_enabled(
    driver_cmd: Optional[str],
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    features: Optional[CuaDriverFeatures] = None,
) -> bool:
    session = detect_linux_session(env)
    mode = configured_wayland_mode(config)
    # The opt-in cannot create a compositor socket. Even an explicit enabled
    # setting is only an override for driver feature policy, never for physical
    # session validity.
    if mode == "disabled" or session.kind not in {"wayland", "wayland-xwayland"}:
        return False
    if mode == "enabled":
        return True
    return (features or probe_driver_features(driver_cmd, env)).wayland_native


def native_wayland_child_env(
    driver_cmd: Optional[str], config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    out = dict(os.environ if env is None else env)
    if sys.platform == "linux" and native_wayland_enabled(driver_cmd, config, out):
        out[WAYLAND_ENABLE_ENV] = "1"
    else:
        out[WAYLAND_ENABLE_ENV] = "0"
    return out


def _run(args: Sequence[str], env: Mapping[str, str]) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            env=dict(env),
            stdin=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _service_active(service: str, env: Mapping[str, str]) -> bool:
    return _run(["systemctl", "--user", "is-active", service], env) == "active"


def _bus_name_owned(name: str, env: Mapping[str, str]) -> bool:
    return name in _run(["busctl", "--user", "--no-pager", "--list"], env)


def _restore_token_path(env: Mapping[str, str]) -> Path:
    config_dir = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config_dir) / "cua-driver" / "libei-persistent.token"


def diagnose_linux_desktop(
    driver_cmd: Optional[str],
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    os_release_path: Path | str = "/etc/os-release",
) -> Dict[str, Any]:
    """Distribution-neutral Linux desktop diagnosis for doctor and Desktop."""
    e = dict(os.environ if env is None else env)
    session = detect_linux_session(e)
    distribution = detect_linux_distribution(os_release_path)
    features = probe_driver_features(driver_cmd, e)
    mode = configured_wayland_mode(config)
    native_enabled = native_wayland_enabled(driver_cmd, config, e, features=features)
    portal_owner = _bus_name_owned("org.freedesktop.portal.Desktop", e)
    atspi_owner = _bus_name_owned("org.a11y.Bus", e)
    services = {
        "portal": _service_active("xdg-desktop-portal.service", e),
        "pipewire": _service_active("pipewire.service", e),
        "atspi": _service_active("at-spi-dbus-bus.service", e),
    }
    caps = LinuxComputerUseCapabilities(
        session_type=session.kind,
        compositor=session.compositor,
        desktop=session.desktop,
        accessibility_tree=atspi_owner,
        restore_token_present=_restore_token_path(e).is_file(),
    )
    if session.kind in {"x11", "wayland-stale-xwayland"}:
        caps.window_enumeration = caps.app_scoped_capture = caps.window_scoped_capture = caps.desktop_capture = True
        caps.background_element_actions = caps.background_pixel_actions = True
        caps.foreground_pointer_input = caps.foreground_keyboard_input = caps.target_activation = caps.focus_restore = True
        caps.capture_path, caps.input_path, caps.activation_path = "x11", "x11", "x11_ewmh"
        if session.kind == "wayland-stale-xwayland":
            caps.degraded_reasons.append(
                "WAYLAND_DISPLAY is stale; native Wayland is refused and the usable DISPLAY/X11 route is retained."
            )
    elif session.kind in {"wayland", "wayland-xwayland"}:
        caps.window_enumeration = True
        if not native_enabled:
            caps.hard_failures.append("Native Wayland is disabled or the installed driver did not advertise native Wayland support.")
        if not atspi_owner:
            caps.degraded_reasons.append("AT-SPI D-Bus is unavailable; semantic inspection and safe element actions are unavailable.")
        if not portal_owner:
            caps.degraded_reasons.append("xdg-desktop-portal D-Bus ownership is unavailable; portal capture/input cannot request consent.")
        if not services["pipewire"]:
            caps.degraded_reasons.append("PipeWire user service is inactive; PipeWire portal capture is unavailable.")
        wlroots = session.compositor in {"hyprland", "sway", "wlroots"}
        if features.portal_capture and portal_owner and services["pipewire"]:
            caps.desktop_capture = True
            caps.capture_path = "pipewire_portal_capture"
            caps.consent_expected = True
        elif wlroots and features.wayland_native:
            caps.capture_path = "wlroots_native_capture_candidate"
        else:
            caps.capture_path = "unavailable"
        if features.portal_input and portal_owner:
            # Feature compilation and portal ownership only make this a
            # candidate. cua-driver's live health report is authoritative for
            # consent, RemoteDesktop/EIS setup, target activation, and safe
            # delivery; Hermes must not promote it to ready on inference.
            caps.input_path = "portal_remote_desktop_input_candidate"
            caps.consent_expected = True
        elif wlroots and features.wayland_native:
            caps.input_path = "wlroots_virtual_pointer_candidate"
        else:
            caps.input_path = "unavailable"
            if session.compositor in {"gnome", "kde"} and not features.portal_input:
                caps.degraded_reasons.append(
                    "The driver was built without portal_input; GNOME/KDE portal input is not available in this artifact."
                )
        caps.activation_path = "driver_target_activation_candidate" if features.wayland_native else "unavailable"
    else:
        caps.hard_failures.append("No graphical Linux session is reachable from this process.")
    if session.kind == "wayland-stale-xwayland":
        caps.degraded_reasons.extend(session.reasons)
    else:
        caps.hard_failures.extend(session.reasons)
    return {
        "distribution": distribution.as_dict(),
        "session": asdict(session),
        "wayland_mode": mode,
        "native_wayland_enabled": native_enabled,
        "driver_features": features.as_dict(),
        "portal_service": services["portal"],
        "pipewire_service": services["pipewire"],
        "atspi_service": services["atspi"],
        "portal_dbus_available": portal_owner,
        "atspi_dbus_available": atspi_owner,
        "capabilities": caps.as_dict(),
    }


def _package_installed(package: str, pacman: str, env: Mapping[str, str]) -> bool:
    return bool(_run([pacman, "-Q", package], env))


def diagnose_arch_packages(report: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Optional Arch package layer. Never probes or recommends pacman elsewhere."""
    e = dict(os.environ if env is None else env)
    distro = report.get("distribution") or {}
    arch_like = bool(distro.get("arch_like"))
    pacman = shutil.which("pacman", path=e.get("PATH"))
    if not arch_like or not pacman:
        reason = "distribution is not Arch-like" if not arch_like else "pacman executable is unavailable"
        return {"applicable": False, "reason": reason, "packages": {}, "missing_packages": [], "selected_portal_package": None}
    compositor = ((report.get("session") or {}).get("compositor") or "")
    portal_package = ARCH_PORTAL_PACKAGES.get(compositor)
    packages = {package: _package_installed(package, pacman, e) for package in ARCH_SHARED_PACKAGES}
    if portal_package:
        packages[portal_package] = _package_installed(portal_package, pacman, e)
    return {
        "applicable": True,
        "reason": None,
        "packages": packages,
        "missing_packages": [name for name, installed in packages.items() if not installed],
        "selected_portal_package": portal_package,
    }


def diagnose_linux_computer_use(
    driver_cmd: Optional[str],
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    *,
    os_release_path: Path | str = "/etc/os-release",
) -> Dict[str, Any]:
    """Complete Linux diagnosis with an Arch-only package enrichment when valid."""
    e = dict(os.environ if env is None else env)
    report = diagnose_linux_desktop(driver_cmd, config, e, os_release_path=os_release_path)
    packages = diagnose_arch_packages(report, e)
    report["arch_packages"] = packages
    # Flat fields retain desktop compatibility while indicating whether package
    # diagnosis was applicable. On non-Arch they are empty—not synthetic misses.
    report["packages"] = packages["packages"]
    report["missing_packages"] = packages["missing_packages"]
    report["selected_portal_package"] = packages["selected_portal_package"]
    if packages["applicable"] and packages["selected_portal_package"] in packages["missing_packages"]:
        report["capabilities"]["degraded_reasons"].append(
            f"Expected {packages['selected_portal_package']} for the detected desktop is missing."
        )
    return report


def arch_install_hint(report: Mapping[str, Any]) -> Optional[str]:
    packages = report.get("arch_packages") or report
    if not packages.get("applicable", True):
        return None
    missing = packages.get("missing_packages") or []
    if not missing:
        return None
    return "Install only the missing packages for this desktop: sudo pacman -S " + " ".join(str(name) for name in missing)
