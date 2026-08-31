"""Cua-driver backend (macOS, Windows, Linux): MCP over stdio to `cua-driver`. The async `mcp` SDK runs on a
background loop (``cua_backend_session``); the same tool surface works on all three platforms, and per-host gaps
(no DISPLAY, missing AT-SPI, TCC) surface via `hermes computer-use doctor` instead of failing silently. Install
with `hermes computer-use install`. The macOS path uses private SkyLight SPIs that can break on OS updates.
Siblings: ``cua_backend_driver`` (binary/contract/update), ``cua_backend_capture`` + ``cua_backend_input``
(mixins), ``cua_backend_parse``, ``cua_backend_session`` (bridge + session + CLI fallback), ``cua_backend_daemon``
(private daemon + macOS app identity). Siblings look this module's config/policy helpers up lazily."""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)

_MISSING = object()


def _mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x field rename.

    mcp 2.0 renamed model fields to snake_case, keeping camelCase only as a
    serialization alias that pydantic does not expose to attribute access. A
    plain ``getattr(result, "isError", False)`` therefore reads False for
    *every* result on 2.x — a denied or failed cua-driver call would be
    treated as a success. Reading both spellings keeps this correct on either
    SDK generation.

    Deliberately duplicated from ``tools.mcp_tool.mcp_field`` rather than
    imported: computer_use talks to cua-driver over its own stdio client and
    does not otherwise load the (much larger) config-driven MCP client module.
    """
    value = getattr(obj, snake, _MISSING)
    if value is not _MISSING:
        return value
    value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value


def _action_result_from(
    name: str,
    ok: bool,
    message: str,
    meta: Dict[str, Any],
    structured: Dict[str, Any],
    *,
    requested_delivery: Optional[str] = None,
) -> ActionResult:
    """Build an ActionResult, lifting cua-driver's structured verdict.

    All structured fields are additive: a driver that omits
    ``structuredContent`` (or any individual field) leaves the corresponding
    ActionResult attribute ``None``, so callers and tests see unchanged
    behavior on old drivers. See the action response shape in
    cua-driver's mcp-tool-notes and NousResearch/hermes-agent#67052.
    """
    sc = structured if isinstance(structured, dict) else {}

    def _pick(key: str) -> Any:
        # structuredContent is canonical; fall back to a flattened meta copy.
        if key in sc:
            return sc.get(key)
        return meta.get(key)

    verified = _pick("verified")
    if not isinstance(verified, bool):
        verified = None
    effect = _pick("effect")
    if not isinstance(effect, str):
        effect = None
    escalation = _pick("escalation")
    if not isinstance(escalation, dict):
        escalation = None
    path = _pick("path")
    if not isinstance(path, str):
        path = None
    degraded = _pick("degraded")
    if not isinstance(degraded, bool):
        degraded = None
    # Refusal/limitation code — drivers spell it "code" or "reason_code".
    code = _pick("code") or _pick("reason_code")
    if not isinstance(code, str):
        code = None
    # Echo the delivery mode the caller actually requested (the driver's
    # `path` records the rung that ran; this records what we asked for).
    delivery_mode = requested_delivery if isinstance(requested_delivery, str) else None

    return ActionResult(
        ok=ok,
        action=name,
        message=message,
        meta=meta,
        verified=verified,
        effect=effect,
        escalation=escalation,
        path=path,
        degraded=degraded,
        delivery_mode=delivery_mode,
        code=code,
    )



# ---------------------------------------------------------------------------
# Update checking
# ---------------------------------------------------------------------------
#
# cua-driver ships a native `check-update` verb (and a `check_for_update` MCP
# tool) that compares the installed binary against the latest GitHub release —
# the source of truth — and caches the result (~20h). We prefer that over a
# hardcoded version floor, which would rot and can't know what "latest" is.
#
# There is intentionally no version *pin* knob: the upstream installer always
# fetches the latest release, so a `HERMES_CUA_DRIVER_VERSION` env var would
# only have *looked* like it pinned. For a reproducible version, point
# `HERMES_CUA_DRIVER_CMD` at a specific binary instead.

_CUA_DRIVER_CMD_ENV = "HERMES_CUA_DRIVER_CMD"
_CUA_DRIVER_DEFAULT_CMD = "cua-driver"
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP transport (fallback when the
                            # driver doesn't expose `manifest` — see
                            # `_resolve_mcp_invocation` below)

# Whole-screen / desktop capture. cua-driver is a window-oriented driver —
# its `get_window_state` / `screenshot` tools capture a single window (by
# pid + window_id), and there is no MCP tool that captures the entire virtual
# desktop or an arbitrary monitor as one image. But the OS shell surfaces
# themselves (the desktop backdrop and the taskbar/menu-bar) are real windows
# that show up in `list_windows`, so "click the taskbar" is reachable by
# targeting those windows.
#
# Two distinct whole-screen intents, two lanes:
#   * app="screen" (or "fullscreen"/"full screen"/"all") → a real composited
#     capture of everything currently displayed, via cua-driver's
#     `get_desktop_state`. Pixels only — no element tree.
#   * app="desktop" → the OS shell/desktop window (wallpaper + icons) resolved
#     through list_windows, WITH interactable elements (desktop icons).
_FULL_SCREEN_SENTINELS = {"screen", "fullscreen", "full screen", "all"}
_DESKTOP_SHELL_SENTINELS = {"desktop"}
# Backwards-compatible union — membership means "some whole-screen intent".
_SCREEN_CAPTURE_SENTINELS = _FULL_SCREEN_SENTINELS | _DESKTOP_SHELL_SENTINELS

# Known shell/desktop window identifiers across platforms. Matched
# case-insensitively as a substring against both the window's app_name and
# its title (cua-driver surfaces the Win32 class name / app name here).
#   Windows: Progman / WorkerW back the desktop; Shell_TrayWnd is the taskbar.
#   macOS:   Finder owns the desktop; the menu bar / Dock are the shell.
_DESKTOP_WINDOW_NAMES = (
    "progman", "workerw", "program manager",  # Windows desktop
    "shell_traywnd", "taskbar",               # Windows taskbar
    "finder", "desktop", "dock",              # macOS desktop / shell
)

# Linux/X11 can surface GNOME Shell / desktop backdrop windows before real app
# windows and cua-driver 0.6.x currently does not assign a useful z-order for
# them. These windows are targetable X11 windows but do not produce screenshots
# through get_window_state, so default app capture must skip them.
_NON_APP_WINDOW_TITLE_PREFIXES = (
    "@!",          # GNOME Shell background/monitor helper windows
    "Desktop",
    "gnome-shell",
    "GNOME Shell",
)


# Env var cua-driver reads to gate its anonymous usage telemetry (PostHog).
# Setting it to "0" disables telemetry; absence => the binary's own default
# (telemetry ON upstream).
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"
_CUA_NATIVE_WAYLAND_ENV_VAR = "CUA_DRIVER_RS_ENABLE_WAYLAND"


def _computer_use_cfg() -> Dict[str, Any]:
    """The ``computer_use`` config block, or ``{}`` when config is unreadable."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        return (load_config() or {}).get("computer_use") or {}
    return {}

def _cua_no_overlay() -> bool:
    """Pass ``--no-overlay``? ``computer_use.no_overlay`` overrides; else off on macOS (cursor-overlay redraw
    loop can peg a core after a session), headless Linux / WSL2 / containers, and Linux X11 (the overlay is a
    fullscreen always-on-top all-workspaces window with no compositor-owned lifecycle, so an unclean session
    end can leave it wedged over every app); on for Windows and Linux Wayland (compositor owns the surface).

    Reads ``computer_use.no_overlay``. Default ``None`` (auto-detect):
    disable the overlay where idle CPU burn or an X11 desktop wedge is a
    known failure mode — macOS (cursor-overlay vImage redraw loop,
    #28152/#47032), headless Linux / WSL2 / containers, and Linux X11
    (fullscreen always-on-top overlay window that can get stuck over every
    workspace after an unclean session end) — and keep it on Windows and
    Linux Wayland. Explicit ``True`` / ``False`` overrides auto-detection.
    """
    val = _computer_use_cfg().get("no_overlay")
    if val is not None:
        return bool(val)
    # Auto-detect: macOS overlay can peg a core indefinitely after a
    # computer_use session (#47032). Prefer off until the driver teardown
    # is solid; set computer_use.no_overlay: false to keep the cursor.
    if sys.platform == "darwin":
        return True
    if sys.platform != "linux":
        return False
    if not os.environ.get("DISPLAY"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            if "microsoft" in f.read().lower():
                return True
    except Exception:
        pass
    # Linux/X11: the cursor overlay is a fullscreen, always-on-top,
    # all-workspaces X11 window (save-unders path). An unclean session end
    # (agent interrupted mid-capture, stale target window) can leave it stuck
    # above every app on every workspace, wedging desktop input until the app
    # restarts — the same failure class as the HUD window on Mutter/X11
    # (#83473). There is no compositor-owned surface to tear down with the
    # client connection, so default the overlay off on X11 too; set
    # computer_use.no_overlay: false to keep the cursor. Wayland keeps it: the
    # compositor owns the overlay surface lifecycle there.
    if os.environ.get("XDG_SESSION_TYPE") != "wayland" and not os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


def _cua_telemetry_disabled() -> bool:
    """True unless ``computer_use.cua_telemetry`` opts in (unreadable config fails SAFE toward disabling)."""
    return not bool(_computer_use_cfg().get("cua_telemetry", False))

def _cua_configured_permission_mode() -> str:
    """``computer_use.permission_mode``: ``standard`` (default) or ``bounded``; unknown values fall closed to
    ``standard``. ``unrestricted`` is deliberately NOT a config value — it stays tied to the per-session YOLO
    toggle so a stale config line can never silently bypass approvals."""
    raw = str(_computer_use_cfg().get("permission_mode", "standard") or "").strip().lower()
    return raw if raw in {"standard", "bounded"} else "standard"


def _cua_capability_manifest() -> Optional[str]:
    """Path of the reviewed capability manifest for bounded mode, or None.

    Reads ``computer_use.capability_manifest``.  Existence is validated by
    ``_EmbeddedCuaDaemon`` so a missing file fails loudly at session start
    instead of silently degrading the authorization story.
    """
    raw = _computer_use_cfg().get("capability_manifest")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _manifest_is_mode_independent(path: str) -> bool:
    """True when this manifest may accompany any permission mode: v1/v2 declare ``mode: bounded`` and abort
    startup under an unrestricted runtime; v3 has no mode and is the ceiling the driver accepts alongside any
    mode. Unreadable / unparseable -> False (forwarding one would turn a working session into a hard startup
    failure; bounded forwards unconditionally anyway)."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception:
        logger.debug("could not read capability manifest %s", path, exc_info=True)
        return False
    version = parsed.get("version") if isinstance(parsed, dict) else None
    return isinstance(version, int) and not isinstance(version, bool) and version >= 3


def _computer_use_max_image_dimension() -> Optional[int]:
    """``computer_use.max_image_dimension`` longest-edge cap (default 1456 = aux-vision downscale); ``0``/negative -> None."""
    try:
        dim = int(_computer_use_cfg().get("max_image_dimension", 1456))
    except (TypeError, ValueError):
        dim = 1456
    return dim if dim > 0 else None

def cua_driver_child_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Env for spawning cua-driver: ``base_env`` (default ``os.environ``) plus ``CUA_DRIVER_RS_TELEMETRY_ENABLED=0``
    unless the user opted in, plus the native-Wayland bridge (``computer_use.native_wayland`` config opt-in, only when
    the child has a Wayland display). Used by every spawn site (MCP, status, doctor, install) so CLI and gateway
    runtimes share one policy."""
    env = dict(os.environ if base_env is None else base_env)
    if _cua_telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    if sys.platform == "linux" and env.get("WAYLAND_DISPLAY") and bool(_computer_use_cfg().get("native_wayland", False)):
        env[_CUA_NATIVE_WAYLAND_ENV_VAR] = "1"
    return env

def sanitized_cua_driver_env() -> Dict[str, str]:
    """``cua_driver_child_env()`` with Hermes provider secrets stripped — cua-driver is a third-party binary and must
    never inherit API keys. Falls back to the unsanitized telemetry env if the sanitizer can't import."""
    env = cua_driver_child_env()
    with contextlib.suppress(Exception):
        # cua-driver is a third-party binary — never hand it provider API keys via inherited env (same
        # policy as the manifest probe and MCP spawn; #53503/#55709/#58889 lineage).
        from tools.environments.local import _sanitize_subprocess_env
        return _sanitize_subprocess_env(env)
    return env

def _run_quiet(argv: List[str], *, timeout: float, swallow: Any = (), **kw: Any) -> Any:
    """``subprocess.run`` for short probe verbs: text mode, stdin=DEVNULL unless overridden (older drivers fall into a
    stdin-reading mode on unknown verbs; EOF makes them exit fast instead of blocking until the timeout), output
    captured unless the caller redirects it. Exceptions in ``swallow`` return None; others raise."""
    kw.setdefault("stdin", subprocess.DEVNULL)
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    "stdout" in kw or kw.setdefault("capture_output", True)
    try:
        return subprocess.run(argv, text=True, timeout=timeout, stdin=kw.pop("stdin"), encoding=kw.pop("encoding"),
                              errors=kw.pop("errors"), **kw)
    except swallow:
        return None

def _run_driver(driver_cmd: str, *args: str, timeout: float, swallow: Any = ()) -> Any:
    """Run a short cua-driver verb with the sanitized env and hidden window."""
    return _run_quiet([driver_cmd, *args], timeout=timeout, swallow=swallow, encoding="utf-8",
                      errors="replace", creationflags=windows_hide_flags(), env=sanitized_cua_driver_env())

def _linux_session_locked() -> Optional[bool]:
    """Is the graphical session locked? (Linux; best-effort.) A locked KDE/GNOME session freezes renderers and
    half-disables the AX tree, so discovery legitimately returns nothing — which otherwise reads as a driver bug.
    True/False when loginctl answers, None when unavailable (non-Linux, no systemd-logind, probe failure)."""
    # Auto-detect: macOS overlay can peg a core indefinitely after a computer_use session (#47032). Prefer
    # off until the driver teardown is solid; set computer_use.no_overlay: false to keep the cursor.
    if sys.platform != "linux":
        return None
    try:
        proc = _run_quiet(["loginctl", "list-sessions", "--no-legend"], timeout=2.0)
        seats = [line.split()[0] for line in proc.stdout.splitlines() if len(line.split()) >= 2 and "seat" in line]
        if proc.returncode != 0 or not seats:
            return None
        return not any("LockedHint=no" in _run_quiet(["loginctl", "show-session", s, "-p", "LockedHint"], timeout=2.0).stdout
                       for s in seats)
    except Exception:
        return None

def _empty_discovery_reason() -> str:
    """One-line diagnosis for 'window discovery found nothing'."""
    if _linux_session_locked() is True:
        return ("the desktop session is LOCKED (loginctl LockedHint=yes) — unlock the screen; "
                "a locked compositor hides windows and freezes app renderers")
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return "no DISPLAY is set — X11/XWayland is not reachable from this process"
    if sys.platform == "darwin":
        # Headless Mac / asleep panel: ScreenCaptureKit has 0 shareable
        # displays while TCC grants look fine (#67165, #52925 lineage).
        return (
            "window discovery returned no windows; on macOS this usually "
            "means no shareable display (headless Mac or panel asleep) — "
            "wake the display or attach a monitor/HDMI dummy, then run "
            "`hermes computer-use doctor`"
        )
    return (
        "window discovery returned no windows; run `hermes computer-use "
        "doctor` (display reachability, AX capability)"
    )


def _z_index_uninformative(windows: List[Dict[str, Any]]) -> bool:
    """True when every window shares the same z_index (common on Linux/X11)."""
    if not windows:
        return True
    return len({w.get("z_index", 0) for w in windows}) <= 1


def _parse_xprop_net_active_window(stdout: str) -> Optional[int]:
    """Parse ``xprop -root _NET_ACTIVE_WINDOW`` stdout into a window id.

    Accepts the common ``window id # 0x...`` form and falls back to the first
    hex token. Returns None for empty/unparseable output.
    """
    text = stdout or ""
    match = re.search(r"window id # (0x[0-9a-fA-F]+)", text)
    if not match:
        match = re.search(r"(0x[0-9a-fA-F]+)", text)
    if not match:
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


def _linux_x11_active_window_id() -> Optional[int]:
    """Best-effort read of ``_NET_ACTIVE_WINDOW`` via xprop. Never raises."""
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return _parse_xprop_net_active_window(proc.stdout or "")


def _is_real_app_window(w: Dict[str, Any]) -> bool:
    """Return False for desktop/shell helper windows that capture as empty."""
    title = w.get("title", "")
    return not any(
        title.startswith(p) or title.lower().startswith(p.lower())
        for p in _NON_APP_WINDOW_TITLE_PREFIXES
    )


def _select_capture_target(
    windows: List[Dict[str, Any]],
    *,
    app_requested: bool,
    exact_target: bool = False,
) -> Dict[str, Any]:
    """Select the best window for capture from normalized list_windows output.

    Callers pass windows already sorted by ``z_index`` descending (higher =
    frontmost). When ordering is informative, keep that frontmost contract.
    For unqualified default captures (no app filter and no exact
    pid/window_id) on Linux, desktop/shell helper windows (GNOME ``ding``
    "Desktop Icons", ``@!x,y;BDHF`` backdrop helpers) are skipped first —
    they are targetable X11 windows but capture as empty. Then, when every
    remaining candidate shares the same ``z_index`` (tied or unknown, the
    common Linux/X11 case), prefer ``_NET_ACTIVE_WINDOW`` over list order
    (#58026). Exact-target captures must not pay for an ``xprop`` probe.
    """
    candidates = [w for w in windows if not w["off_screen"]]
    pool = candidates
    if not exact_target and not app_requested and sys.platform == "linux":
        real_apps = [w for w in candidates if _is_real_app_window(w)]
        if real_apps:
            pool = real_apps
        if pool and _z_index_uninformative(pool):
            active_id = _linux_x11_active_window_id()
            if active_id is not None:
                for w in pool:
                    if w.get("window_id") == active_id:
                        return w
    if pool:
        return pool[0]
    return windows[0]


def _wsl_windows_path_to_posix(path: str) -> str:
    """Translate a Windows absolute manifest command when Hermes runs in WSL.

    Windows cua-driver manifests can report ``C:\\Users\\...\\cua-driver.exe``
    even though the Hermes process uses POSIX subprocess spawning inside WSL.
    The same file is reachable through DrvFS as ``/mnt/c/Users/...``.
    Non-Windows paths and non-WSL hosts are returned unchanged.
    """
    if not re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    try:
        from hermes_constants import is_wsl

        if not is_wsl():
            return path
    except Exception:
        return path
    win = PureWindowsPath(path)
    drive = (win.drive or "").rstrip(":").lower()
    if not drive:
        return path
    return os.path.join("/mnt", drive, *(str(part) for part in win.parts[1:]))


def _resolve_cua_driver_app_path(driver_cmd: str) -> Optional[str]:
    """Return the CuaDriver.app bundle that CARRIES *driver_cmd*, if any.

    Deliberately derived from the resolved driver binary path only — no
    /Applications or ~/Applications fallback. A fallback candidate can be a
    DIFFERENT install than the driver the manifest resolved (stale copy,
    side-by-side version), and launching it would run code the resolution
    chain never validated. If the resolved driver does not live inside an
    app bundle, the caller fails closed with install guidance.
    """
    resolved_driver_cmd = os.path.realpath(driver_cmd)
    marker = ".app/Contents/MacOS/"
    marker_index = resolved_driver_cmd.find(marker)
    if marker_index < 0:
        return None
    candidate = resolved_driver_cmd[: marker_index + len(".app")]
    executable = os.path.join(candidate, "Contents", "MacOS", "cua-driver")
    if os.path.isfile(executable) and os.access(executable, os.X_OK):
        return candidate
    return None


# The only bundle identity the private daemon may launch through, and the
# teams that sign official cua-driver releases. Exact matches only: a
# suffixed identifier ("com.trycua.driver.evil") or a different non-empty
# team is an impostor bundle, not a variant.
_CUA_DRIVER_BUNDLE_ID = "com.trycua.driver"
_CUA_DRIVER_TEAM_IDS = ("4YEC26S9KF", "YCK386LBJ7")


def _validate_cua_driver_app_signature(app_path: str) -> None:
    """Fail closed unless *app_path* is the genuinely-signed CuaDriver.app.

    Launching via ``/usr/bin/open`` hands LaunchServices whatever bundle sits
    at the path, so the TCC-identity fix must not become a launcher for
    arbitrary apps: require ``codesign -dv`` to report EXACTLY
    ``Identifier=com.trycua.driver`` and an expected TeamIdentifier.
    ``TeamIdentifier=not set`` (unsigned/ad-hoc dev builds) is allowed only
    when ``computer_use.allow_unsigned_driver: true`` is set in config.yaml —
    the escape hatch for local driver development, never the default. Raises
    RuntimeError on any mismatch or when codesign is unavailable/fails.
    """
    codesign = shutil.which("codesign")
    if not codesign:
        raise RuntimeError(
            "codesign is required to verify CuaDriver.app before launching it."
        )
    try:
        proc = subprocess.run(
            [codesign, "-dv", app_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not verify CuaDriver.app signature: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"CuaDriver.app at {app_path} is not code-signed; refusing to launch it "
            f"({(proc.stderr or '').strip()})"
        )
    # codesign -dv reports on stderr.
    fields = {}
    for line in (proc.stderr or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    identifier = fields.get("Identifier", "")
    team = fields.get("TeamIdentifier", "")
    if identifier != _CUA_DRIVER_BUNDLE_ID:
        raise RuntimeError(
            f"CuaDriver.app at {app_path} has identifier {identifier!r}, "
            f"expected {_CUA_DRIVER_BUNDLE_ID!r}; refusing to launch it."
        )
    if team in _CUA_DRIVER_TEAM_IDS:
        return
    if team in ("", "not set") and _computer_use_cfg().get("allow_unsigned_driver") is True:
        return
    raise RuntimeError(
        f"CuaDriver.app at {app_path} is signed by team {team!r}, expected one of "
        f"{_CUA_DRIVER_TEAM_IDS!r}; refusing to launch it. (Set "
        "computer_use.allow_unsigned_driver: true in config.yaml only for "
        "local unsigned driver builds.)"
    )


def _embedded_daemon_spawn_command(
    driver_cmd: str,
    serve_args: List[str],
    *,
    platform: str,
    app_path: Optional[str] = None,
) -> List[str]:
    """Build the private-daemon launch while preserving macOS TCC identity."""
    if platform != "darwin":
        return [driver_cmd, *serve_args]
    resolved_app = app_path or _resolve_cua_driver_app_path(driver_cmd)
    if not resolved_app:
        raise RuntimeError(
            "CuaDriver.app is required for private computer-use sessions on macOS. "
            "Run `hermes computer-use install` to restore it."
        )
    _validate_cua_driver_app_signature(resolved_app)
    return [
        "/usr/bin/open",
        "-n",
        "-g",
        "-a",
        resolved_app,
        "--args",
        *serve_args,
    ]


class _EmbeddedCuaDaemon:
    """Private daemon for a non-standard permission mode.

    Cua Driver permission mode is immutable after daemon startup.  Reusing the
    machine-wide daemon would therefore let one Hermes session's YOLO choice
    affect another session.  A private embedded daemon gives the requesting
    session its own socket, runtime, and launch-time authorization. On macOS
    the runtime is launched through CuaDriver.app so TCC remains attached to
    ``com.trycua.driver`` instead of the embedding host's ad-hoc signature:

    * ``unrestricted`` — explicit Hermes YOLO; launch-time risk
      acknowledgement via ``--dangerously-bypass-approvals``.
    * ``bounded`` — a user-reviewed capability manifest
      (``computer_use.capability_manifest`` in config.yaml) approved at
      launch via ``--approve-capability-manifest``.  The manifest, not a
      runtime prompt, is the authorization boundary; calls outside it fail
      closed inside cua-driver.

    The manifest is a ceiling, not a mode.  cua-driver accepts it alongside
    any permission mode and it "can narrow a profile but never widen it", so
    a configured manifest is forwarded here even when YOLO selected
    ``unrestricted`` — that pairing is what bounds an approval-bypassed run
    to declared scope.  It stays mandatory for ``bounded`` and optional
    everywhere else.
    """

    _START_TIMEOUT_SECONDS = 15.0

    def __init__(
        self,
        driver_cmd: str,
        permission_mode: str,
        capability_manifest: Optional[str] = None,
    ) -> None:
        if permission_mode not in {"unrestricted", "bounded"}:
            raise ValueError(
                "embedded permission override supports unrestricted or bounded only"
            )
        self.capability_manifest: Optional[str] = None
        manifest = str(capability_manifest or "").strip()
        if not manifest and permission_mode == "bounded":
            raise ValueError(
                "bounded permission mode requires computer_use.capability_manifest"
            )
        if manifest:
            manifest = os.path.abspath(os.path.expanduser(manifest))
            if not os.path.isfile(manifest):
                raise ValueError(
                    f"capability manifest not found: {manifest}"
                )
            self.capability_manifest = manifest
        # bounded always forwards — the driver validates it there. Any other
        # mode only accepts a v3 (mode-independent) manifest; forwarding a
        # legacy one would abort startup instead of bounding the run.
        self.manifest_applies = bool(self.capability_manifest) and (
            permission_mode == "bounded"
            or _manifest_is_mode_independent(str(self.capability_manifest))
        )
        if self.capability_manifest and not self.manifest_applies:
            logger.warning(
                "computer_use.capability_manifest is a legacy (v1/v2) manifest, "
                "which cua-driver only accepts in bounded mode — it will NOT "
                "bound this %s session. Migrate the manifest to version 3 to "
                "keep a ceiling on approval-bypassed runs.",
                permission_mode,
            )
        self.permission_mode = permission_mode
        self._driver_cmd = driver_cmd
        self._command = driver_cmd
        self._mcp_args: List[str] = list(_CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = False
        self._running = False
        self._launch_via_app = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_thread: Optional[threading.Thread] = None
        token = uuid.uuid4().hex[:12]
        if sys.platform == "win32":
            self.socket_path = rf"\\.\pipe\hermes-cua-{token}"
        else:
            self.socket_path = os.path.join(
                tempfile.gettempdir(), f"hc-{token}.sock"
            )

    def child_env(self) -> Dict[str, str]:
        env = cua_driver_child_env()
        env["CUA_DRIVER_PERMISSION_MODE"] = self.permission_mode
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def _drain_stderr(self, process: Any) -> None:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return
        try:
            for line in stream:
                text = str(line).strip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("embedded cua-driver: %s", text)
        except Exception:
            pass

    def start(self) -> None:
        if self._running:
            return
        from tools.environments.local import _sanitize_subprocess_env

        if not self._driver_cmd:
            self._driver_cmd = resolve_cua_driver_cmd() or ""
        if not self._driver_cmd:
            raise RuntimeError(cua_driver_install_hint())
        self._command, self._mcp_args = _resolve_mcp_invocation(self._driver_cmd)
        env = _sanitize_subprocess_env(self.child_env())
        serve_args = [
            "serve",
            "--embedded",
            "--socket",
            self.socket_path,
            "--no-permissions-gate",
            "--permission-mode",
            self.permission_mode,
        ]
        if self.permission_mode == "unrestricted":
            serve_args.append("--dangerously-bypass-approvals")
        # A v3 manifest is a ceiling, not a mode: cua-driver accepts it
        # alongside any permission mode and it "can narrow a profile but never
        # widen it". Attaching it to unrestricted is what bounds an
        # approval-bypassed run to declared scope, so pass it whenever it
        # applies — not only for bounded, which used to drop it for every
        # other mode.
        if self.manifest_applies:
            serve_args.extend(
                [
                    "--capability-manifest",
                    str(self.capability_manifest),
                    "--approve-capability-manifest",
                ]
            )
        # The private daemon owns the platform cursor overlay. Applying the
        # policy only to its MCP proxy leaves this long-lived serve process
        # free to create a full-screen overlay before session tuning runs.
        # Must be appended BEFORE the macOS app-launch wrapping so the flag
        # travels inside `open ... --args` with the rest of the serve args.
        serve_args = _mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command)
        self._launch_via_app = sys.platform == "darwin"
        command = _embedded_daemon_spawn_command(
            self._command,
            serve_args,
            platform=sys.platform,
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._owns_runtime = True
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process,),
            name="hermes-cua-daemon-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        deadline = time.monotonic() + self._START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None and (
                not self._launch_via_app or return_code != 0
            ):
                detail = "; ".join(self._stderr_tail) or "no diagnostic output"
                raise RuntimeError(
                    f"embedded cua-driver exited during startup: {detail}"
                )
            try:
                probe = subprocess.run(
                    [self._command, "status", "--socket", self.socket_path],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    env=env,
                )
            except (OSError, subprocess.SubprocessError):
                probe = None
            if probe is not None and probe.returncode == 0:
                self._running = True
                return
            time.sleep(0.1)

        self.stop()
        detail = "; ".join(self._stderr_tail) or "daemon did not become ready"
        raise RuntimeError(f"embedded cua-driver startup timed out: {detail}")

    def proxy_invocation(self) -> Tuple[str, List[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [
            *self._mcp_args,
            "--embedded",
            "--socket",
            self.socket_path,
        ]

    def stop(self) -> None:
        process = self._process
        self._process = None
        owns_runtime = self._owns_runtime
        self._owns_runtime = False
        self._running = False
        if owns_runtime:
            from tools.environments.local import _sanitize_subprocess_env

            try:
                subprocess.run(
                    [self._command, "stop", "--socket", self.socket_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    env=_sanitize_subprocess_env(self.child_env()),
                )
            except (OSError, subprocess.SubprocessError):
                pass
        if process is not None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        if sys.platform != "win32" and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass


def _resolve_mcp_invocation(
    driver_cmd: str,
    *,
    timeout: float = 6.0,
) -> Tuple[str, List[str]]:
    """Return ``(command, args)`` that spawn cua-driver's stdio MCP server.

    Surface 8 of NousResearch/hermes-agent#47072: instead of hardcoding
    ``["mcp"]`` we ask the driver itself via ``cua-driver manifest``
    (trycua/cua#1961). The manifest carries a stable ``mcp_invocation``
    pointer with both ``command`` and ``args``, so a future cua-driver
    that renames or relocates the subcommand keeps working without a
    Hermes patch.

    Falls back to ``(driver_cmd, ["mcp"])`` for older drivers that don't
    expose ``manifest``, or any indeterminate failure — the wrapper must
    not refuse to start just because the discovery hop failed.

    When ``computer_use.no_overlay`` is enabled (or auto-detected — macOS,
    headless/WSL2/X11 Linux), ``--no-overlay`` is appended to suppress the
    cursor overlay rendering loop that can consume CPU indefinitely when idle
    (#28152, #47032).  Older drivers that don't recognise the flag will
    reject it; callers should fall back to the no-overlay invocation on
    spawn failure.
    """
    try:
        from tools.environments.local import _sanitize_subprocess_env
        proc = subprocess.run(
            [driver_cmd, "manifest"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
            # cua-driver is a third-party binary — never hand it provider
            # API keys via inherited env (same policy as the MCP and CLI
            # fallback spawns below; #53503/#55709/#58889 lineage).
            env=_sanitize_subprocess_env(cua_driver_child_env()),
        )
    except Exception:
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    try:
        manifest = json.loads(out)
    except (ValueError, TypeError):
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    if not isinstance(manifest, dict):
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    invocation = manifest.get("mcp_invocation")
    if not isinstance(invocation, dict):
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    args = invocation.get("args")
    command = invocation.get("command")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return driver_cmd, _mcp_args_with_overlay_flag(list(_CUA_DRIVER_ARGS), driver_cmd=driver_cmd)
    if not isinstance(command, str) or not command:
        # The driver knows the subcommand but didn't surface its own path.
        # Keep our resolved driver_cmd; the args are still authoritative.
        return driver_cmd, _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)
    # A Windows-installed cua-driver can hand a WSL-hosted Hermes an absolute
    # ``C:\...`` command; translate it to its DrvFS ``/mnt/<drive>/...`` form
    # BEFORE the path-separator check (backslash is not a separator on POSIX,
    # so the raw Windows string would otherwise be discarded here).
    command = _wsl_windows_path_to_posix(command)
    if not _has_path_separator(command):
        # A manifest may legitimately retain the generic ``cua-driver`` name.
        # Under a GUI's thin PATH that would lose the resolved user-local path
        # and fail at MCP spawn, so preserve the concrete command we verified.
        return driver_cmd, _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)
    # Manifest surfaced a relocated executable — probe THAT binary for
    # `--no-overlay` support rather than the system-resolved one, so a
    # wrapper/relocation with a different feature set doesn't crash on
    # an unknown flag (or silently keep an unwanted overlay).
    return command, _mcp_args_with_overlay_flag(args, driver_cmd=command)


def _mcp_args_with_overlay_flag(
    args: List[str],
    driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD,
) -> List[str]:
    """Return *args* with ``--no-overlay`` appended when configured and supported."""
    if _cua_no_overlay() and _cua_driver_supports_no_overlay(driver_cmd):
        return [*args, "--no-overlay"]
    return list(args)


@functools.lru_cache(maxsize=1)
def _cua_driver_supports_no_overlay(driver_cmd: str) -> bool:
    """True if the installed cua-driver recognises ``--no-overlay``.

    Probes ``<driver> --help`` once and caches the result.  Older
    drivers (< 0.6.x) reject unknown flags, so passing ``--no-overlay``
    would crash the MCP spawn.
    """
    try:
        # cua-driver is a third-party binary — never hand it provider
        # API keys via inherited env (same policy as the manifest probe
        # and MCP spawn; #53503/#55709/#58889 lineage).
        from tools.environments.local import _sanitize_subprocess_env
        proc = subprocess.run(
            [driver_cmd, "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3.0,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
            env=_sanitize_subprocess_env(cua_driver_child_env()),
        )
        help_text = (proc.stdout or "") + (proc.stderr or "")
        return "--no-overlay" in help_text
    except Exception:
        return False

# Regex to parse element lines from get_window_state AX tree markdown.
#
# cua-driver renders each actionable node as one of:
#   - [N] AXRole "label"                         (quoted label, classic)
#   - [N] AXRole = "value"                        (value form, e.g. AXStaticText/AXPopUpButton)
#   - [N] AXRole (label)                          (parenthesised label, e.g. AXButton (Dark))
#   - [N] AXRole (order) id=Label                 (order number + id= label, newer builds)
#   - [N] AXRole id=Label                         (id= label only)
#   - [N] AXRole                                  (no label)
# followed by trailing metadata like [help="..." actions=[...]].
#
# Earlier the regex only matched the quoted and id= forms, so the very common
# `(label)` and `= "value"` forms (System Settings buttons, static text, popups)
# came back with an empty label — which made label-driven clicking impossible.
# A parenthesised group that is purely digits is an ORDER index, not a label, so
# it is excluded and we fall through to the id= label.
#
# Group 1: element index   Group 2: AX role
# Groups 3-6: the label in value / quoted / paren / id= form (whichever matched)
_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)'
    r'(?:'
      r'\s*=\s*"([^"]*)"'              # = "value"
      r'|\s+"([^"]*)"'                 # "value"
      r'|\s+\((?!\d+\))([^)]*)\)'      # (value) but not a pure-digit (order) number
    r')?'
    r'(?:\s+(?:\(\d+\)\s+)?id=([^\s\[\]]+))?',  # optional id=value (after an optional (order))
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return sys.platform == "darwin"


def _has_path_separator(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value)


def _candidate_cua_driver_commands(override: Optional[str] = None) -> List[str]:
    """Return candidate cua-driver commands in resolution order.

    ``override`` is authoritative when supplied. Otherwise a non-empty
    ``HERMES_CUA_DRIVER_CMD`` is authoritative; only when neither is set do we
    use PATH and canonical install locations.

    Desktop apps launched from Finder/Dock often inherit a narrow PATH that
    omits user-local install directories. The upstream cua-driver installer
    commonly places the binary under ``~/.local/bin`` on POSIX systems, so a
    Hermes Desktop/TUI session can otherwise filter out the `computer_use`
    tool even though `hermes computer-use doctor` succeeds from a login shell.
    """
    configured = (override if override is not None else os.environ.get(_CUA_DRIVER_CMD_ENV, "")).strip()
    if configured:
        # An explicit override is authoritative: if it is wrong, report the
        # driver missing instead of silently picking a different binary.
        return [configured]

    candidates = [_CUA_DRIVER_DEFAULT_CMD]
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(
            home, "AppData", "Local"
        )
        candidates.extend([
            # Official cua-driver installer location on Windows. Freshly
            # installed sessions inherit a stale PATH, so PATH lookup alone
            # misses it until every Hermes process is restarted.
            os.path.join(
                local_app_data, "Programs", "Cua", "cua-driver", "bin", "cua-driver.exe"
            ),
            os.path.join(home, ".local", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver"),
        ])
    else:
        candidates.extend([
            os.path.join(home, ".local", "bin", "cua-driver"),
            os.path.join(home, ".cargo", "bin", "cua-driver"),
            "/opt/homebrew/bin/cua-driver",
            "/usr/local/bin/cua-driver",
        ])
    return candidates


def resolve_cua_driver_cmd(override: Optional[str] = None) -> Optional[str]:
    """Resolve the cua-driver executable for every runtime/status surface.

    A supplied override (or ``HERMES_CUA_DRIVER_CMD``) is never silently
    replaced by another binary. Otherwise resolve PATH first, then canonical
    user-local installation locations used by the official installer.
    """
    for candidate in _candidate_cua_driver_commands(override):
        expanded = os.path.expanduser(candidate)
        if _has_path_separator(expanded):
            if shutil.which(expanded):
                return expanded
        else:
            resolved = shutil.which(expanded)
            if resolved:
                return resolved
    return None


def cua_driver_binary_available() -> bool:
    """True if `cua-driver` resolves via env, PATH, or known install paths."""
    return resolve_cua_driver_cmd() is not None


_CUA_DRIVER_RUNTIME_CONTRACT_MIN = (0, 20, 0)
_CUA_DRIVER_RUNTIME_CONTRACT_ARGS = {
    "mcp": {"--socket", "--grant"},
    "serve": {
        "--socket",
        "--permission-mode",
        "--capability-manifest",
        "--approve-capability-manifest",
        "--embedded",
    },
    "stop": {"--socket"},
}


def cua_driver_runtime_contract_status(binary: Optional[str] = None) -> Dict[str, Any]:
    """Report whether a local driver can host Hermes' 0.20 integration."""
    resolved = binary or resolve_cua_driver_cmd()
    if not resolved:
        return {
            "ready": False,
            "binary": None,
            "version": None,
            "reason": "cua-driver is not installed",
        }

    try:
        from tools.environments.local import _sanitize_subprocess_env

        result = subprocess.run(
            [resolved, "manifest"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0 if sys.platform == "win32" else 5.0,
            stdin=subprocess.DEVNULL,
            env=_sanitize_subprocess_env(cua_driver_child_env()),
            creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ready": False,
            "binary": resolved,
            "version": None,
            "reason": f"manifest check failed: {exc}",
        }

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "manifest command failed").strip()
        return {
            "ready": False,
            "binary": resolved,
            "version": None,
            "reason": detail.splitlines()[-1][:200],
        }

    try:
        manifest = json.loads(result.stdout or "")
    except (TypeError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        return {
            "ready": False,
            "binary": resolved,
            "version": None,
            "reason": "driver manifest is missing or invalid",
        }

    raw_version = str(manifest.get("binary_version") or "").strip()
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", raw_version)
    if not match:
        return {
            "ready": False,
            "binary": resolved,
            "version": raw_version or None,
            "reason": "driver manifest does not report a semantic version",
        }
    version = tuple(int(part) for part in match.groups())
    if version < _CUA_DRIVER_RUNTIME_CONTRACT_MIN:
        return {
            "ready": False,
            "binary": resolved,
            "version": raw_version,
            "reason": "Hermes computer use requires cua-driver 0.20.0 or newer",
        }

    invocation = manifest.get("mcp_invocation")
    invocation_args = invocation.get("args") if isinstance(invocation, dict) else None
    if not (
        isinstance(invocation_args, list)
        and invocation_args
        and all(isinstance(arg, str) for arg in invocation_args)
    ):
        return {
            "ready": False,
            "binary": resolved,
            "version": raw_version,
            "reason": "driver manifest does not provide an MCP launch command",
        }

    advertised: Dict[str, set[str]] = {}
    for command in manifest.get("subcommands") or []:
        if not isinstance(command, dict) or not isinstance(command.get("name"), str):
            continue
        advertised[command["name"]] = {
            arg["name"]
            for arg in command.get("args") or []
            if isinstance(arg, dict) and isinstance(arg.get("name"), str)
        }

    missing = []
    for command, required_args in _CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items():
        for arg in sorted(required_args - advertised.get(command, set())):
            missing.append(f"{command} {arg}")
    if missing:
        return {
            "ready": False,
            "binary": resolved,
            "version": raw_version,
            "reason": "driver manifest is missing: " + ", ".join(missing),
        }

    return {
        "ready": True,
        "binary": resolved,
        "version": raw_version,
        "reason": "",
    }


def cua_driver_update_check(*, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Run ``cua-driver check-update --json`` and return its parsed state.

    The payload mirrors the ``check_for_update`` MCP tool:
    ``{current_version, latest_version, update_available, ...}``.

    ``timeout`` defaults to 8s on POSIX and 25s on Windows — first-spawn of
    the exe there routinely eats several seconds in Defender/SmartScreen
    scanning, and a false timeout is expensive: callers treat ``None`` as
    indeterminate, and the ``install_cua_driver(upgrade=True)`` path used to
    fall through to a full multi-minute reinstall on it.

    Returns ``None`` (callers should stay quiet) when the result is
    indeterminate: the binary is missing, the driver is too old to support
    the verb (it predates trycua/cua#1734), the GitHub check failed (an
    ``error`` field is set), or the output didn't parse. Best-effort; never
    raises.
    """
    if timeout is None:
        timeout = 25.0 if sys.platform == "win32" else 8.0
    driver_cmd = resolve_cua_driver_cmd()
    if not driver_cmd:
        return None
    try:
        from tools.environments.local import _sanitize_subprocess_env
        proc = subprocess.run(
            [driver_cmd, "check-update", "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            # Some older drivers don't have the verb and fall through to a
            # stdin-reading mode rather than erroring — DEVNULL gives them EOF
            # so they exit fast instead of blocking until the timeout.
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
            # Sanitized like every other cua-driver spawn: third-party
            # binary, no inherited provider keys (#53503/#55709/#58889).
            env=_sanitize_subprocess_env(cua_driver_child_env()),
        )
    except Exception:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        # Older drivers don't have the verb: usage goes to stderr, stdout empty.
        return None
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("error"):
        # A failed check (exit 1) carries its reason in `error` — indeterminate.
        return None
    return data


def cua_driver_update_nudge() -> Optional[str]:
    """One-line "an update is available" message, or ``None`` when up to date,
    indeterminate, or the driver is too old to report."""
    state = cua_driver_update_check()
    if not state or not state.get("update_available"):
        return None
    latest = state.get("latest_version") or "?"
    current = state.get("current_version") or "?"
    return (
        f"cua-driver {latest} is available (you have {current}); "
        f"update with `hermes computer-use install --upgrade`."
    )


_update_checked = False
# One auto-repair attempt per process: when the runtime-contract gate fails for something a reinstall fixes
# (old version, missing manifest verbs) run the standard install path once instead of telling the user to.
# Guarded so a failing installer can't loop — the second start() goes straight to the error.
_contract_repair_attempted = False

def _maybe_repair_runtime_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Try one automatic driver repair; return the post-repair contract (or the original when no repair was
    attempted / it failed). Never raises. An explicit ``HERMES_CUA_DRIVER_CMD`` override is authoritative even
    when broken, and a missing binary means installation was never requested."""
    global _contract_repair_attempted
    if contract.get("ready") or _contract_repair_attempted or os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip() or not contract.get("binary"):
        return contract
    _contract_repair_attempted = True
    logger.info("computer_use: installed cua-driver is not usable (%s); attempting automatic repair",
                contract.get("reason") or "runtime contract is incomplete")
    try:
        from hermes_cli.tools_config import install_cua_driver
        repaired = install_cua_driver(upgrade=False, show_installer_progress=False)
    except Exception as exc:
        logger.warning("computer_use: automatic cua-driver repair failed: %s", exc)
        return contract
    with contextlib.suppress(Exception):
        return cua_driver_runtime_contract_status() if repaired else contract
    return contract

def _maybe_nudge_update() -> None:
    """Emit an update nudge at most once per process, off-thread so the (cached, ~20h) GitHub poll never blocks
    the first computer_use action."""
    global _update_checked
    if _update_checked:
        return
    _update_checked = True

    def _run() -> None:
        with contextlib.suppress(Exception):
            msg = cua_driver_update_nudge()
            msg and logger.info("computer_use: %s", msg)

    threading.Thread(target=_run, name="cua-driver-update-check", daemon=True).start()


def cua_driver_install_hint() -> str:
    if sys.platform == "win32":
        installer = (
            '  irm https://raw.githubusercontent.com/trycua/cua/main/'
            'libs/cua-driver/scripts/install.ps1 | iex'
        )
    else:
        installer = (
            '  /bin/bash -c "$(curl -fsSL '
            'https://raw.githubusercontent.com/trycua/cua/main/'
            'libs/cua-driver/scripts/install.sh)"'
        )
    return (
        "cua-driver is not installed. Install with one of:\n"
        "  hermes computer-use install\n"
        "Or run the upstream installer directly:\n"
        f"{installer}\n"
        "Or run `hermes tools` and enable the Computer Use toolset to install it automatically."
    )


def _parse_elements_from_tree(markdown: str) -> List[UIElement]:
    """Parse UIElement list from get_window_state AX tree markdown.

    Last-resort fallback for cua-driver builds that don't carry the
    canonical ``structuredContent.elements`` array (see
    ``_parse_elements_from_structured`` — Surface 2 of #47072 prefers
    that path).

    Captures the label whichever form cua-driver used: ``= "value"``,
    ``"quoted"``, ``(parenthesised)``, or ``id=Label``. Bounds always
    come back ``(0, 0, 0, 0)`` because the markdown surface doesn't
    carry them — yet another reason to prefer the structured path;
    element-index clicks don't need them (the driver resolves the index
    to a frame internally).
    """
    elements = []
    for m in _ELEMENT_LINE_RE.finditer(markdown):
        # groups 3-6: value / quoted / paren / id= label (first non-None wins)
        label = m.group(3) or m.group(4) or m.group(5) or m.group(6) or ""
        elements.append(UIElement(
            index=int(m.group(1)),
            role=m.group(2),
            label=label,
            bounds=(0, 0, 0, 0),
        ))
    return elements


def _parse_elements_from_structured(raw_elements: List[Dict[str, Any]]) -> List[UIElement]:
    """Surface 2 of NousResearch/hermes-agent#47072: read the canonical
    ``structuredContent.elements`` array cua-driver-rs emits on every
    ``get_window_state`` response (trycua/cua#1961).

    Each entry has at minimum ``element_index``, ``role``, ``label``;
    ``frame`` (``{x, y, w, h}``) is included whenever the AT-SPI /
    AXFrame call returned usable bounds. Older code parsed the same
    information out of the markdown tree via a regex (lossy: bounds
    were always ``(0, 0, 0, 0)``) — this path preserves the real
    frame so downstream consumers (e.g. ``UIElement.center()``) work
    against pixel coordinates instead of just the index lookup.

    Unknown / malformed entries are skipped rather than failing the
    whole walk — the wrapper degrades to "fewer elements" rather than
    "no elements" on a bad row.
    """
    elements: List[UIElement] = []
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("element_index")
        if not isinstance(idx, int):
            continue
        role = raw.get("role") if isinstance(raw.get("role"), str) else ""
        label = raw.get("label") if isinstance(raw.get("label"), str) else ""
        frame = raw.get("frame") if isinstance(raw.get("frame"), dict) else None
        bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        if frame:
            try:
                bounds = (
                    int(frame.get("x", 0)),
                    int(frame.get("y", 0)),
                    int(frame.get("w", 0)),
                    int(frame.get("h", 0)),
                )
            except (TypeError, ValueError):
                bounds = (0, 0, 0, 0)
        # Surface 6: opaque element_token. cua-driver-rs format is
        # `s{snapshot_hex}:{index}`. We treat it as a black-box string —
        # the driver owns the parse + LRU semantics.
        raw_token = raw.get("element_token")
        token = raw_token if isinstance(raw_token, str) and raw_token else None
        elements.append(UIElement(
            index=idx,
            role=role,
            label=label,
            bounds=bounds,
            element_token=token,
        ))
    return elements


def _image_dimensions_from_bytes(raw: bytes) -> Tuple[int, int]:
    """Best-effort PNG/JPEG dimension sniffing without extra dependencies."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        if width > 0 and height > 0:
            return width, height

    if raw.startswith(b"\xff\xd8"):
        i = 2
        n = len(raw)
        while i + 9 < n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > n:
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > n:
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                if segment_len >= 7:
                    height = int.from_bytes(raw[i + 3:i + 5], "big")
                    width = int.from_bytes(raw[i + 5:i + 7], "big")
                    if width > 0 and height > 0:
                        return width, height
                break
            i += segment_len

    return 0, 0


def _split_tree_text(full_text: str) -> Tuple[str, str]:
    """Split get_window_state text into (summary_line, tree_markdown)."""
    lines = full_text.split("\n", 1)
    summary = lines[0]
    tree = lines[1] if len(lines) > 1 else ""
    return summary, tree


def _parse_key_combo(keys: str) -> Tuple[Optional[str], List[str]]:
    """Parse a key string like 'cmd+s' into (key, modifiers).

    Returns (key, modifiers) where key is the non-modifier key and modifiers
    is a list of modifier names (cmd, shift, option, ctrl).
    """
    MODIFIER_NAMES = {"cmd", "command", "shift", "option", "alt", "ctrl", "control", "fn"}
    KEY_ALIASES = {"command": "cmd", "alt": "option", "control": "ctrl"}

    parts = [p.strip().lower() for p in re.split(r'[+\-]', keys) if p.strip()]
    modifiers = []
    key = None
    for part in parts:
        normalized = KEY_ALIASES.get(part, part)
        if normalized in MODIFIER_NAMES:
            modifiers.append(normalized)
        else:
            key = part  # last non-modifier wins
    return key, modifiers


# ---------------------------------------------------------------------------
# Asyncio bridge — one long-lived loop on a background thread
# ---------------------------------------------------------------------------

class _AsyncBridge:
    """Runs one asyncio loop on a daemon thread; marshals coroutines from the caller."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, daemon=True, name="cua-driver-loop")
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("cua-driver asyncio bridge failed to start")

    def run(self, coro, timeout: Optional[float] = 30.0) -> Any:
        from agent.async_utils import safe_schedule_threadsafe
        if not self._loop or not self._thread or not self._thread.is_alive():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("cua-driver bridge not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("cua-driver bridge not started")
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._loop = None


# ---------------------------------------------------------------------------
# MCP session (lazy, shared across tool calls)
# ---------------------------------------------------------------------------

class _CuaDriverSession:
    """Holds the mcp ClientSession. Spawned lazily; re-entered on drop.

    Lifecycle ownership: a single long-running coroutine
    (`_lifecycle_coro`) opens both the stdio_client and ClientSession
    contexts, populates capabilities, sets `_ready_event`, and then waits
    on `_shutdown_event`. When shutdown is signalled the same coroutine
    closes the contexts — keeping anyio's cancel-scope task-identity
    invariant intact (the bridge schedules each `bridge.run(coro)` as a
    NEW task, so opening contexts in one and closing them in another
    raises "Attempted to exit cancel scope in a different task").
    Tool calls run in their own short-lived tasks; they only touch the
    session object, never the surrounding contexts.
    """

    def __init__(
        self,
        bridge: _AsyncBridge,
        embedded_daemon: Optional[_EmbeddedCuaDaemon] = None,
    ) -> None:
        self._bridge = bridge
        self._embedded_daemon = embedded_daemon
        self._session = None
        self._lock = threading.Lock()
        self._started = False
        # Surface 4 of NousResearch/hermes-agent#47072: per-tool
        # capability-token sets, populated from `tools/list` at session
        # init. Keys are tool names (e.g. "click", "get_window_state");
        # values are sets of capability strings (e.g.
        # "accessibility.element_tokens", "input.keyboard.type.terminal_safe").
        # Empty until the session starts; consumers should call
        # `supports_capability` rather than reading directly.
        self._capabilities: Dict[str, set] = {}
        # Raw input schemas are the compatibility source of truth for action
        # properties.  cua-driver 0.9-era builds advertise delivery_mode in
        # inputSchema while intentionally omitting the old, fabricated
        # ``input.delivery_mode`` capability token.
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._capability_version: str = ""
        # Lifecycle plumbing — see class docstring above.
        self._ready_event = threading.Event()
        self._shutdown_event: Optional[asyncio.Event] = None  # created on bridge loop
        self._lifecycle_future = None  # concurrent.futures.Future
        self._setup_error: Optional[BaseException] = None
        # Stable driver-side identity declared through start_session.
        # Used to revive a logical ended-session rejection without
        # recursive call_tool re-entry or backend-owned state (#71166).
        self._declared_session_id: Optional[str] = None
        self._transport_generation = 0
        self._transport_reset_callback: Optional[Any] = None

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("cua-driver session not started")

    async def _lifecycle_coro(self) -> None:
        """Long-lived owner of the stdio MCP contexts. Opens, signals
        ready, blocks on shutdown, then cleans up. enter + exit happen
        in the SAME asyncio task, so anyio's cancel-scope invariant
        holds — fixing the "Attempted to exit cancel scope in a
        different task than it was entered in" warning emitted by the
        previous _aenter/_aexit split.
        """
        import time as _time
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.environments.local import _sanitize_subprocess_env

        # Build the shutdown event on the loop's thread so the asyncio
        # primitive belongs to the correct loop.
        self._shutdown_event = asyncio.Event()
        _t0 = _time.monotonic()
        # Phase marker surfaced by the ready-timeout error (issue #57025):
        # when startup wedges, the caller reports HOW FAR it got instead of
        # an opaque "never reached ready".
        self._startup_phase = "binary-check"

        try:
            driver_cmd = resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(cua_driver_install_hint())

            # Surface 8: ask cua-driver itself which subcommand spawns
            # the MCP server, instead of hardcoding ["mcp"]. Falls back
            # transparently for older drivers / any discovery failure.
            self._startup_phase = "manifest-discovery"
            if self._embedded_daemon is not None:
                command, args = self._embedded_daemon.proxy_invocation()
                child_env = self._embedded_daemon.child_env()
            else:
                command, args = _resolve_mcp_invocation(driver_cmd)
                child_env = cua_driver_child_env()
            _t_manifest = _time.monotonic()
            params = StdioServerParameters(
                command=command,
                args=args,
                # Apply the telemetry policy first (default: disabled), then
                # sanitize Hermes-managed secrets out of the child env.
                env=_sanitize_subprocess_env(child_env),
            )

            async with stdio_client(params) as (read, write):
                self._startup_phase = "mcp-initialize"
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    _t_init = _time.monotonic()
                    # Populate capabilities + capability_version BEFORE
                    # exposing the session to callers, so the first
                    # tool call already sees them.
                    self._startup_phase = "capability-discovery"
                    await self._populate_capabilities(session)
                    self._session = session
                    self._startup_phase = "ready"
                    self._ready_event.set()
                    logger.info(
                        "cua-driver session ready in %.1fs "
                        "(manifest=%.1fs, mcp_init=%.1fs)",
                        _time.monotonic() - _t0,
                        _t_manifest - _t0,
                        _t_init - _t_manifest,
                    )
                    # Hold the contexts open until stop() / restart asks
                    # us to wind down. Tool calls run as their own tasks
                    # on the same loop and touch self._session directly.
                    await self._shutdown_event.wait()
        except BaseException as e:
            # Capture both ordinary errors and anyio CancelledError.
            # The caller (start()) inspects this to surface setup
            # failures to the synchronous world.
            self._setup_error = e
            self._ready_event.set()
            raise
        finally:
            # Clearing _session before the contexts unwind would let a
            # racing call_tool see None during teardown — but the
            # outer context-manager exits AFTER this block, so set to
            # None here is fine: stop() has already flipped _started.
            self._session = None
            # Reset _started so a session that dies for ANY reason (MCP
            # connection drop, driver crash, unexpected coro exit) is
            # re-enterable: the next start()/call sees _started False and
            # rebuilds the session instead of hanging forever on a dead one
            # via _require_started(). On the normal stop() path this is a
            # harmless idempotent no-op (stop() already set it False). A
            # plain bool write is atomic in CPython, so this is safe from
            # the bridge-loop thread without taking self._lock (which stop()
            # may hold while awaiting this coro's future). See #55048 Bug 1.
            self._started = False

    async def _populate_capabilities(self, session: Any) -> None:
        """Surface 4: cache per-tool capability sets + capability_version
        from tools/list. Soft prerequisite — discovery failure leaves
        the map empty and supports_capability degrades to False."""
        self._capabilities = {}
        self._tool_schemas = {}
        self._capability_version = ""
        try:
            tools_list = await session.list_tools()
            for tool in getattr(tools_list, "tools", []) or []:
                tool_name = getattr(tool, "name", None)
                if not isinstance(tool_name, str):
                    continue
                caps = getattr(tool, "capabilities", None)
                if caps is None:
                    # Some MCP SDKs forward custom fields via
                    # `model_extra` (Pydantic v2) instead of attributes.
                    extra = getattr(tool, "model_extra", None) or {}
                    caps = extra.get("capabilities")
                if isinstance(caps, list):
                    self._capabilities[tool_name] = {
                        c for c in caps if isinstance(c, str)
                    }
                else:
                    self._capabilities[tool_name] = set()
                schema = _mcp_field(tool, "input_schema", "inputSchema")
                if schema is None:
                    schema = (getattr(tool, "model_extra", None) or {}).get(
                        "inputSchema"
                    )
                self._tool_schemas[tool_name] = (
                    dict(schema) if isinstance(schema, dict) else {}
                )
            # capability_version is a top-level sibling of `tools` on the
            # tools/list response. cua-driver-core/src/tool.rs:354 emits
            # it; cua-driver-core/src/protocol.rs:150 leaves it OUT of
            # initialize — so we discover here, not there.
            cv = getattr(tools_list, "capability_version", None)
            if cv is None:
                extra = getattr(tools_list, "model_extra", None) or {}
                cv = extra.get("capability_version")
            if isinstance(cv, str):
                self._capability_version = cv
        except Exception as e:
            logger.debug("cua-driver tools/list capability discovery failed: %s", e)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._bridge.start()
            self._start_lifecycle_locked()
            self._started = True

    def _start_lifecycle_locked(self) -> None:
        """Spawn the lifecycle owner and wait for it to reach ready.
        Caller must hold self._lock."""
        # Reset per-session state.
        self._ready_event = threading.Event()
        self._setup_error = None
        self._shutdown_event = None
        # Fire-and-forget schedule on the bridge loop. The future tracks
        # completion of the WHOLE lifecycle (open → wait → close), not
        # just the open step — start() waits on _ready_event separately.
        loop = self._bridge._loop
        if loop is None:
            raise RuntimeError("cua-driver bridge not started")
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(
            self._lifecycle_coro(), loop
        )
        if not self._ready_event.wait(timeout=30.0):
            # Best-effort: signal shutdown if the future is still alive.
            self._signal_shutdown_locked()
            # Surface which startup phase wedged (issue #57025) — "doctor
            # passes but the wrapper times out" reports are undiagnosable
            # from a bare "never reached ready".
            phase = getattr(self, "_startup_phase", "unknown")
            from hermes_constants import display_hermes_home
            raise RuntimeError(
                "cua-driver session never reached ready (timeout 30s; "
                f"stuck in phase: {phase}). "
                "Run `hermes computer-use doctor` and check "
                f"{display_hermes_home()}/logs/agent.log for the phase timings."
            )
        # If setup failed, the lifecycle coroutine set _setup_error
        # before setting _ready_event. Re-raise it on the caller's thread.
        if self._setup_error is not None:
            raise RuntimeError(
                f"cua-driver session setup failed: {self._setup_error}"
            ) from self._setup_error
        self._transport_generation += 1
        if self._transport_generation > 1:
            self._notify_transport_reset()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_lifecycle_locked()

    def set_transport_reset_callback(self, callback: Any) -> None:
        """Register a synchronous cache invalidation hook for transport swaps."""
        self._transport_reset_callback = callback

    def _notify_transport_reset(self) -> None:
        callback = getattr(self, "_transport_reset_callback", None)
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            logger.debug("cua-driver transport reset callback failed: %s", exc)

    def _stop_lifecycle_locked(self) -> None:
        """Signal shutdown + wait for the lifecycle coroutine to unwind.
        Caller must hold self._lock."""
        self._signal_shutdown_locked()
        fut = self._lifecycle_future
        if fut is None:
            return
        try:
            # 5s budget for context unwind (stdio_client teardown).
            fut.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            logger.warning("cua-driver session shutdown timed out (5s)")
        except Exception as e:
            # Real shutdown errors (not the previous cancel-scope race
            # which is now structurally impossible) still get surfaced.
            logger.warning("cua-driver shutdown error: %s", e)
        finally:
            self._lifecycle_future = None

    def _signal_shutdown_locked(self) -> None:
        """Set the asyncio shutdown event from the caller's thread."""
        loop = self._bridge._loop
        event = self._shutdown_event
        if loop is not None and event is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # Loop closed — nothing to signal.
                pass

    async def _call_tool_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await self._session.call_tool(name, args)
        return _extract_tool_result(result)

    # ── Capability detection (Surface 4 of #47072) ────────────────────
    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        """Return True when the connected cua-driver advertises the given
        capability token (trycua/cua#1961 capability vocabulary).

        When ``tool`` is given, scope the check to that specific tool's
        advertised capability set. When omitted, return True if ANY tool
        advertises the capability — useful for "is this feature available
        anywhere on the driver" probes.

        Always returns False before the session is started (so consumers
        on a dead/uninitialised wrapper degrade rather than crash).
        """
        if tool is not None:
            return capability in self._capabilities.get(tool, set())
        return any(capability in caps for caps in self._capabilities.values())

    def _has_tool(self, name: str) -> bool:
        """Return True when ``tools/list`` advertised a tool by this name.

        Used to route capture(): cua-driver dropped the standalone
        ``screenshot`` tool and folded full-window PNG capture into
        ``get_window_state`` (whose own description notes it "Also captures
        a PNG screenshot of the specified window"). Older drivers that still
        expose ``screenshot`` keep using it; newer ones fall through to
        ``get_window_state``.

        Returns False when discovery hasn't populated the map yet — callers
        treat that as "unknown" and probe defensively rather than trusting it.
        """
        return name in self._capabilities

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        """Return whether a live action schema accepts ``property_name``.

        This deliberately inspects tools/list rather than guessing from the
        package version or requiring a capability token the driver never
        shipped.  A missing/invalid schema fails closed.
        """
        schema = getattr(self, "_tool_schemas", {}).get(tool, {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        return isinstance(properties, dict) and property_name in properties

    @property
    def capabilities_discovered(self) -> bool:
        """True once ``tools/list`` populated the per-tool map. When False,
        ``_has_tool`` answers are not trustworthy (discovery failed or the
        session hasn't started) and capture() should probe defensively."""
        return bool(self._capabilities)

    @property
    def capability_version(self) -> str:
        """Driver-advertised capability vocabulary version (empty string
        when the driver predates the field — older builds had no version)."""
        return self._capability_version

    @staticmethod
    def _logical_error_text(result: Dict[str, Any]) -> str:
        """Flatten a logical MCP error into text for narrow classification."""
        chunks: List[str] = []
        for value in (result.get("data"), result.get("structuredContent")):
            if isinstance(value, str):
                chunks.append(value)
            elif value is not None:
                try:
                    chunks.append(json.dumps(value, sort_keys=True))
                except (TypeError, ValueError):
                    chunks.append(str(value))
        return "\n".join(chunks)

    @classmethod
    def _is_ended_session_result(cls, result: Any) -> bool:
        """Recognise cua-driver's explicit recoverable ended-session result."""
        if not isinstance(result, dict) or result.get("isError") is not True:
            return False
        message = cls._logical_error_text(result).lower()
        return (
            "session" in message
            and ("has ended" in message or "session ended" in message)
            and "start_session" in message
        )

    def _revive_declared_session_once(
        self,
        name: str,
        args: Dict[str, Any],
        first_result: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Revive the stable session and replay one rejected tool call once."""
        session_id = self._declared_session_id
        if not session_id or name in self._LIFECYCLE_CALLS:
            return first_result

        logger.warning(
            "cua-driver session %s ended during %s; reviving and retrying once",
            session_id,
            name,
        )
        revive_result = self._bridge.run(
            self._call_tool_async("start_session", {"session": session_id}),
            timeout=timeout,
        )
        if revive_result.get("isError") is True:
            logger.warning(
                "cua-driver session %s could not be revived: %s",
                session_id,
                self._logical_error_text(revive_result),
            )
            return first_result

        # Return the second result as-is. A second rejection is surfaced; no loop.
        return self._bridge.run(
            self._call_tool_async(name, args),
            timeout=timeout,
        )

    def _restore_declared_session_after_transport_reset(self, timeout: float) -> None:
        """Re-attach the public label inside a replacement private lifecycle."""
        session_id = getattr(self, "_declared_session_id", None)
        if not session_id:
            return
        result = self._bridge.run(
            self._call_tool_async("start_session", {"session": session_id}),
            timeout=timeout,
        )
        if result.get("isError") is True:
            logger.warning(
                "cua-driver public session label %s could not be restored: %s",
                session_id,
                self._logical_error_text(result),
            )

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        """Return True for MCP/stdio failures that are recoverable by reconnecting."""
        name = exc.__class__.__name__
        module = getattr(exc.__class__, "__module__", "")
        return (
            name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}
            or (module.startswith("anyio") and "Resource" in name)
            or isinstance(exc, (BrokenPipeError, EOFError))
        )

    @staticmethod
    def _is_transient_daemon_error(exc: Exception) -> bool:
        """Return True for the cua-driver daemon-proxy EAGAIN congestion error.

        On macOS the ``cua-driver mcp`` bridge forwards calls to the CuaDriver
        daemon over a non-blocking unix socket. Heavier ops (notably
        ``get_window_state``, which walks the AX tree and captures a PNG) can
        come back as an ``MCPError`` carrying ``Resource temporarily
        unavailable (os error 35)`` — POSIX EAGAIN — when the socket buffer is
        momentarily full. This is transient by definition: the same call
        succeeds when retried after a short pause (which is why spaced-out
        single calls work while rapid/large ones intermittently fail). Detect
        it by message so we can retry with backoff rather than surfacing an
        empty 0x0 capture to the model. See the EAGAIN diagnosis in
        references/catalog-add-troubleshooting (apple-music skill) and the
        cua-driver daemon-proxy note.
        """
        msg = str(exc)
        return (
            "Resource temporarily unavailable" in msg
            or "os error 35" in msg
            or "daemon transport error" in msg
            or "daemon proxy" in msg
        )

    def _restart_session_locked(self) -> None:
        """Recreate the MCP session after the daemon/stdin transport was closed.
        Caller must hold self._lock (the reconnect-once retry path holds it)."""
        if self._started:
            try:
                self._stop_lifecycle_locked()
            except Exception as e:
                logger.debug("cua-driver session cleanup before reconnect failed: %s", e)
        self._started = False
        # Clear stale capability state; the next start populates from scratch.
        self._capabilities = {}
        self._tool_schemas = {}
        self._capability_version = ""
        self._start_lifecycle_locked()
        self._started = True

    def _call_tool_via_cli(self, name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Fallback transport: invoke ``cua-driver call <tool> <json>`` as a
        subprocess instead of going through the stdio MCP bridge.

        The ``cua-driver mcp`` stdio bridge can persistently fail to forward
        heavier calls (notably ``get_window_state``) to the daemon with POSIX
        EAGAIN, while the plain ``cua-driver call`` path — which talks to the
        daemon over its own socket — keeps working. When the MCP path gives up,
        we retry over the CLI and remap the JSON into the same dict shape that
        ``_extract_tool_result`` produces, so callers (capture(), _action(),
        list_windows parsing) are transport-agnostic.

        For ``get_window_state`` we route the screenshot to a temp file via
        ``screenshot_out_file`` so the daemon returns a tiny JSON body (a path)
        instead of a multi-megabyte base64 blob — the large payload is what
        congests the daemon socket and triggers EAGAIN in the first place. We
        read the PNG back from disk and base64-encode it ourselves. The CLI
        call is itself retried a few times with backoff, since the underlying
        daemon socket can still be momentarily busy.
        """
        import subprocess as _subprocess
        import tempfile as _tempfile
        import time as _time
        from tools.environments.local import _sanitize_subprocess_env

        call_args = dict(args)
        shot_file: Optional[str] = None
        if name == "get_window_state" and "screenshot_out_file" not in call_args:
            fd, shot_file = _tempfile.mkstemp(prefix="cua_shot_", suffix=".png")
            os.close(fd)
            call_args["screenshot_out_file"] = shot_file

        driver_command = resolve_cua_driver_cmd()
        if not driver_command:
            raise RuntimeError(cua_driver_install_hint())
        child_env = cua_driver_child_env()
        socket_args: List[str] = []
        embedded_daemon = getattr(self, "_embedded_daemon", None)
        if embedded_daemon is not None:
            driver_command = embedded_daemon.proxy_invocation()[0]
            child_env = embedded_daemon.child_env()
            socket_args = ["--socket", embedded_daemon.socket_path]
        cmd = [
            driver_command,
            "call",
            name,
            json.dumps(call_args),
            *socket_args,
        ]
        attempts = 4
        backoff = 0.5
        parsed: Any = None
        last_err = ""
        try:
            for attempt in range(attempts):
                try:
                    proc = _subprocess.run(
                        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(15.0, timeout),
                        creationflags=windows_hide_flags(),
                        env=_sanitize_subprocess_env(child_env),
                    )
                except Exception as e:  # pragma: no cover - subprocess spawn failure
                    raise RuntimeError(f"cua-driver CLI fallback for {name} failed to spawn: {e}") from e

                out = (proc.stdout or "").strip()
                last_err = out[:200] or (proc.stderr or "")[:200]
                # "daemon is not running" is a PERMANENT condition for this
                # invocation (`cua-driver call` requires the machine-wide
                # daemon socket, which Linux installs typically never start —
                # Hermes talks to the direct `cua-driver mcp` runtime
                # instead). Retrying with backoff burns ~3.5s of sleeps per
                # fallback for an outcome that cannot change; fail fast so
                # callers surface a diagnosable error immediately.
                if "daemon is not running" in out or "daemon is not running" in (proc.stderr or ""):
                    raise RuntimeError(
                        f"cua-driver CLI fallback for {name} unavailable: the "
                        "machine-wide cua-driver daemon is not running (the "
                        "CLI transport requires it; the MCP runtime does not)."
                    )
                start = min(
                    (i for i in (out.find("{"), out.find("[")) if i != -1),
                    default=-1,
                )
                if start != -1:
                    try:
                        candidate = json.loads(out[start:])
                    except json.JSONDecodeError:
                        candidate = None
                    if candidate is not None:
                        parsed = candidate
                        break
                # No JSON (EAGAIN warning / empty) — retry with backoff.
                if attempt < attempts - 1:
                    logger.warning(
                        "cua-driver CLI fallback for %s got no JSON "
                        "(attempt %d/%d); retrying in %.1fs",
                        name, attempt + 1, attempts, backoff,
                    )
                    _time.sleep(backoff)
                    backoff *= 2

            if parsed is None:
                raise RuntimeError(
                    f"cua-driver CLI fallback for {name} returned no JSON after "
                    f"{attempts} attempts: {last_err}"
                )

            # Remap structured JSON into {data, images, structuredContent, isError}.
            images: List[str] = []
            data: Any = None
            structured: Optional[Dict] = parsed if isinstance(parsed, dict) else None
            is_error = False
            if isinstance(parsed, dict):
                # Current cua-driver CLI responses may report logical failures
                # in-band even when the subprocess itself exits successfully.
                # Preserve that bit so stateful callers can fail closed.
                is_error = parsed.get("isError") is True or parsed.get("is_error") is True
                shot = parsed.get("screenshot_png_b64")
                if not shot:
                    # Screenshot was routed to a file (ours or the daemon's choice).
                    fpath = parsed.get("screenshot_file_path") or shot_file
                    if fpath and os.path.exists(fpath):
                        try:
                            with open(fpath, "rb") as fh:
                                shot = base64.b64encode(fh.read()).decode("ascii")
                        except Exception as e:
                            logger.debug("cua-driver CLI fallback: failed reading %s: %s", fpath, e)
                if shot:
                    images.append(shot)
                tree = parsed.get("tree_markdown")
                if tree is not None:
                    ec = parsed.get("element_count")
                    summary = f"{ec} elements" if ec is not None else ""
                    data = f"{summary}\n{tree}" if summary else tree
            return {
                "data": data,
                "images": images,
                "structuredContent": structured,
                "isError": is_error,
            }
        finally:
            if shot_file and os.path.exists(shot_file):
                try:
                    os.remove(shot_file)
                except OSError:
                    pass

    # Lifecycle handshake calls issued BY start()/stop() themselves — these
    # must not trigger the auto-restart guard below, or start() would recurse
    # into start() when the session-start hasn't flipped _started yet.
    _LIFECYCLE_CALLS = frozenset({"start_session", "end_session"})

    # Retrying these calls after a broken transport is safe. The first call
    # either had no side effect or is explicitly idempotent. Mutations stay
    # out of this set because a lost response does not prove they failed.
    _TRANSPORT_REPLAY_SAFE_TOOLS = frozenset({
        "get_cursor_position",
        "get_displays",
        "get_screen_size",
        "get_window_state",
        "list_apps",
        "list_windows",
    })

    # Set when an MCP call timed out (#74799): a timed-out session is
    # wedged for all later calls, so it is torn down and recreated before
    # the next non-lifecycle call_tool. Class-level default so tests that
    # bypass __init__ see a healthy (non-suspect) session.
    _timeout_suspect = False

    @classmethod
    def _transport_replay_is_safe(cls, name: str) -> bool:
        return name in cls._TRANSPORT_REPLAY_SAFE_TOOLS

    @staticmethod
    def _unknown_transport_outcome(name: str, exc: Exception) -> Dict[str, Any]:
        message = (
            f"cua-driver transport failed during {name}; the action outcome is "
            "unknown, so Hermes did not replay it. Take fresh state before "
            "deciding whether to act again."
        )
        return {
            "data": message,
            "images": [],
            "image_mime_types": [],
            "structuredContent": {
                "ok": False,
                "code": "transport_outcome_unknown",
                "message": message,
                "operation": name,
                "next_step": "fresh_state",
                "detail": str(exc),
            },
            "isError": True,
        }

    @staticmethod
    def _timeout_outcome(name: str, exc: Exception) -> Dict[str, Any]:
        """Fail-closed result for an MCP call that hit its deadline (#74799).

        The action MAY have taken effect on the remote screen before the
        response was lost — the same effect_disposition=unknown principle as
        ``_unknown_transport_outcome`` — so the timed-out call is never
        silently replayed here; the caller decides after taking fresh state.
        """
        message = (
            f"cua-driver MCP call {name} timed out; the action outcome is "
            "unknown and may still have taken effect on the remote screen. "
            "The session has been marked suspect and will be recreated before "
            "the next computer-use call. Take fresh state before deciding "
            "whether to act again."
        )
        return {
            "data": message,
            "images": [],
            "image_mime_types": [],
            "structuredContent": {
                "ok": False,
                "code": "timeout_outcome_unknown",
                "message": message,
                "operation": name,
                "next_step": "fresh_state",
                "detail": str(exc),
            },
            "isError": True,
        }

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        # A prior MCP timeout (#74799) marks the session suspect: it may be
        # wedged for every later call. Recreate it before this call so a
        # single timeout never poisons the rest of the computer-use session.
        # Healthy sessions (flag clear) are never restarted here.
        if self._timeout_suspect and name not in self._LIFECYCLE_CALLS:
            logger.warning(
                "cua-driver session suspect after earlier MCP timeout; "
                "recreating before %s",
                name,
            )
            with self._lock:
                self._restart_session_locked()
            self._timeout_suspect = False
            self._restore_declared_session_after_transport_reset(timeout)

        # A prior session may have died (MCP drop / driver crash): its
        # lifecycle coro reset _started to False in its finally (#55048).
        if not self._started and name not in self._LIFECYCLE_CALLS:
            logger.warning(
                "cua-driver session not active on %s; (re)starting before call", name
            )
            self.start()
            self._restore_declared_session_after_transport_reset(timeout)
        self._require_started()

        try:
            result = self._bridge.run(
                self._call_tool_async(name, args),
                timeout=timeout,
            )
        except Exception as e:
            if isinstance(e, concurrent.futures.TimeoutError):
                # MCP deadline hit (#74799): the session is suspect and must
                # be recreated before the next call. Fail closed — the action
                # may have taken effect on the remote screen, so never replay
                # it here; surface the uncertainty instead (#74799).
                self._timeout_suspect = True
                logger.warning(
                    "cua-driver MCP timed out on %s; marking session suspect "
                    "for recreation before the next call",
                    name,
                )
                return self._timeout_outcome(name, e)
            if self._is_transient_daemon_error(e):
                if not self._transport_replay_is_safe(name):
                    self._notify_transport_reset()
                    return self._unknown_transport_outcome(name, e)
                logger.warning(
                    "cua-driver MCP transport failed on %s (%s); "
                    "falling back to CLI transport", name, e,
                )
                return self._call_tool_via_cli(name, args, timeout)
            if not self._is_closed_session_error(e):
                raise
            logger.warning("cua-driver MCP session closed during %s; reconnecting once", name)
            with self._lock:
                self._restart_session_locked()
            self._restore_declared_session_after_transport_reset(timeout)
            if not self._transport_replay_is_safe(name):
                return self._unknown_transport_outcome(name, e)
            result = self._bridge.run(
                self._call_tool_async(name, args),
                timeout=timeout,
            )

        # Remember only a successfully declared stable identity. Failed
        # start_session calls must not leave stale recovery state behind.
        if name == "start_session" and result.get("isError") is not True:
            declared_id = args.get("session")
            if isinstance(declared_id, str) and declared_id:
                self._declared_session_id = declared_id

        if self._is_ended_session_result(result):
            result = self._revive_declared_session_once(name, args, result, timeout)

        if (
            name == "end_session"
            and result.get("isError") is not True
            and args.get("session") == self._declared_session_id
        ):
            self._declared_session_id = None
        return result


def _extract_tool_result(mcp_result: Any) -> Dict[str, Any]:
    """Convert an mcp CallToolResult into a plain dict.

    cua-driver returns a mix of text parts, image parts, and structuredContent.
    We flatten into:
      {
        "data": <text or parsed json>,
        "images": [b64, ...],
        "image_mime_types": [mime, ...],   # parallel to `images`, "" when absent
        "structuredContent": <dict|None>,
        "isError": bool,
      }
    structuredContent is populated from the MCP result's structuredContent field
    (MCP spec §2024-11-05+) and takes precedence for structured data like
    list_windows window arrays.

    `image_mime_types` is the explicit `mimeType` cua-driver emits on every
    image part as of trycua/cua#1961 (Surface 7 of
    NousResearch/hermes-agent#47072). Each entry corresponds index-for-index
    with `images`; an empty string entry signals the part carried no
    mimeType (older cua-driver build), and the caller should fall back to
    base64-prefix sniffing.
    """
    data: Any = None
    images: List[str] = []
    image_mime_types: List[str] = []
    # Use identity, not truthiness: unittest mocks and proxy objects commonly
    # synthesize truthy attributes that were never present in the real result.
    is_error = _mcp_field(mcp_result, "is_error", "isError", False) is True
    structured: Optional[Dict] = (
        _mcp_field(mcp_result, "structured_content", "structuredContent") or None
    )
    text_chunks: List[str] = []
    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_chunks.append(getattr(part, "text", "") or "")
        elif ptype == "image":
            b64 = getattr(part, "data", None)
            if b64:
                images.append(b64)
                mime = _mcp_field(part, "mime_type", "mimeType") or ""
                image_mime_types.append(mime)
    if text_chunks:
        joined = "\n".join(t for t in text_chunks if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined
    return {
        "data": data,
        "images": images,
        "image_mime_types": image_mime_types,
        "structuredContent": structured,
        "isError": is_error,
    }


def _image_from_tool_result(out: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Pull a (png_b64, mime_type) pair out of a flattened tool result.

    cua-driver delivers window screenshots in two shapes depending on tool +
    transport:

      * As an MCP ``image`` content part — surfaced by ``_extract_tool_result``
        in ``out["images"]`` with a parallel ``image_mime_types`` entry. This
        is what ``get_window_state`` emits over the stdio MCP transport.
      * As a base64 field inside ``structuredContent`` —
        ``screenshot_png_b64`` (+ ``screenshot_mime_type``). This is what
        ``get_window_state`` returns when its structured payload carries the
        image instead of a content part (newer driver builds; also the shape
        seen via the ``cua-driver call`` CLI surface).

    Checking both makes capture() robust to either delivery shape, so the
    image never silently drops just because the driver moved it between the
    content list and structuredContent. Returns ``(None, None)`` when neither
    location carries an image.
    """
    images = out.get("images") or []
    if images and images[0]:
        mimes = out.get("image_mime_types") or []
        mime = mimes[0] if mimes and mimes[0] else None
        return images[0], mime

    structured = out.get("structuredContent") or {}
    b64 = structured.get("screenshot_png_b64") or structured.get("png_b64")
    if b64:
        mime = (
            structured.get("screenshot_mime_type")
            or structured.get("mime_type")
            or None
        )
        return b64, mime

    return None, None


def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer, rejecting booleans and malformed values."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _is_placeholder_id(value: Any) -> bool:
    """True when *value* is a schema-filler id rather than a real target.

    Several providers emit every declared schema property on every tool call,
    filling unused optional integers with ``0``. A non-positive id cannot name
    a window, so treating it as a targeting request drops the caller's ``app=``
    and fails the capture. Malformed non-numeric values are deliberately NOT
    placeholders: those still reach the existing validation error rather than
    being silently ignored.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return False
    try:
        return int(value) <= 0
    except ValueError:
        return False


def _ingest_windows(raw_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise cua-driver ``list_windows`` entries, dropping unusable ones.

    Every downstream operation needs both an integer ``pid`` (for
    get_window_state / action tools) and ``window_id`` (for screenshot /
    element clicks), so a window missing either is uncapturable.

    Crucially, on X11 a window's PID comes from the *optional*
    ``_NET_WM_PID`` property — the desktop root, panels, and
    override-redirect popups routinely omit it, so the driver reports
    ``pid: null`` for them. Coercing every entry unconditionally
    (``int(w["pid"])``) let one such window abort enumeration of the real,
    targetable windows. We skip the unusable entries instead so capture()
    and focus_app() still find the windows that matter.

    ``z_index`` follows CUA Driver semantics: higher = closer to front.
    Wayland may return ``z_index: null`` (undefined stacking order); we
    treat null as the lowest priority so real windows still sort above
    desktop/root windows, and the backmost never ends up selected as the
    capture target.
    """
    windows: List[Dict[str, Any]] = []
    for w in raw_windows:
        # Compatibility envelopes are untrusted input: skip non-dict members
        # instead of raising AttributeError on one malformed record.
        if not isinstance(w, dict):
            continue
        pid_int = _positive_int(w.get("pid"))
        window_id_int = _positive_int(w.get("window_id"))
        if pid_int is None or window_id_int is None:
            continue
        z_raw = w.get("z_index")
        z_index = z_raw if isinstance(z_raw, (int, float)) and not isinstance(z_raw, bool) else 0
        app_name = w.get("app_name", "")
        title = w.get("title", "")
        windows.append({
            "app_name": app_name if isinstance(app_name, str) else "",
            "pid": pid_int,
            "window_id": window_id_int,
            # cua-driver 0.6.x on Linux may return JSON null here.
            # Only explicit False means off-screen; null means unknown.
            "off_screen": w.get("is_on_screen") is False,
            "title": title if isinstance(title, str) else "",
            "z_index": z_index,
        })
    return windows


def _windows_from_tool_result(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list_windows payloads across cua-driver result shapes."""
    structured = out.get("structuredContent")
    if isinstance(structured, dict):
        windows = structured.get("windows")
        if isinstance(windows, list) and windows:
            return windows

    data = out.get("data")
    if isinstance(data, dict):
        windows = data.get("windows")
        if isinstance(windows, list) and windows:
            return windows
        legacy_windows = data.get("_legacy_windows")
        if isinstance(legacy_windows, list) and legacy_windows:
            return legacy_windows

    windows = out.get("windows")
    if isinstance(windows, list) and windows:
        return windows
    legacy_windows = out.get("_legacy_windows")
    if isinstance(legacy_windows, list) and legacy_windows:
        return legacy_windows
    return []


def _apps_from_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    apps: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for summary in _ingest_windows(windows):
        name = summary["app_name"]
        if not name:
            continue
        key = (name, summary["pid"])
        if key in seen:
            continue
        seen.add(key)
        apps.append({"name": name, "pid": summary["pid"]})
    return apps


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP."""

    def __init__(self, permission_mode: str = "standard") -> None:
        if permission_mode not in {"standard", "bounded", "unrestricted"}:
            raise ValueError(f"unsupported cua-driver permission mode: {permission_mode}")
        self.permission_mode = permission_mode
        self._embedded_daemon: Optional[_EmbeddedCuaDaemon] = None
        if permission_mode != "standard":
            # Manifest: mandatory for bounded (the daemon validates it), optional for unrestricted where it still
            # caps what an approval-bypassed run may touch.
            raw = _computer_use_cfg().get("capability_manifest")
            self._embedded_daemon = _EmbeddedCuaDaemon(
                resolve_cua_driver_cmd() or "", permission_mode,
                capability_manifest=raw.strip() if isinstance(raw, str) and raw.strip() else None)
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge, self._embedded_daemon)
        # Sticky target (set by capture()/focus_app(), used by actions): `_active_pid`, `_active_window_id`, `_last_app`,
        # `_last_target` (exact identity for capture_after — Linux app names may be generic, e.g. several unrelated Qt
        # windows all say Qt6Application), `_snapshot_tokens` (element_index -> element_token, attached to actions so
        # cua-driver reports "stale" instead of silently re-resolving).
        self._clear_active_target()
        # Public session label (one per Hermes run) sent as `session` on every call: owns the cursor color and
        # gives config/recording state a stable owner across transport restarts. Part of the 0.20 runtime contract.
        self._session_id: str = f"hermes-{uuid.uuid4().hex[:12]}"
        self._session.set_transport_reset_callback(self._handle_transport_reset)

    def _handle_transport_reset(self) -> None:
        """Invalidate every capability minted by the replaced transport."""
        self._clear_active_target()

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        contract = cua_driver_runtime_contract_status()
        if not contract.get("ready"):
            contract = _maybe_repair_runtime_contract(contract)
        if not contract.get("ready"):
            raise RuntimeError(f"cua-driver is not ready: {contract.get('reason') or 'runtime contract is incomplete'}. "
                               + ("Update the binary selected by HERMES_CUA_DRIVER_CMD or remove that override."
                                  if os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip() else "Run `hermes computer-use install` to repair it."))
        _maybe_nudge_update()
        # `mcp` is an optional extra: lazy-install on first use (gated by `security.allow_lazy_installs`); failure
        # raises FeatureUnavailable with the exact `uv pip install` hint.
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.computer_use", prompt=False)
        importlib.invalidate_caches()  # a just-installed package may not be importable yet
        with contextlib.ExitStack() as rollback:  # a failed start stops the private daemon, then re-raises
            if self._embedded_daemon is not None:
                rollback.callback(self._embedded_daemon.stop) and self._embedded_daemon.start()
            self._session.start()
            rollback.pop_all()
        # Declare this run's identity. Non-fatal: cua-driver accepts anonymous calls (cursor won't render), so degrade.
        self._best_effort("start_session failed (continuing anonymous)",
                          self._session.call_tool, "start_session", {"session": self._session_id})
        # Post-handshake tuning guards on `_started`: before the handshake flips it, call_tool would re-enter
        # session.start() (stubbed start() recurses).
        if self._session._started:
            max_dim = _computer_use_max_image_dimension()
            if max_dim:  # smaller screenshots cost less over the daemon socket and per turn
                self._best_effort("set_config(max_image_dimension) failed",
                                  self.set_config, max_image_dimension=max_dim)
            if _cua_no_overlay():  # belt-and-suspenders when --no-overlay is unsupported or ignored
                self._best_effort("set_agent_cursor_enabled failed",
                                  self.set_agent_cursor_enabled, False, cursor_id=self._session_id)

    def stop(self) -> None:
        # Best-effort end_session so the driver cleans per-session state (cursor overlay, recording ownership,
        # config overrides); the connection drop below releases daemon-side state regardless.
        if self._session._started:
            self._best_effort("end_session failed (continuing teardown)",
                              self._session.call_tool, "end_session", {"session": self._session_id})
        with contextlib.ExitStack() as teardown:  # every step runs even if one raised (LIFO: session, bridge, daemon)
            self._embedded_daemon is None or teardown.callback(self._embedded_daemon.stop)
            teardown.callback(self._bridge.stop)
            teardown.callback(self._session.stop)

    @staticmethod
    def _best_effort(what: str, fn, *args: Any, **kwargs: Any) -> None:
        """Run a non-fatal driver call, logging (debug) instead of raising."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.debug("cua-driver %s: %s", what, e)

    def is_available(self) -> bool:
        return sys.platform in ("darwin", "win32", "linux") and cua_driver_binary_available()  # other Unix-likes untested E2E

    def _clear_active_target(self) -> None:
        """Forget a capture/focus target so a failed lookup cannot misroute input."""
        self._active_pid = self._active_window_id = self._last_app = self._last_target = None
        # Surface 6 of NousResearch/hermes-agent#47072: per-snapshot `element_index -> element_token` map
        # populated on capture(). Action tools (click/scroll/set_value/...) attach the matching token
        # alongside `element_index` so cua-driver detects "stale" explicitly instead of silently
        # re-resolving to a different element. Cleared whenever a fresh capture overwrites the snapshot
        # context.
        self._snapshot_tokens: Dict[int, str] = {}

    def _failed_capture(self, mode: str, message: str = "") -> CaptureResult:
        """Return an empty capture after disarming any prior target context."""
        self._clear_active_target()
        return CaptureResult(
            mode=mode,
            width=0,
            height=0,
            png_b64=None,
            elements=[],
            app="",
            window_title=message,
            png_bytes_len=0,
        )

    def _call_capture_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a capture-stage tool and disarm state on transport or logical failure."""
        try:
            out = self._session.call_tool(name, args)
        except Exception:
            self._clear_active_target()
            raise
        if out.get("isError") is True:
            message = out.get("data")
            self._clear_active_target()
            raise RuntimeError(
                f"cua-driver {name} failed"
                + (f": {message}" if isinstance(message, str) and message else "")
            )
        return out

    def _load_windows(self) -> List[Dict[str, Any]]:
        """Load normalized visible windows, with the shared CLI recovery path.

        Windows are sorted by ``z_index`` **descending**: CUA Driver
        defines higher values as closer to the front, so the frontmost
        window ends up at index 0 — which is what ``capture()`` and
        ``focus_app()`` pick as the default target.  ``_ingest_windows``
        already normalised null ``z_index`` (Wayland) to 0, so those
        windows sort to the back.
        """
        out = self._call_capture_tool(
            "list_windows",
            {"on_screen_only": True, "session": self._session_id},
        )
        windows = _ingest_windows(_windows_from_tool_result(out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        if windows:
            return windows

        logger.warning(
            "cua-driver list_windows returned no windows over MCP; "
            "re-fetching via CLI transport",
        )
        try:
            cli_out = self._session._call_tool_via_cli(
                "list_windows",
                {"on_screen_only": True, "session": self._session_id},
                20.0,
            )
        except Exception as exc:
            logger.error("cua-driver CLI re-fetch for list_windows failed: %s", exc)
            return []
        if cli_out.get("isError") is True:
            logger.error("cua-driver CLI re-fetch for list_windows returned an error")
            self._clear_active_target()
            return []
        windows = _ingest_windows(_windows_from_tool_result(cli_out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        return windows

    def _match_windows_for_app(
        self, windows: List[Dict[str, Any]], app: str
    ) -> List[Dict[str, Any]]:
        """Resolve ``app=`` through exact names before convenience substrings.

        Linux ``list_windows`` can omit an app name while ``list_apps`` retains
        name/bundle-ID metadata. Exact direct names and exact metadata aliases
        must win over substring matches: querying ``Code`` must not silently
        select ``Visual Studio Code`` merely because it is frontmost.
        """
        app_lower = app.strip().lower()
        if not app_lower:
            return []

        direct_exact = [
            w for w in windows
            if app_lower == str(w.get("app_name", "")).strip().lower()
        ]
        if direct_exact:
            return direct_exact

        try:
            running_apps = self.list_apps()
        except Exception as exc:
            # A title can still be the only usable identity on X11 when app
            # enumeration is unavailable, so retain the constrained title
            # fallback below instead of treating this as a hard no-match.
            logger.debug("computer_use list_apps fallback failed for %r: %s", app, exc)
            running_apps = []

        exact_pids: set[int] = set()
        partial_pids: set[int] = set()
        for raw_app in running_apps:
            if not isinstance(raw_app, dict) or raw_app.get("running") is False:
                continue
            raw_pid = raw_app.get("pid")
            if isinstance(raw_pid, bool) or not isinstance(raw_pid, (int, str)):
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid <= 0:
                continue

            aliases = {
                value.strip().lower()
                for key in ("bundle_id", "bundleId", "name", "app_name", "display_name")
                if isinstance((value := raw_app.get(key)), str) and value.strip()
            }
            if app_lower in aliases:
                exact_pids.add(pid)
            elif any(app_lower in alias for alias in aliases):
                partial_pids.add(pid)

        metadata_exact = [w for w in windows if w.get("pid") in exact_pids]
        if metadata_exact:
            return metadata_exact

        direct_partial = [
            w for w in windows
            if app_lower in str(w.get("app_name", "")).lower()
        ]
        if direct_partial:
            return direct_partial

        metadata_partial = [w for w in windows if w.get("pid") in partial_pids]
        if metadata_partial:
            return metadata_partial

        # Some X11 backends expose a title but no app name. Restrict this final
        # fallback to nameless rows so a localized app name is not overridden
        # merely because its title happens to be in the caller's language.
        return [
            w for w in windows
            if not str(w.get("app_name", "")).strip()
            and app_lower in str(w.get("title", "")).lower()
        ]

    def _capture_full_screen(self, mode: str) -> CaptureResult:
        """Capture the whole displayed screen via cua-driver's desktop lane.

        Uses `get_desktop_state` — a composited grab of everything currently
        on screen (like PrtScn) — instead of resolving a single window through
        `list_windows`. This is what "screenshot my screen" means: previously
        the `screen` sentinel resolved to the OS shell window (Progman /
        WorkerW on Windows), which is the wallpaper + icons layer and never
        shows the windows stacked above it.

        Bonus resilience (2ndNatureAI, #60081): this lane works even when
        Windows UIA enumeration (`list_windows` / `list_apps`) hangs
        (trycua/cua#2110/#2113), because it never enumerates.

        Returns pixels only — a composited image has no single accessibility
        tree, so `elements` is always empty regardless of requested mode. The
        result carries a `note` telling the model how to reach the
        interactive lanes.
        """
        self._clear_active_target()
        previous_scope: Optional[str] = None
        try:
            cfg = self._session.call_tool(
                "get_config", {"session": self._session_id}, timeout=10.0,
            )
            sc = cfg.get("structuredContent") or {}
            if isinstance(sc, dict):
                val = sc.get("capture_scope")
                if isinstance(val, str):
                    previous_scope = val
        except Exception as e:
            logger.debug("cua-driver get_config before full-screen capture failed: %s", e)

        try:
            if previous_scope != "desktop":
                self._session.call_tool(
                    "set_config",
                    {"key": "capture_scope", "value": "desktop",
                     "session": self._session_id},
                    timeout=10.0,
                )
            out = self._call_capture_tool(
                "get_desktop_state", {"session": self._session_id},
            )
        finally:
            if previous_scope and previous_scope != "desktop":
                try:
                    self._session.call_tool(
                        "set_config",
                        {"key": "capture_scope", "value": previous_scope,
                         "session": self._session_id},
                        timeout=10.0,
                    )
                except Exception as e:
                    logger.debug("cua-driver restore capture_scope failed: %s", e)

        png_b64, image_mime_type = _image_from_tool_result(out)
        if not png_b64:
            return self._failed_capture(
                mode,
                "<get_desktop_state returned no image; the driver may "
                "predate the desktop capture lane — try "
                "capture(app='<AppName>') for a specific window>",
            )
        structured = out.get("structuredContent") or {}
        width = int(structured.get("screenshot_width")
                    or structured.get("screen_width") or 0)
        height = int(structured.get("screenshot_height")
                     or structured.get("screen_height") or 0)
        png_bytes_len = 0
        try:
            raw = base64.b64decode(png_b64, validate=False)
            png_bytes_len = len(raw)
            detected_width, detected_height = _image_dimensions_from_bytes(raw)
            if detected_width and detected_height:
                width = detected_width
                height = detected_height
        except Exception:
            png_bytes_len = len(png_b64) * 3 // 4
        return CaptureResult(
            mode="vision",
            width=width,
            height=height,
            png_b64=png_b64,
            elements=[],
            app="screen",
            window_title="Full screen (composited)",
            png_bytes_len=png_bytes_len,
            image_mime_type=image_mime_type,
            note=(
                "full-screen capture has no interactable elements; to act on "
                "what you see, call capture(app='<AppName>') for that app's "
                "clickable element list, or capture(app='desktop') for the "
                "desktop shell (wallpaper icons / taskbar) with elements"
            ),
        )

    # ── Capture ────────────────────────────────────────────────────
    def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> CaptureResult:
        """Capture the frontmost on-screen window or an exact known target.

        Maps hermes `capture(mode, app)` → cua-driver `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision).
        """
        # Step 1: enumerate on-screen windows to find target pid/window_id.
        # Surface 3 of NousResearch/hermes-agent#47072: read the canonical
        # `structuredContent.windows` array directly. Pre-fix the wrapper
        # also kept a text-line regex (`_WINDOW_LINE_RE`) as a fallback for
        # cua-driver builds that predated structuredContent; the supersede
        # PR's effective minimum (trycua/cua#1961 + #1908) is well past
        # that, so the fallback is gone — the wrapper now treats the
        # structured shape as the only contract.
        # Drop schema-filler ids before they can be read as a targeting
        # request, so `capture(app=...)` and frontmost capture still work for
        # models that emit every optional property zero-filled.
        if _is_placeholder_id(pid):
            pid = None
        if _is_placeholder_id(window_id):
            window_id = None
        # Step 0: explicit full-screen capture — a composited grab of
        # everything displayed, via get_desktop_state. Bypasses window
        # enumeration entirely (also keeps screenshots working when Windows
        # UIA enumeration hangs — trycua/cua#2110/#2113, #60081).
        # app='desktop' intentionally does NOT take this lane: it resolves to
        # the shell/desktop window below so desktop icons stay clickable.
        if (
            pid is None
            and window_id is None
            and app
            and app.strip().lower() in _FULL_SCREEN_SENTINELS
        ):
            return self._capture_full_screen(mode)
        # An exact pid/window pair is both the stable capture_after target and
        # the escape hatch when app/window discovery is unavailable on X11.
        if pid is not None or window_id is not None:
            if pid is None or window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires both pid and window_id>",
                )
            target_pid = _positive_int(pid)
            target_window_id = _positive_int(window_id)
            if target_pid is None or target_window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires positive integer pid and window_id>",
                )
            windows = [{
                "app_name": app or "",
                "pid": target_pid,
                "window_id": target_window_id,
                "off_screen": False,
                "title": "",
                "z_index": 0,
            }]
        else:
            try:
                windows = self._load_windows()
            except Exception:
                self._clear_active_target()
                raise
            if not windows:
                # Diagnose instead of returning a bare 0x0: the dominant
                # real-world cause on Linux is a locked desktop session.
                return self._failed_capture(mode, _empty_discovery_reason())

        # Filter by app name (case-insensitive substring) if requested.
        # When the filter matches nothing, surface that explicitly instead of
        # silently capturing the frontmost window — on macOS the `app_name`
        # returned by list_windows is the localized name (e.g. "計算機"), so
        # `app="Calculator"` legitimately matches no windows on a non-English
        # system and the caller needs to retry with the localized name.
        if pid is None and window_id is None and app and app.strip().lower() in _DESKTOP_SHELL_SENTINELS:
            # Desktop-shell request (app='desktop'): resolve to the OS
            # shell/desktop window (the desktop backdrop or the
            # taskbar/menu-bar) via list_windows. Unlike the full-screen lane
            # above, this carries the shell's interactable elements (desktop
            # icons), so "click the taskbar" / "open the recycle bin" work.
            def _is_desktop_window(w: Dict[str, Any]) -> bool:
                haystack = f"{w.get('app_name', '')} {w.get('title', '')}".lower()
                return any(name in haystack for name in _DESKTOP_WINDOW_NAMES)

            desktop = [w for w in windows if _is_desktop_window(w)]
            if not desktop:
                return self._failed_capture(
                    mode,
                    (
                        f"<no desktop/shell window found for app={app!r}; "
                        f"cua-driver captures one window at a time and exposes "
                        f"no whole-virtual-desktop or per-monitor capture. "
                        f"Call list_apps / capture(app='<AppName>') to target a "
                        f"specific window instead. On Windows the taskbar is "
                        f"'Shell_TrayWnd' and the desktop is 'Progman'.>"
                    ),
                )
            # Prefer the desktop backdrop (Progman/WorkerW/Finder) over the
            # taskbar when both are present, so a bare "screen" capture shows
            # the full desktop rather than just the task strip.
            windows = sorted(
                desktop,
                key=lambda w: 0 if any(
                    n in f"{w.get('app_name', '')} {w.get('title', '')}".lower()
                    for n in ("progman", "workerw", "program manager", "finder", "desktop")
                ) else 1,
            )
        elif pid is None and window_id is None and app:
            filtered = self._match_windows_for_app(windows, app)
            if not filtered:
                return self._failed_capture(
                    mode,
                    (
                        f"<no on-screen window matched app={app!r}; "
                        f"call list_apps to see available app names or bundle IDs "
                        f"(macOS reports localized names, e.g. '計算機' "
                        f"instead of 'Calculator'; some Linux/Qt apps only "
                        f"resolve via list_apps metadata)>"
                    ),
                )
            windows = filtered

        # Pick first on-screen window (sorted by z_index / z-order above).
        # On Linux, unqualified default captures skip desktop/shell helper
        # windows and, with tied/unknown z_index, may additionally consult
        # _NET_ACTIVE_WINDOW (#58026).
        target = _select_capture_target(
            windows,
            app_requested=bool(app),
            exact_target=pid is not None or window_id is not None,
        )
        self._active_pid = target["pid"]
        self._active_window_id = target["window_id"]
        self._snapshot_tokens = {}  # prior snapshot's tokens: disarm before any capture so an exception can't pair them
        self._last_target = {"pid": self._active_pid, "window_id": self._active_window_id}

    def launch_app(self, *, bundle_id: Optional[str] = None, name: Optional[str] = None,
                   urls: Optional[List[str]] = None, additional_arguments: Optional[List[str]] = None,
                   creates_new_application_instance: bool = False) -> Dict[str, Any]:
        """Idempotent launch returning ``{pid, bundle_id, name, windows[]}``. ``creates_new_application_instance=True``
        forces a fresh instance so concurrent runs touching the same app get isolated windows."""
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        args: Dict[str, Any] = {"session": self._session_id, **{k: v for k, v in (
            ("bundle_id", bundle_id), ("name", name), ("urls", urls and list(urls)),
            ("additional_arguments", additional_arguments and list(additional_arguments)),
            ("creates_new_application_instance", creates_new_application_instance or None)) if v}}
        out = self._session.call_tool("launch_app", args)
        return out["structuredContent"] or {"data": out["data"]}

    def bring_to_front(self, *, pid: int, window_id: Optional[int] = None) -> ActionResult:
        """Activate a window so subsequent foreground-dispatched input lands on it."""
        args: Dict[str, Any] = {"pid": int(pid), **({} if window_id is None else {"window_id": int(window_id)})}
        # Strict live schema with no session property: a standalone native focus op, not a session-scoped input action.
        return self._action("bring_to_front", args, inject_session=False)

    # ── Pointer + display introspection ─────────────────────────────

    def move_cursor(self, x: int, y: int) -> ActionResult:
        """Move the agent-cursor *overlay* to a screen point. This is a
        visual hint — it does NOT move the real OS pointer (cua-driver
        explicitly avoids stealing pointer focus). The overlay glides
        smoothly to the target, so consumers use it before a click to
        give a visible "where the agent is going" cue."""
        return self._action("move_cursor", {"x": int(x), "y": int(y)})

    def get_cursor_position(self) -> Tuple[int, int]:
        """Return the *real* OS cursor position in screen points
        (origin top-left)."""
        out = self._session.call_tool(
            "get_cursor_position", {"session": self._session_id}
        )
        sc = out.get("structuredContent") or {}
        return int(sc.get("x", 0)), int(sc.get("y", 0))

    def get_screen_size(self) -> Dict[str, Any]:
        """Return the logical size of the main display in points plus
        its backing scale factor. Shape:
        ``{width, height, backing_scale_factor}``."""
        out = self._session.call_tool(
            "get_screen_size", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    def zoom(self, *, window_id: int, x: float, y: float, w: float, h: float,
             factor: float = 1.0, format: str = "jpeg",
             quality: int = 85) -> Dict[str, Any]:
        """Return a JPEG / PNG of a sub-region of a window, optionally
        scaled. cua-driver supports zoom-to-rect for callers that need
        a higher-resolution view of a specific element."""
        return self._session.call_tool("zoom", {
            "window_id": int(window_id),
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "factor": float(factor),
            "format": format, "quality": int(quality),
            "session": self._session_id,
        })

    # ── Agent cursor (overlay) ──────────────────────────────────────
    #
    # Sessions (start_session/end_session, wired in start/stop) own the
    # cursor. These knobs tune its appearance + behavior per-session.
    # All accept an optional `cursor_id` to address a specific cursor
    # when the run drives multiple (rare); the default is this run's
    # session id.

    def set_agent_cursor_enabled(self, enabled: bool, *,
                                 cursor_id: Optional[str] = None) -> ActionResult:
        """Toggle the agent cursor overlay's visibility for this run."""
        return self._action("set_agent_cursor_enabled",
                            {"enabled": bool(enabled), **({"cursor_id": cursor_id} if cursor_id else {})})

    def set_config(self, **config) -> ActionResult:
        """Set cua-driver config keys (e.g. ``max_image_dimension``); unknown keys pass through — cua-driver validates."""
        return self._action("set_config", dict(config))

    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None, *, timeout: float = 30.0) -> Dict[str, Any]:
        """Generic escape hatch: call any cua-driver MCP tool by name. ``session`` is injected via setdefault, so
        this is the supported path for tools the wrapper does not type-wrap (preferred over ``self._session.call_tool``)."""
        payload = dict(args) if args else {}
        payload.setdefault("session", self._session_id)
        return self._session.call_tool(name, payload, timeout=timeout)

    def _action(self, name: str, args: Dict[str, Any], *, inject_session: bool = True) -> ActionResult:
        # Attach the snapshot's `element_token` to an `element_index` call so a superseded snapshot yields an explicit
        # 'stale' error. Two ways to establish support, the live input schema first: cua-driver 0.21+ stopped
        # publishing per-tool `capabilities[]` while still accepting `element_token` in its schema, and it REFUSES a
        # bare `element_index` (`snapshot_id_required`) — gating on the capability alone broke EVERY element click and
        # left only pixel clicks working. The capability check stays so older drivers that shipped the vocabulary keep
        # working; drivers advertising neither (`additionalProperties: false`) must never see the property.
        idx = args.get("element_index")
        token = self._snapshot_tokens.get(idx) if isinstance(idx, int) else None
        if token and (self._session.supports_input_property(name, "element_token")
                      or self._session.supports_capability("accessibility.element_tokens", tool=name)):
            args["element_token"] = token
        if inject_session:  # setdefault preserves any explicit session a caller already supplied
            args.setdefault("session", self._session_id)
        try:
            out = self._session.call_tool(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        data = out["data"]
        structured = out.get("structuredContent") or {}
        message = (str(data.get("message", "")) if isinstance(data, dict) else data if isinstance(data, str) else "") \
            or (str(structured.get("message", "")) if isinstance(structured, dict) else "")
        # Merge data + structuredContent into meta, structured winning on overlap (canonical verdict surface).
        meta = {k: v for part in (data, structured) if isinstance(part, dict) for k, v in part.items()}
        return _action_result_from(name, not out["isError"], message, meta, structured,
                                   requested_delivery=args.get("delivery_mode"))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import PureWindowsPath  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import base64  # noqa: F401,E402
import concurrent.futures  # noqa: F401,E402
from collections import deque  # noqa: F401,E402
import functools  # noqa: F401,E402
import json  # noqa: F401,E402
import re  # noqa: F401,E402
import shutil  # noqa: F401,E402
import tempfile  # noqa: F401,E402
import time  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'CaptureResult': ('tools.computer_use.backend', 'CaptureResult'),
    'UIElement': ('tools.computer_use.backend', 'UIElement'),
    'cua_driver_install_hint': ('tools.computer_use.cua_backend_driver', 'cua_driver_install_hint'),
    'cua_driver_update_check': ('tools.computer_use.cua_backend_driver', 'cua_driver_update_check'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
