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
from tools.computer_use.backend import ActionResult, ComputerUseBackend
from tools.computer_use.cua_backend_capture import _CaptureMixin
from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
from tools.computer_use.cua_backend_driver import (
    _CUA_DRIVER_CMD_ENV, cua_driver_binary_available, cua_driver_runtime_contract_status, cua_driver_update_nudge,
    resolve_cua_driver_cmd)
from tools.computer_use.cua_backend_input import _InputMixin
from tools.computer_use.cua_backend_parse import _action_result_from
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession

logger = logging.getLogger(__name__)
# cua-driver's anonymous PostHog telemetry gate ("0" disables; absent => ON upstream).
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

    Explicit ``True`` / ``False`` overrides auto-detection. See #28152, #47032.
    """
    val = _computer_use_cfg().get("no_overlay")
    if val is not None or sys.platform != "linux":
        return bool(val) if val is not None else sys.platform == "darwin"
    wsl = False
    with contextlib.suppress(Exception), open("/proc/version", encoding="utf-8") as f:
        wsl = "microsoft" in f.read().lower()
    return wsl or not os.environ.get("DISPLAY") or (
        # Linux/X11: the cursor overlay is a fullscreen, always-on-top, all-workspaces X11 window
        # (save-unders path). An unclean session end (agent interrupted mid-capture, stale target window)
        # can leave it stuck above every app on every workspace, wedging desktop input until the app
        # restarts — the same failure class as the HUD window on Mutter/X11 (#83473). There is no
        # compositor-owned surface to tear down with the client connection, so default the overlay off on
        # X11 too; set computer_use.no_overlay: false to keep the cursor. Wayland keeps it: the compositor
        # owns the overlay surface lifecycle there.
        os.environ.get("XDG_SESSION_TYPE") != "wayland" and not os.environ.get("WAYLAND_DISPLAY"))

def _cua_telemetry_disabled() -> bool:
    """True unless ``computer_use.cua_telemetry`` opts in (unreadable config fails SAFE toward disabling)."""
    return not bool(_computer_use_cfg().get("cua_telemetry", False))

def _cua_configured_permission_mode() -> str:
    """``computer_use.permission_mode``: ``standard`` (default) or ``bounded``; unknown values fall closed to
    ``standard``. ``unrestricted`` is deliberately NOT a config value — it stays tied to the per-session YOLO
    toggle so a stale config line can never silently bypass approvals."""
    raw = str(_computer_use_cfg().get("permission_mode", "standard") or "").strip().lower()
    return "bounded" if raw == "bounded" else "standard"

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

def _empty_discovery_reason(remote: bool = False) -> str:
    """One-line diagnosis for 'window discovery found nothing'."""
    if remote:
        # Remote transport: the local session is irrelevant — windows come from the bridge host.
        return ("remote desktop returned no windows — check the host bridge connection and the "
                "remote desktop session state")
    if _linux_session_locked() is True:
        return ("the desktop session is LOCKED (loginctl LockedHint=yes) — unlock the screen; "
                "a locked compositor hides windows and freezes app renderers")
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return "no DISPLAY is set — X11/XWayland is not reachable from this process"
    if sys.platform == "darwin":  # headless Mac / asleep panel: ScreenCaptureKit has 0 shareable displays while TCC looks fine
        return ("window discovery returned no windows; on macOS this usually means no shareable display (headless Mac or "
                "panel asleep) — wake the display or attach a monitor/HDMI dummy, then run `hermes computer-use doctor`")
    return "window discovery returned no windows; run `hermes computer-use doctor` (display reachability, AX capability)"

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


class CuaDriverBackend(_CaptureMixin, _InputMixin, ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP."""

    def __init__(self, permission_mode: str = "standard", remote_config: Optional[Any] = None) -> None:
        if permission_mode not in {"standard", "bounded", "unrestricted"}:
            raise ValueError(f"unsupported cua-driver permission mode: {permission_mode}")
        # Remote transport selection is explicit (provider registry contract): the backend
        # receives the resolved remote config from whoever built it — today the remote
        # provider (tools/computer_use/remote_provider.py), tomorrow any plugin. Self-resolving
        # from config here would make ``computer_use.provider: local`` click a remote desktop
        # whenever ``computer_use.remote`` happens to be present in config.yaml, which is the
        # silent-selection failure the registry exists to prevent.
        self._remote_config = remote_config
        self.permission_mode = permission_mode
        self._embedded_daemon: Optional[_EmbeddedCuaDaemon] = None
        if permission_mode != "standard" and self._remote_config is None:
            # Manifest: mandatory for bounded (the daemon validates it), optional for unrestricted where it still
            # caps what an approval-bypassed run may touch. Skip the embedded daemon when connecting remotely —
            # the remote host owns the driver process and its permission mode.
            raw = _computer_use_cfg().get("capability_manifest")
            self._embedded_daemon = _EmbeddedCuaDaemon(
                resolve_cua_driver_cmd() or "", permission_mode,
                capability_manifest=raw.strip() if isinstance(raw, str) and raw.strip() else None)
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(
            self._bridge, self._embedded_daemon,
            remote_config=self._remote_config)
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

    def start(self) -> None:
        # Remote transport: skip the local driver contract check — the remote host owns the driver.
        if self._remote_config is None:
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
        # Remote transport: the bridge host owns the driver, so the local binary is irrelevant.
        if self._remote_config is not None:
            return True
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

    def _set_active_target(self, target: Dict[str, Any]) -> None:
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

    def set_agent_cursor_enabled(self, enabled: bool, *, cursor_id: Optional[str] = None) -> ActionResult:
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
        # 'stale' error. Gated on the per-tool capability: older drivers (`additionalProperties: false`) must never see it.
        idx = args.get("element_index")
        token = self._snapshot_tokens.get(idx) if isinstance(idx, int) else None
        if token and self._session.supports_capability("accessibility.element_tokens", tool=name):
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
