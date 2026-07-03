"""Cua-driver backend (macOS, Windows, Linux).

Speaks MCP over stdio to `cua-driver`. The Python `mcp` SDK is async, so we
run a dedicated asyncio event loop on a background thread and marshal sync
calls through it.

The same `cua-driver call <tool>` surface (click, type_text, hotkey, drag,
scroll, screenshot, launch_app, list_apps, list_windows, get_window_state,
move_cursor, wait) works identically across macOS, Windows, and Linux —
cua-driver's PARITY matrix marks the action tools VERIFIED on macOS and
Windows in the cross-platform Rust port (`cua-driver-rs`).

Linux is the most recent runtime (X11 today, Wayland via XWayland; pure-
Wayland progress tracked upstream). It is enabled in
`check_computer_use_requirements` alongside macOS and Windows. The plumbing
in this file is OS-agnostic; per-host gaps (no DISPLAY, missing AT-SPI,
etc.) surface as specific blocked checks via `hermes computer-use doctor`
rather than failing silently.

Install:
  - **macOS**:
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"
  - **Windows** (PowerShell):
      irm https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1 | iex

After install, `cua-driver` is on $PATH and supports `cua-driver mcp` (stdio
transport) which is what we invoke.

The macOS path uses private SkyLight SPIs (SLEventPostToPid,
SLPSPostEventRecordTo, _AXObserverAddNotificationAndCheckRemote) that aren't
Apple-public and can break on OS updates. The Windows path in cua-driver-rs
uses stable Win32 APIs (SendInput + UI Automation) — not subject to the
same SPI breakage class.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


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

_CUA_DRIVER_CMD = os.environ.get("HERMES_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP transport (fallback when the
                            # driver doesn't expose `manifest` — see
                            # `_resolve_mcp_invocation` below)

# Whole-screen / desktop capture. cua-driver is a window-oriented driver —
# its `get_window_state` / `screenshot` tools capture a single window (by
# pid + window_id), and there is no MCP tool that captures the entire virtual
# desktop or an arbitrary monitor as one image. But the OS shell surfaces
# themselves (the desktop backdrop and the taskbar/menu-bar) are real windows
# that show up in `list_windows`, so "show me my screen" / "click the taskbar"
# is reachable by targeting those windows. When `app` is one of these
# sentinels, capture() resolves to the desktop/shell window instead of an
# application window.
_SCREEN_CAPTURE_SENTINELS = {"screen", "desktop", "fullscreen", "full screen", "all"}

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


# Env var cua-driver reads to gate its anonymous usage telemetry (PostHog).
# Setting it to "0" disables telemetry; absence => the binary's own default
# (telemetry ON upstream).
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"


def _cua_telemetry_disabled() -> bool:
    """True when Hermes should disable cua-driver telemetry for this user.

    Reads ``computer_use.cua_telemetry`` from config.yaml. Default is False
    (telemetry off). Any failure to read config fails SAFE — toward the
    privacy-preserving default of telemetry disabled.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cu = cfg.get("computer_use") or {}
        # opt-in flag: True => user wants telemetry => do NOT disable.
        return not bool(cu.get("cua_telemetry", False))
    except Exception:
        # Config unreadable — default to disabling telemetry (fail safe).
        return True


def cua_driver_child_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return the environment dict for spawning cua-driver.

    Starts from ``base_env`` (defaults to ``os.environ``) and, when telemetry
    is disabled (the default), injects ``CUA_DRIVER_RS_TELEMETRY_ENABLED=0``.
    When the user has opted in, the var is left untouched so cua-driver uses
    its own default. Used by every cua-driver spawn site (MCP backend, status,
    doctor, install) so the policy is applied consistently.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if _cua_telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    return env


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
    """
    try:
        from tools.environments.local import _sanitize_subprocess_env
        proc = subprocess.run(
            [driver_cmd, "manifest"],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
            # cua-driver is a third-party binary — never hand it provider
            # API keys via inherited env (same policy as the MCP and CLI
            # fallback spawns below; #53503/#55709/#58889 lineage).
            env=_sanitize_subprocess_env(cua_driver_child_env()),
        )
    except Exception:
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out:
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    try:
        manifest = json.loads(out)
    except (ValueError, TypeError):
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    if not isinstance(manifest, dict):
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    invocation = manifest.get("mcp_invocation")
    if not isinstance(invocation, dict):
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    args = invocation.get("args")
    command = invocation.get("command")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return driver_cmd, list(_CUA_DRIVER_ARGS)
    if not isinstance(command, str) or not command:
        # The driver knows the subcommand but didn't surface its own path.
        # Keep our resolved driver_cmd; the args are still authoritative.
        return driver_cmd, args
    return command, args

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


def cua_driver_binary_available() -> bool:
    """True if `cua-driver` is on $PATH or HERMES_CUA_DRIVER_CMD resolves."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


def cua_driver_update_check(*, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """Run ``cua-driver check-update --json`` and return its parsed state.

    The payload mirrors the ``check_for_update`` MCP tool:
    ``{current_version, latest_version, update_available, ...}``.

    Returns ``None`` (callers should stay quiet) when the result is
    indeterminate: the binary is missing, the driver is too old to support
    the verb (it predates trycua/cua#1734), the GitHub check failed (an
    ``error`` field is set), or the output didn't parse. Best-effort; never
    raises.
    """
    try:
        from tools.environments.local import _sanitize_subprocess_env
        proc = subprocess.run(
            [_CUA_DRIVER_CMD, "check-update", "--json"],
            capture_output=True, text=True, timeout=timeout,
            # Some older drivers don't have the verb and fall through to a
            # stdin-reading mode rather than erroring — DEVNULL gives them EOF
            # so they exit fast instead of blocking until the timeout.
            stdin=subprocess.DEVNULL,
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


def _maybe_nudge_update() -> None:
    """Emit an update nudge at most once per process, off-thread so the
    (cached, ~20h) GitHub poll never blocks the first computer_use action."""
    global _update_checked
    if _update_checked:
        return
    _update_checked = True

    def _run() -> None:
        try:
            msg = cua_driver_update_nudge()
        except Exception:
            return
        if msg:
            logger.info("computer_use: %s", msg)

    threading.Thread(
        target=_run, name="cua-driver-update-check", daemon=True
    ).start()


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

    def __init__(self, bridge: _AsyncBridge) -> None:
        self._bridge = bridge
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
        self._capability_version: str = ""
        # Lifecycle plumbing — see class docstring above.
        self._ready_event = threading.Event()
        self._shutdown_event: Optional[asyncio.Event] = None  # created on bridge loop
        self._lifecycle_future = None  # concurrent.futures.Future
        self._setup_error: Optional[BaseException] = None

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
            if not cua_driver_binary_available():
                raise RuntimeError(cua_driver_install_hint())

            # Surface 8: ask cua-driver itself which subcommand spawns
            # the MCP server, instead of hardcoding ["mcp"]. Falls back
            # transparently for older drivers / any discovery failure.
            self._startup_phase = "manifest-discovery"
            command, args = _resolve_mcp_invocation(_CUA_DRIVER_CMD)
            _t_manifest = _time.monotonic()
            params = StdioServerParameters(
                command=command,
                args=args,
                # Apply the telemetry policy first (default: disabled), then
                # sanitize Hermes-managed secrets out of the child env.
                env=_sanitize_subprocess_env(cua_driver_child_env()),
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

    async def _populate_capabilities(self, session: Any) -> None:
        """Surface 4: cache per-tool capability sets + capability_version
        from tools/list. Soft prerequisite — discovery failure leaves
        the map empty and supports_capability degrades to False."""
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

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_lifecycle_locked()

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
        come back as an ``McpError`` carrying ``Resource temporarily
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

        cmd = [_CUA_DRIVER_CMD, "call", name, json.dumps(call_args)]
        attempts = 4
        backoff = 0.5
        parsed: Any = None
        last_err = ""
        try:
            for attempt in range(attempts):
                try:
                    proc = _subprocess.run(
                        cmd, capture_output=True, text=True, timeout=max(15.0, timeout),
                        env=_sanitize_subprocess_env(cua_driver_child_env()),
                    )
                except Exception as e:  # pragma: no cover - subprocess spawn failure
                    raise RuntimeError(f"cua-driver CLI fallback for {name} failed to spawn: {e}") from e

                out = (proc.stdout or "").strip()
                last_err = out[:200] or (proc.stderr or "")[:200]
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
            if isinstance(parsed, dict):
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
            return {"data": data, "images": images, "structuredContent": structured, "isError": False}
        finally:
            if shot_file and os.path.exists(shot_file):
                try:
                    os.remove(shot_file)
                except OSError:
                    pass

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        self._require_started()
        # The cua-driver daemon proxy returns POSIX EAGAIN ("Resource
        # temporarily unavailable") for heavier calls like get_window_state when
        # its non-blocking socket buffer is full. On some machines/builds this
        # is persistent for get_window_state over the MCP stdio bridge, while
        # the direct CLI transport keeps working. So: try the MCP path ONCE,
        # and on the transient/transport error fall straight through to the CLI
        # transport (which has its own retry + screenshot-to-file mitigation)
        # rather than burning a long backoff chain on a path that won't recover.
        try:
            return self._bridge.run(self._call_tool_async(name, args), timeout=timeout)
        except Exception as e:
            if self._is_transient_daemon_error(e):
                logger.warning(
                    "cua-driver MCP transport failed on %s (%s); "
                    "falling back to CLI transport", name, e,
                )
                return self._call_tool_via_cli(name, args, timeout)
            if not self._is_closed_session_error(e):
                raise
            # Daemon restart closes the cached stdio channel. Reconnect once and
            # retry exactly one more time — never loop, to avoid hammering a
            # genuinely dead daemon.
            logger.warning("cua-driver MCP session closed during %s; reconnecting once", name)
            with self._lock:
                self._restart_session_locked()
            return self._bridge.run(self._call_tool_async(name, args), timeout=timeout)


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
    is_error = bool(getattr(mcp_result, "isError", False))
    structured: Optional[Dict] = getattr(mcp_result, "structuredContent", None) or None
    text_chunks: List[str] = []
    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_chunks.append(getattr(part, "text", "") or "")
        elif ptype == "image":
            b64 = getattr(part, "data", None)
            if b64:
                images.append(b64)
                mime = getattr(part, "mimeType", None) or ""
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
    """
    windows: List[Dict[str, Any]] = []
    for w in raw_windows:
        pid, window_id = w.get("pid"), w.get("window_id")
        if pid is None or window_id is None:
            continue
        try:
            pid_int, window_id_int = int(pid), int(window_id)
        except (TypeError, ValueError):
            continue
        windows.append({
            "app_name": w.get("app_name", ""),
            "pid": pid_int,
            "window_id": window_id_int,
            "off_screen": not w.get("is_on_screen", True),
            "title": w.get("title", ""),
            "z_index": w.get("z_index", 0),
        })
    return windows


def _windows_from_tool_result(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list_windows payloads across cua-driver result shapes."""
    structured = out.get("structuredContent")
    if isinstance(structured, dict):
        windows = structured.get("windows")
        if isinstance(windows, list):
            return windows

    data = out.get("data")
    if isinstance(data, dict):
        windows = data.get("windows")
        if isinstance(windows, list):
            return windows
        legacy_windows = data.get("_legacy_windows")
        if isinstance(legacy_windows, list):
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

    def __init__(self) -> None:
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge)
        # Sticky context — updated by capture(), used by action tools.
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._last_app: Optional[str] = None  # last app name targeted via capture/focus_app
        # Surface 6 of NousResearch/hermes-agent#47072: per-snapshot
        # `element_index -> element_token` map populated on capture().
        # Action tools (click/scroll/set_value/...) attach the matching
        # token alongside `element_index` so cua-driver detects "stale"
        # explicitly instead of silently re-resolving to a different
        # element. Cleared whenever a fresh capture overwrites the
        # snapshot context.
        self._snapshot_tokens: Dict[int, str] = {}
        # Per-instance cua-driver session id. cua-driver's MCP server
        # instructions ask every consumer to declare a stable session
        # at the start of a run (start_session) and tear it down at
        # the end (end_session). Doing so:
        #   - Gets a distinct agent-cursor color per Hermes run, with
        #     overlay rendering visualising where actions land
        #     (without moving the real OS cursor).
        #   - Isolates per-session config + recording ownership so
        #     concurrent Hermes runs / subagents don't step on each
        #     other.
        # We mint a UUID4-based id once per CuaDriverBackend instance —
        # one Hermes run = one backend = one session — and pass it as
        # `session` on every cua-driver tool call. Sessions are an
        # additive feature on the cua-driver side: when our id is
        # unknown to the driver (older builds), the tool calls
        # degrade to the anonymous / unsynced path documented in the
        # MCP server instructions.
        self._session_id: str = f"hermes-{uuid.uuid4().hex[:12]}"

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        _maybe_nudge_update()
        # The MCP client SDK (`mcp`) is an optional dependency (the
        # `computer-use` / `mcp` extras), not part of Hermes' minimal core.
        # Lazy-install it on first use — the same pattern every other optional
        # backend uses — so users never hit an opaque `No module named 'mcp'`
        # at invoke time. Auto-install is gated by `security.allow_lazy_installs`
        # (default on); when it's disabled or fails, ensure() raises
        # FeatureUnavailable carrying an actionable `uv pip install mcp==…`
        # hint, which surfaces via the backend-unavailable path in tool.py.
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.computer_use", prompt=False)
        # A just-installed package may not be importable until the import
        # machinery's caches are refreshed within this process.
        import importlib
        importlib.invalidate_caches()
        self._session.start()

        # Declare the run's session identity to cua-driver. From the
        # cua-driver server instructions: "start_session(session) once
        # at the start of a run → declares THIS run's identity (a
        # stable id you choose). Pass that same `session` on every
        # action below. It owns your agent cursor (a distinct color
        # per id) and follows the run across apps/windows." Failure
        # to start the session is non-fatal — cua-driver's tools
        # accept anonymous calls (the cursor just won't render),
        # so we degrade rather than abort.
        try:
            self._session.call_tool("start_session", {"session": self._session_id})
        except Exception as e:
            logger.debug("cua-driver start_session failed (continuing anonymous): %s", e)

    def stop(self) -> None:
        # Tear the cua-driver session down before disconnecting so the
        # driver can clean up per-session state (cursor overlay, recording
        # ownership, config overrides). Best-effort — even if it fails,
        # the connection drop below releases the daemon-side state via
        # the session_end hook cua-driver registers internally.
        if self._session._started:
            try:
                self._session.call_tool("end_session", {"session": self._session_id})
            except Exception as e:
                logger.debug("cua-driver end_session failed (continuing teardown): %s", e)
        try:
            self._session.stop()
        finally:
            self._bridge.stop()

    def is_available(self) -> bool:
        # cua-driver runs on macOS, Windows, and Linux. The Linux path is
        # the most recent addition (X11 + Wayland both supported upstream
        # as of mid-2026). Override the platform check at your own risk:
        # other Unix-likes haven't been exercised end-to-end.
        if sys.platform not in ("darwin", "win32", "linux"):
            return False
        return cua_driver_binary_available()

    # ── Capture ────────────────────────────────────────────────────
    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        """Capture the frontmost on-screen window (optionally filtered by app name).

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
        lw_out = self._session.call_tool(
            "list_windows",
            {"on_screen_only": True, "session": self._session_id},
        )

        def _windows_from(out: Dict[str, Any]) -> List[Dict[str, Any]]:
            wins_ = _ingest_windows(_windows_from_tool_result(out))
            # Sort by z_index descending (lowest z_index = frontmost on macOS).
            wins_.sort(key=lambda w: w["z_index"])
            return wins_

        windows = _windows_from(lw_out)

        # If the MCP bridge returned an empty/degenerate window list (flaky
        # session), re-fetch over the CLI transport before giving up — otherwise
        # the caller sees a silent 0x0 capture even though windows exist.
        if not windows:
            logger.warning(
                "cua-driver list_windows returned no windows over MCP; "
                "re-fetching via CLI transport",
            )
            try:
                cli_lw = self._session._call_tool_via_cli(
                    "list_windows",
                    {"on_screen_only": True, "session": self._session_id},
                    20.0,
                )
                windows = _windows_from(cli_lw)
            except Exception as cli_exc:
                logger.error("cua-driver CLI re-fetch for list_windows failed: %s", cli_exc)

        if not windows:
            return CaptureResult(mode=mode, width=0, height=0, png_b64=None,
                                 elements=[], app="", window_title="", png_bytes_len=0)

        # Filter by app name (case-insensitive substring) if requested.
        # When the filter matches nothing, surface that explicitly instead of
        # silently capturing the frontmost window — on macOS the `app_name`
        # returned by list_windows is the localized name (e.g. "計算機"), so
        # `app="Calculator"` legitimately matches no windows on a non-English
        # system and the caller needs to retry with the localized name.
        if app and app.strip().lower() in _SCREEN_CAPTURE_SENTINELS:
            # Whole-screen / desktop request. cua-driver has no virtual-desktop
            # capture tool, so resolve to the OS shell/desktop window (the
            # desktop backdrop or the taskbar/menu-bar), which list_windows
            # does surface. This makes "show me my screen" and "click the
            # taskbar" work; a single image still can't span multiple monitors
            # — that's a driver limitation, not a wrapper one.
            def _is_desktop_window(w: Dict[str, Any]) -> bool:
                haystack = f"{w.get('app_name', '')} {w.get('title', '')}".lower()
                return any(name in haystack for name in _DESKTOP_WINDOW_NAMES)

            desktop = [w for w in windows if _is_desktop_window(w)]
            if not desktop:
                return CaptureResult(
                    mode=mode, width=0, height=0, png_b64=None,
                    elements=[], app="",
                    window_title=(
                        f"<no desktop/shell window found for app={app!r}; "
                        f"cua-driver captures one window at a time and exposes "
                        f"no whole-virtual-desktop or per-monitor capture. "
                        f"Call list_apps / capture(app='<AppName>') to target a "
                        f"specific window instead. On Windows the taskbar is "
                        f"'Shell_TrayWnd' and the desktop is 'Progman'.>"
                    ),
                    png_bytes_len=0,
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
        elif app:
            app_lower = app.lower()
            filtered = [w for w in windows if app_lower in w["app_name"].lower()]
            if not filtered:
                return CaptureResult(
                    mode=mode, width=0, height=0, png_b64=None,
                    elements=[], app="",
                    window_title=(
                        f"<no on-screen window matched app={app!r}; "
                        f"call list_apps to see available app names "
                        f"(macOS reports localized names, e.g. '計算機' "
                        f"instead of 'Calculator')>"
                    ),
                    png_bytes_len=0,
                )
            windows = filtered

        # Pick first on-screen window (sorted by z_index / z-order above).
        target = next((w for w in windows if not w["off_screen"]), windows[0])
        self._active_pid = target["pid"]
        self._active_window_id = target["window_id"]
        app_name = target["app_name"]
        # Record the resolved app name so capture_after= follow-ups can re-target
        # the same app rather than falling back to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name

        # Step 2: capture.
        png_b64: Optional[str] = None
        image_mime_type: Optional[str] = None
        elements: List[UIElement] = []
        width = height = 0
        window_title = ""

        if mode == "vision":
            # Plain screenshot, no AX walk. cua-driver dropped the standalone
            # `screenshot` tool (≥0.5.x) and folded full-window PNG capture
            # into `get_window_state`. Route accordingly:
            #   * Driver advertises `screenshot` (older builds) → use it; it's
            #     the cheapest path (no AX tree walked server-side).
            #   * Otherwise (current drivers) → call `get_window_state` but
            #     DISCARD the AX tree/elements, returning only the PNG. Vision
            #     mode's whole contract is "just the pixels, no element noise",
            #     so we drop everything but the image.
            # When capability discovery hasn't run (empty map), we don't trust
            # a negative `_has_tool` answer — we still try `screenshot` first
            # and fall back if the driver rejects it, so the path self-heals on
            # any driver version.
            use_screenshot = (
                self._session._has_tool("screenshot")
                or not self._session.capabilities_discovered
            )
            sc_out: Optional[Dict[str, Any]] = None
            if use_screenshot:
                sc_out = self._session.call_tool(
                    "screenshot",
                    {
                        "window_id": self._active_window_id,
                        "format": "jpeg",
                        "quality": 85,
                        "session": self._session_id,
                    },
                )
                png_b64, image_mime_type = _image_from_tool_result(sc_out)
                if not png_b64:
                    # Driver had no usable `screenshot` (e.g. "Unknown tool:
                    # screenshot" on ≥0.5.x, or an empty image part). Fall
                    # through to the get_window_state path below.
                    sc_out = None

            if sc_out is None:
                gws_out = self._session.call_tool(
                    "get_window_state",
                    {
                        "pid": self._active_pid,
                        "window_id": self._active_window_id,
                        "session": self._session_id,
                    },
                )
                png_b64, image_mime_type = _image_from_tool_result(gws_out)
                # Still grab the window title — it's cheap and useful in the
                # vision response — but deliberately leave `elements` empty so
                # vision stays free of AX-tree noise.
                text = gws_out["data"] if isinstance(gws_out["data"], str) else ""
                _, tree = _split_tree_text(text)
                wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
                if wt:
                    window_title = wt.group(1)

            if not png_b64:
                # Both MCP attempts came back imageless without raising (flaky
                # bridge dropping the heavy payload) — re-fetch the window
                # state over the CLI transport, which embeds a screenshot.
                logger.warning(
                    "cua-driver vision capture returned no image over MCP "
                    "(window_id=%s); re-fetching via CLI transport",
                    self._active_window_id,
                )
                try:
                    cli_out = self._session._call_tool_via_cli(
                        "get_window_state",
                        {
                            "pid": self._active_pid,
                            "window_id": self._active_window_id,
                            "session": self._session_id,
                        },
                        30.0,
                    )
                    if cli_out.get("images"):
                        png_b64 = cli_out["images"][0]
                        image_mime_type = "image/png"
                except Exception as cli_exc:
                    logger.error(
                        "cua-driver CLI re-fetch for vision screenshot failed: %s", cli_exc,
                    )
        else:
            # get_window_state: AX tree + screenshot.
            gws_out = self._session.call_tool(
                "get_window_state",
                {
                    "pid": self._active_pid,
                    "window_id": self._active_window_id,
                    "session": self._session_id,
                },
            )
            # The persistent MCP session can return a degenerate result —
            # empty/partial data with NO exception — when the bridge is flaky
            # (e.g. it reconnected mid-call and dropped the heavy
            # get_window_state payload). That surfaces to the model as a silent
            # 0x0 capture. Detect "no screenshot AND no parseable tree" and
            # force a one-shot CLI-transport re-fetch, which talks to the daemon
            # over a different socket and returns the full result. This is
            # distinct from the EAGAIN McpError path (handled in call_tool);
            # here the MCP call "succeeded" but gave us nothing usable.
            def _gws_is_empty(out: Dict[str, Any]) -> bool:
                if out.get("images"):
                    return False
                sc_ = out.get("structuredContent") or {}
                # Modern drivers carry the payload in structuredContent
                # (elements array / embedded screenshot) with no markdown
                # tree — that is NOT an empty result.
                if sc_.get("elements") or sc_.get("screenshot_png_b64"):
                    return False
                txt = out.get("data") if isinstance(out.get("data"), str) else ""
                _, tr = _split_tree_text(txt or "")
                return not (tr and tr.strip())

            if _gws_is_empty(gws_out):
                logger.warning(
                    "cua-driver get_window_state returned an empty result over MCP "
                    "(pid=%s window_id=%s); re-fetching via CLI transport",
                    self._active_pid, self._active_window_id,
                )
                try:
                    cli_out = self._session._call_tool_via_cli(
                        "get_window_state",
                        {
                            "pid": self._active_pid,
                            "window_id": self._active_window_id,
                            "session": self._session_id,
                        },
                        30.0,
                    )
                    if not _gws_is_empty(cli_out):
                        gws_out = cli_out
                except Exception as cli_exc:
                    logger.error(
                        "cua-driver CLI re-fetch for get_window_state failed: %s", cli_exc,
                    )

            text = gws_out["data"] if isinstance(gws_out["data"], str) else ""
            summary, tree = _split_tree_text(text)

            # Parse element count from summary e.g. "✅ AppName — 42 elements, turn 3..."
            m = re.search(r'(\d+)\s+elements?', summary)

            # Surface 2 of NousResearch/hermes-agent#47072: prefer the
            # canonical structuredContent.elements array (trycua/cua#1961).
            # Falls back to markdown regex parsing for cua-driver builds
            # that didn't carry the structured shape — those bounds come
            # back (0,0,0,0); the structured path preserves real frames.
            sc_elements = (gws_out.get("structuredContent") or {}).get("elements")
            if isinstance(sc_elements, list) and sc_elements:
                elements = _parse_elements_from_structured(sc_elements)
            else:
                elements = _parse_elements_from_tree(tree) if tree else []

            # Surface 6: refresh the snapshot-token cache from this
            # capture. Tokens are tied to a specific cua-driver snapshot
            # — when a fresh capture lands, the prior snapshot's tokens
            # are stale, so we overwrite the whole map (and clear it
            # entirely when the new capture carries none).
            self._snapshot_tokens = {
                e.index: e.element_token
                for e in elements
                if e.element_token
            }

            # Image may arrive as an MCP image part or inside
            # structuredContent (screenshot_png_b64) depending on the driver
            # build — _image_from_tool_result handles both.
            png_b64, image_mime_type = _image_from_tool_result(gws_out)

            # Extract window title from the AX tree first AXWindow line.
            wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
            if wt:
                window_title = wt.group(1)

        png_bytes_len = 0
        if png_b64:
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
            mode=mode,
            width=width,
            height=height,
            png_b64=png_b64,
            elements=elements,
            app=app_name,
            window_title=window_title,
            png_bytes_len=png_bytes_len,
            image_mime_type=image_mime_type,
        )

    # ── Pointer ────────────────────────────────────────────────────
    def click(
        self,
        *,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="click",
                                message="No active window — call capture() first.")

        # Choose tool by click_count only — single-vs-double — and pass the
        # button through to `click`'s `button` enum (Surface 5 of
        # NousResearch/hermes-agent#47072). cua-driver-rs gained an explicit
        # `button: "left"|"right"|"middle"` arg on `click` in trycua/cua#1961
        # which rejects unknown buttons; before that, `middle` was silently
        # mapped to a left-click via name-routing through `right_click`.
        # `right_click`/`middle_click` MCP tools are deprecated aliases —
        # kept around but no longer invoked from here.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return ActionResult(ok=False, action="click",
                                message=f"unknown button {button!r} — expected left, right, middle.")
        tool = "double_click" if click_count == 2 else "click"

        args: Dict[str, Any] = {"pid": pid, "button": button_norm}
        if element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for element_index click.")
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        else:
            return ActionResult(ok=False, action=tool,
                                message="click requires element= or x/y.")
        if modifiers:
            args["modifier"] = modifiers

        return self._action(tool, args)

    def drag(
        self,
        *,
        from_element: Optional[int] = None,
        to_element: Optional[int] = None,
        from_xy: Optional[Tuple[int, int]] = None,
        to_xy: Optional[Tuple[int, int]] = None,
        button: str = "left",
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="drag",
                                message="No active window — call capture() first.")
        args: Dict[str, Any] = {"pid": pid}
        if from_element is not None and to_element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for element-based drag.")
            args["from_element"] = from_element
            args["to_element"] = to_element
            args["window_id"] = self._active_window_id
        elif from_xy is not None and to_xy is not None:
            args["from_x"], args["from_y"] = int(from_xy[0]), int(from_xy[1])
            args["to_x"], args["to_y"] = int(to_xy[0]), int(to_xy[1])
        else:
            return ActionResult(ok=False, action="drag",
                                message="drag requires from_element/to_element or from_coordinate/to_coordinate.")
        return self._action("drag", args)

    def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        modifiers: Optional[List[str]] = None,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="scroll",
                                message="No active window — call capture() first.")
        args: Dict[str, Any] = {
            "pid": pid,
            "direction": direction,
            "amount": max(1, min(50, amount)),
        }
        if element is not None and self._active_window_id is not None:
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            args["x"] = x
            args["y"] = y
        return self._action("scroll", args)

    # ── Keyboard ───────────────────────────────────────────────────
    def type_text(self, text: str) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="type_text",
                                message="No active window — call capture() first.")
        return self._action("type_text", {"pid": pid, "text": text})

    def key(self, keys: str) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="key",
                                message="No active window — call capture() first.")

        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return ActionResult(ok=False, action="key",
                                message=f"Could not parse key from '{keys}'.")

        if modifiers:
            # hotkey requires at least one modifier + one key.
            return self._action("hotkey", {"pid": pid, "keys": modifiers + [key_name]})
        else:
            return self._action("press_key", {"pid": pid, "key": key_name})

    # ── Value setter ────────────────────────────────────────────────
    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        pid = self._active_pid
        window_id = self._active_window_id
        if pid is None or window_id is None:
            return ActionResult(ok=False, action="set_value",
                                message="No active window — call capture() first.")
        if element is None:
            return ActionResult(ok=False, action="set_value",
                                message="set_value requires element= (element index).")
        args: Dict[str, Any] = {
            "pid": pid,
            "window_id": window_id,
            "element_index": element,
            "value": value,
        }
        return self._action("set_value", args)

    # ── Introspection ──────────────────────────────────────────────
    def list_apps(self) -> List[Dict[str, Any]]:
        out = self._session.call_tool("list_apps", {"session": self._session_id})
        data = out["data"]
        structured = out.get("structuredContent")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            apps = data.get("apps")
            if isinstance(apps, list):
                return apps
            if isinstance(structured, dict):
                structured_apps = structured.get("apps")
                if isinstance(structured_apps, list):
                    return structured_apps
            return _apps_from_windows(_windows_from_tool_result(out))
        if isinstance(structured, dict):
            apps = structured.get("apps")
            if isinstance(apps, list):
                return apps
            return _apps_from_windows(_windows_from_tool_result(out))
        # list_apps returns plain text — parse app lines.
        if isinstance(data, str):
            apps = []
            for line in data.splitlines():
                m = re.search(r'(.+?)\s+\(pid\s+(\d+)\)', line)
                if m:
                    apps.append({"name": m.group(1).strip(), "pid": int(m.group(2))})
            return apps
        return []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Target an app for subsequent actions without stealing system focus.

        cua-driver background-automation never needs to bring a window to the
        front: capture(app=...) already selects the right window via
        list_windows. We implement focus_app as a pure window-selector —
        enumerate on-screen windows, find the best match for *app*, and store
        its pid/window_id so that subsequent click/type calls hit the right
        process.

        raise_window=True is intentionally ignored: stealing the user's focus
        is exactly what this backend is designed to avoid.
        """
        lw_out = self._session.call_tool(
            "list_windows",
            {"on_screen_only": True, "session": self._session_id},
        )
        windows = _ingest_windows(_windows_from_tool_result(lw_out))
        windows.sort(key=lambda w: w["z_index"])

        app_lower = app.lower()
        matched = [w for w in windows if app_lower in w["app_name"].lower()]
        # Don't silently fall back to the frontmost window when the filter
        # matches nothing — that hides the real failure (often a localized
        # macOS app name mismatch, e.g. caller passed "Calculator" but
        # list_windows returns "計算機").
        target = matched[0] if matched else None
        if target:
            self._active_pid = target["pid"]
            self._active_window_id = target["window_id"]
            self._last_app = target["app_name"]  # preserve for capture_after= follow-ups
            return ActionResult(
                ok=True, action="focus_app",
                message=f"Targeted {target['app_name']} (pid {self._active_pid}, "
                        f"window {self._active_window_id}) without raising window.",
            )
        return ActionResult(ok=False, action="focus_app",
                            message=f"No on-screen window found for app '{app}'.")

    # ── App lifecycle ────────────────────────────────────────────────
    #
    # cua-driver exposes launch_app / kill_app / bring_to_front as a
    # complete set. focus_app() above is a *window-selector* (no
    # process state change); these methods drive the process layer.

    def launch_app(
        self,
        *,
        bundle_id: Optional[str] = None,
        name: Optional[str] = None,
        urls: Optional[List[str]] = None,
        additional_arguments: Optional[List[str]] = None,
        creates_new_application_instance: bool = False,
    ) -> Dict[str, Any]:
        """Idempotent launch. Returns ``{pid, bundle_id, name, windows[]}``
        so callers can skip an extra ``list_windows`` round-trip before
        ``get_window_state``.

        ``creates_new_application_instance=True`` forces a new instance
        even if the app is already running — use it when concurrent
        runs may touch the same app so each session gets its own
        isolated window."""
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        args: Dict[str, Any] = {"session": self._session_id}
        if bundle_id:
            args["bundle_id"] = bundle_id
        if name:
            args["name"] = name
        if urls:
            args["urls"] = list(urls)
        if additional_arguments:
            args["additional_arguments"] = list(additional_arguments)
        if creates_new_application_instance:
            args["creates_new_application_instance"] = True
        out = self._session.call_tool("launch_app", args)
        return out["structuredContent"] or {"data": out["data"]}

    def kill_app(self, *, pid: int) -> ActionResult:
        """Terminate by pid. Equivalent to ``kill -9`` on POSIX,
        ``taskkill /F`` on Windows."""
        return self._action("kill_app", {"pid": int(pid)})

    def bring_to_front(self, *, pid: int,
                       window_id: Optional[int] = None) -> ActionResult:
        """Activate a window so subsequent foreground-dispatched input
        lands on it. cua-driver's docstring notes this is the cheaper
        path than per-call SetForegroundWindow flashes."""
        args: Dict[str, Any] = {"pid": int(pid)}
        if window_id is not None:
            args["window_id"] = int(window_id)
        return self._action("bring_to_front", args)

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
        args: Dict[str, Any] = {"enabled": bool(enabled)}
        if cursor_id:
            args["cursor_id"] = cursor_id
        return self._action("set_agent_cursor_enabled", args)

    def set_agent_cursor_motion(self, *,
                                glide_ms: Optional[float] = None,
                                dwell_ms: Optional[float] = None,
                                idle_hide_ms: Optional[float] = None,
                                cursor_id: Optional[str] = None) -> ActionResult:
        """Tune the overlay's motion timings — glide duration, post-click
        dwell, idle-hide delay. Each None means "leave at current value"."""
        args: Dict[str, Any] = {}
        if glide_ms is not None:
            args["glide_ms"] = float(glide_ms)
        if dwell_ms is not None:
            args["dwell_ms"] = float(dwell_ms)
        if idle_hide_ms is not None:
            args["idle_hide_ms"] = float(idle_hide_ms)
        if cursor_id:
            args["cursor_id"] = cursor_id
        return self._action("set_agent_cursor_motion", args)

    def set_agent_cursor_style(self, *,
                               gradient_colors: Optional[List[str]] = None,
                               bloom_color: Optional[str] = None,
                               image_path: Optional[str] = None,
                               cursor_id: Optional[str] = None) -> ActionResult:
        """Customise the cursor body. ``gradient_colors`` are CSS hex
        strings tip→tail; ``bloom_color`` is the radial halo; an
        ``image_path`` (.svg/.png/.ico) replaces the silhouette
        entirely. Empty values revert to the palette default."""
        args: Dict[str, Any] = {}
        if gradient_colors is not None:
            args["gradient_colors"] = list(gradient_colors)
        if bloom_color is not None:
            args["bloom_color"] = bloom_color
        if image_path is not None:
            args["image_path"] = image_path
        if cursor_id:
            args["cursor_id"] = cursor_id
        return self._action("set_agent_cursor_style", args)

    def get_agent_cursor_state(self, *,
                               cursor_id: Optional[str] = None) -> Dict[str, Any]:
        """Return ``{x, y, config: {cursor_color, cursor_icon, ...},
        enabled}`` for this run's cursor (or the named ``cursor_id``)."""
        args: Dict[str, Any] = {"session": self._session_id}
        if cursor_id:
            args["cursor_id"] = cursor_id
        out = self._session.call_tool("get_agent_cursor_state", args)
        return out.get("structuredContent") or {}

    # ── Recording / replay ──────────────────────────────────────────

    def start_recording(self, *, output_dir: str,
                        record_video: bool = False) -> Dict[str, Any]:
        """Enable trajectory recording (per-turn screenshots + action
        JSON) to ``output_dir``. ``record_video=True`` ALSO captures
        the main display to ``<output_dir>/recording.mp4`` (H.264).
        Recording ownership is keyed by this run's session id so
        concurrent runs don't fight over the recorder."""
        out = self._session.call_tool("start_recording", {
            "output_dir": output_dir,
            "record_video": bool(record_video),
            "session": self._session_id,
        })
        return out.get("structuredContent") or {}

    def stop_recording(self) -> Dict[str, Any]:
        """Disable recording and finalise the mp4 (if video was on).
        Returns the recorder's final state including ``last_video_path``."""
        out = self._session.call_tool("stop_recording", {
            "session": self._session_id,
        })
        return out.get("structuredContent") or {}

    def get_recording_state(self) -> Dict[str, Any]:
        """Return the current recorder state without changing it.
        Shape: ``{recording, enabled, output_dir, next_turn,
        last_video_path, last_error, owner, video_active}``."""
        out = self._session.call_tool(
            "get_recording_state", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    def replay_trajectory(self, *, trajectory_dir: str,
                          dry_run: bool = False,
                          speed_factor: float = 1.0) -> Dict[str, Any]:
        """Replay a prior recording's turn stream by re-invoking each
        turn's tool call in lexical order. ``dry_run=True`` logs without
        actually firing the tools."""
        return self._session.call_tool("replay_trajectory", {
            "trajectory_dir": trajectory_dir,
            "dry_run": bool(dry_run),
            "speed_factor": float(speed_factor),
            "session": self._session_id,
        })

    def install_ffmpeg(self) -> Dict[str, Any]:
        """Bootstrap ffmpeg for ``start_recording(record_video=True)``
        on Linux / Windows. macOS records natively via ScreenCaptureKit
        and doesn't need ffmpeg."""
        return self._session.call_tool(
            "install_ffmpeg", {"session": self._session_id}
        )

    # ── Config ──────────────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        """Return the current cua-driver runtime config."""
        out = self._session.call_tool(
            "get_config", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    def set_config(self, **config) -> ActionResult:
        """Set cua-driver config keys. Common keys include
        ``max_image_dimension`` (image-output resizing), recording
        flags, etc. Unknown keys are passed through verbatim — cua-driver
        validates against its own schema."""
        return self._action("set_config", dict(config))

    # ── Lower-level introspection ───────────────────────────────────

    def get_accessibility_tree(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of running regular apps +
        on-screen visible windows with bounds, z-order, owner pid.
        Roughly the data ``list_windows`` exposes, in one call. Most
        callers should prefer ``capture()`` / ``focus_app()`` which
        already use this shape internally."""
        out = self._session.call_tool(
            "get_accessibility_tree", {"session": self._session_id}
        )
        return out.get("structuredContent") or {"data": out["data"]}

    # ── Browser page tool ───────────────────────────────────────────

    def page(self, *, pid: int, action: str,
             **page_args: Any) -> Dict[str, Any]:
        """Interact with a browser page loaded in a running app (Chrome,
        Safari, Edge, ...). cua-driver routes through CDP / Apple Events
        / AX tree depending on the target. ``action`` + ``page_args``
        shape depends on the requested operation (e.g. ``action="eval"``
        takes ``js: str``); see cua-driver's ``page`` tool description
        for the full grammar."""
        args: Dict[str, Any] = {
            "pid": int(pid),
            "action": action,
            "session": self._session_id,
        }
        args.update(page_args)
        return self._session.call_tool("page", args)

    # ── Generic escape hatch ────────────────────────────────────────

    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None,
                  *, timeout: float = 30.0) -> Dict[str, Any]:
        """Call any cua-driver MCP tool by name with arbitrary args.
        ``session`` is injected (preserves the caller's explicit one
        via setdefault). For tools the wrapper doesn't already type-
        wrap, this is the supported escape hatch — preferred over
        reaching for ``self._session.call_tool`` directly because it
        keeps the session-id contract consistent with everything else."""
        payload = dict(args) if args else {}
        payload.setdefault("session", self._session_id)
        return self._session.call_tool(name, payload, timeout=timeout)

    # ── Internal ───────────────────────────────────────────────────
    def _maybe_attach_element_token(self, tool: str, args: Dict[str, Any]) -> None:
        """Surface 6: when the wrapper is about to call a token-capable
        tool with `element_index`, look up the matching `element_token`
        from the last snapshot and attach it. cua-driver-rs's contract
        for combined args is documented in trycua/cua#1961:

          "element_token takes precedence over element_index when both
           supplied. Returns an explicit 'stale' error if the snapshot
           has been superseded."

        Gated on the per-tool capability claim so we don't send the
        field to drivers that predate the surface (which would reject
        the schema with `additionalProperties: false`).
        """
        idx = args.get("element_index")
        if not isinstance(idx, int):
            return
        token = self._snapshot_tokens.get(idx)
        if not token:
            return
        if not self._session.supports_capability(
            "accessibility.element_tokens", tool=tool
        ):
            return
        args["element_token"] = token

    def _action(self, name: str, args: Dict[str, Any]) -> ActionResult:
        # Attach the snapshot's element_token whenever the call carries
        # an element_index and the target tool advertises support.
        self._maybe_attach_element_token(name, args)
        # Carry this run's session id so the cua-driver agent cursor
        # and per-session state (config overrides, recording ownership)
        # stay tied to this run. setdefault preserves any explicit
        # session a caller already supplied.
        args.setdefault("session", self._session_id)
        try:
            out = self._session.call_tool(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        ok = not out["isError"]
        message = ""
        data = out["data"]
        if isinstance(data, dict):
            message = str(data.get("message", ""))
        elif isinstance(data, str):
            message = data
        return ActionResult(ok=ok, action=name, message=message,
                            meta=data if isinstance(data, dict) else {})
