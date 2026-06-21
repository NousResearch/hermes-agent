"""Cua-driver backend (macOS + Windows).

Speaks MCP over stdio to `cua-driver`. The Python `mcp` SDK is async, so we
run a dedicated asyncio event loop on a background thread and marshal sync
calls through it.

The same `cua-driver call <tool>` surface (click, type_text, hotkey, drag,
scroll, screenshot, launch_app, list_apps, list_windows, get_window_state,
move_cursor, wait) works identically across macOS + Windows — cua-driver's
PARITY matrix marks every action tool VERIFIED on Windows in the
cross-platform Rust port (`cua-driver-rs`).

Linux support exists in cua-driver-rs but is alpha today — Linux PARITY
rows are mostly OPEN, not VERIFIED — so it's gated off in
`check_computer_use_requirements` until that flips upstream. The plumbing
in this file is OS-agnostic, so flipping that gate later is one-line.

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
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
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
        proc = subprocess.run(
            [driver_cmd, "manifest"],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
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
# Handles two output formats from different cua-driver versions:
#   Classic:  "  - [N] AXRole \"label\""
#   New:       "[N] AXRole (order) id=Label"
#
# Group 1: element index
# Group 2: AX role
# Group 3: quoted label (classic format)
# Group 4: id= label (new format)
_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)(?:\s+"([^"]*)"|(?:\s+\(\d+\))?\s+id=([^\s\[\]]*))?' ,
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
        proc = subprocess.run(
            [_CUA_DRIVER_CMD, "check-update", "--json"],
            capture_output=True, text=True, timeout=timeout,
            # Some older drivers don't have the verb and fall through to a
            # stdin-reading mode rather than erroring — DEVNULL gives them EOF
            # so they exit fast instead of blocking until the timeout.
            stdin=subprocess.DEVNULL,
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

    Handles both the classic ``"label"``-quoted format and the newer
    ``id=Label`` format introduced in cua-driver v0.1.6. Bounds always
    come back ``(0, 0, 0, 0)`` because the markdown surface doesn't
    carry them — yet another reason to prefer the structured path.
    """
    elements = []
    for m in _ELEMENT_LINE_RE.finditer(markdown):
        # group(3) = quoted label (classic); group(4) = id= label (new)
        label = m.group(3) or m.group(4) or ""
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
        elements.append(UIElement(
            index=idx,
            role=role,
            label=label,
            bounds=bounds,
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
    """Holds the mcp ClientSession. Spawned lazily; re-entered on drop."""

    def __init__(self, bridge: _AsyncBridge) -> None:
        self._bridge = bridge
        self._session = None
        self._exit_stack = None
        self._lock = threading.Lock()
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("cua-driver session not started")

    async def _aenter(self) -> None:
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.environments.local import _sanitize_subprocess_env

        if not cua_driver_binary_available():
            raise RuntimeError(cua_driver_install_hint())

        # Surface 8: ask cua-driver itself which subcommand spawns the MCP
        # server, instead of hardcoding ["mcp"]. Falls back transparently
        # for older drivers (or any indeterminate discovery failure).
        command, args = _resolve_mcp_invocation(_CUA_DRIVER_CMD)
        params = StdioServerParameters(
            command=command,
            args=args,
            env=_sanitize_subprocess_env(dict(os.environ)),
        )
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._exit_stack = stack
        self._session = session

    async def _aexit(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("cua-driver shutdown error: %s", e)
        self._exit_stack = None
        self._session = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._bridge.start()
            self._bridge.run(self._aenter(), timeout=15.0)
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._bridge.run(self._aexit(), timeout=5.0)
            finally:
                self._started = False

    async def _call_tool_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await self._session.call_tool(name, args)
        return _extract_tool_result(result)

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

    def _restart_session_locked(self) -> None:
        """Recreate the MCP session after the daemon/stdin transport was closed."""
        try:
            if self._started:
                self._bridge.run(self._aexit(), timeout=5.0)
        except Exception as e:
            logger.debug("cua-driver session cleanup before reconnect failed: %s", e)
        self._started = False
        self._bridge.run(self._aenter(), timeout=15.0)
        self._started = True

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        self._require_started()
        try:
            return self._bridge.run(self._call_tool_async(name, args), timeout=timeout)
        except Exception as e:
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


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP (macOS + Windows)."""

    def __init__(self) -> None:
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge)
        # Sticky context — updated by capture(), used by action tools.
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._last_app: Optional[str] = None  # last app name targeted via capture/focus_app

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

    def stop(self) -> None:
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
        lw_out = self._session.call_tool("list_windows", {"on_screen_only": True})
        raw_windows = (lw_out.get("structuredContent") or {}).get("windows") or []
        windows = [
            {
                "app_name": w.get("app_name", ""),
                "pid": int(w["pid"]),
                "window_id": int(w["window_id"]),
                "off_screen": not w.get("is_on_screen", True),
                "title": w.get("title", ""),
                "z_index": w.get("z_index", 0),
            }
            for w in raw_windows
        ]
        # Sort by z_index descending (lowest z_index = frontmost on macOS).
        windows.sort(key=lambda w: w["z_index"])

        if not windows:
            return CaptureResult(mode=mode, width=0, height=0, png_b64=None,
                                 elements=[], app="", window_title="", png_bytes_len=0)

        # Filter by app name (case-insensitive substring) if requested.
        # When the filter matches nothing, surface that explicitly instead of
        # silently capturing the frontmost window — on macOS the `app_name`
        # returned by list_windows is the localized name (e.g. "計算機"), so
        # `app="Calculator"` legitimately matches no windows on a non-English
        # system and the caller needs to retry with the localized name.
        if app:
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
            # screenshot tool: just the PNG, no AX walk.
            sc_out = self._session.call_tool(
                "screenshot",
                {"window_id": self._active_window_id, "format": "jpeg", "quality": 85},
            )
            if sc_out["images"]:
                png_b64 = sc_out["images"][0]
                # Pick up the explicit mimeType cua-driver attaches to image
                # parts (Surface 7). Empty string means the driver didn't
                # carry one — callers will fall back to magic-byte sniffing.
                mimes = sc_out.get("image_mime_types") or []
                image_mime_type = mimes[0] if mimes and mimes[0] else None
        else:
            # get_window_state: AX tree + optional screenshot.
            gws_out = self._session.call_tool(
                "get_window_state",
                {"pid": self._active_pid, "window_id": self._active_window_id},
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

            if gws_out["images"]:
                png_b64 = gws_out["images"][0]
                mimes = gws_out.get("image_mime_types") or []
                image_mime_type = mimes[0] if mimes and mimes[0] else None

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
        out = self._session.call_tool("list_apps", {})
        data = out["data"]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("apps", [])
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
        lw_out = self._session.call_tool("list_windows", {"on_screen_only": True})
        raw_windows = (lw_out.get("structuredContent") or {}).get("windows") or []
        windows = [
            {
                "app_name": w.get("app_name", ""),
                "pid": int(w["pid"]),
                "window_id": int(w["window_id"]),
                "z_index": w.get("z_index", 0),
            }
            for w in raw_windows
        ]
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

    # ── Internal ───────────────────────────────────────────────────
    def _action(self, name: str, args: Dict[str, Any]) -> ActionResult:
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
