"""Cua-driver backend (macOS only).

Speaks MCP over stdio to `cua-driver`. The Python `mcp` SDK is async, so we
run a dedicated asyncio event loop on a background thread and marshal sync
calls through it.

Install: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"`

After install, `cua-driver` is on $PATH and supports `cua-driver mcp` (stdio
transport) which is what we invoke.

The private SkyLight SPIs cua-driver uses (SLEventPostToPid, SLPSPostEvent-
RecordTo, _AXObserverAddNotificationAndCheckRemote) are not Apple-public and
can break on OS updates. Pin the installed version via `HERMES_CUA_DRIVER_
VERSION` if you want reproducibility across an OS bump.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
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
# Version pinning
# ---------------------------------------------------------------------------

PINNED_CUA_DRIVER_VERSION = os.environ.get("HERMES_CUA_DRIVER_VERSION", "0.5.0")

_CUA_DRIVER_CMD = os.environ.get("HERMES_CUA_DRIVER_CMD", "cua-driver")
_CUA_DRIVER_ARGS = ["mcp"]  # stdio MCP transport

# Regex to parse list_windows text output lines:
#   "- AppName (pid 12345) "Title" [window_id: 67890]"
_WINDOW_LINE_RE = re.compile(
    r'^-\s+(.+?)\s+\(pid\s+(\d+)\)\s+.*\[window_id:\s+(\d+)\]',
    re.MULTILINE,
)

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


def cua_driver_install_hint() -> str:
    return (
        "cua-driver is not installed. Install with one of:\n"
        "  hermes computer-use install\n"
        "Or run the upstream installer directly:\n"
        '  /bin/bash -c "$(curl -fsSL '
        'https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"\n'
        "Or run `hermes tools` and enable the Computer Use toolset to install it automatically."
    )


def _parse_windows_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse window records from list_windows text output."""
    windows = []
    for m in _WINDOW_LINE_RE.finditer(text):
        windows.append({
            "app_name": m.group(1).strip(),
            "pid": int(m.group(2)),
            "window_id": int(m.group(3)),
            "off_screen": "[off-screen]" in m.group(0),
        })
    return windows


def _parse_elements_from_tree(markdown: str) -> List[UIElement]:
    """Parse UIElement list from get_window_state AX tree markdown.

    Captures the label whichever form cua-driver used: ``= "value"``,
    ``"quoted"``, ``(parenthesised)``, or ``id=Label``. Bounds are not present
    in the markdown rendering, so they remain (0,0,0,0); element-index clicks do
    not need them (the driver resolves the index to a frame internally).
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

        if not cua_driver_binary_available():
            raise RuntimeError(cua_driver_install_hint())

        params = StdioServerParameters(
            command=_CUA_DRIVER_CMD,
            args=_CUA_DRIVER_ARGS,
            env={**os.environ},
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
        """Recreate the MCP session after the daemon/stdin transport was closed."""
        try:
            if self._started:
                self._bridge.run(self._aexit(), timeout=5.0)
        except Exception as e:
            logger.debug("cua-driver session cleanup before reconnect failed: %s", e)
        self._started = False
        self._bridge.run(self._aenter(), timeout=15.0)
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
                        cmd, capture_output=True, text=True, timeout=max(15.0, timeout)
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
        "structuredContent": <dict|None>,
        "isError": bool,
      }
    structuredContent is populated from the MCP result's structuredContent field
    (MCP spec §2024-11-05+) and takes precedence for structured data like
    list_windows window arrays.
    """
    data: Any = None
    images: List[str] = []
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
    if text_chunks:
        joined = "\n".join(t for t in text_chunks if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined
    return {"data": data, "images": images, "structuredContent": structured, "isError": is_error}


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend. macOS-only via cua-driver MCP."""

    def __init__(self) -> None:
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge)
        # Sticky context — updated by capture(), used by action tools.
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._last_app: Optional[str] = None  # last app name targeted via capture/focus_app

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        self._session.start()

    def stop(self) -> None:
        try:
            self._session.stop()
        finally:
            self._bridge.stop()

    def is_available(self) -> bool:
        if not _is_macos():
            return False
        return cua_driver_binary_available()

    # ── Capture ────────────────────────────────────────────────────
    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        """Capture the frontmost on-screen window (optionally filtered by app name).

        Maps hermes `capture(mode, app)` → cua-driver `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision).
        """
        # Step 1: enumerate on-screen windows to find target pid/window_id.
        lw_out = self._session.call_tool("list_windows", {"on_screen_only": True})

        def _windows_from(out: Dict[str, Any]) -> List[Dict[str, Any]]:
            sc_ = out.get("structuredContent") or {}
            raw_ = sc_.get("windows") if sc_ else None
            if raw_:
                wins_ = [
                    {
                        "app_name": w.get("app_name", ""),
                        "pid": int(w["pid"]),
                        "window_id": int(w["window_id"]),
                        "off_screen": not w.get("is_on_screen", True),
                        "title": w.get("title", ""),
                        "z_index": w.get("z_index", 0),
                    }
                    for w in raw_
                ]
                wins_.sort(key=lambda w: w["z_index"])
                return wins_
            raw_text_ = out["data"] if isinstance(out.get("data"), str) else ""
            return _parse_windows_from_text(raw_text_)

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
                    "list_windows", {"on_screen_only": True}, 20.0,
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
            else:
                # Empty screenshot result over MCP (flaky bridge) — re-fetch the
                # window state via the CLI transport, which embeds a screenshot.
                logger.warning(
                    "cua-driver screenshot returned no image over MCP "
                    "(window_id=%s); re-fetching via CLI transport",
                    self._active_window_id,
                )
                try:
                    cli_out = self._session._call_tool_via_cli(
                        "get_window_state",
                        {"pid": self._active_pid, "window_id": self._active_window_id},
                        30.0,
                    )
                    if cli_out.get("images"):
                        png_b64 = cli_out["images"][0]
                except Exception as cli_exc:
                    logger.error(
                        "cua-driver CLI re-fetch for vision screenshot failed: %s", cli_exc,
                    )
        else:
            # get_window_state: AX tree + optional screenshot.
            gws_out = self._session.call_tool(
                "get_window_state",
                {"pid": self._active_pid, "window_id": self._active_window_id},
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
                        {"pid": self._active_pid, "window_id": self._active_window_id},
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
            if tree and not gws_out["images"]:
                # ax mode — no screenshot
                elements = _parse_elements_from_tree(tree)
            elif gws_out["images"]:
                png_b64 = gws_out["images"][0]
                elements = _parse_elements_from_tree(tree)

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

        # Choose tool based on button and click_count.
        if button == "right":
            tool = "right_click"
        elif click_count == 2:
            tool = "double_click"
        else:
            tool = "click"

        args: Dict[str, Any] = {"pid": pid}
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
        sc = lw_out.get("structuredContent") or {}
        raw_windows = sc.get("windows") if sc else None
        if raw_windows:
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
        else:
            raw_text = lw_out["data"] if isinstance(lw_out["data"], str) else ""
            windows = _parse_windows_from_text(raw_text)

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
