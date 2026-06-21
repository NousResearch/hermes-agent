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
import binascii
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
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

_AX_HELPER_SOURCE = Path(__file__).with_name("ax_geometry_helper.swift")
_AX_HELPER_BINARY = Path(tempfile.gettempdir()) / "hermes-ax-geometry-helper"

_GEOMETRY_MISSING_ATTRS: Dict[str, Any] = {
    "bounds_available": False,
    "bounds_source": "tree_markdown",
    "geometry_status": "missing",
    "geometry_source": "tree_markdown",
    "bounds_coordinate_space": "unknown",
    "bounds_confidence": 0.0,
}

_BOUND_ALIASES = (
    "bounds_image_pixels",
    "window_bounds",
    "bounds",
    "screen_bounds",
    "frame",
    "rect",
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


def _window_subtree_markdown(markdown: str) -> str:
    """Return the first AXWindow subtree from app-level cua-driver markdown."""
    lines = markdown.splitlines()
    first = next((line for line in lines if line.strip()), "")
    if "AXApplication" not in first:
        return markdown

    start = None
    start_indent = 0
    for i, line in enumerate(lines):
        if "AXWindow" in line:
            start = i
            start_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return markdown

    selected = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip():
            selected.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= start_indent:
            break
        selected.append(line)
    return "\n".join(selected)


def _parse_elements_from_tree(markdown: str) -> List[UIElement]:
    """Parse UIElement list from get_window_state AX tree markdown.

    Handles both the classic ``"label"``-quoted format and the newer
    ``id=Label`` format introduced in cua-driver v0.1.6.
    """
    elements = []
    scoped_markdown = _window_subtree_markdown(markdown)
    for m in _ELEMENT_LINE_RE.finditer(scoped_markdown):
        role = m.group(2)
        if role.startswith("AXMenu"):
            continue
        # group(3) = quoted label (classic); group(4) = id= label (new)
        label = m.group(3) or m.group(4) or ""
        elements.append(UIElement(
            index=int(m.group(1)),
            role=role,
            label=label,
            bounds=(0, 0, 0, 0),
            attributes=dict(_GEOMETRY_MISSING_ATTRS),
        ))
    return elements


def _geometry_attrs(
    *,
    available: bool,
    status: str,
    source: str,
    coordinate_space: str,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "bounds_available": bool(available),
        "geometry_status": status,
        "geometry_source": source,
        "bounds_coordinate_space": coordinate_space,
        "bounds_confidence": float(confidence),
    }


def _coerce_bounds(value: Any) -> Tuple[int, int, int, int]:
    """Accept common rect shapes and return an integer x/y/w/h tuple."""
    if isinstance(value, dict):
        return (
            int(float(value.get("x", value.get("left", 0)) or 0)),
            int(float(value.get("y", value.get("top", 0)) or 0)),
            int(float(value.get("w", value.get("width", 0)) or 0)),
            int(float(value.get("h", value.get("height", 0)) or 0)),
        )
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(float(v or 0)) for v in value)  # type: ignore[return-value]
    return (0, 0, 0, 0)


def _bounds_nonzero(bounds: Tuple[int, int, int, int]) -> bool:
    return bool(bounds[2] > 0 and bounds[3] > 0)


def _structured_geometry_source(root_key: str) -> str:
    return "cua_driver.ui_elements" if root_key == "ui_elements" else "cua_driver.elements"


def _image_dimensions_from_b64(b64: str) -> Tuple[int, int]:
    """Best-effort image dimension sniff for PNG/JPEG base64 payloads."""
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        return 0, 0
    return _image_dimensions_from_bytes(raw)


def _window_bounds_tuple(w: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return _coerce_bounds(w.get("bounds"))


def _window_area(w: Dict[str, Any]) -> int:
    _x, _y, width, height = _window_bounds_tuple(w)
    return max(0, int(width or 0)) * max(0, int(height or 0))


def _looks_like_real_window(w: Dict[str, Any]) -> bool:
    """Heuristic filter for Dock thumbnails/menu-bar pseudo windows."""
    _x, _y, width, height = _window_bounds_tuple(w)
    width = int(width or 0)
    height = int(height or 0)
    if width <= 0 or height <= 0:
        return True
    if height <= 40:
        return False
    if width <= 96 and height <= 96:
        return False
    if _window_area(w) < 50_000:
        return False
    return True


def _choose_target_window(windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the best actual content window from z-sorted candidates."""
    if not windows:
        return None
    real_windows = [w for w in windows if _looks_like_real_window(w)]
    if real_windows:
        return next((w for w in real_windows if not w.get("off_screen")), real_windows[0])
    return next((w for w in windows if not w.get("off_screen")), windows[0])


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


def _compile_ax_geometry_helper() -> Optional[str]:
    """Build the read-only Swift AX helper on demand."""
    if not _is_macos() or not _AX_HELPER_SOURCE.exists():
        return None
    swiftc = shutil.which("swiftc")
    if not swiftc:
        return None
    try:
        if (
            _AX_HELPER_BINARY.exists()
            and _AX_HELPER_BINARY.stat().st_mtime >= _AX_HELPER_SOURCE.stat().st_mtime
        ):
            return str(_AX_HELPER_BINARY)
        result = subprocess.run(
            [swiftc, str(_AX_HELPER_SOURCE), "-o", str(_AX_HELPER_BINARY)],
            text=True,
            capture_output=True,
            timeout=12,
        )
        if result.returncode != 0:
            logger.debug("AX geometry helper compile failed: %s", result.stderr.strip())
            return None
        return str(_AX_HELPER_BINARY)
    except Exception as e:
        logger.debug("AX geometry helper compile failed: %s", e)
        return None


def _run_ax_geometry_helper(
    *,
    pid: int,
    window_title: str = "",
    window_index: int = 0,
    window_bounds: Optional[Tuple[int, int, int, int]] = None,
    max_depth: int = 5,
    max_nodes: int = 300,
    timeout: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Return read-only AX geometry JSON for a process/window, best-effort."""
    helper = _compile_ax_geometry_helper()
    if not helper:
        return None
    args = [
        helper,
        "--pid", str(pid),
        "--max-depth", str(max_depth),
        "--max-nodes", str(max_nodes),
    ]
    if window_bounds:
        args.extend(["--window-bounds", ",".join(str(int(v)) for v in window_bounds)])
    if window_title:
        args.extend(["--window-title", window_title])
    else:
        args.extend(["--window-index", str(window_index)])
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.debug("AX geometry helper timed out for pid=%s", pid)
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        logger.debug("AX geometry helper failed for pid=%s: %s", pid, e)
        return {"ok": False, "error": str(e)}
    payload_text = result.stdout.strip()
    if not payload_text:
        logger.debug("AX geometry helper returned no JSON: %s", result.stderr.strip())
        return {"ok": False, "error": "empty_output", "stderr": result.stderr.strip()}
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        logger.debug("AX geometry helper returned invalid JSON: %s", payload_text[:200])
        return {"ok": False, "error": "invalid_json", "stderr": result.stderr.strip()}
    if result.returncode != 0:
        payload.setdefault("ok", False)
        if payload.get("error") == "window_not_found" and window_title:
            retry_args = [
                helper,
                "--pid", str(pid),
                "--window-index", str(window_index),
                "--max-depth", str(max_depth),
                "--max-nodes", str(max_nodes),
            ]
            try:
                retry = subprocess.run(retry_args, text=True, capture_output=True, timeout=timeout)
                retry_text = retry.stdout.strip()
                if retry_text:
                    retry_payload = json.loads(retry_text)
                    if retry.returncode == 0 or retry_payload.get("ok"):
                        retry_payload.setdefault("selection_fallback", "window_index_after_title_miss")
                        return retry_payload
            except Exception as e:
                logger.debug("AX geometry helper title fallback failed for pid=%s: %s", pid, e)
    return payload


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _norm_role(value: Any) -> str:
    role = _norm_text(value)
    if role.startswith("ax"):
        role = role[2:]
    return re.sub(r"[^a-z0-9]", "", role)


def _role_compatible(cua_role: str, ax_role: str) -> bool:
    left = _norm_role(cua_role)
    right = _norm_role(ax_role)
    if not left or not right:
        return False
    if left == right:
        return True
    aliases = {
        "textarea": {"text", "textfield", "textview"},
        "textfield": {"text", "textarea", "textview"},
        "statictext": {"text"},
        "checkbox": {"check box"},
    }
    return right in aliases.get(left, set()) or left in aliases.get(right, set())


def _label_matches(cua_label: str, ax_label: str) -> bool:
    left = _norm_text(cua_label)
    right = _norm_text(ax_label)
    if not left or not right:
        return False
    return left == right or (len(left) >= 3 and left in right) or (len(right) >= 3 and right in left)


def _ax_node_label(node: Dict[str, Any]) -> str:
    for key in ("text", "title", "description", "value", "placeholder", "help", "identifier", "role_description"):
        value = node.get(key)
        if value not in (None, "", []):
            return str(value)
    attrs = node.get("attributes")
    if isinstance(attrs, dict):
        for key in ("AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue", "AXHelp", "AXIdentifier", "AXRoleDescription"):
            value = attrs.get(key)
            if value not in (None, "", []):
                return str(value)
    return ""


def _ax_node_bounds(node: Dict[str, Any]) -> Tuple[int, int, int, int]:
    position = node.get("position") or node.get("AXPosition")
    size = node.get("size") or node.get("AXSize")
    if isinstance(position, dict) and isinstance(size, dict):
        return (
            int(float(position.get("x", 0) or 0)),
            int(float(position.get("y", 0) or 0)),
            int(float(size.get("width", size.get("w", 0)) or 0)),
            int(float(size.get("height", size.get("h", 0)) or 0)),
        )
    rect = node.get("bounds") or node.get("frame") or node.get("rect")
    return _coerce_bounds(rect)


def _flatten_ax_nodes(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    roots = payload.get("roots") or []
    out: List[Dict[str, Any]] = []

    def visit(node: Any, order: int) -> int:
        if not isinstance(node, dict):
            return order
        bounds = _ax_node_bounds(node)
        if _bounds_nonzero(bounds):
            item = dict(node)
            item["_bounds_screen"] = bounds
            item["_label"] = _ax_node_label(node)
            item["_role"] = str(node.get("role") or "")
            item["_order"] = order
            out.append(item)
            order += 1
        for child in node.get("children") or []:
            order = visit(child, order)
        return order

    order = 0
    for root in roots:
        order = visit(root, order)
    return out


def _screen_to_window_bounds(
    screen_bounds: Tuple[int, int, int, int],
    *,
    window_bounds: Tuple[int, int, int, int],
    capture_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    sx, sy, sw, sh = screen_bounds
    wx, wy, ww, wh = window_bounds
    cw, ch = capture_size
    local_x = sx - wx
    local_y = sy - wy
    if ww > 0 and wh > 0 and cw > 0 and ch > 0 and (cw != ww or ch != wh):
        scale_x = cw / ww
        scale_y = ch / wh
        return (
            int(round(local_x * scale_x)),
            int(round(local_y * scale_y)),
            int(round(sw * scale_x)),
            int(round(sh * scale_y)),
        )
    return (int(local_x), int(local_y), int(sw), int(sh))


def _needs_geometry_fallback(elements: List[UIElement]) -> bool:
    return any(not _bounds_nonzero(e.bounds) or e.attributes.get("bounds_available") is False for e in elements)


def _merge_ax_geometry(
    elements: List[UIElement],
    payload: Dict[str, Any],
    *,
    window_bounds: Tuple[int, int, int, int],
    capture_size: Tuple[int, int],
) -> int:
    """Conservatively fill missing Cua element bounds from AX helper nodes."""
    candidates = _flatten_ax_nodes(payload)
    used: set[int] = set()
    merged = 0

    for element in elements:
        if _bounds_nonzero(element.bounds) and element.attributes.get("bounds_available") is not False:
            continue
        role_matches = [
            i for i, node in enumerate(candidates)
            if i not in used and _role_compatible(element.role, str(node.get("_role") or ""))
        ]
        if not role_matches:
            element.attributes.setdefault("geometry_match_error", "no_role_match")
            continue

        label = element.label
        if label:
            label_matches = [
                i for i in role_matches
                if _label_matches(label, str(candidates[i].get("_label") or ""))
            ]
            if len(label_matches) == 1:
                chosen = label_matches[0]
            elif len(label_matches) > 1:
                element.attributes["geometry_match_error"] = "ambiguous"
                continue
            else:
                element.attributes.setdefault("geometry_match_error", "no_label_match")
                continue
        else:
            chosen = role_matches[0]

        node = candidates[chosen]
        element.bounds = _screen_to_window_bounds(
            node["_bounds_screen"],
            window_bounds=window_bounds,
            capture_size=capture_size,
        )
        if not _bounds_nonzero(element.bounds):
            element.attributes["geometry_match_error"] = "zero_after_conversion"
            continue
        used.add(chosen)
        element.attributes.pop("geometry_match_error", None)
        element.attributes.update(_geometry_attrs(
            available=True,
            status="derived",
            source="hermes_ax_map",
            coordinate_space="window_logical",
            confidence=0.72,
        ))
        element.attributes["geometry_ax_role"] = node.get("_role")
        element.attributes["geometry_ax_label"] = node.get("_label")
        merged += 1
    return merged


def _is_window_content_element(element: UIElement) -> bool:
    """Return false for app-level menu structures outside a window screenshot."""
    return element.role not in {"AXMenuBar", "AXMenuBarItem", "AXMenu"}


def _normalize_structured_screen_bounds(
    elements: List[UIElement],
    *,
    window_bounds: Tuple[int, int, int, int],
    capture_size: Tuple[int, int],
) -> None:
    """Convert structured Cua AXPosition+AXSize screen rects to window-local rects."""
    for element in elements:
        source = str(element.attributes.get("geometry_source") or "")
        bounds_source = str(element.attributes.get("bounds_source") or "")
        if source != "cua_driver.elements" or bounds_source != "AXPosition+AXSize":
            continue
        element.bounds = _screen_to_window_bounds(
            element.bounds,
            window_bounds=window_bounds,
            capture_size=capture_size,
        )
        element.attributes.update(_geometry_attrs(
            available=_bounds_nonzero(element.bounds),
            status="direct" if _bounds_nonzero(element.bounds) else "missing",
            source="cua_driver.elements",
            coordinate_space="window_logical",
            confidence=1.0 if _bounds_nonzero(element.bounds) else 0.0,
        ))


def _elements_from_ax_payload(
    payload: Dict[str, Any],
    *,
    window_bounds: Tuple[int, int, int, int],
    capture_size: Tuple[int, int],
    max_elements: int = 120,
) -> List[UIElement]:
    """Create coordinate-addressable elements directly from AX helper output."""
    out: List[UIElement] = []
    for node in _flatten_ax_nodes(payload):
        role = str(node.get("_role") or "")
        if role.startswith("AXMenu") or role == "AXApplication":
            continue
        bounds = _screen_to_window_bounds(
            node["_bounds_screen"],
            window_bounds=window_bounds,
            capture_size=capture_size,
        )
        if not _bounds_nonzero(bounds):
            continue
        label = str(node.get("_label") or "")
        out.append(UIElement(
            index=len(out) + 1,
            role=role,
            label=label,
            bounds=bounds,
            attributes={
                **_geometry_attrs(
                    available=True,
                    status="derived",
                    source="hermes_ax_map",
                    coordinate_space="window_logical",
                    confidence=0.68,
                ),
                "action_backend": "coordinate",
                "synthetic_element": True,
                "geometry_ax_role": role,
                "geometry_ax_label": label,
            },
        ))
        if len(out) >= max_elements:
            break
    return out


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

        params = StdioServerParameters(
            command=_CUA_DRIVER_CMD,
            args=_CUA_DRIVER_ARGS,
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
        self._active_window: Optional[Dict[str, Any]] = None
        self._active_elements: Dict[int, UIElement] = {}
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
    def _set_active_window(self, target: Dict[str, Any]) -> None:
        self._active_pid = int(target["pid"])
        self._active_window_id = int(target["window_id"])
        self._active_window = dict(target)

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        """Capture the frontmost on-screen window (optionally filtered by app name).

        Maps hermes `capture(mode, app)` → cua-driver `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision).
        """
        # Step 1: enumerate on-screen windows to find target pid/window_id.
        lw_out = self._session.call_tool("list_windows", {"on_screen_only": True})

        # Prefer structuredContent.windows (MCP 2024-11-05+); fall back to
        # text-line parsing for older cua-driver builds.
        sc = lw_out.get("structuredContent") or {}
        raw_windows = sc.get("windows") if sc else None
        if raw_windows is None and isinstance(lw_out.get("data"), dict):
            raw_windows = lw_out["data"].get("windows")
        if raw_windows:
            windows = [
                {
                    "app_name": w.get("app_name", ""),
                    "pid": int(w["pid"]),
                    "window_id": int(w["window_id"]),
                    "off_screen": not w.get("is_on_screen", True),
                    "title": w.get("title", ""),
                    "z_index": w.get("z_index", 0),
                    "bounds": _window_bounds_tuple(w),
                }
                for w in raw_windows
            ]
            # Sort by z_index descending (lowest z_index = frontmost on macOS).
            windows.sort(key=lambda w: w["z_index"])
        else:
            raw_text = lw_out["data"] if isinstance(lw_out["data"], str) else ""
            windows = _parse_windows_from_text(raw_text)

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
        self._set_active_window(target)
        app_name = target["app_name"]
        # Record the resolved app name so capture_after= follow-ups can re-target
        # the same app rather than falling back to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name

        # Step 2: capture.
        png_b64: Optional[str] = None
        elements: List[UIElement] = []
        width = height = 0
        window_title = str(target.get("title") or "")
        bx, by, bw, bh = _window_bounds_tuple(target)
        if bw and bh:
            width, height = int(bw), int(bh)

        if mode == "vision":
            # screenshot tool: just the PNG, no AX walk.
            sc_out = self._session.call_tool(
                "screenshot",
                {"window_id": self._active_window_id, "format": "jpeg", "quality": 85},
            )
            if sc_out["images"]:
                png_b64 = sc_out["images"][0]
        else:
            # get_window_state: AX tree + optional screenshot.
            gws_out = self._session.call_tool(
                "get_window_state",
                {"pid": self._active_pid, "window_id": self._active_window_id},
            )
            state: Dict[str, Any] = {}
            if isinstance(gws_out.get("data"), dict):
                state.update(gws_out["data"])
            if isinstance(gws_out.get("structuredContent"), dict):
                state.update(gws_out["structuredContent"] or {})

            text = gws_out["data"] if isinstance(gws_out.get("data"), str) else ""
            summary, tree_from_text = _split_tree_text(text) if text else ("", "")
            tree = state.get("tree_markdown") or state.get("tree") or tree_from_text

            # Parse element count from summary e.g. "✅ AppName — 42 elements, turn 3..."
            m = re.search(r'(\d+)\s+elements?', summary)
            raw_element_source = ""
            raw_elements = None
            for key in ("elements", "ui_elements"):
                if isinstance(state.get(key), list):
                    raw_element_source = _structured_geometry_source(key)
                    raw_elements = state.get(key)
                    break
            if isinstance(raw_elements, list):
                elements = [
                    _parse_element(e, source=raw_element_source)
                    for e in raw_elements
                    if isinstance(e, dict)
                ]
                _normalize_structured_screen_bounds(
                    elements,
                    window_bounds=(int(bx or 0), int(by or 0), int(bw or 0), int(bh or 0)),
                    capture_size=(int(width or 0), int(height or 0)),
                )
                elements = [e for e in elements if _is_window_content_element(e)]
            elif tree:
                elements = _parse_elements_from_tree(tree)

            if gws_out["images"]:
                png_b64 = gws_out["images"][0]

            # Extract window title from the AX tree first AXWindow line.
            wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
            if wt:
                window_title = wt.group(1)
            elif state.get("window_title") or state.get("title"):
                window_title = str(state.get("window_title") or state.get("title") or "")

            if state.get("screenshot_width") and state.get("screenshot_height"):
                width, height = int(state["screenshot_width"]), int(state["screenshot_height"])
            elif state.get("width") and state.get("height") and not (width and height):
                width, height = int(state["width"]), int(state["height"])

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

        if mode != "vision" and (not elements or _needs_geometry_fallback(elements)):
            try:
                helper_payload = _run_ax_geometry_helper(
                    pid=self._active_pid or int(target.get("pid", 0) or 0),
                    window_title=window_title,
                    window_index=0,
                    window_bounds=(int(bx or 0), int(by or 0), int(bw or 0), int(bh or 0)),
                    max_depth=int(os.environ.get("HERMES_AX_GEOMETRY_MAX_DEPTH", "5")),
                    max_nodes=int(os.environ.get("HERMES_AX_GEOMETRY_MAX_NODES", "300")),
                    timeout=float(os.environ.get("HERMES_AX_GEOMETRY_TIMEOUT", "3")),
                )
                if helper_payload and helper_payload.get("ok"):
                    if elements:
                        merged = _merge_ax_geometry(
                            elements,
                            helper_payload,
                            window_bounds=(int(bx or 0), int(by or 0), int(bw or 0), int(bh or 0)),
                            capture_size=(int(width or 0), int(height or 0)),
                        )
                        if merged:
                            logger.debug("AX geometry helper filled %s/%s element bounds", merged, len(elements))
                    else:
                        elements = _elements_from_ax_payload(
                            helper_payload,
                            window_bounds=(int(bx or 0), int(by or 0), int(bw or 0), int(bh or 0)),
                            capture_size=(int(width or 0), int(height or 0)),
                        )
                        if elements:
                            logger.debug("AX geometry helper synthesized %s coordinate-backed elements", len(elements))
                elif helper_payload:
                    reason = str(helper_payload.get("error") or "helper_failed")
                    for e in elements:
                        if not _bounds_nonzero(e.bounds):
                            e.attributes.setdefault("geometry_helper_error", reason)
            except Exception as e:
                logger.debug("AX geometry fallback failed: %s", e)
                for element in elements:
                    if not _bounds_nonzero(element.bounds):
                        element.attributes.setdefault("geometry_helper_error", str(e))

        self._active_elements = {e.index: e for e in elements}
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

    def capture_active(self, mode: str = "som") -> CaptureResult:
        """Capture the currently selected target without reselecting frontmost window."""
        if self._active_window:
            target = dict(self._active_window)
            self._set_active_window(target)
            app_name = target.get("app_name", "")
            bx, by, bw, bh = _window_bounds_tuple(target)
            if mode == "vision":
                sc_out = self._session.call_tool(
                    "screenshot",
                    {"window_id": self._active_window_id, "format": "jpeg", "quality": 85},
                )
                png_b64 = sc_out["images"][0] if sc_out["images"] else None
                width = height = png_bytes_len = 0
                if png_b64:
                    try:
                        raw = base64.b64decode(png_b64, validate=False)
                        png_bytes_len = len(raw)
                        width, height = _image_dimensions_from_bytes(raw)
                    except Exception:
                        png_bytes_len = len(png_b64) * 3 // 4
                return CaptureResult(
                    mode=mode,
                    width=width or int(bw or 0),
                    height=height or int(bh or 0),
                    png_b64=png_b64,
                    elements=[],
                    app=app_name,
                    window_title=str(target.get("title") or ""),
                    png_bytes_len=png_bytes_len,
                )
            # Preserve the selected target by asking capture to re-target the
            # same resolved app rather than falling back to frontmost.
            return self.capture(mode=mode, app=app_name or self._last_app)
        return self.capture(mode=mode)

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
            coordinate_element = self._active_elements.get(int(element)) if self._active_elements else None
            if (
                coordinate_element is not None
                and coordinate_element.attributes.get("action_backend") == "coordinate"
                and _bounds_nonzero(coordinate_element.bounds)
            ):
                cx, cy = coordinate_element.center()
                args["x"] = cx
                args["y"] = cy
                if self._active_window_id is not None:
                    args["window_id"] = self._active_window_id
            else:
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


def _parse_element(d: Dict[str, Any], *, source: str = "cua_driver.elements") -> UIElement:
    bounds_key = ""
    bounds = (0, 0, 0, 0)
    for key in _BOUND_ALIASES:
        if key in d and d.get(key) is not None:
            candidate = _coerce_bounds(d.get(key))
            if _bounds_nonzero(candidate) or not bounds_key:
                bounds_key = key
                bounds = candidate
            if _bounds_nonzero(candidate):
                break

    label = ""
    for key in ("label", "title", "description", "value", "identifier", "role_description"):
        value = d.get(key)
        if value not in (None, ""):
            label = str(value)
            break

    attrs = {
        k: v for k, v in d.items()
        if k not in {
            "index", "element_index", "elementIndex", "role", "label", "title",
            "description", "value", "identifier", "role_description", "app", "pid",
            "window_id", "windowId", *_BOUND_ALIASES,
        }
    }
    attrs["geometry_bounds_key"] = bounds_key or None
    if _bounds_nonzero(bounds):
        coord_space = (
            "screen_logical"
            if bounds_key == "screen_bounds" or d.get("bounds_source") == "AXPosition+AXSize"
            else "window_logical"
        )
        attrs.update(_geometry_attrs(
            available=True,
            status="direct",
            source=source,
            coordinate_space=coord_space,
            confidence=1.0,
        ))
    else:
        attrs.update(_geometry_attrs(
            available=False,
            status="missing",
            source=source if source.startswith("cua_driver.") else "none",
            coordinate_space="unknown",
            confidence=0.0,
        ))

    return UIElement(
        index=int(d.get("index", d.get("element_index", d.get("elementIndex", 0))) or 0),
        role=str(d.get("role", "") or ""),
        label=label,
        bounds=bounds,
        app=str(d.get("app", "") or ""),
        pid=int(d.get("pid", 0) or 0),
        window_id=int(d.get("window_id", d.get("windowId", 0)) or 0),
        attributes=attrs,
    )
