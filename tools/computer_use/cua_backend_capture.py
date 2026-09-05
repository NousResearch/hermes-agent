"""Capture side of the cua-driver backend: window discovery, capture-target selection and capture()/list_windows()/
list_apps()/focus_app() (mixed into ``CuaDriverBackend``). Logger name stays ``tools.computer_use.cua_backend`` so
log-based tests and operators see one backend logger."""

from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, UIElement
from tools.computer_use.cua_backend_input import _BTF_UNSUPPORTED_MSG
from tools.computer_use.cua_backend_parse import (
    _apps_from_windows, _image_dimensions_from_bytes, _image_from_tool_result, _ingest_windows, _is_placeholder_id,
    _is_real_app_window, _parse_elements_from_structured, _parse_elements_from_tree, _parse_xprop_net_active_window,
    _positive_int, _split_tree_text, _windows_from_tool_result, _z_index_uninformative,
)

logger = logging.getLogger("tools.computer_use.cua_backend")

# Whole-screen intents: app="screen"/... -> composited `get_desktop_state` (pixels only); app="desktop" -> the OS
# shell window via list_windows, WITH interactable elements (icons, taskbar).
_FULL_SCREEN_SENTINELS = {"screen", "fullscreen", "full screen", "all"}
_DESKTOP_SHELL_SENTINELS = {"desktop"}
# Shell window identifiers (substring of app_name + title, case-insensitive). Windows: Progman/WorkerW =
# desktop, Shell_TrayWnd = taskbar; macOS: Finder/Dock. The backdrop subset is preferred over the taskbar.
_DESKTOP_WINDOW_NAMES = ("progman", "workerw", "program manager", "shell_traywnd", "taskbar", "finder", "desktop", "dock")
_DESKTOP_BACKDROP_NAMES = ("progman", "workerw", "program manager", "finder", "desktop")
_NO_DESKTOP_WINDOW_MSG = ("<no desktop/shell window found for app={app!r}; cua-driver captures one window at a time "
                          "and exposes no whole-virtual-desktop or per-monitor capture. Call list_apps / "
                          "capture(app='<AppName>') to target a specific window instead. On Windows the taskbar is "
                          "'Shell_TrayWnd' and the desktop is 'Progman'.>")
_NO_APP_MATCH_MSG = ("<no on-screen window matched app={app!r}; call list_apps to see available app names or bundle "
                     "IDs (macOS reports localized names, e.g. '計算機' instead of 'Calculator'; some Linux/Qt apps "
                     "only resolve via list_apps metadata)>")
_NO_DESKTOP_IMAGE_MSG = ("<get_desktop_state returned no image; the driver may predate the desktop capture lane — "
                         "try capture(app='<AppName>') for a specific window>")
_FULL_SCREEN_NOTE = ("full-screen capture has no interactable elements; to act on what you see, call "
                     "capture(app='<AppName>') for that app's clickable element list, or capture(app='desktop') for "
                     "the desktop shell (wallpaper icons / taskbar) with elements")

def _linux_x11_active_window_id() -> Optional[int]:
    """Best-effort read of ``_NET_ACTIVE_WINDOW`` via xprop. Never raises."""
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    try:
        proc = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=2, check=False, stdin=subprocess.DEVNULL)
    except Exception:
        return None
    return _parse_xprop_net_active_window(proc.stdout or "") if proc.returncode == 0 else None

def _select_capture_target(windows: List[Dict[str, Any]], *, app_requested: bool,
                           exact_target: bool = False, allow_local_probe: bool = True) -> Dict[str, Any]:
    """Best window from z-sorted (frontmost-first) list_windows output. Unqualified default captures on
    Linux (no app filter, no exact target) skip desktop/shell helper windows first — targetable but capture
    as empty — and when every remaining candidate shares one ``z_index`` (the common X11 case)
    ``_NET_ACTIVE_WINDOW`` beats list order. Exact-target captures never pay for the ``xprop`` probe.

    Callers pass windows already sorted by ``z_index`` descending (higher = frontmost). When ordering is
    informative, keep that frontmost contract. See #58026.

    ``allow_local_probe`` is False for remote CUA sessions: xprop would probe the agent's local X11
    display, not the remote desktop the cua-driver is driving.
    """
    pool = [w for w in windows if not w["off_screen"]]
    if allow_local_probe and not exact_target and not app_requested and sys.platform == "linux":
        pool = [w for w in pool if _is_real_app_window(w)] or pool
        if pool and _z_index_uninformative(pool):
            active_id = _linux_x11_active_window_id()
            if active_id is not None and (hit := [w for w in pool if w.get("window_id") == active_id]):
                return hit[0]
    return pool[0] if pool else windows[0]

def _sorted_windows(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalised list_windows rows, ``z_index`` DESCENDING (frontmost first = default capture/focus target)."""
    return sorted(_ingest_windows(_windows_from_tool_result(out)), key=lambda w: w["z_index"], reverse=True)

def _tree_and_title(out: Dict[str, Any]) -> Tuple[str, str]:
    """``(tree_markdown, window_title)`` from a get_window_state result."""
    tree = _split_tree_text(data if isinstance((data := out.get("data")), str) else "")[1]
    return tree, (match.group(1) if (match := re.search(r'AXWindow\s+"([^"]+)"', tree)) else "")

def _gws_is_empty(out: Dict[str, Any]) -> bool:
    """True when a get_window_state result carries neither a screenshot nor a parseable tree. Modern
    drivers put the payload in structuredContent with no markdown tree — that is NOT empty."""
    sc_ = out.get("structuredContent") or {}
    return not (out.get("images") or sc_.get("elements") or sc_.get("screenshot_png_b64")
                or _tree_and_title(out)[0].strip())

def _png_metrics(png_b64: str, width: int, height: int) -> Tuple[int, int, int]:
    """``(png_bytes_len, width, height)``; the sniffed size wins when the bytes carry a readable PNG/JPEG header."""
    try:
        raw = base64.b64decode(png_b64, validate=False)
    except Exception:
        return len(png_b64) * 3 // 4, width, height
    detected_width, detected_height = _image_dimensions_from_bytes(raw)
    return len(raw), *((detected_width, detected_height) if detected_width and detected_height else (width, height))

def _is_desktop_window(w: Dict[str, Any], names: Tuple[str, ...] = _DESKTOP_WINDOW_NAMES) -> bool:
    return any(name in f"{w.get('app_name', '')} {w.get('title', '')}".lower() for name in names)


class _CaptureMixin:
    """capture()/list_windows()/list_apps()/focus_app() and their window-discovery helpers."""

    @contextmanager
    def _disarming(self) -> Iterator[None]:
        """Forget the sticky target when the wrapped capture-stage step raises."""
        try:
            yield
        except Exception:
            self._clear_active_target()
            raise

    def _failed_capture(self, mode: str, message: str = "") -> CaptureResult:
        """Return an empty capture after disarming any prior target context."""
        self._clear_active_target()
        return CaptureResult(mode=mode, width=0, height=0, window_title=message)

    def _call_capture_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a capture-stage tool and disarm state on transport or logical failure."""
        with self._disarming():
            out = self._session.call_tool(name, args)
            if out.get("isError") is True:
                message = out.get("data")
                raise RuntimeError(f"cua-driver {name} failed"
                                   + (f": {message}" if isinstance(message, str) and message else ""))
        return out

    def _cli_refetch(self, name: str, args: Dict[str, Any], timeout: float, what: str,
                     warning: str, *warning_args: Any) -> Optional[Dict[str, Any]]:
        """MCP came back empty/imageless without raising: log *warning*, then a one-shot call over the CLI
        transport (different daemon socket). None on failure."""
        logger.warning(warning, *warning_args)
        try:
            cli_out = self._session._call_tool_via_cli(name, args, timeout)
        except Exception as cli_exc:
            logger.error("cua-driver CLI re-fetch for %s failed: %s", what, cli_exc)
            return None
        if cli_out.get("isError") is not True:
            return cli_out
        if name == "list_windows":
            logger.error("cua-driver CLI re-fetch for list_windows returned an error")
        self._clear_active_target()
        return None

    def _fetch_or_refetch(self, name: str, args: Dict[str, Any], timeout: float, what: str,
                          empty: Callable[[Dict[str, Any]], bool], warning: str, *warning_args: Any) -> Dict[str, Any]:
        """``_call_capture_tool`` whose result, when *empty*, is replaced by a non-empty CLI re-fetch (the
        MCP result stands when the CLI fails too)."""
        out = self._call_capture_tool(name, args)
        cli_out = self._cli_refetch(name, args, timeout, what, warning, *warning_args) if empty(out) else None
        return cli_out if cli_out is not None and not empty(cli_out) else out

    def list_windows(self) -> List[Dict[str, Any]]:
        """Visible windows frontmost-first, re-fetching over the CLI transport when MCP returns nothing."""
        return _sorted_windows(self._fetch_or_refetch(
            "list_windows", {"on_screen_only": True, "session": self._session_id}, 20.0, "list_windows",
            lambda out: not _sorted_windows(out),
            "cua-driver list_windows returned no windows over MCP; re-fetching via CLI transport"))

    def _match_windows_for_app(self, windows: List[Dict[str, Any]], app: str) -> List[Dict[str, Any]]:
        """Resolve ``app=``: exact window names, then exact list_apps aliases (Linux ``list_windows`` can
        omit the app name that ``list_apps`` keeps), then substrings — querying ``Code`` must not silently
        select ``Visual Studio Code`` because it is frontmost."""
        app_lower = app.strip().lower()
        _name = lambda w: str(w.get("app_name", "")).lower()  # noqa: E731
        direct_exact = [w for w in windows if app_lower and app_lower == _name(w).strip()]
        if not app_lower or direct_exact:
            return direct_exact
        try:
            running_apps = self.list_apps()
        except Exception as exc:
            # A title can still be the only usable identity on X11 when app enumeration is unavailable,
            # so keep the title fallback below.
            logger.debug("computer_use list_apps fallback failed for %r: %s", app, exc)
            running_apps = []
        exact_pids, partial_pids = set(), set()
        for raw_app in running_apps:
            pid = _positive_int(raw_app.get("pid")) if isinstance(raw_app, dict) else None
            if pid is not None and raw_app.get("running") is not False:
                aliases = {value.strip().lower() for key in ("bundle_id", "bundleId", "name", "app_name", "display_name")
                           if isinstance((value := raw_app.get(key)), str) and value.strip()}
                if app_lower in aliases:
                    exact_pids.add(pid)
                elif any(app_lower in alias for alias in aliases):
                    partial_pids.add(pid)
        # Some X11 backends expose a title but no app name. Restrict the final fallback to nameless rows so
        # a localized app name is not overridden merely because its title happens to be in the caller's language.
        tiers = ([w for w in windows if w.get("pid") in exact_pids],
                 [w for w in windows if app_lower in _name(w)],
                 [w for w in windows if w.get("pid") in partial_pids],
                 [w for w in windows if not _name(w).strip() and app_lower in str(w.get("title", "")).lower()])
        return next((matched for matched in tiers if matched), [])

    def _resolve_capture_windows(self, mode: str, app: Optional[str], pid: Optional[int],
                                 window_id: Optional[int]) -> "List[Dict[str, Any]] | CaptureResult":
        """Candidate windows for capture(), or a failed CaptureResult."""
        if pid is not None or window_id is not None:
            # An exact pid/window pair is both the stable capture_after target and the escape hatch when
            # discovery is unavailable on X11.
            if pid is None or window_id is None:
                return self._failed_capture(mode, "<capture targeting requires both pid and window_id>")
            if (target_pid := _positive_int(pid)) is None or (target_window_id := _positive_int(window_id)) is None:
                return self._failed_capture(mode, "<capture targeting requires positive integer pid and window_id>")
            return [{"app_name": app or "", "pid": target_pid, "window_id": target_window_id, "off_screen": False,
                     "title": "", "z_index": 0}]
        with self._disarming():
            windows = self.list_windows()
        if not windows:
            # Diagnose instead of a bare 0x0: the dominant real-world cause on Linux is a locked desktop session.
            from tools.computer_use import cua_backend as _cb
            return self._failed_capture(mode, _cb._empty_discovery_reason(
                remote=getattr(self, "_remote_config", None) is not None))
        if not app:
            return windows
        if app.strip().lower() in _DESKTOP_SHELL_SENTINELS:
            # Desktop-shell request: the OS shell window WITH its interactable elements (desktop icons), so
            # "click the taskbar" works. Prefer the backdrop (Progman/WorkerW/Finder) over the taskbar so the
            # capture shows the full desktop rather than the task strip.
            desktop = sorted((w for w in windows if _is_desktop_window(w)),
                             key=lambda w: not _is_desktop_window(w, _DESKTOP_BACKDROP_NAMES))
            return desktop or self._failed_capture(mode, _NO_DESKTOP_WINDOW_MSG.format(app=app))
        # When the filter matches nothing, say so instead of silently capturing the frontmost window — on
        # macOS list_windows returns the localized app name (e.g. "計算機"), so `app="Calculator"` legitimately misses.
        return self._match_windows_for_app(windows, app) or self._failed_capture(mode, _NO_APP_MATCH_MSG.format(app=app))

    def _gws_args(self) -> Dict[str, Any]:
        return {"pid": self._active_pid, "window_id": self._active_window_id, "session": self._session_id}

    def _capture_vision(self) -> Tuple[Optional[str], Optional[str], List[UIElement], str]:
        """Pixels only, ``elements`` always empty: ``(png_b64, mime, [], window_title)``. Drivers advertising the
        cheaper standalone ``screenshot`` tool use it; current drivers folded PNG capture into ``get_window_state``
        (tree DISCARDED here). Before discovery ran we still try ``screenshot`` first and fall back, so the path
        self-heals on any driver version."""
        png_b64, image_mime_type, window_title = None, None, ""
        if self._session._has_tool("screenshot") or not self._session.capabilities_discovered:
            png_b64, image_mime_type = _image_from_tool_result(self._call_capture_tool("screenshot", {
                "window_id": self._active_window_id, "format": "jpeg", "quality": 85, "session": self._session_id}))
        if not png_b64:
            # "Unknown tool: screenshot" or an empty image part -> get_window_state. The title is cheap
            # and useful; `elements` stays empty by contract.
            gws_out = self._call_capture_tool("get_window_state", self._gws_args())
            (png_b64, image_mime_type), (_, window_title) = _image_from_tool_result(gws_out), _tree_and_title(gws_out)
        if not png_b64:
            cli_out = self._cli_refetch(
                "get_window_state", self._gws_args(), 30.0, "vision screenshot",
                "cua-driver vision capture returned no image over MCP (window_id=%s); re-fetching via CLI transport",
                self._active_window_id) or {}
            if cli_out.get("images"):
                png_b64, image_mime_type = cli_out["images"][0], "image/png"
        return png_b64, image_mime_type, [], window_title

    def _capture_window_state(self) -> Tuple[Optional[str], Optional[str], List[UIElement], str]:
        """AX tree + screenshot. Returns ``(png_b64, mime, elements, window_title)``."""
        # A flaky bridge can return a degenerate result (no screenshot AND no parseable tree) WITHOUT raising
        # — a silent 0x0 to the model. Distinct from the EAGAIN path handled in call_tool: here MCP "succeeded".
        gws_out = self._fetch_or_refetch(
            "get_window_state", self._gws_args(), 30.0, "get_window_state", _gws_is_empty,
            "cua-driver get_window_state returned an empty result over MCP (pid=%s window_id=%s); re-fetching via CLI "
            "transport", self._active_pid, self._active_window_id)
        tree, window_title = _tree_and_title(gws_out)
        # Prefer the canonical structuredContent.elements (real frames); the markdown regex fallback yields
        # (0,0,0,0) bounds.
        # Surface 2 of NousResearch/hermes-agent#47072: prefer the canonical structuredContent.elements
        # array (trycua/cua#1961). Falls back to markdown regex parsing for cua-driver builds that didn't
        # carry the structured shape — those bounds come back (0,0,0,0); the structured path preserves real
        # frames.
        sc_elements = (gws_out.get("structuredContent") or {}).get("elements")
        elements = (_parse_elements_from_structured(sc_elements) if isinstance(sc_elements, list) and sc_elements
                    else _parse_elements_from_tree(tree) if tree else [])
        # Tokens are tied to this snapshot: overwrite the whole map (and clear it when the new capture carries none).
        self._snapshot_tokens = {e.index: e.element_token for e in elements if e.element_token}
        return *_image_from_tool_result(gws_out), elements, window_title

    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult:
        """Capture the frontmost on-screen window or an exact known target: `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision). Only the structured
        ``structuredContent.windows`` shape is supported."""
        # Schema-filler ids (models zero-fill optional properties) must not read as a targeting request.
        pid, window_id = [None if _is_placeholder_id(v) else v for v in (pid, window_id)]
        exact_target = pid is not None or window_id is not None
        # Full-screen lane bypasses enumeration entirely (also keeps screenshots working when Windows UIA
        # enumeration hangs). app='desktop' deliberately does NOT take it: desktop icons stay clickable.
        if not exact_target and app and app.strip().lower() in _FULL_SCREEN_SENTINELS:
            return self._capture_full_screen(mode)
        windows = self._resolve_capture_windows(mode, app, pid, window_id)
        if isinstance(windows, CaptureResult):
            return windows
        self._set_active_target(target := _select_capture_target(
            windows, app_requested=bool(app), exact_target=exact_target,
            allow_local_probe=getattr(self, "_remote_config", None) is None))
        app_name = target["app_name"]
        # Record the resolved app so capture_after= follow-ups re-target the same app rather than falling back
        # to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name or app or ""
        png_b64, image_mime_type, elements, window_title = (
            self._capture_vision() if mode == "vision" else self._capture_window_state())
        png_bytes_len, width, height = _png_metrics(png_b64, 0, 0) if png_b64 else (0, 0, 0)
        return CaptureResult(mode=mode, width=width, height=height, png_b64=png_b64, elements=elements, app=app_name,
                             window_title=window_title, png_bytes_len=png_bytes_len, image_mime_type=image_mime_type)

    def _capture_full_screen(self, mode: str) -> CaptureResult:
        """Composited PrtScn-style grab via `get_desktop_state` (the shell window would only show wallpaper + icons).
        Never enumerates, so it also works when Windows UIA hangs. Pixels only — `elements` is empty and `note` points
        the model at the interactive lanes. ``capture_scope`` is switched to desktop for the call and restored afterwards.

        Bonus resilience (2ndNatureAI, #60081): this lane works even when Windows UIA enumeration
        (`list_windows` / `list_apps`) hangs (trycua/cua#2110/#2113), because it never enumerates.
        """
        self._clear_active_target()
        previous_scope: Optional[str] = None
        try:
            sc = self._session.call_tool("get_config", {"session": self._session_id}, timeout=10.0).get("structuredContent")
            previous_scope = sc["capture_scope"] if isinstance(sc, dict) and isinstance(sc.get("capture_scope"), str) else None
        except Exception as e:
            logger.debug("cua-driver get_config before full-screen capture failed: %s", e)
        _set_scope = lambda value: self._session.call_tool(  # noqa: E731
            "set_config", {"key": "capture_scope", "value": value, "session": self._session_id}, timeout=10.0)
        try:
            if previous_scope != "desktop":
                _set_scope("desktop")
            out = self._call_capture_tool("get_desktop_state", {"session": self._session_id})
        finally:
            if previous_scope and previous_scope != "desktop":
                try:
                    _set_scope(previous_scope)
                except Exception as e:
                    logger.debug("cua-driver restore capture_scope failed: %s", e)
        png_b64, image_mime_type = _image_from_tool_result(out)
        if not png_b64:
            return self._failed_capture(mode, _NO_DESKTOP_IMAGE_MSG)
        sc = out.get("structuredContent") or {}
        png_bytes_len, width, height = _png_metrics(png_b64, int(sc.get("screenshot_width") or sc.get("screen_width") or 0),
                                                    int(sc.get("screenshot_height") or sc.get("screen_height") or 0))
        return CaptureResult(mode="vision", width=width, height=height, png_b64=png_b64, app="screen",
                             window_title="Full screen (composited)", png_bytes_len=png_bytes_len,
                             image_mime_type=image_mime_type, note=_FULL_SCREEN_NOTE)

    def list_apps(self) -> List[Dict[str, Any]]:
        out = self._session.call_tool("list_apps", {"session": self._session_id})
        structured, data = out.get("structuredContent"), out.get("data")
        # structuredContent is canonical; empty lists fall through so a populated compatibility envelope
        # (older drivers, CLI fallback) can still recover, then apps derived from the windows payload.
        candidates = (lambda: structured.get("apps") if isinstance(structured, dict) else None,
                      lambda: data, lambda: data.get("apps") if isinstance(data, dict) else None,
                      lambda: out.get("apps"), lambda: _apps_from_windows(_windows_from_tool_result(out)))
        for candidate in candidates:
            if isinstance((apps := candidate()), list) and apps:
                return apps
        # Old text-only drivers retain a small, name/PID-only fallback.
        return [{"name": m.group(1).strip(), "pid": int(m.group(2))}
                for m in map(re.compile(r'(.+?)\s+\(pid\s+(\d+)\)').search, data.splitlines()) if m] if isinstance(data, str) else []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Pure window-selector (store pid/window_id so later input hits the right process) — background
        automation never needs to raise a window. ``raise_window=True`` is explicit, separately approved,
        and uses the standalone ``bring_to_front`` tool."""
        with self._disarming():
            matched = self._match_windows_for_app(self.list_windows(), app)
        # No silent fallback to the frontmost window: that hides the real failure (often a localized macOS
        # app-name mismatch).
        if not matched:
            self._clear_active_target()
            return ActionResult(ok=False, action="focus_app", message=f"No on-screen window found for app '{app}'.")
        self._set_active_target(target := matched[0])
        self._last_app = target["app_name"] or app  # retained for back-compat diagnostics
        if not raise_window:
            return ActionResult(ok=True, action="focus_app", message=f"Targeted {target['app_name']} (pid "
                                f"{self._active_pid}, window {self._active_window_id}) without raising window.")
        if not self._session._has_tool("bring_to_front"):
            return ActionResult(ok=False, action="focus_app", code="bring_to_front_unsupported", message=_BTF_UNSUPPORTED_MSG)
        focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
        if focused.ok:
            focused.action = "focus_app"
            focused.meta["target_selected"] = True
        return focused
