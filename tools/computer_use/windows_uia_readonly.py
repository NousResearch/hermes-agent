"""Windows UIA read-only backend.

This module is intentionally read-only and side-effect-light. It does not call
click/type/drag, does not focus windows, does not import or use SendInput,
pyautogui, or pynput, and does not mutate native Windows UI. It can enumerate
UIA metadata through an injected test desktop or an optional pywinauto Desktop
object when available.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WindowsUiaReadOnlyCapabilities:
    available: bool
    backend: str = "pywinauto-uia-readonly"
    platform: str = ""
    read_only: bool = True
    mutation_allowed: bool = False
    input_fallback_enabled: bool = False
    optional_dependency: str = "pywinauto"
    reason: str = ""
    supports: tuple[str, ...] = (
        "list_windows",
        "snapshot_tree",
        "element_capabilities",
    )
    forbidden_actions: tuple[str, ...] = (
        "click",
        "drag",
        "type_text",
        "key",
        "set_value",
        "send_input",
        "pyautogui",
        "pynput",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowsUiaElementSnapshot:
    name: str = ""
    automation_id: str = ""
    control_type: str = ""
    class_name: str = ""
    process_id: int | None = None
    hwnd: int | None = None
    bounding_rectangle: tuple[int, int, int, int] | None = None
    is_enabled: bool | None = None
    is_offscreen: bool | None = None
    is_password: bool | None = None
    supported_patterns: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WindowsUiaReadOnlyBackend:
    """Read-only capability/snapshot backend for Windows UIA metadata."""

    def __init__(
        self,
        *,
        platform: str | None = None,
        desktop_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.platform = platform or sys.platform
        self._desktop_factory = desktop_factory

    def capabilities(self) -> WindowsUiaReadOnlyCapabilities:
        if not _is_windows(self.platform):
            return WindowsUiaReadOnlyCapabilities(
                available=False,
                platform=self.platform,
                reason="Windows UIA read-only backend is only meaningful on Windows.",
            )
        if self._desktop_factory is not None:
            return WindowsUiaReadOnlyCapabilities(
                available=True,
                platform=self.platform,
                reason="Injected read-only UIA desktop factory is configured.",
            )
        if importlib.util.find_spec("pywinauto") is None:
            return WindowsUiaReadOnlyCapabilities(
                available=False,
                platform=self.platform,
                reason="Optional pywinauto dependency is not installed; live UIA enumeration is unavailable.",
            )
        return WindowsUiaReadOnlyCapabilities(
            available=True,
            platform=self.platform,
            reason="Optional pywinauto dependency is installed; read-only UIA enumeration is available.",
        )

    def list_windows(self) -> dict[str, Any]:
        caps = self.capabilities()
        windows: list[dict[str, Any]] = []
        if caps.available:
            try:
                windows = [self._snapshot_element(window).to_dict() for window in self._top_level_windows()]
            except Exception as exc:
                return self._error_result(caps, f"UIA read-only window enumeration failed: {exc}", windows=[])
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "windows": windows,
            "reason": caps.reason,
            "next_step": "Use snapshot_tree/element_capabilities for bounded read-only inspection; no click/type/drag here.",
        }

    def snapshot_tree(
        self,
        *,
        root_selector: dict[str, Any] | None = None,
        max_depth: int = 3,
        max_elements: int = 200,
    ) -> dict[str, Any]:
        caps = self.capabilities()
        depth_limit = max(0, min(int(max_depth), 10))
        element_limit = max(1, min(int(max_elements), 1000))
        elements: list[dict[str, Any]] = []
        if caps.available:
            try:
                roots = self._select_roots(root_selector or {})
                for root in roots:
                    self._walk(root, depth=0, max_depth=depth_limit, max_elements=element_limit, out=elements)
                    if len(elements) >= element_limit:
                        break
            except Exception as exc:
                return self._error_result(
                    caps,
                    f"UIA read-only tree snapshot failed: {exc}",
                    root_selector=root_selector or {},
                    max_depth=depth_limit,
                    max_elements=element_limit,
                    elements=elements,
                )
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "root_selector": root_selector or {},
            "max_depth": depth_limit,
            "max_elements": element_limit,
            "elements": elements,
            "reason": caps.reason,
        }

    def element_capabilities(self, *, selector: dict[str, Any] | None = None) -> dict[str, Any]:
        caps = self.capabilities()
        selected = WindowsUiaElementSnapshot().to_dict()
        matched = False
        if caps.available:
            try:
                roots = self._select_roots({})
                for root in roots:
                    for element in self._iter_tree(root, max_depth=10, max_elements=1000):
                        snapshot = self._snapshot_element(element)
                        if _matches_selector(snapshot, selector or {}):
                            selected = snapshot.to_dict()
                            matched = True
                            break
                    if matched:
                        break
            except Exception as exc:
                return self._error_result(
                    caps,
                    f"UIA read-only element capability lookup failed: {exc}",
                    selector=selector or {},
                    matched=False,
                    capabilities=selected,
                )
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "selector": selector or {},
            "matched": matched,
            "capabilities": selected,
            "reason": caps.reason,
        }

    def _desktop(self) -> Any:
        if self._desktop_factory is not None:
            return self._desktop_factory()
        from pywinauto import Desktop  # type: ignore[import-not-found]

        return Desktop(backend="uia")

    def _top_level_windows(self) -> list[Any]:
        desktop = self._desktop()
        if hasattr(desktop, "windows"):
            return list(desktop.windows())
        if hasattr(desktop, "children"):
            return list(desktop.children())
        return []

    def _select_roots(self, selector: dict[str, Any]) -> list[Any]:
        roots = self._top_level_windows()
        if not selector:
            return roots
        selected = []
        for root in roots:
            snapshot = self._snapshot_element(root)
            if _matches_selector(snapshot, selector):
                selected.append(root)
        return selected

    def _walk(
        self,
        element: Any,
        *,
        depth: int,
        max_depth: int,
        max_elements: int,
        out: list[dict[str, Any]],
    ) -> None:
        if len(out) >= max_elements:
            return
        out.append(self._snapshot_element(element).to_dict())
        if depth >= max_depth:
            return
        for child in _safe_children(element):
            self._walk(child, depth=depth + 1, max_depth=max_depth, max_elements=max_elements, out=out)
            if len(out) >= max_elements:
                return

    def _iter_tree(self, root: Any, *, max_depth: int, max_elements: int) -> Iterable[Any]:
        out: list[Any] = []

        def walk(element: Any, depth: int) -> None:
            if len(out) >= max_elements:
                return
            out.append(element)
            if depth >= max_depth:
                return
            for child in _safe_children(element):
                walk(child, depth + 1)

        walk(root, 0)
        return out

    def _snapshot_element(self, element: Any) -> WindowsUiaElementSnapshot:
        info = getattr(element, "element_info", element)
        is_password = _is_password_element(element, info)
        name = str(getattr(info, "name", "") or "")
        if is_password:
            name = "<redacted password field>"
        rect = _rectangle_tuple(element)
        return WindowsUiaElementSnapshot(
            name=name,
            automation_id=str(getattr(info, "automation_id", "") or ""),
            control_type=str(getattr(info, "control_type", "") or ""),
            class_name=str(getattr(info, "class_name", "") or ""),
            process_id=_none_or_int(getattr(info, "process_id", None)),
            hwnd=_none_or_int(getattr(info, "handle", None) or getattr(info, "native_window_handle", None)),
            bounding_rectangle=rect,
            is_enabled=_safe_bool_call(element, "is_enabled"),
            is_offscreen=_inverse_optional_bool(_safe_bool_call(element, "is_visible")),
            is_password=is_password,
            supported_patterns=_supported_patterns(element),
        )

    def _error_result(self, caps: WindowsUiaReadOnlyCapabilities, reason: str, **extra: Any) -> dict[str, Any]:
        result = {
            "available": False,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "reason": reason,
        }
        result.update(extra)
        return result


def windows_uia_readonly_capability_status(*, platform: str | None = None) -> dict[str, Any]:
    return WindowsUiaReadOnlyBackend(platform=platform).capabilities().to_dict()


def _safe_children(element: Any) -> list[Any]:
    try:
        if hasattr(element, "children"):
            return list(element.children())
        if hasattr(element, "descendants"):
            return list(element.descendants())
    except Exception:
        return []
    return []


def _rectangle_tuple(element: Any) -> tuple[int, int, int, int] | None:
    try:
        rect = element.rectangle()
    except Exception:
        return None
    left = getattr(rect, "left", None)
    top = getattr(rect, "top", None)
    right = getattr(rect, "right", None)
    bottom = getattr(rect, "bottom", None)
    if None in {left, top, right, bottom}:
        return None
    return int(left), int(top), int(right), int(bottom)


def _safe_bool_call(element: Any, method: str) -> bool | None:
    try:
        fn = getattr(element, method)
    except Exception:
        return None
    try:
        return bool(fn())
    except Exception:
        return None


def _inverse_optional_bool(value: bool | None) -> bool | None:
    if value is None:
        return None
    return not value


def _is_password_element(element: Any, info: Any) -> bool:
    for attr in ("is_password", "_password"):
        try:
            value = getattr(element, attr)
            if callable(value):
                value = value()
            if bool(value):
                return True
        except Exception:
            pass
    text = " ".join(
        str(getattr(info, attr, "") or "").lower()
        for attr in ("control_type", "class_name", "automation_id", "name")
    )
    return "password" in text or "pwd" in text


def _supported_patterns(element: Any) -> tuple[str, ...]:
    try:
        patterns = element.get_supported_patterns()
    except Exception:
        return ()
    return tuple(str(pattern) for pattern in patterns)


def _matches_selector(snapshot: WindowsUiaElementSnapshot, selector: dict[str, Any]) -> bool:
    if not selector:
        return True
    data = snapshot.to_dict()
    for key, value in selector.items():
        if value in (None, ""):
            continue
        if data.get(key) != value:
            return False
    return True


def _none_or_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _is_windows(platform_id: str) -> bool:
    return platform_id.startswith("win")
