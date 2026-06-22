"""Windows UIA read-only backend skeleton.

This module is intentionally read-only and side-effect-light. It does not call
click/type/drag, does not focus windows, does not import or use SendInput,
pyautogui, or pynput, and does not mutate native Windows UI. It only reports
capabilities and returns safe placeholder readback structures for future UIA
inspection work.
"""

from __future__ import annotations

import importlib.util
import sys
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
    """Read-only capability/snapshot skeleton for future Windows UIA work."""

    def __init__(self, *, platform: str | None = None) -> None:
        self.platform = platform or sys.platform

    def capabilities(self) -> WindowsUiaReadOnlyCapabilities:
        if not _is_windows(self.platform):
            return WindowsUiaReadOnlyCapabilities(
                available=False,
                platform=self.platform,
                reason="Windows UIA read-only backend is only meaningful on Windows.",
            )
        if importlib.util.find_spec("pywinauto") is None:
            return WindowsUiaReadOnlyCapabilities(
                available=False,
                platform=self.platform,
                reason="Optional pywinauto dependency is not installed; skeleton remains dry-run/status-only.",
            )
        return WindowsUiaReadOnlyCapabilities(
            available=False,
            platform=self.platform,
            reason="Optional pywinauto dependency is installed, but live UIA enumeration remains disabled in this v0 skeleton.",
        )

    def list_windows(self) -> dict[str, Any]:
        caps = self.capabilities()
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "windows": [],
            "reason": caps.reason,
            "next_step": "Implement bounded UIA tree enumeration in a future read-only criterion; no click/type/drag here.",
        }

    def snapshot_tree(
        self,
        *,
        root_selector: dict[str, Any] | None = None,
        max_depth: int = 3,
        max_elements: int = 200,
    ) -> dict[str, Any]:
        caps = self.capabilities()
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "root_selector": root_selector or {},
            "max_depth": max(0, min(int(max_depth), 10)),
            "max_elements": max(1, min(int(max_elements), 1000)),
            "elements": [],
            "reason": caps.reason,
        }

    def element_capabilities(self, *, selector: dict[str, Any] | None = None) -> dict[str, Any]:
        caps = self.capabilities()
        return {
            "available": caps.available,
            "backend": caps.backend,
            "read_only": True,
            "mutation_allowed": False,
            "would_execute_native_input": False,
            "selector": selector or {},
            "matched": False,
            "capabilities": WindowsUiaElementSnapshot().to_dict(),
            "reason": caps.reason,
        }


def windows_uia_readonly_capability_status(*, platform: str | None = None) -> dict[str, Any]:
    return WindowsUiaReadOnlyBackend(platform=platform).capabilities().to_dict()


def _is_windows(platform_id: str) -> bool:
    return platform_id.startswith("win")
