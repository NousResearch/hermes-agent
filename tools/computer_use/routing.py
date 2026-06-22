"""Pure routing decisions for safe desktop-control requests.

This module intentionally has no side effects and does not call browser,
vision, terminal, file, or native GUI backends. It is a small policy helper for
explaining the safest available surface for a desktop-like task before any
mutation-capable tool is considered.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class DesktopRouteDecision:
    """Decision returned by :func:`route_desktop_request`."""

    route: str
    available: bool
    reason: str
    platform: str
    read_only: bool
    next_safe_action: str


_SENSITIVE_TASK_PATTERN = re.compile(
    r"\b(password|passcode|2fa|mfa|totp|otp|security|payment|checkout|bank|"
    r"wire transfer|permission dialog|approve login|login prompt)\b",
    re.IGNORECASE,
)


_FILEISH_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


_SCREENSHOT_WORDS = ("screenshot", "screen shot", "image", "png", "jpg", "jpeg")


def route_desktop_request(
    *,
    intent: str = "inspect",
    target: str = "",
    task: str = "",
    read_only: bool = True,
    platform: Optional[str] = None,
    browser_available: bool = True,
    file_available: bool = True,
    terminal_available: bool = True,
    vision_available: bool = True,
    computer_use_available: bool = False,
    windows_uia_readonly_available: bool = False,
) -> DesktopRouteDecision:
    """Choose the narrowest safe surface for a desktop-control-like request.

    The helper is deliberately dependency-light and capability-injected. Callers
    provide current backend/tool availability; this function only classifies and
    explains the route. It never clicks, types, opens windows, reads files, or
    probes live GUI state.
    """

    platform_id = platform or sys.platform
    normalized_target = (target or "").strip()
    normalized_task = (task or "").strip()
    normalized_intent = (intent or "inspect").strip().lower()
    wants_mutation = (not read_only) or normalized_intent in {"operate", "mutate", "control"}

    if wants_mutation and _is_sensitive_task(normalized_task):
        return _decision(
            "unsupported",
            False,
            "Sensitive desktop tasks such as passwords, payments, 2FA, security prompts, or permission dialogs are not routed to native GUI automation.",
            platform_id,
            read_only,
            "Ask the user for explicit guidance or use a non-GUI, audited workflow instead.",
        )

    target_kind = _classify_target(normalized_target)

    if target_kind == "url":
        if browser_available:
            return _decision(
                "browser",
                True,
                "URL target detected; browser automation is the safest cross-platform route.",
                platform_id,
                read_only,
                "Use the browser toolset for the web UI task.",
            )
        return _decision(
            "unsupported",
            False,
            "URL target detected, but browser automation is not available in this session.",
            platform_id,
            read_only,
            "Ask the user to enable/fix browser automation or provide page content/screenshots.",
        )

    if target_kind == "file":
        if file_available and terminal_available:
            return _decision(
                "file_terminal",
                True,
                "local path/config/log target detected; file plus terminal tools avoid GUI automation.",
                platform_id,
                read_only,
                "Inspect or edit the local artifact with file tools, verifying with terminal commands.",
            )
        return _decision(
            "unsupported",
            False,
            "local path/config/log target detected, but file and terminal tools are not both available.",
            platform_id,
            read_only,
            "Enable file/terminal tooling or ask the user to paste the relevant content.",
        )

    if target_kind == "screenshot":
        if vision_available:
            return _decision(
                "vision",
                True,
                "Screenshot or image target detected; vision analysis is read-only and avoids GUI mutation.",
                platform_id,
                True,
                "Analyze the provided image with the vision toolset.",
            )
        return _decision(
            "unsupported",
            False,
            "Screenshot or image target detected, but vision tooling is not available.",
            platform_id,
            True,
            "Ask the user for a text description or enable vision tooling.",
        )

    if _is_windows(platform_id):
        if wants_mutation:
            return _decision(
                "unsupported",
                False,
                "Windows native GUI mutation is not supported in safe routing v0.",
                platform_id,
                read_only,
                "Use terminal/file/browser/vision alternatives, or wait for an explicitly approved Windows backend.",
            )
        if windows_uia_readonly_available:
            return _decision(
                "windows_uia_readonly",
                True,
                "Windows native app read-only inspection can route to the injected Windows UIA read-only capability.",
                platform_id,
                True,
                "Use the Windows UIA backend only for capture/list/read-only inspection.",
            )
        return _decision(
            "unsupported",
            False,
            "Windows UIA read-only backend is not available; native desktop automation remains unavailable on this platform.",
            platform_id,
            True,
            "Prefer terminal/file for local tasks or vision for user-provided screenshots.",
        )

    if _is_macos(platform_id):
        if computer_use_available:
            return _decision(
                "computer_use",
                True,
                "macOS native desktop target can route to computer_use because the backend is available.",
                platform_id,
                read_only,
                "Use computer_use with existing approval gates for any mutation.",
            )
        return _decision(
            "unsupported",
            False,
            "macOS native desktop target detected, but computer_use is not available.",
            platform_id,
            read_only,
            "Install/enable the supported backend or use browser/terminal/file/vision alternatives.",
        )

    return _decision(
        "unsupported",
        False,
        f"No safe native desktop route is available for platform {platform_id!r}.",
        platform_id,
        read_only,
        "Prefer browser, terminal/file, or vision routes when they fit the task.",
    )


def _decision(
    route: str,
    available: bool,
    reason: str,
    platform: str,
    read_only: bool,
    next_safe_action: str,
) -> DesktopRouteDecision:
    return DesktopRouteDecision(
        route=route,
        available=available,
        reason=reason,
        platform=platform,
        read_only=read_only,
        next_safe_action=next_safe_action,
    )


def _is_sensitive_task(task: str) -> bool:
    return bool(task and _SENSITIVE_TASK_PATTERN.search(task))


def _classify_target(target: str) -> str:
    lowered = target.lower()
    if _is_url(target):
        return "url"
    if lowered in _SCREENSHOT_WORDS or any(word in lowered for word in _SCREENSHOT_WORDS):
        return "screenshot"
    if lowered.startswith(("file:", "path:", "log:", "config:")) or _looks_like_path(target):
        return "file"
    return "native"


def _is_url(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_like_path(target: str) -> bool:
    if not target:
        return False
    win_path = PureWindowsPath(target)
    posix_path = PurePosixPath(target)
    if win_path.drive or target.startswith(("/", "~/", ".\\", "./", "..\\", "../")):
        return True
    suffix = win_path.suffix or posix_path.suffix
    return suffix.lower() in _FILEISH_SUFFIXES


def _is_windows(platform_id: str) -> bool:
    return platform_id.lower().startswith("win")


def _is_macos(platform_id: str) -> bool:
    return platform_id.lower() == "darwin"


__all__ = ["DesktopRouteDecision", "route_desktop_request"]
