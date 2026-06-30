"""In-process bridge from Python browser tools to the visible Desktop BrowserPane.

The renderer owns Electron webviews, so tool code cannot touch the page directly.
The TUI/Desktop gateway registers a runner here; generic browser_* tools call this
neutral module and receive the renderer's browser.desktop.command response.

When no runner is registered the tools keep their legacy provider behavior. Once
a runner is registered and a matching visible Desktop session exists, errors are
real errors and must not fall back to a second hidden browser reality.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

DesktopBrowserCommandRunner = Callable[
    [str, str, dict[str, Any]],
    Mapping[str, Any],
]

_runner: Callable[..., Mapping[str, Any]] | None = None


def set_desktop_browser_command_runner(runner: Callable[..., Mapping[str, Any]] | None) -> None:
    """Register the process-local Desktop BrowserPane command runner."""
    global _runner
    _runner = runner


def clear_desktop_browser_command_runner() -> None:
    """Clear the registered runner. Primarily for tests."""
    global _runner
    _runner = None


def has_desktop_browser_command_runner() -> bool:
    return _runner is not None


def run_desktop_browser_command(
    session_key: str,
    command: str,
    params: Mapping[str, Any] | None = None,
    *,
    tab_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """Run a command against the visible BrowserPane if a runner is registered.

    Returns ``None`` only when no Desktop runner exists in this process. A dict
    with ``desktop_unavailable: True`` means the runner exists but no visible
    Desktop session matches this task/session key; callers may use their legacy
    non-visible browser provider in that case. All other ``ok: false`` responses
    are authoritative visible-browser failures and should be surfaced to the
    agent instead of silently spawning another browser.
    """
    runner = _runner
    if runner is None:
        return None
    try:
        result = runner(
            str(session_key or "default"),
            str(command or ""),
            dict(params or {}),
            tab_id=tab_id,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive; runner tests cover normal errors
        return {"ok": False, "error": f"visible browser bridge failed: {exc}"}
    if result is None:
        return {"ok": False, "desktop_unavailable": True, "error": "visible browser is unavailable"}
    return dict(result)
