#!/usr/bin/env python3
"""Read which OS window sits directly underneath the Hermes desktop window.

The window list lives with the OS, so this round-trips through the gateway's
blocking-prompt bridge like `read_terminal`: ``window.read.request`` -> the renderer's
main process (native window enumeration) -> ``window.read.respond``.
"""

import json
import socket
from typing import Callable, Optional

from tools.read_terminal_tool import read_pane
from tools.registry import registry


def _agent_host(payload: dict) -> Optional[dict]:
    """Describe the machine gap when the window is not one we can drive.

    The renderer sets ``agent_on_this_machine`` false when the desktop app is
    driving a remote gateway, because only it can know: an SSH tunnel makes the
    client look like loopback from here. It sends the bare flag and we name
    ourselves, so no host or connection detail has to cross.

    Absent on a local session, so the common case costs nothing.
    """
    if payload.get("agent_on_this_machine") is not False:
        return None

    try:
        host = socket.gethostname().strip()
    except Exception:
        host = ""

    where = f"on {host}" if host else "on another machine"

    return {
        "same_machine": False,
        "name": host or None,
        "note": (
            f"This window is on the user's screen. You are running {where}, so "
            "computer_use drives that machine's desktop and cannot click, type "
            "into, or screenshot this window. Say so and tell the user what to "
            "do in it, rather than acting somewhere they aren't looking."
        ),
    }


def read_window_below_tool(callback: Optional[Callable] = None) -> str:
    """Return the window underneath the Hermes window as a JSON string."""
    result = read_pane(callback, (), (
        "read_window_below is only available in the Hermes desktop app.",
        "",
        "Failed to read the window below: ",
        "Could not determine the window underneath (the desktop app did "
        "not answer, or window enumeration is unavailable on this system).",
    ))
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return result

    if isinstance(payload, dict) and "agent_on_this_machine" in payload:
        agent_host = _agent_host(payload)
        payload.pop("agent_on_this_machine", None)

        if agent_host:
            payload["agent_host"] = agent_host

        return json.dumps(payload, ensure_ascii=False)

    return result


READ_WINDOW_BELOW_SCHEMA = {
    "name": "read_window_below",
    "description": (
        "Identify the app window directly behind the Hermes desktop window "
        "(what the user is working in). JSON: {window: {app, title, bounds, "
        "id}, frontmost, platform}. An `agent_host` key appears only when you "
        "are running on a different machine than the user's screen — its "
        "`note` says what you can and cannot do with the window, so relay it "
        "rather than trying anyway. title may be empty when the OS withholds "
        "it (noted in `note`); where windows cannot be enumerated at all, "
        "{error, platform} says what would fix it — relay that instead of "
        "retrying. Metadata only; never captures pixels."
    ),
    "parameters": {
        "type": "object", "properties": {}
    },
}


registry.register(
    name="read_window_below",
    toolset="desktop_ui",
    schema=READ_WINDOW_BELOW_SCHEMA,
    handler=lambda args, **kw: read_window_below_tool(callback=kw.get("callback")),
    emoji="🪟",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'tool_error': ('tools.registry', 'tool_error'),
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
