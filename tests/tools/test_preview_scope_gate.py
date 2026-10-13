"""tool_call scope gate for session-gated GUI tools (#120413).

drive_preview / desktop_preview live in the ``desktop_ui`` toolset: a session
without that surface (TUI, CLI, webhook, ...) can never reach them, not even
via tool_search. The bridge must fail fast with the real reason instead of
the generic "not available ... Use tool_search", which sends the model on a
round-trip tool_search can never satisfy. Desktop-scoped sessions keep
working (the call reaches the tool).
"""

import json

import model_tools
from agent.tool_executor import _unwrap_tool_search_call

TUI_SCOPE = ["coding", "project"]
DESKTOP_SCOPE = ["coding", "desktop_ui", "project"]


class _Agent:
    def __init__(self, enabled):
        self.enabled_toolsets = enabled
        self.disabled_toolsets = None


def _bridge(name, scope):
    return model_tools.handle_function_call(
        "tool_call", {"name": name, "arguments": {"action": "elements"}},
        enabled_toolsets=list(scope), disabled_toolsets=None)


def test_bridge_tui_scope_names_the_gui_surface():
    err = json.loads(_bridge("drive_preview", TUI_SCOPE))["error"]
    assert "desktop" in err.lower()
    assert "use tool_search to find" not in err.lower()


def test_bridge_tui_scope_desktop_preview_names_the_gui_surface():
    err = json.loads(_bridge("desktop_preview", TUI_SCOPE))["error"]
    assert "desktop" in err.lower()
    assert "use tool_search to find" not in err.lower()


def test_unwrap_tui_scope_names_the_gui_surface():
    _, _, block = _unwrap_tool_search_call(
        _Agent(TUI_SCOPE), "tool_call",
        {"name": "drive_preview", "arguments": {"action": "elements"}})
    assert block is not None and "desktop" in block.lower()
    assert "use tool_search to find" not in block.lower()


def test_bridge_desktop_scope_still_reaches_the_tool():
    # Headless: no GUI callback, so the tool itself fail-fasts with its
    # desktop-only reason — the point is the scope gate let it through.
    err = json.loads(_bridge("drive_preview", DESKTOP_SCOPE))["error"]
    assert "desktop" in err.lower()


def test_unwrap_desktop_scope_not_blocked():
    name, _, block = _unwrap_tool_search_call(
        _Agent(DESKTOP_SCOPE), "tool_call",
        {"name": "drive_preview", "arguments": {"action": "elements"}})
    assert block is None and name == "drive_preview"


def test_non_surface_tool_keeps_generic_message():
    err = json.loads(_bridge("computer_use", TUI_SCOPE))["error"]
    assert "not available in this session" in err
