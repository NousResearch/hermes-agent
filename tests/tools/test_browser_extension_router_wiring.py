"""Wiring regression tests for the browser extension router.

These guard the *registry wiring* — that every ``browser_*`` handler routes
through :func:`tools.browser_extension_router.routed_browser_handler` with
the tool's action name, its raw args, and its identity kwargs, instead of
calling the legacy backend directly. The routing contract itself is tested
by ``test_browser_extension_router.py``; here we only pin the plumbing.
"""

import json

import pytest

from tools.registry import registry


@pytest.fixture(autouse=True)
def _route_spy(monkeypatch):
    """Replace the wrapper with a spy that records the route and then runs
    the legacy fallback, so each test proves the handler is wired without
    exercising real routing or a real browser backend."""
    calls = []

    def spy(action, args, *, fallback, task_id=None, session_id=None, tool_call_id=None, guard=None,
            result_guard=None):
        calls.append(
            {
                "action": action,
                "args": dict(args),
                "task_id": task_id,
                "session_id": session_id,
                "tool_call_id": tool_call_id,
            }
        )
        return fallback()

    from tools import browser_tool
    from tools import browser_cdp_tool

    monkeypatch.setattr(browser_tool, "routed_browser_handler", spy)
    monkeypatch.setattr(browser_cdp_tool, "routed_browser_handler", spy)
    monkeypatch.setattr(
        browser_tool,
        "browser_navigate",
        lambda url="", task_id=None, local_browser=False: "legacy-nav",
    )
    monkeypatch.setattr(browser_cdp_tool, "browser_cdp", lambda *a, **k: "legacy-cdp")
    return calls


BROWSER_ACTIONS = [
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_get_images",
    "browser_vision",
    "browser_console",
]


def test_every_browser_registry_handler_routes_through_wrapper(_route_spy):
    for name in BROWSER_ACTIONS:
        _route_spy.clear()
        handler = registry.get_entry(name).handler
        args = {"url": "https://example.test", "ref": "@e1", "text": "hi", "direction": "down", "key": "Enter"}
        result = handler(dict(args), task_id="task-fixture", session_id="session-fixture")
        assert result is not None
        assert len(_route_spy) == 1, f"{name} did not route through the wrapper"
        route = _route_spy[0]
        assert route["action"] == name
        assert route["task_id"] == "task-fixture"
        assert route["session_id"] == "session-fixture"




def test_browser_cdp_handler_routes_through_wrapper(_route_spy):
    handler = registry.get_entry("browser_cdp").handler
    args = {"method": "Target.getTargets", "params": {"filter": []}}
    result = handler(dict(args), task_id="task-fixture", session_id="session-fixture")
    assert result == "legacy-cdp"
    route = _route_spy[0]
    assert route["action"] == "browser_cdp"
    assert route["args"] == args
    assert route["task_id"] == "task-fixture"
    assert route["session_id"] == "session-fixture"


def test_routed_tools_are_all_allowed_through_the_broker():
    """A routed browser tool the broker would not forward is a tool that can never reach a controller."""
    from gateway.browser_control_broker import BROWSER_CONTROL_CAPABILITIES
    from tools import browser_tool

    routed = [name for name, _emoji, check_fn, *_ in browser_tool._BROWSER_TOOL_TABLE if check_fn is None]
    assert "browser_batch" in routed
    assert [name for name in routed if name not in BROWSER_CONTROL_CAPABILITIES] == []


def test_browser_batch_routes_a_valid_batch_and_refuses_an_invalid_one_before_routing(_route_spy, monkeypatch):
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, "browser_batch", lambda actions, stop_on_error=True, task_id=None: "legacy-batch")
    handler = registry.get_entry("browser_batch").handler
    steps = [{"action": "click", "ref": "@e1"}, {"action": "press", "key": "Enter"}]

    result = handler({"actions": steps}, task_id="task-fixture", session_id="session-fixture")

    assert len(_route_spy) == 1 and _route_spy[0]["action"] == "browser_batch"
    assert _route_spy[0]["args"] == {"actions": steps}
    assert result == "legacy-batch"

    _route_spy.clear()
    refused = json.loads(handler({"actions": [{"action": "back"}] * 21}, task_id="task-fixture"))
    assert refused["success"] is False and "at most 20" in refused["error"]
    assert _route_spy == []


ROUTED_TOOL_ARGS = {
    "browser_fill": {"ref": "@e1", "value": "hello"},
    "browser_select": {"ref": "@e2", "value": "Two"},
    "browser_hover": {"ref": "@e3"},
    "browser_drag": {"ref": "@e3", "to_ref": "@e4"},
    "browser_scroll_to": {"ref": "@e9"},
    "browser_tabs": {},
    "browser_tab_activate": {"tab_id": 7},
    "browser_tab_create": {"url": "https://example.test"},
    "browser_tab_close": {"tab_id": 7},
    "browser_tab_group": {"tab_ids": [7, 8]},
    "browser_tab_ungroup": {"tab_ids": [7, 8]},
    "browser_screenshot": {"ref": "@e5"},
}


@pytest.mark.parametrize("name", sorted(ROUTED_TOOL_ARGS))
def test_every_new_browser_tool_routes_through_the_wrapper_with_its_raw_args(_route_spy, monkeypatch, name):
    from tools import browser_tool

    monkeypatch.setattr(browser_tool, name, lambda *a, **k: "legacy-result")
    args = ROUTED_TOOL_ARGS[name]

    result = registry.get_entry(name).handler(dict(args), task_id="task-fixture", session_id="session-fixture")

    assert result == "legacy-result"
    assert [(r["action"], r["args"], r["task_id"], r["session_id"]) for r in _route_spy] == [
        (name, args, "task-fixture", "session-fixture")]
