"""Hermes' URL checks on the browser extension lane.

The extension refuses bad schemes and embedded credentials but has no secret-in-URL check and no operator
website policy. These tests go through the registry handler with a fake broker and assert that a URL that
Hermes refuses never reaches ``broker.dispatch``, for navigate and for navigate steps inside a batch.
"""

import json

import pytest

from tools import browser_tool
from tools.browser_extension_router import route_browser_tool
from tools.registry import registry

SECRET_URL = "https://example.test/cb?key=sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"


class FakeBroker:
    def __init__(self):
        self.dispatched = []

    def scope_for_session(self, **identity):
        return object()

    def select(self, scope, action):
        return object()

    def dispatch(self, scope, *, action, arguments, tool_call_id=""):
        self.dispatched.append((action, arguments))
        return json.dumps({"success": True})


@pytest.fixture
def broker(monkeypatch):
    """A bound extension controller that can run everything, and an operator policy that blocks blocked.test."""
    import gateway.browser_control_broker as gw
    import tools.browser_extension_router as router

    fake = FakeBroker()
    monkeypatch.setattr(gw, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(gw, "get_browser_control_broker", lambda: fake)
    monkeypatch.setattr(router, "_bound_identity", lambda: ("session", "principal", "transport"))
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: (
        {"message": "Blocked by website policy", "host": "blocked.test", "rule": "blocked.test", "source": "config"}
        if "blocked.test" in url else None))
    return fake


def _call(name, args):
    return json.loads(registry.get_entry(name).handler(args, task_id="task", session_id="session"))


@pytest.mark.parametrize("url", [
    SECRET_URL, "https://blocked.test/page", "http://169.254.169.254/latest/meta-data/",
])
def test_a_refused_navigate_never_reaches_the_controller(broker, url):
    out = _call("browser_navigate", {"url": url})

    assert out["success"] is False and out["error"].startswith("Blocked")
    assert broker.dispatched == []


@pytest.mark.parametrize("url", [SECRET_URL, "https://blocked.test/page", "http://169.254.169.254/latest/meta-data/"])
def test_a_refused_tab_create_never_reaches_the_controller(broker, url):
    out = _call("browser_tab_create", {"url": url})

    assert out["success"] is False and out["error"].startswith("Blocked")
    assert broker.dispatched == []


def test_an_allowed_tab_create_is_sent_with_the_normalized_url(broker):
    _call("browser_tab_create", {"url": "http://localhost:3000/Köln"})

    assert broker.dispatched == [("browser_tab_create", {"url": "http://localhost:3000/K%C3%B6ln"})]


def test_the_operator_policy_refusal_names_the_rule(broker):
    out = _call("browser_navigate", {"url": "https://blocked.test/page"})

    assert out["blocked_by_policy"] == {"host": "blocked.test", "rule": "blocked.test", "source": "config"}


def test_local_and_private_addresses_stay_reachable_through_the_extension(broker):
    assert _call("browser_navigate", {"url": "http://localhost:3000/app"}) == {"success": True}
    assert _call("browser_navigate", {"url": "http://192.168.1.10/admin"}) == {"success": True}

    assert [args["url"] for _action, args in broker.dispatched] == ["http://localhost:3000/app", "http://192.168.1.10/admin"]


def test_the_url_that_was_checked_is_the_url_that_is_sent(broker):
    _call("browser_navigate", {"url": "https://example.test/Köln"})

    assert broker.dispatched == [("browser_navigate", {"url": "https://example.test/K%C3%B6ln"})]


def test_a_blocked_navigate_step_refuses_the_whole_batch(broker):
    steps = [
        {"action": "click", "ref": "@e1"},
        {"action": "navigate", "url": "https://example.test/ok"},
        {"action": "navigate", "url": "https://blocked.test/page"},
        {"action": "press", "key": "Enter"},
    ]

    out = _call("browser_batch", {"actions": steps})

    assert out["success"] is False
    assert "step 3 (navigate)" in out["error"] and "Nothing was run" in out["error"]
    assert broker.dispatched == []


def test_a_secret_in_a_navigate_step_refuses_the_batch(broker):
    out = _call("browser_batch", {"actions": [{"action": "navigate", "url": SECRET_URL}]})

    assert out["success"] is False and "step 1 (navigate)" in out["error"]
    assert broker.dispatched == []


def test_an_allowed_batch_is_sent_with_normalized_urls_and_the_callers_args_are_untouched(broker):
    steps = [{"action": "navigate", "url": "https://example.test/Köln"}, {"action": "click", "ref": "@e1"}]
    args = {"actions": steps, "stop_on_error": False}

    _call("browser_batch", args)

    assert broker.dispatched == [("browser_batch", {
        "actions": [{"action": "navigate", "url": "https://example.test/K%C3%B6ln"}, {"action": "click", "ref": "@e1"}],
        "stop_on_error": False})]
    assert steps[0]["url"] == "https://example.test/Köln"


@pytest.mark.parametrize("name, args", [
    ("browser_tab_group", {"tab_ids": [7, 8]}), ("browser_tab_ungroup", {"tab_ids": [7, 8]}),
])
def test_the_tab_group_tools_open_no_url_and_go_through_untouched(broker, name, args):
    _call(name, args)

    assert broker.dispatched == [(name, args)]


def test_the_guard_does_not_run_on_the_built_in_lane():
    seen = []
    result = route_browser_tool(
        "browser_navigate", {"url": SECRET_URL}, fallback=lambda: "legacy", broker=FakeBroker(), enabled=False,
        guard=lambda action, args, route: seen.append(action) or (args, "refused"))

    assert result == "legacy" and seen == []
