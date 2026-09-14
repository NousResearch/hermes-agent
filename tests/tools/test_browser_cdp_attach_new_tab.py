"""browser_navigate must not hijack an existing tab when attached to the user's own browser via CDP."""

import json

import pytest

import tools.browser_tool as bt
from tools import browser_tool_session as bt_session


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    bt._active_sessions.clear()
    bt._last_active_session_key.clear()
    monkeypatch.setattr(bt, "_secret_url_error_normalized", lambda url: (url, None))
    monkeypatch.setattr(bt, "_navigation_session_key", lambda task_id, url: task_id)
    monkeypatch.setattr(bt, "_is_local_sidecar_key", lambda key: False)
    monkeypatch.setattr(bt, "_url_policy_error", lambda url, auto_local=False: None)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(bt, "_post_redirect_block", lambda *a: None)
    monkeypatch.setattr(bt, "_attach_auto_snapshot", lambda response, key: None)
    yield
    bt._active_sessions.clear()
    bt._last_active_session_key.clear()


def _install(monkeypatch, features, tab_ok=True):
    session = {"session_name": "s", "features": features}
    calls = []

    def run(key, command, args=None, timeout=None):
        calls.append((command, list(args or [])))
        if command == "tab":
            return {"success": tab_ok, "error": None if tab_ok else "boom"}
        return {"success": True, "data": {"title": "T", "url": args[0]}}

    monkeypatch.setattr(bt_session, "_get_session_info", lambda key: session)
    monkeypatch.setattr(bt_session, "_run_browser_command", run)
    return calls


def test_cdp_override_opens_new_tab_before_first_open_only(monkeypatch):
    calls = _install(monkeypatch, {"cdp_override": True})

    assert json.loads(bt.browser_navigate("https://example.com", task_id="t"))["success"] is True
    assert calls == [("tab", ["new"]), ("open", ["https://example.com"])]

    calls.clear()
    bt.browser_navigate("https://example.org", task_id="t")
    assert calls == [("open", ["https://example.org"])]


def test_local_session_never_opens_new_tab(monkeypatch):
    calls = _install(monkeypatch, {"local": True})

    bt.browser_navigate("https://example.com", task_id="t")
    assert calls == [("open", ["https://example.com"])]


def test_failed_new_tab_refuses_to_navigate(monkeypatch):
    calls = _install(monkeypatch, {"cdp_override": True}, tab_ok=False)

    result = json.loads(bt.browser_navigate("https://example.com", task_id="t"))
    assert result["success"] is False
    assert "refusing to navigate" in result["error"]
    assert ("open", ["https://example.com"]) not in calls
