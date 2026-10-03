"""Regression for #120382: exact relying-party and child-frame origin checks."""
import asyncio
import json
from unittest.mock import patch

from agent.vault_store import VaultStore
from tools import browser_vault_tool as vault
from tools.browser_supervisor import CDPSupervisor
from tools.browser_supervisor_frames import FrameInfo


def test_focus_login_uses_owning_page_dom_for_oopif_even_without_frame_tree():
    supervisor = CDPSupervisor("test", "ws://localhost")
    supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
    supervisor._frames["owned"] = FrameInfo("owned", "https://identity.test", "https://identity.test",
                                             None, True, "child-session")
    supervisor._frames["foreign"] = FrameInfo("foreign", "https://identity.test", "https://identity.test",
                                               None, True, "other-session")
    calls = []

    async def cdp(method, params=None, *, session_id=None, timeout=10):
        calls.append((method, session_id))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [
                {"targetId": "blank", "type": "page", "url": "http://localhost/blank"},
                {"targetId": "rp", "type": "page", "url": "https://site.test/login"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "rp-session"}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": session_id == "child-session"}}}
        if method == "DOM.getDocument":
            return {"result": {"root": {"nodeId": 1}}}
        if method == "DOM.querySelectorAll":
            return {"result": {"nodeIds": [2]}}
        if method == "DOM.describeNode":
            return {"result": {"node": {"frameId": "owned"}}}
        if method == "Page.getFrameTree":
            return {"result": {"frameTree": {"frame": {"id": "rp"}}}}
        return {"result": {}}

    supervisor._cdp = cdp
    async def noop(*args, **kwargs):
        pass
    supervisor._enable_page_domains = noop
    supervisor._install_dialog_bridge = noop
    with patch("tools.browser_supervisor._schedule", side_effect=lambda coro, loop, **kw: asyncio.run(coro)):
        result = supervisor.focus_page("https://site.test", accept="password", frame_accept="password")
    assert result["ok"] and result["frame_id"] == "owned"
    assert ("Runtime.evaluate", "other-session") not in calls
    assert ("Runtime.evaluate", "child-session") in calls


def test_focus_login_finds_oopif_inside_same_origin_wrapper_not_foreign_tab():
    supervisor = CDPSupervisor("test", "ws://localhost")
    supervisor._loop = type("Loop", (), {"is_running": lambda self: True})()
    supervisor._frames["nested"] = FrameInfo("nested", "https://identity.test/login", "https://identity.test",
                                              "wrapper", True, "nested-session")
    supervisor._frames["foreign"] = FrameInfo("foreign", "https://identity.test/login", "https://identity.test",
                                               None, True, "foreign-session")
    calls = []

    async def cdp(method, params=None, *, session_id=None, timeout=10):
        calls.append((method, session_id))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [
                {"targetId": "foreign-tab", "type": "page", "url": "https://elsewhere.test/"},
                {"targetId": "rp", "type": "page", "url": "https://site.test/login"}]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "rp-session"}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": session_id in {"nested-session", "foreign-session"}}}}
        if method == "DOM.getDocument":
            return {"result": {"root": {"nodeId": 1}}}
        if method == "DOM.querySelectorAll":
            return {"result": {"nodeIds": {1: [2], 3: [4]}.get(params["nodeId"], [])}}
        if method == "DOM.describeNode":
            return {"result": {"node": {2: {"frameId": "wrapper", "contentDocument": {"nodeId": 3}},
                                         4: {"frameId": "nested"}}[params["nodeId"]]}}
        return {"result": {}}

    supervisor._cdp = cdp
    async def noop(*args, **kwargs):
        pass
    supervisor._enable_page_domains = noop
    supervisor._install_dialog_bridge = noop
    with patch("tools.browser_supervisor._schedule", side_effect=lambda coro, loop, **kw: asyncio.run(coro)):
        result = supervisor.focus_page("https://site.test", accept="password", frame_accept="password")
    assert result["ok"] and result["frame_id"] == "nested"
    assert ("Runtime.evaluate", "foreign-session") not in calls
    assert ("DOM.querySelectorAll", "rp-session") in calls


def test_cross_origin_login_requires_explicit_pair_consent_and_keeps_password_blind(tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "site", {"identifier": "a@b.test", "password": "iframe-only-canary",
                                             "identifier_type": "email"}, origin="https://site.test")
    controls = [{"autocomplete": "current-password", "formIndex": 0, "index": 0,
                 "name": "password", "type": "password"}]
    expressions = []

    def frame_eval(task, frame, expression):
        assert frame == "child"
        if expression == "location.href":
            return {"success": True, "result": "https://identity.test/login"}
        if "iframe-only-canary" in expression:
            expressions.append(expression)
            return {"success": True, "result": {"filled": 1}}
        return {"success": True, "result": controls}

    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_login_frame", return_value="child"), \
         patch.object(vault, "_frame_eval", side_effect=frame_eval), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="deny") as consent:
        denied = json.loads(vault.browser_vault_fill(meta.id))
        assert denied["error_type"] == "frame_origin_declined"
        assert not expressions
        consent.return_value = "accept"
        raw = vault.browser_vault_fill(meta.id)
    assert json.loads(raw)["filled_fields"] == 1
    assert "iframe-only-canary" not in raw
    assert len(expressions) == 1
    assert "https://site.test" in expressions[0]
    assert "https://identity.test" in expressions[0]
    assert "location.ancestorOrigins" in expressions[0]


def test_cross_origin_login_refuses_unknown_child_origin(tmp_path):
    store = VaultStore(base_dir=tmp_path / "vault")
    meta = store.add_item("login", "site", {"identifier": "a@b.test", "password": "canary",
                                             "identifier_type": "email"}, origin="https://site.test")
    with patch("agent.vault_store.get_vault_store", return_value=store), \
         patch.object(vault, "_focus_login_frame", return_value="child"), \
         patch.object(vault, "_frame_eval", return_value={"success": False}), \
         patch("tools.approval_prompt.request_elicitation_consent") as consent:
        result = json.loads(vault.browser_vault_fill(meta.id))
    assert result["error_type"] == "frame_origin_unknown"
    consent.assert_not_called()
