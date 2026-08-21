"""Regression tests for browser CDP target creation without focus theft."""

import asyncio
import json


def test_supervisor_creates_background_target(monkeypatch):
    from tools.browser_supervisor import CDPSupervisor

    calls = []
    supervisor = object.__new__(CDPSupervisor)
    supervisor._page_session_id = None

    async def fake_cdp(method, params=None, **kwargs):
        calls.append({"method": method, "params": params, "kwargs": kwargs})
        if method == "Target.getTargets":
            return {"result": {"targetInfos": []}}
        if method == "Target.createTarget":
            return {"result": {"targetId": "synthetic-target"}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "synthetic-session"}}
        return {"result": {}}

    async def fake_dialog_bridge(session_id):
        calls.append({"method": "_install_dialog_bridge", "session_id": session_id})

    monkeypatch.setattr(supervisor, "_cdp", fake_cdp)
    monkeypatch.setattr(supervisor, "_install_dialog_bridge", fake_dialog_bridge)
    asyncio.run(supervisor._attach_initial_page())

    create = next(call for call in calls if call["method"] == "Target.createTarget")
    assert (create["params"] or {}).get("background") is True


def test_raw_browser_cdp_creates_background_target(monkeypatch):
    import tools.browser_cdp_tool as tool

    calls = {}

    async def fake_call(endpoint, method, params, target_id, timeout):
        calls.update(
            {
                "endpoint": endpoint,
                "method": method,
                "params": dict(params),
                "target_id": target_id,
                "timeout": timeout,
            }
        )
        return {"targetId": "synthetic-target"}

    monkeypatch.setattr(tool, "_WS_AVAILABLE", True)
    monkeypatch.setattr(tool, "_resolve_cdp_endpoint", lambda: "ws://synthetic-endpoint")
    monkeypatch.setattr(tool, "_browser_cdp_private_guard", lambda **kwargs: None)
    monkeypatch.setattr(tool, "_cdp_call", fake_call)
    monkeypatch.setattr(tool, "_run_async", lambda coroutine: asyncio.run(coroutine))

    result = tool.browser_cdp("Target.createTarget", {"url": "about:blank"})

    payload = json.loads(result)
    assert payload["result"]["targetId"] == "synthetic-target"
    assert calls["params"]["background"] is True
