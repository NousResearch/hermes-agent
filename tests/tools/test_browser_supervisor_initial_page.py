"""Startup must survive extension popups that cannot enable the Page domain.

The transport double represents the real observed CDP failure; supervisor
selection, domain setup and session binding run through production code.
"""

import asyncio

import pytest

from tools.browser_supervisor import CDPSupervisor


@pytest.mark.parametrize("healthy_page_available", [True, False])
def test_startup_survives_an_extension_popup_before_web_targets(monkeypatch, healthy_page_available):
    popup = {"targetId": "popup", "type": "page", "url": "chrome-extension://fixture/popup.html", "title": "Popup", "attached": False}
    web = {"targetId": "web", "type": "page", "url": "https://example.test/login", "title": "Login", "attached": False}
    targets = [popup, web] if healthy_page_available else [popup]
    sessions = {}
    supervisor = CDPSupervisor(task_id="initial-page-regression", cdp_url="ws://127.0.0.1:1/devtools/browser/fixture")

    async def transport(method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": list(targets)}}
        if method == "Target.createTarget":
            created = {"targetId": "blank", "type": "page", "url": params["url"], "title": "", "attached": False}
            targets.append(created)
            return {"result": {"targetId": "blank"}}
        if method == "Target.attachToTarget":
            target = next(t for t in targets if t["targetId"] == params["targetId"])
            sid = "session-" + target["targetId"]
            sessions[sid] = target
            return {"result": {"sessionId": sid}}
        if method == "Page.enable" and sessions[session_id]["url"].startswith("chrome-extension://"):
            raise TimeoutError("The extension popup does not answer Page.enable")
        return {"result": {}}

    monkeypatch.setattr(supervisor, "_cdp", transport)
    asyncio.run(supervisor._attach_initial_page())
    selected = sessions[supervisor._page_session_id]
    assert selected["url"] == ("https://example.test/login" if healthy_page_available else "about:blank")
    assert popup in targets
