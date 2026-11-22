"""The supervisor's fallback tab must not raise a headed browser window over the user's work (#156748)."""

from __future__ import annotations

import asyncio

from tools import browser_supervisor as bs


def test_initial_page_is_created_in_background(monkeypatch):
    supervisor = bs.CDPSupervisor(task_id="initial-page", cdp_url="ws://127.0.0.1:9222")
    sent = []

    async def cdp(method, params=None, **_kwargs):
        sent.append((method, params))
        return {"result": {
            "Target.getTargets": {"targetInfos": []},
            "Target.createTarget": {"targetId": "new-tab"},
            "Target.attachToTarget": {"sessionId": "session-1"},
        }.get(method, {})}

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(supervisor, "_cdp", cdp)
    monkeypatch.setattr(supervisor, "_enable_page_domains", noop)
    monkeypatch.setattr(supervisor, "_install_dialog_bridge", noop)

    asyncio.run(supervisor._attach_initial_page())

    assert ("Target.createTarget", {"url": "about:blank", "background": True}) in sent
