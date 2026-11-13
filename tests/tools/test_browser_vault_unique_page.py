"""OTP page discovery refuses ambiguous CDP targets without changing the active page."""
import asyncio
import threading
from unittest.mock import Mock

import pytest

from tools import browser_supervisor as module


@pytest.mark.parametrize("matches", [1, 2])
def test_unique_otp_target_selection(monkeypatch, matches):
    supervisor = object.__new__(module.CDPSupervisor)
    supervisor._loop = Mock()
    supervisor._loop.is_running.return_value = True
    supervisor._state_lock = threading.Lock()
    supervisor._page_session_id = "previous"
    detached = []

    async def cdp(method, params=None, **kwargs):
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"targetId": str(i), "type": "page", "url": "https://example.test"}
                                                 for i in range(matches)]}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": params["targetId"]}}
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": True}}}
        if method == "Target.detachFromTarget":
            detached.append(params["sessionId"])
        return {}

    async def noop(*args, **kwargs):
        return None

    supervisor._cdp = cdp
    supervisor._enable_page_domains = noop
    supervisor._install_dialog_bridge = noop
    monkeypatch.setattr(module, "_schedule", lambda coro, loop, timeout: asyncio.run(coro))
    result = supervisor.focus_page("https://example.test", accept="true", unique=True)
    assert result["ok"] is (matches == 1)
    assert supervisor._page_session_id == ("0" if matches == 1 else "previous")
    assert detached == ([] if matches == 1 else ["0", "1"])
