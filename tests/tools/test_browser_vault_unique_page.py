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
    supervisor._driven_target_id = None
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
    assert result.get("session_id") == ("0" if matches == 1 else None)
    assert supervisor._page_session_id == ("0" if matches == 1 else "previous")
    assert detached == ([] if matches == 1 else ["0", "1"])


def test_enter_code_refuses_if_session_moves_after_unique_discovery(monkeypatch):
    import json
    from tools import browser_vault_tool as tool

    supervisor = Mock()
    supervisor.focus_page.return_value = {"ok": True, "session_id": "selected"}
    monkeypatch.setattr(tool, "_page_backend", lambda _: None)
    monkeypatch.setattr(tool, "_ensure_supervisor", lambda _: supervisor)
    monkeypatch.setattr(tool, "_saved_login", lambda _: (None, None))
    monkeypatch.setattr(tool, "_current_page_check", lambda _: tool._CheckedPage("https://example.test", "later"))
    result = json.loads(tool.browser_vault_enter_code())
    assert result["error_type"] == "supervisor_required"
    supervisor.evaluate_runtime.assert_not_called()
