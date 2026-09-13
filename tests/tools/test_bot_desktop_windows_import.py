"""Native Windows: Bot Desktop lease must import, and ordinary computer_use must survive.

Bot Screen is Linux-only. The lease module is still imported on every nonempty
``computer_use`` action and on ``display.status``, so a module-level ``fcntl``
import must not take down those paths on win32.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.windows_only


def test_lease_module_imports_without_fcntl():
    from tools.bot_desktop import lease

    assert lease.fcntl is None


def test_computer_use_capture_reaches_the_existing_backend(monkeypatch):
    from tools.computer_use import tool

    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": object())

    def _dispatch(backend, action, args, **kwargs):
        return json.dumps({"ok": True, "action": action, "reached_backend": True})

    monkeypatch.setattr(tool, "_dispatch", _dispatch)
    raw = tool.handle_computer_use({"action": "capture"})
    assert isinstance(raw, str)
    res = json.loads(raw)
    blob = json.dumps(res)
    assert "fcntl" not in blob.lower()
    assert res.get("ok") is True
    assert res.get("reached_backend") is True


def test_display_status_jsonrpc_reports_unsupported():
    import tui_gateway.server as server

    resp = server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "display.status", "params": {}}
    )
    assert "error" not in resp, resp
    result = resp["result"]
    assert result["supported"] is False
    assert "lease" in result


def test_lease_mutation_refuses_without_a_unix_lock():
    from tools.bot_desktop import lease

    with pytest.raises(lease.LeaseLockUnavailable):
        lease.acquire("viewer")
    assert lease.get().holder == lease.AGENT
