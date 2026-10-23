"""Vault fill attaches its supervisor through the sandbox port forward when the browser runs in the terminal backend."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from tools import browser_tool, browser_tool_cdp, browser_tool_session, browser_vault_tool
from tools.bot_desktop import runtime, sandbox_host
from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.environments import streams

SANDBOX_CDP = "ws://127.0.0.1:9223/devtools/browser/abc"


@pytest.fixture
def sandboxed(monkeypatch):
    attach = Mock(return_value=object())
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get", lambda task_id: None)
    monkeypatch.setattr(SUPERVISOR_REGISTRY, "get_or_start", attach)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(browser_tool_session, "_run_browser_command",
                        lambda *a, **k: {"success": True, "data": {"cdpUrl": SANDBOX_CDP}})
    monkeypatch.setattr(browser_tool_session, "_browser_in_sandbox", lambda: True)
    monkeypatch.setattr(browser_tool_cdp, "_get_dialog_policy_config", lambda: ("dismiss", 5))
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: object())
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: None)
    return attach


def test_sandboxed_browser_attaches_through_the_forwarded_port(sandboxed, monkeypatch):
    monkeypatch.setattr(streams, "forward_port", lambda env, port, *, user=None: 45001)
    assert browser_vault_tool._ensure_supervisor("t1") is sandboxed.return_value
    assert sandboxed.call_args.kwargs["cdp_url"] == "ws://127.0.0.1:45001/devtools/browser/abc"


def test_failed_forward_never_attaches_to_the_host_loopback(sandboxed, monkeypatch):
    monkeypatch.setattr(streams, "forward_port", Mock(side_effect=OSError("exec stream closed")))
    assert browser_vault_tool._ensure_supervisor("t1") is None
    sandboxed.assert_not_called()
