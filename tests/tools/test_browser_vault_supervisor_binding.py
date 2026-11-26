"""Vault fills must follow the current browser session, never a stale supervisor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools import browser_vault_tool


def test_ensure_supervisor_replaces_a_stale_endpoint(monkeypatch):
    """Changing browser daemons for one task must not retain the former page."""
    current_url = "ws://current.example/devtools/browser/current"
    stale = SimpleNamespace(cdp_url="ws://stale.example/devtools/browser/stale")
    current = SimpleNamespace(cdp_url=current_url)
    calls = []

    class Registry:
        def get(self, task_id):
            assert task_id == "task-1"
            return stale

        def get_or_start(self, **kwargs):
            calls.append(kwargs)
            return current

    from tools import browser_session, browser_session_cdp, browser_supervisor

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", Registry())
    monkeypatch.setattr(browser_session, "session_for_task", lambda task_id: {"cdp_url": current_url})
    monkeypatch.setattr(browser_session_cdp, "resolve_cdp_override", lambda url: url)
    monkeypatch.setattr(browser_session_cdp, "get_dialog_policy_config", lambda: ("accept", 30.0))

    assert browser_vault_tool._ensure_supervisor("task-1") is current
    assert calls == [{
        "task_id": "task-1",
        "cdp_url": current_url,
        "dialog_policy": "accept",
        "dialog_timeout_s": 30.0,
    }]


def test_ensure_supervisor_refuses_a_stale_endpoint_without_current_cdp(monkeypatch):
    """A stale task supervisor is not a substitute for a current browser session."""
    stale = SimpleNamespace(cdp_url="ws://stale.example/devtools/browser/stale")

    class Registry:
        def get(self, task_id):
            return stale

        def get_or_start(self, **kwargs):
            raise AssertionError("must not attach without a current CDP endpoint")

    from tools import browser_session, browser_supervisor

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", Registry())
    monkeypatch.setattr(browser_session, "session_for_task", lambda task_id: {})

    assert browser_vault_tool._ensure_supervisor("task-1") is None