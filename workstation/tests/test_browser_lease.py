from __future__ import annotations

import json
import pytest

from tools.close_preview_tool import close_preview_tool
from workstation.browser_session import (
    BrowserControlLeaseManager,
    BrowserControlMode,
    HumanTakeoverActiveError,
    UnifiedBrowserSession,
)


def test_human_takeover_lease_blocks_destructive_actions():
    """Scenario 4: Human takeover blocks destructive agent actions until resumed."""
    mgr = BrowserControlLeaseManager.get_instance()
    task_id = "task_takeover_1"

    # Initially AGENT mode
    assert mgr.get_lease(task_id).mode == BrowserControlMode.AGENT

    # Human takeover requested (e.g. login or captcha)
    lease = mgr.request_human_control(task_id, "User logging into Instagram account")
    assert lease.mode == BrowserControlMode.HUMAN

    # Destructive actions are strictly blocked
    with pytest.raises(HumanTakeoverActiveError, match="Human Takeover is active"):
        mgr.assert_action_allowed(task_id, "close_preview")

    with pytest.raises(HumanTakeoverActiveError, match="Human Takeover is active"):
        mgr.assert_action_allowed(task_id, "browser_navigate")

    with pytest.raises(HumanTakeoverActiveError, match="Human Takeover is active"):
        mgr.assert_action_allowed(task_id, "close_tab")

    # close_preview_tool returns graceful tool error when blocked
    res_raw = close_preview_tool(task_id=task_id)
    res = json.loads(res_raw)
    assert res.get("error") is True or "Human Takeover is active" in str(res)

    # Safe actions (e.g. non-destructive read) are permitted
    mgr.assert_action_allowed(task_id, "read")
    mgr.assert_action_allowed(task_id, "snapshot")

    # Human user finishes and releases control
    resumed = mgr.resume_agent_control(task_id)
    assert resumed.mode == BrowserControlMode.AGENT

    # Now destructive actions are allowed again
    mgr.assert_action_allowed(task_id, "close_preview")
    mgr.assert_action_allowed(task_id, "browser_navigate")


def test_workstation_routed_browser_handler_enforces_lease(monkeypatch):
    """Scenario: workstation_routed_browser_handler raises HumanTakeoverActiveError when human lease is active."""
    from tools.browser_workstation import workstation_routed_browser_handler

    mgr = BrowserControlLeaseManager.get_instance()
    task_id = "task_route_lease_1"
    mgr.request_human_control(task_id, "Manual CAPTCHA solving")

    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.setattr("tools.browser_workstation.workstation_controller_available", lambda force=False: True)

    with pytest.raises(HumanTakeoverActiveError, match="Human Takeover is active"):
        workstation_routed_browser_handler(
            "browser_navigate",
            {"url": "https://example.com"},
            fallback=lambda: "fallback",
            task_id=task_id,
        )

    mgr.resume_agent_control(task_id)
