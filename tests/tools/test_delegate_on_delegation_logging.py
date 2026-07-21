"""Regression test for #7198: debug logging for on_delegation exceptions.

When the on_delegation notification handler in delegate_task catches an
exception from the subagent-stop hook (or the memory manager's on_delegation
call), it must DEBUG-log the failure with enough context to diagnose
(task index + error message) instead of silently swallowing it.
"""
import logging
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def test_on_delegation_exception_is_debug_logged(caplog):
    """When on_delegation notification raises, the debug log must capture it."""
    from tools.delegate_tool import delegate_task

    # Minimal mock parent agent
    parent = MagicMock()
    parent._credential_pool = None
    parent.session_id = "test-session"
    parent._memory_manager = MagicMock()
    # Make on_delegation raise to trigger the outer handler
    parent._memory_manager.on_delegation.side_effect = RuntimeError("provider down")

    with patch("run_agent.AIAgent") as MockAgent:
        child = MagicMock()
        child.run.return_value = ("done", [{"summary": "result", "status": "completed", "duration_seconds": 0.1}])
        child.session_id = "child-1"
        child.session_estimated_cost_usd = 0.0
        child.session_cost_source = "none"
        MockAgent.return_value = child

        with caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"):
            delegate_task(
                goal="test task",
                parent_agent=parent,
                task_count=1,
            )

    # The debug log must mention on_delegation and the error
    delegation_logs = [
        r for r in caplog.records
        if "on_delegation" in r.message
    ]
    assert len(delegation_logs) >= 1, (
        "Expected a DEBUG log about on_delegation failure, got none"
    )
    assert "provider down" in delegation_logs[0].message or "failed" in delegation_logs[0].message.lower()
