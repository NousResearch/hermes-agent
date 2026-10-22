"""Vault inspection and secret evaluation must follow the last navigation."""
from unittest.mock import Mock

import pytest

from tools import browser_vault_tool as vault


@pytest.fixture
def routed_browser(monkeypatch):
    from tools import browser_supervisor, browser_tool, browser_tool_cdp, browser_tool_session

    task, effective = "vault-task", "vault-task::local"
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {task: effective})
    monkeypatch.setattr(browser_tool, "_active_sessions", {
        task: {"owner_task_id": task, "session_key": task},
        effective: {"owner_task_id": task, "session_key": effective},
    })
    cloud = Mock()
    local = Mock()
    local.evaluate_runtime.return_value = {"ok": True, "result": "local-result"}
    registry = Mock()
    registry.get.side_effect = lambda key: {task: cloud, effective: local}.get(key)
    registry.get_or_start.return_value = local
    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", registry)
    run = Mock(return_value={"success": True, "data": {"cdpUrl": "ws://127.0.0.1:1234/fixture"}})
    monkeypatch.setattr(browser_tool_session, "_run_browser_command", run)
    monkeypatch.setattr(browser_tool_cdp, "_resolve_cdp_override", lambda url: url)
    monkeypatch.setattr(browser_tool_cdp, "_get_dialog_policy_config", lambda: ("dismiss", 10))
    return task, effective, cloud, local, registry, run


def test_inspection_and_secret_use_navigation_supervisor(routed_browser):
    task, effective, cloud, local, registry, run = routed_browser
    assert vault._eval_js(task, "inspection")["result"] == "local-result"
    assert vault._eval_js_secret(task, "synthetic-fill")["result"] == "local-result"
    cloud.evaluate_runtime.assert_not_called()
    run.assert_not_called()


def test_attach_and_fallback_keep_navigation_session(routed_browser):
    task, effective, cloud, local, registry, run = routed_browser
    registry.get.side_effect = lambda key: cloud if key == task else None
    assert vault._ensure_supervisor(task) is local
    run.assert_called_once_with(effective, "get", ["cdp-url"])
    assert registry.get_or_start.call_args.kwargs["task_id"] == effective

    run.reset_mock()
    run.return_value = {"success": True, "data": {"result": "fallback-result"}}
    assert vault._eval_js(task, "inspection")["result"] == "fallback-result"
    run.assert_called_once_with(effective, "eval", ["inspection"])

    run.reset_mock()
    run.return_value = {"success": False}
    assert vault._eval_js_secret(task, "synthetic-fill")["error_type"] == "supervisor_required"
    run.assert_called_once_with(effective, "get", ["cdp-url"])
    cloud.evaluate_runtime.assert_not_called()
