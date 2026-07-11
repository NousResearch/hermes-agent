"""Real guard-path coverage for no-callback approval retries."""

import json

import tools.approval as approval


SESSION = "fallback-execution-session"


def _reset_state():
    with approval._lock:
        approval._gateway_queues.clear()
        approval._gateway_notify_cbs.clear()
        approval._session_approved.clear()
        approval._permanent_approved.clear()
        approval._pending.clear()
        approval._pending_by_session.clear()


def _gateway_session(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    token = approval.set_current_session_key(SESSION)
    _reset_state()
    return token


def _finish_session(token):
    approval.reset_current_session_key(token)
    _reset_state()


def test_terminal_fallback_retry_consumes_exact_approval(monkeypatch):
    token = _gateway_session(monkeypatch)
    try:
        command = "rm -rf /tmp/fallback-a"
        pending = approval.check_all_command_guards(command, "local")
        assert pending["status"] == "pending_approval"
        for key in ("request_id", "argument_hash", "operation", "tool_name"):
            assert key in pending

        assert approval.resolve_gateway_approval(
            SESSION, "once", request_id=pending["request_id"]
        ) == 1

        changed = approval.check_all_command_guards("rm -rf /tmp/fallback-b", "local")
        assert changed["approved"] is False
        assert changed.get("status") != "pending_approval"

        pending = approval.check_all_command_guards(command, "local")
        assert approval.resolve_gateway_approval(
            SESSION, "once", request_id=pending["request_id"]
        ) == 1
        retried = approval.check_all_command_guards(command, "local")
        assert retried["approved"] is True
        assert pending["request_id"] not in approval._pending
        assert SESSION not in approval._pending_by_session
    finally:
        _finish_session(token)


def test_execute_code_fallback_retry_consumes_exact_approval(monkeypatch):
    token = _gateway_session(monkeypatch)
    try:
        code = "import os\nprint(1)"
        pending = approval.check_execute_code_guard(code, "local")
        assert pending["status"] == "pending_approval"
        for key in ("request_id", "argument_hash", "operation", "tool_name"):
            assert key in pending

        assert approval.resolve_gateway_approval(
            SESSION, "once", request_id=pending["request_id"]
        ) == 1

        changed = approval.check_execute_code_guard("import os\nprint(2)", "local")
        assert changed["approved"] is False
        assert changed.get("status") != "pending_approval"

        pending = approval.check_execute_code_guard(code, "local")
        assert approval.resolve_gateway_approval(
            SESSION, "once", request_id=pending["request_id"]
        ) == 1
        retried = approval.check_execute_code_guard(code, "local")
        assert retried["approved"] is True
        assert pending["request_id"] not in approval._pending
        assert SESSION not in approval._pending_by_session
    finally:
        _finish_session(token)


def test_terminal_fallback_deny_retry_is_blocked_with_reason(monkeypatch):
    token = _gateway_session(monkeypatch)
    try:
        command = "rm -rf /tmp/fallback-denied"
        pending = approval.check_all_command_guards(command, "local")
        reason = "wrong directory"
        assert approval.resolve_gateway_approval(
            SESSION,
            "deny",
            request_id=pending["request_id"],
            reason=reason,
        ) == 1

        retried = approval.check_all_command_guards(command, "local")
        assert retried["approved"] is False
        assert retried["status"] == "blocked"
        assert retried["outcome"] == "denied"
        assert retried["resolution_reason"] == reason
        assert retried["deny_reason"] == reason
        assert reason in retried["message"]
        assert pending["request_id"] not in approval._pending
    finally:
        _finish_session(token)


def test_execute_code_fallback_deny_retry_is_blocked_with_reason(monkeypatch):
    token = _gateway_session(monkeypatch)
    try:
        code = "import os\nprint('denied')"
        pending = approval.check_execute_code_guard(code, "local")
        reason = "use the terminal tool"
        assert approval.resolve_gateway_approval(
            SESSION,
            "deny",
            request_id=pending["request_id"],
            reason=reason,
        ) == 1

        retried = approval.check_execute_code_guard(code, "local")
        assert retried["approved"] is False
        assert retried["status"] == "blocked"
        assert retried["outcome"] == "denied"
        assert retried["resolution_reason"] == reason
        assert retried["deny_reason"] == reason
        assert reason in retried["message"]
        assert pending["request_id"] not in approval._pending
    finally:
        _finish_session(token)


def _pending_identity_payload(operation):
    return {
        "approved": False,
        "status": "pending_approval",
        "approval_pending": True,
        "request_id": f"{operation}-request",
        "argument_hash": f"{operation}-hash",
        "operation": operation,
        "tool_name": operation,
        "created_at": 1.0,
        "expires_at": 2.0,
        "command": "redacted",
        "description": "approval required",
        "pattern_key": operation,
    }


def test_terminal_public_tool_preserves_pending_identity(monkeypatch):
    from tools import terminal_tool

    pending = _pending_identity_payload("terminal")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {
        "env_type": "local", "cwd": ".", "timeout": 30,
    })
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _: "default")
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *args, **kwargs: pending)
    monkeypatch.setitem(terminal_tool._active_environments, "default", object())

    result = json.loads(terminal_tool.terminal_tool("rm -rf /tmp/a"))
    for key in ("request_id", "argument_hash", "operation", "tool_name"):
        assert result[key] == pending[key]


def test_execute_code_public_tool_preserves_pending_identity(monkeypatch):
    from tools import code_execution_tool, terminal_tool

    pending = _pending_identity_payload("execute_code")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(approval, "check_execute_code_guard", lambda *args, **kwargs: pending)

    result = json.loads(code_execution_tool.execute_code("print(1)", task_id="identity-test"))
    for key in ("request_id", "argument_hash", "operation", "tool_name"):
        assert result[key] == pending[key]


def test_pending_metadata_is_deep_copied_and_hash_is_required():
    _reset_state()
    arguments = {"nested": {"value": 1}}
    request = approval.submit_pending(
        SESSION,
        {
            "operation": "execute_code",
            "tool_name": "execute_code",
            "arguments": arguments,
        },
    )
    arguments["nested"]["value"] = 2
    request["arguments"]["nested"]["value"] = 3

    stored = approval.get_pending_approval(request["request_id"])
    assert stored is not None
    assert stored["arguments"] == {"nested": {"value": 1}}
    assert approval.resolve_gateway_approval(
        SESSION, "once", request_id=request["request_id"]
    ) == 1
    assert approval.consume_pending_approval(SESSION, request["request_id"]) is None
    assert approval.consume_pending_approval(
        SESSION,
        request["request_id"],
        request_hash=stored["argument_hash"],
    ) is None
