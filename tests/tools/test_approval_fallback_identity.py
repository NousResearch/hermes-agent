"""Regression tests for identity and fail-closed fallback approvals."""

import time

import tools.approval as approval


SESSION = "fallback-identity-session"


def _clear_state():
    with approval._lock:
        approval._pending.clear()
        getattr(approval, "_pending_by_session", {}).clear()


def _submit(command="rm -rf /tmp/a", **extra):
    data = {
        "operation": "terminal",
        "tool_name": "terminal",
        "arguments": {"command": command},
        "pattern_key": "recursive delete",
        "requester": "user-1",
        "channel": "api",
    }
    data.update(extra)
    return approval.submit_pending(SESSION, data)


class TestFallbackApprovalIdentity:
    def setup_method(self):
        _clear_state()

    def teardown_method(self):
        _clear_state()

    def test_two_pending_approvals_are_independently_addressable(self):
        first = _submit(command="rm -rf /tmp/a")
        second = _submit(command="rm -rf /tmp/b")

        assert first["request_id"] != second["request_id"]
        assert first["argument_hash"] != second["argument_hash"]
        assert first["operation"] == "terminal"
        assert first["tool_name"] == "terminal"
        assert first["policy_key"] == "recursive delete"
        assert first["requester"] == "user-1"
        assert first["session_key"] == SESSION
        assert first["channel"] == "api"
        assert first["created_at"] <= first["expires_at"]
        assert approval.resolve_gateway_approval(
            SESSION,
            "once",
            request_id=second["request_id"],
            request_hash=second["argument_hash"],
        ) == 1
        assert approval._pending[second["request_id"]]["resolution"] == "once"
        assert approval._pending[first["request_id"]]["resolution"] is None
        assert approval.resolve_gateway_approval(
            SESSION,
            "once",
            request_id=second["request_id"],
            request_hash=second["argument_hash"],
        ) == 0

    def test_changed_arguments_fail_closed(self):
        request = _submit(command="rm -rf /tmp/a")

        assert approval.resolve_gateway_approval(
            SESSION,
            "once",
            request_id=request["request_id"],
            request_hash="changed-arguments",
        ) == 0
        assert approval._pending[request["request_id"]]["resolution"] is None

    def test_expired_request_fails_closed(self):
        request = _submit(expires_at=time.time() - 1)

        assert approval.resolve_gateway_approval(
            SESSION,
            "once",
            request_id=request["request_id"],
            request_hash=request["argument_hash"],
        ) == 0
        assert approval._pending[request["request_id"]]["status"] == "expired"
        assert approval._pending[request["request_id"]]["resolution"] is None

    def test_post_restart_in_memory_request_fails_closed(self):
        request = _submit()
        with approval._lock:
            approval._pending.clear()
            getattr(approval, "_pending_by_session", {}).clear()

        assert approval.resolve_gateway_approval(
            SESSION,
            "once",
            request_id=request["request_id"],
            request_hash=request["argument_hash"],
        ) == 0

    def test_legacy_session_resolution_is_fifo_and_resolve_all_is_preserved(self, caplog):
        caplog.set_level("INFO")
        first = _submit(command="rm -rf /tmp/a")
        second = _submit(command="rm -rf /tmp/b")
        third = _submit(command="rm -rf /tmp/c")

        assert approval.resolve_gateway_approval(SESSION, "once") == 1
        assert approval._pending[first["request_id"]]["resolution"] == "once"
        assert approval._pending[second["request_id"]]["resolution"] is None
        assert "session_fifo" in caplog.text
        assert first["request_id"] in caplog.text

        assert approval.resolve_gateway_approval(SESSION, "deny", resolve_all=True) == 2
        assert approval._pending[second["request_id"]]["resolution"] == "deny"
        assert approval._pending[third["request_id"]]["resolution"] == "deny"
