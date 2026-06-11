import json
import threading
import time


class _Source:
    platform = type("P", (), {"value": "telegram"})()
    chat_id = "chat1"
    thread_id = "topic1"
    user_id = "user1"
    message_id = "m1"


class _OtherSource:
    platform = type("P", (), {"value": "telegram"})()
    chat_id = "chat1"
    thread_id = "topic1"
    user_id = "user2"
    message_id = "m2"


def test_request_sensitive_input_unavailable_does_not_leak(monkeypatch):
    from tools import sensitive_input_tool
    from tools import secret_capture_gateway

    monkeypatch.setattr(secret_capture_gateway, "get_notify", lambda session_key: None)
    result = json.loads(sensitive_input_tool.request_sensitive_input("MY_SECRET", "Send it"))
    assert result["success"] is False
    assert result["secret_capture_status"] == "unavailable"
    assert "super-secret-value" not in json.dumps(result)


def test_request_sensitive_input_uses_cli_secret_callback_when_available(monkeypatch):
    from tools import sensitive_input_tool
    from tools import secret_capture_gateway
    from tools import skills_tool

    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "s-cli")
    monkeypatch.setattr(secret_capture_gateway, "get_notify", lambda session_key: None)
    skills_tool.set_secret_capture_callback(
        lambda env_var, prompt: {
            "success": True,
            "stored_as": env_var,
            "validated": False,
            "skipped": False,
            "value": "super-secret-value",
        }
    )
    try:
        result = json.loads(sensitive_input_tool.request_sensitive_input("MY_SECRET", "Send it"))
    finally:
        skills_tool.set_secret_capture_callback(None)

    assert result["success"] is True
    assert result["stored_as"] == "MY_SECRET"
    assert result["secret_capture_status"] == "stored"
    assert "super-secret-value" not in json.dumps(result)


def test_request_sensitive_input_stores_safe_receipt(monkeypatch):
    from tools import sensitive_input_tool
    from tools import secret_capture_gateway

    saved = {}
    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "s1")

    def fake_save(key, value):
        saved["key"] = key
        saved["value"] = value
        return {"success": True, "stored_as": key, "value": value, "secret": value}

    monkeypatch.setattr("hermes_cli.config.save_env_value_secure", fake_save)

    def notify(entry):
        def resolve():
            time.sleep(0.05)
            secret_capture_gateway.resolve_gateway_secret(entry.secret_id, "super-secret-value")
        threading.Thread(target=resolve, daemon=True).start()

    secret_capture_gateway.register_notify("s1", notify)
    try:
        result = json.loads(sensitive_input_tool.request_sensitive_input("MY_SECRET", "Send it", timeout_seconds=2))
    finally:
        secret_capture_gateway.unregister_notify("s1")

    assert saved == {"key": "MY_SECRET", "value": "super-secret-value"}
    assert result["success"] is True
    assert result["stored_as"] == "MY_SECRET"
    assert result["secret_capture_status"] == "stored"
    assert "super-secret-value" not in json.dumps(result)


def test_secret_capture_resolve_is_single_use_and_source_bound():
    from tools import secret_capture_gateway

    entry = secret_capture_gateway.register("sid-atomic", "session-atomic", "MY_SECRET", "Send it")
    entry.bind_source(_Source())
    try:
        assert secret_capture_gateway.resolve_gateway_secret("sid-atomic", "wrong", source=_OtherSource()) is False
        assert secret_capture_gateway.resolve_gateway_secret("sid-atomic", "first", source=_Source()) is True
        assert secret_capture_gateway.resolve_gateway_secret("sid-atomic", "second", source=_Source()) is False
        assert secret_capture_gateway.cancel_gateway_secret("sid-atomic", "button_cancelled", source=_Source()) is False
        assert entry.value == "first"
        assert entry.cancelled is False
    finally:
        secret_capture_gateway.clear_session("session-atomic")


def test_secret_capture_timeout_marks_reason_and_finalize_callback():
    from tools import secret_capture_gateway

    statuses = []
    entry = secret_capture_gateway.register("sid-timeout", "session-timeout", "MY_SECRET", "Send it")
    secret_capture_gateway.register_finalize("session-timeout", lambda e, status: statuses.append((e.secret_id, status)))
    try:
        resolved = secret_capture_gateway.wait_for_response("sid-timeout", timeout=0)
        assert resolved is entry
        assert resolved is not None
        assert resolved.cancelled is True
        assert resolved.reason == "timeout"
        assert statuses == [("sid-timeout", "timeout")]
        assert secret_capture_gateway.resolve_gateway_secret("sid-timeout", "late") is False
    finally:
        secret_capture_gateway.unregister_notify("session-timeout")


def test_request_sensitive_input_timeout_reports_timeout(monkeypatch):
    from tools import sensitive_input_tool
    from tools import secret_capture_gateway

    monkeypatch.setattr("tools.approval.get_current_session_key", lambda default="": "s-timeout")
    secret_capture_gateway.register_notify("s-timeout", lambda entry: None)
    try:
        result = json.loads(sensitive_input_tool.request_sensitive_input("MY_SECRET", "Send it", timeout_seconds=0))
    finally:
        secret_capture_gateway.unregister_notify("s-timeout")

    assert result["success"] is False
    assert result["secret_capture_status"] == "timeout"
    assert result["skipped"] is True
