import json

from tools import notification_tool


def test_email_adapter_unavailable_without_sendgrid_key(monkeypatch):
    monkeypatch.setattr(notification_tool.sql, "runtime_env", lambda: {})
    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)

    result = notification_tool._email_adapter_send({"to_email": "client@example.com", "subject": "Hi", "text": "Hello"})

    assert result["ok"] is False
    assert result["configured"] is False
    assert result["adapter"] == "sendgrid"
    assert result["status"] == "unavailable"


def test_sendgrid_payload_is_provider_specific_behind_generic_contract(monkeypatch):
    captured = {}
    monkeypatch.setattr(notification_tool.sql, "runtime_env", lambda: {
        "SENDGRID_API_KEY": "test-key",
        "SENDGRID_FROM_EMAIL": "zeus@sitiouno.com",
        "SENDGRID_FROM_NAME": "Zeus",
    })

    def fake_request(path, body):
        captured["path"] = path
        captured["body"] = body
        return {"ok": True, "status": 202, "data": None}

    monkeypatch.setattr(notification_tool, "_sendgrid_request", fake_request)

    result = notification_tool._email_adapter_send({
        "to_email": "client@example.com",
        "to_name": "Client",
        "subject": "Quote ready",
        "text": "Please review your quote.",
        "html": "<p>Please review your quote.</p>",
        "metadata": {"workspace_id": "ws-1"},
    })

    assert result["ok"] is True
    assert result["adapter"] == "sendgrid"
    assert captured["path"] == "/v3/mail/send"
    assert captured["body"]["from"] == {"email": "zeus@sitiouno.com", "name": "Zeus"}
    assert captured["body"]["personalizations"][0]["to"] == [{"email": "client@example.com", "name": "Client"}]
    assert captured["body"]["custom_args"] == {"workspace_id": "ws-1"}


def test_notification_email_send_handler_returns_adapter_result(monkeypatch):
    monkeypatch.setattr(notification_tool, "_email_adapter_send", lambda payload: {"ok": True, "adapter": "sendgrid", "status": "sent", "request_id": "abc"})

    payload = json.loads(notification_tool._handle_email_send({
        "to_email": "client@example.com",
        "subject": "Hello",
        "text": "Body",
    }))

    assert payload["ok"] is True
    assert payload["result"]["adapter"] == "sendgrid"


def test_toolset_exports_notification_tools():
    import toolsets

    notification_tools = set(toolsets.TOOLSETS["notifications"]["tools"])

    assert "notification_status" in notification_tools
    assert "notification_email_send" in notification_tools
