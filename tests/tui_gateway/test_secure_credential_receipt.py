"""Desktop secure-credential receipts must never be mistaken for secret text."""

from __future__ import annotations


def _install_secret_callback(monkeypatch, block_result):
    from tools import project_tools, skills_tool, terminal_tool
    from tui_gateway import server

    captured = {}
    monkeypatch.setattr(terminal_tool, "set_sudo_password_callback", lambda _callback: None)
    monkeypatch.setattr(project_tools, "set_project_workspace_callback", lambda _callback: None)
    monkeypatch.setattr(
        skills_tool,
        "set_secret_capture_callback",
        lambda callback: captured.setdefault("callback", callback),
    )
    monkeypatch.setattr(server, "_block", lambda *_args, **_kwargs: block_result)

    server._wire_callbacks("session-1")
    return captured["callback"]


def test_stored_receipt_returns_success_without_resaving(monkeypatch):
    from hermes_cli import config

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the non-secret receipt was passed to secret storage")

    monkeypatch.setattr(config, "save_env_value_secure", forbidden)
    callback = _install_secret_callback(monkeypatch, {"stored": True})

    result = callback("A", "Enter a credential")

    assert result == {
        "success": True,
        "stored_as": "A",
        "validated": False,
        "skipped": False,
        "message": "ok",
    }


def test_legacy_string_response_still_uses_secure_storage(monkeypatch):
    from hermes_cli import config

    saved = []

    def save(key, value):
        saved.append((key, value))
        return {"success": True, "stored_as": key, "validated": True}

    monkeypatch.setattr(config, "save_env_value_secure", save)
    callback = _install_secret_callback(monkeypatch, "  secret with spaces  ")

    result = callback("API_KEY", "Enter a credential")

    assert saved == [("API_KEY", "  secret with spaces  ")]
    assert result["skipped"] is False
