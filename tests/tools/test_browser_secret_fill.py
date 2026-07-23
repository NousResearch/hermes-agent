from __future__ import annotations

import json
import subprocess

import pytest


@pytest.fixture(autouse=True)
def _disable_camofox(monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)


def test_resolve_onepassword_secret_ref_uses_op_read(monkeypatch):
    import tools.browser_tool as bt

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="secret-value\n", stderr="")

    monkeypatch.setattr(bt, "_find_onepassword_cli", lambda: "/usr/bin/op")
    monkeypatch.setattr(bt.subprocess, "run", fake_run)

    value = bt._resolve_onepassword_secret(secret_ref="op://Private/Login/password")

    assert value == "secret-value"
    assert calls[0][0] == ["/usr/bin/op", "read", "op://Private/Login/password"]
    assert calls[0][1]["capture_output"] is True


def test_resolve_onepassword_item_field_matches_purpose(monkeypatch):
    import tools.browser_tool as bt

    payload = {
        "fields": [
            {"id": "one", "label": "Email", "purpose": "USERNAME", "value": "person@example.com"},
            {"id": "two", "label": "Hidden", "purpose": "PASSWORD", "value": "secret-value"},
        ]
    }

    monkeypatch.setattr(bt, "_run_onepassword", lambda _args: json.dumps(payload))

    assert bt._resolve_onepassword_secret(item="Example", field="password") == "secret-value"


def test_browser_secret_fill_does_not_echo_secret(monkeypatch):
    import tools.browser_tool as bt

    inserted = {}

    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: f"session:{task_id}")
    monkeypatch.setattr(bt, "_resolve_onepassword_secret", lambda **_kwargs: "super-secret-value")
    monkeypatch.setattr(bt, "_run_browser_command", lambda *a, **kw: {"success": True})
    monkeypatch.setattr(bt, "_clear_focused_browser_field", lambda task_id: {"success": True})

    def fake_insert(task_id, text):
        inserted["task_id"] = task_id
        inserted["text"] = text
        return {"success": True}

    monkeypatch.setattr(bt, "_insert_text_via_cdp", fake_insert)

    result = json.loads(
        bt.browser_secret_fill(
            ref="e2",
            item="Example Login",
            field="password",
            task_id="task-1",
        )
    )

    assert inserted == {"task_id": "session:task-1", "text": "super-secret-value"}
    assert result == {
        "success": True,
        "filled": True,
        "element": "@e2",
        "source": "1password",
        "cleared": True,
    }
    assert "super-secret-value" not in json.dumps(result)


def test_browser_secret_fill_insert_error_does_not_echo_secret(monkeypatch):
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(bt, "_resolve_onepassword_secret", lambda **_kwargs: "super-secret-value")
    monkeypatch.setattr(bt, "_run_browser_command", lambda *a, **kw: {"success": True})
    monkeypatch.setattr(bt, "_clear_focused_browser_field", lambda task_id: {"success": True})
    monkeypatch.setattr(
        bt,
        "_insert_text_via_cdp",
        lambda task_id, text: {"success": False, "error": "CDP text insertion failed."},
    )

    result = json.loads(bt.browser_secret_fill(ref="@e2", secret_ref="op://Private/Login/password"))

    assert result["success"] is False
    assert result["error"] == "CDP text insertion failed."
    assert "super-secret-value" not in json.dumps(result)
