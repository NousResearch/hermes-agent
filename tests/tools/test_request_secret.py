"""Tests for the assistant-invokable secure secret request tool."""

import json

from tools import skills_tool
from tools.registry import registry


def test_request_secret_routes_to_existing_capture_callback(monkeypatch):
    calls = []

    def fake_capture(env_var, prompt, metadata):
        calls.append((env_var, prompt, metadata))
        return {
            "success": True,
            "stored_as": env_var,
            "validated": True,
            "skipped": False,
            "message": "ok",
        }

    monkeypatch.setattr(skills_tool, "capture_secret", fake_capture)

    result = json.loads(
        skills_tool.request_secret(
            " PENPOT_MCP_TOKEN ",
            " Enter the fresh Penpot MCP token ",
            task_id="session-123",
        )
    )

    assert result == {
        "success": True,
        "stored_as": "PENPOT_MCP_TOKEN",
        "validated": True,
        "skipped": False,
        "message": "ok",
    }
    assert calls == [
        (
            "PENPOT_MCP_TOKEN",
            "Enter the fresh Penpot MCP token",
            {"source": "assistant_secret_request", "task_id": "session-123"},
        )
    ]


def test_request_secret_never_returns_secret_value(monkeypatch):
    secret = "do-not-return-this-value"
    monkeypatch.setattr(
        skills_tool,
        "capture_secret",
        lambda *_args, **_kwargs: {
            "success": True,
            "stored_as": "PENPOT_MCP_TOKEN",
            "validated": True,
            "skipped": False,
            "message": "ok",
        },
    )

    output = skills_tool.request_secret("PENPOT_MCP_TOKEN", "Enter token")

    assert secret not in output
    assert json.loads(output)["success"] is True


def test_request_secret_tool_is_registered_in_skills_toolset():
    entry = registry.get_entry("request_secret")

    assert entry is not None
    assert entry.toolset == "skills"
    assert entry.schema["parameters"]["required"] == ["env_var", "prompt"]
