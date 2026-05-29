"""Tests for tools/agent_message_tool.py."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.agent_message_tool import (
    AGENT_MESSAGE_SCHEMA,
    _available_profiles,
    _build_agent_message_command,
    _profile_exists,
    _validate_profile_name,
    agent_message_tool,
)
from tools.registry import registry


def _make_profile(root: Path, name: str) -> None:
    if name == "default":
        home = root
    else:
        home = root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  default: test\n", encoding="utf-8")


def test_validate_profile_name_rejects_shell_metacharacters():
    for bad in ["../narvi", "narvi;rm -rf /", "narvi test", "-narvi", ""]:
        try:
            _validate_profile_name(bad)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"accepted unsafe profile name: {bad!r}")


def test_profile_discovery_uses_configured_profiles_root(tmp_path):
    _make_profile(tmp_path, "default")
    _make_profile(tmp_path, "narvi")

    assert _profile_exists("default", root=tmp_path)
    assert _profile_exists("narvi", root=tmp_path)
    assert _available_profiles(root=tmp_path) == ["default", "narvi"]


def test_build_command_is_argv_list_no_shell(monkeypatch):
    monkeypatch.setattr("tools.agent_message_tool._resolve_hermes_executable", lambda: "/bin/hermes")

    command = _build_agent_message_command("narvi", "hello; still one arg")

    assert command == [
        "/bin/hermes",
        "--profile",
        "narvi",
        "chat",
        "-Q",
        "-q",
        "hello; still one arg",
    ]


def test_agent_message_sync_success(monkeypatch, tmp_path):
    _make_profile(tmp_path, "narvi")
    monkeypatch.setattr("tools.agent_message_tool.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("tools.agent_message_tool._resolve_hermes_executable", lambda: "/bin/hermes")

    completed = SimpleNamespace(
        returncode=0,
        stdout="session_id: 20260529_test\nNarvi reply\n",
        stderr="",
    )

    with patch("tools.agent_message_tool.subprocess.run", return_value=completed) as run_mock:
        result = json.loads(
            agent_message_tool(
                {
                    "target_profile": "narvi",
                    "message": "Can you hear me?",
                    "timeout_seconds": 12,
                }
            )
        )

    assert result["success"] is True
    assert result["target_profile"] == "narvi"
    assert result["session_id"] == "20260529_test"
    assert result["reply"] == "Narvi reply"
    run_mock.assert_called_once()
    command = run_mock.call_args.args[0]
    kwargs = run_mock.call_args.kwargs
    assert isinstance(command, list)
    assert kwargs["timeout"] == 12
    assert "shell" not in kwargs
    assert command[-1] == "Can you hear me?"


def test_agent_message_timeout_returns_partial_output(monkeypatch, tmp_path):
    _make_profile(tmp_path, "narvi")
    monkeypatch.setattr("tools.agent_message_tool.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("tools.agent_message_tool._resolve_hermes_executable", lambda: "/bin/hermes")

    exc = subprocess.TimeoutExpired(
        cmd=["/bin/hermes"],
        timeout=1,
        output="partial stdout",
        stderr="partial stderr",
    )
    with patch("tools.agent_message_tool.subprocess.run", side_effect=exc):
        result = json.loads(
            agent_message_tool(
                {
                    "target_profile": "narvi",
                    "message": "slow request",
                    "timeout_seconds": 1,
                }
            )
        )

    assert result["success"] is False
    assert "timed out" in result["error"]
    assert "partial stdout" in result["partial_output"]
    assert "partial stderr" in result["partial_output"]


def test_agent_message_missing_profile_reports_available(monkeypatch, tmp_path):
    _make_profile(tmp_path, "aragorn")
    monkeypatch.setattr("tools.agent_message_tool.get_default_hermes_root", lambda: tmp_path)

    result = json.loads(agent_message_tool({"target_profile": "narvi", "message": "hello"}))

    assert result["success"] is False
    assert "not found" in result["error"]
    assert result["available_profiles"] == ["aragorn"]


def test_agent_message_registered_in_messaging_toolset():
    from toolsets import TOOLSETS, resolve_toolset

    entry = registry.get_entry("agent_message")

    assert entry is not None
    assert entry.toolset == "messaging"
    assert entry.schema is AGENT_MESSAGE_SCHEMA
    assert "agent_message" in TOOLSETS["messaging"]["tools"]
    assert "agent_message" in resolve_toolset("messaging")
