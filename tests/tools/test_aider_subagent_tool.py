"""Tests for the Aider subagent tool."""

import json
import subprocess

import tools.aider_subagent_tool as aider_tool
from toolsets import TOOLSETS, resolve_toolset


def test_build_default_command():
    assert aider_tool._build_aider_command("Fix the tests") == [
        "aider",
        "--message",
        "Fix the tests",
        "--yes",
    ]


def test_build_model_command_prefixes_openrouter():
    assert aider_tool._build_aider_command(
        "Fix the tests",
        model="deepseek/deepseek-v4-flash",
    ) == [
        "aider",
        "--message",
        "Fix the tests",
        "--yes",
        "--model",
        "openrouter/deepseek/deepseek-v4-flash",
    ]


def test_build_model_command_preserves_openrouter_prefix():
    command = aider_tool._build_aider_command(
        "Fix the tests",
        model="openrouter/deepseek/deepseek-v4-flash",
    )

    assert command[-2:] == ["--model", "openrouter/deepseek/deepseek-v4-flash"]


def test_missing_instruction_returns_error():
    result = json.loads(aider_tool.aider_subagent("  "))

    assert result["status"] == "error"
    assert "instruction" in result["error"]


def test_missing_aider_binary_returns_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("AIDER_BIN", raising=False)
    monkeypatch.setattr(aider_tool.shutil, "which", lambda _name: None)

    result = json.loads(aider_tool.aider_subagent("Fix the tests", workdir=str(tmp_path)))

    assert result["status"] == "error"
    assert "Aider executable not found" in result["error"]
    assert result["command"] == "aider --message '<instruction>' --yes"


def test_command_preview_redacts_full_instruction():
    command = aider_tool._build_aider_command(
        "Fix this and include SECRET_TOKEN=abc123",
        model="deepseek/deepseek-v4-flash",
    )

    preview = aider_tool._command_preview(command)

    assert "SECRET_TOKEN" not in preview
    assert "<instruction>" in preview
    assert "--model openrouter/deepseek/deepseek-v4-flash" in preview


def test_subprocess_receives_expected_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(aider_tool.shutil, "which", lambda _name: "aider")
    monkeypatch.setattr(aider_tool.subprocess, "run", fake_run)

    result = json.loads(
        aider_tool.aider_subagent(
            "Fix the tests",
            model="deepseek/deepseek-v4-flash",
            workdir=str(tmp_path),
            timeout_seconds=42,
        )
    )

    assert result["status"] == "completed"
    assert result["command"] == (
        "aider --message '<instruction>' --yes --model openrouter/deepseek/deepseek-v4-flash"
    )
    assert captured["command"] == [
        "aider",
        "--message",
        "Fix the tests",
        "--yes",
        "--model",
        "openrouter/deepseek/deepseek-v4-flash",
    ]
    assert captured["cwd"] == str(tmp_path)
    assert captured["timeout"] == 42


def test_toolset_exposes_aider_subagent():
    assert TOOLSETS["aider"]["tools"] == ["aider_subagent"]
    assert "aider_subagent" in resolve_toolset("hermes-cli")