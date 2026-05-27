"""Tests for the PR-Agent wrapper tool."""

import json
import subprocess

import tools.pr_agent_tool as pr_tool
from toolsets import TOOLSETS, resolve_toolset


def test_build_default_review_command():
    assert pr_tool._build_pr_agent_command("https://github.com/org/repo/pull/1") == [
        "pr-agent",
        "--pr_url=https://github.com/org/repo/pull/1",
        "review",
        "--config.publish_output=false",
        "--config.model=openrouter/deepseek/deepseek-v4-flash",
        "--config.fallback_models=[]",
        "--config.custom_model_max_tokens=32000",
        "--config.max_model_tokens=32000",
        "--config.temperature=0.2",
    ]


def test_build_ask_command_requires_question():
    result = json.loads(
        pr_tool.pr_agent("https://github.com/org/repo/pull/1", action="ask", workdir=".")
    )

    assert result["status"] == "error"
    assert "question" in result["error"]


def test_build_ask_command_with_model_prefix():
    command = pr_tool._build_pr_agent_command(
        "https://github.com/org/repo/pull/1",
        action="ask",
        question="Summarize this PR.",
        model="deepseek/deepseek-v4-flash",
        publish_output=True,
    )

    assert command[:4] == [
        "pr-agent",
        "--pr_url=https://github.com/org/repo/pull/1",
        "ask",
        "Summarize this PR.",
    ]
    assert "--config.publish_output=true" in command
    assert "--config.model=openrouter/deepseek/deepseek-v4-flash" in command


def test_config_path_is_passed_after_action():
    command = pr_tool._build_pr_agent_command(
        "https://github.com/org/repo/pull/1",
        action="ask",
        question="Summarize this PR.",
        config_path="/tmp/pr_agent.toml",
    )

    assert command[2:5] == ["ask", "Summarize this PR.", "--config_path=/tmp/pr_agent.toml"]


def test_env_file_loads_openrouter_and_github_tokens(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY='or-secret'\nGITHUB_TOKEN=gh-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB__USER_TOKEN", raising=False)
    monkeypatch.setattr(pr_tool, "_run_gh_auth_token", lambda: None)

    env = pr_tool._build_pr_agent_env(
        "openrouter/deepseek/deepseek-v4-flash",
        env_file=str(env_file),
    )

    assert env["OPENROUTER_API_KEY"] == "or-secret"
    assert env["OPENAI_API_KEY"] == "or-secret"
    assert env["GITHUB__USER_TOKEN"] == "gh-secret"


def test_missing_pr_agent_binary_returns_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("PR_AGENT_BIN", raising=False)
    monkeypatch.setattr(pr_tool.shutil, "which", lambda _name: None)

    result = json.loads(
        pr_tool.pr_agent(
            "https://github.com/org/repo/pull/1",
            action="ask",
            question="Summarize this PR.",
            workdir=str(tmp_path),
        )
    )

    assert result["status"] == "error"
    assert "PR-Agent executable not found" in result["error"]


def test_subprocess_receives_minimal_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(pr_tool.shutil, "which", lambda name: "pr-agent" if name == "pr-agent" else None)
    monkeypatch.setattr(pr_tool.subprocess, "run", fake_run)

    result = json.loads(
        pr_tool.pr_agent(
            "https://github.com/org/repo/pull/1",
            action="ask",
            question="Summarize this PR.",
            workdir=str(tmp_path),
            timeout_seconds=42,
        )
    )

    assert result["status"] == "completed"
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"]["OPENROUTER_API_KEY"] == "or-secret"
    assert captured["env"]["OPENAI_API_KEY"] == "or-secret"
    assert captured["env"]["GITHUB__USER_TOKEN"] == "gh-secret"
    assert "OPENAI_BASE_URL" not in captured["env"]
    assert captured["command"][2:4] == ["ask", "Summarize this PR."]


def test_toolset_exposes_pr_agent():
    assert TOOLSETS["pr_agent"]["tools"] == ["pr_agent"]
    assert "pr_agent" in resolve_toolset("hermes-cli")
    assert "pr_agent" in resolve_toolset("hermes-acp")