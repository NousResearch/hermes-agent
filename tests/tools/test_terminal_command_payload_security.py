"""Security invariants for large terminal command payloads."""

import json
import logging

import tools.terminal_tool as terminal_tool


def _config(cwd):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 1,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


def _patch_terminal(monkeypatch, tmp_path, env):
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(
        terminal_tool, "_get_env_config", lambda: _config(str(tmp_path))
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_container_task_id",
        lambda value: value or "default",
    )


def test_original_command_is_guarded_before_environment_execution(
    monkeypatch, tmp_path
):
    command = "rm -rf guarded-marker\n" + ("# payload\n" * 1200)
    observed = []

    class Env:
        env = {}

        def execute(self, command, **kwargs):
            raise AssertionError("denied command reached environment execution")

    _patch_terminal(monkeypatch, tmp_path, Env())

    def deny(candidate, env_type, **kwargs):
        observed.append(candidate)
        return {"approved": False, "description": "dangerous command"}

    monkeypatch.setattr(terminal_tool, "_check_all_guards", deny)

    result = json.loads(terminal_tool.terminal_tool(command))

    assert observed == [command]
    assert result["status"] == "blocked"


def test_short_command_preview_force_redacts_secret():
    secret = "sk-proj-PreviewSecret123456789012345"

    preview = terminal_tool._safe_command_preview(
        f"curl -H 'Authorization: Bearer {secret}' https://example.invalid"
    )

    assert secret not in preview
    assert "***" in preview


def test_long_multiline_command_preview_contains_only_metadata():
    marker = "FULL_SCRIPT_BODY_MUST_NOT_BE_LOGGED_4ca1"
    command = marker + "\n" + ("echo payload\n" * 100)

    preview = terminal_tool._safe_command_preview(command)

    assert marker not in preview
    assert "echo payload" not in preview
    assert f"chars={len(command)}" in preview
    assert f"lines={command.count(chr(10)) + 1}" in preview


def test_retry_log_omits_long_command_body(monkeypatch, tmp_path, caplog):
    marker = "RETRY_LOG_SCRIPT_BODY_916e"
    command = marker + "\n" + ("echo payload\n" * 100)

    class FailingEnv:
        env = {}

        def execute(self, command, **kwargs):
            raise RuntimeError("backend unavailable")

    _patch_terminal(monkeypatch, tmp_path, FailingEnv())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *args, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda seconds: None)

    with caplog.at_level(logging.WARNING, logger="tools.terminal_tool"):
        terminal_tool.terminal_tool(command)

    assert marker not in caplog.text
    assert "echo payload" not in caplog.text
    assert f"chars={len(command)}" in caplog.text
