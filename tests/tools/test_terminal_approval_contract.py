import json
from unittest.mock import MagicMock


def _make_env_config(tmp_path):
    return {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


def _run_terminal(monkeypatch, tmp_path, approval):
    import tools.terminal_tool as terminal_tool_module

    mock_env = MagicMock()

    monkeypatch.setattr(
        terminal_tool_module, "_get_env_config", lambda: _make_env_config(tmp_path)
    )
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: approval,
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    result = json.loads(terminal_tool_module.terminal_tool(command="rm -rf /tmp/demo"))
    return result, mock_env


def test_pending_approval_uses_short_error_and_preserves_human_prompt(monkeypatch, tmp_path):
    approval = {
        "approved": False,
        "status": "pending_approval",
        "command": "rm -rf /tmp/demo",
        "description": "dangerous deletion",
        "pattern_key": "dangerous:rm_rf",
        "message": "WARNING: Asking the user for approval.\n\nrm -rf /tmp/demo",
    }

    result, mock_env = _run_terminal(monkeypatch, tmp_path, approval)

    assert result["status"] == "pending_approval"
    assert result["approval_pending"] is True
    assert result["error"] == "Command blocked pending user approval."
    assert "Asking the user for approval" not in result["error"]
    assert result["approval_message"] == approval["message"]
    assert result["description"] == "dangerous deletion"
    assert result["pattern_key"] == "dangerous:rm_rf"
    mock_env.execute.assert_not_called()


def test_blocked_approval_keeps_structure_without_leaking_warning_text(monkeypatch, tmp_path):
    approval = {
        "approved": False,
        "status": "blocked",
        "description": "dangerous deletion",
        "pattern_key": "dangerous:rm_rf",
        "message": "BLOCKED: Command denied by user. Do NOT retry this command.",
    }

    result, mock_env = _run_terminal(monkeypatch, tmp_path, approval)

    assert result["status"] == "blocked"
    assert result["error"] == (
        "Command denied: dangerous deletion. "
        "Use the approval prompt to allow it, or rephrase the command."
    )
    assert result["approval_message"] == approval["message"]
    assert result["description"] == "dangerous deletion"
    assert result["pattern_key"] == "dangerous:rm_rf"
    assert "Do NOT retry this command" not in result["error"]
    mock_env.execute.assert_not_called()
