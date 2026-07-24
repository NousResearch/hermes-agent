"""Regression: execute_code must preserve CWD across environment recreation.

When the terminal environment is cleaned up mid-conversation and
``_get_or_create_env`` rebuilds it, the new environment must use the
last-known CWD — not the config default.  This is the execute_code
sibling of the file-tools fix in #26211.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _force_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")


@patch("tools.terminal_tool._active_environments", new_callable=dict)
@patch("tools.terminal_tool._get_env_config")
@patch("tools.terminal_tool._create_environment")
def test_get_or_create_env_uses_last_known_cwd(
    mock_create_env, mock_config, mock_active
):
    from tools.code_execution_tool import _get_or_create_env
    from tools.file_tools import _last_known_cwd

    mock_env = MagicMock()
    mock_create_env.return_value = mock_env
    mock_config.return_value = {
        "env_type": "local",
        "cwd": "/default/config/path",
        "timeout": 30,
    }

    task_id = "default"
    _last_known_cwd[task_id] = "/Users/user/project"

    try:
        _get_or_create_env(task_id)

        create_call = mock_create_env.call_args
        assert create_call is not None
        args = create_call.args if create_call.args else ()
        kwargs = create_call.kwargs if create_call.kwargs else {}
        cwd_passed = kwargs.get("cwd", args[2] if len(args) >= 3 else None)
        assert cwd_passed == "/Users/user/project", (
            f"Expected last-known CWD '/Users/user/project', got {cwd_passed!r}"
        )
    finally:
        _last_known_cwd.pop(task_id, None)
        mock_active.pop(task_id, None)


@patch("tools.terminal_tool._active_environments", new_callable=dict)
@patch("tools.terminal_tool._get_env_config")
@patch("tools.terminal_tool._create_environment")
def test_get_or_create_env_falls_back_to_config_when_no_last_known(
    mock_create_env, mock_config, mock_active
):
    from tools.code_execution_tool import _get_or_create_env
    from tools.file_tools import _last_known_cwd

    mock_env = MagicMock()
    mock_create_env.return_value = mock_env
    mock_config.return_value = {
        "env_type": "local",
        "cwd": "/default/config/path",
        "timeout": 30,
    }

    task_id = "default"
    _last_known_cwd.pop(task_id, None)

    try:
        _get_or_create_env(task_id)

        create_call = mock_create_env.call_args
        assert create_call is not None
        args = create_call.args if create_call.args else ()
        kwargs = create_call.kwargs if create_call.kwargs else {}
        cwd_passed = kwargs.get("cwd", args[2] if len(args) >= 3 else None)
        assert cwd_passed == "/default/config/path", (
            f"Expected config default '/default/config/path', got {cwd_passed!r}"
        )
    finally:
        mock_active.pop(task_id, None)
