"""terminal.allow_background: false disables every path that starts a tracked background process."""
import json
from unittest.mock import patch

from tests.tools.test_terminal_foreground_timeout_cap import _make_env_config


def _run(monkeypatch, tmp_path, **kwargs):
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("TERMINAL_ALLOW_BACKGROUND", "false")
    with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config(cwd=str(tmp_path))), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        return json.loads(terminal_tool(**kwargs))


def test_background_true_is_refused(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, command="echo x", background=True)
    assert "allow_background" in result["error"]


def test_over_cap_foreground_timeout_is_refused_not_promoted(monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    result = _run(monkeypatch, tmp_path, command=f"echo x >> {marker}", timeout=9999)
    assert "allow_background" in result["error"]
    assert not marker.exists()


def test_foreground_still_runs_without_yield_handler(monkeypatch, tmp_path):
    from tools import terminal_tool as tt

    monkeypatch.setenv("TERMINAL_ALLOW_BACKGROUND", "false")
    assert tt._yield_kwargs("echo x", env_type="local", cwd=None, effective_task_id="t",
                            task_id=None, session_key="") == {}
    result = _run(monkeypatch, tmp_path, command="echo live", timeout=30)
    assert result["exit_code"] == 0 and "live" in result["output"]


def test_config_key_bridges_to_env():
    from hermes_cli.config import apply_terminal_config_to_env

    env = apply_terminal_config_to_env(env={}, config={"terminal": {"allow_background": False}}, override=True)
    assert env["TERMINAL_ALLOW_BACKGROUND"] == "False"
