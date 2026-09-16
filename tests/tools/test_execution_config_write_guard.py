"""Execution tools must not persist changes to the active security config."""

import json

import hermes_cli.config as hermes_config
from tools.code_execution_tool import execute_code


def test_execute_code_restores_active_config_after_direct_write(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    original = b"approvals:\n  mode: smart\n"
    config_path.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()

    result = json.loads(execute_code(
        code=f"open({str(config_path)!r}, 'w', encoding='utf-8').write('approvals:\\n  mode: off\\n')",
        task_id="config-write-guard",
        reset=True,
    ))

    assert result["status"] == "error"
    assert "modified the active Hermes config.yaml" in result["error"]
    assert config_path.read_bytes() == original


def test_execute_code_removes_new_active_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    hermes_config._LOAD_CONFIG_CACHE.clear()

    result = json.loads(execute_code(
        code=f"open({str(config_path)!r}, 'w', encoding='utf-8').write('hooks: {{}}\\n')",
        task_id="config-create-guard",
        reset=True,
    ))

    assert result["status"] == "error"
    assert not config_path.exists()