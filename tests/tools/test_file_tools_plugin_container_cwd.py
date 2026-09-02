"""The file-tools env creation path applies the container cwd guard through the same predicate the
terminal tool uses, so a plugin backend declaring ``is_container`` gets a host-path override
sanitized exactly like docker does (#101013)."""

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool
import tools.terminal_tool_config as ttc


def _capture_create(monkeypatch):
    seen = {}

    def _fake_create(config, env_type, **kwargs):
        seen["cwd"] = kwargs["cwd"]
        return object()

    monkeypatch.setattr(terminal_tool, "_create_configured_env", _fake_create)
    monkeypatch.setattr(terminal_tool, "_select_image", lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, "_resolve_task_host_cwd", lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda _tid: None)
    return seen


def test_plugin_container_backend_gets_the_same_host_cwd_guard_as_docker(monkeypatch, tmp_path):
    host_cwd = str(tmp_path / "Users" / "me" / "workspace")
    monkeypatch.setattr(terminal_tool, "_get_env_config",
                        lambda: {"env_type": "mycloud", "cwd": "/workspace", "timeout": 60})
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _tid: {"cwd": host_cwd})
    monkeypatch.setattr(ttc, "_plugin_env_flag", lambda env_type, attr, default=False: attr == "is_container")
    seen = _capture_create(monkeypatch)

    env_type, _ = ft._create_terminal_env_for_file_ops("sess", "sess")

    assert env_type == "mycloud"
    assert seen["cwd"] == "/workspace"  # host override dropped, not fed to the sandbox


def test_plugin_non_container_backend_keeps_the_host_cwd(monkeypatch, tmp_path):
    host_cwd = str(tmp_path / "proj")
    monkeypatch.setattr(terminal_tool, "_get_env_config",
                        lambda: {"env_type": "myremote", "cwd": "/workspace", "timeout": 60})
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _tid: {"cwd": host_cwd})
    monkeypatch.setattr(ttc, "_plugin_env_flag", lambda env_type, attr, default=False: False)
    seen = _capture_create(monkeypatch)

    ft._create_terminal_env_for_file_ops("sess", "sess")

    assert seen["cwd"] == host_cwd
