"""Kanban assignments survive profile loading (regression for #73556)."""
import pytest
import yaml


@pytest.mark.parametrize("backend", ["local", "docker"])
def test_dispatched_workspace_survives_profile_loading(monkeypatch, tmp_path, backend):
    from hermes_cli import kanban_db as kb
    from hermes_cli.config import apply_terminal_config_to_env
    from tests.hermes_cli.test_kanban_worker_terminal_cwd import _capture_spawn_env
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tools.terminal_tool import _get_env_config, _resolve_config_cwd, _resolve_task_host_cwd

    root = tmp_path / "home"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    selected = tmp_path / "selected"
    selected.mkdir()
    cfg = {"terminal": {"backend": backend, "cwd": str(root),
                        "docker_mount_cwd_to_workspace": True}, "toolsets": ["kanban"]}
    (profile / "config.yaml").write_text(yaml.safe_dump(cfg))
    (root / "config.yaml").write_text("toolsets: [kanban]\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    captured = _capture_spawn_env(kb, monkeypatch, str(selected))
    env = captured["env"]
    apply_terminal_config_to_env(env=env, config=cfg)
    assert env["TERMINAL_CWD"] == str(selected.resolve())
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE"):
        monkeypatch.setenv(key, env[key])
    token = install_profile_terminal_scope(profile)
    try:
        config = _get_env_config()
        expected = "/workspace" if backend == "docker" else str(selected.resolve())
        assert config["cwd"] == expected
        if backend == "docker":
            assert _resolve_task_host_cwd(config, "worker") == str(selected.resolve())
        assert _resolve_config_cwd(backend, True)[0] == expected
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    finally:
        reset_terminal_scope(token)
    token = install_profile_terminal_scope(profile)
    try:
        assert _get_env_config()["host_cwd" if backend == "docker" else "cwd"] == str(root.resolve())
    finally:
        reset_terminal_scope(token)


@pytest.mark.parametrize("entrypoint", ["bridge", "scope", "probe"])
def test_invalid_assignment_never_falls_back_to_profile(monkeypatch, tmp_path, entrypoint):
    from hermes_cli.config import apply_terminal_config_to_env
    from tools.terminal_scope import build_profile_terminal_scope, TerminalPolicyUnavailable
    from tools.terminal_tool import _resolve_config_cwd
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "missing"))
    cfg = {"terminal": {"backend": "local", "cwd": str(tmp_path)}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    actions = {
        "bridge": lambda: apply_terminal_config_to_env(config=cfg),
        "scope": lambda: build_profile_terminal_scope(tmp_path),
        "probe": lambda: _resolve_config_cwd("docker", True),
    }
    with pytest.raises((ValueError, TerminalPolicyUnavailable), match="existing absolute directory"):
        actions[entrypoint]()
