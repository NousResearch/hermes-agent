"""Desktop host cwd must not replace an unmounted Docker profile's execution cwd."""

import json
import os

import pytest


@pytest.mark.parametrize("mount", [False, True])
def test_registration_uses_the_target_profiles_mount_policy(
    tmp_path, monkeypatch, mount
):
    import hermes_constants as hc
    from tools import terminal_tool
    from tools.terminal_scope import (
        install_and_reset_profile_terminal_scope,
        terminal_env,
    )
    from tui_gateway import server

    home = tmp_path / "home"
    profile = home / "profiles" / "beta"
    profile.mkdir(parents=True)
    workspace = tmp_path / "host-project"
    workspace.mkdir()
    (profile / "config.yaml").write_text(
        json.dumps(
            {
                "terminal": {
                    "backend": "docker",
                    "cwd": "/workspace",
                    "docker_mount_cwd_to_workspace": mount,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", str(not mount).lower())
    before = dict(os.environ)
    session = {
        "session_key": "profile-cwd",
        "cwd": str(workspace),
        "explicit_cwd": True,
        "source": "desktop",
        "profile_home": str(profile),
    }
    vars(server)["_register_session_cwd"](session)

    token = hc.set_hermes_home_override(str(profile))
    try:
        overrides = terminal_tool.resolve_task_overrides("profile-cwd")
        assert overrides == (
            {"cwd": str(workspace), "cwd_source": "session"} if mount
            else {"cwd": "", "cwd_source": "process", "host_workspace": str(workspace)}
        )
        if not mount:
            with install_and_reset_profile_terminal_scope(profile):
                assert terminal_env("TERMINAL_CWD") == "/workspace"
    finally:
        terminal_tool.clear_task_env_overrides("profile-cwd")
        hc.reset_hermes_home_override(token)
    assert session["cwd"] == str(workspace)
    assert os.environ == before


@pytest.mark.platforms("posix")
def test_unmounted_registration_preserves_configured_container_cwd(tmp_path):
    from tools import terminal_tool
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
    from tui_gateway import server

    token = set_terminal_scope(
        {
            "TERMINAL_ENV": "docker",
            "TERMINAL_CWD": "/work",
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "false",
        }
    )
    try:
        vars(server)["_register_session_cwd"](
            {
                "session_key": "configured-cwd",
                "cwd": str(tmp_path),
                "explicit_cwd": True,
            }
        )
        config = terminal_tool._get_env_config()
        assert config["cwd"] == "/work"
        assert (
            terminal_tool._resolve_command_cwd(
                workdir=None,
                default_cwd=config["cwd"],
                session_key="configured-cwd",
                env_type="docker",
            )
            == "/work"
        )
    finally:
        terminal_tool.clear_task_env_overrides("configured-cwd")
        reset_terminal_scope(token)


def test_a_bad_unrelated_terminal_setting_does_not_drop_the_registration(tmp_path):
    """Only the backend and the mount flag decide; an unparsable ``TERMINAL_TIMEOUT`` (which the
    full terminal config would raise on, inside the caller's blanket suppress) changes nothing."""
    from tools import terminal_tool
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
    from tui_gateway import server

    token = set_terminal_scope({
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "false",
        "TERMINAL_TIMEOUT": "5m",
    })
    try:
        server._register_session_cwd({"session_key": "bad-timeout", "cwd": str(tmp_path), "explicit_cwd": True})
        assert terminal_tool.resolve_task_overrides("bad-timeout") == {
            "cwd": "", "cwd_source": "process", "host_workspace": str(tmp_path),
        }
    finally:
        terminal_tool.clear_task_env_overrides("bad-timeout")
        reset_terminal_scope(token)


def test_an_unreadable_profile_keeps_the_desktop_cwd_record(tmp_path, monkeypatch):
    """The routed profile's refusal scope raises on every terminal read; the host-side record
    (project tools read it) must still be written."""
    import hermes_constants as hc
    from tools import terminal_tool
    from tui_gateway import server

    home = tmp_path / "home"
    profile = home / "profiles" / "bad"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("terminal: [unclosed", encoding="utf-8")
    workspace = tmp_path / "host-project"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    vars(server)["_register_session_cwd"]({
        "session_key": "bad-profile", "cwd": str(workspace), "explicit_cwd": True,
        "source": "desktop", "profile_home": str(profile),
    })

    token = hc.set_hermes_home_override(str(profile))
    try:
        assert terminal_tool.resolve_task_overrides("bad-profile").get("cwd") == str(workspace)
    finally:
        terminal_tool.clear_task_env_overrides("bad-profile")
        hc.reset_hermes_home_override(token)
