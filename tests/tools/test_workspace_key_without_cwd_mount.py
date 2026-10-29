"""Persistent Docker must not split a profile container when the cwd is not mounted.

``_workspace_container_key`` keys one container per (profile, workspace) so a
mounted cwd cannot leak across folders (#140665). The mount itself is gated on
``docker_mount_cwd_to_workspace``. The split was not. A Desktop project chat
on a profile that leaves the flag false then gets an empty ``:ws:`` home.
"""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def _persistent_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)


def _register(task_id, cwd):
    terminal_tool.register_task_env_overrides(
        task_id, {"cwd": cwd, "cwd_source": "session"}
    )


def test_unmounted_session_cwd_keeps_the_profile_container(monkeypatch, tmp_path):
    """Negative control. Old code returns ``profile:<name>:ws:<digest>``."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false")
    ws_a = tmp_path / "proj-a"
    ws_b = tmp_path / "proj-b"
    ws_a.mkdir()
    ws_b.mkdir()
    keys = []
    for sid, ws in (("desktop-a", ws_a), ("desktop-b", ws_b)):
        _register(sid, str(ws))
        tokens = set_session_vars(session_key=sid, profile="tenant-a")
        try:
            keys.append(terminal_tool._resolve_container_task_id(sid))
        finally:
            clear_session_vars(tokens)
            terminal_tool.clear_task_env_overrides(sid)
    assert keys == ["profile:tenant-a", "profile:tenant-a"]


def test_unmounted_default_profile_stays_on_default(monkeypatch, tmp_path):
    """A default-profile Desktop folder must not mint ``default:ws:<digest>``."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false")
    ws = tmp_path / "proj"
    ws.mkdir()
    _register("desktop-default", str(ws))
    tokens = set_session_vars(session_key="desktop-default", profile="")
    try:
        key = terminal_tool._resolve_container_task_id("desktop-default")
    finally:
        clear_session_vars(tokens)
        terminal_tool.clear_task_env_overrides("desktop-default")
    assert key == "default"


def test_unmounted_already_split_key_collapses_to_the_profile(monkeypatch):
    """``ensure_task_env`` re-resolves a ``:ws:`` key. Mount off must drop it."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false")
    tokens = set_session_vars(session_key="desktop-a", profile="tenant-a")
    try:
        key = terminal_tool._resolve_container_task_id(
            "profile:tenant-a:ws:abc123def4567890"
        )
    finally:
        clear_session_vars(tokens)
    assert key == "profile:tenant-a"


def test_mounted_session_cwd_still_splits_per_workspace(monkeypatch, tmp_path):
    """The #140665 split stays when the profile mounts the cwd."""
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    keys = []
    for sid, name in (("desktop-a", "proj-a"), ("desktop-b", "proj-b")):
        ws = tmp_path / name
        ws.mkdir()
        _register(sid, str(ws))
        tokens = set_session_vars(session_key=sid, profile="tenant-a")
        try:
            keys.append(terminal_tool._resolve_container_task_id(sid))
        finally:
            clear_session_vars(tokens)
            terminal_tool.clear_task_env_overrides(sid)
    assert keys[0] != keys[1]
    assert keys[0].startswith("profile:tenant-a:ws:")
    assert keys[1].startswith("profile:tenant-a:ws:")
