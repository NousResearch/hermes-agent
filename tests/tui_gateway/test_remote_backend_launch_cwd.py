"""A TUI/dashboard session on a sandbox backend never sends the gateway's launch directory to the sandbox.

The hosted image runs the gateway with cwd ``/opt/data`` (the Hermes home). A session created with no workspace kept
that host directory as its cwd, and ``_terminal_task_cwd_with_source`` registered it as the terminal cwd of a modal /
daytona / sprites / plugin sandbox, where it does not exist: every command failed ``cd: /opt/data: No such file or
directory`` (exit 126). The backend's own default applies instead, as on the messaging gateway.
"""

from __future__ import annotations

import pytest

from tools import terminal_tool
from tui_gateway import server


@pytest.fixture
def launch_dir(tmp_path, monkeypatch):
    """The gateway's launch directory: an existing host dir nobody picked, with no configured terminal.cwd."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    for name in ("TERMINAL_ENV", "TERMINAL_CWD", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    launch = tmp_path / "opt-data"
    launch.mkdir()
    monkeypatch.chdir(launch)
    return launch


def _backend(monkeypatch, name: str, **env: str) -> None:
    monkeypatch.setenv("TERMINAL_ENV", name)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _fresh_session(key: str) -> dict:
    """What ``session.create`` stores for a TUI session that sent no cwd (no ``HERMES_TUI_CWD``)."""
    explicit, cwd, _remote = server._resolve_create_cwd({}, "tui", None)
    return {"session_key": key, "cwd": cwd, "explicit_cwd": explicit, "profile_home": None, "source": "tui"}


@pytest.mark.parametrize("backend", ["modal", "daytona", "singularity"])
def test_sandbox_session_does_not_register_the_launch_dir(launch_dir, monkeypatch, backend):
    _backend(monkeypatch, backend)
    session = _fresh_session(f"launch-{backend}")
    assert session["cwd"] == str(launch_dir) and not session["explicit_cwd"]

    assert server._terminal_task_cwd_with_source(session) == ("", "process")

    server._register_session_cwd(session)
    try:
        assert terminal_tool.get_session_cwd(session["session_key"]) is None
        # The per-command cwd is the backend default, never the host launch dir.
        assert terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="/root", session_key=session["session_key"], env_type=backend) == "/root"
    finally:
        terminal_tool.clear_task_env_overrides(session["session_key"])


def test_plugin_sandbox_backend_does_not_register_the_launch_dir(launch_dir, monkeypatch):
    """A plugin backend (sprites) declaring ``is_remote``/``is_container``."""
    import agent.terminal_env_registry as registry

    _backend(monkeypatch, "sprites")
    monkeypatch.setattr(registry, "provider_flag", lambda name, attr, default=False: name == "sprites")
    session = _fresh_session("launch-sprites")

    assert server._terminal_task_cwd_with_source(session) == ("", "process")
    assert server._persisted_session_cwd(session) is None


def test_plugin_host_backend_keeps_the_launch_dir(launch_dir, monkeypatch):
    """A plugin backend declaring neither flag runs on this host: the launch dir stays its cwd."""
    import agent.terminal_env_registry as registry

    _backend(monkeypatch, "hostrunner")
    monkeypatch.setattr(registry, "provider_flag", lambda name, attr, default=False: False)
    session = _fresh_session("launch-hostrunner")

    assert server._terminal_task_cwd_with_source(session) == (str(launch_dir), "session")


def test_explicit_workspace_still_wins_on_a_sandbox_backend(launch_dir, monkeypatch, tmp_path):
    _backend(monkeypatch, "modal")
    project = tmp_path / "project"
    project.mkdir()
    explicit, cwd, _remote = server._resolve_create_cwd({"cwd": str(project)}, "tui", None)
    session = {"session_key": "explicit", "cwd": cwd, "explicit_cwd": explicit, "profile_home": None}

    assert explicit
    assert server._terminal_task_cwd_with_source(session) == (str(project), "session")


def test_configured_sandbox_cwd_still_wins(launch_dir, monkeypatch):
    _backend(monkeypatch, "modal", TERMINAL_CWD="/workspace")
    session = _fresh_session("configured")

    assert server._terminal_task_cwd_with_source(session) == ("/workspace", "process")


def test_local_backend_keeps_the_launch_dir(launch_dir, monkeypatch):
    session = _fresh_session("local")

    assert server._terminal_task_cwd_with_source(session) == (str(launch_dir), "session")
    assert server._persisted_session_cwd(session) == str(launch_dir)


@pytest.mark.parametrize("mount", ["true", "false"])
def test_docker_keeps_the_host_dir(launch_dir, monkeypatch, mount):
    """docker is unchanged: with ``docker_mount_cwd_to_workspace`` the host dir is the bind-mount source, tagged
    ``session`` so per-session isolation mounts it (terminal_tool remaps it to /workspace); without it the session
    keeps its workspace container key."""
    _backend(monkeypatch, "docker", TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=mount)
    session = _fresh_session(f"docker-mount-{mount}")

    assert server._terminal_task_cwd_with_source(session) == (str(launch_dir), "session")
    assert server._persisted_session_cwd(session) == str(launch_dir)


def test_sandbox_resume_drops_a_stored_hermes_home_cwd(launch_dir, monkeypatch):
    """A row persisted before the fix with the Hermes home as cwd is not re-adopted as the sandbox workspace."""
    hermes_home = str(server._hermes_home)
    _backend(monkeypatch, "modal")
    assert server._resumable_stored_cwd(hermes_home, None) == ""

    monkeypatch.setenv("TERMINAL_ENV", "local")
    assert server._resumable_stored_cwd(hermes_home, None) == hermes_home


def test_sandbox_path_completion_lists_the_backend_cwd(launch_dir, monkeypatch):
    """``complete.path`` with no terminal cwd lists relative to the backend's own cwd, never the host dir."""
    _backend(monkeypatch, "modal")
    seen = {}

    def fake_entries(search_dir, session_key):
        seen["dir"] = search_dir
        return [("notes.md", False)]

    monkeypatch.setattr(server, "_backend_dir_entries", fake_entries)
    resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": "no"}})

    assert seen["dir"] == "."
    assert [i["text"] for i in resp["result"]["items"]] == ["notes.md"]
