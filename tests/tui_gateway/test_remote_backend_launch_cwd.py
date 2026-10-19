"""A TUI/dashboard session on a sandbox backend never sends the gateway's launch directory to the sandbox.

The hosted image runs the gateway with cwd ``/opt/data`` (the Hermes home). A session created with no workspace kept
that host directory as its cwd, and ``_terminal_task_cwd_with_source`` registered it as the terminal cwd of a modal /
daytona / sprites / plugin sandbox, where it does not exist: every command failed ``cd: /opt/data: No such file or
directory`` (exit 126). The backend's own default applies instead, as on the messaging gateway.
"""

from __future__ import annotations

import threading

import pytest

from agent.terminal_env_provider import TerminalEnvironmentProvider
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


class _Sprites(TerminalEnvironmentProvider):
    """A plugin sandbox backend (remote, container), like hermes-plugin-sprites."""

    name = "sprites"
    is_remote = True
    is_container = True

    def is_available(self):
        return True

    def create_environment(self, **kwargs):
        raise NotImplementedError


class _HostRunner(TerminalEnvironmentProvider):
    """A plugin backend that runs commands on this host (neither remote nor a container)."""

    name = "hostrunner"
    is_remote = False
    is_container = False

    def is_available(self):
        return True

    def create_environment(self, **kwargs):
        raise NotImplementedError


@pytest.fixture
def clean_registry():
    from agent import terminal_env_registry as registry

    registry._reset_for_tests()
    yield registry
    registry._reset_for_tests()


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


def test_plugin_sandbox_backend_does_not_register_the_launch_dir(launch_dir, monkeypatch, clean_registry):
    """A plugin backend (sprites) declaring ``is_remote``/``is_container``."""
    _backend(monkeypatch, "sprites")
    clean_registry.register_provider(_Sprites())
    session = _fresh_session("launch-sprites")

    assert server._terminal_task_cwd_with_source(session) == ("", "process")
    assert server._persisted_session_cwd(session) is None


def test_plugin_host_backend_keeps_the_launch_dir(launch_dir, monkeypatch, clean_registry):
    """A plugin backend declaring neither flag runs on this host: the launch dir stays its cwd."""
    _backend(monkeypatch, "hostrunner")
    clean_registry.register_provider(_HostRunner())
    session = _fresh_session("launch-hostrunner")

    assert server._terminal_task_cwd_with_source(session) == (str(launch_dir), "session")


def test_plugin_classification_reads_the_bound_profiles_registry(launch_dir, tmp_path, clean_registry):
    """A provider registered only in a named profile's scope classifies that profile's sessions, even when the
    caller (session.create) has not bound the profile's home yet."""
    profile = tmp_path / "profiles" / "runner"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("terminal:\n  backend: hostrunner\n", encoding="utf-8")
    clean_registry.register_provider(_HostRunner(), scope=str(profile))
    session = {"session_key": "named-runner", "cwd": str(launch_dir), "explicit_cwd": False,
               "profile_home": str(profile), "source": "tui"}

    assert server._host_cwd_is_terminal_cwd(str(profile))
    assert server._terminal_task_cwd_with_source(session) == (str(launch_dir), "session")
    # Unregistered in that scope: unknown plugin, treated as remote.
    clean_registry._reset_for_tests()
    assert not server._host_cwd_is_terminal_cwd(str(profile))


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
    """A row persisted before the fix with a launch dir (the Hermes home, its subprocess home) as cwd is not
    re-adopted as the sandbox workspace."""
    hermes_home = server.get_hermes_home()
    (hermes_home / "home").mkdir(exist_ok=True)
    _backend(monkeypatch, "modal")
    for launch in (str(hermes_home), str(hermes_home / "home")):
        assert server._resumable_stored_cwd(launch, None) == ""

    monkeypatch.setenv("TERMINAL_ENV", "local")
    assert server._resumable_stored_cwd(str(hermes_home), None) == str(hermes_home)


@pytest.mark.parametrize("launch", ["os-home", "root"])
def test_sandbox_resume_drops_a_stored_desktop_launch_dir(launch_dir, monkeypatch, tmp_path, launch):
    """Legacy rows holding a desktop launch dir (the OS home, ``/``) are not adopted as a sandbox workspace, by
    hydration or ``session.resume`` (both go through ``_resumable_stored_cwd``); local still adopts them."""
    from hermes_state import SessionDB

    stored = str(server.Path.home()) if launch == "os-home" else "/"
    _backend(monkeypatch, "modal")
    assert server._resumable_stored_cwd(stored, None) == ""

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("legacy", source="desktop", model="m", cwd=stored)
        monkeypatch.setitem(server._sessions, "s-legacy", {"session_key": "legacy", "source": "desktop",
                                                           "cwd": str(launch_dir), "profile_home": None})
        monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta", lambda *a, **k: None)
        server._hydrate_session_cwd("s-legacy", "legacy", db, None)
        session = server._sessions["s-legacy"]
        assert not session.get("explicit_cwd")
        assert server._terminal_task_cwd_with_source(session) == ("", "process")
    finally:
        db.close()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    assert server._resumable_stored_cwd(stored, None) == stored


def test_sandbox_resume_keeps_a_workspace_picked_under_the_hermes_root(launch_dir, monkeypatch):
    """A folder the user picked under the Hermes root is a workspace, not a launch dir: it persists and resumes."""
    picked = server.get_hermes_home() / "projects" / "chosen"
    picked.mkdir(parents=True)
    _backend(monkeypatch, "modal")
    session = {"session_key": "picked", "source": "desktop", "cwd": str(picked), "explicit_cwd": True,
               "profile_home": None}

    stored = server._persisted_session_cwd(session)
    assert stored == str(picked)
    assert server._resumable_stored_cwd(stored, None) == str(picked)


def test_sandbox_hydrated_workspace_stays_the_terminal_cwd(launch_dir, monkeypatch, tmp_path):
    """A sandbox session hydrated from its stored workspace marks it explicit, so the terminal uses it rather than
    reading it as an unpicked launch dir."""
    from hermes_state import SessionDB

    project = tmp_path / "project"
    project.mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _backend(monkeypatch, "modal")
        db.create_session("hydrated", source="tui", model="m", cwd=str(project))
        monkeypatch.setitem(server._sessions, "s-hydrated",
                            {"session_key": "hydrated", "source": "tui", "cwd": str(launch_dir),
                             "profile_home": None})
        monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta", lambda *a, **k: None)
        server._hydrate_session_cwd("s-hydrated", "hydrated", db, None)
        session = server._sessions["s-hydrated"]

        assert session["cwd"] == str(project) and session["explicit_cwd"] is True
        assert server._terminal_task_cwd_with_source(session) == (str(project), "session")
    finally:
        db.close()


def test_compute_host_carries_workspace_provenance(launch_dir, monkeypatch, tmp_path):
    """An isolated turn's frame carries ``explicit_cwd``, so the compute host's rebuilt session keeps a picked
    sandbox workspace as its terminal cwd."""
    import io
    import types

    from tui_gateway.compute_host import ComputeHost

    project = tmp_path / "project"
    project.mkdir()
    _backend(monkeypatch, "modal")
    parent = {"session_key": "picked", "cwd": str(project), "explicit_cwd": True, "history": [],
              "history_lock": threading.Lock(), "profile_home": None, "source": "desktop"}
    frame = server._compute_host_turn_frame("rid", "child", parent, "hi")
    assert frame["explicit_cwd"] is True

    captured = {}

    def fake_init_session(sid, key, agent, history, **kwargs):
        captured.update(kwargs)
        monkeypatch.setitem(server._sessions, sid, {"agent": agent, "session_key": key, "cwd": kwargs.get("cwd"),
                                                    "explicit_cwd": kwargs.get("explicit_cwd", False),
                                                    "profile_home": None})

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **kw: types.SimpleNamespace(session_id=key))
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda agent, db: False)
    monkeypatch.setattr(server, "_init_session", fake_init_session)
    server._sessions.pop("child", None)
    child = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)._build_server_session(
        server, {**frame, "sid": "child", "history": []}, "child")

    assert captured["explicit_cwd"] is True
    assert server._terminal_task_cwd_with_source(child) == (str(project), "session")


def test_compute_host_minimal_record_keeps_workspace_provenance(launch_dir, monkeypatch, tmp_path):
    """When ``_init_session`` fails, the compute host's minimal record still carries the picked workspace."""
    import io
    import types

    from tui_gateway.compute_host import ComputeHost

    project = tmp_path / "project"
    project.mkdir()
    _backend(monkeypatch, "modal")

    def failing_init_session(*_a, **_k):
        raise RuntimeError("side machinery unavailable")

    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **kw: types.SimpleNamespace(session_id=key))
    monkeypatch.setattr(server, "_transfer_db_to_agent", lambda agent, db: False)
    monkeypatch.setattr(server, "_init_session", failing_init_session)
    server._sessions.pop("child-min", None)
    frame = {"sid": "child-min", "session_key": "picked-min", "cwd": str(project), "explicit_cwd": True,
             "history": [], "source": "desktop"}
    try:
        child = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)._build_server_session(server, frame, "child-min")
        assert child["explicit_cwd"] is True and child["source"] == "desktop"
        assert server._terminal_task_cwd_with_source(child) == (str(project), "session")
    finally:
        server._sessions.pop("child-min", None)


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
