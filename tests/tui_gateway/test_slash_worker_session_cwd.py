"""The slash worker runs in its session's project, and follows it when the project moves.

#151827 (port of #63759): the worker parents the session's stdio MCP servers, which read their
project from its cwd (Hindsight: ``${PWD}``). Spawned in the gateway's launch dir, every session's
worker MCP fleet answered with the launch dir's project.
"""

from __future__ import annotations

import os

import pytest

from tui_gateway import server


@pytest.fixture
def spawned(monkeypatch):
    calls = []

    class _Proc:
        stdout = stderr = iter(())
        stdin = None

        def poll(self):
            return 0

    def _spawn(argv, purpose, **kw):
        calls.append(dict(kw, argv=list(argv)))
        return _Proc()

    monkeypatch.setattr(server, "spawn_owned", _spawn)
    return calls


def test_worker_spawns_in_the_session_project_and_pins_terminal_cwd(spawned, tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", "/gateway/launch")
    server._SlashWorker("sess", "", cwd=str(tmp_path))

    assert spawned[0]["cwd"] == str(tmp_path)
    assert spawned[0]["env"]["TERMINAL_CWD"] == str(tmp_path)


def test_worker_without_a_host_project_keeps_the_gateway_cwd(spawned):
    server._SlashWorker("sess", "", cwd=None)

    assert spawned[0]["cwd"] == os.getcwd()


@pytest.fixture
def local_backend(monkeypatch):
    monkeypatch.setattr(server, "_effective_terminal_backend", lambda: "local")


def test_process_cwd_is_the_session_cwd_on_the_local_backend(local_backend, tmp_path):
    assert server._local_session_process_cwd({"cwd": str(tmp_path)}) == str(tmp_path)


def test_process_cwd_refuses_a_remote_backend_path(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_effective_terminal_backend", lambda: "ssh")

    assert server._local_session_process_cwd({"cwd": str(tmp_path)}) is None


def test_process_cwd_never_hands_popen_a_missing_dir(local_backend, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_heal_dead_cwd", lambda cwd: cwd)  # healing finds nothing either
    assert server._local_session_process_cwd({"cwd": str(tmp_path / "deleted")}) is None


@pytest.fixture
def live_session(local_backend, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta", lambda *_a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)
    restarts = []
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: restarts.append(sid))
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    session = {"session_key": "k", "cwd": str(tmp_path / "old"), "source": "desktop", "slash_worker": object()}
    server._sessions["ui"] = session
    yield session, restarts, tmp_path
    server._sessions.pop("ui", None)


def _set_cwd(cwd) -> dict:
    return server._methods["session.cwd.set"]("r", {"session_id": "ui", "cwd": str(cwd)})


def test_a_project_move_restarts_the_live_worker(live_session):
    _session, restarts, tmp_path = live_session
    response = _set_cwd(tmp_path / "new")

    assert "error" not in response, response
    assert restarts == ["ui"]


def test_setting_the_same_project_again_keeps_the_worker(live_session):
    _session, restarts, tmp_path = live_session
    _set_cwd(str(tmp_path / "old") + os.sep)

    assert restarts == []


def test_a_session_without_a_worker_does_not_spawn_one(live_session):
    session, restarts, tmp_path = live_session
    session["slash_worker"] = None
    _set_cwd(tmp_path / "new")

    assert restarts == []


def test_worker_argv_carries_the_session_cwd(spawned):
    """The child must receive the project as ``--session-cwd`` so it can re-pin after its cli
    import; off the local backend there is no project and the flag stays absent (review finding 7)."""
    server._SlashWorker("sess", "", cwd="/tmp/proj")
    argv = spawned[0]["argv"]
    assert "--session-cwd" in argv, argv
    assert argv[argv.index("--session-cwd") + 1] == "/tmp/proj"

    server._SlashWorker("sess", "", cwd=None)
    assert "--session-cwd" not in spawned[1]["argv"]


def test_slash_worker_child_repins_the_session_cwd_after_the_cli_import(monkeypatch, tmp_path):
    """Gate twin: the child's ``import cli`` runs the dotenv bridge, which re-applies an explicit
    ``terminal.cwd`` from config.yaml and clobbers the spawn's ``TERMINAL_CWD`` pin; the child
    must re-pin the session project afterwards or the MCP children it parents resolve to the
    configured cwd instead of the session's project (review finding 7)."""
    import sys
    import types

    import tui_gateway.slash_worker as sw

    project, other = tmp_path / "proj", tmp_path / "other"
    project.mkdir()
    other.mkdir()

    class _FakeCLI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(HermesCLI=_FakeCLI))
    monkeypatch.setattr(sw, "_prepare_slash_worker_runtime", lambda: None)
    monkeypatch.setattr(sw.sys, "stdin", type("S", (), {"readline": lambda self: ""})())
    # The bridge's damage, as the child finds its env after the import: the config's terminal.cwd won.
    monkeypatch.setenv("TERMINAL_CWD", str(other))
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "s",
                                      "--session-cwd", str(project)])
    from agent.runtime_cwd import clear_session_cwd, scoped_session_cwd
    try:
        sw.main()
        assert os.environ["TERMINAL_CWD"] == str(project), \
            "the cli import's terminal.cwd bridge clobbered the spawn pin and the child never re-pinned it"
        assert scoped_session_cwd() == str(project), \
            "the session cwd ContextVar was never pinned in the child"
    finally:
        clear_session_cwd()
