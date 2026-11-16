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


def test_a_remote_backend_move_never_restarts_the_worker(live_session, monkeypatch):
    """On main the restart decision compared _local_session_process_cwd twice (both None on a
    remote backend, so it never restarted); the raw-cwd compare (review finding 8a) made every
    remote move restart a worker that was never cwd-pinned. Gate the restart on the backend: the
    remote worker runs in the gateway's launch dir whatever its remote cwd says (T00018 residual 3)."""
    monkeypatch.setattr(server, "_effective_terminal_backend", lambda: "ssh")
    session, restarts, tmp_path = live_session
    _set_cwd(tmp_path / "new")

    assert restarts == [], "a remote-backend move restarted a worker that was never cwd-pinned"
    assert session["cwd"] == str(tmp_path / "new")  # the move itself still lands


def test_a_project_move_persists_the_cwd_exactly_once(live_session, monkeypatch, tmp_path):
    """The pre-move process-cwd read must not route through the healing display getter: on a dead
    cwd it persisted the healed fallback + scheduled a git-meta probe BEFORE the real set, so one
    move wrote the DB twice (review finding 8a). The restart decision compares the raw stored
    values; the move persists exactly the final one."""
    import os as _os
    session, _restarts, tmp = live_session
    persists = []
    monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta",
                        lambda s, cwd, **k: persists.append(cwd))
    session["cwd"] = str(tmp / "deleted")  # a removed worktree: the healing getter would persist
    server._move_session_cwd("ui", session, str(tmp / "new"))
    assert persists == [str(tmp / "new")], \
        f"the move persisted more than the final cwd: {persists}"
    assert session["cwd"] == str(tmp / "new")


def test_a_worker_restart_spawns_in_the_session_project(local_backend, monkeypatch, tmp_path):
    """``_restart_slash_worker`` (model switch, compression, a project move) must hand the session
    project to the respawned worker as ``cwd=``, like the first-use spawn does — the restarted
    worker parents the session's stdio MCP children and they read their project from its cwd
    (T00014 test gap: no test asserted the restart's cwd=)."""
    (tmp_path / "proj").mkdir()
    session = {"session_key": "k", "cwd": str(tmp_path / "proj"), "slash_worker": object()}
    spawned_kwargs = []

    class _RecordingWorker:
        def __init__(self, *args, **kwargs):
            spawned_kwargs.append(kwargs)
            self.stderr_tail = []

        def close(self):
            pass

    attached = []
    monkeypatch.setattr(server, "_SlashWorker", _RecordingWorker)
    monkeypatch.setattr(server, "_resolve_model", lambda: "m")
    monkeypatch.setattr(server, "_attach_worker", lambda sid, s, w: attached.append(sid))

    server._restart_slash_worker("ui", session)

    assert attached == ["ui"]
    assert spawned_kwargs[0].get("cwd") == str(tmp_path / "proj"), \
        "the restarted worker was not spawned in the session project"


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
    import hermes_cli.config as config_mod
    try:
        sw.main()
        assert os.environ["TERMINAL_CWD"] == str(project), \
            "the cli import's terminal.cwd bridge clobbered the spawn pin and the child never re-pinned it"
        assert scoped_session_cwd() == str(project), \
            "the session cwd ContextVar was never pinned in the child"
    finally:
        clear_session_cwd()
        # main() pinned for real: the env write and the bridge wrapper must not leak into a
        # test appended after this one.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        left = getattr(config_mod.apply_terminal_config_to_env, "__wrapped__", None)
        if getattr(config_mod.apply_terminal_config_to_env, "_keeps_session_cwd_pin", False) and left is not None:
            config_mod.apply_terminal_config_to_env = left


def test_the_childs_pin_survives_the_real_terminal_config_bridge(monkeypatch):
    """The discriminative twin of the re-pin test above: the clobberer is REAL. Every reload
    funnels through ``apply_terminal_config_to_env`` — the ``import cli`` bridge, MCP
    discovery's ``load_hermes_dotenv`` (the ``_prepare_slash_worker_runtime`` pass), and
    terminal_tool's once-per-process launcher bridge — and an explicit ``terminal.cwd`` in
    config.yaml rewrites ``TERMINAL_CWD`` over the spawn pin. The stubbed-prep main() test
    above passes vacuously against that; here the pin must survive the actual bridge calls
    (review finding 7, empirically reproduced)."""
    from pathlib import Path

    import hermes_cli.config as config_mod
    from agent.runtime_cwd import clear_session_cwd, scoped_session_cwd
    from hermes_constants import get_hermes_home
    import tui_gateway.slash_worker as sw

    project = Path(get_hermes_home()) / "pin-project"
    project.mkdir()
    configured = Path(get_hermes_home()) / "from-config-cwd"
    configured.mkdir()
    config_path = Path(get_hermes_home()) / "config.yaml"
    config_path.write_text(f"terminal:\n  cwd: {configured}\n", encoding="utf-8")
    original_bridge = config_mod.apply_terminal_config_to_env
    try:
        # A sibling test may have left ITS wrapper installed (main() pins for real); unwrap it
        # so this test pins against the real bridge, not a previous test's project.
        inherited = getattr(original_bridge, "__wrapped__", None)
        if getattr(original_bridge, "_keeps_session_cwd_pin", False) and inherited is not None:
            original_bridge = inherited
            config_mod.apply_terminal_config_to_env = original_bridge
        sw._pin_session_cwd(str(project))
        assert os.environ["TERMINAL_CWD"] == str(project)
        # The real damage, first via the discovery reload's funnel ...
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
        assert os.environ["TERMINAL_CWD"] == str(project), \
            "the dotenv reload's terminal bridge clobbered the session pin"
        # ... and via the launcher bridge (terminal_tool fires it later with override=True) —
        # through the module attribute, the seam every consumer late-imports through.
        config_mod.apply_terminal_config_to_env(env=None, override=True)
        assert os.environ["TERMINAL_CWD"] == str(project), \
            "the launcher terminal bridge clobbered the session pin"
        assert scoped_session_cwd() == str(project), \
            "the session cwd ContextVar was never pinned in the child"
    finally:
        config_mod.apply_terminal_config_to_env = original_bridge
        config_path.unlink(missing_ok=True)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        clear_session_cwd()
