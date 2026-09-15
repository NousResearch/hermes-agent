"""Terminal-scope invariants for the TUI gateway's off-turn profile entrypoints under multiplexing.

``tools/terminal_tool.py`` no longer bridges a profile's ``terminal.*`` into ``os.environ`` under a
home override, so every secondary-profile entrypoint must bind the owning terminal scope itself:
side workers (``prompt.background`` / ``prompt.btw`` / ``preview.restart``) and agent builds
(eager resume, ``session.branch``) used to bind only the home (+ secrets) and ran a
``terminal.backend: docker`` secondary on the launch process's ``local`` backend.

Launch-profile turns bind a scope once any secondary is served (#107422); that scope must carry
the launch process's env-only policy (``TERMINAL_ENV=ssh`` from systemd / a launcher, no file to
rebuild it from) via the snapshot frozen at activation, while a later ambient write from a
secondary context still never becomes the launch turn's authority.

Review findings on #108440 (andrexibiza); #107442 (ehz0ah).
"""

import os
import threading
import types
from unittest.mock import patch

import pytest

from tools import terminal_tool as tt
from tools.terminal_scope import get_terminal_scope
from tui_gateway import launch_terminal_policy as ltp
from tui_gateway import server


@pytest.fixture(autouse=True)
def _launch_local_env(monkeypatch):
    """The launch process runs the local backend; secondaries must never see it."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", False)
    monkeypatch.setattr(ltp, "_snapshot", None)
    monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", lambda _h: {})


def _secondary(tmp_path):
    home = tmp_path / "profiles" / "secondary"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  docker_image: secondary:test\n", encoding="utf-8")
    return home


def _observe(got):
    got.update(scope_bound=get_terminal_scope() is not None, backend=tt._get_env_config()["env_type"])


def _run_side_worker(session, body):
    done = threading.Event()
    with patch.object(server, "_emit", lambda *a, **k: done.set()):
        server._spawn_side_agent(1, session, "side-task", "parent", "background.complete", body,
                                 cwd=session["cwd"])
        assert done.wait(15), "side worker did not finish"


def test_secondary_side_worker_runs_its_own_terminal_backend(tmp_path):
    session = {"profile_home": str(_secondary(tmp_path)), "cwd": str(tmp_path)}
    got = {}

    def body():
        _observe(got)
        return "done"

    _run_side_worker(session, body)
    assert got == {"scope_bound": True, "backend": "docker"}
    assert os.environ["TERMINAL_ENV"] == "local"  # never an ambient write
    assert get_terminal_scope() is None

    # A body that blows up still releases the scope (and the worker still reports).
    seen = {}

    def exploding():
        _observe(seen)
        raise RuntimeError("side turn blew up")

    _run_side_worker(session, exploding)
    assert seen["backend"] == "docker"
    assert get_terminal_scope() is None
    assert os.environ["TERMINAL_ENV"] == "local"


def test_secondary_branch_build_runs_its_own_terminal_backend(tmp_path):
    session = {"profile_home": str(_secondary(tmp_path)), "cwd": str(tmp_path)}
    got = {}

    def build(*a, **k):
        _observe(got)
        return types.SimpleNamespace()

    with patch.object(server, "_profile_session_db", return_value=(None, False)), \
            patch.object(server, "_make_agent_in_context", build), \
            patch.object(server, "_init_session"), patch.object(server, "_transfer_db_to_agent"):
        server._build_branch_agent(session, "branch-sid", "branch-key", [], "gui")
        assert got == {"scope_bound": True, "backend": "docker"}
        assert get_terminal_scope() is None

        def failing(*a, **k):
            _observe(got)
            raise RuntimeError("build blew up")

        with patch.object(server, "_make_agent_in_context", failing), pytest.raises(RuntimeError):
            server._build_branch_agent(session, "branch-sid", "branch-key", [], "gui")
    assert get_terminal_scope() is None
    assert os.environ["TERMINAL_ENV"] == "local"


class _StopAfterPolicy(Exception):
    pass


def _launch_turn_policy(launch_home):
    """Run ``_prepare_turn_input`` for a launch-profile turn up to the terminal-scope bind; returns
    the policy the terminal tool would use, with every bound scope released afterwards."""
    st = server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=False)

    def stop(*a):
        raise _StopAfterPolicy()

    try:
        with patch.object(server, "_hermes_home", launch_home), \
                patch.object(server, "_served_profile_homes", {launch_home.parent / "other"}), \
                patch.object(server, "_wire_callbacks", stop):
            with pytest.raises(_StopAfterPolicy):
                server._prepare_turn_input("launch-sid", {"session_key": "launch-key"}, st, "hello", [])
            assert get_terminal_scope() is not None
            return tt._get_env_config()
    finally:
        from tools.approval_context import reset_current_session_key
        from tools.terminal_scope import reset_terminal_scope
        if st.scopes.terminal is not None:
            reset_terminal_scope(st.scopes.terminal)
        if st.scopes.approval is not None:
            reset_current_session_key(st.scopes.approval)
        server._clear_session_context(st.scopes.session_tokens)


def test_launch_turn_keeps_env_only_ssh_policy_once_multiplexing_is_active(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")
    ltp.capture_launch_terminal_env()  # multiplex activation: first secondary served

    cfg = _launch_turn_policy(launch)
    assert (cfg["env_type"], cfg["ssh_host"]) == ("ssh", "example.test")
    assert get_terminal_scope() is None


def test_launch_turn_ignores_ambient_terminal_env_written_after_activation(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")
    ltp.capture_launch_terminal_env()
    # A secondary context later poisons the process env (the pre-#108440 latch shape).
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "bee/img:1")

    cfg = _launch_turn_policy(launch)
    assert cfg["env_type"] == "ssh"
    assert cfg["ssh_host"] == "example.test"
    assert cfg.get("docker_image") != "bee/img:1"
    assert os.environ["TERMINAL_ENV"] == "docker"  # observed, never rewritten


@pytest.mark.parametrize("entrypoint", ["side-worker", "branch", "explicit-build-scope"])
def test_launch_off_turn_entrypoints_keep_frozen_terminal_policy(tmp_path, monkeypatch, entrypoint):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")
    ltp.capture_launch_terminal_env()
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "secondary:test")
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", {tmp_path / "secondary"})
    session = {"profile_home": None, "cwd": str(tmp_path)}
    got = {}

    def observe(*args, **kwargs):
        _observe(got)
        got["host"] = tt._get_env_config()["ssh_host"]
        return types.SimpleNamespace()

    if entrypoint == "side-worker":
        _run_side_worker(session, observe)
    elif entrypoint == "branch":
        with patch.object(server, "_profile_session_db", return_value=(None, False)), \
                patch.object(server, "_make_agent_in_context", observe), \
                patch.object(server, "_init_session"), patch.object(server, "_transfer_db_to_agent"):
            server._build_branch_agent(session, "launch-branch", "launch-key", [], "gui")
    else:
        # Eager builds supply the explicit launch home instead of profile_home=None.
        with server._profile_build_scope(launch):
            observe()

    assert got == {"scope_bound": True, "backend": "ssh", "host": "example.test"}
    assert get_terminal_scope() is None
    assert os.environ["TERMINAL_ENV"] == "docker"
