"""A minted CLI turn's workspace mounts under per-session docker isolation (#140665).

A one-shot ``hermes chat`` (``-q``/``--query-file``/``--oneshot``, scripts) runs in a
single-session process: its launch cwd — ``--in DIR`` included — IS that run's
workspace. The conversation loop mints a fresh task_id for such turns (the gateway
always passes its session ids through), and per-session isolation resolves the
``/workspace`` bind source from the session-scoped cwd override table alone — which
nobody populated for a CLI run, so the container started with no workspace bind at
all while ``docker_mount_cwd_to_workspace`` was on.

These tests pin ``register_cli_turn_workspace`` (the CLI-side counterpart of the
gateway's ``_register_session_cwd``) and its consumption by
``_resolve_task_host_cwd``:

* isolation + docker ⇒ the minted id gets a ``cwd_source: "session"`` override
  derived from ``TERMINAL_CWD``/the launch cwd, and that directory mounts;
* shared/persistent containers and other backends register nothing (the legacy
  ``host_cwd`` mount path stays untouched);
* a container-path or dead ``TERMINAL_CWD`` registers but still mounts nothing
  (the host-side checks in ``_resolve_task_host_cwd`` stay authoritative);
* ``_bind_turn_identity`` registers only the id it minted — a caller-supplied
  task_id (gateway sessions) is left to whoever owns its workspace provenance.
"""

import os

import pytest

from agent.turn_context import _bind_turn_identity
from tools import terminal_tool


class _StubAgent:
    """Attribute sink for ``_bind_turn_identity`` (it only setattrs/getattrs)."""

    session_id = "sess-1"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate override/alias/cwd-record state and pin the config bridge off."""
    before_overrides = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    with terminal_tool._session_cwd_lock:
        before_cwd = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    # The config→env bridge is one-shot; mark it done so tests control env vars.
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(before_cwd)


def _enable_isolation(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")


def _mount_config(host_cwd="/Users/prev/dev/oldrepo"):
    return {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": host_cwd,
    }


class TestRegisterCliTurnWorkspace:
    def test_isolation_binds_launch_cwd_as_session_workspace(self, monkeypatch):
        _enable_isolation(monkeypatch)
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        overrides = terminal_tool.resolve_task_overrides("cli-turn-1")
        assert overrides["cwd_source"] == "session"
        assert overrides["cwd"] == os.path.abspath(os.getcwd())

    def test_terminal_cwd_wins_over_launch_cwd(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert terminal_tool.resolve_task_overrides("cli-turn-1")["cwd"] == str(
            tmp_path
        )

    def test_persistent_containers_register_nothing(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert terminal_tool.resolve_task_overrides("cli-turn-1") == {}

    def test_local_backend_registers_nothing(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert terminal_tool.resolve_task_overrides("cli-turn-1") == {}


class TestMountResolutionConsumesCliWorkspace:
    def test_cli_workspace_mounts_for_its_container(self, monkeypatch, tmp_path):
        """The reported bug: the one-shot session's container must bind its own
        workspace, not start with an empty /workspace."""
        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert terminal_tool._resolve_task_host_cwd(
            _mount_config(), "cli-turn-1"
        ) == str(tmp_path)

    def test_container_path_terminal_cwd_mounts_nothing(self, monkeypatch):
        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", "/workspace/already-inside")
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert (
            terminal_tool._resolve_task_host_cwd(_mount_config(), "cli-turn-1") is None
        )

    def test_dead_terminal_cwd_mounts_nothing(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "gone"))
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        assert (
            terminal_tool._resolve_task_host_cwd(_mount_config(), "cli-turn-1") is None
        )

    def test_mount_flag_off_means_no_mount(self, monkeypatch, tmp_path):
        _enable_isolation(monkeypatch)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        terminal_tool.register_cli_turn_workspace("cli-turn-1")
        cfg = _mount_config()
        cfg["docker_mount_cwd_to_workspace"] = False
        assert terminal_tool._resolve_task_host_cwd(cfg, "cli-turn-1") is None


class TestBindTurnIdentityIntegration:
    def test_minted_id_gets_the_cli_workspace(self, monkeypatch):
        _enable_isolation(monkeypatch)
        effective_task_id, _turn_id = _bind_turn_identity(
            _StubAgent(), None, None, None, None, None
        )
        overrides = terminal_tool.resolve_task_overrides(effective_task_id)
        assert overrides["cwd_source"] == "session"
        assert overrides["cwd"] == os.path.abspath(os.getcwd())

    def test_caller_supplied_id_is_left_alone(self, monkeypatch):
        """A gateway/subagent turn hands its own task_id through; its workspace
        provenance belongs to the caller, never to the CLI binding."""
        _enable_isolation(monkeypatch)
        _bind_turn_identity(_StubAgent(), "gateway:sess-1", None, None, None, None)
        assert terminal_tool.resolve_task_overrides("gateway:sess-1") == {}

    def test_shared_mode_mints_id_without_override(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
        effective_task_id, _turn_id = _bind_turn_identity(
            _StubAgent(), None, None, None, None, None
        )
        assert terminal_tool.resolve_task_overrides(effective_task_id) == {}
