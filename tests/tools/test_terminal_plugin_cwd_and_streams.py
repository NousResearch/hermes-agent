"""Plugin terminal backends declare where commands start (``default_cwd``) and whether Bot Desktop,
computer_use and the browser can follow the terminal into them (``supports_streams`` plus the
environment's ``exec_prefix``), instead of the core assuming ``/root`` and refusing every plugin."""

import subprocess
from types import SimpleNamespace

import pytest

from agent import terminal_env_registry as reg
from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools import file_tools, file_tools_paths, terminal_tool
from tools.bot_desktop import placement
from tools.environments import streams
from tools.terminal_tool_backends import _create_environment


class _SpaceEnv:
    def __init__(self, cwd="/", home="/home/cua"):
        # Like a sandbox that learns its home only once connected.
        self.cwd = home if cwd == "~" else cwd

    def execute(self, command, **kwargs):
        return {"output": "", "exit_code": 0}

    def cleanup(self):
        pass

    def exec_prefix(self, *, user=None, interactive=True):
        return ["spacectl", "exec", "box-1", "--"]


class _PlainProvider(TerminalEnvironmentProvider):
    name = "plainbox"
    display_name = "PlainBox"

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default", image=None, container_config=None, **kwargs):
        return _SpaceEnv(cwd, home=f"/home/{task_id}")  # every sandbox has its own home


class _SpaceProvider(_PlainProvider):
    name = "spacebox"
    supports_streams = True

    @property
    def default_cwd(self):
        return "/home/space"


class _HomeProvider(_PlainProvider):
    name = "homebox"

    @property
    def default_cwd(self):
        return "~"


@pytest.fixture(autouse=True)
def _providers(monkeypatch):
    reg._reset_for_tests()
    for provider in (_PlainProvider(), _SpaceProvider(), _HomeProvider()):
        reg.register_provider(provider)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(file_tools_paths, "_ssh_home_failed_at", {})
    yield
    reg._reset_for_tests()


def _create(backend, task_id, cwd="~"):
    return _create_environment(backend, image="", cwd=cwd, timeout=10, task_id=task_id)


def test_plugin_default_cwd_replaces_root_and_absorbs_host_paths(monkeypatch):
    assert terminal_tool._resolve_config_cwd("spacebox", False) == ("/home/space", None)
    monkeypatch.setenv("TERMINAL_CWD", "/Users/me/project")  # host-shaped: never valid in the sandbox
    assert terminal_tool._resolve_config_cwd("spacebox", False) == ("/home/space", None)
    monkeypatch.delenv("TERMINAL_CWD")
    assert terminal_tool._resolve_config_cwd("plainbox", False) == ("/root", None)


def test_home_default_stays_per_sandbox_and_is_never_expanded_on_the_host():
    assert terminal_tool._resolve_config_cwd("homebox", False) == ("~", None)
    alice, bob = _create("homebox", "alice"), _create("homebox", "bob")
    # A second sandbox still gets "~" and resolves its OWN home, not the first one's.
    assert terminal_tool._resolve_config_cwd("homebox", False) == ("~", None)
    assert (alice._hermes_sandbox_home, bob._hermes_sandbox_home) == ("/home/alice", "/home/bob")


def test_a_cd_inside_the_sandbox_home_survives_while_host_workspaces_are_still_dropped(monkeypatch):
    env = _create("homebox", "alice")
    def resolve():
        return terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd="~", session_key="alice", env_type="homebox", env=env)

    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda _key: "/home/alice/project")
    assert resolve() == "/home/alice/project"
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda _key: "/home/mark/workspace")
    assert resolve() == "~"


def test_file_tools_anchor_relative_paths_at_their_own_sandbox_home(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "homebox")
    monkeypatch.setattr(file_tools, "_get_file_ops",
                        lambda task_id="default": SimpleNamespace(env=_create("homebox", task_id)))
    assert file_tools_paths._container_default_cwd("sess") == "/home/sess"


@pytest.mark.parametrize(("backend", "expected"), [("spacebox", placement.TERMINAL), ("plainbox", placement.REFUSED)])
def test_placement_follows_a_plugin_backend_only_when_it_declares_streams(monkeypatch, backend, expected):
    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)
    monkeypatch.setattr(placement, "_plugin_host_isolated", lambda: False)
    assert placement.resolve().where == expected


def test_placement_refuses_a_streaming_plugin_under_host_isolation(monkeypatch):
    # The plugin host returns the environment as an opaque placeholder, so exec_prefix() is unreachable.
    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "spacebox")
    monkeypatch.setattr(placement, "_plugin_host_isolated", lambda: True)
    where = placement.resolve()
    assert where.where == placement.REFUSED
    assert "plugins.isolation" in where.reason and "placement: gateway" in where.reason


def test_streams_spawn_through_the_plugin_environments_exec_prefix(monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: spawned.append(argv) or object())
    streams.open_stream(_SpaceEnv(), ["cua-driver", "mcp"], child_env={"DISPLAY": ":20"})
    assert spawned[0][:6] == ["spacectl", "exec", "box-1", "--", "bash", "-c"]
    assert "export DISPLAY=:20;" in spawned[0][6] and "exec cua-driver mcp" in spawned[0][6]


def test_a_declared_home_is_kept_when_terminal_cwd_starts_the_sandbox_elsewhere(monkeypatch):
    env = _create("spacebox", "t1", cwd="/workspace")  # terminal.cwd picked another in-sandbox directory
    assert env._hermes_sandbox_home == "/home/space"
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda _key: "/home/space/project")
    assert terminal_tool._resolve_command_cwd(
        workdir=None, default_cwd="/workspace", session_key="t1", env_type="spacebox", env=env) == "/home/space/project"


def test_a_string_exec_prefix_is_refused_instead_of_split_into_characters():
    class _StringPrefixEnv(_SpaceEnv):
        def exec_prefix(self, *, user=None, interactive=True):
            return "spacectl exec box-1 --"

    assert streams.exec_prefix(_StringPrefixEnv()) is None


def test_a_failed_sandbox_bring_up_is_not_retried_on_every_path_resolution(monkeypatch):
    from tools import terminal_tool_lifecycle

    attempts = []

    def _unreachable(task_id="default"):
        attempts.append(task_id)
        raise ConnectionError("sandbox unreachable")

    monkeypatch.setenv("TERMINAL_ENV", "homebox")
    monkeypatch.setattr(terminal_tool_lifecycle, "get_active_env", lambda _task_id: None)
    monkeypatch.setattr(file_tools, "_get_file_ops", _unreachable)
    assert file_tools_paths._container_default_cwd("sess") is None
    assert file_tools_paths._container_default_cwd("sess") is None
    assert attempts == ["sess"]


def test_desktop_scratch_dir_works_for_an_environment_without_get_temp_dir():
    from tools.bot_desktop import sandbox_host

    assert not hasattr(_SpaceEnv(), "get_temp_dir")
    assert sandbox_host._remote_dir(_SpaceEnv(), "p1") == "/tmp/hermes-bot-desktop/p1"


def test_missing_desktop_stack_hint_names_no_config_key_for_a_plugin_backend(monkeypatch):
    from tools.bot_desktop import runtime, sandbox_host

    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "spacebox")
    monkeypatch.setattr(placement, "_plugin_host_isolated", lambda: False)
    monkeypatch.setattr(placement, "terminal_environment", lambda *, create=True: _SpaceEnv())
    monkeypatch.setattr(sandbox_host, "missing_binaries", lambda env: ["Xvnc"])
    monkeypatch.setattr(sandbox_host, "published_env", lambda env, profile: {})
    blocker = runtime.status().blocker
    assert "terminal.spacebox_image" not in blocker
    assert "spacebox plugin" in blocker and "placement: gateway" in blocker
