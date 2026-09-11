"""Per-turn terminal scope isolation under profile multiplexing (#68559 class).

One multiplexed process serves several profiles, but terminal.* used to
resolve through the process-global ``TERMINAL_*`` env vars bridged once at
startup — so every routed profile inherited the launch profile's backend,
cwd, docker mounts and shared-container key (#68559, #94200, #101132,
#95470). ``tools.terminal_scope`` installs the routed profile's COMPLETE
terminal policy as a ContextVar at each profile boundary; readers resolve
ONLY from it (omitted key → defined default, never ``os.environ``) and an
unresolvable policy fails closed.
"""

import json
import os

import pytest

from tools.terminal_scope import (
    TerminalPolicyRefusal,
    TerminalPolicyUnavailable,
    get_terminal_scope,
    install_profile_terminal_scope,
    reset_terminal_scope,
    set_terminal_scope,
    terminal_env,
)
from tools import browser_tool_cloud as bt_cloud

_LAUNCH_CWD = "/home/launch-user/private"
_LAUNCH_VOLUMES = '["/host/secret:/data:rw"]'


@pytest.fixture(autouse=True)
def _polluted_launch_env(monkeypatch, tmp_path):
    """Launch profile A bridged a docker backend with sensitive policy into
    the process env; every test proves a routed profile observes none of it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CWD", _LAUNCH_CWD)
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", _LAUNCH_VOLUMES)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "alpha-shared")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "10.10.0.103")
    monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", lambda _h: {})
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda _h: None)
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    yield


def _profile(tmp_path, name, config_yaml="", dotenv=""):
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    if config_yaml:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def test_no_scope_keeps_process_env_behavior():
    """Single-process CLI/TUI (no scope bound) is byte-identical to before."""
    assert terminal_env("TERMINAL_ENV") == "docker"
    assert terminal_env("TERMINAL_SSH_HOST") == "10.10.0.103"


def test_scoped_read_never_falls_through_to_process_env():
    """Omitted key under a scope → defined default, NOT the ambient value."""
    token = set_terminal_scope({"TERMINAL_ENV": "local"})
    try:
        assert terminal_env("TERMINAL_ENV") == "local"
        assert terminal_env("TERMINAL_SSH_HOST") == ""
        assert terminal_env("TERMINAL_DOCKER_VOLUMES", "[]") == "[]"
        assert os.environ["TERMINAL_ENV"] == "docker"  # never mutated
    finally:
        reset_terminal_scope(token)


@pytest.mark.parametrize(
    "config_yaml,dotenv",
    [
        pytest.param("terminal:\n  backend: local\n  cwd: {cwd}\n", "", id="config-yaml"),
        pytest.param("", "TERMINAL_ENV=local\nTERMINAL_CWD={cwd}\n", id="dotenv-only"),
    ],
)
def test_routed_turn_reads_every_terminal_consumer_from_profile(
    tmp_path, config_yaml, dotenv
):
    """Leak matrix through the REAL gateway boundary: a routed local profile
    with its own cwd must be seen as such by every terminal.* consumer —
    terminal_tool config, container key resolution, docker media translation,
    file_tools/runtime_cwd cwd anchors, and the browser/env_probe backend
    checks — with none of launch profile A's docker policy showing through."""
    import gateway.run as gw
    import tools.terminal_tool as tt
    from agent import runtime_cwd
    from gateway.platforms import base as gbase
    from tools import browser_tool, env_probe, file_tools_paths

    b_cwd = tmp_path / "b-work"
    b_cwd.mkdir()
    home = _profile(
        tmp_path, "bee",
        config_yaml.format(cwd=b_cwd), dotenv.format(cwd=b_cwd),
    )

    with gw._profile_runtime_scope(home):
        cfg = tt._get_env_config()
        assert cfg["env_type"] == "local"
        assert cfg["cwd"] == str(b_cwd)
        assert cfg["docker_volumes"] == []
        assert cfg["docker_shared_container_key"] == ""
        assert tt._resolve_container_task_id(None) == "default"
        assert gbase._parse_docker_volume_mounts() == []
        assert not any(
            "alpha-shared" in c for c in gbase._docker_sandbox_dir_candidates("agent:bee:x")
        )
        assert file_tools_paths._configured_terminal_cwd() == str(b_cwd)
        assert runtime_cwd.resolve_agent_cwd() == b_cwd
        assert bt_cloud._is_local_backend() is True
        # env_probe bails out with "" for remote backends; a local profile
        # must not be treated as remote just because the launch env is docker.
        assert env_probe._resolve_terminal_backend() == "local"
    assert get_terminal_scope() is None
    # Process env untouched — the launch profile's own turns are unchanged.
    assert os.environ["TERMINAL_DOCKER_VOLUMES"] == _LAUNCH_VOLUMES


def test_profile_omitting_keys_gets_defaults_not_launch_values(tmp_path):
    """#101132/#95470: a docker profile that does NOT set docker_volumes or
    docker_shared_container_key must not inherit the launch profile's."""
    import gateway.run as gw
    import tools.terminal_tool as tt

    home = _profile(tmp_path, "bee", "terminal:\n  backend: docker\n")
    with gw._profile_runtime_scope(home):
        cfg = tt._get_env_config()
        assert cfg["env_type"] == "docker"
        assert cfg["docker_volumes"] == []
        assert cfg["docker_shared_container_key"] == ""
        assert cfg["ssh_host"] == ""
        assert cfg["cwd"] != _LAUNCH_CWD
    assert json.loads(os.environ["TERMINAL_DOCKER_VOLUMES"])  # A unchanged


def test_malformed_profile_config_refuses_execution(tmp_path):
    """Unresolvable policy → refusal scope; terminal_tool refuses instead of
    running under the launch process's ambient policy (fail closed)."""
    from tools.terminal_tool import terminal_tool

    home = _profile(tmp_path, "broken", "terminal: [unclosed\n")
    token = install_profile_terminal_scope(home)
    try:
        assert isinstance(get_terminal_scope(), TerminalPolicyRefusal)
        with pytest.raises(TerminalPolicyUnavailable):
            terminal_env("TERMINAL_ENV")
        result = terminal_tool(command="whoami")
        assert "terminal policy unavailable" in result
    finally:
        reset_terminal_scope(token)


def test_gateway_runtime_scope_resets_on_error(tmp_path):
    import gateway.run as gw

    home = _profile(tmp_path, "qa", "terminal:\n  backend: local\n")
    with pytest.raises(RuntimeError):
        with gw._profile_runtime_scope(home):
            assert terminal_env("TERMINAL_ENV") == "local"
            raise RuntimeError("turn blew up")
    assert get_terminal_scope() is None


def test_tui_and_cron_boundaries_bind_and_reset(tmp_path):
    import tui_gateway.server as server
    from tools.terminal_scope import install_and_reset_profile_terminal_scope

    home = _profile(tmp_path, "dash", "terminal:\n  backend: local\n")
    with server._session_profile_runtime_scope({"profile_home": str(home)}):
        assert terminal_env("TERMINAL_ENV") == "local"
        assert terminal_env("TERMINAL_SSH_HOST") == ""
    assert get_terminal_scope() is None
    with install_and_reset_profile_terminal_scope(home):  # cron fire helper
        assert terminal_env("TERMINAL_ENV") == "local"
    assert get_terminal_scope() is None


def test_config_list_and_dict_values_are_json_not_repr(tmp_path):
    """config.yaml list/dict terminal keys must be JSON so terminal_tool's
    json.loads path succeeds. str() produces Python repr and drops the tool."""
    from tools.terminal_scope import build_profile_terminal_scope

    home = _profile(
        tmp_path,
        "docker-lists",
        "\n".join(
            [
                "terminal:",
                "  backend: docker",
                "  docker_forward_env:",
                "    - EMAIL_HOME_ADDRESS",
                "  docker_volumes:",
                "    - /tmp/a:/data",
                "  docker_env:",
                "    FOO: bar",
                "  docker_extra_args:",
                "    - --network=host",
                "",
            ]
        ),
    )
    scope = build_profile_terminal_scope(home)
    assert json.loads(scope["TERMINAL_DOCKER_FORWARD_ENV"]) == ["EMAIL_HOME_ADDRESS"]
    assert json.loads(scope["TERMINAL_DOCKER_VOLUMES"]) == ["/tmp/a:/data"]
    assert json.loads(scope["TERMINAL_DOCKER_ENV"]) == {"FOO": "bar"}
    assert json.loads(scope["TERMINAL_DOCKER_EXTRA_ARGS"]) == ["--network=host"]

    import gateway.run as gw
    import tools.terminal_tool as tt

    with gw._profile_runtime_scope(home):
        cfg = tt._get_env_config()
        assert cfg["docker_forward_env"] == ["EMAIL_HOME_ADDRESS"]
        assert cfg["docker_volumes"] == ["/tmp/a:/data"]
        assert cfg["docker_env"] == {"FOO": "bar"}
        assert cfg["docker_extra_args"] == ["--network=host"]


def test_dotenv_json_strings_stay_json_strings(tmp_path):
    """The .env path already stores JSON text; str() must keep that payload."""
    from tools.terminal_scope import build_profile_terminal_scope

    home = _profile(
        tmp_path,
        "dotenv-json",
        "terminal:\n  backend: docker\n",
        'TERMINAL_DOCKER_FORWARD_ENV=["EMAIL_HOME_ADDRESS"]\n'
        'TERMINAL_DOCKER_VOLUMES=["/tmp/a:/data"]\n',
    )
    scope = build_profile_terminal_scope(home)
    assert json.loads(scope["TERMINAL_DOCKER_FORWARD_ENV"]) == ["EMAIL_HOME_ADDRESS"]
    assert json.loads(scope["TERMINAL_DOCKER_VOLUMES"]) == ["/tmp/a:/data"]


def test_launch_terminal_scope_preserves_ambient_env(tmp_path, monkeypatch):
    """#107422 / Finding 3: launch scope preserves ambient process environment
    (e.g. TERMINAL_ENV=ssh, TERMINAL_SSH_HOST) when launch config.yaml has no terminal
    section, while secondary profile scopes never inherit ambient env."""
    from tools.terminal_scope import (
        build_launch_terminal_scope,
        build_profile_terminal_scope,
    )

    launch_home = tmp_path / "launch-profile"
    secondary_home = tmp_path / "profiles" / "sec"
    launch_home.mkdir(parents=True, exist_ok=True)
    secondary_home.mkdir(parents=True, exist_ok=True)
    (launch_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (secondary_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    launch_scope = build_launch_terminal_scope(launch_home)
    assert launch_scope["TERMINAL_ENV"] == "ssh"
    assert launch_scope["TERMINAL_SSH_HOST"] == "example.test"

    # Secondary profile must NOT inherit ambient TERMINAL_ENV=ssh
    sec_scope = build_profile_terminal_scope(secondary_home)
    assert sec_scope["TERMINAL_ENV"] == "local"
    assert sec_scope.get("TERMINAL_SSH_HOST", "") == ""

    # Explicit launch config backend overrides ambient env
    (launch_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n", encoding="utf-8"
    )
    launch_scope_docker = build_launch_terminal_scope(launch_home)
    assert launch_scope_docker["TERMINAL_ENV"] == "docker"
    assert launch_scope_docker["TERMINAL_SSH_HOST"] == "example.test"


def test_prepare_turn_input_binds_launch_scope_once_multiplexing_is_active(
    tmp_path, monkeypatch
):
    """#107422 / Finding 4: real prompt_turn._prepare_turn_input path binds the
    launch profile's terminal scope once _served_profile_homes is non-empty."""
    from tools.terminal_scope import get_terminal_scope, reset_terminal_scope, terminal_env
    import tui_gateway.server as srv
    from tui_gateway.prompt_turn import _TurnRun

    launch_home = tmp_path / "launch-turn-home"
    launch_home.mkdir(parents=True, exist_ok=True)
    (launch_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(srv, "_hermes_home", str(launch_home), raising=False)

    # Single profile mode: _served_profile_homes is empty
    monkeypatch.setattr(srv, "_served_profile_homes", set(), raising=False)

    import threading
    session = {
        "session_key": "test_launch_sess",
        "profile_home": None,
        "history": [],
        "history_lock": threading.Lock(),
        "cols": 80,
        "agent": None,
    }
    st_single = _TurnRun(
        agent=None, one_turn_restore=True, terminal_callback=None, receipt_committed=False
    )
    # Monkeypatch helper dependencies that _prepare_turn_input runs
    monkeypatch.setattr(srv, "_wire_callbacks", lambda sid: None, raising=False)
    monkeypatch.setattr(srv, "_sync_bot_capabilities", lambda sid, s: None, raising=False)
    monkeypatch.setattr(srv, "_session_cwd", lambda s: str(launch_home), raising=False)
    monkeypatch.setattr(srv, "_register_session_cwd", lambda s: None, raising=False)
    monkeypatch.setattr(srv, "make_stream_renderer", lambda cols: None, raising=False)
    monkeypatch.setattr(srv, "_start_turn_voice", lambda: (None, None), raising=False)
    monkeypatch.setattr(srv, "_turn_notes", lambda sid, s: [], raising=False)

    srv._prepare_turn_input("sid1", session, st_single, "hello", [])
    assert st_single.scopes.terminal is None
    assert get_terminal_scope() is None

    # Now activate multiplexing by registering a secondary served profile
    sec_home = tmp_path / "profiles" / "sec"
    sec_home.mkdir(parents=True, exist_ok=True)
    served = {sec_home}
    monkeypatch.setattr(srv, "_served_profile_homes", served, raising=False)

    st_multi = _TurnRun(
        agent=None, one_turn_restore=True, terminal_callback=None, receipt_committed=False
    )
    srv._prepare_turn_input("sid1", session, st_multi, "hello", [])
    try:
        assert st_multi.scopes.terminal is not None
        assert get_terminal_scope() is not None
        assert terminal_env("TERMINAL_ENV") == "local"
    finally:
        reset_terminal_scope(st_multi.scopes.terminal)
    assert get_terminal_scope() is None


def test_profile_build_scope_binds_terminal_scope(tmp_path):
    """#107422 / Finding 2: _profile_build_scope binds the terminal scope for the
    profile so eager resume and branch builds resolve routed terminal policy."""
    from tui_gateway.methods_session import _profile_build_scope
    from tools.terminal_scope import get_terminal_scope, terminal_env

    sec_home = tmp_path / "profiles" / "docker-resume"
    sec_home.mkdir(parents=True, exist_ok=True)
    (sec_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  docker_image: test/eager:1\n",
        encoding="utf-8",
    )

    assert get_terminal_scope() is None
    with _profile_build_scope(sec_home):
        assert get_terminal_scope() is not None
        assert terminal_env("TERMINAL_ENV") == "docker"
        assert terminal_env("TERMINAL_DOCKER_IMAGE") == "test/eager:1"
    assert get_terminal_scope() is None


def test_spawn_side_agent_binds_terminal_scope_on_worker(tmp_path, monkeypatch):
    """#107422 / Finding 1: _spawn_side_agent binds the secondary profile's terminal
    scope for the full worker lifetime on daemon thread."""
    import threading
    import tui_gateway.server as srv
    from tools.terminal_scope import get_terminal_scope, terminal_env

    sec_home = tmp_path / "profiles" / "docker-worker"
    sec_home.mkdir(parents=True, exist_ok=True)
    (sec_home / "config.yaml").write_text(
        "terminal:\n  backend: docker\n", encoding="utf-8"
    )

    observed = {}
    event_done = threading.Event()

    def body():
        observed["terminal_env"] = terminal_env("TERMINAL_ENV")
        observed["scope_present"] = get_terminal_scope() is not None
        event_done.set()
        return "ok"

    monkeypatch.setattr(srv, "_emit", lambda ev, parent, payload: None, raising=False)
    monkeypatch.setattr(srv, "_session_cwd", lambda s: str(tmp_path), raising=False)

    session = {"profile_home": str(sec_home), "session_key": "s-side"}
    srv._spawn_side_agent("rid", session, "task_1", "parent_1", "event_1", body)

    assert event_done.wait(timeout=5.0), "side agent worker timed out"
    assert observed["scope_present"] is True
    assert observed["terminal_env"] == "docker"


