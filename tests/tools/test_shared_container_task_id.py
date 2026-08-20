"""
Regression tests for the shared-container task_id mapping.

The top-level agent and all delegate_task subagents share a single
terminal sandbox keyed by ``"default"``.  ``_resolve_container_task_id``
is the sole gatekeeper for which tool-call task_ids go to the shared
container vs. get their own isolated sandbox.  RL / benchmark
environments opt in to isolation by calling
``register_task_env_overrides(task_id, {...})`` before the agent loop;
every other task_id collapses back to ``"default"``.

If you change the collapse logic, update both the helper and these
tests -- see `hermes-agent-dev` skill, "Why do subagents get their own
containers?" section, and the Container lifecycle paragraph under
Docker Backend in ``website/docs/user-guide/configuration.md``.
"""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_tenki_shaped_task_id_does_not_bypass_non_tenki_collapse():
    assert (
        terminal_tool._resolve_environment_cache_key(
            "tenki:foreign-profile:default",
            "docker",
        )
        == "default"
    )


def test_cwd_only_override_collapses_to_default():
    """CWD-only overrides (ACP adapter workspace tracking) must NOT trigger
    container isolation — they should collapse to the shared 'default'
    container so all surfaces (TUI, gateway, dashboard) share one sandbox.
    Regression for #37361."""
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("acp-session-abc")
            == "default"
        )
    finally:
        terminal_tool.clear_task_env_overrides("acp-session-abc")


def test_env_type_override_keeps_own_id():
    """env_type is an isolation key — must trigger per-task container."""
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "sandbox", "cwd": "/work"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("bench-env")
            == "bench-env"
        )
    finally:
        terminal_tool.clear_task_env_overrides("bench-env")


def test_multiplexed_tenki_profiles_isolate_config_auth_and_tool_caches(
    monkeypatch,
    tmp_path,
):
    """Two profiles in one gateway process must never share a Tenki sandbox,
    workspace, credential, file-ops wrapper, or execute-code environment."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli import config as hermes_config
    from hermes_constants import get_hermes_home
    from tools import code_execution_tool, file_tools
    from tools.tenki_config import resolve_tenki_auth_token

    # This test exercises process-wide cleanup, so give it an isolated
    # lifecycle registry instead of consuming any environment a prior test
    # intentionally left alive.
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_retiring_environments", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    for home, workspace, token in (
        (home_a, "workspace-a", "token-a"),
        (home_b, "workspace-b", "token-b"),
    ):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "terminal:\n"
            "  backend: tenki\n"
            "  cwd: /home/tenki\n"
            f"  tenki_workspace_id: {workspace}\n",
            encoding="utf-8",
        )
        (home / ".env").write_text(
            f"TENKI_AUTH_TOKEN={token}\n",
            encoding="utf-8",
        )

    # These process-wide values belong to neither scoped profile. Config and
    # auth resolution must override/fail closed around them.
    monkeypatch.setenv("HERMES_PROFILE", "primary-process-profile")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TENKI_WORKSPACE_ID", "process-workspace-leak")
    monkeypatch.setenv("TENKI_AUTH_TOKEN", "process-token-leak")
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    created = []

    class FakeTenkiEnv:
        def __init__(self, *, cwd, profile_home, workspace, token, task_id):
            self.cwd = cwd
            self.profile_home = profile_home
            self.workspace = workspace
            self.token = token
            self.task_id = task_id
            self.cleanup_calls = 0

        def execute(self, _command, **_kwargs):
            return {
                "output": f"{self.profile_home}:{self.workspace}:{self.token}",
                "returncode": 0,
            }

        def cleanup(self):
            self.cleanup_calls += 1

    def fake_create_environment(**kwargs):
        env = FakeTenkiEnv(
            cwd=kwargs["cwd"],
            profile_home=str(get_hermes_home()),
            workspace=kwargs["container_config"]["tenki_workspace_id"],
            token=resolve_tenki_auth_token(),
            task_id=kwargs["task_id"],
        )
        created.append(env)
        return env

    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        fake_create_environment,
    )

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    hermes_config._LOAD_CONFIG_CACHE.clear()
    hermes_config._RAW_CONFIG_CACHE.clear()
    try:
        with _profile_runtime_scope(home_a):
            terminal_tool.terminal_tool("printf a", task_id="shared", force=True)
            env_a = terminal_tool.get_active_env("shared")
            file_ops_a = file_tools._get_file_ops("shared")
            code_env_a, code_backend_a = code_execution_tool._get_or_create_env(
                "shared"
            )

        with _profile_runtime_scope(home_b):
            terminal_tool.terminal_tool("printf b", task_id="shared", force=True)
            env_b = terminal_tool.get_active_env("shared")
            file_ops_b = file_tools._get_file_ops("shared")
            code_env_b, code_backend_b = code_execution_tool._get_or_create_env(
                "shared"
            )

        assert len(created) == 2
        assert env_a is created[0]
        assert env_b is created[1]
        assert env_a is not env_b
        assert (env_a.workspace, env_a.token) == ("workspace-a", "token-a")
        assert (env_b.workspace, env_b.token) == ("workspace-b", "token-b")
        assert env_a.profile_home == str(home_a)
        assert env_b.profile_home == str(home_b)
        assert env_a.task_id != env_b.task_id
        assert env_a.task_id.startswith("tenki:")
        assert env_b.task_id.startswith("tenki:")

        assert file_ops_a.env is env_a
        assert file_ops_b.env is env_b
        assert file_ops_a is not file_ops_b
        assert (code_env_a, code_backend_a) == (env_a, "tenki")
        assert (code_env_b, code_backend_b) == (env_b, "tenki")

        # Process shutdown operates on canonical registry keys outside either
        # profile scope. It must still retire both profile-owned wrappers.
        assert terminal_tool.cleanup_all_environments() == 2
        assert env_a.cleanup_calls == 1
        assert env_b.cleanup_calls == 1
        assert not terminal_tool._active_environments
    finally:
        set_multiplex_active(previous_multiplex)
        for env in created:
            for key, value in list(terminal_tool._active_environments.items()):
                if value is env:
                    terminal_tool._active_environments.pop(key, None)
                    terminal_tool._last_activity.pop(key, None)
                    terminal_tool._creation_locks.pop(key, None)
        file_tools.clear_file_ops_cache()
        hermes_config._LOAD_CONFIG_CACHE.clear()
        hermes_config._RAW_CONFIG_CACHE.clear()
