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
tests -- see "Why do subagents get their own containers?" and the
Container lifecycle paragraph under Docker Backend in
``website/docs/user-guide/configuration.md``.
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


# --- Cross-profile SSH-leak isolation (commit e00f940a9, re-applied) ---------
#
# When a session key is present (WebUI/gateway), each session must own its own
# slot in _active_environments so switching from profile A (ssh_host=10.0.0.1)
# to profile B (ssh_host=10.0.0.2) cannot reuse A's SSHEnvironment. Without this
# the shared "default" slot silently runs commands on the wrong remote host.


def test_session_key_scopes_to_its_own_slot(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"


def test_distinct_session_keys_get_distinct_slots(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    a = terminal_tool._resolve_container_task_id(None)
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-B")
    b = terminal_tool._resolve_container_task_id(None)
    assert a == "session:sess-A"
    assert b == "session:sess-B"
    assert a != b


def test_subagent_collapses_onto_parent_session(monkeypatch):
    # Subagents inherit the parent's session key, so they share the parent's
    # container (the #16177 intent) rather than a global "default".
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    assert (
        terminal_tool._resolve_container_task_id("subagent-3-cafef00d")
        == "session:sess-A"
    )


def test_rl_override_wins_over_session_key(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-A")
    terminal_tool.register_task_env_overrides("tb2-z", {"docker_image": "z:1"})
    try:
        assert terminal_tool._resolve_container_task_id("tb2-z") == "tb2-z"
    finally:
        terminal_tool.clear_task_env_overrides("tb2-z")


def test_no_session_key_still_defaults(monkeypatch):
    # CLI mode: no session key -> unchanged "default" behaviour.
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    assert terminal_tool._resolve_container_task_id(None) == "default"


# --- Production gateway path: session key bound via ContextVars ---------------
#
# The tests above set HERMES_SESSION_KEY through os.environ, which only
# exercises the os.getenv() *fallback* branch of the scoping logic. Real
# gateway turns never write this process-global env var — they bind the
# identity through gateway.session_context.set_session_vars(), which stores it
# in a ContextVar, and _resolve_container_task_id reads it back via
# get_session_env(). These companion tests cover that production path with
# HERMES_SESSION_KEY absent from os.environ.


def test_session_key_from_contextvar_without_environ(monkeypatch):
    # Prove the fix works on the gateway path: HERMES_SESSION_KEY is NOT in
    # os.environ; the key lives only in the ContextVar bound by the gateway.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    tokens = set_session_vars(session_key="sess-ctx")
    try:
        assert (
            terminal_tool._resolve_container_task_id(None) == "session:sess-ctx"
        )
        # Subagents inherit the same ContextVar and collapse onto the parent.
        assert (
            terminal_tool._resolve_container_task_id("subagent-1-cafe")
            == "session:sess-ctx"
        )
    finally:
        clear_session_vars(tokens)


def test_contextvar_session_key_wins_over_environ(monkeypatch):
    # Two concurrent gateway sessions in one process must not cross-contaminate:
    # the ContextVar is authoritative even when a *different* value lingers in
    # os.environ (e.g. a CLI-set or previously-leaked global). The container
    # slot must follow the ContextVar-bound session, not the process global.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_SESSION_KEY", "sess-ENV")
    tokens = set_session_vars(session_key="sess-CTX")
    try:
        assert (
            terminal_tool._resolve_container_task_id(None) == "session:sess-CTX"
        )
    finally:
        clear_session_vars(tokens)


# --- Persistent Docker is PROFILE-scoped, not session-scoped ------------------
#
# Product contract: TERMINAL_ENV=docker + container_persistent:true means ONE
# long-lived container per Hermes profile, shared by every session of that
# profile (CLI, gateway chats, WebUI). The a270c4ade session-key fallback must
# NOT fragment persistent Docker into per-session containers; it exists for
# backends where cross-session reuse is dangerous (SSH).


def _persistent_docker(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")


def test_persistent_docker_default_profile_shares_default_container(monkeypatch):
    # A gateway session of the default profile lands on the SAME container key
    # CLI mode uses — "default" — not a per-session key.
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="agent:main:telegram:dm:123", profile="")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "default"
    finally:
        clear_session_vars(tokens)


def test_persistent_docker_two_sessions_same_profile_share_container(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="agent:main:telegram:dm:123", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="agent:main:discord:guild:456", profile="work")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == b == "profile:work"


def test_persistent_docker_distinct_profiles_get_distinct_containers(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    tokens = set_session_vars(session_key="sess-X", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="sess-X", profile="research")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == "profile:work"
    assert b == "profile:research"
    assert a != b


def test_nonpersistent_docker_keeps_session_scoping(monkeypatch):
    # container_persistent:false is an explicit isolation statement (#82731) —
    # the profile-scope gate must not fire there.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)


def test_ssh_backend_keeps_session_scoping(monkeypatch):
    # The original a270c4ade leak: SSH environments must stay session-scoped
    # regardless of profile, or profile A's SSHEnvironment leaks into B.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)


# --- Trusted-profiles shared container opt-in (#84671) ------------------------


def test_shared_key_unifies_profiles(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    tokens = set_session_vars(session_key="s1", profile="work")
    try:
        a = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_key="s2", profile="research")
    try:
        b = terminal_tool._resolve_container_task_id(None)
    finally:
        clear_session_vars(tokens)
    assert a == b == "shared:team/workspace"


def test_shared_key_applies_to_cli_no_session(monkeypatch):
    # CLI (no session key) must land in the same shared container as gateway
    # sessions, or the opt-in splits the container it exists to unify.
    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    assert terminal_tool._resolve_container_task_id(None) == "shared:team/workspace"


def test_empty_shared_key_keeps_profile_scoping(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars

    _persistent_docker(monkeypatch)
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "")
    tokens = set_session_vars(session_key="s1", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "profile:work"
    finally:
        clear_session_vars(tokens)


def test_shared_key_ignored_outside_persistent_docker(monkeypatch):
    # The opt-in is a persistent-Docker concept only: SSH keeps session
    # scoping even when the key is set.
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "team/workspace")
    tokens = set_session_vars(session_key="sess-A", profile="work")
    try:
        assert terminal_tool._resolve_container_task_id(None) == "session:sess-A"
    finally:
        clear_session_vars(tokens)
