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


def test_literal_default_stays_default():
    assert terminal_tool._resolve_container_task_id("default") == "default"


def test_subagent_task_id_collapses_to_default():
    # delegate_task constructs IDs like "subagent-<N>-<uuid_hex>"; these
    # should share the parent's container, not spin up their own.
    assert terminal_tool._resolve_container_task_id("subagent-0-deadbeef") == "default"
    assert terminal_tool._resolve_container_task_id("subagent-42-cafef00d") == "default"


def test_arbitrary_session_id_collapses_to_default():
    # Session UUIDs or anything else without an override still collapse.
    assert terminal_tool._resolve_container_task_id("sess-123e4567-e89b-12d3") == "default"


def test_rl_task_with_override_keeps_its_own_id():
    # RL / benchmark pattern: register a per-task image, then the task_id
    # must survive ``_resolve_container_task_id`` so the rollout lands in
    # its own sandbox.
    terminal_tool.register_task_env_overrides(
        "tb2-task-fix-git", {"docker_image": "tb2:fix-git", "cwd": "/app"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("tb2-task-fix-git")
            == "tb2-task-fix-git"
        )
    finally:
        terminal_tool.clear_task_env_overrides("tb2-task-fix-git")


def test_cleared_override_collapses_again():
    terminal_tool.register_task_env_overrides("tb2-x", {"docker_image": "x:y"})
    assert terminal_tool._resolve_container_task_id("tb2-x") == "tb2-x"
    terminal_tool.clear_task_env_overrides("tb2-x")
    assert terminal_tool._resolve_container_task_id("tb2-x") == "default"


def test_get_active_env_reads_shared_container_from_subagent_id():
    """``get_active_env`` must see the shared ``"default"`` sandbox when
    called with a subagent's task_id, so the agent loop's turn-budget
    enforcement reads the real env (not None) during delegation."""
    sentinel = object()
    terminal_tool._active_environments["default"] = sentinel
    try:
        assert terminal_tool.get_active_env("subagent-7-cafe") is sentinel
        assert terminal_tool.get_active_env(None) is sentinel
        assert terminal_tool.get_active_env("default") is sentinel
    finally:
        terminal_tool._active_environments.pop("default", None)


def test_get_active_env_honours_rl_override():
    rl_env = object()
    default_env = object()
    terminal_tool._active_environments["default"] = default_env
    terminal_tool._active_environments["rl-42"] = rl_env
    terminal_tool.register_task_env_overrides("rl-42", {"docker_image": "x"})
    try:
        # With an override registered, lookup returns the task's own env,
        # not the shared "default" one.
        assert terminal_tool.get_active_env("rl-42") is rl_env
    finally:
        terminal_tool.clear_task_env_overrides("rl-42")
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._active_environments.pop("rl-42", None)


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


def test_cwd_plus_docker_image_keeps_own_id():
    """When overrides include both cwd AND docker_image, isolation must
    still be honoured (RL/benchmark pattern with explicit cwd)."""
    terminal_tool.register_task_env_overrides(
        "rl-with-cwd", {"docker_image": "myimg:latest", "cwd": "/workspace"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("rl-with-cwd")
            == "rl-with-cwd"
        )
    finally:
        terminal_tool.clear_task_env_overrides("rl-with-cwd")


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
