"""
Regression tests for container task_id mapping.

Top-level CLI calls use ``None`` and share ``"default"``. Gateway/ACP calls
use per-session task IDs and must keep isolated sandboxes. ``delegate_task``
children use separate child IDs for file-state/UI bookkeeping, then explicitly
alias those IDs to the parent's sandbox key while the child is running.
"""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_task_routing_state():
    """Ensure no stray overrides or aliases from other tests leak in."""
    before_overrides = dict(terminal_tool._task_env_overrides)
    before_aliases = dict(terminal_tool._task_container_aliases)
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_container_aliases.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    terminal_tool._task_container_aliases.clear()
    terminal_tool._task_container_aliases.update(before_aliases)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_literal_default_stays_default():
    assert terminal_tool._resolve_container_task_id("default") == "default"


def test_unaliased_subagent_task_id_collapses_to_default_for_cli_compatibility():
    assert terminal_tool._resolve_container_task_id("sa-0-deadbeef") == "default"
    assert terminal_tool._resolve_container_task_id("subagent-42-cafef00d") == "default"


def test_gateway_session_id_keeps_its_own_container_key():
    assert (
        terminal_tool._resolve_container_task_id("sess-123e4567-e89b-12d3")
        == "sess-123e4567-e89b-12d3"
    )


def test_subagent_alias_maps_child_to_parent_session_container():
    terminal_tool.register_task_container_alias("sa-0-deadbeef", "gateway-session-1")
    assert terminal_tool._resolve_container_task_id("sa-0-deadbeef") == "gateway-session-1"


def test_subagent_alias_maps_child_to_default_for_cli_parent():
    terminal_tool.register_task_container_alias("sa-0-deadbeef", None)
    assert terminal_tool._resolve_container_task_id("sa-0-deadbeef") == "default"


def test_cleared_subagent_alias_collapses_to_default_again():
    terminal_tool.register_task_container_alias("sa-0-deadbeef", "gateway-session-1")
    assert terminal_tool._resolve_container_task_id("sa-0-deadbeef") == "gateway-session-1"
    terminal_tool.clear_task_container_alias("sa-0-deadbeef")
    assert terminal_tool._resolve_container_task_id("sa-0-deadbeef") == "default"


def test_rl_task_with_override_keeps_its_own_id():
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


def test_cleared_override_preserves_non_subagent_task_id():
    terminal_tool.register_task_env_overrides("tb2-x", {"docker_image": "x:y"})
    assert terminal_tool._resolve_container_task_id("tb2-x") == "tb2-x"
    terminal_tool.clear_task_env_overrides("tb2-x")
    assert terminal_tool._resolve_container_task_id("tb2-x") == "tb2-x"


def test_get_active_env_reads_aliased_parent_container_from_subagent_id():
    sentinel = object()
    terminal_tool._active_environments["gateway-session-1"] = sentinel
    terminal_tool.register_task_container_alias("sa-7-cafe", "gateway-session-1")
    try:
        assert terminal_tool.get_active_env("sa-7-cafe") is sentinel
    finally:
        terminal_tool._active_environments.pop("gateway-session-1", None)


def test_get_active_env_keeps_distinct_gateway_sessions():
    session_a = object()
    session_b = object()
    terminal_tool._active_environments["gateway-session-a"] = session_a
    terminal_tool._active_environments["gateway-session-b"] = session_b
    try:
        assert terminal_tool.get_active_env("gateway-session-a") is session_a
        assert terminal_tool.get_active_env("gateway-session-b") is session_b
    finally:
        terminal_tool._active_environments.pop("gateway-session-a", None)
        terminal_tool._active_environments.pop("gateway-session-b", None)


def test_get_active_env_honours_rl_override():
    rl_env = object()
    default_env = object()
    terminal_tool._active_environments["default"] = default_env
    terminal_tool._active_environments["rl-42"] = rl_env
    terminal_tool.register_task_env_overrides("rl-42", {"docker_image": "x"})
    try:
        assert terminal_tool.get_active_env("rl-42") is rl_env
    finally:
        terminal_tool.clear_task_env_overrides("rl-42")
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._active_environments.pop("rl-42", None)


class _CleanupProbe:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def test_cleanup_vm_resolves_session_id_to_shared_container():
    """API/session task IDs must tear down the real shared sandbox key."""
    env = _CleanupProbe()
    terminal_tool._active_environments["default"] = env
    terminal_tool._last_activity["default"] = 123.0
    terminal_tool._creation_locks["default"] = object()
    try:
        terminal_tool.cleanup_vm("api-session-A")

        assert env.cleaned is True
        assert "default" not in terminal_tool._active_environments
        assert "default" not in terminal_tool._last_activity
        assert "default" not in terminal_tool._creation_locks
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)
        terminal_tool._creation_locks.pop("default", None)


def test_cleanup_vm_honours_override_task_id():
    """Isolated benchmark task IDs should still clean their own sandbox."""
    env = _CleanupProbe()
    terminal_tool.register_task_env_overrides("rl-cleanup", {"docker_image": "x"})
    terminal_tool._active_environments["rl-cleanup"] = env
    try:
        terminal_tool.cleanup_vm("rl-cleanup")

        assert env.cleaned is True
        assert "rl-cleanup" not in terminal_tool._active_environments
    finally:
        terminal_tool.clear_task_env_overrides("rl-cleanup")
        terminal_tool._active_environments.pop("rl-cleanup", None)
