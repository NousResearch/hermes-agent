"""A delegate_task child's role is decided by the TARGET profile's config, not the caller's.

``_build_child_agent`` binds the target profile's scope (``delegated_profile_scope(profile_home)``)
for the child's credentials and model, but the ``delegation.orchestrator_enabled`` kill switch used
to be read *before* that binding — so it was resolved against the caller's ``config.yaml``.

On a multi-profile fleet that is wrong in both directions: a caller that had the switch off
silently downgraded every routed child to ``leaf``, even when the profile that actually runs the
child had opted into orchestrator behaviour. The target scope is the scope the child runs in, so
it is the scope that decides the child's role.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


def _parent():
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._credential_pool = MagicMock(name="caller-pool")
    return parent


def _build_child(profile_home):
    from tools.delegate_tool import _build_child_agent

    with (
        patch("run_agent.AIAgent") as MockAgent,
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
    ):
        MockAgent.return_value = MagicMock()
        return _build_child_agent(
            task_index=0,
            goal="Coordinate safely",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=_parent(),
            task_count=1,
            profile_home=profile_home,
        )


def test_orchestrator_switch_is_read_in_the_target_profile_scope(tmp_path, monkeypatch):
    """The switch is resolved inside the target profile's scope, not the caller's."""
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    seen: dict[str, str] = {}

    def _fake_switch():
        from hermes_constants import get_hermes_home
        seen["home"] = str(get_hermes_home())
        return True

    with patch("tools.delegate_tool._get_orchestrator_enabled", _fake_switch):
        _build_child(target_home)

    assert seen.get("home") == str(target_home), (
        "orchestrator_enabled was resolved outside the target profile scope"
    )


def test_switch_read_in_the_target_scope_changes_the_child_role(tmp_path, monkeypatch):
    """A target profile that turns the switch off downgrades the child, whatever the caller says."""
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    # The switch is off in the caller's scope and on in the target's: reading it in the
    # caller's scope would downgrade the child to leaf.
    def _scoped_switch():
        from hermes_constants import get_hermes_home
        return get_hermes_home() == target_home

    with patch("tools.delegate_tool._get_orchestrator_enabled", _scoped_switch):
        child = _build_child(target_home)

    assert child._delegate_role == "orchestrator"


def test_switch_off_in_the_target_scope_downgrades_the_child(tmp_path, monkeypatch):
    """The reverse: the target profile opts out, the caller opts in."""
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    def _scoped_switch():
        from hermes_constants import get_hermes_home
        return get_hermes_home() == caller_home

    with patch("tools.delegate_tool._get_orchestrator_enabled", _scoped_switch):
        child = _build_child(target_home)

    assert child._delegate_role == "leaf"


def test_switch_is_read_in_the_caller_scope_when_there_is_no_profile_route(tmp_path, monkeypatch):
    """Without a profile route there is no target scope: the caller's own scope decides."""
    caller_home = tmp_path / "caller"
    caller_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    seen: dict[str, str] = {}

    def _fake_switch():
        from hermes_constants import get_hermes_home
        seen["home"] = str(get_hermes_home())
        return True

    with patch("tools.delegate_tool._get_orchestrator_enabled", _fake_switch):
        child = _build_child(None)

    assert seen.get("home") == str(caller_home)
    assert child._delegate_role == "orchestrator"


@pytest.mark.parametrize("depth,expected", [(0, "orchestrator"), (1, "leaf")])
def test_depth_budget_still_degrades_the_role(tmp_path, monkeypatch, depth, expected):
    """The target-scope read does not change the depth budget rule."""
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    from tools.delegate_tool import _build_child_agent

    parent = _parent()
    parent._delegate_depth = depth

    with (
        patch("run_agent.AIAgent") as MockAgent,
        patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
    ):
        MockAgent.return_value = MagicMock()
        child = _build_child_agent(
            task_index=0,
            goal="Coordinate safely",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=parent,
            task_count=1,
            profile_home=target_home,
        )

    assert child._delegate_role == expected
