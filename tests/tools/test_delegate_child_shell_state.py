"""A delegate_task child's shell exports stay in the child (#84536).

At spawn, ChildRun.seed_workspace() seeds the child's cwd record and aliases the child to the
parent's environment (one sandbox, #16177: the parent sees the child's files). That environment
owns one bash snapshot every command sources and re-dumps, so a child's ``export``/``unset``
became the parent's and its siblings' shell state. Contract: copy-on-spawn, as for the cwd; the
child starts from the parent's exports, later changes stay in the child.

Adapted from the regression test posted on the issue by ar-labs-agent.
"""
from __future__ import annotations

import json
import uuid

import pytest

from tests.tools import _child_env_fixtures

child_env = _child_env_fixtures.child_env  # fixture: real local terminal, isolated


def _seed_child(parent: str, child: str) -> None:
    """What ChildRun.seed_workspace() does for a delegate_task child at spawn."""
    from tools.terminal_tool import get_session_cwd, record_session_cwd, register_container_alias

    record_session_cwd(child, get_session_cwd(parent))
    register_container_alias(child, parent)


def _sh(command: str, task_id: str, *, child: bool) -> str:
    from agent.delegation_context import delegated_child_context
    from tools.terminal_tool import terminal_tool

    if child:
        with delegated_child_context():
            raw = terminal_tool(command, task_id=task_id)
    else:
        raw = terminal_tool(command, task_id=task_id)
    result = json.loads(raw)
    assert result.get("exit_code", 0) == 0 and not result.get("error"), result
    return result["output"].strip()


@pytest.fixture(params=["gateway-session", "cli"])
def tasks(request, child_env):
    """A parent that exported PARENT_SET, then spawned two children (and one grandchild)."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.terminal_tool import clear_session_cwd, clear_task_env_overrides
    from tools.terminal_tool_lifecycle import cleanup_all_environments

    tokens = None
    session_key = f"agent:main:local:dm:repro-{uuid.uuid4().hex[:8]}"
    if request.param == "gateway-session":
        tokens = set_session_vars(platform="local", session_key=session_key, session_id=session_key.rsplit(":", 1)[-1])
    parent = f"parent-{uuid.uuid4().hex[:8]}"
    children = [f"subagent-{i}-{uuid.uuid4().hex[:8]}" for i in (0, 1)]
    try:
        _sh("export PARENT_SET=from-parent", parent, child=False)
        for child in children:
            _seed_child(parent, child)
        yield parent, children
    finally:
        cleanup_all_environments()
        for task_id in (parent, *children):
            clear_task_env_overrides(task_id)
        clear_session_cwd(session_key)
        if tokens is not None:
            clear_session_vars(tokens)


def test_control_child_inherits_what_the_parent_exported_before_the_spawn(tasks):
    _parent, (child, _sibling) = tasks
    assert _sh('printf %s "${PARENT_SET-unset}"', child, child=True) == "from-parent"


def test_control_parent_sees_files_the_child_writes(tasks):
    parent, (child, _sibling) = tasks
    _sh("printf written-by-child > child-file.txt", child, child=True)
    assert _sh("cat child-file.txt", parent, child=False) == "written-by-child"


def test_child_export_persists_in_the_child_but_not_the_parent(tasks):
    parent, (child, _sibling) = tasks
    _sh("export LEAK_PROBE=from-child", child, child=True)
    assert _sh('printf %s "${LEAK_PROBE-unset}"', child, child=True) == "from-child"
    assert _sh('printf %s "${LEAK_PROBE-unset}"', parent, child=False) == "unset"


def test_child_unset_does_not_remove_the_parents_export(tasks):
    parent, (child, _sibling) = tasks
    _sh("unset PARENT_SET", child, child=True)
    assert _sh('printf %s "${PARENT_SET-unset}"', child, child=True) == "unset"
    assert _sh('printf %s "${PARENT_SET-unset}"', parent, child=False) == "from-parent"


def test_child_export_does_not_reach_a_sibling(tasks):
    _parent, (child, sibling) = tasks
    _sh("export SIBLING_PROBE=from-first-child", child, child=True)
    assert _sh('printf %s "${SIBLING_PROBE-unset}"', sibling, child=True) == "unset"


def test_grandchild_starts_from_its_parent_child_and_stays_out_of_it(tasks):
    parent, (child, _sibling) = tasks
    _sh("export CHILD_SET=from-child", child, child=True)
    grandchild = f"{child}-g"
    _seed_child(child, grandchild)
    try:
        assert _sh('printf %s "${CHILD_SET-unset}"', grandchild, child=True) == "from-child"
        _sh("export GRAND_SET=from-grandchild", grandchild, child=True)
        assert _sh('printf %s "${GRAND_SET-unset}"', child, child=True) == "unset"
        assert _sh('printf %s "${GRAND_SET-unset}"', parent, child=False) == "unset"
    finally:
        from tools.terminal_tool import clear_task_env_overrides
        clear_task_env_overrides(grandchild)
