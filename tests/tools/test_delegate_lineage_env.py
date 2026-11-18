"""HERMES_SESSION_PARENT_ID: a subagent's local terminal names the session that dispatched it.

The shell test drives the public ``delegate_task`` (real ``_build_child_agent``, ``_ChildRun``
threads and the local terminal backend). Only the model is fake: the child's ``run_conversation``
issues the terminal and ``delegate_task`` calls a model would. It imports only names that predate
the variable, so on a tree without it the shell assertions fail, not the imports.
"""

from __future__ import annotations

import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.tools import _child_env_fixtures

child_env = _child_env_fixtures.child_env  # fixture: real local terminal, isolated HOME/HERMES_HOME

PARENT_ID = "HERMES_SESSION_PARENT_ID"
PROBE = 'printf "%s|%s" "${HERMES_SESSION_ID-missing}" "${HERMES_SESSION_PARENT_ID-missing}"'
NEST, LEAF = "Nest one level and probe the shell", "Probe the shell"


def _shell(task_id: str | None) -> tuple[str, str]:
    """(HERMES_SESSION_ID, HERMES_SESSION_PARENT_ID) as a command in the terminal sees them."""
    from tools.terminal_tool import terminal_tool

    result = json.loads(terminal_tool(PROBE, task_id=task_id))
    assert result.get("exit_code", 0) == 0 and not result.get("error"), result
    session_id, parent_id = result["output"].strip().split("|")
    return session_id, parent_id


def _agent_fields(agent, *, session_id: str, depth: int, task_id: str | None) -> None:
    agent.session_id, agent._delegate_depth, agent._current_task_id = session_id, depth, task_id
    agent.base_url, agent.api_key = "https://openrouter.ai/api/v1", "test-key"
    agent.provider, agent.api_mode, agent.model, agent.platform = "openrouter", "chat_completions", "m", "cli"
    agent._session_db = agent._print_fn = agent.tool_progress_callback = agent.thinking_callback = None
    agent.providers_allowed = agent.providers_ignored = agent.providers_order = agent.provider_sort = None
    agent._active_children, agent._active_children_lock = [], threading.Lock()


def _fake_model_agent(seen: dict):
    """Stands in for ``run_agent.AIAgent``: the real class keeps ``parent_session_id`` as
    ``_parent_session_id`` (agent/agent_init.py); its turn runs a fixed tool script."""

    def build(**kwargs):
        child = MagicMock()
        _agent_fields(child, session_id=kwargs.get("session_id") or f"sub-{uuid.uuid4().hex[:8]}", depth=1,
                      task_id=None)
        child._parent_session_id = kwargs.get("parent_session_id")

        def run_conversation(user_message, task_id=None, **_kwargs):
            from tools.delegate_tool import delegate_task

            child._current_task_id = task_id
            if NEST not in user_message:
                seen["grandchild"] = (child.session_id, *_shell(task_id))
                return {"final_response": "probed", "completed": True, "api_calls": 1}
            seen["child"] = (child.session_id, *_shell(task_id))
            reply = json.loads(delegate_task(goal=LEAF, parent_agent=child))
            assert reply["results"][0]["status"] == "completed", reply
            seen["child_after_nested"] = (child.session_id, *_shell(task_id))
            # What context compression does mid-run: a new session id, rebound for this worker.
            from agent.conversation_compression import _rebind_session_context

            child.session_id = f"{child.session_id}-segment-2"
            _rebind_session_context(child.session_id)
            seen["child_after_rebind"] = (child.session_id, *_shell(task_id))
            return {"final_response": "nested", "completed": True, "api_calls": 2}

        child.run_conversation.side_effect = run_conversation
        return child

    return build


@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_subagent_shell_names_its_direct_dispatcher_and_nothing_leaks(child_env, monkeypatch, surface):
    from gateway.session_context import clear_session_vars, get_session_env, set_current_session_id, set_session_vars
    from tools.delegate_tool import delegate_task
    from tools.terminal_tool_lifecycle import cleanup_all_environments

    # This Hermes was itself started from a subagent's shell: its environment carries that run's value.
    monkeypatch.setenv(PARENT_ID, "outer-launcher")
    top = f"top-{uuid.uuid4().hex[:8]}"
    tokens = None
    if surface == "gateway":
        tokens = set_session_vars(platform="local", session_key=f"agent:main:local:dm:{top}", session_id=top)
    else:  # a CLI process never binds session vars (the latch is process-wide, so reset it)
        monkeypatch.setattr("gateway.session_context._session_context_engaged", False)
        set_current_session_id(top)
    parent = MagicMock()
    _agent_fields(parent, session_id=top, depth=0, task_id=f"task-{top}")
    seen: dict = {}
    try:
        before = _shell(parent._current_task_id)
        with (patch("run_agent.AIAgent", side_effect=_fake_model_agent(seen)),
              patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
              patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True)):
            reply = json.loads(delegate_task(goal=NEST, role="orchestrator", parent_agent=parent))
        assert reply["results"][0]["status"] == "completed", reply
        child_id = seen["child"][0]
        assert seen["child"] == (child_id, child_id, top)
        grandchild_id = seen["grandchild"][0]
        assert seen["grandchild"] == (grandchild_id, grandchild_id, child_id)  # direct parent, not the top session
        assert seen["child_after_nested"] == (child_id, child_id, top)
        assert seen["child_after_rebind"] == (f"{child_id}-segment-2", f"{child_id}-segment-2", top)
        # The same shell, before and after, in the top-level session: no dispatcher, inherited or leaked.
        assert before == _shell(parent._current_task_id) == (top, "missing")
        assert get_session_env(PARENT_ID) == ""
    finally:
        cleanup_all_environments()
        if tokens is not None:
            clear_session_vars(tokens)


def test_backend_env_overlay_cannot_set_it(child_env, monkeypatch):
    """A configured terminal env (``LocalEnvironment(env=...)``) can't stamp a dispatcher either."""
    from tools.environments.local import LocalEnvironment

    monkeypatch.setenv(PARENT_ID, "outer-launcher")
    monkeypatch.setattr("gateway.session_context._session_context_engaged", False)  # CLI process
    shell = LocalEnvironment(cwd=str(child_env), env={PARENT_ID: "configured"})
    try:
        result = shell.execute(f'printf %s "${{{PARENT_ID}-missing}}"')
        assert (result["returncode"], result["output"].strip()) == (0, "missing")
    finally:
        shell.cleanup()


def test_unknown_dispatcher_is_absent_and_hides_the_enclosing_runs():
    from agent.delegation_context import delegated_child_context
    from tools.environments.local import build_subprocess_env

    with delegated_child_context("child", parent_session_id="top"):
        assert build_subprocess_env(base={})[PARENT_ID] == "top"
        with delegated_child_context("grandchild"):
            assert PARENT_ID not in build_subprocess_env(base={})
        with delegated_child_context():  # construction scope: binds no run identity
            assert build_subprocess_env(base={})[PARENT_ID] == "top"
