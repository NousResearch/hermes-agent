"""delegate_task inherit_context: which parent attribute the inherited transcript is read from.

The gateway AIAgent keeps its live transcript in ``_session_messages`` and has no
``conversation_history`` attribute; the CLI console path uses ``conversation_history``.
``_session_messages`` wins when present -- including when it is present but EMPTY, which means
there is nothing to inherit, not "try the other attribute".
"""
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate_inherit_context import _msg, _parent
from tools.delegate_tool import _build_child_agent

BOOT = [{"role": "user", "content": "boot seed"}]


def _prefill(parent):
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="Continue the work", context=None, toolsets=None, model=None,
                           max_iterations=10, parent_agent=parent, task_count=1, inherit_context=True)
    return MockAgent.call_args.kwargs["prefill_messages"]


def test_gateway_parent_without_conversation_history_inherits_session_messages():
    parent = _parent([_msg("user", "GATEWAY_FACT deploy window is 22:00")])
    del parent.conversation_history
    assert "GATEWAY_FACT" in _prefill(parent)[0]["content"]


def test_cli_parent_without_session_messages_falls_back_to_conversation_history():
    parent = _parent()
    del parent._session_messages
    parent.conversation_history = [_msg("user", "CLI_FACT the branch is release/2")]
    assert "CLI_FACT" in _prefill(parent)[0]["content"]


def test_present_but_empty_session_messages_does_not_fall_back_to_conversation_history():
    parent = _parent([])
    parent.conversation_history = [_msg("user", "STALE_FACT from an earlier transcript")]
    assert _prefill(parent) == BOOT
