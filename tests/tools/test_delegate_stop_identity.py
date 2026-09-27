from types import SimpleNamespace
from unittest.mock import patch

from tools.delegate_tool_results import _fire_subagent_stop_hooks


def test_stop_hook_preserves_stable_child_identity_after_session_rotation():
    child = SimpleNamespace(session_id="rotated-child", _subagent_id="sa-stable")
    parent = SimpleNamespace(session_id="parent", _current_turn_id="turn")
    entry = {"task_index": 0, "status": "completed", "summary": "done"}
    with patch("hermes_cli.plugins.invoke_hook") as hook:
        _fire_subagent_stop_hooks([entry], {0: child}, parent)
    assert hook.call_args.args == ("subagent_stop",)
    assert hook.call_args.kwargs["child_session_id"] == "rotated-child"
    assert hook.call_args.kwargs["child_subagent_id"] == "sa-stable"
