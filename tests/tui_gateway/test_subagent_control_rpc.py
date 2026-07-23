"""JSON-RPC tests for operator-facing live subagent controls."""

from unittest.mock import patch

from tui_gateway import server


def test_subagent_steer_rpc_targets_one_live_child():
    with patch("tools.delegate_tool.steer_subagent", return_value=True) as steer:
        result = server._methods["subagent.steer"](
            1,
            {"subagent_id": "child-7", "instruction": "Focus on the failing test"},
        )

    assert result["result"] == {"accepted": True, "subagent_id": "child-7"}
    steer.assert_called_once_with("child-7", "Focus on the failing test")


def test_subagent_steer_rpc_rejects_malformed_input_without_lookup():
    with patch("tools.delegate_tool.steer_subagent") as steer:
        missing_id = server._methods["subagent.steer"](
            1, {"instruction": "new direction"}
        )
        missing_instruction = server._methods["subagent.steer"](
            2, {"subagent_id": "child-7", "instruction": "   "}
        )

    assert missing_id["error"] == {"code": 4000, "message": "subagent_id required"}
    assert missing_instruction["error"] == {
        "code": 4000,
        "message": "instruction required",
    }
    steer.assert_not_called()


def test_subagent_steer_rpc_keeps_operator_scope_unrestricted():
    """The TUI RPC intentionally does not pass a model delegation controller."""
    with patch("tools.delegate_tool.steer_subagent", return_value=False) as steer:
        server._methods["subagent.steer"](
            1, {"subagent_id": "selected-child", "instruction": "stop exploring"}
        )

    steer.assert_called_once_with("selected-child", "stop exploring")
