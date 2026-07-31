"""Handler-level tests for TUI ``command.dispatch`` /learn gating."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from agent.learn_prompt import LEARN_UNAVAILABLE_MESSAGE, build_learn_prompt

_READ_ONLY = {"memory", "skills_list", "skill_view"}
_WRITABLE = _READ_ONLY | {"skill_manage"}


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    mod._pending.clear()
    mod._answers.clear()
    mod._live_transports.clear()


def _dispatch(server, agent, arg="auth flow"):
    sid = "learn-session"
    server._sessions[sid] = {
        "session_key": sid,
        "agent": agent,
    }
    return server.handle_request(
        {
            "id": "learn-rid",
            "method": "command.dispatch",
            "params": {"name": "learn", "arg": arg, "session_id": sid},
        }
    )


def test_tui_learn_read_only_returns_unavailable_error(server):
    agent = MagicMock(valid_tool_names=_READ_ONLY)
    resp = _dispatch(server, agent)
    assert "error" in resp
    assert resp["error"]["code"] == 4003
    assert resp["error"]["message"] == LEARN_UNAVAILABLE_MESSAGE


def test_tui_learn_writable_returns_send_payload(server):
    agent = MagicMock(valid_tool_names=_WRITABLE)
    resp = _dispatch(server, agent, arg="focus on oauth")
    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == build_learn_prompt("focus on oauth")
