"""Handler-level tests for TUI ``command.dispatch`` /learn gating (#74249).

``/learn`` authors a skill via ``skill_manage``; on a read-only skill tool set
(``skills_list`` + ``skill_view``, no ``skill_manage``) the turn would be doomed
— the model is invited to call a tool that is not in its schema — so the
handler must return the 4003 unavailable error instead of submitting.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.learn_prompt import LEARN_UNAVAILABLE_MESSAGE, build_learn_prompt

_READ_ONLY = {"memory", "skills_list", "skill_view"}
_WRITABLE = _READ_ONLY | {"skill_manage"}


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Mocks are scoped to the initial import only (mirrors test_goal_command.py).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(mod, "_hermes_home", home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_sig", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


def _make_session(server, sid, agent):
    server._sessions[sid] = {
        "session_key": sid,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": agent,
    }


def _dispatch(server, sid, arg="auth flow"):
    return server._methods["command.dispatch"](
        1, {"name": "learn", "arg": arg, "session_id": sid}
    )


def test_tui_learn_read_only_returns_unavailable_error(server):
    _make_session(server, "learn-session", MagicMock(valid_tool_names=_READ_ONLY))
    resp = _dispatch(server, "learn-session")
    assert "error" in resp
    assert resp["error"]["code"] == 4003
    assert resp["error"]["message"] == LEARN_UNAVAILABLE_MESSAGE


def test_tui_learn_writable_returns_send_payload(server):
    _make_session(server, "learn-session", MagicMock(valid_tool_names=_WRITABLE))
    resp = _dispatch(server, "learn-session", arg="focus on oauth")
    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == build_learn_prompt("focus on oauth")
