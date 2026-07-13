"""Tests for mid-turn assistant message delivery over the TUI gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_interim_message")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


def test_agent_callback_emits_interim_message_without_settling_turn(server, monkeypatch):
    inflight_turn = {"assistant_text": "full attempted reply"}
    server._sessions["sid-1"] = {
        "inflight_turn": inflight_turn,
        "running": True,
    }
    emitted = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )

    callback = server._agent_cbs("sid-1")["interim_assistant_callback"]
    callback("full attempted reply", already_streamed=False)

    assert emitted == [
        (
            "message.interim",
            "sid-1",
            {"text": "full attempted reply", "already_streamed": False},
        )
    ]
    assert server._sessions["sid-1"]["inflight_turn"] is inflight_turn
    assert server._sessions["sid-1"]["running"] is True
