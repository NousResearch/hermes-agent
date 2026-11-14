"""``slash.exec`` /reasoning and /yolo change the LIVE session, not a slash-worker copy.

The slash worker is a separate HermesCLI: a worker run of ``/reasoning high`` or ``/yolo`` reported
the change while the session's agent kept its old effort and its approvals (ported from
rlaope/oh-my-hermes#2729, MIT, which had to route around it in its own TUI).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import server


def _exec(sid: str, command: str) -> dict:
    return server._methods["slash.exec"]("rid", {"session_id": sid, "command": command})


def test_slash_reasoning_and_yolo_reach_the_live_session() -> None:
    from tools.approval import is_session_yolo_enabled
    from tools.approval_yolo import toggle_session_yolo

    agent = SimpleNamespace(reasoning_config=None, service_tier=None, model="m", provider="p",
                            session_id="live-key")
    session = {"session_key": "live-key", "agent": agent, "history": [], "slash_worker": None}
    worker_spawn = patch.object(server, "_SlashWorker", side_effect=AssertionError("worker must not run"))
    try:
        with patch.dict(server._sessions, {"s1": session}), worker_spawn, \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), patch.object(server, "_emit"), \
                patch.object(server, "_session_db"):
            out = _exec("s1", "/reasoning high")["result"]["output"]
            assert "high" in out and agent.reasoning_config == {"enabled": True, "effort": "high"}
            write_key.assert_not_called()  # session scope: config.yaml untouched

            assert "ON" in _exec("s1", "/yolo")["result"]["output"]
            assert is_session_yolo_enabled("live-key")
            assert "OFF" in _exec("s1", "/yolo off")["result"]["output"]
            assert not is_session_yolo_enabled("live-key")
    finally:
        toggle_session_yolo("live-key", False)


def test_unrecognised_reasoning_form_still_reaches_the_worker() -> None:
    worker = SimpleNamespace(run=lambda cmd: f"worker:{cmd}", pop_seed=lambda: "")
    session = {"session_key": "k", "agent": None, "slash_worker": worker}
    with patch.dict(server._sessions, {"s2": session}):
        assert _exec("s2", "/reasoning bogus")["result"]["output"] == "worker:/reasoning bogus"
