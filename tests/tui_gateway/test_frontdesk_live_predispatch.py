import io
import importlib
import json
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.task_registry import STATUS_CANCELLED


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        importlib.reload(mod)


def _session(frontdesk_enabled: bool = True, running: bool = False):
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "model", "messages": []}
    return {
        "agent": agent,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "attached_images": [],
        "cols": 80,
        "frontdesk_live_enabled": frontdesk_enabled,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "pending_title": None,
        "running": running,
        "session_key": "s1",
        "transport": None,
    }


def _events(buf: io.StringIO):
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_stop_never_returns_busy_when_frontdesk_enabled(server):
    sid = "s1"
    session = _session(frontdesk_enabled=True, running=True)
    session["agent_ready"].set()
    server._sessions[sid] = session
    buf = io.StringIO()
    server._real_stdout = buf

    resp = server.handle_request({
        "id": "r1",
        "method": "prompt.submit",
        "params": {"session_id": sid, "text": "멈춰"},
    })

    assert "error" not in resp
    assert resp["result"]["status"] == "frontdesk"
    session["agent"].interrupt.assert_called_once_with("멈춰")
    assert any(
        msg.get("params", {}).get("type") == "message.complete"
        and "stopped" in msg.get("params", {}).get("payload", {}).get("text", "")
        for msg in _events(buf)
    )


def test_short_korean_chat_question_starts_agent_build_when_frontdesk_enabled(server):
    sid = "s1"
    session = _session(frontdesk_enabled=True, running=False)
    session["agent_ready"].set()
    server._sessions[sid] = session

    with patch.object(server, "_start_agent_build") as start_build, \
         patch.object(server, "_run_prompt_submit") as run_prompt:
        resp = server.handle_request({
            "id": "r-chat",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "지금 뭐 하고 있어?"},
        })

    assert "error" not in resp
    assert resp["result"]["status"] == "streaming"
    start_build.assert_called_once_with(sid, session)
    session["agent"].run_conversation.assert_not_called()


def test_slash_status_does_not_enter_frontdesk_prompt_gate(server):
    sid = "s1"
    session = _session(frontdesk_enabled=True, running=False)
    session["agent_ready"].set()
    server._sessions[sid] = session

    with patch.object(server, "_start_agent_build") as start_build, \
         patch.object(server, "_run_prompt_submit") as run_prompt:
        resp = server.handle_request({
            "id": "r-status",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "/status"},
        })

    assert "error" not in resp
    assert resp["result"]["status"] == "streaming"
    start_build.assert_called_once_with(sid, session)


def test_worker_request_does_not_start_agent_build_when_no_lane(server):
    sid = "s1"
    session = _session(frontdesk_enabled=True, running=False)
    session["agent_ready"].set()
    server._sessions[sid] = session
    buf = io.StringIO()
    server._real_stdout = buf

    with patch.object(server, "_start_agent_build") as start_build:
        resp = server.handle_request({
            "id": "r-worker",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "워커 레인에 배당해서 이 회귀를 조사해줘"},
        })

    assert "error" not in resp
    assert resp["result"]["status"] == "frontdesk"
    start_build.assert_not_called()
    session["agent"].run_conversation.assert_not_called()
    assert any(
        msg.get("params", {}).get("type") == "message.complete"
        and "worker lane unavailable" in msg.get("params", {}).get("payload", {}).get("text", "")
        for msg in _events(buf)
    )
    runtime = session.get("_orchestration_runtime")
    assert runtime is not None
    tasks = runtime.task_registry.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == STATUS_CANCELLED


def test_korean_followup_steers_when_frontdesk_enabled(server):
    sid = "s1"
    session = _session(frontdesk_enabled=True, running=True)
    session["agent_ready"].set()
    session["agent"].steer.return_value = True
    server._sessions[sid] = session

    resp = server.handle_request({
        "id": "r-steer",
        "method": "prompt.submit",
        "params": {"session_id": sid, "text": "빠니니를 파는 곳도 찾아보고 있어야지"},
    })

    assert "error" not in resp
    assert resp["result"]["status"] == "frontdesk"
    session["agent"].steer.assert_called_once_with("빠니니를 파는 곳도 찾아보고 있어야지")


def test_frontdesk_disabled_preserves_existing_busy_behavior(server):
    sid = "s1"
    session = _session(frontdesk_enabled=False, running=True)
    session["agent_ready"].set()
    server._sessions[sid] = session

    resp = server.handle_request({
        "id": "r2",
        "method": "prompt.submit",
        "params": {"session_id": sid, "text": "멈춰"},
    })

    assert resp["error"]["code"] == 4009
    assert "session busy" in resp["error"]["message"]
    session["agent"].interrupt.assert_not_called()


def test_frontdesk_disabled_starts_existing_model_path_when_idle(server):
    sid = "s1"
    session = _session(frontdesk_enabled=False, running=False)
    session["agent_ready"].set()
    server._sessions[sid] = session

    with patch.object(server, "_start_agent_build") as start_build, \
         patch.object(server, "_run_prompt_submit") as run_prompt:
        resp = server.handle_request({
            "id": "r-disabled",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "지금 뭐 하고 있어?"},
        })

    assert "error" not in resp
    assert resp["result"]["status"] == "streaming"
    start_build.assert_called_once_with(sid, session)
