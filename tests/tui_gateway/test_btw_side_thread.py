"""TUI sidecar ``prompt.btw`` flags (tui_gateway/methods_prompt.py).

``--explore`` runs a side thread off the session's live agent and streams each step as
``btw.progress`` before the ``btw.complete`` report; ``--threads|--stop|--promote`` reply at once
with ``{text, prompt}`` so the client submits a promotion on its own turn path.
"""

from __future__ import annotations

import threading

import pytest

from agent import side_thread
from tests.tui_gateway.test_secret_capture_session import _gateway

SID = "session-btw"


@pytest.fixture
def sidecar(monkeypatch):
    server, server_requests, _skills = _gateway(monkeypatch)
    frames, done = [], threading.Event()

    def write(frame):
        frames.append(frame)
        if (frame.get("params") or {}).get("type") == "btw.complete":
            done.set()
        return True

    monkeypatch.setattr(server, "write_json", write)
    agent = type("Agent", (), {"_session_messages": [{"role": "user", "content": "fix it"}]})()
    with server._sessions_lock:
        server._sessions[SID] = {"agent": agent, "session_key": "key", "history": []}
    side_thread._registry.clear()
    try:
        yield server, frames, done, agent
    finally:
        with server._sessions_lock:
            server._sessions.pop(SID, None)
        server_requests.reset_for_tests()
        side_thread._registry.clear()


def _events(frames, kind):
    return [f["params"]["payload"] for f in frames if (f.get("params") or {}).get("type") == kind]


def test_explore_streams_progress_then_reports(sidecar, monkeypatch):
    server, frames, done, agent = sidecar
    seen = {}

    def run(thread, parent, history, *, on_progress):
        seen.update(parent=parent, history=history)
        thread.tool_calls.append("search_files")
        on_progress("search_files · retry")
        thread._finish("done", answer="retry lives in net.py:40")
        return thread.answer

    monkeypatch.setattr("agent.side_thread.explore_refusal", lambda _agent: None)
    monkeypatch.setattr("agent.side_thread.run_side_exploration", run)
    reply = server._methods["prompt.btw"]("rid", {"session_id": SID, "text": "--explore find the retry logic"})
    task_id = reply["result"]["task_id"]
    assert done.wait(10)

    assert seen == {"parent": agent, "history": agent._session_messages}
    assert _events(frames, "btw.progress") == [{"task_id": task_id, "text": "search_files · retry"}]
    (complete,) = _events(frames, "btw.complete")
    assert complete["task_id"] == task_id and "retry lives in net.py:40" in complete["text"]
    # The thread is known under the RPC's own task id, so the client's notice id promotes it.
    assert side_thread.find_thread(SID, task_id).status == "done"


def test_promote_replies_with_the_turn_to_submit(sidecar):
    server, frames, _done, _agent = sidecar
    thread = side_thread.register_thread(SID, "find the retry logic")
    thread._finish("done", answer="net.py:40")
    reply = server._methods["prompt.btw"]("rid", {"session_id": SID, "text": f"--promote {thread.thread_id}"})
    assert reply["result"]["prompt"] == side_thread.build_promotion_prompt(thread)
    assert thread.thread_id in reply["result"]["text"]
    assert "task_id" not in reply["result"] and frames == []


def test_explore_refusal_replies_at_once(sidecar, monkeypatch):
    server, frames, _done, _agent = sidecar
    monkeypatch.setattr("agent.side_thread.explore_refusal",
                        lambda _agent: side_thread.SideThreadUnavailable("provider"))
    reply = server._methods["prompt.btw"]("rid", {"session_id": SID, "text": "--explore x"})
    assert "task_id" not in reply["result"] and reply["result"]["text"]
    assert side_thread.list_threads(SID) == [] and frames == []
